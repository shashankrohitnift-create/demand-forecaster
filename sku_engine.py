"""
Multi-SKU Demand Forecasting Engine  (v2)
=========================================
Adds to the original pipeline:
  1. MOVING SEASONALITY  — festival dates (Ramadan/Eid) are supplied as a calendar.
     Each month gets "fraction of the month covered by the event" as a regressor,
     so the peak follows the festival across calendar months. Fixed annual shape
     (e.g. summer) is captured with Fourier terms.
  2. CONFIGURABLE SPLIT  — you choose how many months train vs test (holdout).
  3. ROP / ROQ           — reorder point & quantity per SKU for DC replenishment.

ROUTING
  regular demand   (ADI < cut, enough history) -> SARIMAX + calendar regressors
                                                  (falls back to Holt-Winters, then naive)
  intermittent     (ADI >= cut)                -> Croston / SBA (flat rate)
  short history                                -> seasonal-naive
Every SKU is scored on the holdout against seasonal-naive; if it loses, naive is used.
"""

import numpy as np
import pandas as pd
import warnings
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# ----------------------------- defaults (overridable) -----------------------------
HORIZON        = 6      # months to forecast into the future
TEST_MONTHS    = 6      # months held out to score accuracy
SEASON         = 12
FOURIER_K      = 2      # Fourier pairs for fixed annual seasonality
ADI_CUT        = 1.32
CV2_CUT        = 0.49
MIN_MONTHS     = 24     # min history for SARIMAX / Holt-Winters

# Default UAE festival calendar. Replace/extend via the app or by editing this list.
# (name, start YYYY-MM-DD, end YYYY-MM-DD) inclusive.
DEFAULT_EVENTS = [
    ("ramadan", "2022-04-02", "2022-05-01"), ("eid_fitr", "2022-05-02", "2022-05-04"),
    ("ramadan", "2023-03-23", "2023-04-20"), ("eid_fitr", "2023-04-21", "2023-04-23"),
    ("ramadan", "2024-03-11", "2024-04-09"), ("eid_fitr", "2024-04-10", "2024-04-12"),
    ("ramadan", "2025-03-01", "2025-03-29"), ("eid_fitr", "2025-03-30", "2025-04-01"),
    ("ramadan", "2026-02-18", "2026-03-19"), ("eid_fitr", "2026-03-20", "2026-03-22"),
]


# ============================ data prep ============================
def load_long(path, date_col="date", sku_col="sku", qty_col="qty", sheet=None):
    """Read a long-format sales file. For Excel, picks the sheet that actually
    contains the required columns (so a leading 'readme' sheet doesn't break it)."""
    if str(path).lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        book = pd.read_excel(path, sheet_name=sheet)
        if isinstance(book, dict):                      # all sheets
            want = {date_col.lower(), sku_col.lower(), qty_col.lower()}
            pick = None
            for name, d in book.items():
                cols = {str(c).strip().lower() for c in d.columns}
                if want.issubset(cols):
                    pick = d; break
            df = pick if pick is not None else list(book.values())[0]
        else:
            df = book
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={date_col: "date", sku_col: "sku", qty_col: "qty"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0)
    keep = ["date", "sku", "qty"] + (["rep"] if "rep" in df.columns else [])
    return df.dropna(subset=["date"])[keep]


def to_monthly_matrix(df, key="sku"):
    """Long daily -> wide monthly (rows = month, cols = series key). Missing months = 0."""
    m = (df.set_index("date").groupby(key)["qty"].resample("MS").sum().reset_index())
    mat = m.pivot(index="date", columns=key, values="qty")
    full = pd.date_range(mat.index.min(), mat.index.max(), freq="MS")
    return mat.reindex(full).fillna(0.0)


SEP = " || "   # separator between SKU and salesman inside a series key


def apply_grouping(df, mode="sku", sep=SEP):
    """Build the series key the forecast runs on.

    mode:
      "sku"      one series per SKU (default; use this for DC replenishment)
      "sku_rep"  one series per SKU x country

    Splitting demand across countries makes each series smaller and spikier,
    which usually lowers accuracy. Use the split for territory/target views,
    not for stock planning.
    """
    d = df.copy()
    if mode == "sku_rep" and "rep" in d.columns:
        d["series"] = d["sku"].astype(str) + sep + d["rep"].astype(str)
    else:
        d["series"] = d["sku"].astype(str)
    return d


def split_series_key(out, mode, sep=SEP):
    """Turn the series key into separate SKU / Country columns."""
    df = out.copy()
    if mode == "sku_rep":
        parts = df["SKU"].astype(str).str.split(sep, n=1, expand=True, regex=False)
        df.insert(0, "Country", parts[1] if parts.shape[1] > 1 else "")
        df["SKU"] = parts[0]
        cols = ["SKU", "Country"] + [c for c in df.columns if c not in ("SKU", "Country")]
        df = df[cols]
    return df


def split_impact(df, mode, sep=" | "):
    """Quantify what the split does: series count, and how many become intermittent."""
    base = to_monthly_matrix(apply_grouping(df, "sku"), "series")
    out = {"series_sku": base.shape[1]}
    b_int = sum(1 for c in base.columns if classify(base[c])[0] in ("intermittent", "lumpy"))
    out["intermittent_sku"] = b_int
    if mode == "sku_rep" and "rep" in df.columns:
        sp = to_monthly_matrix(apply_grouping(df, mode, sep), "series")
        out["series_split"] = sp.shape[1]
        out["intermittent_split"] = sum(
            1 for c in sp.columns if classify(sp[c])[0] in ("intermittent", "lumpy"))
    return out


# ============================ moving seasonality ============================
def event_features(month_index, events):
    """Fraction of each month covered by each named event -> one column per event.

    This is what makes seasonality *moving*: a Ramadan spanning 11 Mar - 9 Apr puts
    ~0.68 in March and ~0.30 in April, so the model learns the lift and applies it
    wherever the festival falls in a future year.
    """
    names = sorted({n for n, _, _ in events})
    feat = pd.DataFrame(0.0, index=month_index, columns=[f"ev_{n}" for n in names])
    for name, s, e in events:
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        for m in month_index:
            m_end = m + pd.offsets.MonthEnd(0)
            lo, hi = max(m, s), min(m_end, e)
            if hi >= lo:
                feat.at[m, f"ev_{name}"] += ((hi - lo).days + 1) / m_end.day
    return feat.clip(upper=1.0)


def fourier_features(month_index, period=None, K=None):
    """Smooth fixed annual seasonality (e.g. summer), independent of festivals."""
    period = period or SEASON
    K = K or FOURIER_K
    t = np.arange(len(month_index))
    cols = {}
    for k in range(1, K + 1):
        cols[f"sin{k}"] = np.sin(2 * np.pi * k * t / period)
        cols[f"cos{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols, index=month_index)


def build_exog(full_index, events):
    return pd.concat([event_features(full_index, events),
                      fourier_features(full_index)], axis=1)


# ============================ classification ============================
def classify(y):
    nz = y[y > 0]
    if len(nz) < 2:
        return "sparse", np.nan, np.nan
    adi = len(y) / len(nz)
    cv2 = (nz.std(ddof=0) / nz.mean()) ** 2
    if adi < ADI_CUT and cv2 < CV2_CUT:      q = "smooth"
    elif adi < ADI_CUT:                      q = "erratic"
    elif cv2 < CV2_CUT:                      q = "intermittent"
    else:                                    q = "lumpy"
    return q, adi, cv2


# ============================ forecasters ============================
def f_sarimax(y, exog_hist, exog_future, order=(1, 1, 1)):
    res = sm.tsa.SARIMAX(y, exog=exog_hist, order=order,
                         enforce_stationarity=False,
                         enforce_invertibility=False).fit(disp=False, maxiter=200)
    return np.asarray(res.forecast(steps=len(exog_future), exog=exog_future), float)


def f_holt_winters(y, h):
    seasonal = "mul" if np.min(y) > 0 else "add"
    m = ExponentialSmoothing(y, trend="add", seasonal=seasonal,
                             seasonal_periods=SEASON, damped_trend=True).fit()
    return np.asarray(m.forecast(h), float)


def f_croston(y, h, alpha=0.1, sba=True):
    v = np.asarray(y, float)
    z = p = None; q = 1; first = True
    for x in v:
        if x > 0:
            if first: z, p, first = x, q, False
            else:     z += alpha * (x - z); p += alpha * (q - p)
            q = 1
        else:
            q += 1
    if z is None or not p:
        return np.zeros(h)
    return np.full(h, z / p * ((1 - alpha / 2) if sba else 1.0))


DORMANT_MONTHS = 6        # no sales for this long -> treat as winding down
ZERO_FLOOR_SHARE = 0.35   # floor as a share of recent monthly level
NAIVE_BLEND = 0.5   # 0 = pure same-month-last-year, 1 = pure recent level


def recent_level(y, months=6):
    """What the SKU is selling *now* — mean of the last `months` months.

    Deliberately does NOT fall back to older history: a SKU that stopped selling
    should forecast ~0, not be resurrected by sales from a year ago.
    """
    v = np.asarray(y, float)
    if v.size == 0:
        return 0.0
    tail = v[-months:] if v.size >= months else v
    return float(np.mean(tail))


def f_naive(y, h, blend=None):
    """Seasonal naive, damped toward the recent level.

    Pure same-month-last-year copies single months verbatim, so a month that
    happened to be zero last year forecasts zero even when the SKU is clearly
    selling now. Blending with the recent level keeps the seasonal shape while
    preventing indefensible zeros.
    """
    blend = NAIVE_BLEND if blend is None else blend
    v = np.asarray(y, float)
    if v.size == 0:
        return np.zeros(h)
    lvl = recent_level(y)
    if v.size >= SEASON:
        base = v[-SEASON:]
        seas = np.array([base[i % SEASON] for i in range(h)], dtype=float)
        return (1 - blend) * seas + blend * lvl
    return np.full(h, lvl)


def wmape(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    d = np.abs(a).sum()
    return np.abs(a - f).sum() / d if d > 0 else np.nan


# ============================ per-SKU fit ============================
def smart_round(values):
    """Round sensibly for the scale of the SKU.

    Rounding everything to whole numbers destroys low-volume SKUs: a carton SKU
    selling ~0.5/month would round to 0 every month. Keep decimals when the
    numbers are small, whole units when they are not.
    """
    v = np.asarray(values, float)
    peak = float(np.nanmax(v)) if v.size else 0.0
    if peak < 10:
        return np.round(v, 2)      # e.g. 0.56 cartons
    if peak < 100:
        return np.round(v, 1)
    return np.round(v, 0)


def trim_to_launch(y, min_keep=6):
    """Drop leading zero months that precede a SKU's first ever sale.

    reindex(...).fillna(0) creates zeros for months before a SKU existed. Counting
    those as 'no demand' inflates ADI, misroutes new SKUs to Croston and produces a
    near-zero forecast. Pre-launch months are absence of data, not zero demand.
    Trailing zeros are KEPT — those may be a genuine stop in selling.
    """
    v = np.asarray(y, float)
    nz = np.nonzero(v > 0)[0]
    if len(nz) == 0:
        return y, 0
    first = int(nz[0])
    if first == 0:
        return y, 0
    if len(v) - first < min_keep:      # too little real history to model; keep as is
        return y, 0
    return y.iloc[first:], first


def fit_one(sku, series, exog_all, horizon, test_months):
    y = series.astype(float)
    y.index = pd.DatetimeIndex(y.index, freq="MS")
    y, skipped = trim_to_launch(y)
    n = len(y)
    quad, adi, cv2 = classify(y)

    fut_index = pd.date_range(y.index.max() + pd.offsets.MonthBegin(1),
                              periods=horizon, freq="MS")

    # ---- choose candidate ----
    if quad in ("smooth", "erratic") and n >= MIN_MONTHS:
        candidate = "sarimax"
    elif quad in ("intermittent", "lumpy"):
        candidate = "croston"
    else:
        candidate = "naive"

    # ---- backtest on the holdout ----
    def predict(kind, train_y, steps, idx_for_exog):
        if kind == "sarimax":
            return f_sarimax(train_y, exog_all.loc[train_y.index], exog_all.loc[idx_for_exog])
        if kind == "croston":
            return f_croston(train_y, steps)
        if kind == "holt_winters":
            return f_holt_winters(train_y, steps)
        return f_naive(train_y, steps)

    w_cand = w_naive = w_hw = np.nan
    if test_months > 0 and n > test_months + 3:
        tr, te = y.iloc[:-test_months], y.iloc[-test_months:]
        for kind, slot in (("candidate", "c"), ("naive", "n"), ("hw", "h")):
            pass
        try:    w_cand = wmape(te, predict(candidate, tr, test_months, te.index))
        except Exception: w_cand = np.nan
        try:    w_naive = wmape(te, predict("naive", tr, test_months, te.index))
        except Exception: w_naive = np.nan
        if candidate == "sarimax":
            try: w_hw = wmape(te, predict("holt_winters", tr, test_months, te.index))
            except Exception: w_hw = np.nan

    method, note = candidate, ""
    if candidate == "sarimax" and np.isnan(w_cand):
        method, note, w_cand = "holt_winters", "sarimax failed -> Holt-Winters", w_hw
    if pd.notna(w_cand) and pd.notna(w_naive) and w_naive < w_cand:
        method = "naive"
        note = (note + "; " if note else "") + "naive beat candidate"

    # ---- final forecast on full history ----
    try:
        fc = predict(method, y, horizon, fut_index)
    except Exception:
        fc = f_naive(y, horizon); method += " -> naive"
    fc = np.clip(np.asarray(fc, float), 0, None)

    # Dormancy guard: a SKU with no sales for a long stretch is likely discontinued.
    # Croston keeps quoting an old rate forever because it only looks at non-zero
    # months, so decay the forecast toward zero instead of resurrecting the SKU.
    v_all = np.asarray(y, float)
    nz_idx = np.nonzero(v_all > 0)[0]
    dormant = (len(v_all) - 1 - int(nz_idx[-1])) if nz_idx.size else len(v_all)
    if dormant >= DORMANT_MONTHS:
        fc = fc * max(0.0, 1.0 - (dormant - DORMANT_MONTHS + 1) / 6.0)
        note = (note + "; " if note else "") + f"no sales for {dormant} months — decayed toward zero"

    # Sanity floor: if the SKU clearly sold in the recent past, a (near) zero
    # forecast is indefensible. Lift it to a conservative share of recent demand.
    lvl = recent_level(y)
    recent_nz = int((np.asarray(y, float)[-6:] > 0).sum())
    if lvl > 0 and recent_nz >= 2:
        floor = ZERO_FLOOR_SHARE * lvl
        if np.nanmax(fc) < floor:                 # whole forecast collapsed
            fc = np.maximum(fc, floor)
            note = (note + "; " if note else "") + "raised off zero (recent sales present)"
        else:
            fc = np.maximum(fc, floor * 0.25)     # stop individual months hitting 0

    fc = smart_round(fc)

    if skipped:
        note = (note + "; " if note else "") + f"new SKU: first sale after {skipped} mo, pre-launch months excluded"
    return dict(sku=sku, months=n, pattern=quad, adi=adi, cv2=cv2, method=method,
                wmape=w_cand, wmape_naive=w_naive, wmape_hw=w_hw, note=note,
                forecast=fc, fut_index=fut_index, skipped=skipped)


def run(mat, events=None, horizon=None, test_months=None):
    """Fit every SKU. Returns (results DataFrame, future month index)."""
    events = events if events is not None else DEFAULT_EVENTS
    horizon = horizon or HORIZON
    test_months = TEST_MONTHS if test_months is None else test_months

    full_index = pd.date_range(mat.index.min(),
                               mat.index.max() + pd.offsets.MonthBegin(horizon + 1),
                               freq="MS")
    exog_all = build_exog(full_index, events)

    rows = [fit_one(s, mat[s], exog_all, horizon, test_months) for s in mat.columns]
    fut_index = rows[0]["fut_index"]
    out = pd.DataFrame([{
        "SKU": r["sku"], "Months": r["months"], "Pre-launch skipped": r["skipped"],
        "Pattern": r["pattern"],
        "ADI": r["adi"], "CV2": r["cv2"], "Method": r["method"],
        "WMAPE": r["wmape"], "WMAPE_naive": r["wmape_naive"],
        "Beats_naive": ("—" if pd.isna(r["wmape"]) or pd.isna(r["wmape_naive"])
                        else ("YES" if r["wmape"] < r["wmape_naive"] else "no")),
        "Note": r["note"],
        **{fut_index[i].strftime("%b-%y"): r["forecast"][i] for i in range(horizon)},
    } for r in rows])
    return out, fut_index


# ============================ ROP / ROQ ============================
Z_TABLE = {80: 0.84, 85: 1.04, 90: 1.28, 95: 1.65, 98: 2.05, 99: 2.33}


def z_for(service_level):
    keys = sorted(Z_TABLE)
    nearest = min(keys, key=lambda k: abs(k - service_level))
    return Z_TABLE[nearest]


def compute_rop_roq(out, fut_index, lead_time_days=14, service_level=95,
                    days_cover=30, days_per_month=30, peak_aware=True):
    """Reorder point & quantity per SKU.

    ROP = demand during lead time + safety stock
          safety stock = z x forecast error (WMAPE) x lead-time demand
    ROQ = daily demand x days of cover

    peak_aware: also reports a peak-month ROP, because using the average of the
    forecast months understates a festival month where demand can double.
    """
    z = z_for(service_level)
    month_cols = [d.strftime("%b-%y") for d in fut_index]
    rows = []
    for _, r in out.iterrows():
        vals = pd.to_numeric(r[month_cols], errors="coerce").astype(float)
        avg_month = float(np.nanmean(vals)) if len(vals) else 0.0
        peak_month = float(np.nanmax(vals)) if len(vals) else 0.0
        err = float(r["WMAPE"]) if pd.notna(r["WMAPE"]) else 0.30  # cautious default
        daily = avg_month / days_per_month
        dlt = daily * lead_time_days
        ss = z * err * dlt
        rop = dlt + ss
        roq = daily * days_cover
        peak_daily = peak_month / days_per_month
        peak_rop = peak_daily * lead_time_days + z * err * peak_daily * lead_time_days
        def rr(x, big=avg_month):
            # keep decimals for low-volume SKUs; whole units otherwise
            if big < 10:  return round(float(x), 2)
            if big < 100: return round(float(x), 1)
            return round(float(x))
        rows.append({
            "SKU": r["SKU"], "Pattern": r["Pattern"], "Method": r["Method"],
            "Avg monthly demand": rr(avg_month),
            "Peak monthly demand": rr(peak_month),
            "Daily demand": round(daily, 3) if avg_month < 10 else round(daily, 1),
            "Lead-time demand": rr(dlt),
            "Forecast error": err,
            "Safety stock": rr(ss),
            "ROP": rr(rop),
            "ROQ": rr(roq),
            "Order-up-to": rr(rop + roq),
            "ROP at peak month": rr(peak_rop),
        })
    df = pd.DataFrame(rows)
    if not peak_aware:
        df = df.drop(columns=["Peak monthly demand", "ROP at peak month"])
    return df


# ============================ sample data ============================
def make_sample(n_sku=12, seed=6):
    """3 years of daily sales with a MOVING Ramadan (Apr-22, Apr-23, Mar-24)."""
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31")
    days = pd.date_range(start, end, freq="D")
    ram = {2022: ("2022-04-01", "2022-04-28"),
           2023: ("2023-04-01", "2023-04-28"),
           2024: ("2024-03-01", "2024-03-29")}
    ram = {k: (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in ram.items()}
    summer = {6: 1.18, 7: 1.28, 8: 1.18}
    kinds = ["smooth", "erratic", "intermittent", "lumpy"]
    recs = []
    all_months = sorted({(d.year, d.month) for d in days})
    for k in range(n_sku):
        kind = kinds[k % 4]
        base = rng.uniform(40, 140)
        if kind == "intermittent":
            active = {m for m in all_months if rng.random() < 0.45}
        elif kind == "lumpy":
            active = {m for m in all_months if rng.random() < 0.30}
        else:
            active = set(all_months)
        for d in days:
            if (d.year, d.month) not in active:
                continue
            v = base * (1 + 0.05 * ((d - start).days / 730)) * summer.get(d.month, 1.0)
            a, b = ram[d.year]
            if a <= d <= b:
                v *= 2.4
            if kind == "smooth":
                v = max(0, rng.normal(v, v * 0.08))
            elif kind == "erratic":
                v = max(0, rng.normal(v, v * 0.45))
            elif kind == "intermittent":
                v = rng.normal(v * 0.4, v * 0.2) if rng.random() < 0.30 else 0
            else:
                v = rng.normal(v * 3, v) if rng.random() < 0.15 else 0
            v = max(round(v / 30), 0)
            if v > 0:
                recs.append((d, f"SKU-{k:02d}-{kind[:4].upper()}", v))
    return pd.DataFrame(recs, columns=["date", "sku", "qty"])


# ============================ workbook export ============================
def write_workbook(out, rop_df, path):
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.utils.dataframe import dataframe_to_rows

    wb = openpyxl.Workbook()
    HEAD = Font(name="Arial", bold=True, color="FFFFFF")
    FILL = PatternFill("solid", fgColor="2C2668")
    thin = Side(style="thin", color="D9D9D9")
    BORD = Border(left=thin, right=thin, top=thin, bottom=thin)
    GRN = PatternFill("solid", fgColor="E2EFDA")

    def dump(ws, df, pct_cols=(), green_cols=()):
        for j, c in enumerate(df.columns, 1):
            x = ws.cell(1, j, str(c)); x.font = HEAD; x.fill = FILL
            x.alignment = Alignment(horizontal="center", wrap_text=True)
            ws.column_dimensions[get_column_letter(j)].width = max(10, min(20, len(str(c)) + 4))
        for i, row in enumerate(dataframe_to_rows(df, index=False, header=False), 2):
            for j, v in enumerate(row, 1):
                x = ws.cell(i, j, v); x.border = BORD; x.font = Font(name="Arial")
                col = df.columns[j - 1]
                if col in pct_cols: x.number_format = "0.0%"
                if col in green_cols: x.fill = GRN; x.font = Font(name="Arial", bold=True)
        ws.freeze_panes = "B2"

    ws1 = wb.active; ws1.title = "Forecast"
    dump(ws1, out, pct_cols=("WMAPE", "WMAPE_naive"))
    ws2 = wb.create_sheet("ROP-ROQ")
    dump(ws2, rop_df, pct_cols=("Forecast error",), green_cols=("ROP", "ROQ"))
    wb.save(path)


if __name__ == "__main__":
    long = make_sample()
    mat = to_monthly_matrix(long)
    out, fut = run(mat, DEFAULT_EVENTS, horizon=6, test_months=6)
    rop = compute_rop_roq(out, fut)
    write_workbook(out, rop, "SKU_Forecast_Results.xlsx")
    print(out[["SKU", "Pattern", "Method", "WMAPE", "WMAPE_naive", "Beats_naive"]].to_string(index=False))
    print()
    print(rop[["SKU", "Avg monthly demand", "Lead-time demand", "Safety stock", "ROP", "ROQ"]].to_string(index=False))
