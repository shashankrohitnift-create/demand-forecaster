"""
Diagnostic — why is one SKU forecasting zero?
=============================================
Edit the two lines below, then run:   python check.py
"""
import pandas as pd
import numpy as np
import sku_engine as E

# ----------------- EDIT THESE TWO -----------------
FILE = "TEMPLATE_sales_data_upload.xlsx"        # your uploaded sales file
SKU  = "HSMGFPT2216X10"              # the SKU showing zero
# --------------------------------------------------

print("engine file :", E.__file__)
print("has smart_round :", hasattr(E, "smart_round"), " (must be True — if False you are running the OLD engine)")
print("has trim_to_launch :", hasattr(E, "trim_to_launch"))
print("-" * 70)

df = E.load_long(FILE)
print(f"file loaded: {len(df):,} rows, {df['sku'].nunique()} SKUs, "
      f"{df['date'].min().date()} -> {df['date'].max().date()}")

sub = df[df["sku"].astype(str).str.strip() == SKU]
print(f"\nrows for {SKU}: {len(sub)}")
if len(sub) == 0:
    print("  !! No rows matched. Check for spelling / trailing spaces.")
    print("  Similar SKU codes in the file:")
    for s in df["sku"].astype(str).unique():
        if SKU[:8].lower() in s.lower():
            print("   ", repr(s))
    raise SystemExit

print(f"  qty total   : {sub['qty'].sum()}")
print(f"  qty min/max : {sub['qty'].min()} / {sub['qty'].max()}")
print(f"  date range  : {sub['date'].min().date()} -> {sub['date'].max().date()}")

mat = E.to_monthly_matrix(df)
s = mat[SKU]
print(f"\nmonthly series ({len(s)} months):")
for d, v in s.items():
    print(f"   {d:%b-%Y}: {v:>8.3f}")

y2, skipped = E.trim_to_launch(s.copy())
print(f"\ntrim_to_launch -> skipped {skipped} pre-launch months, kept {len(y2)}")
q, adi, cv2 = E.classify(y2)
print(f"classify -> pattern={q}  ADI={adi:.2f}  CV2={cv2:.2f}")

out, fut = E.run(mat, E.DEFAULT_EVENTS, horizon=6, test_months=3)
r = out[out["SKU"] == SKU].iloc[0]
print("\nRESULT")
print("  months used     :", r["Months"])
print("  pre-launch skip :", r["Pre-launch skipped"])
print("  pattern         :", r["Pattern"])
print("  method          :", r["Method"])
print("  WMAPE           :", r["WMAPE"])
print("  note            :", r["Note"])
print("  FORECAST        :", [r[d.strftime("%b-%y")] for d in fut])

rop = E.compute_rop_roq(out, fut)
print("\nROP row:")
print(rop[rop["SKU"] == SKU].to_string(index=False))
