"""
Multi-SKU Demand Forecaster — Web App
=====================================
Features
  • Moving seasonality  — editable festival calendar (Ramadan/Eid) in the UI
  • Train / Test split  — you choose how much history trains vs tests the model
  • ROP / ROQ           — reorder point & quantity for the DC
  • Guide tab           — every term on the screen explained

Run:    streamlit run app.py
Share:  streamlit run app.py --server.address 0.0.0.0 --server.port 8501
"""
import os, tempfile
import numpy as np
import pandas as pd
import streamlit as st
import sku_engine as engine

# ---------------- cached heavy steps ----------------
@st.cache_data(show_spinner=False)
def _read_upload(file_bytes, name):
    """Parse the uploaded file once. Cached on the file's bytes."""
    import io
    bio = io.BytesIO(file_bytes)
    return pd.read_csv(bio) if name.lower().endswith(".csv") else pd.read_excel(bio)


@st.cache_data(show_spinner=False)
def _monthly(long_df, mode):
    return engine.to_monthly_matrix(engine.apply_grouping(long_df, mode), "series")


@st.cache_data(show_spinner=False)
def _impact(long_df, mode):
    return engine.split_impact(long_df, mode)


@st.cache_data(show_spinner="Fitting models — this can take a minute on a large range…")
def _run_forecast(mat, events, horizon, test_months, lead_time, service, days_cover):
    out, fut = engine.run(mat, list(events), horizon=horizon, test_months=test_months)
    rop = engine.compute_rop_roq(out, fut, lead_time_days=lead_time,
                                 service_level=service, days_cover=days_cover)
    return out, fut, rop


st.set_page_config(page_title="Demand Forecaster", page_icon="🎯",
                   layout="wide", initial_sidebar_state="expanded")


# --------------------------- optional password ---------------------------
def check_password():
    try:
        pw = st.secrets["APP_PASSWORD"]
    except Exception:
        pw = None
    if not pw or st.session_state.get("authed"):
        return True
    entered = st.text_input("Enter access password", type="password")
    if entered == pw:
        st.session_state["authed"] = True
        return True
    if entered:
        st.error("Wrong password.")
    return False


if not check_password():
    st.stop()

# ----------------------------- theme -----------------------------
st.markdown("""
<style>
:root{
  --accent:#7c6cf5; --accent2:#22d3a6; --warn:#f0a63a; --bad:#f0614f;
  --panel:#171a23; --panel2:#1e2230; --line:#2b3040; --muted:#98a0b3;
}
.stApp { background: radial-gradient(1200px 600px at 15% -10%, #1d2233 0%, #0e1117 55%); }
section[data-testid="stSidebar"] { background:#141824; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color:#cfd4e3; }

/* hero */
.hero{ background:linear-gradient(120deg,#221c4d 0%,#1a2340 45%,#14213a 100%);
  border:1px solid #2f3350; border-radius:18px; padding:22px 26px; margin:2px 0 18px;
  box-shadow:0 10px 34px rgba(0,0,0,.35);}
.hero h1{ margin:0 0 6px; font-size:30px; font-weight:800; letter-spacing:-.02em; color:#f3f4fa;}
.hero p{ margin:0; color:#b9c0d4; font-size:15px; max-width:80ch;}
.hero .pill{ display:inline-block; font-size:11px; letter-spacing:.14em; text-transform:uppercase;
  color:#b3a9ff; background:rgba(124,108,245,.14); border:1px solid rgba(124,108,245,.35);
  padding:4px 11px; border-radius:20px; margin-bottom:12px;}
.hero .chips{ margin-top:14px; display:flex; gap:8px; flex-wrap:wrap;}
.hero .chip{ font-size:12px; color:#cdd3e6; background:rgba(255,255,255,.05);
  border:1px solid var(--line); padding:5px 12px; border-radius:8px;}

/* metric cards */
div[data-testid="stMetric"]{ background:var(--panel); border:1px solid var(--line);
  border-radius:14px; padding:14px 16px; box-shadow:0 2px 10px rgba(0,0,0,.25);}
div[data-testid="stMetric"] label p{ color:var(--muted)!important; font-size:12px!important;
  text-transform:uppercase; letter-spacing:.07em;}
div[data-testid="stMetricValue"]{ font-size:27px!important; font-weight:750!important; color:#f0f2f8!important;}

/* tabs */
.stTabs [data-baseweb="tab-list"]{ gap:6px; border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{ background:transparent; border-radius:10px 10px 0 0;
  padding:9px 18px; color:var(--muted); font-weight:600;}
.stTabs [aria-selected="true"]{ background:var(--panel); color:#fff!important;
  border:1px solid var(--line); border-bottom:2px solid var(--accent);}

/* buttons */
.stButton>button{ border-radius:11px; font-weight:700; border:1px solid #3b3f58;
  padding:.55rem 1.15rem; transition:.15s;}
.stButton>button[kind="primary"]{ background:linear-gradient(95deg,#7c6cf5,#5b8cf7); border:none;}
.stButton>button[kind="primary"]:hover{ filter:brightness(1.12); transform:translateY(-1px);}

/* section headings */
h3{ color:#e7eaf4!important; font-weight:700!important; letter-spacing:-.01em;}
.sect{ display:flex; align-items:center; gap:9px; margin:22px 0 8px;
  font-size:17px; font-weight:750; color:#e9ecf6;}
.sect .ic{ width:30px;height:30px;border-radius:9px; display:flex;align-items:center;
  justify-content:center; background:rgba(124,108,245,.16); border:1px solid rgba(124,108,245,.35);
  font-size:15px;}

/* dataframes */
div[data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:12px; overflow:hidden;}
/* alerts a bit softer */
div[data-testid="stNotification"]{ border-radius:12px; }
/* widget accents -> purple instead of default red */
:root{ --primary-color:#7c6cf5; }
div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]{
  background-color:#7c6cf5!important; border-color:#7c6cf5!important;}
div[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div:first-child{
  background:#7c6cf5!important;}
div[data-testid="stSlider"] [data-testid="stThumbValue"]{ color:#a99cff!important;}
div[data-testid="stSlider"] div[style*="rgb(255, 75, 75)"]{ background:#7c6cf5!important;}
[data-testid="stWidgetLabel"] + div div[style*="rgb(255, 75, 75)"]{ background:#7c6cf5!important;}
div[style*="background-color: rgb(255, 75, 75)"]{ background-color:#7c6cf5!important;}
div[style*="background: rgb(255, 75, 75)"]{ background:#7c6cf5!important;}
div[data-baseweb="slider"] div[role="slider"]{ background:#7c6cf5!important; border-color:#7c6cf5!important;}
div[data-baseweb="slider"] [data-testid="stSliderTickBar"]~div>div{ background:#7c6cf5!important;}
div[data-baseweb="slider"] div[data-testid="stTickBar"]{ color:var(--muted);}
.stSlider [data-baseweb="slider"] > div > div { background:#7c6cf5 !important; }
.stSlider [data-baseweb="slider"] [role="slider"] { background:#7c6cf5 !important; box-shadow:0 0 0 3px rgba(124,108,245,.25)!important;}
label[data-baseweb="radio"] div[aria-checked="true"]{ background:#7c6cf5!important; border-color:#7c6cf5!important;}
input[type="checkbox"]:checked, label[data-baseweb="checkbox"] span[aria-checked="true"]{
  background:#7c6cf5!important; border-color:#7c6cf5!important;}
/* readable body text */
.stApp, .stApp p, .stApp li, .stMarkdown p { color:#c3cade; }
.stCaption, div[data-testid="stCaptionContainer"] p { color:#8f97ab!important; }
section[data-testid="stSidebar"] label p { color:#c3cade!important; font-weight:600;}
/* inputs */
div[data-baseweb="select"]>div, div[data-baseweb="input"]>div{
  background:#1b1f2c!important; border-color:#333a4f!important; border-radius:10px!important;}
/* guide cards */
.gcard{ background:var(--panel); border:1px solid var(--line); border-left:3px solid var(--accent);
  border-radius:0 12px 12px 0; padding:14px 18px; margin:10px 0;}
.gcard h4{ margin:0 0 6px; color:#eef0f7; font-size:15px;}
.gcard p{ margin:0; color:#b6bdd0; font-size:13.5px; line-height:1.55;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <span class="pill">Demand Planning</span>
  <h1>🎯 Multi-SKU Demand Forecaster</h1>
  <p>Picks the right method for how each product actually sells, follows moving festivals like
     Ramadan into the correct month, proves itself on history it never saw, and turns the result
     into reorder points for the DC.</p>
  <div class="chips">
    <span class="chip">📅 Moving seasonality</span>
    <span class="chip">🧪 Backtested accuracy</span>
    <span class="chip">📦 ROP &amp; ROQ</span>
    <span class="chip">🌍 SKU or Country level</span>
  </div>
</div>
""", unsafe_allow_html=True)


_FLOW_SVG = '<svg viewBox="0 0 1000 152" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;margin:14px 0 4px;"><defs><marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#7c6cf5"/></marker><linearGradient id="g1" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#252a3d"/><stop offset="100%" stop-color="#1b1f2e"/></linearGradient></defs><g font-family="system-ui,Segoe UI,Arial"><rect x="6" y="34" width="170" height="74" rx="13" fill="url(#g1)" stroke="#3a3f5a"/><text x="91" y="60" font-size="19" text-anchor="middle">📄</text><text x="91" y="80" font-size="13" fill="#e8ebf5" text-anchor="middle" font-weight="700">1 · Your sales</text><text x="91" y="96" font-size="10.5" fill="#98a0b3" text-anchor="middle">date · SKU · qty</text><rect x="212" y="34" width="170" height="74" rx="13" fill="url(#g1)" stroke="#3a3f5a"/><text x="297" y="60" font-size="19" text-anchor="middle">🧭</text><text x="297" y="80" font-size="13" fill="#e8ebf5" text-anchor="middle" font-weight="700">2 · Classify</text><text x="297" y="96" font-size="10.5" fill="#98a0b3" text-anchor="middle">how often · how even</text><rect x="418" y="34" width="170" height="74" rx="13" fill="url(#g1)" stroke="#3a3f5a"/><text x="503" y="60" font-size="19" text-anchor="middle">🎛️</text><text x="503" y="80" font-size="13" fill="#e8ebf5" text-anchor="middle" font-weight="700">3 · Right method</text><text x="503" y="96" font-size="10.5" fill="#98a0b3" text-anchor="middle">+ festival calendar</text><rect x="624" y="34" width="170" height="74" rx="13" fill="url(#g1)" stroke="#3a3f5a"/><text x="709" y="60" font-size="19" text-anchor="middle">🧪</text><text x="709" y="80" font-size="13" fill="#e8ebf5" text-anchor="middle" font-weight="700">4 · Prove it</text><text x="709" y="96" font-size="10.5" fill="#98a0b3" text-anchor="middle">test on hidden months</text><rect x="830" y="34" width="164" height="74" rx="13" fill="#1b3a34" stroke="#22d3a6"/><text x="912" y="60" font-size="19" text-anchor="middle">📦</text><text x="912" y="80" font-size="13" fill="#c9f7e9" text-anchor="middle" font-weight="700">5 · Order plan</text><text x="912" y="96" font-size="10.5" fill="#7fd8bf" text-anchor="middle">ROP · ROQ</text><g stroke="#7c6cf5" stroke-width="2" marker-end="url(#ar)"><line x1="178" y1="71" x2="206" y2="71"/><line x1="384" y1="71" x2="412" y2="71"/><line x1="590" y1="71" x2="618" y2="71"/><line x1="796" y1="71" x2="824" y2="71"/></g><text x="709" y="130" font-size="10.5" fill="#f0a63a" text-anchor="middle">↻ if it loses to a simple guess, fall back</text></g></svg>'
_CAL_SVG = '<svg viewBox="0 0 1000 190" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;"><g font-family="system-ui,Segoe UI,Arial"><text x="8" y="34" font-size="12" fill="#cfd4e3" font-weight="700">Last year</text><text x="8" y="86" font-size="12" fill="#cfd4e3" font-weight="700">This year</text><text x="8" y="146" font-size="12" fill="#22d3a6" font-weight="700">Forecast</text><rect x="92" y="18" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="126" y="35" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Jan</text><rect x="166" y="18" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="200" y="35" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Feb</text><rect x="240" y="18" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="274" y="35" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Mar</text><rect x="314" y="18" width="68" height="24" rx="6" fill="#f0a63a" stroke="#343a4e"/><text x="348" y="35" font-size="10.5" fill="#1a1205" text-anchor="middle" font-weight="700">Apr★</text><rect x="388" y="18" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="422" y="35" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">May</text><rect x="462" y="18" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="496" y="35" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Jun</text><rect x="536" y="18" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="570" y="35" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Jul</text><rect x="610" y="18" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="644" y="35" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Aug</text><rect x="684" y="18" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="718" y="35" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Sep</text><rect x="758" y="18" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="792" y="35" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Oct</text><rect x="832" y="18" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="866" y="35" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Nov</text><rect x="906" y="18" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="940" y="35" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Dec</text><rect x="92" y="70" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="126" y="87" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Jan</text><rect x="166" y="70" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="200" y="87" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Feb</text><rect x="240" y="70" width="68" height="24" rx="6" fill="#f0a63a" stroke="#343a4e"/><text x="274" y="87" font-size="10.5" fill="#1a1205" text-anchor="middle" font-weight="700">Mar★</text><rect x="314" y="70" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="348" y="87" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Apr</text><rect x="388" y="70" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="422" y="87" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">May</text><rect x="462" y="70" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="496" y="87" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Jun</text><rect x="536" y="70" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="570" y="87" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Jul</text><rect x="610" y="70" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="644" y="87" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Aug</text><rect x="684" y="70" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="718" y="87" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Sep</text><rect x="758" y="70" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="792" y="87" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Oct</text><rect x="832" y="70" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="866" y="87" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Nov</text><rect x="906" y="70" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="940" y="87" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Dec</text><rect x="92" y="130" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="126" y="147" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Jan</text><rect x="166" y="130" width="68" height="24" rx="6" fill="#22d3a6" stroke="#343a4e"/><text x="200" y="147" font-size="10.5" fill="#05231b" text-anchor="middle" font-weight="700">Feb★</text><rect x="240" y="130" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="274" y="147" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Mar</text><rect x="314" y="130" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="348" y="147" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Apr</text><rect x="388" y="130" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="422" y="147" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">May</text><rect x="462" y="130" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="496" y="147" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Jun</text><rect x="536" y="130" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="570" y="147" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Jul</text><rect x="610" y="130" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="644" y="147" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Aug</text><rect x="684" y="130" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="718" y="147" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Sep</text><rect x="758" y="130" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="792" y="147" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Oct</text><rect x="832" y="130" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="866" y="147" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Nov</text><rect x="906" y="130" width="68" height="24" rx="6" fill="#232735" stroke="#343a4e"/><text x="940" y="147" font-size="10.5" fill="#8f97ab" text-anchor="middle" font-weight="400">Dec</text><path d="M330,46 C330,58 256,58 256,66" stroke="#f0a63a" stroke-width="1.6" fill="none" stroke-dasharray="4 3"/><path d="M256,98 C256,112 182,116 182,126" stroke="#22d3a6" stroke-width="1.6" fill="none" stroke-dasharray="4 3"/><text x="500" y="180" font-size="11" fill="#98a0b3" text-anchor="middle">★ = the festival peak. It slides ~11 days earlier each year — the model follows it.</text></g></svg>'
_ROP_SVG = '<svg viewBox="0 0 1000 210" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;"><g font-family="system-ui,Segoe UI,Arial"><rect x="70" y="150" width="880" height="34" fill="rgba(240,97,79,.13)"/><text x="80" y="171" font-size="11" fill="#f0937f">safety stock — the buffer</text><line x1="70" y1="118" x2="950" y2="118" stroke="#f0a63a" stroke-width="1.7" stroke-dasharray="7 5"/><text x="76" y="112" font-size="11.5" fill="#f0a63a" font-weight="700">REORDER POINT — order when stock touches this</text><polyline points="70,42 300,118 400,152 400,46 630,118 730,152 730,46 950,100" fill="none" stroke="#7c6cf5" stroke-width="3" stroke-linejoin="round"/><circle cx="300" cy="118" r="5.5" fill="#f0a63a"/><circle cx="630" cy="118" r="5.5" fill="#f0a63a"/><text x="300" y="105" font-size="10.5" fill="#f0a63a" text-anchor="middle">order placed</text><circle cx="400" cy="152" r="5.5" fill="#22d3a6"/><circle cx="730" cy="152" r="5.5" fill="#22d3a6"/><text x="412" y="200" font-size="10.5" fill="#22d3a6">stock arrives</text><line x1="300" y1="192" x2="400" y2="192" stroke="#6c7489" stroke-width="1.2"/><text x="350" y="186" font-size="10.5" fill="#8f97ab" text-anchor="middle">lead time</text><line x1="748" y1="46" x2="748" y2="152" stroke="#22d3a6" stroke-width="1.6"/><text x="758" y="96" font-size="11" fill="#22d3a6" font-weight="700">ROQ</text><text x="758" y="110" font-size="10" fill="#7fd8bf">how much to order</text><line x1="70" y1="184" x2="950" y2="184" stroke="#3a4054" stroke-width="1.4"/><text x="22" y="46" font-size="11" fill="#8f97ab">stock</text></g></svg>'

tab_run, tab_guide = st.tabs(["  ▶  Forecast  ", "  📘  Guide  "])


# ============================== GUIDE TAB ==============================
with tab_guide:
    st.markdown("""
<div style="background:#171a23;border:1px solid #2b3040;border-radius:16px;padding:20px 24px;margin-bottom:6px;">
  <h2 style="margin:0 0 4px;color:#f2f4fa;font-size:23px;">How this works</h2>
  <p style="margin:0;color:#a8b0c5;font-size:14px;">The whole process in five steps — then every term on the screen, explained.</p>
</div>
""", unsafe_allow_html=True)

    # ---------------- visual process flow ----------------
    st.markdown(_FLOW_SVG, unsafe_allow_html=True)

    # ---------------- moving festival visual ----------------
    st.markdown('<div class="sect"><span class="ic">📅</span>Why the festival calendar matters</div>',
                unsafe_allow_html=True)
    st.markdown(_CAL_SVG, unsafe_allow_html=True)
    st.warning("**Enter next year's festival dates.** The model can only place next year's peak "
               "if you tell it when the festival is. Missing future dates is the #1 cause of a bad forecast.")

    # ---------------- demand patterns visual ----------------
    st.markdown('<div class="sect"><span class="ic">🧭</span>The four demand patterns</div>',
                unsafe_allow_html=True)
    pats = [("Smooth","sells most months, steady","#22d3a6",[6,5,6,6,5,6,6,5],"SARIMAX"),
            ("Erratic","regular but jumpy sizes","#7c6cf5",[2,8,3,9,4,7,3,8],"SARIMAX"),
            ("Intermittent","many empty months","#f0a63a",[0,6,0,0,7,0,0,5],"Croston"),
            ("Lumpy","rare AND jumpy","#f0614f",[0,9,0,0,0,2,0,0],"Croston")]
    cols = st.columns(4)
    for col,(nm,desc,c,vals,meth) in zip(cols,pats):
        bars = "".join(
            f'<rect x="{6+i*17}" y="{54-(v/9*44)}" width="12" height="{max(v/9*44,2)}" rx="2" fill="{c if v else "#2a2f3e"}"/>'
            for i,v in enumerate(vals))
        col.markdown(f"""
<div style="background:#171a23;border:1px solid #2b3040;border-top:3px solid {c};border-radius:12px;padding:12px;">
  <div style="font-weight:750;color:#eef0f7;font-size:14px;">{nm}</div>
  <div style="color:#98a0b3;font-size:11.5px;margin-bottom:6px;">{desc}</div>
  <svg viewBox="0 0 145 60" style="width:100%;height:56px;">{bars}</svg>
  <div style="font-size:11px;color:{c};font-weight:700;margin-top:4px;">→ {meth}</div>
</div>""", unsafe_allow_html=True)

    # ---------------- ROP visual ----------------
    st.markdown('<div class="sect"><span class="ic">📦</span>How ROP and ROQ work</div>',
                unsafe_allow_html=True)
    st.markdown(_ROP_SVG, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sect"><span class="ic">📖</span>Every term, explained</div>',
                unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("""
<div class="gcard"><h4>Forecast horizon</h4><p>How many months into the <b>future</b> to predict.</p></div>
<div class="gcard"><h4>Test months (holdout)</h4><p>Recent months <b>hidden</b> from the model so it can be scored fairly. The rest trains it. 36 months with Test = 12 → trains on 24, tested on 12. Set to 0 once you trust it, to forecast using every month.</p></div>
<div class="gcard"><h4>ADI &amp; CV² cutoffs</h4><p>ADI = how often it sells. CV² = how uneven the sizes are. Together they decide the demand pattern.</p></div>
<div class="gcard"><h4>WMAPE</h4><p>Average error on the hidden months, as a % of total sales. <b>Lower is better</b> — under 20% is good for a regular seller.</p></div>
<div class="gcard"><h4>Beats naive</h4><p><b>YES</b> = the model earned its place. <b>no</b> = the simple "same month last year" guess did better, so we fall back to it.</p></div>
""", unsafe_allow_html=True)
    with g2:
        st.markdown("""
<div class="gcard"><h4>Lead time</h4><p>Days from placing an order to it arriving, door to door.</p></div>
<div class="gcard"><h4>Service level</h4><p>How often you want to avoid a stockout. 95% is standard; higher means more safety stock.</p></div>
<div class="gcard"><h4>Days of cover</h4><p>How long one order should last. 30 ≈ a month's stock per order.</p></div>
<div class="gcard"><h4>Safety stock</h4><p>The buffer, sized by <b>each product's own forecast error</b> — accurate products need less stock for the same protection.</p></div>
<div class="gcard"><h4>ROP at peak month</h4><p>The reorder point recalculated on the busiest forecast month. <b>Use this going into a festival</b> — the normal ROP uses the average and would under-stock you.</p></div>
""", unsafe_allow_html=True)

    st.info("⚠️ **Stop-start products score badly on purpose.** Intermittent and lumpy SKUs often "
            "show errors above 100% — that is normal for demand that arrives in rare bursts. Judge "
            "them on the total over a quarter, not month by month. A flat forecast there is the correct answer.")


# ============================== RUN TAB ==============================
with tab_run:
    # ---------------- sidebar ----------------
    with st.sidebar:
        st.header("⚙️ Settings")
        horizon = st.slider("Forecast horizon (months ahead)", 1, 18, 6,
                            help="How far into the future to predict.")
        test_months = st.slider("Test months (holdout)", 0, 18, 6,
                                help="Recent months hidden from the model so it can be scored. "
                                     "0 = no test, use all data to forecast.")
        season = st.selectbox("Season length (months)", [12, 4], index=0)

        with st.expander("Advanced — pattern cutoffs"):
            adi_cut = st.number_input("ADI cutoff", 1.0, 3.0, 1.32, 0.01)
            cv2_cut = st.number_input("CV² cutoff", 0.1, 2.0, 0.49, 0.01)
            min_months = st.number_input("Min months for advanced model", 13, 48, 24, 1)

        st.markdown("---")
        st.subheader("📦 Stock settings")
        lead_time = st.number_input("Lead time (days)", 1, 180, 14)
        service = st.select_slider("Service level %", [80, 85, 90, 95, 98, 99], value=95)
        days_cover = st.number_input("Days of cover (ROQ)", 7, 180, 30)

        st.markdown("---")
        source = st.radio("Data source", ["Upload my file", "Use sample data"])

    # push settings into engine
    engine.SEASON, engine.ADI_CUT, engine.CV2_CUT, engine.MIN_MONTHS = \
        season, adi_cut, cv2_cut, min_months

    # ---------------- festival calendar ----------------
    st.markdown('<div class="sect"><span class="ic">📅</span>Festival calendar — moving seasonality</div>', unsafe_allow_html=True)
    st.caption("Edit dates, add rows for new festivals, and **include next year's dates** so the "
               "peak is placed in the right future month. Delete a row by selecting it and pressing Delete.")
    if "events_df" not in st.session_state:
        st.session_state["events_df"] = pd.DataFrame(
            [{"event": n, "start": pd.Timestamp(s), "end": pd.Timestamp(e)}
             for n, s, e in engine.DEFAULT_EVENTS])
    with st.expander("📅 Festival dates — click to edit", expanded=False):
        events_df = st.data_editor(
            st.session_state["events_df"], num_rows="dynamic", use_container_width=True,
            column_config={
                "event": st.column_config.TextColumn("Festival", help="e.g. ramadan, eid_fitr, national_day"),
                "start": st.column_config.DateColumn("Starts"),
                "end": st.column_config.DateColumn("Ends"),
            }, key="events_editor")

    def events_as_tuples(df):
        out = []
        for _, r in df.iterrows():
            if pd.isna(r.get("event")) or pd.isna(r.get("start")) or pd.isna(r.get("end")):
                continue
            out.append((str(r["event"]).strip(),
                        pd.Timestamp(r["start"]).strftime("%Y-%m-%d"),
                        pd.Timestamp(r["end"]).strftime("%Y-%m-%d")))
        return out

    # ---------------- data intake ----------------
    st.markdown('<div class="sect"><span class="ic">📄</span>Sales data</div>', unsafe_allow_html=True)
    long_df = None

    def guess(cols, keys):
        for k in keys:
            for c in cols:
                if k in str(c).lower():
                    return c
        return cols[0]

    if source == "Upload my file":
        up = st.file_uploader("Daily sales — one row per date · SKU · quantity",
                              type=["csv", "xlsx"])
        if up is not None:
            raw = _read_upload(up.getvalue(), up.name)
            st.dataframe(raw.head(), use_container_width=True)
            cols = list(raw.columns)
            c1, c2, c3 = st.columns(3)
            dcol = c1.selectbox("Date column", cols, index=cols.index(guess(cols, ["date", "day", "doc"])))
            scol = c2.selectbox("SKU column", cols, index=cols.index(guess(cols, ["sku", "item", "part", "material", "code"])))
            qcol = c3.selectbox("Qty column", cols, index=cols.index(guess(cols, ["qty", "sales", "quantity", "demand", "units"])))
            rep_options = ["(none)"] + cols
            rguess = guess(cols, ["country", "nation", "market", "region", "territory"])
            rcol = st.selectbox("Country column (optional)", rep_options,
                                index=rep_options.index(rguess) if rguess in cols else 0,
                                help="Needed only if you want to forecast per country.")
            picked = [dcol, scol, qcol] + ([rcol] if rcol != "(none)" else [])
            if len(set(picked)) != len(picked):
                st.error("The same column is selected for more than one role. "
                         "Pick a different column for each of Date / SKU / Qty"
                         + (" / Sales ID." if rcol != "(none)" else "."))
                st.stop()
            # build by position, not by rename — avoids collisions when the file
            # already contains columns literally named date/sku/qty/rep
            t = pd.DataFrame({
                "date": raw[dcol],
                "sku":  raw[scol].astype(str).str.strip(),
                "qty":  raw[qcol],
            })
            if rcol != "(none)":
                t["rep"] = raw[rcol].astype(str).str.strip()
            t["date"] = pd.to_datetime(t["date"], errors="coerce")
            t["qty"] = pd.to_numeric(t["qty"], errors="coerce").fillna(0)
            bad_dates = int(t["date"].isna().sum())
            t = t.dropna(subset=["date"])
            t = t[t["sku"].notna() & (t["sku"] != "") & (t["sku"].str.lower() != "nan")]
            if bad_dates:
                st.warning(f"{bad_dates:,} rows had an unreadable date and were skipped. "
                           "If that is most of your file, the date column is stored as text — "
                           "fix with Data ▸ Text to Columns ▸ Date, or pick a different column.")
            if t.empty:
                st.error("No usable rows after reading. Check the Date / SKU / Qty selections above.")
                st.stop()
            long_df = t
    else:
        st.info("Sample data: 12 SKUs, 3 years, with Ramadan moving Apr-22 → Apr-23 → Mar-24.")
        long_df = engine.make_sample()

    # ---------------- forecast level (SKU vs salesperson) ----------------
    group_mode = "sku"
    if long_df is not None and "rep" in long_df.columns:
        st.markdown('<div class="sect"><span class="ic">🌍</span>Forecast level</div>', unsafe_allow_html=True)
        by_rep = st.checkbox("Forecast SKU × Country",
                             value=False,
                             help="Off = one forecast per SKU (recommended for stock planning). "
                                  "On = a separate forecast for every SKU/country combination.")
        if by_rep:
            group_mode = "sku_rep"
            st.caption("Forecasts each SKU separately for each country. "
                       "Results show SKU and Country in two columns.")
            imp = _impact(long_df, group_mode)
            i1, i2, i3 = st.columns(3)
            i1.metric("Series at SKU level", imp["series_sku"])
            i2.metric("Series after split", imp.get("series_split", "—"),
                      delta=f"+{imp.get('series_split',0)-imp['series_sku']}")
            base_pct = imp["intermittent_sku"] / max(imp["series_sku"], 1)
            split_pct = imp.get("intermittent_split", 0) / max(imp.get("series_split", 1), 1)
            i3.metric("Sparse series after split", f"{split_pct:.0%}",
                      delta=f"{(split_pct-base_pct)*100:+.0f} pts vs SKU level",
                      delta_color="inverse",
                      help="Intermittent/lumpy series. More sparse = harder to forecast.")
            st.warning(
                "**What splitting does to the numbers.** Demand is divided across countries, so "
                "each series is smaller and spikier. Expect: more series classified intermittent "
                "(flat Croston forecasts), higher error (WMAPE), and more SKUs losing to naive. "
                "The split forecasts also will **not** add up to the SKU-level forecast — and the "
                "SKU-level one is usually the more accurate.\n\n"
                "**Use the split for country planning. Use SKU level for DC stock (ROP/ROQ) if one DC "
                "serves several countries — otherwise the pooled stock will be under-sized.**")
            st.caption("Tip: if a country is newly opened or paused, its series starts or stops abruptly "
                       "and the model reads that as a new product or a demand collapse.")

    # ---------------- split preview ----------------
    if long_df is not None and len(long_df):
        mat_preview = _monthly(long_df, group_mode)
        n_months = len(mat_preview)
        train_months = n_months - test_months
        c1, c2, c3 = st.columns(3)
        c1.metric("Months of data", n_months)
        c2.metric("Train on", f"{train_months} months")
        c3.metric("Test on", f"{test_months} months" if test_months else "no test")
        if test_months and train_months < engine.SEASON + 3:
            st.warning(f"Only {train_months} months left to train on. Reduce Test months, "
                       "or the model will fall back to the simple method.")
        if test_months == 0:
            st.info("Test months = 0 → no accuracy check this run. Use this once you trust the model.")

    run_btn = st.button("🚀  Run forecast", type="primary", disabled=(long_df is None), use_container_width=True)

    if run_btn and long_df is not None:
        evs = events_as_tuples(events_df)
        if not evs:
            st.warning("No festivals entered — the model will still run, but moving-festival "
                       "peaks will not be placed correctly.")
        mat = _monthly(long_df, group_mode)
        out, fut, rop = _run_forecast(mat, tuple(evs), horizon, test_months,
                                      lead_time, service, days_cover)
        st.session_state.update(out=out, mat=mat, fut=fut, rop=rop, level=group_mode)

    # ---------------- results ----------------
    if "out" in st.session_state:
        out, mat, fut, rop = (st.session_state[k] for k in ("out", "mat", "fut", "rop"))
        st.success(f"Done — {len(out)} SKUs forecast.")

        reg = out[out["Pattern"].isin(["smooth", "erratic"])]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("SKUs", len(out))
        m2.metric("Regular sellers", len(reg))
        if reg["WMAPE"].notna().any():
            m3.metric("Median error (regular)", f"{reg['WMAPE'].median():.0%}",
                      help="WMAPE on the hidden test months. Lower is better.")
        m4.metric("Beat the naive guess", int((out["Beats_naive"] == "YES").sum()))

        c1, c2 = st.columns(2)
        c1.markdown("**Demand patterns**"); c1.bar_chart(out["Pattern"].value_counts())
        c2.markdown("**Methods used**");    c2.bar_chart(out["Method"].value_counts())

        st.markdown('<div class="sect"><span class="ic">🔍</span>Inspect one SKU</div>', unsafe_allow_html=True)
        pick = st.selectbox("SKU", out["SKU"].tolist())
        row = out[out["SKU"] == pick].iloc[0]
        mcols = [d.strftime("%b-%y") for d in fut]
        hist = mat[pick]
        chart = pd.DataFrame(index=list(mat.index) + list(fut))
        chart["Actual"] = list(hist.values) + [None] * len(mcols)
        chart["Forecast"] = [None] * len(hist) + [row[c] for c in mcols]
        st.line_chart(chart)
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Pattern", row["Pattern"])
        d2.metric("Method", row["Method"])
        d3.metric("Error (WMAPE)", f"{row['WMAPE']:.0%}" if pd.notna(row["WMAPE"]) else "no test")
        d4.metric("Beats naive", row["Beats_naive"])
        if row["Note"]:
            st.caption(f"Note: {row['Note']}")
        if row["Pattern"] in ("intermittent", "lumpy"):
            st.caption("This is a stop-start seller — a flat forecast is the correct answer here, "
                       "and a high error % is normal.")

        lvl = st.session_state.get("level", "sku")
        st.markdown("### " + ("All SKU × Country — forecast" if lvl == "sku_rep"
                              else "All SKUs — forecast"))
        out_disp = engine.split_series_key(out, lvl)
        st.dataframe(out_disp.style.format({"ADI": "{:.2f}", "CV2": "{:.2f}",
                                            "WMAPE": "{:.0%}", "WMAPE_naive": "{:.0%}"}),
                     use_container_width=True, height=320)

        st.markdown('<div class="sect"><span class="ic">📦</span>Reorder points &amp; quantities</div>', unsafe_allow_html=True)
        if st.session_state.get("level", "sku") != "sku":
            st.error("⚠️ These ROP/ROQ figures are **per country**, not per SKU. If one DC serves "
                     "several countries, using split figures will under-size the pooled stock. "
                     "Turn off 'Forecast SKU × Country' and re-run before setting DC stock levels.")
        st.caption(f"Lead time {lead_time} days · service {service}% · {days_cover} days of cover. "
                   "Use **ROP at peak month** going into a festival.")
        rop_disp = engine.split_series_key(rop, lvl)
        st.dataframe(rop_disp.style.format({"Forecast error": "{:.0%}"}),
                     use_container_width=True, height=300)

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as t:
            path = t.name
        engine.write_workbook(out_disp, rop_disp, path)
        with open(path, "rb") as f:
            data = f.read()
        os.unlink(path)
        st.download_button("⬇ Download Excel workbook (forecast + ROP/ROQ)", data,
                           file_name="SKU_Forecast_Results.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
