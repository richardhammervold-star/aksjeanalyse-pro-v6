# app_v6.py
# 📈 Aksjeanalyse – Pro v6 Dashboard
# Fleksible horisonter (A/B/C), uavhengige eps-filtre og robuste Close-hentere
# Build: v6.3 – oktober 2025

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from pandas.tseries.offsets import BusinessDay
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

# -----------------------------
# Sideoppsett + enkel styling
# -----------------------------
st.set_page_config(
    page_title="Aksjeanalyse – Pro v6 Dashboard",
    layout="wide",
)

st.title("📈 Aksjeanalyse – Pro v6 Dashboard")
st.caption("Fleksible horisonter (A/B/C), uavhengige eps-filtre og robuste Close-hentere • Build: v6.3 – oktober 2025")

# -----------------------------
# Presets (kan utvides)
# -----------------------------
PRESETS = {
    "OBX (Norge)": [
        "EQNR.OL","DNB.OL","MOWI.OL","NHY.OL","TEL.OL","ORK.OL","YAR.OL","KOG.OL","AKRBP.OL","TGS.OL"
    ],
    "USA – Megacaps": [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","XOM"
    ],
}

# -----------------------------
# Sidebar – kontroller
# -----------------------------
st.sidebar.header("⚙️ Innstillinger")

preset = st.sidebar.selectbox("Hurtigvalg gruppe", list(PRESETS.keys()))
if st.sidebar.button("📥 Last inn preset"):
    st.session_state["tickers"] = PRESETS[preset]

default_tickers = ["AAPL","MSFT","NVDA","AMZN","EQNR.OL"]
tickers = st.session_state.get("tickers", default_tickers)

user_tickers = st.sidebar.text_area(
    "Tickere (komma/linjer). Tomt felt = bruk valgt preset.",
    value=", ".join(tickers), height=140
)
tickers = [t.strip().upper() for chunk in user_tickers.split("\n") for t in chunk.split(",") if t.strip()]

col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    start_date = st.date_input("Startdato", pd.to_datetime("2019-01-01"))
with col_d2:
    end_date = st.date_input("Sluttdato", pd.Timestamp.today())

st.sidebar.markdown("---")
st.sidebar.subheader("Horisont (handelsdager)")
col_hA, col_hB, col_hC = st.sidebar.columns(3)
with col_hA:
    H_A = st.number_input("A (vises som '3 dager')", min_value=1, max_value=60, value=3, step=1)
with col_hB:
    H_B = st.number_input("B (vises som '7 dager')", min_value=1, max_value=120, value=7, step=1)
with col_hC:
    H_C = st.number_input("C (vises som '14 dager')", min_value=1, max_value=252, value=14, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Støyfilter (eps, %) – per horisont")
col_eA, col_eB, col_eC = st.sidebar.columns(3)
with col_eA:
    epsA = st.number_input("EPS A (%)", min_value=0.0, max_value=50.0, value=1.0, step=0.10, format="%.2f")
with col_eB:
    epsB = st.number_input("EPS B (%)", min_value=0.0, max_value=50.0, value=3.0, step=0.10, format="%.2f")
with col_eC:
    epsC = st.number_input("EPS C (%)", min_value=0.0, max_value=50.0, value=5.0, step=0.10, format="%.2f")

st.sidebar.markdown("---")
st.sidebar.subheader("Terskler for anbefaling")
same_thr = st.sidebar.checkbox("Bruk samme terskler på alle horisonter", value=True)

def slider_pair(label_buy, label_sell, buy_default=0.60, sell_default=0.40):
    b = st.sidebar.slider(label_buy, 0.50, 0.90, buy_default, 0.01)
    s = st.sidebar.slider(label_sell, 0.10, 0.50, sell_default, 0.01)
    return b, s

if same_thr:
    b_all, s_all = slider_pair("KJØP hvis sannsynlighet >", "SELG hvis sannsynlighet <", 0.60, 0.40)
    THR = {"A": (b_all, s_all), "B": (b_all, s_all), "C": (b_all, s_all)}
else:
    bA, sA = slider_pair("A • KJØP >", "A • SELG <", 0.60, 0.40)
    bB, sB = slider_pair("B • KJØP >", "B • SELG <", 0.60, 0.40)
    bC, sC = slider_pair("C • KJØP >", "C • SELG <", 0.60, 0.40)
    THR = {"A": (bA, sA), "B": (bB, sB), "C": (bC, sC)}

st.sidebar.markdown("---")
want_excel = st.sidebar.checkbox("Eksporter Excel (flere ark)", value=False)
want_csv   = st.sidebar.checkbox("Eksporter CSV (samlet)", value=True)

# Kart over horisonter og eps
HORIZONS = {"A": int(H_A), "B": int(H_B), "C": int(H_C)}
EPS_MAP  = {"A": float(epsA), "B": float(epsB), "C": float(epsC)}

st.sidebar.caption("Tips: Norske aksjer bruker .OL (EQNR.OL). Valuta = '=X' (EURNOK=X). Råvarer: CL=F, GC=F.")

# -----------------------------
# Datahjelpere (med caching)
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_history(ticker: str, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    return df

# -----------------------------
# ROBUST henter av Close → 1D Series
# -----------------------------
def _get_close_series(df: pd.DataFrame) -> pd.Series:
    """
    Returner en 1D 'Close'-Series uansett om df har enkel kolonne,
    DataFrame med én 'Close', eller MultiIndex-kolonner.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    close_obj = None

    # 1) Direkte kolonne 'Close'
    if "Close" in df.columns:
        close_obj = df["Close"]

    # 2) MultiIndex: prøv å hente nivå 'Close'
    if close_obj is None and isinstance(df.columns, pd.MultiIndex):
        try:
            # behold DataFrame; velg første kolonne under
            close_obj = df.xs("Close", axis=1, level=0, drop_level=False)
        except Exception:
            pass

    # 3) Andre navn/tupler som ligner på 'Close'
    if close_obj is None:
        candidates = [c for c in df.columns
                      if (isinstance(c, tuple) and str(c[0]).lower() == "close")
                      or (isinstance(c, str) and c.lower() == "close")]
        if candidates:
            close_obj = df[candidates[0]]

    # Konverter til 1D Series
    if isinstance(close_obj, pd.DataFrame):
        if close_obj.shape[1] >= 1:
            close_series = close_obj.iloc[:, 0]
        else:
            return pd.Series(dtype=float)
    elif isinstance(close_obj, pd.Series):
        close_series = close_obj
    else:
        return pd.Series(dtype=float)

    return pd.to_numeric(close_series, errors="coerce").astype(float)

# -----------------------------
# Indikatorer (bruker robust Close)
# -----------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Enkel felles feature-pipe."""
    out = df.copy()
    close = _get_close_series(df)

    if close.empty:
        # Retur bare indexen for konsistens
        return pd.DataFrame(index=df.index).assign(Close=np.nan)

    ret1 = close.pct_change()
    out["ret1"] = ret1
    out["ret3"] = close.pct_change(3)
    out["ret5"] = close.pct_change(5)
    out["ma5"] = close.rolling(5).mean()
    out["ma20"] = close.rolling(20).mean()
    out["vol10"] = ret1.rolling(10).std()
    out["trend20"] = (out["ma20"] - out["ma5"]) / out["ma20"]

    # RSI(14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    out["rsi14"] = 100 - (100 / (1 + rs))

    # EMA, MACD, Bollinger%
    out["ema20"] = close.ewm(span=20, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()
    out["ema_gap"] = (out["ema20"] - out["ema50"]) / out["ema50"]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd
    out["macd_sig"] = signal
    out["macd_hist"] = macd - signal

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    out["bb_pct"] = (close - lower) / (upper - lower)
    out["bb_width"] = (upper - lower) / ma20

    return out

# -----------------------------
# Label (robust mot Close-format)
# -----------------------------
def make_label_eps(df: pd.DataFrame, horizon_days: int, eps_pct: float) -> pd.Series:
    """
    Lag 1/0/NaN label basert på prisendring over horizon_days og støyfilter eps_pct (%).
    Robust mot ulike Close-kolonneformat fra yfinance.
    """
    close = _get_close_series(df)
    if close.empty:
        return pd.Series(dtype=float, name=f"Target_{horizon_days}")

    fwd = close.shift(-horizon_days) / close - 1.0
    eps = float(eps_pct) / 100.0

    arr = np.where(fwd > eps, 1,
          np.where(fwd < -eps, 0, np.nan)).astype("float64")

    return pd.Series(arr, index=close.index, name=f"Target_{horizon_days}")

# -----------------------------
# Modellering per horisont
# -----------------------------
FEATURES_ALL = [
    "ret1","ret3","ret5","ma5","ma20","vol10","trend20","rsi14",
    "ema20","ema50","ema_gap","bb_pct","bb_width","macd","macd_sig","macd_hist"
]

def fit_predict_single_horizon(df_raw: pd.DataFrame, horizon_days: int, eps: float):
    """
    Bygger features, lager label med eps-filter, trener modell,
    returnerer proba-serie (hele tidsrekken), validerings-metrics og siste indeks.
    """
    feats = add_indicators(df_raw)
    y = make_label_eps(feats, horizon_days, eps)

    # tilgjengelige features
    avail = [c for c in FEATURES_ALL if c in feats.columns and feats[c].notna().any()]
    if not avail or y.name not in feats.columns.union([y.name]):
        # smal fallback
        X = feats[avail] if avail else pd.DataFrame(index=feats.index)
    else:
        X = feats[avail]

    pack = pd.concat([X, y], axis=1).dropna()
    if pack.empty or len(pack) < 80:
        # nøytral fallback
        proba = pd.Series(0.5, index=(pack.index if len(pack) else feats.index), name="proba")
        return proba, np.nan, np.nan, 0.5, (pack.index[-1] if len(pack) else feats.index[-1] if len(feats) else None)

    Xp = pack[avail]
    yp = pack[y.name].astype(int)

    # Minst 2 klasser
    if len(np.unique(yp)) < 2:
        proba = pd.Series(0.5, index=pack.index, name="proba")
        return proba, np.nan, np.nan, 0.5, pack.index[-1]

    # Enkel walk-forward-ish: del i 70/30 tidlig/val
    n = len(pack)
    split = int(n * 0.70)
    Xtr, Xva = Xp.iloc[:split], Xp.iloc[split:]
    ytr, yva = yp.iloc[:split], yp.iloc[split:]

    scaler = RobustScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xall_s = scaler.transform(Xp)

    base = GradientBoostingClassifier(random_state=0)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(Xtr_s, ytr)

    p_val = clf.predict_proba(Xva_s)[:, 1] if len(Xva) else np.array([])
    p_all = clf.predict_proba(Xall_s)[:, 1]
    proba_full = pd.Series(p_all, index=Xp.index, name="proba")

    # Finn opt_thr som max accuracy på val
    opt_thr = 0.5
    acc = auc = np.nan
    if len(Xva):
        cand = np.linspace(0.3, 0.7, 41)
        accs = [accuracy_score(yva, (p_val >= t).astype(int)) for t in cand]
        opt_thr = float(cand[int(np.argmax(accs))])
        acc = float(np.max(accs))
        try:
            auc = float(roc_auc_score(yva, p_val))
        except Exception:
            auc = np.nan

    last_idx = pack.index[-1]
    return proba_full, acc, auc, opt_thr, last_idx

def analyze_ticker_multi(df_raw: pd.DataFrame, horizons: dict, eps_map: dict) -> dict:
    """
    Tren og prediker for A/B/C, returner dict med nøkler 'A','B','C'
    og felter: proba (Series), acc, auc, opt_thr, last_idx
    """
    out = {}
    for key in ["A", "B", "C"]:
        H = int(horizons[key])
        eps = float(eps_map[key])
        proba, acc, auc, opt_thr, last_idx = fit_predict_single_horizon(df_raw, H, eps)
        out[key] = {
            "proba": proba,
            "acc": acc,
            "auc": auc,
            "opt_thr": opt_thr,
            "last_idx": last_idx,   # siste treningsindeks
            "H": H,
            "eps": eps
        }
    return out

# -----------------------------
# Utils
# -----------------------------
def rec_from_prob(prob: float, buy_thr: float, sell_thr: float) -> str:
    if np.isnan(prob):
        return "HOLD"
    if prob > buy_thr:
        return "BUY"
    if prob < sell_thr:
        return "SELL"
    return "HOLD"

def expected_business_date(last_idx, horizon_days: int) -> str:
    """Legg til handelsdager fra siste treningsindeks (ikke kalenderdager)."""
    if last_idx is None or pd.isna(last_idx):
        return "—"
    try:
        return (pd.to_datetime(last_idx) + BusinessDay(horizon_days)).strftime("%Y-%m-%d")
    except Exception:
        return "—"

def last_proba(pack_key: dict, default=0.5) -> float:
    try:
        s = pack_key["proba"]
        if len(s) == 0:
            return default
        v = float(s.iloc[-1])
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default

def style_df(df: pd.DataFrame, fmt_map: dict):
    styler = df.style.format(fmt_map)
    try:
        styler = styler.hide_index()
    except Exception:
        try:
            styler = styler.hide(axis="index")
        except Exception:
            pass
    return styler

# -----------------------------
# Kjør skann
# -----------------------------
run = st.button("🔎 Skann og sammenlign (A/B/C)")
results = []

if run:
    progress = st.progress(0)
    status = st.empty()

    for i, t in enumerate(tickers, start=1):
        status.write(f"Henter og analyserer: {t} ({i}/{len(tickers)})")
        try:
            df_raw = fetch_history(t, start=start_date, end=end_date)
        except Exception:
            df_raw = pd.DataFrame()

        if df_raw is None or df_raw.empty:
            results.append({
                "Ticker": t,
                f"Prob_A": np.nan, f"Rec_A": "HOLD", f"Date_A": "—", f"EPS_A(%)": np.nan,
                f"Prob_B": np.nan, f"Rec_B": "HOLD", f"Date_B": "—", f"EPS_B(%)": np.nan,
                f"Prob_C": np.nan, f"Rec_C": "HOLD", f"Date_C": "—", f"EPS_C(%)": np.nan,
                "Acc": np.nan, "AUC": np.nan, "Composite": np.nan
            })
            progress.progress(i / len(tickers))
            continue

        pack = analyze_ticker_multi(df_raw, HORIZONS, EPS_MAP)

        pA = last_proba(pack["A"])
        pB = last_proba(pack["B"])
        pC = last_proba(pack["C"])

        dA = expected_business_date(pack["A"]["last_idx"], pack["A"]["H"])
        dB = expected_business_date(pack["B"]["last_idx"], pack["B"]["H"])
        dC = expected_business_date(pack["C"]["last_idx"], pack["C"]["H"])

        bA, sA = THR["A"]; bB, sB = THR["B"]; bC, sC = THR["C"]

        rA = rec_from_prob(pA, max(bA, pack["A"]["opt_thr"]), sA)
        rB = rec_from_prob(pB, max(bB, pack["B"]["opt_thr"]), sB)
        rC = rec_from_prob(pC, max(bC, pack["C"]["opt_thr"]), sC)

        probs = [x for x in [pA, pB, pC] if not np.isnan(x)]
        comp = float(np.mean(probs)) if probs else np.nan

        accs = [pack[k]["acc"] for k in ["A","B","C"] if not np.isnan(pack[k]["acc"])]
        aucs = [pack[k]["auc"] for k in ["A","B","C"] if not np.isnan(pack[k]["auc"])]
        acc = float(np.mean(accs)) if accs else np.nan
        auc = float(np.mean(aucs)) if aucs else np.nan

        results.append({
            "Ticker": t,
            "Prob_A": pA, "Rec_A": rA, "Date_A": dA, "EPS_A(%)": pack["A"]["eps"],
            "Prob_B": pB, "Rec_B": rB, "Date_B": dB, "EPS_B(%)": pack["B"]["eps"],
            "Prob_C": pC, "Rec_C": rC, "Date_C": dC, "EPS_C(%)": pack["C"]["eps"],
            "Acc": acc, "AUC": auc, "Composite": comp
        })

        progress.progress(i / len(tickers))

    status.empty()
    progress.empty()

# -----------------------------
# Visning
# -----------------------------
if run:
    df = pd.DataFrame(results)

    # Tving numerisk for formatering
    num_cols = ["Prob_A","Prob_B","Prob_C","Acc","AUC","Composite","EPS_A(%)","EPS_B(%)","EPS_C(%)"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader(f"🟩 {H_A} handelsdager frem (A)")
        dA_tab = df[["Ticker","Prob_A","Rec_A","Date_A","EPS_A(%)","Acc"]].copy().sort_values("Prob_A", ascending=False)
        st.dataframe(
            style_df(dA_tab, {"Prob_A": "{:.2%}","Acc": "{:.2%}","EPS_A(%)": "{:.2f}"}),
            use_container_width=True
        )

    with c2:
        st.subheader(f"🟦 {H_B} handelsdager frem (B)")
        dB_tab = df[["Ticker","Prob_B","Rec_B","Date_B","EPS_B(%)","Acc"]].copy().sort_values("Prob_B", ascending=False)
        st.dataframe(
            style_df(dB_tab, {"Prob_B": "{:.2%}","Acc": "{:.2%}","EPS_B(%)": "{:.2f}"}),
            use_container_width=True
        )

    with c3:
        st.subheader(f"🟧 {H_C} handelsdager frem (C)")
        dC_tab = df[["Ticker","Prob_C","Rec_C","Date_C","EPS_C(%)","Acc"]].copy().sort_values("Prob_C", ascending=False)
        st.dataframe(
            style_df(dC_tab, {"Prob_C": "{:.2%}","Acc": "{:.2%}","EPS_C(%)": "{:.2f}"}),
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("📋 Sammenligningstabell (alle horisonter)")

    cmp_df = df[[
        "Ticker",
        "Prob_A","Rec_A","Date_A","EPS_A(%)",
        "Prob_B","Rec_B","Date_B","EPS_B(%)",
        "Prob_C","Rec_C","Date_C","EPS_C(%)",
        "Acc","AUC","Composite"
    ]].sort_values("Composite", ascending=False)

    st.dataframe(
        style_df(
            cmp_df,
            {
                "Prob_A":"{:.2%}","Prob_B":"{:.2%}","Prob_C":"{:.2%}",
                "Acc":"{:.2%}","AUC":"{:.3f}","Composite":"{:.2%}",
                "EPS_A(%)":"{:.2f}","EPS_B(%)":"{:.2f}","EPS_C(%)":"{:.2f}"
            }
        ),
        use_container_width=True
    )

    # -------------------------
    # Eksport
    # -------------------------
    st.markdown("---")
    st.subheader("📤 Eksport")

    if want_csv:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Last ned CSV (alle kolonner)",
            data=csv_bytes,
            file_name=f"scan_v6_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

    if want_excel:
        try:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                dA_tab.to_excel(writer, index=False, sheet_name=f"A_{H_A}d")
                dB_tab.to_excel(writer, index=False, sheet_name=f"B_{H_B}d")
                dC_tab.to_excel(writer, index=False, sheet_name=f"C_{H_C}d")
                cmp_df.to_excel(writer, index=False, sheet_name="Comparison")
            buf.seek(0)
            xls = buf.getvalue()
            st.download_button(
                "⬇️ Last ned Excel (A/B/C/Comparison)",
                data=xls,
                file_name=f"scan_v6_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.info(f"Excel-eksport feilet: {e}")

else:
    st.info("Velg/skriv tickere i sidepanelet og trykk **🔎 Skann og sammenlign** for å starte.")
























