# app_v6.py
# 📈 Aksjeanalyse – Pro v6 Dashboard
# v6.3 — fleksible horisonter (A/B/C), uavhengige eps-filtre, robuste fallbacks

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from datetime import datetime
from pandas.tseries.offsets import BusinessDay

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score

# ------------------------------------------------------------
# Sideoppsett
# ------------------------------------------------------------
st.set_page_config(page_title="Aksjeanalyse – Pro v6", layout="wide")

st.title("📈 Aksjeanalyse – Pro v6 Dashboard")
st.caption("Fleksible horisonter (A/B/C), uavhengige eps-filtre og robuste fallbacks.")

# ------------------------------------------------------------
# Presets (du kan legge til/endre fritt)
# ------------------------------------------------------------
PRESETS = {
    "OBX (Norge)": [
        "EQNR.OL","DNB.OL","MOWI.OL","NHY.OL","TEL.OL","ORK.OL","YAR.OL","KOG.OL","AKRBP.OL","TGS.OL"
    ],
    "USA – Megacaps": ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","XOM"],
}

# ------------------------------------------------------------
# Hjelpere
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_history(ticker: str, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        return df
    except Exception:
        return pd.DataFrame()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Lett, robust feature-pipe. Alle numeriske kolonner coerces."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    # Coerce 'Close' -> float
    close = pd.to_numeric(out.get("Close", pd.Series(index=out.index)), errors="coerce").astype(float)

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
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    out["rsi14"] = 100 - (100 / (1 + rs))

    # EMA, MACD, Bollinger
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

    # Drop alt-null-rammer
    if set(out.columns) == set(df.columns):
        # ingen nye features (skjedde noe feil) -> returner tomt
        return pd.DataFrame(index=df.index)

    return out

FEATURES_ALL = [
    "ret1","ret3","ret5","ma5","ma20","vol10","trend20","rsi14",
    "ema20","ema50","ema_gap","bb_pct","bb_width","macd","macd_sig","macd_hist"
]

def make_label_eps(df: pd.DataFrame, horizon_days: int, eps_pct: float) -> pd.Series:
    """Mål: 1 hvis fwd>eps, 0 hvis fwd<-eps, ellers NaN. Alltid 1D Series."""
    if df is None or df.empty or "Close" not in df:
        return pd.Series(dtype=float)
    close = pd.to_numeric(df["Close"], errors="coerce").astype(float)
    fwd = close.shift(-horizon_days) / close - 1.0
    eps = (eps_pct or 0.0) / 100.0
    arr = np.where(fwd > eps, 1, np.where(fwd < -eps, 0, np.nan))
    return pd.Series(arr, index=df.index, name=f"Target_{horizon_days}")

def clean_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Stripp til gitte features, fyll hull, numeric only."""
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.loc[:, [c for c in cols if c in df.columns]].copy()
    if x.empty:
        return pd.DataFrame()
    # Fyll inn med forward/backward fill & median
    x = x.fillna(method="ffill").fillna(method="bfill")
    x = x.apply(pd.to_numeric, errors="coerce")
    med = x.median(numeric_only=True)
    x = x.fillna(med)
    return x

def fit_predict_single_horizon(df_raw: pd.DataFrame, horizon_days: int, eps_pct: float):
    """
    Returnerer:
      proba_full (Series), acc (float), auc (float), opt_thr (float), last_idx (Timestamp|None)
    Robust mot manglende kolonner og for få klasser.
    """
    out_nan = (pd.Series(dtype=float), np.nan, np.nan, 0.5, None)

    feats = add_indicators(df_raw)
    if feats is None or feats.empty or "Close" not in feats:
        return out_nan

    y = make_label_eps(feats, horizon_days, eps_pct)
    base_feats = [c for c in FEATURES_ALL if c in feats.columns and feats[c].notna().any()]
    if len(base_feats) == 0 or len(y) != len(feats):
        return out_nan

    # Pakke: kun gyldige rader
    pack = pd.concat([feats[base_feats], y], axis=1).dropna()
    if pack.empty or len(pack) < 80:
        # for lite data
        proba = pd.Series(0.5, index=(pack.index if len(pack) else feats.index), name="proba")
        return (proba, np.nan, np.nan, 0.5, (pack.index[-1] if len(pack) else None))

    X = pack.loc[:, base_feats]
    yv = pack[y.name]

    # Må ha minst 2 klasser
    uniq = np.unique(yv.values.astype(int))
    if len(uniq) < 2:
        proba = pd.Series(0.5, index=pack.index, name="proba")
        return (proba, np.nan, np.nan, 0.5, pack.index[-1])

    # Walk-forward (enkel variant)
    n = len(X)
    anchors = [int(n*0.60), int(n*0.75), int(n*0.90)]
    probs = pd.Series(np.nan, index=X.index)
    val_accs, val_aucs, thrs = [], [], []

    scaler = RobustScaler()
    for a in anchors:
        tr0, tr1 = 0, a
        va0, va1 = a, min(a + int(n*0.08), n-1)
        Xtr, ytr = X.iloc[tr0:tr1], yv.iloc[tr0:tr1]
        Xva, yva = X.iloc[va0:va1], yv.iloc[va0:va1]
        if len(Xva)==0 or len(np.unique(ytr.astype(int)))<2:
            continue

        scaler.fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xva_s = scaler.transform(Xva)

        base = GradientBoostingClassifier(random_state=0)
        clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        clf.fit(Xtr_s, ytr.astype(int))
        p = clf.predict_proba(Xva_s)[:, 1]
        probs.iloc[va0:va1] = p

        grid = np.linspace(0.3, 0.7, 41)
        accs = [accuracy_score(yva.astype(int), (p>=t).astype(int)) for t in grid]
        t_star = float(grid[int(np.argmax(accs))])
        thrs.append(t_star)
        val_accs.append(max(accs))
        try:
            val_aucs.append(roc_auc_score(yva.astype(int), p))
        except Exception:
            pass

    # Tren endelig modell
    scaler.fit(X)
    Xs = scaler.transform(X)
    base = GradientBoostingClassifier(random_state=0)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(Xs, yv.astype(int))
    proba_full = clf.predict_proba(Xs)[:, 1]
    proba_full = pd.Series(proba_full, index=X.index, name="proba")

    acc = float(np.nanmean(val_accs)) if val_accs else np.nan
    auc = float(np.nanmean(val_aucs)) if val_aucs else np.nan
    opt_thr = float(np.nanmean(thrs)) if thrs else 0.5
    return proba_full, acc, auc, opt_thr, X.index[-1]

def analyze_ticker_multi(df_raw: pd.DataFrame, horizons: dict, eps_map: dict):
    """
    horizons: {"A": int_days, "B": int_days, "C": int_days}
    eps_map:  {"A": float(%),  "B": float(%),  "C": float(%)}
    """
    out = {}
    if df_raw is None or df_raw.empty or "Close" not in df_raw:
        for k in ["A","B","C"]:
            out[k] = {"proba": pd.Series(dtype=float), "acc": np.nan, "auc": np.nan, "opt_thr": 0.5, "last_date": None}
        return out

    for k in ["A","B","C"]:
        H = int(horizons.get(k, 3))
        eps = float(eps_map.get(k, 0.0))
        proba, acc, auc, opt_thr, last_idx = fit_predict_single_horizon(df_raw, H, eps)
        out[k] = {
            "proba": proba,
            "acc": acc,
            "auc": auc,
            "opt_thr": opt_thr,
            "last_date": last_idx,
        }
    return out

# --------------------- FIXED last_proba (ferdig try/except) -------------------
def last_proba(pack, key, default=np.nan):
    """Hent sist kjente sannsynlighet for gitt nøkkel ('A','B','C')."""
    try:
        s = pack[key]["proba"]
        if len(s) == 0:
            return default
        v = float(s.iloc[-1])
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default
# ------------------------------------------------------------------------------

def rec_from_prob(prob: float, buy_thr: float, sell_thr: float) -> str:
    if np.isnan(prob):
        return "HOLD"
    if prob > buy_thr:
        return "BUY"
    if prob < sell_thr:
        return "SELL"
    return "HOLD"

def expected_date(last_date, business_days: int):
    if last_date is None or pd.isna(last_date):
        return "—"
    try:
        return (pd.to_datetime(last_date) + BusinessDay(business_days)).strftime("%Y-%m-%d")
    except Exception:
        return "—"

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

# ------------------------------------------------------------
# Sidebar UI
# ------------------------------------------------------------
st.sidebar.header("⚙️ Innstillinger")

preset = st.sidebar.selectbox("Hurtigvalg gruppe", list(PRESETS.keys()))
if st.sidebar.button("📥 Last inn preset"):
    st.session_state["tickers_v6"] = PRESETS[preset]

tickers_default = PRESETS[preset][:5]
tickers = st.session_state.get("tickers_v6", tickers_default)
user_tickers = st.sidebar.text_area(
    "Tickere (komma/linjer). Tomt felt = bruk valgt preset.",
    value=", ".join(tickers), height=120
)
tickers = [t.strip().upper() for chunk in user_tickers.split("\n") for t in chunk.split(",") if t.strip()]

colA, colB = st.sidebar.columns(2)
with colA:
    start_date = st.date_input("Startdato", pd.to_datetime("2019-01-01"))
with colB:
    end_date = st.date_input("Sluttdato", pd.Timestamp.today())

st.sidebar.markdown("### Horisont (handelsdager)")
dA = st.sidebar.slider("A (vises som '3 dager')", 1, 30, 3)
dB = st.sidebar.slider("B (vises som '7 dager')", 1, 30, 7)
dC = st.sidebar.slider("C (vises som '14 dager')", 1, 60, 14)

st.sidebar.markdown("### Støyfilter (eps, %)")
epsA = st.sidebar.number_input("EPS A (%)", value=0.20, min_value=0.0, max_value=50.0, step=0.10)
epsB = st.sidebar.number_input("EPS B (%)", value=1.00, min_value=0.0, max_value=50.0, step=0.10)
epsC = st.sidebar.number_input("EPS C (%)", value=3.00, min_value=0.0, max_value=50.0, step=0.10)

st.sidebar.markdown("### Terskler for anbefaling")
same_thr = st.sidebar.checkbox("Bruk samme terskler på alle horisonter", value=True)
if same_thr:
    b_all = st.sidebar.slider("KJØP hvis sannsynlighet >", 0.50, 0.90, 0.60, 0.01)
    s_all = st.sidebar.slider("SELG hvis sannsynlighet <", 0.10, 0.50, 0.40, 0.01)
    THR = {"A": (b_all, s_all), "B": (b_all, s_all), "C": (b_all, s_all)}
else:
    bA = st.sidebar.slider("A • KJØP >", 0.50, 0.90, 0.60, 0.01)
    sA = st.sidebar.slider("A • SELG <", 0.10, 0.50, 0.40, 0.01)
    bB = st.sidebar.slider("B • KJØP >", 0.50, 0.90, 0.60, 0.01)
    sB = st.sidebar.slider("B • SELG <", 0.10, 0.50, 0.40, 0.01)
    bC = st.sidebar.slider("C • KJØP >", 0.50, 0.90, 0.60, 0.01)
    sC = st.sidebar.slider("C • SELG <", 0.10, 0.50, 0.40, 0.01)
    THR = {"A": (bA, sA), "B": (bB, sB), "C": (bC, sC)}

want_excel = st.sidebar.checkbox("Eksporter Excel (flere ark)", value=False)

HORIZONS = {"A": dA, "B": dB, "C": dC}
EPS_MAP   = {"A": epsA, "B": epsB, "C": epsC}

# ------------------------------------------------------------
# Kjør
# ------------------------------------------------------------
run = st.button("🔎 Skann og sammenlign (A/B/C)")
results = []

if run:
    progress = st.progress(0)
    status = st.empty()

    for i, t in enumerate(tickers, start=1):
        status.write(f"Henter og analyserer: {t} ({i}/{len(tickers)})")
        raw = fetch_history(t, start_date, end_date)

        if raw is None or raw.empty or "Close" not in raw:
            results.append({
                "Ticker": t,
                "Prob_A": np.nan, "Rec_A": "HOLD", "Date_A": "—", "EPS_A(%)": epsA,
                "Prob_B": np.nan, "Rec_B": "HOLD", "Date_B": "—", "EPS_B(%)": epsB,
                "Prob_C": np.nan, "Rec_C": "HOLD", "Date_C": "—", "EPS_C(%)": epsC,
                "Acc": np.nan, "AUC": np.nan
            })
            progress.progress(i/len(tickers))
            continue

        pack = analyze_ticker_multi(raw, HORIZONS, EPS_MAP)

        pA = last_proba(pack, "A", default=np.nan)
        pB = last_proba(pack, "B", default=np.nan)
        pC = last_proba(pack, "C", default=np.nan)

        dA_str = expected_date(pack["A"]["last_date"], HORIZONS["A"])
        dB_str = expected_date(pack["B"]["last_date"], HORIZONS["B"])
        dC_str = expected_date(pack["C"]["last_date"], HORIZONS["C"])

        rA = rec_from_prob(pA, max(THR["A"][0], pack["A"]["opt_thr"]), THR["A"][1])
        rB = rec_from_prob(pB, max(THR["B"][0], pack["B"]["opt_thr"]), THR["B"][1])
        rC = rec_from_prob(pC, max(THR["C"][0], pack["C"]["opt_thr"]), THR["C"][1])

        accs = [x for x in [pack["A"]["acc"], pack["B"]["acc"], pack["C"]["acc"]] if not np.isnan(x)]
        aucs = [x for x in [pack["A"]["auc"], pack["B"]["auc"], pack["C"]["auc"]] if not np.isnan(x)]
        acc = float(np.mean(accs)) if accs else np.nan
        auc = float(np.mean(aucs)) if aucs else np.nan

        results.append({
            "Ticker": t,
            "Prob_A": pA, "Rec_A": rA, "Date_A": dA_str, "EPS_A(%)": epsA,
            "Prob_B": pB, "Rec_B": rB, "Date_B": dB_str, "EPS_B(%)": epsB,
            "Prob_C": pC, "Rec_C": rC, "Date_C": dC_str, "EPS_C(%)": epsC,
            "Acc": acc, "AUC": auc
        })
        progress.progress(i/len(tickers))

    status.empty()
    progress.empty()

# ------------------------------------------------------------
# Visning
# ------------------------------------------------------------
if run:
    df = pd.DataFrame(results)

    for c in ["Prob_A","Prob_B","Prob_C","Acc","AUC"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader(f"🟩 {HORIZONS['A']} dager frem")
        dA_tab = df[["Ticker","Prob_A","Rec_A","Date_A","EPS_A(%)","Acc"]].sort_values("Prob_A", ascending=False)
        st.dataframe(style_df(dA_tab, {"Prob_A":"{:.2%}","Acc":"{:.2%}"}), use_container_width=True)

    with c2:
        st.subheader(f"🟦 {HORIZONS['B']} dager frem")
        dB_tab = df[["Ticker","Prob_B","Rec_B","Date_B","EPS_B(%)","Acc"]].sort_values("Prob_B", ascending=False)
        st.dataframe(style_df(dB_tab, {"Prob_B":"{:.2%}","Acc":"{:.2%}"}), use_container_width=True)

    with c3:
        st.subheader(f"🟧 {HORIZONS['C']} dager frem")
        dC_tab = df[["Ticker","Prob_C","Rec_C","Date_C","EPS_C(%)","Acc"]].sort_values("Prob_C", ascending=False)
        st.dataframe(style_df(dC_tab, {"Prob_C":"{:.2%}","Acc":"{:.2%}"}), use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Sammenligningstabell (alle horisonter)")
    cmp = df[[
        "Ticker",
        "Prob_A","Rec_A","Date_A","EPS_A(%)",
        "Prob_B","Rec_B","Date_B","EPS_B(%)",
        "Prob_C","Rec_C","Date_C","EPS_C(%)",
        "Acc","AUC"
    ]].sort_values(["Prob_A","Prob_B","Prob_C"], ascending=False)

    st.dataframe(style_df(cmp, {
        "Prob_A":"{:.2%}","Prob_B":"{:.2%}","Prob_C":"{:.2%}","Acc":"{:.2%}","AUC":"{:.3f}"
    }), use_container_width=True)

    # --------------------------------------------------------
    # Excel-eksport (valgfritt)
    # --------------------------------------------------------
    if want_excel:
        try:
            import io
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                dA_tab.to_excel(writer, index=False, sheet_name=f"A_{HORIZONS['A']}d")
                dB_tab.to_excel(writer, index=False, sheet_name=f"B_{HORIZONS['B']}d")
                dC_tab.to_excel(writer, index=False, sheet_name=f"C_{HORIZONS['C']}d")
                cmp.to_excel(writer, index=False, sheet_name="Comparison")
            buf.seek(0)
            st.download_button(
                "⬇️ Last ned Excel (A/B/C + Comparison)",
                data=buf.getvalue(),
                file_name=f"scan_v6_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.info(f"Excel-eksport feilet: {e}")

else:
    st.info("Velg/skriv tickere i sidepanelet og trykk **🔎 Skann og sammenlign (A/B/C)** for å starte.")























