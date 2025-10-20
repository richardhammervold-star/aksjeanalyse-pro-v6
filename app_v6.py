# app_v6.py
# 📈 Aksjeanalyse – Pro v6 Dashboard
# Multi-horisont (1d / 3d / 5d) – side om side
# Build: v6.0 – oktober 2025

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

# -----------------------------
# Sideoppsett + enkel styling
# -----------------------------
st.set_page_config(
    page_title="Aksjeanalyse – Pro v6 Dashboard",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main { background-color: #0E1117; }
    .stMarkdown, .stText, .stDataFrame { color: #E6E6E6 !important; }
    .prob-1d { color:#16a34a; font-weight:600; }
    .prob-3d { color:#3b82f6; font-weight:600; }
    .prob-5d { color:#f97316; font-weight:600; }
    .rec-buy  { background: rgba(22,163,74,0.15); border:1px solid rgba(22,163,74,0.35); border-radius:12px; padding:6px 10px; }
    .rec-hold { background: rgba(148,163,184,0.10); border:1px solid rgba(148,163,184,0.25); border-radius:12px; padding:6px 10px; }
    .rec-sell { background: rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.35); border-radius:12px; padding:6px 10px; }
    .soft-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius:16px; padding:12px 14px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 Aksjeanalyse – Pro v6 Dashboard")
st.caption("Multi-horisont modell (1d / 3d / 5d) • Build: v6.0 – oktober 2025")

# -----------------------------
# Presets
# -----------------------------
PRESETS = {
    "OBX (Norge)": [
        "EQNR.OL","DNB.OL","MOWI.OL","NHY.OL","TEL.OL","ORK.OL","YAR.OL","KOG.OL","AKRBP.OL",
        "TGS.OL","SUBC.OL","SALM.OL","AUTO.OL","ELK.OL","NOD.OL","PGS.OL","ADE.OL","BONHR.OL",
        "OTELLO.OL","KID.OL","NRC.OL","VAR.OL","KIT.OL","SCHB.OL","AGS.OL","NEL.OL"
    ],
    "USA – Megacaps": [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","XOM","UNH","V","JNJ","WMT","PG","MA","AVGO","HD","MRK","PEP"
    ],
    "Europa – Blue chips": [
        "NESN.SW","NOVN.SW","ROG.SW","SAP.DE","SIE.DE","MBG.DE","ASML.AS","AD.AS","AIR.PA","OR.PA",
        "MC.PA","SAN.PA","ULVR.L","HSBA.L","SHEL.L","BP.L","BATS.L","RIO.L","AZN.L","GSK.L"
    ],
    "DAX40 (utvalg)": [
        "SAP.DE","SIE.DE","MBG.DE","DTE.DE","BMW.DE","ALV.DE","BAS.DE","MUV2.DE","BAYN.DE","VOW3.DE",
        "IFX.DE","ADS.DE","HEI.DE","HEN3.DE","RWE.DE","LIN.DE","PUM.DE","FME.DE","ENR.DE","DB1.DE"
    ],
    "CAC40 (utvalg)": [
        "OR.PA","MC.PA","AIR.PA","BNP.PA","SU.PA","ENGI.PA","GLE.PA","DG.PA","AI.PA","UG.PA",
        "SAN.PA","ORA.PA","KER.PA","SGO.PA","CAP.PA","STLAP.PA","ACA.PA","RNO.PA","EDEN.PA","PUB.PA"
    ],
    "FTSE100 (utvalg)": [
        "SHEL.L","BP.L","HSBA.L","AZN.L","ULVR.L","RIO.L","BATS.L","GSK.L","DGE.L","GLEN.L",
        "VOD.L","LSEG.L","BARC.L","AV.L","NG.L","BA.L","AAL.L","PHNX.L","REL.L","BTI"
    ],
    "OMX30 (utvalg)": [
        "VOLV-B.ST","ERIC-B.ST","SAND.ST","ATCO-A.ST","ATCO-B.ST","ESSITY-B.ST","SWED-A.ST","SEB-A.ST","ALFA.ST","TELIA.ST",
        "ABB.ST","HEXA-B.ST","SKF-B.ST","BOL.ST","INVE-B.ST","EVO.ST","KINV-B.ST","NDA-SE.ST","MTG-B.ST","SCA-B.ST"
    ],
    "Råvarer": ["CL=F","BZ=F","NG=F","GC=F","SI=F","HG=F","ZC=F","ZW=F","ZS=F"],
    "Valuta (Forex)": ["EURUSD=X","USDJPY=X","GBPUSD=X","AUDUSD=X","USDCAD=X","USDCHF=X","EURNOK=X","USDNOK=X","EURGBP=X"],
    "Krypto": ["BTC-USD","ETH-USD","SOL-USD","XRP-USD","ADA-USD"],
    "USA – Teknologi": [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","AMD","CRM","ORCL","NFLX","NOW","INTU","PANW"
    ]
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
    value=", ".join(tickers), height=120
)
tickers = [t.strip().upper() for chunk in user_tickers.split("\n") for t in chunk.split(",") if t.strip()]

col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    start_date = st.date_input("Startdato", pd.to_datetime("2019-01-01"))
with col_d2:
    end_date = st.date_input("Sluttdato", pd.Timestamp.today())

st.sidebar.markdown("---")
st.sidebar.subheader("Terskler")
same_thr = st.sidebar.checkbox("Bruk samme terskler på alle horisonter", value=False)

def slider_pair(label_buy, label_sell, buy_default=0.65, sell_default=0.50):
    b = st.sidebar.slider(label_buy, 0.50, 0.90, buy_default, 0.01)
    s = st.sidebar.slider(label_sell, 0.10, 0.50, sell_default, 0.01)
    return b, s

if same_thr:
    b_all, s_all = slider_pair("KJØP hvis sannsynlighet >", "SELG hvis sannsynlighet <", 0.65, 0.50)
    thr = {"1d": (b_all, s_all), "3d": (b_all, s_all), "5d": (b_all, s_all)}
else:
    b1, s1 = slider_pair("1 dag • KJØP >", "1 dag • SELG <", 0.65, 0.50)
    b3, s3 = slider_pair("3 dager • KJØP >", "3 dager • SELG <", 0.65, 0.50)
    b5, s5 = slider_pair("5 dager • KJØP >", "5 dager • SELG <", 0.65, 0.50)
    thr = {"1d": (b1, s1), "3d": (b3, s3), "5d": (b5, s5)}

st.sidebar.markdown("---")
want_excel = st.sidebar.checkbox("Eksporter Excel (flere ark)", value=True)
eps = st.sidebar.number_input("Støyfilter (eps, %)", value=0.2, min_value=0.0, max_value=2.0, step=0.1,
                              help="Bevegelser med absolutt avkastning < eps% behandles som støy.")
st.sidebar.caption("Tips: Norske aksjer bruker .OL (EQNR.OL). Valuta = '=X' (EURNOK=X). Råvarer: CL=F, GC=F.")

# -----------------------------
# Datahjelpere (med caching)
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_history(ticker: str, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Rask felles feature-pipe som kan gjenbrukes på tvers av horisonter."""
    df = df.copy()
    close = df["Close"].astype(float)
    ret1 = close.pct_change()
    df["ret1"] = ret1
    df["ret3"] = close.pct_change(3)
    df["ret5"] = close.pct_change(5)
    df["ma5"] = close.rolling(5).mean()
    df["ma20"] = close.rolling(20).mean()
    df["vol10"] = ret1.rolling(10).std()
    df["trend20"] = (df["ma20"] - df["ma5"]) / df["ma20"]

    # RSI(14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi14"] = 100 - (100 / (1 + rs))

    # EMA og MACD og Bollinger%
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema_gap"] = (df["ema20"] - df["ema50"]) / df["ema50"]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd
    df["macd_sig"] = signal
    df["macd_hist"] = macd - signal

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    df["bb_pct"] = (close - lower) / (upper - lower)
    df["bb_width"] = (upper - lower) / ma20

    return df

# ---- PATCHED: trygg 1D label ----
def make_label(df: pd.DataFrame, horizon: int, eps_frac: float) -> pd.Series:
    """Mål: retning over neste N dager. eps_frac i % -> brøk. Returnerer alltid 1D Series."""
    eps = eps_frac / 100.0
    close = df["Close"].astype(float)
    fwd = close.shift(-horizon) / close - 1.0

    # Bruk .to_numpy() og .ravel() for å sikre 1D
    fwd_np = fwd.to_numpy()
    arr = np.where(fwd_np > eps, 1, np.where(fwd_np < -eps, 0, np.nan))
    arr = np.asarray(arr, dtype="float64").ravel()

    return pd.Series(arr, index=df.index, name=f"Target_{horizon}")

FEATURES_ALL = [
    "ret1","ret3","ret5","ma5","ma20","vol10","trend20","rsi14",
    "ema20","ema50","ema_gap","bb_pct","bb_width","macd","macd_sig","macd_hist"
]

def walkforward_fit_predict(X: pd.DataFrame, y: pd.Series):
    """Expanding walk-forward CV for validering + tren endelig modell på hele datasettet."""
    n = len(X)
    if n < 120 or y.isna().sum() > 0:
        return pd.Series([np.nan]*n, index=X.index), np.nan, np.nan, 0.5

    anchors = [int(n*0.60), int(n*0.70), int(n*0.80)]
    probs = pd.Series(np.nan, index=X.index)
    val_accs, val_aucs, thrs = [], [], []

    for a in anchors:
        tr0, tr1 = 0, a
        va0, va1 = a, min(a + int(n*0.10), n-1)
        Xtr, ytr = X.iloc[tr0:tr1], y.iloc[tr0:tr1]
        Xva, yva = X.iloc[va0:va1], y.iloc[va0:va1]

        if len(Xva)==0 or len(np.unique(ytr.dropna().astype(int))) < 2:
            continue

        scaler = RobustScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xva_s = scaler.transform(Xva)

        base = GradientBoostingClassifier(random_state=0)
        clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        clf.fit(Xtr_s, ytr.astype(int))

        p = clf.predict_proba(Xva_s)[:, 1]
        probs.iloc[va0:va1] = p

        # beste cutoff (maks accuracy)
        cand = np.linspace(0.3, 0.7, 41)
        accs = [accuracy_score(yva.astype(int), (p >= t).astype(int)) for t in cand]
        t_star = float(cand[int(np.argmax(accs))])
        thrs.append(t_star)

        val_accs.append(max(accs))
        try:
            val_aucs.append(roc_auc_score(yva.astype(int), p))
        except Exception:
            pass

    # Tren endelig modell på hele datasettet for "dagens" proba
    scaler = RobustScaler().fit(X)
    Xs = scaler.transform(X)
    base = GradientBoostingClassifier(random_state=0)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(Xs, y.astype(int))
    proba_full = clf.predict_proba(Xs)[:, 1]
    proba_full = pd.Series(proba_full, index=X.index)

    acc = float(np.nanmean(val_accs)) if len(val_accs) else np.nan
    auc = float(np.nanmean(val_aucs)) if len(val_aucs) else np.nan
    opt_thr = float(np.nanmean(thrs)) if len(thrs) else 0.5
    return proba_full, acc, auc, opt_thr

# ---- PATCHED: robust mot manglende feature-kolonner ----
def analyze_ticker_multi(df_raw: pd.DataFrame, eps_pct: float) -> dict:
    """
    Bygger indikatorer én gang og trener tre modeller (1/3/5d).
    ROBUST mot manglende/rare feature-kolonner på alle steg.
    """
    out = {}

    # 0) Tomt/ugyldig datagrunnlag
    if df_raw is None or df_raw.empty or "Close" not in df_raw:
        for key in ["1d", "3d", "5d"]:
            out[key] = {"proba": pd.Series(dtype=float), "acc": np.nan, "auc": np.nan,
                        "opt_thr": 0.5, "last_date": None}
        return out

    # 1) Indikatorer
    df = add_indicators(df_raw)

    # 2) Feature-kandidater som faktisk finnes og har minst én ikke-NaN
    feat_base = [c for c in FEATURES_ALL if c in df.columns and df[c].notna().any()]
    if len(feat_base) == 0:
        for key in ["1d", "3d", "5d"]:
            out[key] = {"proba": pd.Series(dtype=float), "acc": np.nan, "auc": np.nan,
                        "opt_thr": 0.5, "last_date": None}
        return out

    for H, key in [(1, "1d"), (3, "3d"), (5, "5d")]:
        # 3) Label
        y = make_label(df, H, eps_pct)
        if y is None or len(y) != len(df):
            out[key] = {"proba": pd.Series(dtype=float), "acc": np.nan, "auc": np.nan,
                        "opt_thr": 0.5, "last_date": None}
            continue

        # 4) Sjekk tilgjengelige features mot df igjen
        available = [c for c in feat_base if c in df.columns]
        if not available:
            out[key] = {"proba": pd.Series(dtype=float), "acc": np.nan, "auc": np.nan,
                        "opt_thr": 0.5, "last_date": None}
            continue

        # 5) Slå sammen (uten subset i dropna for å unngå KeyError)
        pack = pd.concat([df[available], y], axis=1)

        # 6) Kryssjekk kolonnenavn mot pack (kan avvike i sjeldne tilfeller)
        cols_in_pack = [c for c in available if c in pack.columns]
        if not cols_in_pack or y.name not in pack.columns:
            out[key] = {"proba": pd.Series(dtype=float), "acc": np.nan, "auc": np.nan,
                        "opt_thr": 0.5, "last_date": None}
            continue

        pack = pack.dropna()

if pack.empty or len(pack) < 120:
    # Nøytral fallback = 50 % sannsynlighet
    neut_index = pack.index if len(pack) else df.index
    neut = pd.Series(0.5, index=neut_index, name="proba")
    out[key] = {
        "proba": neut,
        "acc": np.nan,
        "auc": np.nan,
        "opt_thr": 0.5,
        "last_date": (pack.index[-1] if len(pack) else None),
    }
    continue
    
# 7) Velg X/Y med faktiske kolonner i pack
X  = pack.loc[:, cols_in_pack]
yv = pack[y.name]

# 8) Må ha minst to klasser for å trene – ellers returnér 0.5 som nøytral sannsynlighet
if len(np.unique(yv.values.astype(int))) < 2:
    neut = pd.Series(0.5, index=pack.index, name="proba")
    out[key] = {
        "proba": neut,           # 50 % sannsynlighet
        "acc": np.nan,
        "auc": np.nan,
        "opt_thr": 0.5,
        "last_date": pack.index[-1],
    }
    continue

# 9) Tren og lagre
proba_full, acc, auc, opt_thr = walkforward_fit_predict(X, yv)


out[key] = {
    "proba": proba_full,
    "acc": acc,
    "auc": auc,
    "opt_thr": opt_thr,
    "last_date": pack.index[-1],
}


return out

def rec_from_prob(prob: float, buy_thr: float, sell_thr: float) -> str:
    if np.isnan(prob):
        return "HOLD"
    if prob > buy_thr:
        return "BUY"
    if prob < sell_thr:
        return "SELL"
    return "HOLD"

def expected_date(last_date: pd.Timestamp, horizon_days: int) -> str:
    if last_date is None or pd.isna(last_date):
        return "—"
    return (pd.to_datetime(last_date) + pd.Timedelta(days=horizon_days)).strftime("%Y-%m-%d")

# -----------------------------
# Kjør skann
# -----------------------------
run = st.button("🔎 Skann og sammenlign (1d / 3d / 5d)")
results = []

if run:
    progress = st.progress(0)
    status = st.empty()

        for i, t in enumerate(tickers, start=1):
        status.write(f"Henter og analyserer: {t}  ({i}/{len(tickers)})")
        try:
            df_raw = fetch_history(t, start=start_date, end=end_date)
        except Exception:
            df_raw = pd.DataFrame()

        if df_raw is None or df_raw.empty or "Close" not in df_raw:
            results.append({
                "Ticker": t,
                "Prob_1d": np.nan, "Rec_1d": "HOLD", "Date_1d": "—",
                "Prob_3d": np.nan, "Rec_3d": "HOLD", "Date_3d": "—",
                "Prob_5d": np.nan, "Rec_5d": "HOLD", "Date_5d": "—",
                "Delta_5d_1d": np.nan,
                "Acc": np.nan, "AUC": np.nan, "Composite": np.nan
            })
            progress.progress(i / len(tickers))
            continue

        # 🚀 Viktig: denne linjen skal ha 8 mellomrom (samme som if/try/except)
        pack = analyze_ticker_multi(df_raw, eps_pct=eps)

        # Hent siste proba per horisont med fallback 0.5
        def last_proba(key, default=0.5):
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

        p1 = last_proba("1d")
        p3 = last_proba("3d")
        p5 = last_proba("5d")

        date1 = expected_date(pack["1d"]["last_date"], 1)
        date3 = expected_date(pack["3d"]["last_date"], 3)
        date5 = expected_date(pack["5d"]["last_date"], 5)

        b1,s1 = thr["1d"]; b3,s3 = thr["3d"]; b5,s5 = thr["5d"]
        r1 = rec_from_prob(p1, max(b1, pack["1d"]["opt_thr"]), s1)
        r3 = rec_from_prob(p3, max(b3, pack["3d"]["opt_thr"]), s3)
        r5 = rec_from_prob(p5, max(b5, pack["5d"]["opt_thr"]), s5)

        probs = [x for x in [p1,p3,p5] if not np.isnan(x)]
        comp = float(np.mean(probs)) if probs else np.nan
        accs = [pack[k]["acc"] for k in ["1d","3d","5d"] if not np.isnan(pack[k]["acc"])]
        aucs = [pack[k]["auc"] for k in ["1d","3d","5d"] if not np.isnan(pack[k]["auc"])]
        acc = float(np.mean(accs)) if accs else np.nan
        auc = float(np.mean(aucs)) if aucs else np.nan

        results.append({
            "Ticker": t,
            "Prob_1d": p1, "Rec_1d": r1, "Date_1d": date1,
            "Prob_3d": p3, "Rec_3d": r3, "Date_3d": date3,
            "Prob_5d": p5, "Rec_5d": r5, "Date_5d": date5,
            "Delta_5d_1d": (p5 - p1) if (not np.isnan(p5) and not np.isnan(p1)) else np.nan,
            "Acc": acc, "AUC": auc, "Composite": comp
        })

        progress.progress(i/len(tickers))

    status.empty()
    progress.empty()

def style_df(df: pd.DataFrame, fmt_map: dict):
    styler = df.style.format(fmt_map)
    # Pandas 2.x: hide_index() finnes ikke – bruk hide(axis="index")
    try:
        styler = styler.hide_index()
    except Exception:
        try:
            styler = styler.hide(axis="index")
        except Exception:
            pass
    return styler

# -----------------------------
# Visning – tre kolonner
# -----------------------------
if run:
    df = pd.DataFrame(results)

    # Sørg for numeriske kolonner (coerce None -> NaN -> tall-format fungerer)
    num_cols = ["Prob_1d", "Prob_3d", "Prob_5d", "Acc", "AUC", "Delta_5d_1d", "Composite"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("🟩 1 dag frem")
        df1 = df[["Ticker", "Prob_1d", "Rec_1d", "Date_1d", "Acc"]].copy()
        df1 = df1.sort_values("Prob_1d", ascending=False)
        st.dataframe(
            style_df(df1, {"Prob_1d": "{:.2%}", "Acc": "{:.2%}"}),
            use_container_width=True
        )

    with c2:
        st.subheader("🟦 3 dager frem")
        df3 = df[["Ticker", "Prob_3d", "Rec_3d", "Date_3d", "Acc"]].copy()
        df3 = df3.sort_values("Prob_3d", ascending=False)
        st.dataframe(
            style_df(df3, {"Prob_3d": "{:.2%}", "Acc": "{:.2%}"}),
            use_container_width=True
        )

    with c3:
        st.subheader("🟧 5 dager frem")
        df5 = df[["Ticker", "Prob_5d", "Rec_5d", "Date_5d", "Acc", "Delta_5d_1d"]].copy()
        df5 = df5.sort_values("Prob_5d", ascending=False)
        st.dataframe(
            style_df(df5, {"Prob_5d": "{:.2%}", "Acc": "{:.2%}", "Delta_5d_1d": "{:.2%}"}),
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("📋 Sammenligningstabell (alle horisonter)")

    cmp_df = df[[
        "Ticker",
        "Prob_1d","Rec_1d","Date_1d",
        "Prob_3d","Rec_3d","Date_3d",
        "Prob_5d","Rec_5d","Date_5d",
        "Delta_5d_1d","Acc","AUC","Composite"
    ]].sort_values("Composite", ascending=False)

    st.dataframe(
        style_df(
            cmp_df,
            {
                "Prob_1d": "{:.2%}",
                "Prob_3d": "{:.2%}",
                "Prob_5d": "{:.2%}",
                "Delta_5d_1d": "{:.2%}",
                "Acc": "{:.2%}",
                "AUC": "{:.3f}",
                "Composite": "{:.2%}",
            }
        ),
        use_container_width=True
    )

    # -------------------------
    # Detalj: graf per ticker
    # -------------------------
    st.markdown("---")
    st.subheader("📊 Detaljvisning (pris + sannsynlighet 1d/3d/5d)")
    sel_list = df["Ticker"].tolist() if not df.empty else []
    sel = st.selectbox("Velg ticker", sel_list)
    if sel:
        try:
            raw = fetch_history(sel, start=start_date, end=end_date)
            pack = analyze_ticker_multi(raw, eps_pct=eps)
            plot_df = pd.DataFrame({
                "Close": raw["Close"],
                "Prob_1d": pack["1d"]["proba"].reindex(raw.index),
                "Prob_3d": pack["3d"]["proba"].reindex(raw.index),
                "Prob_5d": pack["5d"]["proba"].reindex(raw.index),
            }).dropna(subset=["Close"])

            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(plot_df.index, plot_df["Close"])
            ax1.set_xlabel("Dato")
            ax1.set_ylabel("Pris")

            ax2 = ax1.twinx()
            if "Prob_1d" in plot_df: ax2.plot(plot_df.index, plot_df["Prob_1d"], alpha=0.9)
            if "Prob_3d" in plot_df: ax2.plot(plot_df.index, plot_df["Prob_3d"], alpha=0.9)
            if "Prob_5d" in plot_df: ax2.plot(plot_df.index, plot_df["Prob_5d"], alpha=0.9)
            ax2.axhline(0.5, linestyle="--", alpha=0.6)
            ax2.set_ylabel("Sannsynlighet")

            plt.title(f"{sel}: Pris + sannsynlighet (1d/3d/5d)")
            st.pyplot(fig)
        except Exception as e:
            st.info(f"Kunne ikke vise graf for {sel}: {e}")

    # -------------------------
    # Eksport + historikk
    # -------------------------
    st.markdown("---")
    st.subheader("📤 Eksport")

    # CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Last ned CSV (alle horisonter)",
        data=csv_bytes,
        file_name=f"scan_v6_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

    # Excel (flere ark)
    if want_excel:
        try:
            import io
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                df1.to_excel(writer, index=False, sheet_name="1d")
                df3.to_excel(writer, index=False, sheet_name="3d")
                df5.to_excel(writer, index=False, sheet_name="5d")
                cmp_df.to_excel(writer, index=False, sheet_name="Comparison")
            buf.seek(0)
            xls = buf.getvalue()
            st.download_button(
                "⬇️ Last ned Excel (1d/3d/5d/Comparison)",
                data=xls,
                file_name=f"scan_v6_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.info(f"Excel-eksport feilet: {e}")

    st.subheader("🗓️ Historikk (lagre)")
    if "history_v6" not in st.session_state:
        st.session_state["history_v6"] = pd.DataFrame(columns=df.columns)
    if st.button("💾 Legg dagens resultat til historikk"):
        hist = st.session_state["history_v6"]
        tmp = df.copy()
        tmp["SavedAtUTC"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M")
        st.session_state["history_v6"] = pd.concat([hist, tmp], ignore_index=True)
        st.success("Lagt til historikk i denne økten.")

    st.dataframe(
        style_df(
            st.session_state["history_v6"].tail(200),
            {
                "Prob_1d": "{:.2%}",
                "Prob_3d": "{:.2%}",
                "Prob_5d": "{:.2%}",
                "Delta_5d_1d": "{:.2%}",
                "Acc": "{:.2%}",
                "AUC": "{:.3f}",
                "Composite": "{:.2%}",
            }
        ),
        use_container_width=True
    )

else:
    st.info("Velg/skriv tickere i sidepanelet og trykk **🔎 Skann og sammenlign** for å starte.")


















