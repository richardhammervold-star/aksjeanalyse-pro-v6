# app_v6.py
# 📈 Aksjeanalyse – Pro v6 Dashboard
# Tre valgbare horisonter (A/B/C) med egne eps-filter og anbefalinger
# Robust håndtering av yfinance DataFrames (1D, 2D, MultiIndex)

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
# Sideoppsett
# -----------------------------
st.set_page_config(page_title="Aksjeanalyse – Pro v6", layout="wide")
st.title("📈 Aksjeanalyse – Pro v6 Dashboard")
st.caption("Tre justerbare horisonter (A/B/C) • Robust Close-serie • Walk-forward evaluering")

# -----------------------------
# Presets (kan utvides)
# -----------------------------
PRESETS = {
    "OBX (Norge)": [
        "EQNR.OL","DNB.OL","MOWI.OL","NHY.OL","TEL.OL","ORK.OL","YAR.OL","KOG.OL","AKRBP.OL","TGS.OL"
    ],
    "USA – Megacaps": ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","XOM"],
}

# -----------------------------
# Streamlit – sidebar
# -----------------------------
st.sidebar.header("⚙️ Innstillinger")

preset = st.sidebar.selectbox("Hurtigvalg gruppe", list(PRESETS.keys()))
if st.sidebar.button("📥 Last inn preset"):
    st.session_state["tickers"] = PRESETS[preset]

default_tickers = PRESETS[preset][:8]
tickers = st.session_state.get("tickers", default_tickers)
user_tickers = st.sidebar.text_area(
    "Tickere (komma/linjer). Tomt felt = bruk valgt preset.",
    value=", ".join(tickers), height=120
)
tickers = [t.strip().upper() for line in user_tickers.splitlines() for t in line.split(",") if t.strip()]

col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    start_date = st.date_input("Startdato", pd.to_datetime("2019-01-01"))
with col_d2:
    end_date = st.date_input("Sluttdato", pd.Timestamp.today())

st.sidebar.markdown("---")
st.sidebar.subheader("Horisonter (handelsdager)")
dA = st.sidebar.slider("Horisont A (vises som 'A')", 1, 30, 3)
dB = st.sidebar.slider("Horisont B (vises som 'B')", 1, 30, 7)
dC = st.sidebar.slider("Horisont C (vises som 'C')", 1, 30, 14)

st.sidebar.subheader("Støyfilter pr. horisont (eps, %)")
epsA = st.sidebar.number_input("EPS_A (%)", value=1.00, min_value=0.0, max_value=50.0, step=0.10)
epsB = st.sidebar.number_input("EPS_B (%)", value=3.00, min_value=0.0, max_value=50.0, step=0.10)
epsC = st.sidebar.number_input("EPS_C (%)", value=5.00, min_value=0.0, max_value=50.0, step=0.10)

st.sidebar.markdown("---")
st.sidebar.subheader("Terskler for anbefaling")
same_thr = st.sidebar.checkbox("Bruk samme terskler på alle horisonter", value=True)

def slider_pair(label_buy, label_sell, buy_default=0.60, sell_default=0.40):
    b = st.sidebar.slider(label_buy, 0.50, 0.90, buy_default, 0.01)
    s = st.sidebar.slider(label_sell, 0.10, 0.50, sell_default, 0.01)
    return b, s

if same_thr:
    b_all, s_all = slider_pair("KJØP hvis sannsynlighet >", "SELG hvis sannsynlighet <", 0.60, 0.40)
    thr = {"A": (b_all, s_all), "B": (b_all, s_all), "C": (b_all, s_all)}
else:
    bA, sA = slider_pair("[A] KJØP >", "[A] SELG <", 0.60, 0.40)
    bB, sB = slider_pair("[B] KJØP >", "[B] SELG <", 0.60, 0.40)
    bC, sC = slider_pair("[C] KJØP >", "[C] SELG <", 0.60, 0.40)
    thr = {"A": (bA, sA), "B": (bB, sB), "C": (bC, sC)}

st.sidebar.markdown("---")
want_excel = st.sidebar.checkbox("Eksporter Excel (flere ark)", value=False)

# -----------------------------
# Datahjelpere
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_history(ticker: str, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        return df
    except Exception:
        return pd.DataFrame()

def extract_close_series(df: pd.DataFrame) -> pd.Series:
    """
    Returner en 1D float-Series for Close (eller Adj Close), robust mot
    MultiIndex/2D fra yfinance. Faller tilbake til første numeriske kolonne.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    if isinstance(df.columns, pd.MultiIndex):
        for lvl0 in ("Close", "Adj Close"):
            try:
                sub = df.xs(lvl0, axis=1, level=0, drop_level=False)
                if isinstance(sub, pd.DataFrame):
                    sub = sub.iloc[:, 0] if sub.shape[1] > 1 else sub.squeeze()
                return pd.to_numeric(sub, errors="coerce").astype(float)
            except Exception:
                pass
        num = df.select_dtypes(include=[np.number])
        if not num.empty:
            s = num.iloc[:, 0].squeeze()
            return pd.to_numeric(s, errors="coerce").astype(float)
        return pd.Series(dtype=float)

    for cand in ("Close", "Adj Close", "close", "adjclose"):
        if cand in df.columns:
            s = df[cand]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0] if s.shape[1] > 1 else s.squeeze()
            return pd.to_numeric(s, errors="coerce").astype(float)

    num = df.select_dtypes(include=[np.number])
    if not num.empty:
        s = num.iloc[:, 0].squeeze()
        return pd.to_numeric(s, errors="coerce").astype(float)

    return pd.Series(dtype=float)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Enkel indikator-pipe, robust Close-serie."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()

    close = extract_close_series(out)
    if close.empty:
        return pd.DataFrame()

    # Daglige returer + litt tech
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

    # EMA + MACD + Bollinger%
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

def make_label_eps(df: pd.DataFrame, horizon_days: int, eps_pct: float) -> pd.Series:
    """Mål: retning over neste N handelsdager, med nøytral sone ±eps%."""
    if df is None or df.empty:
        return pd.Series(dtype=float)

    close = extract_close_series(df)
    if close.empty:
        return pd.Series(dtype=float)

    eps = float(eps_pct) / 100.0
    fwd = close.shift(-horizon_days) / close - 1.0
    arr = np.where(fwd > eps, 1, np.where(fwd < -eps, 0, np.nan))
    return pd.Series(arr.astype("float64"), index=df.index, name=f"Target_{horizon_days}")

FEATURES_ALL = [
    "ret1","ret3","ret5","ma5","ma20","vol10","trend20","rsi14",
    "ema20","ema50","ema_gap","bb_pct","bb_width","macd","macd_sig","macd_hist"
]

def walkforward_fit_predict(X: pd.DataFrame, y: pd.Series):
    """Expanding walk-forward validering + tren endelig modell."""
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

        cand = np.linspace(0.3, 0.7, 41)
        accs = [accuracy_score(yva.astype(int), (p >= t).astype(int)) for t in cand]
        t_star = float(cand[int(np.argmax(accs))])
        thrs.append(t_star)

        val_accs.append(max(accs))
        try:
            val_aucs.append(roc_auc_score(yva.astype(int), p))
        except Exception:
            pass

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

def fit_predict_single_horizon(df_raw: pd.DataFrame, horizon_days: int, eps_pct: float):
    """Bygg features, label og tren modell for én horisont."""
    feats = add_indicators(df_raw)
    if feats.empty:
        return {
            "proba": pd.Series(dtype=float),
            "acc": np.nan, "auc": np.nan, "opt_thr": 0.5, "last_date": None
        }

    y = make_label_eps(df_raw, horizon_days, eps_pct)
    # velg features som finnes og har data
    avail = [c for c in FEATURES_ALL if c in feats.columns and feats[c].notna().any()]
    if not avail:
        return {
            "proba": pd.Series(dtype=float),
            "acc": np.nan, "auc": np.nan, "opt_thr": 0.5, "last_date": None
        }

    pack = pd.concat([feats[avail], y], axis=1).dropna()
    if pack.empty or len(pack) < 80:
        return {
            "proba": pd.Series(0.5, index=(pack.index if len(pack) else feats.index)),
            "acc": np.nan, "auc": np.nan, "opt_thr": 0.5,
            "last_date": (pack.index[-1] if len(pack) else None)
        }

    X = pack[avail]
    yv = pack[y.name]

    if len(np.unique(yv.values.astype(int))) < 2:
        return {
            "proba": pd.Series(0.5, index=pack.index, name="proba"),
            "acc": np.nan, "auc": np.nan, "opt_thr": 0.5, "last_date": pack.index[-1]
        }

    proba_full, acc, auc, opt_thr = walkforward_fit_predict(X, yv)
    return {
        "proba": proba_full,
        "acc": acc, "auc": auc, "opt_thr": opt_thr, "last_date": pack.index[-1]
    }

def analyze_ticker_multi(df_raw: pd.DataFrame, horizons: dict, eps_map: dict):
    """
    Tren tre horisonter (A/B/C). horizons={'A':3,'B':7,'C':14}, eps_map={'A':1.0,...}
    """
    out = {}
    if df_raw is None or df_raw.empty:
        for k in horizons.keys():
            out[k] = {"proba": pd.Series(dtype=float), "acc": np.nan, "auc": np.nan,
                      "opt_thr": 0.5, "last_date": None}
        return out

    for key, H in horizons.items():
        out[key] = fit_predict_single_horizon(df_raw, int(H), float(eps_map.get(key, 0.0)))
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
run = st.button("🔎 Skann og sammenlign (A / B / C)")
results = []

HORIZONS = {"A": dA, "B": dB, "C": dC}
EPS_MAP   = {"A": epsA, "B": epsB, "C": epsC}

if run:
    progress = st.progress(0.0)
    status = st.empty()

    for i, t in enumerate(tickers, start=1):
        status.write(f"Henter og analyserer: {t} ({i}/{len(tickers)})")
        df_raw = fetch_history(t, start=start_date, end=end_date)

        if df_raw is None or df_raw.empty:
            results.append({
                "Ticker": t,
                "Prob_A": np.nan, "Rec_A": "HOLD", "Date_A": "—", "EPS_A(%)": epsA,
                "Prob_B": np.nan, "Rec_B": "HOLD", "Date_B": "—", "EPS_B(%)": epsB,
                "Prob_C": np.nan, "Rec_C": "HOLD", "Date_C": "—", "EPS_C(%)": epsC,
                "Acc": np.nan, "AUC": np.nan, "Composite": np.nan
            })
            progress.progress(i/len(tickers))
            continue

        pack = analyze_ticker_multi(df_raw, HORIZONS, EPS_MAP)

        def last_proba(key, default=np.nan):
            try:
                s = pack[key]["proba"]
                if len(s) == 0:
                    return default
                v = float(s.iloc[-1])# app_v6.py
# 📈 Aksjeanalyse – Pro v6 Dashboard
# Fleksible horisonter (A/B/C), uavhengige eps-filtre og feature-valg
# Build: v6.3 – oktober 2025

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

# -----------------------------
# Sideoppsett + enkel styling
# -----------------------------
st.set_page_config(page_title="Aksjeanalyse – Pro v6 Dashboard", layout="wide")

st.title("📈 Aksjeanalyse – Pro v6 Dashboard")
st.caption("Fleksible horisonter (A/B/C) • uavhengige eps-filtre og features • Build: v6.3 – oktober 2025")

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
tickers = [t.strip().upper() for chunk in user_tickers.split("\n")
           for t in chunk.split(",") if t.strip()]

col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    start_date = st.date_input("Startdato", pd.to_datetime("2019-01-01"))
with col_d2:
    end_date = st.date_input("Sluttdato", pd.Timestamp.today())

st.sidebar.markdown("---")
st.sidebar.subheader("Horisont (handelsdager)")

H_A = st.sidebar.number_input("Horisont A (vises som 'A')", min_value=1, max_value=60, value=3, step=1)
H_B = st.sidebar.number_input("Horisont B (vises som 'B')", min_value=1, max_value=60, value=7, step=1)
H_C = st.sidebar.number_input("Horisont C (vises som 'C')", min_value=1, max_value=60, value=14, step=1)
HORIZONS = {"A": H_A, "B": H_B, "C": H_C}

st.sidebar.markdown("---")
st.sidebar.subheader("Støyfilter (eps, %) per horisont")

same_eps = st.sidebar.checkbox("Bruk samme eps for alle horisonter", value=False)
if same_eps:
    eps_all = st.sidebar.number_input("Eps for A/B/C (%)", value=0.20, min_value=0.0, max_value=50.0, step=0.10)
    EPS_MAP = {"A": eps_all, "B": eps_all, "C": eps_all}
else:
    eps_A = st.sidebar.number_input("Eps A (%)", value=0.20, min_value=0.0, max_value=50.0, step=0.10)
    eps_B = st.sidebar.number_input("Eps B (%)", value=0.50, min_value=0.0, max_value=50.0, step=0.10)
    eps_C = st.sidebar.number_input("Eps C (%)", value=1.00, min_value=0.0, max_value=50.0, step=0.10)
    EPS_MAP = {"A": eps_A, "B": eps_B, "C": eps_C}

st.sidebar.markdown("---")
st.sidebar.subheader("Features")

# Full liste over beregnede features
FEATURES_ALL = [
    "ret1","ret3","ret5",
    "ma5","ma20",
    "vol10","trend20",
    "rsi14",
    "ema20","ema50","ema_gap",
    "bb_pct","bb_width",
    "macd","macd_sig","macd_hist",
]

same_feats = st.sidebar.checkbox("Bruk samme features på alle horisonter", value=True)
default_feats = ["ret1","ret3","ret5","rsi14","ema_gap","macd","macd_hist","bb_pct","bb_width","vol10"]

if same_feats:
    feats_common = st.sidebar.multiselect("Velg features (A/B/C)", FEATURES_ALL, default=default_feats)
    FEATS_MAP = {"A": feats_common, "B": feats_common, "C": feats_common}
else:
    feats_A = st.sidebar.multiselect("Features for A", FEATURES_ALL, default=default_feats)
    feats_B = st.sidebar.multiselect("Features for B", FEATURES_ALL, default=default_feats)
    feats_C = st.sidebar.multiselect("Features for C", FEATURES_ALL, default=default_feats)
    FEATS_MAP = {"A": feats_A, "B": feats_B, "C": feats_C}

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
    bA, sA = slider_pair("A: KJØP >", "A: SELG <", 0.60, 0.40)
    bB, sB = slider_pair("B: KJØP >", "B: SELG <", 0.60, 0.40)
    bC, sC = slider_pair("C: KJØP >", "C: SELG <", 0.60, 0.40)
    THR = {"A": (bA, sA), "B": (bB, sB), "C": (bC, sC)}

st.sidebar.markdown("---")
want_excel = st.sidebar.checkbox("Eksporter Excel (flere ark)", value=False)

# -----------------------------
# Datahjelpere (med caching)
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_history(ticker: str, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    return df

def add_indicators(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Bygger feature-sett. Robust mot typer/NaN."""
    df = df_raw.copy()
    if "Close" not in df:
        return pd.DataFrame()

    close = pd.to_numeric(df["Close"], errors="coerce").astype(float)
    df["ret1"]  = close.pct_change(1)
    df["ret3"]  = close.pct_change(3)
    df["ret5"]  = close.pct_change(5)

    df["ma5"]   = close.rolling(5).mean()
    df["ma20"]  = close.rolling(20).mean()
    df["vol10"] = df["ret1"].rolling(10).std()

    # trend
    with np.errstate(divide="ignore", invalid="ignore"):
        df["trend20"] = (df["ma20"] - df["ma5"]) / df["ma20"]

    # RSI(14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs   = gain / loss
    df["rsi14"] = 100 - (100 / (1 + rs))

    # EMA og MACD
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    df["ema20"] = ema20
    df["ema50"] = ema50
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ema_gap"] = (ema20 - ema50) / ema50

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    df["macd"]      = macd
    df["macd_sig"]  = sig
    df["macd_hist"] = macd - sig

    # Bollinger%
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    with np.errstate(divide="ignore", invalid="ignore"):
        df["bb_pct"]   = (close - lower) / (upper - lower)
        df["bb_width"] = (upper - lower) / ma20

    # Konverter alt til numerisk (trygt), behold index
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def make_label(df: pd.DataFrame, horizon: int, eps_pct: float) -> pd.Series:
    """Binær label med nøytral sone ±eps_pct. 1D Series, indeks = df.index."""
    if "Close" not in df:
        return pd.Series(dtype="float64")
    eps = float(eps_pct) / 100.0
    close = pd.to_numeric(df["Close"], errors="coerce").astype(float)
    fwd = close.shift(-int(horizon)) / close - 1.0
    arr = np.where(fwd > eps, 1, np.where(fwd < -eps, 0, np.nan))
    return pd.Series(arr, index=df.index, name=f"Target_{horizon}")

def rec_from_prob(prob: float, buy_thr: float, sell_thr: float) -> str:
    if np.isnan(prob):
        return "HOLD"
    if prob >= buy_thr:
        return "BUY"
    if prob <= sell_thr:
        return "SELL"
    return "HOLD"

def expected_business_date(last_date, horizon_days: int) -> str:
    if last_date is None or pd.isna(last_date):
        return "—"
    try:
        dt = pd.to_datetime(last_date) + BusinessDay(int(horizon_days))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return "—"

def fit_predict_single_horizon(df_raw: pd.DataFrame,
                               horizon: int,
                               eps_pct: float,
                               selected_feats: list):
    """
    Returnerer dict:
      - proba: Series (sannsynlighet 0..1)
      - last_date: Timestamp
      - acc / auc: validerings-score (NaN om ikke tilgjengelig)
      - opt_thr: 0.5 (placeholder for kompatibilitet)
    Robust: faller tilbake til 0.50 hvis data/klasser er utilstrekkelig.
    """
    out = {"proba": pd.Series(dtype=float),
           "last_date": None,
           "acc": np.nan,
           "auc": np.nan,
           "opt_thr": 0.5}

    if df_raw is None or df_raw.empty or "Close" not in df_raw:
        return out

    feats = add_indicators(df_raw)
    if feats.empty:
        return out

    # Label
    y = make_label(feats, horizon, eps_pct)
    if y.empty or len(y) != len(feats):
        return out

    # Velg features som faktisk finnes og har minst en ikke-NaN
    selected_feats = list(selected_feats) if selected_feats else []
    avail = [c for c in selected_feats if c in feats.columns and feats[c].notna().any()]
    if len(avail) == 0:
        # Ingen features -> nøytral serie på samme indeks
        out["proba"] = pd.Series(0.5, index=feats.index, name="proba")
        out["last_date"] = feats.index[-1] if len(feats) else None
        return out

    # Pakke sammen og droppe rader med NaN i X eller y
    pack = pd.concat([feats[avail], y], axis=1)
    pack = pack.dropna()
    if pack.empty or len(pack) < 120:
        out["proba"] = pd.Series(0.5, index=(pack.index if len(pack) else feats.index), name="proba")
        out["last_date"] = pack.index[-1] if len(pack) else (feats.index[-1] if len(feats) else None)
        return out

    X = pack.loc[:, avail]
    yv = pack[y.name].astype(int)

    # Må ha minst to klasser
    if len(np.unique(yv.values)) < 2:
        out["proba"] = pd.Series(0.5, index=pack.index, name="proba")
        out["last_date"] = pack.index[-1]
        return out

    # Enkelt hold-out: siste 20% som validering
    n = len(X)
    split = max(int(n * 0.8), 1)
    Xtr, Xva = X.iloc[:split], X.iloc[split:]
    ytr, yva = yv.iloc[:split], yv.iloc[split:]

    scaler = RobustScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xall_s = scaler.transform(X)

    base = GradientBoostingClassifier(random_state=0)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(Xtr_s, ytr)

    proba_all = clf.predict_proba(Xall_s)[:, 1]
    out["proba"] = pd.Series(proba_all, index=pack.index, name="proba")
    out["last_date"] = pack.index[-1]

    # Valideringsscore hvis vi har valideringssett
    if len(Xva) > 0:
        pva = clf.predict_proba(Xva_s)[:, 1]
        try:
            out["acc"] = accuracy_score(yva, (pva >= 0.5).astype(int))
        except Exception:
            out["acc"] = np.nan
        try:
            out["auc"] = roc_auc_score(yva, pva)
        except Exception:
            out["auc"] = np.nan

    return out

def analyze_ticker_multi(df_raw: pd.DataFrame,
                         horizons: dict,
                         eps_map: dict,
                         feats_map: dict) -> dict:
    """
    Kjører tre uavhengige modeller (A/B/C).
    Returnerer dict pr nøkkel i horizons (f.eks 'A','B','C').
    """
    out = {}
    for key, H in horizons.items():
        eps = float(eps_map.get(key, 0.0))
        feats = feats_map.get(key, [])
        out[key] = fit_predict_single_horizon(df_raw, int(H), eps, feats)
    return out

# -----------------------------
# Kjør skann
# -----------------------------
run = st.button("🔎 Skann og sammenlign (A / B / C)")
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

        if df_raw is None or df_raw.empty or "Close" not in df_raw:
            results.append({
                "Ticker": t,
                "Prob_A": np.nan, "Rec_A": "HOLD", "Date_A": "—", "EPS_A(%)": np.nan,
                "Prob_B": np.nan, "Rec_B": "HOLD", "Date_B": "—", "EPS_B(%)": np.nan,
                "Prob_C": np.nan, "Rec_C": "HOLD", "Date_C": "—", "EPS_C(%)": np.nan,
                "Acc": np.nan, "AUC": np.nan,
            })
            progress.progress(i / len(tickers))
            continue

        pack = analyze_ticker_multi(df_raw, HORIZONS, EPS_MAP, FEATS_MAP)

        def last_proba(key, default=np.nan):
            try:
                s = pack[key]["proba"]
                if len(s) == 0:
                    return default
                v = float(s.iloc[-1])
                return v if not np.isnan(v) else default
            except Exception:
                return default

        pA = last_proba("A"); pB = last_proba("B"); pC = last_proba("C")
        dA = expected_business_date(pack["A"]["last_date"], HORIZONS["A"])
        dB = expected_business_date(pack["B"]["last_date"], HORIZONS["B"])
        dC = expected_business_date(pack["C"]["last_date"], HORIZONS["C"])

        bA, sA = THR["A"]; bB, sB = THR["B"]; bC, sC = THR["C"]
        rA = rec_from_prob(pA, bA, sA)
        rB = rec_from_prob(pB, bB, sB)
        rC = rec_from_prob(pC, bC, sC)

        # Enkle samle-scorer (snitt av tilgjengelige)
        accs = [pack[k]["acc"] for k in ["A","B","C"] if not np.isnan(pack[k]["acc"])]
        aucs = [pack[k]["auc"] for k in ["A","B","C"] if not np.isnan(pack[k]["auc"])]
        acc = float(np.mean(accs)) if accs else np.nan
        auc = float(np.mean(aucs)) if aucs else np.nan

        results.append({
            "Ticker": t,
            "Prob_A": pA, "Rec_A": rA, "Date_A": dA, "EPS_A(%)": EPS_MAP["A"],
            "Prob_B": pB, "Rec_B": rB, "Date_B": dB, "EPS_B(%)": EPS_MAP["B"],
            "Prob_C": pC, "Rec_C": rC, "Date_C": dC, "EPS_C(%)": EPS_MAP["C"],
            "Acc": acc, "AUC": auc,
        })

        progress.progress(i / len(tickers))

    status.empty()
    progress.empty()

# -----------------------------
# Visning
# -----------------------------
def style_df(df: pd.DataFrame, fmt_map: dict):
    styler = df.style.format(fmt_map)
    try:
        styler = styler.hide(axis="index")
    except Exception:
        pass
    return styler

if run:
    df = pd.DataFrame(results)

    # Sørg for numeriske kolonner
    for c in ["Prob_A","Prob_B","Prob_C","Acc","AUC","EPS_A(%)","EPS_B(%)","EPS_C(%)"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader(f"🟩 Horisont A ({HORIZONS['A']} handelsdager)")
        dA = df[["Ticker","Prob_A","Rec_A","Date_A","EPS_A(%)","Acc"]].copy()
        dA = dA.sort_values("Prob_A", ascending=False, na_position="last")
        st.dataframe(style_df(dA, {"Prob_A":"{:.2%}","EPS_A(%)":"{:.2f}","Acc":"{:.2%}"}), use_container_width=True)

    with c2:
        st.subheader(f"🟦 Horisont B ({HORIZONS['B']} handelsdager)")
        dB = df[["Ticker","Prob_B","Rec_B","Date_B","EPS_B(%)","Acc"]].copy()
        dB = dB.sort_values("Prob_B", ascending=False, na_position="last")
        st.dataframe(style_df(dB, {"Prob_B":"{:.2%}","EPS_B(%)":"{:.2f}","Acc":"{:.2%}"}), use_container_width=True)

    with c3:
        st.subheader(f"🟧 Horisont C ({HORIZONS['C']} handelsdager)")
        dC = df[["Ticker","Prob_C","Rec_C","Date_C","EPS_C(%)","Acc"]].copy()
        dC = dC.sort_values("Prob_C", ascending=False, na_position="last")
        st.dataframe(style_df(dC, {"Prob_C":"{:.2%}","EPS_C(%)":"{:.2f}","Acc":"{:.2%}"}), use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Sammenligningstabell (alle horisonter)")

    cmp_df = df[[
        "Ticker",
        "Prob_A","Rec_A","Date_A","EPS_A(%)",
        "Prob_B","Rec_B","Date_B","EPS_B(%)",
        "Prob_C","Rec_C","Date_C","EPS_C(%)",
        "Acc","AUC"
    ]].sort_values(["Prob_A","Prob_B","Prob_C"], ascending=False, na_position="last")

    st.dataframe(
        style_df(cmp_df, {
            "Prob_A":"{:.2%}","Prob_B":"{:.2%}","Prob_C":"{:.2%}",
            "EPS_A(%)":"{:.2f}","EPS_B(%)":"{:.2f}","EPS_C(%)":"{:.2f}",
            "Acc":"{:.2%}","AUC":"{:.3f}",
        }), use_container_width=True
    )

    if want_excel:
        try:
            import io
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                dA.to_excel(writer, index=False, sheet_name=f"A_{HORIZONS['A']}d")
                dB.to_excel(writer, index=False, sheet_name=f"B_{HORIZONS['B']}d")
                dC.to_excel(writer, index=False, sheet_name=f"C_{HORIZONS['C']}d")
                cmp_df.to_excel(writer, index=False, sheet_name="Comparison")
            buf.seek(0)
            st.download_button(
                "⬇️ Last ned Excel (A/B/C/Comparison)",
                data=buf.getvalue(),
                file_name=f"scan_v6_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.info(f"Excel-eksport feilet: {e}")

else:
    st.info("Velg/skriv tickere i sidepanelet og trykk **🔎 Skann og sammenlign** for å starte.")







































