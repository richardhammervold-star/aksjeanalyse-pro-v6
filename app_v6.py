# app_v6.py
# 📈 Aksjeanalyse – Pro v6
# Fleksible horisonter (A/B/C), uavhengige eps-filtre og feature-valg
# Build: v6.4 – oktober 2025

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

import matplotlib.pyplot as plt

# -----------------------------
# Sideoppsett + enkel styling
# -----------------------------
st.set_page_config(page_title="Aksjeanalyse – Pro v6", layout="wide")

st.title("📈 Aksjeanalyse – Pro v6 Dashboard")
st.caption("Fleksible horisonter (A/B/C), uavhengige eps-filtre og feature-valg • Build v6.4")

# -----------------------------
# 📦 Presets (komplett + favoritter)
# -----------------------------
if "favorites" not in st.session_state:
    st.session_state["favorites"] = ["AAPL", "MSFT", "EQNR.OL", "BTC-USD", "USDNOK=X"]

PRESETS = {
    # ======== ⭐ FAVORITTER ========
    "⭐ Favoritter (mine tickere)": st.session_state["favorites"],

    # ======== 🇳🇴 NORGE ========
    "🇳🇴 OBX (Norge)": [
        "EQNR.OL","DNB.OL","MOWI.OL","NHY.OL","TEL.OL","ORK.OL","YAR.OL","KOG.OL","AKRBP.OL",
        "TGS.OL","SALM.OL","AUTO.OL","ELK.OL","NOD.OL","PGS.OL","ADE.OL","BONHR.OL",
        "OTELLO.OL","KID.OL","NRC.OL","VAR.OL","KIT.OL","SCHB.OL","AGS.OL","NEL.OL"
    ],
    "📊 Euronext Growth Oslo": [
        "NEL.OL","SCATC.OL","RECSI.OL","QFUEL.OL","VOW.OL","HYPRO.OL","HYN.OL",
        "OTELLO.OL","LINK.OL","TECH.OL","SOFTX.OL","KOMP.OL","AUTO.OL","PLAY.OL","CHG.OL",
        "EIOF.OL","DOF.OL","FROY.OL","BONHR.OL","NAPA.OL"
    ],
    "⚙️ Norsk industri / energi": [
        "EQNR.OL","AKRBP.OL","VAR.OL","NHY.OL","YAR.OL","SCATC.OL","HYPRO.OL","NOD.OL","NEL.OL","KOG.OL","TGS.OL"
    ],
    "💻 Norsk teknologi": [
        "OTELLO.OL","LINK.OL","TECH.OL","SOFTX.OL","PLAY.OL","KID.OL","AUTO.OL","ADE.OL","KOMP.OL"
    ],

    # ======== 🇺🇸 USA ========
    "🇺🇸 USA – Megacaps": [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","XOM",
        "UNH","V","JNJ","WMT","PG","MA","AVGO","HD","MRK","PEP"
    ],
    "🇺🇸 USA – Teknologi / AI": [
        "AAPL","MSFT","NVDA","GOOGL","META","AMZN","AMD","CRM","ORCL","INTU",
        "ADBE","NOW","SNOW","NET","PANW","DDOG","PLTR","SHOP","UBER","SQ"
    ],
    "🇺🇸 USA – Fornybar / Clean Energy": [
        "ENPH","SEDG","RUN","NEE","PLUG","BLDP","TSLA","BE","FSLR","SPWR"
    ],

    # ======== 🌍 VERDEN ========
    "🌍 Verden – Topp 50 selskaper": [
        # USA
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","BRK-B","JPM","V","PG","JNJ","MA","UNH","XOM",
        # Europa
        "NESN.SW","NOVN.SW","ROG.SW","SAP.DE","SIE.DE","ASML.AS","OR.PA","MC.PA","AIR.PA","ULVR.L",
        # Asia
        "TM","SONY","TSM","BABA","TCEHY","005930.KS","9984.T","NTDOY","HMC","SMFG",
        # Fornybar / Energi
        "NEE","BP.L","SHEL.L","EQNR.OL","ENI.MI","TOTF.PA",
        # Krypto-relatert / Tech
        "COIN","MSTR","SQ","PYPL","SHOP"
    ],

    # ======== 💱 VALUTA ========
    "💱 Valuta (Forex – hovedpar)": [
        "EURUSD=X","USDJPY=X","GBPUSD=X","AUDUSD=X","USDCAD=X","USDCHF=X","NZDUSD=X"
    ],
    "💱 Valuta (Nordic & Exotic)": [
        "EURNOK=X","USDNOK=X","SEKNOK=X","EURSEK=X","USDSEK=X",
        "EURDKK=X","USDTRY=X","USDZAR=X","USDCNH=X","USDPLN=X","USDMXN=X","USDINR=X"
    ],

    # ======== ⚙️ RÅVARER ========
    "⚙️ Råvarer – Energi": [
        "CL=F","BZ=F","NG=F","RB=F","HO=F","QM=F","QG=F"
    ],
    "🪨 Råvarer – Metaller": [
        "GC=F","SI=F","HG=F","PL=F","PA=F","ALI=F","ZNC=F","NID=F","AL=F","NI=F"
    ],
    "🌾 Råvarer – Landbruk": [
        "ZC=F","ZW=F","ZS=F","KC=F","CC=F","CT=F","SB=F","LBS=F","OJ=F","FC=F","LC=F","LH=F"
    ],

    # ======== 🪙 Krypto ========
    "🪙 Krypto": [
        "BTC-USD","ETH-USD","SOL-USD","XRP-USD","ADA-USD","BNB-USD","DOGE-USD",
        "AVAX-USD","DOT-USD","LTC-USD","LINK-USD","ATOM-USD","TON-USD","NEAR-USD","HBAR-USD"
    ],
}

# -----------------------------
# ⚙️ Sidebar – Favorittfunksjon
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("⭐ Favoritter")

fav_input = st.sidebar.text_area(
    "Mine favoritt-tickere (komma- eller linje-separert)",
    value=", ".join(st.session_state["favorites"]),
    height=80
)

if st.sidebar.button("💾 Lagre som favoritter"):
    st.session_state["favorites"] = [
        t.strip().upper()
        for chunk in fav_input.split("\n")
        for t in chunk.split(",")
        if t.strip()
    ]
    st.sidebar.success("Favoritter oppdatert! Du finner dem i preset-listen øverst.")


# -----------------------------
# Hjelpere – robust pris, indikatorer, labels
# -----------------------------
def get_close_series(df_raw: pd.DataFrame) -> pd.Series:
    """
    Returner en ren 1D float Series med Close-priser uansett hvordan df_raw ser ut.
    Håndterer single-level, MultiIndex, 'Adj Close' og tilfeller der en DataFrame returneres.
    """
    import pandas as pd
    import numpy as np

    if not isinstance(df_raw, pd.DataFrame):
        return pd.Series(dtype=float)

    # 1) Single-level
    for candidate in ["Close", "Adj Close", "close", "adjclose", "AdjClose"]:
        if candidate in df_raw.columns:
            s = df_raw[candidate]
            if isinstance(s, pd.DataFrame):
                if s.shape[1] > 0:
                    s = s.iloc[:, 0]
                else:
                    return pd.Series(np.nan, index=df_raw.index, dtype=float)
            return pd.to_numeric(s, errors="coerce").astype(float)

    # 2) MultiIndex
    if hasattr(df_raw.columns, "nlevels") and df_raw.columns.nlevels > 1:
        for col in df_raw.columns:
            if isinstance(col, tuple) and any(str(part).lower() == "close" for part in col):
                s = df_raw[col]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0] if s.shape[1] > 0 else pd.Series(np.nan, index=df_raw.index)
                return pd.to_numeric(s, errors="coerce").astype(float)
        for col in df_raw.columns:
            if isinstance(col, tuple) and any(str(part).lower() in ("adj close", "adjclose") for part in col):
                s = df_raw[col]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0] if s.shape[1] > 0 else pd.Series(np.nan, index=df_raw.index)
                return pd.to_numeric(s, errors="coerce").astype(float)

    # 3) Fallback
    return pd.Series(np.nan, index=df_raw.index, dtype=float)


def add_indicators(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Bygger feature-set med robust uthenting av 'Close'."""
    close = get_close_series(df_raw)
    out = pd.DataFrame(index=close.index)
    out["Close"] = close

    out["ret1"]   = close.pct_change(1)
    out["ret3"]   = close.pct_change(3)
    out["ret5"]   = close.pct_change(5)
    out["ma5"]    = close.rolling(5).mean()
    out["ma20"]   = close.rolling(20).mean()
    out["vol10"]  = out["ret1"].rolling(10).std()
    out["trend20"] = (out["ma20"] - out["ma5"]) / out["ma20"]

    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs    = gain / loss
    out["rsi14"] = 100 - (100 / (1 + rs))

    out["ema20"]   = close.ewm(span=20, adjust=False).mean()
    out["ema50"]   = close.ewm(span=50, adjust=False).mean()
    out["ema_gap"] = (out["ema20"] - out["ema50"]) / out["ema50"]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    out["macd"]      = macd
    out["macd_sig"]  = sig
    out["macd_hist"] = macd - sig

    ma20  = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    out["bb_pct"]   = (close - lower) / (upper - lower)
    out["bb_width"] = (upper - lower) / ma20

    return out


def make_label_eps(df_raw: pd.DataFrame, horizon_days: int, eps_pct: float) -> pd.Series:
    """
    1/0/NaN basert på fremtidig avkastning og eps-terskel (%).
    """
    close = get_close_series(df_raw)
    fwd   = close.shift(-int(horizon_days)) / close - 1.0
    eps   = float(eps_pct) / 100.0
    arr = np.where(fwd > eps, 1, np.where(fwd < -eps, 0, np.nan))
    arr = np.asarray(arr, dtype="float64").ravel()
    return pd.Series(arr, index=close.index, name=f"Target_{int(horizon_days)}")


FEATURES_ALL = [
    "ret1","ret3","ret5","ma5","ma20","vol10","trend20","rsi14",
    "ema20","ema50","ema_gap","bb_pct","bb_width","macd","macd_sig","macd_hist"
]

# -----------------------------
# Læring / walk-forward
# -----------------------------
def walkforward_fit_predict(X: pd.DataFrame, y: pd.Series):
    """
    Expanding walk-forward: returnerer proba (Series), mean acc, mean auc, optimal cutoff.
    """
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


# -----------------------------
# Datahentere
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_history(ticker: str, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    return df


# -----------------------------
# Analyse – flerhorisont
# -----------------------------
def analyze_ticker_multi(df_raw: pd.DataFrame,
                         horizons: dict,
                         eps_map: dict,
                         feats_map: dict) -> dict:
    """
    df_raw: råpriser fra yfinance
    horizons: {"A": daysA, "B": daysB, "C": daysC}
    eps_map:  {"A": epsA,  "B": epsB,  "C": epsC}
    feats_map: {"A": [features...], ...}
    """
    out = {}

    if df_raw is None or df_raw.empty:
        for k in horizons.keys():
            out[k] = {"proba": pd.Series(dtype=float), "acc": np.nan, "auc": np.nan,
                      "opt_thr": 0.5, "last_date": None}
        return out

    feats_all = add_indicators(df_raw)

    for key, H in horizons.items():
        y = make_label_eps(df_raw, H, eps_map.get(key, 0.0))
        # Feature-sett for denne horisonten:
        sel = feats_map.get(key, FEATURES_ALL)
        # Kryss mot det som finnes og har data
        avail = [c for c in sel if c in feats_all.columns and feats_all[c].notna().any()]

        if not avail or y is None or len(y) != len(feats_all):
            out[key] = {"proba": pd.Series(dtype=float), "acc": np.nan, "auc": np.nan,
                        "opt_thr": 0.5, "last_date": None}
            continue

        pack = pd.concat([feats_all[avail], y], axis=1).dropna()
        if pack.empty or len(pack) < 60:
            out[key] = {
                "proba": pd.Series(0.5, index=(pack.index if len(pack) else feats_all.index), name="proba"),
                "acc": np.nan, "auc": np.nan, "opt_thr": 0.5,
                "last_date": (pack.index[-1] if len(pack) else None),
            }
            continue

        X = pack[avail]
        yv = pack[y.name]
        if len(np.unique(yv.values.astype(int))) < 2:
            neut = pd.Series(0.5, index=pack.index, name="proba")
            out[key] = {"proba": neut, "acc": np.nan, "auc": np.nan,
                        "opt_thr": 0.5, "last_date": pack.index[-1]}
            continue

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


def expected_date_business(last_date: pd.Timestamp, horizon_days: int) -> str:
    """Legger til **handelsdager** (ikke kalenderdager)."""
    if last_date is None or pd.isna(last_date):
        return "—"
    # +1 BusinessDay fordi modellen gir sannsynlighet for neste H-dagers frem i tid
    return (pd.to_datetime(last_date) + BusinessDay(int(horizon_days))).strftime("%Y-%m-%d")


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
    start_date = st.sidebar.date_input("Startdato", pd.to_datetime("2019-01-01"))
with col_d2:
    end_date = st.sidebar.date_input("Sluttdato", pd.Timestamp.today())

st.sidebar.markdown("---")
st.sidebar.subheader("Horisont (handelsdager)")

dA = st.sidebar.slider("Horisont A (vises som '3 dager')", 1, 30, 3)
dB = st.sidebar.slider("Horisont B (vises som '7 dager')", 1, 30, 7)
dC = st.sidebar.slider("Horisont C (vises som '14 dager')", 1, 30, 14)
HORIZONS = {"A": dA, "B": dB, "C": dC}

st.sidebar.markdown("---")
st.sidebar.subheader("Støyfilter (eps, %)")

epsA = st.sidebar.number_input("EPS A (%)", value=1.00, min_value=0.0, max_value=50.0, step=0.10)
epsB = st.sidebar.number_input("EPS B (%)", value=3.00, min_value=0.0, max_value=50.0, step=0.10)
epsC = st.sidebar.number_input("EPS C (%)", value=5.00, min_value=0.0, max_value=50.0, step=0.10)
EPS_MAP = {"A": epsA, "B": epsB, "C": epsC}

st.sidebar.markdown("---")
st.sidebar.subheader("Features")

same_feats = st.sidebar.checkbox("Bruk samme features på alle horisonter", value=True)
if same_feats:
    feats_common = st.sidebar.multiselect("Velg features (gjelder A/B/C)", FEATURES_ALL, default=FEATURES_ALL)
    FEATS_MAP = {"A": feats_common, "B": feats_common, "C": feats_common}
else:
    fA = st.sidebar.multiselect("Features A", FEATURES_ALL, default=FEATURES_ALL)
    fB = st.sidebar.multiselect("Features B", FEATURES_ALL, default=FEATURES_ALL)
    fC = st.sidebar.multiselect("Features C", FEATURES_ALL, default=FEATURES_ALL)
    FEATS_MAP = {"A": fA, "B": fB, "C": fC}

st.sidebar.markdown("---")
st.sidebar.subheader("Terskler for anbefaling")

same_thr = st.sidebar.checkbox("Bruk samme terskler på alle horisonter", value=True)
if same_thr:
    b_all = st.sidebar.slider("KJØP hvis sannsynlighet >", 0.50, 0.90, 0.60, 0.01)
    s_all = st.sidebar.slider("SELG hvis sannsynlighet <", 0.10, 0.50, 0.40, 0.01)
    THR = {"A": (b_all, s_all), "B": (b_all, s_all), "C": (b_all, s_all)}
else:
    bA = st.sidebar.slider("A • KJØP >", 0.50, 0.90, 0.60, 0.01); sA = st.sidebar.slider("A • SELG <", 0.10, 0.50, 0.40, 0.01)
    bB = st.sidebar.slider("B • KJØP >", 0.50, 0.90, 0.60, 0.01); sB = st.sidebar.slider("B • SELG <", 0.10, 0.50, 0.40, 0.01)
    bC = st.sidebar.slider("C • KJØP >", 0.50, 0.90, 0.60, 0.01); sC = st.sidebar.slider("C • SELG <", 0.10, 0.50, 0.40, 0.01)
    THR = {"A": (bA, sA), "B": (bB, sB), "C": (bC, sC)}

st.sidebar.markdown("---")
want_excel = st.sidebar.checkbox("Eksporter Excel (flere ark)", value=False)

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
            raw = fetch_history(t, start=start_date, end=end_date)
        except Exception:
            raw = pd.DataFrame()

        if raw is None or raw.empty:
            results.append({
                "Ticker": t,
                "Prob_A": np.nan, "Rec_A": "HOLD", "Date_A": "—", "EPS_A(%)": np.nan,
                "Prob_B": np.nan, "Rec_B": "HOLD", "Date_B": "—", "EPS_B(%)": np.nan,
                "Prob_C": np.nan, "Rec_C": "HOLD", "Date_C": "—", "EPS_C(%)": np.nan,
                "Acc": np.nan, "AUC": np.nan
            })
            progress.progress(i/len(tickers))
            continue

        pack = analyze_ticker_multi(raw, HORIZONS, EPS_MAP, FEATS_MAP)

        def last_proba(key, default=np.nan):
            try:
                s = pack[key]["proba"]
                if len(s) == 0:
                    return default
                v = float(s.iloc[-1])
                return v if not np.isnan(v) else default
            except Exception:
                return default

        pA, pB, pC = last_proba("A"), last_proba("B"), last_proba("C")
        dA_str = expected_date_business(pack["A"]["last_date"], HORIZONS["A"])
        dB_str = expected_date_business(pack["B"]["last_date"], HORIZONS["B"])
        dC_str = expected_date_business(pack["C"]["last_date"], HORIZONS["C"])

        bA, sA = THR["A"]; bB, sB = THR["B"]; bC, sC = THR["C"]
        rA = rec_from_prob(pA, max(bA, pack["A"]["opt_thr"]), sA)
        rB = rec_from_prob(pB, max(bB, pack["B"]["opt_thr"]), sB)
        rC = rec_from_prob(pC, max(bC, pack["C"]["opt_thr"]), sC)

        accs = [pack[k]["acc"] for k in ["A","B","C"] if not np.isnan(pack[k]["acc"])]
        aucs = [pack[k]["auc"] for k in ["A","B","C"] if not np.isnan(pack[k]["auc"])]
        acc = float(np.mean(accs)) if accs else np.nan
        auc = float(np.mean(aucs)) if aucs else np.nan

        results.append({
            "Ticker": t,
            "Prob_A": pA, "Rec_A": rA, "Date_A": dA_str, "EPS_A(%)": EPS_MAP["A"],
            "Prob_B": pB, "Rec_B": rB, "Date_B": dB_str, "EPS_B(%)": EPS_MAP["B"],
            "Prob_C": pC, "Rec_C": rC, "Date_C": dC_str, "EPS_C(%)": EPS_MAP["C"],
            "Acc": acc, "AUC": auc
        })

        progress.progress(i/len(tickers))

    status.empty()
    progress.empty()

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
# Visning
# -----------------------------
if run:
    df = pd.DataFrame(results)

    num_cols = ["Prob_A","Prob_B","Prob_C","Acc","AUC"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader(f"🟩 {HORIZONS['A']} handelsdager frem")
        dA = df[["Ticker", "Prob_A", "Rec_A", "Date_A", "EPS_A(%)", "Acc"]].copy()
        dA = dA.sort_values("Prob_A", ascending=False)
        st.dataframe(style_df(dA, {"Prob_A":"{:.2%}", "Acc":"{:.2%}", "AUC":"{:.3f}"}), use_container_width=True)

    with c2:
        st.subheader(f"🟦 {HORIZONS['B']} handelsdager frem")
        dB = df[["Ticker", "Prob_B", "Rec_B", "Date_B", "EPS_B(%)", "Acc"]].copy()
        dB = dB.sort_values("Prob_B", ascending=False)
        st.dataframe(style_df(dB, {"Prob_B":"{:.2%}", "Acc":"{:.2%}", "AUC":"{:.3f}"}), use_container_width=True)

    with c3:
        st.subheader(f"🟧 {HORIZONS['C']} handelsdager frem")
        dC = df[["Ticker", "Prob_C", "Rec_C", "Date_C", "EPS_C(%)", "Acc"]].copy()
        dC = dC.sort_values("Prob_C", ascending=False)
        st.dataframe(style_df(dC, {"Prob_C":"{:.2%}", "Acc":"{:.2%}", "AUC":"{:.3f}"}), use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Sammenligningstabell (alle horisonter)")

    cmp_df = df[[
        "Ticker",
        "Prob_A","Rec_A","Date_A","EPS_A(%)",
        "Prob_B","Rec_B","Date_B","EPS_B(%)",
        "Prob_C","Rec_C","Date_C","EPS_C(%)",
        "Acc","AUC"
    ]].sort_values("Prob_A", ascending=False)

    st.dataframe(
        style_df(
            cmp_df,
            {
                "Prob_A": "{:.2%}", "Prob_B": "{:.2%}", "Prob_C": "{:.2%}",
                "Acc": "{:.2%}", "AUC": "{:.3f}"
            }
        ),
        use_container_width=True
    )

    # -------------------------
    # Detalj: graf per ticker
    # -------------------------
    st.markdown("---")
    st.subheader("📊 Detaljvisning (pris + sannsynlighet A/B/C)")
    sel_list = df["Ticker"].tolist() if not df.empty else []
    sel = st.selectbox("Velg ticker", sel_list)
    if sel:
        try:
            raw = fetch_history(sel, start=start_date, end=end_date)
            feats = add_indicators(raw)
            pack = analyze_ticker_multi(raw, HORIZONS, EPS_MAP, FEATS_MAP)
            plot_df = pd.DataFrame({
                "Close": feats["Close"],
                "Prob_A": pack["A"]["proba"].reindex(feats.index),
                "Prob_B": pack["B"]["proba"].reindex(feats.index),
                "Prob_C": pack["C"]["proba"].reindex(feats.index),
            }).dropna(subset=["Close"])

            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(plot_df.index, plot_df["Close"])
            ax1.set_xlabel("Dato")
            ax1.set_ylabel("Pris")

            ax2 = ax1.twinx()
            if "Prob_A" in plot_df: ax2.plot(plot_df.index, plot_df["Prob_A"], alpha=0.9)
            if "Prob_B" in plot_df: ax2.plot(plot_df.index, plot_df["Prob_B"], alpha=0.9)
            if "Prob_C" in plot_df: ax2.plot(plot_df.index, plot_df["Prob_C"], alpha=0.9)
            ax2.axhline(0.5, linestyle="--", alpha=0.6)
            ax2.set_ylabel("Sannsynlighet")

            plt.title(f"{sel}: Pris + sannsynlighet (A/B/C)")
            st.pyplot(fig)
        except Exception as e:
            st.info(f"Kunne ikke vise graf for {sel}: {e}")

    # -------------------------
    # Eksport
    # -------------------------
    st.markdown("---")
    st.subheader("📤 Eksport")

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Last ned CSV (alle horisonter)",
        data=csv_bytes,
        file_name=f"scan_v6_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
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
    st.info("Velg/skriv tickere i sidepanelet og trykk **🔎 Skann og sammenlign (A/B/C)** for å starte.")





























