# app_v6.py
# 📈 Aksjeanalyse – Pro v6 Dashboard
# Multi-horisont (A/B/C) med justerbare handelsdager og EPS per horisont
# Build: v6.4 – oktober 2025

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from pandas.tseries.offsets import BusinessDay

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score


# -----------------------------
# Sideoppsett + enkel styling
# -----------------------------
st.set_page_config(page_title="Aksjeanalyse – Pro v6 Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #0E1117; }
    .stMarkdown, .stText, .stDataFrame { color: #E6E6E6 !important; }
    .soft-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius:16px; padding:12px 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Aksjeanalyse – Pro v6 Dashboard")
st.caption("Multi-horisont modell (A/B/C) • justerbare handelsdager • EPS per horisont • Build: v6.4 – oktober 2025")


# -----------------------------
# Presets
# -----------------------------
PRESETS = {
    "OBX (Norge)": [
        "EQNR.OL","DNB.OL","MOWI.OL","NHY.OL","TEL.OL","ORK.OL","YAR.OL","KOG.OL","AKRBP.OL","TGS.OL"
    ],
    "USA – Megacaps": [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","XOM","UNH","V","JNJ","WMT","PG","MA","AVGO","HD","MRK","PEP"
    ],
}

DEFAULT_TICKERS = ["EQNR.OL", "DNB.OL", "MOWI.OL", "NHY.OL", "TEL.OL"]


# -----------------------------
# Sidebar – kontroller
# -----------------------------
st.sidebar.header("⚙️ Innstillinger")

preset = st.sidebar.selectbox("Hurtigvalg gruppe", list(PRESETS.keys()))
if st.sidebar.button("📥 Last inn preset"):
    st.session_state["tickers"] = PRESETS[preset]

tickers_str = st.sidebar.text_area(
    "Tickere (komma/linjer). Tomt felt = bruk valgt preset.",
    value=", ".join(st.session_state.get("tickers", DEFAULT_TICKERS)), height=120
)
TICKERS = [t.strip().upper() for chunk in tickers_str.split("\n") for t in chunk.split(",") if t.strip()]

col_dates = st.sidebar.columns(2)
with col_dates[0]:
    START = st.date_input("Startdato", pd.to_datetime("2019-01-01"))
with col_dates[1]:
    END = st.date_input("Sluttdato", pd.Timestamp.today())

st.sidebar.markdown("---")

st.sidebar.subheader("Horisont (handelsdager)")
cA, cB, cC = st.sidebar.columns(3)
with cA:
    dA = st.number_input("A", min_value=1, max_value=60, value=3, step=1, help="Antall handelsdager for horisont A")
with cB:
    dB = st.number_input("B", min_value=1, max_value=60, value=7, step=1, help="Antall handelsdager for horisont B")
with cC:
    dC = st.number_input("C", min_value=1, max_value=60, value=14, step=1, help="Antall handelsdager for horisont C")

HORIZONS = {"A": int(dA), "B": int(dB), "C": int(dC)}

st.sidebar.markdown("---")
st.sidebar.subheader("Støyfilter (eps, %)")

use_same_eps = st.sidebar.checkbox("Bruk samme eps på alle horisonter", value=False)
if use_same_eps:
    eps_all = st.sidebar.number_input("EPS alle", value=0.50, min_value=0.0, max_value=50.0, step=0.10)
    EPS_MAP = {"A": eps_all, "B": eps_all, "C": eps_all}
else:
    cA, cB, cC = st.sidebar.columns(3)
    with cA:
        epsA = st.number_input("EPS A (%)", value=0.50, min_value=0.0, max_value=50.0, step=0.10)
    with cB:
        epsB = st.number_input("EPS B (%)", value=1.00, min_value=0.0, max_value=50.0, step=0.10)
    with cC:
        epsC = st.number_input("EPS C (%)", value=3.00, min_value=0.0, max_value=50.0, step=0.10)
    EPS_MAP = {"A": epsA, "B": epsB, "C": epsC}

st.sidebar.markdown("---")
st.sidebar.subheader("Features")

FEATURES_ALL = [
    "ret1","ret3","ret5","ma5","ma20","vol10","trend20","rsi14",
    "ema20","ema50","ema_gap","bb_pct","bb_width","macd","macd_sig","macd_hist"
]

use_same_feats = st.sidebar.checkbox("Bruk samme features på alle horisonter", value=True)
if use_same_feats:
    FEATS_COMMON = st.sidebar.multiselect("Velg features", FEATURES_ALL, default=FEATURES_ALL)
    FEATS_MAP = {"A": FEATS_COMMON, "B": FEATS_COMMON, "C": FEATS_COMMON}
else:
    cA, cB, cC = st.sidebar.columns(3)
    with cA:
        featsA = st.multiselect("Features A", FEATURES_ALL, default=FEATURES_ALL)
    with cB:
        featsB = st.multiselect("Features B", FEATURES_ALL, default=FEATURES_ALL)
    with cC:
        featsC = st.multiselect("Features C", FEATURES_ALL, default=FEATURES_ALL)
    FEATS_MAP = {"A": featsA, "B": featsB, "C": featsC}

st.sidebar.markdown("---")
st.sidebar.subheader("Terskler for anbefaling")

thr_box = st.sidebar.checkbox("Bruk samme terskler på alle horisonter", value=True)
def thr_pair(lbl_buy, lbl_sell, bdef=0.60, sdef=0.40):
    b = st.sidebar.slider(lbl_buy, 0.50, 0.90, bdef, 0.01)
    s = st.sidebar.slider(lbl_sell, 0.10, 0.50, sdef, 0.01)
    return b, s

if thr_box:
    b_all, s_all = thr_pair("KJØP hvis sannsynlighet >", "SELG hvis sannsynlighet <", 0.60, 0.40)
    THR = {"A": (b_all, s_all), "B": (b_all, s_all), "C": (b_all, s_all)}
else:
    bA, sA = thr_pair("A: KJØP >", "A: SELG <", 0.60, 0.40)
    bB, sB = thr_pair("B: KJØP >", "B: SELG <", 0.60, 0.40)
    bC, sC = thr_pair("C: KJØP >", "C: SELG <", 0.60, 0.40)
    THR = {"A": (bA, sA), "B": (bB, sB), "C": (bC, sC)}

st.sidebar.markdown("---")
want_excel = st.sidebar.checkbox("Eksporter Excel (flere ark)", value=False)
st.sidebar.caption("Tips: Norske aksjer bruker .OL (EQNR.OL).")


# -----------------------------
# Datahjelpere (med caching)
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_history(ticker: str, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    # Standardiser index som DatetimeIndex uten tz
    try:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    except Exception:
        pass
    return df


# -----------------------------
# ROBUST Close + indikatorer
# -----------------------------
def get_close_series(df: pd.DataFrame) -> pd.Series:
    """
    Returner en 1D Serie med 'Close' (eller 'Adj Close') uansett om input har
    MultiIndex/2D/rare kolonnestrukturer. Fyller med NaN dersom ingenting finnes.
    """
    if df is None or df.empty:
        return pd.Series(index=pd.Index([], name="Date"), dtype="float64")

    if "Close" in df.columns:
        c = df["Close"]
    elif "Adj Close" in df.columns:
        c = df["Adj Close"]
    elif isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        if "Close" in lvl0:
            c = df.xs("Close", level=0, axis=1)
        elif "Adj Close" in lvl0:
            c = df.xs("Adj Close", level=0, axis=1)
        else:
            c = pd.Series(index=df.index, dtype="float64")
    else:
        c = pd.Series(index=df.index, dtype="float64")

    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0] if c.shape[1] > 0 else pd.Series(index=df.index, dtype="float64")

    c = pd.to_numeric(pd.Series(c, index=df.index).squeeze(), errors="coerce")
    c.name = "Close"
    return c


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Bygger felles indikatorer. Sikrer at Close alltid blir 1D numerisk Serie."""
    out = df.copy()

    close = get_close_series(out)

    ret1 = close.pct_change()
    out["Close"] = close
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

    # EMA / MACD
    out["ema20"] = close.ewm(span=20, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()
    out["ema_gap"] = (out["ema20"] - out["ema50"]) / out["ema50"]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd
    out["macd_sig"] = sig
    out["macd_hist"] = macd - sig

    # Bollinger% og bredde
    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    up = ma20 + 2 * sd20
    lo = ma20 - 2 * sd20
    out["bb_pct"] = (close - lo) / (up - lo)
    out["bb_width"] = (up - lo) / ma20

    return out


# -----------------------------
# Labels og modellering
# -----------------------------
def make_label_eps(df_feats: pd.DataFrame, horizon_days: int, eps_pct: float) -> pd.Series:
    """
    Binær label med nøytral sone (eps): 1 hvis fremtidig avkastning > +eps, 0 hvis < -eps, NaN ellers.
    horizon_days tolkes som handelsdager (vi bruker indeksens trinn ~ business days).
    """
    eps = eps_pct / 100.0
    c = df_feats["Close"].astype(float)
    # shift(-horizon_days) ~ frem i tid
    fwd = c.shift(-horizon_days) / c - 1.0
    arr = np.where(fwd > eps, 1, np.where(fwd < -eps, 0, np.nan))
    return pd.Series(arr, index=df_feats.index, name=f"Target_{horizon_days}")


def walkforward_fit_predict(X: pd.DataFrame, y: pd.Series):
    """
    Expanding walk-forward for valid., og tren endelig modell for pred-proba over hele serien.
    Returnerer (proba Serie, val_acc, val_auc, opt_thr)
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

        cls = np.unique(ytr.dropna().astype(int))
        if len(Xva)==0 or len(cls) < 2:
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

    # Tren på hele serien (for proba_full)
    scaler = RobustScaler().fit(X)
    Xs = scaler.transform(X)
    base = GradientBoostingClassifier(random_state=0)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(Xs, y.astype(int))
    proba_full = pd.Series(clf.predict_proba(Xs)[:, 1], index=X.index)

    acc = float(np.nanmean(val_accs)) if len(val_accs) else np.nan
    auc = float(np.nanmean(val_aucs)) if len(val_aucs) else np.nan
    opt_thr = float(np.nanmean(thrs)) if len(thrs) else 0.5

    return proba_full, acc, auc, opt_thr


def fit_predict_single_horizon(df_raw: pd.DataFrame, H: int, eps: float, feat_list: list[str]):
    """
    Bygg features, lag label, tren modell, retur (proba, acc, auc, opt_thr, last_date)
    """
    out_empty = (pd.Series(dtype=float), np.nan, np.nan, 0.5, None)

    if df_raw is None or df_raw.empty:
        return out_empty

    feats = add_indicators(df_raw)

    # Behold bare features som faktisk finnes (med minst én ikke-NaN)
    feat_avail = [c for c in feat_list if c in feats.columns and feats[c].notna().any()]
    if len(feat_avail) == 0:
        return out_empty

    y = make_label_eps(feats, H, eps)
    if y is None or len(y) != len(feats):
        return out_empty

    # Pakke og dropp rader med NaN i features + y
    pack = pd.concat([feats[feat_avail], y], axis=1)
    drop_cols = feat_avail + [y.name]
    try:
        pack = pack.dropna(subset=drop_cols)
    except Exception:
        # fall-back hvis subset skaper trøbbel
        pack = pack.dropna()

    if pack.empty or len(pack) < 60:
        last_date = pack.index[-1] if len(pack) else None
        return (pd.Series(0.5, index=(pack.index if len(pack) else feats.index)), np.nan, np.nan, 0.5, last_date)

    X = pack.loc[:, feat_avail]
    yv = pack[y.name].astype(int)

    if len(np.unique(yv.values)) < 2:
        last_date = pack.index[-1]
        return (pd.Series(0.5, index=pack.index), np.nan, np.nan, 0.5, last_date)

    proba_full, acc, auc, opt_thr = walkforward_fit_predict(X, yv)
    last_date = pack.index[-1]
    return (proba_full, acc, auc, opt_thr, last_date)


def analyze_ticker_multi(df_raw: pd.DataFrame, HORIZONS: dict, EPS_MAP: dict, FEATS_MAP: dict):
    result = {}
    for key, H in HORIZONS.items():
        eps = float(EPS_MAP.get(key, 0.0))
        feat_list = list(FEATS_MAP.get(key, FEATURES_ALL))
        proba, acc, auc, opt_thr, last_date = fit_predict_single_horizon(df_raw, H, eps, feat_list)
        result[key] = {
            "proba": proba,
            "acc": acc,
            "auc": auc,
            "opt_thr": opt_thr,
            "last_date": last_date,
        }
    return result


# -----------------------------
# Visning og kjøring
# -----------------------------
def rec_from_prob(prob: float, buy_thr: float, sell_thr: float) -> str:
    if np.isnan(prob):
        return "HOLD"
    if prob > buy_thr:
        return "BUY"
    if prob < sell_thr:
        return "SELL"
    return "HOLD"


def expected_date(last_date: pd.Timestamp, business_days: int) -> str:
    if last_date is None or pd.isna(last_date):
        return "—"
    try:
        return (pd.to_datetime(last_date) + business_days * BusinessDay()).strftime("%Y-%m-%d")
    except Exception:
        return "—"


def last_proba(d, key: str, default=0.5):
    try:
        s = d[key]["proba"]
        if len(s) == 0:
            return default
        v = float(s.iloc[-1])
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default


run = st.button("🔎 Skann og sammenlign (A/B/C)")
rows = []

if run:
    progress = st.progress(0.0)
    status = st.empty()

    for i, t in enumerate(TICKERS, start=1):
        status.write(f"Henter og analyserer: {t} ({i}/{len(TICKERS)})")
        try:
            raw = fetch_history(t, start=START, end=END)
        except Exception:
            raw = pd.DataFrame()

        if raw is None or raw.empty:
            rows.append({
                "Ticker": t,
                "Prob_A": np.nan, "Rec_A": "HOLD", "Date_A": "—",
                "Prob_B": np.nan, "Rec_B": "HOLD", "Date_B": "—",
                "Prob_C": np.nan, "Rec_C": "HOLD", "Date_C": "—",
                "Acc": np.nan, "AUC": np.nan, "Composite": np.nan
            })
            progress.progress(i / len(TICKERS))
            continue

        pack = analyze_ticker_multi(raw, HORIZONS, EPS_MAP, FEATS_MAP)

        pA = last_proba(pack, "A")
        pB = last_proba(pack, "B")
        pC = last_proba(pack, "C")

        dA_str = expected_date(pack["A"]["last_date"], HORIZONS["A"])
        dB_str = expected_date(pack["B"]["last_date"], HORIZONS["B"])
        dC_str = expected_date(pack["C"]["last_date"], HORIZONS["C"])

        bA, sA = THR["A"]; bB, sB = THR["B"]; bC, sC = THR["C"]

        rA = rec_from_prob(pA, max(bA, pack["A"]["opt_thr"]), sA)
        rB = rec_from_prob(pB, max(bB, pack["B"]["opt_thr"]), sB)
        rC = rec_from_prob(pC, max(bC, pack["C"]["opt_thr"]), sC)

        probs = [x for x in [pA, pB, pC] if not np.isnan(x)]
        comp = float(np.mean(probs)) if probs else np.nan

        accs = [pack[k]["acc"] for k in ["A", "B", "C"] if not np.isnan(pack[k]["acc"])]
        aucs = [pack[k]["auc"] for k in ["A", "B", "C"] if not np.isnan(pack[k]["auc"])]
        acc = float(np.mean(accs)) if accs else np.nan
        auc = float(np.mean(aucs)) if aucs else np.nan

        rows.append({
            "Ticker": t,
            "Prob_A": pA, "Rec_A": rA, "Date_A": dA_str,
            "Prob_B": pB, "Rec_B": rB, "Date_B": dB_str,
            "Prob_C": pC, "Rec_C": rC, "Date_C": dC_str,
            "Acc": acc, "AUC": auc, "Composite": comp
        })

        progress.progress(i / len(TICKERS))

    status.empty()
    progress.empty()

    df = pd.DataFrame(rows)

    # tallformat
    for col in ["Prob_A","Prob_B","Prob_C","Acc","AUC","Composite"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader(f"🟩 A ({HORIZONS['A']} hd)")
        dA = df[["Ticker", "Prob_A", "Rec_A", "Date_A", "Acc"]].sort_values("Prob_A", ascending=False)
        st.dataframe(dA.style.format({"Prob_A":"{:.2%}","Acc":"{:.2%}"}), use_container_width=True)

    with c2:
        st.subheader(f"🟦 B ({HORIZONS['B']} hd)")
        dB = df[["Ticker", "Prob_B", "Rec_B", "Date_B", "Acc"]].sort_values("Prob_B", ascending=False)
        st.dataframe(dB.style.format({"Prob_B":"{:.2%}","Acc":"{:.2%}"}), use_container_width=True)

    with c3:
        st.subheader(f"🟧 C ({HORIZONS['C']} hd)")
        dC = df[["Ticker", "Prob_C", "Rec_C", "Date_C", "Acc"]].sort_values("Prob_C", ascending=False)
        st.dataframe(dC.style.format({"Prob_C":"{:.2%}","Acc":"{:.2%}"}), use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Sammenligningstabell (alle horisonter)")
    cmp = df[[
        "Ticker",
        "Prob_A","Rec_A","Date_A",
        "Prob_B","Rec_B","Date_B",
        "Prob_C","Rec_C","Date_C",
        "Acc","AUC","Composite"
    ]].sort_values("Composite", ascending=False)

    st.dataframe(cmp.style.format({
        "Prob_A":"{:.2%}","Prob_B":"{:.2%}","Prob_C":"{:.2%}",
        "Acc":"{:.2%}","AUC":"{:.3f}","Composite":"{:.2%}"
    }), use_container_width=True)

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
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            dA.to_excel(writer, index=False, sheet_name=f"A_{HORIZONS['A']}d")
            dB.to_excel(writer, index=False, sheet_name=f"B_{HORIZONS['B']}d")
            dC.to_excel(writer, index=False, sheet_name=f"C_{HORIZONS['C']}d")
            cmp.to_excel(writer, index=False, sheet_name="Comparison")
        buf.seek(0)
        st.download_button(
            "⬇️ Last ned Excel (A/B/C/Comparison)",
            data=buf.getvalue(),
            file_name=f"scan_v6_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    st.info("Velg/skriv tickere i sidepanelet og trykk **🔎 Skann og sammenlign** for å starte.")


























