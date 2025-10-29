# app_v6.py — Aksjeanalyse Pro v6 (robust)
# v6.4 – fleksible horisonter (A/B/C), egne eps-filtre og feature-valg per horisont

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
# UI & presets
# -----------------------------
st.set_page_config(page_title="Aksjeanalyse – Pro v6", layout="wide")
st.title("📈 Aksjeanalyse – Pro v6 Dashboard")
st.caption("Fleksible horisonter (A/B/C), uavhengige eps-filtre og feature-valg • v6.4")

PRESETS = {
    "OBX (Norge)": [
        "EQNR.OL","DNB.OL","MOWI.OL","NHY.OL","TEL.OL","ORK.OL","YAR.OL","KOG.OL",
        "AKRBP.OL","TGS.OL"
    ],
    "USA – Megacaps": ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","XOM"],
}

# Standard full liste av features som denne appen tilbyr
FEATURES_ALL = [
    "ret1","ret3","ret5","ma5","ma20","vol10","trend20","rsi14",
    "ema20","ema50","ema_gap","bb_pct","bb_width","macd","macd_sig","macd_hist",
]

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Innstillinger")

preset = st.sidebar.selectbox("Hurtigvalg gruppe", list(PRESETS.keys()))
if st.sidebar.button("📥 Last inn preset"):
    st.session_state["tickers"] = PRESETS[preset]

default_tickers = ["EQNR.OL","DNB.OL","MOWI.OL","NHY.OL","TEL.OL","AKRBP.OL","TGS.OL"]
tickers_text = st.sidebar.text_area(
    "Tickere (komma/linjer). Tomt felt = bruk valgt preset.",
    value=", ".join(st.session_state.get("tickers", default_tickers)),
    height=120
)
tickers = [t.strip().upper() for chunk in tickers_text.split("\n") for t in chunk.split(",") if t.strip()]

col_date1, col_date2 = st.sidebar.columns(2)
with col_date1:
    start_date = st.date_input("Startdato", pd.to_datetime("2019-01-01"))
with col_date2:
    end_date = st.date_input("Sluttdato", pd.Timestamp.today())

st.sidebar.markdown("---")

st.sidebar.subheader("Horisont (handelsdager)")
col_hA, col_hB, col_hC = st.sidebar.columns(3)
with col_hA:
    H_A = st.number_input("A", min_value=1, max_value=60, value=3, step=1, help="Antall børsdager (A)")
with col_hB:
    H_B = st.number_input("B", min_value=1, max_value=60, value=7, step=1, help="Antall børsdager (B)")
with col_hC:
    H_C = st.number_input("C", min_value=1, max_value=60, value=14, step=1, help="Antall børsdager (C)")

st.sidebar.subheader("Støyfilter (eps, %)")
epsA = st.sidebar.number_input("EPS A (%)", 0.0, 50.0, 1.00, 0.10,
                               help="Absolutt avkastning < eps% behandles som nøytral (A).")
epsB = st.sidebar.number_input("EPS B (%)", 0.0, 50.0, 3.00, 0.10,
                               help="Absolutt avkastning < eps% behandles som nøytral (B).")
epsC = st.sidebar.number_input("EPS C (%)", 0.0, 50.0, 5.00, 0.10,
                               help="Absolutt avkastning < eps% behandles som nøytral (C).")

HORIZONS = {"A": H_A, "B": H_B, "C": H_C}
EPS_MAP  = {"A": epsA, "B": epsB, "C": epsC}

st.sidebar.markdown("---")
st.sidebar.subheader("Features (innganger)")

same_feats = st.sidebar.checkbox("Bruk samme features på alle horisonter", value=True)

def _feat_multiselect(label, default=None):
    return st.sidebar.multiselect(label, FEATURES_ALL, default=default or FEATURES_ALL)

if same_feats:
    feats_all = _feat_multiselect("Velg features", default=FEATURES_ALL)
    FEATS_MAP = {"A": feats_all, "B": feats_all, "C": feats_all}
else:
    featsA = _feat_multiselect("Features for A", default=FEATURES_ALL)
    featsB = _feat_multiselect("Features for B", default=FEATURES_ALL)
    featsC = _feat_multiselect("Features for C", default=FEATURES_ALL)
    FEATS_MAP = {"A": featsA, "B": featsB, "C": featsC}

st.sidebar.markdown("---")
st.sidebar.subheader("Terskler for anbefaling")
same_thr = st.sidebar.checkbox("Bruk samme terskler på alle horisonter", value=True)

def _thr_pair(lbl_buy, lbl_sell, b0=0.60, s0=0.40):
    b = st.sidebar.slider(lbl_buy, 0.50, 0.90, b0, 0.01)
    s = st.sidebar.slider(lbl_sell, 0.10, 0.50, s0, 0.01)
    return b, s

if same_thr:
    b_all, s_all = _thr_pair("KJØP hvis sannsynlighet >", "SELG hvis sannsynlighet <", 0.60, 0.40)
    THR = {"A": (b_all, s_all), "B": (b_all, s_all), "C": (b_all, s_all)}
else:
    bA, sA = _thr_pair("A • KJØP >", "A • SELG <", 0.60, 0.40)
    bB, sB = _thr_pair("B • KJØP >", "B • SELG <", 0.60, 0.40)
    bC, sC = _thr_pair("C • KJØP >", "C • SELG <", 0.60, 0.40)
    THR = {"A": (bA, sA), "B": (bB, sB), "C": (bC, sC)}

st.sidebar.markdown("---")
want_excel = st.sidebar.checkbox("Eksporter Excel (flere ark)", value=False)
st.sidebar.caption("Dato for anbefaling = siste treningsdato + valgt antall **handelsdager**.")

# -----------------------------
# Hjelpefunksjoner (data & label)
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_history(ticker: str, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Bygger felles indikatorer. Sikrer Close er numerisk."""
    out = df.copy()
    if "Close" not in out.columns:
        out["Close"] = np.nan
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")

    close = out["Close"]
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

    # Bollinger % og bredde
    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    up = ma20 + 2*sd20
    lo = ma20 - 2*sd20
    out["bb_pct"] = (close - lo) / (up - lo)
    out["bb_width"] = (up - lo) / ma20

    return out

def make_label_eps(feats: pd.DataFrame, horizon_days: int, eps_pct: float) -> pd.Series:
    """Label: 1 hvis fwd > eps, 0 hvis fwd < -eps, ellers NaN."""
    eps = float(eps_pct) / 100.0
    close = pd.to_numeric(feats["Close"], errors="coerce")
    fwd = close.shift(-horizon_days) / close - 1.0
    arr = np.where(fwd > eps, 1, np.where(fwd < -eps, 0, np.nan))
    return pd.Series(arr.astype("float64"), index=feats.index, name=f"Target_{horizon_days}")

# -----------------------------
# Modellering – robust og defensiv
# -----------------------------
def fit_predict_single_horizon(df_raw: pd.DataFrame, horizon_days: int, eps: float, feature_list: list[str]):
    """
    Bygger features og label, trener robust modell og returnerer:
    proba_full (Serie), acc, auc, opt_thr, last_idx (siste treningsindeks)
    """
    feats = add_indicators(df_raw)
    y = make_label_eps(feats, horizon_days, eps)

    # Tilgjengelige features (eksisterer og har minst en ikke-NaN)
    avail = [c for c in feature_list if c in feats.columns and feats[c].notna().any()]

    X_all = feats[avail].copy() if avail else pd.DataFrame(index=feats.index)

    # Train-pack: dropp rader hvor features eller label mangler
    subset_cols = [c for c in avail if c in feats.columns]
    if y.name in feats.columns:
        subset_cols.append(y.name)
    # Hvis vi ikke har noe å trene på -> nøytral fallback
    if not subset_cols:
        neutral = pd.Series(0.5, index=feats.index, name="proba")
        last_idx = feats.index[-1] if len(feats.index) else None
        return neutral, np.nan, np.nan, 0.5, last_idx

    pack = pd.concat([X_all, y], axis=1)
    # Bruk kun kolonner som faktisk finnes i pack
    subset_cols = [c for c in subset_cols if c in pack.columns]
    if not subset_cols:
        neutral = pd.Series(0.5, index=feats.index, name="proba")
        last_idx = feats.index[-1] if len(feats.index) else None
        return neutral, np.nan, np.nan, 0.5, last_idx

    pack = pack.dropna(subset=subset_cols)

    if pack.empty or len(pack) < 80:
        base_index = pack.index if len(pack) else feats.index
        proba = pd.Series(0.5, index=base_index, name="proba")
        last_idx = base_index[-1] if len(base_index) else None
        return proba, np.nan, np.nan, 0.5, last_idx

    yv = pack[y.name].astype(int)
    if len(np.unique(yv)) < 2:
        proba = pd.Series(0.5, index=pack.index, name="proba")
        return proba, np.nan, np.nan, 0.5, pack.index[-1]

    Xv = pack[avail] if avail else pd.DataFrame(index=pack.index)

    # 70/30 tids-splitt
    n = len(pack)
    split = int(n * 0.70)
    Xtr, Xva = Xv.iloc[:split], Xv.iloc[split:]
    ytr, yva = yv.iloc[:split], yv.iloc[split:]

    scaler = RobustScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xva_s = scaler.transform(Xva) if len(Xva) else np.empty((0, Xtr.shape[1]))

    base = GradientBoostingClassifier(random_state=0)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(Xtr_s, ytr)

    # Val-metrics + optimal terskel hvis vi har valdata
    acc = auc = np.nan
    opt_thr = 0.5
    if len(Xva):
        p_val = clf.predict_proba(Xva_s)[:, 1]
        cand = np.linspace(0.3, 0.7, 41)
        accs = [accuracy_score(yva, (p_val >= t).astype(int)) for t in cand]
        opt_thr = float(cand[int(np.argmax(accs))])
        acc = float(np.max(accs))
        try:
            auc = float(roc_auc_score(yva, p_val))
        except Exception:
            auc = np.nan

    # Prediker proba for hele X_all (fyll NA defensivt)
    X_pred = X_all.copy()
    if not X_pred.empty:
        X_pred = X_pred.fillna(method="ffill").fillna(method="bfill")
        X_pred = X_pred.fillna(X_pred.median(numeric_only=True))
        Xs = scaler.transform(X_pred)
        p_all = clf.predict_proba(Xs)[:, 1]
        proba_full = pd.Series(p_all, index=X_pred.index, name="proba")
    else:
        proba_full = pd.Series(0.5, index=feats.index, name="proba")

    return proba_full, acc, auc, opt_thr, pack.index[-1]

def analyze_ticker_multi(df_raw: pd.DataFrame, horizons: dict, eps_map: dict, feats_map: dict) -> dict:
    out = {}
    for key in ["A", "B", "C"]:
        H = int(horizons[key])
        eps = float(eps_map[key])
        feat_list = feats_map.get(key, FEATURES_ALL)

        proba, acc, auc, opt_thr, last_idx = fit_predict_single_horizon(df_raw, H, eps, feat_list)

        out[key] = {
            "proba": proba,
            "acc": acc,
            "auc": auc,
            "opt_thr": opt_thr,
            "last_date": last_idx,
        }
    return out

# -----------------------------
# Presentasjon
# -----------------------------
def rec_from_prob(prob, buy_thr, sell_thr) -> str:
    if np.isnan(prob):
        return "HOLD"
    if prob >= buy_thr:
        return "BUY"
    if prob <= sell_thr:
        return "SELL"
    return "HOLD"

def expected_date(last_ts, horizon_days: int) -> str:
    if last_ts is None or pd.isna(last_ts):
        return "—"
    try:
        return (pd.to_datetime(last_ts) + BusinessDay(horizon_days)).strftime("%Y-%m-%d")
    except Exception:
        return "—"

def last_proba(pack: dict, key: str, default=0.5) -> float:
    try:
        s = pack[key]["proba"]
        if s is None or len(s) == 0:
            return default
        v = float(s.iloc[-1])
        return v if not np.isnan(v) else default
    except Exception:
        return default

# -----------------------------
# Kjør skann
# -----------------------------
run = st.button("🔎 Skann og sammenlign (A/B/C)")
rows = []

if run:
    prog = st.progress(0.0)
    stat = st.empty()

    for i, t in enumerate(tickers, start=1):
        stat.write(f"Henter og analyserer: **{t}** ({i}/{len(tickers)})")
        try:
            raw = fetch_history(t, start_date, end_date)
        except Exception:
            raw = pd.DataFrame()

        if raw is None or raw.empty or "Close" not in raw:
            rows.append({
                "Ticker": t,
                "Prob_A": np.nan, "Rec_A": "HOLD", "Date_A": "—", "EPS_A(%)": epsA,
                "Prob_B": np.nan, "Rec_B": "HOLD", "Date_B": "—", "EPS_B(%)": epsB,
                "Prob_C": np.nan, "Rec_C": "HOLD", "Date_C": "—", "EPS_C(%)": epsC,
                "Acc": np.nan, "AUC": np.nan, "Composite": np.nan
            })
            prog.progress(i/len(tickers))
            continue

        pack = analyze_ticker_multi(raw, HORIZONS, EPS_MAP, FEATS_MAP)

        pA = last_proba(pack, "A"); pB = last_proba(pack, "B"); pC = last_proba(pack, "C")
        dA = expected_date(pack["A"]["last_date"], H_A)
        dB = expected_date(pack["B"]["last_date"], H_B)
        dC = expected_date(pack["C"]["last_date"], H_C)

        bA, sA = THR["A"]; bB, sB = THR["B"]; bC, sC = THR["C"]
        rA = rec_from_prob(pA, max(bA, pack["A"]["opt_thr"]), sA)
        rB = rec_from_prob(pB, max(bB, pack["B"]["opt_thr"]), sB)
        rC = rec_from_prob(pC, max(bC, pack["C"]["opt_thr"]), sC)

        # aggreger enkle metrics på tvers (dersom finnes)
        accs = [pack[k]["acc"] for k in ["A","B","C"] if not np.isnan(pack[k]["acc"])]
        aucs = [pack[k]["auc"] for k in ["A","B","C"] if not np.isnan(pack[k]["auc"])]
        acc = float(np.mean(accs)) if accs else np.nan
        auc = float(np.mean(aucs)) if aucs else np.nan
        composite = float(np.mean([x for x in [pA,pB,pC] if not np.isnan(x)])) if any(not np.isnan(x) for x in [pA,pB,pC]) else np.nan

        rows.append({
            "Ticker": t,
            "Prob_A": pA, "Rec_A": rA, "Date_A": dA, "EPS_A(%)": epsA,
            "Prob_B": pB, "Rec_B": rB, "Date_B": dB, "EPS_B(%)": epsB,
            "Prob_C": pC, "Rec_C": rC, "Date_C": dC, "EPS_C(%)": epsC,
            "Acc": acc, "AUC": auc, "Composite": composite
        })
        prog.progress(i/len(tickers))

    prog.empty(); stat.empty()

    df = pd.DataFrame(rows)

    # kolonne-formatering
    num_fmt = {"Prob_A": "{:.2%}", "Prob_B": "{:.2%}", "Prob_C": "{:.2%}",
               "Acc": "{:.2%}", "AUC": "{:.3f}", "Composite": "{:.2%}"}

    c1,c2,c3 = st.columns(3)
    with c1:
        st.subheader("🟩 A-horisont")
        dA = df[["Ticker","Prob_A","Rec_A","Date_A","EPS_A(%)"]].sort_values("Prob_A", ascending=False)
        st.dataframe(dA.style.format({"Prob_A":"{:.2%}"}), use_container_width=True)
    with c2:
        st.subheader("🟦 B-horisont")
        dB = df[["Ticker","Prob_B","Rec_B","Date_B","EPS_B(%)"]].sort_values("Prob_B", ascending=False)
        st.dataframe(dB.style.format({"Prob_B":"{:.2%}"}), use_container_width=True)
    with c3:
        st.subheader("🟧 C-horisont")
        dC = df[["Ticker","Prob_C","Rec_C","Date_C","EPS_C(%)"]].sort_values("Prob_C", ascending=False)
        st.dataframe(dC.style.format({"Prob_C":"{:.2%}"}), use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Sammenligningstabell (alle horisonter)")
    cmp = df[["Ticker",
              "Prob_A","Rec_A","Date_A","EPS_A(%)",
              "Prob_B","Rec_B","Date_B","EPS_B(%)",
              "Prob_C","Rec_C","Date_C","EPS_C(%)",
              "Acc","AUC","Composite"]].sort_values("Composite", ascending=False)
    st.dataframe(cmp.style.format(num_fmt), use_container_width=True)

    if want_excel:
        try:
            import io
            with pd.ExcelWriter(io.BytesIO(), engine="xlsxwriter") as writer:
                dA.to_excel(writer, index=False, sheet_name=f"A_{H_A}d")
                dB.to_excel(writer, index=False, sheet_name=f"B_{H_B}d")
                dC.to_excel(writer, index=False, sheet_name=f"C_{H_C}d")
                cmp.to_excel(writer, index=False, sheet_name="Comparison")
                writer.book.filename.seek(0)
                data = writer.book.filename.getvalue()
            st.download_button(
                "⬇️ Last ned Excel (A/B/C/Comparison)",
                data=data,
                file_name=f"scan_v6_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.info(f"Excel-eksport feilet: {e}")

else:
    st.info("Velg tickere og trykk **🔎 Skann og sammenlign (A/B/C)** for å starte.")

























