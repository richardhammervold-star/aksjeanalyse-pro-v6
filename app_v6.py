# app_v6.py
# Aksjeanalyse – Pro v6 (robust)
# Multi-horisont A/B/C (handelsdager) med uavhengige eps-filtre og valgbare features
# Build: v6.4 – oktober 2025

import io
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

from datetime import datetime
from pandas.tseries.offsets import BusinessDay

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

# ========================= PAGE & STYLE =========================
st.set_page_config(page_title="Aksjeanalyse – Pro v6", layout="wide")

st.markdown(
    """
    <style>
      .main { background-color: #0E1117; }
      .soft-card { background: rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
                   border-radius:14px; padding:12px 14px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 Aksjeanalyse – Pro v6 Dashboard")
st.caption("Fleksible horisonter (A/B/C), uavhengige eps-filtre og feature-valg • Build v6.4")

# ========================= PRESETS =========================
PRESETS = {
    "OBX (Norge)": [
        "EQNR.OL","DNB.OL","MOWI.OL","NHY.OL","TEL.OL","ORK.OL","YAR.OL","KOG.OL",
        "AKRBP.OL","TGS.OL","SUBC.OL","SALM.OL","AUTO.OL","NEL.OL"
    ],
    "USA – Megacaps": [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","XOM",
        "UNH","V","JNJ","WMT","PG","MA","AVGO","HD","MRK","PEP"
    ],
}

# ========================= SIDEBAR =========================
st.sidebar.header("⚙️ Innstillinger")
preset = st.sidebar.selectbox("Hurtigvalg gruppe", list(PRESETS.keys()))
if st.sidebar.button("📥 Last inn preset"):
    st.session_state["tickers"] = PRESETS[preset]

default_tickers = ["EQNR.OL", "DNB.OL"]
tickers_current = st.session_state.get("tickers", default_tickers)

user_tickers = st.sidebar.text_area(
    "Tickere (komma/linjer). Tomt felt = bruk valgt preset.",
    value=", ".join(tickers_current), height=120
)
TICKERS = [t.strip().upper() for chunk in user_tickers.split("\n") for t in chunk.split(",") if t.strip()]

col_date1, col_date2 = st.sidebar.columns(2)
with col_date1:
    start_date = st.date_input("Startdato", pd.to_datetime("2019-01-01"))
with col_date2:
    end_date = st.date_input("Sluttdato", pd.Timestamp.today())

st.sidebar.markdown("---")
st.sidebar.subheader("Horisont (handelsdager)")

dA = st.sidebar.number_input("Horisont A (vises som '3 dager')", min_value=1, max_value=60, value=3, step=1)
dB = st.sidebar.number_input("Horisont B (vises som '7 dager')", min_value=1, max_value=120, value=7, step=1)
dC = st.sidebar.number_input("Horisont C (vises som '14 dager')", min_value=1, max_value=250, value=14, step=1)

HORIZONS = {"A": int(dA), "B": int(dB), "C": int(dC)}

st.sidebar.markdown("---")
st.sidebar.subheader("Støyfilter (eps, %) – pr. horisont")

epsA = st.sidebar.number_input("EPS A (%)", value=1.00, min_value=0.0, max_value=50.0, step=0.10)
epsB = st.sidebar.number_input("EPS B (%)", value=3.00, min_value=0.0, max_value=50.0, step=0.10)
epsC = st.sidebar.number_input("EPS C (%)", value=5.00, min_value=0.0, max_value=50.0, step=0.10)

EPS_MAP = {"A": float(epsA), "B": float(epsB), "C": float(epsC)}

# ========================= FEATURE VALG =========================
FEATURES_ALL = [
    "ret1","ret3","ret5","ma5","ma20","vol10","trend20","rsi14",
    "ema20","ema50","ema_gap","bb_pct","bb_width","macd","macd_sig","macd_hist"
]

st.sidebar.markdown("---")
st.sidebar.subheader("Features")
same_feats = st.sidebar.checkbox("Bruk samme features på alle horisonter", value=True)

if same_feats:
    feats_same = st.sidebar.multiselect(
        "Velg features", FEATURES_ALL,
        default=["ret1","ret3","ma5","ma20","ema_gap","macd","macd_sig","bb_pct","bb_width","rsi14"]
    )
    FEATS_MAP = {"A": feats_same, "B": feats_same, "C": feats_same}
else:
    featsA = st.sidebar.multiselect(
        "Features A", FEATURES_ALL,
        default=["ret1","ret3","ma5","ema_gap","macd","macd_sig","rsi14"], key="fa"
    )
    featsB = st.sidebar.multiselect(
        "Features B", FEATURES_ALL,
        default=["ret1","ret5","ma20","ema_gap","bb_pct","bb_width","rsi14"], key="fb"
    )
    featsC = st.sidebar.multiselect(
        "Features C", FEATURES_ALL,
        default=["ret1","ret5","ma20","ema50","ema_gap","macd","macd_sig","bb_pct"], key="fc"
    )
    FEATS_MAP = {"A": featsA, "B": featsB, "C": featsC}

# ========================= TERSKLER =========================
st.sidebar.markdown("---")
st.sidebar.subheader("Terskler for anbefaling")

same_thr = st.sidebar.checkbox("Bruk samme terskler på alle horisonter", value=True)

def slider_pair(lbl_buy, lbl_sell, buy_default=0.60, sell_default=0.40):
    b = st.sidebar.slider(lbl_buy, 0.50, 0.90, buy_default, 0.01)
    s = st.sidebar.slider(lbl_sell, 0.10, 0.50, sell_default, 0.01)
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
want_excel = st.sidebar.checkbox("Eksporter Excel (flere ark)", value=True)

# ========================= DATA HELPERS =========================
@st.cache_data(show_spinner=False)
def fetch_history(ticker: str, start, end):
    return yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

# ====================== ROBUST MODEL CORE ======================
# (Dette er limt inn slik at feilene du hadde forsvinner)

def add_indicators(df_raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df_raw.index).copy()
    close = pd.to_numeric(df_raw.get("Close", pd.Series(index=df_raw.index)), errors="coerce").astype(float)
    out["Close"] = close
    out["ret1"] = close.pct_change(1)
    out["ret3"] = close.pct_change(3)
    out["ret5"] = close.pct_change(5)
    out["ma5"] = close.rolling(5).mean()
    out["ma20"] = close.rolling(20).mean()
    out["vol10"] = out["ret1"].rolling(10).std()
    out["trend20"] = (out["ma20"] - out["ma5"]) / out["ma20"]

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    out["rsi14"] = 100 - (100 / (1 + rs))

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

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + 2*std20
    lower = ma20 - 2*std20
    out["bb_pct"] = (close - lower) / (upper - lower)
    out["bb_width"] = (upper - lower) / ma20

    return out

def make_label_eps(df_raw: pd.DataFrame, horizon_days: int, eps_pct: float) -> pd.Series:
    close = pd.to_numeric(df_raw.get("Close", pd.Series(index=df_raw.index)), errors="coerce").astype(float)
    fwd = close.shift(-int(horizon_days)) / close - 1.0
    eps = float(eps_pct) / 100.0
    arr = np.where(fwd > eps, 1, np.where(fwd < -eps, 0, np.nan))
    arr = np.asarray(arr, dtype="float64").ravel()
    return pd.Series(arr, index=df_raw.index, name=f"Target_{int(horizon_days)}")

def _neutral_series(idx: pd.Index, fill: float = 0.5, name: str = "proba") -> pd.Series:
    return pd.Series(fill, index=idx, name=name)

def fit_predict_single_horizon(df_raw: pd.DataFrame, H: int, eps: float, feat_list: list[str] | None = None):
    feats_all = add_indicators(df_raw)
    if not feat_list:
        feat_list = [c for c in FEATURES_ALL if c in feats_all.columns]
    else:
        feat_list = [c for c in feat_list if c in feats_all.columns]
    if not feat_list:
        return _neutral_series(df_raw.index), np.nan, np.nan, 0.5, None

    X = feats_all[feat_list]
    y = make_label_eps(df_raw, H, eps)

    pack = pd.concat([X, y], axis=1).dropna()
    if pack.empty or len(pack) < 60:
        return _neutral_series(df_raw.index), np.nan, np.nan, 0.5, (pack.index[-1] if len(pack) else None)

    Xp = pack[feat_list]
    yp = pack[y.name]
    uniq = np.unique(yp.astype(int))
    if len(uniq) < 2:
        return _neutral_series(df_raw.index), np.nan, np.nan, 0.5, pack.index[-1]

    n = len(Xp)
    anchors = [int(n*0.60), int(n*0.70), int(n*0.80)]
    probs_val = pd.Series(np.nan, index=Xp.index)
    accs, aucs, thrs = [], [], []

    for a in anchors:
        tr0, tr1 = 0, a
        va0, va1 = a, min(a + int(n*0.10), n-1)
        if va1 <= va0:
            continue

        Xtr, ytr = Xp.iloc[tr0:tr1], yp.iloc[tr0:tr1]
        Xva, yva = Xp.iloc[va0:va1], yp.iloc[va0:va1]
        if len(np.unique(ytr.astype(int))) < 2 or len(Xva) == 0:
            continue

        scaler = RobustScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xva_s = scaler.transform(Xva)

        base = GradientBoostingClassifier(random_state=0)
        clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        clf.fit(Xtr_s, ytr.astype(int))

        pv = clf.predict_proba(Xva_s)[:, 1]
        probs_val.iloc[va0:va1] = pv

        cand = np.linspace(0.30, 0.70, 41)
        acc_list = [accuracy_score(yva.astype(int), (pv >= t).astype(int)) for t in cand]
        t_star = float(cand[int(np.argmax(acc_list))])
        thrs.append(t_star)
        accs.append(max(acc_list))
        try:
            aucs.append(roc_auc_score(yva.astype(int), pv))
        except Exception:
            pass

    scaler = RobustScaler().fit(Xp)
    Xs = scaler.transform(Xp)
    base = GradientBoostingClassifier(random_state=0)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(Xs, yp.astype(int))

    p_full = clf.predict_proba(Xs)[:, 1]
    p_full = pd.Series(p_full, index=Xp.index, name="proba").reindex(df_raw.index)

    acc = float(np.nanmean(accs)) if accs else np.nan
    auc = float(np.nanmean(aucs)) if aucs else np.nan
    opt_thr = float(np.nanmean(thrs)) if thrs else 0.5
    last_idx = pack.index[-1]
    return p_full, acc, auc, opt_thr, last_idx

def analyze_ticker_multi(df_raw: pd.DataFrame, horizons_map: dict[str, int], eps_map: dict[str, float], feats_map: dict[str, list[str]] | None = None) -> dict:
    if feats_map is None:
        feats_map = {}
    out = {}
    if df_raw is None or df_raw.empty or "Close" not in df_raw.columns:
        for k in horizons_map.keys():
            out[k] = {"proba": _neutral_series(pd.Index([], dtype="datetime64[ns]")),
                      "acc": np.nan, "auc": np.nan, "opt_thr": 0.5, "last_date": None}
        return out

    for key, H in horizons_map.items():
        eps = float(eps_map.get(key, 0.0))
        flist = feats_map.get(key, FEATURES_ALL)
        proba, acc, auc, opt_thr, last_idx = fit_predict_single_horizon(df_raw, int(H), eps, flist)
        out[key] = {"proba": proba, "acc": acc, "auc": auc, "opt_thr": opt_thr, "last_date": last_idx}
    return out

# ====================== APP LOGIKK ======================

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
    return (pd.to_datetime(last_date) + BusinessDay(business_days)).strftime("%Y-%m-%d")

run = st.button("🔎 Skann og sammenlign (A/B/C)")
results = []

if run:
    progress = st.progress(0)
    status = st.empty()

    for i, t in enumerate(TICKERS, start=1):
        status.write(f"Henter og analyserer: {t} ({i}/{len(TICKERS)})")
        try:
            df_raw = fetch_history(t, start=start_date, end=end_date)
        except Exception:
            df_raw = pd.DataFrame()

        if df_raw is None or df_raw.empty or "Close" not in df_raw:
            results.append({
                "Ticker": t,
                "Prob_A": np.nan, "Rec_A": "HOLD", "Date_A": "—",
                "Prob_B": np.nan, "Rec_B": "HOLD", "Date_B": "—",
                "Prob_C": np.nan, "Rec_C": "HOLD", "Date_C": "—",
                "Acc": np.nan, "AUC": np.nan, "Composite": np.nan
            })
            progress.progress(i / len(TICKERS))
            continue

        pack = analyze_ticker_multi(df_raw, HORIZONS, EPS_MAP, FEATS_MAP)

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

        pA = last_proba("A")
        pB = last_proba("B")
        pC = last_proba("C")

        dA_str = expected_date(pack["A"]["last_date"], HORIZONS["A"])
        dB_str = expected_date(pack["B"]["last_date"], HORIZONS["B"])
        dC_str = expected_date(pack["C"]["last_date"], HORIZONS["C"])

        bA, sA = THR["A"]
        bB, sB = THR["B"]
        bC, sC = THR["C"]

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
            "Prob_A": pA, "Rec_A": rA, "Date_A": dA_str,
            "Prob_B": pB, "Rec_B": rB, "Date_B": dB_str,
            "Prob_C": pC, "Rec_C": rC, "Date_C": dC_str,
            "Acc": acc, "AUC": auc, "Composite": comp
        })

        progress.progress(i / len(TICKERS))

    status.empty()
    progress.empty()

# ========================= VISNING =========================
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

if run:
    df = pd.DataFrame(results)
    for c in ["Prob_A","Prob_B","Prob_C","Acc","AUC","Composite"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader(f"🟩 {HORIZONS['A']} dager frem")
        dfa = df[["Ticker","Prob_A","Rec_A","Date_A","Acc"]].copy()
        dfa = dfa.sort_values("Prob_A", ascending=False)
        st.dataframe(style_df(dfa, {"Prob_A":"{:.2%}","Acc":"{:.2%}"}), use_container_width=True)

    with c2:
        st.subheader(f"🟦 {HORIZONS['B']} dager frem")
        dfb = df[["Ticker","Prob_B","Rec_B","Date_B","Acc"]].copy()
        dfb = dfb.sort_values("Prob_B", ascending=False)
        st.dataframe(style_df(dfb, {"Prob_B":"{:.2%}","Acc":"{:.2%}"}), use_container_width=True)

    with c3:
        st.subheader(f"🟧 {HORIZONS['C']} dager frem")
        dfc = df[["Ticker","Prob_C","Rec_C","Date_C","Acc"]].copy()
        dfc = dfc.sort_values("Prob_C", ascending=False)
        st.dataframe(style_df(dfc, {"Prob_C":"{:.2%}","Acc":"{:.2%}"}), use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Sammenligningstabell (alle horisonter)")
    cmp_df = df[[
        "Ticker",
        "Prob_A","Rec_A","Date_A",
        "Prob_B","Rec_B","Date_B",
        "Prob_C","Rec_C","Date_C",
        "Acc","AUC","Composite"
    ]].sort_values("Composite", ascending=False)

    st.dataframe(
        style_df(
            cmp_df,
            {"Prob_A":"{:.2%}","Prob_B":"{:.2%}","Prob_C":"{:.2%}",
             "Acc":"{:.2%}","AUC":"{:.3f}","Composite":"{:.2%}"}
        ),
        use_container_width=True
    )

    # ----------------- Detaljplot -----------------
    st.markdown("---")
    st.subheader("📊 Detaljvisning (pris + sannsynlighet A/B/C)")
    sel = st.selectbox("Velg ticker", df["Ticker"].tolist() if not df.empty else [])
    if sel:
        try:
            raw = fetch_history(sel, start=start_date, end=end_date)
            pack = analyze_ticker_multi(raw, HORIZONS, EPS_MAP, FEATS_MAP)
            plot_df = pd.DataFrame({
                "Close": raw["Close"],
                f"Prob_{HORIZONS['A']}d": pack["A"]["proba"].reindex(raw.index),
                f"Prob_{HORIZONS['B']}d": pack["B"]["proba"].reindex(raw.index),
                f"Prob_{HORIZONS['C']}d": pack["C"]["proba"].reindex(raw.index),
            }).dropna(subset=["Close"])

            fig, ax1 = plt.subplots(figsize=(10,4))
            ax1.plot(plot_df.index, plot_df["Close"])
            ax1.set_xlabel("Dato"); ax1.set_ylabel("Pris")

            ax2 = ax1.twinx()
            for col in [c for c in plot_df.columns if col.startswith("Prob_")]:
                ax2.plot(plot_df.index, plot_df[col], alpha=0.9)
            ax2.axhline(0.5, linestyle="--", alpha=0.6); ax2.set_ylabel("Sannsynlighet")

            plt.title(f"{sel}: Pris + sannsynlighet ({HORIZONS['A']}/{HORIZONS['B']}/{HORIZONS['C']} handelsdager)")
            st.pyplot(fig)
        except Exception as e:
            st.info(f"Kunne ikke vise graf for {sel}: {e}")

    # ----------------- Eksport -----------------
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
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                dfa.to_excel(writer, index=False, sheet_name=f"A_{HORIZONS['A']}d")
                dfb.to_excel(writer, index=False, sheet_name=f"B_{HORIZONS['B']}d")
                dfc.to_excel(writer, index=False, sheet_name=f"C_{HORIZONS['C']}d")
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
    st.info("Velg/skriv tickere i sidepanelet og trykk **🔎 Skann og sammenlign (A/B/C)** for å starte.")


























