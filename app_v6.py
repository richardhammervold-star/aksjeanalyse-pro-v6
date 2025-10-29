# app_v6.py
# 📈 Aksjeanalyse – Pro v6 Dashboard (business-day everywhere)
# Tre horisonter (A/B/C) – standard 3 / 7 / 14 handelsdager frem
# Build: v6.3 – oktober 2025

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from pandas.tseries.offsets import BDay

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score

import matplotlib.pyplot as plt

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
    unsafe_allow_html=True
)

st.title("📈 Aksjeanalyse – Pro v6 Dashboard")
st.caption("Alle horisonter og datoer beregnes i **handelsdager** (BDay). Build: v6.3 – oktober 2025")

# -----------------------------
# Presets
# -----------------------------
PRESETS = {
    "OBX (Norge)": [
        "EQNR.OL","DNB.OL","MOWI.OL","NHY.OL","TEL.OL","ORK.OL","YAR.OL","KOG.OL",
        "AKRBP.OL","TGS.OL"
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

default_tickers = ["EQNR.OL","DNB.OL","MOWI.OL"]
tickers = st.session_state.get("tickers", default_tickers)

user_tickers = st.sidebar.text_area(
    "Tickere (komma/linjer). Tomt = bruk preset over.",
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
dA = st.sidebar.slider("Horisont A", 1, 30, 3)
dB = st.sidebar.slider("Horisont B", 1, 30, 7)
dC = st.sidebar.slider("Horisont C", 1, 30, 14)

st.sidebar.subheader("Støyfilter (EPS, %)")
epsA = st.sidebar.number_input("EPS_A (A-horisont)", value=0.20, min_value=0.0, max_value=50.0, step=0.10)
epsB = st.sidebar.number_input("EPS_B (B-horisont)", value=0.30, min_value=0.0, max_value=50.0, step=0.10)
epsC = st.sidebar.number_input("EPS_C (C-horisont)", value=0.50, min_value=0.0, max_value=50.0, step=0.10)

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
    bA, sA = slider_pair(f"{dA} hd • KJØP >", f"{dA} hd • SELG <", 0.60, 0.40)
    bB, sB = slider_pair(f"{dB} hd • KJØP >", f"{dB} hd • SELG <", 0.60, 0.40)
    bC, sC = slider_pair(f"{dC} hd • KJØP >", f"{dC} hd • SELG <", 0.60, 0.40)
    thr = {"A": (bA, sA), "B": (bB, sB), "C": (bC, sC)}

st.sidebar.markdown("---")
want_excel = st.sidebar.checkbox("Eksporter Excel (flere ark)", value=True)
st.sidebar.caption("«hd» = handelsdager (BDay). EPS = minimumsbevegelse som ikke regnes som støy.")

# -----------------------------
# Datahjelpere (med caching)
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_history(ticker: str, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Felles feature-pipe."""
    if df is None or df.empty or "Close" not in df:
        return pd.DataFrame()

    out = df.copy()
    close = pd.to_numeric(out["Close"], errors="coerce").astype(float)

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
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd
    out["macd_sig"] = signal
    out["macd_hist"] = macd - signal

    # Bollinger %
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    out["bb_pct"] = (close - lower) / (upper - lower)
    out["bb_width"] = (upper - lower) / ma20

    return out

def make_label_eps(df: pd.DataFrame, horizon_days: int, eps_pct: float) -> pd.Series:
    """
    Binær target i **handelsdager**:
      1 hvis avkastning over +eps, 0 hvis under -eps, ellers NaN.
    Vi forskyver tidsindeksen med BDay(horizon_days) og aligner tilbake.
    """
    if df is None or df.empty or "Close" not in df:
        return pd.Series(dtype=float)

    close = pd.to_numeric(df["Close"], errors="coerce").astype(float)

    # Fremtidig pris etter 'horizon_days' handelsdager:
    fut_idx = close.index + BDay(int(horizon_days))
    fut = pd.Series(close.values, index=fut_idx).reindex(close.index)

    fwd_ret = fut / close - 1.0
    eps = eps_pct / 100.0

    arr = np.where(fwd_ret > eps, 1,
                   np.where(fwd_ret < -eps, 0, np.nan))
    y = pd.Series(arr.astype("float64"), index=df.index, name=f"Target_{horizon_days}")
    return y

FEATURES_ALL = [
    "ret1","ret3","ret5","ma5","ma20","vol10","trend20","rsi14",
    "ema20","ema50","ema_gap","bb_pct","bb_width","macd","macd_sig","macd_hist"
]

def fit_predict_single_horizon(df_raw: pd.DataFrame, horizon_days: int, eps_pct: float):
    """
    En robust, enkel modell som trener på alle tilgjengelige datapunkter (etter rensing).
    Alle tidspunkter og target er i **handelsdager**.
    """
    out = {
        "proba": pd.Series(dtype=float),
        "acc": np.nan,
        "auc": np.nan,
        "opt_thr": 0.5,
        "last_date": None,
    }
    if df_raw is None or df_raw.empty or "Close" not in df_raw:
        return out

    df = add_indicators(df_raw)
    if df.empty:
        return out

    y = make_label_eps(df, horizon_days, eps_pct)
    if y.empty:
        return out

    cols = [c for c in FEATURES_ALL if c in df.columns]
    if not cols:
        return out

    pack = pd.concat([df[cols], y], axis=1).dropna()
    if pack.empty or len(pack) < 120:
        # for lite data – nøytral
        out["proba"] = pd.Series(0.5, index=df.index, name="proba")
        out["last_date"] = df.index[-1]
        return out

    X = pack[cols].copy()
    yv = pack[y.name].astype(int).copy()

    if len(np.unique(yv)) < 2:
        out["proba"] = pd.Series(0.5, index=df.index, name="proba")
        out["last_date"] = pack.index[-1]
        return out

    # enkel holdout (siste 10%)
    n = len(X)
    split = max(int(n*0.9), n-20)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = yv.iloc[:split], yv.iloc[split:]

    scaler = RobustScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    base = GradientBoostingClassifier(random_state=0)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(Xtr_s, ytr)

    # eval
    pte = clf.predict_proba(Xte_s)[:, 1]
    try:
        acc = accuracy_score(yte, (pte >= 0.5).astype(int))
        auc = roc_auc_score(yte, pte)
    except Exception:
        acc, auc = np.nan, np.nan

    # full proba
    scaler_full = RobustScaler().fit(X)
    Xs_full = scaler_full.transform(X)
    clf_full = CalibratedClassifierCV(GradientBoostingClassifier(random_state=0),
                                      method="isotonic", cv=3).fit(Xs_full, yv)
    p_full = clf_full.predict_proba(Xs_full)[:, 1]
    proba_full = pd.Series(p_full, index=pack.index).reindex(df.index).fillna(method="ffill")

    out["proba"] = proba_full
    out["acc"] = float(acc) if acc==acc else np.nan
    out["auc"] = float(auc) if auc==auc else np.nan
    out["opt_thr"] = 0.5  # baseline
    out["last_date"] = pack.index[-1]
    return out

def analyze_ticker_multi(df_raw: pd.DataFrame, horizons: dict, eps_dict: dict) -> dict:
    """
    horizons: {"A": dA, "B": dB, "C": dC} (handelsdager)
    eps_dict: {"A": epsA, "B": epsB, "C": epsC} (i %)
    """
    out = {}
    for key in ["A", "B", "C"]:
        out[key] = fit_predict_single_horizon(df_raw, horizons[key], eps_dict[key])
    return out

def rec_from_prob(prob: float, buy_thr: float, sell_thr: float) -> str:
    if np.isnan(prob): return "HOLD"
    if prob > buy_thr: return "BUY"
    if prob < sell_thr: return "SELL"
    return "HOLD"

def expected_date(last_date: pd.Timestamp, horizon_days: int) -> str:
    """Legg til **handelsdager** (BDay), ikke kalenderdager."""
    if last_date is None or pd.isna(last_date):
        return "—"
    return (pd.to_datetime(last_date) + BDay(int(horizon_days))).strftime("%Y-%m-%d")

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

    HORIZONS = {"A": dA, "B": dB, "C": dC}
    EPS = {"A": epsA, "B": epsB, "C": epsC}

    for i, t in enumerate(tickers, start=1):
        status.write(f"Henter og analyserer: {t}  ({i}/{len(tickers)})")
        try:
            df_raw = fetch_history(t, start=start_date, end=end_date)
        except Exception:
            df_raw = pd.DataFrame()

        if df_raw is None or df_raw.empty or "Close" not in df_raw:
            results.append({
                "Ticker": t,
                "Prob_A": np.nan, "Rec_A": "HOLD", "Date_A": "—", "EPS_A(%)": EPS["A"],
                "Prob_B": np.nan, "Rec_B": "HOLD", "Date_B": "—", "EPS_B(%)": EPS["B"],
                "Prob_C": np.nan, "Rec_C": "HOLD", "Date_C": "—", "EPS_C(%)": EPS["C"],
                "Acc": np.nan, "AUC": np.nan, "Composite": np.nan
            })
            progress.progress(i/len(tickers))
            continue

        pack = analyze_ticker_multi(df_raw, HORIZONS, EPS)

        def last_proba(key, default=0.5):
            s = pack[key]["proba"]
            if s is None or len(s)==0: return default
            v = float(s.iloc[-1])
            if np.isnan(v): return default
            return v

        # siste proba pr horisont
        pA = last_proba("A"); pB = last_proba("B"); pC = last_proba("C")
        dateA = expected_date(pack["A"]["last_date"], HORIZONS["A"])
        dateB = expected_date(pack["B"]["last_date"], HORIZONS["B"])
        dateC = expected_date(pack["C"]["last_date"], HORIZONS["C"])

        # terskler
        bA, sA = thr["A"]; bB, sB = thr["B"]; bC, sC = thr["C"]
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
            "Prob_A": pA, "Rec_A": rA, "Date_A": dateA, "EPS_A(%)": EPS["A"],
            "Prob_B": pB, "Rec_B": rB, "Date_B": dateB, "EPS_B(%)": EPS["B"],
            "Prob_C": pC, "Rec_C": rC, "Date_C": dateC, "EPS_C(%)": EPS["C"],
            "Acc": acc, "AUC": auc, "Composite": comp
        })

        progress.progress(i/len(tickers))

    status.empty()
    progress.empty()

# -----------------------------
# Visning – tre kolonner
# -----------------------------
if run:
    df = pd.DataFrame(results)

    # sikre tallkolonner
    for c in ["Prob_A","Prob_B","Prob_C","Acc","AUC","Composite"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader(f"🟩 {dA} handelsdager frem")
        dfa = df[["Ticker","Prob_A","Rec_A","Date_A","EPS_A(%)","Acc"]].copy()
        dfa = dfa.sort_values("Prob_A", ascending=False)
        st.dataframe(style_df(dfa, {"Prob_A":"{:.2%}","Acc":"{:.2%}","EPS_A(%)":"{:.2f}"}), use_container_width=True)

    with c2:
        st.subheader(f"🟦 {dB} handelsdager frem")
        dfb = df[["Ticker","Prob_B","Rec_B","Date_B","EPS_B(%)","Acc"]].copy()
        dfb = dfb.sort_values("Prob_B", ascending=False)
        st.dataframe(style_df(dfb, {"Prob_B":"{:.2%}","Acc":"{:.2%}","EPS_B(%)":"{:.2f}"}), use_container_width=True)

    with c3:
        st.subheader(f"🟧 {dC} handelsdager frem")
        dfc = df[["Ticker","Prob_C","Rec_C","Date_C","EPS_C(%)","Acc"]].copy()
        dfc = dfc.sort_values("Prob_C", ascending=False)
        st.dataframe(style_df(dfc, {"Prob_C":"{:.2%}","Acc":"{:.2%}","EPS_C(%)":"{:.2f}"}), use_container_width=True)

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
            {"Prob_A":"{:.2%}","Prob_B":"{:.2%}","Prob_C":"{:.2%}",
             "Acc":"{:.2%}","AUC":"{:.3f}","Composite":"{:.2%}",
             "EPS_A(%)":"{:.2f}","EPS_B(%)":"{:.2f}","EPS_C(%)":"{:.2f}"}
        ),
        use_container_width=True
    )

    # -------------------------
    # Detalj: graf per ticker
    # -------------------------
    st.markdown("---")
    st.subheader("📊 Detaljvisning (pris + sannsynlighet)")
    sel_list = df["Ticker"].tolist() if not df.empty else []
    sel = st.selectbox("Velg ticker", sel_list)
    if sel:
        try:
            raw = fetch_history(sel, start=start_date, end=end_date)
            pack = analyze_ticker_multi(raw, {"A":dA,"B":dB,"C":dC}, {"A":epsA,"B":epsB,"C":epsC})
            plot_df = pd.DataFrame({
                "Close": raw["Close"],
                f"Prob_{dA}": pack["A"]["proba"].reindex(raw.index),
                f"Prob_{dB}": pack["B"]["proba"].reindex(raw.index),
                f"Prob_{dC}": pack["C"]["proba"].reindex(raw.index),
            }).dropna(subset=["Close"])

            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(plot_df.index, plot_df["Close"])
            ax1.set_xlabel("Dato")
            ax1.set_ylabel("Pris")

            ax2 = ax1.twinx()
            if f"Prob_{dA}" in plot_df: ax2.plot(plot_df.index, plot_df[f"Prob_{dA}"], alpha=0.9, label=f"{dA} hd")
            if f"Prob_{dB}" in plot_df: ax2.plot(plot_df.index, plot_df[f"Prob_{dB}"], alpha=0.9, label=f"{dB} hd")
            if f"Prob_{dC}" in plot_df: ax2.plot(plot_df.index, plot_df[f"Prob_{dC}"], alpha=0.9, label=f"{dC} hd")
            ax2.axhline(0.5, linestyle="--", alpha=0.6)
            ax2.set_ylabel("Sannsynlighet")
            ax2.legend(loc="upper left")

            plt.title(f"{sel}: Pris + sannsynlighet ({dA}/{dB}/{dC} handelsdager)")
            st.pyplot(fig)
        except Exception as e:
            st.info(f"Kunne ikke vise graf for {sel}: {e}")

    # -------------------------
    # Eksport + historikk
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
                dfa.to_excel(writer, index=False, sheet_name=f"{dA}hd")
                dfb.to_excel(writer, index=False, sheet_name=f"{dB}hd")
                dfc.to_excel(writer, index=False, sheet_name=f"{dC}hd")
                cmp_df.to_excel(writer, index=False, sheet_name="Comparison")
            buf.seek(0)
            xls = buf.getvalue()
            st.download_button(
                f"⬇️ Last ned Excel ({dA}/{dB}/{dC} handelsdager)",
                data=xls,
                file_name=f"scan_v6_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.info(f"Excel-eksport feilet: {e}")

else:
    st.info("Velg/skriv tickere i sidepanelet og trykk **🔎 Skann og sammenlign** for å starte.")










































