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
                v = float(s.iloc[-1])
                return v if not np.isnan(v) else default
            except Exception:
                return default

        pA, pB, pC = last_proba("A"), last_proba("B"), last_proba("C")
        dateA = expected_date(pack["A"]["last_date"], dA)
        dateB = expected_date(pack["B"]["last_date"], dB)
        dateC = expected_date(pack["C"]["last_date"], dC)

        bA, sA = thr["A"]; bB, sB = thr["B"]; bC, sC = thr["C"]
        rA = rec_from_prob(pA, max(bA, pack["A"]["opt_thr"]), sA)
        rB = rec_from_prob(pB, max(bB, pack["B"]["opt_thr"]), sB)
        rC = rec_from_prob(pC, max(bC, pack["C"]["opt_thr"]), sC)

        probs = [x for x in [pA, pB, pC] if not np.isnan(x)]
        comp = float(np.mean(probs)) if probs else np.nan

        accs = [pack[k]["acc"] for k in ["A", "B", "C"] if not np.isnan(pack[k]["acc"])]
        aucs = [pack[k]["auc"] for k in ["A", "B", "C"] if not np.isnan(pack[k]["auc"])]
        acc = float(np.mean(accs)) if accs else np.nan
        auc = float(np.mean(aucs)) if aucs else np.nan

        results.append({
            "Ticker": t,
            "Prob_A": pA, "Rec_A": rA, "Date_A": dateA, "EPS_A(%)": epsA,
            "Prob_B": pB, "Rec_B": rB, "Date_B": dateB, "EPS_B(%)": epsB,
            "Prob_C": pC, "Rec_C": rC, "Date_C": dateC, "EPS_C(%)": epsC,
            "Acc": acc, "AUC": auc, "Composite": comp
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

    for c in ["Prob_A","Prob_B","Prob_C","Acc","AUC","Composite"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader(f"🟩 {dA} handelsdager frem (A)")
        dfA = df[["Ticker", "Prob_A", "Rec_A", "Date_A", "EPS_A(%)", "Acc"]].copy()
        dfA = dfA.sort_values("Prob_A", ascending=False)
        st.dataframe(style_df(dfA, {"Prob_A": "{:.2%}", "Acc": "{:.2%}"}), use_container_width=True)

    with c2:
        st.subheader(f"🟦 {dB} handelsdager frem (B)")
        dfB = df[["Ticker", "Prob_B", "Rec_B", "Date_B", "EPS_B(%)", "Acc"]].copy()
        dfB = dfB.sort_values("Prob_B", ascending=False)
        st.dataframe(style_df(dfB, {"Prob_B": "{:.2%}", "Acc": "{:.2%}"}), use_container_width=True)

    with c3:
        st.subheader(f"🟧 {dC} handelsdager frem (C)")
        dfC = df[["Ticker", "Prob_C", "Rec_C", "Date_C", "EPS_C(%)", "Acc"]].copy()
        dfC = dfC.sort_values("Prob_C", ascending=False)
        st.dataframe(style_df(dfC, {"Prob_C": "{:.2%}", "Acc": "{:.2%}"}), use_container_width=True)

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
                "Prob_A": "{:.2%}", "Prob_B": "{:.2%}", "Prob_C": "{:.2%}",
                "Acc": "{:.2%}", "AUC": "{:.3f}", "Composite": "{:.2%}"
            }
        ),
        use_container_width=True
    )

    # -------------------------
    # Detalj: graf per ticker
    # -------------------------
    st.markdown("---")
    st.subheader("📊 Detaljvisning (pris + sannsynlighet A/B/C)")
    sel = st.selectbox("Velg ticker for graf", df["Ticker"].tolist() if not df.empty else [])
    if sel:
        try:
            raw = fetch_history(sel, start=start_date, end=end_date)
            pack = analyze_ticker_multi(raw, HORIZONS, EPS_MAP)

            plot_df = pd.DataFrame({
                "Close": extract_close_series(raw),
                f"Prob_{dA}": pack["A"]["proba"].reindex(raw.index),
                f"Prob_{dB}": pack["B"]["proba"].reindex(raw.index),
                f"Prob_{dC}": pack["C"]["proba"].reindex(raw.index),
            }).dropna(subset=["Close"])

            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(plot_df.index, plot_df["Close"])
            ax1.set_xlabel("Dato")
            ax1.set_ylabel("Pris")

            ax2 = ax1.twinx()
            if f"Prob_{dA}" in plot_df: ax2.plot(plot_df.index, plot_df[f"Prob_{dA}"], alpha=0.9, label=f"P({dA}d)")
            if f"Prob_{dB}" in plot_df: ax2.plot(plot_df.index, plot_df[f"Prob_{dB}"], alpha=0.9, label=f"P({dB}d)")
            if f"Prob_{dC}" in plot_df: ax2.plot(plot_df.index, plot_df[f"Prob_{dC}"], alpha=0.9, label=f"P({dC}d)")
            ax2.axhline(0.5, linestyle="--", alpha=0.6)
            ax2.set_ylabel("Sannsynlighet")
            ax2.legend(loc="upper left")

            plt.title(f"{sel}: Pris + sannsynlighet (A={dA}d / B={dB}d / C={dC}d)")
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
                dfA.to_excel(writer, index=False, sheet_name=f"A_{dA}d")
                dfB.to_excel(writer, index=False, sheet_name=f"B_{dB}d")
                dfC.to_excel(writer, index=False, sheet_name=f"C_{dC}d")
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




































