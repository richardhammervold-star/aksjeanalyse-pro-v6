# app_v6.py
# 📈 Aksjeanalyse – Pro v6 Dashboard
# Multi-horisont (A/B/C – default 1d / 3d / 5d) – side om side
# Build: v6.2 – oktober 2025

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
    .rec-buy  { background: rgba(22,163,74,0.15); border:1px solid rgba(22,163,74,0.35); border-radius:12px; padding:6px 10px; }
    .rec-hold { background: rgba(148,163,184,0.10); border:1px solid rgba(148,163,184,0.25); border-radius:12px; padding:6px 10px; }
    .rec-sell { background: rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.35); border-radius:12px; padding:6px 10px; }
    .soft-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius:16px; padding:12px 14px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 Aksjeanalyse – Pro v6 Dashboard")
st.caption("Multi-horisont modell (A/B/C – default 1/3/5 dager) • Build: v6.2 – oktober 2025")

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
    "Råvarer": ["CL=F","BZ=F","NG=F","GC=F","SI=F","HG=F","ZC=F","ZW=F","ZS=F"],
    "Valuta (Forex)": ["EURUSD=X","USDJPY=X","GBPUSD=X","AUDUSD=X","USDCAD=X","USDCHF=X","EURNOK=X","USDNOK=X","EURGBP=X"],
    "Krypto": ["BTC-USD","ETH-USD","SOL-USD","XRP-USD","ADA-USD"],
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
    end_date   = st.date_input("Sluttdato", pd.Timestamp.today())

st.sidebar.markdown("---")
st.sidebar.subheader("Horisont (i dager)")
H_A = int(st.sidebar.slider("Horisont A (vises som '1 dag')", 1, 30, 1))
H_B = int(st.sidebar.slider("Horisont B (vises som '3 dager')", 1, 30, 3))
H_C = int(st.sidebar.slider("Horisont C (vises som '5 dager')", 1, 30, 5))

HORIZONS = [(H_A, "A"), (H_B, "B"), (H_C, "C")]  # interne navn

st.sidebar.markdown("---")
st.sidebar.subheader("Terskler for anbefaling")
same_thr = st.sidebar.checkbox("Bruk samme terskler på alle horisonter", value=True)

def slider_pair(label_buy, label_sell, buy_default=0.60, sell_default=0.40):
    b = st.sidebar.slider(label_buy, 0.50, 0.90, buy_default, 0.01)
    s = st.sidebar.slider(label_sell, 0.10, 0.50, sell_default, 0.01)
    return b, s

if same_thr:
    b_all, s_all = slider_pair("KJØP hvis sannsynlighet >", "SELG hvis sannsynlighet <", 0.60, 0.40)
    thr = {k: (b_all, s_all) for _, k in HORIZONS}
else:
    bA, sA = slider_pair(f"{H_A}d • KJØP >", f"{H_A}d • SELG <", 0.60, 0.40)
    bB, sB = slider_pair(f"{H_B}d • KJØP >", f"{H_B}d • SELG <", 0.60, 0.40)
    bC, sC = slider_pair(f"{H_C}d • KJØP >", f"{H_C}d • SELG <", 0.60, 0.40)
    thr = {"A": (bA, sA), "B": (bB, sB), "C": (bC, sC)}

st.sidebar.markdown("---")
want_excel = st.sidebar.checkbox("Eksporter Excel (flere ark)", value=True)
eps = st.sidebar.number_input("Støyfilter (eps, %)", value=0.20, min_value=0.0, max_value=2.0, step=0.1,
                              help="Bevegelser med absolutt avkastning < eps% behandles som støy.")
st.sidebar.caption("Tips: Norske aksjer bruker .OL (EQNR.OL). Valuta = '=X' (EURNOK=X). Råvarer: CL=F, GC=F.")

# -----------------------------
# Datahjelpere (med caching)
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_history(ticker: str, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    return df

def get_close_series(df: pd.DataFrame) -> pd.Series:
    """
    Returner en 1D float-Serie for 'Close' uansett kolonnelayout (Single/MultiIndex).
    Fallback: første numeriske kolonne. Tom serie om ingenting finnes.
    """
    try:
        if "Close" in df.columns:
            s = df["Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            return pd.to_numeric(s.squeeze(), errors="coerce").astype(float)

        if isinstance(df.columns, pd.MultiIndex):
            if "Close" in df.columns.get_level_values(-1):
                s = df.xs("Close", axis=1, level=-1)
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                return pd.to_numeric(s.squeeze(), errors="coerce").astype(float)
    except Exception:
        pass

    num = df.select_dtypes(include=[np.number])
    if not num.empty:
        return pd.to_numeric(num.iloc[:, 0], errors="coerce").astype(float)

    return pd.Series(np.nan, index=df.index, dtype=float)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Rask felles feature-pipe som kan gjenbrukes på tvers av horisonter."""
    df = df.copy()
    close = get_close_series(df)

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

    # EMA / MACD / Bollinger
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

# ---- trygg label (1D) ----
def make_label(df: pd.DataFrame, horizon: int, eps_frac: float) -> pd.Series:
    """
    Mål: retning over neste N dager. eps_frac i % -> brøk.
    Returnerer alltid 1D Series.
    """
    eps_b = float(eps_frac) / 100.0
    close = get_close_series(df)
    fwd = close.shift(-horizon) / close - 1.0
    arr = np.where(fwd > eps_b, 1, np.where(fwd < -eps_b, 0, np.nan))
    return pd.Series(np.asarray(arr, dtype="float64").ravel(), index=df.index, name=f"Target_{horizon}")

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
        if va1 <= va0:
            continue
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

    # Tren endelig modell på hele datasettet
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

def analyze_ticker_multi(df_raw: pd.DataFrame, eps_pct: float, horizons: list[tuple[int,str]]) -> dict:
    """
    Bygger indikatorer én gang og trener modeller for alle 'horizons'.
    horizons: liste av (antall_dager, key) – f.eks. [(1,"A"), (3,"B"), (5,"C")]
    """
    out = {}
    if df_raw is None or df_raw.empty:
        for _, key in horizons:
            out[key] = {"proba": pd.Series(dtype=float), "acc": np.nan, "auc": np.nan,
                        "opt_thr": 0.5, "last_date": None}
        return out

    df = add_indicators(df_raw)

    feat_base = [c for c in FEATURES_ALL if c in df.columns and df[c].notna().any()]
    if len(feat_base) == 0:
        for _, key in horizons:
            out[key] = {"proba": pd.Series(dtype=float), "acc": np.nan, "auc": np.nan,
                        "opt_thr": 0.5, "last_date": None}
        return out

    for H, key in horizons:
        y = make_label(df, H, eps_pct)
        if y is None or len(y) != len(df):
            out[key] = {"proba": pd.Series(dtype=float), "acc": np.nan, "auc": np.nan,
                        "opt_thr": 0.5, "last_date": None}
            continue

        available = [c for c in feat_base if c in df.columns]
        if not available:
            out[key] = {"proba": pd.Series(dtype=float), "acc": np.nan, "auc": np.nan,
                        "opt_thr": 0.5, "last_date": None}
            continue

        pack = pd.concat([df[available], y], axis=1).dropna()
        if pack.empty or len(pack) < 120:
            # nøytral fallback
            neut_index = pack.index if len(pack) else df.index
            out[key] = {
                "proba": pd.Series(0.5, index=neut_index, name="proba"),
                "acc": np.nan, "auc": np.nan, "opt_thr": 0.5,
                "last_date": (pack.index[-1] if len(pack) else None),
            }
            continue

        X  = pack.loc[:, available]
        yv = pack[y.name]
        if len(np.unique(yv.values.astype(int))) < 2:
            neut = pd.Series(0.5, index=pack.index, name="proba")
            out[key] = {"proba": neut, "acc": np.nan, "auc": np.nan,
                        "opt_thr": 0.5, "last_date": pack.index[-1]}
            continue

        proba_full, acc, auc, opt_thr = walkforward_fit_predict(X, yv)
        out[key] = {
            "proba": proba_full, "acc": acc, "auc": auc,
            "opt_thr": opt_thr, "last_date": pack.index[-1],
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
        status.write(f"Henter og analyserer: {t} ({i}/{len(tickers)})")
        try:
            df_raw = fetch_history(t, start=start_date, end=end_date)
        except Exception:
            df_raw = pd.DataFrame()

        if df_raw is None or df_raw.empty:
            results.append({
                "Ticker": t,
                f"Prob_{H_A}d": np.nan, f"Rec_{H_A}d": "HOLD", f"Date_{H_A}d": "—",
                f"Prob_{H_B}d": np.nan, f"Rec_{H_B}d": "HOLD", f"Date_{H_B}d": "—",
                f"Prob_{H_C}d": np.nan, f"Rec_{H_C}d": "HOLD", f"Date_{H_C}d": "—",
                "Delta_C_A": np.nan, "Acc": np.nan, "AUC": np.nan, "Composite": np.nan
            })
            progress.progress(i/len(tickers))
            continue

        pack = analyze_ticker_multi(df_raw, eps_pct=eps, horizons=HORIZONS)

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

        pA = last_proba("A"); pB = last_proba("B"); pC = last_proba("C")
        dateA = expected_date(pack["A"]["last_date"], H_A)
        dateB = expected_date(pack["B"]["last_date"], H_B)
        dateC = expected_date(pack["C"]["last_date"], H_C)

        bA,sA = thr["A"]; bB,sB = thr["B"]; bC,sC = thr["C"]
        rA = rec_from_prob(pA, max(bA, pack["A"]["opt_thr"]), sA)
        rB = rec_from_prob(pB, max(bB, pack["B"]["opt_thr"]), sB)
        rC = rec_from_prob(pC, max(bC, pack["C"]["opt_thr"]), sC)

        probs = [x for x in [pA,pB,pC] if not np.isnan(x)]
        comp = float(np.mean(probs)) if probs else np.nan
        accs = [pack[k]["acc"] for k in ["A","B","C"] if not np.isnan(pack[k]["acc"])]
        aucs = [pack[k]["auc"] for k in ["A","B","C"] if not np.isnan(pack[k]["auc"])]
        acc = float(np.mean(accs)) if accs else np.nan
        auc = float(np.mean(aucs)) if aucs else np.nan

        results.append({
            "Ticker": t,
            f"Prob_{H_A}d": pA, f"Rec_{H_A}d": rA, f"Date_{H_A}d": dateA,
            f"Prob_{H_B}d": pB, f"Rec_{H_B}d": rB, f"Date_{H_B}d": dateB,
            f"Prob_{H_C}d": pC, f"Rec_{H_C}d": rC, f"Date_{H_C}d": dateC,
            "Delta_C_A": (pC - pA) if (not np.isnan(pC) and not np.isnan(pA)) else np.nan,
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
# Visning – tre kolonner
# -----------------------------
if run:
    df = pd.DataFrame(results)

    num_cols = [
        f"Prob_{H_A}d", f"Prob_{H_B}d", f"Prob_{H_C}d",
        "Acc", "AUC", "Delta_C_A", "Composite"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader(f"🟩 {H_A} dag{'er' if H_A>1 else ''} frem")
        df1 = df[["Ticker", f"Prob_{H_A}d", f"Rec_{H_A}d", f"Date_{H_A}d", "Acc"]].copy()
        df1 = df1.sort_values(f"Prob_{H_A}d", ascending=False)
        st.dataframe(
            style_df(df1, {f"Prob_{H_A}d": "{:.2%}", "Acc": "{:.2%}"}),
            use_container_width=True
        )

    with c2:
        st.subheader(f"🟦 {H_B} dag{'er' if H_B>1 else ''} frem")
        df3 = df[["Ticker", f"Prob_{H_B}d", f"Rec_{H_B}d", f"Date_{H_B}d", "Acc"]].copy()
        df3 = df3.sort_values(f"Prob_{H_B}d", ascending=False)
        st.dataframe(
            style_df(df3, {f"Prob_{H_B}d": "{:.2%}", "Acc": "{:.2%}"}),
            use_container_width=True
        )

    with c3:
        st.subheader(f"🟧 {H_C} dag{'er' if H_C>1 else ''} frem")
        df5 = df[["Ticker", f"Prob_{H_C}d", f"Rec_{H_C}d", f"Date_{H_C}d", "Acc", "Delta_C_A"]].copy()
        df5 = df5.sort_values(f"Prob_{H_C}d", ascending=False)
        st.dataframe(
            style_df(df5, {f"Prob_{H_C}d": "{:.2%}", "Acc": "{:.2%}", "Delta_C_A": "{:.2%}"}),
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("📋 Sammenligningstabell (alle horisonter)")
    cmp_df = df[[
        "Ticker",
        f"Prob_{H_A}d", f"Rec_{H_A}d", f"Date_{H_A}d",
        f"Prob_{H_B}d", f"Rec_{H_B}d", f"Date_{H_B}d",
        f"Prob_{H_C}d", f"Rec_{H_C}d", f"Date_{H_C}d",
        "Delta_C_A","Acc","AUC","Composite"
    ]].sort_values("Composite", ascending=False)

    st.dataframe(
        style_df(
            cmp_df,
            {
                f"Prob_{H_A}d": "{:.2%}",
                f"Prob_{H_B}d": "{:.2%}",
                f"Prob_{H_C}d": "{:.2%}",
                "Delta_C_A": "{:.2%}",
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
    st.subheader("📊 Detaljvisning (pris + sannsynlighet A/B/C)")
    sel_list = df["Ticker"].tolist() if not df.empty else []
    sel = st.selectbox("Velg ticker", sel_list)
    if sel:
        try:
            raw = fetch_history(sel, start=start_date, end=end_date)
            pack = analyze_ticker_multi(raw, eps_pct=eps, horizons=HORIZONS)
            close = get_close_series(raw)

            plot_df = pd.DataFrame({
                "Close": close,
                f"Prob_{H_A}d": pack["A"]["proba"].reindex(raw.index),
                f"Prob_{H_B}d": pack["B"]["proba"].reindex(raw.index),
                f"Prob_{H_C}d": pack["C"]["proba"].reindex(raw.index),
            }).dropna(subset=["Close"])

            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(plot_df.index, plot_df["Close"])
            ax1.set_xlabel("Dato")
            ax1.set_ylabel("Pris")

            ax2 = ax1.twinx()
            if f"Prob_{H_A}d" in plot_df: ax2.plot(plot_df.index, plot_df[f"Prob_{H_A}d"], alpha=0.9)
            if f"Prob_{H_B}d" in plot_df: ax2.plot(plot_df.index, plot_df[f"Prob_{H_B}d"], alpha=0.9)
            if f"Prob_{H_C}d" in plot_df: ax2.plot(plot_df.index, plot_df[f"Prob_{H_C}d"], alpha=0.9)
            ax2.axhline(0.5, linestyle="--", alpha=0.6)
            ax2.set_ylabel("Sannsynlighet")

            plt.title(f"{sel}: Pris + sannsynlighet ({H_A}/{H_B}/{H_C} dager)")
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
                df1.to_excel(writer, index=False, sheet_name=f"{H_A}d")
                df3.to_excel(writer, index=False, sheet_name=f"{H_B}d")
                df5.to_excel(writer, index=False, sheet_name=f"{H_C}d")
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
                f"Prob_{H_A}d": "{:.2%}",
                f"Prob_{H_B}d": "{:.2%}",
                f"Prob_{H_C}d": "{:.2%}",
                "Delta_C_A": "{:.2%}",
                "Acc": "{:.2%}",
                "AUC": "{:.3f}",
                "Composite": "{:.2%}",
            }
        ),
        use_container_width=True
    )

else:
    st.info("Velg/skriv tickere i sidepanelet og trykk **🔎 Skann og sammenlign** for å starte.")


























