# app_v6.py
# 📈 Aksjeanalyse – Pro v6 Dashboard
# Multi-horisont (A/B/C – justerbare dager) – side om side
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
st.caption("Multi-horisont modell (A/B/C – justerbare dager) • Build: v6.2 – oktober 2025")

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
    "OMX30 (utvalg)": [
        "VOLV-B.ST","ERIC-B.ST","SAND.ST","ATCO-A.ST","ATCO-B.ST","ESSITY-B.ST","SWED-A.ST","SEB-A.ST","ALFA.ST","TELIA.ST",
        "ABB.ST","HEXA-B.ST","SKF-B.ST","BOL.ST","INVE-B.ST","EVO.ST","KINV-B.ST","NDA-SE.ST","MTG-B.ST","SCA-B.ST"
    ],
    "Råvarer": ["CL=F","BZ=F","NG=F","GC=F","SI=F","HG=F","ZC=F","ZW=F","ZS=F"],
    "Valuta (Forex)": ["EURUSD=X","USDJPY=X","GBPUSD=X","AUDUSD=X","USDCAD=X","USDCHF=X","EURNOK=X","USDNOK=X","EURGBP=X"],
    "Krypto": ["BTC-USD","ETH-USD","SOL-USD","XRP-USD","ADA-USD"]
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

# Dato
col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    start_date = st.sidebar.date_input("Startdato", pd.to_datetime("2019-01-01"))
with col_d2:
    end_date = st.sidebar.date_input("Sluttdato", pd.Timestamp.today())

st.sidebar.markdown("---")

# Justerbare horisonter (1–30 dager)
st.sidebar.subheader("Horisont (antall dager)")
H_A = int(st.sidebar.slider("Horisont A", 1, 30, 1, 1))
H_B = int(st.sidebar.slider("Horisont B", 1, 30, 3, 1))
H_C = int(st.sidebar.slider("Horisont C", 1, 30, 5, 1))
HORIZONS = [("A", H_A), ("B", H_B), ("C", H_C)]  # id, dager

st.sidebar.markdown("---")
st.sidebar.subheader("Terskler for anbefaling")
same_thr = st.sidebar.checkbox("Bruk samme terskler på alle horisonter", value=True)

def slider_pair(label_buy, label_sell, buy_default=0.60, sell_default=0.40):
    b = st.sidebar.slider(label_buy, 0.50, 0.90, buy_default, 0.01)
    s = st.sidebar.slider(label_sell, 0.10, 0.50, sell_default, 0.01)
    return b, s

if same_thr:
    b_all, s_all = slider_pair("KJØP hvis sannsynlighet >", "SELG hvis sannsynlighet <", 0.60, 0.40)
    thr = {hid: (b_all, s_all) for hid, _ in HORIZONS}
else:
    thr = {}
    for hid, days in HORIZONS:
        b, s = slider_pair(f"{hid} ({days}d) • KJØP >", f"{hid} ({days}d) • SELG <", 0.60, 0.40)
        thr[hid] = (b, s)

st.sidebar.markdown("---")
want_excel = st.sidebar.checkbox("Eksporter Excel (flere ark)", value=True)
eps = st.sidebar.number_input(
    "Støyfilter (eps, %)",
    value=0.20, min_value=0.0, max_value=2.0, step=0.10,
    help="Bevegelser med absolutt avkastning < eps% behandles som støy (nøytral)."
)
st.sidebar.caption("Tips: Norske aksjer bruker .OL (EQNR.OL). Valuta = '=X' (EURNOK=X). Råvarer: CL=F, GC=F.")

# -----------------------------
# Datahjelpere (med caching)
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_history(ticker: str, start, end):
    return yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

def get_close_series(df: pd.DataFrame) -> pd.Series:
    """
    Returner en 1-D Series med Close (eller Adj Close) uansett om
    df har vanlige kolonner eller MultiIndex. Tom Series hvis ikke mulig.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    # Vanlige kolonner
    for name in ["Close", "Adj Close"]:
        if name in df.columns:
            s = df[name]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            return pd.to_numeric(s.squeeze(), errors="coerce")

    # MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        for name in ["Close", "Adj Close"]:
            if name in level0:
                s = df[name]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                return pd.to_numeric(s.squeeze(), errors="coerce")

    return pd.Series(dtype=float, index=df.index)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Sikker feature-pipe som tåler MultiIndex og rare datasett."""
    close = get_close_series(df)
    if close.empty:
        return pd.DataFrame(index=df.index)

    out = pd.DataFrame(index=close.index)

    # Basale features
    out["ret1"]  = close.pct_change()
    out["ret3"]  = close.pct_change(3)
    out["ret5"]  = close.pct_change(5)
    out["ma5"]   = close.rolling(5).mean()
    out["ma20"]  = close.rolling(20).mean()
    out["vol10"] = close.pct_change().rolling(10).std()
    out["trend20"] = (out["ma20"] - out["ma5"]) / out["ma20"]

    # RSI(14)
    delta    = close.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs       = avg_gain / avg_loss
    out["rsi14"] = 100 - (100 / (1 + rs))

    # EMA og MACD og Bollinger%
    out["ema20"] = close.ewm(span=20, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()
    out["ema_gap"] = (out["ema20"] - out["ema50"]) / out["ema50"]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"]     = macd
    out["macd_sig"] = signal
    out["macd_hist"] = macd - signal

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    out["bb_pct"]   = (close - lower) / (upper - lower)
    out["bb_width"] = (upper - lower) / ma20

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(method="ffill").fillna(method="bfill")
    return out

def make_label(close_series: pd.Series, horizon: int, eps_frac: float) -> pd.Series:
    """Binary label: 1 hvis fwd > eps, 0 hvis fwd < -eps, ellers NaN."""
    eps = eps_frac / 100.0
    close = pd.to_numeric(close_series, errors="coerce")
    fwd = close.shift(-horizon) / close - 1.0
    arr = np.where(fwd > eps, 1, np.where(fwd < -eps, 0, np.nan))
    return pd.Series(arr.astype("float64"), index=close.index, name=f"Target_{horizon}")

FEATURES_ALL = [
    "ret1","ret3","ret5","ma5","ma20","vol10","trend20","rsi14",
    "ema20","ema50","ema_gap","bb_pct","bb_width","macd","macd_sig","macd_hist"
]

def fit_predict_single_horizon(df_raw: pd.DataFrame, horizon: int, eps_pct: float):
    """
    Enkeltløp: bygg features, lag labels, tren kalibrert GBDT og gi sannsynlighet (hele historikken).
    Robust mot lite/rare data – gir nøytral 0.5 hvis det ikke går.
    """
    feats = add_indicators(df_raw)
    if feats.empty:
        return pd.Series(dtype=float), np.nan, np.nan, 0.5, None

    close = get_close_series(df_raw)
    y = make_label(close, horizon, eps_pct)

    cols = [c for c in FEATURES_ALL if c in feats.columns and feats[c].notna().any()]
    if not cols or y.isna().all():
        return pd.Series(dtype=float), np.nan, np.nan, 0.5, None

    pack = pd.concat([feats[cols], y], axis=1).dropna()
    if pack.empty or len(pack) < 120:
        # For kort historikk – gi 0.5
        neut = pd.Series(0.5, index=(pack.index if len(pack) else feats.index), name="proba")
        last_idx = (pack.index[-1] if len(pack) else (feats.index[-1] if len(feats) else None))
        return neut, np.nan, np.nan, 0.5, last_idx

    X = pack[cols]
    yv = pack[y.name].astype(int)

    # Krever to klasser for å trene
    if len(np.unique(yv.values)) < 2:
        neut = pd.Series(0.5, index=pack.index, name="proba")
        return neut, np.nan, np.nan, 0.5, pack.index[-1]

    # Enkel validering: siste 10% som "val"
    n = len(X)
    split = max(int(n * 0.90), min(300, n - 1))
    Xtr, Xva = X.iloc[:split], X.iloc[split:]
    ytr, yva = yv.iloc[:split], yv.iloc[split:]

    scaler = RobustScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xva_s = scaler.transform(Xva)

    base = GradientBoostingClassifier(random_state=0)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(Xtr_s, ytr)

    pva = clf.predict_proba(Xva_s)[:, 1]
    try:
        acc = accuracy_score(yva, (pva >= 0.5).astype(int))
        auc = roc_auc_score(yva, pva) if len(np.unique(yva)) == 2 else np.nan
    except Exception:
        acc, auc = np.nan, np.nan

    # Tren på hele samplet for full serie
    scaler = RobustScaler().fit(X)
    Xs = scaler.transform(X)
    clf = CalibratedClassifierCV(GradientBoostingClassifier(random_state=0), method="isotonic", cv=3)
    clf.fit(Xs, yv)
    proba_full = clf.predict_proba(Xs)[:, 1]
    proba_full = pd.Series(proba_full, index=X.index, name="proba")

    # Enkel cut-off-optimalisering på val (om mulig)
    opt_thr = 0.5
    if len(Xva):
        cand = np.linspace(0.3, 0.7, 41)
        try:
            accs = [accuracy_score(yva, (pva >= t).astype(int)) for t in cand]
            opt_thr = float(cand[int(np.argmax(accs))])
        except Exception:
            pass

    return proba_full, float(acc) if acc==acc else np.nan, float(auc) if auc==auc else np.nan, float(opt_thr), pack.index[-1]

def analyze_ticker_multi(df_raw: pd.DataFrame, eps_pct: float, horizons: list[tuple[str,int]]):
    """
    Kjør tre justerbare horisonter (id, dager) og returner et pack per horisont-id.
    """
    out = {}
    if df_raw is None or df_raw.empty:
        for hid, _ in horizons:
            out[hid] = {"proba": pd.Series(dtype=float), "acc": np.nan, "auc": np.nan,
                        "opt_thr": 0.5, "last_date": None, "days": _}
        return out

    for hid, days in horizons:
        p, acc, auc, opt_thr, last_idx = fit_predict_single_horizon(df_raw, days, eps_pct)
        out[hid] = {"proba": p, "acc": acc, "auc": auc, "opt_thr": opt_thr, "last_date": last_idx, "days": days}
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

        pack = analyze_ticker_multi(df_raw, eps_pct=eps, horizons=HORIZONS)

        # Hent siste sannsynlighet for hver horisont
        def last_proba(hid, default=np.nan):
            try:
                s = pack[hid]["proba"]
                if len(s) == 0:
                    return default
                v = float(s.iloc[-1])
                if np.isnan(v):
                    return default
                return v
            except Exception:
                return default

        # Les ut A/B/C
        pA = last_proba("A"); dA = pack["A"]["days"]
        pB = last_proba("B"); dB = pack["B"]["days"]
        pC = last_proba("C"); dC = pack["C"]["days"]

        dateA = expected_date(pack["A"]["last_date"], dA)
        dateB = expected_date(pack["B"]["last_date"], dB)
        dateC = expected_date(pack["C"]["last_date"], dC)

        bA, sA = thr["A"]; bB, sB = thr["B"]; bC, sC = thr["C"]
        rA = rec_from_prob(pA, max(bA, pack["A"]["opt_thr"]), sA)
        rB = rec_from_prob(pB, max(bB, pack["B"]["opt_thr"]), sB)
        rC = rec_from_prob(pC, max(bC, pack["C"]["opt_thr"]), sC)

        probs = [x for x in [pA, pB, pC] if not np.isnan(x)]
        comp = float(np.mean(probs)) if probs else np.nan

        accs = [pack[h]["acc"] for h in ["A","B","C"] if not np.isnan(pack[h]["acc"])]
        aucs = [pack[h]["auc"] for h in ["A","B","C"] if not np.isnan(pack[h]["auc"])]
        acc = float(np.mean(accs)) if accs else np.nan
        auc = float(np.mean(aucs)) if aucs else np.nan

        results.append({
            "Ticker": t,
            f"Prob_{dA}d": pA, f"Rec_{dA}d": rA, f"Date_{dA}d": dateA,
            f"Prob_{dB}d": pB, f"Rec_{dB}d": rB, f"Date_{dB}d": dateB,
            f"Prob_{dC}d": pC, f"Rec_{dC}d": rC, f"Date_{dC}d": dateC,
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

    # Oppdag faktiske dagstall for kolonner
    daysA, daysB, daysC = H_A, H_B, H_C
    cA = [f"Ticker", f"Prob_{daysA}d", f"Rec_{daysA}d", f"Date_{daysA}d", "Acc"]
    cB = [f"Ticker", f"Prob_{daysB}d", f"Rec_{daysB}d", f"Date_{daysB}d", "Acc"]
    cC = [f"Ticker", f"Prob_{daysC}d", f"Rec_{daysC}d", f"Date_{daysC}d", "Acc"]

    # Sørg for numeriske kolonner
    for c in df.columns:
        if c.startswith("Prob_") or c in ["Acc","AUC","Composite"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader(f"🟩 {daysA} dag(er) frem")
        df1 = df[cA].copy()
        df1 = df1.sort_values(f"Prob_{daysA}d", ascending=False, na_position="last")
        st.dataframe(
            style_df(df1, {f"Prob_{daysA}d":"{:.2%}", "Acc":"{:.2%}"}),
            use_container_width=True
        )

    with c2:
        st.subheader(f"🟦 {daysB} dag(er) frem")
        df3 = df[cB].copy()
        df3 = df3.sort_values(f"Prob_{daysB}d", ascending=False, na_position="last")
        st.dataframe(
            style_df(df3, {f"Prob_{daysB}d":"{:.2%}", "Acc":"{:.2%}"}),
            use_container_width=True
        )

    with c3:
        st.subheader(f"🟧 {daysC} dag(er) frem")
        df5 = df[cC + ["Composite"]].copy()
        df5 = df5.sort_values(f"Prob_{daysC}d", ascending=False, na_position="last")
        st.dataframe(
            style_df(df5, {f"Prob_{daysC}d":"{:.2%}", "Acc":"{:.2%}", "Composite":"{:.2%}"}),
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("📋 Sammenligningstabell (alle horisonter)")

    cmp_cols = ["Ticker",
                f"Prob_{daysA}d", f"Rec_{daysA}d", f"Date_{daysA}d",
                f"Prob_{daysB}d", f"Rec_{daysB}d", f"Date_{daysB}d",
                f"Prob_{daysC}d", f"Rec_{daysC}d", f"Date_{daysC}d",
                "Acc","AUC","Composite"]
    cmp_df = df[cmp_cols].sort_values("Composite", ascending=False, na_position="last")

    st.dataframe(
        style_df(
            cmp_df,
            {
                f"Prob_{daysA}d":"{:.2%}",
                f"Prob_{daysB}d":"{:.2%}",
                f"Prob_{daysC}d":"{:.2%}",
                "Acc":"{:.2%}",
                "AUC":"{:.3f}",
                "Composite":"{:.2%}",
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
            plot_df = pd.DataFrame({"Close": close})
            for hid, days in HORIZONS:
                key = f"Prob_{hid}"
                plot_df[key] = pack[hid]["proba"].reindex(plot_df.index)

            plot_df = plot_df.dropna(subset=["Close"])

            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(plot_df.index, plot_df["Close"])
            ax1.set_xlabel("Dato")
            ax1.set_ylabel("Pris")

            ax2 = ax1.twinx()
            for hid, _ in HORIZONS:
                key = f"Prob_{hid}"
                if key in plot_df:
                    ax2.plot(plot_df.index, plot_df[key], alpha=0.9, label=key)
            ax2.axhline(0.5, linestyle="--", alpha=0.6)
            ax2.set_ylabel("Sannsynlighet")
            ax2.legend(loc="upper left")

            plt.title(f"{sel}: Pris + sannsynlighet (A/B/C)")
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
                df[cA].to_excel(writer, index=False, sheet_name=f"{daysA}d")
                df[cB].to_excel(writer, index=False, sheet_name=f"{daysB}d")
                df[cC + ["Composite"]].to_excel(writer, index=False, sheet_name=f"{daysC}d")
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
                f"Prob_{daysA}d":"{:.2%}",
                f"Prob_{daysB}d":"{:.2%}",
                f"Prob_{daysC}d":"{:.2%}",
                "Acc":"{:.2%}",
                "AUC":"{:.3f}",
                "Composite":"{:.2%}",
            }
        ),
        use_container_width=True
    )

else:
    st.info("Velg/skriv tickere i sidepanelet, justér horisonter og trykk **🔎 Skann og sammenlign** for å starte.")






























