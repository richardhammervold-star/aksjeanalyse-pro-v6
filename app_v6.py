# app_v6.py
# 📈 Aksjeanalyse – Pro v6 Dashboard (justerbare horisonter)
# Build: v6.2 – oktober 2025

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
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
st.caption("Multi-horisont modell (justerbare horisonter A/B/C) • Build: v6.2 – oktober 2025")

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
    "USA – Teknologi": [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","AMD","CRM","ORCL","NFLX","NOW","INTU","PANW"
    ],
    "Krypto": ["BTC-USD","ETH-USD","SOL-USD","XRP-USD","ADA-USD"],
    "Valuta (Forex)": ["EURUSD=X","USDJPY=X","GBPUSD=X","AUDUSD=X","USDCAD=X","USDCHF=X","EURNOK=X","USDNOK=X","EURGBP=X"],
    "Råvarer": ["CL=F","BZ=F","NG=F","GC=F","SI=F","HG=F","ZC=F","ZW=F","ZS=F"]
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
    start_date = st.sidebar.date_input("Startdato", pd.to_datetime("2019-01-01"))
with col_d2:
    end_date = st.sidebar.date_input("Sluttdato", pd.Timestamp.today())

st.sidebar.markdown("---")
st.sidebar.subheader("Horisont (i dager)")
hA = st.sidebar.slider("Horisont A (vises som '1 dag frem')", 1, 30, 1)
hB = st.sidebar.slider("Horisont B (vises som '3 dager frem')", 1, 30, 3)
hC = st.sidebar.slider("Horisont C (vises som '5 dager frem')", 1, 30, 5)
HORIZONS = [hA, hB, hC]

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
    bA, sA = slider_pair(f"{hA} d • KJØP >", f"{hA} d • SELG <", 0.60, 0.40)
    bB, sB = slider_pair(f"{hB} d • KJØP >", f"{hB} d • SELG <", 0.60, 0.40)
    bC, sC = slider_pair(f"{hC} d • KJØP >", f"{hC} d • SELG <", 0.60, 0.40)
    thr = {"A": (bA, sA), "B": (bB, sB), "C": (bC, sC)}

st.sidebar.markdown("---")
want_excel = st.sidebar.checkbox("Eksporter Excel (flere ark)", value=True)
eps = st.sidebar.number_input("Støyfilter (eps, % – nøytral sone rundt 0)", value=0.0, min_value=0.0, max_value=2.0, step=0.1)
st.sidebar.caption("Tips: Norske aksjer bruker .OL (EQNR.OL). Valuta = '=X' (EURNOK=X). Råvarer: CL=F, GC=F.")

# -----------------------------
# Datahjelpere (med caching)
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_history(ticker: str, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    return df

# -----------------------------
# Indikatorer + labels
# -----------------------------
FEATURES_ALL = [
    "ret1","ret3","ma5","ma20","vol10","rsi14",
    "bb_pct","bb_width","macd","macd_sig","macd_hist"
]

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in df.columns:
        # tom, returner tom ramme – håndteres oppstrøms
        return pd.DataFrame(index=df.index)

    # Trygg konvertering
    close = pd.to_numeric(df["Close"], errors="coerce")
    out = pd.DataFrame(index=df.index).copy()

    # Enkle returer/MA/vol
    out["ret1"]  = close.pct_change()
    out["ret3"]  = close.pct_change(3)
    out["ma5"]   = close.rolling(5).mean()
    out["ma20"]  = close.rolling(20).mean()
    out["vol10"] = close.pct_change().rolling(10).std()

    # RSI(14)
    delta    = close.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs       = avg_gain / avg_loss
    out["rsi14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    out["macd"]     = macd
    out["macd_sig"] = sig
    out["macd_hist"] = macd - sig

    # Bollinger
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + 2*std20
    lower = ma20 - 2*std20
    out["bb_pct"]   = (close - lower) / (upper - lower)
    out["bb_width"] = (upper - lower) / ma20

    # Rens
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(method="ffill").fillna(method="bfill")
    return out

def make_label(df_close: pd.Series, horizon: int, eps_frac: float) -> pd.Series:
    """1 hvis pris om N dager er > eps over i dag, 0 hvis < -eps, ellers 0.5 -> vi gjør 0/1 og masker nøytral ut etterpå."""
    eps = eps_frac / 100.0
    close = pd.to_numeric(df_close, errors="coerce")
    fwd = close.shift(-horizon) / close - 1.0
    y = pd.Series(np.where(fwd > eps, 1,
                  np.where(fwd < -eps, 0, np.nan)),
                  index=close.index, name=f"Target_{horizon}")
    return y

# -----------------------------
# Modellering (robust)
# -----------------------------
def fit_predict_single_horizon(df_raw: pd.DataFrame, horizon: int, eps_pct: float):
    """Returner sannsynlighetsserie (samme index som df_raw) for valgt horisont."""
    if df_raw is None or df_raw.empty or "Close" not in df_raw:
        return pd.Series(dtype=float), np.nan

    feats = add_indicators(df_raw)
    if feats.empty:
        return pd.Series(dtype=float), np.nan

    # Velg features som faktisk finnes
    available = [c for c in FEATURES_ALL if c in feats.columns]
    if not available:
        return pd.Series(dtype=float), np.nan

    y = make_label(df_raw["Close"], horizon, eps_pct)
    pack = pd.concat([feats[available], y], axis=1)
    pack = pack.dropna()
    if pack.empty or len(pack) < 60:
        # for lite data
        return pd.Series(0.5, index=(pack.index if len(pack) else feats.index), name="proba"), np.nan

    X = pack[available].copy()
    yv = pack[y.name].astype(int)

    # Robust skalering + kalibrert GB
    scaler = RobustScaler().fit(X)
    Xs = scaler.transform(X)
    base = GradientBoostingClassifier(random_state=0)
    clf  = CalibratedClassifierCV(base, method="isotonic", cv=3)

    try:
        clf.fit(Xs, yv)
        proba = clf.predict_proba(Xs)[:, 1]
        proba = pd.Series(proba, index=pack.index, name="proba")
    except Exception:
        proba = pd.Series(0.5, index=pack.index, name="proba")

    # Merk: vi returnerer serie (for å kunne plukke siste verdi)
    return proba, pack.index[-1]

def analyze_ticker_multi(df_raw: pd.DataFrame, eps_pct: float, horizons: list[int]):
    """Trener tre uavhengige modeller – én pr horisont – og returnerer pakke."""
    out = {}
    for H in horizons:
        p, last_idx = fit_predict_single_horizon(df_raw, H, eps_pct)
        out[H] = {"proba": p, "last_date": last_idx}
    return out

# -----------------------------
# Hjelpere
# -----------------------------
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
                "Prob_A": np.nan, "Rec_A": "HOLD", "Date_A": "—",
                "Prob_B": np.nan, "Rec_B": "HOLD", "Date_B": "—",
                "Prob_C": np.nan, "Rec_C": "HOLD", "Date_C": "—",
                "Delta_C_A": np.nan
            })
            progress.progress(i / len(tickers))
            continue

        pack = analyze_ticker_multi(df_raw, eps_pct=eps, horizons=HORIZONS)

        def last_prob_for(H, default=0.5):
            try:
                s = pack[H]["proba"]
                if len(s) == 0:
                    return default
                v = float(s.iloc[-1])
                return v if not np.isnan(v) else default
            except Exception:
                return default

        pA = last_prob_for(HORIZONS[0])
        pB = last_prob_for(HORIZONS[1])
        pC = last_prob_for(HORIZONS[2])

        dateA = expected_date(pack[HORIZONS[0]]["last_date"], HORIZONS[0])
        dateB = expected_date(pack[HORIZONS[1]]["last_date"], HORIZONS[1])
        dateC = expected_date(pack[HORIZONS[2]]["last_date"], HORIZONS[2])

        bA,sA = thr["A"]; bB,sB = thr["B"]; bC,sC = thr["C"]
        rA = rec_from_prob(pA, bA, sA)
        rB = rec_from_prob(pB, bB, sB)
        rC = rec_from_prob(pC, bC, sC)

        results.append({
            "Ticker": t,
            "Prob_A": pA, "Rec_A": rA, "Date_A": dateA,
            "Prob_B": pB, "Rec_B": rB, "Date_B": dateB,
            "Prob_C": pC, "Rec_C": rC, "Date_C": dateC,
            "Delta_C_A": (pC - pA) if (not np.isnan(pC) and not np.isnan(pA)) else np.nan,
        })

        progress.progress(i / len(tickers))

    status.empty()
    progress.empty()

# -----------------------------
# Visning – tre kolonner
# -----------------------------
if run:
    df = pd.DataFrame(results)
    # Sørg for numerisk formatering
    for c in ["Prob_A","Prob_B","Prob_C","Delta_C_A"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader(f"🟩 {hA} dag(er) frem")
        df1 = df[["Ticker", "Prob_A", "Rec_A", "Date_A"]].copy()
        df1 = df1.sort_values("Prob_A", ascending=False)
        st.dataframe(style_df(df1, {"Prob_A": "{:.2%}"}), use_container_width=True)

    with c2:
        st.subheader(f"🟦 {hB} dag(er) frem")
        df2 = df[["Ticker", "Prob_B", "Rec_B", "Date_B"]].copy()
        df2 = df2.sort_values("Prob_B", ascending=False)
        st.dataframe(style_df(df2, {"Prob_B": "{:.2%}"}), use_container_width=True)

    with c3:
        st.subheader(f"🟧 {hC} dag(er) frem")
        df3 = df[["Ticker", "Prob_C", "Rec_C", "Date_C", "Delta_C_A"]].copy()
        df3 = df3.sort_values("Prob_C", ascending=False)
        st.dataframe(style_df(df3, {"Prob_C": "{:.2%}", "Delta_C_A": "{:.2%}"}), use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Sammenligningstabell (alle horisonter)")
    cmp_df = df[[
        "Ticker",
        "Prob_A","Rec_A","Date_A",
        "Prob_B","Rec_B","Date_B",
        "Prob_C","Rec_C","Date_C",
        "Delta_C_A"
    ]].sort_values("Prob_C", ascending=False)

    st.dataframe(
        style_df(
            cmp_df,
            {
                "Prob_A": "{:.2%}",
                "Prob_B": "{:.2%}",
                "Prob_C": "{:.2%}",
                "Delta_C_A": "{:.2%}",
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
            plot_df = pd.DataFrame({
                "Close": raw["Close"],
                f"Prob_{hA}d": pack[HORIZONS[0]]["proba"].reindex(raw.index),
                f"Prob_{hB}d": pack[HORIZONS[1]]["proba"].reindex(raw.index),
                f"Prob_{hC}d": pack[HORIZONS[2]]["proba"].reindex(raw.index),
            }).dropna(subset=["Close"])

            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(plot_df.index, plot_df["Close"])
            ax1.set_xlabel("Dato")
            ax1.set_ylabel("Pris")

            ax2 = ax1.twinx()
            if f"Prob_{hA}d" in plot_df: ax2.plot(plot_df.index, plot_df[f"Prob_{hA}d"], alpha=0.9)
            if f"Prob_{hB}d" in plot_df: ax2.plot(plot_df.index, plot_df[f"Prob_{hB}d"], alpha=0.9)
            if f"Prob_{hC}d" in plot_df: ax2.plot(plot_df.index, plot_df[f"Prob_{hC}d"], alpha=0.9)
            ax2.axhline(0.5, linestyle="--", alpha=0.6)
            ax2.set_ylabel("Sannsynlighet")

            plt.title(f"{sel}: Pris + sannsynlighet ({hA}/{hB}/{hC}d)")
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
                df1.to_excel(writer, index=False, sheet_name=f"{hA}d")
                df2.to_excel(writer, index=False, sheet_name=f"{hB}d")
                df3.to_excel(writer, index=False, sheet_name=f"{hC}d")
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
    st.info("Velg/skriv tickere i sidepanelet, justér horisonter og terskler – og trykk **🔎 Skann og sammenlign** for å starte.")




























