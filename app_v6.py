# app_v6.py
# 📈 Aksjeanalyse – Pro v6 Dashboard
# Multi-horisont (1d / 3d / 5d) – side om side
# Build: v6.1 – oktober 2025

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
st.set_page_config(page_title="Aksjeanalyse – Pro v6 Dashboard", layout="wide")

st.title("📈 Aksjeanalyse – Pro v6 Dashboard")
st.caption("Multi-horisont modell (1d / 3d / 5d) • Build: v6.1 – oktober 2025")

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
    value=", ".join(tickers),
    height=120
)
tickers = [t.strip().upper() for chunk in user_tickers.split("\n") for t in chunk.split(",") if t.strip()]

col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    start_date = st.date_input("Startdato", pd.to_datetime("2015-01-02"))
with col_d2:
    end_date = st.date_input("Sluttdato", pd.Timestamp.today())

st.sidebar.markdown("---")
st.sidebar.subheader("Terskler")
same_thr = st.sidebar.checkbox("Bruk samme terskler på alle horisonter", value=False)

def slider_pair(label_buy, label_sell, buy_default=0.65, sell_default=0.50):
    b = st.sidebar.slider(label_buy, 0.50, 0.90, buy_default, 0.01)
    s = st.sidebar.slider(label_sell, 0.10, 0.50, sell_default, 0.01)
    return b, s

if same_thr:
    b_all, s_all = slider_pair("KJØP hvis sannsynlighet >", "SELG hvis sannsynlighet <", 0.65, 0.50)
    thr = {"1d": (b_all, s_all), "3d": (b_all, s_all), "5d": (b_all, s_all)}
else:
    b1, s1 = slider_pair("1 dag • KJØP >", "1 dag • SELG <", 0.65, 0.50)
    b3, s3 = slider_pair("3 dager • KJØP >", "3 dager • SELG <", 0.65, 0.50)
    b5, s5 = slider_pair("5 dager • KJØP >", "5 dager • SELG <", 0.65, 0.50)
    thr = {"1d": (b1, s1), "3d": (b3, s3), "5d": (b5, s5)}

st.sidebar.markdown("---")
want_excel = st.sidebar.checkbox("Eksporter Excel (flere ark)", value=True)
eps = st.sidebar.number_input(
    "Støyfilter (eps, % – nøytral sone ±eps)",
    value=0.2, min_value=0.0, max_value=2.0, step=0.1
)

# -----------------------------
# Datahjelpere (med caching)
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_history(ticker: str, start, end) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Rask feature-pipe."""
    df = df.copy()
    close = df["Close"].astype(float)

    df["ret1"]  = close.pct_change()
    df["ret3"]  = close.pct_change(3)
    df["ret5"]  = close.pct_change(5)
    df["ma5"]   = close.rolling(5).mean()
    df["ma20"]  = close.rolling(20).mean()
    df["vol10"] = df["ret1"].rolling(10).std()
    df["trend20"] = (df["ma20"] - df["ma5"]) / df["ma20"]

    delta = close.diff()
    gain  = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi14"] = 100 - (100 / (1 + rs))

    df["ema20"]   = close.ewm(span=20, adjust=False).mean()
    df["ema50"]   = close.ewm(span=50, adjust=False).mean()
    df["ema_gap"] = (df["ema20"] - df["ema50"]) / df["ema50"]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"]     = macd
    df["macd_sig"] = signal
    df["macd_hist"]= macd - signal

    ma20  = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    df["bb_pct"]   = (close - lower) / (upper - lower)
    df["bb_width"] = (upper - lower) / ma20

    return df

# ---- 1D label med nøytral sone (eps %) ----
def make_label(df: pd.DataFrame, horizon: int, eps_frac: float) -> pd.Series:
    """
    Label: 1 hvis fremtidig avkastning > +eps, 0 hvis < -eps, ellers NaN (nøytral).
    eps_frac er prosent. Returnerer ALLTID en 1D Series med samme index som df.
    """
    eps = float(eps_frac) / 100.0

    close = df["Close"].astype(float)
    fwd = close.shift(-horizon) / close - 1.0        # framtidsavkastning

    # Tving til ren 1D numpy-array (noen miljøer kan gi (n,1))
    fwd_np = np.asarray(fwd, dtype="float64").reshape(-1)

    # 1 (opp), 0 (ned), NaN (nøytral sone)
    arr = np.where(fwd_np > eps, 1.0, np.where(fwd_np < -eps, 0.0, np.nan))
    arr = np.asarray(arr, dtype="float64").reshape(-1)

    return pd.Series(arr, index=df.index, name=f"Target_{horizon}")

FEATURES_ALL = [
    "ret1","ret3","ret5","ma5","ma20","vol10","trend20","rsi14",
    "ema20","ema50","ema_gap","bb_pct","bb_width","macd","macd_sig","macd_hist"
]

def walkforward_fit_predict(X: pd.DataFrame, y: pd.Series):
    """
    Expanding walk-forward CV + trening av endelig modell.
    Dropp NaN i y (nøytral-sone) for trening/validering, men produser sannsynligheter
    for HELE datasettet til slutt.
    """
    # Behold en kopi av full index for slutt-prognose
    full_index = X.index

    # Kun rader med gyldig label brukes til trening/validering
    mask = y.notna()
    X_tr = X.loc[mask]
    y_tr = y.loc[mask]

    n = len(X_tr)
    # Ikke nok data -> returner nøytral 0.5
    if n < 120 or len(np.unique(y_tr.astype(int))) < 2:
        return pd.Series(0.5, index=full_index), np.nan, np.nan, 0.5

    anchors = [int(n * 0.60), int(n * 0.70), int(n * 0.80)]
    # Vi samler valideringsproba for rader som faktisk brukes i CV
    cv_probs = pd.Series(np.nan, index=X_tr.index)
    val_accs, val_aucs, thrs = [], [], []

    for a in anchors:
        tr0, tr1 = 0, a
        va0, va1 = a, min(a + int(n * 0.10), n - 1)

        Xtr, ytr = X_tr.iloc[tr0:tr1], y_tr.iloc[tr0:tr1]
        Xva, yva = X_tr.iloc[va0:va1], y_tr.iloc[va0:va1]
        if len(Xva) == 0 or len(np.unique(ytr.astype(int))) < 2:
            continue

        scaler = RobustScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xva_s = scaler.transform(Xva)

        base = GradientBoostingClassifier(random_state=0)
        clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        clf.fit(Xtr_s, ytr.astype(int))

        p = clf.predict_proba(Xva_s)[:, 1]
        cv_probs.loc[Xva.index] = p

        # Finn beste cutoff for accuracy
        cand = np.linspace(0.3, 0.7, 41)
        accs = [accuracy_score(yva.astype(int), (p >= t).astype(int)) for t in cand]
        t_star = float(cand[int(np.argmax(accs))])
        thrs.append(t_star)

        val_accs.append(max(accs))
        try:
            val_aucs.append(roc_auc_score(yva.astype(int), p))
        except Exception:
            pass

    # Tren endelig modell på all tilgjengelig treningsdata (uten NaN i y)
    scaler_full = RobustScaler().fit(X_tr)
    X_tr_s = scaler_full.transform(X_tr)

    base = GradientBoostingClassifier(random_state=0)
    clf_full = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf_full.fit(X_tr_s, y_tr.astype(int))

    # Prediker sannsynlighet for HELE datasettet (inkl. rader med nøytral y)
    X_all_s = scaler_full.transform(X)
    proba_full = clf_full.predict_proba(X_all_s)[:, 1]
    proba_full = pd.Series(proba_full, index=full_index)

    acc = float(np.nanmean(val_accs)) if len(val_accs) else np.nan
    auc = float(np.nanmean(val_aucs)) if len(val_aucs) else np.nan
    opt_thr = float(np.nanmean(thrs)) if len(thrs) else 0.5

    return proba_full, acc, auc, opt_thr

def _empty_result(idx=None):
    return {
        "proba": pd.Series(np.nan, index=(idx if idx is not None else []), dtype="float64"),
        "acc": np.nan, "auc": np.nan, "opt_thr": np.nan, "last_date": None
    }

def analyze_ticker_multi(df_raw: pd.DataFrame, eps_pct: float) -> dict:
    """Bygg indikatorer én gang og tren tre modeller. Returnér NaN når noe mangler."""
    out = {}

    if df_raw is None or df_raw.empty or "Close" not in df_raw:
        for key in ["1d","3d","5d"]:
            out[key] = _empty_result()
        return out

    df = add_indicators(df_raw)

    feat_base = [c for c in FEATURES_ALL if c in df.columns and df[c].notna().any()]
    if len(feat_base) == 0:
        for key in ["1d","3d","5d"]:
            out[key] = _empty_result()
        return out

    for H, key in [(1, "1d"), (3, "3d"), (5, "5d")]:
        y = make_label(df, H, eps_pct)
        if y is None or len(y) != len(df):
            out[key] = _empty_result()
            continue

        available = [c for c in feat_base if c in df.columns]
        if not available:
            out[key] = _empty_result()
            continue

        pack = pd.concat([df[available], y], axis=1)

        cols_in_pack = [c for c in available if c in pack.columns]
        if not cols_in_pack or y.name not in pack.columns:
            out[key] = _empty_result()
            continue

        pack = pack.dropna()
        if pack.empty or len(pack) < 60:
            out[key] = _empty_result(pack.index if len(pack) else None)
            if len(pack):
                out[key]["last_date"] = pack.index[-1]
            continue

        X  = pack.loc[:, cols_in_pack]
        yv = pack[y.name]

        if len(np.unique(yv.values.astype(int))) < 2:
            out[key] = _empty_result(pack.index)
            out[key]["last_date"] = pack.index[-1]
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

        if df_raw is None or df_raw.empty or "Close" not in df_raw:
            results.append({
                "Ticker": t,
                "Prob_1d": np.nan, "Rec_1d": "HOLD", "Date_1d": "—",
                "Prob_3d": np.nan, "Rec_3d": "HOLD", "Date_3d": "—",
                "Prob_5d": np.nan, "Rec_5d": "HOLD", "Date_5d": "—",
                "Delta_5d_1d": np.nan,
                "Acc": np.nan, "AUC": np.nan, "Composite": np.nan
            })
            progress.progress(i / len(tickers))
            continue

        pack = analyze_ticker_multi(df_raw, eps_pct=eps)

        # henter siste gyldige proba – default = NaN
        def last_proba(key, default=np.nan):
            try:
                s = pack[key]["proba"]
                if len(s) == 0:
                    return default
                v = float(s.iloc[-1])
                return v if not np.isnan(v) else default
            except Exception:
                return default

        p1 = last_proba("1d"); p3 = last_proba("3d"); p5 = last_proba("5d")

        date1 = expected_date(pack["1d"]["last_date"], 1)
        date3 = expected_date(pack["3d"]["last_date"], 3)
        date5 = expected_date(pack["5d"]["last_date"], 5)

        b1, s1 = thr["1d"]; b3, s3 = thr["3d"]; b5, s5 = thr["5d"]
        r1 = rec_from_prob(p1, max(b1, pack["1d"]["opt_thr"]) if not np.isnan(pack["1d"]["opt_thr"]) else b1, s1)
        r3 = rec_from_prob(p3, max(b3, pack["3d"]["opt_thr"]) if not np.isnan(pack["3d"]["opt_thr"]) else b3, s3)
        r5 = rec_from_prob(p5, max(b5, pack["5d"]["opt_thr"]) if not np.isnan(pack["5d"]["opt_thr"]) else b5, s5)

        probs = [x for x in [p1,p3,p5] if not np.isnan(x)]
        comp = float(np.mean(probs)) if probs else np.nan

        accs = [pack[k]["acc"] for k in ["1d","3d","5d"] if not np.isnan(pack[k]["acc"])]
        aucs = [pack[k]["auc"] for k in ["1d","3d","5d"] if not np.isnan(pack[k]["auc"])]
        acc = float(np.mean(accs)) if accs else np.nan
        auc = float(np.mean(aucs)) if aucs else np.nan

        results.append({
            "Ticker": t,
            "Prob_1d": p1, "Rec_1d": r1, "Date_1d": date1,
            "Prob_3d": p3, "Rec_3d": r3, "Date_3d": date3,
            "Prob_5d": p5, "Rec_5d": r5, "Date_5d": date5,
            "Delta_5d_1d": (p5 - p1) if (not np.isnan(p5) and not np.isnan(p1)) else np.nan,
            "Acc": acc, "AUC": auc, "Composite": comp
        })

        progress.progress(i / len(tickers))

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

    # Sørg for numeriske kolonner
    num_cols = ["Prob_1d","Prob_3d","Prob_5d","Acc","AUC","Delta_5d_1d","Composite"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("🟩 1 dag frem")
        df1 = df[["Ticker","Prob_1d","Rec_1d","Date_1d","Acc"]].copy().sort_values("Prob_1d", ascending=False)
        st.dataframe(style_df(df1, {"Prob_1d":"{:.2%}","Acc":"{:.2%}"}), use_container_width=True)

    with c2:
        st.subheader("🟦 3 dager frem")
        df3 = df[["Ticker","Prob_3d","Rec_3d","Date_3d","Acc"]].copy().sort_values("Prob_3d", ascending=False)
        st.dataframe(style_df(df3, {"Prob_3d":"{:.2%}","Acc":"{:.2%}"}), use_container_width=True)

    with c3:
        st.subheader("🟧 5 dager frem")
        df5 = df[["Ticker","Prob_5d","Rec_5d","Date_5d","Acc","Delta_5d_1d"]].copy().sort_values("Prob_5d", ascending=False)
        st.dataframe(style_df(df5, {"Prob_5d":"{:.2%}","Acc":"{:.2%}","Delta_5d_1d":"{:.2%}"}), use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Sammenligningstabell (alle horisonter)")
    cmp_df = df[[
        "Ticker",
        "Prob_1d","Rec_1d","Date_1d",
        "Prob_3d","Rec_3d","Date_3d",
        "Prob_5d","Rec_5d","Date_5d",
        "Delta_5d_1d","Acc","AUC","Composite"
    ]].sort_values("Composite", ascending=False)
    st.dataframe(
        style_df(cmp_df, {
            "Prob_1d":"{:.2%}","Prob_3d":"{:.2%}","Prob_5d":"{:.2%}",
            "Delta_5d_1d":"{:.2%}","Acc":"{:.2%}","AUC":"{:.3f}","Composite":"{:.2%}"
        }),
        use_container_width=True
    )

    # -------------------------
    # Detalj: graf per ticker
    # -------------------------
    st.markdown("---")
    st.subheader("📊 Detaljvisning (pris + sannsynlighet 1d/3d/5d)")
    sel_list = df["Ticker"].tolist() if not df.empty else []
    sel = st.selectbox("Velg ticker", sel_list)
    if sel:
        try:
            raw = fetch_history(sel, start=start_date, end=end_date)
            pack = analyze_ticker_multi(raw, eps_pct=eps)
            plot_df = pd.DataFrame({
                "Close": raw["Close"],
                "Prob_1d": pack["1d"]["proba"].reindex(raw.index),
                "Prob_3d": pack["3d"]["proba"].reindex(raw.index),
                "Prob_5d": pack["5d"]["proba"].reindex(raw.index),
            }).dropna(subset=["Close"])

            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(plot_df.index, plot_df["Close"])
            ax1.set_xlabel("Dato")
            ax1.set_ylabel("Pris")

            ax2 = ax1.twinx()
            if "Prob_1d" in plot_df: ax2.plot(plot_df.index, plot_df["Prob_1d"], alpha=0.9)
            if "Prob_3d" in plot_df: ax2.plot(plot_df.index, plot_df["Prob_3d"], alpha=0.9)
            if "Prob_5d" in plot_df: ax2.plot(plot_df.index, plot_df["Prob_5d"], alpha=0.9)
            ax2.axhline(0.5, linestyle="--", alpha=0.6)
            ax2.set_ylabel("Sannsynlighet")

            plt.title(f"{sel}: Pris + sannsynlighet (1d/3d/5d)")
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
                df1.to_excel(writer, index=False, sheet_name="1d")
                df3.to_excel(writer, index=False, sheet_name="3d")
                df5.to_excel(writer, index=False, sheet_name="5d")
                cmp_df.to_excel(writer, index=False, sheet_name="Comparison")
            buf.seek(0)
            xls = buf.getvalue()
            st.download_button(
                "⬇️ Last ned Excel (1d/3d/5d/Comparison)",
                data=xls,
                file_name=f"scan_v6_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.info(f"Excel-eksport feilet: {e}")

else:
    st.info("Velg/skriv tickere i sidepanelet og trykk **🔎 Skann og sammenlign** for å starte.")
































