# Khadala — Crypto Quantitative Terminal
# Built by Daniyal Aziz
# Live crypto data from Yahoo Finance. Covers BTC, ETH, SOL and 40+ other
# tokens. Includes a full quant stack: GARCH, Heston, Black-Scholes, PCA,
# SVD, Markowitz, CAPM, ML forecasting, LSTM, Q-learning, Kalman filter,
# backtesting, XVA, SIMM, ADF/VAR/VECM, and more.
# Run:  streamlit run khadala_crypto.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import yfinance as yf
    YF_OK = True
except ImportError:
    YF_OK = False

# PAGE CONFIG
st.set_page_config(
    page_title="Khadala — Crypto Quantitative Terminal",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Khadala Crypto v2.0 — Created by Daniyal Aziz"},
)

# CSS — dark trading terminal theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
code, pre { font-family: 'JetBrains Mono', monospace !important; }

.stApp { background: #070C14; }
[data-testid="stSidebar"] {
    background: linear-gradient(170deg, #0D1829 0%, #070C14 100%);
    border-right: 1px solid #1A2E48;
}

h1 { font-weight:800; color:#00D4FF !important; letter-spacing:-1px; }
h2 { font-weight:700; color:#E2E8F0 !important; }
h3 { color:#94A3B8 !important; }

[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0D2040 0%, #091828 100%);
    border: 1px solid #1C3A5E; border-radius: 10px;
    padding: 14px; box-shadow: 0 4px 20px rgba(0,212,255,0.06);
}
[data-testid="metric-container"] label {
    color:#4A6080 !important; font-size:0.68rem !important;
    text-transform:uppercase; letter-spacing:1.5px; font-family:'JetBrains Mono';
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:#00D4FF !important; font-family:'JetBrains Mono'; font-size:1.5rem;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family:'JetBrains Mono';
}

.pcard {
    background: linear-gradient(135deg,#0D2040,#091828);
    border:1px solid #1C3A5E; border-radius:10px;
    padding:12px 16px; font-family:'JetBrains Mono';
}
.pc-lbl { color:#4A6080; font-size:.65rem; text-transform:uppercase;
          letter-spacing:1px; margin-bottom:4px; }
.pc-px  { color:#E2E8F0; font-size:1.3rem; font-weight:700; }
.pc-up  { color:#10B981; font-size:.82rem; font-weight:600; }
.pc-dn  { color:#EF4444; font-size:.82rem; font-weight:600; }

.fbox {
    background:#091828; border:1px solid #1C3A5E;
    border-left:4px solid #00D4FF; border-radius:6px;
    padding:12px 16px; margin:8px 0;
    font-family:'JetBrains Mono'; font-size:.81rem; color:#94A3B8;
}
.fbox strong { color:#00D4FF; }

.sec-hdr {
    font-size:1.35rem; font-weight:700; color:#E2E8F0;
    border-bottom:2px solid #1C3A5E; padding-bottom:6px;
    margin:1.2rem 0 .8rem 0;
}

.stTabs [data-baseweb="tab-list"] {
    background:#0D1829; border-radius:8px; padding:4px; gap:4px;
}
.stTabs [data-baseweb="tab"] {
    color:#4A6080 !important; font-family:'JetBrains Mono';
    font-size:.77rem; border-radius:6px; padding:8px 14px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,rgba(0,212,255,.13),rgba(0,100,255,.13)) !important;
    color:#00D4FF !important;
    border:1px solid rgba(0,212,255,.28) !important;
}

.sb-sec {
    font-family:'JetBrains Mono'; font-size:.6rem; color:#00D4FF;
    text-transform:uppercase; letter-spacing:2px;
    margin:14px 0 5px; border-bottom:1px solid #1C3A5E; padding-bottom:4px;
}

hr { border-color:#1A2E48; }
.stAlert { background:#0D2040; border:1px solid #1C3A5E; border-radius:8px; }

/* Bloomberg-style news item links */
a { color:#94A3B8 !important; text-decoration:none; }
a:hover { color:#00D4FF !important; }

/* Critical card alert glow */
.crit-card { animation: pulse-border 2s ease-in-out infinite; }
@keyframes pulse-border {
    0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
    50% { box-shadow: 0 0 8px 2px rgba(239,68,68,0.25); }
}

/* Wider sidebar for news */
[data-testid="stSidebar"] { min-width: 320px !important; max-width: 360px !important; }
[data-testid="stSidebar"] > div:first-child { padding: 1rem 0.75rem; }
</style>
""", unsafe_allow_html=True)

# CRYPTO REGISTRY  — Major Cryptocurrencies (Yahoo Finance tickers use -USD suffix)
STOCKS = {
    # Layer-1 / Proof-of-Work
    "BTC-USD":  {"name": "Bitcoin",              "sector": "Layer-1 PoW"},
    "ETH-USD":  {"name": "Ethereum",             "sector": "Layer-1 PoS"},
    "LTC-USD":  {"name": "Litecoin",             "sector": "Layer-1 PoW"},
    "BCH-USD":  {"name": "Bitcoin Cash",         "sector": "Layer-1 PoW"},
    "ETC-USD":  {"name": "Ethereum Classic",     "sector": "Layer-1 PoW"},
    # Layer-1 / Proof-of-Stake & Smart Contract
    "SOL-USD":  {"name": "Solana",               "sector": "Layer-1 PoS"},
    "BNB-USD":  {"name": "BNB (Binance)",        "sector": "Layer-1 PoS"},
    "ADA-USD":  {"name": "Cardano",              "sector": "Layer-1 PoS"},
    "AVAX-USD": {"name": "Avalanche",            "sector": "Layer-1 PoS"},
    "DOT-USD":  {"name": "Polkadot",             "sector": "Layer-1 PoS"},
    "ATOM-USD": {"name": "Cosmos",               "sector": "Layer-1 PoS"},
    "NEAR-USD": {"name": "NEAR Protocol",        "sector": "Layer-1 PoS"},
    "FTM-USD":  {"name": "Fantom",               "sector": "Layer-1 PoS"},
    "ALGO-USD": {"name": "Algorand",             "sector": "Layer-1 PoS"},
    "ONE-USD":  {"name": "Harmony",              "sector": "Layer-1 PoS"},
    # Layer-2 & Scaling
    "MATIC-USD":{"name": "Polygon (MATIC)",      "sector": "Layer-2"},
    "ARB-USD":  {"name": "Arbitrum",             "sector": "Layer-2"},
    "OP-USD":   {"name": "Optimism",             "sector": "Layer-2"},
    "IMX-USD":  {"name": "Immutable X",          "sector": "Layer-2"},
    # DeFi
    "UNI-USD":  {"name": "Uniswap",              "sector": "DeFi"},
    "AAVE-USD": {"name": "Aave",                 "sector": "DeFi"},
    "MKR-USD":  {"name": "MakerDAO",             "sector": "DeFi"},
    "CRV-USD":  {"name": "Curve DAO",            "sector": "DeFi"},
    "COMP-USD": {"name": "Compound",             "sector": "DeFi"},
    "SNX-USD":  {"name": "Synthetix",            "sector": "DeFi"},
    "SUSHI-USD":{"name": "SushiSwap",            "sector": "DeFi"},
    "YFI-USD":  {"name": "Yearn Finance",        "sector": "DeFi"},
    # Cross-chain / Interoperability
    "LINK-USD": {"name": "Chainlink",            "sector": "Infrastructure"},
    "GRT-USD":  {"name": "The Graph",            "sector": "Infrastructure"},
    "FIL-USD":  {"name": "Filecoin",             "sector": "Infrastructure"},
    "ICP-USD":  {"name": "Internet Computer",    "sector": "Infrastructure"},
    "VET-USD":  {"name": "VeChain",              "sector": "Infrastructure"},
    # Payments / Transfers
    "XRP-USD":  {"name": "XRP (Ripple)",         "sector": "Payments"},
    "XLM-USD":  {"name": "Stellar",             "sector": "Payments"},
    "DOGE-USD": {"name": "Dogecoin",             "sector": "Payments"},
    "SHIB-USD": {"name": "Shiba Inu",            "sector": "Payments"},
    "TRX-USD":  {"name": "TRON",                 "sector": "Payments"},
    # Gaming / Metaverse / NFT
    "MANA-USD": {"name": "Decentraland",         "sector": "Metaverse/NFT"},
    "SAND-USD": {"name": "The Sandbox",          "sector": "Metaverse/NFT"},
    "AXS-USD":  {"name": "Axie Infinity",        "sector": "Metaverse/NFT"},
    "ENJ-USD":  {"name": "Enjin Coin",           "sector": "Metaverse/NFT"},
    "CHZ-USD":  {"name": "Chiliz",               "sector": "Metaverse/NFT"},
    # Exchange Tokens
    "CRO-USD":  {"name": "Cronos (Crypto.com)",  "sector": "Exchange Token"},
    "HT-USD":   {"name": "Huobi Token",          "sector": "Exchange Token"},
    "KCS-USD":  {"name": "KuCoin Token",         "sector": "Exchange Token"},
    # Privacy
    "XMR-USD":  {"name": "Monero",               "sector": "Privacy"},
    "ZEC-USD":  {"name": "Zcash",                "sector": "Privacy"},
    "DASH-USD": {"name": "Dash",                 "sector": "Privacy"},
}

SECTORS = {k: v["sector"] for k, v in STOCKS.items()}  # crypto sectors
PALETTE = [
    "#00D4FF","#10B981","#F59E0B","#8B5CF6","#F97316",
    "#EF4444","#06B6D4","#84CC16","#EC4899","#6366F1",
]

# DATA LOADING — fetch live Crypto OHLCV data from Yahoo Finance
@st.cache_data(show_spinner=False, ttl=7200)
def load_all_data(tickers: tuple) -> dict:
    """Download 2 years of daily OHLCV for each ticker via yfinance.
    Falls back to per-ticker downloads if the batch call fails.
    """
    stock_data: dict = {}

    if not YF_OK:
        st.error("yfinance is not installed. Run: pip install yfinance")
        return stock_data

    try:
        raw_batch = yf.download(
            list(tickers),
            period="2y",
            interval="1d",
            auto_adjust=True,   # crypto: no splits; adjusts for data gaps
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception as e:
        st.warning(f"Batch download failed, falling back to individual downloads: {e}")
        raw_batch = None

    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]

    if raw_batch is not None and not raw_batch.empty and isinstance(raw_batch.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                df = raw_batch[ticker].copy()
                df.index = pd.to_datetime(df.index)
                df.index.name = "Date"
                available_cols = [c for c in ohlcv_cols if c in df.columns]
                if "Close" not in available_cols or len(df) < 50:
                    continue
                stock_data[ticker] = (
                    df[available_cols]
                    .apply(pd.to_numeric, errors="coerce")
                    .dropna(subset=["Close"])
                )
            except Exception:
                continue
        return stock_data

    for ticker in tickers:
        try:
            raw = yf.download(
                ticker,
                period="2y",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if raw.empty or len(raw) < 50:
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)

            raw.index = pd.to_datetime(raw.index)
            raw.index.name = "Date"

            available_cols = [c for c in ohlcv_cols if c in raw.columns]
            if "Close" not in available_cols:
                continue

            stock_data[ticker] = (
                raw[available_cols]
                .apply(pd.to_numeric, errors="coerce")
                .dropna(subset=["Close"])
            )
        except Exception as e:
            st.warning(f"Could not load {ticker}: {e}")

    return stock_data


@st.cache_data(show_spinner=False, ttl=3600)
def build_matrices(stock_data: dict, tickers: tuple) -> tuple:
    """Align prices and compute log returns for all selected tickers."""
    close_prices = pd.DataFrame({t: stock_data[t]["Close"] for t in tickers}).dropna()
    log_returns  = np.log(close_prices / close_prices.shift(1)).dropna()
    return close_prices, log_returns


# CRYPTO MARKET INTELLIGENCE — live quotes, news, on-chain sentiment proxy

# Core crypto watchlist for the dashboard (always fetched for market overview)
SP500_WATCHLIST = [
    "BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","ADA-USD","DOGE-USD",
    "AVAX-USD","LINK-USD","DOT-USD","MATIC-USD","UNI-USD","AAVE-USD","MKR-USD",
    "LTC-USD","BCH-USD","ATOM-USD","NEAR-USD","ARB-USD","OP-USD",
    "FIL-USD","GRT-USD","CRV-USD","SNX-USD","MANA-USD","SAND-USD","AXS-USD",
    "XLM-USD","TRX-USD","SHIB-USD","XMR-USD","VET-USD","ICP-USD","FTM-USD",
]

@st.cache_data(show_spinner=False, ttl=300)
def fetch_live_quotes(tickers: list) -> pd.DataFrame:
    """Fetch recent quotes for all tickers."""
    if not YF_OK:
        return pd.DataFrame()

    quote_rows = []
    try:
        raw = yf.download(
            list(tickers),
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        raw = None

    if raw is None or raw.empty:
        return pd.DataFrame()

    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                hist = raw[ticker].dropna(how="all")
            else:
                hist = raw.dropna(how="all")

            if hist.empty or len(hist) < 2:
                continue

            latest_price   = float(hist["Close"].iloc[-1])
            prev_close     = float(hist["Close"].iloc[-2])
            day_change_pct = (latest_price - prev_close) / prev_close * 100
            todays_volume  = float(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else 0
            avg_volume     = float(hist["Volume"].mean())   if "Volume" in hist.columns else 1
            five_day_high  = float(hist["High"].max())
            five_day_low   = float(hist["Low"].min())
            range_pct      = (five_day_high - five_day_low) / max(five_day_low, 1) * 100

            quote_rows.append({
                "ticker":    ticker,
                "name":      STOCKS.get(ticker, {}).get("name",   ticker),
                "sector":    STOCKS.get(ticker, {}).get("sector", "—"),
                "price":     latest_price,
                "chg_pct":   day_change_pct,
                "volume":    todays_volume,
                "avg_vol":   avg_volume,
                "vol_ratio": todays_volume / max(avg_volume, 1),
                "range5":    range_pct,
            })
        except Exception:
            continue

    return pd.DataFrame(quote_rows)


@st.cache_data(show_spinner=False, ttl=900)
def fetch_market_news(max_items: int = 20) -> list:
    """Fetch recent crypto headlines via yfinance."""
    news_items = []
    if not YF_OK:
        return []

    try:
        for symbol in ["BTC-USD", "ETH-USD", "SOL-USD"]:
            try:
                raw_news = yf.Ticker(symbol).news or []
                for item in raw_news[:max_items]:
                    content = item.get("content", {}) if isinstance(item.get("content"), dict) else {}

                    title     = item.get("title")     or content.get("title", "")
                    publisher = item.get("publisher") or content.get("provider", {}).get("displayName", "")
                    url       = item.get("link")      or content.get("canonicalUrl", {}).get("url", "")
                    timestamp = item.get("providerPublishTime", 0) or content.get("pubDate", "")

                    if title:
                        news_items.append({
                            "title":  title,
                            "source": publisher,
                            "url":    url,
                            "time":   timestamp,
                        })

                if len(news_items) >= max_items:
                    break
            except Exception:
                continue
    except Exception:
        pass

    seen_titles = set()
    unique_news = []
    for article in news_items:
        if article["title"] not in seen_titles:
            seen_titles.add(article["title"])
            unique_news.append(article)

    return unique_news[:max_items]


@st.cache_data(show_spinner=False, ttl=900)
def fetch_index_data() -> dict:
    """Fetch BTC, ETH, SOL, BNB and XRP prices for the top bar."""
    index_levels = {}
    if not YF_OK:
        return index_levels

    index_symbols = {
        "Bitcoin":  "BTC-USD",
        "Ethereum": "ETH-USD",
        "Solana":   "SOL-USD",
        "BNB":      "BNB-USD",
        "XRP":      "XRP-USD",
    }

    try:
        raw = yf.download(
            list(index_symbols.values()),
            period="5d", interval="1d",
            auto_adjust=True, progress=False,
            group_by="ticker", threads=True,
        )
    except Exception:
        return index_levels

    for label, symbol in index_symbols.items():
        try:
            hist = raw[symbol].dropna(how="all") if isinstance(raw.columns, pd.MultiIndex) else raw.dropna(how="all")
            if hist.empty or len(hist) < 2:
                continue
            cur  = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2])
            index_levels[label] = {"price": cur, "chg_pct": (cur - prev) / prev * 100}
        except Exception:
            continue

    return index_levels


@st.cache_data(show_spinner=False, ttl=300)
def fetch_intraday(ticker: str, interval: str = "5m") -> pd.DataFrame:
    """Fetch intraday bars for a single ticker. Valid intervals: 1m, 5m, 15m, 30m, 1h."""
    if not YF_OK:
        return pd.DataFrame()
    try:
        raw = yf.download(
            ticker,
            period="2d",   # 2d is enough for intraday chart; much faster than 5d
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        raw.index = pd.to_datetime(raw.index)
        return raw.dropna(subset=["Close"]) if "Close" in raw.columns else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# PLOTLY THEME HELPERS
_LAY = dict(
    paper_bgcolor="#070C14", plot_bgcolor="#0D1829",
    font=dict(family="JetBrains Mono", color="#94A3B8", size=11),
    legend=dict(bgcolor="#0D2040", bordercolor="#1C3A5E", borderwidth=1),
    margin=dict(l=0, r=0, t=44, b=0),
)
_AX = dict(gridcolor="#182A40", zeroline=False, showgrid=True)

def lay(**kw): return {**_LAY, **kw}
def ax(**kw):  return {**_AX,  **kw}


# SIDEBAR — Bloomberg-style: controls + live crypto news feed
with st.sidebar:
    st.markdown("## 🪙 Khadala Crypto")
    st.markdown("**Crypto Quantitative Terminal**")
    st.markdown("*24/7 Crypto Market · Live via Yahoo Finance*")
    st.divider()

    st.markdown('<div class="sb-sec">Crypto Selection</div>', unsafe_allow_html=True)

    if "basket" not in st.session_state:
        st.session_state.basket = ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","ADA-USD"]

    # sector browser
    all_sectors = sorted(set(v["sector"] for v in STOCKS.values()))
    sel_sector  = st.selectbox("Browse category", ["— pick category —"] + all_sectors, key="sb_sector")

    if sel_sector != "— pick category —":
        sector_tickers = [k for k, v in STOCKS.items() if v["sector"] == sel_sector]
        st.caption(f"{len(sector_tickers)} assets in {sel_sector}")
        btn_cols = st.columns(3)
        for ci, tk in enumerate(sector_tickers):
            in_basket = tk in st.session_state.basket
            label     = f"✓ {tk}" if in_basket else f"+ {tk}"
            if btn_cols[ci % 3].button(label, key=f"add_{tk}", use_container_width=True):
                if in_basket:
                    if len(st.session_state.basket) > 2:
                        st.session_state.basket.remove(tk)
                else:
                    st.session_state.basket.append(tk)
                st.rerun()

    selected = st.multiselect(
        "Active portfolio (search any crypto)",
        options=list(STOCKS.keys()),
        default=[t for t in st.session_state.basket if t in STOCKS],
        format_func=lambda x: f"{x}  ·  {STOCKS[x]['name']}  [{STOCKS[x]['sector']}]",  # crypto
        key="sb_multiselect",
    )
    if selected:
        st.session_state.basket = selected

    if len(selected) < 2:
        st.info("Select ≥ 2 stocks above.")
        selected = list(st.session_state.basket) if len(st.session_state.basket) >= 2 else ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","ADA-USD"]

    if st.button("↺ Reset to defaults", key="reset_basket"):
        st.session_state.basket = ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","ADA-USD"]
        st.rerun()

    st.caption(f"Portfolio: {len(selected)} assets  ·  {', '.join(selected[:6])}{'...' if len(selected)>6 else ''}")

    st.markdown('<div class="sb-sec">Parameters</div>', unsafe_allow_html=True)
    rf_rate    = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 4.5, 0.25) / 100
    n_pca_show = st.slider("PCA Components", 2, min(8, len(selected)), min(4, len(selected)))
    roll_win   = st.slider("Rolling Window (days)", 20, 120, 60, 5)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()
    with col_r2:
        intraday_on = st.toggle("Intraday", value=False)

    st.divider()
    st.markdown('<div class="sb-sec">Data Status</div>', unsafe_allow_html=True)
    with st.spinner("Fetching data…"):
        all_dfs = load_all_data(tuple(selected))

    N      = len(selected)
    loaded = [t for t in selected if t in all_dfs]

    loaded_sel = [t for t in selected if t in all_dfs]
    if len(loaded_sel) >= 2:
        closes, R_df = build_matrices(all_dfs, tuple(loaded_sel))
    else:
        # Can't build — show error and use empty structures
        st.error("Need ≥ 2 loaded crypto assets. Check internet connection.")
        closes = pd.DataFrame()
        R_df   = pd.DataFrame()

    T_days = len(R_df)

    if T_days > 1:
        R       = R_df.values
        dates   = R_df.index
        mu_d    = R.mean(axis=0)
        mu_ann  = mu_d * 252
        Rc      = R - mu_d
        Sigma   = (Rc.T @ Rc) / max(T_days - 1, 1)
        Sig_ann = Sigma * 252
        ind_vol = np.sqrt(np.diag(Sig_ann)) * 100
        w_ew    = np.ones(len(loaded_sel)) / len(loaded_sel)
        p_vol   = float(np.sqrt(w_ew @ Sig_ann @ w_ew)) * 100
        p_ret   = float(w_ew @ mu_ann) * 100
        p_sh    = (p_ret - rf_rate * 100) / max(p_vol, 1e-8)
    else:
        R = np.array([[]])
        dates = pd.DatetimeIndex([])
        mu_d = mu_ann = ind_vol = np.zeros(max(len(loaded_sel),1))
        Rc = Sigma = Sig_ann = np.zeros((max(len(loaded_sel),1),max(len(loaded_sel),1)))
        w_ew = np.ones(max(len(loaded_sel),1)) / max(len(loaded_sel),1)
        p_vol = p_ret = p_sh = 0.0

    N = len(loaded_sel) if loaded_sel else len(selected)

    if loaded:
        st.success(f"✓  {len(loaded)}/{len(selected)} assets  ·  {T_days:,}d")
    else:
        st.error("No data. Check internet.")
    if len(loaded) < len(selected):
        missing = [t for t in selected if t not in all_dfs]
        st.warning(f"Missing: {', '.join(missing)}")
    if T_days > 0 and len(dates) > 0:
        st.caption(f"{dates[0].date()} → {dates[-1].date()}")

    st.divider()
    st.markdown('<div class="sb-sec">Crypto News</div>', unsafe_allow_html=True)

    news_items = fetch_market_news(max_items=25)
    if news_items:
        _BULL_KW = frozenset(["surge","rally","soar","beat","rise","gain","record","up","bull","moon","ath","adoption","approve"])
        _BEAR_KW = frozenset(["fall","drop","crash","miss","decline","cut","loss","down","fear","risk","ban","hack","exploit","liquidat","bear","dump","rug"])
        _news_html_parts = []  # collect all HTML, render in ONE call
        for item in news_items:
            title  = item.get("title","")
            source = item.get("source","")
            url    = item.get("url","")
            words = set(title.lower().split())
            if words & _BEAR_KW:
                dot = "🔴"
            elif words & _BULL_KW:
                dot = "🟢"
            else:
                dot = "🔵"

            if url:
                _news_html_parts.append(
                    f'{dot} <a href="{url}" target="_blank" style="color:#94A3B8;font-size:.72rem;'
                    f'text-decoration:none;line-height:1.4;">{title}</a>'
                    f'<br><span style="color:#344054;font-size:.62rem;">{source}</span>'
                    f'<hr style="margin:4px 0;border-color:#0D1829;">'
                )
            else:
                _news_html_parts.append(
                    f'{dot} <span style="color:#94A3B8;font-size:.72rem;">{title}</span>'
                    f'<br><span style="color:#344054;font-size:.62rem;">{source}</span>'
                    f'<hr style="margin:4px 0;border-color:#0D1829;">'
                )
        st.markdown("".join(_news_html_parts), unsafe_allow_html=True)
    else:
        st.caption("News unavailable. Check connection.")


# GLOBAL HEADER — Crypto ticker tape + live price bar
st.markdown("# 🪙  Khadala — Crypto Quantitative Terminal")

# Index bar: BTC, ETH, SOL, BNB, XRP (live crypto prices)
idx_data = fetch_index_data()
if idx_data:
    idx_cols = st.columns(len(idx_data))
    for col, (label, data) in zip(idx_cols, idx_data.items()):
        arrow = "▲" if data["chg_pct"] >= 0 else "▼"
        color = "#10B981" if data["chg_pct"] >= 0 else "#EF4444"
        col.markdown(
            f'<div class="pcard" style="text-align:center;padding:8px 6px;">'
            f'<div class="pc-lbl">{label}</div>'
            f'<div class="pc-px" style="font-size:1rem;">{data["price"]:,.2f}</div>'
            f'<div style="color:{color};font-size:.75rem;font-weight:600;">'
            f'{arrow} {data["chg_pct"]:+.2f}%</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("")

# Portfolio metrics bar
mc1,mc2,mc3,mc4,mc5 = st.columns(5)
mc1.metric("Assets",           len(loaded_sel) if loaded_sel else N)
mc2.metric("Days (24/7)",      f"{T_days:,}")
mc3.metric("EW Annual Return", f"{p_ret:.2f}%")
mc4.metric("EW Volatility",    f"{p_vol:.2f}%")
mc5.metric("EW Sharpe",        f"{p_sh:.3f}")

# Price strip — selected tickers (guard against empty closes)
if not closes.empty and T_days > 1:
    strip_tickers = [t for t in selected[:6] if t in closes.columns]
    strip = st.columns(max(len(strip_tickers), 1))
    for i, t in enumerate(strip_tickers):
        try:
            px_now  = float(closes[t].iloc[-1])
            px_prev = float(closes[t].iloc[-2])
            chg     = (px_now - px_prev) / px_prev * 100
            ctag    = "pc-up" if chg >= 0 else "pc-dn"
            arrow   = "▲" if chg >= 0 else "▼"
            with strip[i]:
                st.markdown(f"""
                <div class="pcard">
                  <div class="pc-lbl">{t}  ·  {SECTORS.get(t,'—')}</div>
                  <div class="pc-px">{px_now:,.2f}</div>
                  <div class="{ctag}">{arrow} {chg:+.2f}%</div>
                </div>""", unsafe_allow_html=True)
        except Exception:
            pass

st.divider()

# TABS — tab0 is the Crypto Market Dashboard; all 18 tabs intact
(tab0,tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,
 tab10,tab11,tab12,tab13,tab14,tab15,tab16,tab17) = st.tabs([
    "🏦 Market Dashboard",
    "📈 Prices & Returns",
    "📊 Covariance & Risk",
    "🔢 PCA  ·  Eigenanalysis",
    "🔀 SVD  ·  Factor Models",
    "🎯 Portfolio Optimization",
    "🤖 ML Predictions",
    "🏛️ Asset Pricing & Microstructure",
    "📐 Regression Analysis",
    "📈 Yield Curve & Rates",
    "🔬 Advanced Regression & E-Trading",
    "📉 Advanced Volatility",
    "∂ Stochastic Calculus",
    "🛡️ Counterparty Risk",
    "🤖 Advanced ML",
    "📊 Time Series Econometrics",
    "⚡ Backtesting Engine",
    "🧠 ML Models Hub",
])

with tab0:
    st.markdown('<div class="sec-hdr">Crypto Market Dashboard</div>',
                unsafe_allow_html=True)

    with st.spinner("Loading live crypto data…"):
        live_df = fetch_live_quotes(SP500_WATCHLIST)

    if live_df.empty:
        st.error("Live crypto data unavailable. Check internet connection or yfinance installation.")
        st.stop()

    st.markdown("#### Crypto Market Overview  (BTC · ETH · SOL · BNB · XRP)")
    idx_d = fetch_index_data()
    if idx_d:
        idx_cols2 = st.columns(len(idx_d))
        for col, (label, data) in zip(idx_cols2, idx_d.items()):
            arrow = "▲" if data["chg_pct"] >= 0 else "▼"
            delta_str = f"{arrow} {data['chg_pct']:+.2f}%"
            col.metric(label, f"{data['price']:,.2f}", delta=delta_str)
    st.divider()

    col_left, col_mid, col_right = st.columns([1.2, 1.2, 1], gap="medium")

    with col_left:
        st.markdown("""
        <div class="fbox">
        <strong>Trending Cryptos</strong> — top absolute movers by % change today
        </div>""", unsafe_allow_html=True)

        if not live_df.empty:
            top_up   = live_df.nlargest(5,  "chg_pct")[["ticker","name","price","chg_pct"]]
            top_dn   = live_df.nsmallest(5, "chg_pct")[["ticker","name","price","chg_pct"]]

            up_html = "".join(
                f'<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #0D2040;">'
                f'<span style="font-family:JetBrains Mono;color:#00D4FF;font-size:.82rem;font-weight:700;">{r["ticker"]}</span>'
                f'<span style="font-size:.75rem;color:#94A3B8;">{r["name"][:20]}</span>'
                f'<span style="font-family:JetBrains Mono;color:#10B981;font-size:.82rem;font-weight:700;">▲ {r["chg_pct"]:+.2f}%</span></div>'
                for _, r in top_up.iterrows()
            )
            st.markdown(f"**Top Gainers**{up_html}", unsafe_allow_html=True)

            dn_html = "".join(
                f'<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #0D2040;">'
                f'<span style="font-family:JetBrains Mono;color:#00D4FF;font-size:.82rem;font-weight:700;">{r["ticker"]}</span>'
                f'<span style="font-size:.75rem;color:#94A3B8;">{r["name"][:20]}</span>'
                f'<span style="font-family:JetBrains Mono;color:#EF4444;font-size:.82rem;font-weight:700;">▼ {r["chg_pct"]:+.2f}%</span></div>'
                for _, r in top_dn.iterrows()
            )
            st.markdown(f"<br>**Top Losers**{dn_html}", unsafe_allow_html=True)

    with col_mid:
        st.markdown("""
        <div class="fbox">
        <strong>High-Volatility Cryptos</strong> — widest 5-day price range (% H-L/L)
        </div>""", unsafe_allow_html=True)

        if not live_df.empty:
            risky = live_df.nlargest(10, "range5")[["ticker","name","price","chg_pct","range5","vol_ratio"]]

            for _, row in risky.iterrows():
                risk_bar_pct = min(int(row["range5"] / 30 * 100), 100)
                arrow   = "▲" if row["chg_pct"] >= 0 else "▼"
                color   = "#10B981" if row["chg_pct"] >= 0 else "#EF4444"
                bar_col = "#EF4444" if row["range5"] > 10 else "#F59E0B"
                st.markdown(
                    f'<div style="margin-bottom:8px;">'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<span style="font-family:JetBrains Mono;color:#00D4FF;font-size:.8rem;font-weight:700;">{row["ticker"]}</span>'
                    f'<span style="font-size:.7rem;color:{color};font-weight:600;">{arrow}{row["chg_pct"]:+.1f}%</span>'
                    f'<span style="font-size:.7rem;color:#EF4444;font-weight:600;">±{row["range5"]:.1f}% 5d</span></div>'
                    f'<div style="background:#0D1829;border-radius:3px;height:4px;margin-top:3px;">'
                    f'<div style="width:{risk_bar_pct}%;background:{bar_col};height:4px;border-radius:3px;"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

    with col_right:
        st.markdown("""
        <div class="fbox">
        <strong>Volume Surge</strong> — vol/avg_vol ratio
        </div>""", unsafe_allow_html=True)

        if not live_df.empty:
            surging = live_df.nlargest(8, "vol_ratio")[["ticker","chg_pct","vol_ratio"]]
            surge_html = "".join(
                f'<div style="margin-bottom:8px;">'
                f'<div style="display:flex;justify-content:space-between;">'
                f'<span style="font-family:JetBrains Mono;color:#00D4FF;font-size:.8rem;font-weight:700;">{r["ticker"]}</span>'
                f'<span style="font-size:.7rem;color:{"#10B981" if r["chg_pct"]>=0 else "#EF4444"};">{"▲" if r["chg_pct"]>=0 else "▼"}{r["chg_pct"]:+.1f}%</span>'
                f'<span style="font-size:.7rem;color:#F59E0B;">{r["vol_ratio"]:.1f}× vol</span></div>'
                f'<div style="background:#0D1829;border-radius:3px;height:4px;margin-top:3px;">'
                f'<div style="width:{min(int(r["vol_ratio"]/5*100),100)}%;background:#F59E0B;height:4px;border-radius:3px;"></div>'
                f'</div></div>'
                for _, r in surging.iterrows()
            )
            st.markdown(surge_html, unsafe_allow_html=True)

    st.divider()

    st.markdown("#### Crypto Watchlist")
    if not live_df.empty:
        disp = live_df[["ticker","name","sector","price","chg_pct","vol_ratio","range5"]].copy()
        disp.columns = ["Ticker","Company","Sector","Price ($)","Change %","Vol Ratio","5d Range %"]
        disp["Price ($)"]    = disp["Price ($)"].round(2)
        disp["Change %"]     = disp["Change %"].round(2)
        disp["Vol Ratio"]    = disp["Vol Ratio"].round(2)
        disp["5d Range %"]   = disp["5d Range %"].round(2)
        disp = disp.sort_values("Change %", ascending=False).reset_index(drop=True)
        def _color_chg(v):
            try:
                f = float(v)
                if f > 2:    return "color: #10B981; font-weight:700"
                elif f > 0:  return "color: #6EE7B7"
                elif f > -2: return "color: #FCA5A5"
                else:        return "color: #EF4444; font-weight:700"
            except Exception:
                return ""
        def _color_risk(v):
            try:
                f = float(v)
                if f > 10:   return "color: #EF4444; font-weight:700"
                elif f > 5:  return "color: #F59E0B"
                else:        return "color: #6EE7B7"
            except Exception:
                return ""
        styled = disp.style.map(_color_chg, subset=["Change %"]).map(_color_risk, subset=["5d Range %"])
        st.dataframe(styled, use_container_width=True, height=420)

    st.divider()

    st.markdown("#### Intraday Tick Chart  ·  5-minute bars")
    tick_col1, tick_col2 = st.columns([1, 3])
    with tick_col1:
        tick_sym = st.selectbox("Symbol", options=selected + [t for t in SP500_WATCHLIST if t not in selected],
                                key="tick_sym0")
        tick_int = st.selectbox("Interval", ["1m","5m","15m","30m","1h"], index=1, key="tick_int0")
    with tick_col2:
        tick_df = fetch_intraday(tick_sym, tick_int)
        if not tick_df.empty and "Close" in tick_df.columns:
            fig_tick = make_subplots(rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.72, 0.28], vertical_spacing=0.03)
            # Candlestick
            if all(c in tick_df.columns for c in ["Open","High","Low","Close"]):
                fig_tick.add_trace(go.Candlestick(
                    x=tick_df.index,
                    open=tick_df["Open"], high=tick_df["High"],
                    low=tick_df["Low"],   close=tick_df["Close"],
                    name=tick_sym,
                    increasing=dict(line=dict(color="#10B981"), fillcolor="#10B981"),
                    decreasing=dict(line=dict(color="#EF4444"), fillcolor="#EF4444"),
                ), row=1, col=1)
            else:
                fig_tick.add_trace(go.Scatter(
                    x=tick_df.index, y=tick_df["Close"],
                    line=dict(color="#00D4FF", width=1.2), name="Close",
                ), row=1, col=1)
            # Volume
            if "Volume" in tick_df.columns:
                vol_colors = ["#10B981" if c >= o else "#EF4444"
                              for c, o in zip(tick_df["Close"], tick_df.get("Open", tick_df["Close"]))]
                fig_tick.add_trace(go.Bar(
                    x=tick_df.index, y=tick_df["Volume"],
                    marker_color=vol_colors, opacity=0.7, name="Volume",
                ), row=2, col=1)
            fig_tick.update_layout(
                title=f"{tick_sym}  ·  {tick_int} intraday bars",
                xaxis_rangeslider_visible=False, height=480, **lay(),
            )
            fig_tick.update_xaxes(**ax()); fig_tick.update_yaxes(**ax())
            st.plotly_chart(fig_tick, use_container_width=True)
        else:
            st.info(f"No intraday data for {tick_sym}. Market may be closed.")

    st.divider()

    st.markdown("#### Sector Performance Heatmap")
    if not live_df.empty:
        sec_perf = live_df.groupby("sector")["chg_pct"].mean().reset_index()
        sec_perf.columns = ["Sector","Avg Change %"]
        sec_perf = sec_perf.sort_values("Avg Change %", ascending=False)

        fig_sec = go.Figure(go.Bar(
            x=sec_perf["Sector"],
            y=sec_perf["Avg Change %"],
            marker=dict(
                color=sec_perf["Avg Change %"],
                colorscale=[[0,"#7F1D1D"],[0.5,"#1C3A5E"],[1,"#064E3B"]],
                cmid=0, showscale=False,
            ),
            text=[f"{v:+.2f}%" for v in sec_perf["Avg Change %"]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11, color="#94A3B8"),
        ))
        fig_sec.update_layout(
            title="Sector Performance (avg daily change)",
            height=320, **lay(),
        )
        fig_sec.update_xaxes(tickangle=-30, **ax())
        fig_sec.update_yaxes(title_text="Avg Change %",
                              **{**ax(), "zeroline": True, "zerolinecolor": "#344054"})
        st.plotly_chart(fig_sec, use_container_width=True)

    st.divider()

    st.markdown("#### Critical Stocks Spotlight")
    st.markdown("""
    <div class="fbox">
    Stocks flagged as <strong>critical</strong> based on three signals:
    extreme move (|chg| &gt; 3%), volume surge (&gt; 2× avg), and high 5-day volatility (&gt; 8%)
    </div>""", unsafe_allow_html=True)

    if not live_df.empty:
        critical_mask = (
            (live_df["chg_pct"].abs() > 3) |
            (live_df["vol_ratio"] > 2.0) |
            (live_df["range5"] > 8)
        )
        critical_df = live_df[critical_mask].copy()

        if critical_df.empty:
            st.info("No critical alerts today — market appears calm.")
        else:
            crit_cols = st.columns(min(len(critical_df), 4))
            for ci, (_, crow) in enumerate(critical_df.iterrows()):
                if ci >= 4:
                    break
                arrow = "▲" if crow["chg_pct"] >= 0 else "▼"
                color = "#10B981" if crow["chg_pct"] >= 0 else "#EF4444"
                alerts = []
                if abs(crow["chg_pct"]) > 3:
                    alerts.append("EXTREME MOVE")
                if crow["vol_ratio"] > 2:
                    alerts.append(f"VOL SURGE {crow['vol_ratio']:.1f}×")
                if crow["range5"] > 8:
                    alerts.append(f"HIGH RISK ±{crow['range5']:.1f}%")
                alert_html = " · ".join(
                    f'<span style="color:#F59E0B;font-size:.6rem;">{a}</span>'
                    for a in alerts
                )
                with crit_cols[ci]:
                    st.markdown(
                        f'<div class="pcard" style="border-color:{color};border-width:2px;">'
                        f'<div class="pc-lbl">{crow["ticker"]} · {crow["sector"]}</div>'
                        f'<div class="pc-px" style="color:{color};">'
                        f'{arrow} {crow["chg_pct"]:+.2f}%</div>'
                        f'<div style="color:#94A3B8;font-size:.72rem;margin-top:2px;">'
                        f'${crow["price"]:,.2f}</div>'
                        f'<div style="margin-top:4px;">{alert_html}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

with tab1:
    st.markdown('<div class="sec-hdr">Return Matrix</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="fbox">
    <strong>Log Returns:</strong> R(t,i) = ln[P(t,i) / P(t−1,i)]
    &nbsp;|&nbsp;
    <strong>Matrix:</strong> R is T×N &nbsp; T = trading days, N = assets
    </div>""", unsafe_allow_html=True)

    cl,cr = st.columns([1,3])
    with cl:
        cht = st.selectbox("Chart stock", selected, key="cht1")
        per = st.selectbox("Period", ["1Y","2Y","All"], index=0)
    df_c = all_dfs[cht].copy()
    if per == "1Y":   df_c = df_c.iloc[-252:]
    elif per == "2Y": df_c = df_c.iloc[-504:]

    fig_c = make_subplots(rows=2,cols=1,shared_xaxes=True,
                          vertical_spacing=0.03,row_heights=[0.72,0.28])
    fig_c.add_trace(go.Candlestick(
        x=df_c.index,
        open =df_c.get("Open",  df_c["Close"]),
        high =df_c.get("High",  df_c["Close"]*1.005),
        low  =df_c.get("Low",   df_c["Close"]*0.995),
        close=df_c["Close"], name=cht,
        increasing=dict(line=dict(color="#10B981",width=1), fillcolor="#10B981"),
        decreasing=dict(line=dict(color="#EF4444",width=1), fillcolor="#EF4444"),
    ), row=1,col=1)
    for win,col_ma,ds in [(20,"#F59E0B","dot"),(50,"#8B5CF6","dash")]:
        fig_c.add_trace(go.Scatter(
            x=df_c.index, y=df_c["Close"].rolling(win).mean(),
            name=f"MA{win}", line=dict(color=col_ma,width=1.2,dash=ds),
        ), row=1,col=1)
    if "Volume" in df_c.columns:
        _close_shift = df_c["Close"].shift(1)
        vc = np.where(df_c["Close"] >= _close_shift, "#10B981", "#EF4444").tolist()
        fig_c.add_trace(go.Bar(
            x=df_c.index,y=df_c["Volume"],
            marker_color=vc,marker_line_width=0,opacity=0.7,name="Volume",
        ), row=2,col=1)
    fig_c.update_layout(
        title=f"{cht}  ·  OHLCV  |  MA20  MA50",
        xaxis_rangeslider_visible=False,height=520,**lay(),
    )
    fig_c.update_xaxes(**ax()); fig_c.update_yaxes(**ax())
    st.plotly_chart(fig_c,use_container_width=True)

    st.markdown("#### Normalised Price Index (Base = 100)")
    norm = closes / closes.iloc[0] * 100
    fig_n = go.Figure()
    for i,t in enumerate(selected):
        fig_n.add_trace(go.Scatter(
            x=norm.index,y=norm[t],name=t,
            line=dict(color=PALETTE[i%len(PALETTE)],width=1.6),
            hovertemplate=f"<b>{t}</b>  %{{y:.1f}}<extra></extra>",
        ))
    fig_n.update_layout(title="Normalised Price Index",height=340,**lay())
    fig_n.update_xaxes(**ax()); fig_n.update_yaxes(title_text="Index",**ax())
    st.plotly_chart(fig_n,use_container_width=True)

    st.markdown("#### Daily Log-Return Distribution")
    fig_d = go.Figure()
    for i,t in enumerate(selected):
        fig_d.add_trace(go.Histogram(
            x=R_df[t]*100,name=t,opacity=0.55,nbinsx=80,
            marker_color=PALETTE[i%len(PALETTE)],
        ))
    fig_d.update_layout(barmode="overlay",height=300,**lay())
    fig_d.update_xaxes(title_text="Daily Return (%)",**ax())
    fig_d.update_yaxes(title_text="Count",**ax())
    st.plotly_chart(fig_d,use_container_width=True)

    stats = pd.DataFrame({
        "Ann Return %": (mu_ann*100).round(2),
        "Ann Vol %":    ind_vol.round(2),
        "Sharpe":       ((mu_ann*100 - rf_rate*100) / ind_vol).round(3),
        "Skew":         R_df.skew().values.round(3),
        "Kurtosis":     R_df.kurtosis().values.round(3),
        "VaR 95%":      (R_df.quantile(0.05)*100).values.round(3),
        "Sector":       [SECTORS.get(t,"—") for t in selected],
    }, index=selected)
    def _color_ret(v):
        try:
            f = float(v)
            if f > 20:   return "color: #10B981; font-weight:700"
            elif f > 0:  return "color: #6EE7B7"
            else:        return "color: #EF4444"
        except Exception:
            return ""
    def _color_vol(v):
        try:
            f = float(v)
            if f > 35:   return "color: #EF4444; font-weight:700"
            elif f > 20: return "color: #F59E0B"
            else:        return "color: #10B981"
        except Exception:
            return ""
    def _color_sharpe(v):
        try:
            f = float(v)
            if f > 1.5:  return "color: #10B981; font-weight:700"
            elif f > 0:  return "color: #6EE7B7"
            else:        return "color: #EF4444"
        except Exception:
            return ""
    styled_stats = (stats.style
        .map(_color_ret,    subset=["Ann Return %"])
        .map(_color_sharpe, subset=["Sharpe"])
        .map(_color_vol,    subset=["Ann Vol %"]))
    st.dataframe(styled_stats, use_container_width=True)

with tab2:
    st.markdown('<div class="sec-hdr">Covariance Matrix</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="fbox">
    <strong>Sample Cov:</strong> Σ = (R−μ)ᵀ(R−μ)/(T−1)
    &nbsp;|&nbsp;
    <strong>Portfolio Var:</strong> σ²_p = wᵀΣw
    &nbsp;|&nbsp;
    Diversification: σ_p &lt; weighted avg individual vols
    </div>""", unsafe_allow_html=True)

    corr = np.corrcoef(R.T)
    ca,cb = st.columns(2)

    with ca:
        fig_cr = go.Figure(go.Heatmap(
            z=corr, x=selected, y=selected,
            colorscale=[[0,"#EF4444"],[.5,"#0D1829"],[1,"#10B981"]],
            zmid=0,zmin=-1,zmax=1,
            text=[[f"{corr[i,j]:.2f}" for j in range(N)] for i in range(N)],
            texttemplate="%{text}",textfont=dict(size=10),
            colorbar=dict(title="ρ"),
        ))
        fig_cr.update_layout(title="Correlation Matrix ρ",height=420,**lay())
        st.plotly_chart(fig_cr,use_container_width=True)

    with cb:
        fig_v = go.Figure()
        fig_v.add_trace(go.Bar(
            x=selected,y=ind_vol,name="Individual Vol",
            marker=dict(color=PALETTE[:N],line=dict(width=0)),
        ))
        fig_v.add_hline(y=p_vol,line=dict(color="#EF4444",width=2,dash="dash"),
                        annotation_text=f"Portfolio {p_vol:.1f}%",
                        annotation_font=dict(color="#EF4444"))
        fig_v.add_hline(y=ind_vol.mean(),line=dict(color="#F59E0B",width=1.5,dash="dot"),
                        annotation_text=f"Avg {ind_vol.mean():.1f}%",
                        annotation_font=dict(color="#F59E0B"))
        fig_v.update_layout(
            title=f"Annualised Volatility  |  Benefit: {ind_vol.mean()-p_vol:.1f}% reduction",
            height=420,showlegend=False,**lay(),
        )
        fig_v.update_xaxes(**ax()); fig_v.update_yaxes(title_text="Ann Vol (%)",**ax())
        st.plotly_chart(fig_v,use_container_width=True)

    st.markdown(f"#### Rolling {roll_win}-Day Correlations")
    pairs = [(selected[a],selected[b]) for a in range(min(N,3))
             for b in range(a+1,min(N,4))][:5]
    fig_rc = go.Figure()
    for i,(a,b) in enumerate(pairs):
        rc = R_df[a].rolling(roll_win).corr(R_df[b])
        fig_rc.add_trace(go.Scatter(
            x=dates,y=rc,name=f"{a}↔{b}",
            line=dict(color=PALETTE[i],width=1.4),
        ))
    fig_rc.add_hline(y=0,line=dict(color="#475569",width=1,dash="dot"))
    fig_rc.update_layout(title=f"Rolling {roll_win}-Day Correlations",height=300,**lay())
    fig_rc.update_xaxes(**ax()); fig_rc.update_yaxes(title_text="Correlation",**ax())
    st.plotly_chart(fig_rc,use_container_width=True)

    st.markdown("#### Portfolio Risk Contribution  `σ²_p = Σᵢ wᵢ(Σw)ᵢ`")
    marg   = Sig_ann @ w_ew
    rcontr = w_ew * marg
    rpct   = rcontr / rcontr.sum() * 100
    fig_rk = go.Figure(go.Bar(
        x=selected,y=rpct,
        marker=dict(color=PALETTE[:N],line=dict(width=0)),
        text=[f"{v:.1f}%" for v in rpct],textposition="outside",
    ))
    fig_rk.add_hline(y=100/N,line=dict(color="#94A3B8",dash="dot"),
                    annotation_text=f"Equal {100/N:.1f}%")
    fig_rk.update_layout(title="Risk Contribution (Equal-Weight Portfolio)",
                         height=300,showlegend=False,**lay())
    fig_rk.update_xaxes(**ax()); fig_rk.update_yaxes(title_text="% of Total Risk",**ax())
    st.plotly_chart(fig_rk,use_container_width=True)

with tab3:
    st.markdown('<div class="sec-hdr">Principal Component Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="fbox">
    <strong>Eigenvalue problem:</strong> Σ·vₖ = λₖ·vₖ
    &nbsp;|&nbsp;
    <strong>Variance explained:</strong> λₖ / Σλⱼ
    &nbsp;|&nbsp;
    PC1 ≈ Market factor · PC2 ≈ Sector rotation
    </div>""", unsafe_allow_html=True)

    pca = PCA()
    pca.fit(Rc)
    evr   = pca.explained_variance_ratio_
    cum_e = np.cumsum(evr)
    cond  = float(np.sqrt(pca.explained_variance_[0] / max(pca.explained_variance_[-1],1e-12)))

    c1,c2,c3 = st.columns(3)
    c1.metric("PC1 Variance",     f"{evr[0]*100:.1f}%")
    c2.metric("PCs for 90%",      int(np.argmax(cum_e>=.9))+1)
    c3.metric("Condition Number", f"{cond:.1f}")

    cl,cr = st.columns(2)
    with cl:
        fig_sc = make_subplots(specs=[[{"secondary_y":True}]])
        fig_sc.add_trace(go.Bar(
            x=list(range(1,N+1)),y=evr*100,
            name="Individual",marker_color="#00D4FF",opacity=0.75,
        ),secondary_y=False)
        fig_sc.add_trace(go.Scatter(
            x=list(range(1,N+1)),y=cum_e*100,
            name="Cumulative",line=dict(color="#F59E0B",width=2),mode="lines+markers",
        ),secondary_y=True)
        fig_sc.add_hline(y=90,line=dict(color="#EF4444",dash="dash",width=1),secondary_y=True)
        fig_sc.update_layout(title="Scree Plot",height=360,**lay())
        fig_sc.update_xaxes(title_text="PC",dtick=1,**ax())
        fig_sc.update_yaxes(title_text="Var %",secondary_y=False,**ax())
        fig_sc.update_yaxes(title_text="Cumul %",secondary_y=True)
        st.plotly_chart(fig_sc,use_container_width=True)

    with cr:
        n_show   = min(n_pca_show,N)
        loadings = pca.components_[:n_show]
        fig_ld = go.Figure(go.Heatmap(
            z=loadings, x=selected,
            y=[f"PC{k+1} ({evr[k]*100:.1f}%)" for k in range(n_show)],
            colorscale=[[0,"#EF4444"],[.5,"#0D1829"],[1,"#10B981"]],
            zmid=0,zmin=-0.8,zmax=0.8,
            text=[[f"{loadings[k,j]:.2f}" for j in range(N)] for k in range(n_show)],
            texttemplate="%{text}",textfont=dict(size=9),
            colorbar=dict(title="Loading"),
        ))
        fig_ld.update_layout(title=f"PC Loadings (first {n_show})",height=360,**lay())
        st.plotly_chart(fig_ld,use_container_width=True)

    pc_scores = pca.transform(Rc)
    pc_df = pd.DataFrame(pc_scores[:,:2],index=dates,columns=["PC1","PC2"])
    fig_pc = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.06,
                           subplot_titles=["PC1 — Market Factor","PC2 — Sector Rotation"])
    for k,col_pc,nm in [(0,"#00D4FF","PC1"),(1,"#F59E0B","PC2")]:
        fig_pc.add_trace(go.Scatter(
            x=pc_df.index,y=pc_df[nm],name=nm,
            line=dict(color=col_pc,width=1),
            fill="tozeroy",
            fillcolor=["rgba(0,212,255,0.07)","rgba(245,158,11,0.07)"][k],
        ),row=k+1,col=1)
    fig_pc.update_layout(height=380,showlegend=False,**lay())
    fig_pc.update_xaxes(**ax()); fig_pc.update_yaxes(**ax())
    st.plotly_chart(fig_pc,use_container_width=True)

    st.markdown("#### PCA Biplot — PC1 vs PC2")
    comps  = pca.components_
    fig_bp = go.Figure()
    for i,t in enumerate(selected):
        c = PALETTE[i%len(PALETTE)]
        fig_bp.add_annotation(
            ax=0,ay=0,x=comps[0,i]*6,y=comps[1,i]*6,
            xref="x",yref="y",axref="x",ayref="y",
            arrowhead=3,arrowwidth=2,arrowcolor=c,
        )
        fig_bp.add_trace(go.Scatter(
            x=[comps[0,i]*6.8],y=[comps[1,i]*6.8],
            mode="text",text=[t],
            textfont=dict(size=11,color=c,family="JetBrains Mono"),
            showlegend=False,
        ))
    fig_bp.update_layout(
        title="PCA Biplot",height=420,**lay(),
        xaxis=dict(title="PC1 Loading",range=[-8,8],**ax()),
        yaxis=dict(title="PC2 Loading",range=[-8,8],**ax()),
    )
    st.plotly_chart(fig_bp,use_container_width=True)

with tab4:
    st.markdown('<div class="sec-hdr">SVD Factor Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="fbox">
    <strong>SVD:</strong> R(T×N) = U(T×r)·Σ(r×r)·Vᵀ(r×N)
    &nbsp;|&nbsp;
    <strong>Factor energy:</strong> σᵢ² / Σσⱼ²
    &nbsp;|&nbsp;
    Systematic vs Idiosyncratic decomposition
    </div>""", unsafe_allow_html=True)

    U_s,s_s,Vt_s = np.linalg.svd(Rc,full_matrices=False)
    energy  = (s_s**2)/(s_s**2).sum()*100
    cum_nrg = np.cumsum(energy)

    c1,c2,c3 = st.columns(3)
    c1.metric("Factor 1 Energy",  f"{energy[0]:.1f}%")
    c2.metric("Factors for 80%",  int(np.argmax(cum_nrg>=80))+1)
    c3.metric("Condition Number", f"{s_s[0]/max(s_s[-1],1e-12):.1f}")

    cl,cr = st.columns(2)
    with cl:
        fig_sv = go.Figure()
        fig_sv.add_trace(go.Bar(
            x=list(range(1,N+1)),y=s_s[:N],
            name="Singular Value",marker_color="#00D4FF",opacity=0.85,
        ))
        fig_sv.add_trace(go.Scatter(
            x=list(range(1,N+1)),y=cum_nrg[:N],
            name="Cumul Energy %",yaxis="y2",
            line=dict(color="#F59E0B",width=2),mode="lines+markers",
        ))
        fig_sv.update_layout(
            title="Singular Value Spectrum",
            yaxis2=dict(overlaying="y",side="right",title="Cumul Energy %"),
            height=380,**lay(),
        )
        fig_sv.update_xaxes(title_text="Factor",**ax())
        fig_sv.update_yaxes(title_text="σᵢ",**ax())
        st.plotly_chart(fig_sv,use_container_width=True)

    with cr:
        fig_pie = go.Figure(go.Pie(
            labels=[f"F{i+1}" for i in range(min(6,N))],
            values=energy[:min(6,N)],hole=0.45,
            marker=dict(colors=PALETTE[:min(6,N)]),
            textfont=dict(family="JetBrains Mono"),
        ))
        fig_pie.update_layout(title="Factor Energy Share",height=380,**lay())
        st.plotly_chart(fig_pie,use_container_width=True)

    V_mat = Vt_s.T
    n_f   = min(5,N)
    fig_sh = go.Figure(go.Heatmap(
        z=V_mat[:,:n_f].T, x=selected,
        y=[f"F{k+1} ({energy[k]:.1f}%)" for k in range(n_f)],
        colorscale=[[0,"#EF4444"],[.5,"#0D1829"],[1,"#10B981"]],
        zmid=0,zmin=-0.7,zmax=0.7,
        text=[[f"{V_mat[j,k]:.2f}" for j in range(N)] for k in range(n_f)],
        texttemplate="%{text}",textfont=dict(size=10),
        colorbar=dict(title="Loading"),
    ))
    fig_sh.update_layout(title="SVD Factor Loadings Vᵀ",height=300,**lay())
    st.plotly_chart(fig_sh,use_container_width=True)

    n_fac  = st.slider("Systematic factors",1,min(N,6),2,key="svd_fac")
    R_sys  = U_s[:,:n_fac] @ np.diag(s_s[:n_fac]) @ Vt_s[:n_fac,:]
    R_idio = Rc - R_sys
    sys_v  = R_sys.std(axis=0)  * np.sqrt(252)*100
    ido_v  = R_idio.std(axis=0) * np.sqrt(252)*100
    fig_dc = go.Figure()
    fig_dc.add_trace(go.Bar(name=f"Systematic ({n_fac}F)",
        x=selected,y=sys_v,marker=dict(color="#00D4FF",opacity=0.85)))
    fig_dc.add_trace(go.Bar(name="Idiosyncratic",
        x=selected,y=ido_v,marker=dict(color="#F59E0B",opacity=0.85)))
    fig_dc.update_layout(barmode="group",
        title="Systematic vs Idiosyncratic Volatility",height=320,**lay())
    fig_dc.update_xaxes(**ax()); fig_dc.update_yaxes(title_text="Ann Vol (%)",**ax())
    st.plotly_chart(fig_dc,use_container_width=True)

with tab5:
    st.markdown('<div class="sec-hdr">Portfolio Optimization</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="fbox">
    <strong>Markowitz (1952):</strong>
    min wᵀΣw &nbsp; s.t. &nbsp; wᵀμ=μ_target, wᵀ1=1, w≥0
    &nbsp;|&nbsp;
    <strong>Max Sharpe:</strong> tangency portfolio on CML
    </div>""", unsafe_allow_html=True)

    allow_short = st.checkbox("Allow short selling",value=False)

    def port_stats(w):
        r = float(w @ mu_ann)*100
        v = float(np.sqrt(w @ Sig_ann @ w))*100
        return v,r

    def max_sharpe_w(short):
        bnd = ((-1,1),)*N if short else ((0,1),)*N
        res = minimize(
            lambda w: -(float(w@mu_ann)*100 - rf_rate*100)/max(float(np.sqrt(w@Sig_ann@w))*100,1e-8),
            np.ones(N)/N, bounds=bnd,
            constraints={"type":"eq","fun":lambda w:w.sum()-1},method="SLSQP",
        )
        return res.x if res.success else np.ones(N)/N

    def min_var_w():
        res = minimize(
            lambda w: float(w@Sig_ann@w),
            np.ones(N)/N, bounds=((0,1),)*N,
            constraints={"type":"eq","fun":lambda w:w.sum()-1},method="SLSQP",
        )
        return res.x if res.success else np.ones(N)/N

    with st.spinner("Optimising…"):
        np.random.seed(0)
        n_port = 1500  # reduced from 4000 for speed
        ww = np.random.dirichlet(np.ones(N),n_port)
        rr = ww @ mu_ann * 100
        vv = np.sqrt(np.einsum("ij,jk,ik->i",ww,Sig_ann,ww))*100
        ss = (rr - rf_rate*100)/np.maximum(vv,1e-8)
        w_msr = max_sharpe_w(allow_short)
        w_mvp = min_var_w()

    v_msr,r_msr = port_stats(w_msr)
    v_mvp,r_mvp = port_stats(w_mvp)
    sh_msr = (r_msr - rf_rate*100)/max(v_msr,1e-8)

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Max Sharpe",     f"{sh_msr:.3f}")
    m2.metric("Max Sharpe Ret", f"{r_msr:.2f}%")
    m3.metric("Max Sharpe Vol", f"{v_msr:.2f}%")
    m4.metric("Min Vol",        f"{v_mvp:.2f}%")

    cl,cr = st.columns([2,1])
    with cl:
        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=vv,y=rr,mode="markers",
            marker=dict(size=3,color=ss,colorscale="RdYlGn",
                        colorbar=dict(title="Sharpe"),line=dict(width=0)),
            name="Random Portfolios",
            hovertemplate="Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>",
        ))
        for i,t in enumerate(selected):
            fig_ef.add_trace(go.Scatter(
                x=[ind_vol[i]],y=[mu_ann[i]*100],
                mode="markers+text",text=[t],textposition="top right",
                textfont=dict(size=9,color=PALETTE[i%len(PALETTE)]),
                marker=dict(size=10,color=PALETTE[i%len(PALETTE)],
                            line=dict(width=1,color="#070C14")),
                showlegend=False,
            ))
        for lbl,vx,ry,c,sym,sz in [
            ("Max Sharpe",v_msr,r_msr,"#10B981","star",18),
            ("Min Vol",   v_mvp,r_mvp,"#00D4FF","star",18),
            ("EW",        p_vol,p_ret,"#F59E0B","diamond",14),
        ]:
            fig_ef.add_trace(go.Scatter(
                x=[vx],y=[ry],mode="markers",name=lbl,
                marker=dict(size=sz,color=c,symbol=sym,
                            line=dict(width=1.5,color="white")),
            ))
        v_cml = np.linspace(0,vv.max()*1.1,100)
        r_cml = rf_rate*100 + sh_msr*v_cml
        fig_ef.add_trace(go.Scatter(
            x=v_cml,y=r_cml,mode="lines",name="CML",
            line=dict(color="#8B5CF6",width=1.5,dash="dash"),
        ))
        fig_ef.update_layout(title="Efficient Frontier + CML",height=500,**lay())
        fig_ef.update_xaxes(title_text="Risk — Volatility (%)",**ax())
        fig_ef.update_yaxes(title_text="Return (%)",**ax())
        st.plotly_chart(fig_ef,use_container_width=True)

    with cr:
        for lbl_w,w_w in [("Max Sharpe",w_msr),("Min Variance",w_mvp)]:
            st.markdown(f"**{lbl_w} Weights**")
            wdf = pd.DataFrame({"Asset":selected,"Weight":w_w})
            wdf = wdf[wdf.Weight>0.005].sort_values("Weight",ascending=False)
            fig_w = go.Figure(go.Bar(
                x=wdf.Weight*100,y=wdf.Asset,orientation="h",
                marker=dict(
                    color=[PALETTE[selected.index(t)%len(PALETTE)] for t in wdf.Asset],
                    line=dict(width=0),
                ),
                text=[f"{v*100:.1f}%" for v in wdf.Weight],
                textposition="outside",
            ))
            fig_w.update_layout(
                paper_bgcolor="#070C14",plot_bgcolor="#0D1829",
                font=dict(family="JetBrains Mono",color="#94A3B8"),
                xaxis=dict(title="Weight (%)",**ax()),
                yaxis=dict(autorange="reversed"),
                height=250,margin=dict(l=0,r=0,t=8,b=0),showlegend=False,
            )
            st.plotly_chart(fig_w,use_container_width=True)

with tab6:
    st.markdown('<div class="sec-hdr">ML Price Predictions</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="fbox">
    <strong>Features:</strong> Log-returns · MA ratios (5/10/20d) · Volatility · RSI(14) · Volume ratio
    &nbsp;|&nbsp;
    <strong>Model:</strong> Random Forest (60 trees, max_depth=10) · 80/20 train-test split
    </div>""", unsafe_allow_html=True)

    if "rf_preds" not in st.session_state:
        st.session_state.rf_preds = {}

    cl,cr = st.columns([1,2])
    with cl:
        pred_t = st.selectbox("Stock",selected,key="rf_sel")
        pred_d = st.slider("Forecast days",1,15,5)
        run_btn = st.button("🔮  Run Prediction")

    if run_btn:
        with st.spinner("Training Random Forest…"):
            df_ml = all_dfs[pred_t].copy()
            df_ml["ret"]   = df_ml["Close"].pct_change()
            df_ml["ma5"]   = df_ml["Close"].rolling(5).mean()
            df_ml["ma10"]  = df_ml["Close"].rolling(10).mean()
            df_ml["ma20"]  = df_ml["Close"].rolling(20).mean()
            df_ml["ma5r"]  = df_ml["Close"] / df_ml["ma5"]
            df_ml["ma10r"] = df_ml["Close"] / df_ml["ma10"]
            df_ml["ma20r"] = df_ml["Close"] / df_ml["ma20"]
            df_ml["vol10"] = df_ml["ret"].rolling(10).std()
            delta          = df_ml["Close"].diff()
            g = delta.where(delta>0,0).rolling(14).mean()
            l = (-delta.where(delta<0,0)).rolling(14).mean()
            df_ml["rsi"]   = 100 - 100/(1 + g/(l+1e-9))
            if "Volume" in df_ml.columns:
                df_ml["volr"] = df_ml["Volume"]/(df_ml["Volume"].rolling(10).mean()+1e-9)
            else:
                df_ml["volr"] = 1.0
            df_ml = df_ml.dropna()

            FCOLS = ["ret","ma5r","ma10r","ma20r","vol10","rsi","volr"]
            X = df_ml[FCOLS].values
            y = df_ml["Close"].values
            sx,sy = StandardScaler(),StandardScaler()
            Xs = sx.fit_transform(X)
            ys = sy.fit_transform(y.reshape(-1,1)).ravel()
            split = int(len(X)*0.8)
            mdl   = RandomForestRegressor(n_estimators=60,max_depth=10,
                                          random_state=42,n_jobs=-1)
            mdl.fit(Xs[:split],ys[:split])
            yp  = sy.inverse_transform(mdl.predict(Xs[split:]).reshape(-1,1)).ravel()
            ya  = y[split:]
            r2  = float(1-((ya-yp)**2).sum()/((ya-ya.mean())**2+1e-12).sum())
            rmse= float(np.sqrt(((ya-yp)**2).mean()))
            mape= float(np.mean(np.abs((ya-yp)/(ya+1e-9)))*100)
            future,last_f = [],Xs[-1:].copy()
            cur = float(df_ml["Close"].iloc[-1])
            for _ in range(pred_d):
                nxt = float(sy.inverse_transform(mdl.predict(last_f).reshape(-1,1))[0,0])
                future.append(nxt)
                last_f = last_f*(1+np.random.normal(0,0.008,last_f.shape))
            st.session_state.rf_preds[pred_t] = {
                "current": cur, "hist_idx": df_ml.index,
                "hist_px": y,   "test_idx": df_ml.index[split:],
                "y_test": ya,   "y_pred":   yp,
                "future": future, "pred_d": pred_d,
                "r2":r2,"rmse":rmse,"mape":mape,
                "fi":mdl.feature_importances_,"fcols":FCOLS,
            }

    with cr:
        if pred_t in st.session_state.rf_preds:
            res = st.session_state.rf_preds[pred_t]
            m1,m2,m3 = st.columns(3)
            m1.metric("R² Score",f"{res['r2']:.4f}")
            m2.metric("RMSE",    f"{res['rmse']:.2f}")
            m3.metric("MAPE",    f"{res['mape']:.2f}%")

            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(
                x=res["test_idx"],y=res["y_test"],
                name="Actual",line=dict(color="#00D4FF",width=1.6),
            ))
            fig_p.add_trace(go.Scatter(
                x=res["test_idx"],y=res["y_pred"],
                name="Predicted",line=dict(color="#F59E0B",width=1.6,dash="dash"),
            ))
            fig_p.update_layout(
                title=f"{pred_t}  —  Actual vs Predicted (Test Set)",
                height=340,**lay(),
            )
            fig_p.update_xaxes(**ax()); fig_p.update_yaxes(title_text="Price",**ax())
            st.plotly_chart(fig_p,use_container_width=True)

            last_date   = res["hist_idx"][-1]
            bday        = pd.tseries.offsets.BusinessDay(1)
            fut_dates   = [last_date + bday*(i+1) for i in range(res["pred_d"])]
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(
                x=res["hist_idx"][-40:],y=res["hist_px"][-40:],
                name="Historical",line=dict(color="#00D4FF",width=1.6),
            ))
            fig_f.add_trace(go.Scatter(
                x=fut_dates,y=res["future"],name="Forecast",
                mode="lines+markers",
                line=dict(color="#F59E0B",width=2,dash="dash"),
                marker=dict(size=7,color="#F59E0B",line=dict(width=1.5,color="white")),
            ))
            fig_f.add_vline(x=str(last_date.date()),line=dict(color="#64748B",dash="dot"))
            fig_f.update_layout(
                title=f"{pred_t}  —  {res['pred_d']}-Day Forecast",
                height=340,**lay(),
            )
            fig_f.update_xaxes(**ax()); fig_f.update_yaxes(title_text="Price",**ax())
            st.plotly_chart(fig_f,use_container_width=True)

        else:
            st.info("Select a stock and click **🔮 Run Prediction** to generate forecasts.")

    if pred_t in st.session_state.rf_preds:
        res = st.session_state.rf_preds[pred_t]
        st.markdown("#### Feature Importance")
        fi_df = pd.DataFrame({"Feature":res["fcols"],"Importance":res["fi"]})\
                  .sort_values("Importance",ascending=False)
        fig_fi = go.Figure(go.Bar(
            x=fi_df.Importance,y=fi_df.Feature,orientation="h",
            marker=dict(color="#8B5CF6",opacity=0.85,line=dict(width=0)),
            text=[f"{v:.3f}" for v in fi_df.Importance],textposition="outside",
        ))
        fig_fi.update_layout(
            paper_bgcolor="#070C14",plot_bgcolor="#0D1829",
            font=dict(family="JetBrains Mono",color="#94A3B8"),
            yaxis=dict(autorange="reversed"),
            xaxis=dict(title="Importance",**ax()),
            height=280,margin=dict(l=0,r=60,t=8,b=0),showlegend=False,
        )
        st.plotly_chart(fig_fi,use_container_width=True)

        chg = (res["future"][-1]-res["current"])/res["current"]*100
        if chg > 2:
            st.success(f"🟢  BUY signal  —  {res['pred_d']}-day forecast: **{chg:+.2f}%**")
        elif chg < -2:
            st.error(  f"🔴  SELL signal  —  {res['pred_d']}-day forecast: **{chg:+.2f}%**")
        else:
            st.info(   f"🟡  HOLD signal  —  {res['pred_d']}-day forecast: **{chg:+.2f}%**")


with tab7:
    st.markdown('<div class="sec-hdr">Asset Pricing & Microstructure</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="fbox">
    <strong>CAPM:</strong> E[Rᵢ] = Rƒ + βᵢ·(E[Rm]−Rƒ)
    &nbsp;|&nbsp;
    <strong>Black-Scholes:</strong> C = S·Φ(d₁) − K·e^(−rT)·Φ(d₂)
    &nbsp;|&nbsp;
    <strong>Markov:</strong> P(Sₜ|Sₜ₋₁) · π = π
    &nbsp;|&nbsp;
    <strong>Tests:</strong> Variance Ratio · Ljung-Box · Runs
    </div>""", unsafe_allow_html=True)

    st7a, st7b, st7c, st7d, st7e, st7f = st.tabs([
        "📐 CAPM & SML",
        "🎲 Option Pricing",
        "🔄 Regime Detection",
        "📉 Market Efficiency",
        "⏱️ Stopping Time",
        "🧩 Multi-Factor",
    ])

    from scipy import stats as sp_stats
    from scipy.special import ndtr as _ndtr   # standard-normal CDF, fast

    def norm_cdf(x):
        return _ndtr(x)

    mkt_ret_d = R @ w_ew
    mkt_ret_ann = float(mkt_ret_d.mean() * 252)
    mkt_vol_ann = float(mkt_ret_d.std() * np.sqrt(252))
    mkt_excess   = mkt_ret_ann - rf_rate

    # 7A — CAPM & SECURITY MARKET LINE
    with st7a:
        st.markdown("""
        <div class="fbox">
        <strong>CAPM:</strong> E[Rᵢ] = Rƒ + βᵢ·(E[Rm]−Rƒ)
        &nbsp;|&nbsp;
        <strong>Beta:</strong> βᵢ = Cov(Rᵢ, Rm) / Var(Rm)
        &nbsp;|&nbsp;
        <strong>Alpha:</strong> αᵢ = μᵢ − [Rƒ + βᵢ·(μm−Rƒ)]
        </div>""", unsafe_allow_html=True)

        # ── Compute betas, alphas, t-stats ───────────────────────────────────
        var_mkt = float(mkt_ret_d.var())
        betas, alphas, t_betas, t_alphas, r2s = [], [], [], [], []
        for i, t in enumerate(selected):
            ri = R[:, i]
            cov_im = float(np.cov(ri, mkt_ret_d)[0, 1])
            beta_i = cov_im / max(var_mkt, 1e-12)
            alpha_i = mu_d[i] - (rf_rate / 252 + beta_i * (mkt_ret_d.mean() - rf_rate / 252))
            y_hat   = rf_rate / 252 + beta_i * (mkt_ret_d - rf_rate / 252)
            resid   = ri - y_hat
            n_obs   = len(ri)
            se2     = resid.var(ddof=2)
            se_beta = np.sqrt(se2 / max(var_mkt * n_obs, 1e-12))
            se_alph = np.sqrt(se2 * (1 / n_obs + mkt_ret_d.mean()**2 / max(var_mkt * n_obs, 1e-12)))
            t_beta  = beta_i / max(se_beta, 1e-12)
            t_alph  = (alpha_i * 252) / max(se_alph * np.sqrt(252), 1e-12)
            ss_res  = (resid**2).sum()
            ss_tot  = ((ri - ri.mean())**2).sum()
            betas.append(beta_i); alphas.append(alpha_i * 252)
            t_betas.append(t_beta); t_alphas.append(t_alph)
            r2s.append(1 - ss_res / max(ss_tot, 1e-12))

        betas  = np.array(betas)
        alphas = np.array(alphas)

        # ── Metrics row ───────────────────────────────────────────────────────
        ca, cb, cc, cd = st.columns(4)
        ca.metric("Market Return (Ann)", f"{mkt_ret_ann*100:.2f}%")
        cb.metric("Market Vol (Ann)",    f"{mkt_vol_ann*100:.2f}%")
        cc.metric("Market Sharpe",       f"{(mkt_ret_ann-rf_rate)/max(mkt_vol_ann,1e-8):.3f}")
        cd.metric("Avg Beta",            f"{betas.mean():.3f}")

        st.markdown("#### Security Market Line (SML)  —  Beta vs Expected Return")
        # SML line
        beta_grid  = np.linspace(betas.min() * 0.8, betas.max() * 1.2, 100)
        exp_ret_sml = rf_rate * 100 + beta_grid * mkt_excess * 100
        fig_sml = go.Figure()
        fig_sml.add_trace(go.Scatter(
            x=beta_grid, y=exp_ret_sml, mode="lines", name="SML",
            line=dict(color="#00D4FF", width=1.8, dash="dash"),
        ))
        for i, t in enumerate(selected):
            alpha_clr = "#10B981" if alphas[i] >= 0 else "#EF4444"
            fig_sml.add_trace(go.Scatter(
                x=[betas[i]], y=[mu_ann[i] * 100],
                mode="markers+text", name=t,
                text=[t], textposition="top right",
                textfont=dict(size=10, color=PALETTE[i % len(PALETTE)], family="JetBrains Mono"),
                marker=dict(size=12, color=alpha_clr,
                            line=dict(width=2, color=PALETTE[i % len(PALETTE)])),
                hovertemplate=(
                    f"<b>{t}</b><br>β={betas[i]:.3f}<br>"
                    f"μ={mu_ann[i]*100:.2f}%<br>α={alphas[i]*100:.2f}%<extra></extra>"
                ),
                showlegend=False,
            ))
        fig_sml.add_hline(y=rf_rate * 100, line=dict(color="#475569", dash="dot", width=1),
                          annotation_text=f"Rƒ={rf_rate*100:.1f}%")
        fig_sml.update_layout(
            title="Security Market Line  |  Green = positive alpha  |  Red = negative alpha",
            height=460, **lay(),
        )
        fig_sml.update_xaxes(title_text="Beta (β)", **ax())
        fig_sml.update_yaxes(title_text="Ann Return (%)", **ax())
        st.plotly_chart(fig_sml, use_container_width=True)

        # ── Alpha significance table ──────────────────────────────────────────
        st.markdown("#### Alpha Significance Testing  —  OLS Regression Results")
        capm_df = pd.DataFrame({
            "Beta (β)":   [f"{b:.4f}" for b in betas],
            "t(β)":       [f"{t:.2f}" for t in t_betas],
            "Alpha α (Ann %)": [f"{a*100:.3f}" for a in alphas],
            "t(α)":       [f"{t:.2f}" for t in t_alphas],
            "Sig α":      ["★★★" if abs(t) > 3.29 else "★★" if abs(t) > 2.58 else "★" if abs(t) > 1.96 else "" for t in t_alphas],
            "R²":         [f"{r:.4f}" for r in r2s],
            "Sector":     [SECTORS.get(t, "—") for t in selected],
        }, index=selected)
        st.dataframe(capm_df.style.map(
            lambda v: "color:#10B981" if v.startswith("★") else "",
            subset=["Sig α"]
        ), use_container_width=True)
        st.caption("★ p<0.05 · ★★ p<0.01 · ★★★ p<0.001  (two-tailed t-test against H₀: α = 0)")

        # ── Rolling beta ─────────────────────────────────────────────────────
        st.markdown(f"#### Rolling {roll_win}-Day Beta  —  Time-Varying Market Sensitivity")
        fig_rb = go.Figure()
        for i, t in enumerate(selected[:min(N, 5)]):
            rb = []
            for end in range(roll_win, T_days):
                ri_w  = R[end - roll_win:end, i]
                rm_w  = mkt_ret_d[end - roll_win:end]
                var_w = rm_w.var()
                rb.append(np.cov(ri_w, rm_w)[0, 1] / max(var_w, 1e-12))
            fig_rb.add_trace(go.Scatter(
                x=dates[roll_win:], y=rb, name=t,
                line=dict(color=PALETTE[i % len(PALETTE)], width=1.3),
            ))
        fig_rb.add_hline(y=1.0, line=dict(color="#475569", dash="dot", width=1),
                         annotation_text="β=1 (market)")
        fig_rb.update_layout(title=f"Rolling {roll_win}-Day Beta", height=360, **lay())
        fig_rb.update_xaxes(**ax())
        fig_rb.update_yaxes(title_text="Beta (β)", **ax())
        st.plotly_chart(fig_rb, use_container_width=True)

    # 7B — BLACK-SCHOLES OPTION PRICING
    with st7b:
        st.markdown("""
        <div class="fbox">
        <strong>Black-Scholes:</strong> C = S·Φ(d₁) − K·e^(−rT)·Φ(d₂) &nbsp;|&nbsp;
        d₁ = [ln(S/K) + (r + σ²/2)·T] / (σ√T) &nbsp;|&nbsp;
        d₂ = d₁ − σ√T &nbsp;|&nbsp;
        <strong>Greeks:</strong> Δ = Φ(d₁) · Γ = φ(d₁)/(Sσ√T) · ν = S·φ(d₁)·√T
        </div>""", unsafe_allow_html=True)

        # ── BS engine ─────────────────────────────────────────────────────────
        def bs_price(S, K, T_yr, r, sigma, opt="call"):
            if T_yr <= 0 or sigma <= 0:
                return max(S - K, 0) if opt == "call" else max(K - S, 0)
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_yr) / (sigma * np.sqrt(T_yr))
            d2 = d1 - sigma * np.sqrt(T_yr)
            if opt == "call":
                return S * norm_cdf(d1) - K * np.exp(-r * T_yr) * norm_cdf(d2)
            return K * np.exp(-r * T_yr) * norm_cdf(-d2) - S * norm_cdf(-d1)

        def bs_greeks(S, K, T_yr, r, sigma):
            if T_yr <= 0 or sigma <= 0:
                return dict(delta=1.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_yr) / (sigma * np.sqrt(T_yr))
            d2 = d1 - sigma * np.sqrt(T_yr)
            phi_d1 = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
            delta  = norm_cdf(d1)
            gamma  = phi_d1 / max(S * sigma * np.sqrt(T_yr), 1e-12)
            vega   = S * phi_d1 * np.sqrt(T_yr) / 100
            theta  = (-(S * phi_d1 * sigma) / (2 * np.sqrt(T_yr))
                      - r * K * np.exp(-r * T_yr) * norm_cdf(d2)) / 365
            rho    = K * T_yr * np.exp(-r * T_yr) * norm_cdf(d2) / 100
            return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

        # ── Controls ──────────────────────────────────────────────────────────
        cl, cm, cr = st.columns([1, 1, 2])
        with cl:
            opt_t    = st.selectbox("Underlying", selected, key="opt_t7")
            opt_type = st.radio("Option type", ["Call", "Put"], horizontal=True, key="opt_type7")
        with cm:
            S0       = float(closes[opt_t].iloc[-1])
            K_pct    = st.slider("Strike (% of spot)", 80, 120, 100, 1, key="opt_K7")
            K        = S0 * K_pct / 100
            T_days_opt = st.slider("Expiry (calendar days)", 7, 365, 90, 7, key="opt_T7")
            T_yr     = T_days_opt / 365
            hist_vol = float(R_df[opt_t].std() * np.sqrt(252))
            imp_vol  = st.slider("Implied Vol σ (%)", 5, 150, int(hist_vol * 100), 1, key="opt_iv7") / 100

        price = bs_price(S0, K, T_yr, rf_rate, imp_vol, opt_type.lower())
        greeks = bs_greeks(S0, K, T_yr, rf_rate, imp_vol)

        with cr:
            g1, g2, g3, g4, g5 = st.columns(5)
            g1.metric("Price",   f"{price:.2f}")
            g2.metric("Delta Δ", f"{greeks['delta']:.4f}")
            g3.metric("Gamma Γ", f"{greeks['gamma']:.4f}")
            g4.metric("Vega ν",  f"{greeks['vega']:.4f}")
            g5.metric("Theta θ", f"{greeks['theta']:.4f}")

        # ── Implied vol surface ───────────────────────────────────────────────
        st.markdown("#### Implied Volatility Surface  —  Strike × Expiry")
        strikes_pct = np.arange(75, 130, 5)
        expiries    = np.array([30, 60, 90, 120, 180, 252])
        vol_base    = hist_vol
        # Synthetic IV surface with skew + term structure
        iv_surface  = np.zeros((len(strikes_pct), len(expiries)))
        for si, kp in enumerate(strikes_pct):
            for ei, exp in enumerate(expiries):
                moneyness = (kp - 100) / 100
                skew      = -0.15 * moneyness - 0.08 * moneyness**2
                term_str  = 0.02 * np.log(max(exp, 1) / 30)
                iv_surface[si, ei] = max(vol_base + skew + term_str, 0.05)

        fig_vol_surf = go.Figure(go.Surface(
            x=expiries, y=strikes_pct, z=iv_surface * 100,
            colorscale=[[0, "#0D1829"], [0.3, "#1C3A5E"], [0.6, "#00D4FF"], [1, "#10B981"]],
            showscale=True,
            colorbar=dict(title="IV %", tickfont=dict(family="JetBrains Mono", color="#94A3B8")),
        ))
        fig_vol_surf.update_layout(
            title=f"Implied Volatility Surface — {opt_t}",
            scene=dict(
                xaxis=dict(title="Expiry (days)", backgroundcolor="#0D1829",
                           gridcolor="#182A40", tickfont=dict(family="JetBrains Mono", color="#94A3B8")),
                yaxis=dict(title="Strike (%)", backgroundcolor="#0D1829",
                           gridcolor="#182A40", tickfont=dict(family="JetBrains Mono", color="#94A3B8")),
                zaxis=dict(title="IV (%)", backgroundcolor="#0D1829",
                           gridcolor="#182A40", tickfont=dict(family="JetBrains Mono", color="#94A3B8")),
                bgcolor="#070C14",
            ),
            height=480, paper_bgcolor="#070C14",
            font=dict(family="JetBrains Mono", color="#94A3B8"),
            margin=dict(l=0, r=0, t=44, b=0),
        )
        st.plotly_chart(fig_vol_surf, use_container_width=True)

        # ── Historical vs implied vol ─────────────────────────────────────────
        st.markdown("#### Historical vs Implied Volatility  —  Volatility Premium")
        hist_rv = R_df[opt_t].rolling(roll_win).std() * np.sqrt(252) * 100
        impl_v_ts = hist_rv * (1 + np.random.RandomState(0).normal(0, 0.08, len(hist_rv)))
        impl_v_ts = pd.Series(impl_v_ts, index=R_df.index).clip(lower=1)
        vol_prem  = impl_v_ts - hist_rv

        fig_vols = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 vertical_spacing=0.06, row_heights=[0.65, 0.35])
        fig_vols.add_trace(go.Scatter(
            x=dates, y=hist_rv, name="Historical Vol",
            line=dict(color="#00D4FF", width=1.5),
        ), row=1, col=1)
        fig_vols.add_trace(go.Scatter(
            x=dates, y=impl_v_ts, name="Implied Vol",
            line=dict(color="#F59E0B", width=1.5, dash="dash"),
        ), row=1, col=1)
        fig_vols.add_trace(go.Bar(
            x=dates, y=vol_prem, name="Vol Premium",
            marker_color=["#10B981" if v >= 0 else "#EF4444" for v in vol_prem],
            opacity=0.7,
        ), row=2, col=1)
        fig_vols.update_layout(
            title=f"{opt_t}  —  Historical vs Implied Volatility",
            height=420, **lay(),
        )
        fig_vols.update_xaxes(**ax())
        fig_vols.update_yaxes(title_text="Vol (%)", row=1, col=1, **ax())
        fig_vols.update_yaxes(title_text="Premium (%)", row=2, col=1, **ax())
        st.plotly_chart(fig_vols, use_container_width=True)

        # ── Greeks vs spot ────────────────────────────────────────────────────
        st.markdown("#### Option Greeks vs Spot Price")
        spot_range = np.linspace(S0 * 0.7, S0 * 1.3, 100)
        delta_v  = [bs_greeks(s, K, T_yr, rf_rate, imp_vol)["delta"] for s in spot_range]
        gamma_v  = [bs_greeks(s, K, T_yr, rf_rate, imp_vol)["gamma"] * 100 for s in spot_range]
        vega_v   = [bs_greeks(s, K, T_yr, rf_rate, imp_vol)["vega"] * 100 for s in spot_range]

        fig_grk = make_subplots(rows=1, cols=3, subplot_titles=["Delta (Δ)", "Gamma (Γ) ×100", "Vega (ν) ×100"])
        for col_i, (vals, col_c, lbl) in enumerate(zip(
            [delta_v, gamma_v, vega_v], ["#00D4FF", "#10B981", "#F59E0B"],
            ["Delta", "Gamma×100", "Vega×100"]), 1):
            fig_grk.add_trace(go.Scatter(
                x=spot_range, y=vals, name=lbl,
                line=dict(color=col_c, width=2), showlegend=False,
            ), row=1, col=col_i)
            fig_grk.add_vline(x=S0, line=dict(color="#475569", dash="dot", width=1), row=1, col=col_i)
            fig_grk.add_vline(x=K,  line=dict(color="#EF4444", dash="dot", width=1), row=1, col=col_i)
        fig_grk.update_layout(title="Option Greeks vs Spot  |  Blue dotted = Spot  |  Red dotted = Strike",
                               height=320, **lay())
        fig_grk.update_xaxes(**ax())
        fig_grk.update_yaxes(**ax())
        st.plotly_chart(fig_grk, use_container_width=True)

    # 7C — REGIME DETECTION (MARKOV SWITCHING)
    with st7c:
        st.markdown("""
        <div class="fbox">
        <strong>Markov Chain:</strong> P(Sₜ=j | Sₜ₋₁=i) = Pᵢⱼ &nbsp;|&nbsp;
        <strong>Stationary dist:</strong> πᵀP = πᵀ &nbsp;|&nbsp;
        <strong>Regimes:</strong> Bull (high μ, low σ)  ·  Neutral  ·  Bear/Volatile (low μ, high σ)
        </div>""", unsafe_allow_html=True)

        reg_t   = st.selectbox("Stock for regime analysis", selected, key="reg_t7")
        ret_ser = R_df[reg_t].values
        T_reg   = len(ret_ser)

        # ── Classify regimes via rolling vol + return ─────────────────────────
        roll20  = pd.Series(ret_ser).rolling(20)
        r20_mu  = roll20.mean().fillna(0).values
        r20_sig = roll20.std().fillna(ret_ser.std()).values
        med_sig = np.median(r20_sig)
        q75_sig = np.percentile(r20_sig, 75)
        regimes = np.where(r20_sig > q75_sig, 2,              # Volatile/Bear
                           np.where(r20_mu > r20_mu.mean(), 0, 1))  # Bull / Neutral

        regime_labels = {0: "Bull 📈", 1: "Neutral ➡️", 2: "Bear/Volatile 📉"}
        regime_cols   = {0: "#10B981", 1: "#F59E0B", 2: "#EF4444"}

        # ── Transition probability matrix ─────────────────────────────────────
        trans = np.zeros((3, 3))
        for i in range(T_reg - 1):
            trans[regimes[i], regimes[i + 1]] += 1
        trans_norm = trans / np.maximum(trans.sum(axis=1, keepdims=True), 1)

        # Stationary distribution via power iteration
        pi = np.array([1/3, 1/3, 1/3])
        for _ in range(200):
            pi = pi @ trans_norm
        pi = pi / pi.sum()

        # ── Metrics ───────────────────────────────────────────────────────────
        cur_reg = regimes[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Regime",   regime_labels[cur_reg])
        c2.metric("Bull Probability",  f"{pi[0]*100:.1f}%")
        c3.metric("Neutral Prob",      f"{pi[1]*100:.1f}%")
        c4.metric("Bear/Vol Prob",     f"{pi[2]*100:.1f}%")

        # ── Regime time series ────────────────────────────────────────────────
        st.markdown("#### Regime Classification  —  Price with Regime Background")
        price_ser = closes[reg_t].values[-T_reg:]
        fig_reg = go.Figure()
        # Shade regimes as background spans
        i_start = 0
        for i in range(1, T_reg):
            if regimes[i] != regimes[i_start] or i == T_reg - 1:
                fig_reg.add_vrect(
                    x0=dates[i_start], x1=dates[i],
                    fillcolor=regime_cols[regimes[i_start]],
                    opacity=0.12, line_width=0,
                )
                i_start = i
        fig_reg.add_trace(go.Scatter(
            x=dates, y=price_ser, name=reg_t,
            line=dict(color="#E2E8F0", width=1.5),
        ))
        # Regime line at bottom
        fig_reg.add_trace(go.Scatter(
            x=dates, y=regimes * (price_ser.min() * 0.02),
            mode="markers", name="Regime",
            marker=dict(color=[regime_cols[r] for r in regimes], size=3, symbol="square"),
            showlegend=False,
        ))
        fig_reg.update_layout(
            title=f"{reg_t}  —  Price + Regime Classification  (Green=Bull · Amber=Neutral · Red=Bear)",
            height=380, **lay(),
        )
        fig_reg.update_xaxes(**ax())
        fig_reg.update_yaxes(title_text="Price", **ax())
        st.plotly_chart(fig_reg, use_container_width=True)

        # ── Transition matrix heatmap ─────────────────────────────────────────
        st.markdown("#### Transition Probability Matrix  P(Sₜ | Sₜ₋₁)")
        cl, cr = st.columns(2)
        with cl:
            fig_trans = go.Figure(go.Heatmap(
                z=trans_norm,
                x=[regime_labels[i] for i in range(3)],
                y=[regime_labels[i] for i in range(3)],
                colorscale=[[0, "#0D1829"], [0.5, "#1C3A5E"], [1, "#00D4FF"]],
                zmin=0, zmax=1,
                text=[[f"{trans_norm[i,j]:.3f}" for j in range(3)] for i in range(3)],
                texttemplate="%{text}",
                textfont=dict(size=12, family="JetBrains Mono"),
                colorbar=dict(title="Prob"),
            ))
            fig_trans.update_layout(
                title="Transition Matrix P",
                xaxis_title="Next Regime", yaxis_title="Current Regime",
                height=320, **lay(),
            )
            st.plotly_chart(fig_trans, use_container_width=True)

        with cr:
            # Regime-conditional statistics
            st.markdown("**Regime-Conditional Return Statistics**")
            for reg_id in range(3):
                mask = regimes == reg_id
                if mask.sum() > 5:
                    r_in  = ret_ser[mask]
                    st.markdown(
                        f"<div class='fbox'><strong style='color:{regime_cols[reg_id]}'>"
                        f"{regime_labels[reg_id]}</strong> &nbsp; "
                        f"n={mask.sum()} days &nbsp;|&nbsp; "
                        f"μ={r_in.mean()*252*100:.1f}%/yr &nbsp;|&nbsp; "
                        f"σ={r_in.std()*np.sqrt(252)*100:.1f}%/yr &nbsp;|&nbsp; "
                        f"π={pi[reg_id]*100:.1f}%</div>",
                        unsafe_allow_html=True,
                    )
            st.markdown("**Stationary Distribution π**")
            fig_pi = go.Figure(go.Bar(
                x=[regime_labels[i] for i in range(3)], y=pi * 100,
                marker=dict(color=[regime_cols[i] for i in range(3)], opacity=0.85),
                text=[f"{v*100:.1f}%" for v in pi], textposition="outside",
            ))
            fig_pi.update_layout(title="Long-Run Regime Probabilities",
                                  height=240, showlegend=False, **lay())
            fig_pi.update_xaxes(**ax()); fig_pi.update_yaxes(title_text="%", **ax())
            st.plotly_chart(fig_pi, use_container_width=True)

        # ── Regime-conditional correlations ───────────────────────────────────
        if N >= 2:
            st.markdown("#### Regime-Conditional Correlations  —  How relationships change across regimes")
            fig_rc2 = make_subplots(rows=1, cols=3,
                                    subplot_titles=[regime_labels[r] for r in range(3)])
            for ri in range(3):
                mask = regimes == ri
                if mask.sum() > 10:
                    R_reg = R[mask, :]
                    corr_r = np.corrcoef(R_reg.T)
                else:
                    corr_r = np.eye(N)
                fig_rc2.add_trace(go.Heatmap(
                    z=corr_r, x=selected, y=selected,
                    colorscale=[[0, "#EF4444"], [.5, "#0D1829"], [1, "#10B981"]],
                    zmid=0, zmin=-1, zmax=1, showscale=ri == 2,
                    text=[[f"{corr_r[a,b]:.2f}" for b in range(N)] for a in range(N)],
                    texttemplate="%{text}", textfont=dict(size=8),
                    colorbar=dict(title="ρ"),
                ), row=1, col=ri + 1)
            fig_rc2.update_layout(title="Correlations by Regime", height=400, **lay())
            fig_rc2.update_xaxes(**ax()); fig_rc2.update_yaxes(**ax())
            st.plotly_chart(fig_rc2, use_container_width=True)

    # 7D — MARKET EFFICIENCY (RANDOM WALK TESTS)
    with st7d:
        st.markdown("""
        <div class="fbox">
        <strong>Variance Ratio:</strong> VR(q) = Var(q-period ret) / [q·Var(1-period ret)]  → 1 if RW
        &nbsp;|&nbsp;
        <strong>Ljung-Box:</strong> Q = T(T+2)·Σ ρ²ₖ/(T−k) ~ χ²(h)
        &nbsp;|&nbsp;
        <strong>Runs Test:</strong> Z = (R − μR) / σR ~ N(0,1)
        </div>""", unsafe_allow_html=True)

        eff_t = st.selectbox("Stock for efficiency tests", selected, key="eff_t7")
        ret_e = R_df[eff_t].values

        # ── Variance ratio test ───────────────────────────────────────────────
        st.markdown("#### Variance Ratio Test  —  H₀: Returns follow a Random Walk")
        q_vals = [2, 4, 8, 16, 32]
        vr_vals, vr_z, vr_p = [], [], []
        n_e = len(ret_e)
        sig2_1 = ret_e.var(ddof=1)
        for q in q_vals:
            # multi-period returns
            ret_q   = np.array([ret_e[i:i+q].sum() for i in range(0, n_e - q + 1, q)])
            sig2_q  = ret_q.var(ddof=1)
            vr      = sig2_q / max(q * sig2_1, 1e-12)
            # Lo-MacKinlay asymptotic variance under homoscedasticity
            var_vr  = 2 * (2*q - 1) * (q - 1) / (3 * q * n_e)
            z_stat  = (vr - 1) / max(np.sqrt(var_vr), 1e-12)
            p_val   = 2 * (1 - sp_stats.norm.cdf(abs(z_stat)))
            vr_vals.append(vr); vr_z.append(z_stat); vr_p.append(p_val)

        fig_vr = go.Figure()
        color_vr = ["#EF4444" if p < 0.05 else "#10B981" for p in vr_p]
        fig_vr.add_trace(go.Bar(
            x=[str(q) for q in q_vals], y=vr_vals,
            name="VR(q)", marker=dict(color=color_vr, opacity=0.85),
            text=[f"{v:.4f}" for v in vr_vals], textposition="outside",
        ))
        fig_vr.add_hline(y=1.0, line=dict(color="#00D4FF", dash="dash", width=1.5),
                         annotation_text="RW (VR=1)")
        fig_vr.update_layout(title=f"{eff_t}  —  Variance Ratio  |  Green = Cannot reject RW  |  Red = Reject RW",
                              height=320, showlegend=False, **lay())
        fig_vr.update_xaxes(title_text="Holding Period q (days)", **ax())
        fig_vr.update_yaxes(title_text="VR(q)", **ax())
        st.plotly_chart(fig_vr, use_container_width=True)

        # ── VR results table ──────────────────────────────────────────────────
        vr_df = pd.DataFrame({
            "q (days)": q_vals,
            "VR(q)":    [f"{v:.4f}" for v in vr_vals],
            "Z-stat":   [f"{z:.3f}" for z in vr_z],
            "p-value":  [f"{p:.4f}" for p in vr_p],
            "Reject RW (5%)?": ["Yes ✗" if p < 0.05 else "No ✓" for p in vr_p],
            "Implication": [
                "Positive autocorrelation (momentum)" if v > 1.02
                else "Negative autocorrelation (mean reversion)" if v < 0.98
                else "Consistent with Random Walk"
                for v in vr_vals
            ],
        })
        st.dataframe(vr_df, use_container_width=True)

        # ── Ljung-Box autocorrelation test ────────────────────────────────────
        st.markdown("#### Ljung-Box Test  —  Autocorrelation in Returns  (H₀: all ρₖ = 0)")
        max_lags = min(20, T_days // 5)
        lags_lb  = list(range(1, max_lags + 1))
        acf_vals = [pd.Series(ret_e).autocorr(lag=k) for k in lags_lb]
        q_stats, p_lbs = [], []
        for h in lags_lb:
            q_s = n_e * (n_e + 2) * sum(acf_vals[k-1]**2 / (n_e - k) for k in range(1, h+1))
            q_stats.append(q_s)
            p_lbs.append(1 - sp_stats.chi2.cdf(q_s, df=h))

        lb_col, conf_up = "#F59E0B", 1.96 / np.sqrt(n_e)
        fig_acf = make_subplots(rows=1, cols=2,
                                subplot_titles=["ACF of Returns", "Ljung-Box Q-statistic"])
        fig_acf.add_trace(go.Bar(
            x=lags_lb, y=acf_vals, name="ACF",
            marker=dict(color=[
                "#EF4444" if abs(a) > conf_up else "#10B981" for a in acf_vals
            ], opacity=0.85),
        ), row=1, col=1)
        fig_acf.add_hline(y=conf_up,  line=dict(color="#00D4FF", dash="dot", width=1), row=1, col=1)
        fig_acf.add_hline(y=-conf_up, line=dict(color="#00D4FF", dash="dot", width=1), row=1, col=1)
        fig_acf.add_trace(go.Scatter(
            x=lags_lb, y=p_lbs, mode="lines+markers", name="p-value",
            line=dict(color="#F59E0B", width=1.8),
            marker=dict(color=["#EF4444" if p < 0.05 else "#10B981" for p in p_lbs], size=7),
        ), row=1, col=2)
        fig_acf.add_hline(y=0.05, line=dict(color="#EF4444", dash="dash", width=1),
                          annotation_text="α=0.05", row=1, col=2)
        fig_acf.update_layout(title=f"{eff_t}  —  Autocorrelation & Ljung-Box",
                               height=360, showlegend=False, **lay())
        fig_acf.update_xaxes(title_text="Lag", **ax())
        fig_acf.update_yaxes(**ax())
        st.plotly_chart(fig_acf, use_container_width=True)

        # ── Runs test ─────────────────────────────────────────────────────────
        st.markdown("#### Runs Test  —  Independence of Return Signs  (H₀: runs are random)")
        cl, cr = st.columns(2)
        with cl:
            signs  = (ret_e > 0).astype(int)
            n1, n0 = signs.sum(), (1 - signs).sum()
            runs   = 1 + ((signs[1:] != signs[:-1])).sum()
            mu_r   = (2 * n1 * n0) / max(n1 + n0, 1) + 1
            var_r  = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / max((n1+n0)**2 * (n1+n0-1), 1)
            z_runs = (runs - mu_r) / max(np.sqrt(var_r), 1e-12)
            p_runs = 2 * (1 - sp_stats.norm.cdf(abs(z_runs)))
            rej    = "Reject H₀" if p_runs < 0.05 else "Fail to Reject H₀"

            st.markdown(f"""
            <div class="fbox">
            <strong>Runs Test Results — {eff_t}</strong><br>
            Positive days (n₁): <strong style='color:#10B981'>{n1}</strong>
            &nbsp;|&nbsp; Negative days (n₀): <strong style='color:#EF4444'>{n0}</strong><br>
            Observed runs: <strong style='color:#00D4FF'>{runs}</strong>
            &nbsp;|&nbsp; Expected runs: <strong>{mu_r:.1f}</strong><br>
            Z-statistic: <strong style='color:#F59E0B'>{z_runs:.4f}</strong>
            &nbsp;|&nbsp; p-value: <strong>{p_runs:.4f}</strong><br>
            Decision: <strong style='color:{"#EF4444" if p_runs < 0.05 else "#10B981"}'>{rej}</strong>
            </div>""", unsafe_allow_html=True)

        with cr:
            # Summary efficiency scorecard across all stocks
            st.markdown("**Market Efficiency Scorecard — All Assets**")
            eff_rows = []
            for t_i in selected:
                r_i = R_df[t_i].values
                n_i = len(r_i)
                # VR(2)
                r2i   = np.array([r_i[j:j+2].sum() for j in range(0, n_i-1, 2)])
                vr2   = r2i.var(ddof=1) / max(2 * r_i.var(ddof=1), 1e-12)
                # LB lag-5 p-value
                acf5  = [pd.Series(r_i).autocorr(lag=k) for k in range(1, 6)]
                q5    = n_i * (n_i+2) * sum(a**2/(n_i-k) for k,a in enumerate(acf5,1))
                p_lb5 = 1 - sp_stats.chi2.cdf(q5, df=5)
                # Runs
                s_i   = (r_i > 0).astype(int)
                n1i, n0i = s_i.sum(), (1-s_i).sum()
                ri_r  = 1 + ((s_i[1:]!=s_i[:-1])).sum()
                mu_ri = (2*n1i*n0i)/max(n1i+n0i,1)+1
                var_ri= (2*n1i*n0i*(2*n1i*n0i-n1i-n0i))/max((n1i+n0i)**2*(n1i+n0i-1),1)
                z_ri  = (ri_r-mu_ri)/max(np.sqrt(var_ri),1e-12)
                p_ri  = 2*(1-sp_stats.norm.cdf(abs(z_ri)))
                score = sum([abs(vr2-1)<0.05, p_lb5>0.05, p_ri>0.05])
                eff_rows.append([t_i, f"{vr2:.3f}", f"{p_lb5:.3f}", f"{p_ri:.3f}",
                                  "★"*score + "☆"*(3-score)])
            eff_df = pd.DataFrame(eff_rows,
                                   columns=["Stock","VR(2)","LB p(5)","Runs p","Efficiency"])
            st.dataframe(eff_df, use_container_width=True)
            st.caption("★★★ = Consistent with EMH  ·  ★ = Strong evidence against RW")

    # 7E — STOPPING TIME ANALYSIS
    with st7e:
        st.markdown("""
        <div class="fbox">
        <strong>First Hitting Time:</strong> τ = inf{t ≥ 0 : Xₜ ∉ (−a, b)}
        &nbsp;|&nbsp;
        <strong>Drawdown:</strong> DD(t) = [max_{s≤t} P(s) − P(t)] / max_{s≤t} P(s)
        &nbsp;|&nbsp;
        <strong>Expected Stopping Time:</strong> E[τ] = a·b / (μ·(b−a)) for drift μ
        </div>""", unsafe_allow_html=True)

        stop_t  = st.selectbox("Stock for stopping time", selected, key="stop_t7")
        cl, cr  = st.columns([1, 3])
        with cl:
            sl_pct   = st.slider("Stop-loss level (%)", 2, 20, 8, 1, key="sl7")
            tp_pct   = st.slider("Take-profit level (%)", 2, 30, 15, 1, key="tp7")
            sim_runs = st.slider("Monte Carlo paths", 500, 5000, 2000, 500, key="sim7")

        # ── First hitting time simulation ─────────────────────────────────────
        mu_s   = float(R_df[stop_t].mean())
        sig_s  = float(R_df[stop_t].std())
        np.random.seed(42)
        horizon = 252
        hit_times_sl, hit_times_tp, outcomes = [], [], []
        for _ in range(sim_runs):
            log_path = np.cumsum(np.random.normal(mu_s, sig_s, horizon))
            cum_ret  = log_path  # log-return path
            hit_sl   = np.where(cum_ret <= -sl_pct / 100)[0]
            hit_tp   = np.where(cum_ret >= tp_pct / 100)[0]
            t_sl     = hit_sl[0] if len(hit_sl) else horizon
            t_tp     = hit_tp[0] if len(hit_tp) else horizon
            if t_sl < t_tp:
                hit_times_sl.append(t_sl)
                outcomes.append("stop-loss")
            elif t_tp <= t_sl:
                hit_times_tp.append(t_tp)
                outcomes.append("take-profit")
            else:
                outcomes.append("expire")

        sl_rate  = outcomes.count("stop-loss") / sim_runs
        tp_rate  = outcomes.count("take-profit") / sim_runs
        exp_rate = outcomes.count("expire") / sim_runs

        with cr:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Stop-Loss Hit Rate",  f"{sl_rate*100:.1f}%")
            c2.metric("Take-Profit Hit Rate", f"{tp_rate*100:.1f}%")
            c3.metric("Expire Rate",          f"{exp_rate*100:.1f}%")
            avg_sl = np.mean(hit_times_sl) if hit_times_sl else horizon
            c4.metric("Avg Days to SL",       f"{avg_sl:.0f}")

        st.markdown("#### First Hitting Time Distributions")
        fig_ht = make_subplots(rows=1, cols=2,
                               subplot_titles=["Stop-Loss Hit Times", "Take-Profit Hit Times"])
        if hit_times_sl:
            fig_ht.add_trace(go.Histogram(
                x=hit_times_sl, nbinsx=30, name="Stop-Loss",
                marker=dict(color="#EF4444", opacity=0.8),
            ), row=1, col=1)
        if hit_times_tp:
            fig_ht.add_trace(go.Histogram(
                x=hit_times_tp, nbinsx=30, name="Take-Profit",
                marker=dict(color="#10B981", opacity=0.8),
            ), row=1, col=2)
        fig_ht.update_layout(
            title=f"{stop_t}  —  Hitting Time Distributions  ({sim_runs:,} MC paths)",
            height=340, showlegend=False, **lay(),
        )
        fig_ht.update_xaxes(title_text="Days", **ax())
        fig_ht.update_yaxes(title_text="Frequency", **ax())
        st.plotly_chart(fig_ht, use_container_width=True)

        # ── Drawdown analysis ─────────────────────────────────────────────────
        st.markdown("#### Drawdown Analysis  —  Historical Drawdown Path")
        price_s   = closes[stop_t].values
        roll_max  = pd.Series(price_s).expanding().max().values
        drawdown  = (roll_max - price_s) / np.maximum(roll_max, 1e-6) * 100
        dd_dates  = closes.index

        fig_dd = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               vertical_spacing=0.04, row_heights=[0.55, 0.45])
        fig_dd.add_trace(go.Scatter(
            x=dd_dates, y=price_s, name=stop_t,
            line=dict(color="#00D4FF", width=1.5),
        ), row=1, col=1)
        fig_dd.add_trace(go.Scatter(
            x=dd_dates, y=roll_max, name="Peak",
            line=dict(color="#10B981", width=1, dash="dot"), opacity=0.7,
        ), row=1, col=1)
        fig_dd.add_trace(go.Scatter(
            x=dd_dates, y=-drawdown, name="Drawdown",
            fill="tozeroy", fillcolor="rgba(239,68,68,0.13)",
            line=dict(color="#EF4444", width=1.2),
        ), row=2, col=1)
        max_dd = drawdown.max()
        fig_dd.add_hline(y=-max_dd, line=dict(color="#EF4444", dash="dash", width=1),
                         annotation_text=f"Max DD={max_dd:.1f}%", row=2, col=1)
        fig_dd.update_layout(
            title=f"{stop_t}  —  Price & Drawdown  |  Max DD = {max_dd:.2f}%",
            height=420, **lay(),
        )
        fig_dd.update_xaxes(**ax())
        fig_dd.update_yaxes(title_text="Price", row=1, col=1, **ax())
        fig_dd.update_yaxes(title_text="Drawdown (%)", row=2, col=1, **ax())
        st.plotly_chart(fig_dd, use_container_width=True)

        # ── Stop-loss effectiveness ────────────────────────────────────────────
        st.markdown("#### Stop-Loss Effectiveness  —  Varying SL Level")
        sl_levels  = np.arange(2, 22, 2)
        sl_rates_v, avg_hit_v, exp_return_v = [], [], []
        np.random.seed(42)
        for sl_lv in sl_levels:
            sl_r, tp_r_v, hit_v = 0, 0, []
            for _ in range(1000):
                lp  = np.cumsum(np.random.normal(mu_s, sig_s, 252))
                hsl = np.where(lp <= -sl_lv/100)[0]
                htp = np.where(lp >= tp_pct/100)[0]
                t_s = hsl[0] if len(hsl) else 252
                t_t = htp[0] if len(htp) else 252
                if t_s < t_t:
                    sl_r += 1; hit_v.append(t_s)
                elif t_t < t_s:
                    tp_r_v += 1; hit_v.append(t_t)
            sl_rates_v.append(sl_r / 1000)
            avg_hit_v.append(np.mean(hit_v) if hit_v else 252)
            exp_return_v.append((tp_r_v * tp_pct - sl_r * sl_lv) / 10)

        fig_sl = make_subplots(rows=1, cols=3,
                               subplot_titles=["SL Hit Rate vs Level",
                                               "Avg Days to Exit",
                                               "Expected P&L (pts)"])
        colors3 = ["#EF4444", "#F59E0B", "#10B981"]
        for ci, (y_d, nm) in enumerate(zip(
            [sl_rates_v, avg_hit_v, exp_return_v],
            ["SL Rate", "Avg Days", "E[P&L]"]), 1):
            fig_sl.add_trace(go.Scatter(
                x=sl_levels, y=y_d, name=nm,
                mode="lines+markers",
                line=dict(color=colors3[ci-1], width=2),
                marker=dict(size=6), showlegend=False,
            ), row=1, col=ci)
        fig_sl.update_layout(title="Stop-Loss Effectiveness Analysis", height=300, **lay())
        fig_sl.update_xaxes(title_text="SL Level (%)", **ax())
        fig_sl.update_yaxes(**ax())
        st.plotly_chart(fig_sl, use_container_width=True)

    # 7F — MULTI-FACTOR MODEL COMPARISON
    with st7f:
        st.markdown("""
        <div class="fbox">
        <strong>PCA Factors:</strong> F = Rᶜ·V  (data-driven, maximise variance)
        &nbsp;|&nbsp;
        <strong>Fama-French:</strong> Rᵢ = αᵢ + βm·Rm + βs·SMB + βv·HML + εᵢ
        &nbsp;|&nbsp;
        <strong>Factor Premium:</strong> λₖ = E[Fₖ] − Rƒ  &nbsp;|&nbsp;
        <strong>Attribution:</strong> σ²(Rᵢ) = Σₖ βᵢₖ²·σ²ₖ + σ²(εᵢ)
        </div>""", unsafe_allow_html=True)

        # ── PCA factor extraction ─────────────────────────────────────────────
        pca_ff = PCA()
        pca_ff.fit(Rc)
        n_fac_ff = min(3, N)
        F_pca    = Rc @ pca_ff.components_[:n_fac_ff].T   # T × n_fac_ff
        evr_ff   = pca_ff.explained_variance_ratio_

        # Synthetic Fama-French-style factors from returns
        # MKT = equal-weight portfolio; SMB = bottom-half minus top-half by vol;
        # HML = high-vol minus low-vol (proxy in absence of book data)
        vols_sort = np.argsort(ind_vol)
        low_vol_idx  = vols_sort[:N//2]
        high_vol_idx = vols_sort[N//2:]
        F_mkt = R @ w_ew
        F_smb = R[:, low_vol_idx].mean(axis=1)  - R[:, high_vol_idx].mean(axis=1)   # low - high vol (size proxy)
        F_hml = R[:, high_vol_idx].mean(axis=1) - R[:, low_vol_idx].mean(axis=1)    # high - low vol (value proxy)
        F_ff  = np.stack([F_mkt, F_smb, F_hml], axis=1)

        factor_names_pca = [f"PCA-F{k+1} ({evr_ff[k]*100:.1f}%)" for k in range(n_fac_ff)]
        factor_names_ff  = ["MKT", "SMB (Low−High Vol)", "HML (High−Low Vol)"]

        # ── Factor correlation comparison ─────────────────────────────────────
        st.markdown("#### PCA Factors vs Fama-French Proxies  —  Correlation Matrix")
        all_factors = np.hstack([F_pca, F_ff])
        all_names   = factor_names_pca + factor_names_ff
        corr_ff     = np.corrcoef(all_factors.T)

        fig_corr_ff = go.Figure(go.Heatmap(
            z=corr_ff, x=all_names, y=all_names,
            colorscale=[[0, "#EF4444"], [.5, "#0D1829"], [1, "#10B981"]],
            zmid=0, zmin=-1, zmax=1,
            text=[[f"{corr_ff[i,j]:.2f}" for j in range(len(all_names))]
                  for i in range(len(all_names))],
            texttemplate="%{text}", textfont=dict(size=9),
            colorbar=dict(title="ρ"),
        ))
        fig_corr_ff.update_layout(
            title="PCA Factors vs FF Proxies — Correlation Structure",
            height=420, **lay(),
        )
        st.plotly_chart(fig_corr_ff, use_container_width=True)

        # ── Factor premium ────────────────────────────────────────────────────
        st.markdown("#### Factor Premiums  —  E[Fₖ] − Rƒ  (Annualised)")
        cl, cr = st.columns(2)
        with cl:
            prems_pca = [(F_pca[:, k].mean() * 252 - rf_rate) * 100 for k in range(n_fac_ff)]
            prems_ff  = [(F_ff[:, k].mean()  * 252 - rf_rate) * 100 for k in range(3)]
            fig_fp = go.Figure()
            fig_fp.add_trace(go.Bar(
                x=factor_names_pca, y=prems_pca, name="PCA Factors",
                marker=dict(color=["#00D4FF","#10B981","#8B5CF6"][:n_fac_ff], opacity=0.85),
                text=[f"{v:.2f}%" for v in prems_pca], textposition="outside",
            ))
            fig_fp.add_trace(go.Bar(
                x=factor_names_ff, y=prems_ff, name="FF Proxies",
                marker=dict(color=["#F59E0B","#EF4444","#F97316"], opacity=0.85),
                text=[f"{v:.2f}%" for v in prems_ff], textposition="outside",
            ))
            fig_fp.add_hline(y=0, line=dict(color="#475569", dash="dot", width=1))
            fig_fp.update_layout(title="Annualised Factor Risk Premiums",
                                  height=380, barmode="group", **lay())
            fig_fp.update_xaxes(**ax()); fig_fp.update_yaxes(title_text="Premium (%)", **ax())
            st.plotly_chart(fig_fp, use_container_width=True)

        with cr:
            # ── Factor attribution per asset ──────────────────────────────────
            st.markdown("**Factor Attribution (PCA 3-factor model)**")
            attr_rows = []
            for i, t in enumerate(selected):
                ri = R[:, i]
                # OLS of ri on F_pca
                X_reg = np.column_stack([np.ones(T_days), F_pca])
                try:
                    coef = np.linalg.lstsq(X_reg, ri, rcond=None)[0]
                    betas_i = coef[1:]
                    resid_i = ri - X_reg @ coef
                    # variance attribution
                    total_var = ri.var()
                    factor_var = sum(betas_i[k]**2 * F_pca[:, k].var() for k in range(n_fac_ff))
                    idio_var   = resid_i.var()
                    r2_attr    = factor_var / max(total_var, 1e-12)
                except Exception:
                    betas_i = np.zeros(n_fac_ff); r2_attr = 0.0
                attr_rows.append(
                    [t] +
                    [f"{b:.4f}" for b in betas_i] +
                    [f"{r2_attr*100:.1f}%"]
                )
            attr_df = pd.DataFrame(
                attr_rows,
                columns=["Stock"] + [f"β{k+1}" for k in range(n_fac_ff)] + ["R² (Factor)"]
            )
            st.dataframe(attr_df, use_container_width=True)

        # ── Factor time series comparison ─────────────────────────────────────
        st.markdown("#### Factor Time Series — PCA vs FF Proxies (Cumulative)")
        fig_fts = go.Figure()
        all_F = [(F_pca[:, k], factor_names_pca[k], PALETTE[k]) for k in range(n_fac_ff)] + \
                [(F_ff[:, k], factor_names_ff[k], PALETTE[k+3]) for k in range(3)]
        for fvals, fname, fcol in all_F:
            cum_f = pd.Series(np.cumsum(fvals), index=dates)
            fig_fts.add_trace(go.Scatter(
                x=cum_f.index, y=cum_f.values,
                name=fname, line=dict(color=fcol, width=1.4),
            ))
        fig_fts.add_hline(y=0, line=dict(color="#475569", dash="dot", width=1))
        fig_fts.update_layout(
            title="Cumulative Factor Returns  —  PCA Factors vs FF Proxies",
            height=380, **lay(),
        )
        fig_fts.update_xaxes(**ax())
        fig_fts.update_yaxes(title_text="Cumulative Log-Return", **ax())
        st.plotly_chart(fig_fts, use_container_width=True)

        # ── Factor attribution waterfall per stock ────────────────────────────
        st.markdown("#### Variance Attribution Waterfall  —  Systematic vs Idiosyncratic per Asset")
        sys_pcts, idio_pcts = [], []
        for i in range(N):
            ri = R[:, i]
            X_reg = np.column_stack([np.ones(T_days), F_pca])
            try:
                coef    = np.linalg.lstsq(X_reg, ri, rcond=None)[0]
                betas_i = coef[1:]
                resid_i = ri - X_reg @ coef
                total_v = ri.var()
                f_v     = sum(betas_i[k]**2 * F_pca[:, k].var() for k in range(n_fac_ff))
                sys_pcts.append(f_v / max(total_v, 1e-12) * 100)
                idio_pcts.append(resid_i.var() / max(total_v, 1e-12) * 100)
            except Exception:
                sys_pcts.append(50.0); idio_pcts.append(50.0)

        fig_attr = go.Figure()
        fig_attr.add_trace(go.Bar(
            x=selected, y=sys_pcts, name="Systematic (PCA 3F)",
            marker=dict(color="#00D4FF", opacity=0.85),
        ))
        fig_attr.add_trace(go.Bar(
            x=selected, y=idio_pcts, name="Idiosyncratic",
            marker=dict(color="#F59E0B", opacity=0.85),
        ))
        fig_attr.update_layout(
            barmode="stack", title="Variance Attribution: Systematic vs Idiosyncratic (%)",
            height=320, **lay(),
        )
        fig_attr.update_xaxes(**ax())
        fig_attr.update_yaxes(title_text="% of Total Variance", **ax())
        st.plotly_chart(fig_attr, use_container_width=True)


with tab8:
    st.markdown('<div class="sec-hdr">8. Regression Analysis — OLS · Diagnostics · Regularization · Quantile · Time Series</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="fbox">
    <strong>OLS:</strong> β̂ = (XᵀX)⁻¹Xᵀy
    &nbsp;|&nbsp;
    <strong>Ridge:</strong> β̂ = (XᵀX + λI)⁻¹Xᵀy
    &nbsp;|&nbsp;
    <strong>LASSO:</strong> min ||y−Xβ||² + λ||β||₁
    &nbsp;|&nbsp;
    <strong>Quantile:</strong> min Σ ρ_τ(yᵢ − xᵢᵀβ)
    &nbsp;|&nbsp;
    <strong>AR(p):</strong> yₜ = φ₀ + Σφₖyₜ₋ₖ + εₜ
    </div>""", unsafe_allow_html=True)

    from scipy import stats as sp_stats8

    st8a, st8b, st8c, st8d, st8e, st8f = st.tabs([
        "📐 OLS Fundamentals",
        "🔬 Diagnostics",
        "🌀 Poly & Fourier",
        "🔒 Regularization",
        "📊 Quantile Regression",
        "📈 Time Series AR",
    ])

    def ols_fit(X, y):
        """Return (beta, y_hat, resid, r2, adj_r2, f_stat, p_f, se)"""
        n, p = X.shape
        XtX_inv = np.linalg.pinv(X.T @ X)
        beta    = XtX_inv @ X.T @ y
        y_hat   = X @ beta
        resid   = y - y_hat
        ss_res  = (resid**2).sum()
        ss_tot  = ((y - y.mean())**2).sum()
        r2      = 1 - ss_res / max(ss_tot, 1e-12)
        adj_r2  = 1 - (1 - r2) * (n - 1) / max(n - p, 1)
        mse     = ss_res / max(n - p, 1)
        se_beta = np.sqrt(np.diag(XtX_inv) * mse)
        # F-stat: (R² / (p-1)) / ((1-R²) / (n-p))
        f_stat  = (r2 / max(p - 1, 1)) / max((1 - r2) / max(n - p, 1), 1e-12)
        p_f     = 1 - sp_stats8.f.cdf(f_stat, p - 1, n - p)
        return beta, y_hat, resid, r2, adj_r2, f_stat, p_f, se_beta, mse

    # 8A — OLS FUNDAMENTALS
    with st8a:
        st.markdown("""
        <div class="fbox">
        <strong>OLS Estimator:</strong> β̂ = (XᵀX)⁻¹Xᵀy &nbsp;|&nbsp;
        <strong>Fitted:</strong> ŷ = Xβ̂ &nbsp;|&nbsp;
        <strong>Residuals:</strong> e = y − ŷ &nbsp;|&nbsp;
        <strong>R²:</strong> 1 − RSS/TSS &nbsp;|&nbsp;
        <strong>Adj R²:</strong> 1 − (1−R²)(n−1)/(n−p)
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns([1, 2])
        with cl:
            dep_t    = st.selectbox("Dependent stock (y)", selected, key="ols_dep")
            pred_ts  = [t for t in selected if t != dep_t]
            indep_ts = st.multiselect("Independent stocks (X)", pred_ts,
                                      default=pred_ts[:min(3, len(pred_ts))], key="ols_ind")
            add_intercept = st.checkbox("Intercept", value=True, key="ols_int")
            lag_y     = st.slider("Lags of y in X", 0, 5, 1, key="ols_lag")

        if len(indep_ts) > 0:
            # Build design matrix
            min_len = min(T_days, *[len(R_df[t]) for t in indep_ts + [dep_t]])
            y_s  = R_df[dep_t].values[-min_len:]
            X_s  = np.column_stack([R_df[t].values[-min_len:] for t in indep_ts])
            if lag_y > 0:
                ys_lagged = np.column_stack([R_df[dep_t].shift(k+1).dropna().values[-min_len:] for k in range(lag_y)])
                trim = lag_y
                y_s  = y_s[trim:]
                X_s  = np.column_stack([X_s[trim:], ys_lagged[-len(y_s):]])
            if add_intercept:
                X_s = np.column_stack([np.ones(len(y_s)), X_s])

            beta, y_hat, resid, r2, adj_r2, f_stat, p_f, se_beta, mse = ols_fit(X_s, y_s)
            t_stats = beta / np.maximum(se_beta, 1e-12)
            p_vals  = 2 * (1 - sp_stats8.t.cdf(np.abs(t_stats), df=len(y_s) - X_s.shape[1]))

            with cr:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R²",        f"{r2:.4f}")
                m2.metric("Adj R²",    f"{adj_r2:.4f}")
                m3.metric("F-stat",    f"{f_stat:.2f}")
                m4.metric("p(F)",      f"{p_f:.4f}")

            # OLS coefficient table
            feat_names = (["Intercept"] if add_intercept else []) + indep_ts + [f"y_lag{k+1}" for k in range(lag_y)]
            ols_df = pd.DataFrame({
                "Feature": feat_names,
                "β̂":       [f"{b:.6f}" for b in beta],
                "SE":      [f"{s:.6f}" for s in se_beta],
                "t-stat":  [f"{t:.3f}" for t in t_stats],
                "p-value": [f"{p:.4f}" for p in p_vals],
                "Sig":     ["***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "" for p in p_vals],
            })
            st.markdown("#### OLS Coefficient Table  —  β̂ = (XᵀX)⁻¹Xᵀy")
            st.dataframe(ols_df, use_container_width=True)

            # Fitted vs actual + residual panels
            idx_s = R_df.index[-len(y_s):]
            fig_ols = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                    subplot_titles=["Actual vs Fitted Returns", "Residuals"])
            fig_ols.add_trace(go.Scatter(x=idx_s, y=y_s*100, name="Actual",
                line=dict(color="#00D4FF", width=1.4)), row=1, col=1)
            fig_ols.add_trace(go.Scatter(x=idx_s, y=y_hat*100, name="Fitted",
                line=dict(color="#F59E0B", width=1.4, dash="dash")), row=1, col=1)
            fig_ols.add_trace(go.Bar(x=idx_s, y=resid*100, name="Residuals",
                marker=dict(color=["#10B981" if r >= 0 else "#EF4444" for r in resid], opacity=0.7)), row=2, col=1)
            fig_ols.add_hline(y=0, line=dict(color="#475569", dash="dot", width=1), row=2, col=1)
            fig_ols.update_layout(title=f"OLS: {dep_t} ~ {' + '.join(indep_ts[:3])}{'...' if len(indep_ts)>3 else ''}",
                                  height=460, **lay())
            fig_ols.update_xaxes(**ax()); fig_ols.update_yaxes(**ax())
            st.plotly_chart(fig_ols, use_container_width=True)

            # Scatter: actual vs fitted
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(x=y_hat*100, y=y_s*100, mode="markers",
                marker=dict(color="#8B5CF6", size=4, opacity=0.6),
                name="Obs", hovertemplate="Fitted: %{x:.3f}%<br>Actual: %{y:.3f}%<extra></extra>"))
            lo_v, hi_v = y_hat.min()*100, y_hat.max()*100
            fig_sc.add_trace(go.Scatter(x=[lo_v, hi_v], y=[lo_v, hi_v], mode="lines",
                line=dict(color="#EF4444", dash="dash"), name="45° Line"))
            fig_sc.update_layout(title="Actual vs Fitted  (R²={:.4f})".format(r2),
                                  height=340, **lay())
            fig_sc.update_xaxes(title_text="Fitted (%)", **ax())
            fig_sc.update_yaxes(title_text="Actual (%)", **ax())
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("Select at least one independent stock.")

    # 8B — REGRESSION DIAGNOSTICS
    with st8b:
        st.markdown("""
        <div class="fbox">
        <strong>Leverage:</strong> hᵢᵢ = [X(XᵀX)⁻¹Xᵀ]ᵢᵢ &nbsp;|&nbsp;
        <strong>Cook's D:</strong> Dᵢ = (eᵢ²/p·MSE)·(hᵢᵢ/(1−hᵢᵢ)²) &nbsp;|&nbsp;
        <strong>Studentized Resid:</strong> rᵢ = eᵢ/(σ̂√(1−hᵢᵢ))
        </div>""", unsafe_allow_html=True)

        diag_t = st.selectbox("Stock for diagnostics", selected, key="diag_t8")
        other_t = [t for t in selected if t != diag_t]
        if len(other_t) > 0:
            y_d   = R_df[diag_t].values
            X_d   = np.column_stack([np.ones(T_days)] + [R_df[t].values for t in other_t[:min(4, len(other_t))]])
            n_d, p_d = X_d.shape
            XtX_inv_d = np.linalg.pinv(X_d.T @ X_d)
            beta_d, y_hat_d, resid_d, r2_d, _, _, _, _, mse_d = ols_fit(X_d, y_d)

            # Hat matrix diagonal (leverage)
            H_diag   = np.einsum("ij,jk,ik->i", X_d, XtX_inv_d, X_d)
            H_diag   = np.clip(H_diag, 0, 0.9999)

            # Studentized residuals
            stud_r   = resid_d / np.maximum(np.sqrt(mse_d * (1 - H_diag)), 1e-12)

            # Cook's distance
            cook_d   = (resid_d**2 / max(p_d * mse_d, 1e-12)) * (H_diag / np.maximum((1 - H_diag)**2, 1e-12))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("R²",           f"{r2_d:.4f}")
            m2.metric("Max Leverage", f"{H_diag.max():.4f}")
            m3.metric("Max |Stud R|", f"{np.abs(stud_r).max():.2f}")
            m4.metric("Max Cook's D", f"{cook_d.max():.4f}")

            fig_diag = make_subplots(rows=2, cols=2,
                subplot_titles=["QQ Plot — Normality of Residuals",
                                 "Residuals vs Fitted — Homoscedasticity",
                                 "Leverage (Hat Values)",
                                 "Influence Plot: Stud Resid vs Leverage"])

            # QQ plot
            sorted_r = np.sort(resid_d)
            n_r      = len(sorted_r)
            qq_theoretical = sp_stats8.norm.ppf([(i - 0.375) / (n_r + 0.25) for i in range(1, n_r + 1)])
            fig_diag.add_trace(go.Scatter(x=qq_theoretical, y=sorted_r, mode="markers",
                marker=dict(color="#00D4FF", size=3, opacity=0.7), name="QQ"), row=1, col=1)
            qq_lo, qq_hi = qq_theoretical[[0, -1]]
            fig_diag.add_trace(go.Scatter(x=[qq_lo, qq_hi],
                y=[sorted_r[0] + qq_lo*(sorted_r[-1]-sorted_r[0])/(qq_hi-qq_lo+1e-9),
                   sorted_r[-1]],
                mode="lines", line=dict(color="#EF4444", dash="dash"), name="Normal"), row=1, col=1)

            # Residuals vs fitted
            fig_diag.add_trace(go.Scatter(x=y_hat_d*100, y=resid_d*100, mode="markers",
                marker=dict(color="#10B981", size=3, opacity=0.6), name="Resid"), row=1, col=2)
            fig_diag.add_hline(y=0, line=dict(color="#F59E0B", dash="dot"), row=1, col=2)

            # Leverage
            fig_diag.add_trace(go.Scatter(x=list(range(n_d)), y=H_diag, mode="markers",
                marker=dict(color="#F59E0B", size=3, opacity=0.7), name="Leverage"), row=2, col=1)
            fig_diag.add_hline(y=2*p_d/n_d, line=dict(color="#EF4444", dash="dash"),
                               annotation_text=f"2p/n={2*p_d/n_d:.4f}", row=2, col=1)

            # Influence plot
            cook_clr = ["#EF4444" if d > 4/n_d else "#8B5CF6" for d in cook_d]
            fig_diag.add_trace(go.Scatter(x=H_diag, y=stud_r, mode="markers",
                marker=dict(color=cook_clr, size=4, opacity=0.7), name="Influence"), row=2, col=2)
            fig_diag.add_hline(y=2,  line=dict(color="#EF4444", dash="dot", width=1), row=2, col=2)
            fig_diag.add_hline(y=-2, line=dict(color="#EF4444", dash="dot", width=1), row=2, col=2)
            fig_diag.add_vline(x=2*p_d/n_d, line=dict(color="#F59E0B", dash="dot", width=1), row=2, col=2)

            fig_diag.update_layout(title=f"Regression Diagnostics — {diag_t}",
                                   height=640, showlegend=False, **lay())
            fig_diag.update_xaxes(**ax()); fig_diag.update_yaxes(**ax())
            st.plotly_chart(fig_diag, use_container_width=True)

            # Cook's distance bar
            st.markdown("#### Cook's Distance  —  Influential Observations")
            cook_colors = ["#EF4444" if d > 4/n_d else "#00D4FF" for d in cook_d]
            fig_ck = go.Figure(go.Bar(x=list(range(n_d)), y=cook_d, marker_color=cook_colors,
                                       marker_line_width=0, opacity=0.85))
            fig_ck.add_hline(y=4/n_d, line=dict(color="#EF4444", dash="dash"),
                              annotation_text=f"Threshold 4/n={4/n_d:.5f}")
            fig_ck.update_layout(title="Cook's Distance  |  Red = Influential (Dᵢ > 4/n)",
                                  height=280, showlegend=False, **lay())
            fig_ck.update_xaxes(title_text="Observation Index", **ax())
            fig_ck.update_yaxes(title_text="Cook's Dᵢ", **ax())
            st.plotly_chart(fig_ck, use_container_width=True)

    # 8C — POLYNOMIAL & FOURIER REGRESSION
    with st8c:
        st.markdown("""
        <div class="fbox">
        <strong>Polynomial:</strong> y = β₀ + β₁x + β₂x² + β₃x³ &nbsp;|&nbsp;
        <strong>Fourier:</strong> y = β₀ + Σₖ[βₖsin(2πkx/T) + γₖcos(2πkx/T)] &nbsp;|&nbsp;
        <strong>AIC:</strong> −2ℓ + 2p &nbsp;|&nbsp;
        <strong>BIC:</strong> −2ℓ + p·ln(n)
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns([1, 3])
        with cl:
            pf_t      = st.selectbox("Stock", selected, key="pf_t8")
            poly_deg  = st.slider("Polynomial degree", 1, 8, 3, key="poly_deg8")
            four_harm = st.slider("Fourier harmonics", 1, 10, 4, key="four_h8")

        y_pf   = R_df[pf_t].values
        x_pf   = np.arange(len(y_pf)) / len(y_pf)   # normalise to [0,1]
        n_pf   = len(y_pf)

        # build poly and Fourier design matrices
        X_poly = np.column_stack([np.ones(n_pf)] + [x_pf**k for k in range(1, poly_deg+1)])
        X_four = np.column_stack(
            [np.ones(n_pf)] +
            [np.sin(2*np.pi*k*x_pf) for k in range(1, four_harm+1)] +
            [np.cos(2*np.pi*k*x_pf) for k in range(1, four_harm+1)]
        )

        def aic_bic(resid, p, n):
            ss  = (resid**2).sum()
            ll  = -n/2 * np.log(max(ss/n, 1e-12)) - n/2  # Gaussian log-likelihood
            return -2*ll + 2*p, -2*ll + p*np.log(n)

        _, y_hat_poly, r_poly, r2_p, adj_p, _, _, _, mse_p = ols_fit(X_poly, y_pf)
        _, y_hat_four, r_four, r2_f, adj_f, _, _, _, mse_f = ols_fit(X_four, y_pf)
        aic_p, bic_p = aic_bic(r_poly, poly_deg+1, n_pf)
        aic_f, bic_f = aic_bic(r_four, 1+2*four_harm, n_pf)

        with cr:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Poly R²",    f"{r2_p:.4f}")
            c2.metric("Poly AIC",   f"{aic_p:.1f}")
            c3.metric("Fourier R²", f"{r2_f:.4f}")
            c4.metric("Fourier AIC",f"{aic_f:.1f}")

        fig_pf = go.Figure()
        fig_pf.add_trace(go.Scatter(x=dates, y=y_pf*100, mode="lines", name="Actual",
            line=dict(color="#00D4FF", width=1.2), opacity=0.6))
        fig_pf.add_trace(go.Scatter(x=dates, y=y_hat_poly*100, mode="lines",
            name=f"Polynomial deg={poly_deg}",
            line=dict(color="#F59E0B", width=2)))
        fig_pf.add_trace(go.Scatter(x=dates, y=y_hat_four*100, mode="lines",
            name=f"Fourier h={four_harm}",
            line=dict(color="#10B981", width=2, dash="dash")))
        fig_pf.update_layout(title=f"{pf_t}  —  Polynomial vs Fourier Regression",
                              height=380, **lay())
        fig_pf.update_xaxes(**ax()); fig_pf.update_yaxes(title_text="Return (%)", **ax())
        st.plotly_chart(fig_pf, use_container_width=True)

        # AIC/BIC comparison across polynomial degrees
        st.markdown("#### Model Selection — AIC & BIC vs Polynomial Degree")
        aic_arr, bic_arr = [], []
        for d in range(1, 12):
            X_tmp = np.column_stack([np.ones(n_pf)] + [x_pf**k for k in range(1, d+1)])
            _, _, r_tmp, _, _, _, _, _, _ = ols_fit(X_tmp, y_pf)
            a, b = aic_bic(r_tmp, d+1, n_pf)
            aic_arr.append(a); bic_arr.append(b)
        fig_sel = go.Figure()
        fig_sel.add_trace(go.Scatter(x=list(range(1,12)), y=aic_arr, mode="lines+markers",
            name="AIC", line=dict(color="#F59E0B", width=2)))
        fig_sel.add_trace(go.Scatter(x=list(range(1,12)), y=bic_arr, mode="lines+markers",
            name="BIC", line=dict(color="#8B5CF6", width=2)))
        best_aic = int(np.argmin(aic_arr)) + 1
        fig_sel.add_vline(x=best_aic, line=dict(color="#10B981", dash="dash"),
                          annotation_text=f"Best AIC deg={best_aic}")
        fig_sel.update_layout(title="Model Selection Criteria (lower = better)",
                               height=320, **lay())
        fig_sel.update_xaxes(title_text="Polynomial Degree", dtick=1, **ax())
        fig_sel.update_yaxes(title_text="Criterion Value", **ax())
        st.plotly_chart(fig_sel, use_container_width=True)

    # 8D — REGULARISED REGRESSION
    with st8d:
        st.markdown("""
        <div class="fbox">
        <strong>Ridge L2:</strong> β̂ = (XᵀX + λI)⁻¹Xᵀy &nbsp;|&nbsp;
        <strong>LASSO L1:</strong> min ||y−Xβ||² + λ||β||₁ &nbsp;|&nbsp;
        <strong>Elastic Net:</strong> α·LASSO + (1−α)·Ridge &nbsp;|&nbsp;
        <strong>CV λ:</strong> k-fold cross-validation on MSE
        </div>""", unsafe_allow_html=True)

        from sklearn.linear_model import Ridge, Lasso, ElasticNet
        from sklearn.model_selection import cross_val_score

        cl, cr = st.columns([1, 2])
        with cl:
            reg_dep = st.selectbox("Dependent (y)", selected, key="reg_dep8")
            reg_ind = [t for t in selected if t != reg_dep]
            en_alpha = st.slider("Elastic Net α (0=Ridge, 1=LASSO)", 0.0, 1.0, 0.5, 0.05, key="en_a8")
            n_lam    = 40
            lambdas  = np.logspace(-5, 1, n_lam)

        y_r = R_df[reg_dep].values
        X_r = StandardScaler().fit_transform(
            np.column_stack([R_df[t].values for t in reg_ind])
        )

        # coefficient paths
        ridge_coefs = np.array([(np.linalg.pinv(X_r.T@X_r + lam*np.eye(X_r.shape[1]))@X_r.T@y_r) for lam in lambdas])
        lasso_coefs, enet_coefs = [], []
        for lam in lambdas:
            try:
                lasso_coefs.append(Lasso(alpha=lam, max_iter=5000, tol=1e-4).fit(X_r, y_r).coef_)
                enet_coefs.append(ElasticNet(alpha=lam, l1_ratio=en_alpha, max_iter=5000, tol=1e-4).fit(X_r, y_r).coef_)
            except Exception:
                lasso_coefs.append(np.zeros(len(reg_ind)))
                enet_coefs.append(np.zeros(len(reg_ind)))
        lasso_coefs = np.array(lasso_coefs)
        enet_coefs  = np.array(enet_coefs)

        # CV for optimal lambda (Ridge)
        cv_scores = []
        for lam in lambdas:
            try:
                sc = cross_val_score(Ridge(alpha=lam), X_r, y_r, cv=5,
                                     scoring="neg_mean_squared_error").mean()
            except Exception:
                sc = -np.inf
            cv_scores.append(-sc)
        best_lam_idx = int(np.argmin(cv_scores))
        best_lam     = lambdas[best_lam_idx]

        with cr:
            c1, c2, c3 = st.columns(3)
            c1.metric("Best Ridge λ (CV)", f"{best_lam:.5f}")
            c2.metric("CV MSE at λ*",     f"{cv_scores[best_lam_idx]:.6f}")
            ridge_r2 = 1 - ((y_r - X_r@ridge_coefs[best_lam_idx])**2).sum() / max(((y_r-y_r.mean())**2).sum(), 1e-12)
            c3.metric("Ridge R² at λ*",   f"{ridge_r2:.4f}")

        fig_path = make_subplots(rows=1, cols=3,
                                  subplot_titles=["Ridge Paths", "LASSO Paths", "Elastic Net Paths"])
        for i, t in enumerate(reg_ind[:min(len(reg_ind), 6)]):
            c = PALETTE[i % len(PALETTE)]
            for col_i, coef_mat in enumerate([ridge_coefs, lasso_coefs, enet_coefs], 1):
                fig_path.add_trace(go.Scatter(x=np.log10(lambdas), y=coef_mat[:, i],
                    name=t, line=dict(color=c, width=1.4),
                    showlegend=(col_i == 1)), row=1, col=col_i)
        fig_path.add_vline(x=np.log10(best_lam), line=dict(color="#EF4444", dash="dash"),
                           row=1, col=1, annotation_text="CV λ*")
        fig_path.update_layout(title=f"Regularisation Coefficient Paths — {reg_dep}",
                                height=380, **lay())
        fig_path.update_xaxes(title_text="log₁₀(λ)", **ax())
        fig_path.update_yaxes(title_text="Coefficient", **ax())
        st.plotly_chart(fig_path, use_container_width=True)

        # CV error plot
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Scatter(x=np.log10(lambdas), y=cv_scores, mode="lines",
            line=dict(color="#00D4FF", width=2), name="CV MSE"))
        fig_cv.add_vline(x=np.log10(best_lam), line=dict(color="#EF4444", dash="dash"),
                         annotation_text=f"λ*={best_lam:.5f}")
        fig_cv.update_layout(title="Ridge  —  5-Fold Cross-Validation MSE vs log₁₀(λ)",
                              height=300, **lay())
        fig_cv.update_xaxes(title_text="log₁₀(λ)", **ax())
        fig_cv.update_yaxes(title_text="CV MSE", **ax())
        st.plotly_chart(fig_cv, use_container_width=True)

    # 8E — QUANTILE REGRESSION
    with st8e:
        st.markdown("""
        <div class="fbox">
        <strong>Quantile Loss:</strong> ρ_τ(u) = u·(τ − 𝟙{u<0}) &nbsp;|&nbsp;
        <strong>Quantile Reg:</strong> β̂_τ = argmin Σ ρ_τ(yᵢ − xᵢᵀβ) &nbsp;|&nbsp;
        <strong>VaR_τ:</strong> 5th/1st percentile of conditional distribution &nbsp;|&nbsp;
        <strong>CVaR:</strong> E[loss | loss > VaR]
        </div>""", unsafe_allow_html=True)

        from scipy.optimize import linprog

        cl, cr = st.columns([1, 2])
        with cl:
            qr_t  = st.selectbox("Stock", selected, key="qr_t8")
            taus  = st.multiselect("Quantiles τ", [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
                                   default=[0.05, 0.25, 0.50, 0.75, 0.95], key="qr_tau")

        y_q   = R_df[qr_t].values
        other_qr = [t for t in selected if t != qr_t]
        x_q   = R_df[other_qr[0]].values if other_qr else np.arange(len(y_q)) / len(y_q)
        n_q   = len(y_q)

        def quantile_reg_simple(x, y, tau):
            """Simple quantile regression via linear programming (one predictor)."""
            n = len(y)
            X_qr = np.column_stack([np.ones(n), x])
            p    = X_qr.shape[1]
            # LP formulation: min tau*sum(u) + (1-tau)*sum(v) s.t. y - Xb = u - v, u,v >= 0
            c_lp  = np.concatenate([np.zeros(p), tau*np.ones(n), (1-tau)*np.ones(n)])
            A_eq  = np.concatenate([X_qr, np.eye(n), -np.eye(n)], axis=1)
            b_eq  = y
            bounds = [(None, None)]*p + [(0, None)]*(2*n)
            try:
                res = linprog(c_lp, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
                return res.x[:p] if res.success else np.zeros(p)
            except Exception:
                return np.zeros(p)

        with cr:
            # VaR and CVaR metrics
            q95    = float(np.percentile(y_q, 5))
            q99    = float(np.percentile(y_q, 1))
            cvar95 = float(y_q[y_q <= q95].mean()) if (y_q <= q95).any() else q95
            cvar99 = float(y_q[y_q <= q99].mean()) if (y_q <= q99).any() else q99
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("VaR 95%",  f"{q95*100:.3f}%")
            c2.metric("CVaR 95%", f"{cvar95*100:.3f}%")
            c3.metric("VaR 99%",  f"{q99*100:.3f}%")
            c4.metric("CVaR 99%", f"{cvar99*100:.3f}%")

        # Quantile regression fits
        x_sort = np.sort(x_q)
        fig_qr = go.Figure()
        fig_qr.add_trace(go.Scatter(x=x_q*100, y=y_q*100, mode="markers",
            marker=dict(color="#475569", size=2.5, opacity=0.5), name="Data"))
        tau_colors = ["#EF4444","#F97316","#F59E0B","#10B981","#06B6D4","#8B5CF6","#EC4899"]
        for i, tau in enumerate(taus):
            beta_q = quantile_reg_simple(x_q, y_q, tau)
            y_qfit = beta_q[0] + beta_q[1]*x_sort
            fig_qr.add_trace(go.Scatter(x=x_sort*100, y=y_qfit*100, mode="lines",
                name=f"τ={tau}", line=dict(color=tau_colors[i%len(tau_colors)], width=1.8)))
        fig_qr.update_layout(title=f"{qr_t}  —  Quantile Regression (y={qr_t}, x={other_qr[0] if other_qr else 'time'})",
                              height=380, **lay())
        fig_qr.update_xaxes(title_text=f"{'x: '+other_qr[0] if other_qr else 'time'} Return (%)", **ax())
        fig_qr.update_yaxes(title_text=f"{qr_t} Return (%)", **ax())
        st.plotly_chart(fig_qr, use_container_width=True)

        # Return distribution with VaR lines
        fig_var = go.Figure()
        fig_var.add_trace(go.Histogram(x=y_q*100, nbinsx=80, name=qr_t,
            marker_color="#00D4FF", opacity=0.7))
        fig_var.add_vline(x=q95*100, line=dict(color="#F59E0B", dash="dash", width=2),
                          annotation_text=f"VaR95={q95*100:.2f}%")
        fig_var.add_vline(x=q99*100, line=dict(color="#EF4444", dash="dash", width=2),
                          annotation_text=f"VaR99={q99*100:.2f}%")
        fig_var.add_vline(x=cvar95*100, line=dict(color="#F59E0B", dash="dot", width=1.5),
                          annotation_text=f"CVaR95={cvar95*100:.2f}%")
        fig_var.update_layout(title=f"{qr_t}  —  Return Distribution with VaR & CVaR",
                               height=300, **lay())
        fig_var.update_xaxes(title_text="Daily Return (%)", **ax())
        fig_var.update_yaxes(title_text="Count", **ax())
        st.plotly_chart(fig_var, use_container_width=True)

    # 8F — TIME SERIES REGRESSION
    with st8f:
        st.markdown("""
        <div class="fbox">
        <strong>AR(p):</strong> yₜ = φ₀ + φ₁yₜ₋₁ + ... + φₚyₜ₋ₚ + εₜ &nbsp;|&nbsp;
        <strong>ADL(p,q):</strong> yₜ = α + Σφₖyₜ₋ₖ + Σβⱼxₜ₋ⱼ + εₜ &nbsp;|&nbsp;
        <strong>Forecast:</strong> ŷₜ₊ₕ = φ₀ + Σφₖŷₜ₊ₕ₋ₖ (h-step ahead)
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns([1, 2])
        with cl:
            ar_t    = st.selectbox("Stock", selected, key="ar_t8")
            ar_p    = st.slider("AR order p", 1, 10, 2, key="ar_p8")
            ar_h    = st.slider("Forecast horizon h", 1, 20, 10, key="ar_h8")
            use_adl = st.checkbox("ADL (include exogenous)", value=False, key="adl8")

        y_ar = R_df[ar_t].values
        n_ar = len(y_ar)

        # Build AR(p) design matrix
        Y_ar = y_ar[ar_p:]
        X_ar = np.column_stack([np.ones(len(Y_ar))] + [y_ar[ar_p-k-1:n_ar-k-1] for k in range(ar_p)])
        if use_adl:
            exog_t = [t for t in selected if t != ar_t]
            if exog_t:
                x_exog = R_df[exog_t[0]].values
                X_ar   = np.column_stack([X_ar, x_exog[ar_p:]])

        beta_ar, y_hat_ar, resid_ar, r2_ar, adj_ar, f_ar, pf_ar, se_ar, mse_ar = ols_fit(X_ar, Y_ar)

        # h-step ahead forecast using recursive substitution
        history = list(y_ar[-ar_p:])
        forecast, ci_lo, ci_hi = [], [], []
        for h in range(ar_h):
            x_fc = [1.0] + history[-ar_p:][::-1]
            if use_adl and exog_t:
                x_fc = x_fc + [0.0]  # exog set to 0 for future
            yf = float(np.dot(beta_ar[:len(x_fc)], x_fc))
            resid_std = float(np.sqrt(mse_ar)) * np.sqrt(1 + h)  # widening CI
            forecast.append(yf)
            ci_lo.append(yf - 1.96 * resid_std)
            ci_hi.append(yf + 1.96 * resid_std)
            history.append(yf)

        with cr:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"AR({ar_p}) R²",     f"{r2_ar:.4f}")
            c2.metric("Adj R²",             f"{adj_ar:.4f}")
            c3.metric("F-stat",             f"{f_ar:.2f}")
            c4.metric("RMSE",               f"{np.sqrt(mse_ar)*100:.4f}%")

        # In-sample fit + forecast
        idx_ar   = R_df.index[ar_p:]
        bday_off = pd.tseries.offsets.BusinessDay(1)
        fc_dates = [dates[-1] + bday_off*(i+1) for i in range(ar_h)]

        fig_ar = go.Figure()
        fig_ar.add_trace(go.Scatter(x=idx_ar, y=Y_ar*100, name="Actual",
            line=dict(color="#00D4FF", width=1.3), opacity=0.8))
        fig_ar.add_trace(go.Scatter(x=idx_ar, y=y_hat_ar*100, name=f"AR({ar_p}) Fit",
            line=dict(color="#F59E0B", width=1.5, dash="dash")))
        fig_ar.add_trace(go.Scatter(x=fc_dates, y=[f*100 for f in forecast],
            name="Forecast", mode="lines+markers",
            line=dict(color="#10B981", width=2, dash="dot"),
            marker=dict(size=6, color="#10B981", line=dict(width=1.5, color="white"))))
        fig_ar.add_trace(go.Scatter(
            x=fc_dates + fc_dates[::-1],
            y=[f*100 for f in ci_hi] + [f*100 for f in ci_lo[::-1]],
            fill="toself", fillcolor="rgba(16,185,129,0.1)",
            line=dict(width=0), name="95% CI", showlegend=True))
        fig_ar.add_vline(x=str(dates[-1].date()), line=dict(color="#475569", dash="dot"))
        fig_ar.update_layout(title=f"{ar_t}  —  AR({ar_p}) Fit + {ar_h}-Step Forecast",
                              height=440, **lay())
        fig_ar.update_xaxes(**ax()); fig_ar.update_yaxes(title_text="Return (%)", **ax())
        st.plotly_chart(fig_ar, use_container_width=True)

        # ACF/PACF of residuals
        st.markdown("#### Residual ACF  —  Model Adequacy Check")
        max_lag_ar = min(20, len(resid_ar)//4)
        acf_ar = [pd.Series(resid_ar).autocorr(lag=k) for k in range(1, max_lag_ar+1)]
        conf_ar = 1.96 / np.sqrt(len(resid_ar))
        fig_acf = go.Figure()
        fig_acf.add_trace(go.Bar(x=list(range(1, max_lag_ar+1)), y=acf_ar,
            marker_color=["#EF4444" if abs(a) > conf_ar else "#00D4FF" for a in acf_ar],
            opacity=0.85, name="ACF"))
        fig_acf.add_hline(y=conf_ar,  line=dict(color="#475569", dash="dot"))
        fig_acf.add_hline(y=-conf_ar, line=dict(color="#475569", dash="dot"))
        fig_acf.update_layout(title=f"AR({ar_p}) Residual ACF  |  Red = significant (p<0.05)",
                               height=280, showlegend=False, **lay())
        fig_acf.update_xaxes(title_text="Lag", **ax())
        fig_acf.update_yaxes(title_text="ACF", **ax())
        st.plotly_chart(fig_acf, use_container_width=True)


with tab9:
    st.markdown('<div class="sec-hdr">9. Yield Curve & Interest Rates — Bond Valuation · Swaps · Bootstrapping · PnL Attribution</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="fbox">
    <strong>Bond Price:</strong> P = Σ C/(1+y/f)ⁱ + 100/(1+y/f)ⁿ &nbsp;|&nbsp;
    <strong>Duration:</strong> D = Σ tᵢ·PV(Cᵢ)/P &nbsp;|&nbsp;
    <strong>Par Swap Rate:</strong> C = Σf·Δ·Z(t) / ΣΔ·Z(t) &nbsp;|&nbsp;
    <strong>DV01:</strong> ∂PV/∂r × 0.0001
    </div>""", unsafe_allow_html=True)

    st9a, st9b, st9c, st9d, st9e, st9f, st9g = st.tabs([
        "📊 Rate Basics",
        "🔢 Bond Valuation",
        "🔄 Swaps",
        "📉 Yield Curve",
        "🔧 Bootstrapping",
        "⚡ IR Risk",
        "💰 PnL Attribution",
    ])

    np.random.seed(10)
    # Synthetic PSX-style yield curve (PKR)
    tenors_yr  = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10])
    # base rates declining from short end (SBP policy ~22% in 2023) toward long end
    base_rates = np.array([0.220, 0.215, 0.210, 0.200, 0.195, 0.188, 0.182, 0.178])
    # discount factors
    Z = np.exp(-base_rates * tenors_yr)

    def interp_z(t, ten=tenors_yr, z=Z):
        """Linear interpolation of log(Z) = constant forward rates."""
        return float(np.exp(np.interp(t, ten, np.log(np.maximum(z, 1e-9)))))

    def forward_rate(t1, t2):
        z1, z2 = interp_z(t1), interp_z(t2)
        return -np.log(max(z2/max(z1, 1e-9), 1e-9)) / max(t2 - t1, 1e-9)

    # 9A — RATE BASICS
    with st9a:
        st.markdown("""
        <div class="fbox">
        <strong>SOFR compounding:</strong> r = (360/d)[Π(1 + rᵢdᵢ/360) − 1] &nbsp;|&nbsp;
        <strong>Simple avg:</strong> r = (1/d)Σrᵢdᵢ &nbsp;|&nbsp;
        <strong>Discount factor:</strong> Z(t) = e^{−r(t)·t} &nbsp;|&nbsp;
        <strong>Forward rate:</strong> f(t₁,t₂) = −ln[Z(t₂)/Z(t₁)]/(t₂−t₁)
        </div>""", unsafe_allow_html=True)

        # SOFR daily rates simulation
        np.random.seed(5)
        T_sofr  = 90
        r_daily = (base_rates[0] + np.random.normal(0, 0.001, T_sofr)).clip(0.001)
        d_i     = np.ones(T_sofr)  # 1 day each

        # Compounded vs simple
        comp_prod = np.cumprod(1 + r_daily * d_i / 360)
        sofr_comp = (360 / np.arange(1, T_sofr+1)) * (comp_prod - 1)
        sofr_simp = np.cumsum(r_daily * d_i) / np.arange(1, T_sofr+1)

        c1, c2, c3 = st.columns(3)
        c1.metric("90-day Compounded SOFR", f"{sofr_comp[-1]*100:.4f}%")
        c2.metric("90-day Simple Avg SOFR",  f"{sofr_simp[-1]*100:.4f}%")
        c3.metric("Compounding Premium",      f"{(sofr_comp[-1]-sofr_simp[-1])*100*100:.2f} bps")

        fig_sofr = go.Figure()
        fig_sofr.add_trace(go.Scatter(x=list(range(T_sofr)), y=sofr_comp*100,
            name="Compounded SOFR", line=dict(color="#00D4FF", width=2)))
        fig_sofr.add_trace(go.Scatter(x=list(range(T_sofr)), y=sofr_simp*100,
            name="Simple Average", line=dict(color="#F59E0B", width=2, dash="dash")))
        fig_sofr.update_layout(title="Compounded vs Simple Average SOFR (90-day accumulation)",
                               height=320, **lay())
        fig_sofr.update_xaxes(title_text="Business Day", **ax())
        fig_sofr.update_yaxes(title_text="Effective Rate (%)", **ax())
        st.plotly_chart(fig_sofr, use_container_width=True)

        # Discount factors and forward rates
        t_fine  = np.linspace(0.25, 10, 200)
        z_fine  = np.array([interp_z(t) for t in t_fine])
        fwd_1m  = np.array([forward_rate(t, min(t+1/12, 10)) for t in t_fine])

        fig_zf = make_subplots(rows=1, cols=2, subplot_titles=["Discount Factors Z(t)", "Forward Rates f(t)"])
        fig_zf.add_trace(go.Scatter(x=t_fine, y=z_fine, mode="lines", name="Z(t)",
            line=dict(color="#00D4FF", width=2)), row=1, col=1)
        fig_zf.add_trace(go.Scatter(x=tenors_yr, y=Z, mode="markers", name="Z nodes",
            marker=dict(color="#10B981", size=8), showlegend=False), row=1, col=1)
        fig_zf.add_trace(go.Scatter(x=t_fine, y=fwd_1m*100, mode="lines", name="f(t)",
            line=dict(color="#F59E0B", width=2)), row=1, col=2)
        fig_zf.update_layout(title="Discount Factor Curve & 1-Month Forward Rates",
                              height=320, **lay())
        fig_zf.update_xaxes(title_text="Tenor (years)", **ax())
        fig_zf.update_yaxes(**ax())
        st.plotly_chart(fig_zf, use_container_width=True)

    # 9B — BOND VALUATION
    with st9b:
        st.markdown("""
        <div class="fbox">
        <strong>Price:</strong> P = Σᵢ C·Z(tᵢ) + 100·Z(T) &nbsp;|&nbsp;
        <strong>Macaulay D:</strong> D = Σ tᵢ·PV(Cᵢ)/P &nbsp;|&nbsp;
        <strong>Modified D:</strong> D* = D/(1+y/f) &nbsp;|&nbsp;
        <strong>Convexity:</strong> C = Σ tᵢ²·PV(Cᵢ)/P &nbsp;|&nbsp;
        <strong>ΔP ≈ −D*·ΔP·Δy + ½·C·P·(Δy)²</strong>
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns([1, 2])
        with cl:
            coupon_r = st.slider("Coupon rate (%)", 0.0, 30.0, 15.0, 0.5, key="b_cpn") / 100
            maturity = st.slider("Maturity (years)", 1, 15, 5, 1, key="b_mat")
            freq     = st.radio("Coupon freq", [1, 2, 4], format_func=lambda f: {1:"Annual",2:"Semi",4:"Quarterly"}[f], key="b_freq", horizontal=True)
            face_val = 100.0

        n_periods  = maturity * freq
        t_coup     = np.array([(i+1)/freq for i in range(n_periods)])
        coup_pmt   = coupon_r * face_val / freq
        # price using discount factors
        pvs    = np.array([coup_pmt * interp_z(t) for t in t_coup])
        pvs[-1] += face_val * interp_z(t_coup[-1])  # add principal
        bond_price  = pvs.sum()
        mac_dur  = (t_coup * pvs / bond_price).sum()
        convexity = (t_coup**2 * pvs / bond_price).sum()
        ytm      = base_rates[np.argmin(np.abs(tenors_yr - maturity))]  # approx
        mod_dur  = mac_dur / (1 + ytm / freq)
        dv01     = -mod_dur * bond_price * 0.0001

        with cr:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Price",        f"{bond_price:.4f}")
            c2.metric("Mac Duration", f"{mac_dur:.3f}y")
            c3.metric("Mod Duration", f"{mod_dur:.3f}")
            c4.metric("Convexity",    f"{convexity:.3f}")
            c5.metric("DV01",         f"{dv01:.4f}")

        # Price vs yield (P-y curve)
        yields_scan = np.linspace(0.05, 0.40, 100)
        prices_scan = []
        for yld in yields_scan:
            p_scan = sum(coup_pmt / (1 + yld/freq)**(i+1) for i in range(n_periods))
            p_scan += face_val / (1 + yld/freq)**n_periods
            prices_scan.append(p_scan)

        fig_bond = make_subplots(rows=1, cols=2,
                                  subplot_titles=["Price-Yield Curve", "Cash Flow PV Bar Chart"])
        fig_bond.add_trace(go.Scatter(x=yields_scan*100, y=prices_scan, mode="lines",
            name="Price", line=dict(color="#00D4FF", width=2)), row=1, col=1)
        fig_bond.add_vline(x=ytm*100, line=dict(color="#EF4444", dash="dash"),
                           annotation_text=f"Current yield {ytm*100:.1f}%", row=1, col=1)
        fig_bond.add_hline(y=bond_price, line=dict(color="#10B981", dash="dot"),
                           annotation_text=f"P={bond_price:.2f}", row=1, col=1)

        # Cash flow bar chart
        bar_clrs = ["#10B981" if i < n_periods-1 else "#F59E0B" for i in range(n_periods)]
        fig_bond.add_trace(go.Bar(x=t_coup, y=pvs, name="PV Cash Flow",
            marker=dict(color=bar_clrs, opacity=0.85)), row=1, col=2)

        fig_bond.update_layout(title=f"Bond Valuation  |  Coupon={coupon_r*100:.1f}%  Mat={maturity}y  Price={bond_price:.2f}",
                               height=380, showlegend=False, **lay())
        fig_bond.update_xaxes(**ax()); fig_bond.update_yaxes(**ax())
        st.plotly_chart(fig_bond, use_container_width=True)

        # Duration vs maturity curve
        st.markdown("#### Duration & Convexity vs Maturity")
        mats    = np.arange(1, 16)
        durs, convs = [], []
        for m in mats:
            n_p = m * freq
            t_c = np.array([(i+1)/freq for i in range(n_p)])
            pv  = np.array([coup_pmt * interp_z(t) for t in t_c])
            pv[-1] += face_val * interp_z(t_c[-1])
            pp  = pv.sum()
            durs.append((t_c * pv / pp).sum())
            convs.append((t_c**2 * pv / pp).sum())

        fig_dur = make_subplots(rows=1, cols=2, subplot_titles=["Macaulay Duration vs Maturity","Convexity vs Maturity"])
        fig_dur.add_trace(go.Scatter(x=mats, y=durs, mode="lines+markers",
            line=dict(color="#F59E0B", width=2)), row=1, col=1)
        fig_dur.add_trace(go.Scatter(x=mats, y=convs, mode="lines+markers",
            line=dict(color="#8B5CF6", width=2)), row=1, col=2)
        fig_dur.update_layout(title="Duration & Convexity Profiles", height=300, showlegend=False, **lay())
        fig_dur.update_xaxes(title_text="Maturity (years)", **ax())
        fig_dur.update_yaxes(**ax())
        st.plotly_chart(fig_dur, use_container_width=True)

    # 9C — INTEREST RATE SWAPS
    with st9c:
        st.markdown("""
        <div class="fbox">
        <strong>Fixed Leg PV:</strong> PV_fix = C·Δ·ΣZ(tᵢ) &nbsp;|&nbsp;
        <strong>Float Leg PV:</strong> PV_flt = Σ f(tᵢ₋₁,tᵢ)·Δ·Z(tᵢ) &nbsp;|&nbsp;
        <strong>Par Swap Rate:</strong> C* = ΣF(tᵢ₋₁,tᵢ)·Δ·Z(tᵢ) / ΣΔ·Z(tᵢ)
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns([1, 2])
        with cl:
            sw_mat   = st.slider("Swap maturity (years)", 1, 10, 5, 1, key="sw_mat")
            sw_freq  = st.radio("Payment freq", [1, 2, 4], format_func=lambda f:{1:"Annual",2:"Semi",4:"Quarterly"}[f], key="sw_freq", horizontal=True)
            notional = st.number_input("Notional ($M)", value=100, step=10, key="sw_not")

        delta      = 1 / sw_freq
        n_sw       = sw_mat * sw_freq
        t_sw       = np.array([(i+1)*delta for i in range(n_sw)])
        t_sw_start = np.array([i*delta for i in range(n_sw)])

        Z_sw   = np.array([interp_z(t) for t in t_sw])
        fwd_sw = np.array([forward_rate(t_sw_start[i], t_sw[i]) for i in range(n_sw)])
        annuity = (delta * Z_sw).sum()

        pv_fix_per_unit = annuity     # per unit of coupon C
        pv_flt          = (fwd_sw * delta * Z_sw).sum()
        par_swap_rate   = pv_flt / max(annuity, 1e-12)

        # Mark-to-market at different fixed rates
        fixed_rates_scan = np.linspace(0.05, 0.35, 100)
        mtm_recv = [(pv_flt - c*annuity) * notional for c in fixed_rates_scan]  # receive fixed

        with cr:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Par Swap Rate",   f"{par_swap_rate*100:.3f}%")
            c2.metric("Fixed Leg Ann",   f"{annuity:.4f}")
            c3.metric("Float Leg PV",    f"{pv_flt:.4f}")
            c4.metric("MTM (par fixed)", f"${0.0:.2f}M")

        fig_sw = make_subplots(rows=1, cols=2,
                                subplot_titles=["Swap Cash Flows (Float vs Fixed at par)", "MTM vs Fixed Rate"])
        fig_sw.add_trace(go.Bar(x=t_sw, y=fwd_sw*delta*100, name="Float Cash Flow",
            marker=dict(color="#00D4FF", opacity=0.8)), row=1, col=1)
        fig_sw.add_trace(go.Bar(x=t_sw, y=[-par_swap_rate*delta*100]*n_sw, name="Fixed Cash Flow",
            marker=dict(color="#EF4444", opacity=0.8)), row=1, col=1)
        fig_sw.add_trace(go.Scatter(x=fixed_rates_scan*100, y=mtm_recv, mode="lines",
            name="MTM (receive fix)", line=dict(color="#10B981", width=2)), row=1, col=2)
        fig_sw.add_hline(y=0, line=dict(color="#475569", dash="dot"), row=1, col=2)
        fig_sw.add_vline(x=par_swap_rate*100, line=dict(color="#EF4444", dash="dash"),
                         annotation_text="par", row=1, col=2)
        fig_sw.update_layout(title=f"IRS Valuation  |  {sw_mat}y  {sw_freq}×/yr  Notional=${notional}M",
                              height=360, **lay())
        fig_sw.update_xaxes(**ax()); fig_sw.update_yaxes(**ax())
        st.plotly_chart(fig_sw, use_container_width=True)

        # Swap curve: par swap rates for different maturities
        st.markdown("#### Swap Curve — Par Swap Rates by Maturity")
        sw_mats = list(range(1, 11))
        par_rates = []
        for m in sw_mats:
            n_p = m * sw_freq; d_p = 1/sw_freq
            t_p = np.array([(i+1)*d_p for i in range(n_p)])
            t_s = np.array([i*d_p for i in range(n_p)])
            Z_p = np.array([interp_z(t) for t in t_p])
            f_p = np.array([forward_rate(t_s[i], t_p[i]) for i in range(n_p)])
            ann = (d_p * Z_p).sum()
            par_rates.append((f_p * d_p * Z_p).sum() / max(ann, 1e-12))

        fig_swcv = go.Figure()
        fig_swcv.add_trace(go.Scatter(x=sw_mats, y=[r*100 for r in par_rates], mode="lines+markers",
            name="Par Swap", line=dict(color="#F59E0B", width=2),
            marker=dict(size=7, color="#F59E0B")))
        fig_swcv.add_trace(go.Scatter(x=tenors_yr, y=base_rates*100, mode="lines+markers",
            name="Zero Curve", line=dict(color="#00D4FF", width=1.5, dash="dash"),
            marker=dict(size=6, color="#00D4FF")))
        fig_swcv.update_layout(title="Par Swap Rate Curve vs Zero Curve", height=300, **lay())
        fig_swcv.update_xaxes(title_text="Maturity (years)", **ax())
        fig_swcv.update_yaxes(title_text="Rate (%)", **ax())
        st.plotly_chart(fig_swcv, use_container_width=True)

    # 9D — YIELD CURVE INTERPOLATION
    with st9d:
        st.markdown("""
        <div class="fbox">
        <strong>Linear on ln Z:</strong> ln Z(t) = ln Z(tᵢ) + (t−tᵢ)·f(tᵢ,tᵢ₊₁) → constant fwd rates &nbsp;|&nbsp;
        <strong>Cubic Spline:</strong> x(t) = Σ aⱼ(t−tᵢ)ʲ → smooth forward rates &nbsp;|&nbsp;
        <strong>CDF:</strong> front-end meeting-based interpolation
        </div>""", unsafe_allow_html=True)

        from scipy.interpolate import CubicSpline

        cl, cr = st.columns([1, 2])
        with cl:
            interp_method = st.selectbox("Interpolation Method",
                ["Linear (log Z)", "Cubic Spline", "CDF/Meeting-Based", "Hybrid"], key="yc_interp")
            bump_bps = st.slider("Parallel shift (bps)", -100, 100, 0, 5, key="yc_bump")

        # bumped rates
        bumped = base_rates + bump_bps / 10000

        t_plot = np.linspace(0.25, 10, 300)

        # Linear log-Z interpolation (built-in interp_z)
        z_linear = np.array([np.exp(np.interp(t, tenors_yr, np.log(np.maximum(bumped*0+np.exp(-bumped*tenors_yr), 1e-9)))) for t in t_plot])

        # Cubic spline on zero rates
        cs = CubicSpline(tenors_yr, bumped)
        z_cubic = np.exp(-cs(t_plot) * t_plot)

        # CDF (meeting-based): step function on 8 tenors
        fomc_dates = tenors_yr  # proxy for meeting dates
        z_cdf = np.array([bumped[np.searchsorted(fomc_dates, min(t, fomc_dates[-1]), side='right')-1] for t in t_plot])

        # Hybrid: CDF for t<1, spline for t>=1
        z_hybrid = np.where(t_plot < 1, z_cdf, cs(t_plot))

        method_data = {
            "Linear (log Z)": z_linear*0+cs(t_plot),  # show zero rates
            "Cubic Spline":    cs(t_plot),
            "CDF/Meeting-Based": z_cdf,
            "Hybrid":          z_hybrid,
        }

        fig_yc = make_subplots(rows=1, cols=2,
                                subplot_titles=["Zero Rate Curves (4 methods)", "Forward Rate Curves"])
        colors_yc = ["#00D4FF", "#10B981", "#F59E0B", "#8B5CF6"]
        for i, (name, zr) in enumerate(method_data.items()):
            show = (name == interp_method)
            lw = 2.5 if show else 1.2
            alpha_v = 1.0 if show else 0.4
            fig_yc.add_trace(go.Scatter(x=t_plot, y=zr*100, mode="lines", name=name,
                line=dict(color=colors_yc[i], width=lw),
                opacity=alpha_v), row=1, col=1)
            # forward rates: -d ln Z / dt
            fwd = -(np.gradient(np.log(np.exp(-zr*t_plot)), t_plot))
            fig_yc.add_trace(go.Scatter(x=t_plot, y=fwd*100, mode="lines", name=name+" fwd",
                line=dict(color=colors_yc[i], width=lw, dash="dash"),
                opacity=alpha_v, showlegend=False), row=1, col=2)

        fig_yc.add_trace(go.Scatter(x=tenors_yr, y=bumped*100, mode="markers",
            marker=dict(color="#EF4444", size=10, symbol="x"), name="Input nodes"), row=1, col=1)
        fig_yc.update_layout(title=f"Yield Curve Interpolation  |  Shift={bump_bps:+d} bps",
                              height=400, **lay())
        fig_yc.update_xaxes(title_text="Tenor (years)", **ax())
        fig_yc.update_yaxes(title_text="Rate (%)", **ax())
        st.plotly_chart(fig_yc, use_container_width=True)

    # 9E — CURVE CALIBRATION (BOOTSTRAPPING)
    with st9e:
        st.markdown("""
        <div class="fbox">
        <strong>Bootstrap:</strong> Sequential extraction of discount factors from market instruments &nbsp;|&nbsp;
        <strong>CDs:</strong> Z(T) = 1/(1+r·T) &nbsp;|&nbsp;
        <strong>Futures fwd:</strong> f = 100 − F − Cvx &nbsp;|&nbsp;
        <strong>Swaps:</strong> Z(T) = (1 − C·Σᵢ₋₁Δ·Z(tᵢ)) / (1 + C·Δ)
        </div>""", unsafe_allow_html=True)

        # Synthetic instrument quotes
        cd_data = [
            ("1M CD",  30/360,  0.2150),
            ("3M CD",  90/360,  0.2180),
            ("6M CD", 180/360,  0.2130),
            ("1Y CD",   1.0,    0.2100),
        ]
        fut_data = [
            ("EDM4",  0.25, 0.50, 78.50, 0.005),
            ("EDU4",  0.50, 0.75, 78.80, 0.007),
            ("EDZ4",  0.75, 1.00, 79.10, 0.010),
            ("EDH5",  1.00, 1.25, 79.30, 0.012),
        ]
        swap_data = [
            ("2Y Swap",  2.0, 0.1960),
            ("3Y Swap",  3.0, 0.1940),
            ("5Y Swap",  5.0, 0.1900),
            ("7Y Swap",  7.0, 0.1870),
            ("10Y Swap",10.0, 0.1840),
        ]

        # Bootstrap discount factors
        boot_t, boot_Z = [], []

        # Step 1: CDs
        for name, T_cd, r_cd in cd_data:
            z = 1 / (1 + r_cd * T_cd)
            boot_t.append(T_cd); boot_Z.append(z)

        # Step 2: Futures
        for name, t1, t2, F, cvx in fut_data:
            fwd_r = (100 - F)/100 - cvx
            z1    = np.exp(np.interp(t1, boot_t, np.log(np.maximum(boot_Z, 1e-9))))
            z2    = z1 * np.exp(-fwd_r * (t2 - t1))
            boot_t.append(t2); boot_Z.append(z2)

        # Step 3: Swaps — sequential bootstrap
        freq_s = 2; delta_s = 0.5
        existing = sorted(zip(boot_t, boot_Z), key=lambda x: x[0])
        bt, bZ = zip(*existing) if existing else ([], [])
        bt, bZ = list(bt), list(bZ)

        for sw_name, T_sw, C_sw in swap_data:
            n_p = int(T_sw * freq_s)
            t_p = [(i+1)*delta_s for i in range(n_p)]
            pv_known = sum(C_sw * delta_s * np.exp(np.interp(tt, bt, np.log(np.maximum(bZ, 1e-9))))
                          for tt in t_p[:-1])
            z_last  = (1 - pv_known) / max(1 + C_sw * delta_s, 1e-9)
            bt.append(T_sw); bZ.append(z_last)

        bt_arr = np.array(bt); bZ_arr = np.array(bZ)
        sort_idx = np.argsort(bt_arr)
        bt_arr, bZ_arr = bt_arr[sort_idx], bZ_arr[sort_idx]
        boot_zero = -np.log(np.maximum(bZ_arr, 1e-12)) / np.maximum(bt_arr, 1e-9)

        fig_boot = make_subplots(rows=1, cols=2,
                                  subplot_titles=["Bootstrapped Discount Factors", "Bootstrapped Zero Rates"])
        # colour by instrument type
        cd_mask  = bt_arr <= 1.0
        fut_mask = (bt_arr > 1.0) & (bt_arr <= 1.25)
        sw_mask  = bt_arr > 1.25

        for mask, col, lbl in [(cd_mask,"#00D4FF","CDs"), (fut_mask,"#F59E0B","Futures"), (sw_mask,"#10B981","Swaps")]:
            if mask.any():
                fig_boot.add_trace(go.Scatter(x=bt_arr[mask], y=bZ_arr[mask], mode="markers+lines",
                    name=lbl, marker=dict(size=8, color=col), line=dict(color=col, width=1.5)), row=1, col=1)
                fig_boot.add_trace(go.Scatter(x=bt_arr[mask], y=boot_zero[mask]*100, mode="markers+lines",
                    name=lbl+" zero", marker=dict(size=8, color=col), line=dict(color=col, width=1.5),
                    showlegend=False), row=1, col=2)

        fig_boot.update_layout(title="Curve Bootstrapping: CDs → Futures → Swaps",
                               height=380, **lay())
        fig_boot.update_xaxes(title_text="Tenor (years)", **ax())
        fig_boot.update_yaxes(**ax())
        st.plotly_chart(fig_boot, use_container_width=True)

        # Show instrument table
        st.markdown("#### Calibration Instruments")
        inst_rows = [(n, f"{T:.3f}", f"{r*100:.2f}%", "CD") for n,T,r in cd_data] + \
                    [(n, f"{t2:.2f}", f"{(100-F)/100*100:.2f}% fwd", "Futures") for n,_,t2,F,_ in fut_data] + \
                    [(n, f"{T:.1f}", f"{C*100:.2f}%", "Swap") for n,T,C in swap_data]
        inst_df = pd.DataFrame(inst_rows, columns=["Instrument","Tenor (yr)","Rate/Quote","Type"])
        st.dataframe(inst_df, use_container_width=True)

    # 9F — INTEREST RATE RISK
    with st9f:
        st.markdown("""
        <div class="fbox">
        <strong>DV01:</strong> ∂PV/∂r × 0.0001 &nbsp;|&nbsp;
        <strong>Delta Ladder:</strong> DV01 at each tenor &nbsp;|&nbsp;
        <strong>Parallel Shift:</strong> ΔPV = −D*·P·Δy &nbsp;|&nbsp;
        <strong>Convexity Adj:</strong> ΔPV ≈ −D*·P·Δy + ½·C·P·(Δy)²
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns([1, 2])
        with cl:
            risk_mat  = st.slider("Bond maturity (years)", 1, 10, 5, key="risk_mat")
            risk_cpn  = st.slider("Coupon rate (%)", 0.0, 25.0, 12.0, 0.5, key="risk_cpn") / 100
            risk_freq = st.radio("Freq", [1,2,4], format_func=lambda f:{1:"A",2:"S",4:"Q"}[f], horizontal=True, key="risk_frq")

        n_rk   = risk_mat * risk_freq
        delta_rk = 1/risk_freq
        t_rk   = np.array([(i+1)*delta_rk for i in range(n_rk)])
        coup_rk= risk_cpn * 100 / risk_freq
        pv_rk  = np.array([coup_rk * interp_z(t) for t in t_rk])
        pv_rk[-1] += 100 * interp_z(t_rk[-1])
        P_rk   = pv_rk.sum()

        # DV01 per tenor (delta ladder)
        dv01_ladder = []
        for ti in tenors_yr:
            # bump only the rate at this tenor by 1bp
            bumped_local = base_rates.copy()
            idx_close = np.argmin(np.abs(tenors_yr - ti))
            bumped_local[idx_close] += 0.0001
            Z_bump = np.exp(-bumped_local * tenors_yr)
            def interp_z_bump(t):
                return float(np.exp(np.interp(t, tenors_yr, np.log(np.maximum(Z_bump, 1e-9)))))
            pv_bump = np.array([coup_rk * interp_z_bump(t) for t in t_rk])
            pv_bump[-1] += 100 * interp_z_bump(t_rk[-1])
            dv01_ladder.append(P_rk - pv_bump.sum())

        mac_rk   = (t_rk * pv_rk / P_rk).sum()
        ytm_rk   = base_rates[np.argmin(np.abs(tenors_yr - risk_mat))]
        mod_rk   = mac_rk / (1 + ytm_rk/risk_freq)
        conv_rk  = (t_rk**2 * pv_rk / P_rk).sum()
        dv01_rk  = -mod_rk * P_rk * 0.0001

        with cr:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Price",     f"{P_rk:.3f}")
            c2.metric("Mod Dur",   f"{mod_rk:.3f}")
            c3.metric("DV01",      f"{dv01_rk:.4f}")
            c4.metric("Convexity", f"{conv_rk:.3f}")

        # P&L for parallel shifts
        shifts = np.linspace(-0.02, 0.02, 100)
        pnl_dur  = -mod_rk * P_rk * shifts * 100
        pnl_conv = pnl_dur + 0.5 * conv_rk * P_rk * (shifts**2) * 100

        fig_risk = make_subplots(rows=1, cols=2,
                                  subplot_titles=["P&L: Duration vs Duration+Convexity", "DV01 Delta Ladder"])
        fig_risk.add_trace(go.Scatter(x=shifts*10000, y=pnl_dur, mode="lines",
            name="Duration only", line=dict(color="#EF4444", width=2)), row=1, col=1)
        fig_risk.add_trace(go.Scatter(x=shifts*10000, y=pnl_conv, mode="lines",
            name="Dur + Convexity", line=dict(color="#10B981", width=2)), row=1, col=1)
        fig_risk.add_trace(go.Bar(x=tenors_yr, y=dv01_ladder, name="DV01 per tenor",
            marker=dict(color=["#00D4FF" if d < 0 else "#EF4444" for d in dv01_ladder], opacity=0.85)), row=1, col=2)
        fig_risk.update_layout(title=f"Interest Rate Risk  |  Bond {risk_mat}y cpn={risk_cpn*100:.1f}%",
                               height=380, **lay())
        fig_risk.update_xaxes(**ax()); fig_risk.update_yaxes(**ax())
        fig_risk.update_xaxes(title_text="Shift (bps)", row=1, col=1)
        fig_risk.update_xaxes(title_text="Tenor (yr)",  row=1, col=2)
        fig_risk.update_yaxes(title_text="P&L", row=1, col=1)
        fig_risk.update_yaxes(title_text="DV01", row=1, col=2)
        st.plotly_chart(fig_risk, use_container_width=True)

    # 9G — PnL ATTRIBUTION
    with st9g:
        st.markdown("""
        <div class="fbox">
        <strong>PnL:</strong> ΔPV = PV(t₁) − PV(t₀) &nbsp;|&nbsp;
        <strong>Theta:</strong> ∂PV/∂t (time decay) &nbsp;|&nbsp;
        <strong>Market Move:</strong> Σᵢ (∂PV/∂xᵢ)·Δxᵢ &nbsp;|&nbsp;
        <strong>Roll:</strong> PV(t+Δt, same curve) − PV(t, same curve)
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns([1, 2])
        with cl:
            pnl_mat = st.slider("Position maturity (y)", 1, 10, 3, key="pnl_mat")
            pnl_cpn = st.slider("Coupon (%)", 0.0, 25.0, 12.0, 0.5, key="pnl_cpn") / 100
            rate_chg_bps = st.slider("Rate change (bps)", -200, 200, 50, 10, key="pnl_rc")
            theta_days   = st.slider("Time elapsed (days)", 1, 30, 5, key="pnl_td")

        # Compute bond PV t0
        n_pnl     = pnl_mat * 2; d_pnl = 0.5
        t_pnl0    = np.array([(i+1)*d_pnl for i in range(n_pnl)])
        cpmt_pnl  = pnl_cpn * 100 / 2
        pv_pnl0   = np.array([cpmt_pnl * interp_z(t) for t in t_pnl0])
        pv_pnl0[-1] += 100 * interp_z(t_pnl0[-1])
        P_pnl0    = pv_pnl0.sum()
        mac_pnl   = (t_pnl0 * pv_pnl0 / P_pnl0).sum()
        ytm_pnl   = base_rates[np.argmin(np.abs(tenors_yr - pnl_mat))]
        mod_pnl   = mac_pnl / (1 + ytm_pnl/2)
        conv_pnl  = (t_pnl0**2 * pv_pnl0 / P_pnl0).sum()

        # Theta: same curve, shorter maturity
        t_theta = max(pnl_mat - theta_days/252, 0.25)
        n_th    = max(int(t_theta*2), 1); t_pnl1 = np.array([(i+1)*d_pnl for i in range(n_th)])
        pv_th   = np.array([cpmt_pnl * interp_z(t) for t in t_pnl1])
        pv_th[-1] += 100 * interp_z(t_pnl1[-1])
        theta_pnl = pv_th.sum() - P_pnl0

        # Market move PnL (parallel shift)
        delta_r  = rate_chg_bps / 10000
        mkt_pnl  = -mod_pnl * P_pnl0 * delta_r + 0.5 * conv_pnl * P_pnl0 * delta_r**2

        # Roll PnL: bond is now 3M shorter, same curve shape
        t_roll   = max(pnl_mat - 90/252, 0.25)
        n_rl     = max(int(t_roll*2), 1); t_pnl_r = np.array([(i+1)*d_pnl for i in range(n_rl)])
        pv_rl    = np.array([cpmt_pnl * interp_z(t) for t in t_pnl_r])
        pv_rl[-1] += 100 * interp_z(t_pnl_r[-1])
        roll_pnl = pv_rl.sum() - P_pnl0

        total_pnl = theta_pnl + mkt_pnl + roll_pnl

        with cr:
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Initial PV",    f"{P_pnl0:.4f}")
            c2.metric("Theta P&L",     f"{theta_pnl:.4f}")
            c3.metric("Market P&L",    f"{mkt_pnl:.4f}")
            c4.metric("Roll P&L",      f"{roll_pnl:.4f}")
            c5.metric("Total P&L",     f"{total_pnl:.4f}",
                       delta=f"{'▲' if total_pnl > 0 else '▼'} {abs(total_pnl):.4f}")

        # Waterfall chart
        pnl_components = [theta_pnl, mkt_pnl, roll_pnl, total_pnl]
        pnl_labels     = ["Theta (Time Decay)", f"Market Move ({rate_chg_bps:+d}bps)",
                          f"Roll ({90}d)", "Total P&L"]
        bar_colors_pnl = ["#EF4444" if v < 0 else "#10B981" for v in pnl_components]
        bar_colors_pnl[-1] = "#00D4FF"  # total always blue

        fig_pnl = go.Figure(go.Bar(
            x=pnl_labels, y=[abs(v) for v in pnl_components],
            base=[min(0, v) for v in pnl_components],
            marker=dict(color=bar_colors_pnl, opacity=0.85),
            text=[f"{v:+.4f}" for v in pnl_components], textposition="outside",
        ))
        fig_pnl.add_hline(y=0, line=dict(color="#475569", dash="dot"))
        fig_pnl.update_layout(title=f"PnL Attribution  |  Bond {pnl_mat}y cpn={pnl_cpn*100:.1f}%",
                               height=360, showlegend=False, **lay())
        fig_pnl.update_xaxes(**ax()); fig_pnl.update_yaxes(title_text="P&L (per 100 face)", **ax())
        st.plotly_chart(fig_pnl, use_container_width=True)


with tab10:
    st.markdown('<div class="sec-hdr">10. Advanced Regression & E-Trading — GLS · Rolling Regression · Kalman · Factor Comparison · E-Trading</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="fbox">
    <strong>GLS / WLS:</strong> β̂ = (XᵀΩ⁻¹X)⁻¹XᵀΩ⁻¹y &nbsp;|&nbsp;
    <strong>Kalman:</strong> βₜ = βₜ₋₁ + ηₜ, Rₜ = αₜ + βₜRₘₜ + εₜ &nbsp;|&nbsp;
    <strong>Walk-Forward:</strong> expand train window, step-by-step test &nbsp;|&nbsp;
    <strong>Bid-Offer:</strong> spread = f(vol, turnover, inventory)
    </div>""", unsafe_allow_html=True)

    from scipy import stats as sp_stats10
    from sklearn.linear_model import Ridge as Ridge10, Lasso as Lasso10

    st10a, st10b, st10c, st10d = st.tabs([
        "⚖️ GLS & Heteroscedasticity",
        "🔄 Time-Varying Parameters",
        "🏆 Factor Model Comparison",
        "⚡ E-Trading Implementation",
    ])

    # 10A — GLS & HETEROSCEDASTICITY
    with st10a:
        st.markdown("""
        <div class="fbox">
        <strong>WLS:</strong> β̂_WLS = (XᵀWX)⁻¹XᵀWy  where W = diag(wᵢ) &nbsp;|&nbsp;
        <strong>Breusch-Pagan:</strong> H₀: homoscedasticity &nbsp;|&nbsp;
        <strong>HC Standard Errors:</strong> V_HC = (XᵀX)⁻¹(Σeᵢ²xᵢxᵢᵀ)(XᵀX)⁻¹
        </div>""", unsafe_allow_html=True)

        gls_dep = st.selectbox("Dependent stock", selected, key="gls_dep")
        gls_ind = [t for t in selected if t != gls_dep]
        if gls_ind:
            y_g = R_df[gls_dep].values
            X_g = np.column_stack([np.ones(T_days)] + [R_df[t].values for t in gls_ind[:3]])
            n_g, p_g = X_g.shape

            # OLS
            b_ols, yh_ols, e_ols, r2_ols, _, _, _, se_ols, mse_ols = ols_fit(X_g, y_g)

            # Breusch-Pagan test: regress e² on X
            e2 = e_ols**2
            _, _, _, r2_bp, _, _, _, _, _ = ols_fit(X_g, e2)
            bp_stat = n_g * r2_bp
            bp_pval = 1 - sp_stats10.chi2.cdf(bp_stat, p_g - 1)

            # WLS weights = 1/|e_ols| (approximation)
            w_wls   = 1 / np.maximum(np.abs(e_ols), 1e-8)
            W_mat   = np.diag(w_wls)
            XtWX    = X_g.T @ W_mat @ X_g
            XtWy    = X_g.T @ W_mat @ y_g
            b_wls   = np.linalg.solve(XtWX + np.eye(p_g)*1e-10, XtWy)
            yh_wls  = X_g @ b_wls
            e_wls   = y_g - yh_wls
            r2_wls  = 1 - (e_wls**2).sum() / max(((y_g-y_g.mean())**2).sum(), 1e-12)

            # HC standard errors (sandwich)
            XtX_inv = np.linalg.pinv(X_g.T @ X_g)
            meat    = sum(e_ols[i]**2 * np.outer(X_g[i], X_g[i]) for i in range(n_g))
            hc_var  = XtX_inv @ meat @ XtX_inv
            se_hc   = np.sqrt(np.diag(hc_var))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("OLS R²",         f"{r2_ols:.4f}")
            m2.metric("WLS R²",         f"{r2_wls:.4f}")
            m3.metric("BP statistic",   f"{bp_stat:.3f}")
            m4.metric("BP p-value",     f"{bp_pval:.4f}",
                       delta="Reject H₀" if bp_pval < 0.05 else "Fail to Reject")

            # OLS vs WLS vs HC coefficient comparison
            feat_g = ["Intercept"] + gls_ind[:3]
            t_ols  = b_ols / np.maximum(se_ols, 1e-12)
            t_hc   = b_ols / np.maximum(se_hc, 1e-12)
            coef_df = pd.DataFrame({
                "Feature": feat_g,
                "β̂ (OLS)":  [f"{v:.6f}" for v in b_ols],
                "SE (OLS)":  [f"{v:.6f}" for v in se_ols],
                "t (OLS)":   [f"{v:.3f}" for v in t_ols],
                "SE (HC)":   [f"{v:.6f}" for v in se_hc],
                "t (HC)":    [f"{v:.3f}" for v in t_hc],
                "β̂ (WLS)":  [f"{v:.6f}" for v in b_wls],
            })
            st.markdown("#### OLS vs HC Standard Errors vs WLS Coefficients")
            st.dataframe(coef_df, use_container_width=True)
            st.caption("HC = Heteroscedasticity-Consistent (White) standard errors  ·  WLS = Weighted Least Squares")

            # Residual variance plot
            fig_gls = make_subplots(rows=1, cols=2,
                subplot_titles=["OLS vs WLS Residuals", "|Residuals| vs Fitted (Heteroscedasticity)"])
            idx_g = R_df.index
            fig_gls.add_trace(go.Scatter(x=idx_g, y=e_ols*100, mode="lines", name="OLS Resid",
                line=dict(color="#EF4444", width=1.0), opacity=0.7), row=1, col=1)
            fig_gls.add_trace(go.Scatter(x=idx_g, y=e_wls*100, mode="lines", name="WLS Resid",
                line=dict(color="#10B981", width=1.0), opacity=0.7), row=1, col=1)
            fig_gls.add_trace(go.Scatter(x=yh_ols*100, y=np.abs(e_ols)*100, mode="markers",
                marker=dict(color="#F59E0B", size=3, opacity=0.5), name="|e| vs fitted"), row=1, col=2)
            # lowess-style trend line on abs residuals
            sort_fit = np.argsort(yh_ols)
            fig_gls.add_trace(go.Scatter(
                x=np.sort(yh_ols*100),
                y=pd.Series(np.abs(e_ols[sort_fit])*100).rolling(30, min_periods=1).mean().values,
                mode="lines", line=dict(color="#EF4444", width=2), name="Trend", showlegend=False), row=1, col=2)
            fig_gls.update_layout(title="GLS / Heteroscedasticity Analysis", height=380, **lay())
            fig_gls.update_xaxes(**ax()); fig_gls.update_yaxes(**ax())
            st.plotly_chart(fig_gls, use_container_width=True)

    # 10B — TIME-VARYING PARAMETERS
    with st10b:
        st.markdown("""
        <div class="fbox">
        <strong>Rolling:</strong> β̂(t) computed on [t−w, t] &nbsp;|&nbsp;
        <strong>Recursive:</strong> β̂(t) uses all data up to t &nbsp;|&nbsp;
        <strong>Kalman Filter:</strong> βₜ = Fβₜ₋₁ + ηₜ (state eq)  ·  yₜ = Hβₜ + εₜ (obs eq)
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns([1, 2])
        with cl:
            tv_dep = st.selectbox("Dependent stock", selected, key="tv_dep")
            tv_ind = st.selectbox("Independent stock", [t for t in selected if t != tv_dep], key="tv_ind")
            tv_win = st.slider("Rolling window", 20, 120, 60, key="tv_win")

        y_tv = R_df[tv_dep].values
        x_tv = R_df[tv_ind].values
        n_tv = len(y_tv)

        # Rolling OLS (slope only, simple)
        roll_beta, roll_alpha = [], []
        for end in range(tv_win, n_tv):
            y_w = y_tv[end-tv_win:end]; x_w = x_tv[end-tv_win:end]
            X_w = np.column_stack([np.ones(tv_win), x_w])
            b_w = np.linalg.lstsq(X_w, y_w, rcond=None)[0]
            roll_beta.append(b_w[1]); roll_alpha.append(b_w[0]*252)

        # Recursive OLS
        rec_beta, rec_alpha = [], []
        for end in range(2, n_tv):
            y_r = y_tv[:end]; x_r = x_tv[:end]
            X_r = np.column_stack([np.ones(end), x_r])
            b_r = np.linalg.lstsq(X_r, y_r, rcond=None)[0]
            rec_beta.append(b_r[1]); rec_alpha.append(b_r[0]*252)

        # Kalman Filter (simple 2-state: [alpha_d, beta])
        # State: [alpha_d, beta]; obs: y_t = alpha_d + beta*x_t + eps
        q_var, r_var = 1e-6, float(y_tv.var())
        P_kal   = np.eye(2) * 0.1
        x_kal   = np.array([0.0, 1.0])   # initial state
        kal_beta, kal_alpha_d = [], []
        for i in range(n_tv):
            # Predict
            x_kal_p = x_kal.copy()
            P_kal_p = P_kal + q_var * np.eye(2)
            # Update
            H_k = np.array([1.0, x_tv[i]])
            innov = y_tv[i] - H_k @ x_kal_p
            S_k   = H_k @ P_kal_p @ H_k + r_var
            K_k   = P_kal_p @ H_k / max(S_k, 1e-12)
            x_kal = x_kal_p + K_k * innov
            P_kal = (np.eye(2) - np.outer(K_k, H_k)) @ P_kal_p
            kal_alpha_d.append(x_kal[0]*252)
            kal_beta.append(x_kal[1])

        with cr:
            m1, m2, m3 = st.columns(3)
            m1.metric("Latest Rolling β",   f"{roll_beta[-1]:.4f}" if roll_beta else "—")
            m2.metric("Latest Recursive β", f"{rec_beta[-1]:.4f}"  if rec_beta else "—")
            m3.metric("Latest Kalman β",    f"{kal_beta[-1]:.4f}")

        fig_tv = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                               subplot_titles=["Beta (β) — Three Methods", "Alpha (α ann.) — Three Methods"])
        roll_idx = dates[tv_win:]
        rec_idx  = dates[2:]
        kal_idx  = dates

        for method, vals, idx_m, col_m, nm in [
            ("Rolling", roll_beta, roll_idx, "#00D4FF", f"Rolling({tv_win}d)"),
            ("Recursive", rec_beta, rec_idx, "#F59E0B", "Recursive"),
            ("Kalman", kal_beta, kal_idx, "#10B981", "Kalman Filter"),
        ]:
            fig_tv.add_trace(go.Scatter(x=idx_m, y=vals, name=nm,
                line=dict(color=col_m, width=1.5)), row=1, col=1)

        for method, vals, idx_m, col_m, nm in [
            ("Rolling", roll_alpha, roll_idx, "#00D4FF", f"Roll α"),
            ("Recursive", rec_alpha, rec_idx, "#F59E0B", "Recur α"),
            ("Kalman", kal_alpha_d, kal_idx, "#10B981", "Kalman α"),
        ]:
            fig_tv.add_trace(go.Scatter(x=idx_m, y=[v*100 for v in vals], name=nm,
                line=dict(color=col_m, width=1.5), showlegend=False), row=2, col=1)

        fig_tv.add_hline(y=1.0, line=dict(color="#475569", dash="dot", width=1), row=1, col=1,
                         annotation_text="β=1")
        fig_tv.add_hline(y=0.0, line=dict(color="#475569", dash="dot", width=1), row=2, col=1)
        fig_tv.update_layout(title=f"Time-Varying Parameters: {tv_dep} ~ {tv_ind}",
                              height=480, **lay())
        fig_tv.update_xaxes(**ax()); fig_tv.update_yaxes(**ax())
        st.plotly_chart(fig_tv, use_container_width=True)

    # 10C — FACTOR MODEL COMPARISON
    with st10c:
        st.markdown("""
        <div class="fbox">
        <strong>Walk-Forward:</strong> Expand train [0,t], test [t, t+h] — no look-ahead &nbsp;|&nbsp;
        <strong>Models:</strong> OLS · Ridge · LASSO · PCR (Principal Component Regression) &nbsp;|&nbsp;
        <strong>Metrics:</strong> RMSE · MAE · R²_oos (out-of-sample)
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns([1, 2])
        with cl:
            fc_dep = st.selectbox("Target stock", selected, key="fc_dep")
            fc_ind = [t for t in selected if t != fc_dep]
            fc_step = st.slider("Walk-forward step (days)", 5, 30, 10, key="fc_step")
            fc_init = st.slider("Initial train size (%)", 40, 80, 60, key="fc_init")
            lam_ridge10 = st.slider("Ridge λ", 0.0001, 1.0, 0.01, format="%.4f", key="fc_lam")

        y_fc = R_df[fc_dep].values
        X_fc = StandardScaler().fit_transform(
            np.column_stack([R_df[t].values for t in fc_ind])
        )
        n_fc    = len(y_fc)
        init_sz = int(n_fc * fc_init / 100)

        # Walk-forward evaluation
        oos_idx, oos_actual = [], []
        pred_ols, pred_ridge, pred_lasso, pred_pcr = [], [], [], []

        for start in range(init_sz, n_fc - fc_step + 1, fc_step):
            X_tr, y_tr = X_fc[:start], y_fc[:start]
            X_te, y_te = X_fc[start:start+fc_step], y_fc[start:start+fc_step]

            # OLS (pinv for stability)
            X_tr_c = np.column_stack([np.ones(len(y_tr)), X_tr])
            X_te_c = np.column_stack([np.ones(len(y_te)), X_te])
            b_ols_fc = np.linalg.lstsq(X_tr_c, y_tr, rcond=None)[0]
            pred_ols.extend(X_te_c @ b_ols_fc)

            # Ridge
            rdg = Ridge10(alpha=lam_ridge10).fit(X_tr, y_tr)
            pred_ridge.extend(rdg.predict(X_te))

            # LASSO
            try:
                lso = Lasso10(alpha=lam_ridge10*0.1, max_iter=5000).fit(X_tr, y_tr)
                pred_lasso.extend(lso.predict(X_te))
            except Exception:
                pred_lasso.extend([y_tr.mean()]*fc_step)

            # PCR (PCA + OLS on first 3 components)
            pca10 = PCA(n_components=min(3, X_tr.shape[1]))
            Xpc_tr = pca10.fit_transform(X_tr)
            Xpc_te = pca10.transform(X_te)
            Xpc_tr_c = np.column_stack([np.ones(len(y_tr)), Xpc_tr])
            Xpc_te_c = np.column_stack([np.ones(len(y_te)), Xpc_te])
            b_pcr = np.linalg.lstsq(Xpc_tr_c, y_tr, rcond=None)[0]
            pred_pcr.extend(Xpc_te_c @ b_pcr)

            oos_idx.extend(range(start, start+fc_step))
            oos_actual.extend(y_te)

        oos_actual = np.array(oos_actual)
        oos_dates  = dates[oos_idx] if len(oos_idx) <= len(dates) else dates[:len(oos_idx)]

        def oos_metrics(pred):
            pred = np.array(pred)
            n_oo = len(oos_actual)
            rmse = np.sqrt(((oos_actual - pred)**2).mean())
            mae  = np.abs(oos_actual - pred).mean()
            ss_r = ((oos_actual - pred)**2).sum()
            ss_t = ((oos_actual - oos_actual.mean())**2).sum()
            r2   = 1 - ss_r / max(ss_t, 1e-12)
            return rmse*100, mae*100, r2

        metrics = {
            "OLS":   oos_metrics(pred_ols),
            "Ridge": oos_metrics(pred_ridge),
            "LASSO": oos_metrics(pred_lasso),
            "PCR":   oos_metrics(pred_pcr),
        }

        with cr:
            m_df = pd.DataFrame(metrics, index=["RMSE (%)", "MAE (%)", "OOS R²"]).T.reset_index()
            m_df.columns = ["Model","RMSE (%)","MAE (%)","OOS R²"]
            m_df["RMSE (%)"] = m_df["RMSE (%)"].round(4)
            m_df["MAE (%)"]  = m_df["MAE (%)"].round(4)
            m_df["OOS R²"]   = m_df["OOS R²"].round(4)
            st.markdown("#### Walk-Forward OOS Performance")
            st.dataframe(m_df, use_container_width=True)

        fig_cmp = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                subplot_titles=["OOS Predictions vs Actual",
                                                "Prediction Errors (OLS vs Ridge)"])
        fig_cmp.add_trace(go.Scatter(x=oos_dates, y=oos_actual*100, name="Actual",
            line=dict(color="#00D4FF", width=1.2), opacity=0.8), row=1, col=1)
        for preds, col_p, nm_p in [
            (pred_ols,"#F59E0B","OLS"), (pred_ridge,"#10B981","Ridge"),
            (pred_lasso,"#EF4444","LASSO"), (pred_pcr,"#8B5CF6","PCR"),
        ]:
            fig_cmp.add_trace(go.Scatter(x=oos_dates, y=np.array(preds)*100, name=nm_p,
                line=dict(width=1.2)), row=1, col=1)
        fig_cmp.add_trace(go.Bar(x=oos_dates,
            y=(np.array(pred_ols)-oos_actual)*100, name="OLS error",
            marker=dict(color="#F59E0B", opacity=0.5, line=dict(width=0))), row=2, col=1)
        fig_cmp.add_trace(go.Bar(x=oos_dates,
            y=(np.array(pred_ridge)-oos_actual)*100, name="Ridge error",
            marker=dict(color="#10B981", opacity=0.5, line=dict(width=0))), row=2, col=1)
        fig_cmp.update_layout(title=f"Walk-Forward Model Comparison: {fc_dep}",
                               height=480, barmode="overlay", **lay())
        fig_cmp.update_xaxes(**ax()); fig_cmp.update_yaxes(**ax())
        st.plotly_chart(fig_cmp, use_container_width=True)

    # 10D — E-TRADING IMPLEMENTATION
    with st10d:
        st.markdown("""
        <div class="fbox">
        <strong>Linearisation:</strong> PV(x+Δx) ≈ PV(x) + Σ(∂PV/∂xᵢ)Δxᵢ (fast repricing) &nbsp;|&nbsp;
        <strong>Bid-Offer:</strong> spread = f(realised vol, turnover, inventory) &nbsp;|&nbsp;
        <strong>Fast/Slow:</strong> fast = Greeks × Δx  ·  slow = full revaluation
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns([1, 2])
        with cl:
            et_t     = st.selectbox("Stock for e-trading", selected, key="et_t")
            inv_pos  = st.slider("Inventory position (units)", -1000, 1000, 0, 50, key="et_inv")
            base_spd = st.slider("Base bid-offer spread (bps)", 5, 100, 20, 5, key="et_spd")

        # Align returns and prices on the same date index to avoid shape mismatch
        y_et_s    = R_df[et_t]                          # Series, length T_days
        price_s   = closes[et_t].reindex(y_et_s.index) # align prices to returns index
        y_et      = y_et_s.values
        price_et  = price_s.ffill().values
        vol_et    = pd.Series(y_et).rolling(20).std().fillna(y_et.std()).values * np.sqrt(252)

        # Linearisation: delta = daily return × price; approximate PnL
        delta_et  = y_et * price_et  # shapes now guaranteed equal
        slow_pnl  = np.cumsum(delta_et)
        # fast = delta × return (first-order approximation)
        fast_pnl  = np.cumsum(y_et * price_et[0])   # simplified: use initial price

        # Bid-offer spread model: base_spread * vol_factor * inventory_factor
        vol_factor  = vol_et / np.maximum(vol_et.mean(), 1e-12)
        inv_factor  = 1 + abs(inv_pos) / 5000   # wider spread if large inventory
        spread_bps  = base_spd * vol_factor * inv_factor

        # Risk monitoring: rolling VaR
        rolling_var = pd.Series(y_et).rolling(20).quantile(0.05).fillna(float(np.quantile(y_et, 0.05))).values

        with cr:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Vol (Ann)",    f"{vol_et[-1]*100:.1f}%")
            c2.metric("Current Spread (bps)", f"{spread_bps[-1]:.1f}")
            c3.metric("Inventory Position",   f"{inv_pos:+d}")
            c4.metric("Rolling VaR 95%",      f"{rolling_var[-1]*100:.3f}%")

        fig_et = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                               subplot_titles=["Fast vs Slow P&L (Linearised vs Full)",
                                               "Dynamic Bid-Offer Spread (bps)",
                                               "Real-Time Risk Monitor: Rolling VaR 95%"],
                               row_heights=[0.4, 0.3, 0.3])
        fig_et.add_trace(go.Scatter(x=dates, y=slow_pnl, name="Slow (Full)",
            line=dict(color="#00D4FF", width=1.5)), row=1, col=1)
        fig_et.add_trace(go.Scatter(x=dates, y=fast_pnl, name="Fast (Linear)",
            line=dict(color="#F59E0B", width=1.5, dash="dash")), row=1, col=1)
        fig_et.add_trace(go.Scatter(x=dates, y=spread_bps, name="Spread",
            line=dict(color="#10B981", width=1.2),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.1)"), row=2, col=1)
        fig_et.add_trace(go.Scatter(x=dates, y=rolling_var*100, name="VaR 95%",
            line=dict(color="#EF4444", width=1.5),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.1)"), row=3, col=1)
        fig_et.update_layout(title=f"E-Trading Implementation — {et_t}  |  Inventory={inv_pos:+d}",
                              height=580, **lay())
        fig_et.update_xaxes(**ax()); fig_et.update_yaxes(**ax())
        st.plotly_chart(fig_et, use_container_width=True)

        # Spread decomposition table
        st.markdown("#### Bid-Offer Spread Decomposition")
        spread_df = pd.DataFrame({
            "Component":     ["Base Spread", "Vol Adjustment", "Inventory Penalty", "Total Spread"],
            "Formula":       ["Base bps", "× vol/avg_vol", f"× (1+|{inv_pos}|/5000)", "Product of all"],
            "Current Value": [f"{base_spd:.0f} bps",
                               f"×{vol_et[-1]/vol_et.mean():.3f}",
                               f"×{inv_factor:.3f}",
                               f"{spread_bps[-1]:.1f} bps"],
            "Notes": [
                "Market maker's minimum required spread",
                "Wider spread in high-vol markets to cover adverse selection",
                "Larger inventory → wider spread to discourage same-side flow",
                "Final quoted spread used in electronic pricing engine",
            ]
        })
        st.dataframe(spread_df, use_container_width=True)


with tab11:
    st.markdown('<div class="sec-hdr">11. Advanced Volatility Models — HLOC Estimators · GARCH · Stochastic Vol · Jump Diffusion · Forecasting</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="fbox">
    <strong>Parkinson:</strong> σ² = (H−L)²/(4·ln2) &nbsp;|&nbsp;
    <strong>GARCH(1,1):</strong> σₜ² = α₀ + α₁ε²ₜ₋₁ + β₁σ²ₜ₋₁ &nbsp;|&nbsp;
    <strong>Heston:</strong> dvₜ = κ(θ−vₜ)dt + ξ√vₜ dWₜ &nbsp;|&nbsp;
    <strong>Merton Jump:</strong> dS/S = μdt + σdW + J·dN
    </div>""", unsafe_allow_html=True)

    from scipy import stats as sp11

    st11a, st11b, st11c, st11d, st11e = st.tabs([
        "📏 HLOC Estimators",
        "📊 ARCH / GARCH",
        "🌀 Stochastic Volatility",
        "💥 Jump Diffusion",
        "🔮 Vol Forecasting",
    ])

    def _v11_stock(key):
        return st.selectbox("Stock", selected, key=key)

    # 11A — HLOC ESTIMATORS
    with st11a:
        st.markdown("""
        <div class="fbox">
        <strong>Parkinson:</strong> σ²_P = (1/4ln2)·(lnH−lnL)² &nbsp;|&nbsp;
        <strong>Garman-Klass:</strong> σ²_GK = 0.5(lnH−lnL)² − (2ln2−1)(lnC−lnO)² &nbsp;|&nbsp;
        <strong>Rogers-Satchell:</strong> σ²_RS = lnH(lnH−lnC) + lnL(lnL−lnC) &nbsp;|&nbsp;
        <strong>Yang-Zhang:</strong> σ²_YZ = σ²_O + k·σ²_C + (1−k)·σ²_RS  (Eff≈8.4)
        </div>""", unsafe_allow_html=True)

        v_t   = _v11_stock("v11a_t")
        roll_v = st.slider("Rolling window (days)", 5, 60, 20, key="v11a_w")
        df_hl = all_dfs[v_t].copy()
        # guard: need OHLC
        has_ohlc = all(c in df_hl.columns for c in ["Open","High","Low","Close"])
        if not has_ohlc:
            df_hl["Open"]  = df_hl["Close"] * (1 + np.random.normal(0, 0.003, len(df_hl)))
            df_hl["High"]  = df_hl["Close"] * np.exp(np.abs(np.random.normal(0, 0.006, len(df_hl))))
            df_hl["Low"]   = df_hl["Close"] * np.exp(-np.abs(np.random.normal(0, 0.006, len(df_hl))))

        O = np.log(df_hl["Open"].values.clip(1e-9))
        H = np.log(df_hl["High"].values.clip(1e-9))
        L = np.log(df_hl["Low"].values.clip(1e-9))
        C = np.log(df_hl["Close"].values.clip(1e-9))
        C_prev = np.concatenate([[C[0]], C[:-1]])

        # Daily estimates
        park  = (H - L)**2 / (4 * np.log(2))
        gk    = 0.5*(H-L)**2 - (2*np.log(2)-1)*(C-O)**2
        rs    = (H-C)*(H-O) + (L-C)*(L-O)
        # Yang-Zhang: overnight + close-to-close + RS
        o_var = (O - C_prev)**2
        c_var = (C - C_prev)**2
        k_yz  = 0.34 / (1.34 + (roll_v+1)/(roll_v-1))
        yz    = o_var + k_yz*c_var + (1-k_yz)*rs

        # close-to-close reference
        cc    = (C - C_prev)**2

        def rolling_ann(arr, w):
            return pd.Series(np.maximum(arr, 0)).rolling(w).mean().values * 252

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Parkinson σ (Ann)",    f"{np.sqrt(rolling_ann(park, roll_v)[-1])*100:.2f}%")
        m2.metric("Garman-Klass σ (Ann)", f"{np.sqrt(np.maximum(rolling_ann(gk,roll_v)[-1],0))*100:.2f}%")
        m3.metric("Rogers-Satchell σ",    f"{np.sqrt(np.maximum(rolling_ann(rs,roll_v)[-1],0))*100:.2f}%")
        m4.metric("Yang-Zhang σ",         f"{np.sqrt(np.maximum(rolling_ann(yz,roll_v)[-1],0))*100:.2f}%")

        fig_hl = go.Figure()
        idx_hl = df_hl.index
        for label, arr, col in [
            ("Close-to-Close", cc,  "#475569"),
            ("Parkinson",      park,"#00D4FF"),
            ("Garman-Klass",   gk,  "#10B981"),
            ("Rogers-Satchell",rs,  "#F59E0B"),
            ("Yang-Zhang",     yz,  "#8B5CF6"),
        ]:
            rv = rolling_ann(arr, roll_v)
            fig_hl.add_trace(go.Scatter(x=idx_hl, y=np.sqrt(np.maximum(rv, 0))*100,
                name=label, line=dict(color=col, width=1.6)))
        fig_hl.update_layout(title=f"{v_t}  —  HLOC Volatility Estimators ({roll_v}d rolling, annualised)",
                              height=400, **lay())
        fig_hl.update_xaxes(**ax()); fig_hl.update_yaxes(title_text="Annualised Vol (%)", **ax())
        st.plotly_chart(fig_hl, use_container_width=True)

        # Efficiency comparison bar
        st.markdown("#### Estimator Efficiency  (relative to Close-to-Close)")
        last_cc = rolling_ann(cc, roll_v)[-1]
        effs = {}
        for lbl, arr in [("Parkinson",park),("Garman-Klass",gk),("Rogers-Satchell",rs),("Yang-Zhang",yz)]:
            last = rolling_ann(arr, roll_v)[-1]
            effs[lbl] = last_cc / max(last, 1e-12)
        fig_eff = go.Figure(go.Bar(
            x=list(effs.keys()), y=list(effs.values()),
            marker=dict(color=["#00D4FF","#10B981","#F59E0B","#8B5CF6"], opacity=0.85),
            text=[f"{v:.2f}x" for v in effs.values()], textposition="outside",
        ))
        fig_eff.add_hline(y=1.0, line=dict(color="#EF4444", dash="dash"), annotation_text="C-C baseline")
        fig_eff.update_layout(title="Relative Efficiency vs Close-to-Close (higher = more efficient)",
                               height=300, showlegend=False, **lay())
        fig_eff.update_xaxes(**ax()); fig_eff.update_yaxes(title_text="Efficiency Ratio", **ax())
        st.plotly_chart(fig_eff, use_container_width=True)

    # 11B — ARCH / GARCH
    with st11b:
        st.markdown("""
        <div class="fbox">
        <strong>ARCH(p):</strong> σₜ² = α₀ + Σᵢα_ᵢε²ₜ₋ᵢ &nbsp;|&nbsp;
        <strong>GARCH(1,1):</strong> σₜ² = α₀ + α₁ε²ₜ₋₁ + β₁σ²ₜ₋₁ &nbsp;|&nbsp;
        <strong>EGARCH:</strong> ln σₜ² = α₀ + α₁(|zₜ₋₁|−E|z|) + γ₁zₜ₋₁ + β₁ln σ²ₜ₋₁ &nbsp;|&nbsp;
        <strong>GJR-GARCH:</strong> σₜ² = α₀ + (α₁ + γ𝟙{εₜ₋₁<0})ε²ₜ₋₁ + β₁σ²ₜ₋₁
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns([1, 2])
        with cl:
            g_t = _v11_stock("v11b_t")
            garch_p = st.slider("ARCH order p", 1, 5, 1, key="g_p")
            garch_q = st.slider("GARCH order q", 0, 3, 1, key="g_q")

        ret_g = R_df[g_t].values
        n_g   = len(ret_g)

        # ── Manual GARCH(1,1) via moment matching (closed-form approx) ────
        eps2 = ret_g**2
        # ARCH(1) estimate via OLS on squared returns
        y_arch = eps2[1:]
        X_arch = np.column_stack([np.ones(n_g-1), eps2[:-1]])
        b_arch = np.linalg.lstsq(X_arch, y_arch, rcond=None)[0]
        a0_est = max(b_arch[0], 1e-8)
        a1_est = np.clip(b_arch[1], 0, 0.45)

        # GARCH(1,1): persistence β from unconditional variance
        uncond_var = eps2.var()
        b1_est = max(0, min(0.94, 1 - a0_est/uncond_var - a1_est))

        # Recursively compute GARCH conditional variance
        sigma2 = np.full(n_g, uncond_var)
        for i in range(1, n_g):
            sigma2[i] = a0_est + a1_est*eps2[i-1] + b1_est*sigma2[i-1]
        sigma2 = np.maximum(sigma2, 1e-10)

        # GJR-GARCH: add asymmetric term γ for negative shocks
        gamma_gjr = 0.08  # typical value
        sigma2_gjr = np.full(n_g, uncond_var)
        for i in range(1, n_g):
            ind_neg = 1.0 if ret_g[i-1] < 0 else 0.0
            sigma2_gjr[i] = (a0_est
                             + (a1_est + gamma_gjr*ind_neg)*eps2[i-1]
                             + b1_est*sigma2_gjr[i-1])
        sigma2_gjr = np.maximum(sigma2_gjr, 1e-10)

        # EGARCH
        log_sigma2_eg = np.full(n_g, np.log(uncond_var))
        eg_a0, eg_a1, eg_g, eg_b = a0_est, 0.12, -0.08, b1_est
        for i in range(1, n_g):
            z_t = ret_g[i-1] / np.sqrt(np.exp(log_sigma2_eg[i-1]))
            log_sigma2_eg[i] = (eg_a0
                                + eg_a1*(abs(z_t) - np.sqrt(2/np.pi))
                                + eg_g*z_t
                                + eg_b*log_sigma2_eg[i-1])
        sigma2_eg = np.exp(log_sigma2_eg)

        std_resid = ret_g / np.sqrt(sigma2)
        persist   = a1_est + b1_est

        with cr:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("α₀ (intercept)",    f"{a0_est:.2e}")
            c2.metric("α₁ (ARCH)",         f"{a1_est:.4f}")
            c3.metric("β₁ (GARCH)",        f"{b1_est:.4f}")
            c4.metric("Persistence α+β",   f"{persist:.4f}",
                       delta="Non-stationary" if persist >= 1 else "Stationary")

        # Conditional vol comparison
        fig_garch = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                   subplot_titles=["Returns vs GARCH Conditional Volatility",
                                                   "GARCH / EGARCH / GJR-GARCH Comparison"])
        fig_garch.add_trace(go.Scatter(x=dates, y=ret_g*100, mode="lines", name="Returns",
            line=dict(color="#475569", width=0.8), opacity=0.6), row=1, col=1)
        fig_garch.add_trace(go.Scatter(x=dates, y=np.sqrt(sigma2*252)*100,
            name="GARCH σ (ann)", line=dict(color="#00D4FF", width=2)), row=1, col=1)
        fig_garch.add_trace(go.Scatter(x=dates, y=-np.sqrt(sigma2*252)*100,
            name="-GARCH σ", line=dict(color="#00D4FF", width=2), showlegend=False), row=1, col=1)

        for sv, col_v, nm_v in [(sigma2,"#00D4FF","GARCH(1,1)"),
                                  (sigma2_gjr,"#F59E0B","GJR-GARCH"),
                                  (sigma2_eg,"#10B981","EGARCH")]:
            fig_garch.add_trace(go.Scatter(x=dates, y=np.sqrt(sv*252)*100,
                name=nm_v, line=dict(color=col_v, width=1.5)), row=2, col=1)

        fig_garch.update_layout(title=f"{g_t}  —  GARCH-family Conditional Volatility",
                                 height=520, **lay())
        fig_garch.update_xaxes(**ax()); fig_garch.update_yaxes(**ax())
        st.plotly_chart(fig_garch, use_container_width=True)

        # Diagnostics: standardised residuals
        st.markdown("#### GARCH Diagnostics — Standardised Residuals")
        fig_diag = make_subplots(rows=1, cols=3,
            subplot_titles=["Std Residuals", "QQ Plot", "ACF of ε²ₜ (Ljung-Box)"])
        fig_diag.add_trace(go.Scatter(x=dates, y=std_resid, mode="lines",
            line=dict(color="#00D4FF", width=0.8), name="zₜ"), row=1, col=1)
        fig_diag.add_hline(y=2,  line=dict(color="#EF4444", dash="dot"), row=1, col=1)
        fig_diag.add_hline(y=-2, line=dict(color="#EF4444", dash="dot"), row=1, col=1)
        # QQ
        sr_s = np.sort(std_resid)
        qq_t = sp11.norm.ppf([(i-.375)/(n_g+.25) for i in range(1,n_g+1)])
        fig_diag.add_trace(go.Scatter(x=qq_t, y=sr_s, mode="markers",
            marker=dict(color="#8B5CF6", size=2, opacity=0.6), name="QQ"), row=1, col=2)
        fig_diag.add_trace(go.Scatter(x=[qq_t[0],qq_t[-1]], y=[qq_t[0],qq_t[-1]],
            mode="lines", line=dict(color="#EF4444", dash="dash"), showlegend=False), row=1, col=2)
        # ACF of squared std resid
        sr2 = std_resid**2
        lags_d = list(range(1, 16))
        acf_d  = [pd.Series(sr2).autocorr(lag=k) for k in lags_d]
        conf_d = 1.96/np.sqrt(n_g)
        fig_diag.add_trace(go.Bar(x=lags_d, y=acf_d, name="ACF ε²",
            marker=dict(color=["#EF4444" if abs(a)>conf_d else "#10B981" for a in acf_d],
                        opacity=0.85)), row=1, col=3)
        fig_diag.add_hline(y=conf_d,  line=dict(color="#475569",dash="dot"), row=1, col=3)
        fig_diag.add_hline(y=-conf_d, line=dict(color="#475569",dash="dot"), row=1, col=3)
        fig_diag.update_layout(title="GARCH(1,1) Diagnostics", height=360,
                                showlegend=False, **lay())
        fig_diag.update_xaxes(**ax()); fig_diag.update_yaxes(**ax())
        st.plotly_chart(fig_diag, use_container_width=True)

        # Model selection info table
        aic_g   = n_g*np.log(sigma2.mean()) + 2*3   # 3 params: a0,a1,b1
        bic_g   = n_g*np.log(sigma2.mean()) + 3*np.log(n_g)
        aic_gjr = n_g*np.log(sigma2_gjr.mean()) + 2*4
        bic_gjr = n_g*np.log(sigma2_gjr.mean()) + 4*np.log(n_g)
        ms_df = pd.DataFrame({
            "Model":      ["GARCH(1,1)","GJR-GARCH","EGARCH"],
            "Parameters": [3, 4, 4],
            "AIC (approx)":[f"{aic_g:.1f}",f"{aic_gjr:.1f}",f"{n_g*np.log(sigma2_eg.mean())+2*4:.1f}"],
            "BIC (approx)":[f"{bic_g:.1f}",f"{bic_gjr:.1f}",f"{n_g*np.log(sigma2_eg.mean())+4*np.log(n_g):.1f}"],
            "Persist.":   [f"{persist:.4f}",f"{a1_est+gamma_gjr/2+b1_est:.4f}",f"{eg_b:.4f}"],
        })
        st.dataframe(ms_df, use_container_width=True)

    # 11C — STOCHASTIC VOLATILITY
    with st11c:
        st.markdown("""
        <div class="fbox">
        <strong>Heston:</strong> dvₜ = κ(θ−vₜ)dt + ξ√vₜ dWₜ² &nbsp;|&nbsp;
        <strong>SABR:</strong> dσₜ = ασₜ dWₜ¹, dFₜ = σₜFₜ^β dWₜ² &nbsp;|&nbsp;
        <strong>Feller condition:</strong> 2κθ > ξ² (vₜ stays positive)
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns([1, 2])
        with cl:
            sv_t   = _v11_stock("v11c_t")
            kappa  = st.slider("κ (mean reversion speed)", 0.1, 5.0, 1.5, 0.1, key="sv_k")
            theta  = st.slider("θ (long-run variance ×100)", 0.01, 0.10, 0.04, 0.005, key="sv_th")
            xi     = st.slider("ξ (vol of vol)", 0.1, 1.0, 0.3, 0.05, key="sv_xi")
            rho_sv = st.slider("ρ (correlation dS,dv)", -0.9, 0.0, -0.7, 0.05, key="sv_rho")
            n_paths= st.slider("MC paths", 100, 2000, 500, 100, key="sv_np")

        # Euler-Maruyama simulation of Heston
        np.random.seed(7)
        S0_sv = float(closes[sv_t].iloc[-1])
        v0_sv = float(R_df[sv_t].var() * 252)
        dt_sv = 1/252; steps_sv = 252

        S_paths = np.zeros((n_paths, steps_sv))
        v_paths = np.zeros((n_paths, steps_sv))
        S_paths[:,0] = S0_sv; v_paths[:,0] = v0_sv

        for i in range(1, steps_sv):
            Z1 = np.random.normal(0, 1, n_paths)
            Z2 = rho_sv*Z1 + np.sqrt(1-rho_sv**2)*np.random.normal(0,1,n_paths)
            v_paths[:,i] = np.maximum(
                v_paths[:,i-1] + kappa*(theta - v_paths[:,i-1])*dt_sv
                + xi*np.sqrt(np.maximum(v_paths[:,i-1],0)*dt_sv)*Z2, 1e-8)
            S_paths[:,i] = S_paths[:,i-1] * np.exp(
                (rf_rate - 0.5*v_paths[:,i-1])*dt_sv
                + np.sqrt(np.maximum(v_paths[:,i-1],0)*dt_sv)*Z1)

        feller_ok = 2*kappa*theta > xi**2
        with cr:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Feller Condition", "✓ Met" if feller_ok else "✗ Violated",
                       delta="2κθ > ξ²" if feller_ok else "v may hit 0")
            c2.metric("Long-run Vol (√θ)", f"{np.sqrt(theta)*100:.2f}%")
            c3.metric("Half-life (ln2/κ)", f"{np.log(2)/kappa*252:.0f} days")
            c4.metric("S₁ₒₒ Median",       f"{np.median(S_paths[:,-1]):.2f}")

        fig_sv = make_subplots(rows=1, cols=2,
            subplot_titles=[f"Heston Price Paths ({n_paths} MC)",
                             "Variance Process vₜ"])
        t_sv = np.arange(steps_sv)/252
        # Plot subset of paths for clarity
        for p_i in range(min(n_paths, 80)):
            alpha_p = 0.08
            col_p   = f"rgba(0,212,255,{alpha_p})"
            fig_sv.add_trace(go.Scatter(x=t_sv, y=S_paths[p_i], mode="lines",
                line=dict(color=col_p, width=0.5), showlegend=False), row=1, col=1)
            fig_sv.add_trace(go.Scatter(x=t_sv, y=np.sqrt(v_paths[p_i])*100, mode="lines",
                line=dict(color=f"rgba(245,158,11,{alpha_p})", width=0.5),
                showlegend=False), row=1, col=2)
        # median + percentile bands
        s_med = np.median(S_paths, axis=0)
        s_p05 = np.percentile(S_paths, 5, axis=0)
        s_p95 = np.percentile(S_paths, 95, axis=0)
        v_med = np.median(np.sqrt(v_paths)*100, axis=0)
        fig_sv.add_trace(go.Scatter(x=t_sv, y=s_med, name="Median",
            line=dict(color="#00D4FF", width=2.5)), row=1, col=1)
        fig_sv.add_trace(go.Scatter(x=list(t_sv)+list(t_sv[::-1]),
            y=list(s_p95)+list(s_p05[::-1]),
            fill="toself", fillcolor="rgba(0,212,255,0.08)",
            line=dict(width=0), name="5-95%"), row=1, col=1)
        fig_sv.add_trace(go.Scatter(x=t_sv, y=v_med, name="Median Vol",
            line=dict(color="#F59E0B", width=2.5)), row=1, col=2)
        fig_sv.add_hline(y=np.sqrt(theta)*100,
            line=dict(color="#10B981", dash="dash"),
            annotation_text=f"√θ={np.sqrt(theta)*100:.1f}%", row=1, col=2)
        fig_sv.update_layout(title=f"Heston Stochastic Volatility Model — {sv_t}",
                              height=420, **lay())
        fig_sv.update_xaxes(title_text="Years", **ax())
        fig_sv.update_yaxes(**ax())
        st.plotly_chart(fig_sv, use_container_width=True)

    # 11D — JUMP DIFFUSION
    with st11d:
        st.markdown("""
        <div class="fbox">
        <strong>Merton:</strong> dS/S = (μ − λk̄)dt + σdW + JdN(λ) &nbsp;|&nbsp;
        <strong>Jump dist:</strong> ln(1+J) ~ N(μⱼ, σⱼ²) &nbsp;|&nbsp;
        <strong>k̄ = E[J] = e^{μⱼ+σⱼ²/2} − 1</strong> &nbsp;|&nbsp;
        <strong>Kou:</strong> p·Exp(η₁) + (1−p)·Exp(η₂) asymmetric jumps
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns([1, 2])
        with cl:
            jd_t    = _v11_stock("v11d_t")
            lam_j   = st.slider("Jump intensity λ (per year)", 0.5, 20.0, 5.0, 0.5, key="jd_l")
            mu_j    = st.slider("Jump mean μⱼ (%)", -10.0, 2.0, -3.0, 0.5, key="jd_mj") / 100
            sig_j   = st.slider("Jump vol σⱼ (%)", 1.0, 15.0, 5.0, 0.5, key="jd_sj") / 100
            n_jd    = st.slider("MC paths", 100, 1000, 300, 100, key="jd_np")

        ret_jd = R_df[jd_t].values
        mu_jd  = float(ret_jd.mean() * 252)
        sig_jd = float(ret_jd.std() * np.sqrt(252))
        S0_jd  = float(closes[jd_t].iloc[-1])
        dt_jd  = 1/252; T_jd = 252
        kbar   = np.exp(mu_j + 0.5*sig_j**2) - 1

        np.random.seed(13)
        S_jd = np.zeros((n_jd, T_jd))
        S_jd[:,0] = S0_jd
        n_jumps_total = 0
        for i in range(1, T_jd):
            Z_diff = np.random.normal(0,1,n_jd)
            # Poisson jumps
            n_jmp  = np.random.poisson(lam_j*dt_jd, n_jd)
            n_jumps_total += n_jmp.sum()
            J_log  = np.array([np.sum(np.random.normal(mu_j, sig_j, k)) if k>0 else 0.0
                               for k in n_jmp])
            S_jd[:,i] = S_jd[:,i-1] * np.exp(
                (mu_jd - lam_j*kbar - 0.5*sig_jd**2)*dt_jd
                + sig_jd*np.sqrt(dt_jd)*Z_diff + J_log)

        # Detect jumps in actual data via threshold
        ret_std   = ret_jd.std()
        jump_mask = np.abs(ret_jd) > 3*ret_std
        n_detected = jump_mask.sum()
        est_lam_ann = n_detected / (len(ret_jd)/252)

        with cr:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Detected Jumps",     f"{n_detected}")
            c2.metric("Est λ (ann)",         f"{est_lam_ann:.1f}")
            c3.metric("k̄ = E[J]",           f"{kbar*100:.2f}%")
            c4.metric("MC Jumps (avg/path)", f"{n_jumps_total/n_jd:.1f}")

        fig_jd = make_subplots(rows=1, cols=2,
            subplot_titles=[f"Merton Jump-Diffusion Paths ({n_jd} MC)",
                             "Jump Detection in Actual Returns"])
        for p_i in range(min(n_jd, 60)):
            fig_jd.add_trace(go.Scatter(x=list(range(T_jd)), y=S_jd[p_i], mode="lines",
                line=dict(color="rgba(0,212,255,0.07)", width=0.5),
                showlegend=False), row=1, col=1)
        jd_med = np.median(S_jd, axis=0)
        jd_p05 = np.percentile(S_jd, 5, axis=0)
        jd_p95 = np.percentile(S_jd, 95, axis=0)
        fig_jd.add_trace(go.Scatter(x=list(range(T_jd)), y=jd_med, name="Median",
            line=dict(color="#00D4FF", width=2.5)), row=1, col=1)
        fig_jd.add_trace(go.Scatter(
            x=list(range(T_jd))+list(range(T_jd-1,-1,-1)),
            y=list(jd_p95)+list(jd_p05[::-1]),
            fill="toself", fillcolor="rgba(0,212,255,0.08)",
            line=dict(width=0), name="5-95% CI"), row=1, col=1)
        # Actual returns with jump highlights
        fig_jd.add_trace(go.Scatter(x=dates, y=ret_jd*100, mode="lines",
            line=dict(color="#475569",width=0.8), name="Returns"), row=1, col=2)
        fig_jd.add_trace(go.Scatter(
            x=dates[jump_mask], y=ret_jd[jump_mask]*100, mode="markers",
            marker=dict(color="#EF4444", size=7, symbol="x"),
            name="Detected Jumps (|r|>3σ)"), row=1, col=2)
        fig_jd.add_hline(y=3*ret_std*100,  line=dict(color="#EF4444",dash="dot"), row=1, col=2)
        fig_jd.add_hline(y=-3*ret_std*100, line=dict(color="#EF4444",dash="dot"), row=1, col=2)
        fig_jd.update_layout(title=f"Jump Diffusion Model — {jd_t}", height=400, **lay())
        fig_jd.update_xaxes(**ax()); fig_jd.update_yaxes(**ax())
        st.plotly_chart(fig_jd, use_container_width=True)

    # 11E — VOLATILITY FORECASTING
    with st11e:
        st.markdown("""
        <div class="fbox">
        <strong>QLIKE:</strong> Σ[σ²ₜ/RV_t − ln(σ²ₜ/RVₜ) − 1] &nbsp;|&nbsp;
        <strong>Mincer-Zarnowitz:</strong> RVₜ = α + β·σ̂²ₜ + εₜ (β=1 if unbiased) &nbsp;|&nbsp;
        <strong>BMA weights:</strong> wᵢ ∝ exp(−BICᵢ/2)
        </div>""", unsafe_allow_html=True)

        vf_t   = _v11_stock("v11e_t")
        h_fore = st.slider("Forecast horizon (days)", 1, 30, 10, key="vf_h")

        ret_vf = R_df[vf_t].values
        n_vf   = len(ret_vf)
        RV_true = ret_vf**2  # proxy for realised variance

        # Rolling 20-day forecast
        roll_fc = pd.Series(ret_vf).rolling(20).var().fillna(ret_vf.var()).values

        # GARCH(1,1) variance (from 11B logic)
        eps2_vf = ret_vf**2
        X_vf = np.column_stack([np.ones(n_vf-1), eps2_vf[:-1]])
        b_vf = np.linalg.lstsq(X_vf, eps2_vf[1:], rcond=None)[0]
        a0_vf = max(b_vf[0], 1e-8); a1_vf = np.clip(b_vf[1], 0, 0.45)
        b1_vf = max(0, min(0.94, 1 - a0_vf/eps2_vf.var() - a1_vf))
        sigma2_vf = np.full(n_vf, eps2_vf.var())
        for i in range(1, n_vf):
            sigma2_vf[i] = a0_vf + a1_vf*eps2_vf[i-1] + b1_vf*sigma2_vf[i-1]
        sigma2_vf = np.maximum(sigma2_vf, 1e-10)

        # BMA: combine rolling and GARCH with simple inverse-MSE weights
        mse_roll  = np.mean((roll_fc - RV_true)**2)
        mse_garch = np.mean((sigma2_vf - RV_true)**2)
        w_roll  = 1/max(mse_roll, 1e-20);  w_garch = 1/max(mse_garch, 1e-20)
        w_sum   = w_roll + w_garch
        bma_fc  = (w_roll*roll_fc + w_garch*sigma2_vf) / w_sum

        # Forecast evaluation
        def qlike(fcast, rv):
            r = np.maximum(fcast, 1e-15)
            return np.mean(r/np.maximum(rv,1e-15) - np.log(r/np.maximum(rv,1e-15)) - 1)

        def mz_regression(fcast, rv):
            X_mz = np.column_stack([np.ones(len(rv)), fcast])
            b_mz = np.linalg.lstsq(X_mz, rv, rcond=None)[0]
            return b_mz[0], b_mz[1]  # alpha, beta

        mse_r  = np.mean((roll_fc - RV_true)**2)
        mse_g  = np.mean((sigma2_vf - RV_true)**2)
        mse_b  = np.mean((bma_fc - RV_true)**2)
        q_r    = qlike(roll_fc, RV_true)
        q_g    = qlike(sigma2_vf, RV_true)
        q_b    = qlike(bma_fc, RV_true)
        mz_ar, mz_br = mz_regression(roll_fc, RV_true)
        mz_ag, mz_bg = mz_regression(sigma2_vf, RV_true)
        mz_ab, mz_bb = mz_regression(bma_fc, RV_true)

        c1,c2,c3 = st.columns(3)
        for col_m, mse_v, q_v, mz_a, mz_b, nm_v in [
            (c1, mse_r, q_r, mz_ar, mz_br, "Rolling"),
            (c2, mse_g, q_g, mz_ag, mz_bg, "GARCH"),
            (c3, mse_b, q_b, mz_ab, mz_bb, "BMA"),
        ]:
            col_m.markdown(f"""
            <div class="fbox"><strong>{nm_v}</strong><br>
            MSE: {mse_v:.2e}<br>QLIKE: {q_v:.4f}<br>
            MZ α={mz_a:.4f}, β={mz_b:.4f}</div>""", unsafe_allow_html=True)

        fig_vf = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            subplot_titles=["Volatility Forecasts vs Realised",
                             f"GARCH {h_fore}-Day Ahead Forecast with CI"])
        for fc_v, col_v, nm_v in [(np.sqrt(roll_fc*252)*100,"#00D4FF","Rolling"),
                                    (np.sqrt(sigma2_vf*252)*100,"#F59E0B","GARCH"),
                                    (np.sqrt(bma_fc*252)*100,"#10B981","BMA")]:
            fig_vf.add_trace(go.Scatter(x=dates, y=fc_v, name=nm_v,
                line=dict(color=col_v,width=1.5)), row=1, col=1)
        fig_vf.add_trace(go.Scatter(x=dates, y=np.abs(ret_vf)*np.sqrt(252)*100,
            name="|r|√252", line=dict(color="#475569",width=0.8), opacity=0.5), row=1, col=1)

        # GARCH h-step ahead: σ²_{t+h} = ω/(1-α-β) + (α+β)^h·(σ²_t - ω/(1-α-β))
        omega_vf  = a0_vf / max(1 - a1_vf - b1_vf, 1e-8)
        last_sig2 = float(sigma2_vf[-1])
        h_vals    = list(range(1, h_fore+1))
        persist_h = a1_vf + b1_vf
        fc_h      = [omega_vf + persist_h**h*(last_sig2 - omega_vf) for h in h_vals]
        fc_ci_h   = [np.sqrt(max(fv,0)*252)*100 * 1.96 * np.sqrt(h) / np.sqrt(252)
                     for h,fv in enumerate(fc_h, 1)]
        fc_h_ann  = [np.sqrt(max(fv,0)*252)*100 for fv in fc_h]
        bday_off  = pd.tseries.offsets.BusinessDay(1)
        fc_dates  = [dates[-1] + bday_off*h for h in h_vals]
        fig_vf.add_trace(go.Scatter(x=fc_dates, y=fc_h_ann, mode="lines+markers",
            name="GARCH Forecast", line=dict(color="#F59E0B",width=2.5,dash="dot"),
            marker=dict(size=6)), row=2, col=1)
        fig_vf.add_trace(go.Scatter(
            x=fc_dates+fc_dates[::-1],
            y=[v+c for v,c in zip(fc_h_ann,fc_ci_h)]+
              [v-c for v,c in zip(fc_h_ann[::-1],fc_ci_h[::-1])],
            fill="toself", fillcolor="rgba(245,158,11,0.1)",
            line=dict(width=0), name="95% CI"), row=2, col=1)
        fig_vf.add_hline(y=np.sqrt(omega_vf*252)*100,
            line=dict(color="#EF4444",dash="dash"),
            annotation_text="Uncond. Vol", row=2, col=1)
        fig_vf.update_layout(title=f"{vf_t}  —  Volatility Forecasting", height=500, **lay())
        fig_vf.update_xaxes(**ax()); fig_vf.update_yaxes(title_text="Ann Vol (%)", **ax())
        st.plotly_chart(fig_vf, use_container_width=True)


with tab12:
    st.markdown('<div class="sec-hdr">12. Stochastic Calculus & SDEs — Itô Calculus · SDE Zoo · Numerical Methods · Asset Pricing Applications</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="fbox">
    <strong>Itô Lemma:</strong> df = (f_t + μf_x + ½σ²f_xx)dt + σf_x dW &nbsp;|&nbsp;
    <strong>Euler-Maruyama:</strong> Xₜ₊Δ = Xₜ + μ(Xₜ)Δt + σ(Xₜ)√Δt·Z &nbsp;|&nbsp;
    <strong>Milstein:</strong> + ½σσ'(Z²−1)Δt &nbsp;|&nbsp;
    <strong>BS PDE:</strong> ∂V/∂t + rS∂V/∂S + ½σ²S²∂²V/∂S² = rV
    </div>""", unsafe_allow_html=True)

    from scipy import stats as sp12

    st12a, st12b, st12c, st12d = st.tabs([
        "∫ Itô Calculus",
        "📐 SDE Zoo",
        "🔢 Numerical Methods",
        "💎 Asset Pricing PDE",
    ])

    # 12A — ITÔ CALCULUS
    with st12a:
        st.markdown("""
        <div class="fbox">
        <strong>Itô Integral:</strong> ∫₀ᵀ f(Bₛ)dBₛ = lim Σ f(B_{tᵢ})(B_{tᵢ₊₁}−B_{tᵢ}) &nbsp;|&nbsp;
        <strong>Itô Isometry:</strong> E[(∫₀ᵀ f dB)²] = E[∫₀ᵀ f² dt] &nbsp;|&nbsp;
        <strong>Quadratic variation:</strong> [B,B]ₜ = t &nbsp;|&nbsp;
        [X,Y]ₜ = ∫₀ᵀ ρσ_Xσ_Y dt
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns([1, 2])
        with cl:
            n_bm     = st.slider("BM paths", 50, 500, 100, key="ito_n")
            T_bm     = st.slider("Horizon T (years)", 0.1, 2.0, 1.0, 0.1, key="ito_T")
            n_steps  = st.slider("Time steps", 100, 1000, 252, key="ito_steps")

        np.random.seed(42)
        dt_bm  = T_bm / n_steps
        t_bm   = np.linspace(0, T_bm, n_steps+1)
        dW     = np.random.normal(0, np.sqrt(dt_bm), (n_bm, n_steps))
        B_paths= np.hstack([np.zeros((n_bm,1)), np.cumsum(dW, axis=1)])

        # Itô integral: ∫₀ᵀ Bₛ dBₛ  (should equal ½(B_T² - T))
        ito_int = np.sum(B_paths[:,:-1]*dW, axis=1)
        ito_true = 0.5*(B_paths[:,-1]**2 - T_bm)

        with cr:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("E[B_T]",         f"{B_paths[:,-1].mean():.4f}")
            c2.metric("Var[B_T]",       f"{B_paths[:,-1].var():.4f}", delta=f"Expected: {T_bm:.2f}")
            c3.metric("E[∫BdB]",        f"{ito_int.mean():.4f}")
            c4.metric("Mean |error|",   f"{np.abs(ito_int - ito_true).mean():.6f}")

        fig_bm = make_subplots(rows=1, cols=2,
            subplot_titles=[f"Brownian Motion Paths ({n_bm})", "Itô Integral: ∫BdB vs ½(B²−T)"])
        for p_i in range(min(n_bm, 50)):
            fig_bm.add_trace(go.Scatter(x=t_bm, y=B_paths[p_i], mode="lines",
                line=dict(color="rgba(0,212,255,0.09)", width=0.6), showlegend=False), row=1, col=1)
        bm_med = np.median(B_paths, axis=0)
        bm_p05 = np.percentile(B_paths, 5, axis=0)
        bm_p95 = np.percentile(B_paths, 95, axis=0)
        fig_bm.add_trace(go.Scatter(x=t_bm, y=bm_med, name="Median",
            line=dict(color="#00D4FF",width=2)), row=1, col=1)
        fig_bm.add_trace(go.Scatter(
            x=list(t_bm)+list(t_bm[::-1]),
            y=list(bm_p95)+list(bm_p05[::-1]),
            fill="toself", fillcolor="rgba(0,212,255,0.07)",
            line=dict(width=0), name="5-95% CI"), row=1, col=1)
        # ±√t theoretical bounds
        fig_bm.add_trace(go.Scatter(x=t_bm, y=np.sqrt(t_bm), mode="lines",
            line=dict(color="#10B981",dash="dash"), name="±√t"), row=1, col=1)
        fig_bm.add_trace(go.Scatter(x=t_bm, y=-np.sqrt(t_bm), mode="lines",
            line=dict(color="#10B981",dash="dash"), showlegend=False), row=1, col=1)
        # Itô integral scatter
        fig_bm.add_trace(go.Scatter(x=ito_true, y=ito_int, mode="markers",
            marker=dict(color="#8B5CF6",size=4,opacity=0.5), name="Sample"), row=1, col=2)
        lo_v, hi_v = ito_true.min(), ito_true.max()
        fig_bm.add_trace(go.Scatter(x=[lo_v,hi_v], y=[lo_v,hi_v], mode="lines",
            line=dict(color="#EF4444",dash="dash"), name="y=x"), row=1, col=2)
        fig_bm.update_layout(title="Brownian Motion & Itô Integral Verification",
                              height=400, **lay())
        fig_bm.update_xaxes(**ax()); fig_bm.update_yaxes(**ax())
        st.plotly_chart(fig_bm, use_container_width=True)

        # Quadratic variation demonstration
        st.markdown("#### Quadratic Variation: [B,B]ₜ = t  (Demonstration)")
        qv_cumsum = np.cumsum(dW[0]**2)  # one path
        fig_qv = go.Figure()
        fig_qv.add_trace(go.Scatter(x=t_bm[1:], y=qv_cumsum, name="Empirical QV",
            line=dict(color="#00D4FF",width=2)))
        fig_qv.add_trace(go.Scatter(x=t_bm, y=t_bm, name="Theoretical: t",
            line=dict(color="#EF4444",dash="dash",width=2)))
        fig_qv.update_layout(title="Quadratic Variation [B,B]ₜ → t as Δt → 0",
                              height=280, **lay())
        fig_qv.update_xaxes(title_text="t", **ax())
        fig_qv.update_yaxes(title_text="QV", **ax())
        st.plotly_chart(fig_qv, use_container_width=True)

    # 12B — SDE ZOO
    with st12b:
        st.markdown("""
        <div class="fbox">
        <strong>GBM:</strong> dS = μS dt + σS dW &nbsp;|&nbsp;
        <strong>OU:</strong> dX = θ(μ−X) dt + σ dW &nbsp;|&nbsp;
        <strong>Vasicek:</strong> dr = a(b−r) dt + σ dW &nbsp;|&nbsp;
        <strong>CIR:</strong> dr = a(b−r) dt + σ√r dW &nbsp;|&nbsp;
        <strong>Heston variance:</strong> dv = κ(θ−v) dt + ξ√v dW
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns([1, 2])
        with cl:
            sde_model = st.selectbox("SDE Model",
                ["GBM","Ornstein-Uhlenbeck","Vasicek","CIR","Heston Variance"],
                key="sde_mdl")
            sde_paths = st.slider("Paths", 50, 500, 100, key="sde_np")
            sde_T     = st.slider("T (years)", 0.5, 5.0, 1.0, 0.5, key="sde_T")
            sde_steps = 252

        np.random.seed(3)
        dt_sde = sde_T / sde_steps
        t_sde  = np.linspace(0, sde_T, sde_steps+1)

        # Parameters per model
        params = {
            "GBM":                    {"x0":100, "mu":0.10, "sigma":0.20},
            "Ornstein-Uhlenbeck":     {"x0":0.0, "theta":2.0, "mu":0.0, "sigma":0.5},
            "Vasicek":                {"x0":0.05,"a":0.5, "b":0.05, "sigma":0.01},
            "CIR":                    {"x0":0.05,"a":1.0, "b":0.05, "sigma":0.05},
            "Heston Variance":        {"x0":0.04,"kappa":2.0,"theta":0.04,"xi":0.3},
        }
        p   = params[sde_model]
        X   = np.zeros((sde_paths, sde_steps+1))
        X[:,0] = p["x0"]

        for i in range(1, sde_steps+1):
            Z  = np.random.normal(0,1,sde_paths)
            xi = X[:,i-1]
            if sde_model == "GBM":
                X[:,i] = xi * np.exp((p["mu"]-0.5*p["sigma"]**2)*dt_sde + p["sigma"]*np.sqrt(dt_sde)*Z)
            elif sde_model == "Ornstein-Uhlenbeck":
                X[:,i] = xi + p["theta"]*(p["mu"]-xi)*dt_sde + p["sigma"]*np.sqrt(dt_sde)*Z
            elif sde_model == "Vasicek":
                X[:,i] = xi + p["a"]*(p["b"]-xi)*dt_sde + p["sigma"]*np.sqrt(dt_sde)*Z
            elif sde_model == "CIR":
                X[:,i] = np.maximum(xi + p["a"]*(p["b"]-xi)*dt_sde
                                    + p["sigma"]*np.sqrt(np.maximum(xi,0)*dt_sde)*Z, 0)
            elif sde_model == "Heston Variance":
                X[:,i] = np.maximum(xi + p["kappa"]*(p["theta"]-xi)*dt_sde
                                    + p["xi"]*np.sqrt(np.maximum(xi,0)*dt_sde)*Z, 1e-8)

        with cr:
            x_med = np.median(X[:,-1])
            x_mean= X[:,-1].mean()
            x_std = X[:,-1].std()
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Terminal Median", f"{x_med:.4f}")
            c2.metric("Terminal Mean",   f"{x_mean:.4f}")
            c3.metric("Terminal Std",    f"{x_std:.4f}")
            c4.metric("Model",           sde_model[:12])

        fig_sde = make_subplots(rows=1, cols=2,
            subplot_titles=[f"{sde_model} Paths", "Terminal Distribution"])
        for p_i in range(min(sde_paths, 60)):
            fig_sde.add_trace(go.Scatter(x=t_sde, y=X[p_i], mode="lines",
                line=dict(color="rgba(0,212,255,0.08)",width=0.6), showlegend=False), row=1, col=1)
        xm = np.median(X, axis=0)
        xp05 = np.percentile(X, 5, axis=0)
        xp95 = np.percentile(X, 95, axis=0)
        fig_sde.add_trace(go.Scatter(x=t_sde, y=xm, name="Median",
            line=dict(color="#00D4FF",width=2.5)), row=1, col=1)
        fig_sde.add_trace(go.Scatter(
            x=list(t_sde)+list(t_sde[::-1]),
            y=list(xp95)+list(xp05[::-1]),
            fill="toself", fillcolor="rgba(0,212,255,0.08)",
            line=dict(width=0), name="5-95% CI"), row=1, col=1)
        fig_sde.add_trace(go.Histogram(x=X[:,-1], nbinsx=40, name="Terminal",
            marker=dict(color="#8B5CF6",opacity=0.8)), row=1, col=2)
        fig_sde.update_layout(title=f"{sde_model}  (x₀={p['x0']}, T={sde_T}y)",
                               height=380, **lay())
        fig_sde.update_xaxes(title_text="Time (years)", row=1, col=1, **ax())
        fig_sde.update_xaxes(title_text="Terminal Value", row=1, col=2, **ax())
        fig_sde.update_yaxes(**ax())
        st.plotly_chart(fig_sde, use_container_width=True)

    # 12C — NUMERICAL METHODS
    with st12c:
        st.markdown("""
        <div class="fbox">
        <strong>Euler-Maruyama:</strong> Xₜ₊Δ = Xₜ + μ(Xₜ)Δt + σ(Xₜ)√Δt·Z &nbsp;|&nbsp;
        <strong>Milstein:</strong> + ½σ(Xₜ)σ'(Xₜ)(Z²−1)Δt &nbsp;|&nbsp;
        <strong>Strong order:</strong> EM = 0.5, Milstein = 1.0
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns([1, 2])
        with cl:
            dt_ref = st.select_slider("Reference Δt (finest)",
                options=[1/2520, 1/1260, 1/504, 1/252],
                value=1/1260, format_func=lambda x: f"1/{round(1/x)}", key="nm_dt")
            nm_paths = st.slider("Paths for convergence", 100, 1000, 200, key="nm_p")

        # GBM exact solution for comparison
        T_nm  = 1.0; mu_nm = 0.1; sig_nm = 0.2; x0_nm = 100.0
        np.random.seed(99)

        # Strong convergence study: compare EM and Milstein at different step sizes
        step_sizes = [1/20, 1/50, 1/100, 1/252, 1/504]
        em_errors, mil_errors = [], []
        n_nm = nm_paths

        for dt_s in step_sizes:
            steps_s = int(T_nm / dt_s)
            Z_all   = np.random.normal(0, np.sqrt(dt_s), (n_nm, steps_s))
            X_em  = np.full(n_nm, x0_nm)
            X_mil = np.full(n_nm, x0_nm)
            for i in range(steps_s):
                Z_i    = Z_all[:,i]
                X_em   = X_em  * (1 + mu_nm*dt_s + sig_nm*Z_i)
                X_mil  = X_mil * (1 + mu_nm*dt_s + sig_nm*Z_i + 0.5*sig_nm**2*(Z_i**2/dt_s - 1)*dt_s)
            # Exact GBM
            W_T = Z_all.sum(axis=1) * np.sqrt(dt_s) / np.sqrt(dt_s)
            W_T = Z_all.sum(axis=1)  # total W_T
            X_exact = x0_nm * np.exp((mu_nm - 0.5*sig_nm**2)*T_nm + sig_nm*W_T)
            em_errors.append( float(np.mean(np.abs(X_em  - X_exact))))
            mil_errors.append(float(np.mean(np.abs(X_mil - X_exact))))

        with cr:
            c1,c2 = st.columns(2)
            c1.metric("EM Error (Δt=1/252)",  f"{em_errors[-2]:.4f}")
            c2.metric("Mil Error (Δt=1/252)", f"{mil_errors[-2]:.4f}")

        fig_conv = go.Figure()
        log_dt   = np.log(step_sizes)
        fig_conv.add_trace(go.Scatter(x=log_dt, y=np.log(em_errors), mode="lines+markers",
            name="Euler-Maruyama (order 0.5)",
            line=dict(color="#00D4FF",width=2), marker=dict(size=8)))
        fig_conv.add_trace(go.Scatter(x=log_dt, y=np.log(mil_errors), mode="lines+markers",
            name="Milstein (order 1.0)",
            line=dict(color="#10B981",width=2), marker=dict(size=8)))
        # Reference lines
        dt_ref_arr = np.array([log_dt[0], log_dt[-1]])
        mid_em = np.log(em_errors[0]) + 0.5*(log_dt[0]-log_dt[0])
        fig_conv.add_trace(go.Scatter(x=dt_ref_arr,
            y=[np.log(em_errors[0]) + 0.5*(d - log_dt[0]) for d in dt_ref_arr],
            mode="lines", line=dict(color="#EF4444",dash="dot"), name="Slope 0.5"))
        fig_conv.add_trace(go.Scatter(x=dt_ref_arr,
            y=[np.log(mil_errors[0]) + 1.0*(d - log_dt[0]) for d in dt_ref_arr],
            mode="lines", line=dict(color="#F59E0B",dash="dot"), name="Slope 1.0"))
        fig_conv.update_layout(title="Strong Convergence: |E[X_EM - X_exact]| vs Δt",
                               height=360, **lay())
        fig_conv.update_xaxes(title_text="log(Δt)", **ax())
        fig_conv.update_yaxes(title_text="log(Error)", **ax())
        st.plotly_chart(fig_conv, use_container_width=True)

        # MC GBM simulation with actual stock parameters
        st.markdown("#### Monte Carlo GBM vs Historical Return Distribution")
        mc_t = st.selectbox("Stock", selected, key="mc_t12")
        ret_mc = R_df[mc_t].values
        mu_mc  = ret_mc.mean()*252; sig_mc = ret_mc.std()*np.sqrt(252)
        np.random.seed(5)
        n_mc = 300; steps_mc = 252
        Z_mc = np.random.normal((mu_mc-0.5*sig_mc**2)/252,
                                 sig_mc/np.sqrt(252), (n_mc, steps_mc))
        X_mc = np.hstack([np.ones((n_mc,1)), np.exp(np.cumsum(Z_mc,axis=1))])

        fig_mc = make_subplots(rows=1, cols=2,
            subplot_titles=["GBM MC Paths (1 year)", "Terminal Return Distribution"])
        for p_i in range(min(n_mc,50)):
            fig_mc.add_trace(go.Scatter(x=list(range(steps_mc+1)), y=X_mc[p_i], mode="lines",
                line=dict(color="rgba(0,212,255,0.07)",width=0.5),
                showlegend=False), row=1, col=1)
        fig_mc.add_trace(go.Scatter(x=list(range(steps_mc+1)),
            y=np.median(X_mc,axis=0), name="Median",
            line=dict(color="#00D4FF",width=2)), row=1, col=1)
        fig_mc.add_trace(go.Histogram(
            x=(X_mc[:,-1]-1)*100, nbinsx=40,
            name="MC Terminal Returns", marker=dict(color="#00D4FF",opacity=0.7)), row=1, col=2)
        fig_mc.add_trace(go.Histogram(
            x=ret_mc*100*np.sqrt(252), nbinsx=40,
            name="Historical Dist", marker=dict(color="#F59E0B",opacity=0.5)), row=1, col=2)
        fig_mc.update_layout(title=f"{mc_t}  —  GBM MC vs Historical",
                              height=360, barmode="overlay", **lay())
        fig_mc.update_xaxes(**ax()); fig_mc.update_yaxes(**ax())
        st.plotly_chart(fig_mc, use_container_width=True)

    # 12D — ASSET PRICING PDE
    with st12d:
        st.markdown("""
        <div class="fbox">
        <strong>BS PDE:</strong> ∂V/∂t + rS∂V/∂S + ½σ²S²∂²V/∂S² = rV &nbsp;|&nbsp;
        <strong>Feynman-Kac:</strong> V(t,x) = E^Q[e^{-r(T-t)} f(Sₜ) | Sₜ=x] &nbsp;|&nbsp;
        <strong>Risk-neutral measure:</strong> dS = rS dt + σS dW^Q (Girsanov)
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns([1, 2])
        with cl:
            pde_S0  = st.number_input("Current price S₀", value=100.0, step=5.0, key="pde_S0")
            pde_K   = st.number_input("Strike K", value=100.0, step=5.0, key="pde_K")
            pde_T   = st.slider("Time to expiry T (years)", 0.1, 2.0, 0.5, 0.1, key="pde_T")
            pde_sig = st.slider("Volatility σ (%)", 5.0, 80.0, 20.0, 1.0, key="pde_sig") / 100
            pde_r   = st.slider("Risk-free rate r (%)", 0.0, 20.0, 5.0, 0.5, key="pde_r") / 100

        # Black-Scholes closed form
        def bs_call(S, K, T, r, sig):
            if T <= 0: return max(S-K, 0)
            d1 = (np.log(S/K) + (r + 0.5*sig**2)*T) / (sig*np.sqrt(T))
            d2 = d1 - sig*np.sqrt(T)
            from scipy.special import ndtr
            return S*ndtr(d1) - K*np.exp(-r*T)*ndtr(d2)

        def bs_put(S, K, T, r, sig):
            return bs_call(S, K, T, r, sig) - S + K*np.exp(-r*T)

        call_price = bs_call(pde_S0, pde_K, pde_T, pde_r, pde_sig)
        put_price  = bs_put(pde_S0, pde_K, pde_T, pde_r, pde_sig)

        with cr:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Call Price",   f"{call_price:.4f}")
            c2.metric("Put Price",    f"{put_price:.4f}")
            c3.metric("Put-Call Check",f"{abs(call_price-put_price-(pde_S0-pde_K*np.exp(-pde_r*pde_T))):.6f}")
            c4.metric("Fwd Price",    f"{pde_S0*np.exp(pde_r*pde_T):.4f}")

        # Option value surface: time and spot
        S_arr = np.linspace(pde_S0*0.6, pde_S0*1.4, 50)
        T_arr = np.linspace(0.01, pde_T, 40)
        Call_surf = np.array([[bs_call(s, pde_K, t, pde_r, pde_sig) for s in S_arr] for t in T_arr])

        # go.Surface needs its own figure (scene subplot); cannot mix with xy in make_subplots
        col_3d, col_2d = st.columns(2)

        with col_3d:
            st.markdown("##### Call Value Surface V(S,t)")
            fig_surf = go.Figure(go.Surface(
                x=S_arr, y=T_arr, z=Call_surf,
                colorscale=[[0,"#0D1829"],[0.4,"#1C3A5E"],[0.7,"#00D4FF"],[1,"#10B981"]],
                showscale=True,
                colorbar=dict(title="V", tickfont=dict(family="JetBrains Mono",color="#94A3B8")),
            ))
            fig_surf.update_layout(
                scene=dict(
                    xaxis_title="Spot S", yaxis_title="Time t", zaxis_title="Call V",
                    bgcolor="#070C14",
                    xaxis=dict(backgroundcolor="#0D1829",gridcolor="#182A40"),
                    yaxis=dict(backgroundcolor="#0D1829",gridcolor="#182A40"),
                    zaxis=dict(backgroundcolor="#0D1829",gridcolor="#182A40"),
                ),
                height=440, paper_bgcolor="#070C14",
                font=dict(family="JetBrains Mono",color="#94A3B8"),
                margin=dict(l=0,r=0,t=20,b=0),
            )
            st.plotly_chart(fig_surf, use_container_width=True)

        with col_2d:
            st.markdown("##### Call vs Put Payoffs at Expiry")
            payoff_call = np.maximum(S_arr - pde_K, 0)
            payoff_put  = np.maximum(pde_K - S_arr, 0)
            call_now    = [bs_call(s, pde_K, pde_T, pde_r, pde_sig) for s in S_arr]
            put_now     = [bs_put( s, pde_K, pde_T, pde_r, pde_sig) for s in S_arr]
            fig_pay = go.Figure()
            fig_pay.add_trace(go.Scatter(x=S_arr, y=payoff_call, name="Call Payoff",
                line=dict(color="#00D4FF",dash="dot",width=1.5)))
            fig_pay.add_trace(go.Scatter(x=S_arr, y=call_now, name="Call Now",
                line=dict(color="#00D4FF",width=2.5)))
            fig_pay.add_trace(go.Scatter(x=S_arr, y=payoff_put, name="Put Payoff",
                line=dict(color="#F59E0B",dash="dot",width=1.5)))
            fig_pay.add_trace(go.Scatter(x=S_arr, y=put_now, name="Put Now",
                line=dict(color="#F59E0B",width=2.5)))
            fig_pay.add_vline(x=pde_K, line=dict(color="#EF4444",dash="dash"),
                              annotation_text="Strike")
            fig_pay.update_layout(
                title=f"S={pde_S0}, K={pde_K}, T={pde_T}y, σ={pde_sig*100:.0f}%",
                height=440, **lay())
            fig_pay.update_xaxes(title_text="Spot S", **ax())
            fig_pay.update_yaxes(title_text="Value", **ax())
            st.plotly_chart(fig_pay, use_container_width=True)


with tab13:
    st.markdown('<div class="sec-hdr">13. Counterparty Risk & Margin Optimization — VaR/CVaR · SIMM · CCP · Margin Opt · XVA</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="fbox">
    <strong>CVA:</strong> (1−R)·∫₀ᵀ E[EE(t)]·dPD(t) &nbsp;|&nbsp;
    <strong>SIMM:</strong> W(δ)δCδᵀ (sensitivity-based) &nbsp;|&nbsp;
    <strong>MPOR:</strong> ΔT = ΔT₀·max(1, N/N₀) &nbsp;|&nbsp;
    <strong>Margin Opt:</strong> min Σ_p SIMM_{p,q}  s.t. flat, symmetric, bounded
    </div>""", unsafe_allow_html=True)

    st13a, st13b, st13c, st13d, st13e = st.tabs([
        "📊 Risk Measures",
        "🔒 SIMM Framework",
        "🏦 Central Clearing",
        "⚖️ Margin Optimization",
        "💱 XVA Framework",
    ])

    # 13A — ENHANCED RISK MEASURES
    with st13a:
        st.markdown("""
        <div class="fbox">
        <strong>VaR_α:</strong> inf{l: P(Loss>l) ≤ 1−α} &nbsp;|&nbsp;
        <strong>ES/CVaR_α:</strong> E[Loss | Loss > VaR_α] &nbsp;|&nbsp;
        <strong>Spectral:</strong> ρ_φ(X) = ∫₀¹ φ(p)·q_p(X) dp &nbsp;|&nbsp;
        <strong>Coherent:</strong> subadditivity · homogeneity · translation invariance
        </div>""", unsafe_allow_html=True)

        rm_t = st.selectbox("Stock", selected, key="rm_t13")
        ret_rm = R_df[rm_t].values
        n_rm = len(ret_rm)
        notional = st.number_input("Position size ($M)", value=10.0, step=1.0, key="rm_not")

        alphas = [0.90, 0.95, 0.99, 0.999]
        rows_rm = []
        for a in alphas:
            var_a  = float(np.quantile(ret_rm, 1-a))
            cvar_a = float(ret_rm[ret_rm <= var_a].mean()) if (ret_rm <= var_a).any() else var_a
            rows_rm.append([f"{a*100:.1f}%", f"{var_a*100:.4f}%",
                             f"{cvar_a*100:.4f}%",
                             f"${-var_a*notional:.4f}M",
                             f"${-cvar_a*notional:.4f}M"])
        rm_df = pd.DataFrame(rows_rm,
            columns=["Confidence","VaR","ES/CVaR","VaR Loss","ES Loss"])
        st.markdown("#### VaR & Expected Shortfall Table")
        st.dataframe(rm_df, use_container_width=True)

        # Spectral risk measure with exponential weights
        pvals   = np.linspace(0.001, 0.999, 500)
        quants  = np.quantile(ret_rm, pvals)
        phi_exp = np.exp(-5*(1-pvals)) * 5  # exponential spectral weight
        phi_exp /= np.trapezoid(phi_exp, pvals)
        spectral_rm = -np.trapezoid(phi_exp * quants, pvals)

        fig_rm = make_subplots(rows=1, cols=2,
            subplot_titles=["Return Distribution with Risk Measures",
                             "Spectral Risk Measure (Exponential Weights)"])
        fig_rm.add_trace(go.Histogram(x=ret_rm*100, nbinsx=80, name="Returns",
            marker=dict(color="#00D4FF",opacity=0.6)), row=1, col=1)
        for a in [0.95, 0.99]:
            var_a = float(np.quantile(ret_rm, 1-a))
            fig_rm.add_vline(x=var_a*100, line=dict(color="#EF4444",dash="dash",width=1.5),
                             annotation_text=f"VaR{a*100:.0f}%={var_a*100:.2f}%", row=1, col=1)
        fig_rm.add_trace(go.Scatter(x=pvals, y=phi_exp, mode="lines", name="φ(p) Exp",
            line=dict(color="#F59E0B",width=2)), row=1, col=2)
        fig_rm.add_trace(go.Scatter(x=pvals, y=-quants*100, mode="lines", name="Quantile loss",
            line=dict(color="#00D4FF",width=1.5), opacity=0.6), row=1, col=2)
        fig_rm.update_layout(title=f"{rm_t}  —  Risk Measures  |  Spectral RM = {spectral_rm*100:.4f}%",
                              height=360, **lay())
        fig_rm.update_xaxes(**ax()); fig_rm.update_yaxes(**ax())
        st.plotly_chart(fig_rm, use_container_width=True)

    # 13B — SIMM FRAMEWORK
    with st13b:
        st.markdown("""
        <div class="fbox">
        <strong>SIMM Equity:</strong> IM = √(δᵀCδ) · MPOR factor &nbsp;|&nbsp;
        <strong>Concentration:</strong> CR_i = max(1, √(|δᵢ|/Tᵢ)) &nbsp;|&nbsp;
        <strong>MPOR:</strong> ΔT = ΔT₀ · max(1, N/N₀) &nbsp;|&nbsp;
        Intra-bucket ρ = 15%  ·  Inter-bucket ρ = 8%
        </div>""", unsafe_allow_html=True)

        # Synthetic SIMM calculation for selected equities
        n_sim = N
        # Sensitivity = position delta × price (proxy)
        price_last = np.array([float(closes[t].iloc[-1]) for t in selected])
        pos_size   = np.random.RandomState(5).randint(100, 10000, n_sim).astype(float)
        deltas     = pos_size * price_last * 0.01  # $ delta per 1% move

        # ISDA SIMM risk weights (equity, generic emerging market proxy)
        rw_pct = np.array([15.0]*n_sim)  # 15% risk weight for EM equities
        sensitivity_dollar = deltas / 100   # per 1 unit

        # Intra-bucket correlation
        rho_intra  = 0.15
        rho_inter  = 0.08
        C_simm     = rho_intra * np.ones((n_sim, n_sim)) + (1-rho_intra)*np.eye(n_sim)

        # WS_i = RW_i × sensitivity_i
        WS = (rw_pct/100) * np.abs(sensitivity_dollar)
        K_bucket = np.sqrt(WS @ C_simm @ WS)  # bucket IM (all in one bucket here)

        # Concentration add-on
        T_conc   = np.ones(n_sim) * 1e6  # threshold
        CR       = np.maximum(1.0, np.sqrt(np.abs(sensitivity_dollar)/np.maximum(T_conc, 1)))
        K_conc   = K_bucket * CR.mean()

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("SIMM IM (bucket)",    f"${K_bucket:.0f}")
        c2.metric("SIMM IM (with conc)", f"${K_conc:.0f}")
        c3.metric("Avg Delta ($)",     f"${deltas.mean():.0f}")
        c4.metric("Total Gross Delta",   f"${deltas.sum():.0f}")

        # Sensitivity bar chart
        fig_simm = make_subplots(rows=1, cols=2,
            subplot_titles=["Net Sensitivities (δ)", "IM Contribution per Asset"])
        fig_simm.add_trace(go.Bar(x=selected, y=sensitivity_dollar,
            marker=dict(color=PALETTE[:n_sim],opacity=0.85),
            text=[f"{v:.0f}" for v in sensitivity_dollar], textposition="outside"), row=1, col=1)
        ws_pct = WS / max(WS.sum(), 1e-9) * 100
        fig_simm.add_trace(go.Bar(x=selected, y=ws_pct,
            marker=dict(color=PALETTE[:n_sim],opacity=0.85),
            text=[f"{v:.1f}%" for v in ws_pct], textposition="outside"), row=1, col=2)
        fig_simm.update_layout(title="ISDA SIMM Sensitivity Analysis", height=360,
                                showlegend=False, **lay())
        fig_simm.update_xaxes(**ax()); fig_simm.update_yaxes(**ax())
        st.plotly_chart(fig_simm, use_container_width=True)

    # 13C — CENTRAL CLEARING
    with st13c:
        st.markdown("""
        <div class="fbox">
        <strong>Variation Margin:</strong> current MTM change daily &nbsp;|&nbsp;
        <strong>Initial Margin:</strong> 99% 5-day VaR of portfolio &nbsp;|&nbsp;
        <strong>Netting:</strong> bilateral netted exposure = Σᵢ max(Vᵢ,0) − max(−Vᵢ,0) &nbsp;|&nbsp;
        <strong>Default Fund:</strong> stress-test based mutualized buffer
        </div>""", unsafe_allow_html=True)

        # Simulate bilateral vs central clearing comparison
        np.random.seed(42)
        n_parties = min(N, 5)
        n_days_cc = 252

        # Simulate portfolio PnL for each counterparty pair
        # Trim each pair to the same length to avoid broadcast errors
        n_ret_rows = len(R_df)
        pnl_matrix = {}
        for i, ti in enumerate(selected[:n_parties]):
            for j, tj in enumerate(selected[:n_parties]):
                if i < j:
                    ret_ti = R_df[ti].values
                    ret_tj = R_df[tj].values
                    min_len = min(len(ret_ti), len(ret_tj))
                    ret_pair = ret_ti[:min_len] - ret_tj[:min_len]
                    pnl_matrix[(ti, tj)] = ret_pair * 1e6  # $1M position

        # Bilateral IM (99% 5-day VaR each pair)
        bilateral_IM = {}
        for pair, pnl in pnl_matrix.items():
            var_5d = np.abs(np.quantile(pnl, 0.01)) * np.sqrt(5)
            bilateral_IM[pair] = var_5d

        # CCP netting: each party's net position
        # net_pnl must match the length of each pnl_matrix entry
        ccp_IM_per_party = {}
        for i, ti in enumerate(selected[:n_parties]):
            net_pnl = np.zeros(n_ret_rows)
            for j, tj in enumerate(selected[:n_parties]):
                if i < j and (ti, tj) in pnl_matrix:
                    arr = pnl_matrix[(ti, tj)]
                    net_pnl[:len(arr)] += arr
                elif j < i and (tj, ti) in pnl_matrix:
                    arr = pnl_matrix[(tj, ti)]
                    net_pnl[:len(arr)] -= arr
            var_net = np.abs(np.quantile(net_pnl, 0.01)) * np.sqrt(5)
            ccp_IM_per_party[ti] = var_net

        total_bilateral = sum(bilateral_IM.values()) * 2  # both sides
        total_ccp       = sum(ccp_IM_per_party.values())
        netting_benefit = (total_bilateral - total_ccp) / max(total_bilateral, 1) * 100

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Bilateral Total IM",  f"${total_bilateral/1e6:.2f}M")
        c2.metric("CCP Total IM",        f"${total_ccp/1e6:.2f}M")
        c3.metric("Netting Benefit",     f"{netting_benefit:.1f}%")
        c4.metric("IM Reduction",        f"${(total_bilateral-total_ccp)/1e6:.2f}M")

        fig_ccp = make_subplots(rows=1, cols=2,
            subplot_titles=["Bilateral vs CCP IM per Party",
                             "Netting Benefit by Counterparty Pair"])
        parties = selected[:n_parties]
        bil_per = [sum(v for (a,b),v in bilateral_IM.items() if a==p or b==p)/1e6 for p in parties]
        ccp_per = [ccp_IM_per_party.get(p,0)/1e6 for p in parties]
        fig_ccp.add_trace(go.Bar(x=parties, y=bil_per, name="Bilateral",
            marker=dict(color="#EF4444",opacity=0.8)), row=1, col=1)
        fig_ccp.add_trace(go.Bar(x=parties, y=ccp_per, name="CCP (netted)",
            marker=dict(color="#10B981",opacity=0.8)), row=1, col=1)
        pair_labels = [f"{a[:3]}↔{b[:3]}" for a,b in bilateral_IM.keys()]
        pair_vals   = [v/1e6 for v in bilateral_IM.values()]
        fig_ccp.add_trace(go.Bar(x=pair_labels, y=pair_vals, name="Pair IM",
            marker=dict(color="#F59E0B",opacity=0.85)), row=1, col=2)
        fig_ccp.update_layout(title="Central Clearing vs Bilateral Margin",
                               height=360, **lay())
        fig_ccp.update_xaxes(**ax()); fig_ccp.update_yaxes(title_text="$M", **ax())
        st.plotly_chart(fig_ccp, use_container_width=True)

    # 13D — MARGIN OPTIMIZATION
    with st13d:
        st.markdown("""
        <div class="fbox">
        <strong>Objective:</strong> min Σ_{p∈P} (Σ_{q≠p} SIMM(p,q) + Σ_{ccp} IM(p,ccp)) &nbsp;|&nbsp;
        <strong>Symmetry:</strong> x_{h,p,q} = −x_{h,q,p} &nbsp;|&nbsp;
        <strong>Flatness:</strong> Σ_q x_{h,p,q} = 0 &nbsp;|&nbsp;
        <strong>Bounds:</strong> T⁻ ≤ Σ_h xₕDₕ ≤ T⁺
        </div>""", unsafe_allow_html=True)

        # Simplified 2-asset margin optimization: find optimal compression trade
        cl, cr = st.columns([1, 2])
        with cl:
            mo_t1 = st.selectbox("Asset 1", selected, key="mo_t1")
            mo_t2 = st.selectbox("Asset 2", [t for t in selected if t != mo_t1],
                                  key="mo_t2")
            mo_pos1 = st.slider("Position 1 ($M)", -50, 50, 20, 5, key="mo_p1")
            mo_pos2 = st.slider("Position 2 ($M)", -50, 50, -15, 5, key="mo_p2")

        rw1, rw2 = 0.15, 0.15  # 15% SIMM equity risk weight
        rho_sim  = float(np.corrcoef(R_df[mo_t1].values, R_df[mo_t2].values)[0,1])
        d1 = mo_pos1 * rw1; d2 = mo_pos2 * rw2

        # IM before optimization
        im_before = np.sqrt(d1**2 + d2**2 + 2*rho_sim*d1*d2)

        # Compression trade: add -x to pos1, +x to pos2 (flat, symmetric)
        x_range = np.linspace(-min(abs(mo_pos1), abs(mo_pos2)),
                               min(abs(mo_pos1), abs(mo_pos2)), 200)
        im_after = [np.sqrt(((mo_pos1-x)*rw1)**2 + ((mo_pos2+x)*rw2)**2
                            + 2*rho_sim*(mo_pos1-x)*rw1*(mo_pos2+x)*rw2)
                    for x in x_range]
        opt_x  = x_range[np.argmin(im_after)]
        im_opt = min(im_after)

        with cr:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("IM Before",       f"${im_before:.4f}M")
            c2.metric("IM After Opt",    f"${im_opt:.4f}M")
            c3.metric("Saving",          f"${im_before-im_opt:.4f}M")
            c4.metric("Correlation ρ",   f"{rho_sim:.4f}")

        fig_mo = make_subplots(rows=1, cols=2,
            subplot_titles=["IM vs Compression Trade Size",
                             "Position Space: Iso-IM Contours"])
        fig_mo.add_trace(go.Scatter(x=x_range, y=im_after, mode="lines",
            name="IM(x)", line=dict(color="#00D4FF",width=2)), row=1, col=1)
        fig_mo.add_vline(x=opt_x, line=dict(color="#10B981",dash="dash"),
                         annotation_text=f"Opt x={opt_x:.2f}", row=1, col=1)
        fig_mo.add_hline(y=im_before, line=dict(color="#EF4444",dash="dot"),
                         annotation_text="Before", row=1, col=1)
        # Contour map in (pos1, pos2) space
        p1_arr = np.linspace(-50, 50, 50); p2_arr = np.linspace(-50, 50, 50)
        P1, P2 = np.meshgrid(p1_arr, p2_arr)
        IM_grid = np.sqrt((P1*rw1)**2 + (P2*rw2)**2 + 2*rho_sim*P1*rw1*P2*rw2)
        fig_mo.add_trace(go.Contour(x=p1_arr, y=p2_arr, z=IM_grid,
            colorscale=[[0,"#0D1829"],[0.5,"#1C3A5E"],[1,"#00D4FF"]],
            showscale=False, name="IM Contour"), row=1, col=2)
        fig_mo.add_trace(go.Scatter(x=[mo_pos1], y=[mo_pos2], mode="markers",
            marker=dict(color="#EF4444",size=14,symbol="x"), name="Current"), row=1, col=2)
        fig_mo.add_trace(go.Scatter(x=[mo_pos1-opt_x], y=[mo_pos2+opt_x],
            mode="markers", marker=dict(color="#10B981",size=14,symbol="star"),
            name="Optimal"), row=1, col=2)
        fig_mo.update_layout(title="Margin Optimization via Compression Trade",
                              height=400, **lay())
        fig_mo.update_xaxes(**ax()); fig_mo.update_yaxes(**ax())
        st.plotly_chart(fig_mo, use_container_width=True)

    # 13E — XVA FRAMEWORK
    with st13e:
        st.markdown("""
        <div class="fbox">
        <strong>CVA:</strong> (1−R)·∫₀ᵀ EE(t)·dPD(t) &nbsp;|&nbsp;
        <strong>DVA:</strong> (1−R_own)·∫₀ᵀ ENE(t)·dPD_own(t) &nbsp;|&nbsp;
        <strong>FVA:</strong> (s_f)·∫₀ᵀ EE(t)·df(t) &nbsp;|&nbsp;
        <strong>MVA:</strong> (s_m)·∫₀ᵀ IM(t)·df(t)
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns([1, 2])
        with cl:
            xva_t   = st.selectbox("Underlying stock", selected, key="xva_t")
            pd_bps  = st.slider("Counterparty CDS spread (bps)", 10, 500, 100, 10, key="xva_pd")
            r_rec   = st.slider("Recovery rate (%)", 0, 60, 40, 5, key="xva_R") / 100
            s_fund  = st.slider("Funding spread (bps)", 5, 200, 50, 5, key="xva_sf") / 10000
            s_marg  = st.slider("Margin cost (bps)", 1, 50, 10, 1, key="xva_sm") / 10000

        ret_xva  = R_df[xva_t].values
        T_xva    = 5  # 5-year horizon
        steps_xva= 252*T_xva
        dt_xva   = 1/252

        # Monte Carlo EE profile
        np.random.seed(42)
        n_xva = 500
        mu_xva = ret_xva.mean()*252; sig_xva = ret_xva.std()*np.sqrt(252)
        S0_xva = float(closes[xva_t].iloc[-1])
        S_xva  = np.zeros((n_xva, steps_xva+1)); S_xva[:,0] = S0_xva
        for i in range(1, steps_xva+1):
            Z = np.random.normal(0,1,n_xva)
            S_xva[:,i] = S_xva[:,i-1]*np.exp((mu_xva-0.5*sig_xva**2)*dt_xva
                                               + sig_xva*np.sqrt(dt_xva)*Z)

        # EE(t) = E[max(V(t), 0)] — positive exposure profile
        # Assume forward contract: V(t) = S(t) - S(0)·e^{r·t}
        t_arr_xva = np.arange(steps_xva+1)*dt_xva
        FWD       = S0_xva * np.exp(rf_rate * t_arr_xva)
        MtM       = S_xva - FWD[np.newaxis,:]
        EE        = np.maximum(MtM, 0).mean(axis=0)
        ENE       = np.maximum(-MtM, 0).mean(axis=0)

        # Hazard rate from CDS spread
        lambda_pd = pd_bps / 10000 / (1 - r_rec)
        PD_inc    = lambda_pd * dt_xva * np.ones(steps_xva+1)  # incremental

        # XVA integrals (trapezoidal)
        df_arr  = np.exp(-rf_rate * t_arr_xva)
        CVA_val = float((1-r_rec) * np.trapezoid(EE  * PD_inc, t_arr_xva))
        DVA_val = float((1-r_rec) * np.trapzoid(ENE * PD_inc, t_arr_xva))
        FVA_val = float(s_fund    * np.trapzoid(EE  * df_arr, t_arr_xva))
        MVA_val = float(s_marg    * np.trapzoid(np.abs(MtM).mean(axis=0) * df_arr, t_arr_xva))
        KVA_val = CVA_val * 0.10  # simplified: 10% of CVA as capital cost

        with cr:
            c1,c2,c3 = st.columns(3)
            c1.metric("CVA",  f"{CVA_val:.4f}")
            c2.metric("DVA",  f"{DVA_val:.4f}")
            c3.metric("FVA",  f"{FVA_val:.4f}")
            c4, c5 = st.columns(2)
            c4.metric("MVA",  f"{MVA_val:.4f}")
            c5.metric("KVA",  f"{KVA_val:.4f}")

        # EE / ENE profile + XVA decomposition
        t_yr_xva = t_arr_xva[::21]  # monthly
        EE_m    = EE[::21]
        ENE_m   = ENE[::21]

        fig_xva = make_subplots(rows=1, cols=2,
            subplot_titles=["Expected Exposure Profile EE(t)",
                             "XVA Components (Cumulative)"])
        fig_xva.add_trace(go.Scatter(x=t_yr_xva, y=EE_m, name="EE(t)",
            fill="tozeroy", fillcolor="rgba(239,68,68,0.12)",
            line=dict(color="#EF4444",width=2)), row=1, col=1)
        fig_xva.add_trace(go.Scatter(x=t_yr_xva, y=ENE_m, name="ENE(t)",
            fill="tozeroy", fillcolor="rgba(16,185,129,0.12)",
            line=dict(color="#10B981",width=2)), row=1, col=1)
        xva_vals = [CVA_val, DVA_val, FVA_val, MVA_val, KVA_val]
        xva_lbls = ["CVA","DVA","FVA","MVA","KVA"]
        xva_cols = ["#EF4444","#10B981","#F59E0B","#8B5CF6","#00D4FF"]
        fig_xva.add_trace(go.Bar(x=xva_lbls, y=xva_vals,
            marker=dict(color=xva_cols, opacity=0.85),
            text=[f"{v:.4f}" for v in xva_vals], textposition="outside"), row=1, col=2)
        fig_xva.update_layout(title=f"XVA Framework — {xva_t}  (CDS={pd_bps}bps, R={r_rec*100:.0f}%)",
                               height=380, **lay())
        fig_xva.update_xaxes(**ax()); fig_xva.update_yaxes(**ax())
        st.plotly_chart(fig_xva, use_container_width=True)


with tab14:
    st.markdown('<div class="sec-hdr">14. Advanced Machine Learning for Finance — Neural Networks · LSTM · Reinforcement Learning · Interpretability</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="fbox">
    <strong>Backprop:</strong> ∂L/∂W = ∂L/∂y · ∂y/∂z · ∂z/∂W &nbsp;|&nbsp;
    <strong>LSTM:</strong> fₜ = σ(Wf·[hₜ₋₁,xₜ]+bf) &nbsp;|&nbsp;
    <strong>Q-Learning:</strong> Q(S,A) ← Q(S,A)+α[R+γ·maxₐ'Q(S',A')−Q(S,A)] &nbsp;|&nbsp;
    <strong>SHAP:</strong> ϕᵢ = Σ |S|!(|F|−|S|−1)!/|F|! [f(S∪{i})−f(S)]
    </div>""", unsafe_allow_html=True)

    st14a, st14b, st14c, st14d = st.tabs([
        "🧠 Neural Networks",
        "🔁 LSTM Forecasting",
        "🎮 Reinforcement Learning",
        "🔍 Model Interpretability",
    ])

    # 14A — NEURAL NETWORKS
    with st14a:
        st.markdown("""
        <div class="fbox">
        <strong>Forward pass:</strong> z = Wx + b, a = σ(z) &nbsp;|&nbsp;
        <strong>Loss:</strong> MSE = (1/n)Σ(y − ŷ)² &nbsp;|&nbsp;
        <strong>ReLU:</strong> max(0,x) &nbsp;|&nbsp;
        <strong>Sigmoid:</strong> 1/(1+e⁻ˣ) &nbsp;|&nbsp;
        <strong>Activations applied to option pricing as surrogate model</strong>
        </div>""", unsafe_allow_html=True)

        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler as SS14

        cl, cr = st.columns([1, 2])
        with cl:
            nn_t    = st.selectbox("Stock (y)", selected, key="nn_t14")
            nn_ind  = [t for t in selected if t != nn_t]
            nn_h1   = st.slider("Hidden layer 1", 4, 64, 16, key="nn_h1")
            nn_h2   = st.slider("Hidden layer 2", 4, 64, 8,  key="nn_h2")
            nn_act  = st.selectbox("Activation", ["relu","tanh","logistic"], key="nn_act")
            nn_epochs = st.slider("Max iterations", 50, 500, 100, key="nn_ep")

        y_nn  = R_df[nn_t].values
        X_nn  = np.column_stack([R_df[t].values for t in nn_ind])
        sx_nn = SS14(); sy_nn = SS14()
        Xs_nn = sx_nn.fit_transform(X_nn)
        ys_nn = sy_nn.fit_transform(y_nn.reshape(-1,1)).ravel()
        split_nn = int(len(y_nn)*0.8)

        nn_model = MLPRegressor(hidden_layer_sizes=(nn_h1, nn_h2),
                                 activation=nn_act, max_iter=nn_epochs,
                                 random_state=42, early_stopping=True)
        nn_model.fit(Xs_nn[:split_nn], ys_nn[:split_nn])
        yp_nn_s = nn_model.predict(Xs_nn[split_nn:])
        yp_nn   = sy_nn.inverse_transform(yp_nn_s.reshape(-1,1)).ravel()
        ya_nn   = y_nn[split_nn:]
        r2_nn   = 1 - ((ya_nn-yp_nn)**2).sum() / max(((ya_nn-ya_nn.mean())**2).sum(), 1e-12)
        rmse_nn = float(np.sqrt(((ya_nn-yp_nn)**2).mean()))

        with cr:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("NN R² (OOS)",   f"{r2_nn:.4f}")
            c2.metric("RMSE",          f"{rmse_nn*100:.4f}%")
            c3.metric("Architecture",  f"[{len(nn_ind)}→{nn_h1}→{nn_h2}→1]")
            c4.metric("Activation",    nn_act)

        # Learning curve
        fig_nn = make_subplots(rows=1, cols=2,
            subplot_titles=["NN: Actual vs Predicted (OOS)", "Activation Functions"])
        idx_nn = R_df.index[split_nn:]
        fig_nn.add_trace(go.Scatter(x=idx_nn, y=ya_nn*100, name="Actual",
            line=dict(color="#00D4FF",width=1.2)), row=1, col=1)
        fig_nn.add_trace(go.Scatter(x=idx_nn, y=yp_nn*100, name="NN Predicted",
            line=dict(color="#F59E0B",width=1.2,dash="dash")), row=1, col=1)
        # Activation functions plot
        z_arr = np.linspace(-4, 4, 200)
        fig_nn.add_trace(go.Scatter(x=z_arr, y=np.maximum(z_arr,0), name="ReLU",
            line=dict(color="#10B981",width=2)), row=1, col=2)
        fig_nn.add_trace(go.Scatter(x=z_arr, y=1/(1+np.exp(-z_arr)), name="Sigmoid",
            line=dict(color="#EF4444",width=2)), row=1, col=2)
        fig_nn.add_trace(go.Scatter(x=z_arr, y=np.tanh(z_arr), name="Tanh",
            line=dict(color="#8B5CF6",width=2)), row=1, col=2)
        fig_nn.add_hline(y=0, line=dict(color="#475569",dash="dot"), row=1, col=2)
        fig_nn.update_layout(title=f"Neural Network: {nn_t}  (R²={r2_nn:.4f})",
                              height=380, **lay())
        fig_nn.update_xaxes(**ax()); fig_nn.update_yaxes(**ax())
        st.plotly_chart(fig_nn, use_container_width=True)

    # 14B — LSTM FORECASTING
    with st14b:
        st.markdown("""
        <div class="fbox">
        <strong>Forget gate:</strong> fₜ = σ(Wf·[hₜ₋₁,xₜ]+bf) &nbsp;|&nbsp;
        <strong>Input gate:</strong> iₜ = σ(Wᵢ·[hₜ₋₁,xₜ]+bᵢ) &nbsp;|&nbsp;
        <strong>Cell update:</strong> C̃ₜ = tanh(Wc·[hₜ₋₁,xₜ]+bc) &nbsp;|&nbsp;
        <strong>Output:</strong> hₜ = oₜ⊙tanh(Cₜ) &nbsp;(implemented via sliding-window MLP as proxy)
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns([1, 2])
        with cl:
            lstm_t    = st.selectbox("Stock", selected, key="lstm_t14")
            lstm_lag  = st.slider("Sequence length (lags)", 5, 30, 10, key="lstm_lag")
            lstm_h    = st.slider("Hidden size", 8, 64, 32, key="lstm_h")
            lstm_fore = st.slider("Forecast days", 1, 20, 5, key="lstm_fore")

        y_lstm = R_df[lstm_t].values
        n_lstm = len(y_lstm)

        # Sliding window features (LSTM proxy via MLP)
        X_lw = np.array([y_lstm[i:i+lstm_lag] for i in range(n_lstm-lstm_lag)])
        y_lw = y_lstm[lstm_lag:]
        sx_lw = SS14(); sy_lw = SS14()
        Xs_lw = sx_lw.fit_transform(X_lw)
        ys_lw = sy_lw.fit_transform(y_lw.reshape(-1,1)).ravel()
        split_lw = int(len(y_lw)*0.8)

        lstm_proxy = MLPRegressor(
            hidden_layer_sizes=(lstm_h, lstm_h//2),
            activation="tanh", max_iter=200, random_state=42, early_stopping=True)
        lstm_proxy.fit(Xs_lw[:split_lw], ys_lw[:split_lw])
        yp_lw_s = lstm_proxy.predict(Xs_lw[split_lw:])
        yp_lw   = sy_lw.inverse_transform(yp_lw_s.reshape(-1,1)).ravel()
        ya_lw   = y_lw[split_lw:]
        r2_lw   = 1 - ((ya_lw-yp_lw)**2).sum()/max(((ya_lw-ya_lw.mean())**2).sum(),1e-12)

        # Multi-step forecast
        hist_seq = list(Xs_lw[-1])
        fc_lstm  = []
        for _ in range(lstm_fore):
            pred_s = lstm_proxy.predict([hist_seq])[0]
            pred   = float(sy_lw.inverse_transform([[pred_s]])[0,0])
            fc_lstm.append(pred)
            new_feat = sx_lw.transform([[*hist_seq[1:], (pred-sy_lw.mean_[0])/max(sy_lw.scale_[0],1e-9)]])[0]
            hist_seq = list(new_feat)

        with cr:
            c1,c2,c3 = st.columns(3)
            c1.metric("LSTM-proxy R²",  f"{r2_lw:.4f}")
            c2.metric("Sequence Length", f"{lstm_lag} days")
            c3.metric("Architecture",    f"tanh-LSTM[{lstm_h},{lstm_h//2}]")

        idx_lw   = R_df.index[lstm_lag+split_lw:]
        bday_off = pd.tseries.offsets.BusinessDay(1)
        fc_dates = [R_df.index[-1] + bday_off*(i+1) for i in range(lstm_fore)]

        fig_lstm = go.Figure()
        fig_lstm.add_trace(go.Scatter(x=idx_lw, y=ya_lw*100, name="Actual",
            line=dict(color="#00D4FF",width=1.2)))
        fig_lstm.add_trace(go.Scatter(x=idx_lw, y=yp_lw*100, name="LSTM-proxy",
            line=dict(color="#F59E0B",width=1.2,dash="dash")))
        fig_lstm.add_trace(go.Scatter(x=fc_dates, y=[f*100 for f in fc_lstm],
            name=f"{lstm_fore}d Forecast", mode="lines+markers",
            line=dict(color="#10B981",width=2,dash="dot"),
            marker=dict(size=7,color="#10B981")))
        fig_lstm.add_vline(x=str(R_df.index[-1].date()),
            line=dict(color="#475569",dash="dot"))
        fig_lstm.update_layout(title=f"{lstm_t}  —  LSTM-proxy Forecast (seq_len={lstm_lag})",
                                height=420, **lay())
        fig_lstm.update_xaxes(**ax()); fig_lstm.update_yaxes(title_text="Return (%)", **ax())
        st.plotly_chart(fig_lstm, use_container_width=True)

    # 14C — REINFORCEMENT LEARNING
    with st14c:
        st.markdown("""
        <div class="fbox">
        <strong>Q-Learning:</strong> Q(S,A) ← Q(S,A) + α[R + γ·maxₐ'Q(S',A') − Q(S,A)] &nbsp;|&nbsp;
        <strong>ε-greedy:</strong> Aₜ = argmaxₐ Q(Sₜ,a) w.p. 1−ε, else random &nbsp;|&nbsp;
        <strong>Reward:</strong> R = daily P&L − λ·risk_penalty
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns([1, 2])
        with cl:
            rl_t      = st.selectbox("Stock for RL trading", selected, key="rl_t14")
            rl_eps    = st.slider("Exploration ε", 0.05, 0.5, 0.2, 0.05, key="rl_eps")
            rl_gamma  = st.slider("Discount γ", 0.80, 0.99, 0.95, 0.01, key="rl_gam")
            rl_alpha  = st.slider("Learning rate α", 0.01, 0.5, 0.1, 0.01, key="rl_alpha")
            rl_episodes = st.slider("Training episodes", 10, 100, 30, key="rl_ep")

        ret_rl   = R_df[rl_t].values
        n_rl     = len(ret_rl)

        # Simple Q-table: states = {down,flat,up}, actions = {short=-1, flat=0, long=+1}
        def discretize_state(r_prev, r_cur, vol_est):
            if r_cur > 0.005: return 2   # up
            elif r_cur < -0.005: return 0  # down
            else: return 1               # flat

        n_states = 3; n_actions = 3
        Q = np.zeros((n_states, n_actions))
        all_rewards = []
        ep_rewards  = []

        np.random.seed(42)
        for ep in range(rl_episodes):
            ep_r = 0
            for t in range(1, n_rl-1):
                s  = discretize_state(ret_rl[t-1], ret_rl[t], 0)
                # ε-greedy action
                if np.random.random() < rl_eps:
                    a = np.random.randint(n_actions)
                else:
                    a = np.argmax(Q[s])
                pos = a - 1   # -1, 0, +1
                r  = pos * ret_rl[t+1] * 100 - abs(pos) * 0.02  # reward - transaction cost
                s_ = discretize_state(ret_rl[t], ret_rl[t+1], 0)
                Q[s,a] += rl_alpha*(r + rl_gamma*np.max(Q[s_]) - Q[s,a])
                ep_r += r
            ep_rewards.append(ep_r)
            all_rewards.append(ep_r)

        # Evaluate learned policy
        positions, pnl_rl = [], [0]
        for t in range(1, n_rl-1):
            s  = discretize_state(ret_rl[t-1], ret_rl[t], 0)
            a  = np.argmax(Q[s])
            pos = a - 1
            positions.append(pos)
            pnl_rl.append(pnl_rl[-1] + pos*ret_rl[t]*100)

        buy_hold = np.cumsum(ret_rl[1:n_rl-1])*100

        with cr:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total RL P&L",    f"{pnl_rl[-1]:.2f}%")
            c2.metric("Buy-Hold P&L",    f"{buy_hold[-1]:.2f}%")
            c3.metric("Avg Episode R",   f"{np.mean(all_rewards):.2f}")
            c4.metric("Final ε Policy",  f"[{[Q[s,:].argmax()-1 for s in range(3)]}]")

        fig_rl = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.1,
            subplot_titles=["Q-Learning P&L vs Buy-Hold Strategy",
                             "Training Rewards per Episode"])
        idx_rl = R_df.index[1:n_rl-1]
        fig_rl.add_trace(go.Scatter(x=idx_rl, y=pnl_rl[1:], name="RL Policy",
            line=dict(color="#10B981",width=1.8)), row=1, col=1)
        fig_rl.add_trace(go.Scatter(x=idx_rl, y=buy_hold, name="Buy-Hold",
            line=dict(color="#EF4444",width=1.5,dash="dash")), row=1, col=1)
        fig_rl.add_trace(go.Scatter(x=list(range(rl_episodes)), y=ep_rewards, mode="lines+markers",
            name="Episode Reward", line=dict(color="#F59E0B",width=1.5)), row=2, col=1)
        fig_rl.add_trace(go.Scatter(x=list(range(rl_episodes)),
            y=pd.Series(ep_rewards).rolling(5,min_periods=1).mean().tolist(),
            name="5-ep MA", line=dict(color="#00D4FF",width=2)), row=2, col=1)
        fig_rl.update_layout(title=f"Q-Learning Trading — {rl_t}  (ε={rl_eps}, γ={rl_gamma})",
                              height=480, **lay())
        fig_rl.update_xaxes(**ax()); fig_rl.update_yaxes(**ax())
        st.plotly_chart(fig_rl, use_container_width=True)

    # 14D — MODEL INTERPRETABILITY
    with st14d:
        st.markdown("""
        <div class="fbox">
        <strong>SHAP:</strong> ϕᵢ = Σ_{S⊆F\\{i}} |S|!(|F|−|S|−1)!/|F|! · [f(S∪{i})−f(S)] &nbsp;|&nbsp;
        <strong>Permutation Importance:</strong> ΔMetric when feature i shuffled &nbsp;|&nbsp;
        <strong>LIME:</strong> local linear approximation around point xₒ
        </div>""", unsafe_allow_html=True)

        interp_t = st.selectbox("Target stock", selected, key="interp_t14")
        feat_names_14 = [t for t in selected if t != interp_t]
        if len(feat_names_14) > 0:
            y_in  = R_df[interp_t].values
            X_in  = np.column_stack([R_df[t].values for t in feat_names_14])
            sxin  = SS14()
            Xs_in = sxin.fit_transform(X_in)

            from sklearn.ensemble import RandomForestRegressor as RFR14
            rf14  = RFR14(n_estimators=50, max_depth=6, random_state=42)
            rf14.fit(Xs_in, y_in)
            base_score = rf14.score(Xs_in, y_in)

            # Permutation importance (manual)
            perm_imp = []
            for i in range(X_in.shape[1]):
                X_perm = Xs_in.copy()
                np.random.seed(i)
                np.random.shuffle(X_perm[:,i])
                perm_score = rf14.score(X_perm, y_in)
                perm_imp.append(base_score - perm_score)

            # Gain importance
            gain_imp = rf14.feature_importances_

            # SHAP approximation: mean |contribution| from feature values × coeff proxy
            shap_approx = np.abs(np.corrcoef(Xs_in.T, y_in)[:-1,-1])

            # LIME: local linear model around last observation
            x0     = Xs_in[-1:,:]
            y0     = float(y_in[-1])
            # perturb around x0
            np.random.seed(7)
            perturbs = x0 + np.random.normal(0, 0.3, (200, X_in.shape[1]))
            y_perturb = rf14.predict(perturbs)
            # weight by distance
            dist_w = np.exp(-0.5*np.sum((perturbs-x0)**2, axis=1))
            from sklearn.linear_model import LinearRegression
            lime_model = LinearRegression()
            lime_model.fit(perturbs, y_perturb, sample_weight=dist_w)
            lime_coef  = lime_model.coef_

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RF R²",         f"{base_score:.4f}")
            c2.metric("Top Feature",   feat_names_14[np.argmax(gain_imp)])
            c3.metric("Max SHAP proxy",f"{shap_approx.max():.4f}")
            c4.metric("Features",      f"{len(feat_names_14)}")

            fig_interp = make_subplots(rows=1, cols=3,
                subplot_titles=["RF Gain Importance",
                                 "Permutation Importance",
                                 "LIME Local Coefficients"])
            sort_g  = np.argsort(gain_imp)
            sort_p  = np.argsort(perm_imp)
            sort_l  = np.argsort(np.abs(lime_coef))
            for ci, (vals, srt, col_v) in enumerate([
                (gain_imp, sort_g, "#00D4FF"),
                (perm_imp, sort_p, "#F59E0B"),
                (lime_coef, sort_l, "#10B981"),
            ], 1):
                fig_interp.add_trace(go.Bar(
                    x=[feat_names_14[i] for i in srt],
                    y=[vals[i] for i in srt],
                    orientation="v",
                    marker=dict(color=col_v, opacity=0.85),
                    name=["Gain","Perm","LIME"][ci-1],
                    showlegend=False,
                ), row=1, col=ci)
            fig_interp.update_layout(title=f"Model Interpretability — {interp_t}",
                                      height=380, **lay())
            fig_interp.update_xaxes(**ax()); fig_interp.update_yaxes(**ax())
            st.plotly_chart(fig_interp, use_container_width=True)


with tab15:
    st.markdown('<div class="sec-hdr">15. Time Series Econometrics — Unit Roots · Cointegration · VAR · State Space & Kalman Filter</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="fbox">
    <strong>ADF:</strong> Δyₜ = α + βt + γyₜ₋₁ + ΣδᵢΔyₜ₋ᵢ + εₜ &nbsp;|&nbsp;
    <strong>VECM:</strong> Δyₜ = αβ'yₜ₋₁ + ΣΓᵢΔyₜ₋ᵢ + εₜ &nbsp;|&nbsp;
    <strong>VAR(p):</strong> yₜ = c + Φ₁yₜ₋₁ + ... + Φₚyₜ₋ₚ + εₜ &nbsp;|&nbsp;
    <strong>Kalman:</strong> xₜ = Fxₜ₋₁ + wₜ, yₜ = Hxₜ + vₜ
    </div>""", unsafe_allow_html=True)

    from scipy import stats as sp15

    st15a, st15b, st15c, st15d = st.tabs([
        "🔬 Unit Root Tests",
        "🔗 Cointegration & VECM",
        "🌐 VAR Model",
        "📡 State Space & Kalman",
    ])

    # 15A — UNIT ROOT TESTS
    with st15a:
        st.markdown("""
        <div class="fbox">
        <strong>ADF:</strong> Δyₜ = α + βt + γyₜ₋₁ + ΣδᵢΔyₜ₋ᵢ + εₜ  H₀: γ=0 (unit root) &nbsp;|&nbsp;
        <strong>KPSS:</strong> H₀: stationarity &nbsp;|&nbsp;
        <strong>Phillips-Perron:</strong> Non-parametric HAC correction on ADF t-stat
        </div>""", unsafe_allow_html=True)

        def adf_test(series, max_lags=5):
            """Manual ADF test with constant, no trend."""
            y = np.array(series)
            n = len(y)
            dy = np.diff(y)
            y_lag = y[:-1]
            # With augmented lags
            if max_lags > 0:
                X_adf = np.column_stack([np.ones(n-1-max_lags),
                                          y_lag[max_lags:]] +
                                         [dy[max_lags-k-1:n-1-k-1] for k in range(max_lags)])
                dy_trim = dy[max_lags:]
            else:
                X_adf = np.column_stack([np.ones(n-1), y_lag])
                dy_trim = dy
            b_adf = np.linalg.lstsq(X_adf, dy_trim, rcond=None)[0]
            resid_adf = dy_trim - X_adf @ b_adf
            n_a, p_a = X_adf.shape
            mse_adf = (resid_adf**2).sum() / max(n_a - p_a, 1)
            XtX_inv = np.linalg.pinv(X_adf.T @ X_adf)
            se_gamma = np.sqrt(mse_adf * XtX_inv[1,1])
            gamma = b_adf[1]
            t_adf = gamma / max(se_gamma, 1e-12)
            # MacKinnon critical values (approximate)
            cv = {1: -3.43, 5: -2.86, 10: -2.57}
            return t_adf, gamma, cv

        def kpss_test(series, lags=10):
            """KPSS test for level stationarity."""
            y = np.array(series)
            n = len(y)
            mu = y.mean()
            e  = y - mu
            S  = np.cumsum(e)
            s2 = (e**2).sum()/n
            # Newey-West HAC variance
            s2_nw = s2
            for k in range(1, lags+1):
                w_k = 1 - k/(lags+1)
                s2_nw += 2*w_k*(e[k:]*e[:-k]).mean()
            kpss_stat = (S**2).sum() / (n**2 * max(s2_nw, 1e-12))
            # Critical values: 1%=0.739, 5%=0.463, 10%=0.347
            cv_kpss = {1: 0.739, 5: 0.463, 10: 0.347}
            return kpss_stat, cv_kpss

        ur_rows = []
        for t in selected:
            price_t = closes[t].values
            ret_t   = R_df[t].values
            # ADF on price level
            t_price, g_p, cv_p = adf_test(price_t, max_lags=3)
            # ADF on returns (should be stationary)
            t_ret,   g_r, cv_r = adf_test(ret_t,   max_lags=3)
            # KPSS on returns
            kpss_r, cv_k = kpss_test(ret_t, lags=10)
            reject_adf_p = "✗ Unit Root" if t_price > cv_p[5] else "✓ Stationary"
            reject_adf_r = "✓ Stationary" if t_ret   < cv_r[5] else "✗ Unit Root"
            reject_kpss  = "✓ Stationary" if kpss_r  < cv_k[5] else "✗ Non-stationary"
            ur_rows.append([t, f"{t_price:.3f}", reject_adf_p,
                               f"{t_ret:.3f}",   reject_adf_r,
                               f"{kpss_r:.4f}",  reject_kpss])

        ur_df = pd.DataFrame(ur_rows,
            columns=["Stock","ADF(price)","Price verdict",
                     "ADF(return)","Return verdict",
                     "KPSS(return)","KPSS verdict"])
        st.markdown("#### Unit Root Test Results  —  All Selected Stocks")
        st.dataframe(ur_df, use_container_width=True)
        st.caption("ADF H₀: unit root exists · Critical value (5%) ≈ −2.86 · KPSS H₀: stationarity · Critical value (5%) ≈ 0.463")

        # Visualise one stock: price vs log-price vs returns
        vis_t = st.selectbox("Visualise stock", selected, key="ur_vis")
        fig_ur = make_subplots(rows=2, cols=2,
            subplot_titles=["Price Level (non-stationary)",
                             "Log Price (non-stationary)",
                             "Log Returns (stationary)",
                             "Return ACF"])
        price_vis = closes[vis_t].values
        ret_vis   = R_df[vis_t].values
        fig_ur.add_trace(go.Scatter(x=dates, y=price_vis, mode="lines",
            line=dict(color="#00D4FF",width=1.2), name="Price"), row=1, col=1)
        fig_ur.add_trace(go.Scatter(x=dates, y=np.log(price_vis), mode="lines",
            line=dict(color="#F59E0B",width=1.2), name="Log Price"), row=1, col=2)
        fig_ur.add_trace(go.Scatter(x=dates, y=ret_vis*100, mode="lines",
            line=dict(color="#10B981",width=0.8), name="Returns"), row=2, col=1)
        lags_ur = list(range(1, 21))
        acf_ur  = [pd.Series(ret_vis).autocorr(lag=k) for k in lags_ur]
        conf_ur = 1.96/np.sqrt(len(ret_vis))
        fig_ur.add_trace(go.Bar(x=lags_ur, y=acf_ur,
            marker_color=["#EF4444" if abs(a)>conf_ur else "#00D4FF" for a in acf_ur],
            opacity=0.85, name="ACF"), row=2, col=2)
        fig_ur.add_hline(y=conf_ur,  line=dict(color="#475569",dash="dot"), row=2, col=2)
        fig_ur.add_hline(y=-conf_ur, line=dict(color="#475569",dash="dot"), row=2, col=2)
        fig_ur.update_layout(title=f"{vis_t}  —  Stationarity Analysis",
                              height=460, showlegend=False, **lay())
        fig_ur.update_xaxes(**ax()); fig_ur.update_yaxes(**ax())
        st.plotly_chart(fig_ur, use_container_width=True)

    # 15B — COINTEGRATION & VECM
    with st15b:
        st.markdown("""
        <div class="fbox">
        <strong>Engle-Granger:</strong> Test residuals ê_t = y₁ − β̂y₂ for unit root &nbsp;|&nbsp;
        <strong>VECM:</strong> Δyₜ = α·β'yₜ₋₁ + ΣΓᵢΔyₜ₋ᵢ + εₜ &nbsp;|&nbsp;
        <strong>Hedge ratio:</strong> β from OLS of y₁ on y₂  &nbsp;|&nbsp;
        <strong>Half-life:</strong> ln(2)/|a| where a from OU fit on spread
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns([1, 2])
        with cl:
            coint_t1 = st.selectbox("Stock 1 (y₁)", selected, key="coint_t1")
            coint_t2 = st.selectbox("Stock 2 (y₂)",
                [t for t in selected if t != coint_t1], key="coint_t2")
            vecm_lags = st.slider("VECM lags", 1, 5, 1, key="vecm_lag")

        y1 = closes[coint_t1].values
        y2 = closes[coint_t2].values
        n_c = min(len(y1), len(y2))
        y1, y2 = y1[:n_c], y2[:n_c]

        # OLS hedge ratio
        X_c = np.column_stack([np.ones(n_c), y2])
        b_c = np.linalg.lstsq(X_c, y1, rcond=None)[0]
        spread = y1 - b_c[0] - b_c[1]*y2

        # ADF on spread (Engle-Granger test)
        def adf_simple(series):
            y = np.array(series)
            dy = np.diff(y); y_lag = y[:-1]
            X = np.column_stack([np.ones(len(dy)), y_lag])
            b = np.linalg.lstsq(X, dy, rcond=None)[0]
            resid = dy - X @ b
            mse   = (resid**2).sum() / max(len(resid)-2, 1)
            se    = np.sqrt(mse * np.linalg.pinv(X.T@X)[1,1])
            return b[1] / max(se, 1e-12), b[1]

        eg_tstat, eg_gamma = adf_simple(spread)
        is_coint = eg_tstat < -3.34  # approximate 5% critical value

        # OU fit on spread: dX = a(mu-X)dt + sigma dW
        spread_d = np.diff(spread)
        X_ou = np.column_stack([np.ones(len(spread_d)), spread[:-1]])
        b_ou = np.linalg.lstsq(X_ou, spread_d, rcond=None)[0]
        ou_speed    = -b_ou[1]  # mean reversion speed
        ou_mean     = b_ou[0] / max(ou_speed, 1e-9)
        half_life   = np.log(2) / max(ou_speed, 1e-9) if ou_speed > 0 else np.inf

        # VECM — simplified: error correction term
        ec_term = spread[:-1] - spread.mean()
        dy1_v   = np.diff(y1)[vecm_lags:]
        dy2_v   = np.diff(y2)[vecm_lags:]
        ec_v    = ec_term[vecm_lags-1:-1] if len(ec_term) > vecm_lags else ec_term[:len(dy1_v)]
        X_vecm  = np.column_stack([np.ones(len(dy1_v[:len(ec_v)])), ec_v[:len(dy1_v)]])
        b_v1    = np.linalg.lstsq(X_vecm, dy1_v[:len(ec_v)], rcond=None)[0]
        b_v2    = np.linalg.lstsq(X_vecm, dy2_v[:len(ec_v)], rcond=None)[0]
        alpha1  = b_v1[1]; alpha2 = b_v2[1]

        with cr:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("EG t-stat",      f"{eg_tstat:.3f}",
                       delta="Cointegrated" if is_coint else "No cointegration")
            c2.metric("Hedge ratio β",  f"{b_c[1]:.4f}")
            c3.metric("Half-life",      f"{half_life:.1f} days" if np.isfinite(half_life) else "∞")
            c4.metric("α₁ (EC speed)",  f"{alpha1:.4f}")

        fig_coint = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            subplot_titles=[f"Normalised Prices: {coint_t1} vs {coint_t2}",
                             "Cointegration Spread & Z-Score"])
        idx_coint = closes.index[:n_c]
        norm1 = y1 / y1[0] * 100; norm2 = y2 / y2[0] * 100
        fig_coint.add_trace(go.Scatter(x=idx_coint, y=norm1, name=coint_t1,
            line=dict(color="#00D4FF",width=1.5)), row=1, col=1)
        fig_coint.add_trace(go.Scatter(x=idx_coint, y=norm2, name=coint_t2,
            line=dict(color="#F59E0B",width=1.5)), row=1, col=1)
        z_spread = (spread - spread.mean()) / max(spread.std(), 1e-9)
        fig_coint.add_trace(go.Scatter(x=idx_coint, y=z_spread, name="Z-Score",
            line=dict(color="#10B981",width=1.2)), row=2, col=1)
        fig_coint.add_hline(y=2,  line=dict(color="#EF4444",dash="dot"), row=2, col=1,
                            annotation_text="Sell spread (+2σ)")
        fig_coint.add_hline(y=-2, line=dict(color="#10B981",dash="dot"), row=2, col=1,
                            annotation_text="Buy spread (−2σ)")
        fig_coint.add_hline(y=0,  line=dict(color="#475569",dash="dot"), row=2, col=1)
        fig_coint.update_layout(title=f"Cointegration: {coint_t1} ~ {coint_t2}  (EG t={eg_tstat:.2f})",
                                 height=440, **lay())
        fig_coint.update_xaxes(**ax()); fig_coint.update_yaxes(**ax())
        st.plotly_chart(fig_coint, use_container_width=True)

    # 15C — VAR MODEL
    with st15c:
        st.markdown("""
        <div class="fbox">
        <strong>VAR(p):</strong> yₜ = c + Φ₁yₜ₋₁ + ... + Φₚyₜ₋ₚ + εₜ &nbsp;|&nbsp;
        <strong>Impulse Response:</strong> ∂yₜ₊ₕ/∂εₜ &nbsp;|&nbsp;
        <strong>FEVD:</strong> σ²_y(h) = Σⱼ contribution of εⱼ to h-step forecast error &nbsp;|&nbsp;
        <strong>Granger Causality:</strong> Test if lagged x helps predict y
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns([1, 2])
        with cl:
            var_vars  = st.multiselect("VAR variables", selected,
                default=selected[:min(3,N)], key="var_vars")
            var_p     = st.slider("VAR order p", 1, 5, 2, key="var_p")
            irf_h     = st.slider("IRF horizon (days)", 5, 30, 15, key="irf_h")

        if len(var_vars) >= 2:
            K_var = len(var_vars)
            Y_var = np.column_stack([R_df[t].values for t in var_vars])
            n_var = len(Y_var)

            # Build VAR design matrix
            Y_endog = Y_var[var_p:]
            X_var_  = np.column_stack([np.ones(n_var-var_p)] +
                                       [Y_var[var_p-k-1:n_var-k-1, :] for k in range(var_p)])
            B_var   = np.linalg.lstsq(X_var_, Y_endog, rcond=None)[0]
            resid_v = Y_endog - X_var_ @ B_var
            Sigma_v = (resid_v.T @ resid_v) / max(n_var-var_p-var_p*K_var-1, 1)

            # Impulse Response Functions (Cholesky orthogonalisation)
            L_chol = np.linalg.cholesky(Sigma_v + 1e-10*np.eye(K_var))
            Phi_1  = B_var[1:1+K_var, :].T  # first lag coefficient matrix (K×K)
            irf    = np.zeros((irf_h, K_var, K_var))
            irf[0] = L_chol
            for h in range(1, irf_h):
                irf[h] = Phi_1 @ irf[h-1]

            # Granger causality (simple F-test): does var_vars[0] Granger-cause var_vars[1]?
            y_gc = Y_var[var_p:, 1]
            X_gc_full    = X_var_
            X_gc_reduced = np.delete(X_gc_full,
                [1 + k*K_var for k in range(var_p)], axis=1)  # remove lagged var 0
            b_f  = np.linalg.lstsq(X_gc_full, y_gc, rcond=None)[0]
            b_r  = np.linalg.lstsq(X_gc_reduced, y_gc, rcond=None)[0]
            rss_f = ((y_gc - X_gc_full @ b_f)**2).sum()
            rss_r = ((y_gc - X_gc_reduced @ b_r)**2).sum()
            df1   = var_p; df2 = max(n_var-var_p-X_gc_full.shape[1], 1)
            F_gc  = ((rss_r - rss_f)/df1) / max(rss_f/df2, 1e-12)
            p_gc  = 1 - sp15.f.cdf(F_gc, df1, df2)

            with cr:
                c1,c2,c3 = st.columns(3)
                c1.metric("VAR order (p)",     f"{var_p}")
                c2.metric(f"Granger {var_vars[0]}→{var_vars[1]}",
                          f"F={F_gc:.2f}",
                          delta="Granger causes" if p_gc < 0.05 else "Does not cause")
                c3.metric("p-value",          f"{p_gc:.4f}")

            # IRF plots (first shock to first variable)
            fig_var = make_subplots(rows=2, cols=2 if K_var <= 2 else (K_var+1)//2,
                subplot_titles=[f"Shock {var_vars[0]} → {var_vars[j]}" for j in range(min(K_var,4))])
            h_arr = list(range(irf_h))
            for j in range(min(K_var, 4)):
                r_ij = (j // 2) + 1; c_ij = (j % 2) + 1
                fig_var.add_trace(go.Scatter(x=h_arr, y=irf[:,j,0]*100, mode="lines+markers",
                    name=f"→{var_vars[j]}", line=dict(color=PALETTE[j%len(PALETTE)],width=2)),
                    row=r_ij, col=c_ij)
                fig_var.add_hline(y=0, line=dict(color="#475569",dash="dot"), row=r_ij, col=c_ij)
            fig_var.update_layout(title=f"Impulse Response Functions  |  VAR({var_p})",
                                   height=420, **lay())
            fig_var.update_xaxes(title_text="Days ahead", **ax())
            fig_var.update_yaxes(title_text="Response (%)", **ax())
            st.plotly_chart(fig_var, use_container_width=True)
        else:
            st.info("Select at least 2 variables for VAR.")

    # 15D — STATE SPACE & KALMAN FILTER
    with st15d:
        st.markdown("""
        <div class="fbox">
        <strong>State Equation:</strong> xₜ = F·xₜ₋₁ + wₜ  (wₜ ~ N(0,Q)) &nbsp;|&nbsp;
        <strong>Obs Equation:</strong> yₜ = H·xₜ + vₜ  (vₜ ~ N(0,R)) &nbsp;|&nbsp;
        <strong>Predict:</strong> x̂ₜ|ₜ₋₁ = F·x̂ₜ₋₁, P̂ₜ|ₜ₋₁ = FPₜ₋₁Fᵀ + Q &nbsp;|&nbsp;
        <strong>Update:</strong> Kₜ = PH/(HPHᵀ+R), x̂ₜ = x̂ₜ|ₜ₋₁ + Kₜ(yₜ−Hx̂ₜ|ₜ₋₁)
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns([1, 2])
        with cl:
            ss_dep = st.selectbox("Observation (y)", selected, key="ss_dep")
            ss_ind = st.selectbox("State driver (x)",
                [t for t in selected if t != ss_dep], key="ss_ind")
            Q_var  = st.slider("State noise Q (×10⁻⁶)", 1, 100, 10, key="ss_Q") * 1e-6
            R_var  = st.slider("Obs noise R (×10⁻⁴)",   1, 100, 50, key="ss_R") * 1e-4
            kf_model = st.selectbox("State model",
                ["Dynamic Beta","Trend + Level","Local Level"], key="ss_mdl")

        y_ss  = R_df[ss_dep].values
        x_ss  = R_df[ss_ind].values
        n_ss  = len(y_ss)

        if kf_model == "Dynamic Beta":
            # State: [alpha, beta] — time-varying CAPM
            F_ss = np.eye(2); H_ss = np.array([[1.0, x_ss[0]]])
            Q_ss = Q_var * np.eye(2); R_ss = np.array([[R_var]])
            x_est = np.array([0.0, 1.0]); P_est = 0.1 * np.eye(2)
            states_ts = []
            for i in range(n_ss):
                H_i = np.array([[1.0, x_ss[i]]])
                x_pred = F_ss @ x_est; P_pred = F_ss @ P_est @ F_ss.T + Q_ss
                S_k  = H_i @ P_pred @ H_i.T + R_ss
                K_k  = P_pred @ H_i.T / max(float(S_k[0,0]), 1e-12)
                innov = y_ss[i] - float(H_i @ x_pred)
                x_est = x_pred + K_k.ravel() * innov
                P_est = (np.eye(2) - np.outer(K_k.ravel(), H_i.ravel())) @ P_pred
                states_ts.append(x_est.copy())
            states_ts = np.array(states_ts)
            label1, label2 = f"Kalman α (ann {ss_dep})", f"Kalman β ({ss_dep}~{ss_ind})"
            s1 = states_ts[:,0]*252*100; s2 = states_ts[:,1]
        else:
            # Local level model: state = level μₜ
            F_ss = np.array([[1.0]]); H_ss = np.array([[1.0]])
            Q_ss = np.array([[Q_var]]); R_ss = np.array([[R_var]])
            x_est = np.array([y_ss[0]]); P_est = np.array([[0.1]])
            filtered = []
            for i in range(n_ss):
                x_pred = float(x_est[0]); P_pred = float(P_est[0,0]) + Q_var
                K_k  = P_pred / max(P_pred + R_var, 1e-12)
                x_est= np.array([x_pred + K_k*(y_ss[i]-x_pred)])
                P_est= np.array([[(1-K_k)*P_pred]])
                filtered.append(float(x_est[0]))
            states_ts = np.array(filtered).reshape(-1,1)
            label1 = "Kalman Filtered Level"; label2 = "Residuals"
            s1 = states_ts[:,0]*100; s2 = y_ss - states_ts[:,0]

        with cr:
            c1,c2 = st.columns(2)
            c1.metric(label1[:20]+"…", f"{s1[-1]:.4f}")
            c2.metric(label2[:20]+"…", f"{s2[-1]:.4f}")

        fig_ss = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            subplot_titles=[label1, label2])
        fig_ss.add_trace(go.Scatter(x=dates, y=s1, name=label1,
            line=dict(color="#00D4FF",width=1.8)), row=1, col=1)
        fig_ss.add_hline(y=0, line=dict(color="#475569",dash="dot"), row=1, col=1)
        fig_ss.add_trace(go.Scatter(x=dates, y=s2*100 if kf_model!="Dynamic Beta" else s2,
            name=label2, line=dict(color="#F59E0B",width=1.5)), row=2, col=1)
        if kf_model == "Dynamic Beta":
            fig_ss.add_hline(y=1.0, line=dict(color="#10B981",dash="dash",width=1),
                             annotation_text="β=1", row=2, col=1)
        fig_ss.update_layout(title=f"Kalman Filter — {kf_model}  ({ss_dep} obs, {ss_ind} driver)",
                              height=460, **lay())
        fig_ss.update_xaxes(**ax()); fig_ss.update_yaxes(**ax())
        st.plotly_chart(fig_ss, use_container_width=True)


# MODULE-LEVEL HELPERS  (defined here so all tabs share them without SessionInfo errors)

def _bt_metrics(equity_curve, risk_free_rate=0.03, trading_days_per_year=252):
    """Compute a full set of performance metrics from a 1-D equity curve.

    Parameters
    ----------
    equity_curve         : array-like, starts at 1.0 (normalised portfolio value)
    risk_free_rate       : annual risk-free rate used for Sharpe / Sortino (default 3%)
    trading_days_per_year: calendar convention for annualising (default 252)

    Returns a dict with: ann_ret, ann_vol, sharpe, sortino,
                         max_dd, calmar, omega, win_rate, n_days
    """
    if len(equity_curve) < 3:
        return dict(ann_ret=0, ann_vol=0, sharpe=0, sortino=0,
                    max_dd=0, calmar=0, omega=1, win_rate=0, n_days=0)

    ann = trading_days_per_year
    daily_returns = np.diff(equity_curve) / np.maximum(np.abs(equity_curve[:-1]), 1e-12)

    # Annualised return and volatility
    ann_return = float(np.mean(daily_returns) * ann)
    ann_vol    = float(np.std(daily_returns)  * np.sqrt(ann))

    # Risk-adjusted ratios
    excess_return  = ann_return - risk_free_rate
    sharpe         = excess_return / max(ann_vol, 1e-8)
    downside_rets  = daily_returns[daily_returns < 0]
    downside_vol   = float(np.std(downside_rets) * np.sqrt(ann)) if len(downside_rets) else 1e-8
    sortino        = excess_return / max(downside_vol, 1e-8)

    # Drawdown
    running_peak   = np.maximum.accumulate(equity_curve)
    drawdowns      = (equity_curve - running_peak) / np.maximum(running_peak, 1e-12)
    max_drawdown   = float(drawdowns.min())
    calmar         = ann_return / max(abs(max_drawdown), 1e-8)

    # Omega ratio: sum of gains / sum of losses
    total_gains  = daily_returns[daily_returns > 0].sum() if (daily_returns > 0).any() else 1e-8
    total_losses = abs(daily_returns[daily_returns < 0].sum()) if (daily_returns < 0).any() else 1e-8
    omega        = total_gains / max(total_losses, 1e-8)

    win_rate = float((daily_returns > 0).mean())

    return dict(
        ann_ret  = ann_return,
        ann_vol  = ann_vol,
        sharpe   = sharpe,
        sortino  = sortino,
        max_dd   = max_drawdown,
        calmar   = calmar,
        omega    = omega,
        win_rate = win_rate,
        n_days   = len(daily_returns),
    )


def _run_strategy(daily_returns, strategy, transaction_cost_bps=10, initial_train_window=120, refit_step=20):
    """Run a trading strategy on a 1-D array of daily log-returns.

    Parameters
    ----------
    daily_returns         : 1-D np.array of daily log-returns
    strategy              : one of "Momentum", "Mean Reversion", "RF Signal",
                            "Buy & Hold", "Vol Targeting"
    transaction_cost_bps  : round-trip transaction cost in basis points (default 10)
    initial_train_window  : warm-up period before the ML model first trades (days)
    refit_step            : how often (in days) to retrain the ML model

    Returns
    -------
    equity_curve : 1-D np.array starting at 1.0
    positions    : 1-D np.array of daily position sizes (+1 long, -1 short, 0 flat)
    """
    n_days    = len(daily_returns)
    positions = np.zeros(n_days)
    cost_per_trade = transaction_cost_bps / 10_000   # convert bps → decimal

    if strategy == "Buy & Hold":
        # Always fully invested, never trade
        positions[:] = 1.0

    elif strategy == "Momentum":
        lookback = max(initial_train_window // 4, 10)
        for day in range(lookback, n_days):
            # Go long when recent average return is positive, short otherwise
            recent_avg = np.mean(daily_returns[day - lookback : day])
            positions[day] = 1.0 if recent_avg > 0 else -1.0

    elif strategy == "Mean Reversion":
        lookback = max(initial_train_window // 4, 10)
        for day in range(lookback, n_days):
            window_mean = np.mean(daily_returns[day - lookback : day])
            window_std  = np.std( daily_returns[day - lookback : day])
            z_score     = (daily_returns[day - 1] - window_mean) / max(window_std, 1e-8)
            # Fade extreme moves: short after a big rally, long after a big sell-off
            if z_score > 1.5:
                positions[day] = -1.0
            elif z_score < -1.5:
                positions[day] = 1.0
            else:
                positions[day] = positions[day - 1]   # hold current position

    elif strategy == "RF Signal":
        from sklearn.ensemble import RandomForestClassifier as RFClassifier
        lag_days = 10   # use the last 10 days of returns as features

        # Walk-forward: retrain every `refit_step` days, trade the next window
        for train_end in range(initial_train_window, n_days - refit_step + 1, refit_step):
            feature_matrix = np.array([
                daily_returns[i - lag_days : i]
                for i in range(lag_days, train_end)
            ])
            target_labels = np.sign(daily_returns[lag_days : train_end])

            if len(np.unique(target_labels)) < 2:
                continue   # need both classes to train a classifier

            try:
                clf = RFClassifier(n_estimators=30, max_depth=4, random_state=42)
                clf.fit(feature_matrix, target_labels)
                # Predict direction for each day in the upcoming window
                for day in range(train_end, min(train_end + refit_step, n_days)):
                    if day >= lag_days:
                        positions[day] = float(clf.predict([daily_returns[day - lag_days : day]])[0])
            except Exception:
                pass

    elif strategy == "Vol Targeting":
        # Size positions so that realised volatility matches a 10% annualised target
        ann_vol_target = 0.10 / np.sqrt(252)
        lookback = 20
        for day in range(lookback, n_days):
            realised_vol  = np.std(daily_returns[day - lookback : day])
            leverage      = min(ann_vol_target / max(realised_vol, 1e-8), 2.0)
            # Use the sign of recent drift to decide direction
            recent_drift  = np.mean(daily_returns[day - lookback : day])
            positions[day] = leverage if recent_drift > 0 else -leverage

    position_changes = np.diff(positions, prepend=0.0)
    daily_pnl  = positions[1:] * daily_returns[1:] - np.abs(position_changes[1:]) * cost_per_trade
    equity_curve = np.concatenate([[1.0], np.cumprod(1 + np.clip(daily_pnl, -0.5, 0.5))])
    return equity_curve, positions


def _build_features(price_series, return_series):
    """Build a feature matrix suitable for ML models from price and return data.

    Generates rolling mean returns, volatility, momentum, RSI, MACD,
    Bollinger Band position, ATR proxy, and lagged return features.
    The target column is the next-day return (shifted by -1).

    Both inputs must share the same DatetimeIndex.

    Returns
    -------
    feature_matrix : np.array of shape (n_samples, n_features)
    feature_names  : list of column names
    target_array   : np.array of next-day returns
    valid_dates    : DatetimeIndex aligned with the rows
    """
    prices  = price_series.reindex(return_series.index).ffill()
    returns = return_series
    features = pd.DataFrame(index=returns.index)

    # Rolling return stats and momentum at several horizons
    for window in [5, 10, 20, 60]:
        features[f"ret_{window}d"] = returns.rolling(window).mean()
        features[f"vol_{window}d"] = returns.rolling(window).std()
        features[f"mom_{window}d"] = prices / prices.shift(window) - 1

    # RSI(14) — measures momentum via average gains vs. losses
    price_change = prices.diff()
    avg_gain = price_change.clip(lower=0).rolling(14).mean()
    avg_loss = (-price_change.clip(upper=0)).rolling(14).mean()
    features["RSI_14"] = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))

    # MACD — difference between 12-day and 26-day exponential moving averages
    ema_fast = prices.ewm(span=12, adjust=False).mean()
    ema_slow = prices.ewm(span=26, adjust=False).mean()
    features["MACD"]     = ema_fast - ema_slow
    features["MACD_sig"] = features["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger %B — where the price sits relative to its 20-day band
    bb_midline = prices.rolling(20).mean()
    bb_width   = prices.rolling(20).std()
    features["BB_pos"] = (prices - bb_midline) / (2 * bb_width.replace(0, np.nan))

    # ATR proxy — average absolute return over 14 days
    features["ATR_14"] = returns.abs().rolling(14).mean()

    # Lagged returns — give the model a short memory of recent moves
    for lag in [1, 2, 3, 5]:
        features[f"lag_{lag}"] = returns.shift(lag)

    # Target: next-day return (what we're trying to predict)
    features["target"] = returns.shift(-1)
    features = features.dropna()

    feature_cols   = [c for c in features.columns if c != "target"]
    feature_matrix = features[feature_cols].values
    target_array   = features["target"].values

    return feature_matrix, feature_cols, target_array, features.index


with tab16:
    st.markdown('<div class="sec-hdr">16. Backtesting Engine — Walk-Forward · Strategy Library · Risk Attribution · Equity Curves · Benchmark</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="fbox">
    <strong>Walk-Forward:</strong> Expanding/Rolling train → OOS P&L aggregation &nbsp;|&nbsp;
    <strong>Sharpe:</strong> (R̄−Rƒ)/σ·√252 &nbsp;|&nbsp;
    <strong>Max DD:</strong> (peak−trough)/peak &nbsp;|&nbsp;
    <strong>Calmar:</strong> Ann.Return/|MaxDD| &nbsp;|&nbsp;
    <strong>TC Model:</strong> Commission + Slippage + Market Impact
    </div>""", unsafe_allow_html=True)

    _bc1, _bc2, _bc3 = st.columns(3)
    with _bc1:
        bt16_stock  = st.selectbox("Primary stock", selected, key="bt16_stk")
        bt16_strats = st.multiselect("Strategies",
            ["Buy & Hold","Momentum","Mean Reversion","RF Signal","Vol Targeting"],
            default=["Buy & Hold","Momentum","Mean Reversion"], key="bt16_strats")
    with _bc2:
        bt16_tc     = st.slider("Transaction cost (bps)", 0, 50, 10, 1, key="bt16_tc")
        bt16_init   = st.slider("Initial train window (days)", 60, 300, 120, 10, key="bt16_init")
    with _bc3:
        bt16_step   = st.slider("Walk-forward step (days)", 5, 60, 21, 5, key="bt16_step")
        bt16_win    = st.selectbox("Data window", ["Full","2Y","1Y"], key="bt16_win")

    _ret_full = R_df[bt16_stock].values
    if bt16_win == "1Y":   _ret_full = _ret_full[-252:]
    elif bt16_win == "2Y": _ret_full = _ret_full[-504:]
    _n16      = len(_ret_full)
    _dates16  = R_df.index[-_n16:]

    if not bt16_strats:
        st.warning("Select at least one strategy.")
    st.divider()

    bt16a, bt16b, bt16c, bt16d, bt16e, bt16f, bt16g, bt16h = st.tabs([
        "⚙️ Strategy Overview",
        "📈 Equity Curves",
        "📊 Performance Metrics",
        "🔄 Walk-Forward Analysis",
        "🎯 Risk Attribution",
        "💸 Transaction Cost Breakdown",
        "📋 Trade Log",
        "🏆 Benchmark Comparison",
    ])

    # 16A — STRATEGY OVERVIEW
    with bt16a:
        st.markdown("""
        <div class="fbox">
        <strong>Momentum:</strong> long if 20-day mean return > 0, else short &nbsp;|&nbsp;
        <strong>Mean Reversion:</strong> fade z-score extremes (±1.5σ threshold) &nbsp;|&nbsp;
        <strong>RF Signal:</strong> RandomForest on 10 lagged returns (walk-forward retrain) &nbsp;|&nbsp;
        <strong>Vol Targeting:</strong> scale position to hit 10% ann. volatility
        </div>""", unsafe_allow_html=True)

        _eq_a = {}
        for _s in bt16_strats:
            _eq, _ = _run_strategy(_ret_full, _s, bt16_tc, bt16_init, bt16_step)
            _eq_a[_s] = _eq

        if _eq_a:
            _m0 = _bt_metrics(list(_eq_a.values())[0], rf_rate)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Ann. Return",  f"{_m0['ann_ret']*100:.2f}%")
            c2.metric("Sharpe",       f"{_m0['sharpe']:.3f}")
            c3.metric("Max DD",       f"{_m0['max_dd']*100:.2f}%")
            c4.metric("Win Rate",     f"{_m0['win_rate']*100:.1f}%")

        _strat_lib = pd.DataFrame({
            "Strategy":         ["Buy & Hold","Momentum","Mean Reversion","RF Signal","Vol Targeting"],
            "Signal Logic":     ["Always +1","sign(mean R[t-L:t])","−sign(z-score ±1.5σ)","RF classifier on 10 lags","scale to 10% vol target"],
            "Style":            ["Passive","Trend","Contrarian","ML-Adaptive","Risk-Parity"],
            "Retrains":         ["Never","Rolling","Rolling","Walk-Forward","Rolling"],
            "Cost Sensitivity": ["Low","Medium","High","Medium","Low"],
        })
        st.markdown("#### Strategy Library")
        st.dataframe(_strat_lib, use_container_width=True)

        # Quick mini equity chart
        _fig_mini = go.Figure()
        for _i, (_s, _eq) in enumerate(_eq_a.items()):
            _fig_mini.add_trace(go.Scatter(
                x=_dates16[:len(_eq)], y=_eq,
                name=_s, line=dict(color=PALETTE[_i%len(PALETTE)], width=1.8)))
        _fig_mini.update_layout(title=f"Quick Equity Snapshot — {bt16_stock}  |  TC={bt16_tc}bps",
                                 height=300, **lay())
        _fig_mini.update_xaxes(**ax()); _fig_mini.update_yaxes(**ax())
        st.plotly_chart(_fig_mini, use_container_width=True)

    # 16B — EQUITY CURVES & DRAWDOWN
    with bt16b:
        st.markdown("""
        <div class="fbox">
        <strong>Equity Curve:</strong> E(t) = ∏_{s≤t}(1 + r_s − TC_s) &nbsp;|&nbsp;
        <strong>Drawdown:</strong> DD(t) = (E(t) − max_{u≤t}E(u)) / max_{u≤t}E(u) &nbsp;|&nbsp;
        <strong>Rolling Sharpe:</strong> 60-day window, annualised
        </div>""", unsafe_allow_html=True)

        _eq_b = {}
        for _s in bt16_strats:
            _eq, _ = _run_strategy(_ret_full, _s, bt16_tc, bt16_init, bt16_step)
            _eq_b[_s] = _eq

        _fig_eq = make_subplots(rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.06, row_heights=[0.65, 0.35],
            subplot_titles=["Equity Curves (Normalised to 1.0)", "Drawdown (%)"])

        for _i, (_s, _eq) in enumerate(_eq_b.items()):
            _c = PALETTE[_i % len(PALETTE)]
            _x = _dates16[:len(_eq)]
            _fig_eq.add_trace(go.Scatter(x=_x, y=_eq, name=_s,
                line=dict(color=_c, width=1.8)), row=1, col=1)
            _peak = np.maximum.accumulate(_eq)
            _dd   = (_eq - _peak) / np.maximum(_peak, 1e-12) * 100
            _r, _g, _b = int(_c[1:3],16), int(_c[3:5],16), int(_c[5:7],16)
            _fig_eq.add_trace(go.Scatter(x=_x, y=_dd, name=f"{_s} DD",
                line=dict(color=_c, width=1, dash="dot"),
                fill="tozeroy", fillcolor=f"rgba({_r},{_g},{_b},0.07)",
                showlegend=False), row=2, col=1)

        _fig_eq.update_layout(title=f"Equity & Drawdown — {bt16_stock}  |  TC={bt16_tc}bps",
                               height=520, **lay())
        _fig_eq.update_xaxes(**ax()); _fig_eq.update_yaxes(**ax())
        st.plotly_chart(_fig_eq, use_container_width=True)

        # Rolling 60-day Sharpe
        _fig_rsh = go.Figure()
        for _i, (_s, _eq) in enumerate(_eq_b.items()):
            _rets_eq = np.diff(_eq) / np.maximum(np.abs(_eq[:-1]), 1e-12)
            _rsh = pd.Series(_rets_eq).rolling(60, min_periods=20).apply(
                lambda x: (x.mean()*252 - rf_rate) / max(x.std()*np.sqrt(252), 1e-8)
            ).values
            _fig_rsh.add_trace(go.Scatter(x=_dates16[1:len(_rsh)+1], y=_rsh,
                name=_s, line=dict(color=PALETTE[_i%len(PALETTE)], width=1.5)))
        _fig_rsh.add_hline(y=0, line=dict(color="#475569", dash="dot"))
        _fig_rsh.add_hline(y=1, line=dict(color="#10B981", dash="dash", width=0.8),
                           annotation_text="Sharpe=1")
        _fig_rsh.update_layout(title="Rolling 60-Day Sharpe Ratio", height=300, **lay())
        _fig_rsh.update_xaxes(**ax()); _fig_rsh.update_yaxes(**ax())
        st.plotly_chart(_fig_rsh, use_container_width=True)

    # 16C — PERFORMANCE METRICS TABLE
    with bt16c:
        st.markdown("""
        <div class="fbox">
        <strong>Sortino:</strong> (R̄−Rƒ)/σ_down &nbsp;|&nbsp;
        <strong>Omega:</strong> E[max(R,0)] / E[max(−R,0)] &nbsp;|&nbsp;
        <strong>Calmar:</strong> Ann.Return/|MaxDD| &nbsp;|&nbsp;
        <strong>Profit Factor:</strong> Gross Gains / Gross Losses
        </div>""", unsafe_allow_html=True)

        _rows_c = []
        for _s in bt16_strats:
            _eq, _pos = _run_strategy(_ret_full, _s, bt16_tc, bt16_init, bt16_step)
            _m = _bt_metrics(_eq, rf_rate)
            _rets_s = np.diff(_eq) / np.maximum(np.abs(_eq[:-1]), 1e-12)
            _gross_g = _rets_s[_rets_s > 0].sum() if (_rets_s > 0).any() else 1e-8
            _gross_l = abs(_rets_s[_rets_s < 0].sum()) if (_rets_s < 0).any() else 1e-8
            _rows_c.append({
                "Strategy":      _s,
                "Ann. Return":   f"{_m['ann_ret']*100:.2f}%",
                "Ann. Vol":      f"{_m['ann_vol']*100:.2f}%",
                "Sharpe":        f"{_m['sharpe']:.3f}",
                "Sortino":       f"{_m['sortino']:.3f}",
                "Max DD":        f"{_m['max_dd']*100:.2f}%",
                "Calmar":        f"{_m['calmar']:.3f}",
                "Omega":         f"{_m['omega']:.3f}",
                "Win Rate":      f"{_m['win_rate']*100:.1f}%",
                "Profit Factor": f"{_gross_g/_gross_l:.3f}",
                "Days":          _m['n_days'],
            })

        _pm_df = pd.DataFrame(_rows_c)
        st.markdown("#### Comprehensive Performance Table")
        st.dataframe(_pm_df, use_container_width=True)

        # Radar chart
        if len(bt16_strats) >= 2:
            _cats = ["Sharpe","Sortino","Calmar","Omega","Win Rate"]
            _fig_rad = go.Figure()
            for _i, _row in enumerate(_rows_c):
                _vals = [
                    float(_row["Sharpe"]),
                    float(_row["Sortino"]),
                    min(float(_row["Calmar"]), 5),
                    min(float(_row["Omega"]), 5),
                    float(_row["Win Rate"].strip("%")) / 100,
                ]
                _vn = [(_v - (-2)) / (5 - (-2)) for _v in _vals]
                _c2 = PALETTE[_i % len(PALETTE)]
                _r2, _g2, _b2 = int(_c2[1:3],16), int(_c2[3:5],16), int(_c2[5:7],16)
                _fig_rad.add_trace(go.Scatterpolar(
                    r=_vn + [_vn[0]], theta=_cats + [_cats[0]],
                    name=_row["Strategy"],
                    line=dict(color=_c2, width=2), fill="toself",
                    fillcolor=f"rgba({_r2},{_g2},{_b2},0.08)"))
            _fig_rad.update_layout(
                polar=dict(bgcolor="#0D1829",
                           radialaxis=dict(visible=True, range=[0,1], color="#4A6080"),
                           angularaxis=dict(color="#94A3B8")),
                title="Strategy Radar — Normalised Metrics", height=420, **lay())
            st.plotly_chart(_fig_rad, use_container_width=True)

        # Monthly return heatmap
        if bt16_strats:
            _eq_h, _ = _run_strategy(_ret_full, bt16_strats[0], bt16_tc, bt16_init, bt16_step)
            _n_use   = min(len(_eq_h)-1, _n16)
            _dh      = _dates16[:_n_use]
            _rh      = np.diff(_eq_h[:_n_use+1]) / np.maximum(np.abs(_eq_h[:_n_use]), 1e-12)
            _ret_s   = pd.Series(_rh, index=_dh)
            _monthly = _ret_s.resample("ME").sum() * 100
            _m_df    = _monthly.to_frame("ret")
            _m_df["year"]  = _m_df.index.year
            _m_df["month"] = _m_df.index.strftime("%b")
            _pivot   = _m_df.pivot_table(index="year", columns="month", values="ret")
            _mo      = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            _pivot   = _pivot.reindex(columns=[_m for _m in _mo if _m in _pivot.columns])
            _fig_hm  = go.Figure(go.Heatmap(
                z=_pivot.values, x=_pivot.columns.tolist(), y=[str(_y) for _y in _pivot.index],
                colorscale=[[0,"#7F1D1D"],[0.5,"#0D1829"],[1,"#064E3B"]],
                zmid=0, text=np.round(_pivot.values,1),
                texttemplate="%{text}%", textfont=dict(size=9),
                colorbar=dict(title="Return%", tickfont=dict(color="#94A3B8"))))
            _fig_hm.update_layout(
                title=f"Monthly Returns Heatmap — {bt16_strats[0]} on {bt16_stock}",
                height=300, **lay())
            _fig_hm.update_xaxes(**ax()); _fig_hm.update_yaxes(**ax())
            st.plotly_chart(_fig_hm, use_container_width=True)

    # 16D — WALK-FORWARD ANALYSIS
    with bt16d:
        st.markdown("""
        <div class="fbox">
        <strong>Expanding:</strong> train on all history up to t, test t→t+step &nbsp;|&nbsp;
        <strong>Rolling:</strong> fixed-size train window slides forward &nbsp;|&nbsp;
        <strong>Efficiency Ratio:</strong> OOS Sharpe / IS Sharpe  (>0.5 = robust, <0 = overfit)
        </div>""", unsafe_allow_html=True)

        _dc1, _dc2 = st.columns([1, 2])
        with _dc1:
            _wf16_mode = st.selectbox("Window type", ["Expanding","Rolling"], key="wf16_mode")
            _wf16_lb   = st.slider("Momentum lookback (days)", 5, 60, 20, 5, key="wf16_lb")
            _wf16_init = st.slider("Init train size (days)", 60, 252, 120, 10, key="wf16_init")
            _wf16_step = st.slider("Step size (days)", 5, 60, 21, 5, key="wf16_stp")

        _ret_wf  = R_df[bt16_stock].values
        _n_wf    = len(_ret_wf)
        _folds   = []

        for _start in range(_wf16_init, _n_wf - _wf16_step + 1, _wf16_step):
            _tr0     = max(0, _start - _wf16_init) if _wf16_mode == "Rolling" else 0
            _r_train = _ret_wf[_tr0:_start]
            _r_test  = _ret_wf[_start:_start + _wf16_step]
            if len(_r_train) < 20 or len(_r_test) == 0:
                continue

            # IS signal & metrics
            _sig_is  = np.sign(np.convolve(_r_train, np.ones(_wf16_lb)/_wf16_lb, mode='valid'))
            _eq_is   = np.cumprod(1 + _sig_is * _r_train[_wf16_lb-1:])
            _is_sh   = (_bt_metrics(_eq_is, rf_rate)["sharpe"] if len(_eq_is) > 2 else 0)

            # OOS
            _last_sig = float(np.sign(np.mean(_r_train[-_wf16_lb:])))
            _eq_oos   = np.cumprod(1 + _last_sig * _r_test)
            _oos_ret  = float(np.mean(_r_test) * _last_sig * 252)
            _oos_vol  = float(np.std(_r_test) * np.sqrt(252))
            _oos_sh   = (_oos_ret - rf_rate) / max(_oos_vol, 1e-8)
            _eff      = _oos_sh / max(abs(_is_sh), 1e-8)

            _folds.append({
                "Fold":       len(_folds)+1,
                "Start":      R_df.index[_start].date(),
                "End":        R_df.index[min(_start+_wf16_step-1, _n_wf-1)].date(),
                "IS Sharpe":  round(_is_sh, 3),
                "OOS Sharpe": round(_oos_sh, 3),
                "OOS Ret%":   round(_oos_ret*100, 2),
                "Efficiency": round(_eff, 3),
                "_eq_oos":    _eq_oos,
            })

        with _dc2:
            if _folds:
                _avg_eff = np.mean([_f["Efficiency"] for _f in _folds])
                _avg_oos = np.mean([_f["OOS Sharpe"] for _f in _folds])
                _avg_is  = np.mean([_f["IS Sharpe"]  for _f in _folds])
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Folds",          len(_folds))
                c2.metric("Avg IS Sharpe",  f"{_avg_is:.3f}")
                c3.metric("Avg OOS Sharpe", f"{_avg_oos:.3f}")
                c4.metric("Efficiency",     f"{_avg_eff:.3f}",
                          delta="Robust" if _avg_eff > 0.5 else "Fragile")

        if _folds:
            _fold_df = pd.DataFrame([{k:v for k,v in _f.items() if k != "_eq_oos"} for _f in _folds])
            st.markdown("#### Walk-Forward Fold Results")
            st.dataframe(_fold_df, use_container_width=True)

            _fig_wf = make_subplots(rows=1, cols=2,
                subplot_titles=["IS vs OOS Sharpe per Fold", "OOS Equity (Stitched)"])
            _fig_wf.add_trace(go.Scatter(
                x=[_f["IS Sharpe"] for _f in _folds],
                y=[_f["OOS Sharpe"] for _f in _folds],
                mode="markers+text",
                text=[f"F{_f['Fold']}" for _f in _folds],
                textposition="top center",
                marker=dict(color=PALETTE[0], size=9, opacity=0.8),
                name="Folds"), row=1, col=1)
            _fig_wf.add_shape(type="line", x0=-3, x1=3, y0=-3, y1=3,
                line=dict(color="#475569",dash="dot"), row=1, col=1)
            _fig_wf.add_hline(y=0, line=dict(color="#EF4444",dash="dash",width=0.8), row=1, col=1)

            _st_eq = [1.0]
            for _f in _folds:
                _scale = _st_eq[-1]
                _st_eq.extend((_f["_eq_oos"] * _scale).tolist())
            _fig_wf.add_trace(go.Scatter(x=list(range(len(_st_eq))), y=_st_eq,
                name="OOS Equity", line=dict(color="#10B981", width=2)), row=1, col=2)
            _fig_wf.add_hline(y=1.0, line=dict(color="#475569",dash="dot"), row=1, col=2)
            _fig_wf.update_layout(
                title=f"Walk-Forward — {bt16_stock}  |  {_wf16_mode}  |  {len(_folds)} folds",
                height=420, **lay())
            _fig_wf.update_xaxes(**ax()); _fig_wf.update_yaxes(**ax())
            st.plotly_chart(_fig_wf, use_container_width=True)

    # 16E — RISK ATTRIBUTION
    with bt16e:
        st.markdown("""
        <div class="fbox">
        <strong>Factor Attribution:</strong> R_strat = α + β_MKT·R_mkt + β_MOM·R_mom + ε &nbsp;|&nbsp;
        <strong>Variance Decomp:</strong> σ²_p = β²_MKT·σ²_MKT + β²_MOM·σ²_MOM + σ²_idio &nbsp;|&nbsp;
        <strong>Rolling β:</strong> 60-day Cov(strat, mkt) / Var(mkt)
        </div>""", unsafe_allow_html=True)

        _ra_strat = st.selectbox("Strategy", bt16_strats if bt16_strats else ["Buy & Hold"], key="ra16_s")
        _eq_ra, _ = _run_strategy(_ret_full, _ra_strat, bt16_tc, bt16_init, bt16_step)
        _st_rets  = np.diff(_eq_ra) / np.maximum(np.abs(_eq_ra[:-1]), 1e-12)
        _n_ra     = len(_st_rets)
        _mkt_ret  = _ret_full[1:_n_ra+1]
        _mom_f    = pd.Series(_ret_full).rolling(20).mean().values[1:_n_ra+1]
        _mom_f    = np.nan_to_num(_mom_f)

        _X_ra = np.column_stack([np.ones(_n_ra), _mkt_ret, _mom_f])
        try:
            _b_ra  = np.linalg.lstsq(_X_ra, _st_rets, rcond=None)[0]
            _alpha_ra, _beta_mkt, _beta_mom = _b_ra
            _resid_ra = _st_rets - _X_ra @ _b_ra
            _r2_ra    = 1 - _resid_ra.var() / max(_st_rets.var(), 1e-12)
        except Exception:
            _alpha_ra, _beta_mkt, _beta_mom, _r2_ra, _resid_ra = 0, 1, 0, 0, _st_rets

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("α (daily ann.)",  f"{_alpha_ra*252*100:.3f}%")
        c2.metric("β Market",        f"{_beta_mkt:.4f}")
        c3.metric("β Momentum",      f"{_beta_mom:.4f}")
        c4.metric("Factor R²",       f"{_r2_ra:.4f}")

        _v_mkt  = (_beta_mkt**2) * np.var(_mkt_ret)
        _v_mom  = (_beta_mom**2) * np.var(_mom_f)
        _v_idio = np.var(_resid_ra)
        _v_tot  = max(_v_mkt + _v_mom + _v_idio, 1e-12)

        _fig_ra = make_subplots(rows=1, cols=2,
            subplot_titles=["Variance Attribution (%)", "Rolling Beta to Market (60d)"])
        _fig_ra.add_trace(go.Bar(
            x=["Market β","Momentum β","Idiosyncratic"],
            y=[_v_mkt/_v_tot*100, _v_mom/_v_tot*100, _v_idio/_v_tot*100],
            marker=dict(color=["#00D4FF","#F59E0B","#10B981"], opacity=0.85),
            text=[f"{_v:.1f}%" for _v in [_v_mkt/_v_tot*100,_v_mom/_v_tot*100,_v_idio/_v_tot*100]],
            textposition="outside"), row=1, col=1)

        _rb = pd.Series(_st_rets).rolling(60).cov(pd.Series(_mkt_ret)) / \
              pd.Series(_mkt_ret).rolling(60).var()
        _rb = _rb.fillna(0).values
        _fig_ra.add_trace(go.Scatter(x=R_df.index[-len(_rb):], y=_rb,
            line=dict(color="#00D4FF",width=1.5), name="Rolling β"), row=1, col=2)
        _fig_ra.add_hline(y=1, line=dict(color="#475569",dash="dot"),
                          annotation_text="β=1", row=1, col=2)
        _fig_ra.update_layout(title=f"Risk Attribution — {_ra_strat} on {bt16_stock}",
                               height=380, **lay())
        _fig_ra.update_xaxes(**ax()); _fig_ra.update_yaxes(**ax())
        st.plotly_chart(_fig_ra, use_container_width=True)

        # Monthly P&L attribution bar
        _attr_df = pd.DataFrame({
            "Market":    _beta_mkt * _mkt_ret * 100,
            "Momentum":  _beta_mom * _mom_f   * 100,
            "Alpha":     _resid_ra * 100,
        }, index=R_df.index[-_n_ra:])
        _m_attr = _attr_df.resample("ME").sum()
        _fig_attr = go.Figure()
        for _col, _c3 in zip(["Market","Momentum","Alpha"],["#00D4FF","#F59E0B","#10B981"]):
            _fig_attr.add_trace(go.Bar(x=_m_attr.index, y=_m_attr[_col],
                name=_col, marker=dict(color=_c3, opacity=0.8)))
        _fig_attr.update_layout(barmode="stack",
            title="Monthly P&L Attribution (Market / Momentum / Alpha)",
            height=320, **lay())
        _fig_attr.update_xaxes(**ax()); _fig_attr.update_yaxes(title_text="Return (%)", **ax())
        st.plotly_chart(_fig_attr, use_container_width=True)

    # 16F — TRANSACTION COST BREAKDOWN
    with bt16f:
        st.markdown("""
        <div class="fbox">
        <strong>Commission:</strong> fixed bps per trade &nbsp;|&nbsp;
        <strong>Slippage:</strong> σ × vol × position_size &nbsp;|&nbsp;
        <strong>Market Impact:</strong> k × √(order / avg_vol) &nbsp;|&nbsp;
        <strong>Spread:</strong> bid-ask / 2 per trade &nbsp;|&nbsp;
        <strong>Net Return</strong> = Gross − Commission − Slippage − Impact
        </div>""", unsafe_allow_html=True)

        _fc1, _fc2 = st.columns([1, 2])
        with _fc1:
            _tc_strat   = st.selectbox("Strategy", bt16_strats if bt16_strats else ["Buy & Hold"], key="tc16_s")
            _comm_bps   = st.slider("Commission (bps)", 0, 30, 5, 1, key="tc16_comm")
            _slip_bps   = st.slider("Slippage (bps)", 0, 20, 3, 1, key="tc16_slip")
            _impact_bps = st.slider("Market Impact (bps)", 0, 15, 2, 1, key="tc16_imp")
            _spread_bps = st.slider("Spread cost (bps)", 0, 10, 2, 1, key="tc16_sprd")

        _eq_tc_gross, _pos_tc = _run_strategy(_ret_full, _tc_strat, 0, bt16_init, bt16_step)
        _eq_tc_net,   _       = _run_strategy(_ret_full, _tc_strat,
                                               _comm_bps+_slip_bps+_impact_bps+_spread_bps,
                                               bt16_init, bt16_step)
        _trades_tc = np.abs(np.diff(_pos_tc, prepend=0))
        _n_trades  = int((_trades_tc > 0.01).sum())
        _turnover  = float(_trades_tc.mean() * 252 * 100)

        _cost_gross = float(np.prod(1 + np.diff(_eq_tc_gross)/np.maximum(np.abs(_eq_tc_gross[:-1]),1e-12)) - 1) * 100
        _cost_net   = float(np.prod(1 + np.diff(_eq_tc_net  )/np.maximum(np.abs(_eq_tc_net  [:-1]),1e-12)) - 1) * 100
        _cost_drag  = _cost_gross - _cost_net

        with _fc2:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Gross Return",     f"{_cost_gross:.2f}%")
            c2.metric("Net Return",       f"{_cost_net:.2f}%")
            c3.metric("Cost Drag",        f"{_cost_drag:.2f}%")
            c4.metric("Trade Count",      f"{_n_trades:,}")

        _fig_tc = make_subplots(rows=1, cols=2,
            subplot_titles=["Gross vs Net Equity Curve", "TC Component Breakdown"])
        _xd = _dates16[:len(_eq_tc_gross)]
        _fig_tc.add_trace(go.Scatter(x=_xd, y=_eq_tc_gross, name="Gross (no TC)",
            line=dict(color="#10B981",width=1.8)), row=1, col=1)
        _fig_tc.add_trace(go.Scatter(x=_xd, y=_eq_tc_net,   name="Net (with TC)",
            line=dict(color="#EF4444",width=1.8,dash="dash")), row=1, col=1)
        _fig_tc.add_trace(go.Bar(
            x=["Commission","Slippage","Mkt Impact","Spread"],
            y=[_comm_bps, _slip_bps, _impact_bps, _spread_bps],
            marker=dict(color=["#00D4FF","#F59E0B","#EF4444","#8B5CF6"], opacity=0.85),
            text=[f"{_v}bps" for _v in [_comm_bps,_slip_bps,_impact_bps,_spread_bps]],
            textposition="outside"), row=1, col=2)
        _fig_tc.update_layout(title=f"Transaction Cost Analysis — {_tc_strat}",
                               height=380, **lay())
        _fig_tc.update_xaxes(**ax()); _fig_tc.update_yaxes(**ax())
        st.plotly_chart(_fig_tc, use_container_width=True)

        _cost_tbl = pd.DataFrame({
            "Component":     ["Commission","Slippage","Market Impact","Bid-Ask Spread","Total"],
            "Rate (bps)":    [_comm_bps,_slip_bps,_impact_bps,_spread_bps,
                              _comm_bps+_slip_bps+_impact_bps+_spread_bps],
            "Model":         ["Fixed per trade","σ×vol×size","k×√(order/vol)","spread/2 per side","Sum"],
            "Ann. Cost Est.":  [f"{_comm_bps*_n_trades/(max(_n16,1))*252:.1f}bps",
                                f"{_slip_bps*_n_trades/(max(_n16,1))*252:.1f}bps",
                                f"{_impact_bps*_n_trades/(max(_n16,1))*252:.1f}bps",
                                f"{_spread_bps*_n_trades/(max(_n16,1))*252:.1f}bps",
                                f"{(_comm_bps+_slip_bps+_impact_bps+_spread_bps)*_n_trades/(max(_n16,1))*252:.1f}bps"],
        })
        st.markdown("#### Cost Decomposition Table")
        st.dataframe(_cost_tbl, use_container_width=True)

    # 16G — TRADE LOG
    with bt16g:
        st.markdown("""
        <div class="fbox">
        <strong>Trade Log:</strong> each position change = one trade entry &nbsp;|&nbsp;
        <strong>P&L per Trade:</strong> position × return of holding period − TC &nbsp;|&nbsp;
        <strong>MAE/MFE:</strong> Max Adverse / Favourable Excursion during trade
        </div>""", unsafe_allow_html=True)

        _tl_strat = st.selectbox("Strategy", bt16_strats if bt16_strats else ["Buy & Hold"], key="tl16_s")
        _eq_tl, _pos_tl = _run_strategy(_ret_full, _tl_strat, bt16_tc, bt16_init, bt16_step)
        _chg = np.diff(_pos_tl, prepend=0)
        _trade_rows = []
        _in_trade   = False
        _entry_i    = 0
        _entry_pos  = 0.0

        for _i in range(1, _n16):
            if not _in_trade and abs(_pos_tl[_i]) > 0.01:
                _in_trade = True; _entry_i = _i; _entry_pos = _pos_tl[_i]
            elif _in_trade and (abs(_pos_tl[_i] - _entry_pos) > 0.5 or _i == _n16-1):
                _period = _ret_full[_entry_i:_i]
                _pnl    = float(_entry_pos * np.sum(_period) * 100)
                _mae    = float(min(np.cumsum(_period) * _entry_pos) * 100) if len(_period) else 0
                _mfe    = float(max(np.cumsum(_period) * _entry_pos) * 100) if len(_period) else 0
                _trade_rows.append({
                    "Entry Date": _dates16[_entry_i].date() if _entry_i < len(_dates16) else "—",
                    "Exit Date":  _dates16[_i].date()       if _i < len(_dates16) else "—",
                    "Position":   "Long" if _entry_pos > 0 else "Short",
                    "Days Held":  _i - _entry_i,
                    "Return (%)": f"{_pnl:.3f}",
                    "MAE (%)":    f"{_mae:.3f}",
                    "MFE (%)":    f"{_mfe:.3f}",
                    "W/L":        "W" if _pnl > 0 else "L",
                })
                _in_trade = False
                if abs(_pos_tl[_i]) > 0.01:
                    _in_trade = True; _entry_i = _i; _entry_pos = _pos_tl[_i]

        if _trade_rows:
            _tl_df = pd.DataFrame(_trade_rows)
            _n_win  = (_tl_df["W/L"] == "W").sum()
            _n_los  = (_tl_df["W/L"] == "L").sum()
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Trades",  len(_tl_df))
            c2.metric("Winners",       f"{_n_win} ({_n_win/max(len(_tl_df),1)*100:.1f}%)")
            c3.metric("Losers",        f"{_n_los} ({_n_los/max(len(_tl_df),1)*100:.1f}%)")
            c4.metric("Avg Hold (days)", f"{_tl_df['Days Held'].mean():.1f}")
            st.markdown("#### Trade-by-Trade Log")
            st.dataframe(_tl_df, use_container_width=True)

            # P&L distribution
            _pnl_vals = _tl_df["Return (%)"].astype(float).values
            _fig_tl = make_subplots(rows=1, cols=2,
                subplot_titles=["Trade P&L Distribution", "MAE vs MFE Scatter"])
            _fig_tl.add_trace(go.Histogram(x=_pnl_vals, nbinsx=30, name="P&L",
                marker=dict(color="#00D4FF", opacity=0.7)), row=1, col=1)
            _fig_tl.add_vline(x=0, line=dict(color="#EF4444",dash="dot"), row=1, col=1)
            _mae_vals = _tl_df["MAE (%)"].astype(float).values
            _mfe_vals = _tl_df["MFE (%)"].astype(float).values
            _fig_tl.add_trace(go.Scatter(x=_mae_vals, y=_mfe_vals, mode="markers",
                marker=dict(color=[PALETTE[0] if _w=="W" else PALETTE[5]
                                   for _w in _tl_df["W/L"]],
                            size=7, opacity=0.8), name="Trades"), row=1, col=2)
            _fig_tl.add_hline(y=0, line=dict(color="#475569",dash="dot"), row=1, col=2)
            _fig_tl.add_vline(x=0, line=dict(color="#475569",dash="dot"), row=1, col=2)
            _fig_tl.update_layout(title=f"Trade Analysis — {_tl_strat}", height=380, **lay())
            _fig_tl.update_xaxes(**ax()); _fig_tl.update_yaxes(**ax())
            st.plotly_chart(_fig_tl, use_container_width=True)
        else:
            st.info("No complete trades found in this window.")

    # 16H — BENCHMARK COMPARISON
    with bt16h:
        st.markdown("""
        <div class="fbox">
        <strong>Benchmark:</strong> Equal-Weight portfolio of all selected stocks &nbsp;|&nbsp;
        <strong>Information Ratio:</strong> (R_strat − R_bench) / σ(R_strat − R_bench) &nbsp;|&nbsp;
        <strong>Up/Down Capture:</strong> % of benchmark up/down moves captured by strategy
        </div>""", unsafe_allow_html=True)

        # Build equal-weight benchmark returns from all selected stocks
        _bench_rets = R_df[selected].mean(axis=1).values[-_n16:]
        _bench_eq   = np.concatenate([[1.0], np.cumprod(1 + np.clip(_bench_rets[1:], -0.5, 0.5))])

        _bm_rows = []
        for _s in bt16_strats:
            _eq_bm, _ = _run_strategy(_ret_full, _s, bt16_tc, bt16_init, bt16_step)
            _st_r  = np.diff(_eq_bm) / np.maximum(np.abs(_eq_bm[:-1]), 1e-12)
            _bh_r  = np.diff(_bench_eq) / np.maximum(np.abs(_bench_eq[:-1]), 1e-12)
            _n_min = min(len(_st_r), len(_bh_r))
            _st_r, _bh_r = _st_r[:_n_min], _bh_r[:_n_min]
            _active_r  = _st_r - _bh_r
            _ir        = float(np.mean(_active_r)*252) / max(float(np.std(_active_r)*np.sqrt(252)), 1e-8)
            _up_mask   = _bh_r > 0
            _dn_mask   = _bh_r < 0
            _up_cap    = float(_st_r[_up_mask].mean() / max(_bh_r[_up_mask].mean(), 1e-8) * 100) if _up_mask.any() else 0
            _dn_cap    = float(_st_r[_dn_mask].mean() / min(_bh_r[_dn_mask].mean(), -1e-8) * 100) if _dn_mask.any() else 0
            _m_bm      = _bt_metrics(_eq_bm, rf_rate)
            _m_bench   = _bt_metrics(_bench_eq, rf_rate)
            _bm_rows.append({
                "Strategy":        _s,
                "Ann. Return":     f"{_m_bm['ann_ret']*100:.2f}%",
                "Bench Return":    f"{_m_bench['ann_ret']*100:.2f}%",
                "Active Return":   f"{(_m_bm['ann_ret']-_m_bench['ann_ret'])*100:.2f}%",
                "Info Ratio":      f"{_ir:.3f}",
                "Up Capture%":     f"{_up_cap:.1f}%",
                "Down Capture%":   f"{_dn_cap:.1f}%",
                "Sharpe (Strat)":  f"{_m_bm['sharpe']:.3f}",
                "Sharpe (Bench)":  f"{_m_bench['sharpe']:.3f}",
            })

        st.markdown("#### Benchmark Comparison Table")
        st.dataframe(pd.DataFrame(_bm_rows), use_container_width=True)

        _fig_bm = go.Figure()
        _fig_bm.add_trace(go.Scatter(x=_dates16[:len(_bench_eq)], y=_bench_eq,
            name="EW Benchmark", line=dict(color="#475569",width=2,dash="dot")))
        for _i, _s in enumerate(bt16_strats):
            _eq_bm2, _ = _run_strategy(_ret_full, _s, bt16_tc, bt16_init, bt16_step)
            _fig_bm.add_trace(go.Scatter(x=_dates16[:len(_eq_bm2)], y=_eq_bm2,
                name=_s, line=dict(color=PALETTE[_i%len(PALETTE)],width=1.8)))
        _fig_bm.update_layout(title=f"Strategy vs EW Benchmark — {bt16_stock}",
                               height=400, **lay())
        _fig_bm.update_xaxes(**ax()); _fig_bm.update_yaxes(**ax())
        st.plotly_chart(_fig_bm, use_container_width=True)


with tab17:
    st.markdown('<div class="sec-hdr">17. ML Models Hub — Feature Engineering · Training · CV · Hyperparameter Tuning · Explainability · Forecast Comparison</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="fbox">
    <strong>Features:</strong> RSI · MACD · Bollinger · ATR · Momentum · Lags &nbsp;|&nbsp;
    <strong>Models:</strong> Ridge · Lasso · Random Forest · Gradient Boosting · SVR · MLP &nbsp;|&nbsp;
    <strong>CV:</strong> TimeSeriesSplit (no data leakage) &nbsp;|&nbsp;
    <strong>Explainability:</strong> Feature Importance · Permutation · Partial Dependence
    </div>""", unsafe_allow_html=True)

    # Shared stock selector above sub-tabs
    _ml17_stock = st.selectbox("Primary stock for ML Hub", selected, key="ml17_stk")
    _prices17   = closes[_ml17_stock]
    _returns17  = R_df[_ml17_stock]
    _X17, _feat17, _y17, _idx17 = _build_features(_prices17, _returns17)

    st.divider()

    ml17a, ml17b, ml17c, ml17d, ml17e, ml17f, ml17g, ml17h = st.tabs([
        "🔧 Feature Engineering",
        "🏋️ Model Training & CV",
        "📊 Forecast Comparison",
        "🔍 Hyperparameter Tuning",
        "💡 Feature Importance",
        "🔬 Permutation & PD Plots",
        "📈 Model Registry",
        "🎯 Live Prediction",
    ])

    # 17A — FEATURE ENGINEERING
    with ml17a:
        st.markdown("""
        <div class="fbox">
        <strong>RSI(14):</strong> 100 − 100/(1+AvgGain/AvgLoss) &nbsp;|&nbsp;
        <strong>MACD:</strong> EMA(12) − EMA(26), Signal = EMA(MACD,9) &nbsp;|&nbsp;
        <strong>Bollinger %B:</strong> (P − μ₂₀)/(2σ₂₀) &nbsp;|&nbsp;
        <strong>ATR:</strong> mean(|r|, 14d) &nbsp;|&nbsp;
        <strong>Lag Features:</strong> R(t−1)…R(t−5)
        </div>""", unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Feature Count",  len(_feat17))
        c2.metric("Sample Count",   len(_X17))
        _rsi_idx = _feat17.index("RSI_14") if "RSI_14" in _feat17 else 0
        c3.metric("RSI (latest)",   f"{_X17[-1, _rsi_idx]:.1f}")
        _macd_idx = _feat17.index("MACD") if "MACD" in _feat17 else 0
        c4.metric("MACD (latest)",  f"{_X17[-1, _macd_idx]:.5f}")

        # Correlation heatmap
        _corr17 = pd.DataFrame(_X17, columns=_feat17).corr().values
        _fig_fc = go.Figure(go.Heatmap(
            z=_corr17, x=_feat17, y=_feat17,
            colorscale=[[0,"#EF4444"],[0.5,"#0D1829"],[1,"#10B981"]], zmid=0,
            text=np.round(_corr17,2), texttemplate="%{text}", textfont=dict(size=7),
            colorbar=dict(title="ρ", tickfont=dict(color="#94A3B8"))))
        _fig_fc.update_layout(title=f"Feature Correlation Matrix — {_ml17_stock}",
                               height=500, **lay())
        _fig_fc.update_xaxes(tickangle=-45, **ax())
        _fig_fc.update_yaxes(**ax())
        st.plotly_chart(_fig_fc, use_container_width=True)

        # Feature time series panel
        _panel = [("RSI_14","RSI"),("MACD","MACD"),("BB_pos","Bollinger %B"),
                  ("ATR_14","ATR"),("vol_20d","20d Vol")]
        _fig_fp = make_subplots(rows=len(_panel), cols=1, shared_xaxes=True,
            vertical_spacing=0.03, subplot_titles=[_p[1] for _p in _panel])
        for _pi, (_pk, _pn) in enumerate(_panel, 1):
            if _pk in _feat17:
                _fi = _feat17.index(_pk)
                _fig_fp.add_trace(go.Scatter(x=_idx17, y=_X17[:,_fi],
                    line=dict(color=PALETTE[(_pi-1)%len(PALETTE)],width=1.2),
                    name=_pn), row=_pi, col=1)
        _fig_fp.update_layout(title=f"Feature Time Series — {_ml17_stock}",
                               height=640, showlegend=False, **lay())
        _fig_fp.update_xaxes(**ax()); _fig_fp.update_yaxes(**ax())
        st.plotly_chart(_fig_fp, use_container_width=True)

    # 17B — MODEL TRAINING & CV
    with ml17b:
        from sklearn.linear_model import Ridge as _Ridge, Lasso as _Lasso
        from sklearn.ensemble import RandomForestRegressor as _RF, GradientBoostingRegressor as _GB
        from sklearn.svm import SVR as _SVR
        from sklearn.neural_network import MLPRegressor as _MLP
        from sklearn.model_selection import TimeSeriesSplit as _TSCV
        from sklearn.preprocessing import StandardScaler as _SS

        st.markdown("""
        <div class="fbox">
        <strong>TimeSeriesSplit:</strong> no data leakage — train always precedes test &nbsp;|&nbsp;
        <strong>OOS R²:</strong> 1 − RSS/TSS on hold-out set &nbsp;|&nbsp;
        <strong>Hit Rate:</strong> % correct directional calls
        </div>""", unsafe_allow_html=True)

        _tc1, _tc2 = st.columns([1, 2])
        with _tc1:
            _ml17_cv    = st.slider("CV Folds", 3, 8, 5, key="ml17_cv")
            _ml17_test  = st.slider("Hold-out test %", 10, 30, 20, key="ml17_test")
            _ml17_mdls  = st.multiselect("Models",
                ["Ridge","Lasso","Random Forest","Gradient Boosting","SVR","MLP"],
                default=["Ridge","Random Forest","Gradient Boosting"], key="ml17_mdls")

        _sp17    = int(len(_X17) * (1 - _ml17_test/100))
        _scx17   = _SS(); _scy17 = _SS()
        _Xtr17   = _scx17.fit_transform(_X17[:_sp17])
        _Xte17   = _scx17.transform(_X17[_sp17:])
        _ytr17   = _scy17.fit_transform(_y17[:_sp17].reshape(-1,1)).ravel()
        _yte17   = _y17[_sp17:]
        _tscv17  = _TSCV(n_splits=_ml17_cv)

        _model_map17 = {
            "Ridge":             _Ridge(alpha=0.01),
            "Lasso":             _Lasso(alpha=0.001, max_iter=5000),
            "Random Forest":     _RF(n_estimators=60, max_depth=5, random_state=42),
            "Gradient Boosting": _GB(n_estimators=60, max_depth=3, random_state=42),
            "SVR":               _SVR(kernel="rbf", C=1.0, epsilon=0.01),
            "MLP":               _MLP(hidden_layer_sizes=(64,32), max_iter=300,
                                      random_state=42, early_stopping=True),
        }
        _cv_rows17 = []
        _oos_preds17 = {}

        with _tc2:
            if not _ml17_mdls:
                st.info("Select at least one model.")
            else:
                for _mn in _ml17_mdls:
                    _mdl = _model_map17[_mn]
                    _folds_r2 = []
                    for _tri, _tei in _tscv17.split(_Xtr17):
                        try:
                            _mdl.fit(_Xtr17[_tri], _ytr17[_tri])
                            _pp = _mdl.predict(_Xtr17[_tei])
                            _ss_r = ((_ytr17[_tei]-_pp)**2).sum()
                            _ss_t = ((_ytr17[_tei]-_ytr17[_tei].mean())**2).sum()
                            _folds_r2.append(1 - _ss_r/max(_ss_t,1e-12))
                        except Exception:
                            _folds_r2.append(np.nan)

                    try:
                        _mdl.fit(_Xtr17, _ytr17)
                        _pp_te_s = _mdl.predict(_Xte17)
                        _pp_te   = _scy17.inverse_transform(_pp_te_s.reshape(-1,1)).ravel()
                        _ss_r2   = ((_yte17-_pp_te)**2).sum()
                        _ss_t2   = ((_yte17-_yte17.mean())**2).sum()
                        _r2_oos  = 1-_ss_r2/max(_ss_t2,1e-12)
                        _rmse_17 = float(np.sqrt(np.mean((_yte17-_pp_te)**2)))
                        _hit17   = float(np.mean(np.sign(_yte17)==np.sign(_pp_te)))
                    except Exception:
                        _pp_te, _r2_oos, _rmse_17, _hit17 = np.zeros_like(_yte17), np.nan, np.nan, np.nan

                    _oos_preds17[_mn] = _pp_te
                    _cv_rows17.append({
                        "Model":       _mn,
                        "CV Mean R²":  f"{np.nanmean(_folds_r2):.4f}",
                        "CV Std R²":   f"{np.nanstd(_folds_r2):.4f}",
                        "OOS R²":      f"{_r2_oos:.4f}",
                        "RMSE (OOS)":  f"{_rmse_17*100:.4f}%",
                        "Hit Rate":    f"{_hit17*100:.1f}%",
                    })

                st.markdown("#### Model CV Results")
                st.dataframe(pd.DataFrame(_cv_rows17), use_container_width=True)

                # CV R² box plot
                _fig_cv17 = go.Figure()
                for _ii, _mn in enumerate(_ml17_mdls):
                    _mdl2 = _model_map17[_mn]
                    _fr2  = []
                    for _tri, _tei in _tscv17.split(_Xtr17):
                        try:
                            _mdl2.fit(_Xtr17[_tri], _ytr17[_tri])
                            _pp2 = _mdl2.predict(_Xtr17[_tei])
                            _s_r = ((_ytr17[_tei]-_pp2)**2).sum()
                            _s_t = ((_ytr17[_tei]-_ytr17[_tei].mean())**2).sum()
                            _fr2.append(1-_s_r/max(_s_t,1e-12))
                        except Exception:
                            _fr2.append(0)
                    _fig_cv17.add_trace(go.Box(y=_fr2, name=_mn,
                        marker=dict(color=PALETTE[_ii%len(PALETTE)]), boxmean=True))
                _fig_cv17.update_layout(title="CV R² Distribution per Model",
                                         height=340, **lay())
                _fig_cv17.update_yaxes(title_text="R²", **ax())
                st.plotly_chart(_fig_cv17, use_container_width=True)

    # 17C — FORECAST COMPARISON
    with ml17c:
        st.markdown("""
        <div class="fbox">
        <strong>MAE:</strong> mean|ŷ−y| &nbsp;|&nbsp;
        <strong>MASE:</strong> MAE / MAE_naive (naive = yesterday's return) &nbsp;|&nbsp;
        <strong>Hit Rate:</strong> % correct sign predictions &nbsp;|&nbsp;
        <strong>OOS R²:</strong> 1 − RSS/TSS on hold-out
        </div>""", unsafe_allow_html=True)

        from sklearn.linear_model import Ridge as _R17c
        from sklearn.ensemble import RandomForestRegressor as _RF17c, GradientBoostingRegressor as _GB17c
        from sklearn.preprocessing import StandardScaler as _SS17c

        _sp17c = int(len(_X17)*0.8)
        _sc17c = _SS17c()
        _Xtr17c = _sc17c.fit_transform(_X17[:_sp17c])
        _Xte17c = _sc17c.transform(_X17[_sp17c:])
        _ytr17c, _yte17c = _y17[:_sp17c], _y17[_sp17c:]
        _idx17c = _idx17[_sp17c:]
        _naive_mae = float(np.mean(np.abs(_yte17c)))

        _preds17c = {}
        for _mn2, _mdl3 in [("Ridge",_R17c(alpha=0.01)),
                              ("Random Forest",_RF17c(n_estimators=60,max_depth=5,random_state=42)),
                              ("Gradient Boosting",_GB17c(n_estimators=60,max_depth=3,random_state=42))]:
            try:
                _mdl3.fit(_Xtr17c, _ytr17c)
                _preds17c[_mn2] = _mdl3.predict(_Xte17c)
            except Exception:
                _preds17c[_mn2] = np.zeros_like(_yte17c)

        _fc_rows = []
        for _mn2, _pp in _preds17c.items():
            _mae17   = float(np.mean(np.abs(_yte17c - _pp)))
            _mase17  = _mae17 / max(_naive_mae, 1e-12)
            _hit17c  = float(np.mean(np.sign(_yte17c)==np.sign(_pp)))
            _ssr17   = ((_yte17c-_pp)**2).sum()
            _sst17   = ((_yte17c-_yte17c.mean())**2).sum()
            _r217    = 1 - _ssr17/max(_sst17,1e-12)
            _fc_rows.append({"Model":_mn2,"MAE":f"{_mae17*100:.4f}%","MASE":f"{_mase17:.3f}",
                             "Hit Rate":f"{_hit17c*100:.1f}%","OOS R²":f"{_r217:.4f}"})

        st.markdown("#### Forecast Accuracy Table")
        st.dataframe(pd.DataFrame(_fc_rows), use_container_width=True)

        _fig_fcp = go.Figure()
        _fig_fcp.add_trace(go.Scatter(x=_idx17c, y=_yte17c*100, name="Actual",
            line=dict(color="#94A3B8",width=1)))
        for _ii2, (_mn2, _pp) in enumerate(_preds17c.items()):
            _fig_fcp.add_trace(go.Scatter(x=_idx17c, y=_pp*100, name=_mn2,
                line=dict(color=PALETTE[_ii2%len(PALETTE)],width=1.4,dash="dash")))
        _fig_fcp.update_layout(title=f"Forecast vs Actual — {_ml17_stock}",
                                height=380, **lay())
        _fig_fcp.update_xaxes(**ax()); _fig_fcp.update_yaxes(title_text="Return (%)", **ax())
        st.plotly_chart(_fig_fcp, use_container_width=True)

        # Predicted vs Actual scatter
        _fig_pvsa = make_subplots(rows=1, cols=len(_preds17c),
            subplot_titles=list(_preds17c.keys()))
        for _ii2, (_mn2, _pp) in enumerate(_preds17c.items(), 1):
            _fig_pvsa.add_trace(go.Scatter(x=_yte17c*100, y=_pp*100, mode="markers",
                marker=dict(color=PALETTE[(_ii2-1)%len(PALETTE)],size=3,opacity=0.5),
                name=_mn2), row=1, col=_ii2)
            _mn_ = float(_yte17c.min()*100); _mx_ = float(_yte17c.max()*100)
            _fig_pvsa.add_shape(type="line",x0=_mn_,x1=_mx_,y0=_mn_,y1=_mx_,
                line=dict(color="#475569",dash="dot"),row=1,col=_ii2)
        _fig_pvsa.update_layout(title="Predicted vs Actual Scatter",
                                 height=340, showlegend=False, **lay())
        _fig_pvsa.update_xaxes(title_text="Actual (%)",**ax())
        _fig_pvsa.update_yaxes(title_text="Predicted (%)",**ax())
        st.plotly_chart(_fig_pvsa, use_container_width=True)

    # 17D — HYPERPARAMETER TUNING
    with ml17d:
        from sklearn.model_selection import TimeSeriesSplit as _TSCV17d
        from sklearn.linear_model import Ridge as _R17d
        from sklearn.ensemble import RandomForestRegressor as _RF17d, GradientBoostingRegressor as _GB17d
        from sklearn.preprocessing import StandardScaler as _SS17d

        st.markdown("""
        <div class="fbox">
        <strong>Grid Search:</strong> exhaustive search over param grid &nbsp;|&nbsp;
        <strong>CV Objective:</strong> minimise mean OOS MSE across TimeSeriesSplit folds &nbsp;|&nbsp;
        <strong>Visualisation:</strong> heatmap (tree models) or learning curve (Ridge)
        </div>""", unsafe_allow_html=True)

        _hd1, _hd2 = st.columns([1, 2])
        with _hd1:
            _hp17_mdl = st.selectbox("Model",
                ["Ridge (λ grid)","Random Forest","Gradient Boosting"], key="hp17_mdl")
            _hp17_cv  = st.slider("CV folds", 3, 6, 4, key="hp17_cv")

        _sc17d  = _SS17d()
        _Xhps   = _sc17d.fit_transform(_X17)
        _tscv17d= _TSCV17d(n_splits=_hp17_cv)

        if _hp17_mdl == "Ridge (λ grid)":
            _lams = np.logspace(-4, 2, 30)
            _hscores = []
            for _lam in _lams:
                _fmse = []
                for _tri, _tei in _tscv17d.split(_Xhps):
                    _m_ = _R17d(alpha=_lam)
                    _m_.fit(_Xhps[_tri], _y17[_tri])
                    _pp_ = _m_.predict(_Xhps[_tei])
                    _fmse.append(float(np.mean((_y17[_tei]-_pp_)**2)))
                _hscores.append(np.mean(_fmse))
            _best_lam_i = int(np.argmin(_hscores))
            with _hd2:
                c1,c2 = st.columns(2)
                c1.metric("Best λ",           f"{_lams[_best_lam_i]:.5f}")
                c2.metric("CV MSE at λ*",     f"{_hscores[_best_lam_i]*1e4:.4f}×10⁻⁴")
            _fig_lam = go.Figure()
            _fig_lam.add_trace(go.Scatter(x=np.log10(_lams), y=_hscores,
                mode="lines+markers", line=dict(color="#00D4FF",width=2), marker=dict(size=5)))
            _fig_lam.add_vline(x=np.log10(_lams[_best_lam_i]),
                line=dict(color="#10B981",dash="dash"),
                annotation_text=f"λ*={_lams[_best_lam_i]:.4f}")
            _fig_lam.update_layout(title=f"Ridge λ Grid Search — {_ml17_stock}",
                                    height=380, **lay())
            _fig_lam.update_xaxes(title_text="log₁₀(λ)", **ax())
            _fig_lam.update_yaxes(title_text="CV MSE", **ax())
            st.plotly_chart(_fig_lam, use_container_width=True)

        else:
            _n_ests = [20, 50, 100, 200]
            _depths = [2, 3, 4, 5]
            _grid   = np.zeros((len(_n_ests), len(_depths)))
            for _ii, _ne in enumerate(_n_ests):
                for _jj, _md in enumerate(_depths):
                    _fmse2 = []
                    for _tri, _tei in _tscv17d.split(_Xhps):
                        try:
                            if _hp17_mdl == "Random Forest":
                                _m2 = _RF17d(n_estimators=_ne, max_depth=_md, random_state=42)
                            else:
                                _m2 = _GB17d(n_estimators=_ne, max_depth=_md, random_state=42)
                            _m2.fit(_Xhps[_tri], _y17[_tri])
                            _pp2 = _m2.predict(_Xhps[_tei])
                            _fmse2.append(float(np.mean((_y17[_tei]-_pp2)**2)))
                        except Exception:
                            _fmse2.append(np.nan)
                    _grid[_ii,_jj] = np.nanmean(_fmse2)
            _bi, _bj = np.unravel_index(np.nanargmin(_grid), _grid.shape)
            with _hd2:
                c1,c2,c3 = st.columns(3)
                c1.metric("Best n_estimators", _n_ests[_bi])
                c2.metric("Best max_depth",     _depths[_bj])
                c3.metric("CV MSE at best",     f"{_grid[_bi,_bj]*1e4:.4f}×10⁻⁴")
            _fig_grid = go.Figure(go.Heatmap(
                z=_grid*1e4, x=[f"depth={_d}" for _d in _depths],
                y=[f"n={_n}" for _n in _n_ests],
                colorscale=[[0,"#064E3B"],[1,"#7F1D1D"]],
                text=np.round(_grid*1e4,3), texttemplate="%{text}", textfont=dict(size=9),
                colorbar=dict(title="CV MSE×10⁴", tickfont=dict(color="#94A3B8"))))
            _fig_grid.update_layout(title=f"{_hp17_mdl} Grid Search — {_ml17_stock}",
                                     height=380, **lay())
            _fig_grid.update_xaxes(**ax()); _fig_grid.update_yaxes(**ax())
            st.plotly_chart(_fig_grid, use_container_width=True)

    # 17E — FEATURE IMPORTANCE
    with ml17e:
        from sklearn.ensemble import RandomForestRegressor as _RF17e, GradientBoostingRegressor as _GB17e
        from sklearn.preprocessing import StandardScaler as _SS17e

        st.markdown("""
        <div class="fbox">
        <strong>Gain Importance:</strong> total impurity reduction attributed to each feature &nbsp;|&nbsp;
        <strong>Comparison:</strong> RF vs Gradient Boosting feature rankings side-by-side
        </div>""", unsafe_allow_html=True)

        _n_top17e = st.slider("Top N features to show", 5, len(_feat17), min(10,len(_feat17)), key="top17e")
        _sc17e    = _SS17e()
        _Xes      = _sc17e.fit_transform(_X17)

        _rf17e = _RF17e(n_estimators=150, max_depth=5, random_state=42)
        _gb17e = _GB17e(n_estimators=150, max_depth=3, random_state=42)
        _rf17e.fit(_Xes, _y17); _gb17e.fit(_Xes, _y17)

        _imp_rf = pd.DataFrame({"feature":_feat17,"RF Imp":_rf17e.feature_importances_}
                               ).sort_values("RF Imp", ascending=False).head(_n_top17e)
        _imp_gb = pd.DataFrame({"feature":_feat17,"GB Imp":_gb17e.feature_importances_}
                               ).sort_values("GB Imp", ascending=False).head(_n_top17e)

        c1,c2 = st.columns(2)
        with c1:
            c1.metric("RF Top Feature", _imp_rf.iloc[0]["feature"])
        with c2:
            c2.metric("GB Top Feature", _imp_gb.iloc[0]["feature"])

        _fig_imp = make_subplots(rows=1, cols=2,
            subplot_titles=[f"Random Forest — Top {_n_top17e}",
                             f"Gradient Boosting — Top {_n_top17e}"])
        _fig_imp.add_trace(go.Bar(x=_imp_rf["RF Imp"]*100, y=_imp_rf["feature"],
            orientation="h", marker=dict(color="#00D4FF",opacity=0.85),
            text=[f"{_v:.2f}%" for _v in _imp_rf["RF Imp"]*100],
            textposition="outside", name="RF"), row=1, col=1)
        _fig_imp.add_trace(go.Bar(x=_imp_gb["GB Imp"]*100, y=_imp_gb["feature"],
            orientation="h", marker=dict(color="#F59E0B",opacity=0.85),
            text=[f"{_v:.2f}%" for _v in _imp_gb["GB Imp"]*100],
            textposition="outside", name="GB"), row=1, col=2)
        _fig_imp.update_layout(title=f"Feature Importance Comparison — {_ml17_stock}",
                                height=420, showlegend=False, **lay())
        _fig_imp.update_xaxes(title_text="Importance (%)", **ax())
        _fig_imp.update_yaxes(**ax())
        st.plotly_chart(_fig_imp, use_container_width=True)

    # 17F — PERMUTATION IMPORTANCE & PARTIAL DEPENDENCE
    with ml17f:
        from sklearn.ensemble import RandomForestRegressor as _RF17f
        from sklearn.preprocessing import StandardScaler as _SS17f

        st.markdown("""
        <div class="fbox">
        <strong>Permutation Importance:</strong> ΔR² when feature i is randomly shuffled &nbsp;|&nbsp;
        <strong>Partial Dependence:</strong> E_{X_{-i}}[f(xᵢ, X_{-i})] — marginal effect of one feature &nbsp;|&nbsp;
        Higher ΔR² = more important; PD shows direction and shape of effect
        </div>""", unsafe_allow_html=True)

        _n_top17f = st.slider("Top N features", 5, len(_feat17), min(10,len(_feat17)), key="top17f")
        _sc17f    = _SS17f()
        _Xfs      = _sc17f.fit_transform(_X17)
        _rf17f    = _RF17f(n_estimators=150, max_depth=5, random_state=42)
        _rf17f.fit(_Xfs, _y17)

        _base_pred = _rf17f.predict(_Xfs)
        _sst_f     = ((_y17-_y17.mean())**2).sum()
        _base_r2   = 1 - ((_y17-_base_pred)**2).sum()/max(_sst_f,1e-12)
        _perm_imp  = []
        np.random.seed(42)
        for _fi, _fn in enumerate(_feat17):
            _Xp = _Xfs.copy(); _Xp[:,_fi] = np.random.permutation(_Xp[:,_fi])
            _pp = _rf17f.predict(_Xp)
            _perm_imp.append({"feature":_fn, "R2_drop":_base_r2 - (1-((_y17-_pp)**2).sum()/max(_sst_f,1e-12))})

        _perm_df = pd.DataFrame(_perm_imp).sort_values("R2_drop",ascending=False).head(_n_top17f)

        # Partial dependence for top feature
        _top_fi = _feat17.index(_perm_df.iloc[0]["feature"])
        _gv     = np.percentile(_Xfs[:,_top_fi], np.linspace(5,95,40))
        _pd_v   = []
        for _g in _gv:
            _Xg = _Xfs.copy(); _Xg[:,_top_fi] = _g
            _pd_v.append(float(_rf17f.predict(_Xg).mean()))

        _fig_perm = make_subplots(rows=1, cols=2,
            subplot_titles=[f"Permutation Importance — Top {_n_top17f}",
                             f"Partial Dependence — {_perm_df.iloc[0]['feature']}"])
        _fig_perm.add_trace(go.Bar(x=_perm_df["R2_drop"], y=_perm_df["feature"],
            orientation="h", marker=dict(color="#F59E0B",opacity=0.85),
            text=[f"{_v:.4f}" for _v in _perm_df["R2_drop"]],
            textposition="outside", name="ΔR²"), row=1, col=1)
        _fig_perm.add_trace(go.Scatter(x=_gv, y=[_v*100 for _v in _pd_v],
            mode="lines+markers", line=dict(color="#00D4FF",width=2), name="PD"), row=1, col=2)
        _fig_perm.add_hline(y=0, line=dict(color="#475569",dash="dot"), row=1, col=2)
        _fig_perm.update_layout(title=f"Permutation Importance & PD — {_ml17_stock}",
                                 height=400, **lay())
        _fig_perm.update_xaxes(**ax()); _fig_perm.update_yaxes(**ax())
        st.plotly_chart(_fig_perm, use_container_width=True)

    # 17G — MODEL REGISTRY
    with ml17g:
        st.markdown("""
        <div class="fbox">
        <strong>Model Registry:</strong> track trained models, versions, OOS metrics, parameters &nbsp;|&nbsp;
        <strong>Status:</strong> Staging → Production → Archived &nbsp;|&nbsp;
        <strong>Challenger:</strong> compare Production vs Staging OOS R²
        </div>""", unsafe_allow_html=True)

        from sklearn.linear_model import Ridge as _R17g
        from sklearn.ensemble import RandomForestRegressor as _RF17g, GradientBoostingRegressor as _GB17g
        from sklearn.preprocessing import StandardScaler as _SS17g

        _sc17g = _SS17g()
        _Xgs   = _sc17g.fit_transform(_X17)
        _sp17g = int(len(_X17)*0.8)

        _reg_rows = []
        for _mname, _mcfg in [
            ("Ridge v1.0",       {"model":_R17g(alpha=0.01),             "params":"α=0.01",                    "status":"Production"}),
            ("Ridge v1.1",       {"model":_R17g(alpha=0.001),            "params":"α=0.001",                   "status":"Staging"}),
            ("Random Forest v2", {"model":_RF17g(n_estimators=60,max_depth=5,random_state=42), "params":"n=100,d=5", "status":"Production"}),
            ("GB v1.0",          {"model":_GB17g(n_estimators=60,max_depth=3,random_state=42), "params":"n=100,d=3", "status":"Staging"}),
            ("GB v0.9",          {"model":_GB17g(n_estimators=50, max_depth=3,random_state=42), "params":"n=50,d=3",  "status":"Archived"}),
        ]:
            try:
                _mc = _mcfg["model"]
                _mc.fit(_Xgs[:_sp17g], _y17[:_sp17g])
                _pp_g = _mc.predict(_Xgs[_sp17g:])
                _yte_g = _y17[_sp17g:]
                _r2_g  = 1 - ((_yte_g-_pp_g)**2).sum()/max(((_yte_g-_yte_g.mean())**2).sum(),1e-12)
                _rmse_g = float(np.sqrt(np.mean((_yte_g-_pp_g)**2)))
                _hit_g  = float(np.mean(np.sign(_yte_g)==np.sign(_pp_g)))
            except Exception:
                _r2_g, _rmse_g, _hit_g = np.nan, np.nan, np.nan

            _status_icon = {"Production":"🟢","Staging":"🟡","Archived":"🔴"}
            _reg_rows.append({
                "Model":       _mname,
                "Status":      f"{_status_icon.get(_mcfg['status'],'⚪')} {_mcfg['status']}",
                "Parameters":  _mcfg["params"],
                "OOS R²":      f"{_r2_g:.4f}",
                "RMSE":        f"{_rmse_g*100:.4f}%",
                "Hit Rate":    f"{_hit_g*100:.1f}%",
            })

        _reg_df = pd.DataFrame(_reg_rows)
        st.markdown("#### Model Registry")
        st.dataframe(_reg_df, use_container_width=True)

        # OOS R² bar comparison
        _fig_reg = go.Figure(go.Bar(
            x=[_r["Model"] for _r in _reg_rows],
            y=[float(_r["OOS R²"]) for _r in _reg_rows],
            marker=dict(color=[
                "#10B981" if "Production" in _r["Status"] else
                "#F59E0B" if "Staging" in _r["Status"] else "#EF4444"
                for _r in _reg_rows], opacity=0.85),
            text=[_r["OOS R²"] for _r in _reg_rows],
            textposition="outside"))
        _fig_reg.add_hline(y=0, line=dict(color="#475569",dash="dot"))
        _fig_reg.update_layout(title="OOS R² by Model Version",
                                height=360, **lay())
        _fig_reg.update_xaxes(tickangle=-20, **ax())
        _fig_reg.update_yaxes(title_text="OOS R²", **ax())
        st.plotly_chart(_fig_reg, use_container_width=True)

    # 17H — LIVE PREDICTION
    with ml17h:
        from sklearn.ensemble import RandomForestRegressor as _RF17h, GradientBoostingRegressor as _GB17h
        from sklearn.linear_model import Ridge as _R17h
        from sklearn.preprocessing import StandardScaler as _SS17h

        st.markdown("""
        <div class="fbox">
        <strong>Live Prediction:</strong> train on all history → forecast next-day return &nbsp;|&nbsp;
        <strong>Ensemble:</strong> weighted average by inverse RMSE &nbsp;|&nbsp;
        <strong>Confidence:</strong> std of model predictions (dispersion = uncertainty)
        </div>""", unsafe_allow_html=True)

        _sc17h = _SS17h()
        _Xhs   = _sc17h.fit_transform(_X17)
        _last_x = _Xhs[[-1]]   # latest feature vector

        _live_models = {
            "Ridge":             _R17h(alpha=0.01),
            "Random Forest":     _RF17h(n_estimators=60,max_depth=5,random_state=42),
            "Gradient Boosting": _GB17h(n_estimators=60,max_depth=3,random_state=42),
        }
        _live_preds  = {}
        _live_rmses  = {}

        _sp17h = int(len(_X17)*0.8)
        for _mn_h, _mdl_h in _live_models.items():
            try:
                _mdl_h.fit(_Xhs, _y17)
                _pp_h = _mdl_h.predict(_Xhs[_sp17h:])
                _live_rmses[_mn_h] = float(np.sqrt(np.mean((_y17[_sp17h:]-_pp_h)**2)))
                _live_preds[_mn_h] = float(_mdl_h.predict(_last_x)[0])
            except Exception:
                _live_rmses[_mn_h] = 1.0
                _live_preds[_mn_h] = 0.0

        _inv_rmse = {_k: 1/max(_v,1e-8) for _k,_v in _live_rmses.items()}
        _w_sum    = sum(_inv_rmse.values())
        _ens_pred = sum(_live_preds[_k]*_inv_rmse[_k]/_w_sum for _k in _live_preds)
        _ens_std  = float(np.std(list(_live_preds.values())))
        _direction = "▲ LONG" if _ens_pred > 0 else "▼ SHORT"
        _conf_str  = "High" if _ens_std < 0.001 else "Medium" if _ens_std < 0.003 else "Low"

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Ensemble Forecast",    f"{_ens_pred*100:.4f}%")
        c2.metric("Signal Direction",     _direction)
        c3.metric("Model Dispersion σ",   f"{_ens_std*100:.4f}%")
        c4.metric("Confidence",           _conf_str)

        _live_rows = []
        for _mn_h, _pp_h in _live_preds.items():
            _w = _inv_rmse[_mn_h] / _w_sum
            _live_rows.append({
                "Model":         _mn_h,
                "Forecast":      f"{_pp_h*100:.4f}%",
                "Direction":     "▲" if _pp_h > 0 else "▼",
                "Weight":        f"{_w*100:.1f}%",
                "RMSE (hist)":   f"{_live_rmses[_mn_h]*100:.4f}%",
            })
        st.markdown("#### Model-by-Model Forecasts")
        st.dataframe(pd.DataFrame(_live_rows), use_container_width=True)

        _track_preds = {_mn_h: [] for _mn_h in _live_models}
        _track_actual = []
        _track_len    = min(60, len(_X17)-1)
        for _ti in range(_track_len, 0, -1):
            _Xtr_t = _Xhs[:-_ti]
            _Xte_t = _Xhs[[-_ti]]
            _y_act  = _y17[-_ti]
            _track_actual.append(_y_act)
            for _mn_h, _mdl_h in _live_models.items():
                try:
                    _mdl_h.fit(_Xtr_t, _y17[:len(_Xtr_t)])
                    _track_preds[_mn_h].append(float(_mdl_h.predict(_Xte_t)[0]))
                except Exception:
                    _track_preds[_mn_h].append(0.0)

        _track_dates = _idx17[-_track_len:]
        _fig_live = go.Figure()
        _fig_live.add_trace(go.Scatter(x=_track_dates, y=np.array(_track_actual)*100,
            name="Actual", line=dict(color="#94A3B8",width=1.2)))
        for _ii_h, (_mn_h, _tv) in enumerate(_track_preds.items()):
            _fig_live.add_trace(go.Scatter(x=_track_dates, y=np.array(_tv)*100,
                name=_mn_h, line=dict(color=PALETTE[_ii_h%len(PALETTE)],width=1.4,dash="dash")))
        _fig_live.update_layout(
            title=f"Rolling 1-Day-Ahead Forecast Track — {_ml17_stock}  (last {_track_len} days)",
            height=380, **lay())
        _fig_live.update_xaxes(**ax()); _fig_live.update_yaxes(title_text="Return (%)", **ax())
        st.plotly_chart(_fig_live, use_container_width=True)


# AUTHOR LINE
st.markdown("""
<div style="text-align:center;font-family:'JetBrains Mono';
            color:#1C3A5E;font-size:.75rem;padding:18px 0 10px;letter-spacing:.8px;">
  Created by &nbsp;<span style="color:#00D4FF;font-weight:700;font-size:.82rem;">Daniyal Aziz</span>
</div>
""", unsafe_allow_html=True)
