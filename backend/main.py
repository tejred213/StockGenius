from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import logging

from curl_cffi.requests import Session as CffiSession
import yfinance as yf
from ml_engine import StockMLEngine
from nifty50 import compare_nifty50, NIFTY50_TICKERS
from smallcap import compare_smallcap, SMALLCAP_TICKERS
from midcap import compare_midcap, MIDCAP_TICKERS
from options_advisor import get_options_recommendation

logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Analytics API")

# Shared curl_cffi session — reuses TCP connections & bypasses bot detection
_session = CffiSession(impersonate="chrome")

@app.get("/api/health")
def health_check():
    return {"status": "awake", "message": "Backend is active"}

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared ML engine instance
ml_engine = StockMLEngine()


# ======================================================================
# Ticker alias map — common names / abbreviations → actual NSE tickers
# ======================================================================
TICKER_ALIASES = {
    # Nifty 50 — full names, short names, and common variations
    "ADANI ENTERPRISES": "ADANIENT.NS", "ADANIENT": "ADANIENT.NS", "ADANI ENT": "ADANIENT.NS",
    "ADANI PORTS": "ADANIPORTS.NS", "ADANIPORTS": "ADANIPORTS.NS",
    "APOLLO HOSPITALS": "APOLLOHOSP.NS", "APOLLOHOSP": "APOLLOHOSP.NS", "APOLLO": "APOLLOHOSP.NS",
    "ASIAN PAINTS": "ASIANPAINT.NS", "ASIANPAINT": "ASIANPAINT.NS", "ASIAN PAINT": "ASIANPAINT.NS",
    "AXIS BANK": "AXISBANK.NS", "AXISBANK": "AXISBANK.NS", "AXIS": "AXISBANK.NS",
    "BAJAJ AUTO": "BAJAJ-AUTO.NS", "BAJAJ-AUTO": "BAJAJ-AUTO.NS", "BAJAJAUTO": "BAJAJ-AUTO.NS",
    "BAJAJ FINANCE": "BAJFINANCE.NS", "BAJFINANCE": "BAJFINANCE.NS", "BAJ FINANCE": "BAJFINANCE.NS",
    "BAJAJ FINSERV": "BAJAJFINSV.NS", "BAJAJFINSV": "BAJAJFINSV.NS", "BAJ FINSERV": "BAJAJFINSV.NS",
    "BPCL": "BPCL.NS", "BHARAT PETROLEUM": "BPCL.NS",
    "BHARTI AIRTEL": "BHARTIARTL.NS", "BHARTIARTL": "BHARTIARTL.NS", "AIRTEL": "BHARTIARTL.NS",
    "BRITANNIA": "BRITANNIA.NS",
    "CIPLA": "CIPLA.NS",
    "COAL INDIA": "COALINDIA.NS", "COALINDIA": "COALINDIA.NS",
    "DIVIS LAB": "DIVISLAB.NS", "DIVISLAB": "DIVISLAB.NS", "DIVI'S LAB": "DIVISLAB.NS", "DIVIS": "DIVISLAB.NS",
    "DR REDDY": "DRREDDY.NS", "DRREDDY": "DRREDDY.NS", "DR REDDYS": "DRREDDY.NS", "DR. REDDY'S": "DRREDDY.NS",
    "EICHER MOTORS": "EICHERMOT.NS", "EICHERMOT": "EICHERMOT.NS", "EICHER": "EICHERMOT.NS",
    "GRASIM": "GRASIM.NS", "GRASIM INDUSTRIES": "GRASIM.NS",
    "HCL TECH": "HCLTECH.NS", "HCLTECH": "HCLTECH.NS", "HCL TECHNOLOGIES": "HCLTECH.NS", "HCL": "HCLTECH.NS",
    "HDFC BANK": "HDFCBANK.NS", "HDFCBANK": "HDFCBANK.NS", "HDFC": "HDFCBANK.NS",
    "HDFC LIFE": "HDFCLIFE.NS", "HDFCLIFE": "HDFCLIFE.NS",
    "HERO MOTOCORP": "HEROMOTOCO.NS", "HEROMOTOCO": "HEROMOTOCO.NS", "HERO": "HEROMOTOCO.NS",
    "HINDALCO": "HINDALCO.NS",
    "HINDUSTAN UNILEVER": "HINDUNILVR.NS", "HINDUNILVR": "HINDUNILVR.NS", "HUL": "HINDUNILVR.NS",
    "ICICI BANK": "ICICIBANK.NS", "ICICIBANK": "ICICIBANK.NS", "ICICI": "ICICIBANK.NS",
    "ITC": "ITC.NS",
    "INDUSIND BANK": "INDUSINDBK.NS", "INDUSINDBK": "INDUSINDBK.NS", "INDUSIND": "INDUSINDBK.NS",
    "INFOSYS": "INFY.NS", "INFY": "INFY.NS",
    "JSW STEEL": "JSWSTEEL.NS", "JSWSTEEL": "JSWSTEEL.NS", "JSW": "JSWSTEEL.NS",
    "KOTAK BANK": "KOTAKBANK.NS", "KOTAKBANK": "KOTAKBANK.NS", "KOTAK": "KOTAKBANK.NS", "KOTAK MAHINDRA": "KOTAKBANK.NS",
    "LARSEN AND TOUBRO": "LT.NS", "LARSEN & TOUBRO": "LT.NS", "L&T": "LT.NS", "LT": "LT.NS",
    "MAHINDRA": "M&M.NS", "M&M": "M&M.NS", "MAHINDRA AND MAHINDRA": "M&M.NS", "MAHINDRA & MAHINDRA": "M&M.NS",
    "MARUTI": "MARUTI.NS", "MARUTI SUZUKI": "MARUTI.NS",
    "NESTLE": "NESTLEIND.NS", "NESTLEIND": "NESTLEIND.NS", "NESTLE INDIA": "NESTLEIND.NS",
    "NTPC": "NTPC.NS",
    "ONGC": "ONGC.NS",
    "POWER GRID": "POWERGRID.NS", "POWERGRID": "POWERGRID.NS",
    "RELIANCE": "RELIANCE.NS", "RELIANCE INDUSTRIES": "RELIANCE.NS", "RIL": "RELIANCE.NS",
    "SBI LIFE": "SBILIFE.NS", "SBILIFE": "SBILIFE.NS",
    "SBI": "SBIN.NS", "SBIN": "SBIN.NS", "STATE BANK": "SBIN.NS", "STATE BANK OF INDIA": "SBIN.NS",
    "SUN PHARMA": "SUNPHARMA.NS", "SUNPHARMA": "SUNPHARMA.NS",
    "TCS": "TCS.NS", "TATA CONSULTANCY": "TCS.NS",
    "TATA CONSUMER": "TATACONSUM.NS", "TATACONSUM": "TATACONSUM.NS",
    "TATA MOTORS": "TATAMOTORS.NS", "TATAMOTORS": "TATAMOTORS.NS",
    "TATA STEEL": "TATASTEEL.NS", "TATASTEEL": "TATASTEEL.NS",
    "TECH MAHINDRA": "TECHM.NS", "TECHM": "TECHM.NS", "TECH M": "TECHM.NS",
    "TITAN": "TITAN.NS", "TITAN COMPANY": "TITAN.NS",
    "ULTRATECH CEMENT": "ULTRACEMCO.NS", "ULTRACEMCO": "ULTRACEMCO.NS", "ULTRATECH": "ULTRACEMCO.NS",
    "UPL": "UPL.NS",
    "WIPRO": "WIPRO.NS",
    "SHRIRAM FINANCE": "SHRIRAMFIN.NS", "SHRIRAMFIN": "SHRIRAMFIN.NS", "SHRIRAM": "SHRIRAMFIN.NS",
    # Common non-Nifty50 stocks
    "ZOMATO": "ZOMATO.NS",
    "PAYTM": "PAYTM.NS",
    "NYKAA": "NYKAA.NS",
    "LIC": "LICI.NS", "LICI": "LICI.NS",
    "VEDANTA": "VEDL.NS", "VEDL": "VEDL.NS",
    "TATA POWER": "TATAPOWER.NS", "TATAPOWER": "TATAPOWER.NS",
    "TATA ELXSI": "TATAELXSI.NS", "TATAELXSI": "TATAELXSI.NS",
    "IRCTC": "IRCTC.NS",
    "HAL": "HAL.NS", "HINDUSTAN AERONAUTICS": "HAL.NS",
    "PNB": "PNB.NS", "PUNJAB NATIONAL BANK": "PNB.NS",
    "BANK OF BARODA": "BANKBARODA.NS", "BANKBARODA": "BANKBARODA.NS", "BOB": "BANKBARODA.NS",
    "CANARA BANK": "CANBK.NS", "CANBK": "CANBK.NS",
    "INDIAN OIL": "IOC.NS", "IOC": "IOC.NS",
    "BHEL": "BHEL.NS",
    "SAIL": "SAIL.NS",
    "GAIL": "GAIL.NS",
    "NHPC": "NHPC.NS",
    # Indices
    "NIFTY 50": "^NSEI", "NIFTY50": "^NSEI", "NIFTY": "^NSEI",
    "NIFTY NEXT 50": "^CRSLDX", "NIFTY NEXT": "^CRSLDX", "NIFTY NEXT50": "^CRSLDX",
    "NIFTY BANK": "^NSEBANK", "BANKNIFTY": "^NSEBANK", "BANK NIFTY": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY MIDCAP 50": "^NSEMDCP50", "NIFTY MIDCAP": "^NSEMDCP50",
}


def normalize_ticker(ticker: str) -> str:
    """
    Resolves a stock name/alias to its actual NSE ticker.
    1. Check the alias map for common names (e.g. 'Infosys' → 'INFY.NS')
    2. If already has an exchange suffix (.NS, .BO) — return as-is
    3. Otherwise append '.NS' as default
    """
    t = ticker.strip().upper()

    # Check alias map first
    if t in TICKER_ALIASES:
        return TICKER_ALIASES[t]

    # If it is an index starting with ^ — leave it
    if t.startswith("^"):
        return t

    # If it already has a dot suffix like .NS, .BO, .L etc — leave it
    if "." in t:
        return t

    return f"{t}.NS"


@app.get("/")
def read_root():
    return {"message": "Stock Analytics Backend is running"}


# ======================================================================
# ML Evaluation Endpoints
# ======================================================================

@app.get("/api/stocks/price/{ticker}")
def get_live_price(ticker: str):
    """
    Fast endpoint to get the live price for a ticker
    without running the full ML evaluation.
    Uses a shared session for connection reuse and has a
    fallback chain: fast_info → history(1d) close.
    """
    normalized = normalize_ticker(ticker)
    tkr = yf.Ticker(normalized, session=_session)

    ltp = None
    previous_close = None

    # --- Attempt 1: fast_info (fastest, real-time) ---
    try:
        fi = tkr.fast_info
        ltp = round(float(fi["lastPrice"]), 2)
        previous_close = round(float(fi.get("previousClose", 0)), 2)
    except Exception as e:
        logger.warning("fast_info failed for %s: %s — falling back to history", normalized, e)

    # --- Attempt 2: latest history close (reliable fallback) ---
    if ltp is None:
        try:
            hist = tkr.history(period="5d")
            if hist is not None and not hist.empty:
                import pandas as pd
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)
                ltp = round(float(hist["Close"].iloc[-1]), 2)
                if len(hist) >= 2:
                    previous_close = round(float(hist["Close"].iloc[-2]), 2)
        except Exception as e2:
            logger.error("history fallback also failed for %s: %s", normalized, e2)
            raise HTTPException(status_code=400, detail=f"Could not fetch price for {normalized}")

    if ltp is None:
        raise HTTPException(status_code=400, detail=f"No price data available for {normalized}")

    day_change = round(ltp - previous_close, 2) if previous_close else None
    day_change_pct = round((day_change / previous_close) * 100, 2) if previous_close and day_change is not None else None

    return {
        "ticker": normalized,
        "ltp": ltp,
        "previous_close": previous_close,
        "day_change": day_change,
        "day_change_pct": day_change_pct,
    }


# Chart interval → (yfinance interval, history window, cache TTL seconds).
# Windows are the deepest yfinance allows for each interval, capped at 1 year:
#   1d/1h can reach a year; 5m caps at 60d, 1m at 7d (yfinance hard limits).
# Intraday bars carry a UNIX-timestamp `time`; daily bars carry a date string.
CHART_INTERVALS = {
    "1d": {"yf_interval": "1d",  "period": "1y",  "ttl": 6 * 3600, "intraday": False},
    "1h": {"yf_interval": "60m", "period": "1y",  "ttl": 1 * 3600, "intraday": True},
    "5m": {"yf_interval": "5m",  "period": "60d", "ttl": 5 * 60,   "intraday": True},
    "1m": {"yf_interval": "1m",  "period": "7d",  "ttl": 2 * 60,   "intraday": True},
}


@app.get("/api/stocks/chart/{ticker}")
def get_chart_data(ticker: str, interval: str = "1d"):
    """
    Returns historical OHLCV (+RSI 7/14) for lightweight-charts.

    Query param `interval` ∈ {1d, 1h, 5m, 1m}. Each interval returns the
    deepest history yfinance supports for it, capped at one year. Intraday
    candles emit `time` as a UNIX timestamp (seconds); daily candles emit a
    `YYYY-MM-DD` string — matching what lightweight-charts expects per mode.
    """
    cfg = CHART_INTERVALS.get(interval)
    if cfg is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid interval '{interval}'. Choose one of: {', '.join(CHART_INTERVALS)}.",
        )

    normalized = normalize_ticker(ticker)
    from cache_manager import CacheManager
    import pandas as pd

    def _fetch():
        import math
        from indicators import _compute_rsi
        df = yf.Ticker(normalized, session=_session).history(
            period=cfg["period"], interval=cfg["yf_interval"]
        )
        if df.empty:
            return []

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Drop rows with any missing OHLC values — these break JSON serialization
        # (yfinance occasionally returns NaN for suspended/holiday/corp-action days).
        df = df.dropna(subset=["Open", "High", "Low", "Close"])

        # Compute RSI 7 and RSI 14 on Close prices.
        # First (period) rows will be NaN — emitted as null so the frontend skips them.
        df["RSI_7"] = _compute_rsi(df["Close"], period=7)
        df["RSI_14"] = _compute_rsi(df["Close"], period=14)

        # MACD (standard 12/26/9) on Close — mirrors indicators.compute_all_indicators.
        # ewm(adjust=False) yields values from the first row (no NaN warm-up), so the
        # early bars are just unstabilized rather than null.
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        def _safe(v):
            v = float(v)
            return round(v, 2) if math.isfinite(v) else None

        # MACD values can be tiny for low-priced small caps — keep more precision.
        def _safe3(v):
            v = float(v)
            return round(v, 3) if math.isfinite(v) else None

        intraday = cfg["intraday"]
        records = []
        for index, row in df.iterrows():
            o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
            # Final safety check — skip any row that still contains NaN or Inf in OHLC
            if not all(math.isfinite(v) for v in (o, h, l, c)):
                continue
            # Intraday → epoch seconds (UTC instant); daily → calendar date string.
            t = int(index.timestamp()) if intraday else index.strftime("%Y-%m-%d")
            records.append({
                "time": t,
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2),
                "rsi_7": _safe(row["RSI_7"]),
                "rsi_14": _safe(row["RSI_14"]),
                "macd": _safe3(row["MACD"]),
                "macd_signal": _safe3(row["MACD_Signal"]),
                "macd_hist": _safe3(row["MACD_Hist"]),
            })
        return records

    result = CacheManager.get_or_fetch(
        key=f"{normalized}_chart_{interval}",
        fetch_fn=_fetch,
        ttl=cfg["ttl"],
        category="data"
    )
    if not result["data"]:
        raise HTTPException(status_code=404, detail="No chart data found.")
    return result["data"]


@app.get("/api/stocks/evaluate/{ticker}")
def evaluate_stock(ticker: str):
    """
    Evaluates a specific stock ticker using the upgraded ML ensemble.
    Returns action signal, trade strategy, confidence, feature importance,
    model accuracy, and full technicals snapshot.
    """
    normalized = normalize_ticker(ticker)
    result = ml_engine.evaluate(normalized)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Not enough data found for ticker '{normalized}'. "
                   "Ensure it has at least 1 year of trading history.",
        )
    return result

@app.get("/api/debug/{ticker}")
def debug_yfinance(ticker: str):
    import yfinance as yf
    try:
        tkr = yf.Ticker(ticker)
        df = tkr.history(period="1mo")
        return {
            "version": yf.__version__,
            "df_empty": df.empty,
            "df_len": len(df),
            "fast_info": type(tkr.fast_info).__name__
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/stocks/nifty50")
def nifty50_comparison():
    """
    Batch-evaluates all Nifty 50 stocks and returns a momentum-ranked leaderboard.
    Cached for 4 hours.
    """
    try:
        return compare_nifty50()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/stocks/options/{symbol}")
def options_advisor(symbol: str, equity_signal: str = "Hold"):
    """
    Options advisor for an NSE symbol. Fetches option chain data,
    analyses OI/volume/PCR, and recommends Call vs Put strategies.

    Query params:
        equity_signal — optional ML equity signal to weight the recommendation
                        (e.g. "Strong Buy", "Sell"). Defaults to "Hold".
    """
    result = get_options_recommendation(symbol.strip().upper(), equity_signal)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ======================================================================
# Small Cap Stocks
# ======================================================================

@app.get("/api/stocks/smallcap")
def smallcap_comparison():
    """Batch-evaluates small cap stocks and returns a momentum-ranked leaderboard."""
    try:
        return compare_smallcap()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ======================================================================
# Mid Cap Stocks
# ======================================================================

@app.get("/api/stocks/midcap")
def midcap_comparison():
    """Batch-evaluates mid cap stocks and returns a momentum-ranked leaderboard."""
    try:
        return compare_midcap()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ======================================================================
# Shared helper — merges Nifty 50 + Mid Cap + Small Cap momentum data (cached)
# ======================================================================

def _get_all_momentum_stocks() -> list[dict]:
    """
    Merges already-cached Nifty 50, Mid Cap, and Small Cap leaderboards.
    Each sub-call has its own 4-hour cache, so this is essentially free.
    """
    combined = []
    try:
        n50 = compare_nifty50()
        combined.extend({**s, "universe": "Nifty 50"} for s in n50.get("leaderboard", []))
    except Exception as e:
        logger.warning("Momentum — nifty50 fetch failed: %s", e)
    try:
        mc = compare_midcap()
        combined.extend({**s, "universe": "Mid Cap"} for s in mc.get("leaderboard", []))
    except Exception as e:
        logger.warning("Momentum — midcap fetch failed: %s", e)
    try:
        sc = compare_smallcap()
        combined.extend({**s, "universe": "Small Cap"} for s in sc.get("leaderboard", []))
    except Exception as e:
        logger.warning("Momentum — smallcap fetch failed: %s", e)

    combined.sort(key=lambda x: x.get("momentum_score", 0), reverse=True)
    for i, r in enumerate(combined, 1):
        r["rank"] = i
    return combined


# ======================================================================
# Momentum Stocks — top movers from Nifty 50 + Small Caps combined
# ======================================================================

@app.get("/api/stocks/momentum")
def momentum_stocks(limit: int = 80):
    """
    Returns the top momentum stocks across Nifty 50 and Small Cap universes.
    Cached for 4 hours, synced with the underlying nifty50/smallcap caches
    so the screener always reflects the freshest leaderboard data.
    """
    from cache_manager import CacheManager, momentum_ttl

    result = CacheManager.get_or_fetch(
        key="momentum_combined",
        fetch_fn=_get_all_momentum_stocks,
        ttl=momentum_ttl(),
        category="data",
    )
    data = result["data"][:limit]
    return {
        "stocks": data,
        "count": len(data),
        "stale": result.get("stale", False),
    }


# ======================================================================
# Strong Buy Stocks — lightweight, derived from cached momentum data
# ======================================================================

def _is_strong_buy(stock: dict) -> tuple[bool, float]:
    """
    Heuristic Strong Buy filter using already-computed momentum data.
    No ML evaluation — reuses cached momentum/technical scores.

    Criteria (all must be true):
      - momentum_score >= 65
      - RSI between 50-80 (strong uptrend, not yet overbought)
      - MACD histogram > 0 (bullish momentum)
      - Price above SMA 50 and SMA 200 (uptrend)
      - ADX >= 20 (trend has strength)
    """
    ms = stock.get("momentum_score", 0)
    rsi = stock.get("rsi", 50)
    macd_h = stock.get("macd_histogram", 0)
    p50 = stock.get("price_above_sma50", 1)
    p200 = stock.get("price_above_sma200", 1)
    adx = stock.get("adx", 0)

    if ms < 65 or rsi < 50 or rsi > 80 or macd_h <= 0 or p50 < 1.0 or p200 < 1.0 or adx < 20:
        return False, 0

    confidence = round(
        0.30 * min(ms / 100, 1)
        + 0.20 * min((rsi - 50) / 30, 1)
        + 0.15 * min(macd_h / 10, 1)
        + 0.15 * min((p50 - 1.0) / 0.15, 1)
        + 0.10 * min((p200 - 1.0) / 0.20, 1)
        + 0.10 * min(adx / 50, 1),
        4,
    ) * 100
    return True, round(confidence, 2)


@app.get("/api/stocks/strong-buys")
def strong_buy_stocks():
    """
    Returns stocks with strong bullish signals, derived from the
    already-cached momentum data. Zero additional API calls or ML training.
    Cached for 4 hours, synced with the underlying momentum data.
    """
    from cache_manager import CacheManager, momentum_ttl

    def _find_strong_buys():
        all_stocks = _get_all_momentum_stocks()
        strong_buys = []
        for stock in all_stocks:
            is_sb, confidence = _is_strong_buy(stock)
            if is_sb:
                strong_buys.append({**stock, "prediction": "Strong Buy", "confidence": confidence})

        strong_buys.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        for i, r in enumerate(strong_buys, 1):
            r["rank"] = i
        return strong_buys

    result = CacheManager.get_or_fetch(
        key="strong_buys_combined",
        fetch_fn=_find_strong_buys,
        ttl=momentum_ttl(),
        category="data",
    )
    return {
        "stocks": result["data"],
        "count": len(result["data"]),
        "stale": result.get("stale", False),
    }


# ======================================================================
# Stock News
# ======================================================================

@app.get("/api/stocks/news/{ticker}")
def get_stock_news(ticker: str):
    """
    Returns recent news articles for a ticker using yfinance.
    """
    normalized = normalize_ticker(ticker)
    try:
        tkr = yf.Ticker(normalized, session=_session)
        news_items = tkr.news or []
        articles = []
        for item in news_items[:10]:
            content = item.get("content", {})
            thumbnail_url = ""
            thumb = content.get("thumbnail")
            if thumb and isinstance(thumb, dict):
                resolutions = thumb.get("resolutions", [])
                if resolutions:
                    thumbnail_url = resolutions[-1].get("url", "")

            articles.append({
                "title": content.get("title", item.get("title", "")),
                "publisher": content.get("provider", {}).get("displayName", ""),
                "link": content.get("canonicalUrl", {}).get("url", item.get("link", "")),
                "published": content.get("pubDate", ""),
                "thumbnail": thumbnail_url,
            })
        return {"ticker": normalized, "articles": articles}
    except Exception as e:
        logger.warning("News fetch failed for %s: %s", normalized, e)
        return {"ticker": normalized, "articles": []}

