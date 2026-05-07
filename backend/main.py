from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import logging

from curl_cffi.requests import Session as CffiSession
import yfinance as yf
from ml_engine import StockMLEngine
from nifty50 import compare_nifty50, NIFTY50_TICKERS
from smallcap import compare_smallcap, SMALLCAP_TICKERS
from options_advisor import get_options_recommendation

logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Analytics API")

# Shared curl_cffi session — reuses TCP connections & bypasses bot detection
_session = CffiSession(impersonate="chrome")

@app.get("/api/health")
def health_check():
    """Endpoint for UptimeRobot to ping to prevent Render cold-starts."""
    return {"status": "awake", "message": "Render backend is active!"}

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


@app.get("/api/stocks/chart/{ticker}")
def get_chart_data(ticker: str, period: str = "1y"):
    """
    Returns historical OHLCV data for lightweight-charts.
    """
    normalized = normalize_ticker(ticker)
    from cache_manager import CacheManager, TTL_PRICES
    import pandas as pd
    
    def _fetch():
        df = yf.Ticker(normalized, session=_session).history(period=period)
        if df.empty: return []
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        records = []
        for index, row in df.iterrows():
            date_str = index.strftime("%Y-%m-%d")
            records.append({
                "time": date_str,
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
            })
        return records

    result = CacheManager.get_or_fetch(
        key=f"{normalized}_chart_{period}",
        fetch_fn=_fetch,
        ttl=TTL_PRICES,
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
# Momentum Stocks — top movers from Nifty 50 + Small Caps combined
# ======================================================================

@app.get("/api/stocks/momentum")
def momentum_stocks(limit: int = 20):
    """
    Returns the top momentum stocks across Nifty 50 and Small Cap universes,
    sorted by momentum_score descending.
    """
    from cache_manager import CacheManager, TTL_NIFTY50

    def _build_momentum():
        combined = []
        try:
            n50 = compare_nifty50()
            combined.extend({**s, "universe": "Nifty 50"} for s in n50.get("leaderboard", []))
        except Exception as e:
            logger.warning("Momentum — nifty50 fetch failed: %s", e)

        try:
            sc = compare_smallcap()
            combined.extend({**s, "universe": "Small Cap"} for s in sc.get("leaderboard", []))
        except Exception as e:
            logger.warning("Momentum — smallcap fetch failed: %s", e)

        combined.sort(key=lambda x: x.get("momentum_score", 0), reverse=True)
        for i, r in enumerate(combined, 1):
            r["rank"] = i
        return combined

    result = CacheManager.get_or_fetch(
        key="momentum_combined",
        fetch_fn=_build_momentum,
        ttl=TTL_NIFTY50,
        category="data",
    )
    data = result["data"][:limit]
    return {
        "stocks": data,
        "count": len(data),
        "stale": result.get("stale", False),
    }


# ======================================================================
# Strong Buy Stocks — ML-evaluated "Strong Buy" signals
# ======================================================================

@app.get("/api/stocks/strong-buys")
def strong_buy_stocks():
    """
    Batch-evaluates stocks from Nifty 50 + Small Cap and returns those
    with a 'Strong Buy' ML signal.
    """
    from cache_manager import CacheManager, TTL_NIFTY50

    def _find_strong_buys():
        all_tickers = (
            [(t["ticker"], t["name"], t["sector"], "Nifty 50") for t in NIFTY50_TICKERS]
            + [(t["ticker"], t["name"], t["sector"], "Small Cap") for t in SMALLCAP_TICKERS]
        )
        strong_buys = []

        for ticker, name, sector, universe in all_tickers:
            try:
                result = ml_engine.evaluate(ticker)
                if result and result.get("prediction") == "Strong Buy":
                    strong_buys.append({
                        "ticker": ticker,
                        "name": name,
                        "sector": sector,
                        "universe": universe,
                        "prediction": result["prediction"],
                        "confidence": result["confidence"],
                        "current_price": result["ltp"],
                        "day_change_pct": result.get("day_change_pct"),
                        "trade_strategy": result["trade_strategy"],
                        "rsi": result["technicals"]["RSI_14"],
                        "macd": result["technicals"]["MACD"],
                    })
            except Exception as e:
                logger.warning("Strong-buy scan — skipped %s: %s", ticker, e)

        strong_buys.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        for i, r in enumerate(strong_buys, 1):
            r["rank"] = i
        return strong_buys

    result = CacheManager.get_or_fetch(
        key="strong_buys_combined",
        fetch_fn=_find_strong_buys,
        ttl=TTL_NIFTY50,
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

