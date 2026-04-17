from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

import yfinance as yf
from ml_engine import StockMLEngine
from nifty50 import compare_nifty50
from options_advisor import get_options_recommendation

app = FastAPI(title="Stock Analytics API")

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
    """
    normalized = normalize_ticker(ticker)
    try:
        tkr_info = yf.Ticker(normalized).fast_info
        ltp = round(float(tkr_info["lastPrice"]), 2)
        previous_close = round(float(tkr_info.get("previousClose", 0)), 2)
        day_change = round(ltp - previous_close, 2) if previous_close else None
        day_change_pct = round((day_change / previous_close) * 100, 2) if previous_close else None
        
        return {
            "ticker": normalized,
            "ltp": ltp,
            "previous_close": previous_close,
            "day_change": day_change,
            "day_change_pct": day_change_pct,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/stocks/chart/{ticker}")
def get_chart_data(ticker: str, period: str = "1y"):
    """
    Returns historical OHLCV data for lightweight-charts.
    """
    normalized = normalize_ticker(ticker)
    from cache_manager import CacheManager, TTL_PRICES
    import pandas as pd
    
    def _fetch():
        df = yf.Ticker(normalized).history(period=period)
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

