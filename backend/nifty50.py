from __future__ import annotations

"""
Nifty 50 Comparison Module — Batch evaluation and momentum ranking
of all 50 Nifty constituent stocks.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from indicators import compute_all_indicators
from cache_manager import CacheManager, TTL_NIFTY50, TTL_PRICES

logger = logging.getLogger(__name__)

# ======================================================================
# Nifty 50 constituents with sector mappings
# ======================================================================

NIFTY50_TICKERS: list[dict[str, str]] = [
    {"ticker": "ADANIENT.NS",   "name": "Adani Enterprises",   "sector": "Industrials"},
    {"ticker": "ADANIPORTS.NS", "name": "Adani Ports",         "sector": "Industrials"},
    {"ticker": "APOLLOHOSP.NS", "name": "Apollo Hospitals",    "sector": "Healthcare"},
    {"ticker": "ASIANPAINT.NS", "name": "Asian Paints",        "sector": "Consumer Goods"},
    {"ticker": "AXISBANK.NS",   "name": "Axis Bank",           "sector": "Financial Services"},
    {"ticker": "BAJAJ-AUTO.NS", "name": "Bajaj Auto",          "sector": "Automobile"},
    {"ticker": "BAJFINANCE.NS", "name": "Bajaj Finance",       "sector": "Financial Services"},
    {"ticker": "BAJAJFINSV.NS", "name": "Bajaj Finserv",       "sector": "Financial Services"},
    {"ticker": "BPCL.NS",       "name": "BPCL",                "sector": "Oil & Gas"},
    {"ticker": "BHARTIARTL.NS", "name": "Bharti Airtel",       "sector": "Telecom"},
    {"ticker": "BRITANNIA.NS",  "name": "Britannia",           "sector": "FMCG"},
    {"ticker": "CIPLA.NS",      "name": "Cipla",               "sector": "Healthcare"},
    {"ticker": "COALINDIA.NS",  "name": "Coal India",          "sector": "Metals & Mining"},
    {"ticker": "DIVISLAB.NS",   "name": "Divi's Lab",          "sector": "Healthcare"},
    {"ticker": "DRREDDY.NS",    "name": "Dr. Reddy's",         "sector": "Healthcare"},
    {"ticker": "EICHERMOT.NS",  "name": "Eicher Motors",       "sector": "Automobile"},
    {"ticker": "GRASIM.NS",     "name": "Grasim Industries",   "sector": "Industrials"},
    {"ticker": "HCLTECH.NS",    "name": "HCL Technologies",    "sector": "IT"},
    {"ticker": "HDFCBANK.NS",   "name": "HDFC Bank",           "sector": "Financial Services"},
    {"ticker": "HDFCLIFE.NS",   "name": "HDFC Life",           "sector": "Financial Services"},
    {"ticker": "HEROMOTOCO.NS", "name": "Hero MotoCorp",       "sector": "Automobile"},
    {"ticker": "HINDALCO.NS",   "name": "Hindalco",            "sector": "Metals & Mining"},
    {"ticker": "HINDUNILVR.NS", "name": "Hindustan Unilever",  "sector": "FMCG"},
    {"ticker": "ICICIBANK.NS",  "name": "ICICI Bank",          "sector": "Financial Services"},
    {"ticker": "ITC.NS",        "name": "ITC",                 "sector": "FMCG"},
    {"ticker": "INDUSINDBK.NS", "name": "IndusInd Bank",       "sector": "Financial Services"},
    {"ticker": "INFY.NS",       "name": "Infosys",             "sector": "IT"},
    {"ticker": "JSWSTEEL.NS",   "name": "JSW Steel",           "sector": "Metals & Mining"},
    {"ticker": "KOTAKBANK.NS",  "name": "Kotak Mahindra Bank", "sector": "Financial Services"},
    {"ticker": "LT.NS",         "name": "Larsen & Toubro",     "sector": "Industrials"},
    {"ticker": "M&M.NS",        "name": "Mahindra & Mahindra", "sector": "Automobile"},
    {"ticker": "MARUTI.NS",     "name": "Maruti Suzuki",       "sector": "Automobile"},
    {"ticker": "NESTLEIND.NS",  "name": "Nestle India",        "sector": "FMCG"},
    {"ticker": "NTPC.NS",       "name": "NTPC",                "sector": "Power"},
    {"ticker": "ONGC.NS",       "name": "ONGC",                "sector": "Oil & Gas"},
    {"ticker": "POWERGRID.NS",  "name": "Power Grid Corp",     "sector": "Power"},
    {"ticker": "RELIANCE.NS",   "name": "Reliance Industries", "sector": "Oil & Gas"},
    {"ticker": "SBILIFE.NS",    "name": "SBI Life Insurance",  "sector": "Financial Services"},
    {"ticker": "SBIN.NS",       "name": "State Bank of India", "sector": "Financial Services"},
    {"ticker": "SUNPHARMA.NS",  "name": "Sun Pharma",          "sector": "Healthcare"},
    {"ticker": "TCS.NS",        "name": "TCS",                 "sector": "IT"},
    {"ticker": "TATACONSUM.NS", "name": "Tata Consumer",       "sector": "FMCG"},
    {"ticker": "TATAMOTORS.NS", "name": "Tata Motors",         "sector": "Automobile"},
    {"ticker": "TATASTEEL.NS",  "name": "Tata Steel",          "sector": "Metals & Mining"},
    {"ticker": "TECHM.NS",      "name": "Tech Mahindra",       "sector": "IT"},
    {"ticker": "TITAN.NS",      "name": "Titan Company",       "sector": "Consumer Goods"},
    {"ticker": "ULTRACEMCO.NS", "name": "UltraTech Cement",    "sector": "Industrials"},
    {"ticker": "UPL.NS",        "name": "UPL",                 "sector": "Chemicals"},
    {"ticker": "WIPRO.NS",      "name": "Wipro",               "sector": "IT"},
    {"ticker": "SHRIRAMFIN.NS", "name": "Shriram Finance",     "sector": "Financial Services"},
]


# ======================================================================
# Public API
# ======================================================================

def compare_nifty50() -> dict[str, Any]:
    """
    Batch-evaluate all Nifty 50 stocks and return a momentum leaderboard.
    Results are cached for 4 hours (TTL_NIFTY50).
    """
    result = CacheManager.get_or_fetch(
        key="nifty50_comparison",
        fetch_fn=_build_comparison,
        ttl=TTL_NIFTY50,
        category="data",
    )
    return {
        "leaderboard": result["data"],
        "count": len(result["data"]),
        "stale": result.get("stale", False),
    }


# ======================================================================
# Internal
# ======================================================================

def _build_comparison() -> list[dict]:
    """Fetch data for every ticker, compute momentum scores, rank."""
    results: list[dict] = []

    for entry in NIFTY50_TICKERS:
        ticker = entry["ticker"]
        try:
            score_data = _score_ticker(ticker)
            if score_data is None:
                continue
            results.append({
                "ticker": ticker,
                "name": entry["name"],
                "sector": entry["sector"],
                **score_data,
            })
        except Exception as exc:
            logger.warning("Nifty50 — skipped %s: %s", ticker, exc)

    # Sort by momentum_score descending
    results.sort(key=lambda x: x["momentum_score"], reverse=True)

    # Add rank
    for i, r in enumerate(results, 1):
        r["rank"] = i

    return results


def _score_ticker(ticker: str) -> dict | None:
    """Compute a composite momentum score for a single ticker."""
    # Fetch with cache
    price_result = CacheManager.get_or_fetch(
        key=f"{ticker}_prices",
        fetch_fn=lambda t=ticker: yf.Ticker(t).history(period="2y"),
        ttl=TTL_PRICES,
        category="data",
    )
    df_raw: pd.DataFrame = price_result["data"]
    if df_raw is None or df_raw.empty or len(df_raw) < 250:
        return None

    # Flatten any MultiIndex columns
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)

    df = compute_all_indicators(df_raw)
    df.dropna(inplace=True)

    if df.empty:
        return None

    latest = df.iloc[-1]

    # --- Composite Momentum Score (0-100) ---
    # Components:
    #   RSI_14          (weight 0.25) — normalized to 0-100
    #   MACD_Histogram  (weight 0.25) — sign + magnitude
    #   Price/SMA_50    (weight 0.25) — above/below moving avg
    #   Price/SMA_200   (weight 0.25) — long-term trend

    rsi_score = float(latest["RSI_14"])  # already 0-100
    macd_hist = float(latest["MACD_Histogram"])
    macd_score = _sigmoid(macd_hist) * 100  # map to 0-100

    p50 = float(latest["Price_SMA50_Ratio"])
    p200 = float(latest["Price_SMA200_Ratio"])
    sma50_score = min(max((p50 - 0.85) / 0.30 * 100, 0), 100)
    sma200_score = min(max((p200 - 0.80) / 0.40 * 100, 0), 100)

    momentum_score = round(
        0.25 * rsi_score + 0.25 * macd_score + 0.25 * sma50_score + 0.25 * sma200_score,
        2,
    )

    return {
        "current_price": round(float(latest["Close"]), 2),
        "rsi": round(float(latest["RSI_14"]), 2),
        "macd_histogram": round(float(latest["MACD_Histogram"]), 4),
        "price_above_sma50": round(p50, 4),
        "price_above_sma200": round(p200, 4),
        "adx": round(float(latest["ADX"]), 2),
        "momentum_score": momentum_score,
    }


def _sigmoid(x: float) -> float:
    """Map any real number to (0, 1) — used to normalize MACD histogram."""
    return 1 / (1 + np.exp(-x))
