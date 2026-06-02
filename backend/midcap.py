from __future__ import annotations

"""
Mid Cap Stocks Module — momentum scoring for Nifty Midcap 50 constituents.
Mirrors the smallcap / nifty50 comparison pattern.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from indicators import compute_all_indicators
from cache_manager import CacheManager, momentum_ttl

logger = logging.getLogger(__name__)

MIDCAP_TICKERS: list[dict[str, str]] = [
    {"ticker": "ABCAPITAL.NS",  "name": "Aditya Birla Capital", "sector": "Financial Services"},
    {"ticker": "ABFRL.NS",      "name": "Aditya Birla Fashion","sector": "Consumer Goods"},
    {"ticker": "ACC.NS",        "name": "ACC",                 "sector": "Cement"},
    {"ticker": "ALKEM.NS",      "name": "Alkem Laboratories",  "sector": "Healthcare"},
    {"ticker": "ASHOKLEY.NS",   "name": "Ashok Leyland",       "sector": "Automobile"},
    {"ticker": "ASTRAL.NS",     "name": "Astral Ltd",          "sector": "Industrials"},
    {"ticker": "AUBANK.NS",     "name": "AU Small Finance Bank","sector": "Financial Services"},
    {"ticker": "AUROPHARMA.NS", "name": "Aurobindo Pharma",    "sector": "Healthcare"},
    {"ticker": "BALKRISIND.NS", "name": "Balkrishna Industries","sector": "Automobile"},
    {"ticker": "BANDHANBNK.NS", "name": "Bandhan Bank",        "sector": "Financial Services"},
    {"ticker": "BHARATFORG.NS", "name": "Bharat Forge",        "sector": "Industrials"},
    {"ticker": "BIOCON.NS",     "name": "Biocon",              "sector": "Healthcare"},
    {"ticker": "BOSCHLTD.NS",   "name": "Bosch",               "sector": "Automobile"},
    {"ticker": "CGPOWER.NS",    "name": "CG Power",            "sector": "Industrials"},
    {"ticker": "COLPAL.NS",     "name": "Colgate-Palmolive",   "sector": "FMCG"},
    {"ticker": "CONCOR.NS",     "name": "Container Corp",      "sector": "Industrials"},
    {"ticker": "CUMMINSIND.NS", "name": "Cummins India",       "sector": "Industrials"},
    {"ticker": "DIXON.NS",      "name": "Dixon Technologies",  "sector": "Consumer Goods"},
    {"ticker": "FEDERALBNK.NS", "name": "Federal Bank",        "sector": "Financial Services"},
    {"ticker": "GMRINFRA.NS",   "name": "GMR Airports",        "sector": "Infrastructure"},
    {"ticker": "GODREJPROP.NS", "name": "Godrej Properties",   "sector": "Real Estate"},
    {"ticker": "HINDPETRO.NS",  "name": "HPCL",                "sector": "Oil & Gas"},
    {"ticker": "IDFCFIRSTB.NS", "name": "IDFC First Bank",     "sector": "Financial Services"},
    {"ticker": "INDHOTEL.NS",   "name": "Indian Hotels",       "sector": "Travel"},
    {"ticker": "INDUSTOWER.NS", "name": "Indus Towers",        "sector": "Telecom"},
    {"ticker": "JUBLFOOD.NS",   "name": "Jubilant Foodworks",  "sector": "FMCG"},
    {"ticker": "LICHSGFIN.NS",  "name": "LIC Housing Finance", "sector": "Financial Services"},
    {"ticker": "LUPIN.NS",      "name": "Lupin",               "sector": "Healthcare"},
    {"ticker": "MARICO.NS",     "name": "Marico",              "sector": "FMCG"},
    {"ticker": "MAXHEALTH.NS",  "name": "Max Healthcare",      "sector": "Healthcare"},
    {"ticker": "MFSL.NS",       "name": "Max Financial",       "sector": "Financial Services"},
    {"ticker": "MPHASIS.NS",    "name": "Mphasis",             "sector": "IT"},
    {"ticker": "MRF.NS",        "name": "MRF",                 "sector": "Automobile"},
    {"ticker": "OBEROIRLTY.NS", "name": "Oberoi Realty",       "sector": "Real Estate"},
    {"ticker": "OFSS.NS",       "name": "Oracle Financial",    "sector": "IT"},
    {"ticker": "PAGEIND.NS",    "name": "Page Industries",     "sector": "Consumer Goods"},
    {"ticker": "PERSISTENT.NS", "name": "Persistent Systems",  "sector": "IT"},
    {"ticker": "PIIND.NS",      "name": "PI Industries",       "sector": "Chemicals"},
    {"ticker": "POLYCAB.NS",    "name": "Polycab India",       "sector": "Industrials"},
    {"ticker": "SAIL.NS",       "name": "SAIL",                "sector": "Metals & Mining"},
    {"ticker": "SRF.NS",        "name": "SRF",                 "sector": "Chemicals"},
    {"ticker": "SUNDARMFIN.NS", "name": "Sundaram Finance",    "sector": "Financial Services"},
    {"ticker": "SUPREMEIND.NS", "name": "Supreme Industries",  "sector": "Industrials"},
    {"ticker": "TATACOMM.NS",   "name": "Tata Communications", "sector": "Telecom"},
    {"ticker": "TIINDIA.NS",    "name": "Tube Investments",    "sector": "Automobile"},
    {"ticker": "TVSMOTOR.NS",   "name": "TVS Motor",           "sector": "Automobile"},
    {"ticker": "UPL.NS",        "name": "UPL",                 "sector": "Chemicals"},
    {"ticker": "VOLTAS.NS",     "name": "Voltas",              "sector": "Consumer Goods"},
    {"ticker": "YESBANK.NS",    "name": "Yes Bank",            "sector": "Financial Services"},
    {"ticker": "ZYDUSLIFE.NS",  "name": "Zydus Lifesciences",  "sector": "Healthcare"},
]


def compare_midcap() -> dict[str, Any]:
    result = CacheManager.get_or_fetch(
        key="midcap_comparison",
        fetch_fn=_build_comparison,
        ttl=momentum_ttl(),
        category="data",
    )
    return {
        "leaderboard": result["data"],
        "count": len(result["data"]),
        "stale": result.get("stale", False),
    }


def _build_comparison() -> list[dict]:
    results: list[dict] = []

    all_tickers = [entry["ticker"] for entry in MIDCAP_TICKERS]
    ticker_to_entry = {entry["ticker"]: entry for entry in MIDCAP_TICKERS}

    try:
        bulk_df = yf.download(
            all_tickers,
            period="2y",
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception as exc:
        logger.error("Mid-cap bulk download failed: %s", exc)
        return results

    for ticker in all_tickers:
        try:
            if len(all_tickers) == 1:
                df_raw = bulk_df.copy()
            else:
                df_raw = bulk_df[ticker].copy()

            df_raw.dropna(how="all", inplace=True)

            if df_raw.empty or len(df_raw) < 250:
                logger.warning("Midcap — insufficient data for %s (%d rows)", ticker, len(df_raw))
                continue

            if isinstance(df_raw.columns, pd.MultiIndex):
                df_raw.columns = df_raw.columns.get_level_values(0)

            score_data = _score_ticker(ticker, df_raw)
            if score_data is None:
                continue

            entry = ticker_to_entry[ticker]
            results.append({
                "ticker": ticker,
                "name": entry["name"],
                "sector": entry["sector"],
                **score_data,
            })
        except Exception as exc:
            logger.warning("Midcap — skipped %s: %s", ticker, exc)

    results.sort(key=lambda x: x["momentum_score"], reverse=True)

    for i, r in enumerate(results, 1):
        r["rank"] = i

    return results


def _score_ticker(ticker: str, df_raw: pd.DataFrame) -> dict | None:
    df = compute_all_indicators(df_raw)
    df.dropna(inplace=True)

    if df.empty:
        return None

    latest = df.iloc[-1]

    rsi_score = float(latest["RSI_14"])
    macd_hist = float(latest["MACD_Histogram"])
    macd_score = _sigmoid(macd_hist) * 100

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
    return 1 / (1 + np.exp(-x))
