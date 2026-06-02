from __future__ import annotations

"""
Small Cap Stocks Module — momentum scoring for Nifty Smallcap 50 constituents.
Mirrors the nifty50 comparison pattern.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from indicators import compute_all_indicators
from cache_manager import CacheManager, momentum_ttl

logger = logging.getLogger(__name__)

SMALLCAP_TICKERS: list[dict[str, str]] = [
    {"ticker": "360ONE.NS",     "name": "360 ONE WAM",         "sector": "Financial Services"},
    {"ticker": "AETHER.NS",     "name": "Aether Industries",   "sector": "Chemicals"},
    {"ticker": "ANGELONE.NS",   "name": "Angel One",           "sector": "Financial Services"},
    {"ticker": "APLAPOLLO.NS",  "name": "APL Apollo Tubes",    "sector": "Metals & Mining"},
    {"ticker": "ATUL.NS",       "name": "Atul Ltd",            "sector": "Chemicals"},
    {"ticker": "BIKAJI.NS",     "name": "Bikaji Foods",        "sector": "FMCG"},
    {"ticker": "BSE.NS",        "name": "BSE Ltd",             "sector": "Financial Services"},
    {"ticker": "CAMPUS.NS",     "name": "Campus Activewear",   "sector": "Consumer Goods"},
    {"ticker": "CDSL.NS",       "name": "CDSL",                "sector": "Financial Services"},
    {"ticker": "CELLO.NS",      "name": "Cello World",         "sector": "Consumer Goods"},
    {"ticker": "CAMS.NS",       "name": "CAMS",                "sector": "Financial Services"},
    {"ticker": "CYIENT.NS",     "name": "Cyient",              "sector": "IT"},
    {"ticker": "DATAPATTNS.NS", "name": "Data Patterns",       "sector": "Defence"},
    {"ticker": "DEVYANI.NS",    "name": "Devyani International","sector": "Consumer Goods"},
    {"ticker": "EASEMYTRIP.NS", "name": "EaseMyTrip",          "sector": "Travel"},
    {"ticker": "FINEORG.NS",    "name": "Fine Organic Ind",    "sector": "Chemicals"},
    {"ticker": "GRINDWELL.NS",  "name": "Grindwell Norton",    "sector": "Industrials"},
    {"ticker": "HAPPSTMNDS.NS", "name": "Happiest Minds",      "sector": "IT"},
    {"ticker": "HOMEFIRST.NS",  "name": "Home First Finance",  "sector": "Financial Services"},
    {"ticker": "IRFC.NS",       "name": "IRFC",                "sector": "Financial Services"},
    {"ticker": "KALYANKJIL.NS", "name": "Kalyan Jewellers",    "sector": "Consumer Goods"},
    {"ticker": "KPITTECH.NS",   "name": "KPIT Technologies",   "sector": "IT"},
    {"ticker": "LATENTVIEW.NS", "name": "Latent View Analytics","sector": "IT"},
    {"ticker": "LXCHEM.NS",     "name": "Laxmi Organic",       "sector": "Chemicals"},
    {"ticker": "MAPMYINDIA.NS", "name": "MapMyIndia",          "sector": "IT"},
    {"ticker": "MEDANTA.NS",    "name": "Medanta (GCHL)",      "sector": "Healthcare"},
    {"ticker": "NUVAMA.NS",     "name": "Nuvama Wealth",       "sector": "Financial Services"},
    {"ticker": "PPLPHARMA.NS",  "name": "Piramal Pharma",      "sector": "Healthcare"},
    {"ticker": "RADICO.NS",     "name": "Radico Khaitan",      "sector": "FMCG"},
    {"ticker": "RAILTEL.NS",    "name": "RailTel Corp",        "sector": "IT"},
    {"ticker": "RAINBOW.NS",    "name": "Rainbow Children's",  "sector": "Healthcare"},
    {"ticker": "RVNL.NS",       "name": "RVNL",                "sector": "Industrials"},
    {"ticker": "ROUTE.NS",      "name": "Route Mobile",        "sector": "IT"},
    {"ticker": "SAPPHIRE.NS",   "name": "Sapphire Foods",      "sector": "Consumer Goods"},
    {"ticker": "SYRMA.NS",      "name": "Syrma SGS Tech",      "sector": "IT"},
    {"ticker": "TARSONS.NS",    "name": "Tarsons Products",    "sector": "Healthcare"},
    {"ticker": "TIINDIA.NS",    "name": "Tube Investments",    "sector": "Automobile"},
    {"ticker": "UTIAMC.NS",     "name": "UTI AMC",             "sector": "Financial Services"},
    {"ticker": "VIJAYA.NS",     "name": "Vijaya Diagnostic",   "sector": "Healthcare"},
    {"ticker": "ZEEL.NS",       "name": "Zee Entertainment",   "sector": "Media"},
]


def compare_smallcap() -> dict[str, Any]:
    result = CacheManager.get_or_fetch(
        key="smallcap_comparison",
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

    all_tickers = [entry["ticker"] for entry in SMALLCAP_TICKERS]
    ticker_to_entry = {entry["ticker"]: entry for entry in SMALLCAP_TICKERS}

    try:
        bulk_df = yf.download(
            all_tickers,
            period="2y",
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception as exc:
        logger.error("Small-cap bulk download failed: %s", exc)
        return results

    for ticker in all_tickers:
        try:
            if len(all_tickers) == 1:
                df_raw = bulk_df.copy()
            else:
                df_raw = bulk_df[ticker].copy()

            df_raw.dropna(how="all", inplace=True)

            if df_raw.empty or len(df_raw) < 250:
                logger.warning("Smallcap — insufficient data for %s (%d rows)", ticker, len(df_raw))
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
            logger.warning("Smallcap — skipped %s: %s", ticker, exc)

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
    import numpy as np
    return 1 / (1 + np.exp(-x))
