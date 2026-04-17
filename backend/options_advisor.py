from __future__ import annotations

"""
Options Advisor Module — Fetches NSE option chain data via nselib,
analyzes OI/volume/PCR, and combines with the equity ML signal to
recommend Call vs Put strategies.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from cache_manager import CacheManager, TTL_OPTION_CHAIN, TTL_FNO_HIST

logger = logging.getLogger(__name__)

# Try to import nselib — it may not be installed yet
try:
    from nselib import derivatives as nse_deriv
    NSELIB_AVAILABLE = True
except ImportError:
    NSELIB_AVAILABLE = False
    logger.warning("nselib not installed — options advisor will be unavailable")


# ======================================================================
# Public API
# ======================================================================

def get_options_recommendation(
    symbol: str,
    equity_signal: str = "Hold",
) -> dict[str, Any]:
    """
    Full options analysis for an NSE symbol.

    Parameters
    ----------
    symbol : str
        NSE symbol like "RELIANCE", "NIFTY", "BANKNIFTY"
    equity_signal : str
        The ML equity signal (Strong Buy / Buy / Hold / Sell / Strong Sell)
        used to weight the final Call/Put recommendation.

    Returns
    -------
    dict with recommendation, reasoning, and all options metrics.
    """
    if not NSELIB_AVAILABLE:
        return {
            "error": "nselib is not installed. Run: pip install nselib",
            "symbol": symbol,
        }

    # 1. Fetch live option chain (cached 2 hours)
    chain_result = CacheManager.get_or_fetch(
        key=f"{symbol}_optchain",
        fetch_fn=lambda: _fetch_live_option_chain(symbol),
        ttl=TTL_OPTION_CHAIN,
        category="data",
    )
    chain_df: pd.DataFrame | None = chain_result["data"]

    if chain_df is None or (isinstance(chain_df, pd.DataFrame) and chain_df.empty):
        return {"error": f"No option chain data available for {symbol}", "symbol": symbol}

    # 2. Fetch historical F&O data (cached 24 hours)
    hist_result = CacheManager.get_or_fetch(
        key=f"{symbol}_fno_hist",
        fetch_fn=lambda: _fetch_fno_history(symbol, days=30),
        ttl=TTL_FNO_HIST,
        category="data",
    )
    hist_df: pd.DataFrame | None = hist_result["data"]

    # 3. Compute options indicators
    options_data = _compute_options_indicators(chain_df, hist_df)

    # 4. Generate recommendation
    recommendation, reasoning = _generate_recommendation(
        equity_signal, options_data
    )

    # 5. Determine suggested strike
    suggested_strike, expiry = _suggest_strike(chain_df, options_data, recommendation)

    return {
        "symbol": symbol,
        "recommendation": recommendation,
        "suggested_strike": suggested_strike,
        "expiry": expiry,
        "reasoning": reasoning,
        "equity_signal": equity_signal,
        "options_data": options_data,
        "data_stale": chain_result.get("stale", False) or hist_result.get("stale", False),
    }


# ======================================================================
# Data fetching
# ======================================================================

def _fetch_live_option_chain(symbol: str) -> pd.DataFrame | None:
    """Fetch the live option chain from NSE via nselib."""
    try:
        data = nse_deriv.nse_live_option_chain(symbol)
        if isinstance(data, pd.DataFrame) and not data.empty:
            return data
        return None
    except Exception as exc:
        logger.error("Failed to fetch option chain for %s: %s", symbol, exc)
        return None


def _fetch_fno_history(symbol: str, days: int = 30) -> pd.DataFrame | None:
    """Fetch F&O Bhav Copies for the last `days` trading days."""
    frames = []
    today = datetime.now()

    for i in range(days):
        date = today - timedelta(days=i)
        date_str = date.strftime("%d-%m-%Y")
        try:
            bhav = nse_deriv.fno_bhav_copy(date_str)
            if isinstance(bhav, pd.DataFrame) and not bhav.empty:
                # Filter for this specific symbol
                sym_data = bhav[
                    bhav["TckrSymb"].str.upper() == symbol.upper()
                ] if "TckrSymb" in bhav.columns else bhav[
                    bhav["SYMBOL"].str.upper() == symbol.upper()
                ] if "SYMBOL" in bhav.columns else pd.DataFrame()

                if not sym_data.empty:
                    sym_data = sym_data.copy()
                    sym_data["Date"] = date
                    frames.append(sym_data)
        except Exception:
            continue  # Weekends/holidays will fail — that's fine

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# ======================================================================
# Indicator computation
# ======================================================================

def _compute_options_indicators(
    chain_df: pd.DataFrame,
    hist_df: pd.DataFrame | None,
) -> dict[str, Any]:
    """Compute PCR, Max Pain, OI buildup, support/resistance levels."""

    indicators: dict[str, Any] = {}

    # --- Normalize column names ---
    chain = chain_df.copy()
    chain.columns = [c.strip().upper() for c in chain.columns]

    # Identify call/put OI and volume columns
    call_oi_col = _find_col(chain, ["CALL_OI", "CE_OI", "CALLS_OI"])
    put_oi_col = _find_col(chain, ["PUT_OI", "PE_OI", "PUTS_OI"])
    call_vol_col = _find_col(chain, ["CALL_VOLUME", "CE_VOLUME", "CALLS_VOLUME", "CE_VOL"])
    put_vol_col = _find_col(chain, ["PUT_VOLUME", "PE_VOLUME", "PUTS_VOLUME", "PE_VOL"])
    strike_col = _find_col(chain, ["STRIKE_PRICE", "STRIKE", "STRIKEPRICE"])

    # --- PCR by OI ---
    total_call_oi = _safe_sum(chain, call_oi_col)
    total_put_oi = _safe_sum(chain, put_oi_col)
    pcr_oi = round(total_put_oi / (total_call_oi + 1), 4) if total_call_oi else None
    indicators["pcr_oi"] = pcr_oi

    # --- PCR by Volume ---
    total_call_vol = _safe_sum(chain, call_vol_col)
    total_put_vol = _safe_sum(chain, put_vol_col)
    pcr_volume = round(total_put_vol / (total_call_vol + 1), 4) if total_call_vol else None
    indicators["pcr_volume"] = pcr_volume

    # --- Max Pain ---
    if strike_col and call_oi_col and put_oi_col:
        indicators["max_pain"] = _calculate_max_pain(chain, strike_col, call_oi_col, put_oi_col)
    else:
        indicators["max_pain"] = None

    # --- Support (highest Put OI) & Resistance (highest Call OI) ---
    if strike_col and put_oi_col:
        max_put_idx = chain[put_oi_col].idxmax() if put_oi_col in chain.columns else None
        indicators["support_from_put_oi"] = (
            float(chain.loc[max_put_idx, strike_col]) if max_put_idx is not None else None
        )
    else:
        indicators["support_from_put_oi"] = None

    if strike_col and call_oi_col:
        max_call_idx = chain[call_oi_col].idxmax() if call_oi_col in chain.columns else None
        indicators["resistance_from_call_oi"] = (
            float(chain.loc[max_call_idx, strike_col]) if max_call_idx is not None else None
        )
    else:
        indicators["resistance_from_call_oi"] = None

    # --- OI Buildup (simplified) ---
    indicators["oi_buildup"] = _classify_oi_buildup(pcr_oi, pcr_volume)

    # --- PCR Trend (30-day from historical data) ---
    indicators["pcr_trend_30d"] = _compute_pcr_trend(hist_df)

    return indicators


def _calculate_max_pain(
    chain: pd.DataFrame,
    strike_col: str,
    call_oi_col: str,
    put_oi_col: str,
) -> float | None:
    """
    Max Pain = strike price at which the total value of all outstanding
    options causes maximum loss for option holders (minimum loss for writers).
    """
    try:
        strikes = chain[strike_col].dropna().unique()
        min_pain = float("inf")
        max_pain_strike = None

        for strike in strikes:
            call_pain = chain.apply(
                lambda row: max(0, float(row[strike_col]) - strike) * float(row[call_oi_col])
                if pd.notna(row.get(call_oi_col)) else 0,
                axis=1,
            ).sum()
            put_pain = chain.apply(
                lambda row: max(0, strike - float(row[strike_col])) * float(row[put_oi_col])
                if pd.notna(row.get(put_oi_col)) else 0,
                axis=1,
            ).sum()
            total_pain = call_pain + put_pain
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = strike

        return float(max_pain_strike) if max_pain_strike is not None else None
    except Exception as exc:
        logger.warning("Max pain calculation failed: %s", exc)
        return None


def _classify_oi_buildup(pcr_oi: float | None, pcr_volume: float | None) -> str:
    """
    Simplified OI buildup classification based on PCR levels.
    """
    if pcr_oi is None:
        return "Unknown"

    if pcr_oi > 1.2:
        if pcr_volume and pcr_volume > 1.0:
            return "Long Buildup"       # Puts dominate, contrarian bullish
        return "Short Covering"
    elif pcr_oi < 0.7:
        if pcr_volume and pcr_volume < 0.8:
            return "Short Buildup"      # Calls dominate, contrarian bearish
        return "Long Unwinding"
    return "Neutral"


def _compute_pcr_trend(hist_df: pd.DataFrame | None) -> str:
    """Determine if PCR has been Rising, Falling, or Flat over the last 30 days."""
    if hist_df is None or hist_df.empty:
        return "Unknown"

    try:
        cols = [c.strip().upper() for c in hist_df.columns]
        hist = hist_df.copy()
        hist.columns = cols

        # Try to compute daily PCR
        oi_col = _find_col(hist, ["OI", "OPEN_INT", "OPENINTEREST", "OPNINTRSTQ"])
        type_col = _find_col(hist, ["OPTIONTYPE", "OPTION_TYPE", "OPTTP"])

        if not oi_col or not type_col:
            return "Unknown"

        hist[oi_col] = pd.to_numeric(hist[oi_col], errors="coerce")

        daily_pcr = []
        for date, group in hist.groupby("Date"):
            calls = group[group[type_col].str.upper().str.contains("CE|CALL", na=False)]
            puts = group[group[type_col].str.upper().str.contains("PE|PUT", na=False)]
            call_oi = calls[oi_col].sum()
            put_oi = puts[oi_col].sum()
            if call_oi > 0:
                daily_pcr.append(put_oi / call_oi)

        if len(daily_pcr) < 5:
            return "Unknown"

        # Simple trend: compare first half average to second half average
        mid = len(daily_pcr) // 2
        first_half = np.mean(daily_pcr[:mid])
        second_half = np.mean(daily_pcr[mid:])

        diff = second_half - first_half
        if diff > 0.05:
            return "Rising"
        elif diff < -0.05:
            return "Falling"
        return "Flat"

    except Exception as exc:
        logger.warning("PCR trend computation failed: %s", exc)
        return "Unknown"


# ======================================================================
# Recommendation logic
# ======================================================================

def _generate_recommendation(
    equity_signal: str,
    options_data: dict[str, Any],
) -> tuple[str, str]:
    """
    Combine equity ML signal with options indicators to generate
    a Call/Put recommendation with reasoning.
    """
    pcr_oi = options_data.get("pcr_oi")
    oi_buildup = options_data.get("oi_buildup", "Unknown")
    max_pain = options_data.get("max_pain")
    support = options_data.get("support_from_put_oi")
    resistance = options_data.get("resistance_from_call_oi")
    pcr_trend = options_data.get("pcr_trend_30d", "Unknown")

    # Score system: positive = bullish (Call), negative = bearish (Put)
    score = 0
    reasons = []

    # --- Equity signal weight (heaviest) ---
    signal_scores = {
        "Strong Buy": 3, "Buy": 1.5, "Hold": 0, "Sell": -1.5, "Strong Sell": -3
    }
    score += signal_scores.get(equity_signal, 0)
    reasons.append(f"ML model signals {equity_signal}")

    # --- PCR contrarian interpretation ---
    if pcr_oi is not None:
        if pcr_oi > 1.2:
            score += 1  # Excess puts = contrarian bullish
            reasons.append(f"PCR at {pcr_oi} (contrarian bullish)")
        elif pcr_oi < 0.7:
            score -= 1  # Excess calls = contrarian bearish
            reasons.append(f"PCR at {pcr_oi} (contrarian bearish)")
        else:
            reasons.append(f"PCR at {pcr_oi} (neutral)")

    # --- OI Buildup ---
    if oi_buildup in ("Long Buildup", "Short Covering"):
        score += 0.5
        reasons.append(f"{oi_buildup} detected")
    elif oi_buildup in ("Short Buildup", "Long Unwinding"):
        score -= 0.5
        reasons.append(f"{oi_buildup} detected")

    # --- Support/Resistance context ---
    if support:
        reasons.append(f"Strong Put OI support at {support}")
    if resistance:
        reasons.append(f"Call OI resistance at {resistance}")
    if max_pain:
        reasons.append(f"Max Pain at {max_pain}")

    # --- Decision ---
    if score >= 2:
        rec = "BUY CALL"
    elif score >= 0.5:
        rec = "MILD CALL BIAS"
    elif score <= -2:
        rec = "BUY PUT"
    elif score <= -0.5:
        rec = "MILD PUT BIAS"
    else:
        rec = "NEUTRAL — AVOID"

    reasoning = ", ".join(reasons)
    return rec, reasoning


def _suggest_strike(
    chain_df: pd.DataFrame,
    options_data: dict[str, Any],
    recommendation: str,
) -> tuple[float | None, str | None]:
    """Suggest a strike price based on OI support/resistance and Max Pain."""
    max_pain = options_data.get("max_pain")
    support = options_data.get("support_from_put_oi")
    resistance = options_data.get("resistance_from_call_oi")

    strike = None
    if "CALL" in recommendation and resistance:
        # Suggest ATM or slightly ITM for calls
        strike = resistance
    elif "PUT" in recommendation and support:
        strike = support
    elif max_pain:
        strike = max_pain

    # Try to extract nearest expiry
    expiry = None
    chain = chain_df.copy()
    chain.columns = [c.strip().upper() for c in chain.columns]
    exp_col = _find_col(chain, ["EXPIRY_DATE", "EXPIRY", "EXPIRYDATE", "XPRY_DATE"])
    if exp_col and exp_col in chain.columns:
        try:
            dates = pd.to_datetime(chain[exp_col], errors="coerce").dropna()
            if not dates.empty:
                nearest = dates.min()
                expiry = nearest.strftime("%Y-%m-%d")
        except Exception:
            pass

    return strike, expiry


# ======================================================================
# Utility helpers
# ======================================================================

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name from a list of candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_sum(df: pd.DataFrame, col: str | None) -> float:
    """Sum a numeric column safely, returning 0 if column doesn't exist."""
    if col and col in df.columns:
        return float(pd.to_numeric(df[col], errors="coerce").sum())
    return 0.0
