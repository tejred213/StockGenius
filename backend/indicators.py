from __future__ import annotations

"""
Feature Engineering Module — 20+ technical indicators across
Trend, Momentum, Volatility, Volume, and Derived categories.
"""

import pandas as pd
import numpy as np


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a yfinance-style OHLCV DataFrame and returns it with
    20+ engineered feature columns appended.  Rows with NaNs from
    warm-up periods are dropped at the end.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # 1 · TREND INDICATORS
    # ------------------------------------------------------------------
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["Signal_Line"]

    # ADX (Average Directional Index) — 14 period
    df["ADX"] = _compute_adx(df, period=14)

    # ------------------------------------------------------------------
    # 2 · MOMENTUM INDICATORS
    # ------------------------------------------------------------------
    df["RSI_14"] = _compute_rsi(df["Close"], period=14)

    # Stochastic %K and %D
    low14 = df["Low"].rolling(window=14).min()
    high14 = df["High"].rolling(window=14).max()
    df["Stoch_K"] = 100 * (df["Close"] - low14) / (high14 - low14 + 1e-10)
    df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()

    # Williams %R
    df["Williams_R"] = -100 * (high14 - df["Close"]) / (high14 - low14 + 1e-10)

    # Rate of Change (12 period)
    df["ROC"] = df["Close"].pct_change(periods=12) * 100

    # Commodity Channel Index (20 period)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    tp_sma = tp.rolling(window=20).mean()
    tp_mad = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["CCI"] = (tp - tp_sma) / (0.015 * tp_mad + 1e-10)

    # ------------------------------------------------------------------
    # 3 · VOLATILITY INDICATORS
    # ------------------------------------------------------------------
    df["BB_Upper"] = df["SMA_20"] + 2 * df["Close"].rolling(window=20).std()
    df["BB_Lower"] = df["SMA_20"] - 2 * df["Close"].rolling(window=20).std()
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / (df["SMA_20"] + 1e-10)

    df["ATR_14"] = _compute_atr(df, period=14)
    df["StdDev_20"] = df["Close"].rolling(window=20).std()

    # ------------------------------------------------------------------
    # 4 · VOLUME INDICATORS
    # ------------------------------------------------------------------
    df["OBV"] = _compute_obv(df)
    df["Volume_SMA_20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_ROC"] = df["Volume"].pct_change(periods=10) * 100

    # ------------------------------------------------------------------
    # 5 · DERIVED / COMPOSITE
    # ------------------------------------------------------------------
    df["Price_SMA50_Ratio"] = df["Close"] / (df["SMA_50"] + 1e-10)
    df["Price_SMA200_Ratio"] = df["Close"] / (df["SMA_200"] + 1e-10)
    df["Golden_Cross"] = (df["SMA_50"] > df["SMA_200"]).astype(int)

    return df


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature columns used for ML training."""
    return [
        # Trend
        "SMA_10", "SMA_20", "SMA_50", "SMA_200",
        "EMA_12", "EMA_26", "MACD", "Signal_Line", "MACD_Histogram", "ADX",
        # Momentum
        "RSI_14", "Stoch_K", "Stoch_D", "Williams_R", "ROC", "CCI",
        # Volatility
        "BB_Width", "ATR_14", "StdDev_20",
        # Volume
        "OBV", "Volume_SMA_20", "Volume_ROC",
        # Derived
        "Price_SMA50_Ratio", "Price_SMA200_Ratio", "Golden_Cross",
    ]


# ======================================================================
#  Private helper functions
# ======================================================================

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr = _compute_atr(df, period)
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    adx = dx.rolling(window=period).mean()
    return adx


def _compute_obv(df: pd.DataFrame) -> pd.Series:
    """Vectorized On-Balance Volume — avoids slow Python loop."""
    direction = np.sign(df["Close"].diff())
    obv = (direction * df["Volume"]).fillna(0).cumsum()
    return obv


# ======================================================================
#  Support & Resistance Levels (Pivot Points)
# ======================================================================

def calculate_support_resistance(row: pd.Series) -> dict:
    """
    Calculates pivot-based support and resistance levels from latest OHLC.
    Uses the standard pivot point methodology:
    - Pivot = (High + Low + Close) / 3
    - R1 = (2 * Pivot) - Low
    - R2 = Pivot + (High - Low)
    - R3 = High + 2 * (Pivot - Low)
    - S1 = (2 * Pivot) - High
    - S2 = Pivot - (High - Low)
    - S3 = Low - 2 * (High - Pivot)

    Args:
        row: A pandas Series with 'High', 'Low', 'Close' keys

    Returns:
        dict with 7 keys: pivot, r1, r2, r3, s1, s2, s3 (all rounded to 2 decimals)
    """
    h = float(row.get("High", 0))
    l = float(row.get("Low", 0))
    c = float(row.get("Close", 0))

    pivot = (h + l + c) / 3
    hl_range = h - l

    r1 = (2 * pivot) - l
    r2 = pivot + hl_range
    r3 = h + 2 * (pivot - l)

    s1 = (2 * pivot) - h
    s2 = pivot - hl_range
    s3 = l - 2 * (h - pivot)

    return {
        "pivot": round(pivot, 2),
        "r1": round(r1, 2),
        "r2": round(r2, 2),
        "r3": round(r3, 2),
        "s1": round(s1, 2),
        "s2": round(s2, 2),
        "s3": round(s3, 2),
    }


def calculate_stoploss_targets(
    sr_levels: dict,
    prediction: str,
    current_price: float
) -> dict | None:
    """
    Calculates three stoploss/target scenarios using auto-adjusted S/R levels.

    Levels are sorted by their position relative to the current price (not
    by their static S1/R1 labels), so the stoploss is always at a real level
    below the entry (Buy) / above the entry (Sell), and the target is always
    at a real level on the profit side. This keeps risk-reward sensible even
    when the price has already moved past one or more pivot levels.

    For Buy signals: SL at supports below price, Target at resistances above
    For Sell signals: SL at resistances above price, Target at supports below
    For Hold: Returns None

    Args:
        sr_levels: dict with keys pivot, r1, r2, r3, s1, s2, s3
        prediction: "Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"
        current_price: current LTP

    Returns:
        dict with keys conservative, moderate, aggressive (each with stoploss,
        target, risk_reward_ratio) or None if prediction is "Hold".
    """
    if prediction == "Hold":
        return None

    is_buy = prediction in ["Buy", "Strong Buy"]

    # Traditional support / resistance pools (no cross-over: SL stays a real
    # support for Buy, real resistance for Sell). Sorted from nearest-to-price
    # outwards so picks degrade gracefully when fewer than 3 levels qualify.
    supports = sorted(
        [sr_levels["s1"], sr_levels["s2"], sr_levels["s3"]], reverse=True
    )  # [highest, mid, lowest]
    resistances = sorted(
        [sr_levels["r1"], sr_levels["r2"], sr_levels["r3"]]
    )  # [lowest, mid, highest]

    # Auto-adjust: keep only levels on the correct side of current price.
    # Stoploss must be below entry (Buy) / above entry (Sell); target the inverse.
    # If filtering empties a pool, fall back to the full list so we still emit
    # numbers (the RR will simply look poor and the trader can judge the setup).
    if is_buy:
        sl_pool = [s for s in supports if s < current_price] or supports
        tgt_pool = [r for r in resistances if r > current_price] or resistances
    else:
        sl_pool = [r for r in resistances if r > current_price] or resistances
        tgt_pool = [s for s in supports if s < current_price] or supports

    def _pick(levels: list, idx: int):
        """Pick the idx-th level if available, else fall back to the last one."""
        if not levels:
            return None
        return levels[min(idx, len(levels) - 1)]

    conservative = {
        "stoploss": _pick(sl_pool, 0),
        "target": _pick(tgt_pool, 0),
    }
    moderate = {
        "stoploss": _pick(sl_pool, 0),
        "target": _pick(tgt_pool, 1),
    }
    aggressive = {
        "stoploss": _pick(sl_pool, 1),
        "target": _pick(tgt_pool, 2),
    }

    def _calc_rr(sl, tgt, price):
        if price is None or sl is None or tgt is None:
            return 0
        sl_dist = abs(price - sl)
        tgt_dist = abs(tgt - price)
        if sl_dist < 1e-10:
            return 0
        return round(tgt_dist / sl_dist, 2)

    for scenario in [conservative, moderate, aggressive]:
        scenario["risk_reward_ratio"] = _calc_rr(
            scenario["stoploss"], scenario["target"], current_price
        )

    return {
        "conservative": conservative,
        "moderate": moderate,
        "aggressive": aggressive,
    }
