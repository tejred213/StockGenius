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
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)
