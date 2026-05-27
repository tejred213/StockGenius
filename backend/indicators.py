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


# ======================================================================
#  Classical Support & Resistance (swing-based, multi-timeframe)
# ======================================================================

def calculate_classical_sr(
    df: pd.DataFrame,
    current_price: float,
    window: int = 5,
    cluster_pct: float = 0.015,
    max_levels: int = 4,
) -> dict:
    """
    Identify classical (chart-drawable) support and resistance levels from
    swing highs and lows in the provided OHLC dataframe.

    Pipeline:
      1. Detect fractal swing points — a bar's High is a swing high if it is
         the max of the (2*window+1)-bar window centered on it (likewise for
         Lows / swing lows).
      2. Cluster nearby swings within `cluster_pct` of each other; each
         cluster becomes one S/R level whose price is the average of its
         member swings.
      3. Rank clusters by touch count (desc) then recency (desc).
      4. Split into resistances (above current price) and supports (below).
      5. Return the top `max_levels` of each, sorted nearest→farthest.

    The dataframe's interval (1H / 4H / Daily) is implicit — caller passes a
    frame at the desired timeframe.

    Args:
        df: OHLCV DataFrame with at least 'High' and 'Low' columns. Index
            should be a DatetimeIndex for `last_touched` to be meaningful.
        current_price: LTP, used to split levels into supports vs resistances.
        window: Fractal lookaround (default 5 → 11-bar centered window).
        cluster_pct: Levels closer than this fraction merge (default 1.5%).
        max_levels: Cap on returned levels per side (default 4).

    Returns:
        {
          "resistances": [{"price": float, "touches": int, "last_touched": "..."}],
          "supports":    [{"price": float, "touches": int, "last_touched": "..."}],
        }
        Empty lists if insufficient data.
    """
    if df is None or df.empty or len(df) < (2 * window + 1):
        return {"resistances": [], "supports": []}

    highs = df["High"].values
    lows = df["Low"].values
    timestamps = df.index

    # 1. Fractal swing detection (vectorized via rolling)
    swing_highs = []  # list of (timestamp, price)
    swing_lows = []
    n = len(df)
    for i in range(window, n - window):
        wh = highs[i - window : i + window + 1]
        wl = lows[i - window : i + window + 1]
        if highs[i] == wh.max():
            swing_highs.append((timestamps[i], float(highs[i])))
        if lows[i] == wl.min():
            swing_lows.append((timestamps[i], float(lows[i])))

    def _cluster(swings: list) -> list:
        """Merge swings within cluster_pct of each other into a single level.

        Returns list of dicts: {price, touches, last_touched (datetime)}.
        """
        if not swings:
            return []
        # Sort by price ascending so contiguous-price members cluster naturally
        swings_sorted = sorted(swings, key=lambda s: s[1])
        clusters: list[list[tuple]] = [[swings_sorted[0]]]
        for ts, px in swings_sorted[1:]:
            ref_px = clusters[-1][-1][1]
            if abs(px - ref_px) / max(ref_px, 1e-10) < cluster_pct:
                clusters[-1].append((ts, px))
            else:
                clusters.append([(ts, px)])
        out = []
        for c in clusters:
            avg_price = sum(p for _, p in c) / len(c)
            touches = len(c)
            last_touched = max(ts for ts, _ in c)
            out.append(
                {
                    "price": round(avg_price, 2),
                    "touches": touches,
                    "last_touched": last_touched,
                }
            )
        return out

    resistance_clusters = _cluster(swing_highs)
    support_clusters = _cluster(swing_lows)

    # 2. Filter by side of current price
    resistances = [c for c in resistance_clusters if c["price"] > current_price]
    supports = [c for c in support_clusters if c["price"] < current_price]

    # 3. Rank: touches desc, then recency desc. Take top N.
    def _rank(levels: list) -> list:
        return sorted(
            levels,
            key=lambda c: (-c["touches"], -c["last_touched"].timestamp()),
        )[:max_levels]

    top_resistances = _rank(resistances)
    top_supports = _rank(supports)

    # 4. Final display order: resistances nearest→farthest from price (ascending)
    #    supports nearest→farthest from price (descending)
    top_resistances.sort(key=lambda c: c["price"])
    top_supports.sort(key=lambda c: -c["price"])

    # 5. Serialize datetime → ISO date string for JSON safety
    def _serialize(levels: list) -> list:
        return [
            {
                "price": lv["price"],
                "touches": lv["touches"],
                "last_touched": lv["last_touched"].strftime("%Y-%m-%d"),
            }
            for lv in levels
        ]

    return {
        "resistances": _serialize(top_resistances),
        "supports": _serialize(top_supports),
    }


def calculate_stoploss_targets(
    classical_sr: dict,
    prediction: str,
    current_price: float,
) -> dict | None:
    """
    Compute three stoploss/target scenarios from classical S/R levels.

    For Buy signals: SL at supports (below price), Target at resistances (above)
    For Sell signals: SL at resistances (above price), Target at supports (below)
    For Hold: Returns None

    Strategy mapping (indices into the sorted level lists, nearest=index 0):
      Conservative: SL=pool[0], Target=tgt[0]   — tightest stop, nearest target
      Moderate:     SL=pool[0], Target=tgt[1]   — same stop, farther target
      Aggressive:   SL=pool[1], Target=tgt[2]   — wider stop, farthest target

    Falls back to the last available element when fewer than 3 levels exist.

    Args:
        classical_sr: dict with keys "resistances" and "supports", each a list
            of {"price", "touches", "last_touched"} (output of
            calculate_classical_sr).
        prediction: "Strong Buy" | "Buy" | "Hold" | "Sell" | "Strong Sell"
        current_price: current LTP

    Returns:
        {
          "conservative": {"stoploss": float, "target": float, "risk_reward_ratio": float},
          "moderate":     {...},
          "aggressive":   {...},
        }
        or None if prediction is "Hold" or the S/R lists are empty.
    """
    if prediction == "Hold":
        return None

    resistances = [lv["price"] for lv in classical_sr.get("resistances", [])]
    supports = [lv["price"] for lv in classical_sr.get("supports", [])]

    if not resistances and not supports:
        return None

    is_buy = prediction in ["Buy", "Strong Buy"]

    # Resistances are sorted ascending (nearest above price first).
    # Supports are sorted descending (nearest below price first).
    if is_buy:
        sl_pool = supports
        tgt_pool = resistances
    else:
        sl_pool = resistances
        tgt_pool = supports

    def _pick(levels: list, idx: int):
        if not levels:
            return None
        return levels[min(idx, len(levels) - 1)]

    conservative = {"stoploss": _pick(sl_pool, 0), "target": _pick(tgt_pool, 0)}
    moderate = {"stoploss": _pick(sl_pool, 0), "target": _pick(tgt_pool, 1)}
    aggressive = {"stoploss": _pick(sl_pool, 1), "target": _pick(tgt_pool, 2)}

    def _calc_rr(sl, tgt, price):
        if price is None or sl is None or tgt is None:
            return 0
        sl_dist = abs(price - sl)
        tgt_dist = abs(tgt - price)
        if sl_dist < 1e-10:
            return 0
        return round(tgt_dist / sl_dist, 2)

    for scenario in (conservative, moderate, aggressive):
        scenario["risk_reward_ratio"] = _calc_rr(
            scenario["stoploss"], scenario["target"], current_price
        )

    return {
        "conservative": conservative,
        "moderate": moderate,
        "aggressive": aggressive,
    }
