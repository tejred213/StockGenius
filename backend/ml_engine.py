from __future__ import annotations

"""
ML Engine — Gradient Boosting + Random Forest ensemble with Time-Series Cross-Validation.

Provides:
  - Action Signal: Strong Buy / Buy / Hold / Sell / Strong Sell
  - Trade Strategy: Swing Trade / Positional / Intraday Avoid
  - Feature Importance (top 10)
  - Cross-validated accuracy metric
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from indicators import compute_all_indicators, get_feature_columns
from cache_manager import CacheManager, TTL_PRICES, TTL_MODEL

import requests
yf_session = requests.Session()
yf_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive"
})

logger = logging.getLogger(__name__)

# ======================================================================
# Label definitions
# ======================================================================

ACTION_LABELS = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]

STRATEGY_LABELS = ["Intraday Avoid", "Swing Trade", "Positional"]


def _encode_action_label(future_return: float) -> str:
    if future_return > 0.03:
        return "Strong Buy"
    if future_return >= 0.01:
        return "Buy"
    if future_return > -0.01:
        return "Hold"
    if future_return >= -0.03:
        return "Sell"
    return "Strong Sell"


def _compute_trade_strategy(row: pd.Series) -> str:
    """
    Classify trade strategy directly from the latest indicators.
    Uses ATR as % of price for volatility, RSI for momentum strength,
    and ADX for trend clarity.

    - Swing Trade:    High volatility + strong directional momentum
    - Positional:     Clear trend (high ADX) + moderate/low volatility
    - Intraday Avoid: Choppy, no clear edge
    """
    close = float(row.get("Close", 1))
    atr = float(row.get("ATR_14", 0))
    rsi = float(row.get("RSI_14", 50))
    adx = float(row.get("ADX", 20))
    macd_hist = float(row.get("MACD_Histogram", 0))
    bb_width = float(row.get("BB_Width", 0))

    atr_pct = (atr / (close + 1e-10)) * 100

    # Momentum: strong if RSI is clearly overbought/oversold
    has_rsi_momentum = rsi > 65 or rsi < 35
    # Trend: MACD histogram magnitude relative to price
    has_macd_signal = abs(macd_hist) / (close + 1e-10) * 100 > 0.1
    # Trend strength
    has_strong_trend = adx > 25

    # High volatility + momentum → Swing Trade
    if atr_pct > 1.8 and (has_rsi_momentum or has_macd_signal) and bb_width > 0.06:
        return "Swing Trade"

    # Clear trend + lower volatility → Positional
    if has_strong_trend and atr_pct <= 1.8:
        return "Positional"

    # Moderate trend with some momentum signal → Positional
    if adx > 20 and (has_rsi_momentum or has_macd_signal) and atr_pct <= 2.5:
        return "Positional"

    return "Intraday Avoid"


# ======================================================================
# Core engine
# ======================================================================

class StockMLEngine:
    """Orchestrates data fetching, feature engineering, training, and inference."""

    def evaluate(self, ticker: str) -> dict | None:
        """
        Full end-to-end evaluation for a single ticker.
        Returns an enriched result dictionary or None if data is insufficient.
        """
        # 1. Fetch price data (with cache)
        price_result = CacheManager.get_or_fetch(
            key=f"{ticker}_prices",
            fetch_fn=lambda: yf.Ticker(ticker, session=yf_session).history(period="2y"),
            ttl=TTL_PRICES,
            category="data",
        )
        df_raw: pd.DataFrame = price_result["data"]

        if df_raw is None or df_raw.empty or len(df_raw) < 60:
            logger.error(f"Insufficient data for {ticker}: df length is {len(df_raw) if df_raw is not None else 0}")
            CacheManager.invalidate(f"{ticker}_prices", "data")
            return None

        # Flatten any MultiIndex columns (yfinance can return them)
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)

        # 2. Feature engineering
        df = compute_all_indicators(df_raw)

        # 3. Create labels
        df["Future_Return"] = df["Close"].pct_change(periods=5).shift(-5)
        df["Action_Label"] = df["Future_Return"].apply(
            lambda r: _encode_action_label(r) if pd.notna(r) else np.nan
        )
        # Drop rows with NaN or infinity in any feature or label column
        feature_cols = get_feature_columns()
        required = feature_cols + ["Action_Label", "Future_Return"]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=required, inplace=True)

        if len(df) < 100:
            return None

        X = df[feature_cols].values
        y_action = df["Action_Label"].values

        # 4. Try to load cached model; otherwise train fresh
        model_result = self._get_or_train_models(
            ticker, X, y_action, feature_cols
        )

        # 5. Inference on the latest data point
        latest_row = df.iloc[-1:]
        X_latest = latest_row[feature_cols].values

        action_pred = self._ensemble_predict(
            model_result["gb_action"],
            model_result["rf_action"],
            X_latest,
        )
        # Strategy is rule-based on the latest data (not model-predicted)
        strategy_pred = _compute_trade_strategy(latest_row.iloc[0])

        # Confidence from probability averaging
        action_proba = self._ensemble_proba(
            model_result["gb_action"],
            model_result["rf_action"],
            X_latest,
        )

        # Feature importance (average of both action models)
        importances = (
            model_result["gb_action"].feature_importances_
            + model_result["rf_action"].feature_importances_
        ) / 2
        top_10_idx = np.argsort(importances)[::-1][:10]
        top_features = [
            {"feature": feature_cols[i], "importance": round(float(importances[i]), 4)}
            for i in top_10_idx
        ]

        # Fetch LTP (Latest Trade Price) — real-time from yfinance fast_info
        try:
            tkr_info = yf.Ticker(ticker, session=yf_session).fast_info
            ltp = round(float(tkr_info["lastPrice"]), 2)
            previous_close = round(float(tkr_info.get("previousClose", 0)), 2)
            day_change = round(ltp - previous_close, 2) if previous_close else None
            day_change_pct = round((day_change / previous_close) * 100, 2) if previous_close else None
        except Exception:
            # Fallback to last close from historical data
            ltp = round(float(df_raw["Close"].iloc[-1]), 2)
            previous_close = round(float(df_raw["Close"].iloc[-2]), 2) if len(df_raw) > 1 else None
            day_change = round(ltp - previous_close, 2) if previous_close else None
            day_change_pct = round((day_change / previous_close) * 100, 2) if previous_close else None

        # Build technicals snapshot
        latest = latest_row.iloc[0]
        technicals = {
            "current_price": round(float(latest["Close"]), 2),
            "SMA_50": round(float(latest["SMA_50"]), 2),
            "SMA_200": round(float(latest["SMA_200"]), 2),
            "RSI_14": round(float(latest["RSI_14"]), 2),
            "MACD": round(float(latest["MACD"]), 2),
            "ADX": round(float(latest["ADX"]), 2),
            "ATR_14": round(float(latest["ATR_14"]), 2),
            "BB_Width": round(float(latest["BB_Width"]), 4),
            "Stoch_K": round(float(latest["Stoch_K"]), 2),
            "OBV": int(latest["OBV"]),
        }

        return {
            "ticker": ticker,
            "ltp": ltp,
            "previous_close": previous_close,
            "day_change": day_change,
            "day_change_pct": day_change_pct,
            "prediction": action_pred,
            "trade_strategy": strategy_pred,
            "confidence": round(float(max(action_proba)) * 100, 2),
            "all_probabilities": {
                str(cls): round(float(p) * 100, 2)
                for cls, p in zip(model_result["action_classes"], action_proba)
            },
            "model_accuracy_cv": model_result["cv_accuracy"],
            "top_features": top_features,
            "technicals": technicals,
            "data_stale": price_result.get("stale", False),
        }

    # ------------------------------------------------------------------
    # Model training & caching
    # ------------------------------------------------------------------

    def _get_or_train_models(
        self,
        ticker: str,
        X: np.ndarray,
        y_action: np.ndarray,
        feature_cols: list[str],
    ) -> dict:
        """Return trained models — from cache if fresh, otherwise train new."""

        def _train() -> dict:
            return self._train_models(X, y_action, feature_cols)

        result = CacheManager.get_or_fetch(
            key=f"{ticker}_model",
            fetch_fn=_train,
            ttl=TTL_MODEL,
            category="model",
        )
        return result["data"]

    def _train_models(
        self,
        X: np.ndarray,
        y_action: np.ndarray,
        feature_cols: list[str],
    ) -> dict:
        logger.info("Training ensemble models…")

        # --- Gradient Boosting for action signals ---
        gb_action = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
        )

        # --- Random Forest for action signals ---
        rf_action = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )

        # Time-series cross-validation (5 folds)
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(rf_action, X, y_action, cv=tscv, scoring="accuracy")
        cv_accuracy = round(float(cv_scores.mean()) * 100, 2)

        # Fit on full data
        gb_action.fit(X, y_action)
        rf_action.fit(X, y_action)

        return {
            "gb_action": gb_action,
            "rf_action": rf_action,
            "action_classes": rf_action.classes_,
            "cv_accuracy": cv_accuracy,
        }

    # ------------------------------------------------------------------
    # Ensemble helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensemble_predict(gb_model, rf_model, X: np.ndarray) -> str:
        """Averaged probability ensemble → pick best class."""
        proba = StockMLEngine._ensemble_proba(gb_model, rf_model, X)
        classes = rf_model.classes_
        return str(classes[np.argmax(proba)])

    @staticmethod
    def _ensemble_proba(gb_model, rf_model, X: np.ndarray) -> np.ndarray:
        """Average the probability outputs of GradientBoosting and Random Forest."""
        rf_classes = list(rf_model.classes_)
        gb_classes = list(gb_model.classes_)

        rf_proba = rf_model.predict_proba(X)[0]
        gb_proba_raw = gb_model.predict_proba(X)[0]

        # Align GB proba to RF class order
        gb_proba_aligned = np.zeros(len(rf_classes))
        for i, cls in enumerate(rf_classes):
            if cls in gb_classes:
                gb_proba_aligned[i] = gb_proba_raw[gb_classes.index(cls)]

        return (rf_proba + gb_proba_aligned) / 2
