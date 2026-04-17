"""
Smart Caching System — fetch once, cache to disk, serve from cache,
auto-refresh after TTL expires on the next request.
"""

import os
import time
import pickle
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Cache directories live under backend/cache/
_CACHE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
_MODEL_CACHE_DIR = os.path.join(_CACHE_ROOT, "model_cache")
_DATA_CACHE_DIR = os.path.join(_CACHE_ROOT, "data_cache")

# Ensure directories exist at import time
for _d in (_CACHE_ROOT, _MODEL_CACHE_DIR, _DATA_CACHE_DIR):
    os.makedirs(_d, exist_ok=True)

# TTL presets (in seconds)
TTL_MODEL = 24 * 3600        # 24 hours — trained models
TTL_PRICES = 6 * 3600        # 6 hours  — stock price data
TTL_OPTION_CHAIN = 2 * 3600  # 2 hours  — live option chain
TTL_FNO_HIST = 24 * 3600     # 24 hours — historical F&O bhav copies
TTL_NIFTY50 = 4 * 3600       # 4 hours  — Nifty 50 comparison

# Stale cache clean-up threshold (7 days)
_CLEANUP_AGE = 7 * 24 * 3600


class CacheEntry:
    """Wrapper that stores data alongside a creation timestamp."""

    def __init__(self, data: Any):
        self.data = data
        self.created_at: float = time.time()


class CacheManager:
    """
    Unified caching layer.

    Usage:
        result = cache.get_or_fetch(
            key="RELIANCE_prices",
            fetch_fn=lambda: yf.Ticker("RELIANCE.NS").history(period="2y"),
            ttl=TTL_PRICES,
            category="data",
        )
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def get_or_fetch(
        key: str,
        fetch_fn: Callable[[], Any],
        ttl: int,
        category: str = "data",
    ) -> dict:
        """
        Returns ``{"data": <value>, "stale": bool, "from_cache": bool}``.

        * Fresh cache hit   → instant return, ``stale=False``
        * Expired cache     → re-fetch; on failure serve stale + warning
        * No cache at all   → fetch; on failure raise
        """
        cache_dir = _MODEL_CACHE_DIR if category == "model" else _DATA_CACHE_DIR
        filepath = os.path.join(cache_dir, f"{key}.pkl")

        entry = CacheManager._load(filepath)

        # 1. Fresh cache hit
        if entry is not None and (time.time() - entry.created_at) < ttl:
            logger.debug("Cache HIT (fresh) for %s", key)
            return {"data": entry.data, "stale": False, "from_cache": True}

        # 2. Expired or missing — try to fetch
        try:
            fresh_data = fetch_fn()
            
            import pandas as pd
            if isinstance(fresh_data, pd.DataFrame) and fresh_data.empty:
                raise ValueError("Fetched DataFrame is empty, bypassing cache to force retry next time.")
                
            CacheManager._save(filepath, CacheEntry(fresh_data))
            logger.info("Cache REFRESHED for %s", key)
            return {"data": fresh_data, "stale": False, "from_cache": False}
        except Exception as exc:
            logger.warning("Fetch failed for %s: %s", key, exc)
            # 3. Serve stale if available
            if entry is not None:
                logger.info("Serving STALE cache for %s", key)
                return {"data": entry.data, "stale": True, "from_cache": True}
            raise  # nothing to fall back on

    @staticmethod
    def invalidate(key: str, category: str = "data") -> None:
        """Remove a specific cache entry."""
        cache_dir = _MODEL_CACHE_DIR if category == "model" else _DATA_CACHE_DIR
        filepath = os.path.join(cache_dir, f"{key}.pkl")
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info("Cache INVALIDATED for %s", key)

    @staticmethod
    def cleanup_old_files() -> int:
        """Delete cache files older than 7 days.  Returns the count of removed files."""
        removed = 0
        now = time.time()
        for dirpath in (_MODEL_CACHE_DIR, _DATA_CACHE_DIR):
            for fname in os.listdir(dirpath):
                fpath = os.path.join(dirpath, fname)
                if os.path.isfile(fpath) and (now - os.path.getmtime(fpath)) > _CLEANUP_AGE:
                    os.remove(fpath)
                    removed += 1
        if removed:
            logger.info("Cache cleanup: removed %d stale files", removed)
        return removed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(filepath: str) -> Optional[CacheEntry]:
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    @staticmethod
    def _save(filepath: str, entry: CacheEntry) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(entry, f)
