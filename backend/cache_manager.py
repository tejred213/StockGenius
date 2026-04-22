"""
Smart Caching System — in-memory TTL cache.
Fetch once, hold in RAM, auto-refresh after TTL expires on the next request.
No pickle / disk I/O — ideal for ephemeral hosting like Render.
"""

import time
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# TTL presets (in seconds)
TTL_MODEL = 24 * 3600        # 24 hours — trained models
TTL_PRICES = 6 * 3600        # 6 hours  — stock price data
TTL_OPTION_CHAIN = 2 * 3600  # 2 hours  — live option chain
TTL_FNO_HIST = 24 * 3600     # 24 hours — historical F&O bhav copies
TTL_NIFTY50 = 4 * 3600       # 4 hours  — Nifty 50 comparison


class CacheEntry:
    """Wrapper that stores data alongside a creation timestamp."""

    __slots__ = ("data", "created_at")

    def __init__(self, data: Any):
        self.data = data
        self.created_at: float = time.time()


# Single in-memory store shared across the process
_store: dict[str, CacheEntry] = {}


class CacheManager:
    """
    Unified in-memory caching layer.

    Usage:
        result = CacheManager.get_or_fetch(
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
        cache_key = f"{category}:{key}"
        entry = _store.get(cache_key)

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

            _store[cache_key] = CacheEntry(fresh_data)
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
        cache_key = f"{category}:{key}"
        if cache_key in _store:
            del _store[cache_key]
            logger.info("Cache INVALIDATED for %s", key)

    @staticmethod
    def clear_all() -> int:
        """Clear the entire in-memory cache.  Returns count of removed entries."""
        count = len(_store)
        _store.clear()
        if count:
            logger.info("Cache cleared: removed %d entries", count)
        return count
