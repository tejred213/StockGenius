"""
Smart Caching System — in-memory TTL cache.
Fetch once, hold in RAM, auto-refresh after TTL expires on the next request.
No pickle / disk I/O — ideal for ephemeral hosting like Render.
"""

import time
import logging
from collections import OrderedDict
from datetime import datetime, timedelta, time as dtime
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# TTL presets (in seconds)
TTL_MODEL = 24 * 3600        # 24 hours — trained models
TTL_PRICES = 6 * 3600        # 6 hours  — stock price data
TTL_OPTION_CHAIN = 2 * 3600  # 2 hours  — live option chain
TTL_FNO_HIST = 24 * 3600     # 24 hours — historical F&O bhav copies
TTL_NIFTY50 = 4 * 3600       # 4 hours  — Nifty 50 comparison (out-of-hours)
TTL_DAILY = 24 * 3600        # 24 hours — reserved for daily-cadence data

TTL_MOMENTUM_MARKET = 15 * 60     # 15 min — momentum lists during NSE hours
TTL_MOMENTUM_OFF_HOURS = 1 * 3600  # 1 hour — momentum lists outside NSE hours

# NSE regular session, IST (India observes no DST, so a fixed UTC+5:30
# offset is exact year-round — no zoneinfo / tzdata dependency needed).
_IST_OFFSET = timedelta(hours=5, minutes=30)
_MARKET_OPEN = dtime(9, 15)
_MARKET_CLOSE = dtime(15, 30)


def is_market_open(now_utc: Optional[datetime] = None) -> bool:
    """True during NSE regular trading hours (Mon-Fri, 09:15-15:30 IST)."""
    now_ist = (now_utc or datetime.utcnow()) + _IST_OFFSET
    if now_ist.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return _MARKET_OPEN <= now_ist.time() <= _MARKET_CLOSE


def momentum_ttl() -> int:
    """
    Cache TTL for momentum / leaderboard data.

    During market hours the list refreshes every 15 minutes so it tracks the
    session. Outside market hours the data is static (prices don't change),
    so we hold it for an hour to avoid pointless re-fetches.
    """
    return TTL_MOMENTUM_MARKET if is_market_open() else TTL_MOMENTUM_OFF_HOURS


class CacheEntry:
    """Wrapper that stores data alongside a creation timestamp."""

    __slots__ = ("data", "created_at")

    def __init__(self, data: Any):
        self.data = data
        self.created_at: float = time.time()


# Single in-memory store shared across the process, ordered least- to
# most-recently-used so eviction can drop from the front.
_store: "OrderedDict[str, CacheEntry]" = OrderedDict()

# Per-category entry caps. TTL alone can't bound memory: a full trading day of
# varied tickers would otherwise accumulate without limit. Trained ensembles run
# ~5MB each and dominate the footprint, so models get the tighter cap.
_MAX_ENTRIES = {"model": 60, "data": 250}
_DEFAULT_MAX_ENTRIES = 250


def _evict_if_needed(category: str) -> None:
    """Drop least-recently-used entries until `category` is within its cap."""
    prefix = f"{category}:"
    limit = _MAX_ENTRIES.get(category, _DEFAULT_MAX_ENTRIES)
    keys = [k for k in _store if k.startswith(prefix)]
    for key in keys[: max(0, len(keys) - limit)]:
        del _store[key]
        logger.info("Cache EVICTED (%s at cap %d): %s", category, limit, key)


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
            _store.move_to_end(cache_key)
            logger.debug("Cache HIT (fresh) for %s", key)
            return {"data": entry.data, "stale": False, "from_cache": True}

        # 2. Expired or missing — try to fetch
        try:
            fresh_data = fetch_fn()

            import pandas as pd
            if isinstance(fresh_data, pd.DataFrame) and fresh_data.empty:
                raise ValueError("Fetched DataFrame is empty, bypassing cache to force retry next time.")

            _store[cache_key] = CacheEntry(fresh_data)
            _store.move_to_end(cache_key)
            _evict_if_needed(category)
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
