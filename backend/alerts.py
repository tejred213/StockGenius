"""
Price & signal alerts API.

Alerts are delivered over Telegram. There is no login system: a browser links
itself to a Telegram chat by sending the bot a one-time token (see
alerts_store's link tokens), and that token is then the browser's `key` for
every alerts call. A background thread started with the app evaluates active
alerts every couple of minutes during market hours and messages the chat for
any that fire. `/api/alerts/check` runs the same check and is kept as a
cron-driven backup.

Endpoints:
    POST   /api/alerts/telegram/link          start linking; returns token + t.me URL
    GET    /api/alerts/telegram/link/{token}  has the user tapped Start yet?
    POST   /api/alerts                        create
    GET    /api/alerts?key=…                  list the linked chat's alerts
    PATCH  /api/alerts/{id}?key=…             change status (pause / resume / re-arm)
    DELETE /api/alerts/{id}?key=…             remove
    POST   /api/alerts/check                  cron-only backup; evaluate + notify
"""

import os
import time
import logging
import threading
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Header
from pydantic import BaseModel, field_validator

import alerts_store as store
from cache_manager import is_market_open
import telegram

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


# ----------------------------------------------------------------------
# Request models
# ----------------------------------------------------------------------
class AlertCreate(BaseModel):
    key: str
    ticker: str
    type: str
    threshold: Optional[float] = None
    target_signal: Optional[str] = None
    note: Optional[str] = None

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        if v not in store.VALID_TYPES:
            raise ValueError(f"type must be one of {sorted(store.VALID_TYPES)}")
        return v


class AlertPatch(BaseModel):
    status: str

    @field_validator("status")
    @classmethod
    def _validate_status(cls, v: str) -> str:
        if v not in store.VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(store.VALID_STATUSES)}")
        return v


# ----------------------------------------------------------------------
# Telegram linking
# ----------------------------------------------------------------------
@router.post("/telegram/link")
def start_link():
    if not telegram.is_configured():
        raise HTTPException(503, "Telegram bot is not configured on the server")
    bot = telegram.get_bot_username()
    if not bot:
        raise HTTPException(503, "Could not reach the Telegram bot — try again shortly")
    token = store.create_link_token()
    return {"token": token, "bot": bot, "url": f"https://t.me/{bot}?start={token}"}


@router.get("/telegram/link/{token}")
def link_status(token: str):
    link = store.get_link(token)
    if not link:
        raise HTTPException(404, "unknown link token")
    return {"linked": link["chat_id"] is not None, "name": link["name"]}


def handle_start(chat_id: str, payload: str, name: str) -> str:
    """Reply for a /start command sent to the bot (called by the telegram poller)."""
    if not payload:
        return (
            "Hi! To get StockGenius alerts here, open the Alerts page on "
            "StockGenius and tap \"Connect Telegram\"."
        )
    if store.link_token(payload, chat_id, name or None):
        return "✅ Connected to StockGenius. Your alerts will be delivered to this chat."
    return (
        "That link has expired or was already used. Open the Alerts page on "
        "StockGenius and tap \"Connect Telegram\" again."
    )


def _chat_for_key(key: str) -> str:
    chat_id = store.chat_for_token(key)
    if not chat_id:
        raise HTTPException(401, "Telegram is not connected — link it from the Alerts page")
    return chat_id


def _owned_alert(alert_id: str, key: str) -> dict:
    chat_id = _chat_for_key(key)
    alert = store.get_alert(alert_id)
    if not alert or alert["chat_id"] != chat_id:
        raise HTTPException(404, "alert not found")
    return alert


# ----------------------------------------------------------------------
# CRUD
# ----------------------------------------------------------------------
@router.post("")
def create_alert(payload: AlertCreate):
    from main import normalize_ticker  # lazy import avoids circular import

    chat_id = _chat_for_key(payload.key)
    ticker = normalize_ticker(payload.ticker)

    # Per-type field requirements
    if payload.type in ("price_above", "price_below", "pct_move"):
        if payload.threshold is None or payload.threshold <= 0:
            raise HTTPException(400, f"{payload.type} requires a positive 'threshold'")
    elif payload.type == "signal":
        if payload.target_signal not in store.VALID_SIGNALS:
            raise HTTPException(
                400, f"signal alert requires 'target_signal' in {sorted(store.VALID_SIGNALS)}"
            )

    alert = store.create_alert(
        chat_id=chat_id,
        ticker=ticker,
        type_=payload.type,
        threshold=payload.threshold,
        target_signal=payload.target_signal,
        note=payload.note,
    )
    return alert


@router.get("")
def list_alerts(key: str = Query(..., description="Link token from /telegram/link")):
    link = store.get_link(key)
    if not link or not link["chat_id"]:
        raise HTTPException(401, "Telegram is not connected — link it from the Alerts page")
    return {"alerts": store.list_alerts(link["chat_id"]), "name": link["name"]}


@router.patch("/{alert_id}")
def patch_alert(alert_id: str, payload: AlertPatch, key: str = Query(...)):
    _owned_alert(alert_id, key)
    if not store.set_status(alert_id, payload.status):
        raise HTTPException(404, "alert not found")
    return store.get_alert(alert_id)


@router.delete("/{alert_id}")
def delete_alert(alert_id: str, key: str = Query(...)):
    _owned_alert(alert_id, key)
    if not store.delete_alert(alert_id):
        raise HTTPException(404, "alert not found")
    return {"deleted": alert_id}


# ----------------------------------------------------------------------
# Cron check
# ----------------------------------------------------------------------
def _evaluate_condition(alert: dict, price: Optional[dict], evaluation: Optional[dict]):
    """
    Returns (fired: bool, value_str: str) for a single alert given the freshly
    fetched price dict and/or ML evaluation dict for its ticker.
    """
    t = alert["type"]

    if t in ("price_above", "price_below", "pct_move"):
        if not price:
            return False, "n/a"
        ltp = price.get("ltp")
        pct = price.get("day_change_pct")
        thr = alert["threshold"]
        if t == "price_above":
            return (ltp is not None and ltp >= thr), f"₹{ltp}"
        if t == "price_below":
            return (ltp is not None and ltp <= thr), f"₹{ltp}"
        if t == "pct_move":
            return (pct is not None and abs(pct) >= thr), f"{pct}%"

    if t == "signal":
        if not evaluation:
            return False, "n/a"
        prediction = evaluation.get("prediction")
        return (prediction == alert["target_signal"]), str(prediction)

    return False, "n/a"


def _build_message(alert: dict, value_str: str) -> str:
    ticker = alert["ticker"].replace(".NS", "").replace(".BO", "")
    t = alert["type"]
    if t == "price_above":
        headline = f"📈 {ticker} crossed above ₹{alert['threshold']:g}"
    elif t == "price_below":
        headline = f"📉 {ticker} dropped below ₹{alert['threshold']:g}"
    elif t == "pct_move":
        headline = f"⚡ {ticker} moved ±{alert['threshold']:g}% today"
    elif t == "signal":
        headline = f"🤖 {ticker} signal is now {alert['target_signal']}"
    else:
        headline = f"{ticker} alert"

    lines = [f"StockGenius alert", "", headline, f"Current: {value_str}"]
    if alert.get("note"):
        lines.append(f"Note: {alert['note']}")
    return "\n".join(lines)


def run_check(include_signals: bool = True) -> dict:
    """
    Evaluates active alerts and sends a Telegram message for any that fire.
    Off-market-hours it returns early without any data fetching.

    Price alerts are cheap (one fast_info call per ticker); signal alerts run the
    ML engine, which reuses cached models/daily prices, so the scheduler checks
    them less often by passing include_signals=False on most ticks.
    """
    if not is_market_open():
        return {"checked": 0, "triggered": 0, "skipped": "market_closed"}

    # The in-process scheduler and the cron endpoint can overlap — serialise so
    # an alert can't be evaluated (and messaged) twice at the same moment.
    with _check_lock:
        return _run_check_locked(include_signals)


def _run_check_locked(include_signals: bool) -> dict:
    from main import get_live_price  # lazy import avoids circular import
    from main import ml_engine

    active = store.list_active_alerts()
    if not include_signals:
        active = [a for a in active if a["type"] != "signal"]
    if not active:
        return {"checked": 0, "triggered": 0, "skipped": 0}

    # Group by ticker so each ticker is fetched / evaluated at most once per run.
    tickers = {a["ticker"] for a in active}
    needs_price = {a["ticker"] for a in active if a["type"] != "signal"}
    needs_signal = {a["ticker"] for a in active if a["type"] == "signal"}

    price_cache: dict[str, Optional[dict]] = {}
    signal_cache: dict[str, Optional[dict]] = {}

    for tk in tickers:
        if tk in needs_price:
            try:
                price_cache[tk] = get_live_price(tk)
            except Exception as exc:
                logger.warning("Alert check — price fetch failed for %s: %s", tk, exc)
                price_cache[tk] = None
        if tk in needs_signal:
            try:
                signal_cache[tk] = ml_engine.evaluate(tk)
            except Exception as exc:
                logger.warning("Alert check — evaluate failed for %s: %s", tk, exc)
                signal_cache[tk] = None

    triggered = 0
    failed = 0
    for alert in active:
        tk = alert["ticker"]
        fired, value_str = _evaluate_condition(
            alert, price_cache.get(tk), signal_cache.get(tk)
        )
        if fired:
            if telegram.send_message(alert["chat_id"], _build_message(alert, value_str)):
                store.mark_triggered(alert["id"], value_str)
                triggered += 1
            else:
                # Leave it active so the next check retries delivery instead of
                # silently consuming the one-shot alert.
                store.mark_checked(alert["id"], value_str)
                failed += 1
                logger.warning("Alert %s fired but Telegram delivery failed", alert["id"])
        else:
            store.mark_checked(alert["id"], value_str)

    return {"checked": len(active), "triggered": triggered, "failed": failed, "skipped": 0}


@router.post("/check")
def check_alerts(x_cron_secret: Optional[str] = Header(default=None)):
    """
    Cron-only backup to the in-process scheduler (see start_scheduler).
    Guarded by the X-Cron-Secret header matching the CRON_SECRET env var.
    """
    expected = os.environ.get("CRON_SECRET")
    if not expected:
        raise HTTPException(503, "CRON_SECRET not configured on the server")
    if x_cron_secret != expected:
        raise HTTPException(401, "invalid cron secret")

    return run_check()


# ----------------------------------------------------------------------
# In-process scheduler
# ----------------------------------------------------------------------
# GitHub Actions cron is far too unreliable to drive alerts on its own — in
# practice "*/15" fires a handful of times a day, landing inside NSE hours about
# once. The Fly machine is always on, so a background thread checks instead.
PRICE_CHECK_INTERVAL = int(os.environ.get("ALERTS_CHECK_INTERVAL", "120"))
SIGNAL_CHECK_INTERVAL = int(os.environ.get("ALERTS_SIGNAL_CHECK_INTERVAL", "900"))

_check_lock = threading.Lock()
_stop_event = threading.Event()
_scheduler_thread: Optional[threading.Thread] = None


def _scheduler_loop() -> None:
    last_signal_check = 0.0
    while not _stop_event.is_set():
        try:
            now = time.monotonic()
            include_signals = now - last_signal_check >= SIGNAL_CHECK_INTERVAL
            result = run_check(include_signals=include_signals)
            if include_signals and result.get("skipped") != "market_closed":
                last_signal_check = now
            if result.get("triggered") or result.get("failed"):
                logger.info("Alert scheduler: %s", result)
        except Exception:
            logger.exception("Alert scheduler tick failed")
        _stop_event.wait(PRICE_CHECK_INTERVAL)


def start_scheduler() -> None:
    global _scheduler_thread
    # Receive /start commands so browsers can link their Telegram chat.
    telegram.start_polling(handle_start)
    if os.environ.get("ALERTS_SCHEDULER", "1") == "0":
        logger.info("Alert scheduler disabled via ALERTS_SCHEDULER=0")
        return
    if _scheduler_thread and _scheduler_thread.is_alive():
        return
    _stop_event.clear()
    _scheduler_thread = threading.Thread(target=_scheduler_loop, name="alerts-scheduler", daemon=True)
    _scheduler_thread.start()
    logger.info(
        "Alert scheduler started (price every %ss, signal every %ss)",
        PRICE_CHECK_INTERVAL, SIGNAL_CHECK_INTERVAL,
    )


def stop_scheduler() -> None:
    _stop_event.set()
    telegram.stop_polling()
