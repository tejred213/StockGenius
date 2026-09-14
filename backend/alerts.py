"""
Price & signal alerts API.

Users create alerts keyed by their WhatsApp phone number (the phone *is* the
identity — there is no login system). A GitHub Actions cron periodically POSTs to
`/api/alerts/check`, which evaluates active alerts and delivers WhatsApp messages
for any that fire.

Endpoints:
    POST   /api/alerts            create
    GET    /api/alerts?phone=…    list a phone's alerts
    PATCH  /api/alerts/{id}       change status (pause / resume / re-arm)
    DELETE /api/alerts/{id}       remove
    POST   /api/alerts/check      cron-only; evaluate + notify
"""

import os
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Header
from pydantic import BaseModel, field_validator

import alerts_store as store
from cache_manager import is_market_open
from whatsapp import send_whatsapp

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


# ----------------------------------------------------------------------
# Request models
# ----------------------------------------------------------------------
class AlertCreate(BaseModel):
    phone: str
    ticker: str
    type: str
    threshold: Optional[float] = None
    target_signal: Optional[str] = None
    note: Optional[str] = None

    @field_validator("phone")
    @classmethod
    def _validate_phone(cls, v: str) -> str:
        v = v.strip().replace(" ", "")
        if not v.startswith("+") or not v[1:].isdigit() or not (8 <= len(v[1:]) <= 15):
            raise ValueError("phone must be E.164 format, e.g. +919876543210")
        return v

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
# CRUD
# ----------------------------------------------------------------------
@router.post("")
def create_alert(payload: AlertCreate):
    from main import normalize_ticker  # lazy import avoids circular import

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
        phone=payload.phone,
        ticker=ticker,
        type_=payload.type,
        threshold=payload.threshold,
        target_signal=payload.target_signal,
        note=payload.note,
    )
    return alert


@router.get("")
def list_alerts(phone: str = Query(..., description="E.164 phone the alerts belong to")):
    return {"alerts": store.list_alerts(phone), "phone": phone}


@router.patch("/{alert_id}")
def patch_alert(alert_id: str, payload: AlertPatch):
    if not store.set_status(alert_id, payload.status):
        raise HTTPException(404, "alert not found")
    return store.get_alert(alert_id)


@router.delete("/{alert_id}")
def delete_alert(alert_id: str):
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


@router.post("/check")
def check_alerts(x_cron_secret: Optional[str] = Header(default=None)):
    """
    Cron-only. Evaluates active alerts and sends WhatsApp for any that fire.
    Guarded by the X-Cron-Secret header matching the CRON_SECRET env var.
    Off-market-hours it returns early without any data fetching.
    """
    expected = os.environ.get("CRON_SECRET")
    if not expected:
        raise HTTPException(503, "CRON_SECRET not configured on the server")
    if x_cron_secret != expected:
        raise HTTPException(401, "invalid cron secret")

    if not is_market_open():
        return {"checked": 0, "triggered": 0, "skipped": "market_closed"}

    from main import get_live_price  # lazy import avoids circular import
    from main import ml_engine

    active = store.list_active_alerts()
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
    for alert in active:
        tk = alert["ticker"]
        fired, value_str = _evaluate_condition(
            alert, price_cache.get(tk), signal_cache.get(tk)
        )
        if fired:
            sent = send_whatsapp(alert["phone"], _build_message(alert, value_str))
            store.mark_triggered(alert["id"], value_str)
            if sent:
                triggered += 1
            else:
                logger.warning("Alert %s fired but WhatsApp delivery failed", alert["id"])
        else:
            store.mark_checked(alert["id"], value_str)

    return {"checked": len(active), "triggered": triggered, "skipped": 0}
