"""
Telegram delivery via the Bot API (free, no messaging window or templates).

Setup: create a bot with @BotFather and set its token as a Fly secret:
    fly secrets set TELEGRAM_BOT_TOKEN=123456:ABC... -a stockgenius-api

Linking a user: the Alerts page sends them to t.me/<bot>?start=<token>. Tapping
Start makes Telegram deliver "/start <token>" to the bot, which we receive by
long-polling getUpdates in a background thread (the Fly machine is always on,
so no public webhook needs registering).

If unconfigured, send_message() logs a warning and returns False instead of
raising — so local dev and unit tests run without a bot token.
"""

import os
import time
import logging
import threading
from typing import Callable, Optional

import requests

logger = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org"
_POLL_TIMEOUT = 30  # seconds Telegram holds a getUpdates long-poll open

_bot_username: Optional[str] = None
_stop_event = threading.Event()
_poll_thread: Optional[threading.Thread] = None


def is_configured() -> bool:
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN"))


def _call(method: str, http_timeout: float = 15, **params) -> Optional[dict]:
    """Call a Bot API method. Returns its `result`, or None on any failure."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return None
    try:
        resp = requests.post(f"{_API_BASE}/bot{token}/{method}", json=params, timeout=http_timeout)
        data = resp.json()
    except Exception as exc:  # network / timeout / non-JSON
        logger.error("Telegram %s error: %s", method, exc)
        return None
    if not data.get("ok"):
        logger.error(
            "Telegram %s failed (%s): %s", method, resp.status_code, data.get("description")
        )
        return None
    return data.get("result")


def send_message(chat_id: str, text: str) -> bool:
    """
    Send a text message. Returns True once Telegram accepts it, False otherwise.
    Never raises — callers treat delivery as best-effort.
    """
    if not is_configured():
        logger.warning("Telegram not configured — skipping message to %s", chat_id)
        return False
    result = _call("sendMessage", chat_id=chat_id, text=text, disable_web_page_preview=True)
    if result is None:
        return False
    logger.info("Telegram sent to %s", chat_id)
    return True


def get_bot_username() -> Optional[str]:
    """The bot's @username (without @), looked up once via getMe."""
    global _bot_username
    if _bot_username is None:
        me = _call("getMe")
        if me:
            _bot_username = me.get("username")
    return _bot_username


# ----------------------------------------------------------------------
# Incoming updates (long polling)
# ----------------------------------------------------------------------
def _poll_loop(on_start: Callable[[str, str, str], Optional[str]]) -> None:
    # getUpdates refuses to run while a webhook is set; clear any stale one.
    _call("deleteWebhook")
    offset = None
    while not _stop_event.is_set():
        params = {"timeout": _POLL_TIMEOUT, "allowed_updates": ["message"]}
        if offset is not None:
            params["offset"] = offset
        updates = _call("getUpdates", http_timeout=_POLL_TIMEOUT + 10, **params)
        if updates is None:
            # Network blip, or 409 while an old machine is still polling during
            # a deploy — back off briefly and retry.
            _stop_event.wait(5)
            continue
        for update in updates:
            offset = update["update_id"] + 1
            try:
                _handle_update(update, on_start)
            except Exception:
                logger.exception("Telegram update handling failed")


def _handle_update(update: dict, on_start: Callable[[str, str, str], Optional[str]]) -> None:
    msg = update.get("message") or {}
    text = (msg.get("text") or "").strip()
    chat = msg.get("chat") or {}
    if not text.startswith("/start") or "id" not in chat:
        return
    parts = text.split(maxsplit=1)
    payload = parts[1] if len(parts) > 1 else ""
    sender = msg.get("from") or {}
    name = sender.get("username") or sender.get("first_name") or ""
    reply = on_start(str(chat["id"]), payload, name)
    if reply:
        _call("sendMessage", chat_id=chat["id"], text=reply)


def start_polling(on_start: Callable[[str, str, str], Optional[str]]) -> None:
    """
    Start the background getUpdates loop. `on_start(chat_id, payload, name)` is
    called for every /start command and returns the reply text to send back.
    """
    global _poll_thread
    if not is_configured():
        logger.warning("TELEGRAM_BOT_TOKEN not set — Telegram linking disabled")
        return
    if _poll_thread and _poll_thread.is_alive():
        return
    _stop_event.clear()
    _poll_thread = threading.Thread(
        target=_poll_loop, args=(on_start,), name="telegram-poller", daemon=True
    )
    _poll_thread.start()
    logger.info("Telegram update polling started")


def stop_polling() -> None:
    _stop_event.set()
