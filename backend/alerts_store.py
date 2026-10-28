"""
Durable storage for price / signal alerts.

The rest of the app is intentionally stateless (in-memory RAM cache only), but
alerts must survive restarts and deploys, so they live in a small SQLite file.
On Fly.io this sits on a mounted volume (see ALERTS_DB_PATH / fly.toml); locally
it falls back to a file next to the code.

Alerts belong to a Telegram chat. A browser proves it owns that chat with a
"link token": the Alerts page creates one, the user sends it to the bot via
/start, and from then on the token acts as that browser's key to the chat's
alerts.

A fresh connection is opened per operation — volume is tiny and traffic is low,
so this avoids any cross-thread sharing concerns with FastAPI's thread pool.
"""

import os
import time
import uuid
import secrets
import sqlite3
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("ALERTS_DB_PATH", os.path.join(os.path.dirname(__file__), "alerts.db"))

VALID_TYPES = {"price_above", "price_below", "pct_move", "signal"}
VALID_STATUSES = {"active", "triggered", "paused"}
VALID_SIGNALS = {"Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"}

# An unlinked token must be sent to the bot within this window.
LINK_TOKEN_TTL = 60 * 60

_initialised = False


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the alerts table if it doesn't exist. Safe to call repeatedly."""
    global _initialised
    # Ensure the parent directory exists (e.g. the Fly volume mount point).
    parent = os.path.dirname(DB_PATH)
    if parent and not os.path.isdir(parent):
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as exc:
            logger.warning("Could not create alerts DB dir %s: %s", parent, exc)

    with _connect() as conn:
        _retire_whatsapp_table(conn)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id            TEXT PRIMARY KEY,
                chat_id       TEXT NOT NULL,
                ticker        TEXT NOT NULL,
                type          TEXT NOT NULL,
                threshold     REAL,
                target_signal TEXT,
                status        TEXT NOT NULL DEFAULT 'active',
                note          TEXT,
                created_at    REAL NOT NULL,
                last_checked  REAL,
                triggered_at  REAL,
                last_value    TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_chat   ON alerts(chat_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS telegram_links (
                token      TEXT PRIMARY KEY,
                chat_id    TEXT,
                name       TEXT,
                created_at REAL NOT NULL,
                linked_at  REAL
            )
            """
        )
    _initialised = True
    logger.info("Alerts DB ready at %s", DB_PATH)


def _retire_whatsapp_table(conn: sqlite3.Connection) -> None:
    """
    Alerts used to be keyed by WhatsApp phone number. Those can't be delivered
    over Telegram, so move the old table aside (kept, not dropped) and let a
    fresh chat-keyed table be created.
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(alerts)")}
    if "phone" in cols:
        conn.execute("ALTER TABLE alerts RENAME TO alerts_whatsapp_legacy")
        conn.execute("DROP INDEX IF EXISTS idx_alerts_phone")
        conn.execute("DROP INDEX IF EXISTS idx_alerts_status")
        logger.info("Moved WhatsApp-era alerts to alerts_whatsapp_legacy")


def _ensure() -> None:
    if not _initialised:
        init_db()


# ----------------------------------------------------------------------
# Telegram link tokens
# ----------------------------------------------------------------------
def create_link_token() -> str:
    _ensure()
    token = secrets.token_urlsafe(24)  # 32 chars; fits Telegram's 64-char /start limit
    now = time.time()
    with _connect() as conn:
        # Opportunistically clear abandoned, never-linked tokens.
        conn.execute(
            "DELETE FROM telegram_links WHERE chat_id IS NULL AND created_at < ?",
            (now - LINK_TOKEN_TTL,),
        )
        conn.execute(
            "INSERT INTO telegram_links (token, created_at) VALUES (?, ?)", (token, now)
        )
    return token


def link_token(token: str, chat_id: str, name: Optional[str]) -> bool:
    """Bind a pending token to a chat. False if unknown, expired or already used."""
    _ensure()
    now = time.time()
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE telegram_links SET chat_id = ?, name = ?, linked_at = ? "
            "WHERE token = ? AND chat_id IS NULL AND created_at >= ?",
            (chat_id, name, now, token, now - LINK_TOKEN_TTL),
        )
        if cur.rowcount > 0:
            return True
        # Telegram re-delivers updates after a restart — the same chat sending
        # the same token again is still a success.
        row = conn.execute(
            "SELECT chat_id FROM telegram_links WHERE token = ?", (token,)
        ).fetchone()
        return bool(row and row["chat_id"] == chat_id)


def get_link(token: str) -> Optional[dict]:
    _ensure()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM telegram_links WHERE token = ?", (token,)
        ).fetchone()
        return dict(row) if row else None


def chat_for_token(token: str) -> Optional[str]:
    link = get_link(token)
    return link["chat_id"] if link else None


# ----------------------------------------------------------------------
# Alerts
# ----------------------------------------------------------------------
def create_alert(
    chat_id: str,
    ticker: str,
    type_: str,
    threshold: Optional[float] = None,
    target_signal: Optional[str] = None,
    note: Optional[str] = None,
) -> dict:
    _ensure()
    alert_id = uuid.uuid4().hex
    row = {
        "id": alert_id,
        "chat_id": chat_id,
        "ticker": ticker,
        "type": type_,
        "threshold": threshold,
        "target_signal": target_signal,
        "status": "active",
        "note": note,
        "created_at": time.time(),
        "last_checked": None,
        "triggered_at": None,
        "last_value": None,
    }
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO alerts
                (id, chat_id, ticker, type, threshold, target_signal,
                 status, note, created_at, last_checked, triggered_at, last_value)
            VALUES
                (:id, :chat_id, :ticker, :type, :threshold, :target_signal,
                 :status, :note, :created_at, :last_checked, :triggered_at, :last_value)
            """,
            row,
        )
    return row


def list_alerts(chat_id: Optional[str] = None) -> list[dict]:
    _ensure()
    with _connect() as conn:
        if chat_id:
            cur = conn.execute(
                "SELECT * FROM alerts WHERE chat_id = ? ORDER BY created_at DESC", (chat_id,)
            )
        else:
            cur = conn.execute("SELECT * FROM alerts ORDER BY created_at DESC")
        return [dict(r) for r in cur.fetchall()]


def list_active_alerts() -> list[dict]:
    _ensure()
    with _connect() as conn:
        cur = conn.execute("SELECT * FROM alerts WHERE status = 'active'")
        return [dict(r) for r in cur.fetchall()]


def get_alert(alert_id: str) -> Optional[dict]:
    _ensure()
    with _connect() as conn:
        cur = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def set_status(alert_id: str, status: str) -> bool:
    """Update status. When re-arming to 'active', clear the previous trigger stamp."""
    _ensure()
    with _connect() as conn:
        if status == "active":
            cur = conn.execute(
                "UPDATE alerts SET status = ?, triggered_at = NULL WHERE id = ?",
                (status, alert_id),
            )
        else:
            cur = conn.execute(
                "UPDATE alerts SET status = ? WHERE id = ?", (status, alert_id)
            )
        return cur.rowcount > 0


def mark_triggered(alert_id: str, last_value: str) -> None:
    _ensure()
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "UPDATE alerts SET status = 'triggered', triggered_at = ?, "
            "last_checked = ?, last_value = ? WHERE id = ?",
            (now, now, last_value, alert_id),
        )


def mark_checked(alert_id: str, last_value: Optional[str] = None) -> None:
    _ensure()
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "UPDATE alerts SET last_checked = ?, last_value = ? WHERE id = ?",
            (now, last_value, alert_id),
        )


def delete_alert(alert_id: str) -> bool:
    _ensure()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
        return cur.rowcount > 0
