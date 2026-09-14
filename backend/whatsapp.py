"""
WhatsApp delivery via the Twilio Messages REST API.

Starts on the free Twilio WhatsApp *sandbox*: the recipient joins once by sending
`join <code>` to the sandbox number, after which the backend can message them.
Going to a production sender + approved template later is just a change of the
three env secrets below (and the body must then match an approved template).

Configuration (Fly secrets):
    TWILIO_ACCOUNT_SID   e.g. ACxxxxxxxx...
    TWILIO_AUTH_TOKEN    the account auth token
    TWILIO_WHATSAPP_FROM e.g. whatsapp:+14155238886   (sandbox number)

If unconfigured, send_whatsapp() logs a warning and returns False instead of
raising — so local dev and unit tests run without Twilio credentials.
"""

import os
import logging

import requests

logger = logging.getLogger(__name__)

_API_BASE = "https://api.twilio.com/2010-04-01/Accounts"


def is_configured() -> bool:
    return bool(
        os.environ.get("TWILIO_ACCOUNT_SID")
        and os.environ.get("TWILIO_AUTH_TOKEN")
        and os.environ.get("TWILIO_WHATSAPP_FROM")
    )


def _to_whatsapp(number: str) -> str:
    """Normalise an E.164 number into Twilio's `whatsapp:+…` address form."""
    number = number.strip()
    if number.startswith("whatsapp:"):
        return number
    return f"whatsapp:{number}"


def send_whatsapp(to_e164: str, body: str) -> bool:
    """
    Send a WhatsApp message. Returns True on a 2xx from Twilio, False otherwise.
    Never raises — callers treat delivery as best-effort.
    """
    sid = os.environ.get("TWILIO_ACCOUNT_SID")
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_ = os.environ.get("TWILIO_WHATSAPP_FROM")

    if not (sid and token and from_):
        logger.warning("WhatsApp not configured — skipping message to %s", to_e164)
        return False

    url = f"{_API_BASE}/{sid}/Messages.json"
    data = {
        "From": _to_whatsapp(from_),
        "To": _to_whatsapp(to_e164),
        "Body": body,
    }
    try:
        resp = requests.post(url, data=data, auth=(sid, token), timeout=15)
        if resp.status_code // 100 == 2:
            logger.info("WhatsApp sent to %s", to_e164)
            return True
        logger.error(
            "WhatsApp send failed (%s) to %s: %s", resp.status_code, to_e164, resp.text[:300]
        )
        return False
    except Exception as exc:  # network / timeout
        logger.error("WhatsApp send error to %s: %s", to_e164, exc)
        return False
