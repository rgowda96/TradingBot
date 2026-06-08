"""Telegram alert delivery (optional — console alerts work without it)."""
import logging
import time

import requests

log = logging.getLogger("notifier")

_MAX_ATTEMPTS = 4


class TelegramNotifier:
    def __init__(self, token, chat_id, timeout=15):
        self.token   = token
        self.chat_id = chat_id
        self.timeout = timeout
        self.enabled = bool(token and chat_id)
        if self.enabled:
            log.info("Telegram alerts enabled.")
        else:
            log.info("Telegram not configured; running in console-only mode.")

    def send(self, text):
        """Send a plain-text message to Telegram. Returns True on success."""
        if not self.enabled:
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = requests.post(
                    url,
                    timeout=self.timeout,
                    data={
                        "chat_id":                  self.chat_id,
                        "text":                     text,
                        "disable_web_page_preview": "true",
                    },
                )
                if resp.status_code == 429:
                    # Telegram tells us exactly how long to wait.
                    retry_after = int(resp.headers.get("Retry-After", 0)) or 2 ** (attempt + 1)
                    log.warning("Telegram rate limited — waiting %ds", retry_after)
                    time.sleep(retry_after)
                    continue
                resp.raise_for_status()
                return True
            except requests.RequestException as exc:
                if attempt == _MAX_ATTEMPTS - 1:
                    log.error("failed to send Telegram alert: %s", exc)
                    return False
                wait = 2 ** attempt
                log.debug("Telegram send failed (attempt %d), retrying in %ds: %s",
                          attempt + 1, wait, exc)
                time.sleep(wait)
        return False
