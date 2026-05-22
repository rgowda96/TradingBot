"""Telegram alert delivery (optional — console alerts work without it)."""
import logging

import requests

log = logging.getLogger("notifier")


class TelegramNotifier:
    def __init__(self, token, chat_id, timeout=15):
        self.token = token
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
        for attempt in range(4):
            try:
                resp = requests.post(
                    url,
                    timeout=self.timeout,
                    data={
                        "chat_id": self.chat_id,
                        "text": text,
                        "disable_web_page_preview": "true",
                    },
                )
                resp.raise_for_status()
                return True
            except requests.RequestException as exc:
                if attempt == 3:
                    log.error("failed to send Telegram alert: %s", exc)
                    return False
        return False
