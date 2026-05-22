"""Telegram alert delivery."""
import logging

import requests

log = logging.getLogger("notifier")


class TelegramNotifier:
    def __init__(self, token, chat_id, timeout=15):
        self.token = token
        self.chat_id = chat_id
        self.timeout = timeout
        self.enabled = bool(token and chat_id)
        if not self.enabled:
            log.warning(
                "Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID "
                "missing); alerts will only be logged."
            )

    def send(self, text):
        """Send an HTML message. Returns True on success."""
        if not self.enabled:
            log.info("ALERT (Telegram disabled):\n%s", text)
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
                        "parse_mode": "HTML",
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
