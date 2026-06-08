"""Telegram alert delivery — DISABLED permanently."""
import logging

log = logging.getLogger("notifier")


class TelegramNotifier:
    """Telegram alerts permanently disabled. All methods are no-ops."""

    def __init__(self, token=None, chat_id=None, timeout=15):
        self.enabled = False
        log.info("Telegram alerts disabled.")

    def send(self, text):
        return False
