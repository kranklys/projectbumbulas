"""Notification helpers for Polymarket Smart Money tracker."""

from __future__ import annotations

import logging

import requests

from src.config import TELEGRAM_CHAT_ID, TELEGRAM_TOKEN

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send messages to Telegram via Bot API."""

    def __init__(self, token: str | None = None, chat_id: str | None = None) -> None:
        self.token = token or TELEGRAM_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID

    def is_configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def send_message(self, message: str) -> bool:
        if not self.is_configured():
            logger.warning("Telegram notifier not configured.")
            return False

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message}

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.exception("Failed to send Telegram message: %s", exc)
            return False

        return True
