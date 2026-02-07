"""Configuration values for Polymarket CLOB API access."""

from __future__ import annotations

import os


BASE_API_URL = os.getenv("POLYMARKET_CLOB_API", "https://clob.polymarket.com")
TRADES_ENDPOINT = "/trades"
DEFAULT_TIMEOUT_S = 10
DEFAULT_TRADE_LIMIT = 50
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
