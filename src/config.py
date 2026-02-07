"""Configuration values for Polymarket CLOB API access."""

from __future__ import annotations

import os


PUBLIC_API_URL = os.getenv("POLYMARKET_PUBLIC_API", "https://gamma-api.polymarket.com")
CLOB_API_URL = os.getenv("POLYMARKET_CLOB_API", "https://clob.polymarket.com")
BASE_API_URL = os.getenv("POLYMARKET_API", PUBLIC_API_URL)
TRADES_ENDPOINT = "/trades"
MARKETS_ENDPOINT = "/markets"
DEFAULT_TIMEOUT_S = 10
DEFAULT_TRADE_LIMIT = 50
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
POLL_INTERVAL_S = int(os.getenv("POLL_INTERVAL_S", "30"))
SAVE_INTERVAL_CYCLES = int(os.getenv("SAVE_INTERVAL_CYCLES", "5"))
RATE_LIMIT_BACKOFF_S = int(os.getenv("RATE_LIMIT_BACKOFF_S", "30"))
MIN_MARKET_VOLUME = float(os.getenv("MIN_MARKET_VOLUME", "1000"))
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "5"))
