"""Polymarket CLOB API client."""

from __future__ import annotations

import logging
from typing import Any

import requests

from src.config import (
    BASE_API_URL,
    DEFAULT_TIMEOUT_S,
    DEFAULT_TRADE_LIMIT,
    TRADES_ENDPOINT,
)

logger = logging.getLogger(__name__)


class PolyClient:
    """Minimal client for fetching public data from Polymarket CLOB."""

    def __init__(self, base_url: str | None = None, timeout_s: int | None = None) -> None:
        self.base_url = base_url or BASE_API_URL
        self.timeout_s = timeout_s or DEFAULT_TIMEOUT_S

    def fetch_latest_trades(self, limit: int = DEFAULT_TRADE_LIMIT) -> list[dict[str, Any]]:
        """Fetch latest trades from the CLOB API."""
        url = f"{self.base_url}{TRADES_ENDPOINT}"
        params = {"limit": limit}

        logger.info("Fetching latest trades from %s", url)

        try:
            response = requests.get(url, params=params, timeout=self.timeout_s)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            logger.exception("Failed to fetch latest trades: %s", exc)
            return []
        except ValueError as exc:
            logger.exception("Failed to parse trades response: %s", exc)
            return []

        if isinstance(payload, dict) and "trades" in payload:
            return payload["trades"]
        if isinstance(payload, list):
            return payload

        logger.warning("Unexpected trades response format: %s", type(payload))
        return []

    def get_market_details(self, condition_id: str) -> dict[str, Any]:
        """Fetch market details by condition ID."""
        url = f"{self.base_url}/markets/{condition_id}"

        logger.info("Fetching market details from %s", url)

        try:
            response = requests.get(url, timeout=self.timeout_s)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            logger.exception("Failed to fetch market details: %s", exc)
            return {}
        except ValueError as exc:
            logger.exception("Failed to parse market details: %s", exc)
            return {}

        if isinstance(payload, dict):
            return payload

        logger.warning("Unexpected market details format: %s", type(payload))
        return {}
