"""Polymarket CLOB API client."""

from __future__ import annotations

import logging
from typing import Any

import time

import requests

from src.config import (
    BASE_API_URL,
    DEFAULT_TIMEOUT_S,
    DEFAULT_TRADE_LIMIT,
    MARKETS_ENDPOINT,
    RATE_LIMIT_BACKOFF_S,
    TRADES_ENDPOINT,
)

logger = logging.getLogger(__name__)


class PolyClient:
    """Minimal client for fetching public data from Polymarket CLOB."""

    def __init__(self, base_url: str | None = None, timeout_s: int | None = None) -> None:
        self.base_url = base_url or BASE_API_URL
        self.timeout_s = timeout_s or DEFAULT_TIMEOUT_S
        self.session = requests.Session()

    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict | list | None:
        try:
            response = self.session.get(url, params=params, timeout=self.timeout_s)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                sleep_for = RATE_LIMIT_BACKOFF_S
                if retry_after:
                    try:
                        sleep_for = int(retry_after)
                    except ValueError:
                        logger.warning("Invalid Retry-After header: %s", retry_after)
                logger.warning("Rate limited (429). Backing off for %ss", sleep_for)
                time.sleep(sleep_for)
                return None
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            logger.exception("Request failed: %s", exc)
            return None
        except ValueError as exc:
            logger.exception("Failed to parse JSON response: %s", exc)
            return None

    def fetch_latest_trades(self, limit: int = DEFAULT_TRADE_LIMIT) -> list[dict[str, Any]]:
        """Fetch latest trades from the CLOB API."""
        url = f"{self.base_url}{TRADES_ENDPOINT}"
        params = {"limit": limit}

        logger.info("Fetching latest trades from %s", url)

        payload = self._get(url, params=params)
        if payload is None:
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

        payload = self._get(url)
        if payload is None:
            return {}

        if isinstance(payload, dict):
            return payload

        logger.warning("Unexpected market details format: %s", type(payload))
        return {}

    def fetch_markets(self, limit: int = 100) -> list[dict[str, Any]]:
        """Fetch market summaries for volume/liquidity analysis."""
        url = f"{self.base_url}{MARKETS_ENDPOINT}"
        params = {"limit": limit}

        logger.info("Fetching market summaries from %s", url)
        payload = self._get(url, params=params)
        if payload is None:
            return []

        if isinstance(payload, dict) and "markets" in payload:
            return payload["markets"]
        if isinstance(payload, list):
            return payload

        logger.warning("Unexpected markets response format: %s", type(payload))
        return []
