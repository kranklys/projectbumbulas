"""Polymarket CLOB API client."""

from __future__ import annotations

import logging
import os
from typing import Any

import time

import requests
from dotenv import load_dotenv

from src.config import (
    BASE_API_URL,
    CLOB_API_URL,
    DEFAULT_TIMEOUT_S,
    DEFAULT_TRADE_LIMIT,
    MARKETS_ENDPOINT,
    PUBLIC_API_URL,
    RATE_LIMIT_BACKOFF_S,
    TRADES_ENDPOINT,
)

logger = logging.getLogger(__name__)

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

RETRY_WAIT_S = 15


class PolyClient:
    """Minimal client for fetching public data from Polymarket CLOB."""

    def __init__(self, base_url: str | None = None, timeout_s: int | None = None) -> None:
        load_dotenv()
        self.base_url = base_url or BASE_API_URL
        self.public_base_url = PUBLIC_API_URL
        self.clob_base_url = CLOB_API_URL
        self.timeout_s = timeout_s or DEFAULT_TIMEOUT_S
        self.session = requests.Session()
        self.api_key = os.getenv("POLYMARKET_API_KEY")
        self.api_secret = os.getenv("POLYMARKET_API_SECRET")
        self.api_passphrase = os.getenv("POLYMARKET_PASSPHRASE")
        self.has_credentials = all(
            [self.api_key, self.api_secret, self.api_passphrase]
        )

    def _get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        allow_fallback: bool = True,
    ) -> dict | list | None:
        if not self.has_credentials:
            logger.info("Please fill in your credentials in the .env file.")
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout_s,
                headers=REQUEST_HEADERS,
            )
            if response.status_code == 401:
                logger.error("API Key Required: received 401 Unauthorized from %s", url)
                if allow_fallback and self.public_base_url:
                    fallback_url = url.replace(self.clob_base_url, self.public_base_url)
                    if fallback_url != url:
                        return self._get(fallback_url, params=params, allow_fallback=False)
                logger.warning("Status %s from %s. Retrying in %ss.", response.status_code, url, RETRY_WAIT_S)
                time.sleep(RETRY_WAIT_S)
                return None
            if response.status_code == 403:
                logger.warning("Status %s from %s. Retrying in %ss.", response.status_code, url, RETRY_WAIT_S)
                time.sleep(RETRY_WAIT_S)
                return None
            if response.status_code == 404:
                logger.warning("Status %s from %s. Retrying in %ss.", response.status_code, url, RETRY_WAIT_S)
                time.sleep(RETRY_WAIT_S)
                return None
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
            logger.warning("Primary trades endpoint failed. Falling back to public markets.")
            return self.fetch_public_markets(limit=10)

        if isinstance(payload, dict) and "trades" in payload:
            return payload["trades"]
        if isinstance(payload, list):
            return payload

        logger.warning("Unexpected trades response format: %s", type(payload))
        return self.fetch_public_markets(limit=10)

    def fetch_public_markets(self, limit: int = 10) -> list[dict[str, Any]]:
        """Fallback: fetch public markets when trades are unavailable."""
        url = f"{self.public_base_url}{MARKETS_ENDPOINT}"
        params = {"active": "true", "limit": limit}
        payload = self._get(url, params=params, allow_fallback=False)
        if payload is None:
            return []
        if isinstance(payload, dict) and "markets" in payload:
            return payload["markets"]
        if isinstance(payload, list):
            return payload
        logger.warning("Unexpected public markets response format: %s", type(payload))
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

    def fetch_historical_data(
        self,
        market_id: str,
        limit: int = 500,
        max_pages: int = 10,
    ) -> list[dict[str, Any]]:
        """Fetch historical trades for a given market/condition ID."""
        url = f"{self.base_url}{TRADES_ENDPOINT}"
        params: dict[str, Any] = {"limit": limit}
        params["marketId"] = market_id
        params["conditionId"] = market_id

        trades: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(max_pages):
            if cursor:
                params["cursor"] = cursor
            payload = self._get(url, params=params)
            if payload is None:
                break
            if isinstance(payload, dict):
                batch = payload.get("trades", [])
                cursor = payload.get("next_cursor") or payload.get("nextCursor")
            elif isinstance(payload, list):
                batch = payload
                cursor = None
            else:
                logger.warning("Unexpected historical trades response format: %s", type(payload))
                break

            if not isinstance(batch, list) or not batch:
                break
            trades.extend([trade for trade in batch if isinstance(trade, dict)])
            if not cursor:
                break

        logger.info("Fetched %s historical trades for market %s", len(trades), market_id)
        return trades
