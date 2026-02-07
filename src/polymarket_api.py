"""Polymarket data client (Gamma + CLOB)."""

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

    def fetch_active_events(self, limit: int = 10) -> list[dict[str, Any]]:
        """Fetch active events from the Gamma API."""
        url = f"{self.public_base_url}/events"
        params = {"active": "true", "closed": "false", "limit": limit}
        payload = self._get(url, params=params, allow_fallback=False)
        if payload is None:
            return []
        if isinstance(payload, list):
            return payload
        logger.warning("Unexpected events response format: %s", type(payload))
        return []

    def fetch_token_price(self, token_id: str) -> dict[str, Any]:
        """Fetch latest token price from the CLOB API."""
        url = f"{self.clob_base_url}/price"
        params = {"token_id": token_id, "side": "buy", "t": int(time.time())}
        payload = self._get(url, params=params, allow_fallback=False)
        if isinstance(payload, dict):
            return payload
        return {}

    def fetch_orderbook(self, token_id: str) -> dict[str, Any]:
        """Fetch orderbook depth from the CLOB API."""
        url = f"{self.clob_base_url}/book"
        params = {"token_id": token_id, "t": int(time.time())}
        payload = self._get(url, params=params, allow_fallback=False)
        if isinstance(payload, dict):
            return payload
        return {}

    def fetch_latest_trades(self, limit: int = DEFAULT_TRADE_LIMIT) -> list[dict[str, Any]]:
        """Build synthetic trade activity from events, prices, and orderbooks."""
        logger.info("Fetching latest activity from Gamma events + CLOB price/book.")
        events = self.fetch_active_events(limit=10)
        if not events:
            logger.warning("No active events returned from Gamma API.")
            return []

        activity: list[dict[str, Any]] = []
        for event in events:
            title = event.get("title") or event.get("question") or "Unknown"
            markets = event.get("markets") or []
            for market in markets:
                token_ids = market.get("clobTokenIds") or []
                for token_id in token_ids:
                    price_payload = self.fetch_token_price(str(token_id))
                    price = price_payload.get("price")
                    book = self.fetch_orderbook(str(token_id))
                    bids = book.get("bids") or []
                    asks = book.get("asks") or []
                    if not bids and not asks:
                        continue
                    max_bid = max(
                        bids,
                        key=lambda entry: float(entry.get("size", 0)),
                        default=None,
                    )
                    max_ask = max(
                        asks,
                        key=lambda entry: float(entry.get("size", 0)),
                        default=None,
                    )
                    for side_label, entry in (("BUY", max_bid), ("SELL", max_ask)):
                        if not entry:
                            continue
                        try:
                            size = float(entry.get("size", 0))
                        except (TypeError, ValueError):
                            size = 0.0
                        if size <= 0:
                            continue
                        entry_price = entry.get("price") or price
                        activity.append(
                            {
                                "marketTitle": title,
                                "price": entry_price,
                                "side": side_label,
                                "amount": size,
                                "token_id": str(token_id),
                                "event_id": event.get("id"),
                            }
                        )
                    if len(activity) >= limit:
                        return activity[:limit]
        return activity[:limit]

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
        """Fetch market details for a given market/condition ID."""
        _ = max_pages
        url = f"{self.base_url}{MARKETS_ENDPOINT}"
        params: dict[str, Any] = {"limit": limit, "id": market_id}
        payload = self._get(url, params=params)
        if payload is None:
            return []
        if isinstance(payload, dict) and "markets" in payload:
            return payload["markets"]
        if isinstance(payload, list):
            return payload
        logger.warning("Unexpected market history response format: %s", type(payload))
        return []
