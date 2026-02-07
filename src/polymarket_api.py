"""Polymarket data client (Gamma + CLOB)."""

from __future__ import annotations

import logging
from typing import Any

import time

import requests

from src.config import (
    BASE_API_URL,
    CLOB_API_URL,
    DEFAULT_TIMEOUT_S,
    DEFAULT_TRADE_LIMIT,
    MARKETS_ENDPOINT,
    PUBLIC_API_URL,
)

logger = logging.getLogger(__name__)

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

class PolyClient:
    """Minimal client for fetching public data from Polymarket CLOB."""

    def __init__(self, base_url: str | None = None, timeout_s: int | None = None) -> None:
        self.base_url = base_url or BASE_API_URL
        self.public_base_url = PUBLIC_API_URL
        self.clob_base_url = CLOB_API_URL
        self.timeout_s = timeout_s or DEFAULT_TIMEOUT_S
        self.session = requests.Session()
        self.last_events_count = 0

    def _get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        allow_fallback: bool = True,
    ) -> dict | list | None:
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout_s,
                headers=REQUEST_HEADERS,
            )
            if response.status_code in {404, 429}:
                logger.warning(
                    "Status %s from %s. Backing off for %ss.",
                    response.status_code,
                    url,
                    5,
                )
                time.sleep(5)
                return None
            if response.status_code >= 400:
                logger.warning("Status %s from %s. Skipping.", response.status_code, url)
                return None
            return response.json()
        except requests.RequestException as exc:
            logger.warning("Request failed: %s", exc)
            return None
        except ValueError as exc:
            logger.warning("Failed to parse JSON response: %s", exc)
            return None

    def fetch_active_events(self, limit: int = 15) -> list[dict[str, Any]]:
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
        """Build synthetic trade activity from Gamma events and CLOB price/book."""
        logger.info("Fetching latest activity from Gamma events + CLOB price/book.")
        events = self.fetch_active_events(limit=15)
        self.last_events_count = len(events)
        if not events:
            logger.warning("No active events returned from Gamma API.")
            return []

        activity: list[dict[str, Any]] = []
        for event in events:
            markets = event.get("markets") or []
            for market in markets:
                title = (
                    market.get("question")
                    or market.get("title")
                    or event.get("title")
                    or "Unknown"
                )
                token_ids = market.get("clobTokenIds") or []
                for token_id in token_ids:
                    token_id = str(token_id)
                    price_payload = self.fetch_token_price(token_id)
                    price = price_payload.get("price")
                    if price is None:
                        continue
                    book = self.fetch_orderbook(token_id)
                    bids = book.get("bids") or []
                    asks = book.get("asks") or []
                    if not bids and not asks:
                        continue
                    for side_label, entries in (("BUY", bids), ("SELL", asks)):
                        for entry in entries:
                            try:
                                size = float(entry.get("size", 0))
                                entry_price = float(entry.get("price", price))
                            except (TypeError, ValueError):
                                continue
                            if size <= 0:
                                continue
                            notional = size * entry_price
                            activity.append(
                                {
                                    "id": entry.get("order_id")
                                    or entry.get("id")
                                    or f"{token_id}:{side_label}:{entry_price}:{size}",
                                    "marketTitle": title,
                                    "price": entry_price,
                                    "side": side_label,
                                    "size": size,
                                    "notional": notional,
                                    "token_id": token_id,
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
