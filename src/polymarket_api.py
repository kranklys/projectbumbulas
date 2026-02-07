"""Polymarket data client using public Gamma + CLOB endpoints."""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
GAMMA_EVENTS_URL = f"{GAMMA_BASE_URL}/events?active=true&closed=false&limit=20"
GAMMA_MARKETS_URL = f"{GAMMA_BASE_URL}/markets"
CLOB_BASE_URL = "https://clob.polymarket.com"
CLOB_PRICE_URL = f"{CLOB_BASE_URL}/price?token_id={{token_id}}&side=buy"
CLOB_BOOK_URL = f"{CLOB_BASE_URL}/book?token_id={{token_id}}"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}


class PolyClient:
    """Minimal client for fetching public data from Polymarket."""

    def __init__(self) -> None:
        self.session = requests.Session()

    def _get_json(self, url: str, params: dict[str, Any] | None = None) -> Any | None:
        try:
            response = self.session.get(url, params=params, headers=REQUEST_HEADERS)
            if response.status_code != 200:
                return None
            return response.json()
        except requests.RequestException:
            return None
        except ValueError:
            return None

    def fetch_markets(self) -> list[dict[str, Any]]:
        """Fetch market data using Gamma events + CLOB price."""
        valid_markets: list[dict[str, Any]] = []
        events = self._get_json(GAMMA_EVENTS_URL) or []
        for event in events:
            for market in event.get("markets", []):
                raw_tokens = market.get("clobTokenIds", [])
                if isinstance(raw_tokens, str):
                    token_ids = json.loads(raw_tokens)
                else:
                    token_ids = raw_tokens
                if not token_ids:
                    continue
                token_id = token_ids[0]
                try:
                    price_url = CLOB_PRICE_URL.format(token_id=token_id)
                    price_resp = self.session.get(price_url, headers=REQUEST_HEADERS)
                    if price_resp.status_code in {404, 429, 500}:
                        continue
                    price_data = price_resp.json()
                    valid_markets.append(
                        {
                            "title": event.get("title"),
                            "price": float(price_data.get("price")),
                            "id": token_id,
                        }
                    )
                except Exception:  # noqa: BLE001
                    continue
        if valid_markets:
            logger.info("✅ PolyClient Rebuilt for Bumbulas")
        return valid_markets

    def fetch_active_events(self, limit: int = 10) -> list[dict[str, Any]]:
        """Fetch active events from Gamma."""
        params = {"active": "true", "closed": "false", "limit": limit}
        payload = self._get_json(f"{GAMMA_BASE_URL}/events", params=params)
        if isinstance(payload, list):
            return payload
        return []

    def fetch_token_price(self, token_id: str) -> dict[str, Any]:
        """Fetch latest token price from the CLOB API."""
        url = f"{CLOB_BASE_URL}/price"
        params = {"token_id": token_id, "side": "buy"}
        payload = self._get_json(url, params=params)
        if isinstance(payload, dict):
            return payload
        return {}

    def fetch_orderbook(self, token_id: str) -> dict[str, Any]:
        """Fetch orderbook depth from the CLOB API."""
        url = f"{CLOB_BASE_URL}/book"
        params = {"token_id": token_id}
        payload = self._get_json(url, params=params)
        if isinstance(payload, dict):
            return payload
        return {}

    def get_market_data(self) -> list[dict[str, Any]]:
        """Return analyzer-ready market data from Gamma + CLOB."""
        return self.fetch_markets()

    def fetch_latest_trades(self) -> list[dict[str, Any]]:
        """Return active markets formatted as trade objects."""
        markets = self.fetch_markets()
        trades: list[dict[str, Any]] = []
        for market in markets:
            trades.append(
                {
                    "id": market.get("id"),
                    "marketTitle": market.get("title"),
                    "price": market.get("price"),
                    "side": "BUY",
                    "size": 0,
                }
            )
        return trades

    def get_market_details(self, condition_id: str) -> dict[str, Any]:
        """Fetch market details by condition ID."""
        payload = self._get_json(f"{GAMMA_MARKETS_URL}/{condition_id}")
        if isinstance(payload, dict):
            return payload
        return {}

    def fetch_historical_data(self, market_id: str, limit: int = 500) -> list[dict[str, Any]]:
        """Fetch market details for a given market/condition ID."""
        params = {"limit": limit, "id": market_id}
        payload = self._get_json(GAMMA_MARKETS_URL, params=params)
        if isinstance(payload, dict) and "markets" in payload:
            return payload["markets"]
        if isinstance(payload, list):
            return payload
        return []


PolymarketAPI = PolyClient
