"""Polymarket data client using public Gamma + CLOB endpoints."""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events?active=true&closed=false&limit=20"
CLOB_PRICE_URL = "https://clob.polymarket.com/price?token_id={token_id}&side=buy"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}


class PolyClient:
    """Minimal client for fetching public data from Polymarket."""

    def __init__(self) -> None:
        self.session = requests.Session()

    def fetch_markets(self) -> list[dict[str, Any]]:
        """Fetch market data using Gamma events + CLOB price."""
        valid_markets: list[dict[str, Any]] = []
        resp = self.session.get(GAMMA_EVENTS_URL, headers=REQUEST_HEADERS)
        events = resp.json()
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


PolymarketAPI = PolyClient
