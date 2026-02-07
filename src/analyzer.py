"""Trade analysis utilities for Polymarket Smart Money tracker."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

WATCHLIST_ADDRESSES = {
    "0x1111111111111111111111111111111111111111",
    "0x2222222222222222222222222222222222222222",
    "0x3333333333333333333333333333333333333333",
}


class TradeAnalyzer:
    """Analyze trades for whale activity and watchlisted wallets."""

    def __init__(self, trades: list[dict[str, Any]]) -> None:
        self.trades = trades

    def find_whale_trades(
        self, trades: list[dict[str, Any]] | None = None, min_usd: float = 10000
    ) -> list[dict[str, Any]]:
        """Return trades with total value above min_usd."""
        trades_to_check = trades if trades is not None else self.trades
        whale_trades = []
        for trade in trades_to_check:
            total_value = self._estimate_trade_value_usd(trade)
            if total_value >= min_usd:
                trade["estimated_usd_value"] = round(total_value, 2)
                whale_trades.append(trade)
        return whale_trades

    def is_watchlist_trade(self, trade: dict[str, Any]) -> bool:
        """Check if the trade was placed by a watchlisted address."""
        address = (
            trade.get("trader")
            or trade.get("taker")
            or trade.get("maker")
            or trade.get("wallet")
        )
        if not address:
            return False
        return str(address).lower() in {addr.lower() for addr in WATCHLIST_ADDRESSES}

    def is_high_impact(self, trade: dict[str, Any], threshold: float = 0.02) -> bool:
        """Detect if a single trade moved price by more than threshold."""
        previous_price = trade.get("previousPrice")
        current_price = trade.get("price")
        if previous_price is None or current_price is None:
            return False
        try:
            previous_price = float(previous_price)
            current_price = float(current_price)
        except (TypeError, ValueError):
            return False
        if previous_price == 0:
            return False
        change = abs(current_price - previous_price) / previous_price
        return change >= threshold

    def _estimate_trade_value_usd(self, trade: dict[str, Any]) -> float:
        """Estimate trade value in USD from common fields."""
        for field in ("notional", "amount", "usdValue", "value"):
            if field in trade:
                try:
                    return float(trade[field])
                except (TypeError, ValueError):
                    continue

        price = trade.get("price")
        size = trade.get("size") or trade.get("quantity") or trade.get("shares")
        if price is not None and size is not None:
            try:
                return float(price) * float(size)
            except (TypeError, ValueError):
                logger.debug("Failed to parse price/size for trade value: %s", trade)

        return 0.0


def compute_signal_score(
    trade: dict[str, Any],
    market_details: dict[str, Any] | None,
    reputation_count: int,
    impact: float | None,
) -> tuple[int, list[str]]:
    """Compute a 0-100 signal score with reasons."""
    reasons: list[str] = []
    score = 0

    value = trade.get("estimated_usd_value")
    if value is not None:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = None

    if value is not None:
        if value >= 50000:
            score += 25
            reasons.append("Very large trade size")
        elif value >= 10000:
            score += 15
            reasons.append("Large trade size")
        elif value >= 5000:
            score += 10
            reasons.append("Notable trade size")

    if impact is not None:
        if impact >= 0.05:
            score += 20
            reasons.append("High price impact")
        elif impact >= 0.02:
            score += 10
            reasons.append("Moderate price impact")

    if reputation_count >= 10:
        score += 20
        reasons.append("Elite trader frequency")
    elif reputation_count >= 5:
        score += 10
        reasons.append("Frequent trader activity")

    if market_details:
        volume = _extract_volume_24h(market_details)
        if volume is not None:
            if volume >= 100000:
                score += 15
                reasons.append("High market liquidity")
            elif volume >= 20000:
                score += 10
                reasons.append("Moderate market liquidity")
            elif volume < 5000:
                score -= 5
                reasons.append("Low market liquidity")

    score = max(0, min(100, score))
    return score, reasons


def track_market_volumes(
    markets: list[dict[str, Any]],
    previous_volumes: dict[str, float],
    min_volume: float = 1000,
    spike_threshold: float = 0.10,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Detect volume spikes based on 24h volume changes."""
    spikes = []
    updated_volumes: dict[str, float] = {}

    for market in markets:
        market_id = str(market.get("conditionId") or market.get("id") or "")
        if not market_id:
            continue
        volume = _extract_volume_24h(market)
        if volume is None:
            continue
        updated_volumes[market_id] = volume
        previous = previous_volumes.get(market_id)
        if previous is None or previous <= 0:
            continue
        change = (volume - previous) / previous
        if volume >= min_volume and change >= spike_threshold:
            spikes.append(
                {
                    "market_id": market_id,
                    "volume": volume,
                    "previous_volume": previous,
                    "change_pct": change * 100,
                    "title": market.get("question")
                    or market.get("title")
                    or market.get("name")
                    or "Unknown",
                    "url": market.get("url") or market.get("marketUrl"),
                }
            )

    return spikes, updated_volumes


def _extract_volume_24h(market: dict[str, Any]) -> float | None:
    for key in ("volume24h", "volume", "volumeUSD", "volume_usd"):
        if key in market:
            try:
                return float(market[key])
            except (TypeError, ValueError):
                return None
    return None
