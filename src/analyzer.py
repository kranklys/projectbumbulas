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
