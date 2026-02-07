"""Utilities for tracking recent trades and detecting momentum signals."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any


class HistoryManager:
    """Track recent trades and detect momentum in a rolling window."""

    def __init__(self, retention_seconds: int = 2 * 60 * 60) -> None:
        self.retention_seconds = retention_seconds
        self._trades: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def add_trade(
        self,
        market_id: str,
        trade_id: str,
        score: int,
        timestamp: float | None = None,
    ) -> None:
        """Store a trade entry for later momentum analysis."""
        event_time = timestamp if timestamp is not None else time.time()
        self._trades[market_id].append(
            {
                "trade_id": trade_id,
                "score": score,
                "timestamp": event_time,
            }
        )
        self.prune(event_time=event_time)

    def detect_momentum(
        self,
        market_id: str,
        min_score: int,
        window_seconds: int = 60 * 60,
        min_trades: int = 3,
    ) -> str | None:
        """Return MOMENTUM_ALERT when enough high-score trades cluster in a window."""
        now = time.time()
        cutoff = now - window_seconds
        recent = [
            trade
            for trade in self._trades.get(market_id, [])
            if trade.get("timestamp", 0) >= cutoff
            and trade.get("score", 0) >= min_score
        ]
        if len(recent) >= min_trades:
            return "MOMENTUM_ALERT"
        return None

    def prune(self, event_time: float | None = None) -> None:
        """Remove trades older than the retention window."""
        now = event_time if event_time is not None else time.time()
        cutoff = now - self.retention_seconds
        for market_id, trades in list(self._trades.items()):
            filtered = [trade for trade in trades if trade.get("timestamp", 0) >= cutoff]
            if filtered:
                self._trades[market_id] = filtered
            else:
                self._trades.pop(market_id, None)
