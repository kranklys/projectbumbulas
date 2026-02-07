"""Simple JSON storage for bot state."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BotStateStorage:
    """Persist processed trade IDs and trader stats to JSON."""

    def __init__(self, path: str = "data/bot_state.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "processed_trade_ids": [],
                "trader_stats": {},
                "tracked_positions": [],
                "market_volumes": {},
                "recent_smart_trades": [],
            }

        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.exception("Failed to load state: %s", exc)
            return {
                "processed_trade_ids": [],
                "trader_stats": {},
                "tracked_positions": [],
                "market_volumes": {},
                "recent_smart_trades": [],
            }

        data.setdefault("processed_trade_ids", [])
        data.setdefault("trader_stats", {})
        data.setdefault("tracked_positions", [])
        data.setdefault("market_volumes", {})
        data.setdefault("recent_smart_trades", [])
        return data

    def cleanup(self, state: dict[str, Any], max_trades: int = 1000) -> None:
        processed = state.get("processed_trade_ids", [])
        if isinstance(processed, list) and len(processed) > max_trades:
            state["processed_trade_ids"] = processed[-max_trades:]
        recent = state.get("recent_smart_trades", [])
        if isinstance(recent, list) and len(recent) > max_trades:
            state["recent_smart_trades"] = recent[-max_trades:]

    def save(self, state: dict[str, Any]) -> None:
        try:
            with self.path.open("w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2, sort_keys=True)
        except OSError as exc:
            logger.exception("Failed to save state: %s", exc)
