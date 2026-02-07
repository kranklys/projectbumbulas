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
            return {"processed_trade_ids": [], "trader_stats": {}}

        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.exception("Failed to load state: %s", exc)
            return {"processed_trade_ids": [], "trader_stats": {}}

        data.setdefault("processed_trade_ids", [])
        data.setdefault("trader_stats", {})
        return data

    def save(self, state: dict[str, Any]) -> None:
        try:
            with self.path.open("w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2, sort_keys=True)
        except OSError as exc:
            logger.exception("Failed to save state: %s", exc)
