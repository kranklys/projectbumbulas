"""Backtesting utilities for signal validation and paper trading."""

from __future__ import annotations

import json
import logging
import os
import time
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any

from src.analyzer import WATCHLIST_ADDRESSES, compute_signal_score

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    score_threshold: int = 70
    horizon_minutes: int = 60
    max_trades: int | None = None


def load_trades(path: str, max_trades: int | None = None) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trades file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        trades = data.get("trades", [])
    else:
        trades = data
    if not isinstance(trades, list):
        raise ValueError("Trades JSON must be a list or contain a 'trades' list.")
    trades = [trade for trade in trades if isinstance(trade, dict)]
    if max_trades is not None:
        trades = trades[:max_trades]
    return trades


def extract_trade_side(trade: dict[str, Any]) -> str | None:
    for key in ("side", "outcome", "direction", "position"):
        if key in trade:
            return str(trade[key]).upper()
    return None


def normalize_trade(trade: dict[str, Any], fallback_index: int) -> dict[str, Any]:
    normalized = dict(trade)
    normalized.setdefault(
        "id",
        trade.get("id")
        or trade.get("tradeId")
        or trade.get("hash")
        or f"backtest-{fallback_index}",
    )
    normalized["condition_id"] = trade.get("conditionId") or trade.get("condition_id")
    normalized["price"] = trade.get("price") or trade.get("pricePerShare") or trade.get(
        "avgPrice"
    )
    normalized["size"] = trade.get("size") or trade.get("quantity") or trade.get("shares")
    normalized["trader"] = (
        trade.get("trader")
        or trade.get("taker")
        or trade.get("maker")
        or trade.get("wallet")
    )
    normalized["side"] = extract_trade_side(trade)
    normalized["timestamp"] = (
        trade.get("timestamp")
        or trade.get("time")
        or trade.get("created_at")
        or trade.get("createdAt")
    )
    return normalized


def estimate_trade_value(trade: dict[str, Any]) -> float | None:
    for field in ("notional", "amount", "usdValue", "value"):
        if field in trade:
            try:
                return float(trade[field])
            except (TypeError, ValueError):
                continue
    price = trade.get("price")
    size = trade.get("size")
    if price is not None and size is not None:
        try:
            return float(price) * float(size)
        except (TypeError, ValueError):
            return None
    return None


def compute_impact(previous_price: float | None, current_price: float | None) -> float | None:
    if previous_price is None or current_price is None:
        return None
    try:
        previous_price = float(previous_price)
        current_price = float(current_price)
    except (TypeError, ValueError):
        return None
    if previous_price == 0:
        return None
    return abs(current_price - previous_price) / previous_price


def build_market_index(
    trades: list[dict[str, Any]],
) -> dict[str, dict[str, list[float]]]:
    market_index: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"timestamps": [], "prices": []}
    )
    for trade in trades:
        condition_id = trade.get("condition_id")
        timestamp = trade.get("timestamp")
        price = trade.get("price")
        if not condition_id or timestamp is None or price is None:
            continue
        try:
            timestamp = float(timestamp)
            price = float(price)
        except (TypeError, ValueError):
            continue
        index = market_index[str(condition_id)]
        index["timestamps"].append(timestamp)
        index["prices"].append(price)
    return market_index


def find_exit_price(
    market_index: dict[str, dict[str, list[float]]],
    condition_id: str,
    entry_time: float,
    horizon_seconds: float,
) -> float | None:
    if condition_id not in market_index:
        return None
    timestamps = market_index[condition_id]["timestamps"]
    prices = market_index[condition_id]["prices"]
    start_idx = bisect_right(timestamps, entry_time)
    end_idx = bisect_right(timestamps, entry_time + horizon_seconds)
    if end_idx <= start_idx:
        return None
    return prices[end_idx - 1]


def compute_roi(entry_price: float, exit_price: float, side: str | None) -> float:
    if entry_price == 0:
        return 0.0
    if side == "NO" or side == "SELL":
        return (entry_price - exit_price) / entry_price
    return (exit_price - entry_price) / entry_price


def calculate_drawdown(returns: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for roi in returns:
        equity *= 1 + roi
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak if peak else 0.0
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown


def run_backtest(trades_path: str, config: BacktestConfig) -> dict[str, Any]:
    raw_trades = load_trades(trades_path, config.max_trades)
    normalized: list[dict[str, Any]] = []
    for index, trade in enumerate(raw_trades):
        normalized.append(normalize_trade(trade, index))

    normalized = [
        trade
        for trade in normalized
        if trade.get("condition_id") and trade.get("price") and trade.get("timestamp")
    ]
    normalized.sort(key=lambda trade: float(trade.get("timestamp", 0)))

    market_index = build_market_index(normalized)
    trader_stats: dict[str, int] = defaultdict(int)
    last_price: dict[str, float] = {}

    total_signals = 0
    executed_signals = 0
    wins = 0
    returns: list[float] = []

    horizon_seconds = config.horizon_minutes * 60
    watchlist = {addr.lower() for addr in WATCHLIST_ADDRESSES}

    for trade in normalized:
        trader = trade.get("trader") or "unknown"
        trader_key = str(trader).lower()
        trader_stats[trader_key] += 1
        reputation_count = trader_stats[trader_key]

        condition_id = str(trade.get("condition_id"))
        current_price = trade.get("price")
        try:
            current_price = float(current_price)
        except (TypeError, ValueError):
            continue

        impact = compute_impact(last_price.get(condition_id), current_price)
        last_price[condition_id] = current_price

        is_watchlisted = trader_key in watchlist
        trade["is_watchlisted"] = is_watchlisted
        trade["estimated_usd_value"] = estimate_trade_value(trade)

        score, _reasons = compute_signal_score(
            trade,
            market_details=None,
            reputation_count=reputation_count,
            impact=impact,
            repeat_offender=is_watchlisted,
        )
        if score < config.score_threshold:
            continue

        total_signals += 1
        entry_time = float(trade.get("timestamp"))
        exit_price = find_exit_price(
            market_index, condition_id, entry_time, horizon_seconds
        )
        if exit_price is None:
            continue

        executed_signals += 1
        roi = compute_roi(current_price, exit_price, trade.get("side"))
        returns.append(roi)
        if roi > 0:
            wins += 1

    hit_rate = (wins / executed_signals) if executed_signals else 0.0
    avg_roi = mean(returns) if returns else 0.0
    max_drawdown = calculate_drawdown(returns)
    total_return = 1.0
    for roi in returns:
        total_return *= 1 + roi
    total_return -= 1

    report = {
        "trades_loaded": len(raw_trades),
        "signals_generated": total_signals,
        "signals_executed": executed_signals,
        "hit_rate": hit_rate,
        "avg_roi": avg_roi,
        "max_drawdown": max_drawdown,
        "total_return": total_return,
        "score_threshold": config.score_threshold,
        "horizon_minutes": config.horizon_minutes,
    }

    output_dir = os.path.join("data", "backtests")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    output_path = os.path.join(output_dir, f"backtest_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    logger.info("Backtest complete. Report saved to %s", output_path)
    logger.info(
        "Signals: %s executed, hit rate: %.2f%%, avg ROI: %.2f%%, max drawdown: %.2f%%",
        executed_signals,
        hit_rate * 100,
        avg_roi * 100,
        max_drawdown * 100,
    )
    return report

