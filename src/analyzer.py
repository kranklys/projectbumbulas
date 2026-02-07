"""Trade analysis utilities for Polymarket Smart Money tracker."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

WATCHLIST_ADDRESSES = {
    "0x1111111111111111111111111111111111111111",
    "0x2222222222222222222222222222222222222222",
    "0x3333333333333333333333333333333333333333",
}
WHALE_TRADE_THRESHOLD = 50


class TradeAnalyzer:
    """Analyze trades for whale activity and watchlisted wallets."""

    def __init__(
        self,
        trades: list[dict[str, Any]],
        watchlist: set[str] | None = None,
    ) -> None:
        self.trades = trades
        self.watchlist = {addr.lower() for addr in (watchlist or WATCHLIST_ADDRESSES)}
        self.market_tracker: dict[str, list[float]] = {}

    @staticmethod
    def _extract_trade_id(trade: dict[str, Any], fallback_index: int) -> str:
        return str(
            trade.get("id")
            or trade.get("tradeId")
            or trade.get("trade_id")
            or trade.get("hash")
            or trade.get("transactionHash")
            or f"fallback-{fallback_index}"
        )

    def filter_new_trades(
        self, seen_trade_ids: set[str]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Filter trades to only those not seen before, updating seen_trade_ids."""
        new_trades: list[dict[str, Any]] = []
        new_trade_ids: list[str] = []
        for index, trade in enumerate(self.trades):
            trade_id = self._extract_trade_id(trade, index)
            trade.setdefault("id", trade_id)
            if trade_id in seen_trade_ids:
                continue
            seen_trade_ids.add(trade_id)
            new_trade_ids.append(trade_id)
            new_trades.append(trade)
        return new_trades, new_trade_ids

    def find_whale_trades(
        self,
        trades: list[dict[str, Any]] | None = None,
        min_usd: float = WHALE_TRADE_THRESHOLD,
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

    def estimate_trade_value_usd(self, trade: dict[str, Any]) -> float:
        """Expose trade value estimation for callers needing consistent sizing."""
        return self._estimate_trade_value_usd(trade)

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
        return str(address).lower() in self.watchlist

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

    def record_whale_trade(self, market_id: str, timestamp: float | None = None) -> int:
        """Track whale trades per market and return count within last hour."""
        event_time = timestamp if timestamp is not None else time.time()
        window_start = event_time - 60 * 60
        history = self.market_tracker.get(market_id, [])
        updated = [t for t in history if t >= window_start]
        updated.append(event_time)
        self.market_tracker[market_id] = updated
        return len(updated)


def compute_signal_score(
    trade: dict[str, Any],
    market_details: dict[str, Any] | None,
    reputation_count: int,
    impact: float | None,
    trader_profile: dict[str, Any] | None = None,
    market_trend: str | None = None,
    market_sentiment: str | None = None,
    repeat_offender: bool = False,
    repeat_offender_bonus: int = 10,
    market_tracker: dict[str, list[float]] | None = None,
    market_id: str | None = None,
    reason_weights: dict[str, float] | None = None,
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

    def weight_for(key: str) -> float:
        if reason_weights is None:
            return 1.0
        return max(0.5, min(1.5, float(reason_weights.get(key, 1.0))))

    if value is not None:
        if value >= 50000:
            score += int(25 * weight_for("trade_size_very_large"))
            reasons.append("Very large trade size")
        elif value >= 10000:
            score += int(15 * weight_for("trade_size_large"))
            reasons.append("Large trade size")
        elif value >= 5000:
            score += int(10 * weight_for("trade_size_notable"))
            reasons.append("Notable trade size")

    if impact is not None:
        if impact >= 0.05:
            score += int(20 * weight_for("impact_high"))
            reasons.append("High price impact")
        elif impact >= 0.02:
            score += int(10 * weight_for("impact_moderate"))
            reasons.append("Moderate price impact")

    if reputation_count >= 10:
        score += int(20 * weight_for("reputation_elite"))
        reasons.append("Elite trader frequency")
    elif reputation_count >= 5:
        score += int(10 * weight_for("reputation_frequent"))
        reasons.append("Frequent trader activity")

    if trade.get("is_watchlisted"):
        score += int(50 * weight_for("watchlist_priority"))
        reasons.append("Watchlist whale priority")

    if trader_profile:
        resolved_count = trader_profile.get("resolved_count", 0)
        winrate = trader_profile.get("winrate")
        avg_profit = trader_profile.get("avg_profit")
        profit_volatility = trader_profile.get("profit_volatility")
        elite = trader_profile.get("elite", False)
        if elite:
            score += int(15 * weight_for("profile_elite"))
            reasons.append("Elite smart wallet profile")
        if resolved_count >= 5 and isinstance(winrate, (int, float)):
            if winrate >= 0.65:
                score += int(10 * weight_for("profile_winrate_strong"))
                reasons.append("Strong win rate history")
            elif winrate >= 0.55:
                score += int(5 * weight_for("profile_winrate_positive"))
                reasons.append("Positive win rate history")
        if isinstance(avg_profit, (int, float)) and avg_profit > 0:
            score += int(5 * weight_for("profile_avg_profit"))
            reasons.append("Positive average outcome")
        if isinstance(profit_volatility, (int, float)) and profit_volatility < 0.05:
            score += int(3 * weight_for("profile_consistent"))
            reasons.append("Consistent outcomes")

    if market_trend == "Up":
        score += int(5 * weight_for("trend_up"))
        reasons.append("Short-term market trend up")
    elif market_trend == "Down":
        score -= int(5 * weight_for("trend_down"))
        reasons.append("Short-term market trend down")

    if market_sentiment == "Bullish":
        score += int(3 * weight_for("sentiment_bullish"))
        reasons.append("Bullish smart wallet sentiment")
    elif market_sentiment == "Bearish":
        score -= int(3 * weight_for("sentiment_bearish"))
        reasons.append("Bearish smart wallet sentiment")

    if repeat_offender:
        score += int(repeat_offender_bonus * weight_for("repeat_offender"))
        reasons.append("Repeat offender watchlist activity")

    if market_tracker is not None and market_id:
        window_start = time.time() - 60 * 60
        recent = [t for t in market_tracker.get(market_id, []) if t >= window_start]
        if len(recent) >= 2:
            score += int(30 * weight_for("market_momentum"))
            reasons.append("Crowd momentum in market")

    if market_details:
        volume = _extract_volume_24h(market_details)
        if volume is not None:
            if volume >= 100000:
                score += int(15 * weight_for("liquidity_high"))
                reasons.append("High market liquidity")
            elif volume >= 20000:
                score += int(10 * weight_for("liquidity_medium"))
                reasons.append("Moderate market liquidity")
            elif volume < 5000:
                score -= int(5 * weight_for("liquidity_low"))
                reasons.append("Low market liquidity")

    score = max(0, min(100, score))
    return score, reasons


def log_high_score_alert(
    trade: dict[str, Any],
    score: int,
    reasons: list[str],
    market_title: str | None = None,
    market_url: str | None = None,
    file_path: str = "alerts.log",
) -> None:
    """Append high-score alert details to a log file."""
    payload = {
        "timestamp": time.time(),
        "trade_id": trade.get("id"),
        "score": score,
        "reasons": reasons,
        "market_title": market_title or "Unknown",
        "market_url": market_url or trade.get("marketUrl") or trade.get("url"),
        "trader": trade.get("trader")
        or trade.get("taker")
        or trade.get("maker")
        or trade.get("wallet"),
        "estimated_usd_value": trade.get("estimated_usd_value"),
    }
    try:
        with open(file_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
    except OSError:
        logger.warning("Failed to write high-score alert to %s", file_path)


def evaluate_signal_density(
    entries: list[dict[str, Any]],
    condition_id: str | None,
    window_seconds: int = 3600,
    min_entries: int = 3,
) -> tuple[str, int]:
    """Classify signal density based on smart wallet entries in a time window."""
    if not condition_id:
        return "Standard", 0
    now = time.time()
    recent_entries = [
        entry
        for entry in entries
        if entry.get("condition_id") == condition_id
        and now - entry.get("timestamp", 0) <= window_seconds
    ]
    count = len(recent_entries)
    if count >= min_entries:
        return "CONFIRMED MOMENTUM", count
    return "Standard", count


def assess_market_risk(
    market_details: dict[str, Any] | None,
    min_volume: float,
    max_spread_pct: float,
) -> tuple[str, list[str]]:
    """Classify market risk based on volume and spread."""
    if not market_details:
        return "Unknown", ["Market details unavailable"]

    reasons: list[str] = []
    volume = _extract_volume_24h(market_details)
    if volume is not None:
        if volume < min_volume:
            reasons.append("Low volume")
        elif volume >= min_volume * 5:
            reasons.append("Healthy volume")

    spread_pct = _estimate_spread_pct(market_details)
    if spread_pct is not None:
        if spread_pct > max_spread_pct:
            reasons.append("Wide spread")
        elif spread_pct <= max_spread_pct / 2:
            reasons.append("Tight spread")

    if "Low volume" in reasons or "Wide spread" in reasons:
        return "High", reasons
    if "Healthy volume" in reasons or "Tight spread" in reasons:
        return "Low", reasons
    return "Medium", reasons


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


def _estimate_spread_pct(market: dict[str, Any]) -> float | None:
    best_bid = market.get("bestBid")
    best_ask = market.get("bestAsk")
    if best_bid is None or best_ask is None:
        return None
    try:
        best_bid = float(best_bid)
        best_ask = float(best_ask)
    except (TypeError, ValueError):
        return None
    if best_ask <= 0:
        return None
    return ((best_ask - best_bid) / best_ask) * 100
