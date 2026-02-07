"""Main loop for Polymarket Smart Money tracker."""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from collections import Counter, deque
from datetime import datetime, timezone

from src.analyzer import (
    TradeAnalyzer,
    WATCHLIST_ADDRESSES,
    assess_market_risk,
    compute_signal_score,
    evaluate_signal_density,
    log_high_score_alert,
    track_market_volumes,
)
from src.notifications import TelegramNotifier
from src.polymarket_api import PolyClient
from src.history_manager import HistoryManager
from src.storage import BotStateStorage
from src.config import (
    MAX_SPREAD_PCT,
    MIN_MARKET_VOLUME,
    POLL_INTERVAL_S,
    SAVE_INTERVAL_CYCLES,
)
from src.backtest import (
    BacktestConfig,
    evaluate_thresholds,
    fetch_trades_for_backtest,
    run_backtest,
    run_backtest_from_trades,
)
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import matplotlib
import matplotlib.pyplot as plt


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

matplotlib.use("Agg")

TRADER_HISTORY_PATH = os.path.join("data", "trader_history.json")
DISCOVERED_WHALES_PATH = os.path.join("data", "discovered_whales.json")
SIGNAL_HISTORY_PATH = os.path.join("data", "signal_history.json")
REPORTS_DIR = os.path.join("data", "reports")
REPEAT_OFFENDER_BONUS = 10
SMART_WALLET_WINDOW_S = 60 * 60
SMART_WALLET_THRESHOLD = 3
MOMENTUM_MIN_SCORE = 85
MOMENTUM_MIN_TRADES = 3
CLEANUP_RETENTION_S = 24 * 60 * 60
FETCH_RETRY_ATTEMPTS = 3
FETCH_RETRY_BACKOFF_S = 2
MARKET_TREND_WINDOW_S = 15 * 60
DISCOVERY_WINDOW_S = 24 * 60 * 60
DISCOVERY_SCORE_THRESHOLD = 70
DISCOVERY_MIN_TRADES = 4
SIGNAL_HISTORY_RETENTION_DAYS = 14
VIRTUAL_TRADES_PATH = os.path.join("data", "virtual_trades.json")
VIRTUAL_TRADE_MIN_AGE_S = 60 * 60
VIRTUAL_TRADE_CHECK_INTERVAL_S = 30 * 60
PERFORMANCE_REPORT_INTERVAL_S = 60 * 60

REASON_KEY_MAP = {
    "Very large trade size": "trade_size_very_large",
    "Large trade size": "trade_size_large",
    "Notable trade size": "trade_size_notable",
    "High price impact": "impact_high",
    "Moderate price impact": "impact_moderate",
    "Elite trader frequency": "reputation_elite",
    "Frequent trader activity": "reputation_frequent",
    "Watchlist whale priority": "watchlist_priority",
    "Elite smart wallet profile": "profile_elite",
    "Strong win rate history": "profile_winrate_strong",
    "Positive win rate history": "profile_winrate_positive",
    "Positive average outcome": "profile_avg_profit",
    "Consistent outcomes": "profile_consistent",
    "Short-term market trend up": "trend_up",
    "Short-term market trend down": "trend_down",
    "Bullish smart wallet sentiment": "sentiment_bullish",
    "Bearish smart wallet sentiment": "sentiment_bearish",
    "Repeat offender watchlist activity": "repeat_offender",
    "Crowd momentum in market": "market_momentum",
    "High market liquidity": "liquidity_high",
    "Moderate market liquidity": "liquidity_medium",
    "Low market liquidity": "liquidity_low",
}

COLOR_MOMENTUM = "\033[95m"
COLOR_WHALE = "\033[94m"
COLOR_RESET = "\033[0m"


def format_trade_message(
    trade: dict,
    market_title: str | None,
    impact: float | None = None,
    reputation: str | None = None,
    critical: bool = False,
    sentiment: str | None = None,
    signal_score: int | None = None,
    reasons: list[str] | None = None,
    risk_label: str | None = None,
    risk_reasons: list[str] | None = None,
    signal_type: str | None = None,
    signal_density: str | None = None,
    priority_multiplier: int = 1,
    market_trend: str | None = None,
    market_regime: str | None = None,
) -> str:
    value = trade.get("estimated_usd_value")
    wallet_pnl = (
        trade.get("pnl")
        or trade.get("profitLoss")
        or trade.get("estimatedPnl")
        or trade.get("estimated_profit")
    )
    trader = (
        trade.get("trader")
        or trade.get("taker")
        or trade.get("maker")
        or trade.get("wallet")
        or "Unknown"
    )
    link = trade.get("marketUrl") or trade.get("url") or "N/A"
    price = trade.get("price") or "N/A"
    title = market_title or "Unknown"
    reputation_label = reputation or "New"
    header = "🔴 CRITICAL ALERT" if critical else "🚨 Smart Trade Detected!"
    sentiment_label = sentiment or "Neutral"
    score_label = f"{signal_score}/100" if signal_score is not None else "N/A"
    reason_text = ", ".join(reasons or []) or "N/A"
    risk_text = risk_label or "Unknown"
    risk_reason_text = ", ".join(risk_reasons or []) or "N/A"
    signal_type_label = signal_type or "Standard"
    density_label = signal_density or "N/A"
    priority_label = f"x{priority_multiplier}"
    trend_label = market_trend or "Unknown"
    regime_label = market_regime or "Normal"

    return (
        f"{header}\n\n"
        f"🎯 Market: {title}\n\n"
        f"💰 Amount: ${value if value is not None else 'N/A'}\n\n"
        f"🧾 Wallet P/L: {wallet_pnl if wallet_pnl is not None else 'N/A'}\n\n"
        f"📈 Price: {price}\n\n"
        f"🧠 Reputation: {reputation_label}\n\n"
        f"🌡️ Sentiment: {sentiment_label}\n\n"
        f"📊 Trend: {trend_label} | Regime: {regime_label}\n\n"
        f"✅ Signal Score: {score_label}\n"
        f"🧩 Reasons: {reason_text}\n\n"
        f"⚠️ Risk: {risk_text}\n"
        f"🧾 Risk Factors: {risk_reason_text}\n\n"
        f"🏷️ Signal Type: {signal_type_label}\n"
        f"📊 Signal Density: {density_label}\n\n"
        f"🚦 Priority: {priority_label}\n\n"
        f"👤 Trader: {trader}\n\n"
        f"🔗 Link: {link}"
    )


def estimate_price_impact(trade: dict) -> float | None:
    previous_price = trade.get("previousPrice")
    current_price = trade.get("price")
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


def _format_trade_feed_entry(
    trade: dict,
    amount: float,
    side: str | None,
    reasoning: list[str] | None,
) -> str:
    market_name = _extract_market_name(trade)
    price = trade.get("price") or "N/A"
    side_label = side or "N/A"
    if side_label.upper() == "BUY":
        side_label = "[green]BUY[/green]"
    elif side_label.upper() == "SELL":
        side_label = "[red]SELL[/red]"
    reasoning_text = "; ".join(reasoning or [])
    insight = f" | Insight: {reasoning_text}" if reasoning_text else ""
    return (
        f"[TRADE] Market: {market_name} | Side: {side_label} | "
        f"Amount: ${amount:,.2f} | Price: {price}{insight}"
    )


def _extract_market_name(trade: dict) -> str:
    return (
        trade.get("marketTitle")
        or trade.get("market_title")
        or trade.get("title")
        or trade.get("question")
        or trade.get("market")
        or "Unknown"
    )


def _build_reasoning(
    trade: dict,
    amount: float,
    side: str | None,
    market_trade_counts: dict[str, int],
) -> list[str]:
    reasons = []
    market_name = _extract_market_name(trade)
    if market_trade_counts.get(market_name, 0) > 1:
        reasons.append(f"Accumulation detected in {market_name}")
    if side and side.upper() == "BUY" and amount >= 1000:
        reasons.append("Strong Bullish Signal")
    impact = estimate_price_impact(trade)
    if impact is not None and impact >= 0.02:
        reasons.append("Momentum Alert")
    if not reasons:
        reasons.append("Notable activity")
    return reasons


def _select_top_opportunity(
    flows: list[dict],
    max_entries: int = 100,
) -> tuple[str | None, float]:
    if not flows:
        return None, 0.0
    recent_entries = flows[-max_entries:]
    totals: dict[str, float] = {}
    for entry in recent_entries:
        market = entry.get("market") or "Unknown"
        totals[market] = totals.get(market, 0.0) + float(entry.get("amount", 0.0))
    top_market = max(totals.items(), key=lambda item: item[1])
    return top_market[0], top_market[1]


def _is_reputable_trade(
    trade: dict,
    analyzer: TradeAnalyzer,
    trader_stats: dict,
    trader_profiles: dict,
) -> bool:
    trader = (
        trade.get("trader")
        or trade.get("taker")
        or trade.get("maker")
        or trade.get("wallet")
    )
    trader_key = str(trader).lower() if trader else ""
    if analyzer.is_watchlist_trade(trade):
        return True
    profile = trader_profiles.get(trader_key, {})
    if profile.get("elite"):
        return True
    return trader_stats.get(trader_key, 0) >= 5


def get_current_market_price(market_details: dict) -> float | None:
    for key in ("lastTradePrice", "price", "last_price"):
        if key in market_details:
            try:
                return float(market_details[key])
            except (TypeError, ValueError):
                continue
    best_bid = market_details.get("bestBid")
    best_ask = market_details.get("bestAsk")
    if best_bid is not None and best_ask is not None:
        try:
            return (float(best_bid) + float(best_ask)) / 2
        except (TypeError, ValueError):
            return None
    return None


def get_trade_id(trade: dict, fallback_index: int) -> str:
    return str(
        trade.get("id")
        or trade.get("tradeId")
        or trade.get("hash")
        or trade.get("transactionHash")
        or f"fallback-{fallback_index}"
    )


def normalize_trade(trade: dict, fallback_index: int) -> dict:
    """Normalize trade fields to a consistent schema."""
    normalized = dict(trade)
    normalized.setdefault("id", get_trade_id(trade, fallback_index))
    trader = (
        trade.get("trader")
        or trade.get("taker")
        or trade.get("maker")
        or trade.get("wallet")
    )
    if trader is not None:
        normalized.setdefault("trader", trader)

    condition_id = trade.get("conditionId") or trade.get("condition_id")
    if condition_id is not None:
        normalized.setdefault("condition_id", condition_id)

    if "price" not in normalized:
        price = trade.get("price") or trade.get("pricePerShare") or trade.get("avgPrice")
        if price is not None:
            normalized["price"] = price

    if "size" not in normalized:
        size = (
            trade.get("size")
            or trade.get("quantity")
            or trade.get("shares")
            or trade.get("amount")
        )
        if size is not None:
            normalized["size"] = size

    if "marketUrl" not in normalized:
        url = trade.get("marketUrl") or trade.get("url")
        if url is not None:
            normalized["marketUrl"] = url

    side = extract_trade_side(trade)
    if side is not None:
        normalized.setdefault("side", side)

    return normalized


def get_reputation_label(count: int, profile: dict | None = None) -> str:
    if profile and profile.get("elite"):
        return "Elite Smart Wallet"
    if count >= 10:
        return "Elite Trader"
    if count >= 5:
        return "Frequent"
    return "New"


def update_trader_profile_metrics(profile: dict) -> None:
    resolved = profile.get("resolved_count", 0)
    wins = profile.get("wins", 0)
    losses = profile.get("losses", 0)
    total_profit = profile.get("total_profit", 0.0)
    if resolved > 0:
        profile["avg_profit"] = total_profit / resolved
    total_outcomes = wins + losses
    if total_outcomes > 0:
        profile["winrate"] = wins / total_outcomes
    profit_history = profile.get("profit_history", [])
    if len(profit_history) >= 2:
        profile["profit_volatility"] = statistics.pstdev(profit_history)
    profile["elite"] = (
        resolved >= 5
        and profile.get("winrate", 0) >= 0.6
        and profile.get("avg_profit", 0) > 0
    )


def refresh_elite_smart_wallets(state: dict) -> None:
    profiles = state.get("trader_profiles", {})
    elites = [
        trader for trader, profile in profiles.items() if profile.get("elite", False)
    ]
    state["elite_smart_wallets"] = elites


def normalize_seen_trades(state: dict, now: float) -> list[dict]:
    seen_entries = state.get("seen_trades")
    if not seen_entries:
        legacy_ids = state.get("seen_trade_ids", [])
        return [
            {"trade_id": str(trade_id), "timestamp": now} for trade_id in legacy_ids
        ]
    normalized = []
    for entry in seen_entries:
        if isinstance(entry, dict) and "trade_id" in entry:
            normalized.append(
                {
                    "trade_id": str(entry.get("trade_id")),
                    "timestamp": float(entry.get("timestamp", now)),
                }
            )
        else:
            normalized.append({"trade_id": str(entry), "timestamp": now})
    return normalized


def cleanup_state_memory(state: dict, history_manager: HistoryManager) -> None:
    now = time.time()
    cutoff = now - CLEANUP_RETENTION_S
    state["seen_trades"] = [
        entry
        for entry in state.get("seen_trades", [])
        if entry.get("timestamp", 0) >= cutoff
    ]
    state["recent_reputable_flows"] = [
        entry
        for entry in state.get("recent_reputable_flows", [])
        if entry.get("timestamp", 0) >= cutoff
    ]
    high_score_history = state.get("high_score_history", {})
    for trader_key, entries in list(high_score_history.items()):
        filtered = [ts for ts in entries if ts >= now - DISCOVERY_WINDOW_S]
        if filtered:
            high_score_history[trader_key] = filtered
        else:
            high_score_history.pop(trader_key, None)
    state["high_score_history"] = high_score_history
    state.pop("seen_trade_ids", None)
    history_manager.prune(event_time=now)


def build_market_url(trade: dict, market_details: dict | None) -> str | None:
    return (
        trade.get("marketUrl")
        or trade.get("url")
        or (market_details.get("url") if market_details else None)
        or (market_details.get("marketUrl") if market_details else None)
    )


def load_discovered_whales(path: str = DISCOVERED_WHALES_PATH) -> set[str]:
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to load discovered whales from %s", path)
        return set()
    if isinstance(data, list):
        return {str(entry).lower() for entry in data}
    return set()


def save_discovered_whales(addresses: set[str], path: str = DISCOVERED_WHALES_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(sorted(addresses), handle, indent=2)


def update_discovery_watchlist(
    state: dict,
    trader_key: str,
    score: int,
    discovered_watchlist: set[str],
) -> bool:
    if score <= DISCOVERY_SCORE_THRESHOLD:
        return False
    now = time.time()
    history = state.get("high_score_history", {})
    timestamps = history.get(trader_key, [])
    timestamps.append(now)
    cutoff = now - DISCOVERY_WINDOW_S
    timestamps = [ts for ts in timestamps if ts >= cutoff]
    history[trader_key] = timestamps
    state["high_score_history"] = history
    if trader_key in discovered_watchlist:
        return False
    if len(timestamps) >= DISCOVERY_MIN_TRADES:
        discovered_watchlist.add(trader_key)
        save_discovered_whales(discovered_watchlist)
        logger.info("Added %s to discovered whales list.", trader_key)
        return True
    return False


class Dashboard:
    def __init__(self) -> None:
        self.console = Console()
        self.live = Live(
            self._render({}, {}),
            console=self.console,
            refresh_per_second=2,
            transient=False,
        )

    def start(self) -> None:
        self.live.start()

    def stop(self) -> None:
        self.live.stop()

    def update(self, stats: dict, state: dict) -> None:
        self.live.update(self._render(stats, state), refresh=True)

    def _render(self, stats: dict, state: dict) -> Group:
        table = Table(title="📊 Polymarket Smart Money Dashboard", expand=True)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Cycle", str(stats.get("cycle", "-")))
        total_trades = stats.get("total_trades_scanned", 0)
        new_trades = stats.get("new_trades", 0)
        table.add_row(
            "Trade Activity",
            f"Total Trades Scanned: {total_trades} | New Trades this Cycle: {new_trades}",
        )
        total_fetched = stats.get("total_trades_fetched", 0)
        table.add_row("Total Trades Fetched", str(total_fetched))
        table.add_row("Whale Trades", str(stats.get("whale_trades", 0)))
        table.add_row("Alerts Sent", str(stats.get("alerts_sent", 0)))
        table.add_row("Elite Wallets", str(len(state.get("elite_smart_wallets", []))))
        top_market = stats.get("top_opportunity_market") or "-"
        top_amount = stats.get("top_opportunity_amount")
        top_amount_text = f"${top_amount:,.2f}" if isinstance(top_amount, (int, float)) else "-"
        table.add_row("TOP OPPORTUNITY", f"{top_market} | {top_amount_text}")
        last_score = stats.get("last_score")
        table.add_row("Last Score", str(last_score) if last_score is not None else "-")
        table.add_row("Last Market", stats.get("last_market", "-"))
        table.add_row("Last URL", stats.get("last_url", "-"))
        table.add_row("Seen Trades (24h)", str(len(state.get("seen_trades", []))))
        feed_entries = stats.get("trade_feed") or []
        feed_text = Text.from_markup("\n".join(feed_entries)) if feed_entries else Text(
            "No trades fetched yet."
        )
        feed_panel = Panel(
            feed_text,
            title="Live Feed (Latest Trades)",
            border_style="cyan",
        )
        return Group(table, feed_panel)


def add_tracked_position(
    state: dict,
    trade_id: str,
    condition_id: str | None,
    entry_price: float | None,
    trader_address: str,
) -> None:
    if not condition_id or entry_price is None:
        return
    tracked_positions = state.get("tracked_positions", [])
    tracked_positions.append(
        {
            "trade_id": trade_id,
            "condition_id": condition_id,
            "entry_price": entry_price,
            "timestamp": time.time(),
            "trader_address": trader_address,
        }
    )
    state["tracked_positions"] = tracked_positions


def format_profit_message(
    market_title: str | None,
    trader: str,
    entry_price: float,
    current_price: float,
    change_pct: float,
) -> str:
    title = market_title or "Unknown"
    return (
        "✅ Profit Update!\n\n"
        f"🎯 Market: {title}\n\n"
        f"👤 Trader: {trader}\n\n"
        f"💰 Entry: {entry_price} → Current: {current_price}\n\n"
        f"📈 Change: {change_pct:+.2f}%"
    )


def format_volume_spike_message(spike: dict) -> str:
    return (
        "🚀 VOLUME SPIKE\n\n"
        f"🎯 Market: {spike.get('title', 'Unknown')}\n\n"
        f"💰 Volume: ${spike.get('volume', 0):,.0f}\n\n"
        f"📈 Change: {spike.get('change_pct', 0):.2f}%\n\n"
        f"🔗 Link: {spike.get('url') or 'N/A'}"
    )


def extract_trade_side(trade: dict) -> str | None:
    for key in ("side", "outcome", "direction", "position"):
        if key in trade:
            return str(trade[key]).upper()
    return None


def update_recent_smart_trades(state: dict, trade: dict) -> None:
    side = extract_trade_side(trade)
    if side not in {"YES", "NO"}:
        return
    recent = state.get("recent_smart_trades", [])
    recent.append({"timestamp": time.time(), "side": side})
    state["recent_smart_trades"] = recent


def update_recent_market_signals(state: dict, condition_id: str | None) -> None:
    if not condition_id:
        return
    recent = state.get("recent_market_signals", [])
    recent.append({"timestamp": time.time(), "condition_id": condition_id})
    state["recent_market_signals"] = recent


def update_recent_smart_wallet_entries(
    state: dict,
    condition_id: str | None,
    trader: str,
) -> None:
    if not condition_id:
        return
    recent = state.get("smart_wallet_entries", [])
    recent.append(
        {"timestamp": time.time(), "condition_id": condition_id, "trader": trader}
    )
    state["smart_wallet_entries"] = recent


def update_recent_volume_spikes(state: dict, market_id: str) -> None:
    recent = state.get("recent_volume_spikes", [])
    recent.append({"timestamp": time.time(), "market_id": market_id})
    state["recent_volume_spikes"] = recent


def get_sentiment(state: dict) -> str:
    recent = state.get("recent_smart_trades", [])
    cutoff = time.time() - 3600
    yes_count = 0
    no_count = 0
    filtered = []
    for entry in recent:
        timestamp = entry.get("timestamp", 0)
        if timestamp >= cutoff:
            filtered.append(entry)
            side = entry.get("side")
            if side == "YES":
                yes_count += 1
            elif side == "NO":
                no_count += 1
    state["recent_smart_trades"] = filtered
    if yes_count > no_count and yes_count > 0:
        return "Bullish"
    if no_count > yes_count and no_count > 0:
        return "Bearish"
    return "Neutral"


def update_market_price_history(
    state: dict, condition_id: str, price: float | None
) -> None:
    if price is None:
        return
    history = state.get("market_price_history", {})
    entries = history.get(condition_id, [])
    entries.append({"timestamp": time.time(), "price": price})
    cutoff = time.time() - MARKET_TREND_WINDOW_S
    history[condition_id] = [entry for entry in entries if entry["timestamp"] >= cutoff]
    state["market_price_history"] = history


def get_market_trend(state: dict, condition_id: str) -> str:
    history = state.get("market_price_history", {}).get(condition_id, [])
    if len(history) < 2:
        return "Flat"
    start = history[0]["price"]
    end = history[-1]["price"]
    if start == 0:
        return "Flat"
    change_pct = (end - start) / start
    if change_pct >= 0.01:
        return "Up"
    if change_pct <= -0.01:
        return "Down"
    return "Flat"


def classify_market_regime(risk_label: str, trend: str) -> str:
    if risk_label == "High":
        return "Defensive"
    if trend == "Up":
        return "Risk-On"
    if trend == "Down":
        return "Risk-Off"
    return "Neutral"


def get_signal_context(
    state: dict, condition_id: str | None
) -> tuple[str, str, int]:
    now = time.time()
    market_signals = state.get("recent_market_signals", [])
    volume_spikes = state.get("recent_volume_spikes", [])
    smart_wallet_entries = state.get("smart_wallet_entries", [])

    signals_15m = sum(1 for s in market_signals if now - s.get("timestamp", 0) <= 900)
    signals_60m = sum(1 for s in market_signals if now - s.get("timestamp", 0) <= 3600)
    density_label = f"15m:{signals_15m} | 60m:{signals_60m}"

    density_status, smart_wallet_count = evaluate_signal_density(
        smart_wallet_entries,
        condition_id,
        window_seconds=SMART_WALLET_WINDOW_S,
        min_entries=SMART_WALLET_THRESHOLD,
    )
    cutoff = now - SMART_WALLET_WINDOW_S
    state["smart_wallet_entries"] = [
        entry
        for entry in smart_wallet_entries
        if entry.get("timestamp", 0) >= cutoff
    ]

    if density_status == "CONFIRMED MOMENTUM":
        density_label = f"{density_label} | smart_wallets:{smart_wallet_count}"
        return "CONFIRMED MOMENTUM", density_label, 2

    if condition_id:
        recent_spike = any(
            s.get("market_id") == condition_id and now - s.get("timestamp", 0) <= 1800
            for s in volume_spikes
        )
        recent_market_signals = sum(
            1
            for s in market_signals
            if s.get("condition_id") == condition_id and now - s.get("timestamp", 0) <= 900
        )
        if recent_spike and recent_market_signals >= 2:
            return "Confirmed", density_label, 1
        if recent_market_signals >= 3:
            return "Momentum", density_label, 1

    return "Standard", density_label, 1


def load_trader_history(path: str = TRADER_HISTORY_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to load trader history from %s", path)
        return {}


def save_trader_history(history: dict, path: str = TRADER_HISTORY_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, sort_keys=True)


def load_signal_history(path: str = SIGNAL_HISTORY_PATH) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to load signal history from %s", path)
        return []
    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]
    return []


def save_signal_history(history: list[dict], path: str = SIGNAL_HISTORY_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, sort_keys=True)


def load_virtual_trades(path: str = VIRTUAL_TRADES_PATH) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to load virtual trades from %s", path)
        return []
    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]
    return []


def save_virtual_trades(trades: list[dict], path: str = VIRTUAL_TRADES_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(trades, handle, indent=2, sort_keys=True)


def update_reason_weights(state: dict, reasons: list[str], won: bool) -> None:
    weights = state.get("reason_weights", {})
    for reason in reasons:
        key = REASON_KEY_MAP.get(reason)
        if not key:
            continue
        current = float(weights.get(key, 1.0))
        if won:
            weights[key] = min(1.5, current + 0.02)
        else:
            weights[key] = max(0.5, current - 0.05)
    state["reason_weights"] = weights


def update_wallet_performance(state: dict, trader_key: str, profit: float, won: bool) -> None:
    performance = state.get("wallet_performance", {})
    entry = performance.get(trader_key, {"profit": 0.0, "wins": 0, "losses": 0})
    entry["profit"] = float(entry.get("profit", 0.0)) + float(profit)
    if won:
        entry["wins"] = int(entry.get("wins", 0)) + 1
    else:
        entry["losses"] = int(entry.get("losses", 0)) + 1
    performance[trader_key] = entry
    state["wallet_performance"] = performance


def verify_virtual_trades(
    client: PolyClient,
    state: dict,
    virtual_trades: list[dict],
) -> list[dict]:
    now = time.time()
    remaining: list[dict] = []
    for trade in virtual_trades:
        entry_time = float(trade.get("timestamp", 0))
        if now - entry_time < VIRTUAL_TRADE_MIN_AGE_S:
            remaining.append(trade)
            continue
        market_id = trade.get("market_id")
        entry_price = trade.get("entry_price")
        if market_id is None or entry_price is None:
            continue
        try:
            entry_price = float(entry_price)
        except (TypeError, ValueError):
            continue
        try:
            market_details = client.get_market_details(str(market_id))
            current_price = get_current_market_price(market_details)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Virtual trade check failed for %s: %s", market_id, exc)
            remaining.append(trade)
            continue
        if current_price is None:
            remaining.append(trade)
            continue
        side = trade.get("side")
        if side == "NO":
            roi = (entry_price - current_price) / entry_price
        else:
            roi = (current_price - entry_price) / entry_price
        won = roi > 0
        trader_key = str(trade.get("trader", "unknown")).lower()
        state.setdefault("wallet_reliability", {})
        reliability = state["wallet_reliability"].get(trader_key, 0.0)
        reliability += 1.0 if won else -1.0
        state["wallet_reliability"][trader_key] = reliability
        totals = state.get("performance_totals", {"predictions": 0, "wins": 0})
        totals["predictions"] = int(totals.get("predictions", 0)) + 1
        if won:
            totals["wins"] = int(totals.get("wins", 0)) + 1
        state["performance_totals"] = totals
        update_wallet_performance(state, trader_key, roi, won)
        update_reason_weights(state, trade.get("reasons", []), won)
    return remaining


def add_virtual_trade(
    virtual_trades: list[dict],
    trade: dict,
    market_id: str,
    entry_price: float,
    reasons: list[str],
    side: str | None,
) -> None:
    virtual_trades.append(
        {
            "timestamp": time.time(),
            "market_id": market_id,
            "entry_price": entry_price,
            "trader": trade.get("trader")
            or trade.get("taker")
            or trade.get("maker")
            or trade.get("wallet"),
            "reasons": reasons,
            "side": side,
        }
    )


def log_performance_report(state: dict, total_predictions: int, wins: int) -> None:
    accuracy = (wins / total_predictions) * 100 if total_predictions else 0.0
    performance = state.get("wallet_performance", {})
    top_wallets = sorted(
        performance.items(),
        key=lambda item: item[1].get("profit", 0.0),
        reverse=True,
    )[:3]
    top_summary = ", ".join(
        f"{wallet[:6]}… profit:{data.get('profit', 0.0):.2f}"
        for wallet, data in top_wallets
    ) or "N/A"
    logger.info(
        "Bot Performance Report | Total Predictions: %s | Accuracy: %.2f%% | Top Wallets: %s",
        total_predictions,
        accuracy,
        top_summary,
    )

def prune_signal_history(history: list[dict]) -> list[dict]:
    cutoff = time.time() - SIGNAL_HISTORY_RETENTION_DAYS * 24 * 60 * 60
    return [entry for entry in history if entry.get("timestamp", 0) >= cutoff]


def record_signal_history(
    history: list[dict],
    trade: dict,
    score: int,
    reasons: list[str],
    market_title: str | None,
    market_url: str | None,
    risk_label: str,
    signal_type: str,
    momentum_alert: str | None,
) -> None:
    history.append(
        {
            "timestamp": time.time(),
            "trade_id": trade.get("id"),
            "market_title": market_title or "Unknown",
            "market_url": market_url or trade.get("marketUrl") or trade.get("url"),
            "trader": trade.get("trader")
            or trade.get("taker")
            or trade.get("maker")
            or trade.get("wallet"),
            "score": score,
            "reasons": reasons,
            "risk": risk_label,
            "signal_type": signal_type,
            "momentum": momentum_alert is not None,
            "estimated_usd_value": trade.get("estimated_usd_value"),
        }
    )


def _format_currency(value: float | None) -> str:
    if value is None:
        return "N/A"
    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def _format_date_label(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")


def generate_signal_charts(
    recent_signals: list[dict],
    report_dir: str,
    report_label: str,
) -> dict[str, str]:
    os.makedirs(report_dir, exist_ok=True)
    chart_paths: dict[str, str] = {}
    if not recent_signals:
        return chart_paths

    timestamps = [entry.get("timestamp", 0) for entry in recent_signals]
    hours = [
        datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:00")
        for ts in timestamps
        if ts
    ]
    hour_counts = Counter(hours)
    hour_labels = sorted(hour_counts.keys())
    counts = [hour_counts[label] for label in hour_labels]

    plt.figure(figsize=(10, 4))
    plt.bar(hour_labels, counts, color="#4C78A8")
    plt.title("Signals per Hour (UTC)")
    plt.xlabel("Hour")
    plt.ylabel("Signals")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    signals_chart = os.path.join(report_dir, f"signals_per_hour_{report_label}.png")
    plt.savefig(signals_chart)
    plt.close()
    chart_paths["signals_per_hour"] = signals_chart

    top_markets = Counter(
        entry.get("market_title", "Unknown") for entry in recent_signals
    ).most_common(5)
    if top_markets:
        labels = [item[0] for item in top_markets]
        values = [item[1] for item in top_markets]
        plt.figure(figsize=(10, 4))
        plt.barh(labels, values, color="#F58518")
        plt.title("Top Markets by Signal Count")
        plt.xlabel("Signals")
        plt.tight_layout()
        markets_chart = os.path.join(report_dir, f"top_markets_{report_label}.png")
        plt.savefig(markets_chart)
        plt.close()
        chart_paths["top_markets"] = markets_chart

    return chart_paths


def generate_daily_report(
    signal_history: list[dict],
    state: dict,
    report_dir: str = REPORTS_DIR,
) -> dict[str, str]:
    now = time.time()
    report_label = _format_date_label(now)
    cutoff = now - 24 * 60 * 60
    recent_signals = [entry for entry in signal_history if entry.get("timestamp", 0) >= cutoff]

    top_signals = sorted(
        recent_signals, key=lambda entry: entry.get("score", 0), reverse=True
    )[:10]
    trader_counts = Counter(
        entry.get("trader", "Unknown") for entry in recent_signals if entry.get("trader")
    ).most_common(5)

    profiles = state.get("trader_profiles", {})
    top_profit = sorted(
        profiles.items(),
        key=lambda item: item[1].get("total_profit", 0),
        reverse=True,
    )[:5]

    charts = generate_signal_charts(recent_signals, report_dir, report_label)

    summary = {
        "date": report_label,
        "signals_count": len(recent_signals),
        "top_signals": top_signals,
        "top_traders": [
            {"trader": trader, "signals": count} for trader, count in trader_counts
        ],
        "top_profit": [
            {
                "trader": trader,
                "total_profit": data.get("total_profit", 0),
                "winrate": data.get("winrate"),
            }
            for trader, data in top_profit
        ],
        "charts": charts,
    }

    os.makedirs(report_dir, exist_ok=True)
    summary_path = os.path.join(report_dir, f"daily_summary_{report_label}.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    html_path = os.path.join(report_dir, f"daily_report_{report_label}.html")
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(
            "<html><head><meta charset='utf-8'>"
            "<title>Daily Bot Report</title>"
            "<style>body{font-family:Arial,sans-serif;padding:20px;} "
            "table{border-collapse:collapse;width:100%;margin-bottom:20px;} "
            "th,td{border:1px solid #ddd;padding:8px;} "
            "th{background:#f2f2f2;text-align:left;}</style>"
            "</head><body>"
            f"<h1>Daily Report - {report_label} (UTC)</h1>"
            f"<p>Total signals: {len(recent_signals)}</p>"
            "<h2>Top Signals</h2>"
            "<table><tr><th>Market</th><th>Score</th><th>Trader</th><th>Value</th><th>Link</th></tr>"
        )
        for entry in top_signals:
            handle.write(
                "<tr>"
                f"<td>{entry.get('market_title','Unknown')}</td>"
                f"<td>{entry.get('score','-')}</td>"
                f"<td>{entry.get('trader','Unknown')}</td>"
                f"<td>{_format_currency(entry.get('estimated_usd_value'))}</td>"
                f"<td><a href='{entry.get('market_url','')}'>{entry.get('market_url','')}</a></td>"
                "</tr>"
            )
        handle.write("</table>")
        handle.write("<h2>Top Traders (Signals)</h2>")
        handle.write("<table><tr><th>Trader</th><th>Signals</th></tr>")
        for trader, count in trader_counts:
            handle.write(f"<tr><td>{trader}</td><td>{count}</td></tr>")
        handle.write("</table>")
        handle.write("<h2>Top Profit Traders</h2>")
        handle.write("<table><tr><th>Trader</th><th>Total Profit</th><th>Winrate</th></tr>")
        for trader, data in top_profit:
            winrate = data.get("winrate")
            winrate_label = f"{winrate:.2%}" if isinstance(winrate, float) else "N/A"
            handle.write(
                "<tr>"
                f"<td>{trader}</td>"
                f"<td>{data.get('total_profit',0):.4f}</td>"
                f"<td>{winrate_label}</td>"
                "</tr>"
            )
        handle.write("</table>")
        if charts.get("signals_per_hour"):
            handle.write("<h2>Signal History Charts</h2>")
            for label, path in charts.items():
                filename = os.path.basename(path)
                handle.write(f"<div><img src='{filename}' style='max-width:100%;'></div>")
        handle.write("</body></html>")

    return {"summary": summary_path, "html": html_path}


def log_signal_event(
    signal_type: str,
    priority_multiplier: int,
    market_title: str | None,
    trader: str,
    trader_count: int,
    score: int,
    risk_label: str,
    reasons: list[str],
    momentum_alert: str | None = None,
) -> None:
    title = market_title or "Unknown"
    trader_label = (
        f"Frequent Trader (seen {trader_count}x)"
        if trader_count > 1
        else "New Whale"
    )
    if momentum_alert:
        header = f"{COLOR_MOMENTUM}🚀 MOMENTUM ALERT{COLOR_RESET}"
    elif signal_type == "CONFIRMED MOMENTUM":
        header = f"{COLOR_MOMENTUM}✅ CONFIRMED MOMENTUM{COLOR_RESET}"
    else:
        header = f"{COLOR_WHALE}🐳 NORMAL WHALE{COLOR_RESET}"
    log_method = logger.info
    if momentum_alert and score >= 90:
        log_method = logger.critical
    elif momentum_alert and score >= MOMENTUM_MIN_SCORE:
        log_method = logger.warning
    log_method(
        "\n%s\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🎯 Market: %s\n"
        "👤 Trader: %s | %s\n"
        "✅ Score: %s/100 | Priority: x%s\n"
        "⚠️ Risk: %s\n"
        "🧩 Reasons: %s\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        header,
        title,
        trader,
        trader_label,
        score,
        priority_multiplier,
        risk_label,
        ", ".join(reasons) or "N/A",
    )


def update_tracked_positions(
    client: PolyClient,
    notifier: TelegramNotifier,
    state: dict,
) -> None:
    markets = client.fetch_markets()
    previous_volumes = state.get("market_volumes", {})
    if markets:
        spikes, updated_volumes = track_market_volumes(markets, previous_volumes)
        state["market_volumes"] = updated_volumes
        for spike in spikes:
            notifier.send_message(format_volume_spike_message(spike))
            update_recent_volume_spikes(state, spike.get("market_id", ""))
    else:
        logger.warning("No market summaries returned for volume spike detection.")

    tracked_positions = state.get("tracked_positions", [])
    if not tracked_positions:
        return

    updated_positions = []
    now = time.time()
    trader_profiles = state.get("trader_profiles", {})

    for position in tracked_positions:
        condition_id = position.get("condition_id")
        entry_price = position.get("entry_price")
        trader = position.get("trader_address", "Unknown")
        timestamp = position.get("timestamp", 0)

        if not condition_id or entry_price is None:
            continue

        market_details = client.get_market_details(str(condition_id))
        if not market_details:
            logger.warning("Market details unavailable for condition_id=%s", condition_id)
            updated_positions.append(position)
            continue
        market_title = (
            market_details.get("question")
            or market_details.get("title")
            or market_details.get("name")
        )
        current_price = get_current_market_price(market_details)

        if current_price is not None:
            change_pct = ((current_price - entry_price) / entry_price) * 100
            if abs(change_pct) >= 5:
                message = format_profit_message(
                    market_title,
                    trader,
                    entry_price,
                    current_price,
                    change_pct,
                )
                notifier.send_message(message)

        is_resolved = (
            market_details.get("isResolved")
            or market_details.get("resolved")
            or str(market_details.get("status", "")).lower() == "resolved"
        )
        is_expired = now - float(timestamp) > 60 * 60 * 24

        if not is_resolved and not is_expired:
            updated_positions.append(position)
        elif is_resolved:
            profile = trader_profiles.get(str(trader).lower(), {})
            profile["resolved_count"] = profile.get("resolved_count", 0) + 1
            current_price = get_current_market_price(market_details)
            if current_price is not None:
                pnl = current_price - entry_price
                total_profit = profile.get("total_profit", 0.0) + pnl
                profile["total_profit"] = total_profit
                if pnl > 0:
                    profile["wins"] = profile.get("wins", 0) + 1
                else:
                    profile["losses"] = profile.get("losses", 0) + 1
                profit_history = profile.get("profit_history", [])
                profit_history.append(pnl)
                profile["profit_history"] = profit_history[-50:]
                update_trader_profile_metrics(profile)
            trader_profiles[str(trader).lower()] = profile

    state["tracked_positions"] = updated_positions
    state["trader_profiles"] = trader_profiles
    refresh_elite_smart_wallets(state)


def process_trades(
    trades: list[dict],
    notifier: TelegramNotifier,
    client: PolyClient,
    state: dict,
    trader_history: dict,
    history_manager: HistoryManager,
    discovered_watchlist: set[str],
    signal_history: list[dict],
    virtual_trades: list[dict],
) -> dict:
    stats: dict[str, object] = {
        "new_trades": 0,
        "whale_trades": 0,
        "alerts_sent": 0,
        "last_score": None,
        "last_market": None,
        "last_url": None,
        "trade_feed": [],
        "top_opportunity_market": None,
        "top_opportunity_amount": None,
        "fetched_trades": 0,
    }
    fetched_trades = []
    for index, trade in enumerate(trades):
        if not isinstance(trade, dict):
            logger.warning("Skipping non-dict trade payload at index %s: %s", index, trade)
            continue
        fetched_trades.append(normalize_trade(trade, index))
    stats["fetched_trades"] = len(fetched_trades)

    now = time.time()
    seen_trades = deque(normalize_seen_trades(state, now), maxlen=2000)
    seen_trade_id_set = {entry.get("trade_id") for entry in seen_trades}
    analyzer = TradeAnalyzer(
        fetched_trades, watchlist=set(WATCHLIST_ADDRESSES) | discovered_watchlist
    )
    new_trades, new_trade_ids = analyzer.filter_new_trades(seen_trade_id_set)
    stats["new_trades"] = len(new_trades)
    if not new_trades:
        logger.warning("No new trades to process after deduplication.")
    for trade_id in new_trade_ids:
        seen_trades.append({"trade_id": trade_id, "timestamp": now})

    trader_stats = state.get("trader_stats", {})
    trader_profiles = state.get("trader_profiles", {})

    market_trade_counts = Counter(_extract_market_name(trade) for trade in fetched_trades)
    feed_entries = []
    min_trade_usd = 50.0
    recent_flows = list(state.get("recent_reputable_flows", []))
    for trade in fetched_trades:
        amount = analyzer.estimate_trade_value_usd(trade)
        side = extract_trade_side(trade)
        market_name = _extract_market_name(trade)
        if amount >= min_trade_usd:
            reasoning = _build_reasoning(trade, amount, side, market_trade_counts)
            logger.info(
                "[ANALYSIS] %s | Market: %s | Amount: $%.2f",
                "; ".join(reasoning),
                market_name,
                amount,
            )
            if _is_reputable_trade(trade, analyzer, trader_stats, trader_profiles):
                recent_flows.append(
                    {
                        "timestamp": now,
                        "market": market_name,
                        "amount": amount,
                    }
                )
        else:
            reasoning = None
        if len(feed_entries) < 10:
            if amount < min_trade_usd:
                feed_entries.append(f"[SKIP] Small trade of ${amount:,.2f}")
            else:
                feed_entries.append(
                    _format_trade_feed_entry(trade, amount, side, reasoning)
                )
    stats["trade_feed"] = feed_entries
    state["recent_reputable_flows"] = recent_flows[-200:]
    top_market, top_amount = _select_top_opportunity(state["recent_reputable_flows"])
    stats["top_opportunity_market"] = top_market
    stats["top_opportunity_amount"] = top_amount
    five_min_cutoff = now - 5 * 60
    recent_window = [
        entry
        for entry in state["recent_reputable_flows"]
        if entry.get("timestamp", 0) >= five_min_cutoff
    ]
    if recent_window:
        window_top, window_amount = _select_top_opportunity(recent_window)
        if window_top:
            logger.info(
                "I recommend watching %s because $%.2f amount of money flowed in during the last 5 minutes.",
                window_top,
                window_amount,
            )

    whale_trades = analyzer.find_whale_trades(new_trades)
    stats["whale_trades"] = len(whale_trades)

    for index, trade in enumerate(whale_trades):
        trade_id = str(trade.get("id") or get_trade_id(trade, index))

        is_watchlisted = analyzer.is_watchlist_trade(trade)
        is_high_impact = analyzer.is_high_impact(trade)
        impact = estimate_price_impact(trade)
        trade["is_watchlisted"] = is_watchlisted

        if is_watchlisted or is_high_impact:
            trader = (
                trade.get("trader")
                or trade.get("taker")
                or trade.get("maker")
                or trade.get("wallet")
                or "Unknown"
            )
            trader_key = str(trader).lower()
            trader_stats[trader_key] = trader_stats.get(trader_key, 0) + 1
            profile = trader_profiles.get(trader_key, {})
            profile["signals"] = profile.get("signals", 0) + 1
            profile["last_seen"] = time.time()
            update_trader_profile_metrics(profile)
            trader_profiles[trader_key] = profile
            reputation = get_reputation_label(trader_stats[trader_key], profile)
            critical = trader_stats[trader_key] > 5
            condition_id = trade.get("conditionId") or trade.get("condition_id")
            market_title = None
            market_details = None
            if condition_id:
                market_details = client.get_market_details(str(condition_id))
                if market_details:
                    market_title = (
                        market_details.get("question")
                        or market_details.get("title")
                        or market_details.get("name")
                    )
            if trade.get("estimated_usd_value") is not None:
                logger.info(
                    "[ANALYSIS] Found trade of $%s on market: %s",
                    trade.get("estimated_usd_value"),
                    market_title or "Unknown",
                )
            current_market_price = (
                get_current_market_price(market_details) if market_details else None
            )
            if condition_id and current_market_price is not None:
                update_market_price_history(state, str(condition_id), current_market_price)
            history_entry = trader_history.get(trader_key, {})
            repeat_offender = is_watchlisted and history_entry.get("count", 0) > 0
            risk_label, risk_reasons = assess_market_risk(
                market_details,
                min_volume=MIN_MARKET_VOLUME,
                max_spread_pct=MAX_SPREAD_PCT,
            )
            market_trend = (
                get_market_trend(state, str(condition_id))
                if condition_id
                else "Flat"
            )
            market_regime = classify_market_regime(risk_label, market_trend)
            market_url = build_market_url(trade, market_details)
            if market_url:
                trade["marketUrl"] = market_url
            entry_price = trade.get("price")
            if entry_price is not None:
                try:
                    entry_price = float(entry_price)
                except (TypeError, ValueError):
                    entry_price = None
            update_recent_smart_trades(state, trade)
            update_recent_market_signals(state, str(condition_id) if condition_id else None)
            if is_watchlisted and condition_id:
                update_recent_smart_wallet_entries(
                    state,
                    str(condition_id),
                    trader_key,
                )
            signal_type, signal_density, priority_multiplier = get_signal_context(
                state,
                str(condition_id) if condition_id else None,
            )
            market_id = str(condition_id) if condition_id else "unknown"
            analyzer.record_whale_trade(market_id)
            sentiment_label = get_sentiment(state)
            score, reasons = compute_signal_score(
                trade,
                market_details,
                trader_stats[trader_key],
                impact,
                trader_profile=profile,
                market_trend=market_trend,
                market_sentiment=sentiment_label,
                repeat_offender=repeat_offender,
                repeat_offender_bonus=REPEAT_OFFENDER_BONUS,
                market_tracker=analyzer.market_tracker,
                market_id=market_id,
                reason_weights=state.get("reason_weights"),
            )
            update_discovery_watchlist(
                state,
                trader_key,
                score,
                discovered_watchlist,
            )
            if score > 80:
                log_high_score_alert(
                    trade,
                    score,
                    reasons,
                    market_title=market_title,
                    market_url=market_url,
                )
                if entry_price is not None and condition_id:
                    add_virtual_trade(
                        virtual_trades,
                        trade,
                        str(condition_id),
                        entry_price,
                        reasons,
                        extract_trade_side(trade),
                    )
            history_manager.add_trade(market_id, trade_id, score)
            momentum_alert = history_manager.detect_momentum(
                market_id,
                min_score=MOMENTUM_MIN_SCORE,
                window_seconds=SMART_WALLET_WINDOW_S,
                min_trades=MOMENTUM_MIN_TRADES,
            )
            if risk_label == "High" and score <= 90:
                logger.info(
                    "🚫 Guarded alert skipped | Risk: %s | Score: %s | Reasons: %s",
                    risk_label,
                    score,
                    ", ".join(risk_reasons) or "N/A",
                )
                trader_history[trader_key] = {
                    "count": history_entry.get("count", 0) + 1,
                    "last_seen": time.time(),
                    "watchlist": is_watchlisted,
                }
                continue
            log_signal_event(
                signal_type,
                priority_multiplier,
                market_title,
                trader_key,
                trader_stats[trader_key],
                score,
                risk_label,
                reasons,
                momentum_alert=momentum_alert,
            )
            message = format_trade_message(
                trade,
                market_title=market_title,
                impact=impact,
                reputation=reputation,
                critical=critical,
                sentiment=sentiment_label,
                signal_score=score,
                reasons=reasons,
                risk_label=risk_label,
                risk_reasons=risk_reasons,
                signal_type=signal_type,
                signal_density=signal_density,
                priority_multiplier=priority_multiplier,
                market_trend=market_trend,
                market_regime=market_regime,
            )
            if momentum_alert:
                message = f"🚀 MOMENTUM ALERT\n\n{message}"
            notifier.send_message(message)
            stats["alerts_sent"] = int(stats.get("alerts_sent", 0)) + 1
            stats["last_score"] = score
            stats["last_market"] = market_title or "Unknown"
            stats["last_url"] = market_url or "N/A"
            record_signal_history(
                signal_history,
                trade,
                score,
                reasons,
                market_title,
                market_url,
                risk_label,
                signal_type,
                momentum_alert,
            )
            trader_history[trader_key] = {
                "count": history_entry.get("count", 0) + 1,
                "last_seen": time.time(),
                "watchlist": is_watchlisted,
            }
            add_tracked_position(
                state,
                trade_id,
                str(condition_id) if condition_id else None,
                entry_price,
                str(trader),
            )
        else:
            logger.info(
                "Skipping whale trade %s | watchlisted=%s | high_impact=%s | trader=%s | value=%s",
                trade_id,
                is_watchlisted,
                is_high_impact,
                trade.get("trader") or "Unknown",
                trade.get("estimated_usd_value"),
            )

    state["seen_trades"] = list(seen_trades)
    state["trader_stats"] = trader_stats
    state["trader_profiles"] = trader_profiles
    refresh_elite_smart_wallets(state)
    return stats


def log_top_traders(state: dict, limit: int = 5) -> None:
    profiles = state.get("trader_profiles", {})
    if not profiles:
        return
    ranked = sorted(
        profiles.items(),
        key=lambda item: (item[1].get("wins", 0), item[1].get("signals", 0)),
        reverse=True,
    )
    summary = []
    for trader, data in ranked[:limit]:
        wins = data.get("wins", 0)
        losses = data.get("losses", 0)
        avg_profit = data.get("avg_profit", 0.0)
        winrate = data.get("winrate")
        signals = data.get("signals", 0)
        elite_flag = "⭐" if data.get("elite") else ""
        winrate_label = f" winrate:{winrate:.2%}" if isinstance(winrate, float) else ""
        summary.append(
            f"{trader[:6]}…{elite_flag} wins:{wins} losses:{losses} avg:{avg_profit:.4f}"
            f"{winrate_label} signals:{signals}"
        )
    logger.info("Top traders: %s", " | ".join(summary))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket Smart Money tracker")
    parser.add_argument(
        "--backtest",
        dest="backtest_path",
        help="Path to historical trades JSON for backtesting.",
    )
    parser.add_argument(
        "--score-threshold",
        type=int,
        default=70,
        help="Minimum signal score to include in the backtest.",
    )
    parser.add_argument(
        "--market-id",
        help="Market/condition ID to fetch historical trades for backtesting.",
    )
    parser.add_argument(
        "--sweep-thresholds",
        help="Comma-separated score thresholds to evaluate for a sweet spot.",
    )
    parser.add_argument(
        "--horizon-minutes",
        type=int,
        default=60,
        help="Minutes after entry to evaluate exit price in backtest.",
    )
    parser.add_argument(
        "--max-trades",
        type=int,
        default=None,
        help="Optional limit on number of trades loaded for backtest.",
    )
    return parser.parse_args()


def main() -> None:
    client = PolyClient()
    notifier = TelegramNotifier()
    storage = BotStateStorage()
    state = storage.load()
    trader_history = load_trader_history()
    discovered_watchlist = load_discovered_whales()
    signal_history = load_signal_history()
    virtual_trades = load_virtual_trades()
    history_manager = HistoryManager(retention_seconds=CLEANUP_RETENTION_S)
    dashboard = Dashboard()
    logger.info("Starting Polymarket Smart Money tracker loop.")
    cycle = 0
    last_virtual_check = 0.0
    last_performance_report = 0.0
    total_trades_scanned = int(state.get("total_trades_scanned", 0))
    total_trades_fetched = int(state.get("total_trades_fetched", 0))

    dashboard.start()
    try:
        while True:
            trades = None
            for attempt in range(1, FETCH_RETRY_ATTEMPTS + 1):
                try:
                    trades = client.fetch_latest_trades()
                    if trades is None:
                        raise ValueError("No trade data returned")
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Trade fetch attempt %s/%s failed: %s",
                        attempt,
                        FETCH_RETRY_ATTEMPTS,
                        exc,
                    )
                    time.sleep(FETCH_RETRY_BACKOFF_S * attempt)
            if not trades:
                logger.warning("No trades returned from API.")
                stats = {
                    "cycle": cycle,
                    "new_trades": 0,
                    "total_trades_scanned": total_trades_scanned,
                    "total_trades_fetched": total_trades_fetched,
                    "trade_feed": ["No trades returned from API."],
                }
            else:
                stats = process_trades(
                    trades,
                    notifier,
                    client,
                    state,
                    trader_history,
                    history_manager,
                    discovered_watchlist,
                    signal_history,
                    virtual_trades,
                )
                stats["cycle"] = cycle
                total_trades_scanned += int(stats.get("new_trades", 0))
                total_trades_fetched += int(stats.get("fetched_trades", 0))
                state["total_trades_scanned"] = total_trades_scanned
                state["total_trades_fetched"] = total_trades_fetched
                stats["total_trades_scanned"] = total_trades_scanned
                stats["total_trades_fetched"] = total_trades_fetched
                if cycle % 10 == 0:
                    update_tracked_positions(client, notifier, state)
                    log_top_traders(state)

            storage.cleanup(state)
            if cycle % SAVE_INTERVAL_CYCLES == 0:
                storage.save(state)
                save_trader_history(trader_history)
                signal_history = prune_signal_history(signal_history)
                save_signal_history(signal_history)
                save_virtual_trades(virtual_trades)

            report_date = _format_date_label(time.time())
            last_report_date = state.get("last_daily_report_date")
            if last_report_date != report_date:
                report_paths = generate_daily_report(signal_history, state)
                state["last_daily_report_date"] = report_date
                logger.info(
                    "Daily report generated: %s (HTML: %s)",
                    report_paths.get("summary"),
                    report_paths.get("html"),
                )

            now = time.time()
            if now - last_virtual_check >= VIRTUAL_TRADE_CHECK_INTERVAL_S:
                virtual_trades = verify_virtual_trades(client, state, virtual_trades)
                save_virtual_trades(virtual_trades)
                last_virtual_check = now

            if now - last_performance_report >= PERFORMANCE_REPORT_INTERVAL_S:
                totals = state.get("performance_totals", {"predictions": 0, "wins": 0})
                log_performance_report(
                    state,
                    int(totals.get("predictions", 0)),
                    int(totals.get("wins", 0)),
                )
                last_performance_report = now

            cleanup_state_memory(state, history_manager)
            dashboard.update(stats, state)
            time.sleep(POLL_INTERVAL_S)
            cycle += 1
    finally:
        dashboard.stop()


if __name__ == "__main__":
    args = parse_args()
    if args.backtest_path or args.market_id:
        config = BacktestConfig(
            score_threshold=args.score_threshold,
            horizon_minutes=args.horizon_minutes,
            max_trades=args.max_trades,
        )
        report = None
        trades: list[dict] | None = None
        if args.backtest_path and os.path.exists(args.backtest_path):
            report = run_backtest(args.backtest_path, config)
        elif args.market_id:
            trades = fetch_trades_for_backtest(args.market_id, args.max_trades)
            report = run_backtest_from_trades(trades, config)
        else:
            raise FileNotFoundError(
                "Backtest file not found and no market_id provided."
            )

        if args.sweep_thresholds and trades is not None:
            thresholds = [
                int(value.strip())
                for value in args.sweep_thresholds.split(",")
                if value.strip().isdigit()
            ]
            if thresholds:
                sweep = evaluate_thresholds(trades, thresholds, args.horizon_minutes)
                best = sweep.get("best", {})
                logger.info(
                    "Sweet spot threshold: %s (profit: %.2f, hit rate: %.2f%%)",
                    best.get("score_threshold"),
                    best.get("total_potential_profit", 0.0),
                    best.get("hit_rate", 0.0) * 100,
                )
    else:
        main()
