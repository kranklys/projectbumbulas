"""Main loop for Polymarket Smart Money tracker."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque

from src.analyzer import (
    TradeAnalyzer,
    assess_market_risk,
    compute_signal_score,
    evaluate_signal_density,
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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

TRADER_HISTORY_PATH = os.path.join("data", "trader_history.json")
REPEAT_OFFENDER_BONUS = 10
SMART_WALLET_WINDOW_S = 60 * 60
SMART_WALLET_THRESHOLD = 3
MOMENTUM_MIN_SCORE = 85
MOMENTUM_MIN_TRADES = 3

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
) -> str:
    value = trade.get("estimated_usd_value")
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

    return (
        f"{header}\n\n"
        f"🎯 Market: {title}\n\n"
        f"💰 Amount: ${value if value is not None else 'N/A'}\n\n"
        f"📈 Price: {price}\n\n"
        f"🧠 Reputation: {reputation_label}\n\n"
        f"🌡️ Sentiment: {sentiment_label}\n\n"
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


def get_reputation_label(count: int) -> str:
    if count >= 10:
        return "Elite Trader"
    if count >= 5:
        return "Frequent"
    return "New"


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
        return "Bullish 🐂"
    if no_count > yes_count and no_count > 0:
        return "Bearish 🐻"
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


def log_signal_event(
    signal_type: str,
    priority_multiplier: int,
    market_title: str | None,
    trader: str,
    score: int,
    risk_label: str,
    reasons: list[str],
    momentum_alert: str | None = None,
) -> None:
    title = market_title or "Unknown"
    if momentum_alert:
        header = f"{COLOR_MOMENTUM}🚀 MOMENTUM ALERT{COLOR_RESET}"
    elif signal_type == "CONFIRMED MOMENTUM":
        header = f"{COLOR_MOMENTUM}✅ CONFIRMED MOMENTUM{COLOR_RESET}"
    else:
        header = f"{COLOR_WHALE}🐳 NORMAL WHALE{COLOR_RESET}"
    logger.info(
        "\n%s\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🎯 Market: %s\n"
        "👤 Trader: %s\n"
        "✅ Score: %s/100 | Priority: x%s\n"
        "⚠️ Risk: %s\n"
        "🧩 Reasons: %s\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        header,
        title,
        trader,
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
                profile["avg_profit"] = total_profit / profile["resolved_count"]
                if pnl > 0:
                    profile["wins"] = profile.get("wins", 0) + 1
                else:
                    profile["losses"] = profile.get("losses", 0) + 1
            trader_profiles[str(trader).lower()] = profile

    state["tracked_positions"] = updated_positions
    state["trader_profiles"] = trader_profiles


def process_trades(
    trades: list[dict],
    notifier: TelegramNotifier,
    client: PolyClient,
    state: dict,
    trader_history: dict,
    history_manager: HistoryManager,
) -> None:
    analyzer = TradeAnalyzer(trades)
    whale_trades = analyzer.find_whale_trades()
    processed_trade_ids = deque(state.get("processed_trade_ids", []), maxlen=1000)
    processed_trade_id_set = set(processed_trade_ids)
    trader_stats = state.get("trader_stats", {})
    trader_profiles = state.get("trader_profiles", {})

    for index, trade in enumerate(whale_trades):
        trade_id = get_trade_id(trade, index)
        if trade_id in processed_trade_id_set:
            continue

        is_watchlisted = analyzer.is_watchlist_trade(trade)
        is_high_impact = analyzer.is_high_impact(trade)
        impact = estimate_price_impact(trade)

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
            trader_profiles[trader_key] = profile
            reputation = get_reputation_label(trader_stats[trader_key])
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
            history_entry = trader_history.get(trader_key, {})
            repeat_offender = is_watchlisted and history_entry.get("count", 0) > 0
            risk_label, risk_reasons = assess_market_risk(
                market_details,
                min_volume=MIN_MARKET_VOLUME,
                max_spread_pct=MAX_SPREAD_PCT,
            )
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
            score, reasons = compute_signal_score(
                trade,
                market_details,
                trader_stats[trader_key],
                impact,
                repeat_offender=repeat_offender,
                repeat_offender_bonus=REPEAT_OFFENDER_BONUS,
            )
            market_id = str(condition_id) if condition_id else "unknown"
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
                processed_trade_ids.append(trade_id)
                processed_trade_id_set.add(trade_id)
                continue
            log_signal_event(
                signal_type,
                priority_multiplier,
                market_title,
                trader_key,
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
                sentiment=get_sentiment(state),
                signal_score=score,
                reasons=reasons,
                risk_label=risk_label,
                risk_reasons=risk_reasons,
                signal_type=signal_type,
                signal_density=signal_density,
                priority_multiplier=priority_multiplier,
            )
            if momentum_alert:
                message = f"🚀 MOMENTUM ALERT\n\n{message}"
            notifier.send_message(message)
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

        processed_trade_ids.append(trade_id)
        processed_trade_id_set.add(trade_id)

    state["processed_trade_ids"] = list(processed_trade_ids)
    state["trader_stats"] = trader_stats
    state["trader_profiles"] = trader_profiles


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
        signals = data.get("signals", 0)
        summary.append(
            f"{trader[:6]}… wins:{wins} losses:{losses} avg:{avg_profit:.4f} signals:{signals}"
        )
    logger.info("Top traders: %s", " | ".join(summary))


def main() -> None:
    client = PolyClient()
    notifier = TelegramNotifier()
    storage = BotStateStorage()
    state = storage.load()
    trader_history = load_trader_history()
    history_manager = HistoryManager()
    logger.info("Starting Polymarket Smart Money tracker loop.")
    cycle = 0

    while True:
        trades = client.fetch_latest_trades()
        if not trades:
            logger.warning("No trades returned from API.")
        else:
            process_trades(
                trades,
                notifier,
                client,
                state,
                trader_history,
                history_manager,
            )
            if cycle % 10 == 0:
                update_tracked_positions(client, notifier, state)
                log_top_traders(state)
            storage.cleanup(state)
            if cycle % SAVE_INTERVAL_CYCLES == 0:
                storage.save(state)
                save_trader_history(trader_history)

        time.sleep(POLL_INTERVAL_S)
        cycle += 1


if __name__ == "__main__":
    main()
