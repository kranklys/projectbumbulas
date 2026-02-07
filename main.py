"""Main loop for Polymarket Smart Money tracker."""

from __future__ import annotations

import logging
import time
from collections import deque

from src.analyzer import (
    TradeAnalyzer,
    assess_market_risk,
    compute_signal_score,
    track_market_volumes,
)
from src.notifications import TelegramNotifier
from src.polymarket_api import PolyClient
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


def get_signal_context(state: dict, condition_id: str | None) -> tuple[str, str]:
    now = time.time()
    market_signals = state.get("recent_market_signals", [])
    volume_spikes = state.get("recent_volume_spikes", [])

    signals_15m = sum(1 for s in market_signals if now - s.get("timestamp", 0) <= 900)
    signals_60m = sum(1 for s in market_signals if now - s.get("timestamp", 0) <= 3600)
    density_label = f"15m:{signals_15m} | 60m:{signals_60m}"

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
            return "Confirmed", density_label
        if recent_market_signals >= 3:
            return "Momentum", density_label

    return "Standard", density_label


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

    state["tracked_positions"] = updated_positions


def process_trades(
    trades: list[dict],
    notifier: TelegramNotifier,
    client: PolyClient,
    state: dict,
) -> None:
    analyzer = TradeAnalyzer(trades)
    whale_trades = analyzer.find_whale_trades()
    processed_trade_ids = deque(state.get("processed_trade_ids", []), maxlen=1000)
    processed_trade_id_set = set(processed_trade_ids)
    trader_stats = state.get("trader_stats", {})

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
            risk_label, risk_reasons = assess_market_risk(
                market_details,
                min_volume=MIN_MARKET_VOLUME,
                max_spread_pct=MAX_SPREAD_PCT,
            )
            if risk_label == "High":
                logger.info(
                    "Skipping alert due to high market risk: %s (%s)",
                    risk_label,
                    ", ".join(risk_reasons),
                )
                processed_trade_ids.append(trade_id)
                processed_trade_id_set.add(trade_id)
                continue
            entry_price = trade.get("price")
            if entry_price is not None:
                try:
                    entry_price = float(entry_price)
                except (TypeError, ValueError):
                    entry_price = None
            logger.info(
                "Smart trade detected | watchlisted=%s | high_impact=%s | trade=%s",
                is_watchlisted,
                is_high_impact,
                trade,
            )
            update_recent_smart_trades(state, trade)
            update_recent_market_signals(state, str(condition_id) if condition_id else None)
            signal_type, signal_density = get_signal_context(
                state,
                str(condition_id) if condition_id else None,
            )
            score, reasons = compute_signal_score(
                trade,
                market_details,
                trader_stats[trader_key],
                impact,
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
            )
            notifier.send_message(message)
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


def main() -> None:
    client = PolyClient()
    notifier = TelegramNotifier()
    storage = BotStateStorage()
    state = storage.load()
    logger.info("Starting Polymarket Smart Money tracker loop.")
    cycle = 0

    while True:
        trades = client.fetch_latest_trades()
        if not trades:
            logger.warning("No trades returned from API.")
        else:
            process_trades(trades, notifier, client, state)
            if cycle % 10 == 0:
                update_tracked_positions(client, notifier, state)
            storage.cleanup(state)
            if cycle % SAVE_INTERVAL_CYCLES == 0:
                storage.save(state)

        time.sleep(POLL_INTERVAL_S)
        cycle += 1


if __name__ == "__main__":
    main()
