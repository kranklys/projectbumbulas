"""Main loop for Polymarket Smart Money tracker."""

from __future__ import annotations

import logging
import time
from collections import deque

from src.analyzer import TradeAnalyzer
from src.notifications import TelegramNotifier
from src.polymarket_api import PolyClient
from src.storage import BotStateStorage
from src.config import POLL_INTERVAL_S, SAVE_INTERVAL_CYCLES


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

    return (
        f"{header}\n\n"
        f"🎯 Market: {title}\n\n"
        f"💰 Amount: ${value if value is not None else 'N/A'}\n\n"
        f"📈 Price: {price}\n\n"
        f"🧠 Reputation: {reputation_label}\n\n"
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


def update_tracked_positions(
    client: PolyClient,
    notifier: TelegramNotifier,
    state: dict,
) -> None:
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
            if condition_id:
                market_details = client.get_market_details(str(condition_id))
                if market_details:
                    market_title = (
                        market_details.get("question")
                        or market_details.get("title")
                        or market_details.get("name")
                    )
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
            message = format_trade_message(
                trade,
                market_title=market_title,
                impact=impact,
                reputation=reputation,
                critical=critical,
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
