"""Main loop for Polymarket Smart Money tracker."""

from __future__ import annotations

import logging
import time

from src.analyzer import TradeAnalyzer
from src.notifications import TelegramNotifier
from src.polymarket_api import PolyClient
from src.storage import BotStateStorage


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


def process_trades(
    trades: list[dict],
    notifier: TelegramNotifier,
    client: PolyClient,
    state: dict,
) -> None:
    analyzer = TradeAnalyzer(trades)
    whale_trades = analyzer.find_whale_trades()
    processed_trade_ids = set(state.get("processed_trade_ids", []))
    trader_stats = state.get("trader_stats", {})

    for index, trade in enumerate(whale_trades):
        trade_id = get_trade_id(trade, index)
        if trade_id in processed_trade_ids:
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
                market_title = (
                    market_details.get("question")
                    or market_details.get("title")
                    or market_details.get("name")
                )
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

        processed_trade_ids.add(trade_id)

    state["processed_trade_ids"] = list(processed_trade_ids)
    state["trader_stats"] = trader_stats


def main() -> None:
    client = PolyClient()
    notifier = TelegramNotifier()
    storage = BotStateStorage()
    state = storage.load()
    logger.info("Starting Polymarket Smart Money tracker loop.")

    while True:
        trades = client.fetch_latest_trades()
        if not trades:
            logger.warning("No trades returned from API.")
        else:
            process_trades(trades, notifier, client, state)
            storage.save(state)

        time.sleep(60)


if __name__ == "__main__":
    main()
