"""Main loop for Polymarket Smart Money tracker."""

from __future__ import annotations

import logging
import time

from src.analyzer import TradeAnalyzer
from src.notifications import TelegramNotifier
from src.polymarket_api import PolyClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def format_trade_message(trade: dict, impact: float | None = None) -> str:
    value = trade.get("estimated_usd_value")
    trader = (
        trade.get("trader")
        or trade.get("taker")
        or trade.get("maker")
        or trade.get("wallet")
        or "Unknown"
    )
    link = trade.get("marketUrl") or trade.get("url") or "N/A"
    impact_pct = f"{impact * 100:.2f}" if impact is not None else "N/A"

    return (
        "🚨 Smart Trade Detected!\n\n"
        f"💰 Value: ${value if value is not None else 'N/A'}\n\n"
        f"📉 Price Change: {impact_pct}%\n\n"
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


def process_trades(trades: list[dict], notifier: TelegramNotifier) -> None:
    analyzer = TradeAnalyzer(trades)
    whale_trades = analyzer.find_whale_trades()

    for trade in whale_trades:
        is_watchlisted = analyzer.is_watchlist_trade(trade)
        is_high_impact = analyzer.is_high_impact(trade)
        impact = estimate_price_impact(trade)

        if is_watchlisted or is_high_impact:
            logger.info(
                "Smart trade detected | watchlisted=%s | high_impact=%s | trade=%s",
                is_watchlisted,
                is_high_impact,
                trade,
            )
            message = format_trade_message(trade, impact=impact)
            notifier.send_message(message)


def main() -> None:
    client = PolyClient()
    notifier = TelegramNotifier()
    logger.info("Starting Polymarket Smart Money tracker loop.")

    while True:
        trades = client.fetch_latest_trades()
        if not trades:
            logger.warning("No trades returned from API.")
        else:
            process_trades(trades, notifier)

        time.sleep(60)


if __name__ == "__main__":
    main()
