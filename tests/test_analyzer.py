from src.analyzer import (
    TradeAnalyzer,
    assess_market_risk,
    compute_signal_score,
    track_market_volumes,
)


def test_find_whale_trades_adds_estimated_value():
    analyzer = TradeAnalyzer(
        [
            {"notional": "12000", "price": 0.4, "size": 100},
            {"amount": 9000, "price": 0.2, "size": 500},
        ]
    )

    whales = analyzer.find_whale_trades()

    assert len(whales) == 1
    assert whales[0]["estimated_usd_value"] == 12000.0


def test_is_watchlist_trade_is_case_insensitive():
    analyzer = TradeAnalyzer([])

    assert analyzer.is_watchlist_trade(
        {"trader": "0x1111111111111111111111111111111111111111"}
    )
    assert analyzer.is_watchlist_trade(
        {"wallet": "0x2222222222222222222222222222222222222222".upper()}
    )
    assert not analyzer.is_watchlist_trade({"trader": "0xabc"})


def test_is_high_impact_threshold():
    analyzer = TradeAnalyzer([])

    assert analyzer.is_high_impact({"previousPrice": 0.4, "price": 0.45})
    assert not analyzer.is_high_impact({"previousPrice": 0.4, "price": 0.401})


def test_compute_signal_score_with_reasons():
    score, reasons = compute_signal_score(
        {"estimated_usd_value": 60000},
        {"volume24h": 120000},
        reputation_count=12,
        impact=0.06,
    )

    assert score >= 80
    assert "Very large trade size" in reasons
    assert "High price impact" in reasons
    assert "Elite trader frequency" in reasons
    assert "High market liquidity" in reasons


def test_assess_market_risk_classifies_high():
    risk, reasons = assess_market_risk(
        {"volume24h": 500, "bestBid": 0.4, "bestAsk": 0.6},
        min_volume=1000,
        max_spread_pct=5,
    )

    assert risk == "High"
    assert "Low volume" in reasons
    assert "Wide spread" in reasons


def test_track_market_volumes_detects_spike():
    markets = [
        {"id": "m1", "volume24h": 2000, "question": "Test", "url": "x"}
    ]
    spikes, updated = track_market_volumes(
        markets, previous_volumes={"m1": 1000}, min_volume=1000, spike_threshold=0.5
    )

    assert len(spikes) == 1
    assert spikes[0]["change_pct"] == 100.0
    assert updated["m1"] == 2000.0
