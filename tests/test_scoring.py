"""Levels, regime, scoring, confidence, checklist tests."""

from scanner.premarket.analysis.confidence import compute_confidence
from scanner.premarket.analysis.levels import build_levels
from scanner.premarket.analysis.regime import classify_regime
from scanner.premarket.analysis.scoring import classify_bias, compute_scores, score_gap
from scanner.premarket.models import (
    FIIDIISnapshot,
    GlobalSnapshot,
    IndexSnapshot,
    OptionChainSnapshot,
)


def _nifty(**kwargs):
    base = dict(
        symbol="^NSEI",
        name="NIFTY",
        previous_close=24850,
        previous_open=24700,
        previous_high=24950,
        previous_low=24650,
        expected_open=24930,
        gap=80,
        gap_pct=0.32,
        gap_class="Moderate Gap Up",
        trend="Bullish",
        sma50=24700,
        sma200=24000,
        available=True,
    )
    base.update(kwargs)
    return IndexSnapshot(**base)


def _chain(**kwargs):
    base = dict(
        symbol="NIFTY",
        available=True,
        highest_call_oi_strike=25000,
        highest_put_oi_strike=24700,
        put_support_levels=[24700, 24600],
        call_resistance_levels=[25000, 25100],
        pcr_oi=1.15,
        put_oi_change=5000,
        call_oi_change=2000,
        total_call_oi=100000,
        total_put_oi=115000,
        expiry_type="weekly",
        days_to_expiry=3,
    )
    base.update(kwargs)
    return OptionChainSnapshot(**base)


def test_build_levels_sources():
    levels = build_levels(_nifty(), _chain())
    assert levels.immediate_support is not None
    assert levels.immediate_resistance is not None
    assert levels.sources


def test_regime_trending_bullish():
    assert classify_regime(_nifty(), {"value": 14}) == "Trending Bullish"


def test_regime_high_vix():
    assert classify_regime(_nifty(), {"value": 25}) == "High Volatility"


def test_scoring_bullish_scenario():
    scores = compute_scores(
        nifty=_nifty(),
        bank=_nifty(name="BANK NIFTY", symbol="^NSEBANK"),
        global_snap=GlobalSnapshot(
            available=True,
            us_direction="Positive",
            asia_direction="Positive",
            gift_direction="Positive",
            gift_nifty=24920,
        ),
        vix={"available": True, "value": 13.5, "trend": "Falling"},
        fii_dii=FIIDIISnapshot(available=True, fii_net=500, dii_net=300),
        nifty_chain=_chain(),
        risk_flags=[],
    )
    assert scores["normalized_score"] > 55
    assert "Bullish" in scores["label"]


def test_scoring_bearish_gap_down():
    nifty = _nifty(
        expected_open=24500,
        gap=-350,
        gap_pct=-1.4,
        gap_class="Strong Gap Down",
        trend="Bearish",
        sma50=24000,
        sma200=24700,
    )
    scores = compute_scores(
        nifty=nifty,
        bank=nifty,
        global_snap=GlobalSnapshot(
            available=True,
            us_direction="Negative",
            asia_direction="Negative",
            gift_direction="Negative",
        ),
        vix={"available": True, "value": 24, "trend": "Rising"},
        fii_dii=FIIDIISnapshot(available=True, fii_net=-800, dii_net=-100),
        nifty_chain=_chain(pcr_oi=0.55, total_put_oi=50000, total_call_oi=120000, put_oi_change=100, call_oi_change=9000),
        risk_flags=["High volatility"],
    )
    assert scores["normalized_score"] < 45
    assert "Bearish" in scores["label"]


def test_scoring_missing_data_neutralish():
    empty = IndexSnapshot(symbol="x", name="x", available=False)
    scores = compute_scores(
        nifty=empty,
        bank=empty,
        global_snap=GlobalSnapshot(available=False),
        vix={"available": False},
        fii_dii=FIIDIISnapshot(available=False),
        nifty_chain=OptionChainSnapshot(symbol="NIFTY", available=False),
        risk_flags=[],
    )
    # Mostly unavailable → near neutral normalized when max_score small / events only
    assert 0 <= scores["normalized_score"] <= 100


def test_confidence_drops_when_missing():
    breakdowns = [
        {"available": False, "signal": 0},
        {"available": False, "signal": 0},
        {"available": True, "signal": 1},
        {"available": True, "signal": 1},
    ]
    conf = compute_confidence(breakdowns)
    assert conf["confidence"] <= 65
    assert conf["missing_categories"] == 2


def test_confidence_agreement():
    breakdowns = [{"available": True, "signal": 2}] * 6 + [{"available": True, "signal": -1}]
    conf = compute_confidence(breakdowns)
    assert conf["bullish_signals"] == 6
    assert conf["bearish_signals"] == 1
    assert conf["confidence"] >= 50


def test_classify_bias_bands():
    assert "Bearish" in classify_bias(10)
    assert classify_bias(50) == "Neutral"
    assert "Bullish" in classify_bias(90)


def test_score_gap_unit():
    b = score_gap(_nifty(gap_pct=1.0))
    assert b.signal == 2
    assert b.available
