"""Tests for beginner-friendly Telegram glance formatting."""

from scanner.premarket.report import (
    format_confirm_telegram,
    format_open_telegram,
    format_premarket_telegram,
)


def _sample_report():
    return {
        "meta": {"report_date": "2026-08-11", "report_time": "09:00"},
        "nifty": {
            "available": True,
            "previous_close": 24450.25,
            "expected_open": 24450.25,
            "gap_class": "Flat",
            "gap_pct": 0.0,
        },
        "banknifty": {
            "available": True,
            "previous_close": 57366.0,
            "gap_class": "Small Gap Up",
            "gap_pct": 0.2,
        },
        "global": {"us_direction": "Negative", "asia_direction": "Mixed", "gift_direction": "unavailable"},
        "vix": {"available": True, "value": 11.87},
        "fii_dii": {"available": True, "fii_net": 1974.0, "dii_net": -1290.0},
        "levels_nifty": {"immediate_support": 24429.0, "immediate_resistance": 24541.0},
        "option_chain_nifty": {"available": False},
        "bias": {"label": "Mild Bullish", "confidence": 67},
        "risk_flags": ["Option-chain unavailable"],
    }


def test_telegram_glance_is_short_and_plain():
    text = format_premarket_telegram(_sample_report())
    assert "Morning Market Glance" in text
    assert "Overall vibe" in text
    assert "Mild Bullish" in text
    assert "What to do" in text
    assert "PCR" not in text
    assert "SMA" not in text
    assert "CHECKLIST" not in text
    assert "normalized" not in text.lower()


def test_open_telegram_glance():
    text = format_open_telegram(
        {
            "meta": {"report_date": "2026-08-11"},
            "premarket_bias": "Mild Bullish",
            "open_915_result": "Confirmed",
            "nifty": {"open": 24500.0},
            "expected_vs_actual_nifty": {"expected": 24450.0, "actual_open": 24500.0},
        }
    )
    assert "9:15 Open Check" in text
    assert "Confirmed" in text
    assert "What to do" in text


def test_confirm_telegram_glance():
    text = format_confirm_telegram(
        {
            "meta": {"report_date": "2026-08-11"},
            "premarket_bias": "Mild Bullish",
            "confirm_930_result": "Invalidated",
            "nifty": {"last": 24300.0},
        }
    )
    assert "9:30 Confirmation" in text
    assert "Invalidated" in text
    assert "not a guaranteed call" in text.lower() or "Decision-support" in text
