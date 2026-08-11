"""Calendar, timezone, duplicate job, and report generation tests."""

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from scanner.premarket.calendar import is_trading_day, should_run_job
from scanner.premarket.confirm import compare_bias
from scanner.premarket.db import finish_job, init_db, save_report, try_acquire_job
from scanner.premarket.report import format_premarket_report


def test_weekend_not_trading_day():
    assert is_trading_day(date(2026, 3, 7)) is False  # Saturday
    assert is_trading_day(date(2026, 3, 8)) is False  # Sunday


def test_weekday_trading_day():
    assert is_trading_day(date(2026, 3, 10)) is True  # Tuesday (not in holiday set)


def test_holiday_not_trading():
    assert is_trading_day(date(2026, 1, 26)) is False  # Republic Day


def test_should_run_job_timezone_window():
    tz = ZoneInfo("Asia/Kolkata")
    now = datetime(2026, 3, 10, 9, 5, tzinfo=tz)
    assert should_run_job("09:00", now=now, window_minutes=15) is True
    assert should_run_job("09:15", now=now, window_minutes=15) is False
    weekend = datetime(2026, 3, 8, 9, 5, tzinfo=tz)
    assert should_run_job("09:00", now=weekend) is False


def test_duplicate_job_protection(tmp_path: Path):
    db = tmp_path / "t.db"
    init_db(db)
    assert try_acquire_job("premarket", "2026-03-10", db) is True
    assert try_acquire_job("premarket", "2026-03-10", db) is False
    finish_job("premarket", "2026-03-10", "success", db_path=db)


def test_compare_bias_confirmation():
    assert compare_bias("Mild Bullish", "bullish") == "Confirmed"
    assert compare_bias("Bearish", "bullish") == "Invalidated"
    assert compare_bias("Mild Bullish", "flat") == "Partially confirmed"
    assert compare_bias("Bullish", "unavailable") == "Insufficient data"


def test_report_generation_handles_missing_fields():
    report = {
        "meta": {"report_date": "2026-03-10", "report_time": "09:00"},
        "nifty": {"previous_close": 24850, "expected_open": None, "gap": None, "gap_pct": None, "trend": "Bullish"},
        "banknifty": {},
        "global": {"us_direction": "unavailable", "asia_direction": "Mixed", "gift_direction": "unavailable"},
        "vix": {"value": None, "trend": None},
        "fii_dii": {},
        "option_chain_nifty": {"error": "DATA UNAVAILABLE"},
        "option_chain_banknifty": {},
        "levels_nifty": {},
        "levels_banknifty": {},
        "checklist": [{"label": "US market supportive", "passed": None}],
        "bias": {
            "label": "Neutral",
            "confidence": 30,
            "total_score": 0,
            "max_score": 2,
            "normalized_score": 50,
            "confidence_reason": "mostly unavailable",
        },
        "scenarios": {"primary": "Wait", "alternative": "Wait"},
        "risk_flags": ["Option-chain: DATA UNAVAILABLE"],
        "regime": "Unclear",
    }
    text = format_premarket_report(report)
    assert "F&O PRE-MARKET REPORT" in text
    assert "DATA UNAVAILABLE" in text
    assert "PRE-MARKET BIAS" in text
    assert "not a prediction" in text.lower() or "directional bias" in text.lower()


def test_save_report_roundtrip(tmp_path: Path):
    db = tmp_path / "t.db"
    report = {
        "meta": {"report_date": "2026-03-10", "report_time": "09:00", "job_type": "premarket"},
        "nifty": {"expected_open": 24900},
        "banknifty": {},
        "vix": {"value": 14.2},
        "fii_dii": {"fii_net": 100, "dii_net": 50},
        "option_chain_nifty": {"pcr_oi": 1.1},
        "option_chain_banknifty": {},
        "bias": {
            "label": "Mild Bullish",
            "confidence": 67,
            "total_score": 3,
            "max_score": 10,
            "normalized_score": 65,
        },
        "regime": "Trending Bullish",
    }
    row_id = save_report(report, db_path=db)
    assert row_id >= 0
