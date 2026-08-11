"""Option-chain OI / PCR / strike tests (no network)."""

from scanner.premarket.collectors.option_chain import (
    analyze_option_records,
    classify_expiry,
    important_strikes_around,
)
from datetime import date


def _sample_records(spot=24850):
    # Strikes around spot with put support below and call resistance above
    rows = []
    for strike, c_oi, p_oi, c_chg, p_chg, c_vol, p_vol in [
        (24700, 1000, 90000, 100, 5000, 200, 800),
        (24800, 2000, 70000, 200, 3000, 300, 600),
        (24850, 5000, 5000, 0, 0, 400, 400),
        (24900, 80000, 3000, 4000, 100, 900, 200),
        (25000, 120000, 1000, 8000, 50, 1000, 100),
    ]:
        rows.append(
            {
                "strikePrice": strike,
                "CE": {
                    "openInterest": c_oi,
                    "changeinOpenInterest": c_chg,
                    "totalTradedVolume": c_vol,
                },
                "PE": {
                    "openInterest": p_oi,
                    "changeinOpenInterest": p_chg,
                    "totalTradedVolume": p_vol,
                },
            }
        )
    return rows


def test_important_strikes_dynamic():
    strikes = important_strikes_around(24850, 50, radius=2)
    assert 24850 in strikes or 24800 in strikes
    assert strikes == sorted(strikes)
    assert len(strikes) == 5
    assert all((s % 50) == 0 for s in strikes)


def test_pcr_and_max_oi():
    snap = analyze_option_records(
        _sample_records(),
        symbol="NIFTY",
        spot=24850,
        expiry="11-Mar-2026",
        strike_interval=50,
    )
    assert snap.available
    assert snap.highest_call_oi_strike == 25000
    assert snap.highest_put_oi_strike == 24700
    assert snap.pcr_oi is not None
    assert snap.pcr_oi > 0
    assert snap.pcr_volume is not None
    assert 24700 in snap.put_support_levels or snap.put_support_levels
    assert snap.call_resistance_levels


def test_classify_expiry_day():
    today = date(2026, 3, 11)
    days, etype = classify_expiry(date(2026, 3, 11), today=today)
    assert days == 0
    assert etype == "expiry_day"


def test_empty_records():
    snap = analyze_option_records([], symbol="NIFTY", spot=100, expiry=None, strike_interval=50)
    assert snap.available is False
