"""Lightweight event / expiry awareness (no random news scraping)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from scanner.premarket.calendar import now_ist
from scanner.premarket.models import OptionChainSnapshot


# Static high-impact windows (update annually). Format: (month, day, label) for fixed dates
# and approximate windows as needed. Prefer marking known recurring risks.
FIXED_EVENTS_2025_2026: list[tuple[date, str]] = [
    (date(2025, 2, 1), "Union Budget"),
    (date(2026, 2, 1), "Union Budget (expected window)"),
]


def _last_thursday(year: int, month: int) -> date:
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    d = next_month - timedelta(days=1)
    while d.weekday() != 3:  # Thursday
        d -= timedelta(days=1)
    return d


def monthly_expiry_nearby(today: date | None = None, window_days: int = 1) -> list[str]:
    today = today or now_ist().date()
    events = []
    for offset in (-1, 0, 1):
        y = today.year
        m = today.month + offset
        while m <= 0:
            m += 12
            y -= 1
        while m > 12:
            m -= 12
            y += 1
        exp = _last_thursday(y, m)
        if abs((exp - today).days) <= window_days:
            events.append(f"Monthly F&O expiry ({exp.isoformat()})")
    return events


def collect_risk_events(
    *,
    nifty_chain: OptionChainSnapshot | None = None,
    bank_chain: OptionChainSnapshot | None = None,
    today: date | None = None,
) -> list[str]:
    """Return human-readable risk flags (events / expiry). Does not invent news."""
    today = today or now_ist().date()
    flags: list[str] = []

    for d, label in FIXED_EVENTS_2025_2026:
        if abs((d - today).days) <= 1:
            flags.append(f"Important event nearby: {label}")

    flags.extend(monthly_expiry_nearby(today))

    for chain in (nifty_chain, bank_chain):
        if chain and chain.available and chain.days_to_expiry is not None:
            if chain.days_to_expiry == 0:
                flags.append(f"{chain.symbol} expiry day — treat option OI with caution")
            elif chain.days_to_expiry <= 2:
                flags.append(f"{chain.symbol} expiry in {chain.days_to_expiry} day(s)")

    return flags
