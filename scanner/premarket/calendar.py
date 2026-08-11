"""Indian market calendar helpers (weekends + NSE holidays)."""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

# NSE holidays 2025–2026 (equity + F&O closed). Configurable via env override later.
# Source: NSE holiday circulars (static fallback when live calendar unavailable).
NSE_HOLIDAYS: set[date] = {
    # 2025
    date(2025, 2, 26),  # Mahashivratri
    date(2025, 3, 14),  # Holi
    date(2025, 3, 31),  # Id-Ul-Fitr (approx / may vary)
    date(2025, 4, 10),  # Mahavir Jayanti
    date(2025, 4, 14),  # Dr Ambedkar Jayanti / Good Friday cluster
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 1),   # Maharashtra Day
    date(2025, 8, 15),  # Independence Day
    date(2025, 8, 27),  # Ganesh Chaturthi
    date(2025, 10, 2),  # Gandhi Jayanti
    date(2025, 10, 21), # Diwali Laxmi Pujan (muhurat may still trade — mark holiday for safety)
    date(2025, 10, 22), # Diwali Balipratipada
    date(2025, 11, 5),  # Guru Nanak Jayanti
    date(2025, 12, 25), # Christmas
    # 2026 (common / announced-style set; update annually)
    date(2026, 1, 26),  # Republic Day
    date(2026, 3, 3),   # Holi (approx)
    date(2026, 3, 26),  # Ram Navami (approx)
    date(2026, 3, 31),  # Id-Ul-Fitr (approx)
    date(2026, 4, 3),   # Good Friday
    date(2026, 4, 14),  # Ambedkar Jayanti
    date(2026, 5, 1),   # Maharashtra Day
    date(2026, 8, 15),  # Independence Day
    date(2026, 10, 2),  # Gandhi Jayanti
    date(2026, 11, 10), # Diwali (approx)
    date(2026, 11, 24), # Guru Nanak Jayanti (approx)
    date(2026, 12, 25), # Christmas
}


def get_tz(name: str = "Asia/Kolkata") -> ZoneInfo:
    return ZoneInfo(name)


def now_ist(tz_name: str = "Asia/Kolkata") -> datetime:
    return datetime.now(get_tz(tz_name))


def is_weekend(d: date) -> bool:
    return d.weekday() >= 5  # Sat=5, Sun=6


def is_nse_holiday(d: date, extra_holidays: set[date] | None = None) -> bool:
    holidays = NSE_HOLIDAYS | (extra_holidays or set())
    return d in holidays


def is_trading_day(d: date | None = None, extra_holidays: set[date] | None = None) -> bool:
    """True if NSE equity/F&O session is expected to be open."""
    if d is None:
        d = now_ist().date()
    if is_weekend(d):
        return False
    if is_nse_holiday(d, extra_holidays):
        return False
    return True


def parse_hhmm(value: str) -> time:
    hour, minute = value.strip().split(":")
    return time(int(hour), int(minute))


def should_run_job(
    job_time: str,
    *,
    tz_name: str = "Asia/Kolkata",
    now: datetime | None = None,
    window_minutes: int = 15,
    extra_holidays: set[date] | None = None,
) -> bool:
    """
    Whether a scheduled job should run now: trading day and within
    [job_time, job_time + window_minutes) in Asia/Kolkata.
    """
    now = now or now_ist(tz_name)
    if now.tzinfo is None:
        now = now.replace(tzinfo=get_tz(tz_name))
    else:
        now = now.astimezone(get_tz(tz_name))

    if not is_trading_day(now.date(), extra_holidays):
        return False

    target = parse_hhmm(job_time)
    start = now.replace(hour=target.hour, minute=target.minute, second=0, microsecond=0)
    end_minute = target.minute + window_minutes
    end = start.replace(
        hour=target.hour + end_minute // 60,
        minute=end_minute % 60,
    )
    return start <= now < end
