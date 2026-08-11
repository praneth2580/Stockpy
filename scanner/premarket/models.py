"""Dataclasses / typed dict helpers for pre-market reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class IndexSnapshot:
    symbol: str
    name: str
    previous_close: float | None = None
    previous_open: float | None = None
    previous_high: float | None = None
    previous_low: float | None = None
    previous_volume: float | None = None
    expected_open: float | None = None
    indication: float | None = None
    gap: float | None = None
    gap_pct: float | None = None
    gap_class: str | None = None
    trend: str | None = None
    sma50: float | None = None
    sma200: float | None = None
    rsi: float | None = None
    available: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GlobalSnapshot:
    us_direction: str = "unavailable"
    asia_direction: str = "unavailable"
    gift_nifty: float | None = None
    gift_nifty_change_pct: float | None = None
    gift_direction: str = "unavailable"
    indices: dict[str, dict[str, Any]] = field(default_factory=dict)
    usd_inr: float | None = None
    crude: float | None = None
    gold: float | None = None
    available: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FIIDIISnapshot:
    fii_buy: float | None = None
    fii_sell: float | None = None
    fii_net: float | None = None
    dii_buy: float | None = None
    dii_sell: float | None = None
    dii_net: float | None = None
    prev_fii_net: float | None = None
    prev_dii_net: float | None = None
    fii_5d_trend: str | None = None
    dii_5d_trend: str | None = None
    fii_futures_net: float | None = None
    fii_options_net: float | None = None
    as_of: str | None = None
    available: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OptionChainSnapshot:
    symbol: str
    expiry: str | None = None
    days_to_expiry: int | None = None
    expiry_type: str | None = None  # weekly / monthly / expiry_day
    spot: float | None = None
    highest_call_oi_strike: float | None = None
    highest_put_oi_strike: float | None = None
    call_oi_change: float | None = None
    put_oi_change: float | None = None
    call_resistance_levels: list[float] = field(default_factory=list)
    put_support_levels: list[float] = field(default_factory=list)
    pcr_oi: float | None = None
    pcr_volume: float | None = None
    important_strikes: list[float] = field(default_factory=list)
    total_call_oi: float | None = None
    total_put_oi: float | None = None
    available: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LevelsSnapshot:
    immediate_support: float | None = None
    major_support: float | None = None
    immediate_resistance: float | None = None
    major_resistance: float | None = None
    sources: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChecklistItem:
    id: str
    category: str
    label: str
    passed: bool | None  # None = unavailable / N/A
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreBreakdown:
    category: str
    signal: int  # -2 .. +2
    weight: float
    contribution: float
    reason: str
    available: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def unavailable(msg: str = "DATA UNAVAILABLE") -> dict[str, Any]:
    return {"available": False, "error": msg}
