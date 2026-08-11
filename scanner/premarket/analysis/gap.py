"""Gap calculation helpers (pure, no network)."""

from __future__ import annotations

from scanner.config import PREMARKET


def classify_gap(gap_pct: float | None, cfg: dict | None = None) -> str | None:
    """Classify opening gap using configurable percentage thresholds."""
    if gap_pct is None:
        return None
    cfg = cfg or PREMARKET
    flat = cfg["gap_flat_pct"]
    small = cfg["gap_small_pct"]
    moderate = cfg["gap_moderate_pct"]
    abs_pct = abs(gap_pct)

    if abs_pct < flat:
        return "Flat"
    direction_up = gap_pct > 0
    if abs_pct < small:
        return "Small Gap Up" if direction_up else "Small Gap Down"
    if abs_pct < moderate:
        return "Moderate Gap Up" if direction_up else "Moderate Gap Down"
    return "Strong Gap Up" if direction_up else "Strong Gap Down"


def compute_gap(
    previous_close: float | None,
    expected_open: float | None,
) -> tuple[float | None, float | None, str | None]:
    if previous_close is None or expected_open is None or previous_close == 0:
        return None, None, None
    gap = expected_open - previous_close
    gap_pct = (gap / previous_close) * 100.0
    return round(gap, 2), round(gap_pct, 4), classify_gap(gap_pct)
