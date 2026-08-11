"""Confidence estimation for pre-market directional bias."""

from __future__ import annotations


def compute_confidence(
    breakdowns: list[dict],
    checklist: list[dict] | None = None,
) -> dict:
    """
    Confidence based on data availability, agreement, and conflicts.
    Never high when most important inputs are missing.
    """
    available = [b for b in breakdowns if b.get("available", True)]
    missing = [b for b in breakdowns if not b.get("available", True)]
    total_cats = len(breakdowns) or 1
    coverage = len(available) / total_cats

    bullish = [b for b in available if b.get("signal", 0) > 0]
    bearish = [b for b in available if b.get("signal", 0) < 0]
    neutral = [b for b in available if b.get("signal", 0) == 0]

    directional = len(bullish) + len(bearish)
    if directional == 0:
        agreement = 0.5
    else:
        majority = max(len(bullish), len(bearish))
        agreement = majority / directional

    # Conflict penalty
    conflict = 0.0
    if bullish and bearish:
        conflict = min(len(bullish), len(bearish)) / max(len(bullish), len(bearish))

    # Signal strength: average |signal| / 2 among available
    if available:
        strength = sum(abs(b.get("signal", 0)) for b in available) / (len(available) * 2)
    else:
        strength = 0.0

    # Checklist availability
    checklist = checklist or []
    if checklist:
        known = [c for c in checklist if c.get("passed") is not None]
        check_cov = len(known) / len(checklist)
    else:
        check_cov = coverage

    raw = (
        0.35 * coverage
        + 0.25 * agreement
        + 0.20 * strength
        + 0.15 * check_cov
        - 0.20 * conflict
    )
    # Hard cap when coverage is poor
    if coverage < 0.4:
        raw = min(raw, 0.45)
    if coverage < 0.25:
        raw = min(raw, 0.30)

    confidence = int(max(0, min(100, round(raw * 100))))

    return {
        "confidence": confidence,
        "bullish_signals": len(bullish),
        "bearish_signals": len(bearish),
        "neutral_signals": len(neutral),
        "missing_categories": len(missing),
        "coverage": round(coverage, 3),
        "reason": (
            f"{len(bullish)} bullish signals, {len(bearish)} bearish signals, "
            f"{len(neutral)} neutral signals; {len(missing)} categories unavailable"
        ),
    }
