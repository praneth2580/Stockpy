"""Historical accuracy metrics for the pre-market bias dashboard."""

from __future__ import annotations

from collections import Counter
from typing import Any

from scanner.premarket.db import list_reports


def _normalize_bias(label: str | None) -> str:
    if not label:
        return "Neutral"
    return label.strip().title()


def _outcome_hit(bias: str, later_outcome: str | None, confirm: str | None) -> bool | None:
    """
    Prefer later_outcome when present; else use 9:30 confirmation.
    Returns True/False/None (unknown).
    """
    if later_outcome:
        lo = later_outcome.lower()
        b = bias.lower()
        if "bull" in b:
            return "bull" in lo or "up" in lo
        if "bear" in b:
            return "bear" in lo or "down" in lo
        return "flat" in lo or "neutral" in lo or "range" in lo
    if confirm:
        if confirm == "Confirmed":
            return True
        if confirm == "Invalidated":
            return False
        if confirm == "Partially confirmed":
            return None
    return None


def compute_accuracy_stats(limit: int = 500) -> dict[str, Any]:
    rows = list_reports(limit=limit)
    total = len(rows)
    bias_counts: Counter = Counter()
    confirm_counts: Counter = Counter()
    hits_by_bias: dict[str, list[bool]] = {}
    segments: dict[str, list[bool | None]] = {
        "normal_days": [],
        "expiry_days": [],
        "high_vix_days": [],
        "gap_up_days": [],
        "gap_down_days": [],
        "trending_days": [],
        "range_bound_days": [],
    }

    confirmed = 0
    confirm_known = 0
    directional_known = 0
    directional_hits = 0

    for row in rows:
        bias = _normalize_bias(row.get("bias"))
        bias_counts[bias] += 1
        payload = row.get("payload") or {}
        confirm = row.get("confirm_930_result") or payload.get("confirm_930_result")
        if confirm:
            confirm_counts[confirm] += 1
            if confirm in {"Confirmed", "Invalidated", "Partially confirmed"}:
                confirm_known += 1
                if confirm == "Confirmed":
                    confirmed += 1

        hit = _outcome_hit(bias, row.get("later_outcome"), confirm)
        hits_by_bias.setdefault(bias, [])
        if hit is not None:
            hits_by_bias[bias].append(hit)
            directional_known += 1
            if hit:
                directional_hits += 1

        regime = (row.get("market_regime") or payload.get("regime") or "").lower()
        vix = row.get("india_vix")
        gap_class = ((payload.get("nifty") or {}).get("gap_class") or "").lower()
        oc = payload.get("option_chain_nifty") or {}
        expiry_day = oc.get("expiry_type") == "expiry_day" or oc.get("days_to_expiry") == 0

        def add_seg(name: str):
            segments[name].append(hit)

        if expiry_day:
            add_seg("expiry_days")
        else:
            add_seg("normal_days")
        if vix is not None and vix >= 22:
            add_seg("high_vix_days")
        if "gap up" in gap_class:
            add_seg("gap_up_days")
        if "gap down" in gap_class:
            add_seg("gap_down_days")
        if "trending" in regime:
            add_seg("trending_days")
        if "range" in regime:
            add_seg("range_bound_days")

    def pct(hits: list[bool]) -> float | None:
        if not hits:
            return None
        return round(100.0 * sum(1 for h in hits if h) / len(hits), 1)

    def seg_pct(vals: list[bool | None]) -> float | None:
        known = [v for v in vals if v is not None]
        return pct(known)

    by_bias_accuracy = {k: pct(v) for k, v in hits_by_bias.items()}

    return {
        "total_reports": total,
        "bias_counts": dict(bias_counts),
        "confirm_930_rate": round(100.0 * confirmed / confirm_known, 1) if confirm_known else None,
        "directional_accuracy": (
            round(100.0 * directional_hits / directional_known, 1) if directional_known else None
        ),
        "directional_sample": directional_known,
        "by_bias_accuracy": by_bias_accuracy,
        "confirm_counts": dict(confirm_counts),
        "segments": {k: seg_pct(v) for k, v in segments.items()},
        "segment_samples": {k: sum(1 for x in v if x is not None) for k, v in segments.items()},
        "note": (
            "Accuracy requires stored confirmations / later outcomes. "
            "Do not claim predictive value until sample size is adequate."
        ),
    }


def format_accuracy_dashboard(stats: dict[str, Any] | None = None) -> str:
    stats = stats or compute_accuracy_stats()
    bc = stats.get("bias_counts") or {}
    lines = [
        "========================================",
        "   Pre-Market Bias Performance",
        "========================================",
        f"Total Reports:       {stats.get('total_reports', 0)}",
        f"Bullish*:            {sum(v for k, v in bc.items() if 'bullish' in k.lower())}",
        f"Bearish*:            {sum(v for k, v in bc.items() if 'bearish' in k.lower())}",
        f"Neutral:             {bc.get('Neutral', 0)}",
        "",
        f"9:30 Confirmation:   {stats.get('confirm_930_rate') if stats.get('confirm_930_rate') is not None else 'N/A'}%",
        f"Directional Accuracy:{stats.get('directional_accuracy') if stats.get('directional_accuracy') is not None else 'N/A'}% "
        f"(n={stats.get('directional_sample', 0)})",
        "",
        "By bias label:",
    ]
    for label, acc in (stats.get("by_bias_accuracy") or {}).items():
        lines.append(f"  {label:<18} {acc if acc is not None else 'N/A'}%")

    lines += ["", "By segment:"]
    samples = stats.get("segment_samples") or {}
    for name, acc in (stats.get("segments") or {}).items():
        n = samples.get(name, 0)
        lines.append(f"  {name:<18} {acc if acc is not None else 'N/A'}% (n={n})")

    lines += ["", stats.get("note", ""), "========================================"]
    return "\n".join(lines)
