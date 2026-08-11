"""Deterministic pre-market scoring engine (transparent, configurable weights)."""

from __future__ import annotations

from scanner.config import PREMARKET
from scanner.premarket.models import (
    FIIDIISnapshot,
    GlobalSnapshot,
    IndexSnapshot,
    OptionChainSnapshot,
    ScoreBreakdown,
)


def _clamp_signal(value: int) -> int:
    return max(-2, min(2, value))


def score_gap(nifty: IndexSnapshot) -> ScoreBreakdown:
    w = PREMARKET["score_weights"]["gap"]
    if not nifty.available or nifty.gap_pct is None:
        return ScoreBreakdown("gap", 0, w, 0.0, "Gap data unavailable", available=False)
    g = nifty.gap_pct
    flat = PREMARKET["gap_flat_pct"]
    small = PREMARKET["gap_small_pct"]
    moderate = PREMARKET["gap_moderate_pct"]
    if abs(g) < flat:
        sig = 0
        reason = "Flat gap"
    elif g >= moderate:
        sig = 2
        reason = f"Strong gap up ({g:.2f}%)"
    elif g >= small:
        sig = 1
        reason = f"Moderate/small gap up ({g:.2f}%)"
    elif g <= -moderate:
        sig = -2
        reason = f"Strong gap down ({g:.2f}%)"
    elif g <= -small:
        sig = -1
        reason = f"Moderate/small gap down ({g:.2f}%)"
    else:
        sig = 1 if g > 0 else -1
        reason = f"Small gap ({g:.2f}%)"
    return ScoreBreakdown("gap", sig, w, sig * w, reason)


def score_trend(name: str, index: IndexSnapshot, weight_key: str) -> ScoreBreakdown:
    w = PREMARKET["score_weights"][weight_key]
    if not index.available or index.trend in (None, "unavailable"):
        return ScoreBreakdown(name, 0, w, 0.0, f"{index.name} trend unavailable", available=False)
    mapping = {"Bullish": 1, "Bearish": -1, "Neutral": 0}
    sig = mapping.get(index.trend, 0)
    return ScoreBreakdown(name, sig, w, sig * w, f"{index.name} trend: {index.trend}")


def score_global(global_snap: GlobalSnapshot) -> ScoreBreakdown:
    w = PREMARKET["score_weights"]["global_markets"]
    if not global_snap.available:
        return ScoreBreakdown("global_markets", 0, w, 0.0, "Global data unavailable", available=False)
    score = 0
    parts = []
    for label, direction in (("US", global_snap.us_direction), ("Asia", global_snap.asia_direction)):
        if direction == "Positive":
            score += 1
            parts.append(f"{label}+")
        elif direction == "Negative":
            score -= 1
            parts.append(f"{label}-")
        elif direction == "Mixed":
            parts.append(f"{label}~")
        else:
            parts.append(f"{label}?")
    sig = _clamp_signal(score)
    return ScoreBreakdown("global_markets", sig, w, sig * w, " ".join(parts), available=True)


def score_gift(global_snap: GlobalSnapshot) -> ScoreBreakdown:
    w = PREMARKET["score_weights"]["gift_nifty"]
    d = global_snap.gift_direction
    if d == "unavailable":
        return ScoreBreakdown("gift_nifty", 0, w, 0.0, "GIFT Nifty unavailable", available=False)
    mapping = {"Positive": 1, "Negative": -1, "Flat": 0, "Mixed": 0}
    sig = mapping.get(d, 0)
    return ScoreBreakdown("gift_nifty", sig, w, sig * w, f"GIFT: {d}")


def score_vix(vix: dict) -> ScoreBreakdown:
    w = PREMARKET["score_weights"]["india_vix"]
    if not vix.get("available") or vix.get("value") is None:
        return ScoreBreakdown("india_vix", 0, w, 0.0, "VIX unavailable", available=False)
    val = vix["value"]
    trend = vix.get("trend")
    if val >= PREMARKET["vix_high"]:
        sig = -2
        reason = f"High VIX {val}"
    elif val >= PREMARKET["vix_elevated"]:
        sig = -1
        reason = f"Elevated VIX {val}"
    elif trend == "Falling":
        sig = 1
        reason = f"VIX falling ({val})"
    elif trend == "Rising":
        sig = -1
        reason = f"VIX rising ({val})"
    else:
        sig = 0
        reason = f"VIX stable ({val})"
    return ScoreBreakdown("india_vix", sig, w, sig * w, reason)


def score_fii_dii(fii: FIIDIISnapshot) -> ScoreBreakdown:
    w = PREMARKET["score_weights"]["fii_dii"]
    if not fii.available:
        return ScoreBreakdown("fii_dii", 0, w, 0.0, "FII/DII unavailable", available=False)
    score = 0
    parts = []
    if fii.fii_net is not None:
        score += 1 if fii.fii_net > 0 else -1 if fii.fii_net < 0 else 0
        parts.append(f"FII net {fii.fii_net}")
    if fii.dii_net is not None:
        score += 1 if fii.dii_net > 0 else -1 if fii.dii_net < 0 else 0
        parts.append(f"DII net {fii.dii_net}")
    if fii.fii_net is None and fii.dii_net is None:
        return ScoreBreakdown("fii_dii", 0, w, 0.0, "FII/DII nets missing", available=False)
    sig = _clamp_signal(score)
    return ScoreBreakdown("fii_dii", sig, w, sig * w, "; ".join(parts))


def score_option_oi(chain: OptionChainSnapshot) -> ScoreBreakdown:
    w = PREMARKET["score_weights"]["option_oi"]
    if not chain.available:
        return ScoreBreakdown("option_oi", 0, w, 0.0, "Option chain unavailable", available=False)
    # Higher put support below / call resistance above is structural, not directional alone.
    # Use relative OI: put OI dominance mildly bullish for cushions.
    if chain.total_put_oi and chain.total_call_oi:
        ratio = chain.total_put_oi / chain.total_call_oi
        if ratio >= 1.2:
            sig = 1
            reason = f"Put OI dominance (ratio {ratio:.2f})"
        elif ratio <= 0.8:
            sig = -1
            reason = f"Call OI dominance (ratio {ratio:.2f})"
        else:
            sig = 0
            reason = f"Balanced OI (ratio {ratio:.2f})"
        return ScoreBreakdown("option_oi", sig, w, sig * w, reason)
    return ScoreBreakdown("option_oi", 0, w, 0.0, "OI totals missing", available=False)


def score_pcr(chain: OptionChainSnapshot) -> ScoreBreakdown:
    w = PREMARKET["score_weights"]["pcr"]
    if not chain.available or chain.pcr_oi is None:
        return ScoreBreakdown("pcr", 0, w, 0.0, "PCR unavailable", available=False)
    pcr = chain.pcr_oi
    # Around expiry, dampen signal
    damp = 0.5 if chain.expiry_type == "expiry_day" else 1.0
    if pcr >= PREMARKET["pcr_bullish"] + 0.2:
        sig = 2
    elif pcr >= PREMARKET["pcr_bullish"]:
        sig = 1
    elif pcr <= PREMARKET["pcr_bearish"] - 0.15:
        sig = -2
    elif pcr <= PREMARKET["pcr_bearish"]:
        sig = -1
    else:
        sig = 0
    if damp < 1 and sig != 0:
        sig = 1 if sig > 0 else -1
        reason = f"PCR={pcr} (dampened: expiry day)"
    else:
        reason = f"PCR={pcr}"
    return ScoreBreakdown("pcr", sig, w, sig * w, reason)


def score_oi_change(chain: OptionChainSnapshot) -> ScoreBreakdown:
    w = PREMARKET["score_weights"]["oi_change"]
    if not chain.available or chain.put_oi_change is None or chain.call_oi_change is None:
        return ScoreBreakdown("oi_change", 0, w, 0.0, "OI change unavailable", available=False)
    diff = chain.put_oi_change - chain.call_oi_change
    if abs(diff) < 1e-6:
        sig = 0
        reason = "OI change balanced"
    elif diff > 0:
        sig = 1
        reason = "Put OI increased more than Call OI"
    else:
        sig = -1
        reason = "Call OI increased more than Put OI"
    if chain.expiry_type == "expiry_day":
        reason += " (expiry caution)"
    return ScoreBreakdown("oi_change", sig, w, sig * w, reason)


def score_support_resistance(nifty: IndexSnapshot, chain: OptionChainSnapshot) -> ScoreBreakdown:
    w = PREMARKET["score_weights"]["support_resistance"]
    ref = nifty.expected_open or nifty.indication or nifty.previous_close
    if ref is None or not chain.available:
        return ScoreBreakdown("support_resistance", 0, w, 0.0, "S/R context unavailable", available=False)
    put = chain.highest_put_oi_strike
    call = chain.highest_call_oi_strike
    if put is None or call is None:
        return ScoreBreakdown("support_resistance", 0, w, 0.0, "Max OI strikes missing", available=False)
    # Closer to put support → milder bullish cushion; near call wall → resistance
    dist_put = abs(ref - put)
    dist_call = abs(call - ref)
    if dist_put < dist_call * 0.6:
        sig = 1
        reason = f"Closer to put support {put}"
    elif dist_call < dist_put * 0.6:
        sig = -1
        reason = f"Closer to call resistance {call}"
    else:
        sig = 0
        reason = f"Between put {put} and call {call}"
    return ScoreBreakdown("support_resistance", sig, w, sig * w, reason)


def score_events(risk_flags: list[str]) -> ScoreBreakdown:
    w = PREMARKET["score_weights"]["events"]
    if any("expiry day" in f.lower() for f in risk_flags):
        return ScoreBreakdown("events", -1, w, -1 * w, "Expiry day risk")
    if any("Important event" in f for f in risk_flags):
        return ScoreBreakdown("events", -1, w, -1 * w, "Major event risk")
    return ScoreBreakdown("events", 0, w, 0.0, "No major event penalty", available=True)


def classify_bias(normalized_0_100: float) -> str:
    bands = PREMARKET["bias_bands"]
    for label, (lo, hi) in bands.items():
        if lo <= normalized_0_100 < hi or (label == "strong_bullish" and normalized_0_100 == 100):
            return label.replace("_", " ").title()
    return "Neutral"


def compute_scores(
    *,
    nifty: IndexSnapshot,
    bank: IndexSnapshot,
    global_snap: GlobalSnapshot,
    vix: dict,
    fii_dii: FIIDIISnapshot,
    nifty_chain: OptionChainSnapshot,
    risk_flags: list[str],
) -> dict:
    breakdowns = [
        score_gap(nifty),
        score_trend("nifty_trend", nifty, "nifty_trend"),
        score_trend("banknifty_trend", bank, "banknifty_trend"),
        score_global(global_snap),
        score_gift(global_snap),
        score_vix(vix),
        score_fii_dii(fii_dii),
        score_option_oi(nifty_chain),
        score_pcr(nifty_chain),
        score_oi_change(nifty_chain),
        score_support_resistance(nifty, nifty_chain),
        score_events(risk_flags),
    ]

    total = sum(b.contribution for b in breakdowns if b.available)
    # Max possible = sum of weights for available categories (signal ±1 baseline for max magnitude uses weight*2)
    max_score = sum(b.weight * 2 for b in breakdowns if b.available)
    if max_score <= 0:
        normalized = 50.0
    else:
        # Map [-max, +max] → [0, 100]
        normalized = ((total + max_score) / (2 * max_score)) * 100.0
        normalized = max(0.0, min(100.0, normalized))

    label = classify_bias(normalized)
    return {
        "breakdowns": [b.to_dict() for b in breakdowns],
        "total_score": round(total, 3),
        "max_score": round(max_score, 3),
        "normalized_score": round(normalized, 2),
        "label": label,
    }
