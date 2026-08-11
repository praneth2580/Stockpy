"""Support / resistance levels from OHLC + option OI (sources labeled)."""

from __future__ import annotations

from scanner.premarket.models import IndexSnapshot, LevelsSnapshot, OptionChainSnapshot


def build_levels(
    index: IndexSnapshot,
    chain: OptionChainSnapshot | None = None,
) -> LevelsSnapshot:
    levels = LevelsSnapshot()
    supports: list[tuple[float, str]] = []
    resistances: list[tuple[float, str]] = []

    if index.previous_low is not None:
        supports.append((index.previous_low, "previous_day_low"))
    if index.previous_close is not None:
        supports.append((index.previous_close, "previous_close"))
        resistances.append((index.previous_close, "previous_close"))
    if index.previous_high is not None:
        resistances.append((index.previous_high, "previous_day_high"))

    # Classic floor pivots from previous session
    if None not in (index.previous_high, index.previous_low, index.previous_close):
        pivot = (index.previous_high + index.previous_low + index.previous_close) / 3.0
        r1 = 2 * pivot - index.previous_low
        s1 = 2 * pivot - index.previous_high
        r2 = pivot + (index.previous_high - index.previous_low)
        s2 = pivot - (index.previous_high - index.previous_low)
        supports.append((round(s1, 2), "pivot_s1"))
        supports.append((round(s2, 2), "pivot_s2"))
        resistances.append((round(r1, 2), "pivot_r1"))
        resistances.append((round(r2, 2), "pivot_r2"))

    if chain and chain.available:
        for lvl in chain.put_support_levels:
            supports.append((lvl, "put_oi"))
        for lvl in chain.call_resistance_levels:
            resistances.append((lvl, "call_oi"))
        if chain.highest_put_oi_strike is not None:
            supports.append((chain.highest_put_oi_strike, "max_put_oi"))
        if chain.highest_call_oi_strike is not None:
            resistances.append((chain.highest_call_oi_strike, "max_call_oi"))

    ref = index.expected_open or index.indication or index.previous_close

    def pick_support(cands: list[tuple[float, str]], immediate: bool) -> tuple[float | None, str | None]:
        if ref is None:
            if not cands:
                return None, None
            cands_sorted = sorted(cands, key=lambda x: x[0], reverse=True)
            return cands_sorted[0] if immediate else cands_sorted[min(1, len(cands_sorted) - 1)]
        below = [(v, s) for v, s in cands if v < ref]
        if not below:
            return None, None
        below_sorted = sorted(below, key=lambda x: x[0], reverse=True)
        if immediate:
            return below_sorted[0]
        return below_sorted[min(1, len(below_sorted) - 1)]

    def pick_resistance(cands: list[tuple[float, str]], immediate: bool) -> tuple[float | None, str | None]:
        if ref is None:
            if not cands:
                return None, None
            cands_sorted = sorted(cands, key=lambda x: x[0])
            return cands_sorted[0] if immediate else cands_sorted[min(1, len(cands_sorted) - 1)]
        above = [(v, s) for v, s in cands if v > ref]
        if not above:
            return None, None
        above_sorted = sorted(above, key=lambda x: x[0])
        if immediate:
            return above_sorted[0]
        return above_sorted[min(1, len(above_sorted) - 1)]

    imm_s, src_is = pick_support(supports, True)
    maj_s, src_ms = pick_support(supports, False)
    imm_r, src_ir = pick_resistance(resistances, True)
    maj_r, src_mr = pick_resistance(resistances, False)

    levels.immediate_support = imm_s
    levels.major_support = maj_s
    levels.immediate_resistance = imm_r
    levels.major_resistance = maj_r
    sources = {}
    if src_is:
        sources["immediate_support"] = src_is
    if src_ms:
        sources["major_support"] = src_ms
    if src_ir:
        sources["immediate_resistance"] = src_ir
    if src_mr:
        sources["major_resistance"] = src_mr
    levels.sources = sources
    return levels
