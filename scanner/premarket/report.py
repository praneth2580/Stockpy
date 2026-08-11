"""Plain-text and Telegram formatting for F&O pre-market reports."""

from __future__ import annotations

from typing import Any


def _fmt(val: Any, digits: int = 2, prefix: str = "", suffix: str = "") -> str:
    if val is None:
        return "DATA UNAVAILABLE"
    if isinstance(val, float):
        return f"{prefix}{val:,.{digits}f}{suffix}"
    return f"{prefix}{val}{suffix}"


def _gap_line(gap: Any, gap_pct: Any) -> str:
    if gap is None or gap_pct is None:
        return "DATA UNAVAILABLE"
    sign = "+" if gap >= 0 else ""
    return f"{sign}{gap:.2f} ({sign}{gap_pct:.2f}%)"


def _check_mark(passed: bool | None) -> str:
    if passed is True:
        return "✓"
    if passed is False:
        return "✗"
    return "—"


def format_premarket_report(report: dict) -> str:
    nifty = report.get("nifty", {})
    bank = report.get("banknifty", {})
    global_ = report.get("global", {})
    vix = report.get("vix", {})
    fii = report.get("fii_dii", {})
    oc_n = report.get("option_chain_nifty", {})
    oc_b = report.get("option_chain_banknifty", {})
    levels_n = report.get("levels_nifty", {})
    levels_b = report.get("levels_banknifty", {})
    checklist = report.get("checklist", [])
    bias = report.get("bias", {})
    scenarios = report.get("scenarios", {})
    risks = report.get("risk_flags", [])
    meta = report.get("meta", {})

    lines = [
        "========================================",
        "       F&O PRE-MARKET REPORT",
        f"       {meta.get('report_time', '09:00')} IST",
        f"       {meta.get('report_date', '')}",
        "========================================",
        "",
        "NIFTY",
        f"Previous Close:       {_fmt(nifty.get('previous_close'))}",
        f"Expected Open:        {_fmt(nifty.get('expected_open'))}",
        f"Gap:                  {_gap_line(nifty.get('gap'), nifty.get('gap_pct'))}",
        f"Classification:       {nifty.get('gap_class') or 'DATA UNAVAILABLE'}",
        f"Trend:                {nifty.get('trend') or 'DATA UNAVAILABLE'}",
        "",
        "BANK NIFTY",
        f"Previous Close:       {_fmt(bank.get('previous_close'))}",
        f"Expected Open:        {_fmt(bank.get('expected_open'))}",
        f"Gap:                  {_gap_line(bank.get('gap'), bank.get('gap_pct'))}",
        f"Classification:       {bank.get('gap_class') or 'DATA UNAVAILABLE'}",
        f"Trend:                {bank.get('trend') or 'DATA UNAVAILABLE'}",
        "",
        "GLOBAL SENTIMENT",
        f"US:                   {global_.get('us_direction', 'unavailable')}",
        f"Asia:                 {global_.get('asia_direction', 'unavailable')}",
        f"GIFT NIFTY:           {global_.get('gift_direction', 'unavailable')}",
    ]
    if global_.get("usd_inr") is not None:
        lines.append(f"USD/INR:              {_fmt(global_.get('usd_inr'))}")
    if global_.get("crude") is not None:
        lines.append(f"Crude:                {_fmt(global_.get('crude'))}")
    if global_.get("gold") is not None:
        lines.append(f"Gold:                 {_fmt(global_.get('gold'))}")

    lines += [
        "",
        "VOLATILITY",
        f"India VIX:            {_fmt(vix.get('value'))}",
        f"Trend:                {vix.get('trend') or 'DATA UNAVAILABLE'}",
        "",
        "FII / DII",
        f"FII Net:              {_fmt(fii.get('fii_net'), suffix=' Cr') if fii.get('fii_net') is not None else 'DATA UNAVAILABLE'}",
        f"DII Net:              {_fmt(fii.get('dii_net'), suffix=' Cr') if fii.get('dii_net') is not None else 'DATA UNAVAILABLE'}",
        "",
        "OPTION CHAIN",
        "",
        "NIFTY",
        f"Expiry:               {oc_n.get('expiry') or 'DATA UNAVAILABLE'}",
        f"Days to expiry:       {oc_n.get('days_to_expiry') if oc_n.get('days_to_expiry') is not None else 'DATA UNAVAILABLE'}",
        f"Expiry type:          {oc_n.get('expiry_type') or 'DATA UNAVAILABLE'}",
        f"Put Support:          {_fmt(oc_n.get('highest_put_oi_strike'), digits=0)}",
        f"Call Resistance:      {_fmt(oc_n.get('highest_call_oi_strike'), digits=0)}",
        f"PCR (OI):             {_fmt(oc_n.get('pcr_oi'), digits=3)}",
        "",
        "BANK NIFTY",
        f"Expiry:               {oc_b.get('expiry') or 'DATA UNAVAILABLE'}",
        f"Put Support:          {_fmt(oc_b.get('highest_put_oi_strike'), digits=0)}",
        f"Call Resistance:      {_fmt(oc_b.get('highest_call_oi_strike'), digits=0)}",
        f"PCR (OI):             {_fmt(oc_b.get('pcr_oi'), digits=3)}",
        "",
        "KEY LEVELS",
        "",
        "NIFTY",
        f"Support:              {_fmt(levels_n.get('immediate_support'))} / {_fmt(levels_n.get('major_support'))}",
        f"Resistance:           {_fmt(levels_n.get('immediate_resistance'))} / {_fmt(levels_n.get('major_resistance'))}",
    ]
    if levels_n.get("sources"):
        lines.append(f"Sources:              {levels_n.get('sources')}")

    lines += [
        "",
        "BANK NIFTY",
        f"Support:              {_fmt(levels_b.get('immediate_support'))} / {_fmt(levels_b.get('major_support'))}",
        f"Resistance:           {_fmt(levels_b.get('immediate_resistance'))} / {_fmt(levels_b.get('major_resistance'))}",
        "",
        f"MARKET REGIME:        {report.get('regime') or 'Unclear'}",
        "",
        "CHECKLIST",
    ]
    for item in checklist:
        mark = _check_mark(item.get("passed"))
        lines.append(f"{mark} {item.get('label')}")

    lines += [
        "",
        "----------------------------------------",
        f"PRE-MARKET BIAS: {bias.get('label', 'NEUTRAL').upper()}",
        f"CONFIDENCE: {bias.get('confidence', 0)}%",
        f"Score: {bias.get('total_score')} / ±{bias.get('max_score')}  (normalized {bias.get('normalized_score')}%)",
        f"Reason: {bias.get('confidence_reason', '')}",
        "----------------------------------------",
        "",
        "PRIMARY SCENARIO",
        scenarios.get("primary", "Wait for market-open confirmation."),
        "",
        "ALTERNATIVE SCENARIO",
        scenarios.get("alternative", "Invalidation levels will clarify after open."),
        "",
        "RISK FLAGS",
    ]
    if risks:
        for r in risks:
            lines.append(f"- {r}")
    else:
        lines.append("- None flagged")

    lines += [
        "",
        "ACTION",
        "Wait for market-open confirmation.",
        "Do not enter a trade solely from this report.",
        "This is a pre-market directional bias, not a prediction.",
        "========================================",
    ]
    return "\n".join(lines)


def format_premarket_telegram(report: dict) -> str:
    """HTML-ish plain summary for Telegram (reuse existing notifier)."""
    text = format_premarket_report(report)
    # Escape minimal HTML-sensitive chars while keeping readability
    return (
        "<b>F&amp;O Pre-Market Report</b>\n"
        "<pre>"
        + text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        + "</pre>"
    )


def build_scenarios(report: dict) -> dict:
    nifty = report.get("nifty", {})
    levels = report.get("levels_nifty", {})
    bias = report.get("bias", {}).get("label", "Neutral")
    support = levels.get("immediate_support")
    resist = levels.get("immediate_resistance")

    primary = "Wait for market-open confirmation."
    alternative = "Monitor key levels after open."

    if support is not None:
        primary = (
            f"If NIFTY holds {support:,.2f}:\n"
            f"→ continuation aligned with '{bias}' bias becomes more likely."
        )
    if resist is not None:
        alternative = (
            f"If NIFTY breaks {resist:,.2f}:\n"
            f"→ the pre-market bias may be invalidated toward the opposite side."
        )
    elif support is not None:
        alternative = (
            f"If NIFTY breaks below {support:,.2f}:\n"
            f"→ bearish follow-through becomes more likely."
        )

    prev = nifty.get("previous_close")
    if prev is not None and "Gap Down" in str(nifty.get("gap_class") or ""):
        alternative = (
            f"If NIFTY fails to reclaim {prev:,.2f}:\n"
            f"→ gap-down continuation risk stays elevated."
        )

    return {"primary": primary, "alternative": alternative}
