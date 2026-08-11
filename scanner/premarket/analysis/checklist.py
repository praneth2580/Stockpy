"""Configurable pre-market checklist evaluation."""

from __future__ import annotations

from scanner.config import PREMARKET
from scanner.premarket.models import (
    ChecklistItem,
    FIIDIISnapshot,
    GlobalSnapshot,
    IndexSnapshot,
    OptionChainSnapshot,
)


def evaluate_checklist(
    *,
    nifty: IndexSnapshot,
    bank: IndexSnapshot,
    global_snap: GlobalSnapshot,
    vix: dict,
    fii_dii: FIIDIISnapshot,
    nifty_chain: OptionChainSnapshot,
    bank_chain: OptionChainSnapshot,
    risk_flags: list[str],
) -> list[ChecklistItem]:
    items: list[ChecklistItem] = []

    def add(cid, category, label, passed, detail=""):
        items.append(ChecklistItem(cid, category, label, passed, detail))

    # Trend
    if nifty.available and nifty.expected_open is not None and nifty.previous_close is not None:
        add(
            "nifty_above_prev",
            "Trend",
            "NIFTY expected open above previous close",
            nifty.expected_open > nifty.previous_close,
            f"{nifty.expected_open} vs {nifty.previous_close}",
        )
    else:
        add("nifty_above_prev", "Trend", "NIFTY expected open above previous close", None, "unavailable")

    if bank.available and bank.expected_open is not None and bank.previous_close is not None:
        add(
            "bank_above_prev",
            "Trend",
            "BANK NIFTY expected open above previous close",
            bank.expected_open > bank.previous_close,
            f"{bank.expected_open} vs {bank.previous_close}",
        )
    else:
        add("bank_above_prev", "Trend", "BANK NIFTY expected open above previous close", None, "unavailable")

    add(
        "nifty_trend_supportive",
        "Trend",
        "Previous session trend supportive",
        True if nifty.trend == "Bullish" else False if nifty.trend == "Bearish" else None,
        nifty.trend or "unavailable",
    )

    # Global
    add(
        "gift_supportive",
        "Global Market",
        "GIFT NIFTY supportive",
        True if global_snap.gift_direction == "Positive" else False if global_snap.gift_direction == "Negative" else None,
        global_snap.gift_direction,
    )
    add(
        "us_supportive",
        "Global Market",
        "US market supportive",
        True if global_snap.us_direction == "Positive" else False if global_snap.us_direction == "Negative" else None,
        global_snap.us_direction,
    )
    add(
        "asia_supportive",
        "Global Market",
        "Asian markets supportive",
        True if global_snap.asia_direction == "Positive" else False if global_snap.asia_direction == "Negative" else None,
        global_snap.asia_direction,
    )

    # Volatility
    if vix.get("available") and vix.get("value") is not None:
        add(
            "vix_supportive",
            "Volatility",
            "India VIX supportive",
            vix["value"] < PREMARKET["vix_elevated"],
            f"VIX={vix['value']}",
        )
        add(
            "vix_no_spike",
            "Volatility",
            "Volatility not showing abnormal spike",
            vix["value"] < PREMARKET["vix_high"] and vix.get("trend") != "Rising",
            f"trend={vix.get('trend')}",
        )
    else:
        add("vix_supportive", "Volatility", "India VIX supportive", None, "unavailable")
        add("vix_no_spike", "Volatility", "Volatility not showing abnormal spike", None, "unavailable")

    # F&O
    if nifty_chain.available:
        put_ok = bool(nifty_chain.put_support_levels)
        call_ok = bool(nifty_chain.call_resistance_levels)
        add("put_oi_support", "F&O", "Put OI provides support", put_ok, str(nifty_chain.put_support_levels[:2]))
        add("call_oi_resist", "F&O", "Call OI provides resistance", call_ok, str(nifty_chain.call_resistance_levels[:2]))
        if nifty_chain.pcr_oi is not None:
            pcr = nifty_chain.pcr_oi
            add(
                "pcr_bias",
                "F&O",
                "PCR supports directional bias",
                pcr >= PREMARKET["pcr_bullish"] or pcr <= PREMARKET["pcr_bearish"],
                f"PCR={pcr}",
            )
        else:
            add("pcr_bias", "F&O", "PCR supports directional bias", None, "unavailable")
        oi_dir = None
        if nifty_chain.put_oi_change is not None and nifty_chain.call_oi_change is not None:
            oi_dir = nifty_chain.put_oi_change > nifty_chain.call_oi_change
        add("oi_change_confirm", "F&O", "OI changes confirm price direction", oi_dir, "")
    else:
        for cid, label in [
            ("put_oi_support", "Put OI provides support"),
            ("call_oi_resist", "Call OI provides resistance"),
            ("pcr_bias", "PCR supports directional bias"),
            ("oi_change_confirm", "OI changes confirm price direction"),
        ]:
            add(cid, "F&O", label, None, "DATA UNAVAILABLE")

    # Institutional
    if fii_dii.available:
        add("fii_supportive", "Institutional", "FII activity supportive", (fii_dii.fii_net or 0) > 0 if fii_dii.fii_net is not None else None, str(fii_dii.fii_net))
        add("dii_supportive", "Institutional", "DII activity supportive", (fii_dii.dii_net or 0) > 0 if fii_dii.dii_net is not None else None, str(fii_dii.dii_net))
        if fii_dii.fii_futures_net is not None:
            add("fii_fut", "Institutional", "FII futures positioning supportive", fii_dii.fii_futures_net > 0, str(fii_dii.fii_futures_net))
        else:
            add("fii_fut", "Institutional", "FII futures positioning supportive", None, "unavailable")
    else:
        add("fii_supportive", "Institutional", "FII activity supportive", None, "DATA UNAVAILABLE")
        add("dii_supportive", "Institutional", "DII activity supportive", None, "DATA UNAVAILABLE")
        add("fii_fut", "Institutional", "FII futures positioning supportive", None, "unavailable")

    # Events
    has_major = any("Important event" in f or "expiry day" in f.lower() for f in risk_flags)
    add(
        "no_major_event_risk",
        "News / Events",
        "No major event risk flagged",
        not has_major,
        "; ".join(risk_flags) if risk_flags else "none",
    )

    return items
