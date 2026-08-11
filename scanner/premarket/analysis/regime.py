"""Simple market-regime classification from measurable indicators."""

from __future__ import annotations

from scanner.config import PREMARKET
from scanner.premarket.models import IndexSnapshot


def classify_regime(
    nifty: IndexSnapshot,
    vix: dict | None = None,
) -> str:
    """
    Returns one of:
    Trending Bullish, Trending Bearish, Range Bound,
    High Volatility, Low Volatility, Unclear
    """
    vix = vix or {}
    vix_val = vix.get("value")

    if vix_val is not None and vix_val >= PREMARKET["vix_high"]:
        return "High Volatility"
    if vix_val is not None and vix_val <= 12:
        # still allow trend override below
        low_vol = True
    else:
        low_vol = False

    if not nifty.available:
        return "Unclear"

    sma50, sma200, close = nifty.sma50, nifty.sma200, nifty.previous_close
    if sma50 is None or sma200 is None or close is None:
        if low_vol:
            return "Low Volatility"
        return "Unclear"

    spread = abs(sma50 - sma200) / close * 100.0 if close else 0
    if spread < 0.4:
        return "Range Bound" if not low_vol else "Low Volatility"

    if sma50 > sma200 and close >= sma50:
        return "Trending Bullish"
    if sma50 < sma200 and close <= sma50:
        return "Trending Bearish"

    if low_vol:
        return "Low Volatility"
    return "Unclear"
