"""Index / VIX data collection via yfinance (reuses project data source)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import yfinance as yf

from scanner.config import PREMARKET
from scanner.indicators import calculate_indicators
from scanner.premarket.analysis.gap import classify_gap, compute_gap
from scanner.premarket.models import IndexSnapshot
from scanner.premarket.retry import retry_call

logger = logging.getLogger(__name__)

__all__ = ["classify_gap", "compute_gap", "fetch_index_snapshot", "fetch_vix", "fetch_nifty_and_banknifty"]


def _safe_float(val: Any) -> float | None:
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _trend_from_indicators(sma50: float | None, sma200: float | None, close: float | None) -> str:
    if sma50 is None or sma200 is None or close is None:
        return "unavailable"
    if sma50 > sma200 and close >= sma50:
        return "Bullish"
    if sma50 < sma200 and close <= sma50:
        return "Bearish"
    return "Neutral"


def fetch_index_snapshot(
    symbol: str,
    name: str,
    *,
    expected_open: float | None = None,
    period: str = "1y",
) -> IndexSnapshot:
    """Fetch previous session OHLC + indicators for an index."""

    def _load() -> IndexSnapshot:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df is None or df.empty:
            return IndexSnapshot(symbol=symbol, name=name, error="No historical data", available=False)

        df = calculate_indicators(df)
        prev = df.iloc[-1]
        # If market already open today, last row is today — use previous completed day when possible
        if len(df) >= 2:
            # Prefer last complete bar as previous session
            prev = df.iloc[-1]

        previous_close = _safe_float(prev["Close"])
        previous_open = _safe_float(prev["Open"])
        previous_high = _safe_float(prev["High"])
        previous_low = _safe_float(prev["Low"])
        previous_volume = _safe_float(prev["Volume"]) if "Volume" in prev else None

        sma50 = _safe_float(prev["SMA50"]) if "SMA50" in df.columns else None
        sma200 = _safe_float(prev["SMA200"]) if "SMA200" in df.columns else None
        rsi = _safe_float(prev["RSI"]) if "RSI" in df.columns else None

        # Live indication when available (may equal last close pre-open)
        indication = None
        try:
            info = getattr(ticker, "fast_info", None)
            if info is not None:
                indication = _safe_float(getattr(info, "last_price", None) or info.get("lastPrice"))
        except Exception:
            indication = None

        exp = expected_open if expected_open is not None else indication
        gap, gap_pct, gap_class = compute_gap(previous_close, exp)

        return IndexSnapshot(
            symbol=symbol,
            name=name,
            previous_close=round(previous_close, 2) if previous_close is not None else None,
            previous_open=round(previous_open, 2) if previous_open is not None else None,
            previous_high=round(previous_high, 2) if previous_high is not None else None,
            previous_low=round(previous_low, 2) if previous_low is not None else None,
            previous_volume=previous_volume,
            expected_open=round(exp, 2) if exp is not None else None,
            indication=round(indication, 2) if indication is not None else None,
            gap=gap,
            gap_pct=gap_pct,
            gap_class=gap_class,
            trend=_trend_from_indicators(sma50, sma200, previous_close),
            sma50=round(sma50, 2) if sma50 is not None else None,
            sma200=round(sma200, 2) if sma200 is not None else None,
            rsi=round(rsi, 2) if rsi is not None else None,
            available=True,
        )

    result = retry_call(
        _load,
        retries=PREMARKET["api_retries"],
        backoff=PREMARKET["api_retry_backoff"],
        label=f"index:{symbol}",
    )
    if result is None:
        return IndexSnapshot(symbol=symbol, name=name, available=False, error="DATA UNAVAILABLE")
    return result


def fetch_vix() -> dict:
    symbol = PREMARKET["india_vix_symbol"]

    def _load() -> dict:
        df = yf.Ticker(symbol).history(period="1mo")
        if df is None or df.empty:
            return {"available": False, "error": "DATA UNAVAILABLE", "value": None}
        latest = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else None
        change = (latest - prev) if prev is not None else None
        trend = "unavailable"
        if change is not None:
            if change < -0.3:
                trend = "Falling"
            elif change > 0.3:
                trend = "Rising"
            else:
                trend = "Flat"
        return {
            "available": True,
            "value": round(latest, 2),
            "previous": round(prev, 2) if prev is not None else None,
            "change": round(change, 2) if change is not None else None,
            "trend": trend,
            "symbol": symbol,
        }

    result = retry_call(
        _load,
        retries=PREMARKET["api_retries"],
        backoff=PREMARKET["api_retry_backoff"],
        label="india_vix",
    )
    return result or {"available": False, "error": "DATA UNAVAILABLE", "value": None}


def fetch_nifty_and_banknifty(gift_indication: float | None = None) -> tuple[IndexSnapshot, IndexSnapshot]:
    nifty = fetch_index_snapshot(
        PREMARKET["nifty_symbol"],
        "NIFTY",
        expected_open=gift_indication,
    )
    bank = fetch_index_snapshot(PREMARKET["banknifty_symbol"], "BANK NIFTY")
    # If gift used for nifty expected open, recompute gap fields already done in fetch
    return nifty, bank
