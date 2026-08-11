"""Global markets, FX, and commodities via yfinance."""

from __future__ import annotations

import logging
from typing import Any

import yfinance as yf

from scanner.config import PREMARKET
from scanner.premarket.models import GlobalSnapshot
from scanner.premarket.retry import retry_call

logger = logging.getLogger(__name__)

US_SYMBOLS = {
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
    "Nasdaq": "^IXIC",
}
ASIA_SYMBOLS = {
    "Nikkei 225": "^N225",
    "Hang Seng": "^HSI",
    "Shanghai Comp": "000001.SS",
}
OTHER = {
    "USD/INR": "INR=X",
    "Crude Oil": "CL=F",
    "Gold": "GC=F",
}


def _change_pct(df) -> float | None:
    if df is None or df.empty or len(df) < 1:
        return None
    close = float(df["Close"].iloc[-1])
    # Prefer previous close for overnight change
    if len(df) >= 2:
        prev = float(df["Close"].iloc[-2])
        if prev:
            return ((close - prev) / prev) * 100.0
    open_ = float(df["Open"].iloc[-1])
    if open_:
        return ((close - open_) / open_) * 100.0
    return None


def _direction_from_changes(changes: list[float | None]) -> str:
    vals = [c for c in changes if c is not None]
    if not vals:
        return "unavailable"
    up = sum(1 for c in vals if c > 0.05)
    down = sum(1 for c in vals if c < -0.05)
    if up > down and up >= max(1, len(vals) // 2):
        return "Positive"
    if down > up and down >= max(1, len(vals) // 2):
        return "Negative"
    if up == 0 and down == 0:
        return "Flat"
    return "Mixed"


def _fetch_one(symbol: str) -> dict[str, Any]:
    def _load() -> dict[str, Any]:
        df = yf.Ticker(symbol).history(period="5d")
        if df is None or df.empty:
            raise ValueError(f"empty history for {symbol}")
        close = float(df["Close"].iloc[-1])
        chg = _change_pct(df)
        return {
            "symbol": symbol,
            "close": round(close, 2),
            "change_pct": round(chg, 3) if chg is not None else None,
            "available": True,
        }

    result = retry_call(
        _load,
        retries=PREMARKET["api_retries"],
        backoff=PREMARKET["api_retry_backoff"],
        label=f"global:{symbol}",
    )
    return result or {"symbol": symbol, "available": False, "error": "DATA UNAVAILABLE"}


def fetch_gift_nifty() -> dict[str, Any]:
    """
    Attempt GIFT Nifty / overnight indication.
    Only uses an explicitly configured symbol — Yahoo symbols for GIFT are unreliable.
    """
    configured = (PREMARKET.get("gift_nifty_symbol") or "").strip()
    if not configured:
        return {"available": False, "error": "DATA UNAVAILABLE", "value": None, "change_pct": None}

    def _load() -> dict[str, Any]:
        t = yf.Ticker(configured)
        df = t.history(period="5d")
        if df is None or df.empty:
            raise ValueError(f"empty {configured}")
        close = float(df["Close"].iloc[-1])
        chg = _change_pct(df)
        return {
            "symbol": configured,
            "value": round(close, 2),
            "change_pct": round(chg, 3) if chg is not None else None,
            "available": True,
        }

    result = retry_call(
        _load,
        retries=2,
        backoff=1.0,
        label=f"gift_nifty:{configured}",
    )
    if result and result.get("available"):
        return result
    return {"available": False, "error": "DATA UNAVAILABLE", "value": None, "change_pct": None}


def fetch_global_snapshot() -> GlobalSnapshot:
    snap = GlobalSnapshot()
    indices: dict[str, dict] = {}

    us_changes = []
    for name, sym in US_SYMBOLS.items():
        data = _fetch_one(sym)
        indices[name] = data
        if not data.get("available"):
            snap.errors.append(f"{name}: UNAVAILABLE")
        us_changes.append(data.get("change_pct"))

    asia_changes = []
    for name, sym in ASIA_SYMBOLS.items():
        data = _fetch_one(sym)
        indices[name] = data
        if not data.get("available"):
            snap.errors.append(f"{name}: UNAVAILABLE")
        asia_changes.append(data.get("change_pct"))

    for name, sym in OTHER.items():
        data = _fetch_one(sym)
        indices[name] = data
        if data.get("available"):
            if name == "USD/INR":
                snap.usd_inr = data.get("close")
            elif name == "Crude Oil":
                snap.crude = data.get("close")
            elif name == "Gold":
                snap.gold = data.get("close")
        else:
            snap.errors.append(f"{name}: UNAVAILABLE")

    gift = fetch_gift_nifty()
    if gift.get("available"):
        snap.gift_nifty = gift.get("value")
        snap.gift_nifty_change_pct = gift.get("change_pct")
        chg = gift.get("change_pct")
        if chg is None:
            snap.gift_direction = "unavailable"
        elif chg > 0.1:
            snap.gift_direction = "Positive"
        elif chg < -0.1:
            snap.gift_direction = "Negative"
        else:
            snap.gift_direction = "Flat"
    else:
        snap.gift_direction = "unavailable"
        snap.errors.append("GIFT Nifty: UNAVAILABLE")

    snap.us_direction = _direction_from_changes(us_changes)
    snap.asia_direction = _direction_from_changes(asia_changes)
    snap.indices = indices
    snap.available = any(v.get("available") for v in indices.values()) or bool(gift.get("available"))
    return snap
