"""9:15 open snapshot and 9:30 confirmation against pre-market bias."""

from __future__ import annotations

import logging
from typing import Any

from scanner.config import PREMARKET
from scanner.premarket.calendar import is_trading_day, now_ist
from scanner.premarket.db import finish_job, get_report, try_acquire_job, update_confirmation
from scanner.premarket.retry import retry_call

logger = logging.getLogger(__name__)


def _latest_open_and_price(symbol: str) -> dict[str, Any]:
    def _load() -> dict[str, Any]:
        import yfinance as yf

        t = yf.Ticker(symbol)
        # Intraday when available
        df = t.history(period="1d", interval="1m")
        if df is None or df.empty:
            df = t.history(period="5d")
        if df is None or df.empty:
            raise ValueError(f"No data for {symbol}")
        open_px = float(df["Open"].iloc[0])
        last_px = float(df["Close"].iloc[-1])
        volume = float(df["Volume"].sum()) if "Volume" in df.columns else None
        return {
            "open": round(open_px, 2),
            "last": round(last_px, 2),
            "volume": volume,
            "available": True,
        }

    return retry_call(
        _load,
        retries=PREMARKET["api_retries"],
        backoff=PREMARKET["api_retry_backoff"],
        label=f"open_snapshot:{symbol}",
    ) or {"available": False, "error": "DATA UNAVAILABLE"}


def _bias_direction(label: str | None) -> str:
    if not label:
        return "neutral"
    low = label.lower()
    if "bullish" in low:
        return "bullish"
    if "bearish" in low:
        return "bearish"
    return "neutral"


def _move_direction(expected: float | None, actual: float | None, last: float | None) -> str:
    ref = expected
    px = last if last is not None else actual
    if ref is None or px is None:
        return "unavailable"
    chg_pct = (px - ref) / ref * 100.0
    if abs(chg_pct) < 0.05:
        return "flat"
    return "bullish" if chg_pct > 0 else "bearish"


def compare_bias(premarket_bias: str | None, move: str) -> str:
    """Return Confirmed / Partially confirmed / Invalidated / Insufficient data."""
    bias_dir = _bias_direction(premarket_bias)
    if move == "unavailable" or bias_dir == "neutral" and move == "flat":
        if move == "unavailable":
            return "Insufficient data"
        return "Partially confirmed"
    if bias_dir == "neutral":
        return "Partially confirmed" if move == "flat" else "Invalidated"
    if move == bias_dir:
        return "Confirmed"
    if move == "flat":
        return "Partially confirmed"
    return "Invalidated"


def run_open_snapshot(*, force: bool = False) -> dict[str, Any]:
    """9:15 AM market-open snapshot vs pre-market expectations."""
    now = now_ist(PREMARKET["timezone"])
    report_date = now.date().isoformat()
    job_type = "open_915"

    if not force and not is_trading_day(now.date()):
        return {"skipped": True, "reason": "Non-trading day"}

    if not force and not try_acquire_job(job_type, report_date):
        return {"skipped": True, "reason": "Duplicate execution blocked"}

    prem = get_report(report_date, "premarket")
    prem_payload = (prem or {}).get("payload") or {}
    bias_label = ((prem_payload.get("bias") or {}).get("label")) if prem_payload else None

    nifty = _latest_open_and_price(PREMARKET["nifty_symbol"])
    bank = _latest_open_and_price(PREMARKET["banknifty_symbol"])

    expected_n = (prem_payload.get("nifty") or {}).get("expected_open")
    expected_b = (prem_payload.get("banknifty") or {}).get("expected_open")

    nifty_move = _move_direction(expected_n or (prem_payload.get("nifty") or {}).get("previous_close"), nifty.get("open"), nifty.get("last"))
    result = compare_bias(bias_label, nifty_move)

    gap_error = None
    if expected_n is not None and nifty.get("open") is not None:
        gap_error = round(nifty["open"] - expected_n, 2)

    snapshot = {
        "meta": {"report_date": report_date, "job_type": job_type, "generated_at": now.isoformat()},
        "premarket_bias": bias_label,
        "nifty": nifty,
        "banknifty": bank,
        "expected_vs_actual_nifty": {
            "expected": expected_n,
            "actual_open": nifty.get("open"),
            "error": gap_error,
        },
        "expected_vs_actual_banknifty": {
            "expected": expected_b,
            "actual_open": bank.get("open"),
            "error": (round(bank["open"] - expected_b, 2) if expected_b is not None and bank.get("open") is not None else None),
        },
        "open_915_result": result,
        "text": (
            f"9:15 Open Snapshot ({report_date})\n"
            f"Pre-market bias: {bias_label or 'N/A'}\n"
            f"NIFTY open: {nifty.get('open', 'UNAVAILABLE')} (expected {expected_n})\n"
            f"BANKNIFTY open: {bank.get('open', 'UNAVAILABLE')} (expected {expected_b})\n"
            f"vs bias: {result}\n"
            f"Hypothesis only — not a trade signal."
        ),
    }

    update_confirmation(
        report_date,
        open_915_result=result,
        nifty_actual_open=nifty.get("open"),
        banknifty_actual_open=bank.get("open"),
        confirmation_payload={"open_915": snapshot},
    )
    finish_job(job_type, report_date, "success")
    return snapshot


def run_confirmation(*, force: bool = False) -> dict[str, Any]:
    """9:30 AM confirmation check."""
    now = now_ist(PREMARKET["timezone"])
    report_date = now.date().isoformat()
    job_type = "confirm_930"

    if not force and not is_trading_day(now.date()):
        return {"skipped": True, "reason": "Non-trading day"}

    if not force and not try_acquire_job(job_type, report_date):
        return {"skipped": True, "reason": "Duplicate execution blocked"}

    prem = get_report(report_date, "premarket")
    prem_payload = (prem or {}).get("payload") or {}
    bias_label = ((prem_payload.get("bias") or {}).get("label")) if prem_payload else None
    open_915 = prem_payload.get("open_915_result") or ((prem or {}).get("open_915_result"))

    nifty = _latest_open_and_price(PREMARKET["nifty_symbol"])
    prev_close = (prem_payload.get("nifty") or {}).get("previous_close")
    move = _move_direction(prev_close, nifty.get("open"), nifty.get("last"))
    result = compare_bias(bias_label, move)

    # Combine with 9:15
    if open_915 == "Confirmed" and result == "Confirmed":
        combined = "Confirmed"
    elif "Invalidated" in (open_915 or "", result):
        combined = "Invalidated"
    elif result == "Insufficient data":
        combined = "Insufficient data"
    else:
        combined = "Partially confirmed"

    confirmation = {
        "meta": {"report_date": report_date, "job_type": job_type, "generated_at": now.isoformat()},
        "premarket_bias": bias_label,
        "open_915_result": open_915,
        "confirm_930_result": combined,
        "nifty": nifty,
        "move_vs_prev_close": move,
        "text": (
            f"9:30 Confirmation ({report_date})\n"
            f"Pre-market bias → Open → 9:30: {combined}\n"
            f"Bias: {bias_label or 'N/A'}\n"
            f"NIFTY last: {nifty.get('last', 'UNAVAILABLE')}\n"
            f"Treat 9:00 report as a hypothesis, not a signal."
        ),
    }

    update_confirmation(
        report_date,
        confirm_930_result=combined,
        nifty_actual_open=nifty.get("open"),
        confirmation_payload={"confirm_930": confirmation},
    )
    finish_job(job_type, report_date, "success")
    return confirmation
