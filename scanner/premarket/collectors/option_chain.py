"""NSE index option-chain collection (best-effort, failure-isolated)."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import requests

from scanner.config import PREMARKET
from scanner.premarket.calendar import now_ist
from scanner.premarket.models import OptionChainSnapshot
from scanner.premarket.retry import retry_call

logger = logging.getLogger(__name__)

NSE_HOME = "https://www.nseindia.com"
OPTION_CHAIN_URLS = [
    "https://www.nseindia.com/api/option-chain-indices?symbol={symbol}",
    "https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol={symbol}",
]

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain",
    "Origin": "https://www.nseindia.com",
    "Connection": "keep-alive",
}


def _nse_session(symbol: str) -> requests.Session:
    sess = requests.Session()
    sess.headers.update(NSE_HEADERS)
    timeout = PREMARKET["api_timeout"]
    sess.get(NSE_HOME, timeout=timeout)
    sess.get(f"{NSE_HOME}/option-chain", timeout=timeout)
    sess.get(f"{NSE_HOME}/option-chain?symbol={symbol}&type=Indices", timeout=timeout)
    return sess


def _parse_expiry(value: str) -> date | None:
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%b-%y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def classify_expiry(expiry: date, today: date | None = None) -> tuple[int, str]:
    today = today or now_ist().date()
    days = (expiry - today).days
    if days == 0:
        expiry_type = "expiry_day"
    elif days <= 7:
        expiry_type = "weekly"
    else:
        # Monthly typically last Thursday; treat >7 as monthly-ish
        expiry_type = "monthly" if days > 7 else "weekly"
    return days, expiry_type


def important_strikes_around(
    spot: float,
    interval: int,
    radius: int | None = None,
) -> list[float]:
    """Dynamic strikes around spot using instrument strike interval."""
    radius = radius if radius is not None else PREMARKET["important_strikes_radius"]
    if interval <= 0:
        interval = 50
    atm = round(spot / interval) * interval
    return [float(atm + i * interval) for i in range(-radius, radius + 1)]


def analyze_option_records(
    records: list[dict],
    *,
    symbol: str,
    spot: float | None,
    expiry: str | None,
    strike_interval: int,
) -> OptionChainSnapshot:
    """Pure analysis of NSE-style option-chain records (testable without network)."""
    snap = OptionChainSnapshot(symbol=symbol, expiry=expiry, spot=spot)

    if expiry:
        exp_date = _parse_expiry(expiry)
        if exp_date:
            days, etype = classify_expiry(exp_date)
            snap.days_to_expiry = days
            snap.expiry_type = etype

    call_oi_by_strike: dict[float, float] = {}
    put_oi_by_strike: dict[float, float] = {}
    call_chg: dict[float, float] = {}
    put_chg: dict[float, float] = {}
    total_call_oi = 0.0
    total_put_oi = 0.0
    total_call_vol = 0.0
    total_put_vol = 0.0

    for row in records:
        strike = row.get("strikePrice")
        if strike is None:
            continue
        strike_f = float(strike)
        ce = row.get("CE") or {}
        pe = row.get("PE") or {}

        c_oi = float(ce.get("openInterest") or 0)
        p_oi = float(pe.get("openInterest") or 0)
        c_coi = float(ce.get("changeinOpenInterest") or 0)
        p_coi = float(pe.get("changeinOpenInterest") or 0)
        c_vol = float(ce.get("totalTradedVolume") or 0)
        p_vol = float(pe.get("totalTradedVolume") or 0)

        call_oi_by_strike[strike_f] = c_oi
        put_oi_by_strike[strike_f] = p_oi
        call_chg[strike_f] = c_coi
        put_chg[strike_f] = p_coi
        total_call_oi += c_oi
        total_put_oi += p_oi
        total_call_vol += c_vol
        total_put_vol += p_vol

    if not call_oi_by_strike and not put_oi_by_strike:
        snap.available = False
        snap.error = "Empty option-chain records"
        return snap

    if call_oi_by_strike:
        snap.highest_call_oi_strike = max(call_oi_by_strike, key=call_oi_by_strike.get)
        # Top 3 call OI above spot as resistance
        above = {k: v for k, v in call_oi_by_strike.items() if spot is None or k >= spot}
        ranked = sorted(above.items(), key=lambda x: x[1], reverse=True)[:3]
        snap.call_resistance_levels = [k for k, _ in ranked]

    if put_oi_by_strike:
        snap.highest_put_oi_strike = max(put_oi_by_strike, key=put_oi_by_strike.get)
        below = {k: v for k, v in put_oi_by_strike.items() if spot is None or k <= spot}
        ranked = sorted(below.items(), key=lambda x: x[1], reverse=True)[:3]
        snap.put_support_levels = [k for k, _ in ranked]

    snap.call_oi_change = sum(call_chg.values())
    snap.put_oi_change = sum(put_chg.values())
    snap.total_call_oi = total_call_oi
    snap.total_put_oi = total_put_oi

    if total_call_oi > 0:
        snap.pcr_oi = round(total_put_oi / total_call_oi, 3)
    if total_call_vol > 0:
        snap.pcr_volume = round(total_put_vol / total_call_vol, 3)

    ref = spot
    if ref is None and snap.highest_call_oi_strike and snap.highest_put_oi_strike:
        ref = (snap.highest_call_oi_strike + snap.highest_put_oi_strike) / 2
    if ref is not None:
        snap.important_strikes = important_strikes_around(ref, strike_interval)

    snap.available = True
    return snap


def fetch_option_chain(symbol: str, strike_interval: int) -> OptionChainSnapshot:
    def _load() -> OptionChainSnapshot:
        sess = _nse_session(symbol)
        last_err: Exception | None = None
        payload = None
        for template in OPTION_CHAIN_URLS:
            url = template.format(symbol=symbol)
            try:
                resp = sess.get(url, timeout=PREMARKET["api_timeout"])
                resp.raise_for_status()
                payload = resp.json()
                break
            except Exception as exc:
                last_err = exc
                continue
        if payload is None:
            raise last_err or ValueError("option-chain fetch failed")

        records = payload.get("records", {}) if isinstance(payload, dict) else {}
        data = records.get("data") or payload.get("data") or []
        expiry_dates = records.get("expiryDates") or payload.get("expiryDates") or []
        spot = None
        try:
            spot = float(records.get("underlyingValue") or payload.get("underlyingValue"))
        except (TypeError, ValueError):
            spot = None

        nearest_expiry = expiry_dates[0] if expiry_dates else None
        filtered = []
        for row in data:
            if nearest_expiry and row.get("expiryDate") and row.get("expiryDate") != nearest_expiry:
                continue
            filtered.append(row)
        if not filtered:
            filtered = data

        return analyze_option_records(
            filtered,
            symbol=symbol,
            spot=spot,
            expiry=nearest_expiry,
            strike_interval=strike_interval,
        )

    result = retry_call(
        _load,
        retries=PREMARKET["api_retries"],
        backoff=PREMARKET["api_retry_backoff"],
        label=f"option_chain:{symbol}",
    )
    if result is None:
        return OptionChainSnapshot(symbol=symbol, available=False, error="DATA UNAVAILABLE")
    return result


def fetch_nifty_banknifty_chains() -> tuple[OptionChainSnapshot, OptionChainSnapshot]:
    nifty = fetch_option_chain(
        PREMARKET["nse_option_nifty"],
        PREMARKET["nifty_strike_interval"],
    )
    bank = fetch_option_chain(
        PREMARKET["nse_option_banknifty"],
        PREMARKET["banknifty_strike_interval"],
    )
    return nifty, bank
