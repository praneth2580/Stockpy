"""FII / DII activity from NSE public endpoints (best-effort)."""

from __future__ import annotations

import logging
from typing import Any

import requests

from scanner.config import PREMARKET
from scanner.premarket.models import FIIDIISnapshot
from scanner.premarket.retry import retry_call

logger = logging.getLogger(__name__)

NSE_HOME = "https://www.nseindia.com"
FII_DII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/reports/fii-dii",
}


def _nse_session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update(NSE_HEADERS)
    sess.get(NSE_HOME, timeout=PREMARKET["api_timeout"])
    return sess


def _to_float(val: Any) -> float | None:
    if val is None or val == "-" or val == "":
        return None
    try:
        return float(str(val).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _trend_from_nets(nets: list[float | None]) -> str | None:
    vals = [n for n in nets if n is not None]
    if len(vals) < 2:
        return None
    positive = sum(1 for n in vals if n > 0)
    negative = sum(1 for n in vals if n < 0)
    if positive >= len(vals) - 1:
        return "Buying"
    if negative >= len(vals) - 1:
        return "Selling"
    return "Mixed"


def fetch_fii_dii() -> FIIDIISnapshot:
    def _load() -> FIIDIISnapshot:
        sess = _nse_session()
        resp = sess.get(FII_DII_URL, timeout=PREMARKET["api_timeout"])
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or not data:
            raise ValueError("Unexpected FII/DII payload")

        rows = data
        latest = rows[0]

        fii = next((r for r in rows if str(r.get("category", "")).upper().startswith("FII")), None)
        dii = next((r for r in rows if str(r.get("category", "")).upper().startswith("DII")), None)

        snap = FIIDIISnapshot(available=True)

        def extract(row: dict | None) -> tuple[float | None, float | None, float | None]:
            if not row:
                return None, None, None
            buy = _to_float(row.get("buyValue") or row.get("buy"))
            sell = _to_float(row.get("sellValue") or row.get("sell"))
            net = _to_float(row.get("netValue") or row.get("net"))
            if net is None and buy is not None and sell is not None:
                net = buy - sell
            return buy, sell, net

        if not fii and not dii:
            raise ValueError(f"Unrecognized FII/DII schema: keys={list(latest.keys())}")

        snap.fii_buy, snap.fii_sell, snap.fii_net = extract(fii)
        snap.dii_buy, snap.dii_sell, snap.dii_net = extract(dii)
        snap.as_of = (fii or dii or {}).get("date")

        fii_rows = [r for r in rows if str(r.get("category", "")).upper().startswith("FII")]
        dii_rows = [r for r in rows if str(r.get("category", "")).upper().startswith("DII")]
        if len(fii_rows) > 1:
            _, _, snap.prev_fii_net = extract(fii_rows[1])
        if len(dii_rows) > 1:
            _, _, snap.prev_dii_net = extract(dii_rows[1])

        snap.fii_5d_trend = _trend_from_nets([_to_float(r.get("netValue")) for r in fii_rows[:5]])
        snap.dii_5d_trend = _trend_from_nets([_to_float(r.get("netValue")) for r in dii_rows[:5]])
        snap.fii_futures_net = None
        snap.fii_options_net = None
        return snap

    result = retry_call(
        _load,
        retries=PREMARKET["api_retries"],
        backoff=PREMARKET["api_retry_backoff"],
        label="fii_dii",
    )
    if result is None:
        return FIIDIISnapshot(available=False, error="DATA UNAVAILABLE")
    return result
