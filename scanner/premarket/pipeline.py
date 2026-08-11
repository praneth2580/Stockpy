"""Orchestrate 9:00 AM F&O pre-market analysis pipeline."""

from __future__ import annotations

import logging
from typing import Any

from scanner.config import PREMARKET
from scanner.premarket.analysis.checklist import evaluate_checklist
from scanner.premarket.analysis.confidence import compute_confidence
from scanner.premarket.analysis.levels import build_levels
from scanner.premarket.analysis.regime import classify_regime
from scanner.premarket.analysis.scoring import compute_scores
from scanner.premarket.calendar import is_trading_day, now_ist
from scanner.premarket.collectors.events import collect_risk_events
from scanner.premarket.collectors.fii_dii import fetch_fii_dii
from scanner.premarket.collectors.global_markets import fetch_global_snapshot
from scanner.premarket.collectors.indices import fetch_nifty_and_banknifty, fetch_vix
from scanner.premarket.collectors.option_chain import fetch_nifty_banknifty_chains
from scanner.premarket.db import finish_job, init_db, save_report, try_acquire_job
from scanner.premarket.report import build_scenarios, format_premarket_report

logger = logging.getLogger(__name__)


def _safe(label: str, fn, fallback):
    try:
        return fn()
    except Exception as exc:
        logger.exception("%s failed: %s", label, exc)
        return fallback


def generate_premarket_report(*, force: bool = False, skip_lock: bool = False) -> dict[str, Any]:
    """
    Collect all available pre-market inputs and build a structured report.
    Individual collector failures do not abort the pipeline.
    """
    init_db()
    now = now_ist(PREMARKET["timezone"])
    report_date = now.date().isoformat()
    job_type = "premarket"

    if not force and not is_trading_day(now.date()):
        msg = f"Skipping pre-market job: {report_date} is not a trading day"
        logger.info(msg)
        return {
            "skipped": True,
            "reason": msg,
            "meta": {"report_date": report_date, "report_time": PREMARKET["report_time"], "job_type": job_type},
        }

    if not skip_lock and not force:
        if not try_acquire_job(job_type, report_date):
            return {
                "skipped": True,
                "reason": "Duplicate execution blocked",
                "meta": {"report_date": report_date, "report_time": PREMARKET["report_time"], "job_type": job_type},
            }

    logger.info("Starting F&O pre-market report for %s", report_date)

    global_snap = _safe("global", fetch_global_snapshot, None)
    gift_val = global_snap.gift_nifty if global_snap else None

    nifty, bank = _safe(
        "indices",
        lambda: fetch_nifty_and_banknifty(gift_indication=gift_val),
        (None, None),
    )
    vix = _safe("vix", fetch_vix, {"available": False, "error": "DATA UNAVAILABLE", "value": None})
    fii_dii = _safe("fii_dii", fetch_fii_dii, None)
    chains = _safe("option_chain", fetch_nifty_banknifty_chains, (None, None))
    nifty_chain, bank_chain = chains if chains else (None, None)

    # Fallbacks for failed objects
    from scanner.premarket.models import FIIDIISnapshot, GlobalSnapshot, IndexSnapshot, OptionChainSnapshot

    if nifty is None:
        nifty = IndexSnapshot(symbol=PREMARKET["nifty_symbol"], name="NIFTY", available=False, error="DATA UNAVAILABLE")
    if bank is None:
        bank = IndexSnapshot(symbol=PREMARKET["banknifty_symbol"], name="BANK NIFTY", available=False, error="DATA UNAVAILABLE")
    if global_snap is None:
        global_snap = GlobalSnapshot(available=False, errors=["DATA UNAVAILABLE"])
    if fii_dii is None:
        fii_dii = FIIDIISnapshot(available=False, error="DATA UNAVAILABLE")
    if nifty_chain is None:
        nifty_chain = OptionChainSnapshot(symbol="NIFTY", available=False, error="DATA UNAVAILABLE")
    if bank_chain is None:
        bank_chain = OptionChainSnapshot(symbol="BANKNIFTY", available=False, error="DATA UNAVAILABLE")

    risk_flags = collect_risk_events(nifty_chain=nifty_chain, bank_chain=bank_chain, today=now.date())
    if vix.get("available") and vix.get("value") is not None and vix["value"] >= PREMARKET["vix_high"]:
        risk_flags.append(f"High volatility (India VIX {vix['value']})")

    levels_n = build_levels(nifty, nifty_chain)
    levels_b = build_levels(bank, bank_chain)
    regime = classify_regime(nifty, vix)

    checklist_objs = evaluate_checklist(
        nifty=nifty,
        bank=bank,
        global_snap=global_snap,
        vix=vix,
        fii_dii=fii_dii,
        nifty_chain=nifty_chain,
        bank_chain=bank_chain,
        risk_flags=risk_flags,
    )
    checklist = [c.to_dict() for c in checklist_objs]

    scores = compute_scores(
        nifty=nifty,
        bank=bank,
        global_snap=global_snap,
        vix=vix,
        fii_dii=fii_dii,
        nifty_chain=nifty_chain,
        risk_flags=risk_flags,
    )
    conf = compute_confidence(scores["breakdowns"], checklist)

    report: dict[str, Any] = {
        "meta": {
            "report_date": report_date,
            "report_time": PREMARKET["report_time"],
            "generated_at": now.isoformat(),
            "job_type": job_type,
            "timezone": PREMARKET["timezone"],
        },
        "nifty": nifty.to_dict(),
        "banknifty": bank.to_dict(),
        "global": global_snap.to_dict(),
        "vix": vix,
        "fii_dii": fii_dii.to_dict(),
        "option_chain_nifty": nifty_chain.to_dict(),
        "option_chain_banknifty": bank_chain.to_dict(),
        "levels_nifty": levels_n.to_dict(),
        "levels_banknifty": levels_b.to_dict(),
        "checklist": checklist,
        "regime": regime,
        "risk_flags": risk_flags,
        "bias": {
            "label": scores["label"],
            "total_score": scores["total_score"],
            "max_score": scores["max_score"],
            "normalized_score": scores["normalized_score"],
            "confidence": conf["confidence"],
            "confidence_reason": conf["reason"],
            "breakdowns": scores["breakdowns"],
        },
        "disclaimer": (
            "Pre-market directional bias only. Not a trade signal. "
            "Wait for market-open confirmation. No guaranteed outcomes."
        ),
    }
    report["scenarios"] = build_scenarios(report)
    report["text"] = format_premarket_report(report)

    try:
        save_report(report)
        if not skip_lock and not force:
            finish_job(job_type, report_date, "success")
        elif force and not skip_lock:
            # Still record success path when forced after lock
            finish_job(job_type, report_date, "success", "forced")
    except Exception as exc:
        logger.exception("Failed to persist report: %s", exc)
        if not skip_lock:
            finish_job(job_type, report_date, "error", str(exc))

    logger.info(
        "Pre-market report complete: bias=%s confidence=%s%%",
        scores["label"],
        conf["confidence"],
    )
    return report
