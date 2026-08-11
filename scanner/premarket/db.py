"""SQLite persistence for pre-market reports and confirmation outcomes."""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from scanner.config import PREMARKET

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SCHEMA = """
CREATE TABLE IF NOT EXISTS premarket_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_date TEXT NOT NULL,
    report_time TEXT NOT NULL,
    job_type TEXT NOT NULL,  -- premarket | open_915 | confirm_930
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    nifty_expected_open REAL,
    nifty_actual_open REAL,
    banknifty_expected_open REAL,
    banknifty_actual_open REAL,
    bias TEXT,
    confidence REAL,
    total_score REAL,
    max_score REAL,
    normalized_score REAL,
    india_vix REAL,
    fii_net REAL,
    dii_net REAL,
    nifty_pcr REAL,
    banknifty_pcr REAL,
    market_regime TEXT,
    open_915_result TEXT,
    confirm_930_result TEXT,
    later_outcome TEXT,
    UNIQUE(report_date, job_type)
);

CREATE TABLE IF NOT EXISTS job_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_type TEXT NOT NULL,
    run_date TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    message TEXT,
    UNIQUE(job_type, run_date)
);
"""


def get_db_path() -> Path:
    path = Path(PREMARKET["db_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = db_path or get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
    logger.info("Premarket DB ready at %s", db_path or get_db_path())


def try_acquire_job(job_type: str, run_date: str, db_path: Path | None = None) -> bool:
    """
    Prevent duplicate execution for the same job_type + date.
    Returns True if this process acquired the lock.
    """
    init_db(db_path)
    now = _utc_now_iso()
    try:
        with connect(db_path) as conn:
            conn.execute(
                "INSERT INTO job_runs (job_type, run_date, started_at, status, message) "
                "VALUES (?, ?, ?, ?, ?)",
                (job_type, run_date, now, "running", None),
            )
        return True
    except sqlite3.IntegrityError:
        logger.warning("Duplicate job blocked: %s on %s", job_type, run_date)
        return False


def finish_job(
    job_type: str,
    run_date: str,
    status: str,
    message: str | None = None,
    db_path: Path | None = None,
) -> None:
    now = _utc_now_iso()
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE job_runs SET finished_at=?, status=?, message=? "
            "WHERE job_type=? AND run_date=?",
            (now, status, message, job_type, run_date),
        )


def save_report(report: dict[str, Any], db_path: Path | None = None) -> int:
    init_db(db_path)
    meta = report.get("meta", {})
    bias = report.get("bias", {})
    nifty = report.get("nifty", {}) or {}
    bank = report.get("banknifty", {}) or {}
    vix = report.get("vix", {}) or {}
    fii = report.get("fii_dii", {}) or {}
    oc_n = report.get("option_chain_nifty", {}) or {}
    oc_b = report.get("option_chain_banknifty", {}) or {}

    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO premarket_reports (
                report_date, report_time, job_type, created_at, payload_json,
                nifty_expected_open, nifty_actual_open,
                banknifty_expected_open, banknifty_actual_open,
                bias, confidence, total_score, max_score, normalized_score,
                india_vix, fii_net, dii_net, nifty_pcr, banknifty_pcr,
                market_regime, open_915_result, confirm_930_result, later_outcome
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(report_date, job_type) DO UPDATE SET
                created_at=excluded.created_at,
                payload_json=excluded.payload_json,
                nifty_expected_open=excluded.nifty_expected_open,
                nifty_actual_open=excluded.nifty_actual_open,
                banknifty_expected_open=excluded.banknifty_expected_open,
                banknifty_actual_open=excluded.banknifty_actual_open,
                bias=excluded.bias,
                confidence=excluded.confidence,
                total_score=excluded.total_score,
                max_score=excluded.max_score,
                normalized_score=excluded.normalized_score,
                india_vix=excluded.india_vix,
                fii_net=excluded.fii_net,
                dii_net=excluded.dii_net,
                nifty_pcr=excluded.nifty_pcr,
                banknifty_pcr=excluded.banknifty_pcr,
                market_regime=excluded.market_regime,
                open_915_result=COALESCE(excluded.open_915_result, open_915_result),
                confirm_930_result=COALESCE(excluded.confirm_930_result, confirm_930_result),
                later_outcome=COALESCE(excluded.later_outcome, later_outcome)
            """,
            (
                meta.get("report_date"),
                meta.get("report_time"),
                meta.get("job_type", "premarket"),
                _utc_now_iso(),
                json.dumps(report, default=str),
                nifty.get("expected_open"),
                nifty.get("actual_open"),
                bank.get("expected_open"),
                bank.get("actual_open"),
                bias.get("label"),
                bias.get("confidence"),
                bias.get("total_score"),
                bias.get("max_score"),
                bias.get("normalized_score"),
                vix.get("value"),
                fii.get("fii_net"),
                fii.get("dii_net"),
                oc_n.get("pcr_oi"),
                oc_b.get("pcr_oi"),
                report.get("regime"),
                report.get("open_915_result"),
                report.get("confirm_930_result"),
                report.get("later_outcome"),
            ),
        )
        return int(cur.lastrowid)


def update_confirmation(
    report_date: str,
    *,
    open_915_result: str | None = None,
    confirm_930_result: str | None = None,
    nifty_actual_open: float | None = None,
    banknifty_actual_open: float | None = None,
    later_outcome: str | None = None,
    confirmation_payload: dict[str, Any] | None = None,
    job_type: str = "premarket",
    db_path: Path | None = None,
) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT payload_json FROM premarket_reports WHERE report_date=? AND job_type=?",
            (report_date, job_type),
        ).fetchone()
        payload = json.loads(row["payload_json"]) if row else {}
        if confirmation_payload:
            payload.setdefault("confirmations", {}).update(confirmation_payload)
        if open_915_result is not None:
            payload["open_915_result"] = open_915_result
        if confirm_930_result is not None:
            payload["confirm_930_result"] = confirm_930_result
        if later_outcome is not None:
            payload["later_outcome"] = later_outcome
        if nifty_actual_open is not None:
            payload.setdefault("nifty", {})["actual_open"] = nifty_actual_open
        if banknifty_actual_open is not None:
            payload.setdefault("banknifty", {})["actual_open"] = banknifty_actual_open

        conn.execute(
            """
            UPDATE premarket_reports SET
                payload_json=?,
                open_915_result=COALESCE(?, open_915_result),
                confirm_930_result=COALESCE(?, confirm_930_result),
                later_outcome=COALESCE(?, later_outcome),
                nifty_actual_open=COALESCE(?, nifty_actual_open),
                banknifty_actual_open=COALESCE(?, banknifty_actual_open)
            WHERE report_date=? AND job_type=?
            """,
            (
                json.dumps(payload, default=str),
                open_915_result,
                confirm_930_result,
                later_outcome,
                nifty_actual_open,
                banknifty_actual_open,
                report_date,
                job_type,
            ),
        )


def get_report(report_date: str, job_type: str = "premarket", db_path: Path | None = None) -> dict | None:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM premarket_reports WHERE report_date=? AND job_type=?",
            (report_date, job_type),
        ).fetchone()
        if not row:
            return None
        data = dict(row)
        data["payload"] = json.loads(data.pop("payload_json"))
        return data


def list_reports(limit: int = 500, db_path: Path | None = None) -> list[dict]:
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM premarket_reports WHERE job_type='premarket' "
            "ORDER BY report_date DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out = []
        for row in rows:
            data = dict(row)
            data["payload"] = json.loads(data.pop("payload_json"))
            out.append(data)
        return out
