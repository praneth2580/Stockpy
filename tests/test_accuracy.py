"""Accuracy dashboard aggregations."""

from pathlib import Path

from scanner.premarket.accuracy import compute_accuracy_stats, format_accuracy_dashboard
from scanner.premarket.db import save_report


def test_accuracy_empty(tmp_path: Path, monkeypatch):
    db = tmp_path / "a.db"
    monkeypatch.setitem(__import__("scanner.config", fromlist=["PREMARKET"]).PREMARKET, "db_path", str(db))
    stats = compute_accuracy_stats()
    assert stats["total_reports"] == 0
    text = format_accuracy_dashboard(stats)
    assert "Total Reports" in text


def test_accuracy_with_rows(tmp_path: Path, monkeypatch):
    db = tmp_path / "a.db"
    monkeypatch.setitem(__import__("scanner.config", fromlist=["PREMARKET"]).PREMARKET, "db_path", str(db))
    for i, (bias, confirm) in enumerate(
        [("Bullish", "Confirmed"), ("Bearish", "Invalidated"), ("Neutral", "Partially confirmed")]
    ):
        save_report(
            {
                "meta": {
                    "report_date": f"2026-03-{10 + i:02d}",
                    "report_time": "09:00",
                    "job_type": "premarket",
                },
                "nifty": {"gap_class": "Moderate Gap Up"},
                "banknifty": {},
                "vix": {"value": 15},
                "fii_dii": {},
                "option_chain_nifty": {"expiry_type": "weekly"},
                "option_chain_banknifty": {},
                "bias": {"label": bias, "confidence": 60, "total_score": 1, "max_score": 5, "normalized_score": 60},
                "regime": "Trending Bullish",
                "confirm_930_result": confirm,
            },
            db_path=db,
        )
    stats = compute_accuracy_stats()
    assert stats["total_reports"] == 3
