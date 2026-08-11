"""Premarket package public API (lazy exports to keep imports light)."""

from __future__ import annotations

__all__ = [
    "generate_premarket_report",
    "run_open_snapshot",
    "run_confirmation",
    "format_premarket_report",
    "format_premarket_telegram",
    "format_open_telegram",
    "format_confirm_telegram",
    "compute_accuracy_stats",
    "format_accuracy_dashboard",
]


def __getattr__(name: str):
    if name == "generate_premarket_report":
        from scanner.premarket.pipeline import generate_premarket_report

        return generate_premarket_report
    if name == "run_open_snapshot":
        from scanner.premarket.confirm import run_open_snapshot

        return run_open_snapshot
    if name == "run_confirmation":
        from scanner.premarket.confirm import run_confirmation

        return run_confirmation
    if name == "format_premarket_report":
        from scanner.premarket.report import format_premarket_report

        return format_premarket_report
    if name == "format_premarket_telegram":
        from scanner.premarket.report import format_premarket_telegram

        return format_premarket_telegram
    if name == "format_open_telegram":
        from scanner.premarket.report import format_open_telegram

        return format_open_telegram
    if name == "format_confirm_telegram":
        from scanner.premarket.report import format_confirm_telegram

        return format_confirm_telegram
    if name == "compute_accuracy_stats":
        from scanner.premarket.accuracy import compute_accuracy_stats

        return compute_accuracy_stats
    if name == "format_accuracy_dashboard":
        from scanner.premarket.accuracy import format_accuracy_dashboard

        return format_accuracy_dashboard
    raise AttributeError(name)
