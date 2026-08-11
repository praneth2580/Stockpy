"""
Central configuration for Stockpy.

Values are loaded from environment variables (and an optional .env file).
Hardcoded defaults keep the existing equity scanner working out of the box.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

_ROOT = Path(__file__).resolve().parent.parent


def _env(key: str, default: str | None = None) -> str | None:
    val = os.getenv(key)
    if val is None or val == "":
        return default
    return val


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Existing equity-scanner settings (kept compatible with runner.settings)
DEFAULT_SETTINGS: dict[str, Any] = {
    "workers": _env_int("STOCKPY_WORKERS", 3),
    "period": _env("STOCKPY_PERIOD", "1y"),
    "telegram_token": _env(
        "TELEGRAM_BOT_TOKEN",
        "5710041825:AAEulSFLC4TBEidHKYcmsmBht-u7_AJUbj4",
    ),
    "telegram_chat_id": _env("TELEGRAM_CHAT_ID", "1175853690"),
}

# Pre-market / F&O configuration
PREMARKET: dict[str, Any] = {
    "timezone": _env("PREMARKET_TIMEZONE", "Asia/Kolkata"),
    "report_time": _env("PREMARKET_REPORT_TIME", "09:00"),
    "market_open_time": _env("MARKET_OPEN_TIME", "09:15"),
    "confirmation_time": _env("CONFIRMATION_TIME", "09:30"),
    "nifty_symbol": _env("NIFTY_SYMBOL", "^NSEI"),
    "banknifty_symbol": _env("BANKNIFTY_SYMBOL", "^NSEBANK"),
    "india_vix_symbol": _env("INDIA_VIX_SYMBOL", "^INDIAVIX"),
    "gift_nifty_symbol": _env("GIFT_NIFTY_SYMBOL", ""),  # empty = try NSE / mark unavailable
    "nse_option_nifty": _env("NSE_OPTION_NIFTY", "NIFTY"),
    "nse_option_banknifty": _env("NSE_OPTION_BANKNIFTY", "BANKNIFTY"),
    "db_path": _env("PREMARKET_DB_PATH", str(_ROOT / "data" / "premarket.db")),
    "notify_enabled": _env_bool("PREMARKET_NOTIFY", True),
    "api_timeout": _env_float("PREMARKET_API_TIMEOUT", 15.0),
    "api_retries": _env_int("PREMARKET_API_RETRIES", 3),
    "api_retry_backoff": _env_float("PREMARKET_API_RETRY_BACKOFF", 1.5),
    # Gap classification thresholds (% of previous close)
    "gap_flat_pct": _env_float("GAP_FLAT_PCT", 0.10),
    "gap_small_pct": _env_float("GAP_SMALL_PCT", 0.30),
    "gap_moderate_pct": _env_float("GAP_MODERATE_PCT", 0.70),
    # Bias classification bands (normalized score 0–100)
    "bias_bands": {
        "strong_bearish": (0.0, 20.0),
        "bearish": (20.0, 35.0),
        "mild_bearish": (35.0, 45.0),
        "neutral": (45.0, 55.0),
        "mild_bullish": (55.0, 65.0),
        "bullish": (65.0, 80.0),
        "strong_bullish": (80.0, 100.0),
    },
    # Scoring weights (absolute contribution; signs applied by signal)
    "score_weights": {
        "gap": _env_float("WEIGHT_GAP", 2.0),
        "nifty_trend": _env_float("WEIGHT_NIFTY_TREND", 2.0),
        "banknifty_trend": _env_float("WEIGHT_BANKNIFTY_TREND", 1.5),
        "global_markets": _env_float("WEIGHT_GLOBAL", 1.5),
        "gift_nifty": _env_float("WEIGHT_GIFT", 1.5),
        "india_vix": _env_float("WEIGHT_VIX", 1.0),
        "fii_dii": _env_float("WEIGHT_FII_DII", 1.5),
        "option_oi": _env_float("WEIGHT_OPTION_OI", 1.5),
        "pcr": _env_float("WEIGHT_PCR", 1.0),
        "oi_change": _env_float("WEIGHT_OI_CHANGE", 1.0),
        "support_resistance": _env_float("WEIGHT_SR", 1.0),
        "events": _env_float("WEIGHT_EVENTS", 0.5),
    },
    "nifty_strike_interval": _env_int("NIFTY_STRIKE_INTERVAL", 50),
    "banknifty_strike_interval": _env_int("BANKNIFTY_STRIKE_INTERVAL", 100),
    "important_strikes_radius": _env_int("IMPORTANT_STRIKES_RADIUS", 5),
    "vix_elevated": _env_float("VIX_ELEVATED", 18.0),
    "vix_high": _env_float("VIX_HIGH", 22.0),
    "pcr_bullish": _env_float("PCR_BULLISH", 1.0),
    "pcr_bearish": _env_float("PCR_BEARISH", 0.7),
}


def apply_config_to_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Merge env-backed defaults into the mutable runner settings dict."""
    for key, value in DEFAULT_SETTINGS.items():
        if key not in settings or settings[key] in (None, ""):
            settings[key] = value
        # Always prefer env when explicitly set
        env_map = {
            "workers": "STOCKPY_WORKERS",
            "period": "STOCKPY_PERIOD",
            "telegram_token": "TELEGRAM_BOT_TOKEN",
            "telegram_chat_id": "TELEGRAM_CHAT_ID",
        }
        env_key = env_map.get(key)
        if env_key and os.getenv(env_key):
            if key == "workers":
                settings[key] = _env_int(env_key, settings[key])
            else:
                settings[key] = os.getenv(env_key)
    return settings
