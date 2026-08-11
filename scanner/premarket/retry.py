"""Shared retry / HTTP helpers for resilient data collection."""

from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

import requests

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_call(
    fn: Callable[[], T],
    *,
    retries: int = 3,
    backoff: float = 1.5,
    exceptions: tuple = (Exception,),
    label: str = "operation",
) -> T | None:
    """
    Execute fn with retries. Returns None on exhausted failures
    instead of raising, so one source cannot crash the pipeline.
    """
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except exceptions as exc:
            last_err = exc
            wait = backoff * attempt
            logger.warning(
                "%s failed (attempt %s/%s): %s — retrying in %.1fs",
                label,
                attempt,
                retries,
                exc,
                wait,
            )
            if attempt < retries:
                time.sleep(wait)
    logger.error("%s unavailable after %s attempts: %s", label, retries, last_err)
    return None


def http_get_json(
    url: str,
    *,
    headers: dict | None = None,
    cookies: dict | None = None,
    timeout: float = 15.0,
    session: requests.Session | None = None,
) -> dict | list | None:
    """GET JSON with timeout. Raises on HTTP/JSON errors for retry_call."""
    sess = session or requests
    resp = sess.get(url, headers=headers, cookies=cookies, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
