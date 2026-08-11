"""Retry helper and API failure isolation."""

from scanner.premarket.retry import retry_call


def test_retry_eventually_succeeds():
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    assert retry_call(flaky, retries=3, backoff=0.01, label="flaky") == "ok"


def test_retry_returns_none_on_exhaustion():
    def always_fail():
        raise RuntimeError("down")

    assert retry_call(always_fail, retries=2, backoff=0.01, label="down") is None
