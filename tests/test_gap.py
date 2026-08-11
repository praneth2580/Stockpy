"""Tests for gap calculation and classification."""

from scanner.premarket.analysis.gap import classify_gap, compute_gap


def test_compute_gap_up():
    gap, pct, klass = compute_gap(24850, 24930)
    assert gap == 80.0
    assert abs(pct - 0.3219) < 0.01
    assert klass is not None
    assert "Gap Up" in klass


def test_compute_gap_down():
    gap, pct, klass = compute_gap(25000, 24700)
    assert gap == -300.0
    assert pct < 0
    assert "Gap Down" in klass


def test_compute_gap_missing():
    assert compute_gap(None, 100) == (None, None, None)
    assert compute_gap(100, None) == (None, None, None)
    assert compute_gap(0, 100) == (None, None, None)


def test_classify_gap_thresholds():
    cfg = {
        "gap_flat_pct": 0.10,
        "gap_small_pct": 0.30,
        "gap_moderate_pct": 0.70,
    }
    assert classify_gap(0.05, cfg) == "Flat"
    assert classify_gap(0.2, cfg) == "Small Gap Up"
    assert classify_gap(0.5, cfg) == "Moderate Gap Up"
    assert classify_gap(1.0, cfg) == "Strong Gap Up"
    assert classify_gap(-0.2, cfg) == "Small Gap Down"
    assert classify_gap(-0.5, cfg) == "Moderate Gap Down"
    assert classify_gap(-1.0, cfg) == "Strong Gap Down"
