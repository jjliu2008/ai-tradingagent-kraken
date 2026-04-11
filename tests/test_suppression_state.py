# tests/test_suppression_state.py
from __future__ import annotations

import json
import time
from pathlib import Path


def test_evaluate_state_normal_stays_normal_when_below_threshold():
    from suppression_state import evaluate_state
    thresholds = {"weak_defensive_enter_threshold": 0.55, "weak_defensive_exit_threshold": 0.40, "cooldown_bars": 4}
    prev = {"state": "normal", "bars_in_state": 5, "bars_above_threshold": 0, "bars_below_threshold": 10}
    result = evaluate_state("FAKEUSD", current_score=0.30, prev_state=prev, thresholds=thresholds)
    assert result["state"] == "normal"
    assert result["allow_new_entries"] is True
    assert result["notional_multiplier"] == 1.0


def test_evaluate_state_transitions_to_weak_defensive_after_n_bars():
    from suppression_state import evaluate_state, WEAK_DEFENSIVE_ENTER_BARS
    thresholds = {"weak_defensive_enter_threshold": 0.55, "weak_defensive_exit_threshold": 0.40, "cooldown_bars": 4}
    prev = {"state": "normal", "bars_in_state": 5,
            "bars_above_threshold": WEAK_DEFENSIVE_ENTER_BARS - 1,
            "bars_below_threshold": 0}
    result = evaluate_state("FAKEUSD", current_score=0.70, prev_state=prev, thresholds=thresholds)
    assert result["state"] == "weak_defensive"
    assert result["allow_new_entries"] is False
    assert result["notional_multiplier"] == 0.5


def test_evaluate_state_returns_to_normal_after_exit_bars():
    from suppression_state import evaluate_state, WEAK_DEFENSIVE_EXIT_BARS
    thresholds = {"weak_defensive_enter_threshold": 0.55, "weak_defensive_exit_threshold": 0.40, "cooldown_bars": 4}
    prev = {"state": "weak_defensive", "bars_in_state": 10,
            "bars_above_threshold": 0,
            "bars_below_threshold": WEAK_DEFENSIVE_EXIT_BARS - 1}
    result = evaluate_state("FAKEUSD", current_score=0.20, prev_state=prev, thresholds=thresholds)
    assert result["state"] == "normal"


def test_compute_and_write_creates_valid_json(tmp_path):
    from suppression_state import compute_and_write, load_state
    out = tmp_path / "suppression_state.json"
    # Empty pairs list — should still write a valid doc
    doc = compute_and_write(pairs=[], out_path=out, history_days=120)
    assert out.exists()
    assert doc["schema_version"] == "1.0"
    assert "portfolio_state" in doc
    loaded = load_state(out)
    assert loaded["schema_version"] == "1.0"


def test_load_state_returns_safe_defaults_when_missing(tmp_path):
    from suppression_state import load_state
    result = load_state(tmp_path / "nonexistent.json")
    assert result["portfolio_state"] == "normal"
    assert result["pairs"] == {}
