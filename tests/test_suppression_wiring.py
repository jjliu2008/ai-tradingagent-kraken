from __future__ import annotations

import json
import types
from pathlib import Path


def test_suppressed_pair_is_skipped(tmp_path, monkeypatch):
    """A pair with allow_new_entries=False must not appear in scan candidates."""
    import universe_scanner_agent as usa

    # Patch load_suppression_state to return a state that blocks FAKEUSD
    state = {
        "portfolio_state": "normal",
        "pairs": {
            "FAKEUSD": {
                "state": "weak_defensive",
                "allow_new_entries": False,
                "notional_multiplier": 0.5,
            }
        }
    }
    monkeypatch.setattr(usa, "load_suppression_state", lambda: state)

    suppressed = usa.suppressed_pairs(state)
    assert "FAKEUSD" in suppressed


def test_normal_pair_not_suppressed(monkeypatch):
    import universe_scanner_agent as usa

    state = {
        "portfolio_state": "normal",
        "pairs": {
            "GIGAUSD": {
                "state": "normal",
                "allow_new_entries": True,
                "notional_multiplier": 1.0,
            }
        }
    }
    monkeypatch.setattr(usa, "load_suppression_state", lambda: state)
    suppressed = usa.suppressed_pairs(state)
    assert "GIGAUSD" not in suppressed


def test_portfolio_off_suppresses_all(monkeypatch):
    import universe_scanner_agent as usa

    state = {
        "portfolio_state": "off",
        "pairs": {
            "GIGAUSD": {"state": "off", "allow_new_entries": False, "notional_multiplier": 0.0},
            "ZECUSD":  {"state": "off", "allow_new_entries": False, "notional_multiplier": 0.0},
        }
    }
    monkeypatch.setattr(usa, "load_suppression_state", lambda: state)
    suppressed = usa.suppressed_pairs(state)
    assert "GIGAUSD" in suppressed
    assert "ZECUSD" in suppressed
