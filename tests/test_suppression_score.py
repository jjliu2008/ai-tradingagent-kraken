# tests/test_suppression_score.py
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest


def _make_frame(n: int, gate: float, atr: float, regime_ok: bool, signal: bool) -> pd.DataFrame:
    return pd.DataFrame({
        "gate_trend_strength_60": [gate] * n,
        "atr_pct": [atr] * n,
        "portfolio_regime_ok": [regime_ok] * n,
        "entry_signal": [signal] * n,
    })


def test_empty_frame_returns_one():
    from research_runtime import compute_pair_weak_score
    assert compute_pair_weak_score(pd.DataFrame()) == 1.0


def test_strong_regime_returns_low_score():
    from research_runtime import compute_pair_weak_score
    frame = _make_frame(8, gate=0.012, atr=0.012, regime_ok=True, signal=True)
    score = compute_pair_weak_score(frame)
    assert 0.0 <= score <= 0.3, f"Expected low score for strong regime, got {score}"


def test_weak_regime_returns_high_score():
    from research_runtime import compute_pair_weak_score
    frame = _make_frame(8, gate=0.000, atr=0.002, regime_ok=False, signal=False)
    score = compute_pair_weak_score(frame)
    assert score >= 0.7, f"Expected high score for weak regime, got {score}"


def test_score_in_unit_interval():
    from research_runtime import compute_pair_weak_score
    for gate in [0.000, 0.005, 0.015]:
        for atr in [0.002, 0.006, 0.015]:
            frame = _make_frame(8, gate=gate, atr=atr, regime_ok=True, signal=False)
            score = compute_pair_weak_score(frame)
            assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"


def test_suppression_thresholds_dict_exists():
    from research_runtime import SUPPRESSION_THRESHOLDS
    assert isinstance(SUPPRESSION_THRESHOLDS, dict)
