# tests/test_segment_diagnostics.py
from __future__ import annotations

import pandas as pd
import pytest


def _make_frame(n: int) -> pd.DataFrame:
    import numpy as np
    ts = list(range(1_700_000_000, 1_700_000_000 + n * 900, 900))
    return pd.DataFrame({
        "ts": ts,
        "gate_trend_strength_60": [0.008] * n,
        "atr_pct": [0.007] * n,
        "efficiency_ratio_8": [0.40] * n,
        "distance_from_vwap": [0.005] * n,
        "close_location": [0.75] * n,
        "volume_ratio": [1.2] * n,
        "component_count": [2] * n,
    })


def test_bucket_gate_trend():
    from segment_diagnostics import bucket_gate_trend
    assert bucket_gate_trend(0.000) == "negative"
    assert bucket_gate_trend(0.001) == "weak"
    assert bucket_gate_trend(0.005) == "moderate"
    assert bucket_gate_trend(0.012) == "strong"


def test_bucket_atr():
    from segment_diagnostics import bucket_atr
    assert bucket_atr(0.002) == "very_low"
    assert bucket_atr(0.005) == "low"
    assert bucket_atr(0.009) == "moderate"
    assert bucket_atr(0.020) == "high"


def test_fingerprint_from_frame():
    from segment_diagnostics import fingerprint_from_frame
    frame = _make_frame(20)
    fp = fingerprint_from_frame(frame)
    assert "gate_trend_bucket" in fp
    assert "atr_bucket" in fp
    assert "efficiency_bucket" in fp
    assert "vwap_dist_bucket" in fp
    assert "close_quality_bucket" in fp
    assert "volume_bucket" in fp
    assert fp["gate_trend_bucket"] == "moderate"


def test_extract_diagnostics_empty_trades():
    from segment_diagnostics import extract_pair_diagnostics
    frame = _make_frame(20)
    split_ts = frame["ts"].iloc[-1] - 30 * 24 * 60 * 60
    result = extract_pair_diagnostics("FAKEUSD", "tc15_only", "base", frame, trades=[], split_ts=split_ts)
    assert "pair" in result
    assert "older60_fingerprint" in result
    assert result["older60_trades"] == 0
