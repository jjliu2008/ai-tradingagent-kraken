# tests/test_pattern_guided_discovery.py
from __future__ import annotations


def test_fingerprint_similarity_identical():
    from pattern_guided_discovery import fingerprint_similarity
    fp = {
        "gate_trend_bucket": "strong", "atr_bucket": "moderate",
        "efficiency_bucket": "directional", "vwap_dist_bucket": "near",
        "close_quality_bucket": "strong", "volume_bucket": "normal",
        "exit_mix_bucket": "take_profit",
    }
    score = fingerprint_similarity(fp, fp)
    assert abs(score - 1.0) < 1e-9


def test_fingerprint_similarity_all_different():
    from pattern_guided_discovery import fingerprint_similarity
    fp_a = {
        "gate_trend_bucket": "strong",  "atr_bucket": "high",
        "efficiency_bucket": "directional", "vwap_dist_bucket": "near",
        "close_quality_bucket": "strong", "volume_bucket": "elevated",
        "exit_mix_bucket": "take_profit",
    }
    fp_b = {
        "gate_trend_bucket": "negative", "atr_bucket": "very_low",
        "efficiency_bucket": "choppy",   "vwap_dist_bucket": "extended",
        "close_quality_bucket": "weak",  "volume_bucket": "low",
        "exit_mix_bucket": "stop_loss",
    }
    score = fingerprint_similarity(fp_a, fp_b)
    assert score < 0.10


def test_is_shadow_eligible_positive_older60():
    from pattern_guided_discovery import is_shadow_eligible
    assert is_shadow_eligible(
        older60_net_pct=0.02, older60_trades=5,
        recent60_net_pct=0.01, recent60_trades=2,
        full_net_pct=0.04, full_trades=5,
        pattern_match_score=0.0, robustness_score=0.05,
    )


def test_is_shadow_eligible_near_flat_needs_pattern_score():
    from pattern_guided_discovery import is_shadow_eligible
    assert not is_shadow_eligible(
        older60_net_pct=0.002, older60_trades=5,
        recent60_net_pct=0.01, recent60_trades=2,
        full_net_pct=0.03, full_trades=5,
        pattern_match_score=0.50,  # below 0.65 threshold
        robustness_score=0.02,
    )


def test_is_shadow_eligible_near_flat_with_good_pattern():
    from pattern_guided_discovery import is_shadow_eligible
    assert is_shadow_eligible(
        older60_net_pct=0.003, older60_trades=4,
        recent60_net_pct=0.01, recent60_trades=2,
        full_net_pct=0.03, full_trades=5,
        pattern_match_score=0.70,  # above 0.65
        robustness_score=0.01,
    )
