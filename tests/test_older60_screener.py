# tests/test_older60_screener.py
from __future__ import annotations

import math


def test_robustness_score_formula():
    """robustness_score equation matches spec exactly."""
    from older60_pair_screener import robustness_score

    score = robustness_score(
        older60_net_pct=0.04,  older60_trades=5,
        recent60_net_pct=0.03, recent60_trades=3,
        full_net_pct=0.06,     full_trades=8,
        full_max_dd_pct=-0.05,
        concentration_share=0.20,
        max_signal_correlation=0.50,
    )
    # older60_term  = 0.04 * min(1.0, sqrt(5/5))   = 0.04
    # recent60_term = 0.03 * min(1.0, sqrt(3/3))   = 0.03
    # full120_term  = 0.06 * min(1.0, sqrt(8/8))   = 0.06
    # drawdown_penalty        = max(0, 0.05) = 0.05
    # concentration_penalty   = max(0, 0.20-0.35)/0.15 = 0.0
    # correlation_penalty     = max(0, 0.50-0.70)/0.30 = 0.0
    # score = 1.5*0.04 + 1.0*0.03 + 0.75*0.06 - 0.5*0.05 - 0.25*0 - 0.25*0
    #       = 0.06 + 0.03 + 0.045 - 0.025 = 0.11
    assert abs(score - 0.11) < 1e-9, f"Got {score}"


def test_robustness_score_penalises_concentration():
    from older60_pair_screener import robustness_score

    low_conc  = robustness_score(0.04,5, 0.03,3, 0.06,8, -0.02, 0.20, 0.50)
    high_conc = robustness_score(0.04,5, 0.03,3, 0.06,8, -0.02, 0.60, 0.50)
    assert high_conc < low_conc


def test_hard_gates_reject_negative_windows():
    from older60_pair_screener import passes_hard_gates

    assert not passes_hard_gates(
        tier="core",
        older60_net_pct=-0.01, older60_trades=6,
        recent60_net_pct=0.02, recent60_trades=4,
        full_net_pct=0.05,     full_trades=9,
    )
    assert not passes_hard_gates(
        tier="shadow",
        older60_net_pct=0.001, older60_trades=4,
        recent60_net_pct=-0.01, recent60_trades=2,
        full_net_pct=0.05,      full_trades=5,
    )


def test_hard_gates_pass_for_valid_core():
    from older60_pair_screener import passes_hard_gates

    assert passes_hard_gates(
        tier="core",
        older60_net_pct=0.02, older60_trades=6,
        recent60_net_pct=0.01, recent60_trades=4,
        full_net_pct=0.04,    full_trades=9,
    )
