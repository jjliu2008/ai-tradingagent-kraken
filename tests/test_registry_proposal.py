# tests/test_registry_proposal.py
from __future__ import annotations


def _make_candidate(pair: str, robustness: float = 0.05, correlation: float = 0.30) -> dict:
    return {
        "pair": pair, "construction": "tc15_tighter_volume_cap",
        "entry_filter": "base", "exit_profile": "base",
        "robustness_score": robustness,
        "full_net_pct": 0.04, "full_trades": 9, "full_max_dd_pct": -0.03,
        "older60_net_pct": 0.02, "older60_trades": 5,
        "recent60_net_pct": 0.02, "recent60_trades": 3,
        "passes_core": True, "passes_shadow": True,
        "max_signal_correlation": correlation,
        "pattern_match_score": 0.75,
    }


def test_build_diff_adds_new_pair_to_shadow():
    from registry_proposal import build_diff
    current_registry = {}
    shadow_candidates = [_make_candidate("NEWUSD")]
    diff = build_diff(current_registry, core_candidates=[], shadow_candidates=shadow_candidates)
    pairs_to_promote = [d["pair"] for d in diff["promote_to_shadow"]]
    assert "NEWUSD" in pairs_to_promote


def test_concentration_gate_blocks_high_concentration():
    from registry_proposal import check_concentration_gate
    existing_shadow = {"GIGAUSD": 0.40}
    blocked, reason = check_concentration_gate("GIGAUSD", existing_shadow, shadow_weight=0.40)
    assert blocked
    assert "concentration" in reason


def test_correlation_gate_blocks_high_correlation():
    from registry_proposal import check_correlation_gate
    blocked, reason = check_correlation_gate(max_signal_correlation=0.80)
    assert blocked
    assert "correlation" in reason


def test_robustness_gate_blocks_non_positive():
    from registry_proposal import check_robustness_gate
    blocked, reason = check_robustness_gate(robustness_score=-0.01)
    assert blocked


def test_build_proposal_writes_json(tmp_path):
    from registry_proposal import build_proposal
    import json

    shadow_candidates = [_make_candidate("NEWUSD")]
    doc = build_proposal(
        run_id="2026-04-11T00:00:00",
        core_candidates=[],
        shadow_candidates=shadow_candidates,
        out_dir=tmp_path,
        before_metrics={"older60_net_pct": -0.015, "recent60_net_pct": 0.30, "full_net_pct": 0.29},
        after_metrics={"older60_net_pct": 0.02, "recent60_net_pct": 0.29, "full_net_pct": 0.30},
    )
    assert (tmp_path / "proposal.json").exists()
    loaded = json.loads((tmp_path / "proposal.json").read_text())
    assert loaded["schema_version"] == "1.0"
    assert "diff" in loaded
    assert "before_metrics" in loaded
    assert "registry_hash_before" in loaded
