from __future__ import annotations


def test_fheusd_is_live_in_active_research_book():
    from research_pair_registry import PAIR_RESEARCH_REGISTRY, active_pairs

    plan = PAIR_RESEARCH_REGISTRY["FHEUSD"]

    assert plan.status == "active_experimental"
    assert "FHEUSD" in active_pairs()
