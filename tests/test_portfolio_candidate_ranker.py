from __future__ import annotations


def test_load_uniform_history_returns_empty_frame_when_cache_missing(tmp_path, monkeypatch):
    import portfolio_candidate_ranker as pcr

    monkeypatch.setattr(pcr, "CACHE_DIR", tmp_path)

    frame = pcr._load_uniform_history("TAOUSD", history_days=60)

    assert frame.empty


def test_select_best_non_active_candidates_ignores_active_pairs_and_keeps_best_variant():
    from portfolio_candidate_ranker import select_best_non_active_candidates

    candidates = [
        {
            "pair": "GIGAUSD",
            "construction": "baseline_or_vst60",
            "entry_filter": "base",
            "exit_profile": "runner",
            "robustness_score": 0.34,
            "older60_net_pct": 0.12,
            "recent60_net_pct": 0.04,
            "full_net_pct": 0.17,
        },
        {
            "pair": "FHEUSD",
            "construction": "trend_gate",
            "entry_filter": "base",
            "exit_profile": "runner",
            "robustness_score": 0.10,
            "older60_net_pct": 0.03,
            "recent60_net_pct": 0.02,
            "full_net_pct": 0.06,
        },
        {
            "pair": "FHEUSD",
            "construction": "trend_gate",
            "entry_filter": "gate50_vwap3",
            "exit_profile": "runner",
            "robustness_score": 0.18,
            "older60_net_pct": 0.05,
            "recent60_net_pct": 0.04,
            "full_net_pct": 0.10,
        },
        {
            "pair": "XDGUSD",
            "construction": "vst60_only",
            "entry_filter": "score45",
            "exit_profile": "runner",
            "robustness_score": 0.16,
            "older60_net_pct": -0.02,
            "recent60_net_pct": 0.12,
            "full_net_pct": 0.09,
        },
    ]

    selected = select_best_non_active_candidates(
        candidates,
        active_pair_names={"GIGAUSD", "BABYUSD"},
    )

    assert [item["pair"] for item in selected] == ["FHEUSD", "XDGUSD"]
    assert selected[0]["entry_filter"] == "gate50_vwap3"
