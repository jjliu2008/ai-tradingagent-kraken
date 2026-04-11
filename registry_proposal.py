# registry_proposal.py
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_pair_registry import PAIR_RESEARCH_REGISTRY

RESULTS_DIR = Path("results")

_MAX_CONCENTRATION    = 0.35
_MAX_CORRELATION      = 0.75
_CLUSTER_PROMOTE_LIMIT = 3


def check_concentration_gate(
    pair: str,
    existing_shadow_weights: dict[str, float],
    shadow_weight: float,
) -> tuple[bool, str]:
    if shadow_weight > _MAX_CONCENTRATION:
        return True, f"concentration {shadow_weight:.1%} exceeds {_MAX_CONCENTRATION:.0%} limit"
    return False, ""


def check_correlation_gate(max_signal_correlation: float) -> tuple[bool, str]:
    if max_signal_correlation > _MAX_CORRELATION:
        return True, f"signal_correlation {max_signal_correlation:.2f} exceeds {_MAX_CORRELATION:.2f}"
    return False, ""


def check_robustness_gate(robustness_score: float) -> tuple[bool, str]:
    if robustness_score <= 0.0:
        return True, f"robustness_score {robustness_score:.4f} <= 0"
    return False, ""


def _registry_hash() -> str:
    content = json.dumps(
        {k: {"c": v.construction, "ef": v.entry_filter, "ep": v.exit_profile}
         for k, v in PAIR_RESEARCH_REGISTRY.items()},
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(content.encode()).hexdigest()[:16]


def _current_active_registry() -> dict[str, Any]:
    return {
        pair: {
            "construction": plan.construction,
            "entry_filter": plan.entry_filter,
            "exit_profile": plan.exit_profile,
            "status": plan.status,
        }
        for pair, plan in PAIR_RESEARCH_REGISTRY.items()
    }


def build_diff(
    current_registry: dict[str, Any],
    core_candidates: list[dict],
    shadow_candidates: list[dict],
    existing_shadow_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    existing_shadow_weights = existing_shadow_weights or {}
    promote: list[dict] = []
    concentration_flags: list[dict] = []
    cluster_counts: dict[str, int] = {}

    for candidate in shadow_candidates + core_candidates:
        pair = candidate["pair"]
        construction = candidate["construction"]
        cluster_id = construction.split("_")[0]
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

        current = current_registry.get(pair, {})
        if current.get("status") in ("active", "active_experimental", "active_frozen"):
            continue

        shadow_weight = existing_shadow_weights.get(pair, 0.0)
        blocked_conc, reason_conc = check_concentration_gate(pair, existing_shadow_weights, shadow_weight)
        blocked_corr, reason_corr = check_correlation_gate(candidate.get("max_signal_correlation", 0.0))
        blocked_rob,  reason_rob  = check_robustness_gate(candidate["robustness_score"])
        blocked_cluster = cluster_counts.get(cluster_id, 0) >= _CLUSTER_PROMOTE_LIMIT

        any_blocked = blocked_conc or blocked_corr or blocked_rob or blocked_cluster
        block_reasons = [r for r in [reason_conc, reason_corr, reason_rob] if r]
        if blocked_cluster:
            block_reasons.append(f"cluster_limit: {cluster_id} already at {_CLUSTER_PROMOTE_LIMIT}")

        entry = {
            "pair": pair,
            "plan": construction,
            "entry_filter": candidate["entry_filter"],
            "exit_profile": candidate["exit_profile"],
            "robustness_score": candidate["robustness_score"],
            "older60_net_pct": candidate["older60_net_pct"],
            "recent60_net_pct": candidate["recent60_net_pct"],
            "approval_required": any_blocked,
            "cluster_id": cluster_id,
            "concentration_weight": shadow_weight,
            "max_signal_correlation_to_active": candidate.get("max_signal_correlation", 0.0),
            "source_run_id": candidate.get("source_run_id", ""),
        }
        if any_blocked:
            entry["block_reasons"] = block_reasons
            concentration_flags.append({"pair": pair, "reasons": block_reasons})
        promote.append(entry)

    current_pairs = set(current_registry.keys())
    proposed_pairs = {c["pair"] for c in shadow_candidates + core_candidates}
    no_change = [
        {"pair": p, "status": current_registry[p].get("status")}
        for p in current_pairs
        if p not in proposed_pairs
    ]

    return {
        "promote_to_shadow":  promote,
        "demote":             [],
        "no_change":          no_change,
        "concentration_flags": concentration_flags,
    }


def build_proposal(
    run_id: str,
    core_candidates: list[dict],
    shadow_candidates: list[dict],
    out_dir: Path,
    before_metrics: dict[str, float],
    after_metrics: dict[str, float],
) -> dict[str, Any]:
    current_registry = _current_active_registry()
    diff = build_diff(current_registry, core_candidates, shadow_candidates)

    doc: dict[str, Any] = {
        "schema_version":       "1.0",
        "run_id":               run_id,
        "source_run_id":        run_id,
        "registry_hash_before": _registry_hash(),
        "core_candidates":      core_candidates,
        "shadow_candidates":    shadow_candidates,
        "diff":                 diff,
        "concentration_flags":  diff["concentration_flags"],
        "before_metrics":       before_metrics,
        "after_metrics":        after_metrics,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "proposal.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


def run_proposal(run_id: str | None = None) -> dict[str, Any]:
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    run_dir    = RESULTS_DIR / "research_runs" / run_id
    latest_dir = RESULTS_DIR / "latest"

    def _load(fname: str) -> Any:
        for d in (run_dir, latest_dir):
            p = d / fname
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        return None

    core_data   = _load("core_candidates.json")   or {"candidates": []}
    shadow_data = _load("shadow_candidates.json") or {"candidates": []}

    before_metrics = {"older60_net_pct": -0.0148, "recent60_net_pct": 0.3066, "full_net_pct": 0.2918}
    after_metrics  = {"older60_net_pct": 0.0, "recent60_net_pct": 0.0, "full_net_pct": 0.0}

    proposal_dir = run_dir / "proposal"
    doc = build_proposal(
        run_id=run_id,
        core_candidates=core_data["candidates"],
        shadow_candidates=shadow_data["candidates"],
        out_dir=proposal_dir,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
    )

    (run_dir / "proposal.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "proposal.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")

    n_clean   = sum(1 for d in doc["diff"]["promote_to_shadow"] if not d["approval_required"])
    n_flagged = sum(1 for d in doc["diff"]["promote_to_shadow"] if d["approval_required"])
    print(f"registry_proposal: run_id={run_id}")
    print(f"  promote_to_shadow: {len(doc['diff']['promote_to_shadow'])} ({n_clean} auto / {n_flagged} needs approval)")
    return doc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    run_proposal(args.run_id)


if __name__ == "__main__":
    main()
