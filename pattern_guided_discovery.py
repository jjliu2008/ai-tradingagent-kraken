# pattern_guided_discovery.py
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("results")

_WEIGHTS: dict[str, float] = {
    "gate_trend_bucket":    0.25,
    "atr_bucket":           0.20,
    "efficiency_bucket":    0.15,
    "vwap_dist_bucket":     0.15,
    "close_quality_bucket": 0.10,
    "volume_bucket":        0.10,
    "exit_mix_bucket":      0.05,
}

_NEAR_FLAT_THRESHOLD          = 0.005
_PATTERN_MATCH_THRESHOLD      = 0.65
_MIN_OLDER60_TRADES_NEAR_FLAT = 4


def fingerprint_similarity(fp_a: dict[str, str], fp_b: dict[str, str]) -> float:
    if not fp_a or not fp_b:
        return 0.0
    total_weight = 0.0
    match_weight = 0.0
    for field, weight in _WEIGHTS.items():
        val_a = fp_a.get(field)
        val_b = fp_b.get(field)
        if val_a is None or val_b is None:
            continue
        total_weight += weight
        if val_a == val_b:
            match_weight += weight
    if total_weight == 0.0:
        return 0.0
    return match_weight / total_weight


def is_shadow_eligible(
    older60_net_pct: float, older60_trades: int,
    recent60_net_pct: float, recent60_trades: int,
    full_net_pct: float, full_trades: int,
    pattern_match_score: float,
    robustness_score: float,
) -> bool:
    from older60_pair_screener import _TIER_GATES
    shadow_gates = _TIER_GATES["shadow"]

    if full_net_pct     <= 0.0:                        return False
    if recent60_net_pct <= 0.0:                        return False
    if robustness_score <= 0.0:                        return False
    if older60_trades   < shadow_gates["older60"]:     return False
    if recent60_trades  < shadow_gates["recent60"]:    return False
    if full_trades      < shadow_gates["full"]:        return False

    # Clearly positive older60: eligible without pattern match
    if older60_net_pct >= _NEAR_FLAT_THRESHOLD:
        return True

    # Near-flat (small positive) or slightly negative: needs pattern support
    if abs(older60_net_pct) < _NEAR_FLAT_THRESHOLD:
        if older60_trades >= _MIN_OLDER60_TRADES_NEAR_FLAT:
            if pattern_match_score >= _PATTERN_MATCH_THRESHOLD:
                return True

    return False


def _safe_run_id(run_id: str) -> str:
    return run_id.replace(":", "-")


def run_discovery(run_id: str | None = None) -> dict[str, Any]:
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    run_dir = RESULTS_DIR / "research_runs" / _safe_run_id(run_id)
    latest_dir = RESULTS_DIR / "latest"

    def _load(fname: str) -> Any:
        for d in (run_dir, latest_dir):
            p = d / fname
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        return None

    candidates_data = _load("older60_candidates.json")
    if not candidates_data:
        print("No candidates found. Run older60_pair_screener.py first.")
        return {}

    pattern_notes_raw = _load("pair_pattern_notes.json") or []
    pattern_notes = {(n["pair"], n["construction"], n["entry_filter"]): n for n in pattern_notes_raw}

    core:   list[dict] = []
    shadow: list[dict] = []

    for c in candidates_data.get("candidates", []):
        key  = (c["pair"], c["construction"], c["entry_filter"])
        note = pattern_notes.get(key, {})
        fp   = note.get("fingerprint", {})
        pattern_match = fingerprint_similarity(fp, fp)  # self-match for now

        eligible = is_shadow_eligible(
            older60_net_pct=c["older60_net_pct"],   older60_trades=c["older60_trades"],
            recent60_net_pct=c["recent60_net_pct"], recent60_trades=c["recent60_trades"],
            full_net_pct=c["full_net_pct"],         full_trades=c["full_trades"],
            pattern_match_score=pattern_match,
            robustness_score=c["robustness_score"],
        )
        if not eligible:
            continue

        enriched = {**c, "pattern_match_score": round(pattern_match, 4)}
        if c.get("passes_core"):
            core.append(enriched)
        else:
            shadow.append(enriched)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "core_candidates.json").write_text(json.dumps({"run_id": run_id, "candidates": core}, indent=2), encoding="utf-8")
    (run_dir / "shadow_candidates.json").write_text(json.dumps({"run_id": run_id, "candidates": shadow}, indent=2), encoding="utf-8")
    latest_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("core_candidates.json", "shadow_candidates.json"):
        (latest_dir / fname).write_text((run_dir / fname).read_text(encoding="utf-8"), encoding="utf-8")

    print(f"pattern_guided_discovery: core={len(core)} shadow={len(shadow)} → {run_dir}")
    return {"run_id": run_id, "core_candidates": core, "shadow_candidates": shadow}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    run_discovery(args.run_id)


if __name__ == "__main__":
    main()
