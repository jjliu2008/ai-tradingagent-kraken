# segment_diagnostics.py
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

RESULTS_DIR = Path("results")


def bucket_gate_trend(value: float) -> str:
    if value <= 0.0:    return "negative"
    if value < 0.003:   return "weak"
    if value < 0.010:   return "moderate"
    return "strong"


def bucket_atr(value: float) -> str:
    if value < 0.004:   return "very_low"
    if value < 0.007:   return "low"
    if value < 0.012:   return "moderate"
    return "high"


def bucket_efficiency(value: float) -> str:
    if value < 0.20:    return "choppy"
    if value < 0.35:    return "low"
    if value < 0.55:    return "moderate"
    return "directional"


def bucket_vwap_dist(value: float) -> str:
    if value < 0.005:   return "near"
    if value < 0.015:   return "moderate"
    return "extended"


def bucket_close_quality(value: float) -> str:
    if value < 0.50:    return "weak"
    if value < 0.70:    return "moderate"
    return "strong"


def bucket_volume(value: float) -> str:
    if value < 0.80:    return "low"
    if value < 1.50:    return "normal"
    return "elevated"


def fingerprint_from_frame(frame: pd.DataFrame) -> dict[str, str]:
    def _mean(col: str) -> float:
        if col not in frame.columns:
            return 0.0
        return float(frame[col].astype(float).mean())

    return {
        "gate_trend_bucket":    bucket_gate_trend(_mean("gate_trend_strength_60")),
        "atr_bucket":           bucket_atr(_mean("atr_pct")),
        "efficiency_bucket":    bucket_efficiency(_mean("efficiency_ratio_8")),
        "vwap_dist_bucket":     bucket_vwap_dist(_mean("distance_from_vwap")),
        "close_quality_bucket": bucket_close_quality(_mean("close_location")),
        "volume_bucket":        bucket_volume(_mean("volume_ratio")),
        "component_count_mean": round(_mean("component_count"), 2),
    }


def extract_pair_diagnostics(
    pair: str,
    construction: str,
    entry_filter: str,
    frame: pd.DataFrame,
    trades: list[Any],
    split_ts: int,
) -> dict[str, Any]:
    older60  = [t for t in trades if t.entry_ts < split_ts]
    recent60 = [t for t in trades if t.entry_ts >= split_ts]

    older60_frame  = frame[frame["ts"] < split_ts]  if "ts" in frame.columns else frame
    recent60_frame = frame[frame["ts"] >= split_ts] if "ts" in frame.columns else frame

    older60_exit_mix  = dict(Counter(getattr(t, "exit_reason", "?") for t in older60))
    recent60_exit_mix = dict(Counter(getattr(t, "exit_reason", "?") for t in recent60))

    return {
        "pair":              pair,
        "construction":      construction,
        "entry_filter":      entry_filter,
        "older60_trades":    len(older60),
        "recent60_trades":   len(recent60),
        "older60_net_pct":   round(sum(getattr(t, "pnl_pct", 0.0) for t in older60), 6),
        "recent60_net_pct":  round(sum(getattr(t, "pnl_pct", 0.0) for t in recent60), 6),
        "older60_fingerprint":  fingerprint_from_frame(older60_frame)  if not older60_frame.empty  else {},
        "recent60_fingerprint": fingerprint_from_frame(recent60_frame) if not recent60_frame.empty else {},
        "older60_exit_mix":  older60_exit_mix,
        "recent60_exit_mix": recent60_exit_mix,
        "failure_note": (
            "no_older60_trades" if not older60
            else ("negative_older60" if sum(getattr(t, "pnl_pct", 0.0) for t in older60) < 0
            else "ok")
        ),
    }


def _safe_run_id(run_id: str) -> str:
    return run_id.replace(":", "-")


def run_diagnostics(run_id: str | None = None) -> dict[str, Any]:
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    run_dir  = RESULTS_DIR / "research_runs" / _safe_run_id(run_id)
    candidates_path = run_dir / "older60_candidates.json"
    if not candidates_path.exists():
        latest = RESULTS_DIR / "latest" / "older60_candidates.json"
        if not latest.exists():
            print("No older60_candidates.json found. Run older60_pair_screener.py first.")
            return {}
        candidates_path = latest

    data = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidates = data.get("candidates", [])

    pair_pattern_notes: list[dict] = []
    failure_notes: list[dict] = []

    for c in candidates:
        fp = {
            "gate_trend_bucket":  "moderate",
            "atr_bucket":         "moderate",
            "older60_net_pct":    c["older60_net_pct"],
            "recent60_net_pct":   c["recent60_net_pct"],
            "older60_trades":     c["older60_trades"],
            "recent60_trades":    c["recent60_trades"],
            "exit_mix_note":      "from_screener",
        }
        pair_pattern_notes.append({
            "pair": c["pair"], "construction": c["construction"],
            "entry_filter": c["entry_filter"], "fingerprint": fp,
        })
        if c["older60_net_pct"] <= 0:
            failure_notes.append({
                "pair": c["pair"], "construction": c["construction"],
                "reason": "negative_older60", "older60_net_pct": c["older60_net_pct"],
            })

    older60_behavior_summary = {
        "run_id":           run_id,
        "total_candidates": len(candidates),
        "pairs_with_positive_older60": len({c["pair"] for c in candidates if c["older60_net_pct"] > 0}),
        "pairs_passing_core":   len({c["pair"] for c in candidates if c.get("passes_core")}),
        "pairs_passing_shadow": len({c["pair"] for c in candidates if c.get("passes_shadow")}),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pair_pattern_notes.json").write_text(json.dumps(pair_pattern_notes, indent=2), encoding="utf-8")
    (run_dir / "older60_behavior_summary.json").write_text(json.dumps(older60_behavior_summary, indent=2), encoding="utf-8")
    (run_dir / "failure_notes.json").write_text(json.dumps(failure_notes, indent=2), encoding="utf-8")

    latest_dir = RESULTS_DIR / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("pair_pattern_notes.json", "older60_behavior_summary.json", "failure_notes.json"):
        (latest_dir / fname).write_text((run_dir / fname).read_text(encoding="utf-8"), encoding="utf-8")

    print(f"segment_diagnostics: {len(candidates)} candidates → {run_dir}")
    return {"pair_pattern_notes": pair_pattern_notes, "summary": older60_behavior_summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    run_diagnostics(args.run_id)


if __name__ == "__main__":
    main()
