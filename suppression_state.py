# suppression_state.py
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import research_runtime as rr
from research_pair_registry import PAIR_RESEARCH_REGISTRY

SUPPRESSION_STATE_PATH = Path("results/suppression_state.json")
UNIFORM_CACHE_DIR = Path("data_cache_walkforward")

# State machine persistence constants
WEAK_DEFENSIVE_ENTER_BARS = 3   # bars above enter threshold → weak_defensive
WEAK_DEFENSIVE_EXIT_BARS  = 4   # bars below exit threshold  → normal
OFF_ENTER_BARS            = 4   # bars meeting portfolio off conditions → off
OFF_EXIT_BARS             = 6   # bars below exit threshold  → weak_defensive

# Portfolio-level off trigger (all three must hold simultaneously)
PORTFOLIO_OFF_WEAK_SCORE      = 0.80  # portfolio mean weak score above this
PORTFOLIO_OFF_GATE_SHARE      = 0.10  # mean gate-open share below this
PORTFOLIO_OFF_SIGNAL_DENSITY  = 0.02  # mean signal density below this


def _load_frame(pair: str, history_days: int = 120) -> pd.DataFrame:
    path = UNIFORM_CACHE_DIR / f"{pair}_15m_{history_days}d_uniform_live.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    if "vwap_k" not in df.columns:
        df["vwap_k"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    if "count" not in df.columns:
        df["count"] = 1
    return df.sort_values("ts").reset_index(drop=True)


def calibrate_thresholds(pairs: list[str], history_days: int = 120) -> dict[str, dict[str, float]]:
    """
    Compute per-pair enter/exit suppression thresholds from densified history.

    Returns:
        dict of pair → {weak_defensive_enter_threshold, weak_defensive_exit_threshold, cooldown_bars}
    """
    thresholds: dict[str, dict[str, float]] = {}
    fallback = {"weak_defensive_enter_threshold": 0.55, "weak_defensive_exit_threshold": 0.40, "cooldown_bars": 4}

    for pair in pairs:
        plan = PAIR_RESEARCH_REGISTRY.get(pair)
        if plan is None:
            thresholds[pair] = fallback.copy()
            continue
        df_raw = _load_frame(pair, history_days)
        if df_raw.empty or len(df_raw) < 16:
            thresholds[pair] = fallback.copy()
            continue
        frame = rr.build_research_frame(df_raw, plan)
        if frame.empty:
            thresholds[pair] = fallback.copy()
            continue

        # Compute rolling weak score per bar (window = 8)
        scores = [
            rr.compute_pair_weak_score(frame.iloc[max(0, i - 8):i])
            for i in range(8, len(frame) + 1)
        ]
        if not scores:
            thresholds[pair] = fallback.copy()
            continue

        s = pd.Series(scores, dtype=float)
        enter = float(np.clip(s.quantile(0.70), 0.35, 0.90))
        exit_thr = float(np.clip(s.quantile(0.45), 0.25, 0.75))
        thresholds[pair] = {
            "weak_defensive_enter_threshold": round(enter, 4),
            "weak_defensive_exit_threshold":  round(exit_thr, 4),
            "cooldown_bars": 4,
        }

    return thresholds


def evaluate_state(
    pair: str,
    current_score: float,
    prev_state: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """
    Evaluate the suppression state machine for one pair.

    prev_state keys: state, bars_in_state, bars_above_threshold, bars_below_threshold
    Returns a new state dict with all suppression_state.json per-pair fields.
    """
    enter_thr = float(thresholds.get("weak_defensive_enter_threshold", 0.55))
    exit_thr  = float(thresholds.get("weak_defensive_exit_threshold", 0.40))

    old_state    = str(prev_state.get("state", "normal"))
    bars_above   = int(prev_state.get("bars_above_threshold", 0))
    bars_below   = int(prev_state.get("bars_below_threshold", 0))
    bars_in_state = int(prev_state.get("bars_in_state", 0))

    # Update persistence counters
    if current_score >= enter_thr:
        bars_above += 1
        bars_below  = 0
    else:
        bars_below += 1
        bars_above  = 0

    # State transitions (off is set at portfolio level, not here)
    new_state = old_state
    if old_state == "normal":
        if bars_above >= WEAK_DEFENSIVE_ENTER_BARS:
            new_state = "weak_defensive"
    elif old_state == "weak_defensive":
        if bars_below >= WEAK_DEFENSIVE_EXIT_BARS:
            new_state = "normal"
    # "off" state: set externally by compute_and_write; never entered here

    if new_state != old_state:
        bars_in_state = 0
    else:
        bars_in_state += 1

    notional_multiplier = 1.0 if new_state == "normal" else 0.5
    allow_new_entries   = new_state == "normal"

    return {
        "state":                 new_state,
        "weak_score":            round(current_score, 4),
        "bars_in_state":         bars_in_state,
        "bars_above_threshold":  bars_above,
        "bars_below_threshold":  bars_below,
        "threshold":             round(enter_thr, 4),
        "exit_threshold":        round(exit_thr, 4),
        "reason_tags":           [],  # populated by compute_and_write
        "notional_multiplier":   notional_multiplier,
        "allow_new_entries":     allow_new_entries,
    }


def compute_and_write(
    pairs: list[str],
    prev_state_path: Path = SUPPRESSION_STATE_PATH,
    out_path: Path = SUPPRESSION_STATE_PATH,
    history_days: int = 120,
) -> dict[str, Any]:
    """
    Main entry point: load densified data, compute weak scores,
    evaluate state machine, write suppression_state.json.
    """
    # Load previous state for persistence
    prev_pairs: dict[str, Any] = {}
    port_bars_above_off  = 0
    port_bars_below_off  = 0
    prev_portfolio_state = "normal"
    if prev_state_path.exists():
        try:
            prev = json.loads(prev_state_path.read_text(encoding="utf-8"))
            prev_pairs = prev.get("pairs", {})
            prev_port_meta = prev.get("portfolio_meta", {})
            port_bars_above_off  = int(prev_port_meta.get("bars_above_off_threshold", 0))
            port_bars_below_off  = int(prev_port_meta.get("bars_below_off_threshold", 0))
            prev_portfolio_state = str(prev_port_meta.get("portfolio_state", "normal"))
        except Exception:
            pass

    # Use cached thresholds or calibrate fresh
    thresholds = dict(rr.SUPPRESSION_THRESHOLDS)
    if not thresholds and pairs:
        thresholds = calibrate_thresholds(pairs, history_days)
        rr.SUPPRESSION_THRESHOLDS.update(thresholds)

    pair_states:              dict[str, Any] = {}
    portfolio_scores:         list[float]    = []
    portfolio_gate_shares:    list[float]    = []
    portfolio_signal_densities: list[float]  = []

    for pair in pairs:
        plan = PAIR_RESEARCH_REGISTRY.get(pair)
        if plan is None:
            continue
        df_raw = _load_frame(pair, history_days)
        if df_raw.empty:
            continue
        frame = rr.build_research_frame(df_raw, plan)
        if frame.empty:
            continue

        score       = rr.compute_pair_weak_score(frame)
        gate_share  = float(frame["portfolio_regime_ok"].fillna(False).astype(float).mean())
        sig_density = float(
            frame["entry_signal"].fillna(False).astype(float).mean()
            if "entry_signal" in frame.columns else 0.0
        )
        portfolio_scores.append(score)
        portfolio_gate_shares.append(gate_share)
        portfolio_signal_densities.append(sig_density)

        pair_thr  = thresholds.get(pair, {
            "weak_defensive_enter_threshold": 0.55,
            "weak_defensive_exit_threshold":  0.40,
            "cooldown_bars": 4,
        })
        prev_pair = prev_pairs.get(pair, {})
        state_info = evaluate_state(pair, score, prev_pair, pair_thr)

        # Add reason tags from latest bar
        last_row = frame.iloc[-1]
        state_info["reason_tags"] = rr.weak_regime_reason_tags(last_row)
        pair_states[pair] = state_info

    # Portfolio-level state with OFF_ENTER_BARS / OFF_EXIT_BARS persistence
    portfolio_state = "normal"
    if portfolio_scores:
        port_score   = float(np.mean(portfolio_scores))
        port_gate    = float(np.mean(portfolio_gate_shares))
        port_density = float(np.mean(portfolio_signal_densities))

        off_conditions_met = (
            port_score   >= PORTFOLIO_OFF_WEAK_SCORE
            and port_gate    <= PORTFOLIO_OFF_GATE_SHARE
            and port_density <= PORTFOLIO_OFF_SIGNAL_DENSITY
        )

        # Update persistence counters
        if off_conditions_met:
            port_bars_above_off += 1
            port_bars_below_off  = 0
        else:
            port_bars_below_off += 1
            port_bars_above_off  = 0

        if prev_portfolio_state == "off":
            # off → weak_defensive requires OFF_EXIT_BARS consecutive bars below threshold
            if port_bars_below_off >= OFF_EXIT_BARS:
                portfolio_state = "weak_defensive"
            else:
                portfolio_state = "off"
        elif port_score >= 0.60 or prev_portfolio_state == "weak_defensive":
            portfolio_state = "weak_defensive"
            # Escalate to off if conditions persist for OFF_ENTER_BARS
            if port_bars_above_off >= OFF_ENTER_BARS:
                portfolio_state = "off"
        # else: portfolio_state stays "normal"
    else:
        port_bars_above_off = 0
        port_bars_below_off = 0

    # Propagate "off" to all pairs
    if portfolio_state == "off":
        for pair in pair_states:
            pair_states[pair]["state"]             = "off"
            pair_states[pair]["allow_new_entries"] = False
            pair_states[pair]["notional_multiplier"] = 0.0

    run_ts = int(time.time())
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    state_doc: dict[str, Any] = {
        "schema_version": "1.0",
        "run_ts":         run_ts,
        "source_run_id":  run_id,
        "bar_ts":         run_ts,
        "portfolio_state": portfolio_state,
        "portfolio_meta": {
            "portfolio_state":          portfolio_state,
            "bars_above_off_threshold": port_bars_above_off,
            "bars_below_off_threshold": port_bars_below_off,
        },
        "pairs":          pair_states,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(state_doc, indent=2), encoding="utf-8")
    return state_doc


def load_state(path: Path = SUPPRESSION_STATE_PATH) -> dict[str, Any]:
    """Load suppression state from JSON. Returns safe defaults if missing or corrupt."""
    if not path.exists():
        return {"schema_version": "1.0", "portfolio_state": "normal", "pairs": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"schema_version": "1.0", "portfolio_state": "normal", "pairs": {}}


def main() -> None:
    import argparse
    from research_pair_registry import active_pairs, shadow_pairs

    parser = argparse.ArgumentParser(description="Compute and write suppression state JSON.")
    parser.add_argument("--include-shadow", action="store_true")
    parser.add_argument("--history-days", type=int, default=120)
    parser.add_argument("--out", default=str(SUPPRESSION_STATE_PATH))
    args = parser.parse_args()

    pairs = list(active_pairs())
    if args.include_shadow:
        pairs = list(dict.fromkeys(pairs + list(shadow_pairs())))

    doc = compute_and_write(pairs=pairs, out_path=Path(args.out), history_days=args.history_days)
    print(f"portfolio_state={doc['portfolio_state']}  pairs={len(doc['pairs'])}")
    for pair, info in doc["pairs"].items():
        print(f"  {pair:<14} state={info['state']:<16} score={info['weak_score']:.3f}  "
              f"allow_entries={info['allow_new_entries']}")


if __name__ == "__main__":
    main()
