# older60_pair_screener.py
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import backtest as bt
import research_runtime as rr
import strategy as strat


RESULTS_DIR       = Path("results")
UNIFORM_CACHE_DIR = Path("data_cache_walkforward")
COMMISSION_PCT    = 0.0026
SLIPPAGE_PCT      = 0.0005
TOP_K             = 3

CONSTRUCTIONS = [
    "tc15_tighter_volume_cap", "tc15_cap_or_mb60", "vst60_only",
    "atr30_only", "baseline_or_atr30", "baseline_or_vst60",
    "union_closehi", "union_volhi", "trend_gate", "baseline_or_tc15",
]
ENTRY_FILTERS = [
    "base", "score45", "close70", "close80", "gate30",
    "volcap45", "close80_volcap60", "gate50_vwap3",
]
EXIT_PROFILES = ["base", "medium", "fast", "tight", "runner"]


def robustness_score(
    older60_net_pct: float,  older60_trades: int,
    recent60_net_pct: float, recent60_trades: int,
    full_net_pct: float,     full_trades: int,
    full_max_dd_pct: float,
    concentration_share: float,
    max_signal_correlation: float,
) -> float:
    older60_term  = older60_net_pct  * min(1.0, math.sqrt(max(0, older60_trades)  / 5.0))
    recent60_term = recent60_net_pct * min(1.0, math.sqrt(max(0, recent60_trades) / 3.0))
    full120_term  = full_net_pct     * min(1.0, math.sqrt(max(0, full_trades)     / 8.0))

    drawdown_penalty      = max(0.0, -float(full_max_dd_pct))
    concentration_penalty = max(0.0, float(concentration_share) - 0.35) / 0.15
    correlation_penalty   = max(0.0, float(max_signal_correlation) - 0.70) / 0.30

    return (
        1.50 * older60_term
        + 1.00 * recent60_term
        + 0.75 * full120_term
        - 0.50 * drawdown_penalty
        - 0.25 * concentration_penalty
        - 0.25 * correlation_penalty
    )


_TIER_GATES = {
    "core":   {"older60": 5, "recent60": 3, "full": 8},
    "shadow": {"older60": 4, "recent60": 1, "full": 4},
}


def passes_hard_gates(
    tier: str,
    older60_net_pct: float, older60_trades: int,
    recent60_net_pct: float, recent60_trades: int,
    full_net_pct: float, full_trades: int,
) -> bool:
    g = _TIER_GATES.get(tier, _TIER_GATES["shadow"])
    if older60_trades  < g["older60"]:  return False
    if recent60_trades < g["recent60"]: return False
    if full_trades     < g["full"]:     return False
    if full_net_pct    <= 0.0:          return False
    if recent60_net_pct <= 0.0:         return False
    if tier == "core" and older60_net_pct <= 0.0:
        return False
    return True


def _load_uniform(pair: str, history_days: int) -> pd.DataFrame:
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


def _summarize_trades(trades: list) -> dict[str, Any]:
    if not trades:
        return {"trades": 0, "net_pct": 0.0, "win_rate": 0.0, "max_dd_pct": 0.0}
    pnls = pd.Series([t.pnl_pct for t in trades], dtype=float)
    equity = (1.0 + pnls).cumprod()
    max_dd = float((equity / equity.cummax() - 1.0).min())
    return {
        "trades": len(trades),
        "net_pct": float(pnls.sum()),
        "win_rate": float((pnls > 0).mean()),
        "max_dd_pct": max_dd,
    }


def screen_pair(
    pair: str,
    df_raw: pd.DataFrame,
    history_days: int,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    if df_raw.empty or len(df_raw) < 96:
        return []

    max_ts   = int(df_raw["ts"].iloc[-1])
    split_ts = max_ts - 60 * 24 * 60 * 60

    from research_pair_registry import PairResearchPlan

    candidates: list[dict[str, Any]] = []

    # Cache frames per (construction, entry_filter).
    # build_ensemble_frame is the expensive step; config_for_plan only changes
    # exit params (min_stop_pct, target_pct, max_hold_bars) which are not used
    # during feature computation — so the frame is identical across exit_profiles.
    # This reduces frame builds from 400 → 80 per pair.
    frame_cache: dict[tuple[str, str], pd.DataFrame] = {}

    for construction in CONSTRUCTIONS:
        # Build base ensemble frame once per construction using default exit config.
        base_dummy = PairResearchPlan(
            pair=pair,
            construction=construction,
            entry_filter="base",
            exit_profile="base",
            status="candidate",
            note="screener candidate",
        )
        try:
            base_frame = strat.build_ensemble_frame(
                df_raw, construction=construction, config=rr.config_for_plan(base_dummy)
            )
        except Exception:
            continue
        if base_frame.empty:
            continue

        for entry_filter in ENTRY_FILTERS:
            key = (construction, entry_filter)
            if key not in frame_cache:
                try:
                    filtered = base_frame.copy()
                    filtered["entry_signal_pre_regime"] = rr.apply_entry_filter(filtered, entry_filter)
                    filtered["portfolio_regime_ok"]     = rr.apply_portfolio_regime_gate(filtered)
                    filtered["entry_signal"]            = (
                        filtered["entry_signal_pre_regime"] & filtered["portfolio_regime_ok"]
                    )
                    frame_cache[key] = rr.add_weak_regime_features(filtered)
                except Exception:
                    frame_cache[key] = pd.DataFrame()

            frame = frame_cache[key]
            if frame.empty:
                continue

            for exit_profile in EXIT_PROFILES:
                try:
                    fake_plan = PairResearchPlan(
                        pair=pair,
                        construction=construction,
                        entry_filter=entry_filter,
                        exit_profile=exit_profile,
                        status="candidate",
                        note="screener candidate",
                    )
                    config = rr.config_for_plan(fake_plan)
                    trades = bt.run_backtest_frame(
                        pair=pair,
                        df=frame,
                        config=config,
                        commission_pct=COMMISSION_PCT,
                        slippage_pct=SLIPPAGE_PCT,
                        construction=construction,
                    )
                    if not trades:
                        continue

                    older60  = [t for t in trades if t.entry_ts < split_ts]
                    recent60 = [t for t in trades if t.entry_ts >= split_ts]
                    full_stats     = _summarize_trades(trades)
                    older60_stats  = _summarize_trades(older60)
                    recent60_stats = _summarize_trades(recent60)

                    score = robustness_score(
                        older60_net_pct=older60_stats["net_pct"],
                        older60_trades=older60_stats["trades"],
                        recent60_net_pct=recent60_stats["net_pct"],
                        recent60_trades=recent60_stats["trades"],
                        full_net_pct=full_stats["net_pct"],
                        full_trades=full_stats["trades"],
                        full_max_dd_pct=full_stats["max_dd_pct"],
                        concentration_share=0.0,
                        max_signal_correlation=0.0,
                    )

                    candidates.append({
                        "pair": pair,
                        "construction": construction,
                        "entry_filter": entry_filter,
                        "exit_profile": exit_profile,
                        "robustness_score": round(score, 6),
                        "full_trades":      full_stats["trades"],
                        "full_net_pct":     round(full_stats["net_pct"], 6),
                        "full_max_dd_pct":  round(full_stats["max_dd_pct"], 6),
                        "older60_trades":   older60_stats["trades"],
                        "older60_net_pct":  round(older60_stats["net_pct"], 6),
                        "recent60_trades":  recent60_stats["trades"],
                        "recent60_net_pct": round(recent60_stats["net_pct"], 6),
                        "passes_core":   passes_hard_gates("core",
                            older60_stats["net_pct"], older60_stats["trades"],
                            recent60_stats["net_pct"], recent60_stats["trades"],
                            full_stats["net_pct"], full_stats["trades"]),
                        "passes_shadow": passes_hard_gates("shadow",
                            older60_stats["net_pct"], older60_stats["trades"],
                            recent60_stats["net_pct"], recent60_stats["trades"],
                            full_stats["net_pct"], full_stats["trades"]),
                    })
                except Exception:
                    continue

    candidates.sort(key=lambda c: c["robustness_score"], reverse=True)
    return candidates[:top_k]


def _safe_run_id(run_id: str) -> str:
    """Sanitize run_id for use as a Windows directory name (colons → hyphens)."""
    return run_id.replace(":", "-")


def run_screener(history_days: int, top_k: int = TOP_K, run_id: str | None = None) -> dict[str, Any]:
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    run_dir = RESULTS_DIR / "research_runs" / _safe_run_id(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    all_candidates: list[dict[str, Any]] = []
    paths = sorted(UNIFORM_CACHE_DIR.glob(f"*_15m_{history_days}d_uniform_live.csv"))

    for path in paths:
        pair = path.name.split("_", 1)[0]
        df_raw = _load_uniform(pair, history_days)
        results = screen_pair(pair, df_raw, history_days, top_k=top_k)
        all_candidates.extend(results)
        print(f"  {pair}: {len(results)} candidates")

    output = {"run_id": run_id, "history_days": history_days, "top_k": top_k, "candidates": all_candidates}
    out_path = run_dir / "older60_candidates.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    latest_dir = RESULTS_DIR / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "older60_candidates.json").write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\nolder60_screener: {len(paths)} pairs -> {len(all_candidates)} candidates -> {out_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-days", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    run_screener(args.history_days, top_k=args.top_k, run_id=args.run_id)


if __name__ == "__main__":
    main()
