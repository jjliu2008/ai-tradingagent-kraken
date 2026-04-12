"""
Pair Sweep
==========
Systematic construction × entry-filter × exit-profile grid on shadow pairs.
Each combination is backtested with the portfolio regime gate applied.
Candidates are immediately screened for signal-timing correlation against
the current active book — phi correlation > CORR_THRESHOLD triggers rejection.

Results written to:
  results/sweep_candidates.json  — all passing combos, sorted by score
  results/sweep_summary.csv      — best combo per pair

Usage:
    python pair_sweep.py                           # all shadow pairs
    python pair_sweep.py --pairs BERAUSD,PARTIUSD  # specific pairs
    python pair_sweep.py --refresh-live            # refetch Kraken history
    python pair_sweep.py --history-days 60         # shorter window
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import backtest as bt
import research_runtime as rr
from research_pair_registry import (
    PAIR_RESEARCH_REGISTRY,
    PairResearchPlan,
    active_pairs,
    shadow_pairs,
)


# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

# Constructions chosen because they produced positive results in at least one
# active or shadow pair — not every construction in strategy.py, just the ones
# with demonstrated evidence.
SWEEP_CONSTRUCTIONS: tuple[str, ...] = (
    "union_closehi",        # KERNEL's construction — best active contributor
    "tc15_cap_or_mb60",     # BABY's construction
    "baseline_or_vst60",    # HOUSE's construction
    "vst60_only",           # FHE's construction
    "trend_gate",           # wide trend-gated family (4 components + 60m gate)
    "union_strong60",       # all components + strong 60m trend
    "tc15_or_atr30",        # two clean component signals
    "baseline_or_atr30",    # broad baseline + ATR momentum
    "union_volhi",          # all components + volume confirmation
)

SWEEP_FILTERS: tuple[str, ...] = (
    "base",
    "close80_volcap60",
    "close80_vwap3",
    "gate50_vwap3",
    "close80_volcap60_vwap3",
)

SWEEP_EXITS: tuple[str, ...] = ("fast", "runner")

# Quality gates
MIN_TRADES      = 4
MIN_WIN_RATE    = 0.50
MIN_NET_PCT     = 0.0

# Correlation gate: phi coefficient against any active pair
CORR_THRESHOLD  = 0.40

COMMISSION_PCT  = 0.0026
SLIPPAGE_PCT    = 0.0005

RESULTS_DIR     = Path("results")
CACHE_DIR       = Path("data_cache_walkforward")


# ---------------------------------------------------------------------------
# History loading (reuses the same cache as general_portfolio_backtest)
# ---------------------------------------------------------------------------

def _load_history(
    pair: str,
    interval: int,
    history_days: int,
    refresh_live: bool,
    trade_count: int,
    trade_pause_sec: float,
) -> pd.DataFrame | None:
    cache_path = CACHE_DIR / f"{pair}_{interval}m_{history_days}d_uniform_live.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path)
    elif not refresh_live:
        # No cache and not asked to fetch — skip silently to avoid rate-limit storms.
        # Pass --refresh-live to fetch history for uncached pairs.
        print(f"  {pair}: no cache — skipping (use --refresh-live to fetch)")
        return None
    else:
        try:
            df = bt.fetch_history(
                pair=pair,
                interval=interval,
                history_days=history_days,
                trade_count=trade_count,
                trade_pause_sec=trade_pause_sec,
            )
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)
        except Exception as exc:
            print(f"  ! Fetch failed for {pair}: {exc}")
            return None

    df.columns = [c.lower() for c in df.columns]
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    if "vwap_k" not in df.columns:
        df["vwap_k"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    if "count" not in df.columns:
        df["count"] = 1

    df = df.sort_values("ts").reset_index(drop=True)
    return df if len(df) >= 100 else None


# ---------------------------------------------------------------------------
# Per-combo helpers
# ---------------------------------------------------------------------------

def _summarize(trades: list[bt.BacktestTrade]) -> dict[str, Any]:
    if not trades:
        return {"trades": 0, "net_pct": 0.0, "win_rate": 0.0,
                "avg_trade_pct": 0.0, "max_dd_pct": 0.0}
    pnls = pd.Series([t.pnl_pct for t in trades], dtype=float)
    equity = (1.0 + pnls).cumprod()
    max_dd = float((equity / equity.cummax() - 1.0).min())
    return {
        "trades":        int(len(trades)),
        "net_pct":       float(pnls.sum()),
        "win_rate":      float((pnls > 0).mean()),
        "avg_trade_pct": float(pnls.mean()),
        "max_dd_pct":    max_dd,
    }


def _run_combo(
    df_raw: pd.DataFrame,
    pair: str,
    construction: str,
    entry_filter: str,
    exit_profile: str,
) -> dict[str, Any] | None:
    """Run one (construction, filter, exit) combo. Returns None if the construction
    is invalid for this pair's data or produces an empty frame."""
    plan = PairResearchPlan(
        pair=pair,
        construction=construction,
        entry_filter=entry_filter,
        exit_profile=exit_profile,
        status="sweep_candidate",
        note="",
    )
    try:
        frame = rr.build_research_frame(df_raw, plan)
    except (ValueError, KeyError):
        return None
    if frame.empty:
        return None

    trades = bt.run_backtest_frame(
        pair=pair,
        df=frame,
        config=rr.config_for_plan(plan),
        commission_pct=COMMISSION_PCT,
        slippage_pct=SLIPPAGE_PCT,
        construction=construction,
    )
    return {"frame": frame, "stats": _summarize(trades)}


# ---------------------------------------------------------------------------
# Correlation screen
# ---------------------------------------------------------------------------

def _phi_correlation(s1: pd.Series, s2: pd.Series) -> float:
    """Phi (Matthews) correlation coefficient between two boolean signal series.
    Aligns on the shared timestamp index. Returns 0 if either series has no variance
    or fewer than 30 overlapping bars."""
    aligned = pd.DataFrame({"a": s1.astype(float), "b": s2.astype(float)}).dropna()
    if len(aligned) < 30:
        return 0.0
    std_a = aligned["a"].std()
    std_b = aligned["b"].std()
    if std_a == 0 or std_b == 0:
        return 0.0
    return float(aligned["a"].corr(aligned["b"]))


def _load_active_signal_series(
    history_days: int,
    interval: int,
    refresh_live: bool,
    trade_count: int,
    trade_pause_sec: float,
) -> dict[str, pd.Series]:
    """
    Build entry_signal_pre_regime series (indexed by ts) for each active pair.
    These form the correlation baseline that candidate signals are checked against.
    """
    result: dict[str, pd.Series] = {}
    for pair in active_pairs():
        plan = PAIR_RESEARCH_REGISTRY.get(pair)
        if plan is None:
            continue
        df_raw = _load_history(pair, interval, history_days, refresh_live,
                               trade_count, trade_pause_sec)
        if df_raw is None:
            continue
        try:
            frame = rr.build_research_frame(df_raw, plan)
        except Exception:
            continue
        if frame.empty:
            continue
        sig = frame.set_index("ts")["entry_signal_pre_regime"].astype(bool)
        result[pair] = sig
        print(f"  Active {pair}: {int(sig.sum())} pre-regime signals over {len(sig)} bars")
    return result


# ---------------------------------------------------------------------------
# Core sweep function
# ---------------------------------------------------------------------------

def sweep_pair(
    pair: str,
    df_raw: pd.DataFrame,
    active_signals: dict[str, pd.Series],
) -> list[dict[str, Any]]:
    """
    Run the full construction × filter × exit grid for one pair.
    Filters by quality thresholds and phi correlation with active pairs.
    Returns all passing candidates sorted by score descending.
    """
    candidates: list[dict[str, Any]] = []
    n_combos = len(SWEEP_CONSTRUCTIONS) * len(SWEEP_FILTERS) * len(SWEEP_EXITS)
    tried = rejected_quality = rejected_corr = 0

    for construction in SWEEP_CONSTRUCTIONS:
        for entry_filter in SWEEP_FILTERS:
            for exit_profile in SWEEP_EXITS:
                tried += 1
                result = _run_combo(df_raw, pair, construction, entry_filter, exit_profile)
                if result is None:
                    continue

                stats = result["stats"]

                # Quality gate
                if (stats["trades"] < MIN_TRADES
                        or stats["net_pct"] <= MIN_NET_PCT
                        or stats["win_rate"] < MIN_WIN_RATE):
                    rejected_quality += 1
                    continue

                # Correlation gate — check phi against each active pair
                candidate_sig = (
                    result["frame"]
                    .set_index("ts")["entry_signal_pre_regime"]
                    .astype(bool)
                )
                max_corr = 0.0
                most_correlated = ""
                for active_pair, active_sig in active_signals.items():
                    corr = _phi_correlation(candidate_sig, active_sig)
                    if corr > max_corr:
                        max_corr = corr
                        most_correlated = active_pair

                if max_corr > CORR_THRESHOLD:
                    rejected_corr += 1
                    continue

                score = stats["net_pct"] * stats["win_rate"]
                candidates.append({
                    "pair":                  pair,
                    "construction":          construction,
                    "entry_filter":          entry_filter,
                    "exit_profile":          exit_profile,
                    "score":                 round(score, 6),
                    "max_corr_with_active":  round(max_corr, 4),
                    "most_correlated_pair":  most_correlated,
                    **{k: (round(v, 6) if isinstance(v, float) else v)
                       for k, v in stats.items()},
                })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    print(
        f"  {pair}: {tried}/{n_combos} valid combos  "
        f"-> {rejected_quality} failed quality  "
        f"-> {rejected_corr} failed corr  "
        f"-> {len(candidates)} passed"
    )
    return candidates


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construction × filter × exit sweep on shadow pairs."
    )
    parser.add_argument(
        "--pairs", default="",
        help="Comma-separated pairs. Defaults to all shadow pairs in the registry.",
    )
    parser.add_argument("--interval",        type=int,   default=15)
    parser.add_argument("--history-days",    type=int,   default=120)
    parser.add_argument("--refresh-live",    action="store_true",
                        help="Ignore cache and refetch history from Kraken.")
    parser.add_argument("--trade-count",     type=int,   default=5000)
    parser.add_argument("--trade-pause-sec", type=float, default=1.2,
                        help="Seconds between Kraken API calls when fetching uncached history.")
    parser.add_argument("--candidates-json",
                        default="results/sweep_candidates.json")
    parser.add_argument("--summary-csv",
                        default="results/sweep_summary.csv")
    args = parser.parse_args()

    pairs_to_sweep = (
        [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
        if args.pairs.strip()
        else list(shadow_pairs())
    )

    n_combos_each = len(SWEEP_CONSTRUCTIONS) * len(SWEEP_FILTERS) * len(SWEEP_EXITS)

    print("\nPAIR SWEEP")
    print(f"  Pairs       : {pairs_to_sweep}")
    print(f"  Constructions: {len(SWEEP_CONSTRUCTIONS)}  "
          f"Filters: {len(SWEEP_FILTERS)}  "
          f"Exits: {len(SWEEP_EXITS)}  "
          f"-> {n_combos_each} combos/pair")
    print(f"  History     : {args.history_days}d @ {args.interval}m bars")
    print(f"  Quality gate: trades>={MIN_TRADES}  "
          f"win_rate>={MIN_WIN_RATE:.0%}  net_pct>{MIN_NET_PCT:.0%}")
    print(f"  Corr gate   : phi<={CORR_THRESHOLD} vs active book")
    print()

    # Build correlation baseline from active pairs
    print("Loading active pair signal series for correlation baseline...")
    active_signals = _load_active_signal_series(
        history_days=args.history_days,
        interval=args.interval,
        refresh_live=args.refresh_live,
        trade_count=args.trade_count,
        trade_pause_sec=args.trade_pause_sec,
    )
    print()

    all_candidates: list[dict[str, Any]] = []
    summary_rows:   list[dict[str, Any]] = []

    for pair in pairs_to_sweep:
        print(f"-- {pair} --")
        df_raw = _load_history(
            pair=pair,
            interval=args.interval,
            history_days=args.history_days,
            refresh_live=args.refresh_live,
            trade_count=args.trade_count,
            trade_pause_sec=args.trade_pause_sec,
        )
        if df_raw is None:
            print(f"  No history — skipped")
            summary_rows.append({"pair": pair, "total_passing_combos": 0})
            print()
            continue

        candidates = sweep_pair(pair, df_raw, active_signals)
        all_candidates.extend(candidates)

        if candidates:
            best = candidates[0]  # already sorted by score
            summary_rows.append({
                "pair":                 pair,
                "best_construction":    best["construction"],
                "best_entry_filter":    best["entry_filter"],
                "best_exit_profile":    best["exit_profile"],
                "best_score":           best["score"],
                "best_net_pct":         best["net_pct"],
                "best_win_rate":        best["win_rate"],
                "best_trades":          best["trades"],
                "max_corr_with_active": best["max_corr_with_active"],
                "most_correlated_pair": best["most_correlated_pair"],
                "total_passing_combos": len(candidates),
            })
        else:
            summary_rows.append({"pair": pair, "total_passing_combos": 0})
        print()

    # Write results
    all_candidates.sort(key=lambda c: c["score"], reverse=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(args.candidates_json, "w") as f:
        json.dump(all_candidates, f, indent=2)

    pd.DataFrame(summary_rows).to_csv(args.summary_csv, index=False)

    print("-- SWEEP COMPLETE --")
    print(f"  Total passing candidates : {len(all_candidates)}")
    print(f"  Candidates JSON          : {args.candidates_json}")
    print(f"  Summary CSV              : {args.summary_csv}")

    if all_candidates:
        print("\nTop 10 candidates:")
        header = f"  {'PAIR':<12} {'CONSTRUCTION':<24} {'FILTER':<28} {'EXIT':<8} "
        header += f"{'TRADES':>6} {'NET':>8} {'WIN':>6} {'CORR':>6}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for c in all_candidates[:10]:
            print(
                f"  {c['pair']:<12} {c['construction']:<24} {c['entry_filter']:<28} "
                f"{c['exit_profile']:<8} {c['trades']:>6} {c['net_pct']:>+.3%} "
                f"{c['win_rate']:>5.0%} {c['max_corr_with_active']:>6.2f}"
            )


if __name__ == "__main__":
    main()
