from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import expanded_screener as ex
import strategy as strat


@dataclass(frozen=True)
class ExitSpec:
    hold_bars: int
    stop_pct: float
    target_pct: float | None = None
    trail_activation_pct: float | None = None
    trail_stop_pct: float | None = None


@dataclass
class EnsembleResult:
    construction: str
    exit_profile: str
    trades_full: int
    net_full: float
    avg_full: float
    win_full: float
    trades_recent: int
    net_recent: float
    avg_recent: float
    win_recent: float
    score: float


def load_interval_frames(history: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    master = ex.add_expanded_features(strat.compute_features(ex.resample_ohlcv(history, 15)))
    master = master.set_index("ts").sort_index()

    cache: dict[int, pd.DataFrame] = {15: master.reset_index()}
    for interval in (30, 60):
        frame = ex.resample_ohlcv(history, interval)
        frame = ex.add_expanded_features(strat.compute_features(frame))
        cache[interval] = frame
    return master, cache


def map_signal(master: pd.DataFrame, interval_df: pd.DataFrame, strategy_name: str, interval: int) -> pd.Series:
    masks = {
        candidate_name: mask.fillna(False)
        for candidate_name, mask, _ in ex.build_all_candidates(interval_df, interval)
    }
    signal_ts = interval_df.loc[masks[strategy_name], "ts"].astype(int)
    effective_ts = signal_ts + (interval - strat.MASTER_INTERVAL_MINUTES) * 60
    mapped = pd.Series(False, index=master.index, dtype=bool)
    mapped.loc[mapped.index.intersection(effective_ts)] = True
    return mapped


def simulate(master: pd.DataFrame, entry_mask: pd.Series, exit_spec: ExitSpec, side_cost_pct: float) -> list[tuple[int, float]]:
    df = master.reset_index()
    trades: list[tuple[int, float]] = []
    i = 0
    last_index = len(df) - 1

    while i < last_index:
        if not bool(entry_mask.iloc[i]):
            i += 1
            continue

        entry_bar = i + 1
        if entry_bar > last_index:
            break

        entry_price = float(df.iloc[entry_bar]["open"]) * (1 + side_cost_pct)
        stop_price = entry_price * (1 - exit_spec.stop_pct)
        target_price = (
            entry_price * (1 + exit_spec.target_pct)
            if exit_spec.target_pct is not None
            else None
        )
        best_high = entry_price
        max_exit_bar = min(last_index, entry_bar + exit_spec.hold_bars)
        exit_bar = max_exit_bar
        exit_price = float(df.iloc[max_exit_bar]["close"]) * (1 - side_cost_pct)

        for bar in range(entry_bar, max_exit_bar + 1):
            row = df.iloc[bar]
            high_price = float(row["high"])
            low_price = float(row["low"])
            close_price = float(row["close"])
            best_high = max(best_high, high_price)

            trail_stop = None
            if (
                exit_spec.trail_activation_pct is not None
                and exit_spec.trail_stop_pct is not None
                and (best_high / entry_price - 1) >= exit_spec.trail_activation_pct
            ):
                trail_stop = best_high * (1 - exit_spec.trail_stop_pct)

            if low_price <= stop_price:
                exit_bar = bar
                exit_price = stop_price * (1 - side_cost_pct)
                break
            if target_price is not None and high_price >= target_price:
                exit_bar = bar
                exit_price = target_price * (1 - side_cost_pct)
                break
            if trail_stop is not None and low_price <= trail_stop:
                exit_bar = bar
                exit_price = trail_stop * (1 - side_cost_pct)
                break
            if bar == max_exit_bar:
                exit_bar = bar
                exit_price = close_price * (1 - side_cost_pct)

        trades.append((int(df.iloc[entry_bar]["ts"]), exit_price / entry_price - 1))
        i = exit_bar + 1

    return trades


def summarize(
    construction: str,
    exit_profile: str,
    trades: list[tuple[int, float]],
    recent_cutoff: int,
) -> EnsembleResult:
    pnls = np.asarray([pnl for _, pnl in trades], dtype=float)
    recent = np.asarray([pnl for ts, pnl in trades if ts >= recent_cutoff], dtype=float)

    def avg(arr: np.ndarray) -> float:
        return float(arr.mean()) if len(arr) else 0.0

    def win(arr: np.ndarray) -> float:
        return float((arr > 0).mean()) if len(arr) else 0.0

    net_full = float(pnls.sum()) if len(pnls) else 0.0
    net_recent = float(recent.sum()) if len(recent) else 0.0
    score = net_full + 1.5 * net_recent + 0.002 * len(pnls)

    return EnsembleResult(
        construction=construction,
        exit_profile=exit_profile,
        trades_full=len(pnls),
        net_full=net_full,
        avg_full=avg(pnls),
        win_full=win(pnls),
        trades_recent=len(recent),
        net_recent=net_recent,
        avg_recent=avg(recent),
        win_recent=win(recent),
        score=score,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GIGA ensemble screener")
    parser.add_argument("--pair", default="GIGAUSD")
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--recent-days", type=int, default=60)
    parser.add_argument("--fee-pct", type=float, default=0.0026)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--results-csv", default="giga_ensemble_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = pd.read_csv(Path(args.cache_dir) / f"{args.pair}_15m_120d_end_latest.csv")
    master, interval_cache = load_interval_frames(history)
    recent_cutoff = int(master.index.max()) - args.recent_days * 24 * 60 * 60
    side_cost_pct = args.fee_pct + args.slippage_pct

    signals = {
        "mb60": map_signal(master, interval_cache[60], "momentum_breakout", 60),
        "mbt30": map_signal(master, interval_cache[30], "momentum_breakout_trail", 30),
        "vst60": map_signal(master, interval_cache[60], "volume_spike_trend", 60),
        "tc30": map_signal(master, interval_cache[30], "triple_confluence", 30),
        "tc15": map_signal(master, interval_cache[15], "triple_confluence", 15),
        "atr30": map_signal(master, interval_cache[30], "atr_squeeze_expand", 30),
    }

    reg60_source = interval_cache[60].copy()
    reg60_source["effective_ts"] = (
        reg60_source["ts"].astype(int) + (60 - strat.MASTER_INTERVAL_MINUTES) * 60
    )
    reg60 = (
        reg60_source.set_index("effective_ts")
        .reindex(master.index)
        .ffill()
        .infer_objects(copy=False)
    )

    constructions = {
        "baseline_15m_exec": signals["mb60"],
        "consensus_2_all": (
            signals["mb60"].astype(int)
            + signals["mbt30"].astype(int)
            + signals["vst60"].astype(int)
            + signals["tc30"].astype(int)
            + signals["tc15"].astype(int)
            + signals["atr30"].astype(int)
        )
        >= 2,
        "breakout_stack": signals["mbt30"] | signals["tc30"] | (signals["mb60"] & signals["tc15"]),
        "high_tf_gate": (
            signals["mbt30"] | signals["tc30"] | signals["atr30"] | signals["tc15"]
        ) & (signals["mb60"] | signals["vst60"]),
        "trend_gate": (
            signals["mbt30"] | signals["tc30"] | signals["atr30"] | signals["tc15"]
        )
        & (reg60["ema_fast"] > reg60["ema_slow"])
        & (reg60["trend_strength"] > 0.0015),
        "strong_trend_gate": (
            signals["mbt30"] | signals["tc30"] | signals["atr30"]
        )
        & (reg60["ema_fast"] > reg60["ema_slow"])
        & (reg60["trend_strength"] > 0.0020)
        & (reg60["momentum_medium"] > 0.010),
        "volume_stack": (signals["vst60"] | signals["mb60"]) & (
            signals["mbt30"] | signals["tc30"] | signals["atr30"]
        ),
        "fast_union": signals["mbt30"] | signals["tc30"] | signals["tc15"] | signals["atr30"],
    }

    exit_profiles = {
        "baseline": ExitSpec(hold_bars=12, stop_pct=0.015, target_pct=0.045),
        "trail": ExitSpec(
            hold_bars=16,
            stop_pct=0.015,
            target_pct=None,
            trail_activation_pct=0.020,
            trail_stop_pct=0.010,
        ),
        "tight": ExitSpec(hold_bars=8, stop_pct=0.012, target_pct=0.030),
        "hybrid": ExitSpec(
            hold_bars=12,
            stop_pct=0.012,
            target_pct=None,
            trail_activation_pct=0.015,
            trail_stop_pct=0.008,
        ),
    }

    rows: list[dict] = []
    for construction, entry_mask in constructions.items():
        for exit_profile, exit_spec in exit_profiles.items():
            if construction == "baseline_15m_exec" and exit_profile != "baseline":
                continue
            trades = simulate(master, entry_mask.fillna(False).astype(bool), exit_spec, side_cost_pct)
            rows.append(
                summarize(
                    construction=construction,
                    exit_profile=exit_profile,
                    trades=trades,
                    recent_cutoff=recent_cutoff,
                ).__dict__
            )

    results = pd.DataFrame(rows).sort_values(
        ["score", "net_recent", "net_full"],
        ascending=[False, False, False],
    )

    float_cols = {
        "net_full",
        "avg_full",
        "win_full",
        "net_recent",
        "avg_recent",
        "win_recent",
        "score",
    }
    formatters = {
        col: (lambda value: f"{value:+.3%}")
        for col in float_cols
    }
    print(f"\n{'=' * 104}")
    print(f" GIGA ENSEMBLE SCREEN | pair={args.pair} | rows={len(results)}")
    print(f"{'=' * 104}")
    cols = [
        "construction",
        "exit_profile",
        "trades_full",
        "net_full",
        "win_full",
        "trades_recent",
        "net_recent",
        "win_recent",
        "score",
    ]
    print(results[cols].head(12).to_string(index=False, formatters=formatters))

    if args.results_csv:
        results.to_csv(args.results_csv, index=False)
        print(f"\nWrote {len(results)} rows to {args.results_csv}")


if __name__ == "__main__":
    main()
