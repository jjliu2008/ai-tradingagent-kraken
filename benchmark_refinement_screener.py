from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import giga_ensemble_screener as gs


def build_regime_gate(master: pd.DataFrame, interval_cache: dict[int, pd.DataFrame]) -> pd.DataFrame:
    reg60_source = interval_cache[60].copy()
    reg60_source["effective_ts"] = (
        reg60_source["ts"].astype(int) + (60 - 15) * 60
    )
    return (
        reg60_source.set_index("effective_ts")
        .reindex(master.index)
        .ffill()
        .infer_objects(copy=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refinement screener around the no-lookahead 15m benchmark"
    )
    parser.add_argument("--pair", default="GIGAUSD")
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--recent-days", type=int, default=60)
    parser.add_argument("--fee-pct", type=float, default=0.0026)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--results-csv", default="benchmark_refinement_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = pd.read_csv(Path(args.cache_dir) / f"{args.pair}_15m_120d_end_latest.csv")
    master, interval_cache = gs.load_interval_frames(history)
    recent_cutoff = int(master.index.max()) - args.recent_days * 24 * 60 * 60
    side_cost_pct = args.fee_pct + args.slippage_pct

    signals = {
        "mb60": gs.map_signal(master, interval_cache[60], "momentum_breakout", 60),
        "vst60": gs.map_signal(master, interval_cache[60], "volume_spike_trend", 60),
        "tc30": gs.map_signal(master, interval_cache[30], "triple_confluence", 30),
        "tc15": gs.map_signal(master, interval_cache[15], "triple_confluence", 15),
        "atr30": gs.map_signal(master, interval_cache[30], "atr_squeeze_expand", 30),
    }

    reg60 = build_regime_gate(master, interval_cache)
    strong_60 = (
        (reg60["ema_fast"] > reg60["ema_slow"])
        & (reg60["trend_strength"] > 0.0015)
    ).fillna(False)
    very_strong_60 = (
        (reg60["ema_fast"] > reg60["ema_slow"])
        & (reg60["trend_strength"] > 0.0030)
    ).fillna(False)
    close_high = (master["close_location"] > 0.75).fillna(False)
    volume_high = (master["volume_ratio"] > 1.2).fillna(False)

    constructions = {
        "baseline_mb60": signals["mb60"],
        "tc15_only": signals["tc15"],
        "vst60_only": signals["vst60"],
        "tc30_only": signals["tc30"],
        "atr30_only": signals["atr30"],
        "baseline_or_tc15": signals["mb60"] | signals["tc15"],
        "baseline_or_tc30": signals["mb60"] | signals["tc30"],
        "baseline_or_atr30": signals["mb60"] | signals["atr30"],
        "baseline_or_vst60": signals["mb60"] | signals["vst60"],
        "tc15_or_tc30": signals["tc15"] | signals["tc30"],
        "tc15_or_atr30": signals["tc15"] | signals["atr30"],
        "tc15_or_vst60": signals["tc15"] | signals["vst60"],
        "core_union_no_mbt30": (
            signals["mb60"] | signals["tc15"] | signals["tc30"] | signals["atr30"] | signals["vst60"]
        ),
        "consensus_pos2": (
            signals["mb60"].astype(int)
            + signals["tc15"].astype(int)
            + signals["tc30"].astype(int)
            + signals["atr30"].astype(int)
            + signals["vst60"].astype(int)
        ) >= 2,
        "tc15_strong60": signals["tc15"] & strong_60,
        "tc15_very60": signals["tc15"] & very_strong_60,
        "vst60_strong60": signals["vst60"] & strong_60,
        "union_strong60": (
            signals["mb60"] | signals["tc15"] | signals["tc30"] | signals["atr30"] | signals["vst60"]
        ) & strong_60,
        "union_closehi": (
            signals["mb60"] | signals["tc15"] | signals["tc30"] | signals["atr30"] | signals["vst60"]
        ) & close_high,
        "union_volhi": (
            signals["mb60"] | signals["tc15"] | signals["tc30"] | signals["atr30"] | signals["vst60"]
        ) & volume_high,
        "baseline_or_tc15_strong60": (signals["mb60"] | signals["tc15"]) & strong_60,
    }

    exit_spec = gs.ExitSpec(hold_bars=12, stop_pct=0.015, target_pct=0.045)

    rows: list[dict] = []
    for construction, entry_mask in constructions.items():
        trades = gs.simulate(master, entry_mask.fillna(False).astype(bool), exit_spec, side_cost_pct)
        rows.append(
            gs.summarize(
                construction=construction,
                exit_profile="baseline",
                trades=trades,
                recent_cutoff=recent_cutoff,
            ).__dict__
        )

    results = pd.DataFrame(rows).sort_values(
        ["net_full", "net_recent", "trades_full"],
        ascending=[False, False, False],
    )

    print(f"\n{'=' * 112}")
    print(f" BENCHMARK REFINEMENT SCREEN | pair={args.pair} | rows={len(results)}")
    print(f"{'=' * 112}")
    cols = [
        "construction",
        "trades_full",
        "net_full",
        "win_full",
        "trades_recent",
        "net_recent",
        "win_recent",
    ]
    formatters = {
        "net_full": lambda value: f"{value:+.3%}",
        "win_full": lambda value: f"{value:.1%}",
        "net_recent": lambda value: f"{value:+.3%}",
        "win_recent": lambda value: f"{value:.1%}",
    }
    print(results[cols].to_string(index=False, formatters=formatters))

    if args.results_csv:
        results.to_csv(args.results_csv, index=False)
        print(f"\nWrote {len(results)} rows to {args.results_csv}")


if __name__ == "__main__":
    main()
