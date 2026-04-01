from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import backtest
import strategy as strat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Focused shadow screener for tc15-only refinements"
    )
    parser.add_argument("--pair", default="GIGAUSD")
    parser.add_argument("--cache-csv", default="data_cache/GIGAUSD_15m_120d_end_latest.csv")
    parser.add_argument("--commission-pct", type=float, default=0.0026)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--recent-days", type=int, default=60)
    parser.add_argument("--results-csv", default="tc15_filter_results.csv")
    return parser.parse_args()


def _summarize(trades: list[backtest.BacktestTrade], recent_cutoff: int) -> dict:
    pnls = pd.Series([trade.pnl_pct for trade in trades], dtype=float)
    recent = pd.Series(
        [trade.pnl_pct for trade in trades if trade.entry_ts >= recent_cutoff],
        dtype=float,
    )
    return {
        "trades_full": int(len(pnls)),
        "net_full": float(pnls.sum()) if len(pnls) else 0.0,
        "win_full": float((pnls > 0).mean()) if len(pnls) else 0.0,
        "max_dd": backtest._max_drawdown([float(x) for x in pnls]) if len(pnls) else 0.0,
        "trades_recent": int(len(recent)),
        "net_recent": float(recent.sum()) if len(recent) else 0.0,
        "win_recent": float((recent > 0).mean()) if len(recent) else 0.0,
    }


def main() -> None:
    args = parse_args()
    history = pd.read_csv(Path(args.cache_csv))
    config = strat.DEFAULT_CONFIG
    frame = strat.build_ensemble_frame(history, construction="tc15_only", config=config).reset_index(drop=True)
    if frame.empty:
        raise SystemExit("No ensemble frame could be built from the supplied history.")

    recent_cutoff = int(frame["ts"].max()) - args.recent_days * 24 * 60 * 60
    base = frame["signal_tc15"].fillna(False)
    positive_60m_mom = (frame["gate_momentum_medium_60"] > 0.0).fillna(False)
    stronger_60m_mom = (frame["gate_momentum_medium_60"] > 0.02).fillna(False)
    not_extreme_close = (frame["close_location"] < 0.98).fillna(False)
    strong_close = (frame["close_location"] >= 0.70).fillna(False)
    capped_volume = (frame["volume_ratio"] <= 6.0).fillna(False)
    capped_volume_tight = (frame["volume_ratio"] <= 4.5).fillna(False)
    compressed = (frame["compression_ratio"] <= 0.85).fillna(False)
    not_loose = (frame["compression_ratio"] <= 1.0).fillna(False)
    not_too_extended = (frame["distance_from_vwap"] <= 0.03).fillna(False)
    not_far_from_vwap = (frame["distance_from_vwap"] <= 0.02).fillna(False)
    mb60 = frame["signal_mb60"].fillna(False)
    atr30 = frame["signal_atr30"].fillna(False)
    tc30 = frame["signal_tc30"].fillna(False)
    vst60 = frame["signal_vst60"].fillna(False)
    tc15_cap = base & capped_volume_tight

    candidates = {
        "tc15_only": base,
        "tc15_no_extreme_volume": base & capped_volume,
        "tc15_tighter_volume_cap": base & capped_volume_tight,
        "tc15_no_exhaust_close": base & not_extreme_close,
        "tc15_no_exhaust_vol": base & not_extreme_close & capped_volume,
        "tc15_60m_mom_up": base & positive_60m_mom,
        "tc15_60m_mom_strong": base & stronger_60m_mom,
        "tc15_not_extended": base & not_too_extended,
        "tc15_not_far_from_vwap": base & not_far_from_vwap,
        "tc15_compressed": base & compressed,
        "tc15_not_loose": base & not_loose,
        "tc15_quality_stack": base & strong_close & capped_volume & compressed & not_too_extended,
        "tc15_quality_stack_loose": base & strong_close & capped_volume & not_loose & not_too_extended,
        "tc15_mom_and_no_exhaust": base & stronger_60m_mom & capped_volume & not_extreme_close,
        "tc15_mom_no_exhaust_vwap": base & stronger_60m_mom & capped_volume & not_extreme_close & not_too_extended,
        "tc15_cap_or_mb60": tc15_cap | mb60,
        "tc15_cap_or_atr30": tc15_cap | atr30,
        "tc15_cap_or_mb60_or_atr30": tc15_cap | mb60 | atr30,
        "tc15_cap_or_tc30": tc15_cap | tc30,
        "tc15_cap_or_vst60": tc15_cap | vst60,
        "tc15_cap_pos60": tc15_cap & positive_60m_mom,
        "tc15_cap_strong60": tc15_cap & stronger_60m_mom,
        "tc15_cap_no_exhaust_close": tc15_cap & not_extreme_close,
        "tc15_cap_no_far_vwap": tc15_cap & not_too_extended,
        "tc15_cap_plus_mb60_strong60": (tc15_cap | mb60) & positive_60m_mom,
        "tc15_cap_plus_atr30_strong60": (tc15_cap | atr30) & positive_60m_mom,
    }

    rows: list[dict] = []
    for name, mask in candidates.items():
        variant_frame = frame.copy()
        variant_frame["entry_signal"] = mask.to_numpy(dtype=bool)
        trades = backtest.run_backtest_frame(
            pair=args.pair,
            df=variant_frame,
            config=config,
            commission_pct=args.commission_pct,
            slippage_pct=args.slippage_pct,
            construction=name,
        )
        rows.append({"construction": name, **_summarize(trades, recent_cutoff)})

    results = pd.DataFrame(rows).sort_values(
        ["net_full", "net_recent", "trades_full"],
        ascending=[False, False, False],
    )

    print(f"\n{'=' * 118}")
    print(f" TC15 FILTER SCREEN | pair={args.pair} | rows={len(results)}")
    print(f"{'=' * 118}")
    cols = [
        "construction",
        "trades_full",
        "net_full",
        "win_full",
        "max_dd",
        "trades_recent",
        "net_recent",
        "win_recent",
    ]
    formatters = {
        "net_full": lambda value: f"{value:+.3%}",
        "win_full": lambda value: f"{value:.1%}",
        "max_dd": lambda value: f"{value:.3%}",
        "net_recent": lambda value: f"{value:+.3%}",
        "win_recent": lambda value: f"{value:.1%}",
    }
    print(results[cols].to_string(index=False, formatters=formatters))

    if args.results_csv:
        results.to_csv(args.results_csv, index=False)
        print(f"\nWrote {len(results)} rows to {args.results_csv}")


if __name__ == "__main__":
    main()
