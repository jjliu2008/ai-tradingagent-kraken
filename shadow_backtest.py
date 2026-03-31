from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import backtest
import strategy as strat


def _recent_cutoff(max_ts: int, recent_days: int) -> int:
    return max_ts - recent_days * 24 * 60 * 60


def _summarize_trades(trades: list[backtest.BacktestTrade], recent_cutoff: int) -> dict:
    pnls = np.asarray([trade.pnl_pct for trade in trades], dtype=float)
    recent_pnls = np.asarray(
        [trade.pnl_pct for trade in trades if trade.entry_ts >= recent_cutoff],
        dtype=float,
    )
    mfe = [trade.mfe_pct for trade in trades]
    holds = [trade.bars_held for trade in trades]
    return {
        "trades_full": len(trades),
        "net_full": float(pnls.sum()) if len(pnls) else 0.0,
        "win_full": float((pnls > 0).mean()) if len(pnls) else 0.0,
        "avg_full": float(pnls.mean()) if len(pnls) else 0.0,
        "max_dd": backtest._max_drawdown([float(x) for x in pnls]) if len(pnls) else 0.0,
        "avg_mfe": float(np.mean(mfe)) if mfe else 0.0,
        "avg_hold": float(np.mean(holds)) if holds else 0.0,
        "trades_recent": len(recent_pnls),
        "net_recent": float(recent_pnls.sum()) if len(recent_pnls) else 0.0,
        "win_recent": float((recent_pnls > 0).mean()) if len(recent_pnls) else 0.0,
        "avg_recent": float(recent_pnls.mean()) if len(recent_pnls) else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shadow backtest ensemble constructions under the same live-faithful execution model"
    )
    parser.add_argument("--pairs", default="GIGAUSD")
    parser.add_argument(
        "--constructions",
        default="baseline_mb60,tc15_tighter_volume_cap",
        help="Comma-separated ensemble construction names",
    )
    parser.add_argument("--interval", type=int, default=strat.MASTER_INTERVAL_MINUTES)
    parser.add_argument("--cache-csv", help="Optional cached history CSV for a single pair")
    parser.add_argument("--history-days", type=int)
    parser.add_argument("--end-ts", type=int)
    parser.add_argument("--trade-count", type=int, default=5000)
    parser.add_argument("--trade-pause-sec", type=float, default=0.8)
    parser.add_argument("--commission-pct", type=float, default=0.0026)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--recent-days", type=int, default=60)
    parser.add_argument("--summary-csv", default="shadow_backtest_summary.csv")
    parser.add_argument("--trades-csv")
    return parser.parse_args()


def _load_history(args: argparse.Namespace, pair: str) -> pd.DataFrame:
    if args.cache_csv:
        return pd.read_csv(args.cache_csv)
    return backtest.fetch_history(
        pair=pair,
        interval=args.interval,
        history_days=args.history_days,
        end_ts=args.end_ts,
        trade_count=args.trade_count,
        trade_pause_sec=args.trade_pause_sec,
    )


def main() -> None:
    args = parse_args()
    config = strat.DEFAULT_CONFIG
    pairs = [pair.strip() for pair in args.pairs.split(",") if pair.strip()]
    constructions = [name.strip() for name in args.constructions.split(",") if name.strip()]
    invalid = [name for name in constructions if name not in strat.ensemble_construction_names()]
    if invalid:
        supported = ", ".join(strat.ensemble_construction_names())
        raise SystemExit(f"Unsupported constructions: {', '.join(invalid)}\nSupported: {supported}")
    if args.cache_csv and len(pairs) != 1:
        raise SystemExit("--cache-csv only supports a single pair")

    pair_histories: dict[str, pd.DataFrame] = {pair: _load_history(args, pair) for pair in pairs}
    max_ts = max(int(df["ts"].max()) for df in pair_histories.values())
    recent_cutoff = _recent_cutoff(max_ts, args.recent_days)

    summary_rows: list[dict] = []
    trade_rows: list[dict] = []

    for construction in constructions:
        construction_trades: list[backtest.BacktestTrade] = []
        for pair, df in pair_histories.items():
            trades = backtest.run_backtest(
                pair=pair,
                df_raw=df,
                config=config,
                commission_pct=args.commission_pct,
                slippage_pct=args.slippage_pct,
                construction=construction,
            )
            print(f"{pair} [{construction}] -> {len(trades)} trades")
            construction_trades.extend(trades)
            trade_rows.extend(asdict(trade) for trade in trades)

        summary_rows.append(
            {
                "construction": construction,
                **_summarize_trades(construction_trades, recent_cutoff),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["net_full", "net_recent", "trades_full"],
        ascending=[False, False, False],
    )
    print(f"\n{'=' * 116}")
    print(" SHADOW BACKTEST SUMMARY")
    print(f"{'=' * 116}")
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
    print(summary[cols].to_string(index=False, formatters=formatters))

    if args.summary_csv:
        summary.to_csv(args.summary_csv, index=False)
        print(f"\nWrote summary to {args.summary_csv}")
    if args.trades_csv and trade_rows:
        pd.DataFrame(trade_rows).to_csv(args.trades_csv, index=False)
        print(f"Wrote trades to {args.trades_csv}")


if __name__ == "__main__":
    main()
