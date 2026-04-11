from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import backtest as bt
import research_runtime as rr
from research_pair_registry import PAIR_RESEARCH_REGISTRY, active_pairs


RESULTS_DIR = Path("results")
UNIFORM_CACHE_DIR = Path("data_cache_walkforward")


def _summarize(trades: list[bt.BacktestTrade]) -> dict[str, Any]:
    if not trades:
        return {
            "trades": 0,
            "net_pct": 0.0,
            "win_rate": 0.0,
            "avg_trade_pct": 0.0,
            "max_dd_pct": 0.0,
        }
    pnls = pd.Series([trade.pnl_pct for trade in trades], dtype=float)
    equity = (1.0 + pnls).cumprod()
    max_dd = float((equity / equity.cummax() - 1.0).min())
    return {
        "trades": int(len(trades)),
        "net_pct": float(pnls.sum()),
        "win_rate": float((pnls > 0).mean()),
        "avg_trade_pct": float(pnls.mean()),
        "max_dd_pct": max_dd,
    }


def _summarize_with_split(
    trades: list,
    split_ts: int,
) -> dict:
    """Summarize trades broken into older60 (before split_ts) and recent60 (at or after)."""
    older = [t for t in trades if t.entry_ts < split_ts]
    recent = [t for t in trades if t.entry_ts >= split_ts]

    def _stats(subset: list) -> dict:
        if not subset:
            return {"trades": 0, "net_pct": 0.0, "win_rate": 0.0, "avg_trade_pct": 0.0, "max_dd_pct": 0.0}
        pnls = pd.Series([t.pnl_pct for t in subset], dtype=float)
        equity = (1.0 + pnls).cumprod()
        max_dd = float((equity / equity.cummax() - 1.0).min())
        return {
            "trades": int(len(subset)),
            "net_pct": float(pnls.sum()),
            "win_rate": float((pnls > 0).mean()),
            "avg_trade_pct": float(pnls.mean()),
            "max_dd_pct": max_dd,
        }

    older_stats = _stats(older)
    recent_stats = _stats(recent)
    return {
        **{f"older60_{k}": v for k, v in older_stats.items()},
        **{f"recent60_{k}": v for k, v in recent_stats.items()},
    }


def _write_cache(pair: str, interval: int, history_days: int, df: pd.DataFrame) -> Path:
    UNIFORM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = UNIFORM_CACHE_DIR / f"{pair}_{interval}m_{history_days}d_uniform_live.csv"
    df.to_csv(path, index=False)
    return path


def _load_uniform_history(
    pair: str,
    interval: int,
    history_days: int,
    refresh_live: bool,
    trade_count: int,
    trade_pause_sec: float,
) -> tuple[pd.DataFrame, Path]:
    cache_path = UNIFORM_CACHE_DIR / f"{pair}_{interval}m_{history_days}d_uniform_live.csv"
    if not refresh_live and cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        df = bt.fetch_history(
            pair=pair,
            interval=interval,
            history_days=history_days,
            trade_count=trade_count,
            trade_pause_sec=trade_pause_sec,
        )
        cache_path = _write_cache(pair, interval, history_days, df)

    df.columns = [column.lower() for column in df.columns]
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    if "vwap_k" not in df.columns:
        df["vwap_k"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    if "count" not in df.columns:
        df["count"] = 1
    return df.sort_values("ts").reset_index(drop=True), cache_path


def run_portfolio_backtest(
    pairs: list[str],
    interval: int,
    history_days: int,
    refresh_live: bool,
    trade_count: int,
    trade_pause_sec: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pair_rows: list[dict[str, Any]] = []
    all_trades: list[bt.BacktestTrade] = []
    recent_30_trades: list[bt.BacktestTrade] = []
    recent_14_trades: list[bt.BacktestTrade] = []
    max_ts = 0

    for pair in pairs:
        plan = PAIR_RESEARCH_REGISTRY[pair]
        df_raw, cache_path = _load_uniform_history(
            pair=pair,
            interval=interval,
            history_days=history_days,
            refresh_live=refresh_live,
            trade_count=trade_count,
            trade_pause_sec=trade_pause_sec,
        )
        if df_raw.empty:
            continue
        max_ts = max(max_ts, int(df_raw["ts"].iloc[-1]))
        frame = rr.build_research_frame(df_raw, plan)
        trades = bt.run_backtest_frame(
            pair=pair,
            df=frame,
            config=rr.config_for_plan(plan),
            commission_pct=0.0026,
            slippage_pct=0.0005,
            construction=plan.construction,
        )
        all_trades.extend(trades)
        pair_rows.append(
            {
                "pair": pair,
                "status": plan.status,
                "construction": plan.construction,
                "entry_filter": plan.entry_filter,
                "exit_profile": plan.exit_profile,
                "source_file": str(cache_path).replace("\\", "/"),
                "bars": int(len(df_raw)),
                **{f"full_{k}": v for k, v in _summarize(trades).items()},
            }
        )

    recent_30_start = max_ts - 30 * 24 * 60 * 60 if max_ts else 0
    recent_14_start = max_ts - 14 * 24 * 60 * 60 if max_ts else 0
    split_ts        = max_ts - 60 * 24 * 60 * 60 if max_ts else 0
    older60_trades:  list = []
    recent60_trades: list = []

    for trade in all_trades:
        if trade.entry_ts < split_ts:
            older60_trades.append(trade)
        else:
            recent60_trades.append(trade)
        if trade.entry_ts >= recent_30_start:
            recent_30_trades.append(trade)
        if trade.entry_ts >= recent_14_start:
            recent_14_trades.append(trade)

    pair_frame = pd.DataFrame(pair_rows)
    if not pair_frame.empty:
        for idx, row in pair_frame.iterrows():
            pair = str(row["pair"])
            pair_trades = [trade for trade in all_trades if trade.pair == pair]
            pair_recent_30 = [trade for trade in pair_trades if trade.entry_ts >= recent_30_start]
            pair_recent_14 = [trade for trade in pair_trades if trade.entry_ts >= recent_14_start]
            pair_older60 = [trade for trade in pair_trades if trade.entry_ts < split_ts]
            pair_recent60 = [trade for trade in pair_trades if trade.entry_ts >= split_ts]
            for prefix, summary in (
                ("recent30", _summarize(pair_recent_30)),
                ("recent14", _summarize(pair_recent_14)),
                ("older60", _summarize(pair_older60)),
                ("recent60", _summarize(pair_recent60)),
            ):
                for key, value in summary.items():
                    pair_frame.loc[idx, f"{prefix}_{key}"] = value
        pair_frame = pair_frame.sort_values("full_net_pct", ascending=False).reset_index(drop=True)

    summary = {
        "history_days": history_days,
        "interval_minutes": interval,
        "active_pairs": int(len(pair_frame)),
        "window_end_ts": int(max_ts),
        "portfolio_regime_gate": rr.PORTFOLIO_REGIME_GATE,
        "full": _summarize(all_trades),
        "recent30": _summarize(recent_30_trades),
        "recent14": _summarize(recent_14_trades),
        "split_ts": int(split_ts),
        "older60": _summarize(older60_trades),
        "recent60": _summarize(recent60_trades),
    }
    return pair_frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Uniform-window portfolio backtest for the active research registry.")
    parser.add_argument("--pairs", default=",".join(active_pairs()))
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--history-days", type=int, default=60)
    parser.add_argument("--refresh-live", action="store_true")
    parser.add_argument("--trade-count", type=int, default=5000)
    parser.add_argument("--trade-pause-sec", type=float, default=0.05)
    parser.add_argument("--summary-json", default="results/general_portfolio_backtest_summary_uniform.json")
    parser.add_argument("--pairs-csv", default="results/general_portfolio_backtest_pairs_uniform.csv")
    args = parser.parse_args()

    pairs = [item.strip().upper() for item in args.pairs.split(",") if item.strip()]
    pair_frame, summary = run_portfolio_backtest(
        pairs=pairs,
        interval=args.interval,
        history_days=args.history_days,
        refresh_live=args.refresh_live,
        trade_count=args.trade_count,
        trade_pause_sec=args.trade_pause_sec,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pair_frame.to_csv(args.pairs_csv, index=False)
    with open(args.summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("UNIFORM PORTFOLIO BACKTEST")
    print(f"pairs={summary['active_pairs']} interval={summary['interval_minutes']}m history_days={summary['history_days']}")
    for label in ("full", "recent30", "recent14"):
        stats = summary[label]
        print(
            f"{label}: trades={stats['trades']} "
            f"net={stats['net_pct']:+.6%} "
            f"win={stats['win_rate']:.2%} "
            f"avg={stats['avg_trade_pct']:+.6%} "
            f"max_dd={stats['max_dd_pct']:+.6%}"
        )


if __name__ == "__main__":
    main()
