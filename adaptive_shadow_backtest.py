from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

import adaptive_strategy_screener as adaptive
import backtest
import expanded_screener as ex
import lower_bar_screener as lbs
import shadow_backtest
import strategy as strat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare adaptive lower-bar challengers against the current 15m baseline under the shared shadow path"
    )
    parser.add_argument("--pairs", default="GIGAUSD,DOGUSD,COQUSD,HYPEUSD")
    parser.add_argument("--history-days", type=int, default=60)
    parser.add_argument("--recent-days", type=int, default=30)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--trade-count", type=int, default=5000)
    parser.add_argument("--trade-pause-sec", type=float, default=0.8)
    parser.add_argument(
        "--cost-scenarios",
        default="taker_like=0.0031,maker_mid=0.0010,maker_optimistic=0.0003",
    )
    parser.add_argument("--summary-csv", default="adaptive_shadow_backtest_summary.csv")
    parser.add_argument("--trades-csv", default="adaptive_shadow_backtest_trades.csv")
    return parser.parse_args()


def _load_baseline_history(pair: str, history_days: int, cache_dir: Path) -> pd.DataFrame:
    cache_120 = cache_dir / f"{pair}_15m_120d_end_latest.csv"
    if cache_120.exists():
        df = pd.read_csv(cache_120).sort_values("ts").reset_index(drop=True)
        cutoff = int(df["ts"].max()) - history_days * 24 * 60 * 60
        return df.loc[df["ts"] >= cutoff].reset_index(drop=True)
    return backtest.fetch_history(
        pair=pair,
        interval=15,
        history_days=history_days,
        trade_pause_sec=0.0,
    )


def _score_frame(frame: pd.DataFrame) -> pd.Series:
    return (
        120.0 * frame["trend_strength"].clip(lower=0.0)
        + 60.0 * frame["momentum_medium"].clip(lower=0.0)
        + 20.0 * frame["momentum_short"].clip(lower=0.0)
        + 12.0 * (frame["volume_ratio"] - 1.0).clip(lower=0.0)
        + 10.0 * (frame["close_location"] - 0.55).clip(lower=0.0)
        + 8.0 * frame.get("market_breadth", pd.Series(0.0, index=frame.index)).clip(lower=0.0)
        + 4.0 * frame.get("strong_breadth", pd.Series(0.0, index=frame.index)).clip(lower=0.0)
        + 4.0 * frame.get("rs_gap", pd.Series(0.0, index=frame.index)).clip(lower=0.0)
    )


def _prepare_candidate_frame(frame: pd.DataFrame, candidate_name: str, interval: int) -> pd.DataFrame:
    candidates = {name: mask for name, mask, _ in adaptive.build_adaptive_candidates(frame, interval)}
    if candidate_name not in candidates:
        available = ", ".join(sorted(candidates))
        raise SystemExit(f"Unknown adaptive candidate '{candidate_name}'. Available: {available}")

    out = frame.copy().reset_index(drop=True)
    out["entry_signal"] = candidates[candidate_name].fillna(False).to_numpy(dtype=bool)
    out["signal_score"] = _score_frame(out).to_numpy(dtype=float)
    out["gate_trend_strength_60"] = out["ctx60_trend_strength"].fillna(0.0).to_numpy(dtype=float)
    out["gate_momentum_medium_60"] = out["ctx60_momentum_medium"].fillna(0.0).to_numpy(dtype=float)
    return out


def _summarize(
    label: str,
    pair: str,
    interval: int,
    cost_model: str,
    trades: list[backtest.BacktestTrade],
    recent_cutoff: int,
) -> dict:
    summary = shadow_backtest._summarize_trades(trades, recent_cutoff)
    return {
        "label": label,
        "pair": pair,
        "interval": interval,
        "cost_model": cost_model,
        **summary,
    }


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    pairs = [pair.strip() for pair in args.pairs.split(",") if pair.strip()]
    cost_scenarios = lbs.parse_cost_scenarios(args.cost_scenarios)
    config = strat.DEFAULT_CONFIG

    interval_frames = adaptive._build_frames(
        pairs=pairs,
        intervals=[3, 5],
        history_days=args.history_days,
        cache_dir=cache_dir,
        trade_count=args.trade_count,
        trade_pause_sec=args.trade_pause_sec,
    )
    baseline_history = _load_baseline_history("GIGAUSD", args.history_days, cache_dir)
    max_ts = max(
        int(baseline_history["ts"].max()),
        max(int(df["ts"].max()) for frames in interval_frames.values() for df in frames.values()),
    )
    recent_cutoff = max_ts - args.recent_days * 24 * 60 * 60

    candidate_specs = [
        ("baseline_tc15_tighter_volume_cap", "GIGAUSD", 15, "baseline"),
        ("giga_5m_freqai_mtf_adaptive_union", "GIGAUSD", 5, "freqai_mtf_adaptive_union"),
        (
            "giga_5m_freqai_mtf_adaptive_union_strongtrend",
            "GIGAUSD",
            5,
            "freqai_mtf_adaptive_union_strongtrend",
        ),
        (
            "giga_5m_freqai_mtf_adaptive_union_strongtrend_vwapcap",
            "GIGAUSD",
            5,
            "freqai_mtf_adaptive_union_strongtrend_vwapcap",
        ),
        ("coq_3m_freqai_mtf_momentum_accel", "COQUSD", 3, "freqai_mtf_momentum_accel"),
    ]

    summary_rows: list[dict] = []
    trade_rows: list[dict] = []
    for cost_label, side_cost_pct in cost_scenarios:
        print(f"\n{'=' * 110}")
        print(f" ADAPTIVE SHADOW BACKTEST | cost_model={cost_label} | side_cost={side_cost_pct:.4%}")
        print(f"{'=' * 110}")
        for label, pair, interval, construction in candidate_specs:
            if construction == "baseline":
                trades = backtest.run_backtest(
                    pair=pair,
                    df_raw=baseline_history,
                    config=config,
                    commission_pct=side_cost_pct,
                    slippage_pct=0.0,
                    construction="tc15_tighter_volume_cap",
                )
            else:
                source_frame = interval_frames[interval][pair]
                candidate_frame = _prepare_candidate_frame(
                    frame=source_frame,
                    candidate_name=construction,
                    interval=interval,
                )
                trades = backtest.run_backtest_frame(
                    pair=pair,
                    df=candidate_frame,
                    config=config,
                    commission_pct=side_cost_pct,
                    slippage_pct=0.0,
                    construction=construction,
                )

            print(f"{label}: {len(trades)} trades")
            summary_rows.append(
                _summarize(
                    label=label,
                    pair=pair,
                    interval=interval,
                    cost_model=cost_label,
                    trades=trades,
                    recent_cutoff=recent_cutoff,
                )
            )
            for trade in trades:
                row = asdict(trade)
                row["label"] = label
                row["cost_model"] = cost_label
                trade_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["cost_model", "net_full", "net_recent", "trades_full"],
        ascending=[True, False, False, False],
    )
    print(f"\n{'=' * 110}")
    print(" ADAPTIVE SHADOW SUMMARY")
    print(f"{'=' * 110}")
    cols = [
        "cost_model",
        "label",
        "pair",
        "interval",
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

    summary.to_csv(args.summary_csv, index=False)
    print(f"\nWrote summary to {args.summary_csv}")
    if trade_rows:
        pd.DataFrame(trade_rows).to_csv(args.trades_csv, index=False)
        print(f"Wrote trades to {args.trades_csv}")


if __name__ == "__main__":
    main()
