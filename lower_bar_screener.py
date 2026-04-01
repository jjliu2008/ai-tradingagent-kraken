from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import backtest
import expanded_screener as ex
import strategy as strat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen 1m/3m/5m strategies from trade-aggregated 1m history"
    )
    parser.add_argument("--pairs", default="GIGAUSD,DOGUSD,COQUSD,HYPEUSD")
    parser.add_argument("--intervals", default="1,3,5")
    parser.add_argument("--history-days", type=int, default=60)
    parser.add_argument("--recent-days", type=int, default=30)
    parser.add_argument("--min-keep-trades", type=int, default=8)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--trade-count", type=int, default=5000)
    parser.add_argument("--trade-pause-sec", type=float, default=0.8)
    parser.add_argument(
        "--cost-scenarios",
        default="taker_like=0.0031,maker_mid=0.0010,maker_optimistic=0.0003",
        help="Comma-separated name=value side-cost assumptions",
    )
    parser.add_argument("--results-csv", default="lower_bar_screen_results.csv")
    return parser.parse_args()


def parse_cost_scenarios(text: str) -> list[tuple[str, float]]:
    scenarios: list[tuple[str, float]] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"Invalid cost scenario '{item}'. Use name=value format.")
        name, value = item.split("=", 1)
        scenarios.append((name.strip(), float(value.strip())))
    if not scenarios:
        raise SystemExit("At least one cost scenario is required.")
    return scenarios


def cache_path(cache_dir: Path, pair: str, history_days: int) -> Path:
    return cache_dir / f"{pair}_1m_{history_days}d_end_latest.csv"


def load_or_fetch_history(
    pair: str,
    history_days: int,
    cache_dir: Path,
    trade_count: int,
    trade_pause_sec: float,
) -> pd.DataFrame:
    path = cache_path(cache_dir, pair, history_days)
    if path.exists():
        print(f"Loading cached 1m history: {path}")
        return pd.read_csv(path)

    df = backtest.fetch_history(
        pair=pair,
        interval=1,
        history_days=history_days,
        trade_count=trade_count,
        trade_pause_sec=trade_pause_sec,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")
    return df


def densify_1m(df: pd.DataFrame) -> pd.DataFrame:
    raw = df.sort_values("ts").reset_index(drop=True)
    full_ts = pd.DataFrame(
        {
            "ts": np.arange(
                int(raw["ts"].iloc[0]),
                int(raw["ts"].iloc[-1]) + 60,
                60,
                dtype=np.int64,
            )
        }
    )
    out = full_ts.merge(raw, on="ts", how="left")
    price_cols = ["open", "high", "low", "close", "vwap_k"]
    out[price_cols] = out[price_cols].ffill()
    out["volume"] = out["volume"].fillna(0.0)
    out["count"] = out["count"].fillna(0).astype(int)
    out = out.dropna(subset=["close"]).reset_index(drop=True)
    for col in price_cols:
        out[col] = out[col].astype(float)
    return out


def evaluate_pair(
    pair: str,
    base_history_1m: pd.DataFrame,
    intervals: list[int],
    side_cost_pct: float,
    recent_days: int,
    min_keep_trades: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    end_ts = int(base_history_1m["ts"].iloc[-1])
    recent_cutoff = end_ts - recent_days * 24 * 60 * 60

    for interval in intervals:
        interval_df = ex.resample_ohlcv(base_history_1m, interval)
        interval_df = ex.add_expanded_features(strat.compute_features(interval_df))
        full_df = interval_df.reset_index(drop=True)
        recent_df = interval_df.loc[interval_df["ts"] >= recent_cutoff].reset_index(drop=True)
        full_candidates = ex.build_all_candidates(full_df, interval)
        recent_candidates = {
            name: mask for name, mask, _ in ex.build_all_candidates(recent_df, interval)
        }

        for strategy_name, full_mask, exit_spec in full_candidates:
            recent_mask = recent_candidates.get(strategy_name)
            if recent_mask is None:
                continue

            full_result = ex.evaluate_window(
                pair=pair,
                interval=interval,
                strategy_name=strategy_name,
                df=full_df,
                entry_mask=full_mask.fillna(False),
                exit_spec=exit_spec,
                side_cost_pct=side_cost_pct,
                window_days=int(round((full_df["ts"].iloc[-1] - full_df["ts"].iloc[0]) / 86400)),
            )
            recent_result = ex.evaluate_window(
                pair=pair,
                interval=interval,
                strategy_name=strategy_name,
                df=recent_df,
                entry_mask=recent_mask.fillna(False),
                exit_spec=exit_spec,
                side_cost_pct=side_cost_pct,
                window_days=recent_days,
            )

            keep = (
                full_result.net_pnl > 0
                and recent_result.net_pnl > 0
                and full_result.trades >= min_keep_trades
                and recent_result.trades >= max(4, min_keep_trades // 2)
            )
            score = full_result.net_pnl + 1.5 * recent_result.net_pnl + 0.01 * full_result.sharpe
            rows.append(
                {
                    "pair": pair,
                    "interval": interval,
                    "strategy": strategy_name,
                    "trades_full": full_result.trades,
                    "net_full": full_result.net_pnl,
                    "win_full": full_result.win_rate,
                    "dd_full": full_result.max_drawdown,
                    "avg_mfe_full": full_result.avg_mfe,
                    "trades_recent": recent_result.trades,
                    "net_recent": recent_result.net_pnl,
                    "win_recent": recent_result.win_rate,
                    "score": score,
                    "verdict": "KEEP" if keep else "KILL",
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    pairs = [pair.strip() for pair in args.pairs.split(",") if pair.strip()]
    intervals = [int(value.strip()) for value in args.intervals.split(",") if value.strip()]
    cost_scenarios = parse_cost_scenarios(args.cost_scenarios)
    cache_dir = Path(args.cache_dir)

    dense_histories: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        history = load_or_fetch_history(
            pair=pair,
            history_days=args.history_days,
            cache_dir=cache_dir,
            trade_count=args.trade_count,
            trade_pause_sec=args.trade_pause_sec,
        )
        dense = densify_1m(history)
        dense_histories[pair] = dense
        print(f"{pair}: sparse={len(history)} dense={len(dense)}")

    frames: list[pd.DataFrame] = []
    for label, side_cost_pct in cost_scenarios:
        print(f"\n{'=' * 100}")
        print(f" LOWER BAR SCREEN | cost_model={label} | side_cost={side_cost_pct:.4%}")
        print(f"{'=' * 100}")
        scenario_frames = [
            evaluate_pair(
                pair=pair,
                base_history_1m=dense_histories[pair],
                intervals=intervals,
                side_cost_pct=side_cost_pct,
                recent_days=args.recent_days,
                min_keep_trades=args.min_keep_trades,
            )
            for pair in pairs
        ]
        scenario_df = pd.concat([frame for frame in scenario_frames if not frame.empty], ignore_index=True)
        scenario_df["cost_model"] = label
        scenario_df["side_cost_pct"] = side_cost_pct
        scenario_df = scenario_df.sort_values(
            ["verdict", "score", "net_recent", "net_full"],
            ascending=[True, False, False, False],
        )
        frames.append(scenario_df)

        cols = [
            "pair",
            "interval",
            "strategy",
            "trades_full",
            "net_full",
            "trades_recent",
            "net_recent",
            "score",
            "verdict",
        ]
        print(scenario_df[cols].head(15).to_string(index=False, float_format=lambda v: f"{v:+.3%}"))

    if not frames:
        raise SystemExit("No results produced.")

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(args.results_csv, index=False)
    print(f"\nWrote {len(results)} rows to {args.results_csv}")


if __name__ == "__main__":
    main()
