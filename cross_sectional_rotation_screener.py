from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import backtest
import expanded_screener as ex
import shadow_backtest
import strategy as strat


@dataclass(frozen=True)
class ExitSpec:
    hold_bars: int
    stop_pct: float
    target_pct: float


@dataclass(frozen=True)
class StrategySpec:
    name: str
    exit_spec: ExitSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-sectional 15m/30m/60m rotation screener over a liquid Kraken spot universe"
    )
    parser.add_argument("--pairs", default="GIGAUSD,SOLUSD,XRPUSD,TAOUSD,HYPEUSD,XDGUSD,ADAUSD,SUIUSD,TRXUSD")
    parser.add_argument("--history-days", type=int, default=60)
    parser.add_argument("--recent-days", type=int, default=30)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--trade-count", type=int, default=5000)
    parser.add_argument("--trade-pause-sec", type=float, default=0.0)
    parser.add_argument("--commission-pct", type=float, default=0.0026)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--summary-csv", default="cross_sectional_rotation_summary.csv")
    parser.add_argument("--trades-csv", default="cross_sectional_rotation_trades.csv")
    return parser.parse_args()


def _cache_path(cache_dir: Path, pair: str, history_days: int) -> Path:
    return cache_dir / f"{pair}_15m_{history_days}d_end_latest.csv"


def _load_history(pair: str, history_days: int, cache_dir: Path, trade_count: int, trade_pause_sec: float) -> pd.DataFrame:
    path = _cache_path(cache_dir, pair, history_days)
    if path.exists():
        return pd.read_csv(path)
    df = backtest.fetch_history(
        pair=pair,
        interval=15,
        history_days=history_days,
        trade_count=trade_count,
        trade_pause_sec=trade_pause_sec,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _align_context(base: pd.DataFrame, higher: pd.DataFrame, higher_interval: int, prefix: str) -> pd.DataFrame:
    cols = [
        "ema_fast",
        "ema_slow",
        "trend_strength",
        "momentum_short",
        "momentum_medium",
        "volume_ratio",
        "compression_ratio",
        "close_location",
        "distance_from_vwap",
        "roll_high_12",
        "roll_low_12",
        "body_pct",
        "green_bar",
    ]
    ctx = higher[["ts"] + cols].copy()
    ctx["effective_ts"] = ctx["ts"].astype(int) + (higher_interval - 15) * 60
    aligned = (
        ctx.set_index("effective_ts")[cols]
        .reindex(base["ts"].astype(int))
        .ffill()
        .reset_index(drop=True)
        .rename(columns={column: f"{prefix}_{column}" for column in cols})
    )
    return aligned


def _build_pair_frame(raw: pd.DataFrame, pair: str) -> pd.DataFrame:
    base = ex.add_expanded_features(strat.compute_features(raw.copy())).reset_index(drop=True)
    frame30 = ex.add_expanded_features(strat.compute_features(ex.resample_ohlcv(raw, 30))).reset_index(drop=True)
    frame60 = ex.add_expanded_features(strat.compute_features(ex.resample_ohlcv(raw, 60))).reset_index(drop=True)
    out = base.copy()
    out["pair"] = pair
    out = pd.concat(
        [
            out,
            _align_context(out, frame30, 30, "ctx30"),
            _align_context(out, frame60, 60, "ctx60"),
        ],
        axis=1,
    )
    return out


def _attach_cross_context(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for pair, frame in frames.items():
        view = frame[["ts", "pair", "momentum_medium", "trend_strength", "atr_pct", "volume_ratio"]].copy()
        view["rs_15"] = view["momentum_medium"] / view["atr_pct"].clip(lower=1e-6)
        view["trend_15"] = view["trend_strength"]
        view["rs_60"] = frame["ctx60_momentum_medium"] / frame["atr_pct"].clip(lower=1e-6)
        rows.append(view)

    universe = pd.concat(rows, ignore_index=True)
    universe["rs_rank_15"] = universe.groupby("ts")["rs_15"].rank(method="dense", ascending=False)
    universe["trend_rank_15"] = universe.groupby("ts")["trend_15"].rank(method="dense", ascending=False)
    universe["rs_rank_60"] = universe.groupby("ts")["rs_60"].rank(method="dense", ascending=False)
    breadth = universe.groupby("ts").agg(
        breadth_15=("momentum_medium", lambda s: float((s > 0).mean())),
        strong_breadth_15=("trend_strength", lambda s: float((s > 0.0015).mean())),
        mean_rs_15=("rs_15", "mean"),
    )
    universe = universe.join(breadth, on="ts")
    universe["rs_gap_15"] = universe["rs_15"] - universe["mean_rs_15"]

    enriched: dict[str, pd.DataFrame] = {}
    for pair, frame in frames.items():
        ctx = universe.loc[universe["pair"] == pair, ["ts", "rs_rank_15", "trend_rank_15", "rs_rank_60", "breadth_15", "strong_breadth_15", "rs_gap_15"]]
        enriched[pair] = pd.concat([frame.reset_index(drop=True), ctx.drop(columns=["ts"]).reset_index(drop=True)], axis=1)
    return enriched


def _signal_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    trend_up = frame["ema_fast"] > frame["ema_slow"]
    ctx30_up = frame["ctx30_ema_fast"] > frame["ctx30_ema_slow"]
    ctx60_up = frame["ctx60_ema_fast"] > frame["ctx60_ema_slow"]
    leader = (frame["rs_rank_15"] <= 2).fillna(False)
    leader60 = (frame["rs_rank_60"] <= 3).fillna(False)
    breadth_ok = (frame["breadth_15"] >= 0.30).fillna(False)
    strong_breadth = (frame["strong_breadth_15"] >= 0.30).fillna(False)
    score_ok = (frame["rs_gap_15"] > 0).fillna(False)
    rank_improving = ((frame["rs_rank_15"].shift(2) >= 4) & (frame["rs_rank_15"] <= 2)).fillna(False)
    persistent = ((frame["rs_rank_15"].rolling(3, min_periods=2).min() <= 2)).fillna(False)

    leader_breakout = (
        leader
        & leader60
        & breadth_ok
        & trend_up
        & ctx30_up
        & ctx60_up
        & (frame["trend_strength"] > 0.0015)
        & (frame["ctx60_trend_strength"] > 0.0008)
        & (frame["close"] > frame["roll_high_12"])
        & (frame["momentum_medium"] > 0.008)
        & (frame["volume_ratio"] > 0.8)
        & (frame["close_location"] > 0.60)
    ).fillna(False)

    persistence_breakout = (
        persistent
        & leader60
        & breadth_ok
        & trend_up
        & (frame["close"] > frame["roll_high_12"])
        & (frame["body_pct"] > 0.003)
        & (frame["volume_ratio"] > 0.8)
        & (frame["close_location"] > 0.58)
    ).fillna(False)

    rotation_emerge = (
        rank_improving
        & leader60
        & score_ok
        & breadth_ok
        & trend_up
        & ctx30_up
        & (frame["mom_accel"] > 0)
        & (frame["close"] > frame["ema_fast"])
        & (frame["close_location"] > 0.55)
        & (frame["volume_ratio"] > 0.7)
    ).fillna(False)

    leader_pullback = (
        leader
        & leader60
        & strong_breadth
        & trend_up
        & ctx60_up
        & (frame["low"] <= frame["ema_fast"])
        & (frame["close"] > frame["ema_fast"])
        & frame["distance_from_vwap"].between(-0.010, 0.020)
        & (frame["volume_ratio"] >= 0.5)
        & (frame["volume_ratio"] <= 2.5)
        & (frame["close_location"] > 0.55)
    ).fillna(False)

    breadth_thrust = (
        (frame["breadth_15"] >= 0.50)
        & (frame["rs_rank_15"] <= 1)
        & trend_up
        & ctx60_up
        & frame["green_bar"]
        & (frame["body_pct"] > 0.004)
        & (frame["close"] > frame["roll_high_12"])
        & (frame["volume_ratio"] > 1.0)
    ).fillna(False)

    higher_low_leader = (
        leader
        & leader60
        & trend_up
        & ctx30_up
        & (frame["low"].shift(1) > frame["roll_low_12"])
        & (frame["close"] > frame["roll_high_12"])
        & (frame["close_location"] > 0.58)
        & (frame["volume_ratio"] > 0.8)
    ).fillna(False)

    return {
        "leader_breakout_15": leader_breakout,
        "persistence_breakout_15": persistence_breakout,
        "rotation_emerge_15": rotation_emerge,
        "leader_pullback_15": leader_pullback,
        "breadth_thrust_15": breadth_thrust,
        "higher_low_leader_15": higher_low_leader,
    }


def _specs() -> dict[str, ExitSpec]:
    return {
        "leader_breakout_15": ExitSpec(hold_bars=5, stop_pct=0.012, target_pct=0.032),
        "persistence_breakout_15": ExitSpec(hold_bars=6, stop_pct=0.012, target_pct=0.035),
        "rotation_emerge_15": ExitSpec(hold_bars=5, stop_pct=0.011, target_pct=0.030),
        "leader_pullback_15": ExitSpec(hold_bars=5, stop_pct=0.010, target_pct=0.025),
        "breadth_thrust_15": ExitSpec(hold_bars=4, stop_pct=0.012, target_pct=0.028),
        "higher_low_leader_15": ExitSpec(hold_bars=5, stop_pct=0.011, target_pct=0.030),
    }


def _candidate_score(row: pd.Series) -> float:
    return (
        120.0 * max(float(row["trend_strength"]), 0.0)
        + 40.0 * max(float(row["momentum_medium"]), 0.0)
        + 20.0 * max(float(row["rs_gap_15"]), 0.0)
        + 10.0 * max(0.0, 4.0 - float(row["rs_rank_15"]))
        + 8.0 * max(float(row["breadth_15"]) - 0.2, 0.0)
        + 8.0 * max(float(row["close_location"]) - 0.5, 0.0)
        + 6.0 * max(float(row["volume_ratio"]) - 0.8, 0.0)
    )


def _run_portfolio_strategy(
    frames: dict[str, pd.DataFrame],
    masks_by_pair: dict[str, pd.Series],
    exit_spec: ExitSpec,
    commission_pct: float,
    slippage_pct: float,
    strategy_name: str,
) -> list[backtest.BacktestTrade]:
    timestamps = sorted({int(ts) for frame in frames.values() for ts in frame["ts"].astype(int).tolist()})
    idx_by_pair = {
        pair: {int(ts): idx for idx, ts in enumerate(frame["ts"].astype(int).tolist())}
        for pair, frame in frames.items()
    }
    trades: list[backtest.BacktestTrade] = []
    pending: dict | None = None
    position: dict | None = None

    for ts in timestamps:
        if pending is not None and position is None and ts >= pending["entry_ts"]:
            pair = pending["pair"]
            frame = frames[pair]
            idx = idx_by_pair[pair].get(ts)
            if idx is not None:
                row = frame.iloc[idx]
                entry_price = float(row["open"]) * (1 + commission_pct + slippage_pct)
                position = {
                    "pair": pair,
                    "entry_idx": idx,
                    "entry_ts": ts,
                    "entry_price": entry_price,
                    "stop_price": entry_price * (1 - exit_spec.stop_pct),
                    "target_price": entry_price * (1 + exit_spec.target_pct),
                    "best_price": entry_price,
                    "score": pending["score"],
                }
                pending = None

        if position is not None:
            pair = position["pair"]
            frame = frames[pair]
            idx = idx_by_pair[pair].get(ts)
            if idx is not None:
                row = frame.iloc[idx]
                high_price = float(row["high"])
                low_price = float(row["low"])
                close_price = float(row["close"])
                position["best_price"] = max(float(position["best_price"]), high_price)

                exit_reason: str | None = None
                exit_price: float | None = None
                if low_price <= float(position["stop_price"]):
                    exit_reason = "STOP_LOSS"
                    exit_price = float(position["stop_price"]) * (1 - commission_pct - slippage_pct)
                elif high_price >= float(position["target_price"]):
                    exit_reason = "TAKE_PROFIT"
                    exit_price = float(position["target_price"]) * (1 - commission_pct - slippage_pct)
                else:
                    bars_held = idx - int(position["entry_idx"])
                    trend_lost = (
                        close_price < float(row["ema_fast"])
                        and float(row["rs_rank_15"]) > 3
                    ) or (float(row["ctx60_momentum_medium"]) < -0.002)
                    if trend_lost:
                        exit_reason = "TREND_LOST"
                        exit_price = close_price * (1 - commission_pct - slippage_pct)
                    elif bars_held >= exit_spec.hold_bars:
                        exit_reason = "TIME_LIMIT"
                        exit_price = close_price * (1 - commission_pct - slippage_pct)

                if exit_reason and exit_price is not None:
                    entry_price = float(position["entry_price"])
                    pnl_pct = (exit_price - entry_price) / entry_price - commission_pct - slippage_pct
                    trades.append(
                        backtest.BacktestTrade(
                            construction=strategy_name,
                            pair=pair,
                            entry_bar=int(position["entry_idx"]),
                            entry_ts=int(position["entry_ts"]),
                            entry_price=entry_price,
                            exit_bar=idx,
                            exit_ts=ts,
                            exit_price=exit_price,
                            exit_reason=exit_reason,
                            pnl_pct=pnl_pct,
                            mfe_pct=(float(position["best_price"]) - entry_price) / entry_price,
                            bars_held=idx - int(position["entry_idx"]),
                            signal_score=float(position["score"]),
                            trend_strength=float(frame.loc[int(position["entry_idx"]), "trend_strength"]),
                            pullback_depth_pct=float(frame.loc[int(position["entry_idx"]), "pullback_depth_pct"]),
                            distance_from_vwap=float(frame.loc[int(position["entry_idx"]), "distance_from_vwap"]),
                            compression_ratio=float(frame.loc[int(position["entry_idx"]), "compression_ratio"]),
                        )
                    )
                    position = None
            continue

        if pending is not None:
            continue

        candidates: list[tuple[float, str, int]] = []
        for pair, frame in frames.items():
            idx = idx_by_pair[pair].get(ts)
            if idx is None or idx + 1 >= len(frame):
                continue
            if not bool(masks_by_pair[pair].iloc[idx]):
                continue
            row = frame.iloc[idx]
            candidates.append((_candidate_score(row), pair, idx))

        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        score, pair, idx = candidates[0]
        next_ts = int(frames[pair].iloc[idx + 1]["ts"])
        pending = {"pair": pair, "entry_ts": next_ts, "score": score}

    return trades


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    pairs = [pair.strip() for pair in args.pairs.split(",") if pair.strip()]

    raw_histories = {
        pair: _load_history(pair, args.history_days, cache_dir, args.trade_count, args.trade_pause_sec)
        for pair in pairs
    }
    frames = {
        pair: _build_pair_frame(raw, pair)
        for pair, raw in raw_histories.items()
    }
    frames = _attach_cross_context(frames)

    max_ts = max(int(frame["ts"].max()) for frame in frames.values())
    recent_cutoff = max_ts - args.recent_days * 24 * 60 * 60

    baseline_pair = "GIGAUSD" if "GIGAUSD" in raw_histories else pairs[0]
    baseline_trades = backtest.run_backtest(
        pair=baseline_pair,
        df_raw=raw_histories[baseline_pair],
        config=strat.DEFAULT_CONFIG,
        commission_pct=args.commission_pct,
        slippage_pct=args.slippage_pct,
        construction="tc15_tighter_volume_cap",
    )
    baseline_net = float(sum(trade.pnl_pct for trade in baseline_trades))
    baseline_count = len(baseline_trades)

    summary_rows: list[dict] = [
        {
            "strategy": "baseline_tc15_tighter_volume_cap",
            "universe_size": 1,
            **shadow_backtest._summarize_trades(baseline_trades, recent_cutoff),
            "beats_baseline": False,
            "verdict": "BASELINE",
        }
    ]
    trade_rows: list[dict] = []
    for trade in baseline_trades:
        row = asdict(trade)
        row["strategy"] = "baseline_tc15_tighter_volume_cap"
        trade_rows.append(row)

    masks_all = {pair: _signal_masks(frame) for pair, frame in frames.items()}
    for name, exit_spec in _specs().items():
        masks_by_pair = {pair: masks_all[pair][name] for pair in pairs}
        trades = _run_portfolio_strategy(
            frames=frames,
            masks_by_pair=masks_by_pair,
            exit_spec=exit_spec,
            commission_pct=args.commission_pct,
            slippage_pct=args.slippage_pct,
            strategy_name=name,
        )
        net_full = float(sum(trade.pnl_pct for trade in trades))
        beats_baseline = net_full > baseline_net and len(trades) > baseline_count
        summary_rows.append(
            {
                "strategy": name,
                "universe_size": len(pairs),
                **shadow_backtest._summarize_trades(trades, recent_cutoff),
                "beats_baseline": beats_baseline,
                "verdict": "KEEP" if beats_baseline else "KILL",
            }
        )
        for trade in trades:
            row = asdict(trade)
            row["strategy"] = name
            trade_rows.append(row)
        print(f"{name}: trades={len(trades)} net={net_full:+.3%}")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["verdict", "net_full", "trades_full"],
        ascending=[True, False, False],
    )
    print(f"\n{'=' * 120}")
    print(" CROSS-SECTIONAL ROTATION SUMMARY")
    print(f"{'=' * 120}")
    cols = [
        "strategy",
        "universe_size",
        "trades_full",
        "net_full",
        "win_full",
        "max_dd",
        "trades_recent",
        "net_recent",
        "win_recent",
        "verdict",
    ]
    formatters = {
        "net_full": lambda value: f"{value:+.3%}",
        "win_full": lambda value: f"{value:.1%}",
        "max_dd": lambda value: f"{value:.3%}",
        "net_recent": lambda value: f"{value:+.3%}",
        "win_recent": lambda value: f"{value:.1%}",
    }
    print(summary_df[cols].to_string(index=False, formatters=formatters))

    summary_df.to_csv(args.summary_csv, index=False)
    print(f"\nWrote summary to {args.summary_csv}")
    if trade_rows:
        pd.DataFrame(trade_rows).to_csv(args.trades_csv, index=False)
        print(f"Wrote trades to {args.trades_csv}")


if __name__ == "__main__":
    main()
