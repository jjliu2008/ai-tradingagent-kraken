from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import expanded_screener as base
import strategy as strat


@dataclass(frozen=True)
class UniverseExitSpec:
    hold_bars: int
    stop_pct: float
    target_pct: float | None = None
    trail_activation_pct: float | None = None
    trail_stop_pct: float | None = None


@dataclass
class UniverseTrade:
    pair: str
    entry_ts: int
    exit_ts: int
    pnl_pct: float
    exit_reason: str
    mfe_pct: float
    bars_held: int


@dataclass
class UniverseSummary:
    interval: int
    strategy: str
    window_days: int
    trades: int
    net_pnl: float
    avg_pnl: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    avg_mfe: float
    sharpe: float
    avg_bars_held: float


def load_histories(pairs: list[str], cache_dir: Path) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        histories[pair] = base.load_history(pair, cache_dir)
    return histories


def build_interval_frames(
    histories: dict[str, pd.DataFrame],
    interval: int,
) -> dict[str, pd.DataFrame]:
    indexed_frames: dict[str, pd.DataFrame] = {}
    common_ts: pd.Index | None = None

    for pair, history in histories.items():
        resampled = base.resample_ohlcv(history, interval)
        features = base.add_expanded_features(strat.compute_features(resampled))
        features = features.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
        indexed = features.set_index("ts").sort_index()
        indexed_frames[pair] = indexed
        common_ts = indexed.index if common_ts is None else common_ts.intersection(indexed.index)

    if common_ts is None or len(common_ts) == 0:
        raise RuntimeError(f"No aligned timestamps for interval {interval}")

    common_ts = common_ts.sort_values()
    aligned: dict[str, pd.DataFrame] = {}
    for pair, frame in indexed_frames.items():
        aligned[pair] = frame.loc[common_ts].reset_index()
    return aligned


def add_universe_context(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    ts_index = next(iter(frames.values()))["ts"]
    close = pd.DataFrame({pair: df["close"].to_numpy() for pair, df in frames.items()}, index=ts_index)
    ema_fast = pd.DataFrame({pair: df["ema_fast"].to_numpy() for pair, df in frames.items()}, index=ts_index)
    ema_slow = pd.DataFrame({pair: df["ema_slow"].to_numpy() for pair, df in frames.items()}, index=ts_index)
    momentum_medium = pd.DataFrame(
        {pair: df["momentum_medium"].to_numpy() for pair, df in frames.items()},
        index=ts_index,
    )
    momentum_long = pd.DataFrame(
        {pair: df["momentum_long"].to_numpy() for pair, df in frames.items()},
        index=ts_index,
    )
    atr_pct = pd.DataFrame({pair: df["atr_pct"].to_numpy() for pair, df in frames.items()}, index=ts_index)
    volume_ratio = pd.DataFrame(
        {pair: df["volume_ratio"].to_numpy() for pair, df in frames.items()},
        index=ts_index,
    )

    rs_raw = momentum_medium / atr_pct.clip(lower=0.0020)
    rs_rank = rs_raw.rank(axis=1, ascending=False, method="first")
    rs_mean = rs_raw.mean(axis=1)
    rs_std = rs_raw.std(axis=1).replace(0, np.nan)
    rs_z = rs_raw.sub(rs_mean, axis=0).div(rs_std, axis=0)

    leader_pair = rs_raw.fillna(-np.inf).idxmax(axis=1)
    top_score = rs_raw.max(axis=1)
    second_score = rs_raw.apply(
        lambda row: row.nlargest(2).iloc[-1] if row.count() >= 2 else np.nan,
        axis=1,
    )
    leader_gap = (top_score - second_score).fillna(0.0)

    breadth_up = (ema_fast > ema_slow).mean(axis=1)
    breadth_mom = (momentum_medium > 0).mean(axis=1)
    breadth_long = (momentum_long > 0).mean(axis=1)
    breadth_volume = (volume_ratio > 1.0).mean(axis=1)
    dispersion = rs_raw.std(axis=1)
    dispersion_low = dispersion < dispersion.rolling(40, min_periods=10).quantile(0.35)
    cross_mean_mom = momentum_medium.mean(axis=1)
    cross_median_mom = momentum_medium.median(axis=1)
    pair_ret_1 = close.pct_change(1)
    ret_rank = pair_ret_1.rank(axis=1, ascending=False, method="first")

    enriched: dict[str, pd.DataFrame] = {}
    for pair, frame in frames.items():
        out = frame.copy()
        out["rs_score"] = rs_raw[pair].to_numpy()
        out["rs_rank"] = rs_rank[pair].to_numpy()
        out["rs_z"] = rs_z[pair].fillna(0.0).to_numpy()
        out["leader_gap"] = leader_gap.to_numpy()
        out["breadth_up"] = breadth_up.to_numpy()
        out["breadth_mom"] = breadth_mom.to_numpy()
        out["breadth_long"] = breadth_long.to_numpy()
        out["breadth_volume"] = breadth_volume.to_numpy()
        out["dispersion"] = dispersion.fillna(0.0).to_numpy()
        out["dispersion_low_prev"] = (
            dispersion_low.astype("boolean").shift(1).fillna(False).astype(bool).to_numpy()
        )
        out["pair_rel_mom"] = (momentum_medium[pair] - cross_mean_mom).to_numpy()
        out["pair_rel_mom_median"] = (momentum_medium[pair] - cross_median_mom).to_numpy()
        out["leader_pair"] = leader_pair.to_numpy()
        out["is_leader"] = (leader_pair == pair).astype(float).to_numpy()
        out["leader_prev"] = (leader_pair.shift(1) == pair).fillna(False).astype(float).to_numpy()
        out["rank_jump"] = (rs_rank[pair].shift(1) - rs_rank[pair]).fillna(0.0).to_numpy()
        out["ret_rank"] = ret_rank[pair].to_numpy()
        enriched[pair] = out
    return enriched


def build_cross_pair_candidates(
    frames: dict[str, pd.DataFrame],
) -> list[tuple[str, dict[str, pd.Series], UniverseExitSpec]]:
    candidates: list[tuple[str, dict[str, pd.Series], UniverseExitSpec]] = []

    leader_rotation_scores: dict[str, pd.Series] = {}
    leader_pullback_scores: dict[str, pd.Series] = {}
    breadth_thrust_scores: dict[str, pd.Series] = {}
    laggard_catchup_scores: dict[str, pd.Series] = {}
    dispersion_expand_scores: dict[str, pd.Series] = {}
    leader_persistence_scores: dict[str, pd.Series] = {}

    for pair, df in frames.items():
        trend_up = df["ema_fast"] > df["ema_slow"]
        breakout_12 = df["close"] > df["roll_high_12"]
        breakout_20 = df["close"] > df["roll_high_20"]
        touched_fast = df["low"] <= df["ema_fast"]
        moderate_volume = df["volume_ratio"].between(0.8, 2.5)
        strong_close = df["close_location"] > 0.65

        leader_rotation = (
            (df["is_leader"] > 0)
            & (df["rs_z"] > 0.45)
            & (df["rank_jump"] >= 1)
            & trend_up
            & breakout_12
            & (df["breadth_up"] >= 0.60)
            & (df["breadth_mom"] >= 0.40)
            & (df["compression_ratio"] < 1.00)
            & strong_close
            & moderate_volume
        )
        leader_rotation_scores[pair] = leader_rotation.astype(float) * (
            1.4
            + df["rs_z"].clip(lower=0)
            + 0.25 * df["leader_gap"].clip(lower=0)
            + 0.2 * df["rank_jump"].clip(lower=0)
        )

        leader_pullback = (
            (df["rs_rank"] <= 2)
            & trend_up
            & (df["breadth_up"] >= 0.60)
            & (df["breadth_long"] >= 0.50)
            & touched_fast
            & (df["close"] > df["ema_fast"])
            & (df["pair_rel_mom"] > -0.003)
            & (df["close_location"] > 0.55)
            & df["green_bar"]
            & df["volume_ratio"].between(0.7, 1.8)
        )
        leader_pullback_scores[pair] = leader_pullback.astype(float) * (
            1.0
            + 0.6 * (3 - df["rs_rank"]).clip(lower=0)
            + df["pair_rel_mom"].clip(lower=-0.002)
            + df["trend_strength"].clip(lower=0)
        )

        breadth_thrust = (
            (df["rs_rank"] <= 2)
            & trend_up
            & breakout_20
            & (df["breadth_up"] >= 0.75)
            & (df["breadth_mom"] >= 0.60)
            & (df["breadth_volume"] >= 0.40)
            & (df["body_pct"] > 0.006)
            & (df["volume_ratio"] > 1.10)
            & (df["close_location"] > 0.70)
        )
        breadth_thrust_scores[pair] = breadth_thrust.astype(float) * (
            1.3
            + df["rs_z"].clip(lower=0)
            + 2.0 * df["body_pct"].clip(lower=0)
        )

        laggard_catchup = (
            (df["rs_rank"].between(2, 4))
            & trend_up
            & (df["breadth_up"] >= 0.60)
            & (df["breadth_long"] >= 0.50)
            & (df["rank_jump"] >= 1)
            & (df["pair_rel_mom"].shift(1) < 0)
            & (df["pair_rel_mom"] > 0)
            & (df["momentum_short"] > 0)
            & (df["close"] > df["ema_fast"])
            & (df["close_location"] > 0.60)
            & (df["volume_ratio"] > 0.95)
        )
        laggard_catchup_scores[pair] = laggard_catchup.astype(float) * (
            0.8
            + 0.4 * df["rank_jump"].clip(lower=0)
            + df["pair_rel_mom"].clip(lower=0)
            + df["momentum_short"].clip(lower=0)
        )

        dispersion_expand = (
            (df["is_leader"] > 0)
            & df["dispersion_low_prev"]
            & trend_up
            & breakout_12
            & (df["rs_z"] > 0.75)
            & (df["volume_ratio"] > 1.0)
            & (df["close_location"] > 0.68)
        )
        dispersion_expand_scores[pair] = dispersion_expand.astype(float) * (
            1.1
            + df["rs_z"].clip(lower=0)
            + 0.2 * df["leader_gap"].clip(lower=0)
        )

        leader_persistence = (
            (df["is_leader"] > 0)
            & (df["leader_prev"] > 0)
            & (df["rs_z"] > 0.60)
            & trend_up
            & breakout_12
            & (df["momentum_medium"] > 0.010)
            & moderate_volume
            & (df["close_location"] > 0.65)
        )
        leader_persistence_scores[pair] = leader_persistence.astype(float) * (
            1.2
            + df["rs_z"].clip(lower=0)
            + df["momentum_medium"].clip(lower=0)
        )

    candidates.append(
        (
            "leader_rotation_breakout",
            leader_rotation_scores,
            UniverseExitSpec(hold_bars=4, stop_pct=0.012, target_pct=0.035),
        )
    )
    candidates.append(
        (
            "leader_pullback_resume",
            leader_pullback_scores,
            UniverseExitSpec(hold_bars=4, stop_pct=0.010, target_pct=0.024),
        )
    )
    candidates.append(
        (
            "breadth_thrust_winner",
            breadth_thrust_scores,
            UniverseExitSpec(hold_bars=4, stop_pct=0.014, target_pct=0.040),
        )
    )
    candidates.append(
        (
            "leader_laggard_catchup",
            laggard_catchup_scores,
            UniverseExitSpec(hold_bars=3, stop_pct=0.010, target_pct=0.025),
        )
    )
    candidates.append(
        (
            "dispersion_expansion_leader",
            dispersion_expand_scores,
            UniverseExitSpec(hold_bars=5, stop_pct=0.012, target_pct=0.035),
        )
    )
    candidates.append(
        (
            "leader_persistence_trail",
            leader_persistence_scores,
            UniverseExitSpec(
                hold_bars=6,
                stop_pct=0.012,
                target_pct=None,
                trail_activation_pct=0.020,
                trail_stop_pct=0.010,
            ),
        )
    )
    return candidates


def simulate_universe_trades(
    frames: dict[str, pd.DataFrame],
    scores: dict[str, pd.Series],
    exit_spec: UniverseExitSpec,
    side_cost_pct: float,
) -> list[UniverseTrade]:
    pairs = list(frames)
    last_index = len(next(iter(frames.values()))) - 1
    trades: list[UniverseTrade] = []
    i = 0

    while i < last_index:
        best_pair: str | None = None
        best_score = 0.0
        for pair in pairs:
            value = scores[pair].iloc[i]
            if pd.notna(value) and float(value) > best_score:
                best_score = float(value)
                best_pair = pair

        if best_pair is None:
            i += 1
            continue

        df = frames[best_pair]
        entry_bar = i + 1
        if entry_bar > last_index:
            break

        entry_ts = int(df.iloc[entry_bar]["ts"])
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
        exit_reason = "TIME_LIMIT"

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
                exit_reason = "STOP_LOSS"
                break
            if target_price is not None and high_price >= target_price:
                exit_bar = bar
                exit_price = target_price * (1 - side_cost_pct)
                exit_reason = "TAKE_PROFIT"
                break
            if trail_stop is not None and low_price <= trail_stop:
                exit_bar = bar
                exit_price = trail_stop * (1 - side_cost_pct)
                exit_reason = "TRAIL_STOP"
                break
            if bar == max_exit_bar:
                exit_bar = bar
                exit_price = close_price * (1 - side_cost_pct)
                exit_reason = "TIME_LIMIT"

        trades.append(
            UniverseTrade(
                pair=best_pair,
                entry_ts=entry_ts,
                exit_ts=int(df.iloc[exit_bar]["ts"]),
                pnl_pct=exit_price / entry_price - 1,
                exit_reason=exit_reason,
                mfe_pct=best_high / entry_price - 1,
                bars_held=exit_bar - entry_bar,
            )
        )
        i = exit_bar + 1

    return trades


def summarize_trades(
    interval: int,
    strategy_name: str,
    trades: list[UniverseTrade],
    window_days: int,
) -> UniverseSummary:
    pnls = [trade.pnl_pct for trade in trades]
    return UniverseSummary(
        interval=interval,
        strategy=strategy_name,
        window_days=window_days,
        trades=len(trades),
        net_pnl=float(np.sum(pnls)) if pnls else 0.0,
        avg_pnl=float(np.mean(pnls)) if pnls else 0.0,
        win_rate=float(np.mean(np.asarray(pnls) > 0)) if pnls else 0.0,
        profit_factor=base.profit_factor(pnls),
        max_drawdown=base.max_drawdown(pnls),
        avg_mfe=float(np.mean([trade.mfe_pct for trade in trades])) if trades else 0.0,
        sharpe=base.trade_sharpe(pnls),
        avg_bars_held=float(np.mean([trade.bars_held for trade in trades])) if trades else 0.0,
    )


def screen_interval(
    histories: dict[str, pd.DataFrame],
    interval: int,
    recent_days: int,
    side_cost_pct: float,
    min_keep_trades: int,
) -> pd.DataFrame:
    frames = add_universe_context(build_interval_frames(histories, interval))
    end_ts = int(next(iter(frames.values()))["ts"].iloc[-1])
    recent_cutoff = end_ts - recent_days * 24 * 60 * 60

    full_candidates = build_cross_pair_candidates(frames)
    recent_frames = {
        pair: df.loc[df["ts"] >= recent_cutoff].reset_index(drop=True)
        for pair, df in frames.items()
    }
    recent_candidates = {
        name: scores
        for name, scores, _ in build_cross_pair_candidates(recent_frames)
    }

    rows: list[dict] = []
    full_window_days = int(
        round(
            (
                next(iter(frames.values()))["ts"].iloc[-1]
                - next(iter(frames.values()))["ts"].iloc[0]
            )
            / 86400
        )
    )

    for strategy_name, full_scores, exit_spec in full_candidates:
        recent_scores = recent_candidates[strategy_name]
        full_trades = simulate_universe_trades(frames, full_scores, exit_spec, side_cost_pct)
        recent_trades = simulate_universe_trades(recent_frames, recent_scores, exit_spec, side_cost_pct)
        full_result = summarize_trades(interval, strategy_name, full_trades, full_window_days)
        recent_result = summarize_trades(interval, strategy_name, recent_trades, recent_days)

        keep = (
            full_result.net_pnl > 0
            and recent_result.net_pnl > 0
            and full_result.trades >= min_keep_trades
            and recent_result.trades >= max(2, min_keep_trades // 2)
        )
        score = (
            full_result.net_pnl
            + 1.5 * recent_result.net_pnl
            + 0.01 * full_result.sharpe
            + 0.002 * full_result.trades
        )

        full_pair_counts = pd.Series([trade.pair for trade in full_trades]).value_counts()
        recent_pair_counts = pd.Series([trade.pair for trade in recent_trades]).value_counts()

        rows.append(
            {
                "interval": interval,
                "strategy": strategy_name,
                "trades_full": full_result.trades,
                "net_full": full_result.net_pnl,
                "avg_full": full_result.avg_pnl,
                "win_full": full_result.win_rate,
                "pf_full": full_result.profit_factor,
                "dd_full": full_result.max_drawdown,
                "mfe_full": full_result.avg_mfe,
                "sharpe_full": full_result.sharpe,
                "bars_full": full_result.avg_bars_held,
                "trades_recent": recent_result.trades,
                "net_recent": recent_result.net_pnl,
                "avg_recent": recent_result.avg_pnl,
                "win_recent": recent_result.win_rate,
                "pf_recent": recent_result.profit_factor,
                "dd_recent": recent_result.max_drawdown,
                "mfe_recent": recent_result.avg_mfe,
                "sharpe_recent": recent_result.sharpe,
                "top_pairs_full": ",".join(
                    f"{pair}:{count}" for pair, count in full_pair_counts.head(3).items()
                ),
                "top_pairs_recent": ",".join(
                    f"{pair}:{count}" for pair, count in recent_pair_counts.head(3).items()
                ),
                "score": score,
                "verdict": "KEEP" if keep else "KILL",
            }
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-pair orthogonal strategy screener")
    parser.add_argument("--pairs", default="GIGAUSD,DOGUSD,SOLUSD,HYPEUSD,COQUSD")
    parser.add_argument("--intervals", default="15,30,60")
    parser.add_argument("--recent-days", type=int, default=60)
    parser.add_argument("--fee-pct", type=float, default=0.0026)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--min-keep-trades", type=int, default=5)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--results-csv", default="orthogonal_screen_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = [pair.strip() for pair in args.pairs.split(",") if pair.strip()]
    intervals = [int(value.strip()) for value in args.intervals.split(",") if value.strip()]
    side_cost_pct = args.fee_pct + args.slippage_pct
    histories = load_histories(pairs, Path(args.cache_dir))

    frames: list[pd.DataFrame] = []
    for interval in intervals:
        print(f"Screening interval {interval}m across {', '.join(pairs)}")
        frames.append(
            screen_interval(
                histories=histories,
                interval=interval,
                recent_days=args.recent_days,
                side_cost_pct=side_cost_pct,
                min_keep_trades=args.min_keep_trades,
            )
        )

    results = pd.concat(frames, ignore_index=True)
    results = results.sort_values(
        ["verdict", "score", "net_recent", "net_full"],
        ascending=[True, False, False, False],
    )

    fmt = lambda v: f"{v:+.3%}" if np.isfinite(v) and abs(v) < 10 else f"{v:.2f}"
    keepers = results.loc[results["verdict"] == "KEEP"].copy()

    print(f"\n{'=' * 96}")
    print(f" ORTHOGONAL SCREEN | {len(results)} combos tested | {len(keepers)} survived")
    print(f"{'=' * 96}")
    cols = [
        "interval",
        "strategy",
        "trades_full",
        "net_full",
        "win_full",
        "pf_full",
        "dd_full",
        "trades_recent",
        "net_recent",
        "win_recent",
        "top_pairs_recent",
        "score",
    ]
    if keepers.empty:
        print("\nNo survivors. Top 10 by score:")
        print(results[cols].head(10).to_string(index=False, float_format=fmt))
    else:
        print("\nSurvivors:")
        print(keepers[cols].to_string(index=False, float_format=fmt))

    if args.results_csv:
        results.to_csv(args.results_csv, index=False)
        print(f"\nWrote {len(results)} rows to {args.results_csv}")


if __name__ == "__main__":
    main()
