from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import expanded_screener as ex
import lower_bar_screener as lbs
import strategy as strat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive lower-bar screener with MTF regime gates and cross-pair context"
    )
    parser.add_argument("--pairs", default="GIGAUSD,DOGUSD,COQUSD,HYPEUSD")
    parser.add_argument("--intervals", default="3,5")
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
    parser.add_argument("--results-csv", default="adaptive_strategy_screen_results.csv")
    return parser.parse_args()


def _align_context(
    base: pd.DataFrame,
    higher: pd.DataFrame,
    base_interval: int,
    higher_interval: int,
    prefix: str,
) -> pd.DataFrame:
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
    ]
    ctx = higher[["ts"] + cols].copy()
    ctx["effective_ts"] = ctx["ts"].astype(int) + (higher_interval - base_interval) * 60
    aligned = (
        ctx.set_index("effective_ts")[cols]
        .reindex(base["ts"].astype(int))
        .ffill()
        .reset_index(drop=True)
        .rename(columns={column: f"{prefix}_{column}" for column in cols})
    )
    return aligned


def _build_pair_frame(dense_1m: pd.DataFrame, pair: str, base_interval: int) -> pd.DataFrame:
    base = ex.resample_ohlcv(dense_1m, base_interval)
    base = ex.add_expanded_features(strat.compute_features(base)).reset_index(drop=True)
    ctx15 = ex.add_expanded_features(strat.compute_features(ex.resample_ohlcv(dense_1m, 15))).reset_index(drop=True)
    ctx60 = ex.add_expanded_features(strat.compute_features(ex.resample_ohlcv(dense_1m, 60))).reset_index(drop=True)

    out = base.copy()
    out["pair"] = pair
    out["base_interval"] = base_interval
    out = pd.concat(
        [
            out,
            _align_context(out, ctx15, base_interval=base_interval, higher_interval=15, prefix="ctx15"),
            _align_context(out, ctx60, base_interval=base_interval, higher_interval=60, prefix="ctx60"),
        ],
        axis=1,
    )
    return out


def _attach_cross_pair_context(pair_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for pair, frame in pair_frames.items():
        view = frame[
            ["ts", "pair", "momentum_short", "momentum_medium", "atr_pct", "trend_strength", "volume_ratio"]
        ].copy()
        atr_scale = view["atr_pct"].clip(lower=1e-6)
        view["rs_score"] = view["momentum_medium"] / atr_scale
        view["impulse_score"] = view["momentum_short"] / atr_scale
        rows.append(view)

    universe = pd.concat(rows, ignore_index=True)
    universe["rs_rank"] = universe.groupby("ts")["rs_score"].rank(method="dense", ascending=False)
    universe["impulse_rank"] = universe.groupby("ts")["impulse_score"].rank(method="dense", ascending=False)

    breadth = universe.groupby("ts").agg(
        market_breadth=("momentum_medium", lambda s: float((s > 0).mean())),
        strong_breadth=("trend_strength", lambda s: float((s > 0.0015).mean())),
        mean_rs=("rs_score", "mean"),
        mean_impulse=("impulse_score", "mean"),
    )
    universe = universe.join(breadth, on="ts")
    universe["rs_gap"] = universe["rs_score"] - universe["mean_rs"]
    universe["impulse_gap"] = universe["impulse_score"] - universe["mean_impulse"]
    universe["leader_recent"] = (
        universe.sort_values(["pair", "ts"])
        .groupby("pair")["rs_rank"]
        .transform(lambda s: s.rolling(3, min_periods=1).min())
        <= 2
    )
    keep_cols = [
        "ts",
        "pair",
        "rs_score",
        "impulse_score",
        "rs_rank",
        "impulse_rank",
        "market_breadth",
        "strong_breadth",
        "mean_rs",
        "mean_impulse",
        "rs_gap",
        "impulse_gap",
        "leader_recent",
    ]
    universe = universe[keep_cols].copy()

    enriched: dict[str, pd.DataFrame] = {}
    for pair, frame in pair_frames.items():
        ctx = universe.loc[universe["pair"] == pair].drop(columns=["pair"]).reset_index(drop=True)
        enriched[pair] = pd.concat([frame.reset_index(drop=True), ctx.drop(columns=["ts"])], axis=1)
    return enriched


def _candidate_thresholds(interval: int) -> dict[str, float]:
    if interval == 3:
        return {
            "trend_base": 0.0015,
            "trend_ctx15": 0.0015,
            "trend_ctx60": 0.0010,
            "mom_base": 0.0080,
            "mom_ctx15": 0.0060,
            "mom_ctx60": 0.0080,
            "body_burst": 0.0055,
        }
    return {
        "trend_base": 0.0018,
        "trend_ctx15": 0.0015,
        "trend_ctx60": 0.0012,
        "mom_base": 0.0100,
        "mom_ctx15": 0.0070,
        "mom_ctx60": 0.0100,
        "body_burst": 0.0065,
    }


def build_adaptive_candidates(df: pd.DataFrame, interval: int) -> list[tuple[str, pd.Series, ex.ExitSpec]]:
    cfg = _candidate_thresholds(interval)

    trend_up = df["ema_fast"] > df["ema_slow"]
    ctx15_up = df["ctx15_ema_fast"] > df["ctx15_ema_slow"]
    ctx60_up = df["ctx60_ema_fast"] > df["ctx60_ema_slow"]
    strong15 = (
        ctx15_up
        & (df["ctx15_trend_strength"] > cfg["trend_ctx15"])
        & (df["ctx15_momentum_medium"] > cfg["mom_ctx15"])
    ).fillna(False)
    regime15 = (
        ctx15_up
        & (df["ctx15_trend_strength"] > (cfg["trend_ctx15"] * 0.5))
        & (df["ctx15_momentum_medium"] > 0.0)
    ).fillna(False)
    strong60 = (
        ctx60_up
        & (df["ctx60_trend_strength"] > cfg["trend_ctx60"])
        & (df["ctx60_momentum_medium"] > cfg["mom_ctx60"])
    ).fillna(False)
    regime60 = (
        ctx60_up
        & (df["ctx60_trend_strength"] > 0.0)
        & (df["ctx60_momentum_medium"] > -0.002)
    ).fillna(False)
    hot60 = (
        ctx60_up
        & (df["ctx60_trend_strength"] > (cfg["trend_ctx60"] * 1.8))
        & (df["ctx60_momentum_medium"] > (cfg["mom_ctx60"] * 1.25))
    ).fillna(False)
    broad_market = (df["market_breadth"] >= 0.50).fillna(False)
    strong_breadth = (df["strong_breadth"] >= 0.50).fillna(False)
    market_ok = (df["market_breadth"] >= 0.25).fillna(False)
    top2 = (df["rs_rank"] <= 2).fillna(False)
    leader = (df["rs_rank"] <= 1).fillna(False)
    rs_positive = (df["rs_gap"] > 0).fillna(False)
    impulse_positive = (df["impulse_gap"] > 0).fillna(False)
    peer_ok = ((df["rs_rank"] <= 3) | (df["rs_gap"] > -0.10)).fillna(False)
    leader_persistent = df["leader_recent"].fillna(False)

    breakout = (
        trend_up
        & (df["trend_strength"] > cfg["trend_base"])
        & (df["close"] > df["roll_high_12"])
        & (df["momentum_medium"] > cfg["mom_base"])
        & (df["volume_ratio"] > 1.0)
        & (df["volume_ratio"] <= 4.5)
        & (df["close_location"] > 0.70)
        & (df["compression_ratio"] < 0.90)
    ).fillna(False)
    breakout_trail = (
        breakout
        & (df["volume_ratio"] > 1.2)
        & (df["compression_ratio"] < 0.85)
        & (df["close_location"] > 0.74)
    ).fillna(False)
    volume_spike = (
        trend_up
        & (df["trend_strength"] > cfg["trend_base"])
        & df["green_bar"]
        & (df["volume_ratio"] > 1.8)
        & (df["body_pct"] > 0.0045)
        & (df["close_location"] > 0.74)
    ).fillna(False)
    quiet_burst = (
        trend_up
        & (df["body_pct"].shift(1).abs() < 0.003)
        & (df["body_pct"] > cfg["body_burst"])
        & (df["volume_ratio"] > 1.3)
        & (df["close_location"] > 0.75)
    ).fillna(False)
    higher_low = (
        trend_up
        & (df["trend_strength"] > cfg["trend_base"])
        & (df["low"].shift(1) > df["roll_low_12"])
        & (df["close"] > df["roll_high_12"])
        & (df["close_location"] > 0.65)
        & (df["volume_ratio"] > 0.9)
    ).fillna(False)
    pullback_resume = (
        trend_up
        & (df["trend_strength"] > (cfg["trend_base"] * 0.8))
        & (df["low"] <= df["ema_fast"])
        & (df["close"] > df["ema_fast"])
        & df["green_bar"]
        & (df["close_location"] > 0.60)
        & (df["volume_ratio"] >= 0.8)
        & (df["volume_ratio"] <= 2.0)
        & df["distance_from_vwap"].between(-0.010, 0.015)
    ).fillna(False)
    consensus_count = (
        breakout.astype(int)
        + volume_spike.astype(int)
        + quiet_burst.astype(int)
        + higher_low.astype(int)
    )

    momentum_breakout = (
        (df["trend_strength"] > 0.0020)
        & trend_up
        & (df["close"] > df["roll_high_12"])
        & (df["momentum_medium"] > 0.015)
        & (df["volume_ratio"] > 1.0)
        & (df["close_location"] > 0.70)
        & (df["compression_ratio"] < 0.90)
    ).fillna(False)
    momentum_breakout_trail = (
        (df["trend_strength"] > 0.0025)
        & trend_up
        & (df["close"] > df["roll_high_12"])
        & (df["momentum_medium"] > 0.020)
        & (df["volume_ratio"] > 1.2)
        & (df["close_location"] > 0.70)
    ).fillna(False)
    higher_low_structure = (
        trend_up
        & (df["trend_strength"] > 0.0020)
        & (df["low"].shift(1) > df["roll_low_12"])
        & (df["close"] > df["roll_high_12"])
        & (df["close_location"] > 0.65)
        & (df["volume_ratio"] > 1.0)
    ).fillna(False)
    triple_confluence = (
        trend_up
        & (df["trend_strength"] > 0.0025)
        & (df["macd_hist"] > 0)
        & (df["macd_hist"] > df["macd_hist"].shift(1))
        & (df["compression_ratio"] < 0.85)
        & (df["volume_ratio"] > 1.1)
        & (df["close_location"] > 0.65)
        & (df["close"] > df["roll_high_12"])
    ).fillna(False)
    momentum_accel = (
        trend_up
        & (df["mom_accel"] > 0.005)
        & (df["momentum_medium"] > 0.010)
        & (df["trend_strength"] > 0.0015)
        & (df["close_location"] > 0.60)
        & (df["volume_ratio"] > 0.9)
    ).fillna(False)
    rsi_momentum_thrust = (
        trend_up
        & (df["rsi_14"] > 60)
        & (df["rsi_14"].shift(1) <= 60)
        & (df["trend_strength"] > 0.0015)
        & (df["close_location"] > 0.60)
        & (df["volume_ratio"] > 0.9)
    ).fillna(False)
    volume_spike_trend = (
        trend_up
        & df["vol_spike"]
        & df["green_bar"]
        & (df["close_location"] > 0.70)
        & (df["trend_strength"] > 0.0015)
        & (df["body_pct"] > 0.005)
    ).fillna(False)
    quiet_then_burst = (
        trend_up
        & (df["body_pct"].shift(1).abs() < 0.003)
        & (df["body_pct"] > 0.008)
        & (df["close_location"] > 0.75)
        & (df["volume_ratio"] > 1.3)
        & (df["trend_strength"] > 0.0015)
    ).fillna(False)

    return [
        (
            "freqai_mtf_momentum_breakout_trail",
            momentum_breakout_trail & regime15 & regime60 & peer_ok & market_ok,
            ex.ExitSpec(
                hold_bars=8,
                stop_pct=0.015,
                target_pct=None,
                trail_activation_pct=0.020,
                trail_stop_pct=0.010,
            ),
        ),
        (
            "freqai_mtf_volume_spike_trend",
            volume_spike_trend & regime15 & regime60 & peer_ok & market_ok,
            ex.ExitSpec(hold_bars=5, stop_pct=0.012, target_pct=0.030),
        ),
        (
            "freqai_mtf_higher_low_structure",
            higher_low_structure & regime15 & regime60 & peer_ok,
            ex.ExitSpec(hold_bars=6, stop_pct=0.012, target_pct=0.030),
        ),
        (
            "freqai_mtf_triple_confluence",
            triple_confluence & regime15 & regime60 & peer_ok,
            ex.ExitSpec(hold_bars=6, stop_pct=0.015, target_pct=0.045),
        ),
        (
            "freqai_mtf_momentum_accel",
            momentum_accel & regime15 & regime60 & peer_ok & market_ok,
            ex.ExitSpec(hold_bars=5, stop_pct=0.015, target_pct=0.040),
        ),
        (
            "freqai_mtf_rsi_thrust",
            rsi_momentum_thrust & regime15 & regime60 & peer_ok & market_ok,
            ex.ExitSpec(hold_bars=5, stop_pct=0.012, target_pct=0.035),
        ),
        (
            "freqai_mtf_quiet_then_burst",
            quiet_then_burst & regime15 & peer_ok & market_ok,
            ex.ExitSpec(hold_bars=5, stop_pct=0.012, target_pct=0.030),
        ),
        (
            "freqai_mtf_adaptive_union",
            ((hot60 & momentum_breakout_trail) | (~hot60 & (higher_low_structure | momentum_accel)))
            & regime15
            & peer_ok
            & market_ok,
            ex.ExitSpec(hold_bars=6, stop_pct=0.012, target_pct=0.035),
        ),
        (
            "freqai_mtf_adaptive_union_strongtrend",
            (
                ((hot60 & momentum_breakout_trail) | (~hot60 & (higher_low_structure | momentum_accel)))
                & regime15
                & peer_ok
                & market_ok
                & (df["trend_strength"] > 0.008)
            ),
            ex.ExitSpec(hold_bars=6, stop_pct=0.012, target_pct=0.035),
        ),
        (
            "freqai_mtf_adaptive_union_strongtrend_vwapcap",
            (
                ((hot60 & momentum_breakout_trail) | (~hot60 & (higher_low_structure | momentum_accel)))
                & regime15
                & peer_ok
                & market_ok
                & (df["trend_strength"] > 0.008)
                & (df["distance_from_vwap"] <= 0.03)
            ),
            ex.ExitSpec(hold_bars=6, stop_pct=0.012, target_pct=0.035),
        ),
        (
            "adaptive_breakout_trail",
            breakout_trail & strong15 & strong60 & top2 & broad_market & rs_positive,
            ex.ExitSpec(
                hold_bars=8,
                stop_pct=0.012,
                target_pct=None,
                trail_activation_pct=0.015,
                trail_stop_pct=0.008,
            ),
        ),
        (
            "adaptive_volume_spike_leader",
            volume_spike & strong15 & strong60 & leader & broad_market & impulse_positive,
            ex.ExitSpec(hold_bars=5, stop_pct=0.010, target_pct=0.030),
        ),
        (
            "adaptive_higher_low",
            higher_low & strong15 & strong60 & top2 & broad_market & rs_positive,
            ex.ExitSpec(hold_bars=6, stop_pct=0.010, target_pct=0.028),
        ),
        (
            "adaptive_quiet_burst",
            quiet_burst & strong15 & top2 & strong_breadth & rs_positive,
            ex.ExitSpec(hold_bars=5, stop_pct=0.010, target_pct=0.028),
        ),
        (
            "adaptive_pullback_resume",
            pullback_resume & strong15 & strong60 & top2 & broad_market,
            ex.ExitSpec(hold_bars=5, stop_pct=0.009, target_pct=0.022),
        ),
        (
            "adaptive_regime_switch",
            ((hot60 & breakout_trail & top2 & broad_market) | (~hot60 & pullback_resume & top2 & strong15))
            & leader_persistent,
            ex.ExitSpec(hold_bars=6, stop_pct=0.010, target_pct=0.028),
        ),
        (
            "adaptive_consensus2",
            (consensus_count >= 2) & strong15 & strong60 & top2 & broad_market,
            ex.ExitSpec(hold_bars=6, stop_pct=0.010, target_pct=0.030),
        ),
    ]


def _evaluate_pair_interval(
    pair: str,
    interval: int,
    df: pd.DataFrame,
    recent_cutoff: int,
    side_cost_pct: float,
    min_keep_trades: int,
) -> list[dict]:
    recent_df = df.loc[df["ts"] >= recent_cutoff].reset_index(drop=True)
    full_df = df.reset_index(drop=True)
    rows: list[dict] = []

    full_candidates = build_adaptive_candidates(full_df, interval)
    recent_candidates = {name: mask for name, mask, _ in build_adaptive_candidates(recent_df, interval)}
    for name, full_mask, exit_spec in full_candidates:
        recent_mask = recent_candidates.get(name)
        if recent_mask is None:
            continue

        full_result = ex.evaluate_window(
            pair=pair,
            interval=interval,
            strategy_name=name,
            df=full_df,
            entry_mask=full_mask.fillna(False),
            exit_spec=exit_spec,
            side_cost_pct=side_cost_pct,
            window_days=int(round((full_df["ts"].iloc[-1] - full_df["ts"].iloc[0]) / 86400)),
        )
        recent_result = ex.evaluate_window(
            pair=pair,
            interval=interval,
            strategy_name=name,
            df=recent_df,
            entry_mask=recent_mask.fillna(False),
            exit_spec=exit_spec,
            side_cost_pct=side_cost_pct,
            window_days=int(round((recent_df["ts"].iloc[-1] - recent_df["ts"].iloc[0]) / 86400)) if len(recent_df) > 1 else 0,
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
                "strategy": name,
                "trades_full": full_result.trades,
                "net_full": full_result.net_pnl,
                "win_full": full_result.win_rate,
                "pf_full": full_result.profit_factor,
                "dd_full": full_result.max_drawdown,
                "mfe_full": full_result.avg_mfe,
                "trades_recent": recent_result.trades,
                "net_recent": recent_result.net_pnl,
                "win_recent": recent_result.win_rate,
                "pf_recent": recent_result.profit_factor,
                "score": score,
                "verdict": "KEEP" if keep else "KILL",
            }
        )
    return rows


def _build_frames(
    pairs: list[str],
    intervals: list[int],
    history_days: int,
    cache_dir: Path,
    trade_count: int,
    trade_pause_sec: float,
) -> dict[int, dict[str, pd.DataFrame]]:
    dense_histories: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        history = lbs.load_or_fetch_history(
            pair=pair,
            history_days=history_days,
            cache_dir=cache_dir,
            trade_count=trade_count,
            trade_pause_sec=trade_pause_sec,
        )
        dense = lbs.densify_1m(history)
        dense_histories[pair] = dense
        print(f"{pair}: sparse={len(history)} dense={len(dense)}")

    interval_frames: dict[int, dict[str, pd.DataFrame]] = {}
    for interval in intervals:
        raw_frames = {
            pair: _build_pair_frame(dense_histories[pair], pair=pair, base_interval=interval)
            for pair in pairs
        }
        interval_frames[interval] = _attach_cross_pair_context(raw_frames)
    return interval_frames


def main() -> None:
    args = parse_args()
    pairs = [pair.strip() for pair in args.pairs.split(",") if pair.strip()]
    intervals = [int(value.strip()) for value in args.intervals.split(",") if value.strip()]
    cost_scenarios = lbs.parse_cost_scenarios(args.cost_scenarios)
    cache_dir = Path(args.cache_dir)

    interval_frames = _build_frames(
        pairs=pairs,
        intervals=intervals,
        history_days=args.history_days,
        cache_dir=cache_dir,
        trade_count=args.trade_count,
        trade_pause_sec=args.trade_pause_sec,
    )
    max_ts = max(int(df["ts"].max()) for frames in interval_frames.values() for df in frames.values())
    recent_cutoff = max_ts - args.recent_days * 24 * 60 * 60

    all_frames: list[pd.DataFrame] = []
    for label, side_cost_pct in cost_scenarios:
        print(f"\n{'=' * 108}")
        print(f" ADAPTIVE SCREEN | cost_model={label} | side_cost={side_cost_pct:.4%}")
        print(f"{'=' * 108}")
        rows: list[dict] = []
        for interval, frames in interval_frames.items():
            for pair, frame in frames.items():
                rows.extend(
                    _evaluate_pair_interval(
                        pair=pair,
                        interval=interval,
                        df=frame,
                        recent_cutoff=recent_cutoff,
                        side_cost_pct=side_cost_pct,
                        min_keep_trades=args.min_keep_trades,
                    )
                )
        scenario_df = pd.DataFrame(rows)
        scenario_df["cost_model"] = label
        scenario_df["side_cost_pct"] = side_cost_pct
        scenario_df = scenario_df.sort_values(
            ["verdict", "score", "net_recent", "net_full"],
            ascending=[True, False, False, False],
        )
        all_frames.append(scenario_df)

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

    results = pd.concat(all_frames, ignore_index=True)
    results.to_csv(args.results_csv, index=False)
    print(f"\nWrote {len(results)} rows to {args.results_csv}")


if __name__ == "__main__":
    main()
