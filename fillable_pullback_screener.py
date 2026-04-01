from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import adaptive_strategy_screener as adaptive
import backtest
import shadow_backtest
import strategy as strat


@dataclass(frozen=True)
class ExitProfile:
    hold_bars: int
    stop_pct: float
    target_pct: float


@dataclass(frozen=True)
class PullbackCandidate:
    name: str
    mask: pd.Series
    anchor: str
    offset_pct: float
    ttl_bars: int
    exit_profile: ExitProfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen 5m pullback-after-impulse strategies with passive-fill realism"
    )
    parser.add_argument("--pairs", default="GIGAUSD,DOGUSD,COQUSD,HYPEUSD")
    parser.add_argument("--history-days", type=int, default=60)
    parser.add_argument("--recent-days", type=int, default=30)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--trade-count", type=int, default=5000)
    parser.add_argument("--trade-pause-sec", type=float, default=0.8)
    parser.add_argument("--side-cost-pct", type=float, default=0.0010)
    parser.add_argument("--min-keep-trades", type=int, default=8)
    parser.add_argument("--min-fill-rate", type=float, default=0.20)
    parser.add_argument("--summary-csv", default="fillable_pullback_summary.csv")
    parser.add_argument("--trades-csv", default="fillable_pullback_trades.csv")
    return parser.parse_args()


def _load_baseline_history(pair: str, history_days: int, cache_dir: Path) -> pd.DataFrame:
    path = cache_dir / f"{pair}_15m_120d_end_latest.csv"
    if path.exists():
        df = pd.read_csv(path).sort_values("ts").reset_index(drop=True)
        cutoff = int(df["ts"].max()) - history_days * 24 * 60 * 60
        return df.loc[df["ts"] >= cutoff].reset_index(drop=True)
    return backtest.fetch_history(
        pair=pair,
        interval=15,
        history_days=history_days,
        trade_pause_sec=0.0,
    )


def _build_frames(
    pairs: list[str],
    history_days: int,
    cache_dir: Path,
    trade_count: int,
    trade_pause_sec: float,
) -> dict[str, pd.DataFrame]:
    frames = adaptive._build_frames(
        pairs=pairs,
        intervals=[5],
        history_days=history_days,
        cache_dir=cache_dir,
        trade_count=trade_count,
        trade_pause_sec=trade_pause_sec,
    )
    return frames[5]


def _candidate_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    ctx15_up = (df["ctx15_ema_fast"] > df["ctx15_ema_slow"]).fillna(False)
    ctx60_up = (df["ctx60_ema_fast"] > df["ctx60_ema_slow"]).fillna(False)
    peer_ok = ((df["rs_rank"] <= 3) | (df["rs_gap"] > -0.12)).fillna(False)
    regime = (
        ctx15_up
        & ctx60_up
        & (df["ctx15_trend_strength"] > 0.0004)
        & (df["ctx60_trend_strength"] > 0.0002)
        & (df["market_breadth"] >= 0.15)
        & peer_ok
    ).fillna(False)

    prev_green = df["green_bar"].shift(1).fillna(False).astype(bool)
    prior_impulse = (
        prev_green
        & (
            (df["momentum_short"].shift(1) > 0.004)
            | (df["momentum_medium"].shift(1) > 0.010)
            | (df["close"].shift(1) > (df["roll_high_12"].shift(1) * 0.998))
        )
        & (df["close_location"].shift(1) > 0.55)
        & (df["volume_ratio"].shift(1) > 0.8)
    ).fillna(False)
    hot_impulse = (
        prior_impulse
        & (df["body_pct"].shift(1) > 0.0045)
        & (df["volume_ratio"].shift(1) <= 4.0)
    ).fillna(False)
    orderly_pullback = (
        (df["volume_ratio"] >= 0.3)
        & (df["volume_ratio"] <= 3.0)
        & df["distance_from_vwap"].between(-0.010, 0.030)
        & (df["close_location"] > 0.30)
        & (df["compression_ratio"] < 1.30)
    ).fillna(False)

    touch_ema = (df["low"] <= df["ema_fast"] * 1.0030).fillna(False)
    touch_vwap = (df["low"] <= df["session_vwap"] * 1.0030).fillna(False)
    retest_breakout = (df["low"] <= df["roll_high_12"] * 1.0040).fillna(False)
    inside_bar = ((df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))).fillna(False)
    soft_resume = ((df["close"] >= df["ema_fast"] * 0.999) & (df["close_location"] > 0.35)).fillna(False)
    green_resume = (df["green_bar"] & (df["close"] > df["open"]) & (df["close_location"] > 0.55)).fillna(False)
    not_extended = (df["distance_from_vwap"] <= 0.035).fillna(False)
    stronger_trend = (df["trend_strength"] > 0.0025).fillna(False)
    score = (
        100.0 * df["trend_strength"].clip(lower=0.0)
        + 35.0 * df["momentum_medium"].clip(lower=0.0)
        + 15.0 * (df["close_location"] - 0.5).clip(lower=0.0)
        + 10.0 * (df["volume_ratio"] - 0.8).clip(lower=0.0)
        + 8.0 * df["rs_gap"].clip(lower=0.0)
        + 6.0 * df["market_breadth"].clip(lower=0.0)
    )
    score_ok = (score > 1.65).fillna(False)

    masks = {
        "ema_reclaim": regime & prior_impulse & orderly_pullback & touch_ema & soft_resume & not_extended,
        "ema_reclaim_strong": regime & hot_impulse & orderly_pullback & touch_ema & green_resume & stronger_trend,
        "vwap_reclaim": regime & prior_impulse & orderly_pullback & touch_vwap & soft_resume,
        "breakout_retest": regime & prior_impulse & orderly_pullback & retest_breakout & soft_resume,
        "inside_resume": regime & prior_impulse & orderly_pullback & inside_bar & touch_ema & soft_resume,
        "hybrid_support": regime
        & prior_impulse
        & orderly_pullback
        & ((touch_ema & touch_vwap) | (touch_ema & retest_breakout))
        & soft_resume,
        "score_filtered_ema": regime & prior_impulse & orderly_pullback & touch_ema & soft_resume & score_ok,
        "score_filtered_breakout": regime & prior_impulse & orderly_pullback & retest_breakout & soft_resume & score_ok,
    }
    return {name: mask.fillna(False) for name, mask in masks.items()}


def _build_candidates(df: pd.DataFrame) -> list[PullbackCandidate]:
    masks = _candidate_masks(df)
    exit_profiles = {
        "quick": ExitProfile(hold_bars=6, stop_pct=0.009, target_pct=0.022),
        "runner": ExitProfile(hold_bars=8, stop_pct=0.010, target_pct=0.030),
    }
    specs: list[PullbackCandidate] = []
    for base_name, anchor in (
        ("ema_reclaim", "ema_fast"),
        ("ema_reclaim_strong", "ema_fast"),
        ("vwap_reclaim", "session_vwap"),
        ("breakout_retest", "roll_high_12"),
        ("inside_resume", "ema_fast"),
        ("hybrid_support", "hybrid_support"),
        ("score_filtered_ema", "ema_fast"),
        ("score_filtered_breakout", "roll_high_12"),
    ):
        for offset_name, offset_pct, ttl_bars in (
            ("flat", 0.0, 1),
            ("5bps", 0.0005, 2),
            ("10bps", 0.0010, 2),
        ):
            for exit_name, exit_profile in exit_profiles.items():
                specs.append(
                    PullbackCandidate(
                        name=f"{base_name}__{offset_name}__{exit_name}",
                        mask=masks[base_name],
                        anchor=anchor,
                        offset_pct=offset_pct,
                        ttl_bars=ttl_bars,
                        exit_profile=exit_profile,
                    )
                )
    return specs


def _entry_anchor(row: pd.Series, anchor: str) -> float:
    if anchor == "ema_fast":
        return float(row["ema_fast"])
    if anchor == "session_vwap":
        return float(row["session_vwap"])
    if anchor == "roll_high_12":
        return float(row["roll_high_12"])
    if anchor == "hybrid_support":
        levels = [
            float(row["ema_fast"]),
            float(row["session_vwap"]),
            float(row["roll_high_12"]),
        ]
        return max(level for level in levels if np.isfinite(level))
    raise ValueError(f"Unsupported anchor '{anchor}'")


def _simulate_candidate(
    pair: str,
    df: pd.DataFrame,
    candidate: PullbackCandidate,
    side_cost_pct: float,
) -> tuple[list[backtest.BacktestTrade], int, int]:
    trades: list[backtest.BacktestTrade] = []
    pending: dict | None = None
    position: dict | None = None
    missed_entries = 0
    total_signals = int(candidate.mask.sum())
    warmup = max(96, strat.DEFAULT_CONFIG.warmup_bars)

    for i in range(warmup, len(df)):
        row = df.iloc[i]
        ts = int(row["ts"])
        open_price = float(row["open"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        close_price = float(row["close"])

        if pending is not None and position is None:
            if low_price <= pending["limit_price"]:
                entry_bar = i
                entry_price = pending["limit_price"] * (1 + side_cost_pct)
                stop_price = entry_price * (1 - candidate.exit_profile.stop_pct)
                target_price = entry_price * (1 + candidate.exit_profile.target_pct)
                position = {
                    "entry_bar": entry_bar,
                    "entry_ts": ts,
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "best_price": entry_price,
                    "signal_score": float(pending["signal_score"]),
                    "entry_row": df.iloc[pending["signal_bar"]],
                }
                pending = None
            elif i > pending["expiry_bar"]:
                pending = None
                missed_entries += 1

        if position is not None:
            position["best_price"] = max(float(position["best_price"]), high_price)
            exit_reason: str | None = None
            exit_price: float | None = None

            if low_price <= float(position["stop_price"]):
                exit_reason = "STOP_LOSS"
                exit_price = float(position["stop_price"]) * (1 - side_cost_pct)
            elif high_price >= float(position["target_price"]):
                exit_reason = "TAKE_PROFIT"
                exit_price = float(position["target_price"]) * (1 - side_cost_pct)
            else:
                bars_held = i - int(position["entry_bar"])
                trend_lost = (
                    (close_price < float(row["ema_fast"]) and close_price < float(row["session_vwap"]))
                    or (float(row["ctx15_momentum_medium"]) < -0.001 and close_price < open_price)
                )
                if trend_lost:
                    exit_reason = "TREND_LOST"
                    exit_price = close_price * (1 - side_cost_pct)
                elif bars_held >= candidate.exit_profile.hold_bars:
                    exit_reason = "TIME_LIMIT"
                    exit_price = close_price * (1 - side_cost_pct)

            if exit_reason and exit_price is not None:
                entry_price = float(position["entry_price"])
                pnl_pct = (exit_price - entry_price) / entry_price - side_cost_pct
                best_price = float(position["best_price"])
                mfe_pct = (best_price - entry_price) / entry_price
                entry_row = position["entry_row"]
                trades.append(
                    backtest.BacktestTrade(
                        construction=candidate.name,
                        pair=pair,
                        entry_bar=int(position["entry_bar"]),
                        entry_ts=int(position["entry_ts"]),
                        entry_price=entry_price,
                        exit_bar=i,
                        exit_ts=ts,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        pnl_pct=pnl_pct,
                        mfe_pct=mfe_pct,
                        bars_held=i - int(position["entry_bar"]),
                        signal_score=float(position["signal_score"]),
                        trend_strength=float(entry_row["trend_strength"]),
                        pullback_depth_pct=float(entry_row["pullback_depth_pct"]),
                        distance_from_vwap=float(entry_row["distance_from_vwap"]),
                        compression_ratio=float(entry_row["compression_ratio"]),
                    )
                )
                position = None
                continue

        if i + 1 >= len(df) or pending is not None or position is not None:
            continue

        if bool(candidate.mask.iloc[i]):
            next_open = float(df.iloc[i + 1]["open"])
            anchor_price = _entry_anchor(row, candidate.anchor)
            if not np.isfinite(anchor_price) or anchor_price <= 0:
                continue
            limit_price = min(next_open * (1 - candidate.offset_pct), anchor_price)
            if not np.isfinite(limit_price) or limit_price <= 0:
                continue
            pending = {
                "limit_price": limit_price,
                "expiry_bar": min(len(df) - 1, i + candidate.ttl_bars),
                "signal_score": float(
                    100.0 * max(float(row["trend_strength"]), 0.0)
                    + 30.0 * max(float(row["momentum_medium"]), 0.0)
                    + 10.0 * max(float(row["rs_gap"]), 0.0)
                ),
                "signal_bar": i,
            }

    if pending is not None:
        missed_entries += 1

    return trades, total_signals, missed_entries


def _baseline_summary(pair: str, history_days: int, cache_dir: Path, recent_cutoff: int) -> tuple[dict, list[dict]]:
    history = _load_baseline_history(pair, history_days, cache_dir)
    trades = backtest.run_backtest(
        pair=pair,
        df_raw=history,
        config=strat.DEFAULT_CONFIG,
        commission_pct=0.0031,
        slippage_pct=0.0,
        construction="tc15_tighter_volume_cap",
    )
    summary = {
        "pair": pair,
        "candidate": "baseline_tc15_tighter_volume_cap",
        "signals": len(trades),
        "fills": len(trades),
        "missed_entries": 0,
        "fill_rate": 1.0 if trades else 0.0,
        **shadow_backtest._summarize_trades(trades, recent_cutoff),
        "verdict": "BASELINE",
        "cost_model": "baseline_taker_ref",
    }
    trade_rows: list[dict] = []
    for trade in trades:
        row = asdict(trade)
        row["candidate"] = "baseline_tc15_tighter_volume_cap"
        row["cost_model"] = "baseline_taker_ref"
        trade_rows.append(row)
    return summary, trade_rows


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    pairs = [pair.strip() for pair in args.pairs.split(",") if pair.strip()]

    frames = _build_frames(
        pairs=pairs,
        history_days=args.history_days,
        cache_dir=cache_dir,
        trade_count=args.trade_count,
        trade_pause_sec=args.trade_pause_sec,
    )
    recent_cutoff = max(int(frame["ts"].max()) for frame in frames.values()) - args.recent_days * 24 * 60 * 60

    summary_rows: list[dict] = []
    trade_rows: list[dict] = []

    baseline_pair = "GIGAUSD" if "GIGAUSD" in pairs else pairs[0]
    baseline_summary, baseline_trades = _baseline_summary(
        pair=baseline_pair,
        history_days=args.history_days,
        cache_dir=cache_dir,
        recent_cutoff=recent_cutoff,
    )
    summary_rows.append(baseline_summary)
    trade_rows.extend(baseline_trades)

    for pair in pairs:
        df = frames[pair].reset_index(drop=True)
        candidates = _build_candidates(df)
        print(f"\n{pair}: screening {len(candidates)} passive pullback variants")
        for candidate in candidates:
            trades, signals, missed_entries = _simulate_candidate(
                pair=pair,
                df=df,
                candidate=candidate,
                side_cost_pct=args.side_cost_pct,
            )
            summary = shadow_backtest._summarize_trades(trades, recent_cutoff)
            fills = len(trades)
            fill_rate = fills / signals if signals else 0.0
            keep = (
                summary["net_full"] > 0
                and summary["net_recent"] > 0
                and summary["trades_full"] >= args.min_keep_trades
                and summary["trades_recent"] >= max(4, args.min_keep_trades // 2)
                and fill_rate >= args.min_fill_rate
            )
            summary_rows.append(
                {
                    "pair": pair,
                    "candidate": candidate.name,
                    "signals": signals,
                    "fills": fills,
                    "missed_entries": missed_entries,
                    "fill_rate": fill_rate,
                    **summary,
                    "verdict": "KEEP" if keep else "KILL",
                    "cost_model": f"maker_mid_passive_{args.side_cost_pct:.4f}",
                }
            )
            for trade in trades:
                row = asdict(trade)
                row["candidate"] = candidate.name
                row["cost_model"] = f"maker_mid_passive_{args.side_cost_pct:.4f}"
                trade_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["verdict", "net_full", "net_recent", "fill_rate", "trades_full"],
        ascending=[True, False, False, False, False],
    )
    print(f"\n{'=' * 132}")
    print(" FILLABLE PULLBACK SUMMARY")
    print(f"{'=' * 132}")
    cols = [
        "pair",
        "candidate",
        "signals",
        "fills",
        "fill_rate",
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
        "fill_rate": lambda value: f"{value:.1%}",
        "net_full": lambda value: f"{value:+.3%}",
        "win_full": lambda value: f"{value:.1%}",
        "max_dd": lambda value: f"{value:.3%}",
        "net_recent": lambda value: f"{value:+.3%}",
        "win_recent": lambda value: f"{value:.1%}",
    }
    print(summary_df[cols].head(20).to_string(index=False, formatters=formatters))

    summary_df.to_csv(args.summary_csv, index=False)
    print(f"\nWrote summary to {args.summary_csv}")
    if trade_rows:
        pd.DataFrame(trade_rows).to_csv(args.trades_csv, index=False)
        print(f"Wrote trades to {args.trades_csv}")


if __name__ == "__main__":
    main()
