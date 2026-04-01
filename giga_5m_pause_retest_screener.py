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
    name: str
    hold_bars: int
    stop_pct: float
    target_pct: float


@dataclass(frozen=True)
class Candidate:
    name: str
    mask: pd.Series
    anchor: str
    offset_pct: float
    ttl_bars: int
    exit_profile: ExitProfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Focused GIGA 5m pause/retest passive-fill screener"
    )
    parser.add_argument("--pair", default="GIGAUSD")
    parser.add_argument("--context-pairs", default="DOGUSD,COQUSD,HYPEUSD")
    parser.add_argument("--history-days", type=int, default=60)
    parser.add_argument("--recent-days", type=int, default=30)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--trade-count", type=int, default=5000)
    parser.add_argument("--trade-pause-sec", type=float, default=0.8)
    parser.add_argument("--side-cost-pct", type=float, default=0.0010)
    parser.add_argument("--min-keep-trades", type=int, default=6)
    parser.add_argument("--min-fill-rate", type=float, default=0.25)
    parser.add_argument("--summary-csv", default="giga_5m_pause_retest_summary.csv")
    parser.add_argument("--trades-csv", default="giga_5m_pause_retest_trades.csv")
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


def _build_frame(
    pair: str,
    context_pairs: list[str],
    history_days: int,
    cache_dir: Path,
    trade_count: int,
    trade_pause_sec: float,
) -> pd.DataFrame:
    pairs = [pair] + [p for p in context_pairs if p and p != pair]
    frames = adaptive._build_frames(
        pairs=pairs,
        intervals=[5],
        history_days=history_days,
        cache_dir=cache_dir,
        trade_count=trade_count,
        trade_pause_sec=trade_pause_sec,
    )
    return frames[5][pair].reset_index(drop=True)


def _impulse_pause_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    prev_green = df["green_bar"].shift(1).fillna(False).astype(bool)
    prev_breakout = (
        prev_green
        & (df["close"].shift(1) > df["roll_high_12"].shift(1) * 0.998)
        & (df["body_pct"].shift(1) > 0.0045)
        & (df["close_location"].shift(1) > 0.65)
        & (df["volume_ratio"].shift(1) > 1.0)
    ).fillna(False)
    prev_impulse = (
        prev_breakout
        & (df["trend_strength"].shift(1) > 0.0020)
        & (df["ctx15_trend_strength"] > 0.0004)
        & (df["ctx60_trend_strength"] > 0.0002)
        & (df["market_breadth"] >= 0.15)
    ).fillna(False)
    prev_hot = (
        prev_impulse
        & (df["momentum_medium"].shift(1) > 0.010)
        & (df["volume_ratio"].shift(1) <= 4.5)
    ).fillna(False)

    pause_small = (
        (df["body_pct"].abs() <= df["body_pct"].shift(1).abs() * 0.75)
        & ((df["high"] - df["low"]) <= (df["high"].shift(1) - df["low"].shift(1)) * 0.95)
    ).fillna(False)
    inside_pause = (
        (df["high"] <= df["high"].shift(1) * 1.001)
        & (df["low"] >= df["low"].shift(1) * 0.999)
    ).fillna(False)
    nr_pause = (
        (df["high"] - df["low"]) <= (df["high"] - df["low"]).rolling(5, min_periods=2).median()
    ).fillna(False)
    higher_low_pause = (
        (df["low"] >= df["low"].shift(1) * 0.999)
        & (df["low"] <= df["ema_fast"] * 1.004)
    ).fillna(False)
    breakout_hold = (
        (df["close"] >= df["roll_high_12"] * 0.998)
        & (df["close"] >= df["ema_fast"] * 0.999)
        & (df["distance_from_vwap"] <= 0.030)
    ).fillna(False)
    soft_resume = (
        (df["close_location"] > 0.35)
        & (df["close"] >= df["open"] * 0.998)
        & (df["volume_ratio"] >= 0.35)
        & (df["volume_ratio"] <= 2.5)
    ).fillna(False)
    not_extended = (df["distance_from_vwap"] <= 0.028).fillna(False)
    score = (
        120.0 * df["trend_strength"].clip(lower=0.0)
        + 30.0 * df["momentum_medium"].clip(lower=0.0)
        + 20.0 * df["momentum_short"].clip(lower=0.0)
        + 10.0 * df["rs_gap"].clip(lower=0.0)
        + 8.0 * df["market_breadth"].clip(lower=0.0)
        + 8.0 * (df["close_location"] - 0.5).clip(lower=0.0)
    )
    score43 = (score > 1.85).fillna(False)

    masks = {
        "inside_pause": prev_impulse & inside_pause & soft_resume & breakout_hold & not_extended,
        "inside_pause_strong": prev_hot & inside_pause & soft_resume & breakout_hold & score43,
        "nr_pause": prev_impulse & pause_small & nr_pause & soft_resume & breakout_hold,
        "higher_low_pause": prev_impulse & higher_low_pause & soft_resume & breakout_hold,
        "ema_pause": prev_impulse & pause_small & (df["low"] <= df["ema_fast"] * 1.003) & soft_resume & breakout_hold,
        "breakout_hold_pause": prev_impulse & pause_small & soft_resume & breakout_hold & not_extended,
        "hybrid_pause": prev_hot & (inside_pause | nr_pause) & higher_low_pause & soft_resume & breakout_hold,
        "score_pause": prev_impulse & pause_small & soft_resume & breakout_hold & score43,
    }
    return {name: mask.fillna(False) for name, mask in masks.items()}


def _build_candidates(df: pd.DataFrame) -> list[Candidate]:
    masks = _impulse_pause_masks(df)
    exit_profiles = (
        ExitProfile("quick", hold_bars=5, stop_pct=0.009, target_pct=0.022),
        ExitProfile("runner", hold_bars=8, stop_pct=0.010, target_pct=0.030),
        ExitProfile("tight", hold_bars=4, stop_pct=0.008, target_pct=0.018),
    )
    anchor_map = {
        "inside_pause": "pause_low",
        "inside_pause_strong": "pause_low",
        "nr_pause": "breakout_support",
        "higher_low_pause": "ema_fast",
        "ema_pause": "ema_fast",
        "breakout_hold_pause": "breakout_support",
        "hybrid_pause": "breakout_support",
        "score_pause": "pause_low",
    }
    specs: list[Candidate] = []
    for name, mask in masks.items():
        for offset_name, offset_pct, ttl_bars in (
            ("flat", 0.0, 1),
            ("5bps", 0.0005, 2),
            ("10bps", 0.0010, 2),
        ):
            for exit_profile in exit_profiles:
                specs.append(
                    Candidate(
                        name=f"{name}__{offset_name}__{exit_profile.name}",
                        mask=mask,
                        anchor=anchor_map[name],
                        offset_pct=offset_pct,
                        ttl_bars=ttl_bars,
                        exit_profile=exit_profile,
                    )
                )
    return specs


def _entry_anchor(row: pd.Series, anchor: str) -> float:
    if anchor == "pause_low":
        return float(row["low"])
    if anchor == "ema_fast":
        return float(row["ema_fast"])
    if anchor == "breakout_support":
        return max(float(row["ema_fast"]), float(row["roll_high_12"]))
    raise ValueError(f"Unsupported anchor '{anchor}'")


def _simulate(
    pair: str,
    df: pd.DataFrame,
    candidate: Candidate,
    side_cost_pct: float,
) -> tuple[list[backtest.BacktestTrade], int, int]:
    trades: list[backtest.BacktestTrade] = []
    pending: dict | None = None
    position: dict | None = None
    missed = 0
    signals = int(candidate.mask.sum())
    warmup = max(96, strat.DEFAULT_CONFIG.warmup_bars)

    for i in range(warmup, len(df)):
        row = df.iloc[i]
        ts = int(row["ts"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        close_price = float(row["close"])

        if pending is not None and position is None:
            if low_price <= pending["limit_price"]:
                entry_price = pending["limit_price"] * (1 + side_cost_pct)
                position = {
                    "entry_bar": i,
                    "entry_ts": ts,
                    "entry_price": entry_price,
                    "stop_price": entry_price * (1 - candidate.exit_profile.stop_pct),
                    "target_price": entry_price * (1 + candidate.exit_profile.target_pct),
                    "best_price": entry_price,
                    "entry_row": df.iloc[pending["signal_bar"]],
                    "signal_score": pending["signal_score"],
                }
                pending = None
            elif i > pending["expiry_bar"]:
                pending = None
                missed += 1

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
                    (close_price < float(row["ema_fast"]) and close_price < float(row["roll_high_12"]) * 0.998)
                    or (float(row["ctx15_momentum_medium"]) < -0.0015)
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

        if pending is None and position is None and i + 1 < len(df) and bool(candidate.mask.iloc[i]):
            anchor_price = _entry_anchor(row, candidate.anchor)
            next_open = float(df.iloc[i + 1]["open"])
            limit_price = min(anchor_price, next_open * (1 - candidate.offset_pct))
            if np.isfinite(limit_price) and limit_price > 0:
                pending = {
                    "limit_price": limit_price,
                    "expiry_bar": min(len(df) - 1, i + candidate.ttl_bars),
                    "signal_bar": i,
                    "signal_score": float(
                        120.0 * max(float(row["trend_strength"]), 0.0)
                        + 20.0 * max(float(row["momentum_medium"]), 0.0)
                        + 10.0 * max(float(row["rs_gap"]), 0.0)
                    ),
                }

    if pending is not None:
        missed += 1
    return trades, signals, missed


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    context_pairs = [p.strip() for p in args.context_pairs.split(",") if p.strip()]
    frame = _build_frame(
        pair=args.pair,
        context_pairs=context_pairs,
        history_days=args.history_days,
        cache_dir=cache_dir,
        trade_count=args.trade_count,
        trade_pause_sec=args.trade_pause_sec,
    )
    recent_cutoff = int(frame["ts"].max()) - args.recent_days * 24 * 60 * 60

    summary_rows: list[dict] = []
    trade_rows: list[dict] = []

    baseline_history = _load_baseline_history(args.pair, args.history_days, cache_dir)
    baseline_trades = backtest.run_backtest(
        pair=args.pair,
        df_raw=baseline_history,
        config=strat.DEFAULT_CONFIG,
        commission_pct=0.0031,
        slippage_pct=0.0,
        construction="tc15_tighter_volume_cap",
    )
    summary_rows.append(
        {
            "pair": args.pair,
            "candidate": "baseline_tc15_tighter_volume_cap",
            "signals": len(baseline_trades),
            "fills": len(baseline_trades),
            "missed_entries": 0,
            "fill_rate": 1.0 if baseline_trades else 0.0,
            **shadow_backtest._summarize_trades(baseline_trades, recent_cutoff),
            "verdict": "BASELINE",
        }
    )
    for trade in baseline_trades:
        row = asdict(trade)
        row["candidate"] = "baseline_tc15_tighter_volume_cap"
        trade_rows.append(row)

    candidates = _build_candidates(frame)
    print(f"{args.pair}: screening {len(candidates)} pause/retest variants")
    for candidate in candidates:
        trades, signals, missed = _simulate(args.pair, frame, candidate, args.side_cost_pct)
        summary = shadow_backtest._summarize_trades(trades, recent_cutoff)
        fills = len(trades)
        fill_rate = fills / signals if signals else 0.0
        keep = (
            summary["net_full"] > 0
            and summary["net_recent"] > 0
            and summary["trades_full"] >= args.min_keep_trades
            and summary["trades_recent"] >= max(3, args.min_keep_trades // 2)
            and fill_rate >= args.min_fill_rate
        )
        summary_rows.append(
            {
                "pair": args.pair,
                "candidate": candidate.name,
                "signals": signals,
                "fills": fills,
                "missed_entries": missed,
                "fill_rate": fill_rate,
                **summary,
                "verdict": "KEEP" if keep else "KILL",
            }
        )
        for trade in trades:
            row = asdict(trade)
            row["candidate"] = candidate.name
            trade_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["verdict", "net_full", "net_recent", "fill_rate", "trades_full"],
        ascending=[True, False, False, False, False],
    )
    print(f"\n{'=' * 120}")
    print(" GIGA 5M PAUSE/RETEST SUMMARY")
    print(f"{'=' * 120}")
    cols = [
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
