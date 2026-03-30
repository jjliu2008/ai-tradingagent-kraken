from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import backtest
import strategy as strat


@dataclass(frozen=True)
class ExitSpec:
    hold_bars: int
    stop_pct: float
    target_pct: float | None = None
    trail_activation_pct: float | None = None
    trail_stop_pct: float | None = None


@dataclass
class SimTrade:
    entry_ts: int
    exit_ts: int
    pnl_pct: float
    exit_reason: str
    mfe_pct: float


@dataclass
class ScreenResult:
    pair: str
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


def compute_rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rsi_7"] = compute_rsi(out["close"], 7)
    out["rsi_14"] = compute_rsi(out["close"], 14)
    out["body_pct"] = (out["close"] - out["open"]) / out["open"]
    out["ema_fast_dist"] = (out["close"] - out["ema_fast"]) / out["ema_fast"]
    out["ema_slow_dist"] = (out["close"] - out["ema_slow"]) / out["ema_slow"]
    out["roll_high_12"] = out["high"].rolling(12).max().shift(1)
    out["roll_high_20"] = out["high"].rolling(20).max().shift(1)
    out["roll_low_20"] = out["low"].rolling(20).min().shift(1)
    out["green_bar"] = out["close"] > out["open"]
    return out


def resample_ohlcv(df: pd.DataFrame, interval: int) -> pd.DataFrame:
    if interval == 15:
        return df.copy().reset_index(drop=True)

    frame = df.copy()
    frame["dt"] = pd.to_datetime(frame["ts"], unit="s", utc=True)
    frame = frame.set_index("dt")
    frame["pv"] = frame["vwap_k"] * frame["volume"]

    agg = (
        frame.resample(f"{interval}min", label="left", closed="left", origin="epoch")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "pv": "sum",
                "volume": "sum",
                "count": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
    )
    agg["vwap_k"] = agg["pv"] / agg["volume"].replace(0, np.nan)
    agg = agg.dropna(subset=["vwap_k"]).reset_index()
    agg["ts"] = agg["dt"].astype("int64") // 1_000_000_000
    return agg[["ts", "open", "high", "low", "close", "vwap_k", "volume", "count"]].reset_index(drop=True)


def load_history(
    pair: str,
    history_days: int,
    end_ts: int | None,
    cache_dir: Path | None,
    trade_count: int,
    trade_pause_sec: float,
) -> pd.DataFrame:
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        end_label = end_ts if end_ts is not None else "latest"
        cache_path = cache_dir / f"{pair}_15m_{history_days}d_end_{end_label}.csv"
        if cache_path.exists():
            print(f"Loading cached history: {cache_path}")
            return pd.read_csv(cache_path)

    df = backtest.fetch_history(
        pair=pair,
        interval=15,
        history_days=history_days,
        end_ts=end_ts,
        trade_count=trade_count,
        trade_pause_sec=trade_pause_sec,
    )
    if cache_path is not None:
        df.to_csv(cache_path, index=False)
        print(f"Saved cache: {cache_path}")
    return df


def build_candidates(df: pd.DataFrame, interval: int) -> list[tuple[str, pd.Series, ExitSpec]]:
    trend_up = df["ema_fast"] > df["ema_slow"]

    candidates: list[tuple[str, pd.Series, ExitSpec]] = []

    if interval == 15:
        candidates.extend(
            [
                (
                    "vwap_reclaim",
                    (
                        (df["trend_strength"] > 0.0010)
                        & trend_up
                        & df["support_touch_pct"].between(-0.006, 0.001)
                        & df["distance_from_vwap"].between(-0.004, 0.004)
                        & (df["close"] > df["trigger_level"])
                        & (df["close_location"] > 0.55)
                        & df["volume_ratio"].between(0.7, 1.8)
                    ),
                    ExitSpec(hold_bars=8, stop_pct=0.008, target_pct=0.018),
                ),
                (
                    "quiet_pullback",
                    (
                        (df["trend_strength"] > 0.0015)
                        & trend_up
                        & df["ema_fast_dist"].between(-0.006, 0.001)
                        & (df["momentum_short"] < 0)
                        & (df["compression_ratio"] < 1.0)
                        & (df["close_location"] > 0.60)
                        & (df["volume_ratio"] < 1.2)
                        & df["green_bar"]
                    ),
                    ExitSpec(hold_bars=8, stop_pct=0.008, target_pct=0.018),
                ),
                (
                    "oversold_snapback",
                    (
                        (df["distance_from_vwap"] < -0.015)
                        & (df["rsi_7"] < 35)
                        & df["green_bar"]
                        & (df["close_location"] > 0.70)
                        & df["volume_ratio"].between(0.7, 2.5)
                    ),
                    ExitSpec(hold_bars=4, stop_pct=0.009, target_pct=0.016),
                ),
                (
                    "higher_low_breakout",
                    (
                        (df["trend_strength"] > 0.0015)
                        & trend_up
                        & (df["low"].shift(1) < df["ema_fast"].shift(1))
                        & (df["close"] > df["roll_high_12"])
                        & (df["close_location"] > 0.65)
                        & (df["volume_ratio"] > 1.0)
                    ),
                    ExitSpec(hold_bars=6, stop_pct=0.008, target_pct=0.020),
                ),
            ]
        )

    if interval == 30:
        candidates.extend(
            [
                (
                    "trend_reclaim",
                    (
                        (df["trend_strength"] > 0.0020)
                        & trend_up
                        & (df["low"] <= df["ema_fast"])
                        & (df["close"] > df["ema_fast"])
                        & df["distance_from_vwap"].between(-0.006, 0.004)
                        & (df["momentum_short"] > -0.004)
                        & (df["close_location"] > 0.55)
                        & df["volume_ratio"].between(0.7, 1.8)
                    ),
                    ExitSpec(
                        hold_bars=6,
                        stop_pct=0.010,
                        target_pct=0.025,
                        trail_activation_pct=0.018,
                        trail_stop_pct=0.007,
                    ),
                ),
                (
                    "oversold_trend_bounce",
                    (
                        (df["trend_strength"] > 0.0005)
                        & trend_up
                        & df["distance_from_vwap"].between(-0.020, -0.006)
                        & (df["rsi_14"] < 40)
                        & df["green_bar"]
                        & (df["close_location"] > 0.65)
                        & (df["volume_ratio"] < 1.6)
                    ),
                    ExitSpec(hold_bars=6, stop_pct=0.012, target_pct=0.025),
                ),
                (
                    "compression_breakout",
                    (
                        (df["trend_strength"] > 0.0015)
                        & trend_up
                        & (df["compression_ratio"] < 0.85)
                        & (df["close"] > df["roll_high_20"])
                        & (df["momentum_medium"] > 0.010)
                        & (df["volume_ratio"] > 1.1)
                        & (df["close_location"] > 0.70)
                    ),
                    ExitSpec(hold_bars=5, stop_pct=0.010, target_pct=0.030),
                ),
            ]
        )

    if interval == 60:
        candidates.extend(
            [
                (
                    "hourly_trend_pullback",
                    (
                        (df["trend_strength"] > 0.0030)
                        & trend_up
                        & df["ema_fast_dist"].between(-0.012, 0.002)
                        & (df["momentum_short"] < 0)
                        & (df["momentum_medium"] > -0.010)
                        & (df["close_location"] > 0.55)
                        & (df["volume_ratio"] < 1.4)
                    ),
                    ExitSpec(
                        hold_bars=6,
                        stop_pct=0.015,
                        target_pct=0.040,
                        trail_activation_pct=0.025,
                        trail_stop_pct=0.012,
                    ),
                ),
                (
                    "hourly_vwap_reclaim",
                    (
                        (df["trend_strength"] > 0.0020)
                        & trend_up
                        & (df["low_vs_vwap"] <= -0.010)
                        & (df["close"] > df["session_vwap"])
                        & df["green_bar"]
                        & (df["close_location"] > 0.65)
                        & df["volume_ratio"].between(0.8, 1.8)
                    ),
                    ExitSpec(hold_bars=5, stop_pct=0.015, target_pct=0.035),
                ),
                (
                    "hourly_mean_reversion",
                    (
                        (df["distance_from_vwap"] < -0.030)
                        & (df["rsi_14"] < 35)
                        & df["green_bar"]
                        & (df["close_location"] > 0.75)
                    ),
                    ExitSpec(hold_bars=4, stop_pct=0.020, target_pct=0.050),
                ),
                (
                    "hourly_breakout",
                    (
                        (df["trend_strength"] > 0.0020)
                        & trend_up
                        & (df["compression_ratio"] < 0.90)
                        & (df["close"] > df["roll_high_12"])
                        & (df["momentum_medium"] > 0.015)
                        & (df["volume_ratio"] > 1.0)
                        & (df["close_location"] > 0.70)
                    ),
                    ExitSpec(hold_bars=5, stop_pct=0.015, target_pct=0.045),
                ),
            ]
        )

    return candidates


def simulate_trades(
    df: pd.DataFrame,
    entry_mask: pd.Series,
    exit_spec: ExitSpec,
    side_cost_pct: float,
) -> list[SimTrade]:
    trades: list[SimTrade] = []
    i = 0
    last_index = len(df) - 1

    while i < last_index:
        if not bool(entry_mask.iloc[i]):
            i += 1
            continue

        entry_bar = i + 1
        if entry_bar > last_index:
            break

        entry_ts = int(df.iloc[entry_bar]["ts"])
        entry_price = float(df.iloc[entry_bar]["open"]) * (1 + side_cost_pct)
        stop_price = entry_price * (1 - exit_spec.stop_pct)
        target_price = entry_price * (1 + exit_spec.target_pct) if exit_spec.target_pct is not None else None
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
            SimTrade(
                entry_ts=entry_ts,
                exit_ts=int(df.iloc[exit_bar]["ts"]),
                pnl_pct=exit_price / entry_price - 1,
                exit_reason=exit_reason,
                mfe_pct=best_high / entry_price - 1,
            )
        )
        i = exit_bar + 1

    return trades


def max_drawdown(pnls: list[float]) -> float:
    if not pnls:
        return 0.0
    equity = np.cumprod(1 + np.asarray(pnls, dtype=float))
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity - peaks) / peaks
    return float(drawdowns.min())


def profit_factor(pnls: list[float]) -> float:
    wins = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p < 0)
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def evaluate_window(
    pair: str,
    interval: int,
    strategy_name: str,
    df: pd.DataFrame,
    entry_mask: pd.Series,
    exit_spec: ExitSpec,
    side_cost_pct: float,
    window_days: int,
) -> ScreenResult:
    trades = simulate_trades(df, entry_mask, exit_spec, side_cost_pct)
    pnls = [trade.pnl_pct for trade in trades]
    return ScreenResult(
        pair=pair,
        interval=interval,
        strategy=strategy_name,
        window_days=window_days,
        trades=len(trades),
        net_pnl=float(np.sum(pnls)) if pnls else 0.0,
        avg_pnl=float(np.mean(pnls)) if pnls else 0.0,
        win_rate=float(np.mean(np.asarray(pnls) > 0)) if pnls else 0.0,
        profit_factor=profit_factor(pnls),
        max_drawdown=max_drawdown(pnls),
        avg_mfe=float(np.mean([trade.mfe_pct for trade in trades])) if trades else 0.0,
    )


def screen_pair(
    pair: str,
    base_history: pd.DataFrame,
    intervals: list[int],
    side_cost_pct: float,
    recent_days: int,
    min_keep_trades: int,
) -> pd.DataFrame:
    end_ts = int(base_history["ts"].iloc[-1])
    recent_cutoff = end_ts - recent_days * 24 * 60 * 60
    rows: list[dict] = []

    for interval in intervals:
        interval_df = resample_ohlcv(base_history, interval)
        interval_df = add_extra_features(strat.compute_features(interval_df))
        full_df = interval_df.reset_index(drop=True)
        recent_df = interval_df.loc[interval_df["ts"] >= recent_cutoff].reset_index(drop=True)

        for strategy_name, full_mask, exit_spec in build_candidates(full_df, interval):
            recent_mask = build_candidates(recent_df, interval)
            recent_mask = next(mask for name, mask, _ in recent_mask if name == strategy_name)

            full_result = evaluate_window(
                pair=pair,
                interval=interval,
                strategy_name=strategy_name,
                df=full_df,
                entry_mask=full_mask.fillna(False),
                exit_spec=exit_spec,
                side_cost_pct=side_cost_pct,
                window_days=int(round((full_df["ts"].iloc[-1] - full_df["ts"].iloc[0]) / 86400)),
            )
            recent_result = evaluate_window(
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
                and recent_result.trades >= max(2, min_keep_trades // 2)
            )

            rows.append(
                {
                    "pair": pair,
                    "interval": interval,
                    "strategy": strategy_name,
                    "trades_120d": full_result.trades,
                    "net_120d": full_result.net_pnl,
                    "avg_120d": full_result.avg_pnl,
                    "win_120d": full_result.win_rate,
                    "pf_120d": full_result.profit_factor,
                    "dd_120d": full_result.max_drawdown,
                    "mfe_120d": full_result.avg_mfe,
                    "trades_60d": recent_result.trades,
                    "net_60d": recent_result.net_pnl,
                    "avg_60d": recent_result.avg_pnl,
                    "win_60d": recent_result.win_rate,
                    "pf_60d": recent_result.profit_factor,
                    "dd_60d": recent_result.max_drawdown,
                    "mfe_60d": recent_result.avg_mfe,
                    "score": full_result.net_pnl + recent_result.net_pnl,
                    "verdict": "KEEP" if keep else "KILL",
                }
            )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Holdout screener for fresh spot-long Kraken strategy families")
    parser.add_argument("--pairs", default="SOLUSD")
    parser.add_argument("--intervals", default="15,30,60")
    parser.add_argument("--history-days", type=int, default=120)
    parser.add_argument("--recent-days", type=int, default=60)
    parser.add_argument("--end-ts", type=int)
    parser.add_argument("--fee-pct", type=float, default=0.0026)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--trade-count", type=int, default=5000)
    parser.add_argument("--trade-pause-sec", type=float, default=0.0)
    parser.add_argument("--min-keep-trades", type=int, default=6)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--results-csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = [pair.strip() for pair in args.pairs.split(",") if pair.strip()]
    intervals = [int(value.strip()) for value in args.intervals.split(",") if value.strip()]
    side_cost_pct = args.fee_pct + args.slippage_pct
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    frames: list[pd.DataFrame] = []
    for pair in pairs:
        history = load_history(
            pair=pair,
            history_days=args.history_days,
            end_ts=args.end_ts,
            cache_dir=cache_dir,
            trade_count=args.trade_count,
            trade_pause_sec=args.trade_pause_sec,
        )
        frames.append(
            screen_pair(
                pair=pair,
                base_history=history,
                intervals=intervals,
                side_cost_pct=side_cost_pct,
                recent_days=args.recent_days,
                min_keep_trades=args.min_keep_trades,
            )
        )

    results = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if results.empty:
        print("No strategy families produced any trades.")
        return

    results = results.sort_values(["verdict", "score", "net_60d", "net_120d"], ascending=[True, False, False, False])
    keepers = results.loc[results["verdict"] == "KEEP"].copy()

    print("\nTop screen results:")
    print(
        results.head(15).to_string(
            index=False,
            float_format=lambda value: f"{value:+.3%}" if np.isfinite(value) and abs(value) < 10 else f"{value:.2f}",
        )
    )

    if keepers.empty:
        print("\nNo survivors met the 60d + 120d keep rule.")
    else:
        print("\nSurvivors:")
        print(
            keepers.to_string(
                index=False,
                float_format=lambda value: f"{value:+.3%}" if np.isfinite(value) and abs(value) < 10 else f"{value:.2f}",
            )
        )
        best = keepers.iloc[0]
        print(
            "\nRecommended baseline: "
            f"{best['pair']} {int(best['interval'])}m {best['strategy']} "
            f"| net_120d={best['net_120d']:+.3%} | net_60d={best['net_60d']:+.3%}"
        )

    if args.results_csv:
        results.to_csv(args.results_csv, index=False)
        print(f"\nWrote screen results to {args.results_csv}")


if __name__ == "__main__":
    main()
