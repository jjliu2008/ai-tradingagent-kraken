from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

import kraken_client as kraken
import strategy as strat


@dataclass
class BacktestTrade:
    construction: str
    pair: str
    entry_bar: int
    entry_ts: int
    entry_price: float
    exit_bar: int
    exit_ts: int
    exit_price: float
    exit_reason: str
    pnl_pct: float
    mfe_pct: float
    bars_held: int
    signal_score: float
    trend_strength: float
    pullback_depth_pct: float
    distance_from_vwap: float
    compression_ratio: float


def _pair_key(raw: dict) -> str:
    return next(key for key in raw if key != "last")


def _fetch_history_ohlc(pair: str, interval: int) -> pd.DataFrame:
    print(f"Fetching {interval}m bars for {pair} via OHLC endpoint...")
    raw = kraken.fetch_ohlc(pair, interval=interval)
    df = strat.parse_ohlc(raw)
    if len(df) > 1:
        df = df.iloc[:-1].reset_index(drop=True)
    start = pd.Timestamp(df["ts"].iloc[0], unit="s", tz="UTC")
    end = pd.Timestamp(df["ts"].iloc[-1], unit="s", tz="UTC")
    print(f"  Bars: {len(df)} | {start} -> {end}")
    return df


def _fetch_history_trades(
    pair: str,
    interval: int,
    history_days: int,
    end_ts: int | None,
    trade_count: int,
    trade_pause_sec: float,
) -> pd.DataFrame:
    end_ts = int(time.time()) if end_ts is None else end_ts
    start_ts = end_ts - history_days * 24 * 60 * 60
    print(
        f"Fetching {interval}m bars for {pair} via trade aggregation..."
        f" {history_days}d window ending {pd.Timestamp(end_ts, unit='s', tz='UTC')}"
    )

    cursor = start_ts * 1_000_000_000
    buckets: dict[int, dict[str, float | int]] = {}
    requests_made = 0

    while True:
        raw: dict | None = None
        for attempt in range(6):
            try:
                raw = kraken.fetch_trades(pair, since=cursor, count=trade_count)
                break
            except Exception as exc:
                wait_seconds = min(10.0, 1.0 + attempt * 2.0)
                print(f"  Retry {attempt + 1}/6 at cursor {cursor}: {exc}")
                time.sleep(wait_seconds)
        if raw is None:
            raise RuntimeError(f"Failed to fetch Kraken trades for {pair} after retries")

        pair_key = _pair_key(raw)
        trades = raw[pair_key]
        next_cursor = int(raw["last"])
        requests_made += 1
        if not trades:
            break

        stop = False
        for price_s, volume_s, trade_time, *_ in trades:
            if trade_time >= end_ts:
                stop = True
                break

            bucket_ts = int(trade_time // (interval * 60) * (interval * 60))
            price = float(price_s)
            volume = float(volume_s)
            bucket = buckets.get(bucket_ts)
            if bucket is None:
                buckets[bucket_ts] = {
                    "ts": bucket_ts,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "vwap_k": price,
                    "volume": volume,
                    "count": 1,
                    "_pv": price * volume,
                }
            else:
                bucket["high"] = max(float(bucket["high"]), price)
                bucket["low"] = min(float(bucket["low"]), price)
                bucket["close"] = price
                bucket["volume"] = float(bucket["volume"]) + volume
                bucket["count"] = int(bucket["count"]) + 1
                bucket["_pv"] = float(bucket["_pv"]) + price * volume

        if requests_made % 100 == 0:
            last_trade_ts = pd.Timestamp(trades[-1][2], unit="s", tz="UTC")
            print(f"  Requests: {requests_made} | last trade seen: {last_trade_ts}")

        if stop or next_cursor <= cursor:
            break
        cursor = next_cursor
        if trade_pause_sec > 0:
            time.sleep(trade_pause_sec)

    if not buckets:
        raise RuntimeError(f"No trade history returned for {pair}")

    rows = [buckets[key] for key in sorted(buckets)]
    df = pd.DataFrame(rows)
    df["vwap_k"] = df["_pv"] / df["volume"]
    df = df[["ts", "open", "high", "low", "close", "vwap_k", "volume", "count"]]

    start = pd.Timestamp(df["ts"].iloc[0], unit="s", tz="UTC")
    end = pd.Timestamp(df["ts"].iloc[-1], unit="s", tz="UTC")
    print(f"  Bars: {len(df)} | {start} -> {end} | requests={requests_made}")
    return df


def fetch_history(
    pair: str,
    interval: int,
    history_days: int | None = None,
    end_ts: int | None = None,
    trade_count: int = 5000,
    trade_pause_sec: float = 0.8,
) -> pd.DataFrame:
    if history_days is None:
        return _fetch_history_ohlc(pair, interval)
    return _fetch_history_trades(pair, interval, history_days, end_ts, trade_count, trade_pause_sec)


def run_backtest(
    pair: str,
    df_raw: pd.DataFrame,
    config: strat.StrategyConfig,
    commission_pct: float,
    slippage_pct: float,
    construction: str = strat.DEFAULT_ENSEMBLE_CONSTRUCTION,
) -> list[BacktestTrade]:
    df = strat.build_ensemble_frame(df_raw, construction=construction, config=config)
    return run_backtest_frame(
        pair=pair,
        df=df,
        config=config,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        construction=construction,
    )


def run_backtest_frame(
    pair: str,
    df: pd.DataFrame,
    config: strat.StrategyConfig,
    commission_pct: float,
    slippage_pct: float,
    construction: str,
) -> list[BacktestTrade]:
    strategy = strat.TrendGateEnsembleStrategy(pair=pair, config=config, construction=construction)
    if df.empty:
        return []

    trades: list[BacktestTrade] = []
    pending_signal: strat.Signal | None = None

    for i in range(strategy.min_master_bars, len(df)):
        window = df.iloc[: i + 1]
        row = window.iloc[-1]
        ts = int(row["ts"])
        close_price = float(row["close"])
        high_price = float(row["high"])
        low_price = float(row["low"])

        if pending_signal is not None:
            entry_price = float(row["open"]) * (1 + commission_pct + slippage_pct)
            strategy.open_trade(
                pending_signal,
                size=1.0,
                entry_price=entry_price,
                exit_mode="standard",
                ai_confidence=0.0,
                entry_bar=i,
                entry_ts=ts,
            )
            pending_signal = None
            continue

        if strategy.trade and strategy.trade.is_open:
            strategy.trade.update_best(high_price)

            exit_reason: str | None = None
            exit_price: float | None = None
            if low_price <= strategy.trade.stop_price:
                exit_reason = "STOP_LOSS"
                exit_price = strategy.trade.stop_price * (1 - commission_pct - slippage_pct)
            elif strategy.trade.target_price is not None and high_price >= strategy.trade.target_price:
                exit_reason = "TAKE_PROFIT"
                exit_price = strategy.trade.target_price * (1 - commission_pct - slippage_pct)
            else:
                exit_reason = strategy.check_exit(window)
                if exit_reason:
                    exit_price = close_price * (1 - commission_pct - slippage_pct)

            if exit_reason and exit_price is not None:
                closed = strategy.close_trade(i, ts, exit_price, exit_reason)
                pnl_pct = (closed.realized_pnl_pct() or 0.0) - commission_pct - slippage_pct
                trades.append(
                    BacktestTrade(
                        construction=construction,
                        pair=pair,
                        entry_bar=closed.entry_bar,
                        entry_ts=closed.entry_ts,
                        entry_price=closed.entry_price,
                        exit_bar=i,
                        exit_ts=ts,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        pnl_pct=pnl_pct,
                        mfe_pct=closed.best_pct,
                        bars_held=closed.bars_held(i),
                        signal_score=closed.signal_score,
                        trend_strength=float(df.loc[closed.entry_bar, "trend_strength"]),
                        pullback_depth_pct=float(df.loc[closed.entry_bar, "pullback_depth_pct"]),
                        distance_from_vwap=float(df.loc[closed.entry_bar, "distance_from_vwap"]),
                        compression_ratio=float(df.loc[closed.entry_bar, "compression_ratio"]),
                    )
                )
            continue

        if bool(row.get("entry_signal", False)) and i + 1 < len(df):
            signal = strat.build_ensemble_signal(
                pair,
                df,
                row_idx=i,
                construction=construction,
                bar_idx=i,
            )
            if signal is not None:
                pending_signal = signal

    return trades


def _trade_sharpe(pnls: list[float]) -> float:
    if len(pnls) < 2:
        return 0.0
    arr = np.asarray(pnls)
    std = arr.std(ddof=1)
    if std == 0:
        return 0.0
    return float(arr.mean() / std * np.sqrt(len(arr)))


def _max_drawdown(pnls: list[float]) -> float:
    if not pnls:
        return 0.0
    equity = np.cumprod(1 + np.asarray(pnls, dtype=float))
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity - peaks) / peaks
    return float(drawdowns.min())


def report(all_trades: list[BacktestTrade]) -> None:
    if not all_trades:
        print("\nNo trades generated.")
        return

    pnls = [trade.pnl_pct for trade in all_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    reasons = pd.Series([trade.exit_reason for trade in all_trades]).value_counts()

    print(f"\n{'=' * 70}")
    print(f" ENSEMBLE BACKTEST | trades={len(all_trades)}")
    print(f"{'=' * 70}")
    print(f" Net PnL:      {np.sum(pnls):.3%}")
    print(f" Win rate:     {(np.mean(np.asarray(pnls) > 0)):.1%}")
    print(f" Avg win:      {np.mean(wins) if wins else 0.0:.3%}")
    print(f" Avg loss:     {np.mean(losses) if losses else 0.0:.3%}")
    print(f" Trade Sharpe: {_trade_sharpe(pnls):.2f}")
    print(f" Max drawdown: {_max_drawdown(pnls):.3%}")
    print(f" Avg hold:     {np.mean([trade.bars_held for trade in all_trades]):.2f} bars")
    print(f" Avg MFE:      {np.mean([trade.mfe_pct for trade in all_trades]):.3%}")
    print("\nExit reasons:")
    print(reasons.to_string())

    df = pd.DataFrame([asdict(trade) for trade in all_trades])
    pair_summary = (
        df.groupby("pair")
        .agg(
            trades=("pnl_pct", "count"),
            net_pnl=("pnl_pct", "sum"),
            win_rate=("pnl_pct", lambda s: float((s > 0).mean())),
            avg_pnl=("pnl_pct", "mean"),
            avg_mfe=("mfe_pct", "mean"),
        )
        .sort_values("net_pnl", ascending=False)
    )
    print("\nPair summary:")
    print(pair_summary.to_string(float_format=lambda x: f"{x:.3%}" if abs(x) < 10 else f"{x:.2f}"))

    print("\nRecent trades:")
    for trade in sorted(all_trades, key=lambda t: t.entry_ts)[-10:]:
        dt = pd.Timestamp(trade.entry_ts, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M")
        entry_fmt = f"{trade.entry_price:>10.6f}" if abs(trade.entry_price) < 1 else f"{trade.entry_price:>10.2f}"
        exit_fmt = f"{trade.exit_price:>10.6f}" if abs(trade.exit_price) < 1 else f"{trade.exit_price:>10.2f}"
        print(
            f" {dt} | {trade.pair:<7} | entry={entry_fmt} | "
            f"exit={exit_fmt} | pnl={trade.pnl_pct:+.3%} | {trade.exit_reason}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest the GIGAUSD ensemble baseline")
    parser.add_argument("--pairs", default="GIGAUSD")
    parser.add_argument("--interval", type=int, default=strat.MASTER_INTERVAL_MINUTES)
    parser.add_argument("--construction", default=strat.DEFAULT_ENSEMBLE_CONSTRUCTION)
    parser.add_argument("--history-days", type=int, help="Fetch a longer history window by aggregating Kraken public trades.")
    parser.add_argument("--end-ts", type=int, help="UTC unix timestamp marking the end of the fetched history window.")
    parser.add_argument("--trade-count", type=int, default=5000)
    parser.add_argument("--trade-pause-sec", type=float, default=0.8)
    parser.add_argument("--commission-pct", type=float, default=0.0026)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--signals-csv")
    parser.add_argument("--breakout-window", type=int, default=strat.DEFAULT_CONFIG.breakout_window)
    parser.add_argument("--min-trend-strength", type=float, default=strat.DEFAULT_CONFIG.min_trend_strength)
    parser.add_argument("--min-momentum-medium", type=float, default=strat.DEFAULT_CONFIG.min_momentum_medium)
    parser.add_argument("--min-volume-ratio", type=float, default=strat.DEFAULT_CONFIG.min_volume_ratio)
    parser.add_argument("--max-compression-ratio", type=float, default=strat.DEFAULT_CONFIG.max_compression_ratio)
    parser.add_argument("--min-close-location", type=float, default=strat.DEFAULT_CONFIG.min_close_location)
    parser.add_argument("--stop-pct", type=float, default=strat.DEFAULT_CONFIG.min_stop_pct)
    parser.add_argument("--target-pct", type=float, default=strat.DEFAULT_CONFIG.target_pct)
    parser.add_argument("--hold-bars", type=int, default=strat.DEFAULT_CONFIG.max_hold_bars)
    parser.add_argument("--fast-hold-bars", type=int, default=strat.DEFAULT_CONFIG.fast_max_hold_bars)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> strat.StrategyConfig:
    return strat.StrategyConfig(
        breakout_window=args.breakout_window,
        min_trend_strength=args.min_trend_strength,
        min_momentum_medium=args.min_momentum_medium,
        min_volume_ratio=args.min_volume_ratio,
        max_compression_ratio=args.max_compression_ratio,
        min_close_location=args.min_close_location,
        min_stop_pct=args.stop_pct,
        target_pct=args.target_pct,
        max_hold_bars=args.hold_bars,
        fast_max_hold_bars=args.fast_hold_bars,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    pairs = [pair.strip() for pair in args.pairs.split(",") if pair.strip()]

    all_trades: list[BacktestTrade] = []
    for pair in pairs:
        df = fetch_history(
            pair=pair,
            interval=args.interval,
            history_days=args.history_days,
            end_ts=args.end_ts,
            trade_count=args.trade_count,
            trade_pause_sec=args.trade_pause_sec,
        )
        trades = run_backtest(
            pair=pair,
            df_raw=df,
            config=config,
            commission_pct=args.commission_pct,
            slippage_pct=args.slippage_pct,
            construction=args.construction,
        )
        print(f"  {pair} [{args.construction}]: {len(trades)} trades")
        all_trades.extend(trades)

    report(all_trades)
    if args.signals_csv and all_trades:
        pd.DataFrame([asdict(trade) for trade in all_trades]).to_csv(args.signals_csv, index=False)
        print(f"\nWrote trades to {args.signals_csv}")


if __name__ == "__main__":
    main()
