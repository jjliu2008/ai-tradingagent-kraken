from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import backtest
import kraken_client as kraken
import shadow_backtest
import strategy as strat


BANNED_BASES = {
    "PAXG",
    "XAUT",
    "USDT",
    "USDC",
    "EUR",
    "GBP",
    "AUD",
    "CAD",
    "CHF",
    "JPY",
}


@dataclass(frozen=True)
class PortfolioSpec:
    name: str
    construction: str
    max_positions: int
    min_breadth: float
    max_rs_rank: int
    min_signal_score: float
    require_gate: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen liquid Kraken spot universe with cross-sectional 15m/30m/60m portfolio selectors"
    )
    parser.add_argument("--history-days", type=int, default=60)
    parser.add_argument("--recent-days", type=int, default=30)
    parser.add_argument("--pairs", help="Optional comma-separated pair override. If set, skip live universe discovery.")
    parser.add_argument("--universe-size", type=int, default=12)
    parser.add_argument("--min-quote-volume", type=float, default=500000.0)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--trade-count", type=int, default=5000)
    parser.add_argument("--trade-pause-sec", type=float, default=0.2)
    parser.add_argument("--commission-pct", type=float, default=0.0026)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--summary-csv", default="cross_sectional_universe_summary.csv")
    parser.add_argument("--trades-csv", default="cross_sectional_universe_trades.csv")
    parser.add_argument("--universe-csv", default="cross_sectional_universe_pairs.csv")
    return parser.parse_args()


def _fetch_liquid_usd_spot_universe(limit: int, min_quote_volume: float) -> pd.DataFrame:
    pairs = kraken.fetch_asset_pairs("info")
    metas: dict[str, dict] = {}
    altnames: list[str] = []

    for key, info in pairs.items():
        alt = str(info.get("altname") or key)
        status = str(info.get("status") or "")
        if status != "online":
            continue
        if not alt.endswith("USD"):
            continue
        if ".M" in key or ".d" in key.lower():
            continue
        base = alt[:-3]
        if base in BANNED_BASES:
            continue
        if any(base.endswith(suffix) for suffix in ("2L", "2S", "3L", "3S")):
            continue
        metas[key] = {
            "pair_key": key,
            "pair": alt,
            "wsname": info.get("wsname"),
            "base": base,
        }
        altnames.append(alt)

    rows: list[dict] = []
    for start in range(0, len(altnames), 20):
        chunk = altnames[start : start + 20]
        raw = kraken.fetch_public_ticker(chunk)
        for key, payload in raw.items():
            meta = metas.get(key)
            if meta is None:
                continue
            try:
                base_volume = float(payload["v"][1])
                vwap_24h = float(payload["p"][1])
                last_price = float(payload["c"][0])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            quote_volume = base_volume * vwap_24h
            if quote_volume < min_quote_volume:
                continue
            rows.append(
                {
                    **meta,
                    "base_volume_24h": base_volume,
                    "vwap_24h": vwap_24h,
                    "last_price": last_price,
                    "quote_volume_24h": quote_volume,
                }
            )

    universe = pd.DataFrame(rows).sort_values("quote_volume_24h", ascending=False).reset_index(drop=True)
    return universe.head(limit)


def _history_cache_path(cache_dir: Path, pair: str, history_days: int) -> Path:
    return cache_dir / f"{pair}_15m_{history_days}d_end_latest.csv"


def _load_or_fetch_history(
    pair: str,
    history_days: int,
    cache_dir: Path,
    trade_count: int,
    trade_pause_sec: float,
) -> pd.DataFrame:
    path = _history_cache_path(cache_dir, pair, history_days)
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


def _attach_cross_sectional_context(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for pair, frame in frames.items():
        view = frame[["ts", "trend_strength", "momentum_medium", "atr_pct", "signal_score"]].copy()
        view["pair"] = pair
        view["rs_score"] = view["momentum_medium"] / view["atr_pct"].clip(lower=1e-6)
        rows.append(view)

    universe = pd.concat(rows, ignore_index=True)
    universe["cs_rs_rank"] = universe.groupby("ts")["rs_score"].rank(method="dense", ascending=False)
    universe["cs_signal_rank"] = universe.groupby("ts")["signal_score"].rank(method="dense", ascending=False)
    breadth = universe.groupby("ts").agg(
        cs_breadth=("momentum_medium", lambda s: float((s > 0).mean())),
        cs_strong_breadth=("trend_strength", lambda s: float((s > 0.0015).mean())),
        cs_mean_rs=("rs_score", "mean"),
    )
    universe = universe.join(breadth, on="ts")
    universe["cs_rs_gap"] = universe["rs_score"] - universe["cs_mean_rs"]

    enriched: dict[str, pd.DataFrame] = {}
    for pair, frame in frames.items():
        ctx = (
            universe.loc[universe["pair"] == pair, ["ts", "cs_rs_rank", "cs_signal_rank", "cs_breadth", "cs_strong_breadth", "cs_mean_rs", "cs_rs_gap"]]
            .reset_index(drop=True)
        )
        enriched[pair] = pd.concat([frame.reset_index(drop=True), ctx.drop(columns=["ts"])], axis=1)
    return enriched


def _portfolio_specs() -> list[PortfolioSpec]:
    return [
        PortfolioSpec("cs_tc15_cap_top2", "tc15_tighter_volume_cap", 1, 0.25, 2, 20.0, False),
        PortfolioSpec("cs_tc15_cap_top3", "tc15_tighter_volume_cap", 1, 0.20, 3, 18.0, False),
        PortfolioSpec("cs_tc15_cap_or_mb60_top3", "tc15_cap_or_mb60", 1, 0.20, 3, 18.0, False),
        PortfolioSpec("cs_baseline_or_tc15_top3", "baseline_or_tc15", 1, 0.20, 3, 18.0, False),
        PortfolioSpec("cs_union_strong60_top3", "union_strong60", 1, 0.20, 3, 18.0, True),
        PortfolioSpec("cs_baseline_mb60_top2", "baseline_mb60", 1, 0.20, 2, 14.0, False),
    ]


def _candidate_score(row: pd.Series) -> float:
    rs_rank = float(row.get("cs_rs_rank", 99.0))
    breadth = float(row.get("cs_breadth", 0.0))
    rs_gap = float(row.get("cs_rs_gap", 0.0))
    gate = float(row.get("gate_trend_strength_60", 0.0))
    return (
        float(row.get("signal_score", 0.0))
        + 8.0 * max(0.0, 4.0 - rs_rank)
        + 18.0 * max(0.0, rs_gap)
        + 10.0 * max(0.0, breadth - 0.2)
        + 2200.0 * max(0.0, gate)
    )


def _run_portfolio_backtest(
    frames: dict[str, pd.DataFrame],
    spec: PortfolioSpec,
    commission_pct: float,
    slippage_pct: float,
) -> list[backtest.BacktestTrade]:
    timestamps = sorted({int(ts) for frame in frames.values() for ts in frame["ts"].astype(int).tolist()})
    by_ts: dict[str, dict[int, int]] = {
        pair: {int(ts): idx for idx, ts in enumerate(frame["ts"].astype(int).tolist())}
        for pair, frame in frames.items()
    }
    strategies = {
        pair: strat.TrendGateEnsembleStrategy(pair=pair, config=strat.DEFAULT_CONFIG, construction=spec.construction)
        for pair in frames
    }
    trades: list[backtest.BacktestTrade] = []
    pending: dict | None = None
    open_pair: str | None = None

    for ts in timestamps:
        if pending is not None and ts >= pending["entry_ts"]:
            pair = pending["pair"]
            frame = frames[pair]
            idx = by_ts[pair].get(ts)
            if idx is not None:
                row = frame.iloc[idx]
                entry_price = float(row["open"]) * (1 + commission_pct + slippage_pct)
                strategies[pair].open_trade(
                    pending["signal"],
                    size=1.0,
                    entry_price=entry_price,
                    exit_mode="standard",
                    ai_confidence=0.0,
                    entry_bar=idx,
                    entry_ts=ts,
                )
                open_pair = pair
                pending = None

        if open_pair is not None:
            frame = frames[open_pair]
            idx = by_ts[open_pair].get(ts)
            if idx is not None:
                strategy = strategies[open_pair]
                row = frame.iloc[idx]
                high_price = float(row["high"])
                low_price = float(row["low"])
                close_price = float(row["close"])
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
                    window = frame.iloc[: idx + 1]
                    exit_reason = strategy.check_exit(window)
                    if exit_reason:
                        exit_price = close_price * (1 - commission_pct - slippage_pct)

                if exit_reason and exit_price is not None:
                    closed = strategy.close_trade(idx, ts, exit_price, exit_reason)
                    pnl_pct = (closed.realized_pnl_pct() or 0.0) - commission_pct - slippage_pct
                    trades.append(
                        backtest.BacktestTrade(
                            construction=spec.name,
                            pair=open_pair,
                            entry_bar=closed.entry_bar,
                            entry_ts=closed.entry_ts,
                            entry_price=closed.entry_price,
                            exit_bar=idx,
                            exit_ts=ts,
                            exit_price=exit_price,
                            exit_reason=exit_reason,
                            pnl_pct=pnl_pct,
                            mfe_pct=closed.best_pct,
                            bars_held=closed.bars_held(idx),
                            signal_score=closed.signal_score,
                            trend_strength=float(frame.loc[closed.entry_bar, "trend_strength"]),
                            pullback_depth_pct=float(frame.loc[closed.entry_bar, "pullback_depth_pct"]),
                            distance_from_vwap=float(frame.loc[closed.entry_bar, "distance_from_vwap"]),
                            compression_ratio=float(frame.loc[closed.entry_bar, "compression_ratio"]),
                        )
                    )
                    open_pair = None
            continue

        if pending is not None:
            continue

        candidates: list[tuple[float, str, int]] = []
        for pair, frame in frames.items():
            idx = by_ts[pair].get(ts)
            if idx is None or idx + 1 >= len(frame):
                continue
            row = frame.iloc[idx]
            if not bool(row.get("entry_signal", False)):
                continue
            if float(row.get("cs_breadth", 0.0)) < spec.min_breadth:
                continue
            if float(row.get("cs_rs_rank", 99.0)) > spec.max_rs_rank:
                continue
            if float(row.get("signal_score", 0.0)) < spec.min_signal_score:
                continue
            if spec.require_gate and not bool(row.get("gate_is_open", False)):
                continue
            candidates.append((_candidate_score(row), pair, idx))

        if not candidates:
            continue

        candidates.sort(key=lambda item: item[0], reverse=True)
        _, pair, idx = candidates[0]
        signal = strat.build_ensemble_signal(
            pair=pair,
            frame=frames[pair],
            row_idx=idx,
            construction=spec.construction,
            bar_idx=idx,
        )
        if signal is None:
            continue
        next_ts = int(frames[pair].iloc[idx + 1]["ts"])
        pending = {
            "pair": pair,
            "entry_ts": next_ts,
            "signal": signal,
        }

    return trades


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)

    if args.pairs:
        pairs = [pair.strip() for pair in args.pairs.split(",") if pair.strip()]
        universe = pd.DataFrame(
            {
                "pair": pairs,
                "quote_volume_24h": [np.nan] * len(pairs),
                "last_price": [np.nan] * len(pairs),
            }
        )
    else:
        universe = _fetch_liquid_usd_spot_universe(args.universe_size, args.min_quote_volume)
    universe.to_csv(args.universe_csv, index=False)
    print("Universe:")
    print(universe[["pair", "quote_volume_24h", "last_price"]].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    raw_histories: dict[str, pd.DataFrame] = {}
    frames_by_construction: dict[str, dict[str, pd.DataFrame]] = {}

    for pair in universe["pair"]:
        raw_histories[pair] = _load_or_fetch_history(
            pair=pair,
            history_days=args.history_days,
            cache_dir=cache_dir,
            trade_count=args.trade_count,
            trade_pause_sec=args.trade_pause_sec,
        )

    for spec in _portfolio_specs():
        built = {
            pair: strat.build_ensemble_frame(df_raw, construction=spec.construction, config=strat.DEFAULT_CONFIG)
            for pair, df_raw in raw_histories.items()
        }
        built = {pair: frame for pair, frame in built.items() if not frame.empty}
        frames_by_construction[spec.name] = _attach_cross_sectional_context(built)

    max_ts = max(int(df["ts"].max()) for df in raw_histories.values())
    recent_cutoff = max_ts - args.recent_days * 24 * 60 * 60

    summary_rows: list[dict] = []
    trade_rows: list[dict] = []

    baseline_pair = "GIGAUSD"
    if baseline_pair in raw_histories:
        baseline_trades = backtest.run_backtest(
            pair=baseline_pair,
            df_raw=raw_histories[baseline_pair],
            config=strat.DEFAULT_CONFIG,
            commission_pct=args.commission_pct,
            slippage_pct=args.slippage_pct,
            construction="tc15_tighter_volume_cap",
        )
        summary_rows.append(
            {
                "strategy": "baseline_tc15_tighter_volume_cap",
                "universe_size": 1,
                **shadow_backtest._summarize_trades(baseline_trades, recent_cutoff),
                "beats_baseline": False,
                "verdict": "BASELINE",
            }
        )
        for trade in baseline_trades:
            row = asdict(trade)
            row["strategy"] = "baseline_tc15_tighter_volume_cap"
            trade_rows.append(row)
        baseline_net = float(sum(trade.pnl_pct for trade in baseline_trades))
        baseline_trades_count = len(baseline_trades)
    else:
        baseline_net = 0.0
        baseline_trades_count = 0

    for spec in _portfolio_specs():
        print(f"\nRunning {spec.name} across {len(frames_by_construction[spec.name])} pairs...")
        trades = _run_portfolio_backtest(
            frames=frames_by_construction[spec.name],
            spec=spec,
            commission_pct=args.commission_pct,
            slippage_pct=args.slippage_pct,
        )
        net_full = float(sum(trade.pnl_pct for trade in trades))
        summary = shadow_backtest._summarize_trades(trades, recent_cutoff)
        beats_baseline = net_full > baseline_net and len(trades) > baseline_trades_count
        summary_rows.append(
            {
                "strategy": spec.name,
                "universe_size": len(frames_by_construction[spec.name]),
                **summary,
                "beats_baseline": beats_baseline,
                "verdict": "KEEP" if beats_baseline else "KILL",
            }
        )
        for trade in trades:
            row = asdict(trade)
            row["strategy"] = spec.name
            trade_rows.append(row)
        print(f"  {spec.name}: trades={len(trades)} net={net_full:+.3%}")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["verdict", "net_full", "trades_full"],
        ascending=[True, False, False],
    )
    print(f"\n{'=' * 120}")
    print(" CROSS-SECTIONAL UNIVERSE SUMMARY")
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
