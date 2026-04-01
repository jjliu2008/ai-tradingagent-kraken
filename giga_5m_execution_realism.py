from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

import adaptive_strategy_screener as adaptive
import backtest
import lower_bar_screener as lbs
import shadow_backtest
import strategy as strat


BEST_GIGA_5M_LABEL = "trend8__score43__vwapna__closena__compna__hold8_t30_s10"


@dataclass
class MakerScenario:
    name: str
    side_cost_pct: float
    entry_offset_pct: float
    ttl_bars: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execution-realism screen for the best GIGA 5m branch challenger"
    )
    parser.add_argument("--history-days", type=int, default=60)
    parser.add_argument("--recent-days", type=int, default=30)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--trade-count", type=int, default=5000)
    parser.add_argument("--trade-pause-sec", type=float, default=0.8)
    parser.add_argument("--summary-csv", default="giga_5m_execution_summary.csv")
    parser.add_argument("--trades-csv", default="giga_5m_execution_trades.csv")
    return parser.parse_args()


def _load_baseline_history(history_days: int, cache_dir: Path) -> pd.DataFrame:
    cache_120 = cache_dir / "GIGAUSD_15m_120d_end_latest.csv"
    if cache_120.exists():
        df = pd.read_csv(cache_120).sort_values("ts").reset_index(drop=True)
        cutoff = int(df["ts"].max()) - history_days * 24 * 60 * 60
        return df.loc[df["ts"] >= cutoff].reset_index(drop=True)
    return backtest.fetch_history(
        pair="GIGAUSD",
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
        + 8.0 * frame["market_breadth"].clip(lower=0.0)
        + 4.0 * frame["strong_breadth"].clip(lower=0.0)
        + 4.0 * frame["rs_gap"].clip(lower=0.0)
    )


def _build_best_variant_frame(history_days: int, cache_dir: Path, trade_count: int, trade_pause_sec: float) -> pd.DataFrame:
    frames = adaptive._build_frames(
        pairs=["GIGAUSD", "DOGUSD", "COQUSD", "HYPEUSD"],
        intervals=[5],
        history_days=history_days,
        cache_dir=cache_dir,
        trade_count=trade_count,
        trade_pause_sec=trade_pause_sec,
    )
    frame = frames[5]["GIGAUSD"].copy().reset_index(drop=True)
    candidates = {name: mask for name, mask, _ in adaptive.build_adaptive_candidates(frame, 5)}
    base = candidates["freqai_mtf_adaptive_union"].fillna(False)
    score = _score_frame(frame)
    mask = base & (frame["trend_strength"] > 0.008) & (score > 43.37)

    out = frame.copy().reset_index(drop=True)
    out["entry_signal"] = mask.to_numpy(dtype=bool)
    out["signal_score"] = score.to_numpy(dtype=float)
    out["gate_trend_strength_60"] = out["ctx60_trend_strength"].fillna(0.0).to_numpy(dtype=float)
    out["gate_momentum_medium_60"] = out["ctx60_momentum_medium"].fillna(0.0).to_numpy(dtype=float)
    return out


def _best_variant_config() -> strat.StrategyConfig:
    return strat.StrategyConfig(
        min_stop_pct=0.010,
        target_pct=0.030,
        max_hold_bars=8,
        fast_max_hold_bars=6,
    )


def _maker_scenarios() -> list[MakerScenario]:
    return [
        MakerScenario(name="taker_ref", side_cost_pct=0.0031, entry_offset_pct=0.0, ttl_bars=1),
        MakerScenario(name="maker_mid_fill_now", side_cost_pct=0.0010, entry_offset_pct=0.0, ttl_bars=1),
        MakerScenario(name="maker_mid_5bps_ttl1", side_cost_pct=0.0010, entry_offset_pct=0.0005, ttl_bars=1),
        MakerScenario(name="maker_mid_10bps_ttl1", side_cost_pct=0.0010, entry_offset_pct=0.0010, ttl_bars=1),
        MakerScenario(name="maker_mid_10bps_ttl2", side_cost_pct=0.0010, entry_offset_pct=0.0010, ttl_bars=2),
        MakerScenario(name="maker_mid_15bps_ttl2", side_cost_pct=0.0010, entry_offset_pct=0.0015, ttl_bars=2),
        MakerScenario(name="maker_mid_20bps_ttl3", side_cost_pct=0.0010, entry_offset_pct=0.0020, ttl_bars=3),
        MakerScenario(name="maker_opt_10bps_ttl2", side_cost_pct=0.0003, entry_offset_pct=0.0010, ttl_bars=2),
    ]


def _simulate_maker_entry(
    df: pd.DataFrame,
    pair: str,
    config: strat.StrategyConfig,
    scenario: MakerScenario,
    construction: str,
) -> tuple[list[backtest.BacktestTrade], int]:
    strategy = strat.TrendGateEnsembleStrategy(pair=pair, config=config, construction=construction)
    if df.empty:
        return [], 0

    trades: list[backtest.BacktestTrade] = []
    pending_order: dict | None = None
    missed_entries = 0

    last_index = len(df) - 1
    for i in range(strategy.min_master_bars, len(df)):
        window = df.iloc[: i + 1]
        row = window.iloc[-1]
        ts = int(row["ts"])
        close_price = float(row["close"])
        high_price = float(row["high"])
        low_price = float(row["low"])

        if pending_order is not None and strategy.trade is None:
            if low_price <= pending_order["limit_price"]:
                entry_price = pending_order["limit_price"] * (1 + scenario.side_cost_pct)
                signal = pending_order["signal"]
                strategy.open_trade(
                    signal,
                    size=1.0,
                    entry_price=entry_price,
                    exit_mode="standard",
                    ai_confidence=0.0,
                    entry_bar=i,
                    entry_ts=ts,
                )
                pending_order = None
            elif i >= pending_order["expiry_bar"]:
                pending_order = None
                missed_entries += 1

        if strategy.trade and strategy.trade.is_open:
            strategy.trade.update_best(high_price)

            exit_reason: str | None = None
            exit_price: float | None = None
            if low_price <= strategy.trade.stop_price:
                exit_reason = "STOP_LOSS"
                exit_price = strategy.trade.stop_price * (1 - scenario.side_cost_pct)
            elif strategy.trade.target_price is not None and high_price >= strategy.trade.target_price:
                exit_reason = "TAKE_PROFIT"
                exit_price = strategy.trade.target_price * (1 - scenario.side_cost_pct)
            else:
                exit_reason = strategy.check_exit(window)
                if exit_reason:
                    exit_price = close_price * (1 - scenario.side_cost_pct)

            if exit_reason and exit_price is not None:
                closed = strategy.close_trade(i, ts, exit_price, exit_reason)
                pnl_pct = (closed.realized_pnl_pct() or 0.0) - scenario.side_cost_pct
                trades.append(
                    backtest.BacktestTrade(
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

        if pending_order is None and bool(row.get("entry_signal", False)) and i + 1 < len(df):
            signal = strat.build_ensemble_signal(
                pair,
                df,
                row_idx=i,
                construction=construction,
                bar_idx=i,
            )
            if signal is not None:
                next_open = float(df.iloc[i + 1]["open"])
                pending_order = {
                    "signal": signal,
                    "limit_price": next_open * (1 - scenario.entry_offset_pct),
                    "expiry_bar": min(last_index, i + scenario.ttl_bars),
                }

    if pending_order is not None:
        missed_entries += 1
    return trades, missed_entries


def _summarize(label: str, scenario: str, trades: list[backtest.BacktestTrade], recent_cutoff: int, signals: int, missed_entries: int) -> dict:
    summary = shadow_backtest._summarize_trades(trades, recent_cutoff)
    fills = len(trades)
    return {
        "label": label,
        "scenario": scenario,
        "signals": signals,
        "fills": fills,
        "missed_entries": missed_entries,
        "fill_rate": (fills / signals) if signals else 0.0,
        **summary,
    }


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)

    baseline_history = _load_baseline_history(args.history_days, cache_dir)
    variant_frame = _build_best_variant_frame(
        history_days=args.history_days,
        cache_dir=cache_dir,
        trade_count=args.trade_count,
        trade_pause_sec=args.trade_pause_sec,
    )
    variant_signals = int(variant_frame["entry_signal"].sum())
    recent_cutoff = max(int(baseline_history["ts"].max()), int(variant_frame["ts"].max())) - args.recent_days * 24 * 60 * 60

    summary_rows: list[dict] = []
    trade_rows: list[dict] = []

    baseline_trades = backtest.run_backtest(
        pair="GIGAUSD",
        df_raw=baseline_history,
        config=strat.DEFAULT_CONFIG,
        commission_pct=0.0031,
        slippage_pct=0.0,
        construction="tc15_tighter_volume_cap",
    )
    summary_rows.append(
        _summarize(
            label="baseline_tc15_tighter_volume_cap",
            scenario="baseline_taker_ref",
            trades=baseline_trades,
            recent_cutoff=recent_cutoff,
            signals=len(baseline_trades),
            missed_entries=0,
        )
    )
    for trade in baseline_trades:
        row = asdict(trade)
        row["label"] = "baseline_tc15_tighter_volume_cap"
        row["scenario"] = "baseline_taker_ref"
        trade_rows.append(row)

    print(f"baseline_tc15_tighter_volume_cap: {len(baseline_trades)} trades")
    config = _best_variant_config()
    for scenario in _maker_scenarios():
        trades, missed = _simulate_maker_entry(
            df=variant_frame,
            pair="GIGAUSD",
            config=config,
            scenario=scenario,
            construction=BEST_GIGA_5M_LABEL,
        )
        print(
            f"{BEST_GIGA_5M_LABEL} [{scenario.name}] "
            f"signals={variant_signals} fills={len(trades)} missed={missed}"
        )
        summary_rows.append(
            _summarize(
                label=BEST_GIGA_5M_LABEL,
                scenario=scenario.name,
                trades=trades,
                recent_cutoff=recent_cutoff,
                signals=variant_signals,
                missed_entries=missed,
            )
        )
        for trade in trades:
            row = asdict(trade)
            row["label"] = BEST_GIGA_5M_LABEL
            row["scenario"] = scenario.name
            trade_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["net_full", "net_recent", "fill_rate"],
        ascending=[False, False, False],
    )
    print(f"\n{'=' * 110}")
    print(" EXECUTION REALISM SUMMARY")
    print(f"{'=' * 110}")
    cols = [
        "label",
        "scenario",
        "signals",
        "fills",
        "fill_rate",
        "net_full",
        "win_full",
        "max_dd",
        "trades_recent",
        "net_recent",
        "win_recent",
    ]
    formatters = {
        "fill_rate": lambda value: f"{value:.1%}",
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
