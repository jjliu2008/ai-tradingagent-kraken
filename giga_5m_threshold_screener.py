from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

import adaptive_strategy_screener as adaptive
import backtest
import lower_bar_screener as lbs
import shadow_backtest
import strategy as strat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Focused threshold sweep for the GIGA 5m adaptive union branch"
    )
    parser.add_argument("--history-days", type=int, default=60)
    parser.add_argument("--recent-days", type=int, default=30)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--trade-count", type=int, default=5000)
    parser.add_argument("--trade-pause-sec", type=float, default=0.8)
    parser.add_argument(
        "--cost-scenarios",
        default="taker_like=0.0031,maker_mid=0.0010,maker_optimistic=0.0003",
    )
    parser.add_argument("--summary-csv", default="giga_5m_threshold_summary.csv")
    parser.add_argument("--trades-csv", default="giga_5m_threshold_trades.csv")
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


def _prepare_frame(frame: pd.DataFrame, entry_mask: pd.Series) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    out["entry_signal"] = entry_mask.fillna(False).to_numpy(dtype=bool)
    out["signal_score"] = _score_frame(out).to_numpy(dtype=float)
    out["gate_trend_strength_60"] = out["ctx60_trend_strength"].fillna(0.0).to_numpy(dtype=float)
    out["gate_momentum_medium_60"] = out["ctx60_momentum_medium"].fillna(0.0).to_numpy(dtype=float)
    return out


def _exit_profiles() -> dict[str, strat.StrategyConfig]:
    return {
        "shared_default": strat.DEFAULT_CONFIG,
        "hold12_t35_s12": strat.StrategyConfig(min_stop_pct=0.012, target_pct=0.035, max_hold_bars=12, fast_max_hold_bars=8),
        "hold8_t35_s12": strat.StrategyConfig(min_stop_pct=0.012, target_pct=0.035, max_hold_bars=8, fast_max_hold_bars=6),
        "hold8_t30_s10": strat.StrategyConfig(min_stop_pct=0.010, target_pct=0.030, max_hold_bars=8, fast_max_hold_bars=6),
        "hold6_t25_s10": strat.StrategyConfig(min_stop_pct=0.010, target_pct=0.025, max_hold_bars=6, fast_max_hold_bars=4),
    }


def _variant_name(
    trend_th: float | None,
    score_th: float | None,
    vwap_cap: float | None,
    close_th: float | None,
    comp_cap: float | None,
    exit_name: str,
) -> str:
    def fmt(prefix: str, value: float | None, scale: int = 1000) -> str:
        if value is None:
            return f"{prefix}na"
        return f"{prefix}{int(round(value * scale))}"

    return "__".join(
        [
            fmt("trend", trend_th, 1000),
            fmt("score", score_th, 1),
            fmt("vwap", vwap_cap, 1000),
            fmt("close", close_th, 100),
            fmt("comp", comp_cap, 100),
            exit_name,
        ]
    )


def _build_variants(frame: pd.DataFrame) -> dict[str, pd.Series]:
    candidates = {name: mask for name, mask, _ in adaptive.build_adaptive_candidates(frame, 5)}
    base = candidates["freqai_mtf_adaptive_union"].fillna(False)
    score = _score_frame(frame)

    variants: dict[str, pd.Series] = {"base": base}
    trend_thresholds = [0.008, 0.009, 0.010]
    score_thresholds = [None, 43.37, 46.0]
    vwap_caps = [None, 0.030, 0.025]
    close_thresholds = [None, 0.75]
    comp_caps = [None, 1.40]

    for trend_th in trend_thresholds:
        for score_th in score_thresholds:
            for vwap_cap in vwap_caps:
                for close_th in close_thresholds:
                    for comp_cap in comp_caps:
                        mask = base.copy()
                        if trend_th is not None:
                            mask &= frame["trend_strength"] > trend_th
                        if score_th is not None:
                            mask &= score > score_th
                        if vwap_cap is not None:
                            mask &= frame["distance_from_vwap"] <= vwap_cap
                        if close_th is not None:
                            mask &= frame["close_location"] > close_th
                        if comp_cap is not None:
                            mask &= frame["compression_ratio"] <= comp_cap
                        name = _variant_name(
                            trend_th=trend_th,
                            score_th=score_th,
                            vwap_cap=vwap_cap,
                            close_th=close_th,
                            comp_cap=comp_cap,
                            exit_name="maskonly",
                        )
                        variants[name] = mask.fillna(False)
    return variants


def _summarize(label: str, trades: list[backtest.BacktestTrade], recent_cutoff: int, cost_model: str) -> dict:
    return {
        "label": label,
        "cost_model": cost_model,
        **shadow_backtest._summarize_trades(trades, recent_cutoff),
    }


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cost_scenarios = lbs.parse_cost_scenarios(args.cost_scenarios)

    frames = adaptive._build_frames(
        pairs=["GIGAUSD", "DOGUSD", "COQUSD", "HYPEUSD"],
        intervals=[5],
        history_days=args.history_days,
        cache_dir=cache_dir,
        trade_count=args.trade_count,
        trade_pause_sec=args.trade_pause_sec,
    )
    giga_frame = frames[5]["GIGAUSD"].copy().reset_index(drop=True)
    baseline_history = _load_baseline_history(args.history_days, cache_dir)
    recent_cutoff = max(int(giga_frame["ts"].max()), int(baseline_history["ts"].max())) - args.recent_days * 24 * 60 * 60

    variants = _build_variants(giga_frame)
    exit_profiles = _exit_profiles()

    summary_rows: list[dict] = []
    trade_rows: list[dict] = []

    for cost_label, side_cost_pct in cost_scenarios:
        print(f"\n{'=' * 112}")
        print(f" GIGA 5M THRESHOLD SWEEP | cost_model={cost_label} | side_cost={side_cost_pct:.4%}")
        print(f"{'=' * 112}")

        baseline_trades = backtest.run_backtest(
            pair="GIGAUSD",
            df_raw=baseline_history,
            config=strat.DEFAULT_CONFIG,
            commission_pct=side_cost_pct,
            slippage_pct=0.0,
            construction="tc15_tighter_volume_cap",
        )
        summary_rows.append(
            _summarize("baseline_tc15_tighter_volume_cap", baseline_trades, recent_cutoff, cost_label)
        )
        for trade in baseline_trades:
            row = asdict(trade)
            row["label"] = "baseline_tc15_tighter_volume_cap"
            row["cost_model"] = cost_label
            trade_rows.append(row)

        print(f"baseline_tc15_tighter_volume_cap: {len(baseline_trades)} trades")
        for variant_name, mask in variants.items():
            prepared = _prepare_frame(giga_frame, mask)
            for exit_name, config in exit_profiles.items():
                label = variant_name.replace("__maskonly", f"__{exit_name}")
                trades = backtest.run_backtest_frame(
                    pair="GIGAUSD",
                    df=prepared,
                    config=config,
                    commission_pct=side_cost_pct,
                    slippage_pct=0.0,
                    construction=label,
                )
                summary_rows.append(_summarize(label, trades, recent_cutoff, cost_label))
                for trade in trades:
                    row = asdict(trade)
                    row["label"] = label
                    row["cost_model"] = cost_label
                    trade_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["cost_model", "net_full", "net_recent", "trades_full"],
        ascending=[True, False, False, False],
    )

    print(f"\n{'=' * 112}")
    print(" TOP THRESHOLD VARIANTS")
    print(f"{'=' * 112}")
    cols = [
        "cost_model",
        "label",
        "trades_full",
        "net_full",
        "win_full",
        "max_dd",
        "trades_recent",
        "net_recent",
        "win_recent",
    ]
    formatters = {
        "net_full": lambda value: f"{value:+.3%}",
        "win_full": lambda value: f"{value:.1%}",
        "max_dd": lambda value: f"{value:.3%}",
        "net_recent": lambda value: f"{value:+.3%}",
        "win_recent": lambda value: f"{value:.1%}",
    }
    top = summary.groupby("cost_model", group_keys=False).head(20)
    print(top[cols].to_string(index=False, formatters=formatters))

    summary.to_csv(args.summary_csv, index=False)
    print(f"\nWrote summary to {args.summary_csv}")
    if trade_rows:
        pd.DataFrame(trade_rows).to_csv(args.trades_csv, index=False)
        print(f"Wrote trades to {args.trades_csv}")


if __name__ == "__main__":
    main()
