from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

import backtest
import strategy as strat
from pair_strategy_registry import PAIR_STRATEGY_REGISTRY, frozen_pairs, registry_pairs


DATA_DIRS = (Path("data_cache"), Path("data_cache_walkforward"))
RESULTS_DIR = Path("results")

ENTRY_FILTERS: dict[str, dict[str, float]] = {
    "base": {},
    "score45": {"min_score": 45.0},
    "score55": {"min_score": 55.0},
    "score60": {"min_score": 60.0},
    "close70": {"min_close_location": 0.70},
    "close75": {"min_close_location": 0.75},
    "close80": {"min_close_location": 0.80},
    "gate20": {"min_gate_trend_strength": 0.0020},
    "gate30": {"min_gate_trend_strength": 0.0030},
    "gate50": {"min_gate_trend_strength": 0.0050},
    "volcap45": {"max_volume_ratio": 4.5},
    "volcap30": {"max_volume_ratio": 3.0},
    "volcap60": {"max_volume_ratio": 6.0},
    "vwap2": {"max_dist_vwap": 0.0200},
    "vwap3": {"max_dist_vwap": 0.0300},
    "score45_close70": {"min_score": 45.0, "min_close_location": 0.70},
    "score45_gate20": {"min_score": 45.0, "min_gate_trend_strength": 0.0020},
    "close70_gate20": {"min_close_location": 0.70, "min_gate_trend_strength": 0.0020},
    "score45_close70_gate20": {
        "min_score": 45.0,
        "min_close_location": 0.70,
        "min_gate_trend_strength": 0.0020,
    },
    "score55_close75_volcap45": {
        "min_score": 55.0,
        "min_close_location": 0.75,
        "max_volume_ratio": 4.5,
    },
    "close80_volcap60": {
        "min_close_location": 0.80,
        "max_volume_ratio": 6.0,
    },
    "close80_vwap3": {
        "min_close_location": 0.80,
        "max_dist_vwap": 0.0300,
    },
    "close80_volcap60_vwap3": {
        "min_close_location": 0.80,
        "max_volume_ratio": 6.0,
        "max_dist_vwap": 0.0300,
    },
    "gate50_vwap2": {
        "min_gate_trend_strength": 0.0050,
        "max_dist_vwap": 0.0200,
    },
    "gate50_vwap3": {
        "min_gate_trend_strength": 0.0050,
        "max_dist_vwap": 0.0300,
    },
    "close80_gate50_vwap3": {
        "min_close_location": 0.80,
        "min_gate_trend_strength": 0.0050,
        "max_dist_vwap": 0.0300,
    },
}

EXIT_PROFILES: dict[str, dict[str, float | int]] = {
    "base": {"min_stop_pct": 0.0150, "target_pct": 0.0450, "max_hold_bars": 12},
    "medium": {"min_stop_pct": 0.0125, "target_pct": 0.0400, "max_hold_bars": 10},
    "fast": {"min_stop_pct": 0.0100, "target_pct": 0.0300, "max_hold_bars": 8},
    "tight": {"min_stop_pct": 0.0120, "target_pct": 0.0350, "max_hold_bars": 8},
    "runner": {"min_stop_pct": 0.0150, "target_pct": 0.0550, "max_hold_bars": 14},
}


def _parse_days(path: Path) -> int:
    for part in path.stem.split("_"):
        if part.endswith("d") and part[:-1].isdigit():
            return int(part[:-1])
    return 0


def load_pair_data(pair: str) -> pd.DataFrame:
    matches: list[Path] = []
    for data_dir in DATA_DIRS:
        matches.extend(sorted(data_dir.glob(f"{pair}_15m_*_latest.csv")))
        matches.extend(sorted(data_dir.glob(f"{pair}_15m_*_live.csv")))
    if not matches:
        raise FileNotFoundError(f"No 15m data found for {pair}")

    best_path = max(matches, key=lambda path: (_parse_days(path), path.stat().st_mtime))
    df = pd.read_csv(best_path)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    if "vwap_k" not in df.columns:
        df["vwap_k"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    if "count" not in df.columns:
        df["count"] = 1
    return df.sort_values("ts").reset_index(drop=True)


def rolling_folds(length: int) -> list[tuple[str, int, int, int, int]]:
    fractions = [
        ("fold1", 0.0, 0.50, 0.50, 0.65),
        ("fold2", 0.0, 0.65, 0.65, 0.80),
        ("fold3", 0.0, 0.80, 0.80, 1.00),
    ]
    folds: list[tuple[str, int, int, int, int]] = []
    for name, train_start_frac, train_end_frac, test_start_frac, test_end_frac in fractions:
        train_start = int(length * train_start_frac)
        train_end = int(length * train_end_frac)
        test_start = int(length * test_start_frac)
        test_end = int(length * test_end_frac)
        if train_end - train_start < 240 or test_end - test_start < 120:
            continue
        folds.append((name, train_start, train_end, test_start, test_end))
    return folds


def apply_entry_filter(frame: pd.DataFrame, filter_name: str) -> pd.Series:
    spec = ENTRY_FILTERS[filter_name]
    mask = frame["entry_signal"].fillna(False).astype(bool)
    if "min_score" in spec:
        mask &= frame["signal_score"].astype(float) >= float(spec["min_score"])
    if "min_close_location" in spec:
        mask &= frame["close_location"].astype(float) >= float(spec["min_close_location"])
    if "max_volume_ratio" in spec:
        mask &= frame["volume_ratio"].astype(float) <= float(spec["max_volume_ratio"])
    if "min_gate_trend_strength" in spec:
        mask &= frame["gate_trend_strength_60"].astype(float) >= float(spec["min_gate_trend_strength"])
    if "max_dist_vwap" in spec:
        mask &= frame["distance_from_vwap"].astype(float) <= float(spec["max_dist_vwap"])
    return mask.fillna(False)


def summarize_trades(pair: str, construction: str, candidate: str, trades: list[backtest.BacktestTrade]) -> dict[str, Any]:
    if not trades:
        return {
            "pair": pair,
            "construction": construction,
            "candidate": candidate,
            "trades": 0,
            "net": 0.0,
            "win": 0.0,
            "avg": 0.0,
            "max_dd": 0.0,
            "score": 0.0,
        }
    pnls = pd.Series([trade.pnl_pct for trade in trades], dtype=float)
    equity = (1.0 + pnls).cumprod()
    max_dd = float((equity / equity.cummax() - 1.0).min())
    score = float(pnls.sum() - 0.50 * abs(max_dd) + 0.0025 * len(trades))
    return {
        "pair": pair,
        "construction": construction,
        "candidate": candidate,
        "trades": int(len(trades)),
        "net": float(pnls.sum()),
        "win": float((pnls > 0).mean()),
        "avg": float(pnls.mean()),
        "max_dd": max_dd,
        "score": score,
    }


def run_candidate_backtest(
    pair: str,
    df_raw: pd.DataFrame,
    construction: str,
    filter_name: str,
    exit_name: str,
    commission_pct: float,
    slippage_pct: float,
) -> list[backtest.BacktestTrade]:
    exit_spec = EXIT_PROFILES[exit_name]
    config = replace(
        strat.DEFAULT_CONFIG,
        min_stop_pct=float(exit_spec["min_stop_pct"]),
        target_pct=float(exit_spec["target_pct"]),
        max_hold_bars=int(exit_spec["max_hold_bars"]),
    )
    frame = strat.build_ensemble_frame(df_raw, construction=construction, config=config)
    if frame.empty:
        return []
    tuned = frame.copy()
    tuned["entry_signal"] = apply_entry_filter(frame, filter_name)
    return backtest.run_backtest_frame(
        pair=pair,
        df=tuned,
        config=config,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        construction=construction,
    )


def aggregate_candidate_tests(
    pair: str,
    construction: str,
    df_raw: pd.DataFrame,
    candidate: str,
    commission_pct: float,
    slippage_pct: float,
) -> dict[str, Any]:
    filter_name, exit_name = candidate.split("|", 1)
    all_test_trades: list[backtest.BacktestTrade] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_name, _, train_end, test_start, test_end in rolling_folds(len(df_raw)):
        train_df = df_raw.iloc[:train_end].reset_index(drop=True)
        test_df = df_raw.iloc[test_start:test_end].reset_index(drop=True)

        train_trades = run_candidate_backtest(
            pair,
            train_df,
            construction,
            filter_name,
            exit_name,
            commission_pct,
            slippage_pct,
        )
        test_trades = run_candidate_backtest(
            pair,
            test_df,
            construction,
            filter_name,
            exit_name,
            commission_pct,
            slippage_pct,
        )
        train_summary = summarize_trades(pair, construction, candidate, train_trades)
        test_summary = summarize_trades(pair, construction, candidate, test_trades)
        all_test_trades.extend(test_trades)
        fold_rows.append(
            {
                "pair": pair,
                "construction": construction,
                "candidate": candidate,
                "fold": fold_name,
                "train_trades": train_summary["trades"],
                "train_net": train_summary["net"],
                "train_score": train_summary["score"],
                "test_trades": test_summary["trades"],
                "test_net": test_summary["net"],
                "test_score": test_summary["score"],
            }
        )

    summary = summarize_trades(pair, construction, candidate, all_test_trades)
    summary["fold_count"] = len(fold_rows)
    summary["filter_name"] = filter_name
    summary["exit_name"] = exit_name
    summary["fold_rows"] = fold_rows
    return summary


def pair_candidates() -> list[str]:
    return [f"{filter_name}|{exit_name}" for filter_name in ENTRY_FILTERS for exit_name in EXIT_PROFILES]


def load_construction_overrides(csv_paths: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw_path in csv_paths:
        path_str = raw_path.strip()
        if not path_str:
            continue
        path = Path(path_str)
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "pair" not in frame.columns or "construction" not in frame.columns:
            continue
        for _, row in frame.iterrows():
            pair = str(row["pair"]).strip().upper()
            construction = str(row["construction"]).strip()
            if pair and construction and pair not in overrides:
                overrides[pair] = construction
    return overrides


def main() -> None:
    default_pairs = sorted(set(registry_pairs()) | {"KERNELUSD", "FHEUSD"})
    parser = argparse.ArgumentParser(description="Constrained rolling OOS tuner for pair-specific strategy winners.")
    parser.add_argument("--pairs", default=",".join(default_pairs))
    parser.add_argument(
        "--construction-csvs",
        default="results/live_walkforward_best_top30.csv,results/live_walkforward_best_top12.csv,results/pair_strategy_best_ge3.csv",
        help="Comma-separated CSVs used to resolve constructions for pairs not already in the registry.",
    )
    parser.add_argument("--commission-pct", type=float, default=0.0026)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--candidates-csv", default="results/rolling_pair_tuner_candidates.csv")
    parser.add_argument("--best-csv", default="results/rolling_pair_tuner_best.csv")
    parser.add_argument("--folds-csv", default="results/rolling_pair_tuner_folds.csv")
    args = parser.parse_args()
    overrides = load_construction_overrides(args.construction_csvs.split(","))

    results_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []

    for pair in [item.strip().upper() for item in args.pairs.split(",") if item.strip()]:
        construction = PAIR_STRATEGY_REGISTRY.get(pair) or overrides.get(pair)
        if construction is None:
            continue

        df = load_pair_data(pair)
        pair_rows: list[dict[str, Any]] = []
        for candidate in pair_candidates():
            summary = aggregate_candidate_tests(
                pair=pair,
                construction=construction,
                df_raw=df,
                candidate=candidate,
                commission_pct=args.commission_pct,
                slippage_pct=args.slippage_pct,
            )
            results_rows.append({k: v for k, v in summary.items() if k != "fold_rows"})
            pair_rows.append(summary)
            fold_rows.extend(summary["fold_rows"])

        frame = pd.DataFrame([{k: v for k, v in row.items() if k != "fold_rows"} for row in pair_rows])
        if frame.empty:
            continue

        ranked = frame.sort_values(["score", "net", "trades"], ascending=False).reset_index(drop=True)
        best = dict(ranked.iloc[0])
        baseline = frame[frame["candidate"] == "base|base"].copy()
        base_row = dict(baseline.iloc[0]) if not baseline.empty else {}

        best_rows.append(
            {
                "pair": pair,
                "frozen": pair in frozen_pairs(),
                "construction": construction,
                "baseline_candidate": base_row.get("candidate", "base|base"),
                "baseline_trades": int(base_row.get("trades", 0) or 0),
                "baseline_net": float(base_row.get("net", 0.0) or 0.0),
                "baseline_score": float(base_row.get("score", 0.0) or 0.0),
                "best_candidate": best["candidate"],
                "best_filter_name": best["filter_name"],
                "best_exit_name": best["exit_name"],
                "best_trades": int(best["trades"]),
                "best_net": float(best["net"]),
                "best_win": float(best["win"]),
                "best_max_dd": float(best["max_dd"]),
                "best_score": float(best["score"]),
                "net_delta_vs_base": float(best["net"]) - float(base_row.get("net", 0.0) or 0.0),
                "score_delta_vs_base": float(best["score"]) - float(base_row.get("score", 0.0) or 0.0),
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results_rows).to_csv(args.candidates_csv, index=False)
    pd.DataFrame(fold_rows).to_csv(args.folds_csv, index=False)
    best_frame = pd.DataFrame(best_rows).sort_values(["score_delta_vs_base", "net_delta_vs_base"], ascending=False)
    best_frame.to_csv(args.best_csv, index=False)

    print("Rolling tuner summary:")
    for _, row in best_frame.iterrows():
        frozen_tag = " [frozen]" if bool(row["frozen"]) else ""
        print(
            f"  {row['pair']:<10} {row['construction']:<24} "
            f"{row['best_candidate']:<28} "
            f"oos={row['best_net']:+.3%}/{int(row['best_trades'])} "
            f"delta={row['net_delta_vs_base']:+.3%}{frozen_tag}"
        )


if __name__ == "__main__":
    main()
