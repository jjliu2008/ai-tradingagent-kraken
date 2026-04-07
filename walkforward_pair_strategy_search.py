from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

import backtest as bt
import find_alpha_pairs as alpha
import strategy as strat
import universe_scanner_agent as usa
from pair_strategy_registry import PAIR_STRATEGY_REGISTRY


CACHE_DIR = Path("data_cache_walkforward")


def current_universe() -> list[str]:
    return list(dict.fromkeys(usa.UNIVERSE))


def fetch_or_cache_15m_history(pair: str, history_days: int, trade_count: int, trade_pause_sec: float) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"{pair}_15m_{history_days}d_live.csv"
    if out.exists():
        return out
    df = bt.fetch_history(
        pair=pair,
        interval=15,
        history_days=history_days,
        trade_count=trade_count,
        trade_pause_sec=trade_pause_sec,
    )
    df.to_csv(out, index=False)
    return out


def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    if "vwap_k" not in df.columns:
        df["vwap_k"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    if "count" not in df.columns:
        df["count"] = 1
    return df.sort_values("ts").reset_index(drop=True)


def split_train_test(df: pd.DataFrame, train_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = max(120, int(len(df) * train_frac))
    cut = min(cut, len(df) - 60)
    train = df.iloc[:cut].reset_index(drop=True)
    test = df.iloc[cut:].reset_index(drop=True)
    return train, test


def score_row(net: float, win: float, trades: int) -> float:
    return net * (0.5 + win) * max(trades, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live-universe walk-forward pair-specific strategy search.")
    parser.add_argument("--history-days", type=int, default=30)
    parser.add_argument("--train-frac", type=float, default=0.67)
    parser.add_argument("--hourly-days", type=int, default=30)
    parser.add_argument("--notional", type=float, default=75.0)
    parser.add_argument("--prefilter-min-trades", type=int, default=1)
    parser.add_argument("--prefilter-top", type=int, default=20)
    parser.add_argument("--trade-count", type=int, default=5000)
    parser.add_argument("--trade-pause-sec", type=float, default=0.15)
    parser.add_argument("--commission-pct", type=float, default=0.0026)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--matrix-csv", default="results/live_walkforward_matrix.csv")
    parser.add_argument("--best-csv", default="results/live_walkforward_best.csv")
    parser.add_argument("--summary-csv", default="results/live_walkforward_summary.csv")
    args = parser.parse_args()

    requested = current_universe()
    tradable, _ = usa.validate_tradable_pairs(requested, notional=args.notional, max_spread_pct=0.03)
    print(f"Tradable universe: {len(tradable)}/{len(requested)}")

    prefilter_rows: list[dict] = []
    for idx, pair in enumerate(tradable, start=1):
        print(f"[prefilter {idx}/{len(tradable)}] {pair}")
        result = alpha.backtest_live(pair, 60, args.notional, days=args.hourly_days)
        if result is None:
            continue
        prefilter_rows.append(
            {
                "pair": pair,
                "trades": result.n_trades,
                "net_usd": result.pnl,
                "win_rate": result.win_rate,
                "pf": result.pf,
                "sharpe": result.sharpe,
                "freq_week": result.freq_week,
            }
        )

    pre = pd.DataFrame(prefilter_rows)
    if pre.empty:
        raise SystemExit("No prefilter results generated.")

    pre = pre[(pre["net_usd"] > 0) & (pre["trades"] >= args.prefilter_min_trades)].copy()
    pre = pre.sort_values(["net_usd", "win_rate", "pf"], ascending=False)
    candidates = pre["pair"].head(args.prefilter_top).tolist()
    if "GIGAUSD" not in candidates and "GIGAUSD" in tradable:
        candidates.insert(0, "GIGAUSD")
    candidates = list(dict.fromkeys(candidates))
    print(f"Walk-forward candidates: {candidates}")

    constructions = list(strat.ensemble_construction_names())
    matrix_rows: list[dict] = []
    best_rows: list[dict] = []

    for idx, pair in enumerate(candidates, start=1):
        print(f"\n[walk-forward {idx}/{len(candidates)}] {pair}")
        if pair == "GIGAUSD":
            chosen = PAIR_STRATEGY_REGISTRY["GIGAUSD"]
            history_path = fetch_or_cache_15m_history(pair, args.history_days, args.trade_count, args.trade_pause_sec)
            df = load_ohlcv_csv(history_path)
            train_df, test_df = split_train_test(df, args.train_frac)
            train_trades = bt.run_backtest(pair, train_df, strat.DEFAULT_CONFIG, args.commission_pct, args.slippage_pct, construction=chosen)
            test_trades = bt.run_backtest(pair, test_df, strat.DEFAULT_CONFIG, args.commission_pct, args.slippage_pct, construction=chosen)
            best_rows.append(
                {
                    "pair": pair,
                    "construction": chosen,
                    "frozen": True,
                    "train_trades": len(train_trades),
                    "train_net": sum(t.pnl_pct for t in train_trades),
                    "test_trades": len(test_trades),
                    "test_net": sum(t.pnl_pct for t in test_trades),
                    "test_win": float((pd.Series([t.pnl_pct for t in test_trades]) > 0).mean()) if test_trades else 0.0,
                }
            )
            continue

        try:
            history_path = fetch_or_cache_15m_history(pair, args.history_days, args.trade_count, args.trade_pause_sec)
        except Exception as exc:
            print(f"  history fetch failed: {exc}")
            continue

        df = load_ohlcv_csv(history_path)
        if len(df) < 240:
            print("  insufficient 15m history")
            continue
        train_df, test_df = split_train_test(df, args.train_frac)

        pair_rows: list[dict] = []
        for construction in constructions:
            try:
                train_trades = bt.run_backtest(
                    pair, train_df, strat.DEFAULT_CONFIG, args.commission_pct, args.slippage_pct, construction=construction
                )
                test_trades = bt.run_backtest(
                    pair, test_df, strat.DEFAULT_CONFIG, args.commission_pct, args.slippage_pct, construction=construction
                )
            except Exception as exc:
                pair_rows.append({"pair": pair, "construction": construction, "error": type(exc).__name__})
                continue

            train_pnls = [t.pnl_pct for t in train_trades]
            test_pnls = [t.pnl_pct for t in test_trades]
            row = {
                "pair": pair,
                "construction": construction,
                "train_trades": len(train_trades),
                "train_net": sum(train_pnls) if train_pnls else 0.0,
                "train_win": float((pd.Series(train_pnls) > 0).mean()) if train_pnls else 0.0,
                "test_trades": len(test_trades),
                "test_net": sum(test_pnls) if test_pnls else 0.0,
                "test_win": float((pd.Series(test_pnls) > 0).mean()) if test_pnls else 0.0,
            }
            row["train_score"] = score_row(row["train_net"], row["train_win"], row["train_trades"])
            pair_rows.append(row)
            matrix_rows.append(row)

        if not pair_rows:
            continue

        frame = pd.DataFrame(pair_rows)
        keep = frame[(frame["train_net"] > 0) & (frame["train_trades"] >= 2)].copy()
        if keep.empty:
            continue
        keep = keep.sort_values(["train_score", "test_net", "test_trades"], ascending=False)
        best_rows.append(dict(keep.iloc[0]))

    Path(args.matrix_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix_rows).to_csv(args.matrix_csv, index=False)
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(args.best_csv, index=False)

    survivors = best_df[(best_df["test_net"] > 0) & (best_df["test_trades"] >= 1)].copy()
    summary = {
        "candidate_count": len(candidates),
        "selected_count": len(best_df),
        "survivor_count": len(survivors),
        "survivor_pairs": ",".join(survivors["pair"].tolist()) if not survivors.empty else "",
        "survivor_total_test_net": float(survivors["test_net"].sum()) if not survivors.empty else 0.0,
        "survivor_total_test_trades": int(survivors["test_trades"].sum()) if not survivors.empty else 0,
    }
    with open(args.summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print("\nBest pair-specific selections:")
    if best_df.empty:
        print("  none")
    else:
        for _, row in best_df.iterrows():
            print(
                f"  {row['pair']:<8} {row['construction']:<24} "
                f"train={row['train_net']:+.3%}/{int(row['train_trades'])} "
                f"test={row['test_net']:+.3%}/{int(row['test_trades'])}"
            )

    print("\nOut-of-sample survivors:")
    if survivors.empty:
        print("  none")
    else:
        for _, row in survivors.iterrows():
            print(
                f"  {row['pair']:<8} {row['construction']:<24} "
                f"test={row['test_net']:+.3%}/{int(row['test_trades'])} win={row['test_win']:.1%}"
            )
        print(
            f"\nCombined OOS: trades={int(survivors['test_trades'].sum())} "
            f"net={survivors['test_net'].sum():+.3%}"
        )

    print(f"\nMatrix saved   -> {args.matrix_csv}")
    print(f"Best saved     -> {args.best_csv}")
    print(f"Summary saved  -> {args.summary_csv}")


if __name__ == "__main__":
    main()
