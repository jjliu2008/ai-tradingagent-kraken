from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import backtest as bt
import strategy as strat
from pair_strategy_registry import frozen_pairs


DATA_DIR = Path("data_cache")


def find_best_15m_data() -> dict[str, tuple[int, Path]]:
    found: dict[str, tuple[int, Path]] = {}
    for path in sorted(DATA_DIR.glob("*_15m_*_end_latest.csv")):
        parts = path.stem.split("_")
        if len(parts) < 3:
            continue
        pair = parts[0].upper()
        try:
            days = int(parts[2].replace("d", ""))
        except ValueError:
            continue
        if pair not in found or days > found[pair][0]:
            found[pair] = (days, path)
    return found


def load_pair(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    if "vwap_k" not in df.columns:
        df["vwap_k"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    if "count" not in df.columns:
        df["count"] = 1
    return df.sort_values("ts").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen pair-specific ensemble constructions on cached 15m data.")
    parser.add_argument("--commission-pct", type=float, default=0.0026)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--min-trades", type=int, default=3)
    parser.add_argument("--include-frozen", action="store_true")
    parser.add_argument("--matrix-csv", default="results/pair_strategy_matrix.csv")
    parser.add_argument("--best-csv", default="results/pair_strategy_best.csv")
    args = parser.parse_args()

    config = strat.DEFAULT_CONFIG
    constructions = list(strat.ensemble_construction_names())
    best_data = find_best_15m_data()
    frozen = set(frozen_pairs())

    rows: list[dict] = []
    for pair, (days, path) in sorted(best_data.items()):
        if not args.include_frozen and pair in frozen:
            continue
        df = load_pair(path)
        for construction in constructions:
            try:
                trades = bt.run_backtest(
                    pair=pair,
                    df_raw=df,
                    config=config,
                    commission_pct=args.commission_pct,
                    slippage_pct=args.slippage_pct,
                    construction=construction,
                )
            except Exception as exc:
                rows.append(
                    {
                        "pair": pair,
                        "days": days,
                        "construction": construction,
                        "error": type(exc).__name__,
                    }
                )
                continue

            pnls = [trade.pnl_pct for trade in trades]
            rows.append(
                {
                    "pair": pair,
                    "days": days,
                    "construction": construction,
                    "trades": len(trades),
                    "net": sum(pnls) if pnls else 0.0,
                    "win": float((pd.Series(pnls) > 0).mean()) if pnls else 0.0,
                    "avg": float(pd.Series(pnls).mean()) if pnls else 0.0,
                    "max_dd": bt._max_drawdown(pnls) if pnls else 0.0,
                    "avg_mfe": float(pd.Series([trade.mfe_pct for trade in trades]).mean()) if pnls else 0.0,
                    "avg_hold": float(pd.Series([trade.bars_held for trade in trades]).mean()) if pnls else 0.0,
                }
            )

    matrix = pd.DataFrame(rows).sort_values(["pair", "net"], ascending=[True, False])
    Path(args.matrix_csv).parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.matrix_csv, index=False)

    filtered = matrix[(matrix.get("trades", 0) >= args.min_trades) & (matrix.get("net", 0.0) > 0)].copy()
    best = filtered.groupby("pair", as_index=False).first()
    best = best.sort_values("net", ascending=False)
    best.to_csv(args.best_csv, index=False)

    print(f"Pairs tested: {matrix['pair'].nunique()}")
    print(f"Positive pair/construction combos: {len(filtered)}")
    print(f"Best-per-pair selections: {len(best)}")
    for _, row in best.iterrows():
        print(
            f"{row['pair']:<8} {row['construction']:<24} "
            f"trades={int(row['trades']):>2} net={row['net']:+.3%} win={row['win']:.1%}"
        )
    print(f"\nMatrix saved -> {args.matrix_csv}")
    print(f"Best saved   -> {args.best_csv}")


if __name__ == "__main__":
    main()
