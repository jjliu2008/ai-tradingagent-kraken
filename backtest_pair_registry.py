from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

import backtest as bt
import strategy as strat
from pair_strategy_registry import PAIR_STRATEGY_REGISTRY


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
    parser = argparse.ArgumentParser(description="Backtest the pair-specific strategy registry on cached 15m data.")
    parser.add_argument("--commission-pct", type=float, default=0.0026)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--trades-csv", default="results/pair_strategy_registry_trades.csv")
    args = parser.parse_args()

    best_data = find_best_15m_data()
    all_trades = []
    missing = []

    for pair, construction in PAIR_STRATEGY_REGISTRY.items():
        payload = best_data.get(pair)
        if payload is None:
            missing.append(pair)
            continue
        _, path = payload
        df = load_pair(path)
        trades = bt.run_backtest(
            pair=pair,
            df_raw=df,
            config=strat.DEFAULT_CONFIG,
            commission_pct=args.commission_pct,
            slippage_pct=args.slippage_pct,
            construction=construction,
        )
        all_trades.extend(trades)

    if missing:
        print("Missing cached data for:", ", ".join(missing))

    if not all_trades:
        print("No trades generated.")
        return

    all_trades = sorted(all_trades, key=lambda trade: (trade.entry_ts, trade.pair))
    bt.report(all_trades)
    trades_df = pd.DataFrame([asdict(trade) for trade in all_trades])
    Path(args.trades_csv).parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(args.trades_csv, index=False)
    print(f"\nRegistry trades saved -> {args.trades_csv}")


if __name__ == "__main__":
    main()
