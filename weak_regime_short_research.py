from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import research_runtime as rr
import strategy as strat


RESULTS_DIR = Path("results")
UNIFORM_CACHE_DIR = Path("data_cache_walkforward")
COMMISSION_PCT = 0.0026
SLIPPAGE_PCT = 0.0005
FEATURE_CONSTRUCTION = "trend_gate"


def _load_uniform_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [column.lower() for column in df.columns]
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    if "vwap_k" not in df.columns:
        df["vwap_k"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    if "count" not in df.columns:
        df["count"] = 1
    return df.sort_values("ts").reset_index(drop=True)


def _build_feature_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    frame = strat.build_ensemble_frame(
        df_raw,
        construction=FEATURE_CONSTRUCTION,
        config=strat.DEFAULT_CONFIG,
    )
    if frame.empty:
        return frame
    out = frame.copy()
    out["entry_signal_pre_regime"] = out["entry_signal"].fillna(False)
    out["portfolio_regime_ok"] = rr.apply_portfolio_regime_gate(out)
    return rr.add_weak_regime_features(out)


def _short_signal_map(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "short_breakdown": (
            frame["weak_regime"]
            & (frame["gate_trend_strength_60"].astype(float) <= -0.0025)
            & (frame["atr_pct"].astype(float) >= 0.0045)
            & (frame["close"].astype(float) < frame["ema_fast"].astype(float))
            & (frame["close"].astype(float) < frame["session_vwap"].astype(float))
            & (frame["momentum_medium"].astype(float) <= -0.0100)
            & (frame["close_location"].astype(float) <= 0.35)
            & (frame["volume_ratio"].astype(float) >= 0.90)
        ).fillna(False),
        "short_failed_reclaim": (
            frame["weak_regime"]
            & (frame["gate_trend_strength_60"].astype(float) <= -0.0015)
            & (frame["close"].astype(float).shift(1) >= frame["session_vwap"].astype(float).shift(1))
            & (frame["close"].astype(float) < frame["session_vwap"].astype(float))
            & (frame["close"].astype(float) < frame["ema_fast"].astype(float))
            & (frame["wick_imbalance"].astype(float) <= -0.05)
            & (frame["close_location"].astype(float) <= 0.45)
        ).fillna(False),
        "short_rollover": (
            frame["weak_regime"]
            & (frame["gate_trend_strength_60"].astype(float) <= -0.0030)
            & (frame["atr_pct"].astype(float) >= 0.0040)
            & (
                frame["close"].astype(float)
                < frame["low"].rolling(12, min_periods=12).min().shift(1).astype(float)
            )
            & (frame["momentum_1"].astype(float) <= -0.0050)
            & (frame["volume_ratio"].astype(float) >= 1.0)
        ).fillna(False),
    }


def _short_backtest(
    frame: pd.DataFrame,
    signal_col: str,
    *,
    stop_pct: float,
    target_pct: float,
    max_hold_bars: int,
    min_hold_bars: int = 1,
) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    pending = False
    current: dict[str, Any] | None = None

    for i in range(96, len(frame) - 1):
        row = frame.iloc[i]

        if pending:
            next_row = frame.iloc[i + 1]
            entry_price = float(next_row["open"]) * (1 - COMMISSION_PCT - SLIPPAGE_PCT)
            current = {
                "entry_idx": i + 1,
                "entry_ts": int(next_row["ts"]),
                "entry_price": entry_price,
                "stop_price": entry_price * (1 + stop_pct),
                "target_price": entry_price * (1 - target_pct),
                "min_hold_bars": min_hold_bars,
                "max_hold_bars": max_hold_bars,
            }
            pending = False
            continue

        if current is not None:
            high_price = float(row["high"])
            low_price = float(row["low"])
            close_price = float(row["close"])
            bars_held = i - int(current["entry_idx"]) + 1
            exit_reason: str | None = None
            exit_price: float | None = None

            if high_price >= float(current["stop_price"]):
                exit_reason = "STOP_LOSS"
                exit_price = float(current["stop_price"]) * (1 + COMMISSION_PCT + SLIPPAGE_PCT)
            elif low_price <= float(current["target_price"]):
                exit_reason = "TAKE_PROFIT"
                exit_price = float(current["target_price"]) * (1 + COMMISSION_PCT + SLIPPAGE_PCT)
            elif bars_held >= int(current["max_hold_bars"]):
                exit_reason = "TIME_LIMIT"
                exit_price = close_price * (1 + COMMISSION_PCT + SLIPPAGE_PCT)
            elif (
                bars_held >= int(current["min_hold_bars"])
                and close_price > float(row["ema_fast"])
                and float(row["momentum_medium"]) > 0
            ):
                exit_reason = "BEAR_LOST"
                exit_price = close_price * (1 + COMMISSION_PCT + SLIPPAGE_PCT)

            if exit_reason and exit_price is not None:
                pnl_pct = (float(current["entry_price"]) - exit_price) / float(current["entry_price"])
                trades.append(
                    {
                        "entry_ts": int(current["entry_ts"]),
                        "pnl_pct": pnl_pct,
                        "bars_held": bars_held,
                        "exit_reason": exit_reason,
                    }
                )
                current = None
            continue

        if bool(row.get(signal_col, False)):
            pending = True

    return trades


def _summarize(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {"trades": 0, "net_pct": 0.0, "win_rate": 0.0, "avg_trade_pct": 0.0}
    pnls = pd.Series([trade["pnl_pct"] for trade in trades], dtype=float)
    return {
        "trades": int(len(trades)),
        "net_pct": float(pnls.sum()),
        "win_rate": float((pnls > 0).mean()),
        "avg_trade_pct": float(pnls.mean()),
    }


def run_research(history_days: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = sorted(UNIFORM_CACHE_DIR.glob(f"*_15m_{history_days}d_uniform_live.csv"))

    for path in paths:
        pair = path.name.split("_", 1)[0]
        df_raw = _load_uniform_history(path)
        frame = _build_feature_frame(df_raw)
        if frame.empty:
            continue

        signals = _short_signal_map(frame)
        recent_start = int(frame["ts"].iloc[-1]) - 60 * 24 * 60 * 60
        for signal_name, signal_mask in signals.items():
            frame[signal_name] = signal_mask
            params = {
                "short_breakdown": {"stop_pct": 0.0150, "target_pct": 0.0300, "max_hold_bars": 10},
                "short_failed_reclaim": {"stop_pct": 0.0120, "target_pct": 0.0250, "max_hold_bars": 8},
                "short_rollover": {"stop_pct": 0.0140, "target_pct": 0.0350, "max_hold_bars": 12},
            }[signal_name]
            trades = _short_backtest(frame, signal_name, **params)
            older60 = [trade for trade in trades if trade["entry_ts"] < recent_start]
            recent60 = [trade for trade in trades if trade["entry_ts"] >= recent_start]
            rows.append(
                {
                    "pair": pair,
                    "signal": signal_name,
                    "weak_regime_share": float(frame["weak_regime"].mean()),
                    **{f"full_{k}": v for k, v in _summarize(trades).items()},
                    **{f"older60_{k}": v for k, v in _summarize(older60).items()},
                    **{f"recent60_{k}": v for k, v in _summarize(recent60).items()},
                }
            )

    pair_frame = pd.DataFrame(rows).sort_values(["full_net_pct", "full_trades"], ascending=[False, False])
    survivor_mask = (
        (pair_frame["full_trades"] >= 8)
        & (pair_frame["full_net_pct"] > 0)
        & (pair_frame["older60_net_pct"] > 0)
        & (pair_frame["recent60_net_pct"] > 0)
    )
    survivors = pair_frame.loc[survivor_mask, ["pair", "signal", "full_trades", "full_net_pct"]]
    summary = {
        "history_days": history_days,
        "pairs_screened": int(pair_frame["pair"].nunique()) if not pair_frame.empty else 0,
        "signals_screened": int(len(pair_frame)),
        "survivor_count": int(len(survivors)),
        "survivors": survivors.to_dict(orient="records"),
        "note": (
            "Features are built from the densified 15m uniform caches using the standard trend_gate "
            "construction so the weak-regime short overlay can be compared on the same true clock."
        ),
    }
    return pair_frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen weak-regime short overlays using local densified 120d caches.")
    parser.add_argument("--history-days", type=int, default=120)
    parser.add_argument("--summary-json", default="results/weak_regime_short_summary.json")
    parser.add_argument("--pairs-csv", default="results/weak_regime_short_pairs.csv")
    args = parser.parse_args()

    pair_frame, summary = run_research(args.history_days)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pair_frame.to_csv(args.pairs_csv, index=False)
    with open(args.summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("WEAK REGIME SHORT RESEARCH")
    print(f"pairs={summary['pairs_screened']} signals={summary['signals_screened']} survivors={summary['survivor_count']}")
    if not pair_frame.empty:
        print(pair_frame.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
