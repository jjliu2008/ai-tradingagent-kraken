from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import backtest as bt
import research_runtime as rr
from book_optimizer import _portfolio_score
from research_pair_registry import PAIR_RESEARCH_REGISTRY, PairResearchPlan, active_pairs


RESULTS_DIR = Path("results")
CACHE_DIR = Path("data_cache_walkforward")
COMMISSION_PCT = 0.0026
SLIPPAGE_PCT = 0.0005


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(candidate.get("robustness_score", 0.0)),
        float(candidate.get("older60_net_pct", 0.0)),
        float(candidate.get("recent60_net_pct", 0.0)),
        float(candidate.get("full_net_pct", 0.0)),
    )


def select_best_non_active_candidates(
    candidates: list[dict[str, Any]],
    active_pair_names: set[str],
) -> list[dict[str, Any]]:
    best_by_pair: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        pair = str(candidate.get("pair", "")).upper()
        if not pair or pair in active_pair_names:
            continue
        enriched = {**candidate, "pair": pair}
        current = best_by_pair.get(pair)
        if current is None or _candidate_sort_key(enriched) > _candidate_sort_key(current):
            best_by_pair[pair] = enriched
    return sorted(best_by_pair.values(), key=_candidate_sort_key, reverse=True)


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return list(payload.get("candidates", []))
    return list(payload)


def _load_uniform_history(pair: str, history_days: int) -> pd.DataFrame:
    path = CACHE_DIR / f"{pair}_15m_{history_days}d_uniform_live.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [column.lower() for column in df.columns]
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    if "vwap_k" not in df.columns:
        df["vwap_k"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    if "count" not in df.columns:
        df["count"] = 1
    return df.sort_values("ts").reset_index(drop=True)


def _summarize(trades: list[bt.BacktestTrade]) -> dict[str, Any]:
    if not trades:
        return {
            "trades": 0,
            "net_pct": 0.0,
            "win_rate": 0.0,
            "avg_trade_pct": 0.0,
            "max_dd_pct": 0.0,
        }
    pnls = pd.Series([trade.pnl_pct for trade in trades], dtype=float)
    equity = (1.0 + pnls).cumprod()
    return {
        "trades": int(len(trades)),
        "net_pct": float(pnls.sum()),
        "win_rate": float((pnls > 0).mean()),
        "avg_trade_pct": float(pnls.mean()),
        "max_dd_pct": float((equity / equity.cummax() - 1.0).min()),
    }


def _window_summaries(trades: list[bt.BacktestTrade], max_ts: int) -> dict[str, dict[str, Any]]:
    recent30_start = max_ts - 30 * 24 * 60 * 60 if max_ts else 0
    recent14_start = max_ts - 14 * 24 * 60 * 60 if max_ts else 0
    split_ts = max_ts - 60 * 24 * 60 * 60 if max_ts else 0
    return {
        "full": _summarize(trades),
        "older60": _summarize([trade for trade in trades if trade.entry_ts < split_ts]),
        "recent60": _summarize([trade for trade in trades if trade.entry_ts >= split_ts]),
        "recent30": _summarize([trade for trade in trades if trade.entry_ts >= recent30_start]),
        "recent14": _summarize([trade for trade in trades if trade.entry_ts >= recent14_start]),
    }


def _run_portfolio(plans: list[PairResearchPlan], history_days: int) -> dict[str, dict[str, Any]]:
    all_trades: list[bt.BacktestTrade] = []
    max_ts = 0
    for plan in plans:
        df_raw = _load_uniform_history(plan.pair, history_days)
        if df_raw.empty:
            continue
        max_ts = max(max_ts, int(df_raw["ts"].iloc[-1]))
        frame = rr.build_research_frame(df_raw, plan)
        trades = bt.run_backtest_frame(
            pair=plan.pair,
            df=frame,
            config=rr.config_for_plan(plan),
            commission_pct=COMMISSION_PCT,
            slippage_pct=SLIPPAGE_PCT,
            construction=plan.construction,
        )
        all_trades.extend(trades)
    return _window_summaries(all_trades, max_ts)


def _plan_from_candidate(candidate: dict[str, Any]) -> PairResearchPlan:
    return PairResearchPlan(
        pair=str(candidate["pair"]).upper(),
        construction=str(candidate["construction"]),
        entry_filter=str(candidate["entry_filter"]),
        exit_profile=str(candidate["exit_profile"]),
        status="candidate",
        note="Portfolio impact candidate from older60 screener.",
    )


def _candidate_has_required_history(pair: str) -> bool:
    return all(
        (CACHE_DIR / f"{pair}_15m_{history_days}d_uniform_live.csv").exists()
        for history_days in (120, 60)
    )


def _compact_summary(summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "full_net_pct": summary["full"]["net_pct"],
        "full_trades": summary["full"]["trades"],
        "full_max_dd_pct": summary["full"]["max_dd_pct"],
        "older60_net_pct": summary["older60"]["net_pct"],
        "recent60_net_pct": summary["recent60"]["net_pct"],
        "recent30_net_pct": summary["recent30"]["net_pct"],
        "recent14_net_pct": summary["recent14"]["net_pct"],
    }


def rank_candidate_portfolio_impacts(
    candidate_path: Path,
    top_n: int,
) -> dict[str, Any]:
    active_roster = [PAIR_RESEARCH_REGISTRY[pair] for pair in active_pairs()]
    active_pair_names = {plan.pair for plan in active_roster}

    baseline120 = _run_portfolio(active_roster, history_days=120)
    baseline60 = _run_portfolio(active_roster, history_days=60)
    baseline_score = _portfolio_score(baseline120, baseline60)

    selected = select_best_non_active_candidates(_load_candidates(candidate_path), active_pair_names=active_pair_names)
    ranked_rows: list[dict[str, Any]] = []

    for candidate in selected:
        candidate_plan = _plan_from_candidate(candidate)
        if not _candidate_has_required_history(candidate_plan.pair):
            continue

        add_roster = [*active_roster, candidate_plan]
        add120 = _run_portfolio(add_roster, history_days=120)
        add60 = _run_portfolio(add_roster, history_days=60)
        add_score = _portfolio_score(add120, add60)

        replacement_rows: list[dict[str, Any]] = []
        for incumbent in active_roster:
            replace_roster = [plan for plan in active_roster if plan.pair != incumbent.pair]
            replace_roster.append(candidate_plan)
            replace120 = _run_portfolio(replace_roster, history_days=120)
            replace60 = _run_portfolio(replace_roster, history_days=60)
            replace_score = _portfolio_score(replace120, replace60)
            replacement_rows.append(
                {
                    "replace_pair": incumbent.pair,
                    "score": replace_score,
                    "score_delta": replace_score - baseline_score,
                    "summary120": _compact_summary(replace120),
                    "summary60": _compact_summary(replace60),
                }
            )

        best_replacement = max(replacement_rows, key=lambda row: row["score"])
        ranked_rows.append(
            {
                "pair": candidate_plan.pair,
                "construction": candidate_plan.construction,
                "entry_filter": candidate_plan.entry_filter,
                "exit_profile": candidate_plan.exit_profile,
                "robustness_score": candidate.get("robustness_score", 0.0),
                "candidate_metrics": {
                    "full_net_pct": candidate.get("full_net_pct", 0.0),
                    "older60_net_pct": candidate.get("older60_net_pct", 0.0),
                    "recent60_net_pct": candidate.get("recent60_net_pct", 0.0),
                    "full_trades": candidate.get("full_trades", 0),
                },
                "add_as_new_pair": {
                    "score": add_score,
                    "score_delta": add_score - baseline_score,
                    "summary120": _compact_summary(add120),
                    "summary60": _compact_summary(add60),
                },
                "best_replacement": best_replacement,
            }
        )

    ranked_rows.sort(
        key=lambda row: (
            row["add_as_new_pair"]["score_delta"],
            row["best_replacement"]["score_delta"],
            row["robustness_score"],
        ),
        reverse=True,
    )

    return {
        "candidate_source": str(candidate_path).replace("\\", "/"),
        "baseline": {
            "pairs": [plan.pair for plan in active_roster],
            "score": baseline_score,
            "summary120": _compact_summary(baseline120),
            "summary60": _compact_summary(baseline60),
        },
        "ranked_candidates": ranked_rows[:top_n],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank non-active pair candidates by portfolio add/replace impact.")
    parser.add_argument("--candidate-json", default="results/latest/older60_candidates.json")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--out-json", default="results/latest/portfolio_candidate_rankings.json")
    parser.add_argument("--out-csv", default="results/latest/portfolio_candidate_rankings.csv")
    args = parser.parse_args()

    result = rank_candidate_portfolio_impacts(Path(args.candidate_json), top_n=args.top_n)

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    csv_rows: list[dict[str, Any]] = []
    for row in result["ranked_candidates"]:
        csv_rows.append(
            {
                "pair": row["pair"],
                "construction": row["construction"],
                "entry_filter": row["entry_filter"],
                "exit_profile": row["exit_profile"],
                "robustness_score": row["robustness_score"],
                "candidate_full_net_pct": row["candidate_metrics"]["full_net_pct"],
                "candidate_older60_net_pct": row["candidate_metrics"]["older60_net_pct"],
                "candidate_recent60_net_pct": row["candidate_metrics"]["recent60_net_pct"],
                "add_score_delta": row["add_as_new_pair"]["score_delta"],
                "add_full120_net_pct": row["add_as_new_pair"]["summary120"]["full_net_pct"],
                "add_older60_net_pct": row["add_as_new_pair"]["summary120"]["older60_net_pct"],
                "best_replace_pair": row["best_replacement"]["replace_pair"],
                "best_replace_score_delta": row["best_replacement"]["score_delta"],
                "best_replace_full120_net_pct": row["best_replacement"]["summary120"]["full_net_pct"],
                "best_replace_older60_net_pct": row["best_replacement"]["summary120"]["older60_net_pct"],
            }
        )
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)

    print(
        f"portfolio_candidate_ranker: baseline={result['baseline']['score']:+.6f} "
        f"ranked={len(result['ranked_candidates'])} -> {out_json}"
    )


if __name__ == "__main__":
    main()
