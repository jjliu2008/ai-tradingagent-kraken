from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from pair_strategy_registry import PAIR_STRATEGY_REGISTRY, frozen_pairs


RESULTS_DIR = Path("results")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _pair_trade_stats(trades: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if trades.empty:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for pair, group in trades.groupby("pair"):
        pnls = group["pnl_pct"].astype(float)
        reasons = Counter(str(value) for value in group["exit_reason"])
        avg_mfe = float(group["mfe_pct"].astype(float).mean()) if "mfe_pct" in group.columns else 0.0
        capture = float(pnls.mean() / avg_mfe) if avg_mfe > 0 else 0.0
        out[str(pair)] = {
            "trades": int(len(group)),
            "net": float(pnls.sum()),
            "win": float((pnls > 0).mean()),
            "avg": float(pnls.mean()),
            "avg_mfe": avg_mfe,
            "capture_ratio": capture,
            "avg_hold": float(group["bars_held"].astype(float).mean()),
            "exit_reasons": dict(reasons),
            "top_exit_reason": reasons.most_common(1)[0][0] if reasons else "",
        }
    return out


def _generic_pair_stats(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    per_pair = payload.get("per_pair") or {}
    out: dict[str, dict[str, Any]] = {}
    for pair, stats in per_pair.items():
        if not stats:
            continue
        out[str(pair)] = {
            "trades": int(stats.get("n_trades", 0) or 0),
            "net_usd": float(stats.get("total_pnl_usd", 0.0) or 0.0),
            "win": float(stats.get("win_rate", 0.0) or 0.0),
            "pf": float(stats.get("profit_factor", 0.0) or 0.0),
            "max_dd_usd": float(stats.get("max_dd_usd", 0.0) or 0.0),
            "exit_reasons": stats.get("exit_reasons") or {},
        }
    return out


def _best_map(frame: pd.DataFrame, pair_col: str = "pair") -> dict[str, dict[str, Any]]:
    if frame.empty:
        return {}
    return {str(row[pair_col]): row.to_dict() for _, row in frame.iterrows()}


def _merge_best_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    usable = [frame for frame in frames if not frame.empty]
    if not usable:
        return pd.DataFrame()
    merged = pd.concat(usable, ignore_index=True)
    sort_cols: list[str] = []
    ascending: list[bool] = []
    if "best_score" in merged.columns:
        sort_cols.append("best_score")
        ascending.append(False)
    elif "score" in merged.columns:
        sort_cols.append("score")
        ascending.append(False)
    if "best_net" in merged.columns:
        sort_cols.append("best_net")
        ascending.append(False)
    elif "test_net" in merged.columns:
        sort_cols.append("test_net")
        ascending.append(False)
    if not sort_cols:
        return merged.drop_duplicates("pair", keep="first")
    merged = merged.sort_values(sort_cols, ascending=ascending)
    return merged.drop_duplicates("pair", keep="first").reset_index(drop=True)


def _hypotheses(top_exit_reason: str, generic_win: float, generic_pf: float) -> list[str]:
    if top_exit_reason == "TREND_LOST":
        return ["stronger_trend_gate", "higher_close_location", "faster_exit_breakout"]
    if top_exit_reason == "TIME_LIMIT":
        return ["atr30_only", "baseline_or_atr30", "shorter_hold_breakout"]
    if top_exit_reason == "STOP_LOSS" and generic_win < 0.25 and generic_pf < 0.8:
        return ["mean_reversion_after_spike", "pullback_resume_30m", "anti_exhaustion_breakout"]
    return ["tight_parameter_sweep", "alternative_family_scan", "drop_if_next_oos_fails"]


def _questions(pair: str, top_exit_reason: str, generic_win: float) -> list[str]:
    base = [
        f"Does {pair} continue cleanly after breakout, or fade within 1-3 bars?",
        f"Is 30m structure cleaner than 15m for {pair}?",
    ]
    if top_exit_reason == "TREND_LOST":
        base.append(f"Is {pair} entering too early before real trend confirmation?")
    elif top_exit_reason == "TIME_LIMIT":
        base.append(f"Is {pair} moving in the right direction but too slowly to hit current targets?")
    elif top_exit_reason == "STOP_LOSS" or generic_win < 0.25:
        base.append(f"Does {pair} behave more like a mean-reversion market than a continuation market?")
    return base


def _verdict(
    pair: str,
    frozen: bool,
    cached_row: dict[str, Any] | None,
    wf_row: dict[str, Any] | None,
    registry_stats: dict[str, Any] | None,
    generic_stats: dict[str, Any] | None,
    reinvent_row: dict[str, Any] | None,
    mean_reversion_row: dict[str, Any] | None,
    tuner_row: dict[str, Any] | None,
) -> tuple[str, str]:
    cached_net = float((cached_row or {}).get("net", 0.0) or 0.0)
    cached_trades = int((cached_row or {}).get("trades", 0) or 0)
    wf_test_net = float((wf_row or {}).get("test_net", 0.0) or 0.0)
    wf_test_trades = int((wf_row or {}).get("test_trades", 0) or 0)
    generic_trades = int((generic_stats or {}).get("trades", 0) or 0)
    generic_win = float((generic_stats or {}).get("win", 0.0) or 0.0)
    generic_pf = float((generic_stats or {}).get("pf", 0.0) or 0.0)
    reinvent_test_trades = int((reinvent_row or {}).get("test_trades", 0) or 0)
    reinvent_test_net = float((reinvent_row or {}).get("test_net", 0.0) or 0.0)
    mr_test_trades = int((mean_reversion_row or {}).get("test_trades", 0) or 0)
    mr_test_net = float((mean_reversion_row or {}).get("test_net", 0.0) or 0.0)
    tuner_delta = float((tuner_row or {}).get("net_delta_vs_base", 0.0) or 0.0)
    tuner_best_net = float((tuner_row or {}).get("best_net", 0.0) or 0.0)
    tuner_best_trades = int((tuner_row or {}).get("best_trades", 0) or 0)

    if frozen:
        if cached_net > 0:
            return "KEEP", "Frozen benchmark remains the anchor until a challenger beats it repeatedly."
        return "TUNE", "Frozen benchmark is weak on the latest slice but still the best validated anchor."

    if tuner_best_trades >= 2 and tuner_best_net > 0 and tuner_delta > 0:
        return "KEEP", "Rolling OOS tuning found a better variant than the current baseline construction."

    if wf_test_trades >= 2 and wf_test_net > 0:
        return "KEEP", "Positive out-of-sample result with enough trades to justify keeping the pair active."

    if wf_test_trades == 1 and wf_test_net > 0:
        return "TUNE", "Out-of-sample is positive but only on one trade; needs more evidence before promotion."

    if wf_test_trades >= 1 and wf_test_net < 0:
        if cached_net > 0 and cached_trades >= 3:
            return "TUNE", "Cached edge exists, but the current out-of-sample slice is negative; refine before reinvention."
        if generic_trades >= 5 and generic_win < 0.25 and generic_pf < 0.8:
            return "REINVENT", "Both generic and out-of-sample results are weak; the family likely mismatches the pair."
        return "REINVENT", "Current family does not hold out of sample and needs a structural rethink."

    if (
        reinvent_test_trades >= 2
        and reinvent_test_net < 0
        and mr_test_trades == 0
        and generic_trades >= 5
        and generic_win < 0.25
    ):
        return "DROP", "Continuation reinvention failed and even mean-reversion did not produce enough tradable evidence."

    if cached_net > 0 and cached_trades >= 3:
        if generic_trades >= 5 and generic_win < 0.25 and generic_pf < 0.8:
            return "REINVENT", "Cached winner exists, but generic evidence shows this pair punishes the current continuation family."
        return "TUNE", "Positive cached evidence exists, but it has not yet survived a broad walk-forward test."

    if generic_trades >= 5 and generic_win < 0.25 and generic_pf < 0.8:
        return "DROP", "The pair is consistently damaging under current families and lacks a validated alternative."

    return "DROP", "There is not enough positive evidence to justify keeping this pair in active research."


def _priority(verdict: str, wf_row: dict[str, Any] | None, generic_stats: dict[str, Any] | None) -> int:
    base = {"REINVENT": 90, "TUNE": 60, "KEEP": 20, "DROP": 10}.get(verdict, 0)
    generic_trades = int((generic_stats or {}).get("trades", 0) or 0)
    wf_test_trades = int((wf_row or {}).get("test_trades", 0) or 0)
    return base + generic_trades + wf_test_trades


def build_diagnostics() -> dict[str, Any]:
    cached_best = _load_csv(RESULTS_DIR / "pair_strategy_best_ge3.csv")
    walkforward_best = _load_csv(RESULTS_DIR / "live_walkforward_best_top30.csv")
    if walkforward_best.empty:
        walkforward_best = _load_csv(RESULTS_DIR / "live_walkforward_best_top12.csv")
    registry_trades = _load_csv(RESULTS_DIR / "pair_strategy_registry_trades.csv")
    universe69 = _load_json(RESULTS_DIR / "backtest_universe69.json")
    reinvent_best = _load_csv(RESULTS_DIR / "reinvent_pair_best.csv")
    mean_reversion_best = _load_csv(RESULTS_DIR / "mean_reversion_pair_best.csv")
    tuner_best = _merge_best_frames(
        _load_csv(RESULTS_DIR / "rolling_pair_tuner_best.csv"),
        _load_csv(RESULTS_DIR / "rolling_pair_tuner_best_60d_expansion.csv"),
    )

    cached_map = _best_map(cached_best)
    wf_map = _best_map(walkforward_best)
    registry_map = _pair_trade_stats(registry_trades)
    generic_map = _generic_pair_stats(universe69)
    reinvent_map = _best_map(reinvent_best)
    mean_reversion_map = _best_map(mean_reversion_best)
    tuner_map = _best_map(tuner_best)

    pairs = sorted(
        set(PAIR_STRATEGY_REGISTRY)
        | set(cached_map)
        | set(wf_map)
        | set(registry_map)
        | set(generic_map)
        | set(reinvent_map)
        | set(mean_reversion_map)
        | set(tuner_map)
    )

    diagnostics: list[dict[str, Any]] = []
    verdict_counter = Counter()
    for pair in pairs:
        frozen = pair in frozen_pairs()
        current_strategy = PAIR_STRATEGY_REGISTRY.get(pair)
        cached_row = cached_map.get(pair)
        wf_row = wf_map.get(pair)
        reg_stats = registry_map.get(pair)
        gen_stats = generic_map.get(pair)
        reinvent_row = reinvent_map.get(pair)
        mean_reversion_row = mean_reversion_map.get(pair)
        tuner_row = tuner_map.get(pair)

        verdict, diagnosis = _verdict(
            pair,
            frozen,
            cached_row,
            wf_row,
            reg_stats,
            gen_stats,
            reinvent_row,
            mean_reversion_row,
            tuner_row,
        )
        verdict_counter[verdict] += 1

        top_exit_reason = ""
        if reg_stats:
            top_exit_reason = str(reg_stats.get("top_exit_reason") or "")
        elif gen_stats:
            exits = gen_stats.get("exit_reasons") or {}
            if exits:
                top_exit_reason = max(exits.items(), key=lambda kv: kv[1])[0]

        generic_win = float((gen_stats or {}).get("win", 0.0) or 0.0)
        generic_pf = float((gen_stats or {}).get("pf", 0.0) or 0.0)

        diagnostics.append(
            {
                "pair": pair,
                "frozen": frozen,
                "current_strategy": current_strategy or (wf_row or {}).get("construction") or (cached_row or {}).get("construction"),
                "verdict": verdict,
                "priority": _priority(verdict, wf_row, gen_stats),
                "diagnosis": diagnosis,
                "cached": cached_row or {},
                "walkforward": wf_row or {},
                "reinvention": reinvent_row or {},
                "mean_reversion": mean_reversion_row or {},
                "tuner": tuner_row or {},
                "registry_stats": reg_stats or {},
                "generic_universe": gen_stats or {},
                "top_exit_reason": top_exit_reason,
                "next_hypotheses": _hypotheses(top_exit_reason, generic_win, generic_pf),
                "research_questions": _questions(pair, top_exit_reason, generic_win),
            }
        )

    diagnostics.sort(key=lambda item: (-item["priority"], item["pair"]))

    portfolio_summary = {
        "registry_pairs": list(PAIR_STRATEGY_REGISTRY.keys()),
        "registry_trade_count": int(len(registry_trades)) if not registry_trades.empty else 0,
        "registry_net_pct": float(registry_trades["pnl_pct"].astype(float).sum()) if not registry_trades.empty else 0.0,
        "registry_win_rate": float((registry_trades["pnl_pct"].astype(float) > 0).mean()) if not registry_trades.empty else 0.0,
        "generic_universe_trade_count": int((universe69.get("combined") or {}).get("n_trades", 0) or 0),
        "generic_universe_total_pnl_usd": float((universe69.get("combined") or {}).get("total_pnl_usd", 0.0) or 0.0),
        "walkforward_survivors": str(
            (
                _load_csv(RESULTS_DIR / "live_walkforward_summary_top30.csv").iloc[0].get("survivor_pairs")
            )
            if (RESULTS_DIR / "live_walkforward_summary_top30.csv").exists()
            else (
                _load_csv(RESULTS_DIR / "live_walkforward_summary_top12.csv").iloc[0].get("survivor_pairs")
                if (RESULTS_DIR / "live_walkforward_summary_top12.csv").exists()
                else ""
            )
        ),
        "verdict_counts": dict(verdict_counter),
    }

    return {"portfolio_summary": portfolio_summary, "pairs": diagnostics}


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["portfolio_summary"]
    lines = [
        "# Pair Diagnostics",
        "",
        "## Portfolio Summary",
        f"- Registry pairs: {', '.join(summary['registry_pairs'])}",
        f"- Registry net: {summary['registry_net_pct']:.3%} across {summary['registry_trade_count']} trades",
        f"- Registry win rate: {summary['registry_win_rate']:.1%}",
        f"- Generic universe total PnL: ${summary['generic_universe_total_pnl_usd']:+.2f} across {summary['generic_universe_trade_count']} trades",
        f"- Walk-forward survivors: {summary['walkforward_survivors'] or 'none'}",
        f"- Reinvention results loaded: {str((RESULTS_DIR / 'reinvent_pair_best.csv').exists())}",
        f"- Rolling tuner results loaded: {str((RESULTS_DIR / 'rolling_pair_tuner_best.csv').exists())}",
        f"- Verdict counts: {summary['verdict_counts']}",
        "",
        "## Pair Verdicts",
    ]

    for item in payload["pairs"]:
        cached = item["cached"]
        wf = item["walkforward"]
        reinvent = item["reinvention"]
        mean_reversion = item["mean_reversion"]
        tuner = item["tuner"]
        generic = item["generic_universe"]
        lines.extend(
            [
                "",
                f"### {item['pair']} - {item['verdict']}",
                f"- Current strategy: {item['current_strategy'] or 'none'}",
                f"- Diagnosis: {item['diagnosis']}",
                f"- Cached best: {cached.get('construction', 'n/a')} | trades={int(cached.get('trades', 0) or 0)} | net={float(cached.get('net', 0.0) or 0.0):+.3%}",
                f"- Walk-forward: {wf.get('construction', 'n/a')} | test trades={int(wf.get('test_trades', 0) or 0)} | test net={float(wf.get('test_net', 0.0) or 0.0):+.3%}",
                f"- Reinvention: {reinvent.get('family', 'n/a')} | test trades={int(reinvent.get('test_trades', 0) or 0)} | test net={float(reinvent.get('test_net', 0.0) or 0.0):+.3%}",
                f"- Mean reversion: {mean_reversion.get('family', 'n/a')} | test trades={int(mean_reversion.get('test_trades', 0) or 0)} | test net={float(mean_reversion.get('test_net', 0.0) or 0.0):+.3%}",
                f"- Tuner best: {tuner.get('best_candidate', 'n/a')} | trades={int(tuner.get('best_trades', 0) or 0)} | net={float(tuner.get('best_net', 0.0) or 0.0):+.3%}",
                f"- Generic universe: trades={int(generic.get('trades', 0) or 0)} | net_usd={float(generic.get('net_usd', 0.0) or 0.0):+.2f}",
                f"- Top exit reason: {item['top_exit_reason'] or 'n/a'}",
                f"- Next hypotheses: {', '.join(item['next_hypotheses'])}",
                "- Research questions:",
            ]
        )
        for question in item["research_questions"]:
            lines.append(f"  - {question}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic pair diagnostics and research verdicts.")
    parser.add_argument("--json-out", default="results/pair_diagnostics.json")
    parser.add_argument("--md-out", default="results/pair_diagnostics.md")
    args = parser.parse_args()

    payload = build_diagnostics()
    json_out = Path(args.json_out)
    md_out = Path(args.md_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_out.write_text(render_markdown(payload), encoding="utf-8")

    summary = payload["portfolio_summary"]
    print(f"Registry net: {summary['registry_net_pct']:.3%} across {summary['registry_trade_count']} trades")
    print(f"Walk-forward survivors: {summary['walkforward_survivors'] or 'none'}")
    print(f"Verdicts: {summary['verdict_counts']}")
    print(f"JSON -> {json_out}")
    print(f"MD   -> {md_out}")


if __name__ == "__main__":
    main()
