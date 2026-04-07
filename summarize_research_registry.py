from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from research_pair_registry import PAIR_RESEARCH_REGISTRY, active_pairs, dropped_pairs, shadow_pairs


RESULTS_DIR = Path("results")


def load_tuner_results() -> pd.DataFrame:
    frames = []
    for path_str in [
        "results/rolling_pair_tuner_best.csv",
        "results/rolling_pair_tuner_best_60d_expansion.csv",
    ]:
        path = Path(path_str)
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["pair", "best_net"], ascending=[True, False])
    return merged.drop_duplicates("pair", keep="first").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the current research-agent pair registry.")
    parser.add_argument("--tuner-csv", default="results/rolling_pair_tuner_best.csv")
    parser.add_argument("--walkforward-csv", default="results/live_walkforward_best_top12.csv")
    parser.add_argument("--summary-json", default="results/research_registry_summary.json")
    parser.add_argument("--summary-md", default="results/research_registry_summary.md")
    args = parser.parse_args()

    tuner = load_tuner_results()
    walkforward = pd.read_csv(args.walkforward_csv) if Path(args.walkforward_csv).exists() else pd.DataFrame()

    active = tuner[tuner["pair"].isin(active_pairs())].copy() if not tuner.empty else pd.DataFrame()
    shadow = tuner[tuner["pair"].isin(shadow_pairs())].copy() if not tuner.empty else pd.DataFrame()

    wf_active = walkforward[walkforward["pair"].isin(active_pairs())].copy() if not walkforward.empty else pd.DataFrame()
    wf_shadow = walkforward[walkforward["pair"].isin(shadow_pairs())].copy() if not walkforward.empty else pd.DataFrame()

    payload = {
        "active_pairs": list(active_pairs()),
        "shadow_pairs": list(shadow_pairs()),
        "dropped_pairs": list(dropped_pairs()),
        "active_oos_trades": int(active["best_trades"].sum()) if not active.empty else 0,
        "active_oos_net": float(active["best_net"].sum()) if not active.empty else 0.0,
        "active_baseline_oos_trades": int(active["baseline_trades"].sum()) if not active.empty else 0,
        "active_baseline_oos_net": float(active["baseline_net"].sum()) if not active.empty else 0.0,
        "active_oos_net_delta": (
            float(active["best_net"].sum()) - float(active["baseline_net"].sum())
            if not active.empty
            else 0.0
        ),
        "active_pairs_detail": active[
            [
                "pair",
                "construction",
                "baseline_candidate",
                "baseline_trades",
                "baseline_net",
                "best_candidate",
                "best_trades",
                "best_net",
                "net_delta_vs_base",
            ]
        ].to_dict(orient="records")
        if not active.empty
        else [],
        "shadow_pairs_detail": shadow[
            ["pair", "construction", "best_candidate", "best_trades", "best_net", "net_delta_vs_base"]
        ].to_dict(orient="records")
        if not shadow.empty
        else [],
        "walkforward_active": wf_active.to_dict(orient="records") if not wf_active.empty else [],
        "walkforward_shadow": wf_shadow.to_dict(orient="records") if not wf_shadow.empty else [],
    }

    Path(args.summary_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Research Registry Summary",
        "",
        "## Active Set",
        f"- Active pairs: {', '.join(payload['active_pairs']) or 'none'}",
        f"- Rolling OOS net: {payload['active_oos_net']:+.3%} across {payload['active_oos_trades']} trades",
        f"- Same-set untuned baseline: {payload['active_baseline_oos_net']:+.3%} across {payload['active_baseline_oos_trades']} trades",
        f"- Net improvement vs untuned baseline: {payload['active_oos_net_delta']:+.3%}",
        "",
        "## Active Detail",
    ]
    for row in payload["active_pairs_detail"]:
        lines.append(
            f"- {row['pair']}: {row['construction']} | "
            f"base={row['baseline_candidate']} {float(row['baseline_net']):+.3%}/{int(row['baseline_trades'])} | "
            f"best={row['best_candidate']} {float(row['best_net']):+.3%}/{int(row['best_trades'])} | "
            f"delta={float(row['net_delta_vs_base']):+.3%}"
        )
    lines.extend(
        [
            "",
            "## Shadow Set",
            f"- Shadow pairs: {', '.join(payload['shadow_pairs']) or 'none'}",
        ]
    )
    for row in payload["shadow_pairs_detail"]:
        lines.append(
            f"- {row['pair']}: {row['construction']} | {row['best_candidate']} | "
            f"oos={float(row['best_net']):+.3%}/{int(row['best_trades'])} | "
            f"delta={float(row['net_delta_vs_base']):+.3%}"
        )
    lines.extend(
        [
            "",
            "## Dropped",
            f"- Dropped pairs: {', '.join(payload['dropped_pairs']) or 'none'}",
            "",
            "## Notes",
            "- Active set is intentionally narrower than the cached registry.",
            "- Promotion is driven by rolling OOS evidence, not just full-sample cached PnL.",
        ]
    )
    Path(args.summary_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Active OOS net: {payload['active_oos_net']:+.3%} across {payload['active_oos_trades']} trades")
    print(f"Active pairs: {', '.join(payload['active_pairs']) or 'none'}")
    print(f"Shadow pairs: {', '.join(payload['shadow_pairs']) or 'none'}")
    print(f"Dropped pairs: {', '.join(payload['dropped_pairs']) or 'none'}")


if __name__ == "__main__":
    main()
