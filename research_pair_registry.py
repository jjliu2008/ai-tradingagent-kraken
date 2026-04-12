from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PairResearchPlan:
    pair: str
    construction: str
    entry_filter: str
    exit_profile: str
    status: str
    note: str


# Current research-agent recommendation set.
# Main book is the last fully validated gated 5-pair portfolio.
PAIR_RESEARCH_REGISTRY: dict[str, PairResearchPlan] = {
    "GIGAUSD": PairResearchPlan(
        pair="GIGAUSD",
        construction="tc15_tighter_volume_cap",
        entry_filter="base",
        exit_profile="base",
        status="active_frozen",
        note="Anchor baseline; still the most validated single-pair strategy and kept frozen in the main book.",
    ),
    "ZECUSD": PairResearchPlan(
        pair="ZECUSD",
        construction="union_volhi",
        entry_filter="close80_volcap60",
        exit_profile="runner",
        status="active_experimental",
        note="Promoted by the densified book optimizer as the strongest fifth-slot addition to the repaired main book.",
    ),
    "XDGUSD": PairResearchPlan(
        pair="XDGUSD",
        construction="baseline_or_atr30",
        entry_filter="volcap45",
        exit_profile="medium",
        status="shadow",
        note="Pre-gate positive, but the shared portfolio regime gate strips too much edge; demoted behind BABYUSD.",
    ),
    "FHEUSD": PairResearchPlan(
        pair="FHEUSD",
        construction="trend_gate",
        entry_filter="gate50_vwap3",
        exit_profile="runner",
        status="active_experimental",
        note="Promoted back into the live research book after the portfolio-aware backtest showed clean improvement versus the current baseline.",
    ),
    "KERNELUSD": PairResearchPlan(
        pair="KERNELUSD",
        construction="union_closehi",
        entry_filter="close80_volcap60",
        exit_profile="runner",
        status="active",
        note="Tightened entry filter to close80_volcap60 after densification: recent-30d flipped +0.63% vs -1.48% on base filter.",
    ),
    "HOUSEUSD": PairResearchPlan(
        pair="HOUSEUSD",
        construction="trend_gate",
        entry_filter="gate50_vwap3",
        exit_profile="runner",
        status="active",
        note="Repaired by the active-pair sweep and promoted back into the core book after strong multi-window densified portfolio impact.",
    ),
    "BABYUSD": PairResearchPlan(
        pair="BABYUSD",
        construction="tc15_cap_or_mb60",
        entry_filter="base",
        exit_profile="fast",
        status="active",
        note="Best replacement for XDGUSD under the shared portfolio gate; fast exit profile preserved from the last validated 5-pair book.",
    ),
    "BERAUSD": PairResearchPlan(
        pair="BERAUSD",
        construction="vst60_only",
        entry_filter="gate50_vwap3",
        exit_profile="runner",
        status="shadow",
        note="Improved after repair, but still not strong enough across the full 120d standardized window.",
    ),
    "PARTIUSD": PairResearchPlan(
        pair="PARTIUSD",
        construction="trend_gate",
        entry_filter="close80_vwap3",
        exit_profile="fast",
        status="shadow",
        note="Repair helped recent performance, but the pair is still negative on the full 120d standardized window.",
    ),
    "APTUSD": PairResearchPlan(
        pair="APTUSD",
        construction="vst60_only",
        entry_filter="base",
        exit_profile="runner",
        status="shadow",
        note="Still positive on densified data, but displaced by stronger multi-window roster additions in the optimizer pass.",
    ),
    "NIGHTUSD": PairResearchPlan(
        pair="NIGHTUSD",
        construction="baseline_or_atr30",
        entry_filter="base",
        exit_profile="base",
        status="shadow",
        note="Cached positive but no convincing OOS activity yet.",
    ),
    "TAOUSD": PairResearchPlan(
        pair="TAOUSD",
        construction="atr30_only",
        entry_filter="score55",
        exit_profile="runner",
        status="shadow",
        note="Tuner improved the pair, but only on one OOS trade.",
    ),
    "SUIUSD": PairResearchPlan(
        pair="SUIUSD",
        construction="union_volhi",
        entry_filter="close80_volcap60",
        exit_profile="runner",
        status="shadow",
        note="Demoted to shadow pending OOS confirmation. Densified data looks clean but has not proven itself out of sample yet.",
    ),
    "ADAUSD": PairResearchPlan(
        pair="ADAUSD",
        construction="baseline_or_tc30",
        entry_filter="base",
        exit_profile="base",
        status="shadow",
        note="Current family remains unconfirmed out of sample.",
    ),
    "ETHUSD": PairResearchPlan(
        pair="ETHUSD",
        construction="tc30_only",
        entry_filter="base",
        exit_profile="base",
        status="shadow",
        note="Positive cached edge exists, but OOS evidence is still too thin.",
    ),
    "COQUSD": PairResearchPlan(
        pair="COQUSD",
        construction="baseline_mb60",
        entry_filter="base",
        exit_profile="base",
        status="drop",
        note="Continuation reinvention failed and mean-reversion was too sparse to trust.",
    ),
    "HYPEUSD": PairResearchPlan(
        pair="HYPEUSD",
        construction="",
        entry_filter="",
        exit_profile="",
        status="drop",
        note="Damaging under current long-only families with no credible replacement.",
    ),
    "SOLUSD": PairResearchPlan(
        pair="SOLUSD",
        construction="",
        entry_filter="",
        exit_profile="",
        status="drop",
        note="Damaging under current long-only families with no credible replacement.",
    ),
}


def active_pairs() -> tuple[str, ...]:
    return tuple(
        plan.pair
        for plan in PAIR_RESEARCH_REGISTRY.values()
        if plan.status in {"active_frozen", "active", "active_experimental"}
    )


def shadow_pairs() -> tuple[str, ...]:
    return tuple(plan.pair for plan in PAIR_RESEARCH_REGISTRY.values() if plan.status == "shadow")


def dropped_pairs() -> tuple[str, ...]:
    return tuple(plan.pair for plan in PAIR_RESEARCH_REGISTRY.values() if plan.status == "drop")
