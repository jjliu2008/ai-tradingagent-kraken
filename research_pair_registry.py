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


# This is the current research-agent recommendation set.
# It is intentionally narrower than the cached registry and is driven by
# walk-forward plus rolling-OOS tuning evidence.
PAIR_RESEARCH_REGISTRY: dict[str, PairResearchPlan] = {
    "GIGAUSD": PairResearchPlan(
        pair="GIGAUSD",
        construction="tc15_tighter_volume_cap",
        entry_filter="base",
        exit_profile="base",
        status="active_frozen",
        note="Anchor baseline; still the most validated single-pair strategy.",
    ),
    "ZECUSD": PairResearchPlan(
        pair="ZECUSD",
        construction="trend_gate",
        entry_filter="close80_volcap60",
        exit_profile="fast",
        status="active",
        note="Recent repair pass removed weak late-breakout entries while preserving positive OOS behavior.",
    ),
    "XDGUSD": PairResearchPlan(
        pair="XDGUSD",
        construction="baseline_or_atr30",
        entry_filter="volcap45",
        exit_profile="medium",
        status="active",
        note="OOS improved after removing extreme-volume entries and tightening exits.",
    ),
    "FHEUSD": PairResearchPlan(
        pair="FHEUSD",
        construction="vst60_only",
        entry_filter="close80_volcap60",
        exit_profile="runner",
        status="active_experimental",
        note="Recent repair pass tightened close-quality and excess-volume drift without killing frequency.",
    ),
    "KERNELUSD": PairResearchPlan(
        pair="KERNELUSD",
        construction="union_closehi",
        entry_filter="base",
        exit_profile="runner",
        status="active_experimental",
        note="Positive walk-forward survivor; runner exit improved rolling OOS.",
    ),
    "HOUSEUSD": PairResearchPlan(
        pair="HOUSEUSD",
        construction="baseline_or_vst60",
        entry_filter="close80_vwap3",
        exit_profile="runner",
        status="active_experimental",
        note="Recent repair pass cut the worst overextended entries by capping distance from VWAP.",
    ),
    "BABYUSD": PairResearchPlan(
        pair="BABYUSD",
        construction="tc15_cap_or_mb60",
        entry_filter="base",
        exit_profile="fast",
        status="active_experimental",
        note="60d tuning converted a sparse setup into a small but clean OOS contributor.",
    ),
    "BERAUSD": PairResearchPlan(
        pair="BERAUSD",
        construction="vst60_only",
        entry_filter="gate50_vwap3",
        exit_profile="medium",
        status="active_experimental",
        note="Recent repair pass required stronger 60m trend plus less extension above VWAP.",
    ),
    "PARTIUSD": PairResearchPlan(
        pair="PARTIUSD",
        construction="trend_gate",
        entry_filter="base",
        exit_profile="runner",
        status="active_experimental",
        note="Runner exit unlocked more of the trend and made the pair additive out of sample.",
    ),
    "APTUSD": PairResearchPlan(
        pair="APTUSD",
        construction="tc15_or_atr30",
        entry_filter="base",
        exit_profile="runner",
        status="active_experimental",
        note="60d history showed the pair was under-observed; runner exit now yields repeat OOS contribution.",
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
        construction="tc30_only",
        entry_filter="base",
        exit_profile="base",
        status="shadow",
        note="Cached positive, but no OOS confirmation yet.",
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
