from __future__ import annotations

# Frozen baseline plus pair-specific constructions selected from the strict
# cached-data matrix screen. These are the current best local candidates, not
# a guarantee of future profitability.
PAIR_STRATEGY_REGISTRY: dict[str, str] = {
    "GIGAUSD": "tc15_tighter_volume_cap",
    "ZECUSD": "trend_gate",
    "XDGUSD": "baseline_or_atr30",
    "NIGHTUSD": "baseline_or_atr30",
    "TAOUSD": "atr30_only",
    "SUIUSD": "tc30_only",
    "ADAUSD": "baseline_or_tc30",
    "COQUSD": "baseline_mb60",
    "ETHUSD": "tc30_only",
}


def frozen_pairs() -> tuple[str, ...]:
    return ("GIGAUSD",)


def registry_pairs() -> tuple[str, ...]:
    return tuple(PAIR_STRATEGY_REGISTRY.keys())
