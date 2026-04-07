from __future__ import annotations

from dataclasses import replace
from typing import Any

import math
import pandas as pd

import strategy as strat
from research_pair_registry import PAIR_RESEARCH_REGISTRY, PairResearchPlan, active_pairs, shadow_pairs


EXIT_PROFILES: dict[str, dict[str, float | int]] = {
    "base": {"min_stop_pct": 0.0150, "target_pct": 0.0450, "max_hold_bars": 12},
    "medium": {"min_stop_pct": 0.0125, "target_pct": 0.0400, "max_hold_bars": 10},
    "fast": {"min_stop_pct": 0.0100, "target_pct": 0.0300, "max_hold_bars": 8},
    "tight": {"min_stop_pct": 0.0120, "target_pct": 0.0350, "max_hold_bars": 8},
    "runner": {"min_stop_pct": 0.0150, "target_pct": 0.0550, "max_hold_bars": 14},
}


def plan_for_pair(pair: str) -> PairResearchPlan | None:
    return PAIR_RESEARCH_REGISTRY.get(pair.upper())


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def active_research_pairs(include_shadow: bool = False) -> list[str]:
    pairs = list(active_pairs())
    if include_shadow:
        pairs.extend(shadow_pairs())
    return list(dict.fromkeys(pairs))


def config_for_plan(plan: PairResearchPlan) -> strat.StrategyConfig:
    exit_spec = EXIT_PROFILES.get(plan.exit_profile, EXIT_PROFILES["base"])
    return replace(
        strat.DEFAULT_CONFIG,
        min_stop_pct=float(exit_spec["min_stop_pct"]),
        target_pct=float(exit_spec["target_pct"]),
        max_hold_bars=int(exit_spec["max_hold_bars"]),
    )


def apply_entry_filter(frame: pd.DataFrame, filter_name: str) -> pd.Series:
    mask = frame["entry_signal"].fillna(False).astype(bool)
    if filter_name == "base":
        return mask
    if filter_name == "score45":
        return mask & (frame["signal_score"].astype(float) >= 45.0)
    if filter_name == "score55":
        return mask & (frame["signal_score"].astype(float) >= 55.0)
    if filter_name == "score60":
        return mask & (frame["signal_score"].astype(float) >= 60.0)
    if filter_name == "close70":
        return mask & (frame["close_location"].astype(float) >= 0.70)
    if filter_name == "close75":
        return mask & (frame["close_location"].astype(float) >= 0.75)
    if filter_name == "close80":
        return mask & (frame["close_location"].astype(float) >= 0.80)
    if filter_name == "gate20":
        return mask & (frame["gate_trend_strength_60"].astype(float) >= 0.0020)
    if filter_name == "gate30":
        return mask & (frame["gate_trend_strength_60"].astype(float) >= 0.0030)
    if filter_name == "gate50":
        return mask & (frame["gate_trend_strength_60"].astype(float) >= 0.0050)
    if filter_name == "volcap45":
        return mask & (frame["volume_ratio"].astype(float) <= 4.5)
    if filter_name == "volcap30":
        return mask & (frame["volume_ratio"].astype(float) <= 3.0)
    if filter_name == "volcap60":
        return mask & (frame["volume_ratio"].astype(float) <= 6.0)
    if filter_name == "vwap2":
        return mask & (frame["distance_from_vwap"].astype(float) <= 0.0200)
    if filter_name == "vwap3":
        return mask & (frame["distance_from_vwap"].astype(float) <= 0.0300)
    if filter_name == "close80_volcap60":
        return (
            mask
            & (frame["close_location"].astype(float) >= 0.80)
            & (frame["volume_ratio"].astype(float) <= 6.0)
        )
    if filter_name == "close80_vwap3":
        return (
            mask
            & (frame["close_location"].astype(float) >= 0.80)
            & (frame["distance_from_vwap"].astype(float) <= 0.0300)
        )
    if filter_name == "close80_volcap60_vwap3":
        return (
            mask
            & (frame["close_location"].astype(float) >= 0.80)
            & (frame["volume_ratio"].astype(float) <= 6.0)
            & (frame["distance_from_vwap"].astype(float) <= 0.0300)
        )
    if filter_name == "gate50_vwap2":
        return (
            mask
            & (frame["gate_trend_strength_60"].astype(float) >= 0.0050)
            & (frame["distance_from_vwap"].astype(float) <= 0.0200)
        )
    if filter_name == "gate50_vwap3":
        return (
            mask
            & (frame["gate_trend_strength_60"].astype(float) >= 0.0050)
            & (frame["distance_from_vwap"].astype(float) <= 0.0300)
        )
    if filter_name == "close80_gate50_vwap3":
        return (
            mask
            & (frame["close_location"].astype(float) >= 0.80)
            & (frame["gate_trend_strength_60"].astype(float) >= 0.0050)
            & (frame["distance_from_vwap"].astype(float) <= 0.0300)
        )
    return mask


def build_research_frame(df_raw: pd.DataFrame, plan: PairResearchPlan) -> pd.DataFrame:
    frame = strat.build_ensemble_frame(df_raw, construction=plan.construction, config=config_for_plan(plan))
    if frame.empty:
        return frame
    out = frame.copy()
    out["entry_signal"] = apply_entry_filter(out, plan.entry_filter)
    return out


def detect_signal(df_raw: pd.DataFrame, plan: PairResearchPlan) -> dict[str, Any] | None:
    frame = build_research_frame(df_raw, plan)
    if frame.empty:
        return None
    row = frame.iloc[-1]
    conditions = [
        name.replace("signal_", "")
        for name in ["signal_mb60", "signal_tc15", "signal_mbt30", "signal_vst60", "signal_tc30", "signal_atr30"]
        if bool(row.get(name, False))
    ]
    return {
        "frame": frame,
        "signal": bool(row.get("entry_signal", False)),
        "score": _safe_float(row.get("signal_score", 0.0), 0.0),
        "component_count": int(row.get("component_count", 0) or 0),
        "conditions": conditions,
        "price": _safe_float(row.get("close", 0.0), 0.0),
    }


def should_trend_exit(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    last = frame.iloc[-1]
    return bool(
        _safe_float(last.get("close", 0.0), 0.0) < _safe_float(last.get("ema_fast", 0.0), 0.0)
        and _safe_float(last.get("momentum_medium", 0.0), 0.0) < 0
    )
