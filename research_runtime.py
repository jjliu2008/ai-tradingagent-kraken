from __future__ import annotations

from dataclasses import replace
from typing import Any

import math
import numpy as np
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

PORTFOLIO_REGIME_GATE: dict[str, float] = {
    "min_gate_trend_strength_60": 0.0060,
    "min_atr_pct": 0.0060,
}

WEAK_REGIME_MODEL: dict[str, float] = {
    "max_gate_trend_strength_60": 0.0020,
    "negative_gate_trend_strength_60": 0.0,
    "max_atr_pct": 0.0055,
    "hard_max_atr_pct": 0.0045,
    "max_efficiency_ratio_8": 0.30,
    "min_trend_efficiency_ratio_8": 0.35,
}

# Per-pair suppression thresholds — populated by suppression_state.calibrate_thresholds().
# Keys are pair names (e.g. "GIGAUSD"). Each value has:
#   weak_defensive_enter_threshold: float  — weak score above this for N bars → weak_defensive
#   weak_defensive_exit_threshold:  float  — weak score below this for N bars → return to normal
#   cooldown_bars:                  int    — minimum dwell bars in normal before re-entering weak_defensive
SUPPRESSION_THRESHOLDS: dict[str, dict[str, float]] = {}


def compute_pair_weak_score(frame: pd.DataFrame, window: int = 8) -> float:
    """
    Multi-factor weak score from the last ``window`` bars, normalized to [0.0, 1.0].

    Higher score = weaker / more concerning regime. Factors and weights:
      0.35  gate_trend_strength_60  (lower gate  → higher score)
      0.25  atr_pct                 (lower ATR   → higher score)
      0.25  portfolio_regime_ok share (lower OK% → higher score)
      0.15  entry_signal density    (fewer fires → higher score)

    Returns 1.0 (worst case) on empty frame.
    """
    if frame.empty:
        return 1.0
    tail = frame.tail(window)

    gate = float(tail["gate_trend_strength_60"].astype(float).mean())
    atr  = float(tail["atr_pct"].astype(float).mean())
    regime_ok_share = float(
        tail["portfolio_regime_ok"].fillna(False).astype(float).mean()
    )
    if "entry_signal" in tail.columns:
        signal_density = float(
            tail["entry_signal"].fillna(False).astype(float).mean()
        )
    else:
        signal_density = 0.0

    # gate: 0.010 → score 0.0 ; 0.0 or below → score 1.0
    gate_score = max(0.0, min(1.0, 1.0 - gate / 0.010))
    # atr: 0.010+ → score 0.0 ; 0.003 or below → score 1.0
    atr_score = max(0.0, min(1.0, 1.0 - max(0.0, atr - 0.003) / 0.007))
    # regime_ok: all OK → 0.0 ; none OK → 1.0
    regime_score = 1.0 - regime_ok_share
    # signal density: ≥25% of bars fire → 0.0 ; 0% → 1.0
    density_score = max(0.0, min(1.0, 1.0 - signal_density / 0.25))

    return (
        0.35 * gate_score
        + 0.25 * atr_score
        + 0.25 * regime_score
        + 0.15 * density_score
    )


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
    # --- L2 microstructure filters (backtestable from OHLCV bar structure) ---
    if filter_name == "wick_bias":
        # Lower wick longer than upper wick = net buying pressure on the bar
        return (
            mask
            & (frame["wick_imbalance"].astype(float) > 0.0)
            & (frame["body_ratio"].astype(float) > 0.20)
        )
    if filter_name == "micro_confirmed":
        # Close80 quality + buying pressure visible in bar microstructure
        return (
            mask
            & (frame["close_location"].astype(float) >= 0.80)
            & (frame["wick_imbalance"].astype(float) > 0.05)
            & (frame["body_ratio"].astype(float) > 0.25)
            & (frame["micro_pressure"].astype(float) > 0.10)
        )
    if filter_name == "micro_confirmed_volcap60":
        return (
            mask
            & (frame["close_location"].astype(float) >= 0.80)
            & (frame["wick_imbalance"].astype(float) > 0.05)
            & (frame["body_ratio"].astype(float) > 0.25)
            & (frame["micro_pressure"].astype(float) > 0.10)
            & (frame["volume_ratio"].astype(float) <= 6.0)
        )
    return mask


def check_l2_entry_gate(l2_features: dict) -> tuple[bool, str]:
    """
    Real-time L2 order book gate applied before any live entry.

    Parameters
    ----------
    l2_features : dict returned by strat.compute_orderbook_features()

    Returns
    -------
    (approved, reason)
        approved=True  → L2 conditions met, proceed with entry
        approved=False → L2 conditions not met, skip this bar
    """
    obi          = float(l2_features.get("obi", 0.5))
    weighted_obi = float(l2_features.get("weighted_obi", 0.5))
    spread_pct   = float(l2_features.get("spread_pct", 0.0))
    depth_ratio  = float(l2_features.get("depth_ratio", 1.0))

    if spread_pct > 0.005:          # > 0.5% spread → illiquid, skip
        return False, f"spread_wide:{spread_pct:.4%}"
    if obi < 0.45:                  # heavy ask-side pressure
        return False, f"obi_bearish:{obi:.3f}"
    if weighted_obi < 0.44:         # weighted book also bearish
        return False, f"weighted_obi_bearish:{weighted_obi:.3f}"
    if depth_ratio < 0.60:          # ask-side has >40% more depth within 1%
        return False, f"depth_ask_heavy:{depth_ratio:.3f}"
    return True, f"L2_ok obi={obi:.3f} w_obi={weighted_obi:.3f} spread={spread_pct:.4%}"


def apply_portfolio_regime_gate(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["gate_trend_strength_60"].astype(float) >= PORTFOLIO_REGIME_GATE["min_gate_trend_strength_60"]
    ) & (
        frame["atr_pct"].astype(float) >= PORTFOLIO_REGIME_GATE["min_atr_pct"]
    )


def add_weak_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    abs_diff = out["close"].astype(float).diff().abs()
    path8 = abs_diff.rolling(8, min_periods=4).sum().replace(0.0, np.nan)
    path16 = abs_diff.rolling(16, min_periods=8).sum().replace(0.0, np.nan)
    out["efficiency_ratio_8"] = (
        (out["close"].astype(float) - out["close"].astype(float).shift(8)).abs() / path8
    ).fillna(0.0)
    out["efficiency_ratio_16"] = (
        (out["close"].astype(float) - out["close"].astype(float).shift(16)).abs() / path16
    ).fillna(0.0)

    trend_soft = out["gate_trend_strength_60"].astype(float) < WEAK_REGIME_MODEL["max_gate_trend_strength_60"]
    trend_negative = (
        out["gate_trend_strength_60"].astype(float)
        < WEAK_REGIME_MODEL["negative_gate_trend_strength_60"]
    )
    vol_soft = out["atr_pct"].astype(float) < WEAK_REGIME_MODEL["max_atr_pct"]
    vol_hard = out["atr_pct"].astype(float) < WEAK_REGIME_MODEL["hard_max_atr_pct"]
    efficiency_low = out["efficiency_ratio_8"].astype(float) < WEAK_REGIME_MODEL["max_efficiency_ratio_8"]

    out["trend_regime"] = (
        out["portfolio_regime_ok"].fillna(False)
        & (out["efficiency_ratio_8"].astype(float) >= WEAK_REGIME_MODEL["min_trend_efficiency_ratio_8"])
    )
    out["weak_regime_trend_soft"] = trend_soft.fillna(False)
    out["weak_regime_trend_negative"] = trend_negative.fillna(False)
    out["weak_regime_vol_soft"] = vol_soft.fillna(False)
    out["weak_regime_vol_hard"] = vol_hard.fillna(False)
    out["weak_regime_efficiency_low"] = efficiency_low.fillna(False)
    out["weak_regime"] = (
        (~out["portfolio_regime_ok"].fillna(False))
        & (
            out["weak_regime_trend_negative"]
            | (out["weak_regime_trend_soft"] & out["weak_regime_efficiency_low"])
            | out["weak_regime_vol_hard"]
        )
    ).fillna(False)
    out["weak_regime_score"] = (
        (~out["portfolio_regime_ok"].fillna(False)).astype(int)
        + out["weak_regime_trend_soft"].astype(int)
        + out["weak_regime_trend_negative"].astype(int)
        + out["weak_regime_vol_soft"].astype(int)
        + out["weak_regime_vol_hard"].astype(int)
        + out["weak_regime_efficiency_low"].astype(int)
    )
    return out


def weak_regime_reason_tags(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if not bool(row.get("portfolio_regime_ok", False)):
        reasons.append("portfolio_gate_closed")
    if bool(row.get("weak_regime_trend_negative", False)):
        reasons.append("trend_negative")
    elif bool(row.get("weak_regime_trend_soft", False)):
        reasons.append("trend_weak")
    if bool(row.get("weak_regime_vol_hard", False)):
        reasons.append("vol_too_low_hard")
    elif bool(row.get("weak_regime_vol_soft", False)):
        reasons.append("vol_too_low")
    if bool(row.get("weak_regime_efficiency_low", False)):
        reasons.append("directional_efficiency_low")
    return reasons


def build_research_frame(df_raw: pd.DataFrame, plan: PairResearchPlan) -> pd.DataFrame:
    frame = strat.build_ensemble_frame(df_raw, construction=plan.construction, config=config_for_plan(plan))
    if frame.empty:
        return frame
    out = frame.copy()
    out["entry_signal_pre_regime"] = apply_entry_filter(out, plan.entry_filter)
    out["portfolio_regime_ok"] = apply_portfolio_regime_gate(out)
    out["entry_signal"] = out["entry_signal_pre_regime"] & out["portfolio_regime_ok"]
    return add_weak_regime_features(out)


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
        "signal_pre_regime": bool(row.get("entry_signal_pre_regime", False)),
        "portfolio_regime_ok": bool(row.get("portfolio_regime_ok", False)),
        "trend_regime": bool(row.get("trend_regime", False)),
        "weak_regime": bool(row.get("weak_regime", False)),
        "weak_regime_score": int(row.get("weak_regime_score", 0) or 0),
        "weak_reasons": weak_regime_reason_tags(row),
        "gate_trend_strength_60": _safe_float(row.get("gate_trend_strength_60", 0.0), 0.0),
        "atr_pct": _safe_float(row.get("atr_pct", 0.0), 0.0),
        "efficiency_ratio_8": _safe_float(row.get("efficiency_ratio_8", 0.0), 0.0),
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
