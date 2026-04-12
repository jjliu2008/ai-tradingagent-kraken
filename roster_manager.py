"""
Roster Manager
==============
Maintains the dynamic active roster: a gate-score–ranked subset of an
approved watchlist built from pair_sweep.py results.

Guardrails
----------
- Roster changes only trigger when a new bar closes (bar_ts advances).
- At most MAX_DAILY_ROTATIONS rotations per UTC day.
- A challenger must outscore its incumbent by MIN_SCORE_GAP_FRAC (hysteresis).
- State is persisted to disk so restarts do not reset the daily rotation cap.

Public interface used by universe_scanner_agent.py
---------------------------------------------------
    roster = RosterManager(max_pos=5)
    roster.load_sweep_candidates()          # populate from sweep results
    plan  = roster.get_plan("BERAUSD")      # PairResearchPlan (registry or watchlist)
    score = RosterManager.compute_gate_score(df_raw)   # static helper

    new_active, log = roster.maybe_rotate(gate_scores, bar_ts)

Standalone check:
    python roster_manager.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import research_runtime as rr
from research_pair_registry import (
    PAIR_RESEARCH_REGISTRY,
    PairResearchPlan,
    active_pairs,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_DAILY_ROTATIONS   = 2
MIN_SCORE_GAP_FRAC    = 0.20    # challenger must beat incumbent by ≥ 20%
MAX_WATCHLIST_PAIRS   = 12      # cap to keep gate-score fetching cheap

ROSTER_STATE_FILE     = Path("runtime/roster/roster_state.json")
SWEEP_CANDIDATES_FILE = Path("results/sweep_candidates.json")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WatchlistEntry:
    pair:                 str
    construction:         str
    entry_filter:         str
    exit_profile:         str
    sweep_score:          float   # net_pct × win_rate from pair_sweep.py
    max_corr_with_active: float   # phi correlation with closest active pair


@dataclass
class _RosterState:
    active:              list[str]
    rotations_today:     int
    last_rotation_date:  str    # "YYYY-MM-DD" UTC
    last_bar_ts:         int    # bar timestamp of the last roster evaluation


# ---------------------------------------------------------------------------
# RosterManager
# ---------------------------------------------------------------------------

class RosterManager:
    """
    Manages the approved watchlist and live active roster.

    The watchlist is populated once from sweep results (call load_sweep_candidates).
    The active roster starts as the current registry active pairs and rotates
    via maybe_rotate at each bar close.
    """

    def __init__(
        self,
        max_pos:              int,
        state_file:           Path = ROSTER_STATE_FILE,
        max_daily_rotations:  int  = MAX_DAILY_ROTATIONS,
        min_score_gap_frac:   float = MIN_SCORE_GAP_FRAC,
    ) -> None:
        self.max_pos             = max_pos
        self.state_file          = state_file
        self.max_daily_rotations = max_daily_rotations
        self.min_score_gap_frac  = min_score_gap_frac
        self.watchlist: dict[str, WatchlistEntry] = {}
        self._state = self._load_state()

    # ------------------------------------------------------------------
    # Watchlist population
    # ------------------------------------------------------------------

    def load_sweep_candidates(
        self,
        path: Path = SWEEP_CANDIDATES_FILE,
        max_candidates: int = MAX_WATCHLIST_PAIRS,
    ) -> int:
        """
        Populate the watchlist from sweep_candidates.json produced by pair_sweep.py.

        Keeps the best combo per pair, then takes the top max_candidates overall
        by sweep score. Pairs already in the active registry are skipped — they
        are controlled by the registry, not the sweep.

        Returns the number of new watchlist entries added.
        """
        if not path.exists():
            return 0

        with open(path) as f:
            candidates: list[dict[str, Any]] = json.load(f)

        # Best combo per pair
        best_per_pair: dict[str, dict[str, Any]] = {}
        for c in candidates:
            pair = c["pair"]
            # Skip pairs that are already fully managed by the registry
            if pair in PAIR_RESEARCH_REGISTRY and PAIR_RESEARCH_REGISTRY[pair].status in {
                "active", "active_frozen", "active_experimental"
            }:
                continue
            if pair not in best_per_pair or c["score"] > best_per_pair[pair]["score"]:
                best_per_pair[pair] = c

        ranked = sorted(best_per_pair.values(), key=lambda c: c["score"], reverse=True)

        added = 0
        for c in ranked[:max_candidates]:
            pair = c["pair"]
            self.watchlist[pair] = WatchlistEntry(
                pair=pair,
                construction=c["construction"],
                entry_filter=c["entry_filter"],
                exit_profile=c["exit_profile"],
                sweep_score=float(c.get("score", 0.0)),
                max_corr_with_active=float(c.get("max_corr_with_active", 0.0)),
            )
            added += 1

        return added

    # ------------------------------------------------------------------
    # Plan lookup
    # ------------------------------------------------------------------

    def get_plan(self, pair: str) -> PairResearchPlan | None:
        """
        Return a PairResearchPlan for any pair in the active roster.

        Checks the research registry first (for originally active pairs).
        Falls back to the watchlist entry for sweep-promoted pairs.
        """
        plan = rr.plan_for_pair(pair)
        if plan is not None:
            return plan

        entry = self.watchlist.get(pair)
        if entry is None:
            return None

        return PairResearchPlan(
            pair=pair,
            construction=entry.construction,
            entry_filter=entry.entry_filter,
            exit_profile=entry.exit_profile,
            status="watchlist_promoted",
            note=f"Sweep candidate (score={entry.sweep_score:.4f}, "
                 f"max_corr={entry.max_corr_with_active:.2f})",
        )

    # ------------------------------------------------------------------
    # Gate score
    # ------------------------------------------------------------------

    @staticmethod
    def compute_gate_score(df_raw: pd.DataFrame) -> float:
        """
        Compute gate_trend_strength_60 × atr_pct from the last completed bar.
        Uses trend_gate construction (cheapest) just to build the feature frame.
        Returns 0.0 on any failure.
        """
        plan = PairResearchPlan(
            pair="_gate_score",
            construction="trend_gate",
            entry_filter="base",
            exit_profile="fast",
            status="internal",
            note="",
        )
        try:
            frame = rr.build_research_frame(df_raw, plan)
        except Exception:
            return 0.0
        if frame.empty:
            return 0.0

        last = frame.iloc[-1]
        g60 = float(last.get("gate_trend_strength_60", 0.0) or 0.0)
        atr = float(last.get("atr_pct", 0.0) or 0.0)
        return g60 * atr

    # ------------------------------------------------------------------
    # Rotation logic
    # ------------------------------------------------------------------

    @property
    def active(self) -> list[str]:
        """Current active pair list."""
        return list(self._state.active)

    def maybe_rotate(
        self,
        gate_scores: dict[str, float],
        bar_ts: int,
    ) -> tuple[list[str], list[str]]:
        """
        Evaluate whether the active roster should rotate.

        Parameters
        ----------
        gate_scores : gate_trend_strength_60 × atr_pct for every pair in the
                      eligible universe (active + watchlist). Computed from the
                      latest closed bar by the agent.
        bar_ts      : timestamp of the latest closed bar. Rotation only triggers
                      when this advances past the last recorded bar_ts.

        Returns
        -------
        (current_active, rotation_log)
            rotation_log is a list of human-readable strings describing any
            changes made; empty list if the roster did not change.
        """
        rotation_log: list[str] = []

        # Gate 1: only evaluate at bar close
        if bar_ts <= self._state.last_bar_ts:
            return list(self._state.active), rotation_log
        self._state.last_bar_ts = bar_ts

        # Gate 2: reset daily counter on UTC date change
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._state.last_rotation_date:
            self._state.rotations_today = 0
            self._state.last_rotation_date = today

        # Gate 3: daily rotation cap
        if self._state.rotations_today >= self.max_daily_rotations:
            self._save_state()
            return list(self._state.active), rotation_log

        # Build full eligible universe: active + watchlist, scored by gate score
        eligible: dict[str, float] = {}
        for pair in self._state.active:
            eligible[pair] = gate_scores.get(pair, 0.0)
        for pair in self.watchlist:
            eligible[pair] = gate_scores.get(pair, 0.0)

        # Rank by gate score; take the top max_pos
        ranked = sorted(eligible.items(), key=lambda kv: kv[1], reverse=True)
        proposed_active = [pair for pair, _ in ranked[: self.max_pos]]
        proposed_set    = set(proposed_active)
        current_set     = set(self._state.active)

        pairs_out = current_set - proposed_set
        pairs_in  = proposed_set - current_set

        # Execute swaps one at a time, respecting all guardrails
        remaining_in = set(pairs_in)
        for out_pair in pairs_out:
            if self._state.rotations_today >= self.max_daily_rotations:
                break
            if not remaining_in:
                break

            # Pick the highest-scoring challenger
            in_pair = max(remaining_in, key=lambda p: gate_scores.get(p, 0.0))

            # Gate 4: minimum score gap (hysteresis)
            incumbent_score  = gate_scores.get(out_pair, 0.0)
            challenger_score = gate_scores.get(in_pair, 0.0)
            required          = incumbent_score * (1.0 + self.min_score_gap_frac)
            if challenger_score < required:
                continue

            # Execute the swap
            self._state.active = [
                in_pair if p == out_pair else p
                for p in self._state.active
            ]
            if in_pair not in self._state.active:
                # out_pair was already replaced; append if slot still needed
                self._state.active.append(in_pair)
            self._state.rotations_today += 1
            remaining_in.discard(in_pair)

            gap_pct = (challenger_score / max(incumbent_score, 1e-9) - 1.0) * 100
            rotation_log.append(
                f"ROTATE out={out_pair} ({incumbent_score:.5f}) "
                f"→ in={in_pair} ({challenger_score:.5f}, +{gap_pct:.1f}%)"
            )

        self._save_state()
        return list(self._state.active), rotation_log

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> _RosterState:
        def _merge_loaded_active(loaded_active: list[str]) -> list[str]:
            merged = list(dict.fromkeys(loaded_active))
            for pair in active_pairs():
                if pair not in merged:
                    merged.append(pair)
            return merged

        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                return _RosterState(
                    active=_merge_loaded_active(list(data.get("active", list(active_pairs())))),
                    rotations_today=int(data.get("rotations_today", 0)),
                    last_rotation_date=str(data.get("last_rotation_date", "")),
                    last_bar_ts=int(data.get("last_bar_ts", 0)),
                )
            except Exception:
                pass
        return _RosterState(
            active=_merge_loaded_active(list(active_pairs())),
            rotations_today=0,
            last_rotation_date="",
            last_bar_ts=0,
        )

    def _save_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "active":             self._state.active,
            "rotations_today":    self._state.rotations_today,
            "last_rotation_date": self._state.last_rotation_date,
            "last_bar_ts":        self._state.last_bar_ts,
            "watchlist": {
                pair: {
                    "construction":         e.construction,
                    "entry_filter":         e.entry_filter,
                    "exit_profile":         e.exit_profile,
                    "sweep_score":          e.sweep_score,
                    "max_corr_with_active": e.max_corr_with_active,
                }
                for pair, e in self.watchlist.items()
            },
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    def status_report(self) -> str:
        lines = [
            f"Active roster ({len(self._state.active)}/{self.max_pos}): "
            f"{self._state.active}",
            f"Rotations today: {self._state.rotations_today}/{self.max_daily_rotations}",
            f"Watchlist pairs ({len(self.watchlist)}): {list(self.watchlist.keys())}",
            f"Last bar ts: {self._state.last_bar_ts}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone check
# ---------------------------------------------------------------------------

def main() -> None:
    roster = RosterManager(max_pos=5)
    n = roster.load_sweep_candidates()
    print(f"Loaded {n} watchlist entries from sweep candidates.")
    print()
    print(roster.status_report())
    print()
    if roster.watchlist:
        print("Watchlist details:")
        for pair, entry in roster.watchlist.items():
            print(
                f"  {pair:<12} {entry.construction:<24} {entry.entry_filter:<28} "
                f"{entry.exit_profile:<8} sweep_score={entry.sweep_score:.4f} "
                f"max_corr={entry.max_corr_with_active:.2f}"
            )


if __name__ == "__main__":
    main()
