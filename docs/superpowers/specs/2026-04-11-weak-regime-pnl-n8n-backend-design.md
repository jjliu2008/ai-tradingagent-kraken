# Design: Older-60d PnL Repair + n8n Backend Solidification

**Date:** 2026-04-11
**Status:** Approved for implementation planning
**Problem:** The long book is profitable overall (+29.18% / 120d) but bleeds in the older-60d window (-1.48%). No validated weak-regime complement exists. The n8n backend is unwired.

---

## Context

Current honest 120d result (`current_main_120d_summary_after_weak.json`):
- Full 120d: +29.18%, 16 trades, max DD -4.19%
- Older 60d: -1.48%
- Recent 60d: +30.66%

Current active pairs: GIGAUSD, BABYUSD, KERNELUSD, HOUSEUSD, ZECUSD

What has been validated:
- Long-only trend/continuation families on densified 15m uniform data
- Three-window scoring gate (full, older60, recent60)
- Weak-regime detector (identifies bad periods but provides no profitable alternative yet)

What has been ruled out:
- Generic mean-reversion weak-mode router
- Mirrored short versions of long strategies
- Broad weak-regime short screens
- Weak-bear short overlays on the long book
- All produced 0 strict survivors

---

## Design Principles

1. Stay inside proven long families. Do not invent new signal logic.
2. Score every candidate on all three windows — older60 as primary, but recent60 and full120 must both stay positive.
3. Suppression is always on. It is not a fallback; it is a standing safety layer.
4. n8n handles orchestration only. Strategy logic stays in Python.
5. No candidate auto-promotes above shadow. Human approval required for active_experimental and above.
6. Judge every idea on: full 120d, older 60d, recent 60d, enough trades per segment, after costs.

---

## Research Architecture (5 Layers)

### Layer 1 — Pair-Specific Family Re-Iteration (`older60_pair_screener.py`)

For each pair in the densified universe, re-run the full proven family library:
- Constructions: `tc15`, `mb60`, `vst60`, `atr30`, union variants
- Existing validated entry filters
- Existing validated exit profiles

This is a per-pair re-evaluation, not a fixed-registry check.

**Evidence bars by tier:**
- `core_candidates`: `older60_trades >= 5`, `recent60_trades >= 3`, `full_trades >= 8`
- `shadow_candidates`: `older60_trades >= 4`, `recent60_trades >= 1`, `full_trades >= 4`

**Scoring objective (constrained, uncertainty-adjusted):**
- Hard gates for `core_candidates`: `older60_net_pct > 0`, `recent60_net_pct > 0`, `full_net_pct > 0`
- Hard gates for `shadow_candidates`: `recent60_net_pct > 0`, `full_net_pct > 0`, plus older60 positive or near-flat rule from Layer 3
- Segment terms:
  - `older60_term = older60_net_pct * min(1.0, sqrt(older60_trades / 5.0))`
  - `recent60_term = recent60_net_pct * min(1.0, sqrt(recent60_trades / 3.0))`
  - `full120_term = full_net_pct * min(1.0, sqrt(full_trades / 8.0))`
- Penalties:
  - `drawdown_penalty = max(0.0, -full_max_dd_pct)`
  - `concentration_penalty = max(0.0, candidate_contribution_share - 0.35) / 0.15`
  - `correlation_penalty = max(0.0, max_signal_correlation_to_active - 0.70) / 0.30`
- Ranking equation:
  - `robustness_score = 1.50 * older60_term + 1.00 * recent60_term + 0.75 * full120_term - 0.50 * drawdown_penalty - 0.25 * concentration_penalty - 0.25 * correlation_penalty`

**Important boundary:**
- Layer 1 ranks pair-plan candidates only
- Exact portfolio replacement checks happen later in `registry_proposal.py`, where the candidate is evaluated in the context of the current roster

**Output:** Top-K plans per pair (default K=3), written to `results/research_runs/<run_id>/older60_candidates.json`

Outputting top-K (not just the winner) allows Layer 2 to compare what differs between the best surviving plans per pair.

---

### Layer 2 — Segment Diagnosis (`segment_diagnostics.py`)

For each pair and each of its top-K candidate plans, extract diagnostics for the older-60d segment:

**Feature distributions per segment:**
- `gate_trend_strength_60` bucket
- `atr_pct` bucket
- efficiency ratio bucket
- VWAP extension
- close quality / close location
- volume ratio
- breakout/compression state

**Trade-level analysis:**
- Exit reason mix (TAKE_PROFIT / STOP_LOSS / TIME_LIMIT / TREND_LOST / BEAR_LOST)
- Trade frequency
- Entry quality distribution
- Contribution to combined portfolio equity curve

**Outputs (structured JSON, consumed programmatically by Layer 3):**
- `results/research_runs/<run_id>/pair_pattern_notes.json`
- `results/research_runs/<run_id>/older60_behavior_summary.json`
- `results/research_runs/<run_id>/failure_notes.json`

Discovery is not "best score wins." It becomes: which plan works, and under what pattern does it work.

---

### Layer 3 — Pattern-Guided Discovery (`pattern_guided_discovery.py`)

Consumes Layer 2 JSON artifacts from the same run (by `source_run_id`).

**Allowed fingerprint fields (pre-specified, no freeform expansion):**
- `gate_trend_strength_60` bucket
- `atr_pct` bucket
- `efficiency_ratio_8` bucket
- `distance_from_vwap` bucket
- `close_location` bucket
- `volume_ratio` bucket
- `component_count`
- top exit reason / exit mix bucket

**Fingerprint similarity score (pre-specified):**
- Weighted bucket-match score:
  - `0.25` trend bucket
  - `0.20` ATR bucket
  - `0.15` efficiency bucket
  - `0.15` VWAP-distance bucket
  - `0.10` close-quality bucket
  - `0.10` volume bucket
  - `0.05` exit-mix bucket
- `pattern_match_score` is normalized to `[0, 1]`

**Logic:**
- Reassign a pair to the proven family that best matches its older-60d pattern fingerprint
- Allow near-flat pairs (`|older60_net_pct| < 0.005`) into shadow only if they have `>=4` older-60d trades, `pattern_match_score >= 0.65`, and positive `robustness_score`
- Dynamic pair-to-family assignment is driven by observed pattern fingerprints, not invented logic
- Still limited to proven building blocks
- A reassignment is only accepted if it beats both:
  - `keep_current_family`
  - `best_raw_segmented_score`
  by positive `robustness_score` and without violating recent60/full120 positivity gates

**Outputs:**
- `results/research_runs/<run_id>/core_candidates.json`
- `results/research_runs/<run_id>/shadow_candidates.json`
- Pattern-conditioned plan recommendations per pair

**Shadow eligibility gate:**
- `older60_net_pct > 0` OR near-flat (|older60_net_pct| < 0.005) with ≥4 older-60d trades and strong pattern match
- `recent60_net_pct > 0`
- `full_net_pct > 0`
- Minimum per-segment trade count satisfied for the `shadow_candidates` tier

---

### Layer 4 — Controlled Research Escalation (opt-in, gated)

Only triggered if Layers 1–3 do not expand the book enough.

**Rules:**
- Restricted to pairs that already show positive directional structure in the older-60d window
- No broad weak-mode families
- No generic mean reversion
- No direct promotion to active — manual review required
- Results land in shadow only; human decides whether to escalate
- Pair-specific only; no cross-pair or portfolio-level logic here

This layer is for targeted experiments, not the default discovery loop.

---

### Layer 5 — Suppression and Defensive Execution (`suppression_state.py`)

**Always on. Not a fallback.**

**Per-pair weak score** (multi-factor, computed from densified history):
- `gate_trend_strength_60`
- `atr_pct`
- Rolling `portfolio_regime_ok` share
- Rolling signal density for that pair

**Per-pair threshold calibration:**
- Computed from each pair's own 120d densified history (not a global gate)
- `weak_defensive` threshold = pair-specific percentile of weak score distribution
- Thresholds stored in `SUPPRESSION_THRESHOLDS` dict in `research_runtime.py`
- Each pair also gets:
  - `weak_defensive_enter_threshold`
  - `weak_defensive_exit_threshold`
  - minimum dwell / cooldown bars before returning to `normal`

**State machine (evaluated at portfolio and pair level independently):**

| State | Trigger | Behavior |
|---|---|---|
| `normal` | Portfolio and pair regime healthy | Full activity, full notional |
| `weak_defensive` | Pair weak score above enter threshold for 2–3 consecutive bars, OR portfolio composite threshold crossed | Drop lowest-conviction pairs first (ranked by older-60d robustness); keep resilient core; optionally reduce notional |
| `off` | Portfolio composite weak score high AND gate-open share very low AND signal density near zero, for 3–4 consecutive bars | No new entries; existing positions managed to exit only |

Deactivation order in `weak_defensive`: pairs ranked by older-60d robustness score from Layer 1 — least robust drops first.

**Anti-whipsaw rules:**
- `weak_defensive -> normal` requires weak score below exit threshold for `>=4` consecutive bars
- `off -> weak_defensive` requires composite weak score below exit threshold and gate-open share recovery for `>=6` consecutive bars
- State changes are recorded with `bars_in_state`, `bars_above_threshold`, and `bars_below_threshold`

**Integration points:**
- `universe_scanner_agent.py` — primary paper/live path (authoritative)
- `agent.py` — development/paper path (also wired)
- Both read `results/suppression_state.json` at startup and on each bar close

---

## Promotion Ladder

```
discovery → shadow → active_experimental → active
```

- **shadow**: auto-promoted by daily n8n workflow only if concentration, robustness, and correlation sanity gates all pass
- **active_experimental**: requires explicit webhook approval (manual trigger)
- **active**: requires explicit webhook approval

**Concentration gate for shadow auto-promotion:**
- Blocked if candidate pushes single-pair shadow-book concentration above 35%
- Blocked if 3+ candidates from the same regime-sensitivity cluster promote in one daily run
- Blocked if `robustness_score <= 0`
- Blocked if max signal correlation to current active book exceeds `0.75`
- Blocked candidates flagged in proposal with `approval_required: true`

No `shadow_experimental` tier. Metadata in proposal artifact carries the distinction:
`approval_required`, `cluster_id`, `concentration_weight`, `source_run_id`

---

## n8n Workflow Architecture

### Daily Research Workflow (cron, once per day)

1. Fetch latest densified bars for all universe pairs
2. Run `older60_pair_screener.py` → top-K candidates per pair
3. Run `segment_diagnostics.py` → pair pattern notes, behavior summary, failure notes
4. Run `pattern_guided_discovery.py` → core and shadow candidates
5. Run `registry_proposal.py` → versioned proposal artifact with diff vs current registry
6. Evaluate concentration, robustness, and correlation gates → auto-apply clean shadow promotions
7. Flag concentration-breaching candidates with `approval_required: true`
8. Send summary alert (Slack/webhook): candidates, impact scores, before/after portfolio metrics

**Artifact directory:** `results/research_runs/<run_id>/`
**Convenience symlink/copy:** `results/latest/`

**Workflow safety / idempotency requirements:**
- `run_id` is immutable and unique per research run
- Artifacts are written to a temp directory and atomically renamed into `results/research_runs/<run_id>/`
- Re-running the same `run_id` is read-only and must not re-apply promotions
- Registry proposals include `registry_hash_before` so apply steps can verify expected state
- Every approval action includes an idempotency token and candidate identifier
- Duplicate webhooks must be safe to replay with no double promotion

### Intraday Agent Workflow (every 15m, bar-close aligned)

1. Evaluate current suppression state (portfolio + per-pair) via `suppression_state.py`
2. Run `universe_scanner_agent.py` with current active registry + suppression state
3. Collect health, open positions, signals fired, blocked reasons
4. Fire alert on: new entry, exit, suppression state change, agent error

**Intraday safety requirements:**
- Suppression evaluation aborts if latest bar timestamp is stale or unchanged
- `suppression_state.json` includes `source_run_id` and bar timestamp to detect stale inputs
- Runtime reads suppression state idempotently at each bar close; partial writes are ignored

### Approval Workflow (triggered by daily workflow output)

- Receives webhook with `run_id` and candidate list
- Displays before/after portfolio metrics and diff
- Accepts explicit approve/reject per candidate
- On approval: promotes candidate in registry, logs to audit trail
- On reject: candidate stays in shadow, reason logged

---

## File / Module Summary

### New files

| File | Purpose |
|---|---|
| `older60_pair_screener.py` | Layer 1: per-pair family re-iteration, constrained scoring, top-K output |
| `segment_diagnostics.py` | Layer 2: feature/exit/pattern extraction per segment → JSON |
| `pattern_guided_discovery.py` | Layer 3: consumes L2 JSON, assigns pairs to best-matching families |
| `suppression_state.py` | Layer 5: per-pair weak score, state machine, persistence, JSON output |
| `registry_proposal.py` | Versioned proposal artifact with diff, metrics, concentration flags |

### Modified files

| File | Change |
|---|---|
| `research_runtime.py` | Add `compute_pair_weak_score()`, add `SUPPRESSION_THRESHOLDS` dict |
| `research_pair_registry.py` | No new tiers; metadata fields added to plan definitions |
| `universe_scanner_agent.py` | Read `suppression_state.json`, respect pair/portfolio state before signal evaluation |
| `agent.py` | Same suppression wiring as `universe_scanner_agent.py` |
| `general_portfolio_backtest.py` | Add `older60_net_pct` and `recent60_net_pct` as first-class outputs |

### Unchanged files

`backtest.py`, `pair_diagnostics.py`, `rolling_pair_tuner.py`, `walkforward_pair_strategy_search.py`, `weak_mode_family_screener.py`, `weak_regime_short_research.py`, `weak_bear_short_screener.py`

---

## Artifact Schemas

### `suppression_state.json`

```json
{
  "schema_version": "1.0",
  "run_ts": 1234567890,
  "source_run_id": "2026-04-11T00:00:00",
  "bar_ts": 1234567800,
  "portfolio_state": "normal",
  "pairs": {
    "GIGAUSD": {
      "state": "normal",
      "weak_score": 0.12,
      "bars_in_state": 0,
      "bars_above_threshold": 0,
      "bars_below_threshold": 5,
      "threshold": 0.45,
      "exit_threshold": 0.38,
      "reason_tags": [],
      "notional_multiplier": 1.0,
      "allow_new_entries": true
    },
    "ZECUSD": {
      "state": "weak_defensive",
      "weak_score": 0.71,
      "bars_in_state": 3,
      "threshold": 0.52,
      "reason_tags": ["low_gate_trend", "low_signal_density"],
      "notional_multiplier": 0.5,
      "allow_new_entries": false
    }
  }
}
```

### `registry_proposals/<run_id>/proposal.json`

```json
{
  "schema_version": "1.0",
  "run_id": "2026-04-11T00:00:00",
  "source_run_id": "2026-04-11T00:00:00",
  "registry_hash_before": "sha256:...",
  "core_candidates": [],
  "shadow_candidates": [],
  "diff": {
    "promote_to_shadow": [
      {
        "pair": "XDGUSD",
        "plan": "tc15_tighter_volume_cap",
        "approval_required": false,
        "cluster_id": "tc15_group",
        "concentration_weight": 0.12,
        "max_signal_correlation_to_active": 0.41,
        "robustness_score": 0.084,
        "source_run_id": "2026-04-11T00:00:00"
      }
    ],
    "demote": [],
    "no_change": []
  },
  "concentration_flags": [],
  "before_metrics": {
    "older60_net_pct": -0.0148,
    "recent60_net_pct": 0.3066,
    "full_net_pct": 0.2918
  },
  "after_metrics": {
    "older60_net_pct": 0.031,
    "recent60_net_pct": 0.298,
    "full_net_pct": 0.301
  }
}
```

---

## Success Criteria

- Older-60d portfolio PnL moves from -1.48% toward breakeven or positive
- Recent-60d remains positive and does not degrade by more than 5 percentage points versus the baseline computed from the same run
- Full-120d remains positive
- Top pair contribution share stays below 45% of full-120d net PnL
- Leave-one-top-contributor-out portfolio remains non-negative on full-120d and positive on recent-60d
- Suppression state correctly identifies weak periods and prevents new entries during them
- Daily n8n workflow runs end-to-end without manual intervention for the research loop
- Intraday n8n workflow fires on 15m bar closes and alerts on state changes
- No candidate reaches `active` without two explicit manual approvals (shadow → active_experimental, active_experimental → active)
- All research artifacts are versioned, run-scoped, and auditable
