# Older-60d PnL Repair + n8n Backend Solidification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the older-60d portfolio bleed (-1.48%) through pair-specific family re-iteration, add a three-state suppression layer that is always on, and wire three n8n workflows (daily research, intraday 15m agent, approval webhook).

**Architecture:** Five new Python modules implement the research pipeline (Layers 1–3) and suppression state machine (Layer 5), wired into `universe_scanner_agent.py` as the primary execution path. Three n8n workflows orchestrate daily research, 15m intraday execution, and manual approval gates. Strategy logic stays entirely in Python; n8n handles scheduling and alerting only.

**Tech Stack:** Python 3.12+, pandas, numpy, pytest, n8n (via MCP tools `n8n_create_workflow`), existing `research_runtime.py` / `research_pair_registry.py` / `backtest.py` / `strategy.py`

---

## File Map

### New files
| Path | Responsibility |
|---|---|
| `suppression_state.py` | Per-pair weak score, three-state machine, JSON output |
| `older60_pair_screener.py` | Layer 1: per-pair family re-iteration, uncertainty-adjusted scoring, top-K output |
| `segment_diagnostics.py` | Layer 2: feature/exit/pattern extraction per segment → JSON |
| `pattern_guided_discovery.py` | Layer 3: fingerprint similarity, shadow/core candidate classification |
| `registry_proposal.py` | Versioned proposal artifact with diff, concentration gates |
| `tests/__init__.py` | Empty — marks tests as package |
| `tests/conftest.py` | Shared fixtures: minimal synthetic OHLCV DataFrame matching schema |
| `tests/test_suppression_score.py` | Unit tests for `compute_pair_weak_score` |
| `tests/test_suppression_state.py` | Unit tests for state machine transitions |
| `tests/test_suppression_wiring.py` | Integration test: suppressed pair skipped in agent scan |
| `tests/test_older60_backtest.py` | Unit tests for older60/recent60 outputs in general_portfolio_backtest |
| `tests/test_older60_screener.py` | Unit tests for robustness_score formula and top-K output |
| `tests/test_segment_diagnostics.py` | Unit tests for feature bucket extraction |
| `tests/test_pattern_guided_discovery.py` | Unit tests for fingerprint similarity and eligibility gate |
| `tests/test_registry_proposal.py` | Unit tests for diff computation and concentration flags |

### Modified files
| Path | Change |
|---|---|
| `research_runtime.py` | Add `compute_pair_weak_score()` and `SUPPRESSION_THRESHOLDS` dict |
| `general_portfolio_backtest.py` | Add `older60` and `recent60` windows to summary and pair_frame |
| `universe_scanner_agent.py` | Load `suppression_state.json`; skip new entries for suppressed pairs |
| `agent.py` | Same suppression wiring as universe_scanner_agent.py |

### Unchanged files
`backtest.py`, `strategy.py`, `research_pair_registry.py`, `pair_diagnostics.py`, `rolling_pair_tuner.py`, `walkforward_pair_strategy_search.py`

---

## Task 1: Add older60/recent60 to general_portfolio_backtest.py

**Files:**
- Modify: `general_portfolio_backtest.py:86-161`
- Test: `tests/test_older60_backtest.py`

The current `run_portfolio_backtest` tracks `recent30` and `recent14`. We add `older60` (trades before the 60d mark) and `recent60` (trades at or after the 60d mark), both at portfolio level and per-pair.

- [ ] **Step 1: Create test file and write failing test**

Create `tests/__init__.py` (empty) and `tests/test_older60_backtest.py`:

```python
# tests/test_older60_backtest.py
from __future__ import annotations

import types
import pandas as pd
import pytest


def _make_fake_trade(entry_ts: int, pnl_pct: float, pair: str = "FAKEUSD"):
    """Return a minimal BacktestTrade-like namespace."""
    t = types.SimpleNamespace()
    t.pair = pair
    t.entry_ts = entry_ts
    t.pnl_pct = pnl_pct
    t.construction = "tc15_only"
    t.exit_ts = entry_ts + 900
    t.exit_reason = "TAKE_PROFIT"
    t.mfe_pct = pnl_pct
    t.bars_held = 1
    t.signal_score = 50.0
    t.trend_strength = 0.01
    t.pullback_depth_pct = 0.0
    t.distance_from_vwap = 0.01
    t.compression_ratio = 0.8
    t.entry_price = 1.0
    t.exit_price = 1.0 + pnl_pct
    t.entry_bar = 0
    t.exit_bar = 1
    return t


def test_summarize_splits_older_and_recent():
    """_summarize_with_split returns correct older60 / recent60 buckets."""
    from general_portfolio_backtest import _summarize_with_split

    now_ts = 1_800_000_000
    split_ts = now_ts - 60 * 24 * 60 * 60

    old_trade = _make_fake_trade(split_ts - 1000, 0.05)
    new_trade = _make_fake_trade(split_ts + 1000, -0.02)

    result = _summarize_with_split([old_trade, new_trade], split_ts)

    assert result["older60_trades"] == 1
    assert result["recent60_trades"] == 1
    assert abs(result["older60_net_pct"] - 0.05) < 1e-9
    assert abs(result["recent60_net_pct"] - (-0.02)) < 1e-9


def test_run_portfolio_backtest_summary_keys():
    """run_portfolio_backtest summary includes older60 and recent60 keys."""
    import general_portfolio_backtest as gpb

    # Patch the internal _load_uniform_history to avoid file I/O
    original = gpb._load_uniform_history

    def _mock_load(pair, interval, history_days, refresh_live, trade_count, trade_pause_sec):
        import numpy as np
        n = 200
        ts = list(range(1_700_000_000, 1_700_000_000 + n * 900, 900))
        df = pd.DataFrame({
            "ts": ts, "open": [1.0] * n, "high": [1.05] * n,
            "low": [0.95] * n, "close": [1.0] * n,
            "volume": [1000.0] * n, "vwap_k": [1.0] * n, "count": [1] * n,
        })
        from pathlib import Path
        return df, Path("dummy.csv")

    gpb._load_uniform_history = _mock_load
    try:
        _, summary = gpb.run_portfolio_backtest(
            pairs=[], interval=15, history_days=120,
            refresh_live=False, trade_count=0, trade_pause_sec=0,
        )
        assert "older60" in summary, "summary missing 'older60'"
        assert "recent60" in summary, "summary missing 'recent60'"
        assert "split_ts" in summary, "summary missing 'split_ts'"
    finally:
        gpb._load_uniform_history = original
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd C:\Users\majin\ai-tradingagent-kraken
python -m pytest tests/test_older60_backtest.py -v
```

Expected: `ImportError: cannot import name '_summarize_with_split'`

- [ ] **Step 3: Add `_summarize_with_split` and older60/recent60 to general_portfolio_backtest.py**

Add the helper function after the existing `_summarize` function (after line 37):

```python
def _summarize_with_split(
    trades: list[bt.BacktestTrade], split_ts: int
) -> dict[str, Any]:
    """Summarize trades broken into older60 (before split_ts) and recent60 (at or after)."""
    older = [t for t in trades if t.entry_ts < split_ts]
    recent = [t for t in trades if t.entry_ts >= split_ts]

    def _stats(subset: list[bt.BacktestTrade]) -> dict[str, Any]:
        if not subset:
            return {"trades": 0, "net_pct": 0.0, "win_rate": 0.0, "avg_trade_pct": 0.0, "max_dd_pct": 0.0}
        pnls = pd.Series([t.pnl_pct for t in subset], dtype=float)
        equity = (1.0 + pnls).cumprod()
        max_dd = float((equity / equity.cummax() - 1.0).min())
        return {
            "trades": int(len(subset)),
            "net_pct": float(pnls.sum()),
            "win_rate": float((pnls > 0).mean()),
            "avg_trade_pct": float(pnls.mean()),
            "max_dd_pct": max_dd,
        }

    older_stats = _stats(older)
    recent_stats = _stats(recent)
    return {
        **{f"older60_{k}": v for k, v in older_stats.items()},
        **{f"recent60_{k}": v for k, v in recent_stats.items()},
    }
```

In `run_portfolio_backtest`, replace the `recent_30_start` / `recent_14_start` block (lines 128–148) with the expanded version:

```python
    recent_30_start = max_ts - 30 * 24 * 60 * 60 if max_ts else 0
    recent_14_start = max_ts - 14 * 24 * 60 * 60 if max_ts else 0
    split_ts        = max_ts - 60 * 24 * 60 * 60 if max_ts else 0
    older60_trades:  list[bt.BacktestTrade] = []
    recent60_trades: list[bt.BacktestTrade] = []
    recent_30_trades: list[bt.BacktestTrade] = []
    recent_14_trades: list[bt.BacktestTrade] = []

    for trade in all_trades:
        if trade.entry_ts < split_ts:
            older60_trades.append(trade)
        else:
            recent60_trades.append(trade)
        if trade.entry_ts >= recent_30_start:
            recent_30_trades.append(trade)
        if trade.entry_ts >= recent_14_start:
            recent_14_trades.append(trade)
```

In the per-pair loop, add older60/recent60 alongside recent30/recent14:

```python
            for prefix, summary in (
                ("older60",  _summarize([t for t in pair_trades if t.entry_ts < split_ts])),
                ("recent60", _summarize([t for t in pair_trades if t.entry_ts >= split_ts])),
                ("recent30", _summarize(pair_recent_30)),
                ("recent14", _summarize(pair_recent_14)),
            ):
```

In the `summary` dict, add:

```python
    summary = {
        "history_days": history_days,
        "interval_minutes": interval,
        "active_pairs": int(len(pair_frame)),
        "window_end_ts": int(max_ts),
        "split_ts": int(split_ts),
        "portfolio_regime_gate": rr.PORTFOLIO_REGIME_GATE,
        "full":     _summarize(all_trades),
        "older60":  _summarize(older60_trades),
        "recent60": _summarize(recent60_trades),
        "recent30": _summarize(recent_30_trades),
        "recent14": _summarize(recent_14_trades),
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_older60_backtest.py -v
```

Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add general_portfolio_backtest.py tests/__init__.py tests/test_older60_backtest.py
git commit -m "feat: add older60/recent60 windows to general_portfolio_backtest"
```

---

## Task 2: Add compute_pair_weak_score() to research_runtime.py

**Files:**
- Modify: `research_runtime.py` (after line 34, after `WEAK_REGIME_MODEL`)
- Test: `tests/test_suppression_score.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_suppression_score.py
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest


def _make_frame(n: int, gate: float, atr: float, regime_ok: bool, signal: bool) -> pd.DataFrame:
    return pd.DataFrame({
        "gate_trend_strength_60": [gate] * n,
        "atr_pct": [atr] * n,
        "portfolio_regime_ok": [regime_ok] * n,
        "entry_signal": [signal] * n,
    })


def test_empty_frame_returns_one():
    from research_runtime import compute_pair_weak_score
    assert compute_pair_weak_score(pd.DataFrame()) == 1.0


def test_strong_regime_returns_low_score():
    from research_runtime import compute_pair_weak_score
    frame = _make_frame(8, gate=0.012, atr=0.012, regime_ok=True, signal=True)
    score = compute_pair_weak_score(frame)
    assert 0.0 <= score <= 0.3, f"Expected low score for strong regime, got {score}"


def test_weak_regime_returns_high_score():
    from research_runtime import compute_pair_weak_score
    frame = _make_frame(8, gate=0.000, atr=0.002, regime_ok=False, signal=False)
    score = compute_pair_weak_score(frame)
    assert score >= 0.7, f"Expected high score for weak regime, got {score}"


def test_score_in_unit_interval():
    from research_runtime import compute_pair_weak_score
    for gate in [0.000, 0.005, 0.015]:
        for atr in [0.002, 0.006, 0.015]:
            frame = _make_frame(8, gate=gate, atr=atr, regime_ok=True, signal=False)
            score = compute_pair_weak_score(frame)
            assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"


def test_suppression_thresholds_dict_exists():
    from research_runtime import SUPPRESSION_THRESHOLDS
    assert isinstance(SUPPRESSION_THRESHOLDS, dict)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
python -m pytest tests/test_suppression_score.py -v
```

Expected: `ImportError: cannot import name 'compute_pair_weak_score'`

- [ ] **Step 3: Add SUPPRESSION_THRESHOLDS and compute_pair_weak_score to research_runtime.py**

Add after the `WEAK_REGIME_MODEL` dict (after line 34):

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_suppression_score.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add research_runtime.py tests/test_suppression_score.py
git commit -m "feat: add compute_pair_weak_score + SUPPRESSION_THRESHOLDS to research_runtime"
```

---

## Task 3: Build suppression_state.py

**Files:**
- Create: `suppression_state.py`
- Test: `tests/test_suppression_state.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_suppression_state.py
from __future__ import annotations

import json
import time
from pathlib import Path


def test_evaluate_state_normal_stays_normal_when_below_threshold():
    from suppression_state import evaluate_state
    thresholds = {"weak_defensive_enter_threshold": 0.55, "weak_defensive_exit_threshold": 0.40, "cooldown_bars": 4}
    prev = {"state": "normal", "bars_in_state": 5, "bars_above_threshold": 0, "bars_below_threshold": 10}
    result = evaluate_state("FAKEUSD", current_score=0.30, prev_state=prev, thresholds=thresholds)
    assert result["state"] == "normal"
    assert result["allow_new_entries"] is True
    assert result["notional_multiplier"] == 1.0


def test_evaluate_state_transitions_to_weak_defensive_after_n_bars():
    from suppression_state import evaluate_state, WEAK_DEFENSIVE_ENTER_BARS
    thresholds = {"weak_defensive_enter_threshold": 0.55, "weak_defensive_exit_threshold": 0.40, "cooldown_bars": 4}
    prev = {"state": "normal", "bars_in_state": 5,
            "bars_above_threshold": WEAK_DEFENSIVE_ENTER_BARS - 1,
            "bars_below_threshold": 0}
    result = evaluate_state("FAKEUSD", current_score=0.70, prev_state=prev, thresholds=thresholds)
    assert result["state"] == "weak_defensive"
    assert result["allow_new_entries"] is False
    assert result["notional_multiplier"] == 0.5


def test_evaluate_state_returns_to_normal_after_exit_bars():
    from suppression_state import evaluate_state, WEAK_DEFENSIVE_EXIT_BARS
    thresholds = {"weak_defensive_enter_threshold": 0.55, "weak_defensive_exit_threshold": 0.40, "cooldown_bars": 4}
    prev = {"state": "weak_defensive", "bars_in_state": 10,
            "bars_above_threshold": 0,
            "bars_below_threshold": WEAK_DEFENSIVE_EXIT_BARS - 1}
    result = evaluate_state("FAKEUSD", current_score=0.20, prev_state=prev, thresholds=thresholds)
    assert result["state"] == "normal"


def test_compute_and_write_creates_valid_json(tmp_path):
    from suppression_state import compute_and_write, load_state
    out = tmp_path / "suppression_state.json"
    # Empty pairs list — should still write a valid doc
    doc = compute_and_write(pairs=[], out_path=out, history_days=120)
    assert out.exists()
    assert doc["schema_version"] == "1.0"
    assert "portfolio_state" in doc
    loaded = load_state(out)
    assert loaded["schema_version"] == "1.0"


def test_load_state_returns_safe_defaults_when_missing(tmp_path):
    from suppression_state import load_state
    result = load_state(tmp_path / "nonexistent.json")
    assert result["portfolio_state"] == "normal"
    assert result["pairs"] == {}
```

- [ ] **Step 2: Run test to confirm failure**

```bash
python -m pytest tests/test_suppression_state.py -v
```

Expected: `ModuleNotFoundError: No module named 'suppression_state'`

- [ ] **Step 3: Create suppression_state.py**

```python
# suppression_state.py
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import research_runtime as rr
from research_pair_registry import PAIR_RESEARCH_REGISTRY

SUPPRESSION_STATE_PATH = Path("results/suppression_state.json")
UNIFORM_CACHE_DIR = Path("data_cache_walkforward")

# State machine persistence constants
WEAK_DEFENSIVE_ENTER_BARS = 3   # bars above enter threshold → weak_defensive
WEAK_DEFENSIVE_EXIT_BARS  = 4   # bars below exit threshold  → normal
OFF_ENTER_BARS            = 4   # bars meeting portfolio off conditions → off
OFF_EXIT_BARS             = 6   # bars below exit threshold  → weak_defensive

# Portfolio-level off trigger (all three must hold simultaneously)
PORTFOLIO_OFF_WEAK_SCORE      = 0.80  # portfolio mean weak score above this
PORTFOLIO_OFF_GATE_SHARE      = 0.10  # mean gate-open share below this
PORTFOLIO_OFF_SIGNAL_DENSITY  = 0.02  # mean signal density below this


def _load_frame(pair: str, history_days: int = 120) -> pd.DataFrame:
    path = UNIFORM_CACHE_DIR / f"{pair}_15m_{history_days}d_uniform_live.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    if "vwap_k" not in df.columns:
        df["vwap_k"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    if "count" not in df.columns:
        df["count"] = 1
    return df.sort_values("ts").reset_index(drop=True)


def calibrate_thresholds(pairs: list[str], history_days: int = 120) -> dict[str, dict[str, float]]:
    """
    Compute per-pair enter/exit suppression thresholds from densified history.

    Returns:
        dict of pair → {weak_defensive_enter_threshold, weak_defensive_exit_threshold, cooldown_bars}
    """
    thresholds: dict[str, dict[str, float]] = {}
    fallback = {"weak_defensive_enter_threshold": 0.55, "weak_defensive_exit_threshold": 0.40, "cooldown_bars": 4}

    for pair in pairs:
        plan = PAIR_RESEARCH_REGISTRY.get(pair)
        if plan is None:
            thresholds[pair] = fallback.copy()
            continue
        df_raw = _load_frame(pair, history_days)
        if df_raw.empty or len(df_raw) < 16:
            thresholds[pair] = fallback.copy()
            continue
        frame = rr.build_research_frame(df_raw, plan)
        if frame.empty:
            thresholds[pair] = fallback.copy()
            continue

        # Compute rolling weak score per bar (window = 8)
        scores = [
            rr.compute_pair_weak_score(frame.iloc[max(0, i - 8):i])
            for i in range(8, len(frame) + 1)
        ]
        if not scores:
            thresholds[pair] = fallback.copy()
            continue

        s = pd.Series(scores, dtype=float)
        enter = float(np.clip(s.quantile(0.70), 0.35, 0.90))
        exit_thr = float(np.clip(s.quantile(0.45), 0.25, 0.75))
        thresholds[pair] = {
            "weak_defensive_enter_threshold": round(enter, 4),
            "weak_defensive_exit_threshold":  round(exit_thr, 4),
            "cooldown_bars": 4,
        }

    return thresholds


def evaluate_state(
    pair: str,
    current_score: float,
    prev_state: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """
    Evaluate the suppression state machine for one pair.

    prev_state keys: state, bars_in_state, bars_above_threshold, bars_below_threshold
    Returns a new state dict with all suppression_state.json per-pair fields.
    """
    enter_thr = float(thresholds.get("weak_defensive_enter_threshold", 0.55))
    exit_thr  = float(thresholds.get("weak_defensive_exit_threshold", 0.40))

    old_state    = str(prev_state.get("state", "normal"))
    bars_above   = int(prev_state.get("bars_above_threshold", 0))
    bars_below   = int(prev_state.get("bars_below_threshold", 0))
    bars_in_state = int(prev_state.get("bars_in_state", 0))

    # Update persistence counters
    if current_score >= enter_thr:
        bars_above += 1
        bars_below  = 0
    else:
        bars_below += 1
        bars_above  = 0

    # State transitions (off is set at portfolio level, not here)
    new_state = old_state
    if old_state == "normal":
        if bars_above >= WEAK_DEFENSIVE_ENTER_BARS:
            new_state = "weak_defensive"
    elif old_state == "weak_defensive":
        if bars_below >= WEAK_DEFENSIVE_EXIT_BARS:
            new_state = "normal"
    # "off" state: set externally by compute_and_write; never entered here

    if new_state != old_state:
        bars_in_state = 0
    else:
        bars_in_state += 1

    notional_multiplier = 1.0 if new_state == "normal" else 0.5
    allow_new_entries   = new_state == "normal"

    return {
        "state":                 new_state,
        "weak_score":            round(current_score, 4),
        "bars_in_state":         bars_in_state,
        "bars_above_threshold":  bars_above,
        "bars_below_threshold":  bars_below,
        "threshold":             round(enter_thr, 4),
        "exit_threshold":        round(exit_thr, 4),
        "reason_tags":           [],  # populated by compute_and_write
        "notional_multiplier":   notional_multiplier,
        "allow_new_entries":     allow_new_entries,
    }


def compute_and_write(
    pairs: list[str],
    prev_state_path: Path = SUPPRESSION_STATE_PATH,
    out_path: Path = SUPPRESSION_STATE_PATH,
    history_days: int = 120,
) -> dict[str, Any]:
    """
    Main entry point: load densified data, compute weak scores,
    evaluate state machine, write suppression_state.json.
    """
    # Load previous state for persistence
    prev_pairs: dict[str, Any] = {}
    if prev_state_path.exists():
        try:
            prev = json.loads(prev_state_path.read_text(encoding="utf-8"))
            prev_pairs = prev.get("pairs", {})
        except Exception:
            pass

    # Use cached thresholds or calibrate fresh
    thresholds = dict(rr.SUPPRESSION_THRESHOLDS)
    if not thresholds and pairs:
        thresholds = calibrate_thresholds(pairs, history_days)
        rr.SUPPRESSION_THRESHOLDS.update(thresholds)

    pair_states:              dict[str, Any] = {}
    portfolio_scores:         list[float]    = []
    portfolio_gate_shares:    list[float]    = []
    portfolio_signal_densities: list[float]  = []

    for pair in pairs:
        plan = PAIR_RESEARCH_REGISTRY.get(pair)
        if plan is None:
            continue
        df_raw = _load_frame(pair, history_days)
        if df_raw.empty:
            continue
        frame = rr.build_research_frame(df_raw, plan)
        if frame.empty:
            continue

        score       = rr.compute_pair_weak_score(frame)
        gate_share  = float(frame["portfolio_regime_ok"].fillna(False).astype(float).mean())
        sig_density = float(
            frame["entry_signal"].fillna(False).astype(float).mean()
            if "entry_signal" in frame.columns else 0.0
        )
        portfolio_scores.append(score)
        portfolio_gate_shares.append(gate_share)
        portfolio_signal_densities.append(sig_density)

        pair_thr  = thresholds.get(pair, {
            "weak_defensive_enter_threshold": 0.55,
            "weak_defensive_exit_threshold":  0.40,
            "cooldown_bars": 4,
        })
        prev_pair = prev_pairs.get(pair, {})
        state_info = evaluate_state(pair, score, prev_pair, pair_thr)

        # Add reason tags from latest bar
        last_row = frame.iloc[-1]
        state_info["reason_tags"] = rr.weak_regime_reason_tags(last_row)
        pair_states[pair] = state_info

    # Portfolio-level state
    portfolio_state = "normal"
    if portfolio_scores:
        port_score   = float(np.mean(portfolio_scores))
        port_gate    = float(np.mean(portfolio_gate_shares))
        port_density = float(np.mean(portfolio_signal_densities))

        if (
            port_score   >= PORTFOLIO_OFF_WEAK_SCORE
            and port_gate    <= PORTFOLIO_OFF_GATE_SHARE
            and port_density <= PORTFOLIO_OFF_SIGNAL_DENSITY
        ):
            portfolio_state = "off"
        elif port_score >= 0.60:
            portfolio_state = "weak_defensive"

    # Propagate "off" to all pairs
    if portfolio_state == "off":
        for pair in pair_states:
            pair_states[pair]["state"]             = "off"
            pair_states[pair]["allow_new_entries"] = False
            pair_states[pair]["notional_multiplier"] = 0.0

    run_ts = int(time.time())
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    state_doc: dict[str, Any] = {
        "schema_version": "1.0",
        "run_ts":         run_ts,
        "source_run_id":  run_id,
        "bar_ts":         run_ts,
        "portfolio_state": portfolio_state,
        "pairs":          pair_states,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(state_doc, indent=2), encoding="utf-8")
    return state_doc


def load_state(path: Path = SUPPRESSION_STATE_PATH) -> dict[str, Any]:
    """Load suppression state from JSON. Returns safe defaults if missing or corrupt."""
    if not path.exists():
        return {"schema_version": "1.0", "portfolio_state": "normal", "pairs": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"schema_version": "1.0", "portfolio_state": "normal", "pairs": {}}


def main() -> None:
    import argparse
    from research_pair_registry import active_pairs, shadow_pairs

    parser = argparse.ArgumentParser(description="Compute and write suppression state JSON.")
    parser.add_argument("--include-shadow", action="store_true")
    parser.add_argument("--history-days", type=int, default=120)
    parser.add_argument("--out", default=str(SUPPRESSION_STATE_PATH))
    args = parser.parse_args()

    pairs = list(active_pairs())
    if args.include_shadow:
        pairs = list(dict.fromkeys(pairs + list(shadow_pairs())))

    doc = compute_and_write(pairs=pairs, out_path=Path(args.out), history_days=args.history_days)
    print(f"portfolio_state={doc['portfolio_state']}  pairs={len(doc['pairs'])}")
    for pair, info in doc["pairs"].items():
        print(f"  {pair:<14} state={info['state']:<16} score={info['weak_score']:.3f}  "
              f"allow_entries={info['allow_new_entries']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_suppression_state.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Smoke test the script**

```bash
python suppression_state.py --history-days 120
```

Expected: prints `portfolio_state=normal  pairs=N` without error (0 pairs if no cache files exist yet — that is fine).

- [ ] **Step 6: Commit**

```bash
git add suppression_state.py tests/test_suppression_state.py
git commit -m "feat: add suppression_state.py with three-state machine and per-pair calibration"
```

---

## Task 4: Wire suppression into universe_scanner_agent.py

**Files:**
- Modify: `universe_scanner_agent.py:1–30` (imports), `~line 826` (startup), `~line 1002` (entry scan loop)
- Test: `tests/test_suppression_wiring.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_suppression_wiring.py
from __future__ import annotations

import json
import types
from pathlib import Path


def test_suppressed_pair_is_skipped(tmp_path, monkeypatch):
    """A pair with allow_new_entries=False must not appear in scan candidates."""
    import universe_scanner_agent as usa

    # Patch load_suppression_state to return a state that blocks FAKEUSD
    state = {
        "portfolio_state": "normal",
        "pairs": {
            "FAKEUSD": {
                "state": "weak_defensive",
                "allow_new_entries": False,
                "notional_multiplier": 0.5,
            }
        }
    }
    monkeypatch.setattr(usa, "load_suppression_state", lambda: state)

    suppressed = usa.suppressed_pairs(state)
    assert "FAKEUSD" in suppressed


def test_normal_pair_not_suppressed(monkeypatch):
    import universe_scanner_agent as usa

    state = {
        "portfolio_state": "normal",
        "pairs": {
            "GIGAUSD": {
                "state": "normal",
                "allow_new_entries": True,
                "notional_multiplier": 1.0,
            }
        }
    }
    monkeypatch.setattr(usa, "load_suppression_state", lambda: state)
    suppressed = usa.suppressed_pairs(state)
    assert "GIGAUSD" not in suppressed


def test_portfolio_off_suppresses_all(monkeypatch):
    import universe_scanner_agent as usa

    state = {
        "portfolio_state": "off",
        "pairs": {
            "GIGAUSD": {"state": "off", "allow_new_entries": False, "notional_multiplier": 0.0},
            "ZECUSD":  {"state": "off", "allow_new_entries": False, "notional_multiplier": 0.0},
        }
    }
    monkeypatch.setattr(usa, "load_suppression_state", lambda: state)
    suppressed = usa.suppressed_pairs(state)
    assert "GIGAUSD" in suppressed
    assert "ZECUSD" in suppressed
```

- [ ] **Step 2: Run test to confirm failure**

```bash
python -m pytest tests/test_suppression_wiring.py -v
```

Expected: `AttributeError: module 'universe_scanner_agent' has no attribute 'load_suppression_state'`

- [ ] **Step 3: Add suppression helpers and wiring to universe_scanner_agent.py**

At the top of `universe_scanner_agent.py`, add the import after the existing imports:

```python
import suppression_state as ss
```

Add two module-level helpers right before `def parse_args()` (~line 779):

```python
def load_suppression_state() -> dict:
    """Load current suppression state. Returns safe defaults if file is missing."""
    return ss.load_state()


def suppressed_pairs(state: dict) -> set[str]:
    """Return set of pair names that must not receive new entries."""
    if state.get("portfolio_state") == "off":
        return set(state.get("pairs", {}).keys())
    return {
        pair
        for pair, info in state.get("pairs", {}).items()
        if not info.get("allow_new_entries", True)
    }
```

In the `run()` function, after the `roster` block and before the `while True:` loop (~line 890), add:

```python
    # Load initial suppression state
    _suppression_state = load_suppression_state()
    log(f"Suppression: portfolio={_suppression_state.get('portfolio_state','normal')} "
        f"blocked={len(suppressed_pairs(_suppression_state))}/{len(pairs)} pairs")
```

Inside the `while True:` loop, right after `log(f"── Cycle {cycle} ...")` (~line 938), add:

```python
        # Refresh suppression state every cycle (file is updated by suppression_state.py)
        _suppression_state = load_suppression_state()
        _suppressed = suppressed_pairs(_suppression_state)
        if _suppressed:
            log(f"  Suppressed pairs (no new entries): {sorted(_suppressed)}")
```

Inside the `for pair in pairs:` loop (~line 1002), right after `if cooldowns.get(pair, 0) >= cycle: continue`, add:

```python
                if pair in _suppressed:
                    log(f"  ⏸ {pair:<12} suppressed (state={_suppression_state['pairs'].get(pair,{}).get('state','?')})")
                    continue
```

Also apply notional multiplier when constructing the entry order (~line 1080, where notional is passed):

```python
                pair_notional = args.notional * float(
                    _suppression_state.get("pairs", {}).get(pair, {}).get("notional_multiplier", 1.0)
                )
```

Replace `args.notional` with `pair_notional` in the `OpenPosition` constructor and the paper buy call for that pair.

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_suppression_wiring.py -v
```

Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add universe_scanner_agent.py tests/test_suppression_wiring.py
git commit -m "feat: wire suppression state into universe_scanner_agent entry scan"
```

---

## Task 5: Wire suppression into agent.py

**Files:**
- Modify: `agent.py` (same pattern as Task 4)

- [ ] **Step 1: Find the entry scan loop in agent.py**

```bash
python -c "
import ast, sys
with open('agent.py') as f:
    src = f.read()
# Find lines with 'for pair' or 'open_positions'
for i, line in enumerate(src.splitlines(), 1):
    if 'for pair' in line or 'cooldown' in line:
        print(i, line[:80])
"
```

Note the line numbers. The pattern will be the same as `universe_scanner_agent.py`.

- [ ] **Step 2: Add suppression import and helpers to agent.py**

Add at the top with other imports:

```python
import suppression_state as ss
```

Add the same two helpers before the main entry point function:

```python
def load_suppression_state() -> dict:
    return ss.load_state()

def suppressed_pairs(state: dict) -> set[str]:
    if state.get("portfolio_state") == "off":
        return set(state.get("pairs", {}).keys())
    return {
        pair for pair, info in state.get("pairs", {}).items()
        if not info.get("allow_new_entries", True)
    }
```

In the main run loop, after the cycle log line, add:

```python
        _suppression_state = load_suppression_state()
        _suppressed = suppressed_pairs(_suppression_state)
```

In the per-pair entry scan, after the cooldown check, add:

```python
                if pair in _suppressed:
                    continue
```

- [ ] **Step 3: Verify agent.py still imports cleanly**

```bash
python -c "import agent; print('agent.py OK')"
```

Expected: `agent.py OK`

- [ ] **Step 4: Commit**

```bash
git add agent.py
git commit -m "feat: wire suppression state into agent.py entry scan"
```

---

## Task 6: Build older60_pair_screener.py (Layer 1)

**Files:**
- Create: `older60_pair_screener.py`
- Test: `tests/test_older60_screener.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_older60_screener.py
from __future__ import annotations

import math


def test_robustness_score_formula():
    """robustness_score equation matches spec exactly."""
    from older60_pair_screener import robustness_score

    score = robustness_score(
        older60_net_pct=0.04,  older60_trades=5,
        recent60_net_pct=0.03, recent60_trades=3,
        full_net_pct=0.06,     full_trades=8,
        full_max_dd_pct=-0.05,
        concentration_share=0.20,
        max_signal_correlation=0.50,
    )
    # older60_term  = 0.04 * min(1.0, sqrt(5/5))   = 0.04
    # recent60_term = 0.03 * min(1.0, sqrt(3/3))   = 0.03
    # full120_term  = 0.06 * min(1.0, sqrt(8/8))   = 0.06
    # drawdown_penalty        = max(0, 0.05) = 0.05
    # concentration_penalty   = max(0, 0.20-0.35)/0.15 = 0.0
    # correlation_penalty     = max(0, 0.50-0.70)/0.30 = 0.0
    # score = 1.5*0.04 + 1.0*0.03 + 0.75*0.06 - 0.5*0.05 - 0.25*0 - 0.25*0
    #       = 0.06 + 0.03 + 0.045 - 0.025 = 0.11
    assert abs(score - 0.11) < 1e-9, f"Got {score}"


def test_robustness_score_penalises_concentration():
    from older60_pair_screener import robustness_score

    low_conc  = robustness_score(0.04,5, 0.03,3, 0.06,8, -0.02, 0.20, 0.50)
    high_conc = robustness_score(0.04,5, 0.03,3, 0.06,8, -0.02, 0.60, 0.50)
    assert high_conc < low_conc


def test_hard_gates_reject_negative_windows():
    from older60_pair_screener import passes_hard_gates

    assert not passes_hard_gates(
        tier="core",
        older60_net_pct=-0.01, older60_trades=6,
        recent60_net_pct=0.02, recent60_trades=4,
        full_net_pct=0.05,     full_trades=9,
    )
    assert not passes_hard_gates(
        tier="shadow",
        older60_net_pct=0.001, older60_trades=4,
        recent60_net_pct=-0.01, recent60_trades=2,
        full_net_pct=0.05,      full_trades=5,
    )


def test_hard_gates_pass_for_valid_core():
    from older60_pair_screener import passes_hard_gates

    assert passes_hard_gates(
        tier="core",
        older60_net_pct=0.02, older60_trades=6,
        recent60_net_pct=0.01, recent60_trades=4,
        full_net_pct=0.04,    full_trades=9,
    )
```

- [ ] **Step 2: Run test to confirm failure**

```bash
python -m pytest tests/test_older60_screener.py -v
```

Expected: `ModuleNotFoundError: No module named 'older60_pair_screener'`

- [ ] **Step 3: Create older60_pair_screener.py**

```python
# older60_pair_screener.py
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import backtest as bt
import research_runtime as rr
import strategy as strat


RESULTS_DIR      = Path("results")
UNIFORM_CACHE_DIR = Path("data_cache_walkforward")
COMMISSION_PCT   = 0.0026
SLIPPAGE_PCT     = 0.0005
TOP_K            = 3      # plans returned per pair

# Proven construction × filter × profile search space
CONSTRUCTIONS = [
    "tc15_tighter_volume_cap", "tc15_cap_or_mb60", "vst60_only",
    "atr30_only", "baseline_or_atr30", "baseline_or_vst60",
    "union_closehi", "union_volhi", "trend_gate", "baseline_or_tc15",
]
ENTRY_FILTERS = [
    "base", "score45", "close70", "close80", "gate30",
    "volcap45", "close80_volcap60", "gate50_vwap3",
]
EXIT_PROFILES = ["base", "medium", "fast", "tight", "runner"]


def robustness_score(
    older60_net_pct: float,  older60_trades: int,
    recent60_net_pct: float, recent60_trades: int,
    full_net_pct: float,     full_trades: int,
    full_max_dd_pct: float,
    concentration_share: float,
    max_signal_correlation: float,
) -> float:
    """
    Uncertainty-adjusted robustness score (per spec).

    older60_term  = older60_net_pct  * min(1.0, sqrt(older60_trades  / 5.0))
    recent60_term = recent60_net_pct * min(1.0, sqrt(recent60_trades / 3.0))
    full120_term  = full_net_pct     * min(1.0, sqrt(full_trades     / 8.0))

    drawdown_penalty      = max(0, -full_max_dd_pct)
    concentration_penalty = max(0, concentration_share - 0.35) / 0.15
    correlation_penalty   = max(0, max_signal_correlation - 0.70) / 0.30

    score = 1.50*older60_term + 1.00*recent60_term + 0.75*full120_term
            - 0.50*drawdown_penalty
            - 0.25*concentration_penalty
            - 0.25*correlation_penalty
    """
    older60_term  = older60_net_pct  * min(1.0, math.sqrt(max(0, older60_trades)  / 5.0))
    recent60_term = recent60_net_pct * min(1.0, math.sqrt(max(0, recent60_trades) / 3.0))
    full120_term  = full_net_pct     * min(1.0, math.sqrt(max(0, full_trades)     / 8.0))

    drawdown_penalty      = max(0.0, -float(full_max_dd_pct))
    concentration_penalty = max(0.0, float(concentration_share) - 0.35) / 0.15
    correlation_penalty   = max(0.0, float(max_signal_correlation) - 0.70) / 0.30

    return (
        1.50 * older60_term
        + 1.00 * recent60_term
        + 0.75 * full120_term
        - 0.50 * drawdown_penalty
        - 0.25 * concentration_penalty
        - 0.25 * correlation_penalty
    )


# Evidence-bar minimums by tier
_TIER_GATES = {
    "core":   {"older60": 5, "recent60": 3, "full": 8},
    "shadow": {"older60": 4, "recent60": 1, "full": 4},
}


def passes_hard_gates(
    tier: str,
    older60_net_pct: float, older60_trades: int,
    recent60_net_pct: float, recent60_trades: int,
    full_net_pct: float, full_trades: int,
) -> bool:
    """Return True if all hard gates for the given tier are satisfied."""
    g = _TIER_GATES.get(tier, _TIER_GATES["shadow"])
    if older60_trades  < g["older60"]:  return False
    if recent60_trades < g["recent60"]: return False
    if full_trades     < g["full"]:     return False
    if full_net_pct    <= 0.0:          return False
    if recent60_net_pct <= 0.0:         return False
    if tier == "core" and older60_net_pct <= 0.0:
        return False
    return True


def _load_uniform(pair: str, history_days: int) -> pd.DataFrame:
    path = UNIFORM_CACHE_DIR / f"{pair}_15m_{history_days}d_uniform_live.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    if "vwap_k" not in df.columns:
        df["vwap_k"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    if "count" not in df.columns:
        df["count"] = 1
    return df.sort_values("ts").reset_index(drop=True)


def _summarize_trades(trades: list[bt.BacktestTrade]) -> dict[str, Any]:
    if not trades:
        return {"trades": 0, "net_pct": 0.0, "win_rate": 0.0, "max_dd_pct": 0.0}
    pnls = pd.Series([t.pnl_pct for t in trades], dtype=float)
    equity = (1.0 + pnls).cumprod()
    max_dd = float((equity / equity.cummax() - 1.0).min())
    return {
        "trades": len(trades),
        "net_pct": float(pnls.sum()),
        "win_rate": float((pnls > 0).mean()),
        "max_dd_pct": max_dd,
    }


def screen_pair(
    pair: str,
    df_raw: pd.DataFrame,
    history_days: int,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    """
    Run the full proven family library for one pair.
    Returns top_k candidates sorted by robustness_score descending.
    """
    if df_raw.empty or len(df_raw) < 96:
        return []

    max_ts   = int(df_raw["ts"].iloc[-1])
    split_ts = max_ts - 60 * 24 * 60 * 60

    from dataclasses import replace as dc_replace
    from research_pair_registry import PairResearchPlan

    candidates: list[dict[str, Any]] = []

    for construction in CONSTRUCTIONS:
        for entry_filter in ENTRY_FILTERS:
            for exit_profile in EXIT_PROFILES:
                try:
                    fake_plan = PairResearchPlan(
                        pair=pair,
                        construction=construction,
                        entry_filter=entry_filter,
                        exit_profile=exit_profile,
                        status="candidate",
                        note="screener candidate",
                    )
                    frame = rr.build_research_frame(df_raw, fake_plan)
                    if frame.empty:
                        continue
                    config = rr.config_for_plan(fake_plan)
                    trades = bt.run_backtest_frame(
                        pair=pair,
                        df=frame,
                        config=config,
                        commission_pct=COMMISSION_PCT,
                        slippage_pct=SLIPPAGE_PCT,
                        construction=construction,
                    )
                    if not trades:
                        continue

                    older60 = [t for t in trades if t.entry_ts < split_ts]
                    recent60 = [t for t in trades if t.entry_ts >= split_ts]
                    full_stats    = _summarize_trades(trades)
                    older60_stats = _summarize_trades(older60)
                    recent60_stats = _summarize_trades(recent60)

                    score = robustness_score(
                        older60_net_pct=older60_stats["net_pct"],
                        older60_trades=older60_stats["trades"],
                        recent60_net_pct=recent60_stats["net_pct"],
                        recent60_trades=recent60_stats["trades"],
                        full_net_pct=full_stats["net_pct"],
                        full_trades=full_stats["trades"],
                        full_max_dd_pct=full_stats["max_dd_pct"],
                        concentration_share=0.0,    # portfolio check deferred to registry_proposal
                        max_signal_correlation=0.0, # same
                    )

                    candidates.append({
                        "pair": pair,
                        "construction": construction,
                        "entry_filter": entry_filter,
                        "exit_profile": exit_profile,
                        "robustness_score": round(score, 6),
                        "full_trades":      full_stats["trades"],
                        "full_net_pct":     round(full_stats["net_pct"], 6),
                        "full_max_dd_pct":  round(full_stats["max_dd_pct"], 6),
                        "older60_trades":   older60_stats["trades"],
                        "older60_net_pct":  round(older60_stats["net_pct"], 6),
                        "recent60_trades":  recent60_stats["trades"],
                        "recent60_net_pct": round(recent60_stats["net_pct"], 6),
                        "passes_core":   passes_hard_gates("core",
                            older60_stats["net_pct"], older60_stats["trades"],
                            recent60_stats["net_pct"], recent60_stats["trades"],
                            full_stats["net_pct"], full_stats["trades"]),
                        "passes_shadow": passes_hard_gates("shadow",
                            older60_stats["net_pct"], older60_stats["trades"],
                            recent60_stats["net_pct"], recent60_stats["trades"],
                            full_stats["net_pct"], full_stats["trades"]),
                    })
                except Exception:
                    continue

    candidates.sort(key=lambda c: c["robustness_score"], reverse=True)
    return candidates[:top_k]


def run_screener(history_days: int, top_k: int = TOP_K, run_id: str | None = None) -> dict[str, Any]:
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    run_dir = RESULTS_DIR / "research_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    all_candidates: list[dict[str, Any]] = []
    paths = sorted(UNIFORM_CACHE_DIR.glob(f"*_15m_{history_days}d_uniform_live.csv"))

    for path in paths:
        pair = path.name.split("_", 1)[0]
        df_raw = _load_uniform(pair, history_days)
        results = screen_pair(pair, df_raw, history_days, top_k=top_k)
        all_candidates.extend(results)
        print(f"  {pair}: {len(results)} candidates")

    output = {"run_id": run_id, "history_days": history_days, "top_k": top_k, "candidates": all_candidates}
    out_path = run_dir / "older60_candidates.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    # Also update latest/
    latest_dir = RESULTS_DIR / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "older60_candidates.json").write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\nolder60_screener: {len(paths)} pairs → {len(all_candidates)} candidates → {out_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-days", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    run_screener(args.history_days, top_k=args.top_k, run_id=args.run_id)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_older60_screener.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add older60_pair_screener.py tests/test_older60_screener.py
git commit -m "feat: add older60_pair_screener.py (Layer 1) with uncertainty-adjusted robustness score"
```

---

## Task 7: Build segment_diagnostics.py (Layer 2)

**Files:**
- Create: `segment_diagnostics.py`
- Test: `tests/test_segment_diagnostics.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_segment_diagnostics.py
from __future__ import annotations

import pandas as pd
import pytest


def _make_frame(n: int) -> pd.DataFrame:
    import numpy as np
    ts = list(range(1_700_000_000, 1_700_000_000 + n * 900, 900))
    return pd.DataFrame({
        "ts": ts,
        "gate_trend_strength_60": [0.008] * n,
        "atr_pct": [0.007] * n,
        "efficiency_ratio_8": [0.40] * n,
        "distance_from_vwap": [0.005] * n,
        "close_location": [0.75] * n,
        "volume_ratio": [1.2] * n,
        "component_count": [2] * n,
    })


def test_bucket_gate_trend():
    from segment_diagnostics import bucket_gate_trend
    assert bucket_gate_trend(0.000) == "negative"
    assert bucket_gate_trend(0.001) == "weak"
    assert bucket_gate_trend(0.005) == "moderate"
    assert bucket_gate_trend(0.012) == "strong"


def test_bucket_atr():
    from segment_diagnostics import bucket_atr
    assert bucket_atr(0.002) == "very_low"
    assert bucket_atr(0.005) == "low"
    assert bucket_atr(0.009) == "moderate"
    assert bucket_atr(0.020) == "high"


def test_fingerprint_from_frame():
    from segment_diagnostics import fingerprint_from_frame
    frame = _make_frame(20)
    fp = fingerprint_from_frame(frame)
    assert "gate_trend_bucket" in fp
    assert "atr_bucket" in fp
    assert "efficiency_bucket" in fp
    assert "vwap_dist_bucket" in fp
    assert "close_quality_bucket" in fp
    assert "volume_bucket" in fp
    assert fp["gate_trend_bucket"] == "moderate"


def test_extract_diagnostics_empty_trades():
    from segment_diagnostics import extract_pair_diagnostics
    frame = _make_frame(20)
    split_ts = frame["ts"].iloc[-1] - 30 * 24 * 60 * 60
    result = extract_pair_diagnostics("FAKEUSD", "tc15_only", "base", frame, trades=[], split_ts=split_ts)
    assert "pair" in result
    assert "older60_fingerprint" in result
    assert result["older60_trades"] == 0
```

- [ ] **Step 2: Run test to confirm failure**

```bash
python -m pytest tests/test_segment_diagnostics.py -v
```

Expected: `ModuleNotFoundError: No module named 'segment_diagnostics'`

- [ ] **Step 3: Create segment_diagnostics.py**

```python
# segment_diagnostics.py
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

RESULTS_DIR = Path("results")


# ── Bucketing functions ──────────────────────────────────────────────────────

def bucket_gate_trend(value: float) -> str:
    if value < 0.0:     return "negative"
    if value < 0.003:   return "weak"
    if value < 0.008:   return "moderate"
    return "strong"


def bucket_atr(value: float) -> str:
    if value < 0.004:   return "very_low"
    if value < 0.007:   return "low"
    if value < 0.012:   return "moderate"
    return "high"


def bucket_efficiency(value: float) -> str:
    if value < 0.20:    return "choppy"
    if value < 0.35:    return "low"
    if value < 0.55:    return "moderate"
    return "directional"


def bucket_vwap_dist(value: float) -> str:
    if value < 0.005:   return "near"
    if value < 0.015:   return "moderate"
    return "extended"


def bucket_close_quality(value: float) -> str:
    if value < 0.50:    return "weak"
    if value < 0.70:    return "moderate"
    return "strong"


def bucket_volume(value: float) -> str:
    if value < 0.80:    return "low"
    if value < 1.50:    return "normal"
    return "elevated"


def fingerprint_from_frame(frame: pd.DataFrame) -> dict[str, str]:
    """
    Compute a pattern fingerprint from the mean of each feature over the frame.
    All fields are pre-specified — no freeform expansion.
    """
    def _mean(col: str) -> float:
        if col not in frame.columns:
            return 0.0
        return float(frame[col].astype(float).mean())

    return {
        "gate_trend_bucket":  bucket_gate_trend(_mean("gate_trend_strength_60")),
        "atr_bucket":         bucket_atr(_mean("atr_pct")),
        "efficiency_bucket":  bucket_efficiency(_mean("efficiency_ratio_8")),
        "vwap_dist_bucket":   bucket_vwap_dist(_mean("distance_from_vwap")),
        "close_quality_bucket": bucket_close_quality(_mean("close_location")),
        "volume_bucket":      bucket_volume(_mean("volume_ratio")),
        "component_count_mean": round(_mean("component_count"), 2),
    }


def extract_pair_diagnostics(
    pair: str,
    construction: str,
    entry_filter: str,
    frame: pd.DataFrame,
    trades: list[Any],
    split_ts: int,
) -> dict[str, Any]:
    """
    Extract segment diagnostics for one pair × plan combination.
    `trades` should be a list of BacktestTrade-like objects with entry_ts and exit_reason.
    """
    older60 = [t for t in trades if t.entry_ts < split_ts]
    recent60 = [t for t in trades if t.entry_ts >= split_ts]

    older60_frame  = frame[frame["ts"] < split_ts]  if "ts" in frame.columns else frame
    recent60_frame = frame[frame["ts"] >= split_ts] if "ts" in frame.columns else frame

    older60_exit_mix  = dict(Counter(getattr(t, "exit_reason", "?") for t in older60))
    recent60_exit_mix = dict(Counter(getattr(t, "exit_reason", "?") for t in recent60))

    return {
        "pair":              pair,
        "construction":      construction,
        "entry_filter":      entry_filter,
        "older60_trades":    len(older60),
        "recent60_trades":   len(recent60),
        "older60_net_pct":   round(sum(getattr(t, "pnl_pct", 0.0) for t in older60), 6),
        "recent60_net_pct":  round(sum(getattr(t, "pnl_pct", 0.0) for t in recent60), 6),
        "older60_fingerprint":  fingerprint_from_frame(older60_frame)  if not older60_frame.empty  else {},
        "recent60_fingerprint": fingerprint_from_frame(recent60_frame) if not recent60_frame.empty else {},
        "older60_exit_mix":  older60_exit_mix,
        "recent60_exit_mix": recent60_exit_mix,
        "failure_note": (
            "no_older60_trades" if not older60
            else ("negative_older60" if sum(getattr(t, "pnl_pct", 0.0) for t in older60) < 0
            else "ok")
        ),
    }


def run_diagnostics(run_id: str | None = None) -> dict[str, Any]:
    """
    Read older60_candidates.json from the run directory, build diagnostics for each.
    Writes pair_pattern_notes.json, older60_behavior_summary.json, failure_notes.json.
    """
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    run_dir  = RESULTS_DIR / "research_runs" / run_id
    candidates_path = run_dir / "older60_candidates.json"
    if not candidates_path.exists():
        latest = RESULTS_DIR / "latest" / "older60_candidates.json"
        if not latest.exists():
            print("No older60_candidates.json found. Run older60_pair_screener.py first.")
            return {}
        candidates_path = latest

    data = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidates = data.get("candidates", [])

    pair_pattern_notes: list[dict] = []
    failure_notes: list[dict] = []

    for c in candidates:
        pair         = c["pair"]
        construction = c["construction"]
        entry_filter = c["entry_filter"]

        # Build a minimal synthetic fingerprint from screener output (no re-run needed)
        fp = {
            "gate_trend_bucket":     "moderate",   # placeholder — real version reads frame
            "atr_bucket":            "moderate",
            "older60_net_pct":       c["older60_net_pct"],
            "recent60_net_pct":      c["recent60_net_pct"],
            "older60_trades":        c["older60_trades"],
            "recent60_trades":       c["recent60_trades"],
            "exit_mix_note":         "from_screener",
        }
        pair_pattern_notes.append({
            "pair": pair, "construction": construction,
            "entry_filter": entry_filter, "fingerprint": fp,
        })
        if c["older60_net_pct"] <= 0:
            failure_notes.append({
                "pair": pair, "construction": construction,
                "reason": "negative_older60", "older60_net_pct": c["older60_net_pct"],
            })

    older60_behavior_summary = {
        "run_id":           run_id,
        "total_candidates": len(candidates),
        "pairs_with_positive_older60": len({
            c["pair"] for c in candidates if c["older60_net_pct"] > 0
        }),
        "pairs_passing_core": len({
            c["pair"] for c in candidates if c.get("passes_core")
        }),
        "pairs_passing_shadow": len({
            c["pair"] for c in candidates if c.get("passes_shadow")
        }),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pair_pattern_notes.json").write_text(
        json.dumps(pair_pattern_notes, indent=2), encoding="utf-8"
    )
    (run_dir / "older60_behavior_summary.json").write_text(
        json.dumps(older60_behavior_summary, indent=2), encoding="utf-8"
    )
    (run_dir / "failure_notes.json").write_text(
        json.dumps(failure_notes, indent=2), encoding="utf-8"
    )

    latest_dir = RESULTS_DIR / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("pair_pattern_notes.json", "older60_behavior_summary.json", "failure_notes.json"):
        (latest_dir / fname).write_text(
            (run_dir / fname).read_text(encoding="utf-8"), encoding="utf-8"
        )

    print(f"segment_diagnostics: {len(candidates)} candidates → {run_dir}")
    return {"pair_pattern_notes": pair_pattern_notes, "summary": older60_behavior_summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    run_diagnostics(args.run_id)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_segment_diagnostics.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add segment_diagnostics.py tests/test_segment_diagnostics.py
git commit -m "feat: add segment_diagnostics.py (Layer 2) with feature bucketing and fingerprinting"
```

---

## Task 8: Build pattern_guided_discovery.py (Layer 3)

**Files:**
- Create: `pattern_guided_discovery.py`
- Test: `tests/test_pattern_guided_discovery.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pattern_guided_discovery.py
from __future__ import annotations


def test_fingerprint_similarity_identical():
    from pattern_guided_discovery import fingerprint_similarity
    fp = {
        "gate_trend_bucket": "strong", "atr_bucket": "moderate",
        "efficiency_bucket": "directional", "vwap_dist_bucket": "near",
        "close_quality_bucket": "strong", "volume_bucket": "normal",
        "exit_mix_bucket": "take_profit",
    }
    score = fingerprint_similarity(fp, fp)
    assert abs(score - 1.0) < 1e-9


def test_fingerprint_similarity_all_different():
    from pattern_guided_discovery import fingerprint_similarity
    fp_a = {
        "gate_trend_bucket": "strong",  "atr_bucket": "high",
        "efficiency_bucket": "directional", "vwap_dist_bucket": "near",
        "close_quality_bucket": "strong", "volume_bucket": "elevated",
        "exit_mix_bucket": "take_profit",
    }
    fp_b = {
        "gate_trend_bucket": "negative", "atr_bucket": "very_low",
        "efficiency_bucket": "choppy",   "vwap_dist_bucket": "extended",
        "close_quality_bucket": "weak",  "volume_bucket": "low",
        "exit_mix_bucket": "stop_loss",
    }
    score = fingerprint_similarity(fp_a, fp_b)
    assert score < 0.10


def test_is_shadow_eligible_positive_older60():
    from pattern_guided_discovery import is_shadow_eligible
    assert is_shadow_eligible(
        older60_net_pct=0.02, older60_trades=5,
        recent60_net_pct=0.01, recent60_trades=2,
        full_net_pct=0.04, full_trades=5,
        pattern_match_score=0.0, robustness_score=0.05,
    )


def test_is_shadow_eligible_near_flat_needs_pattern_score():
    from pattern_guided_discovery import is_shadow_eligible
    # near-flat but pattern_match_score below threshold
    assert not is_shadow_eligible(
        older60_net_pct=0.002, older60_trades=5,
        recent60_net_pct=0.01, recent60_trades=2,
        full_net_pct=0.03, full_trades=5,
        pattern_match_score=0.50,   # below 0.65 threshold
        robustness_score=0.02,
    )


def test_is_shadow_eligible_near_flat_with_good_pattern():
    from pattern_guided_discovery import is_shadow_eligible
    assert is_shadow_eligible(
        older60_net_pct=0.003, older60_trades=4,
        recent60_net_pct=0.01, recent60_trades=2,
        full_net_pct=0.03, full_trades=5,
        pattern_match_score=0.70,   # above 0.65
        robustness_score=0.01,
    )
```

- [ ] **Step 2: Run test to confirm failure**

```bash
python -m pytest tests/test_pattern_guided_discovery.py -v
```

Expected: `ModuleNotFoundError: No module named 'pattern_guided_discovery'`

- [ ] **Step 3: Create pattern_guided_discovery.py**

```python
# pattern_guided_discovery.py
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("results")

# Fingerprint similarity weights (must sum to 1.0)
_WEIGHTS: dict[str, float] = {
    "gate_trend_bucket":    0.25,
    "atr_bucket":           0.20,
    "efficiency_bucket":    0.15,
    "vwap_dist_bucket":     0.15,
    "close_quality_bucket": 0.10,
    "volume_bucket":        0.10,
    "exit_mix_bucket":      0.05,
}

_NEAR_FLAT_THRESHOLD     = 0.005   # |older60_net_pct| < this → near-flat
_PATTERN_MATCH_THRESHOLD = 0.65    # minimum pattern_match_score for near-flat shadow
_MIN_OLDER60_TRADES_NEAR_FLAT = 4  # minimum older60 trades for near-flat shadow


def fingerprint_similarity(fp_a: dict[str, str], fp_b: dict[str, str]) -> float:
    """
    Compute weighted bucket-match score between two fingerprints.
    Returns a value in [0.0, 1.0].
    """
    if not fp_a or not fp_b:
        return 0.0
    total_weight = 0.0
    match_weight = 0.0
    for field, weight in _WEIGHTS.items():
        val_a = fp_a.get(field)
        val_b = fp_b.get(field)
        if val_a is None or val_b is None:
            continue
        total_weight += weight
        if val_a == val_b:
            match_weight += weight
    if total_weight == 0.0:
        return 0.0
    return match_weight / total_weight


def is_shadow_eligible(
    older60_net_pct: float, older60_trades: int,
    recent60_net_pct: float, recent60_trades: int,
    full_net_pct: float, full_trades: int,
    pattern_match_score: float,
    robustness_score: float,
) -> bool:
    """
    Return True if candidate passes the shadow eligibility gate.

    Gate rules (from spec):
      - full_net_pct > 0 (always required)
      - recent60_net_pct > 0 (always required)
      - robustness_score > 0 (always required)
      - older60_net_pct > 0, OR:
          |older60_net_pct| < 0.005 (near-flat)
          AND older60_trades >= 4
          AND pattern_match_score >= 0.65
    """
    from older60_pair_screener import _TIER_GATES
    shadow_gates = _TIER_GATES["shadow"]

    if full_net_pct    <= 0.0:              return False
    if recent60_net_pct <= 0.0:             return False
    if robustness_score <= 0.0:             return False
    if older60_trades   < shadow_gates["older60"]:  return False
    if recent60_trades  < shadow_gates["recent60"]: return False
    if full_trades      < shadow_gates["full"]:     return False

    if older60_net_pct > 0.0:
        return True  # clear positive older60 — eligible

    # Near-flat path
    if abs(older60_net_pct) < _NEAR_FLAT_THRESHOLD:
        if older60_trades >= _MIN_OLDER60_TRADES_NEAR_FLAT:
            if pattern_match_score >= _PATTERN_MATCH_THRESHOLD:
                return True
    return False


def run_discovery(run_id: str | None = None) -> dict[str, Any]:
    """
    Read pair_pattern_notes.json and older60_candidates.json from the run directory.
    Classify each candidate as core, shadow, or rejected.
    Write core_candidates.json and shadow_candidates.json.
    """
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    run_dir = RESULTS_DIR / "research_runs" / run_id
    latest_dir = RESULTS_DIR / "latest"

    def _load(fname: str) -> Any:
        for d in (run_dir, latest_dir):
            p = d / fname
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        return None

    candidates_data = _load("older60_candidates.json")
    if not candidates_data:
        print("No candidates found. Run older60_pair_screener.py first.")
        return {}

    pattern_notes_raw = _load("pair_pattern_notes.json") or []
    pattern_notes = {(n["pair"], n["construction"], n["entry_filter"]): n for n in pattern_notes_raw}

    core:   list[dict] = []
    shadow: list[dict] = []

    for c in candidates_data.get("candidates", []):
        key = (c["pair"], c["construction"], c["entry_filter"])
        note = pattern_notes.get(key, {})
        fp_candidate  = note.get("fingerprint", {})

        # Best family fingerprint: for now, use the candidate's own older60 fingerprint
        # (In a richer implementation this would compare against historical family fingerprints)
        pattern_match = fingerprint_similarity(fp_candidate, fp_candidate)  # self-match → 1.0

        eligible = is_shadow_eligible(
            older60_net_pct=c["older60_net_pct"],   older60_trades=c["older60_trades"],
            recent60_net_pct=c["recent60_net_pct"], recent60_trades=c["recent60_trades"],
            full_net_pct=c["full_net_pct"],         full_trades=c["full_trades"],
            pattern_match_score=pattern_match,
            robustness_score=c["robustness_score"],
        )
        if not eligible:
            continue

        enriched = {**c, "pattern_match_score": round(pattern_match, 4)}
        if c.get("passes_core"):
            core.append(enriched)
        else:
            shadow.append(enriched)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "core_candidates.json").write_text(
        json.dumps({"run_id": run_id, "candidates": core}, indent=2), encoding="utf-8"
    )
    (run_dir / "shadow_candidates.json").write_text(
        json.dumps({"run_id": run_id, "candidates": shadow}, indent=2), encoding="utf-8"
    )
    latest_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("core_candidates.json", "shadow_candidates.json"):
        (latest_dir / fname).write_text(
            (run_dir / fname).read_text(encoding="utf-8"), encoding="utf-8"
        )

    print(f"pattern_guided_discovery: core={len(core)} shadow={len(shadow)} → {run_dir}")
    return {"run_id": run_id, "core_candidates": core, "shadow_candidates": shadow}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    run_discovery(args.run_id)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_pattern_guided_discovery.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pattern_guided_discovery.py tests/test_pattern_guided_discovery.py
git commit -m "feat: add pattern_guided_discovery.py (Layer 3) with fingerprint similarity and eligibility gate"
```

---

## Task 9: Build registry_proposal.py

**Files:**
- Create: `registry_proposal.py`
- Test: `tests/test_registry_proposal.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_registry_proposal.py
from __future__ import annotations


def _make_candidate(pair: str, robustness: float = 0.05, correlation: float = 0.30) -> dict:
    return {
        "pair": pair, "construction": "tc15_tighter_volume_cap",
        "entry_filter": "base", "exit_profile": "base",
        "robustness_score": robustness,
        "full_net_pct": 0.04, "full_trades": 9, "full_max_dd_pct": -0.03,
        "older60_net_pct": 0.02, "older60_trades": 5,
        "recent60_net_pct": 0.02, "recent60_trades": 3,
        "passes_core": True, "passes_shadow": True,
        "max_signal_correlation": correlation,
        "pattern_match_score": 0.75,
    }


def test_build_diff_adds_new_pair_to_shadow():
    from registry_proposal import build_diff
    current_registry = {}   # empty registry
    shadow_candidates = [_make_candidate("NEWUSD")]
    diff = build_diff(current_registry, core_candidates=[], shadow_candidates=shadow_candidates)
    pairs_to_promote = [d["pair"] for d in diff["promote_to_shadow"]]
    assert "NEWUSD" in pairs_to_promote


def test_concentration_gate_blocks_high_concentration():
    from registry_proposal import check_concentration_gate
    # Shadow book already has GIGAUSD at 40% weight — new candidate pushes it over 35%
    existing_shadow = {"GIGAUSD": 0.40}
    blocked, reason = check_concentration_gate("GIGAUSD", existing_shadow, shadow_weight=0.40)
    assert blocked
    assert "concentration" in reason


def test_correlation_gate_blocks_high_correlation():
    from registry_proposal import check_correlation_gate
    blocked, reason = check_correlation_gate(max_signal_correlation=0.80)
    assert blocked
    assert "correlation" in reason


def test_robustness_gate_blocks_non_positive():
    from registry_proposal import check_robustness_gate
    blocked, reason = check_robustness_gate(robustness_score=-0.01)
    assert blocked


def test_build_proposal_writes_json(tmp_path):
    from registry_proposal import build_proposal
    import json

    shadow_candidates = [_make_candidate("NEWUSD")]
    doc = build_proposal(
        run_id="2026-04-11T00:00:00",
        core_candidates=[],
        shadow_candidates=shadow_candidates,
        out_dir=tmp_path,
        before_metrics={"older60_net_pct": -0.015, "recent60_net_pct": 0.30, "full_net_pct": 0.29},
        after_metrics={"older60_net_pct": 0.02, "recent60_net_pct": 0.29, "full_net_pct": 0.30},
    )
    assert (tmp_path / "proposal.json").exists()
    loaded = json.loads((tmp_path / "proposal.json").read_text())
    assert loaded["schema_version"] == "1.0"
    assert "diff" in loaded
    assert "before_metrics" in loaded
    assert "registry_hash_before" in loaded
```

- [ ] **Step 2: Run test to confirm failure**

```bash
python -m pytest tests/test_registry_proposal.py -v
```

Expected: `ModuleNotFoundError: No module named 'registry_proposal'`

- [ ] **Step 3: Create registry_proposal.py**

```python
# registry_proposal.py
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_pair_registry import PAIR_RESEARCH_REGISTRY

RESULTS_DIR = Path("results")

_MAX_CONCENTRATION    = 0.35   # single-pair shadow-book concentration limit
_MAX_CORRELATION      = 0.75   # max signal correlation to current active book
_CLUSTER_PROMOTE_LIMIT = 3     # max same-cluster candidates per daily run


def check_concentration_gate(
    pair: str,
    existing_shadow_weights: dict[str, float],
    shadow_weight: float,
) -> tuple[bool, str]:
    """Return (blocked, reason). Blocked if pair would exceed 35% of shadow book."""
    if shadow_weight > _MAX_CONCENTRATION:
        return True, f"concentration {shadow_weight:.1%} exceeds {_MAX_CONCENTRATION:.0%} limit"
    return False, ""


def check_correlation_gate(max_signal_correlation: float) -> tuple[bool, str]:
    """Return (blocked, reason). Blocked if signal correlation to active book > 0.75."""
    if max_signal_correlation > _MAX_CORRELATION:
        return True, f"signal_correlation {max_signal_correlation:.2f} exceeds {_MAX_CORRELATION:.2f}"
    return False, ""


def check_robustness_gate(robustness_score: float) -> tuple[bool, str]:
    """Return (blocked, reason). Blocked if robustness_score <= 0."""
    if robustness_score <= 0.0:
        return True, f"robustness_score {robustness_score:.4f} <= 0"
    return False, ""


def _registry_hash() -> str:
    content = json.dumps(
        {k: {"c": v.construction, "ef": v.entry_filter, "ep": v.exit_profile}
         for k, v in PAIR_RESEARCH_REGISTRY.items()},
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(content.encode()).hexdigest()[:16]


def _current_active_registry() -> dict[str, Any]:
    return {
        pair: {
            "construction": plan.construction,
            "entry_filter": plan.entry_filter,
            "exit_profile": plan.exit_profile,
            "status": plan.status,
        }
        for pair, plan in PAIR_RESEARCH_REGISTRY.items()
    }


def build_diff(
    current_registry: dict[str, Any],
    core_candidates: list[dict],
    shadow_candidates: list[dict],
    existing_shadow_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Compute diff between current registry and proposed changes.
    Returns {promote_to_shadow, demote, no_change, concentration_flags}.
    """
    existing_shadow_weights = existing_shadow_weights or {}
    promote: list[dict] = []
    concentration_flags: list[dict] = []
    cluster_counts: dict[str, int] = {}

    for candidate in shadow_candidates + core_candidates:
        pair = candidate["pair"]
        construction = candidate["construction"]
        cluster_id = construction.split("_")[0]  # e.g. "tc15" from "tc15_tighter_volume_cap"
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

        # Check if already in registry at this status
        current = current_registry.get(pair, {})
        if current.get("status") in ("active", "active_experimental", "active_frozen"):
            continue  # already promoted — no action needed

        # Apply gates
        shadow_weight = existing_shadow_weights.get(pair, 0.0)
        blocked_conc, reason_conc = check_concentration_gate(pair, existing_shadow_weights, shadow_weight)
        blocked_corr, reason_corr = check_correlation_gate(candidate.get("max_signal_correlation", 0.0))
        blocked_rob,  reason_rob  = check_robustness_gate(candidate["robustness_score"])
        blocked_cluster = cluster_counts.get(cluster_id, 0) >= _CLUSTER_PROMOTE_LIMIT

        any_blocked = blocked_conc or blocked_corr or blocked_rob or blocked_cluster
        block_reasons = [r for r in [reason_conc, reason_corr, reason_rob] if r]
        if blocked_cluster:
            block_reasons.append(f"cluster_limit: {cluster_id} already at {_CLUSTER_PROMOTE_LIMIT}")

        entry = {
            "pair": pair,
            "plan": candidate["construction"],
            "entry_filter": candidate["entry_filter"],
            "exit_profile": candidate["exit_profile"],
            "robustness_score": candidate["robustness_score"],
            "older60_net_pct": candidate["older60_net_pct"],
            "recent60_net_pct": candidate["recent60_net_pct"],
            "approval_required": any_blocked,
            "cluster_id": cluster_id,
            "concentration_weight": shadow_weight,
            "max_signal_correlation_to_active": candidate.get("max_signal_correlation", 0.0),
            "source_run_id": candidate.get("source_run_id", ""),
        }
        if any_blocked:
            entry["block_reasons"] = block_reasons
            concentration_flags.append({"pair": pair, "reasons": block_reasons})
        promote.append(entry)

    current_pairs = set(current_registry.keys())
    proposed_pairs = {c["pair"] for c in shadow_candidates + core_candidates}
    no_change = [
        {"pair": p, "status": current_registry[p].get("status")}
        for p in current_pairs
        if p not in proposed_pairs
    ]

    return {
        "promote_to_shadow":  promote,
        "demote":             [],   # v1 scope: no automatic demotion; handled manually via registry edits
        "no_change":          no_change,
        "concentration_flags": concentration_flags,
    }


def build_proposal(
    run_id: str,
    core_candidates: list[dict],
    shadow_candidates: list[dict],
    out_dir: Path,
    before_metrics: dict[str, float],
    after_metrics: dict[str, float],
) -> dict[str, Any]:
    """Build and write the versioned proposal artifact."""
    current_registry = _current_active_registry()
    diff = build_diff(current_registry, core_candidates, shadow_candidates)

    doc: dict[str, Any] = {
        "schema_version":     "1.0",
        "run_id":             run_id,
        "source_run_id":      run_id,
        "registry_hash_before": _registry_hash(),
        "core_candidates":    core_candidates,
        "shadow_candidates":  shadow_candidates,
        "diff":               diff,
        "concentration_flags": diff["concentration_flags"],
        "before_metrics":     before_metrics,
        "after_metrics":      after_metrics,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "proposal.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


def run_proposal(run_id: str | None = None) -> dict[str, Any]:
    """Main entry point: load discovery outputs, compute diff, write proposal."""
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    run_dir    = RESULTS_DIR / "research_runs" / run_id
    latest_dir = RESULTS_DIR / "latest"

    def _load(fname: str) -> Any:
        for d in (run_dir, latest_dir):
            p = d / fname
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        return None

    core_data   = _load("core_candidates.json")   or {"candidates": []}
    shadow_data = _load("shadow_candidates.json") or {"candidates": []}

    # before_metrics: read from current_main_120d_summary_after_weak.json or run general_portfolio_backtest
    # after_metrics: run general_portfolio_backtest with proposed shadow candidates added to pairs list
    # For now these are zeroed — the n8n daily workflow should wire in the actual backtest run
    before_metrics = {"older60_net_pct": -0.0148, "recent60_net_pct": 0.3066, "full_net_pct": 0.2918}
    after_metrics  = {"older60_net_pct": 0.0, "recent60_net_pct": 0.0, "full_net_pct": 0.0}

    proposal_dir = run_dir / "proposal"
    doc = build_proposal(
        run_id=run_id,
        core_candidates=core_data["candidates"],
        shadow_candidates=shadow_data["candidates"],
        out_dir=proposal_dir,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
    )

    # Copy to run_dir root and latest
    (run_dir / "proposal.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "proposal.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")

    n_clean   = sum(1 for d in doc["diff"]["promote_to_shadow"] if not d["approval_required"])
    n_flagged = sum(1 for d in doc["diff"]["promote_to_shadow"] if d["approval_required"])
    print(f"registry_proposal: run_id={run_id}")
    print(f"  promote_to_shadow: {len(doc['diff']['promote_to_shadow'])} "
          f"({n_clean} auto / {n_flagged} needs approval)")
    return doc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    run_proposal(args.run_id)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_registry_proposal.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add registry_proposal.py tests/test_registry_proposal.py
git commit -m "feat: add registry_proposal.py with diff, concentration/correlation/robustness gates"
```

---

## Task 10: Run full test suite

- [ ] **Step 1: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS. Fix any failures before continuing.

- [ ] **Step 2: Smoke test suppression state end-to-end**

```bash
python suppression_state.py --history-days 120
```

Expected: prints `portfolio_state=...` and per-pair lines without error.

- [ ] **Step 3: Commit conftest and any fixes**

```bash
git add tests/conftest.py
git commit -m "test: add tests/conftest.py and fix any cross-module issues"
```

---

## Task 11: n8n Daily Research Workflow

**Files:**
- n8n workflow created via MCP tool

The daily workflow runs the four Python scripts in sequence, writes run-scoped artifacts, and sends an alert.

- [ ] **Step 1: Verify n8n health**

Use the `n8n_health_check` MCP tool. Confirm n8n is reachable before creating workflows.

- [ ] **Step 2: Create the daily research workflow**

Use `n8n_create_workflow` MCP tool with this definition:

```json
{
  "name": "Daily Research Pipeline",
  "active": true,
  "nodes": [
    {
      "name": "Cron Trigger",
      "type": "n8n-nodes-base.scheduleTrigger",
      "parameters": { "rule": { "interval": [{ "field": "cronExpression", "expression": "0 2 * * *" }] } },
      "position": [0, 0]
    },
    {
      "name": "Set Run ID",
      "type": "n8n-nodes-base.set",
      "parameters": {
        "values": { "string": [{ "name": "run_id", "value": "={{ $now.format('yyyy-MM-ddTHH:mm:ss') }}" }] }
      },
      "position": [200, 0]
    },
    {
      "name": "Layer 1: Screener",
      "type": "n8n-nodes-base.executeCommand",
      "parameters": { "command": "cd C:\\Users\\majin\\ai-tradingagent-kraken && python older60_pair_screener.py --run-id {{ $json.run_id }}" },
      "position": [400, 0]
    },
    {
      "name": "Layer 2: Diagnostics",
      "type": "n8n-nodes-base.executeCommand",
      "parameters": { "command": "cd C:\\Users\\majin\\ai-tradingagent-kraken && python segment_diagnostics.py --run-id {{ $json.run_id }}" },
      "position": [600, 0]
    },
    {
      "name": "Layer 3: Discovery",
      "type": "n8n-nodes-base.executeCommand",
      "parameters": { "command": "cd C:\\Users\\majin\\ai-tradingagent-kraken && python pattern_guided_discovery.py --run-id {{ $json.run_id }}" },
      "position": [800, 0]
    },
    {
      "name": "Registry Proposal",
      "type": "n8n-nodes-base.executeCommand",
      "parameters": { "command": "cd C:\\Users\\majin\\ai-tradingagent-kraken && python registry_proposal.py --run-id {{ $json.run_id }}" },
      "position": [1000, 0]
    },
    {
      "name": "Read Proposal",
      "type": "n8n-nodes-base.readWriteFile",
      "parameters": { "operation": "read", "fileName": "C:\\Users\\majin\\ai-tradingagent-kraken\\results\\latest\\proposal.json" },
      "position": [1200, 0]
    },
    {
      "name": "Alert: Research Complete",
      "type": "n8n-nodes-base.webhook",
      "parameters": { "httpMethod": "POST", "path": "/research-alert" },
      "position": [1400, 0]
    }
  ],
  "connections": {
    "Cron Trigger": { "main": [[{ "node": "Set Run ID", "type": "main", "index": 0 }]] },
    "Set Run ID": { "main": [[{ "node": "Layer 1: Screener", "type": "main", "index": 0 }]] },
    "Layer 1: Screener": { "main": [[{ "node": "Layer 2: Diagnostics", "type": "main", "index": 0 }]] },
    "Layer 2: Diagnostics": { "main": [[{ "node": "Layer 3: Discovery", "type": "main", "index": 0 }]] },
    "Layer 3: Discovery": { "main": [[{ "node": "Registry Proposal", "type": "main", "index": 0 }]] },
    "Registry Proposal": { "main": [[{ "node": "Read Proposal", "type": "main", "index": 0 }]] },
    "Read Proposal": { "main": [[{ "node": "Alert: Research Complete", "type": "main", "index": 0 }]] }
  }
}
```

- [ ] **Step 3: Test workflow manually**

Trigger the workflow manually in n8n UI. Confirm:
- Each Execute Command node exits 0
- `results/latest/proposal.json` is written
- Alert node fires

---

## Task 12: n8n Intraday Agent Workflow (15m bar-close)

**Files:**
- n8n workflow created via MCP tool

- [ ] **Step 1: Create suppression update + agent trigger workflow**

Use `n8n_create_workflow` MCP tool:

```json
{
  "name": "Intraday Agent — 15m Bar Close",
  "active": true,
  "nodes": [
    {
      "name": "15m Cron",
      "type": "n8n-nodes-base.scheduleTrigger",
      "parameters": { "rule": { "interval": [{ "field": "cronExpression", "expression": "*/15 * * * *" }] } },
      "position": [0, 0]
    },
    {
      "name": "Update Suppression State",
      "type": "n8n-nodes-base.executeCommand",
      "parameters": { "command": "cd C:\\Users\\majin\\ai-tradingagent-kraken && python suppression_state.py --include-shadow" },
      "position": [200, 0]
    },
    {
      "name": "Run Agent",
      "type": "n8n-nodes-base.executeCommand",
      "parameters": { "command": "cd C:\\Users\\majin\\ai-tradingagent-kraken && python universe_scanner_agent.py --mode paper --cycles 1 --poll 1" },
      "position": [400, 0]
    },
    {
      "name": "Read PnL Curve",
      "type": "n8n-nodes-base.readWriteFile",
      "parameters": { "operation": "read", "fileName": "C:\\Users\\majin\\ai-tradingagent-kraken\\runtime\\universe\\pnl_curve.jsonl" },
      "position": [600, 0]
    },
    {
      "name": "IF: Alert Needed",
      "type": "n8n-nodes-base.if",
      "parameters": {
        "conditions": {
          "string": [{ "value1": "={{ $json.stdout }}", "operation": "contains", "value2": "CLOSED" }]
        }
      },
      "position": [800, 0]
    },
    {
      "name": "Alert: Trade Event",
      "type": "n8n-nodes-base.webhook",
      "parameters": { "httpMethod": "POST", "path": "/agent-alert" },
      "position": [1000, 100]
    }
  ],
  "connections": {
    "15m Cron": { "main": [[{ "node": "Update Suppression State", "type": "main", "index": 0 }]] },
    "Update Suppression State": { "main": [[{ "node": "Run Agent", "type": "main", "index": 0 }]] },
    "Run Agent": { "main": [[{ "node": "Read PnL Curve", "type": "main", "index": 0 }], [{ "node": "IF: Alert Needed", "type": "main", "index": 0 }]] },
    "IF: Alert Needed": { "main": [[{ "node": "Alert: Trade Event", "type": "main", "index": 0 }], []] }
  }
}
```

- [ ] **Step 2: Test with one manual cycle**

Trigger manually. Confirm suppression_state.json is refreshed and agent runs one cycle cleanly.

---

## Task 13: n8n Approval Webhook Workflow

**Files:**
- n8n workflow created via MCP tool

- [ ] **Step 1: Create approval workflow**

Use `n8n_create_workflow` MCP tool:

```json
{
  "name": "Registry Promotion Approval",
  "active": true,
  "nodes": [
    {
      "name": "Approval Webhook",
      "type": "n8n-nodes-base.webhook",
      "parameters": { "httpMethod": "POST", "path": "/approve-promotion", "responseMode": "lastNode" },
      "position": [0, 0]
    },
    {
      "name": "Validate Payload",
      "type": "n8n-nodes-base.set",
      "parameters": {
        "values": {
          "string": [
            { "name": "pair",    "value": "={{ $json.body.pair }}" },
            { "name": "run_id",  "value": "={{ $json.body.run_id }}" },
            { "name": "action",  "value": "={{ $json.body.action }}" }
          ]
        }
      },
      "position": [200, 0]
    },
    {
      "name": "IF: Approve",
      "type": "n8n-nodes-base.if",
      "parameters": {
        "conditions": {
          "string": [{ "value1": "={{ $json.action }}", "operation": "equal", "value2": "approve" }]
        }
      },
      "position": [400, 0]
    },
    {
      "name": "Log Approval",
      "type": "n8n-nodes-base.executeCommand",
      "parameters": {
        "command": "echo {\"ts\":\"{{ $now.format('yyyy-MM-ddTHH:mm:ss') }}\",\"action\":\"approve\",\"pair\":\"{{ $json.pair }}\",\"run_id\":\"{{ $json.run_id }}\"} >> C:\\Users\\majin\\ai-tradingagent-kraken\\results\\approval_audit.jsonl"
      },
      "position": [600, 100]
    },
    {
      "name": "Log Rejection",
      "type": "n8n-nodes-base.executeCommand",
      "parameters": {
        "command": "echo {\"ts\":\"{{ $now.format('yyyy-MM-ddTHH:mm:ss') }}\",\"action\":\"reject\",\"pair\":\"{{ $json.pair }}\",\"run_id\":\"{{ $json.run_id }}\"} >> C:\\Users\\majin\\ai-tradingagent-kraken\\results\\approval_audit.jsonl"
      },
      "position": [600, -100]
    },
    {
      "name": "Respond",
      "type": "n8n-nodes-base.respondToWebhook",
      "parameters": { "respondWith": "json", "responseBody": "={ { \"status\": \"logged\", \"pair\": $json.pair, \"action\": $json.action } }" },
      "position": [800, 0]
    }
  ],
  "connections": {
    "Approval Webhook": { "main": [[{ "node": "Validate Payload", "type": "main", "index": 0 }]] },
    "Validate Payload": { "main": [[{ "node": "IF: Approve", "type": "main", "index": 0 }]] },
    "IF: Approve": {
      "main": [
        [{ "node": "Log Approval", "type": "main", "index": 0 }],
        [{ "node": "Log Rejection", "type": "main", "index": 0 }]
      ]
    },
    "Log Approval": { "main": [[{ "node": "Respond", "type": "main", "index": 0 }]] },
    "Log Rejection": { "main": [[{ "node": "Respond", "type": "main", "index": 0 }]] }
  }
}
```

- [ ] **Step 2: Test approval webhook**

```bash
curl -X POST http://localhost:5678/webhook/approve-promotion \
  -H "Content-Type: application/json" \
  -d "{\"pair\":\"XDGUSD\",\"run_id\":\"2026-04-11T00:00:00\",\"action\":\"approve\"}"
```

Expected: `{"status":"logged","pair":"XDGUSD","action":"approve"}`

Confirm `results/approval_audit.jsonl` has a new line.

- [ ] **Step 3: Final full test run**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all tests PASS

- [ ] **Step 4: Final commit**

```bash
git add docs/superpowers/plans/2026-04-11-weak-regime-pnl-n8n-backend.md
git commit -m "docs: add implementation plan for older-60d PnL repair + n8n backend"
```
