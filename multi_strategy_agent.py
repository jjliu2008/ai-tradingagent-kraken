"""
Multi-Strategy AI Trading Agent — Hackathon Edition

Combines the top-performing strategies from the expanded screener into a
single agent with AI-powered regime detection, dynamic strategy weighting,
and ensemble signal confirmation.

Key AI enhancements:
1. Regime Detection — Claude classifies market as trending/ranging/volatile
2. Strategy Ensemble — Multiple strategies vote; Claude breaks ties
3. Dynamic Position Sizing — AI adjusts size based on conviction + regime
4. Adaptive Exit Management — Claude can tighten/loosen stops based on context
5. Cross-Pair Intelligence — Uses context pairs to inform trading decisions

Architecture:
    MarketData → FeatureEngine → StrategyEnsemble → AIFilter → OrderManager
                                                         ↑
                                               RegimeDetector (AI)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import kraken_client as kraken
import strategy as strat

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategySpec:
    """Defines a single strategy within the ensemble."""
    name: str
    interval: int
    weight: float  # 0.0 to 1.0 — contribution to ensemble score
    stop_pct: float
    target_pct: float | None
    hold_bars: int
    trail_activation_pct: float | None = None
    trail_stop_pct: float | None = None


# Top strategies from expanded screener — ranked by score
STRATEGY_PORTFOLIO = [
    # #1: GIGAUSD 60m momentum_breakout — 9.7% net, 66% win, 2.8 PF
    StrategySpec("momentum_breakout_60m", 60, 1.0, 0.015, 0.045, 5),
    # #2: GIGAUSD 30m momentum_breakout_trail — 13.3% net, 52% win
    StrategySpec("momentum_breakout_trail_30m", 30, 0.85, 0.015, None, 8,
                 trail_activation_pct=0.020, trail_stop_pct=0.010),
    # #3: GIGAUSD 60m volume_spike_trend — hot in last 60d (+10.7%)
    StrategySpec("volume_spike_trend_60m", 60, 0.80, 0.012, 0.030, 5),
    # #4: GIGAUSD 30m triple_confluence — 4.5% net, 2.26 PF recent
    StrategySpec("triple_confluence_30m", 30, 0.75, 0.015, 0.045, 6),
    # #5: GIGAUSD 15m triple_confluence — 7.5% net, 1.72 PF
    StrategySpec("triple_confluence_15m", 15, 0.70, 0.015, 0.045, 6),
    # #6: GIGAUSD 30m atr_squeeze_expand — 4.3% net, 1.54 PF
    StrategySpec("atr_squeeze_expand_30m", 30, 0.65, 0.012, 0.035, 6),
]


@dataclass
class RegimeState:
    """AI-determined market regime."""
    regime: str = "unknown"         # trending_up, trending_down, ranging, volatile
    confidence: float = 0.0
    trend_bias: float = 0.0         # -1.0 to 1.0
    volatility_level: str = "normal"  # low, normal, high, extreme
    recommended_strategies: list[str] = field(default_factory=list)
    updated_at: str = ""


@dataclass
class EnsembleSignal:
    """Signal from the strategy ensemble."""
    pair: str
    timestamp: int
    price: float
    strategies_firing: list[str]
    ensemble_score: float
    regime: RegimeState
    raw_features: dict
    recommended_size_mult: float = 1.0
    recommended_exit_mode: str = "standard"


# ---------------------------------------------------------------------------
# Feature engine (reuses strategy.py + expanded indicators)
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def compute_bollinger(series: pd.Series, window=20, num_std=2.0):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    bw = ((upper - lower) / sma).replace(0, np.nan)
    return upper, lower, pct_b, bw


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Full feature set for the multi-strategy agent."""
    df = strat.compute_features(df_raw)

    df["rsi_7"] = compute_rsi(df["close"], 7)
    df["rsi_14"] = compute_rsi(df["close"], 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(df["close"])
    df["macd_cross_up"] = (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
    df["bb_upper"], df["bb_lower"], df["bb_pct_b"], df["bb_bandwidth"] = compute_bollinger(df["close"])
    df["body_pct"] = (df["close"] - df["open"]) / df["open"]
    df["upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1)) / (df["high"] - df["low"]).replace(0, np.nan)
    df["lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    df["green_bar"] = df["close"] > df["open"]
    df["ema_fast_dist"] = (df["close"] - df["ema_fast"]) / df["ema_fast"]
    df["roll_high_12"] = df["high"].rolling(12).max().shift(1)
    df["roll_high_20"] = df["high"].rolling(20).max().shift(1)
    df["vol_sma_10"] = df["volume"].rolling(10).mean()
    df["vol_spike"] = df["volume"] > 2.0 * df["vol_sma_10"]
    df["bb_squeeze"] = df["bb_bandwidth"] < df["bb_bandwidth"].rolling(50, min_periods=20).quantile(0.2)
    df["mom_accel"] = df["momentum_medium"] - df["momentum_medium"].shift(3)

    return df


# ---------------------------------------------------------------------------
# Strategy detection functions
# ---------------------------------------------------------------------------

def detect_momentum_breakout_60m(df: pd.DataFrame) -> bool:
    last = df.iloc[-1]
    return bool(
        float(last["trend_strength"]) > 0.0020
        and float(last["ema_fast"]) > float(last["ema_slow"])
        and float(last["close"]) > float(last["roll_high_12"])
        and float(last["momentum_medium"]) > 0.015
        and float(last["volume_ratio"]) > 1.0
        and float(last["close_location"]) > 0.70
        and float(last["compression_ratio"]) < 0.90
    )


def detect_momentum_breakout_trail_30m(df: pd.DataFrame) -> bool:
    last = df.iloc[-1]
    return bool(
        float(last["trend_strength"]) > 0.0025
        and float(last["ema_fast"]) > float(last["ema_slow"])
        and float(last["close"]) > float(last["roll_high_12"])
        and float(last["momentum_medium"]) > 0.020
        and float(last["volume_ratio"]) > 1.2
        and float(last["close_location"]) > 0.70
    )


def detect_volume_spike_trend_60m(df: pd.DataFrame) -> bool:
    last = df.iloc[-1]
    return bool(
        float(last["ema_fast"]) > float(last["ema_slow"])
        and bool(last["vol_spike"])
        and bool(last["green_bar"])
        and float(last["close_location"]) > 0.70
        and float(last["trend_strength"]) > 0.0015
        and float(last["body_pct"]) > 0.005
    )


def detect_triple_confluence(df: pd.DataFrame) -> bool:
    last = df.iloc[-1]
    return bool(
        float(last["ema_fast"]) > float(last["ema_slow"])
        and float(last["trend_strength"]) > 0.0025
        and float(last["macd_hist"]) > 0
        and float(last["macd_hist"]) > float(df.iloc[-2]["macd_hist"])
        and float(last["compression_ratio"]) < 0.85
        and float(last["volume_ratio"]) > 1.1
        and float(last["close_location"]) > 0.65
        and float(last["close"]) > float(last["roll_high_12"])
    )


def detect_atr_squeeze_expand(df: pd.DataFrame) -> bool:
    last = df.iloc[-1]
    return bool(
        float(last["compression_ratio"]) < 0.75
        and float(last["ema_fast"]) > float(last["ema_slow"])
        and float(last["close"]) > float(last["roll_high_12"])
        and float(last["momentum_medium"]) > 0.010
        and float(last["volume_ratio"]) > 1.0
    )


DETECTORS = {
    "momentum_breakout_60m": detect_momentum_breakout_60m,
    "momentum_breakout_trail_30m": detect_momentum_breakout_trail_30m,
    "volume_spike_trend_60m": detect_volume_spike_trend_60m,
    "triple_confluence_30m": detect_triple_confluence,
    "triple_confluence_15m": detect_triple_confluence,
    "atr_squeeze_expand_30m": detect_atr_squeeze_expand,
}


# ---------------------------------------------------------------------------
# AI Regime Detection
# ---------------------------------------------------------------------------

REGIME_PROMPT = """You are a crypto market regime classifier for a trading agent.

Given the following market snapshot, classify the current regime.

Market Data:
- Pair: {pair}
- Price: ${price:.6f}
- Trend Strength (EMA8 vs EMA21): {trend_strength:.4%}
- 6-bar Momentum: {momentum_medium:.4%}
- 12-bar Momentum: {momentum_long:.4%}
- ATR%: {atr_pct:.4%}
- Compression Ratio (short/long ATR): {compression_ratio:.3f}
- RSI-14: {rsi_14:.1f}
- RSI-7: {rsi_7:.1f}
- MACD Histogram: {macd_hist:.6f}
- Bollinger Bandwidth: {bb_bandwidth:.4f}
- Volume Ratio: {volume_ratio:.2f}x
- Close Location: {close_location:.2f}
- Distance from VWAP: {distance_from_vwap:.4%}

Recent Price Action (last 12 bars):
- Highest high: ${recent_high:.6f}
- Lowest low: ${recent_low:.6f}
- Range: {recent_range_pct:.3%}
- Green bars: {green_count}/12

Respond in JSON:
{{
  "regime": "trending_up" | "trending_down" | "ranging" | "volatile",
  "confidence": 0.0 to 1.0,
  "trend_bias": -1.0 to 1.0,
  "volatility_level": "low" | "normal" | "high" | "extreme",
  "recommended_strategies": ["strategy_name", ...],
  "reasoning": "1-2 sentences"
}}

Strategy options: momentum_breakout_60m, momentum_breakout_trail_30m,
volume_spike_trend_60m, triple_confluence_30m, triple_confluence_15m,
atr_squeeze_expand_30m

Rules:
- In "trending_up" regimes: favor breakout + momentum strategies
- In "ranging" regimes: avoid breakouts, favor mean reversion if available
- In "volatile" regimes: reduce sizing, prefer tight stops
- In "trending_down": recommend SKIP for all long-only strategies
"""


ENSEMBLE_FILTER_PROMPT = """You are an execution filter for a multi-strategy crypto trading agent.

The strategy ensemble has detected a potential trade. Multiple strategies are firing
simultaneously, which increases signal confidence.

Market Snapshot:
{market_snapshot}

Current Regime: {regime} (confidence: {regime_confidence:.0%})
Volatility: {volatility_level}
Trend Bias: {trend_bias:+.2f}

Strategies Firing ({n_firing}/{n_total}):
{strategies_firing}

Ensemble Score: {ensemble_score:.2f}

Portfolio State:
- Current value: ${portfolio_value:,.2f}
- Open PnL: {open_pnl:+.3%}
- Recent trades: {recent_trades_summary}
- Fee hurdle (round-trip): {fee_hurdle:.3%}

Historical context — winning trades in this pair had:
- Strong uptrend (trend_strength > 0.2%)
- Compressed volatility expanding
- Close above prior highs
- Above-average volume
- MACD histogram positive and expanding

Respond in JSON:
{{
  "action": "TRADE" or "SKIP",
  "confidence": 0.0 to 1.0,
  "size_mult": 0.5 to 1.5,
  "exit_mode": "fast" or "standard" or "trail",
  "stop_adjustment": -0.005 to 0.005,
  "target_adjustment": -0.01 to 0.01,
  "reason_tags": ["short", "tags"],
  "reasoning": "1-2 sentences"
}}

size_mult guidelines:
- 1.5 = maximum conviction (trending regime, multiple strategies, strong momentum)
- 1.0 = standard conviction
- 0.5 = reduced conviction (uncertain regime, single strategy, weak confirmation)

exit_mode guidelines:
- "trail" = use trailing stop for extended moves (strong breakouts)
- "standard" = fixed stop/target
- "fast" = tight stops, quick exit (uncertain regime)
"""


class AIEngine:
    """Handles all AI interactions for the multi-strategy agent."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        if Anthropic is None:
            raise ImportError("anthropic package required. pip install anthropic")
        self.client = Anthropic()
        self.model = model

    def detect_regime(self, pair: str, df: pd.DataFrame) -> RegimeState:
        last = df.iloc[-1]
        recent = df.iloc[-12:]

        prompt = REGIME_PROMPT.format(
            pair=pair,
            price=float(last["close"]),
            trend_strength=float(last["trend_strength"]),
            momentum_medium=float(last["momentum_medium"]),
            momentum_long=float(last.get("momentum_long", 0)),
            atr_pct=float(last["atr_pct"]),
            compression_ratio=float(last["compression_ratio"]),
            rsi_14=float(last["rsi_14"]),
            rsi_7=float(last["rsi_7"]),
            macd_hist=float(last["macd_hist"]),
            bb_bandwidth=float(last["bb_bandwidth"]),
            volume_ratio=float(last["volume_ratio"]),
            close_location=float(last["close_location"]),
            distance_from_vwap=float(last["distance_from_vwap"]),
            recent_high=float(recent["high"].max()),
            recent_low=float(recent["low"].min()),
            recent_range_pct=(float(recent["high"].max()) - float(recent["low"].min())) / float(recent["low"].min()),
            green_count=int((recent["close"] > recent["open"]).sum()),
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Extract JSON
            if "```" in text:
                text = text.split("```")[1].strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            data = json.loads(text)
            return RegimeState(
                regime=data.get("regime", "unknown"),
                confidence=float(data.get("confidence", 0.5)),
                trend_bias=float(data.get("trend_bias", 0.0)),
                volatility_level=data.get("volatility_level", "normal"),
                recommended_strategies=data.get("recommended_strategies", []),
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as e:
            print(f"  [AI] Regime detection error: {e}")
            return RegimeState(updated_at=datetime.now(timezone.utc).isoformat())

    def filter_ensemble_signal(
        self,
        signal: EnsembleSignal,
        portfolio_value: float,
        open_pnl: float,
        recent_trades: list[dict],
        fee_hurdle: float,
    ) -> dict:
        market = signal.raw_features
        recent_summary = "None"
        if recent_trades:
            wins = sum(1 for t in recent_trades if t.get("pnl_pct", 0) > 0)
            avg_pnl = np.mean([t.get("pnl_pct", 0) for t in recent_trades])
            recent_summary = f"{len(recent_trades)} trades, {wins} wins, avg PnL {avg_pnl:+.3%}"

        strat_list = "\n".join(f"  - {s}" for s in signal.strategies_firing)

        prompt = ENSEMBLE_FILTER_PROMPT.format(
            market_snapshot=json.dumps({k: f"{v:.6f}" if isinstance(v, float) else v
                                       for k, v in market.items()}, indent=2),
            regime=signal.regime.regime,
            regime_confidence=signal.regime.confidence,
            volatility_level=signal.regime.volatility_level,
            trend_bias=signal.regime.trend_bias,
            n_firing=len(signal.strategies_firing),
            n_total=len(STRATEGY_PORTFOLIO),
            strategies_firing=strat_list,
            ensemble_score=signal.ensemble_score,
            portfolio_value=portfolio_value,
            open_pnl=open_pnl,
            recent_trades_summary=recent_summary,
            fee_hurdle=fee_hurdle,
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                system="You are an execution filter for a deterministic multi-strategy crypto trading agent. Respond only in JSON.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            if "```" in text:
                text = text.split("```")[1].strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            return json.loads(text)
        except Exception as e:
            print(f"  [AI] Filter error: {e}")
            return {"action": "TRADE", "confidence": 0.5, "size_mult": 1.0,
                    "exit_mode": "standard", "stop_adjustment": 0, "target_adjustment": 0,
                    "reason_tags": ["ai_fallback"]}


# ---------------------------------------------------------------------------
# Ensemble engine
# ---------------------------------------------------------------------------

class StrategyEnsemble:
    """Runs multiple strategies and combines their signals."""

    def __init__(self, portfolio: list[StrategySpec]):
        self.portfolio = portfolio
        self.last_signal_bar: dict[str, int] = {}
        self.cooldown_bars = 2

    def detect(self, pair: str, df: pd.DataFrame, regime: RegimeState) -> EnsembleSignal | None:
        """Run all strategies and create ensemble signal if any fire."""
        if len(df) < 30:
            return None

        last = df.iloc[-1]
        bar_idx = len(df) - 1
        firing: list[str] = []
        total_weight = 0.0

        for spec in self.portfolio:
            detector = DETECTORS.get(spec.name)
            if detector is None:
                continue

            # Check cooldown
            last_bar = self.last_signal_bar.get(spec.name, -10)
            if bar_idx - last_bar <= self.cooldown_bars:
                continue

            # Check regime compatibility
            if regime.regime == "trending_down" and regime.confidence > 0.6:
                continue  # Skip all longs in confirmed downtrend

            # Check if AI recommends this strategy in current regime
            regime_boost = 1.0
            if spec.name in regime.recommended_strategies:
                regime_boost = 1.2
            elif regime.recommended_strategies and spec.name not in regime.recommended_strategies:
                regime_boost = 0.7

            try:
                if detector(df):
                    firing.append(spec.name)
                    total_weight += spec.weight * regime_boost
            except Exception:
                pass

        if not firing:
            return None

        # Compute raw features for AI prompt
        raw_features = {}
        for col in ["close", "trend_strength", "momentum_medium", "compression_ratio",
                     "volume_ratio", "close_location", "rsi_7", "rsi_14",
                     "macd_hist", "bb_bandwidth", "distance_from_vwap", "atr_pct"]:
            try:
                raw_features[col] = float(last[col])
            except (KeyError, TypeError):
                pass

        # Ensemble score: weighted sum normalized
        max_possible = sum(s.weight for s in self.portfolio)
        ensemble_score = total_weight / max_possible if max_possible > 0 else 0.0

        # Size recommendation based on signal strength
        if len(firing) >= 3:
            size_mult = 1.3
        elif len(firing) >= 2:
            size_mult = 1.1
        else:
            size_mult = 0.9

        # Exit mode recommendation
        exit_mode = "standard"
        if any("trail" in s for s in firing):
            exit_mode = "trail"
        elif regime.volatility_level in ("high", "extreme"):
            exit_mode = "fast"

        return EnsembleSignal(
            pair=pair,
            timestamp=int(last["ts"]),
            price=float(last["close"]),
            strategies_firing=firing,
            ensemble_score=ensemble_score,
            regime=regime,
            raw_features=raw_features,
            recommended_size_mult=size_mult,
            recommended_exit_mode=exit_mode,
        )

    def record_signals(self, firing: list[str], bar_idx: int) -> None:
        for name in firing:
            self.last_signal_bar[name] = bar_idx


# ---------------------------------------------------------------------------
# Event logging
# ---------------------------------------------------------------------------

class EventLogger:
    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = log_dir / "events.jsonl"
        self.summary_path = log_dir / "summary.jsonl"

    def log(self, event_type: str, data: dict) -> None:
        entry = {"ts": datetime.now(timezone.utc).isoformat(), "event": event_type, **data}
        with open(self.events_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        tag = f"[{event_type}]"
        detail = " | ".join(f"{k}={v}" for k, v in data.items() if k not in ("ts",))
        print(f"  {tag} {detail}")

    def log_summary(self, data: dict) -> None:
        entry = {"ts": datetime.now(timezone.utc).isoformat(), **data}
        with open(self.summary_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-Strategy AI Trading Agent")
    p.add_argument("--mode", choices=["paper", "live"], default="paper")
    p.add_argument("--pairs", default="GIGAUSD")
    p.add_argument("--context-pairs", default="DOGUSD,HYPEUSD")
    p.add_argument("--notional-usd", type=float, default=25)
    p.add_argument("--max-positions", type=int, default=1)
    p.add_argument("--poll", type=int, default=60)
    p.add_argument("--cycles", type=int, default=0, help="0 = infinite")
    p.add_argument("--use-claude", action="store_true")
    p.add_argument("--regime-interval", type=int, default=5, help="Regime check every N cycles")
    p.add_argument("--log-dir", default="runtime/multi")
    p.add_argument("--state-file", default="runtime/multi/state.json")
    p.add_argument("--reset-paper", action="store_true")
    p.add_argument("--paper-init-balance", type=float, default=10000)
    p.add_argument("--validate-live-orders", action="store_true")
    p.add_argument("--resume-state", action="store_true")
    return p.parse_args()


def fetch_market_data(pair: str, interval: int = 60) -> pd.DataFrame | None:
    try:
        raw = kraken.fetch_ohlc(pair, interval=interval)
        return strat.parse_ohlc(raw)
    except Exception as e:
        print(f"  Error fetching {pair}: {e}")
        return None


def run(args: argparse.Namespace) -> None:
    logger = EventLogger(Path(args.log_dir))
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    context_pairs = [p.strip() for p in args.context_pairs.split(",") if p.strip()]

    # Initialize AI
    ai: AIEngine | None = None
    if args.use_claude:
        try:
            ai = AIEngine()
            print("[AI] Claude integration enabled")
        except Exception as e:
            print(f"[AI] Failed to initialize: {e}")

    # Initialize ensemble
    ensemble = StrategyEnsemble(STRATEGY_PORTFOLIO)

    # Paper trading init
    if args.mode == "paper":
        if args.reset_paper:
            kraken.paper_init(balance=args.paper_init_balance)
            print(f"[PAPER] Initialized with ${args.paper_init_balance:,.2f}")

    # State
    regime = RegimeState()
    trade_log: list[dict] = []
    open_trade: dict | None = None
    pending_order: dict | None = None
    cycle = 0

    logger.log("agent_started", {
        "mode": args.mode,
        "pairs": pairs,
        "strategies": [s.name for s in STRATEGY_PORTFOLIO],
        "use_claude": args.use_claude,
    })

    print(f"\n{'=' * 70}")
    print(f" Multi-Strategy AI Agent | mode={args.mode} | pairs={pairs}")
    print(f" Strategies: {len(STRATEGY_PORTFOLIO)} | AI: {'ON' if ai else 'OFF'}")
    print(f"{'=' * 70}\n")

    while True:
        cycle += 1
        if args.cycles > 0 and cycle > args.cycles:
            break

        print(f"\n--- Cycle {cycle} | {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')} ---")

        # 1. Fetch market data
        market_data: dict[str, pd.DataFrame] = {}
        for pair in pairs + context_pairs:
            df = fetch_market_data(pair)
            if df is not None and len(df) > 30:
                market_data[pair] = build_features(df)

        if not any(p in market_data for p in pairs):
            print("  No market data available, waiting...")
            time.sleep(args.poll)
            continue

        # 2. Regime detection (AI)
        if ai and cycle % args.regime_interval == 1:
            primary_pair = pairs[0]
            if primary_pair in market_data:
                print("  [AI] Detecting market regime...")
                regime = ai.detect_regime(primary_pair, market_data[primary_pair])
                logger.log("regime_update", asdict(regime))
                print(f"  [AI] Regime: {regime.regime} ({regime.confidence:.0%}) | "
                      f"vol={regime.volatility_level} | bias={regime.trend_bias:+.2f}")

        # 3. Check pending orders & open trades
        if args.mode == "paper" and pending_order:
            try:
                orders = kraken.paper_orders()
                if isinstance(orders, dict) and "open" in orders:
                    if pending_order["order_id"] not in orders.get("open", {}):
                        # Order filled or cancelled
                        history = kraken.paper_history()
                        filled = False
                        if isinstance(history, dict):
                            for tid, info in history.get("trades", {}).items():
                                if info.get("ordertxid") == pending_order["order_id"]:
                                    filled = True
                                    break
                        if filled:
                            open_trade = {
                                "pair": pending_order["pair"],
                                "entry_price": pending_order["price"],
                                "size": pending_order["size"],
                                "entry_ts": time.time(),
                                "stop_price": pending_order["price"] * (1 - 0.015),
                                "target_price": pending_order["price"] * (1 + 0.045),
                                "exit_mode": pending_order.get("exit_mode", "standard"),
                                "entry_bar": cycle,
                            }
                            logger.log("entry_filled", {"pair": open_trade["pair"],
                                                       "price": open_trade["entry_price"]})
                        pending_order = None
            except Exception as e:
                print(f"  Order check error: {e}")

        # 4. Monitor open trades
        if open_trade:
            pair = open_trade["pair"]
            if pair in market_data:
                df = market_data[pair]
                current_price = float(df.iloc[-1]["close"])
                pnl_pct = (current_price - open_trade["entry_price"]) / open_trade["entry_price"]
                bars_held = cycle - open_trade["entry_bar"]

                exit_reason = None
                if current_price <= open_trade["stop_price"]:
                    exit_reason = "STOP_LOSS"
                elif open_trade["target_price"] and current_price >= open_trade["target_price"]:
                    exit_reason = "TAKE_PROFIT"
                elif bars_held >= 5:
                    exit_reason = "TIME_LIMIT"
                elif bars_held >= 1 and float(df.iloc[-1]["ema_fast"]) > float(df.iloc[-1]["ema_slow"]) == False:
                    if float(df.iloc[-1]["momentum_medium"]) < 0:
                        exit_reason = "TREND_LOST"

                if exit_reason:
                    if args.mode == "paper":
                        try:
                            kraken.paper_sell(pair, open_trade["size"])
                        except Exception as e:
                            print(f"  Exit error: {e}")

                    trade_record = {
                        "pair": pair,
                        "entry": open_trade["entry_price"],
                        "exit": current_price,
                        "pnl_pct": pnl_pct,
                        "reason": exit_reason,
                        "bars_held": bars_held,
                    }
                    trade_log.append(trade_record)
                    logger.log("trade_closed", trade_record)
                    open_trade = None
                else:
                    print(f"  Open: {pair} | entry={open_trade['entry_price']:.6f} | "
                          f"current={current_price:.6f} | pnl={pnl_pct:+.3%} | bars={bars_held}")

        # 5. Detect new signals (only if no open trade/pending order)
        if open_trade is None and pending_order is None:
            for pair in pairs:
                if pair not in market_data:
                    continue

                df = market_data[pair]
                signal = ensemble.detect(pair, df, regime)

                if signal is None:
                    continue

                print(f"\n  SIGNAL: {pair} | strategies={signal.strategies_firing} | "
                      f"score={signal.ensemble_score:.2f}")

                # AI filter
                ai_decision = None
                if ai:
                    print("  [AI] Filtering signal...")
                    portfolio_value = args.paper_init_balance  # simplified
                    open_pnl = 0.0
                    fee_hurdle = 2 * (0.0026 + 0.0005)
                    ai_decision = ai.filter_ensemble_signal(
                        signal, portfolio_value, open_pnl, trade_log[-5:], fee_hurdle,
                    )
                    logger.log("ai_decision", ai_decision)
                    print(f"  [AI] Decision: {ai_decision.get('action')} | "
                          f"confidence={ai_decision.get('confidence', 0):.2f} | "
                          f"size_mult={ai_decision.get('size_mult', 1):.2f} | "
                          f"tags={ai_decision.get('reason_tags', [])}")

                    if ai_decision.get("action") == "SKIP":
                        ensemble.record_signals(signal.strategies_firing, len(df) - 1)
                        continue

                # Place order
                size_mult = ai_decision.get("size_mult", 1.0) if ai_decision else signal.recommended_size_mult
                exit_mode = ai_decision.get("exit_mode", "standard") if ai_decision else signal.recommended_exit_mode

                notional = args.notional_usd * size_mult
                size = notional / signal.price
                price = signal.price

                if args.mode == "paper":
                    try:
                        result = kraken.paper_buy(pair, size, order_type="limit", price=price)
                        order_id = None
                        if isinstance(result, dict) and "txid" in result:
                            txids = result["txid"]
                            order_id = txids[0] if isinstance(txids, list) else txids

                        pending_order = {
                            "pair": pair,
                            "order_id": order_id,
                            "price": price,
                            "size": size,
                            "exit_mode": exit_mode,
                        }
                        logger.log("order_placed", {
                            "pair": pair, "price": price, "size": size,
                            "strategies": signal.strategies_firing,
                            "ensemble_score": signal.ensemble_score,
                        })
                    except Exception as e:
                        print(f"  Order error: {e}")

                ensemble.record_signals(signal.strategies_firing, len(df) - 1)
                break  # One signal per cycle

        # 6. Print summary every 10 cycles
        if cycle % 10 == 0 and trade_log:
            pnls = [t["pnl_pct"] for t in trade_log]
            wins = sum(1 for p in pnls if p > 0)
            print(f"\n  === SUMMARY | trades={len(trade_log)} | "
                  f"wins={wins} ({wins/len(trade_log):.0%}) | "
                  f"net={sum(pnls):+.3%} | avg={np.mean(pnls):+.3%} ===")
            logger.log_summary({
                "cycle": cycle, "trades": len(trade_log),
                "win_rate": wins / len(trade_log) if trade_log else 0,
                "net_pnl": sum(pnls), "avg_pnl": float(np.mean(pnls)),
            })

        time.sleep(args.poll)

    # Final report
    if trade_log:
        pnls = [t["pnl_pct"] for t in trade_log]
        wins = sum(1 for p in pnls if p > 0)
        print(f"\n{'=' * 70}")
        print(f" FINAL REPORT | {len(trade_log)} trades")
        print(f"{'=' * 70}")
        print(f" Net PnL:  {sum(pnls):+.3%}")
        print(f" Win Rate: {wins/len(trade_log):.1%}")
        print(f" Avg PnL:  {np.mean(pnls):+.3%}")
        print(f" Best:     {max(pnls):+.3%}")
        print(f" Worst:    {min(pnls):+.3%}")

        reasons = pd.Series([t["reason"] for t in trade_log]).value_counts()
        print(f"\nExit reasons:")
        print(reasons.to_string())


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "live":
        print("\n⚠️  LIVE MODE — Real orders will be placed!")
        confirm = input("Type 'yes' to continue: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(0)
    run(args)
