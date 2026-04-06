"""
Universe Scanner Agent
======================
Watches 15 Kraken pairs simultaneously. Max 3 concurrent positions.
Plots continuous mark-to-market P&L every bar — produces a smooth curve
like a professional trading system, not sparse once-a-week spikes.

How the smooth curve works:
  - Open positions are marked to current market price every poll cycle
  - Realized + unrealized PnL is logged every cycle
  - With 3 positions always working, something is always moving

Usage:
    python universe_scanner_agent.py --mode paper --reset-paper
    python universe_scanner_agent.py --mode paper --poll 60 --max-positions 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import kraken_client as kraken
import strategy as strat

try:
    from anthropic import Anthropic
    _HAS_CLAUDE = True
except ImportError:
    _HAS_CLAUDE = False

load_dotenv()

# ---------------------------------------------------------------------------
# Universe — pairs to watch
# ---------------------------------------------------------------------------

UNIVERSE = [
    # ── Tier A: backtest-validated long-term (120d cached data)
    # GIGAUSD: +20.7% net / 120d | 69% win | 5.07 PF | Sharpe 23
    # ZECUSD:  +4.8%  net / 60d  | 50% win | 3.76 PF
    "GIGAUSD",
    "ZECUSD",
    # ── Tier A: live-scanner confirmed (332-pair scan, 30d live data)
    # REDUSD:  3+ trades | 60%+ win | PF≥2  (scanner Tier A)
    # SUIUSD:  3 trades  | 67% win  | PF=2.12 | Sharpe=15
    "REDUSD",
    "SUIUSD",
    # ── Tier B: 2 trades, 100% win rate, ranked by net P&L
    # KERNELUSD: +$16.78 | CHZUSD: +$15.54 | WIFUSD: +$9.30
    # ZAMAUSD: +$4.73   | CCUSD:  +$4.83   | HYPEUSD: +$2.57
    "KERNELUSD",
    "CHZUSD",
    "WIFUSD",
    "ZAMAUSD",
    "CCUSD",
    "HYPEUSD",
]

# Per-pair minimum vote threshold
# Tier A pairs (more historical evidence) stay at 2/5
# Tier B pairs (2-trade sample) also 2/5 — consensus filter is critical
PAIR_MIN_VOTES: dict[str, int] = {
    "GIGAUSD":   2,
    "ZECUSD":    2,
    "REDUSD":    2,
    "SUIUSD":    2,
    "KERNELUSD": 2,
    "CHZUSD":    2,
    "WIFUSD":    2,
    "ZAMAUSD":   2,
    "CCUSD":     2,
    "HYPEUSD":   2,
}
DEFAULT_MIN_VOTES = 2   # fallback for any pair not in PAIR_MIN_VOTES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_POSITIONS    = 5        # concurrent open positions (more pairs = more concurrency)
BASE_NOTIONAL    = 100.0    # USD per position (reduced from 150 to manage risk across larger universe)
STOP_PCT         = 0.015    # 1.5% stop loss
TARGET_PCT       = 0.040    # 4.0% take profit
MAX_HOLD_BARS    = 10       # exit after N bars if no TP/SL
TREND_EXIT_BARS  = 3        # start checking trend exit after this many bars
COOLDOWN_BARS    = 4        # bars to wait before re-entering same pair
COMMISSION       = 0.0026
SLIPPAGE         = 0.0005
ROUND_TRIP_COST  = (COMMISSION + SLIPPAGE) * 2

MIN_SIGNAL_VOTES = 2   # global default (overridden per-pair by PAIR_MIN_VOTES)
MIN_SIGNAL_SCORE = 2   # alias used by argparse default

# ---------------------------------------------------------------------------
# Features — full set used by all 4 proven signals
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features needed by the 4 proven consensus signals."""
    df = df.copy()
    # EMAs
    df["ema_fast"]  = df["close"].ewm(12, adjust=False).mean()
    df["ema_slow"]  = df["close"].ewm(26, adjust=False).mean()
    df["ema55"]     = df["close"].ewm(55, adjust=False).mean()
    # Volume
    df["vol_ma20"]  = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["vol_ma20"].replace(0, np.nan)
    # ATR / compression
    df["atr"]       = (df["high"] - df["low"]).rolling(14).mean()
    df["atr_pct"]   = df["atr"] / df["close"]
    df["compression_ratio"] = (
        df["atr_pct"] / df["atr_pct"].rolling(24).mean().replace(0, np.nan)
    )
    # Rolling high
    df["roll_high_12"] = df["high"].rolling(12).max().shift(1)
    df["roll_high_20"] = df["high"].rolling(20).max().shift(1)
    # Close location (where did we close in the bar's range)
    bar_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["close_location"] = (df["close"] - df["low"]) / bar_range
    # Trend strength (% above slow EMA)
    df["trend_strength"] = (df["close"] - df["ema_slow"]) / df["ema_slow"]
    # Momentum
    df["momentum_medium"] = df["close"].pct_change(8)
    # MACD
    df["macd"]      = df["ema_fast"] - df["ema_slow"]
    df["macd_sig"]  = df["macd"].ewm(9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]
    # Bollinger Bands
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_upper"]  = sma20 + 2 * std20
    df["bb_lower"]  = sma20 - 2 * std20
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / sma20
    df["bb_squeeze"]= df["bb_width"] < df["bb_width"].rolling(50, min_periods=20).quantile(0.20)
    # VWAP
    vwap_pv  = (df["close"] * df["volume"]).rolling(48).sum()
    vwap_vol = df["volume"].rolling(48).sum()
    df["vwap"]          = vwap_pv / vwap_vol.replace(0, np.nan)
    df["distance_from_vwap"] = (df["close"] - df["vwap"]) / df["vwap"].replace(0, np.nan)
    return df


# ---------------------------------------------------------------------------
# The 4 proven signals — same logic as consensus_agent.py
# (validated: GIGAUSD +16% net / 120d, 56% win, 3.42 PF)
# ---------------------------------------------------------------------------

def _sig_macd_accel(df: pd.DataFrame) -> bool:
    """MACD histogram accelerating up + price above 12-bar high."""
    if len(df) < 30: return False
    last = df.iloc[-1]; prev = df.iloc[-2]; prev2 = df.iloc[-3]
    return bool(
        float(last["ema_fast"]) > float(last["ema_slow"])
        and float(last["macd_hist"]) > 0
        and float(last["macd_hist"]) > float(prev["macd_hist"])
        and float(prev["macd_hist"]) > float(prev2["macd_hist"])
        and float(last["close"]) > float(last["roll_high_12"])
        and 1.0 <= float(last["volume_ratio"]) <= 4.0
        and float(last["close_location"]) > 0.68
    )


def _sig_bb_squeeze(df: pd.DataFrame) -> bool:
    """Bollinger Band squeeze breakout."""
    if len(df) < 30: return False
    last = df.iloc[-1]
    return bool(
        bool(last["bb_squeeze"])
        and float(last["ema_fast"]) > float(last["ema_slow"])
        and float(last["momentum_medium"]) > 0.012
        and float(last["close"]) > float(last["roll_high_12"])
        and 1.0 <= float(last["volume_ratio"]) <= 4.0
        and float(last["close_location"]) > 0.68
    )


def _sig_atr_compress(df: pd.DataFrame) -> bool:
    """ATR compression: volatility tightest in 24 bars, now breaking out."""
    if len(df) < 30: return False
    last = df.iloc[-1]
    return bool(
        float(last["compression_ratio"]) < 0.75
        and float(last["ema_fast"]) > float(last["ema_slow"])
        and float(last["close"]) > float(last["roll_high_12"])
        and float(last["momentum_medium"]) > 0.010
        and float(last["volume_ratio"]) > 1.0
    )


def _sig_sweet_spot(df: pd.DataFrame) -> bool:
    """Sweet-spot breakout: moderate trend, compression, volume."""
    if len(df) < 30: return False
    last = df.iloc[-1]
    ts = float(last["trend_strength"])
    return bool(
        0.0020 <= ts <= 0.0200
        and float(last["ema_fast"]) > float(last["ema_slow"])
        and float(last["compression_ratio"]) < 0.85
        and float(last["close"]) > float(last["roll_high_12"])
        and 1.0 <= float(last["volume_ratio"]) <= 4.0
        and float(last["close_location"]) > 0.70
        and float(last["momentum_medium"]) > 0.012
    )


def _sig_oversold_bounce(df: pd.DataFrame) -> bool:
    """Mean-reversion bounce: prior weakness + reversal bar + volume surge.
    Fires in downtrend conditions where breakout signals cannot.
    """
    if len(df) < 30: return False
    last = df.iloc[-1]
    # Prior 4 bars were all weak (negative momentum)
    prior_mom = [float(df.iloc[-i]["momentum_medium"]) for i in range(2, 6)]
    prior_weak = sum(1 for m in prior_mom if m < -0.003) >= 3
    # Current bar is recovering: MACD hist turning up
    macd_turning = float(last["macd_hist"]) > float(df.iloc[-2]["macd_hist"])
    # Volume surge — capitulation/reversal signature
    vol_surge = float(last["volume_ratio"]) >= 1.4
    # Close in upper 50% of bar (buyers winning)
    closing_up = float(last["close_location"]) > 0.50
    # Not too far below VWAP (avoid falling knives)
    vwap_dist = float(last["distance_from_vwap"])
    near_vwap = -0.06 <= vwap_dist <= 0.005
    return bool(prior_weak and macd_turning and vol_surge and closing_up and near_vwap)


_SIGNALS = [_sig_macd_accel, _sig_bb_squeeze, _sig_atr_compress, _sig_sweet_spot, _sig_oversold_bounce]
_SIG_NAMES = ["macd_accel", "bb_squeeze", "atr_compress", "sweet_spot", "oversold_bounce"]


def score_signal(df: pd.DataFrame) -> tuple[int, list[str]]:
    """
    Run all 4 proven consensus signals. Returns (vote_count, firing_names).
    Need MIN_SIGNAL_VOTES (2+) to enter — same threshold as consensus_agent.
    """
    if len(df) < 55:
        return 0, []
    firing = []
    for fn, name in zip(_SIGNALS, _SIG_NAMES):
        try:
            if fn(df):
                firing.append(name)
        except Exception:
            pass
    return len(firing), firing


# ---------------------------------------------------------------------------
# Open position tracking
# ---------------------------------------------------------------------------

@dataclass
class OpenPosition:
    pair:         str
    entry_price:  float
    size:         float           # units of base asset
    notional_usd: float
    entry_bar:    int
    entry_ts:     float
    stop_price:   float
    target_price: float
    signal_score: int
    conditions:   list[str]
    current_price: float = 0.0
    bars_held:    int = 0

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.current_price <= 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

    @property
    def unrealized_pnl_usd(self) -> float:
        return self.unrealized_pnl_pct * self.notional_usd

    def update_price(self, price: float) -> None:
        self.current_price = price
        self.bars_held += 1

    def should_exit(self) -> Optional[str]:
        if self.current_price <= 0:
            return None
        if self.current_price <= self.stop_price:
            return "STOP_LOSS"
        if self.current_price >= self.target_price:
            return "TAKE_PROFIT"
        if self.bars_held >= MAX_HOLD_BARS:
            return "TIME_LIMIT"
        return None

    def should_trend_exit(self, df: pd.DataFrame) -> bool:
        if self.bars_held < TREND_EXIT_BARS or len(df) < 3:
            return False
        last = df.iloc[-1]
        try:
            ema12 = float(last["ema12"]); ema26 = float(last["ema26"])
            return ema12 < ema26
        except Exception:
            return False


# ---------------------------------------------------------------------------
# PnL curve — records every bar for the smooth chart
# ---------------------------------------------------------------------------

@dataclass
class PnLCurve:
    log_path: Path
    realized_usd:   float = 0.0
    n_wins:         int   = 0
    n_losses:       int   = 0
    trade_log:      list  = field(default_factory=list)

    def record(self, open_positions: dict[str, OpenPosition]) -> dict:
        unrealized_usd = sum(p.unrealized_pnl_usd for p in open_positions.values())
        total_usd      = self.realized_usd + unrealized_usd
        now_ts = datetime.now(timezone.utc)
        point = {
            "ts":             now_ts.timestamp(),
            "ts_iso":         now_ts.isoformat(),
            "realized_usd":   round(self.realized_usd, 4),
            "unrealized_usd": round(unrealized_usd, 4),
            "total_pnl_usd":  round(total_usd, 4),
            # legacy key kept for compatibility
            "total_usd":      round(total_usd, 4),
            "n_open":         len(open_positions),
            "open_positions": [
                {
                    "pair":               p.pair,
                    "entry_price":        p.entry_price,
                    "current_price":      p.current_price,
                    "unrealized_pnl_pct": round(p.unrealized_pnl_pct, 6),
                    "unrealized_pnl_usd": round(p.unrealized_pnl_usd, 4),
                    "bars_held":          p.bars_held,
                    "conditions":         p.conditions,
                }
                for p in open_positions.values()
            ],
            "n_trades":       len(self.trade_log),
            "n_wins":         self.n_wins,
            "n_losses":       self.n_losses,
            "recent_trades":  self.trade_log[-20:],
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(point) + "\n")
        return point

    def close_trade(self, pos: OpenPosition, exit_price: float, reason: str) -> float:
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price - ROUND_TRIP_COST
        pnl_usd = pnl_pct * pos.notional_usd
        self.realized_usd += pnl_usd
        record = {
            "pair":        pos.pair,
            "entry":       pos.entry_price,
            "exit":        exit_price,
            "pnl_pct":     round(pnl_pct, 6),
            "pnl_usd":     round(pnl_usd, 4),
            "reason":      reason,
            "bars_held":   pos.bars_held,
            "score":       pos.signal_score,
            "conditions":  pos.conditions,
            "entry_ts":    pos.entry_ts,
            "exit_ts":     time.time(),
        }
        self.trade_log.append(record)
        if pnl_usd > 0:
            self.n_wins += 1
        else:
            self.n_losses += 1
        return pnl_usd


# ---------------------------------------------------------------------------
# AI filter (optional) — quick single-call filter, not full debate
# ---------------------------------------------------------------------------

AI_FILTER_PROMPT = """\
You are a crypto trading signal filter. A momentum signal just fired.

Pair: {pair}
Conditions met ({score}/5): {conditions}
Recent price data (last 5 bars):
{price_summary}

Is this a high-quality entry? Respond in JSON only:
{{"trade": true or false, "reason": "one short phrase"}}

Rules:
- true if trend is clear, breakout is real, volume confirms
- false if overextended, weak volume, or chasing a spike
"""


class AIFilter:
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        if not _HAS_CLAUDE:
            raise ImportError("pip install anthropic")
        self.client = Anthropic()
        self.model  = model
        self.calls  = 0

    def approve(self, pair: str, score: int, conditions: list[str], df: pd.DataFrame) -> bool:
        try:
            last5 = df.tail(5)[["close", "volume", "atr_pct"]].round(6).to_string(index=False)
            resp  = self.client.messages.create(
                model=self.model, max_tokens=80,
                messages=[{"role": "user", "content": AI_FILTER_PROMPT.format(
                    pair=pair, score=score, conditions=", ".join(conditions),
                    price_summary=last5,
                )}],
            )
            text = resp.content[0].text.strip()
            if "```" in text:
                text = text.split("```")[1].lstrip("json").strip()
            result = json.loads(text)
            self.calls += 1
            return bool(result.get("trade", True))
        except Exception:
            return True   # fail open — let the trade through if AI unavailable


# ---------------------------------------------------------------------------
# Fetching helpers
# ---------------------------------------------------------------------------

def fetch_ohlc_df(pair: str, interval: int = 60) -> Optional[pd.DataFrame]:
    """Fetch OHLC, return computed-feature DataFrame or None.
    Drops the last (incomplete/still-forming) candle — Kraken always appends
    the current open bar which has near-zero volume and unreliable close_location.
    """
    try:
        raw = kraken.fetch_ohlc(pair, interval=interval)
        df  = strat.parse_ohlc(raw)
        df  = df.iloc[:-1]   # drop incomplete current candle
        if len(df) < 60:
            return None
        return compute_features(df)
    except Exception:
        return None


def current_price(pair: str) -> Optional[float]:
    try:
        raw = kraken.fetch_ticker(pair)
        if isinstance(raw, dict):
            data = raw.get(pair) or next(iter(raw.values()), {})
            for key in ("c", "a", "b"):
                v = data.get(key)
                if isinstance(v, list) and v:
                    return float(v[0])
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Universe Scanner — 15-pair concurrent trading")
    p.add_argument("--mode",        choices=["paper", "live"], default="paper")
    p.add_argument("--universe",    default=",".join(UNIVERSE))
    p.add_argument("--max-pos",     type=int,   default=MAX_POSITIONS)
    p.add_argument("--notional",    type=float, default=BASE_NOTIONAL)
    p.add_argument("--poll",        type=int,   default=60,   help="seconds between cycles")
    p.add_argument("--interval",    type=int,   default=15,   help="OHLC interval in minutes")
    p.add_argument("--min-score",   type=int,   default=MIN_SIGNAL_SCORE)
    p.add_argument("--cycles",      type=int,   default=0,    help="0 = run forever")
    p.add_argument("--use-claude",  action="store_true")
    p.add_argument("--ai-model",    default="claude-haiku-4-5-20251001")
    p.add_argument("--reset-paper", action="store_true")
    p.add_argument("--paper-balance", type=float, default=10000.0)
    p.add_argument("--log-dir",     default="runtime/universe")
    return p.parse_args()


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def run(args: argparse.Namespace) -> None:
    log_dir   = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    pnl_curve = PnLCurve(log_path=log_dir / "pnl_curve.jsonl")
    event_log = log_dir / "events.jsonl"

    def emit(event: str, **kw):
        record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **kw}
        with event_log.open("a") as f:
            f.write(json.dumps(record) + "\n")

    pairs = [p.strip() for p in args.universe.split(",") if p.strip()]

    # AI filter (optional)
    ai_filter: Optional[AIFilter] = None
    if args.use_claude and _HAS_CLAUDE:
        try:
            ai_filter = AIFilter(model=args.ai_model)
            log(f"AI filter enabled ({args.ai_model})")
        except Exception as e:
            log(f"AI filter unavailable: {e}")

    if args.mode == "paper" and args.reset_paper:
        kraken.paper_init(balance=args.paper_balance)
        log(f"Paper account reset — balance ${args.paper_balance:,.0f}")

    # State
    open_positions: dict[str, OpenPosition] = {}   # pair → position
    cooldowns:      dict[str, int]           = {}   # pair → cycle cooldown expires
    cycle = 0

    print(f"\n{'═'*65}")
    print(f"  UNIVERSE SCANNER | {args.mode.upper()} | {len(pairs)} pairs | max {args.max_pos} positions")
    print(f"  Notional: ${args.notional:.0f}/trade | Stop: {STOP_PCT:.1%} | Target: {TARGET_PCT:.1%}")
    print(f"  AI filter: {'ON' if ai_filter else 'OFF'} | Min score: {args.min_score}/5")
    print(f"  P&L curve → {pnl_curve.log_path}")
    print(f"{'═'*65}\n")

    emit("agent_started", pairs=pairs, max_pos=args.max_pos, notional=args.notional)

    while True:
        cycle += 1
        if args.cycles > 0 and cycle > args.cycles:
            break

        ts_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
        log(f"── Cycle {cycle} | {ts_str} UTC | open={len(open_positions)}/{args.max_pos} ──")

        # ── 1. Update open positions with fresh prices ──────────────────────
        to_close: list[tuple[str, float, str]] = []

        for pair, pos in list(open_positions.items()):
            price = current_price(pair)
            if price is None:
                continue
            pos.update_price(price)
            reason = pos.should_exit()

            # Also check trend exit using fresh OHLC
            if reason is None and pos.bars_held >= TREND_EXIT_BARS:
                df_fresh = fetch_ohlc_df(pair, interval=args.interval)
                if df_fresh is not None and pos.should_trend_exit(df_fresh):
                    reason = "TREND_LOST"

            if reason:
                to_close.append((pair, price, reason))
            else:
                upnl = pos.unrealized_pnl_pct
                log(f"  ↔ {pair:<12} entry={pos.entry_price:.5g} now={price:.5g} "
                    f"upnl={upnl:+.2%} bars={pos.bars_held}")

        # ── 2. Close positions that hit exit conditions ──────────────────────
        for pair, price, reason in to_close:
            pos = open_positions.pop(pair)
            if args.mode == "paper":
                try:
                    kraken.paper_sell(pair, pos.size)
                except Exception as e:
                    log(f"  ! Exit order failed {pair}: {e}")

            pnl_usd = pnl_curve.close_trade(pos, price, reason)
            emoji   = "✅" if pnl_usd > 0 else "❌"
            log(f"  {emoji} CLOSED {pair:<12} {reason:<12} "
                f"pnl={pos.unrealized_pnl_pct:+.2%} (${pnl_usd:+.2f}) "
                f"bars={pos.bars_held}")
            cooldowns[pair] = cycle + COOLDOWN_BARS
            emit("trade_closed", pair=pair, reason=reason,
                 pnl_pct=pos.unrealized_pnl_pct, pnl_usd=pnl_usd,
                 bars=pos.bars_held, score=pos.signal_score)

        # ── 3. Scan for new entries if slots available ──────────────────────
        if len(open_positions) < args.max_pos:
            candidates: list[tuple[str, int, list[str], pd.DataFrame]] = []

            for pair in pairs:
                if pair in open_positions:
                    continue
                if cooldowns.get(pair, 0) >= cycle:
                    continue

                df = fetch_ohlc_df(pair, interval=args.interval)
                if df is None:
                    continue

                score, conditions = score_signal(df)
                # Use per-pair threshold if defined, else global --min-score
                pair_threshold = PAIR_MIN_VOTES.get(pair, args.min_score)
                log(f"  📊 {pair:<12} score={score}/{pair_threshold} signals=[{', '.join(conditions) if conditions else 'none'}]")
                if score >= pair_threshold:
                    candidates.append((pair, score, conditions, df))

            # Sort by score descending — take best signals first
            candidates.sort(key=lambda x: x[1], reverse=True)

            for pair, score, conditions, df in candidates:
                if len(open_positions) >= args.max_pos:
                    break

                price = float(df["close"].iloc[-1])

                # AI filter (optional)
                if ai_filter is not None:
                    approved = ai_filter.approve(pair, score, conditions, df)
                    if not approved:
                        log(f"  ✗ AI SKIP {pair} score={score}")
                        emit("ai_skip", pair=pair, score=score, conditions=conditions)
                        continue

                # Size
                size = args.notional / price

                # Place order
                order_id = None
                if args.mode == "paper":
                    try:
                        result   = kraken.paper_buy(pair, size)
                        txids    = result.get("txid", []) if isinstance(result, dict) else []
                        order_id = txids[0] if txids else "paper"
                    except Exception as e:
                        log(f"  ! Entry order failed {pair}: {e}")
                        continue

                pos = OpenPosition(
                    pair         = pair,
                    entry_price  = price,
                    size         = size,
                    notional_usd = args.notional,
                    entry_bar    = cycle,
                    entry_ts     = time.time(),
                    stop_price   = price * (1 - STOP_PCT),
                    target_price = price * (1 + TARGET_PCT),
                    signal_score = score,
                    conditions   = conditions,
                    current_price= price,
                )
                open_positions[pair] = pos

                thresh = PAIR_MIN_VOTES.get(pair, args.min_score)
                log(f"  🟢 ENTER  {pair:<12} score={score}/{thresh}+ "
                    f"conds={conditions} "
                    f"price={price:.5g} stop={pos.stop_price:.5g}")
                emit("trade_opened", pair=pair, score=score, conditions=conditions,
                     price=price, notional=args.notional, order_id=order_id)

        # ── 4. Record PnL curve point (mark-to-market) ──────────────────────
        point = pnl_curve.record(open_positions)
        total_usd    = point["total_usd"]
        realized_usd = point["realized_usd"]
        unreal_usd   = point["unrealized_usd"]
        n_trades     = point["n_trades"]

        log(f"  📈 PnL: realized=${realized_usd:+.2f}  "
            f"unrealized=${unreal_usd:+.2f}  "
            f"TOTAL=${total_usd:+.2f}  "
            f"trades={n_trades} ({pnl_curve.n_wins}W/{pnl_curve.n_losses}L)")

        # ── 5. Periodic summary every 20 cycles ────────────────────────────
        if cycle % 20 == 0 and pnl_curve.trade_log:
            pnls = [t["pnl_usd"] for t in pnl_curve.trade_log]
            wins = [p for p in pnls if p > 0]
            log(f"\n  ━━━ SUMMARY ━━━")
            log(f"  Trades: {len(pnls)} | Wins: {len(wins)} ({len(wins)/len(pnls):.0%})")
            log(f"  Realized P&L: ${sum(pnls):+.2f}")
            log(f"  Avg per trade: ${np.mean(pnls):+.2f}")
            pairs_traded = list({t["pair"] for t in pnl_curve.trade_log})
            log(f"  Pairs traded:  {pairs_traded}")
            log("")

        time.sleep(args.poll)

    # ── Final report ────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  FINAL REPORT | Cycles: {cycle}")
    if pnl_curve.trade_log:
        pnls = [t["pnl_usd"] for t in pnl_curve.trade_log]
        wins = [p for p in pnls if p > 0]
        print(f"  Trades:        {len(pnls)}")
        print(f"  Win rate:      {len(wins)/len(pnls):.1%}  ({len(wins)}W / {len(pnls)-len(wins)}L)")
        print(f"  Realized P&L:  ${sum(pnls):+.2f}")
        print(f"  Best trade:    ${max(pnls):+.2f}")
        print(f"  Worst trade:   ${min(pnls):+.2f}")
        print(f"  Avg per trade: ${np.mean(pnls):+.2f}")
        print(f"\n  P&L curve:     {pnl_curve.log_path}")
    else:
        print("  No completed trades.")
    print(f"{'═'*65}")


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "live":
        print("\n⚠️  LIVE MODE — Real orders will be placed!")
        if input("Type 'yes' to confirm: ").strip().lower() != "yes":
            sys.exit(0)
    run(args)
