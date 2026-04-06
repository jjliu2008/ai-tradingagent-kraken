"""
find_alpha_pairs.py — Scan Kraken pairs for GIGA-like alpha
============================================================
Fetches live OHLC from Kraken, runs the full consensus backtest on each pair,
and ranks by quality. Run this on the Pi (needs Kraken network access).

Usage:
    python3 find_alpha_pairs.py
    python3 find_alpha_pairs.py --interval 60 --notional 150 --min-trades 3
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import kraken_client as kc
import strategy as strat
from consensus_agent import compute_all_features, SIGNALS, SIGNAL_NAMES

# ── Params ────────────────────────────────────────────────────────────────────
STOP_PCT    = 0.015
TARGET_PCT  = 0.045
MAX_BARS    = 10
COOLDOWN    = 3
COMMISSION  = 0.0026
SLIPPAGE    = 0.0005
ROUND_TRIP  = (COMMISSION + SLIPPAGE) * 2
MIN_VOTES   = 2

def get_all_usd_pairs() -> list[str]:
    """Fetch every live USD pair from Kraken's AssetPairs API."""
    try:
        raw = kc.fetch_asset_pairs()
        pairs = []
        for name, info in raw.items():
            # Only spot USD pairs, no leveraged/dark-pool variants
            if (info.get("quote") in ("ZUSD", "USD")
                    and info.get("status") == "online"
                    and ".d" not in name):
                # Use the altname (clean name like XBTUSD) if available
                alt = info.get("altname", name)
                pairs.append(alt)
        return sorted(set(pairs))
    except Exception as e:
        print(f"  Warning: could not fetch pair list ({e}), using built-in list")
        return [
            "GIGAUSD","ZECUSD","PEPEUSD","WIFUSD","BONKUSD","FLOKUSD","MEMEUSD",
            "POPCATUSD","MOGUSD","TURBOUSD","GOATUSD","MOONUSD","PENGUUSD",
            "FETUSD","VIRTUALUSD","NEARUSD","TAOUSD","INJUSD","JUPUSD","TIAUSD",
            "DYMUSD","STXUSD","ENAUSD","ETHFIUSD","AAVEUSD","SEIUSD","APTUSD",
            "SUIUSD","ARBUSD","OPUSD","STRKUSD","ZKUSD","ADAUSD","XRPUSD",
            "LINKUSD","DOTUSD","ATOMUSD","AVAXUSD","UNIUSD","LTCUSD","XBTUSD",
            "ETHUSD","SOLUSD","XDGUSD","TRXUSD","NIGHTUSD","HYPEUSD","COQUSD",
        ]


@dataclass
class PairResult:
    pair:       str
    n_trades:   int
    pnl:        float
    win_rate:   float
    pf:         float
    sharpe:     float
    max_dd:     float
    freq_week:  float   # trades per week
    n_bars:     int


def fetch_ohlc_paginated(pair: str, interval: int, days: int = 120) -> pd.DataFrame | None:
    """Fetch up to `days` of OHLC by paginating Kraken (720 bars per call)."""
    import time as _time
    bars_needed = days * 24 * (60 // interval)
    calls_needed = max(1, -(-bars_needed // 720))   # ceiling division

    now_ts = int(_time.time())
    # Walk backwards: oldest chunk first
    chunks = []
    for i in range(calls_needed - 1, -1, -1):
        since_ts = now_ts - (i + 1) * 720 * interval * 60
        try:
            raw = kc.fetch_ohlc(pair, interval=interval, since=since_ts)
            chunk = strat.parse_ohlc(raw)
            if chunk is not None and len(chunk) > 0:
                chunks.append(chunk)
        except Exception:
            pass
        _time.sleep(0.15)   # gentle on Kraken rate limits

    if not chunks:
        return None
    df = pd.concat(chunks).drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    df = df.iloc[:-1]   # drop incomplete current candle
    return df if len(df) >= 100 else None


def backtest_live(pair: str, interval: int, notional: float, days: int = 120) -> PairResult | None:
    """Fetch paginated OHLC and run backtest. Returns None if pair unavailable."""
    try:
        df = fetch_ohlc_paginated(pair, interval=interval, days=days)
        if df is None:
            return None
    except Exception:
        return None

    try:
        dff = compute_all_features(df)
    except Exception:
        return None

    trades = []
    realized = 0.0
    equity_series = []
    cooldown_until = -1
    in_trade = False
    entry_bar = 0
    entry_price = 0.0

    for i in range(60, len(dff)):
        row   = dff.iloc[i]
        close = float(row["close"])

        unrealized = 0.0
        if in_trade:
            unrealized = ((close - entry_price) / entry_price) * notional

        equity_series.append(realized + unrealized)

        if in_trade:
            bars_held   = i - entry_bar
            stop_price  = entry_price * (1 - STOP_PCT)
            target_price = entry_price * (1 + TARGET_PCT)
            reason = None
            if close <= stop_price:
                reason = "SL"
            elif close >= target_price:
                reason = "TP"
            elif bars_held >= MAX_BARS:
                reason = "TL"
            elif bars_held >= 2:
                if float(row.get("ema_fast", close)) < float(row.get("ema_slow", close)):
                    reason = "TR"
            if reason:
                pnl = ((close - entry_price) / entry_price - ROUND_TRIP) * notional
                realized += pnl
                trades.append(pnl)
                in_trade = False
                cooldown_until = i + COOLDOWN
            continue

        if i <= cooldown_until:
            continue

        # trailing window (avoids O(n²) full slice)
        window = dff.iloc[max(0, i - 59):i + 1]
        fires  = [n for fn, n in zip(SIGNALS, SIGNAL_NAMES) if fn(window)]
        if len(fires) >= MIN_VOTES:
            in_trade    = True
            entry_bar   = i
            entry_price = close

    if not trades:
        return PairResult(pair, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, len(dff))

    pnls   = np.array(trades)
    wins   = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    eq     = np.array(equity_series)
    peak   = np.maximum.accumulate(eq)
    dd     = (eq - peak).min()

    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and abs(losses.sum()) > 0 else 99.0
    bars_per_year = 365 * 24 * (60 // interval) * 0.75
    sharpe = (pnls.mean() / pnls.std() * np.sqrt(bars_per_year / max(len(trades), 1))
              if pnls.std() > 0 else 0.0)

    # Estimate days from bar count
    bars_per_day = 24 * (60 // interval) * 0.75
    n_days = len(dff) / bars_per_day
    freq_week = len(trades) / n_days * 7

    return PairResult(
        pair=pair, n_trades=len(trades),
        pnl=float(pnls.sum()), win_rate=float((pnls > 0).mean()),
        pf=float(min(pf, 99)), sharpe=float(sharpe),
        max_dd=float(dd), freq_week=freq_week, n_bars=len(dff),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval",   type=int,   default=60)
    parser.add_argument("--notional",   type=float, default=150.0)
    parser.add_argument("--days",       type=int,   default=120,
                        help="Days of history to fetch (default 120)")
    parser.add_argument("--min-trades", type=int,   default=1,
                        help="Min trades to include in ranking (default 1 — show all profitable)")
    parser.add_argument("--top",        type=int,   default=30,
                        help="Show top N pairs")
    parser.add_argument("--pairs",      default="",
                        help="Comma-separated pairs to test (default: all Kraken USD pairs)")
    args = parser.parse_args()

    # Build candidate list: CLI override → dynamic API fetch → built-in fallback
    if args.pairs:
        candidates = [p.strip().upper() for p in args.pairs.split(",")]
    else:
        print("  Fetching full Kraken USD pair list...", end=" ", flush=True)
        candidates = get_all_usd_pairs()
        print(f"{len(candidates)} pairs found")

    bars_label = f"{args.interval}m"
    print(f"\n{'='*70}")
    print(f"  PAIR ALPHA SCANNER | interval={bars_label} | {args.days}d history | notional=${args.notional}")
    print(f"  Signals: {', '.join(SIGNAL_NAMES)}")
    print(f"  Min votes: {MIN_VOTES}/{len(SIGNALS)} | Testing {len(candidates)} pairs...")
    print(f"  ETA: ~{len(candidates)//3} min — testing every live USD pair on Kraken")
    print(f"{'='*70}\n")

    results = []
    for i, pair in enumerate(candidates):
        print(f"  [{i+1:3d}/{len(candidates)}] {pair:<14} ...", end=" ", flush=True)
        t0 = time.time()
        r  = backtest_live(pair, args.interval, args.notional, days=args.days)
        elapsed = time.time() - t0

        if r is None:
            print("unavailable")
            continue

        if r.n_trades == 0:
            print(f"0 trades ({r.n_bars} bars)")
            continue

        marker = ""
        if r.pnl > 5 and r.win_rate >= 0.50 and r.pf >= 2.0:
            marker = "  ⭐"
        elif r.pnl > 0 and r.win_rate >= 0.40:
            marker = "  ✓"

        print(f"{r.n_trades:2d} trades | {r.pnl:+7.2f} | win={r.win_rate:.0%} | "
              f"PF={r.pf:.2f} | Sharpe={r.sharpe:.1f} | {r.freq_week:.1f}/wk{marker}")
        results.append(r)
        time.sleep(0.25)   # gentle on Kraken API

    # ── Ranked summary ────────────────────────────────────────────────────────
    import math
    all_profitable = [r for r in results if r.pnl > 0]
    qualified      = [r for r in all_profitable if r.n_trades >= args.min_trades]

    def score(r):
        return r.win_rate * math.log(r.pf + 1) * math.log(r.n_trades + 1)

    qualified.sort(key=score, reverse=True)

    print(f"\n{'='*74}")
    print(f"  ALL PROFITABLE PAIRS (>={args.min_trades} trade) — ranked by quality x frequency")
    print(f"{'='*74}")
    print("%-14s %6s %8s %5s %6s %8s %7s  %s" % (
        "Pair", "Trades", "Net P&L", "Win%", "PF", "Sharpe", "Freq/wk", "Tier"))
    print("-" * 74)
    for r in qualified[:args.top]:
        if r.win_rate >= 0.60 and r.pf >= 2.0 and r.n_trades >= 3:
            tier = "⭐ A"
        elif r.win_rate >= 0.50 and r.pf >= 1.5:
            tier = "✓  B"
        elif r.win_rate >= 0.33:
            tier = "   C"
        else:
            tier = "   D"
        print("%-14s %6d %+8.2f %4.0f%% %6.2f %8.1f %7.1f  %s" % (
            r.pair, r.n_trades, r.pnl, r.win_rate * 100,
            r.pf, r.sharpe, r.freq_week, tier))

    if not qualified:
        print("  No profitable pairs found.")

    a_pairs = [r.pair for r in qualified if r.win_rate >= 0.60 and r.pf >= 2.0 and r.n_trades >= 3]
    b_pairs = [r.pair for r in qualified if r.win_rate >= 0.50 and r.pf >= 1.5 and r.pair not in a_pairs]
    c_pairs = [r.pair for r in qualified if r.win_rate >= 0.33 and r.pair not in a_pairs + b_pairs]

    print(f"\n  Tested: {len(results)} | Profitable: {len(all_profitable)} | Qualified: {len(qualified)}")
    if a_pairs:
        print(f"\n  ⭐ TIER A — add to universe now:")
        print(f"     {','.join(a_pairs)}")
    if b_pairs:
        print(f"  ✓  TIER B — worth monitoring:")
        print(f"     {','.join(b_pairs[:15])}")
    if c_pairs:
        print(f"     TIER C — marginal, use cautiously:")
        print(f"     {','.join(c_pairs[:10])}")


if __name__ == "__main__":
    main()
