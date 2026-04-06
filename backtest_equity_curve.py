"""
Mark-to-Market Equity Curve Backtest
======================================
Replays the consensus signals on cached OHLC data and produces a smooth
mark-to-market equity curve — the same curve the live agent generates.

Every 15-minute bar is marked to current price, so open positions update
the equity continuously even between actual trade entries/exits.

Usage:
    python backtest_equity_curve.py
    python backtest_equity_curve.py --pairs GIGAUSD,ZECUSD --notional 500
    python backtest_equity_curve.py --output results/equity_curve.html
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from consensus_agent import compute_all_features, SIGNALS, SIGNAL_NAMES

# ── Parameters ───────────────────────────────────────────────────────────────
STOP_PCT      = 0.015     # 1.5% stop loss
TARGET_PCT    = 0.045     # 4.5% take profit
MAX_BARS      = 10        # bars before time-limit exit
COOLDOWN      = 3         # bars to wait after exit
COMMISSION    = 0.0026    # Kraken taker 0.26%
SLIPPAGE      = 0.0005    # assumed slippage
ROUND_TRIP    = (COMMISSION + SLIPPAGE) * 2
DATA_DIR      = Path("data_cache")

# Per-pair vote thresholds — mirrors universe_scanner_agent.py
# GIGAUSD: votes=1 → -16% net, votes=2 → +20.7% net (must stay at 2)
# ZECUSD:  votes=1 → +15.8% net (positive expectancy, ~6 trades/wk)
PAIR_MIN_VOTES: dict[str, int] = {
    "GIGAUSD": 2,
    "ZECUSD":  2,
    "SOLUSD":  2,
    "ETHUSD":  2,
    "XBTUSD":  2,
    "LINKUSD": 2,
    "AVAXUSD": 2,
    "OPUSD":   2,
}
DEFAULT_MIN_VOTES = 2

# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class EquityPoint:
    bar_index:   int
    timestamp:   float
    equity:      float     # cumulative equity (mark-to-market)
    realized:    float     # realized P&L component
    unrealized:  float     # unrealized P&L component
    n_open:      int
    n_trades:    int

@dataclass
class Trade:
    pair:          str
    entry_bar:     int
    entry_price:   float
    exit_price:    float
    exit_bar:      int
    pnl_net:       float
    reason:        str
    votes:         int
    signals_fired: list[str]


# ── Core backtest with mark-to-market ─────────────────────────────────────────
def backtest_pair_mtm(
    pair: str,
    df_raw: pd.DataFrame,
    notional_usd: float = 150.0,
) -> tuple[list[Trade], list[EquityPoint]]:
    """
    Returns (trades, equity_points) — equity_points logged every bar
    with mark-to-market valuation of open positions.
    """
    try:
        df = compute_all_features(df_raw)
    except Exception as e:
        print(f"  Feature error {pair}: {e}")
        return [], []

    trades:         list[Trade]        = []
    equity_points:  list[EquityPoint]  = []

    realized_pnl_usd = 0.0
    cooldown_until   = -1
    in_trade         = False
    entry_bar        = 0
    entry_price      = 0.0
    active_votes     = 0
    active_signals:  list[str] = []

    for i in range(60, len(df)):
        row   = df.iloc[i]
        close = float(row["close"])
        ts    = float(row.get("ts", i))

        # ── Mark open position to market ──
        unrealized_usd = 0.0
        if in_trade:
            unrealized_pct = (close - entry_price) / entry_price
            unrealized_usd = unrealized_pct * notional_usd - (notional_usd * ROUND_TRIP)

        total_equity = realized_pnl_usd + unrealized_usd
        equity_points.append(EquityPoint(
            bar_index=i, timestamp=ts,
            equity=total_equity, realized=realized_pnl_usd,
            unrealized=unrealized_usd, n_open=1 if in_trade else 0,
            n_trades=len(trades),
        ))

        # ── Manage open trade ──
        if in_trade:
            bars_held   = i - entry_bar
            exit_reason = None

            stop_price   = entry_price * (1 - STOP_PCT)
            target_price = entry_price * (1 + TARGET_PCT)

            if close <= stop_price:
                exit_reason = "STOP_LOSS"
            elif close >= target_price:
                exit_reason = "TAKE_PROFIT"
            elif bars_held >= MAX_BARS:
                exit_reason = "TIME_LIMIT"
            elif bars_held >= 2:
                ema_fast = float(row.get("ema_fast", close))
                ema_slow = float(row.get("ema_slow", close))
                mom      = float(row.get("momentum_medium", 0))
                if ema_fast < ema_slow and mom < 0:
                    exit_reason = "TREND_LOST"

            if exit_reason:
                pnl_pct = (close - entry_price) / entry_price - ROUND_TRIP
                pnl_usd = pnl_pct * notional_usd
                realized_pnl_usd += pnl_usd
                trades.append(Trade(
                    pair=pair, entry_bar=entry_bar, entry_price=entry_price,
                    exit_price=close, exit_bar=i, pnl_net=pnl_usd,
                    reason=exit_reason, votes=active_votes,
                    signals_fired=active_signals,
                ))
                in_trade = False
                cooldown_until = i + COOLDOWN
            continue

        # ── Cooldown guard ──
        if i <= cooldown_until:
            continue

        # ── Check consensus signals (per-pair threshold) ──
        window    = df.iloc[:i + 1]
        fires     = [name for fn, name in zip(SIGNALS, SIGNAL_NAMES) if fn(window)]
        min_votes = PAIR_MIN_VOTES.get(pair, DEFAULT_MIN_VOTES)

        if len(fires) >= min_votes:
            in_trade       = True
            entry_bar      = i
            entry_price    = close
            active_signals = fires
            active_votes   = len(fires)

    return trades, equity_points


# ── Statistics ─────────────────────────────────────────────────────────────────
def compute_stats(trades: list[Trade], equity_points: list[EquityPoint]) -> dict:
    if not trades:
        return {}

    pnls  = np.array([t.pnl_net for t in trades])
    wins  = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    equities = np.array([e.equity for e in equity_points])
    peak     = np.maximum.accumulate(equities)
    dd       = equities - peak
    max_dd   = float(dd.min())

    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and abs(losses.sum()) > 0 else float("inf")

    # Annualised Sharpe from 15-min bars (35040/year)
    daily_pnls = pnls  # trade-level
    if daily_pnls.std() > 0:
        sharpe = (daily_pnls.mean() / daily_pnls.std()) * np.sqrt(35_040 / max(len(trades), 1))
    else:
        sharpe = 0.0

    return {
        "n_trades":     len(trades),
        "n_wins":       int((pnls > 0).sum()),
        "n_losses":     int((pnls <= 0).sum()),
        "win_rate":     float((pnls > 0).mean()),
        "total_pnl_usd":float(pnls.sum()),
        "avg_pnl_usd":  float(pnls.mean()),
        "profit_factor":float(pf),
        "max_dd_usd":   float(max_dd),
        "sharpe":       float(sharpe),
        "exit_reasons": {r: sum(1 for t in trades if t.reason == r)
                         for r in sorted({t.reason for t in trades})},
    }


# ── HTML Chart ─────────────────────────────────────────────────────────────────
CHART_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Universe Scanner — Equity Curve</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  body  {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #0d1117; color: #e6edf3; margin: 0; padding: 20px; }}
  h1    {{ color: #58a6ff; margin: 0 0 4px; font-size: 22px; }}
  .sub  {{ color: #8b949e; margin: 0 0 24px; font-size: 13px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px 18px; }}
  .card .label {{ font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: .5px; }}
  .card .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
  .pos  {{ color: #3fb950; }} .neg {{ color: #f85149; }} .neu {{ color: #79c0ff; }}
  .chart-wrap {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 24px; }}
  .chart-title {{ color: #8b949e; font-size: 13px; margin: 0 0 12px; }}
  table {{ width: 100%; border-collapse: collapse; background: #161b22; border-radius: 8px; overflow: hidden; }}
  th    {{ background: #21262d; color: #8b949e; font-size: 11px; text-transform: uppercase;
           padding: 8px 14px; text-align: right; border-bottom: 1px solid #30363d; }}
  th:first-child {{ text-align: left; }}
  td    {{ padding: 8px 14px; text-align: right; border-bottom: 1px solid #21262d; font-size: 13px; }}
  td:first-child {{ text-align: left; font-weight: 500; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #1c2128; }}
</style>
</head>
<body>
<h1>Universe Scanner — Mark-to-Market Equity Curve</h1>
<p class="sub">Pairs: {pairs_label} | Notional: ${notional}/trade | Stop: {stop_pct}% | Target: {target_pct}% | Min votes: {min_votes}/4</p>

<div class="grid">
  <div class="card"><div class="label">Total P&amp;L</div><div class="value {pnl_cls}">${total_pnl:+.2f}</div></div>
  <div class="card"><div class="label">Win Rate</div><div class="value neu">{win_rate:.0%}</div></div>
  <div class="card"><div class="label">Profit Factor</div><div class="value {pf_cls}">{pf:.2f}x</div></div>
  <div class="card"><div class="label">Sharpe (ann.)</div><div class="value neu">{sharpe:.2f}</div></div>
  <div class="card"><div class="label">Max Drawdown</div><div class="value neg">${max_dd:.2f}</div></div>
  <div class="card"><div class="label">Total Trades</div><div class="value neu">{n_trades}</div></div>
</div>

<div class="chart-wrap">
  <div class="chart-title">Equity Curve — Mark-to-Market (every 15-min bar) + Trend</div>
  <canvas id="equity" height="80"></canvas>
</div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">
<div class="chart-wrap">
  <div class="chart-title">Cumulative Realized P&amp;L per Trade</div>
  <canvas id="cumulative" height="120"></canvas>
</div>
<div class="chart-wrap">
  <div class="chart-title">Per-Trade P&amp;L Waterfall</div>
  <canvas id="waterfall" height="120"></canvas>
</div>
</div>

<h2 style="color:#8b949e;font-size:14px;margin:20px 0 10px">Trade Log</h2>
<table>
  <tr><th>Pair</th><th>#</th><th>Entry</th><th>Exit</th><th>P&amp;L ($)</th><th>Votes</th><th>Signals</th><th>Reason</th></tr>
  {trade_rows}
</table>

<script>
const equityLabels = {equity_labels};
const equityData   = {equity_data};
const tradePoints  = {trade_points};

// Compute linear trend overlay (regression)
function linRegression(data) {{
  const n = data.length;
  const xMean = (n - 1) / 2;
  const yMean = data.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  data.forEach((y, x) => {{ num += (x - xMean) * (y - yMean); den += (x - xMean) ** 2; }});
  const slope = num / den;
  const intercept = yMean - slope * xMean;
  return data.map((_, x) => +(intercept + slope * x).toFixed(4));
}}
const trendData = linRegression(equityData);

// Equity chart with trend overlay
new Chart(document.getElementById('equity'), {{
  type: 'line',
  data: {{
    labels: equityLabels,
    datasets: [
      {{
        label: 'Equity (mark-to-market)',
        data: equityData,
        borderColor: '#58a6ff',
        backgroundColor: 'rgba(88,166,255,0.06)',
        borderWidth: 1.5,
        fill: true,
        pointRadius: 0,
        tension: 0,
        order: 2,
      }},
      {{
        label: 'Trend',
        data: trendData,
        borderColor: '#3fb950',
        backgroundColor: 'transparent',
        borderWidth: 2,
        borderDash: [6, 3],
        pointRadius: 0,
        fill: false,
        order: 1,
      }},
    ]
  }},
  options: {{
    animation: false,
    plugins: {{
      legend: {{ labels: {{ color: '#8b949e', font: {{ size: 11 }} }} }},
      tooltip: {{ callbacks: {{ label: ctx => '$' + ctx.raw.toFixed(2) }} }},
    }},
    scales: {{
      x: {{ display: false }},
      y: {{ ticks: {{ color: '#8b949e', callback: v => '$' + v.toFixed(0) }}, grid: {{ color: '#21262d' }} }}
    }},
  }}
}});

// Waterfall (per-trade PnL bar chart)
const wfLabels = tradePoints.map(t => t.label);
const wfColors = tradePoints.map(t => t.pnl >= 0 ? 'rgba(63,185,80,0.8)' : 'rgba(248,81,73,0.8)');
new Chart(document.getElementById('waterfall'), {{
  type: 'bar',
  data: {{
    labels: wfLabels,
    datasets: [{{ label: 'Trade P&L ($)', data: tradePoints.map(t => t.pnl),
                  backgroundColor: wfColors, borderWidth: 0 }}]
  }},
  options: {{
    animation: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: ctx => '$' + ctx.raw.toFixed(2) }} }},
    }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e', font: {{ size: 10 }} }}, grid: {{ display: false }} }},
      y: {{ ticks: {{ color: '#8b949e', callback: v => '$' + v.toFixed(0) }}, grid: {{ color: '#21262d' }} }}
    }},
  }}
}});

// Cumulative realized PnL chart (per-trade staircase)
const cumLabels = tradePoints.map((t, i) => '#' + (i+1) + ' ' + t.label);
let cumSum = 0;
const cumData = tradePoints.map(t => {{ cumSum += t.pnl; return +cumSum.toFixed(4); }});
const cumColors = cumData.map(v => v >= 0 ? 'rgba(88,166,255,0.7)' : 'rgba(248,81,73,0.7)');
new Chart(document.getElementById('cumulative'), {{
  type: 'line',
  data: {{
    labels: cumLabels,
    datasets: [{{
      label: 'Cumulative Realized P&L',
      data: cumData,
      borderColor: '#3fb950',
      backgroundColor: 'rgba(63,185,80,0.08)',
      borderWidth: 2,
      fill: true,
      pointRadius: 4,
      pointBackgroundColor: cumColors,
      stepped: true,
      tension: 0,
    }}]
  }},
  options: {{
    animation: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: ctx => '$' + ctx.raw.toFixed(2) }} }},
    }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e', font: {{ size: 9 }}, maxRotation: 45 }}, grid: {{ display: false }} }},
      y: {{ ticks: {{ color: '#8b949e', callback: v => '$' + v.toFixed(0) }}, grid: {{ color: '#21262d' }} }}
    }},
  }}
}});
</script>
</body>
</html>
"""


def generate_html(
    all_trades: list[Trade],
    all_equity: list[EquityPoint],
    stats: dict,
    pairs: list[str],
    notional: float,
    output_path: Path,
) -> None:
    # Equity curve — sample every N points to keep HTML small
    step = max(1, len(all_equity) // 2000)
    sampled = all_equity[::step]
    labels = [str(e.bar_index) for e in sampled]
    data   = [round(e.equity, 4) for e in sampled]

    # Per-trade waterfall
    trade_points = json.dumps([
        {"label": f"{t.pair}#{i+1}", "pnl": round(t.pnl_net, 2)}
        for i, t in enumerate(all_trades)
    ])

    # Trade rows
    rows = []
    cumulative = 0.0
    for i, t in enumerate(all_trades):
        cumulative += t.pnl_net
        color = "pos" if t.pnl_net >= 0 else "neg"
        rows.append(
            f"<tr>"
            f"<td>{t.pair}</td>"
            f"<td>{i+1}</td>"
            f"<td>{t.entry_price:.5g}</td>"
            f"<td>{t.exit_price:.5g}</td>"
            f"<td class='{color}'>${t.pnl_net:+.2f}</td>"
            f"<td>{t.votes}</td>"
            f"<td style='font-size:11px'>{', '.join(t.signals_fired)}</td>"
            f"<td>{t.reason}</td>"
            f"</tr>"
        )

    s = stats
    total_pnl = s.get("total_pnl_usd", 0)
    pnl_cls   = "pos" if total_pnl >= 0 else "neg"
    pf        = s.get("profit_factor", 0)
    pf_cls    = "pos" if pf >= 1.5 else ("neg" if pf < 1.0 else "neu")

    html = CHART_TEMPLATE.format(
        pairs_label  = " + ".join(pairs),
        notional     = notional,
        stop_pct     = STOP_PCT * 100,
        target_pct   = TARGET_PCT * 100,
        min_votes    = " / ".join(f"{p}:{v}" for p, v in PAIR_MIN_VOTES.items()),
        total_pnl    = total_pnl,
        pnl_cls      = pnl_cls,
        win_rate     = s.get("win_rate", 0),
        pf           = pf,
        pf_cls       = pf_cls,
        sharpe       = s.get("sharpe", 0),
        max_dd       = abs(s.get("max_dd_usd", 0)),
        n_trades     = s.get("n_trades", 0),
        equity_labels= json.dumps(labels),
        equity_data  = json.dumps(data),
        trade_points = trade_points,
        trade_rows   = "\n  ".join(rows),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"  Chart saved → {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def find_best_data() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(DATA_DIR.glob("*_15m_*.csv")):
        parts = path.stem.split("_")
        if len(parts) < 3:
            continue
        pair = parts[0]
        try:
            days = int(parts[2].replace("d", ""))
        except ValueError:
            continue
        if pair not in found:
            found[pair] = path
        else:
            existing = int(found[pair].stem.split("_")[2].replace("d", ""))
            if days > existing:
                found[pair] = path
    return found


def load_pair(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "ts"})
        required = {"ts", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            return None
        df = df.sort_values("ts").reset_index(drop=True)
        if "vwap_k" not in df.columns:
            df["vwap_k"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        if "count" not in df.columns:
            df["count"] = 1
        return df
    except Exception as e:
        print(f"  Load error {path.name}: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Mark-to-Market Equity Curve Backtest")
    parser.add_argument("--pairs",    default="GIGAUSD,ZECUSD",
                        help="Comma-separated pairs to backtest")
    parser.add_argument("--notional", type=float, default=150.0,
                        help="USD notional per position")
    parser.add_argument("--output",   default="results/equity_curve.html",
                        help="Output HTML path")
    parser.add_argument("--json",     default="results/backtest_stats.json",
                        help="Output JSON stats path")
    args = parser.parse_args()

    pairs      = [p.strip().upper() for p in args.pairs.split(",")]
    all_paths  = find_best_data()
    output_path = Path(args.output)
    json_path   = Path(args.json)

    all_trades:  list[Trade]        = []
    all_equity:  list[EquityPoint]  = []
    per_pair_stats: dict[str, dict] = {}

    print(f"\n{'='*60}")
    print("  MARK-TO-MARKET EQUITY CURVE BACKTEST")
    print(f"  Pairs:    {', '.join(pairs)}")
    print(f"  Notional: ${args.notional}/trade")
    print(f"  Signals:  {', '.join(SIGNAL_NAMES)}")
    votes_str = " | ".join(f"{p}:{v}/4" for p, v in PAIR_MIN_VOTES.items())
    print(f"  Min votes: {votes_str}")
    print(f"{'='*60}\n")

    for pair in pairs:
        path = all_paths.get(pair)
        if path is None:
            print(f"  {pair}: no data file found — skipping")
            continue

        days = path.stem.split("_")[2]
        df   = load_pair(path)
        if df is None or len(df) < 100:
            print(f"  {pair}: insufficient data — skipping")
            continue

        print(f"  {pair} ({days}, {len(df)} bars) ...", end=" ", flush=True)
        trades, equity = backtest_pair_mtm(pair, df, notional_usd=args.notional)
        stats = compute_stats(trades, equity)
        per_pair_stats[pair] = stats

        if trades:
            print(f"{stats['n_trades']} trades | "
                  f"net=${stats['total_pnl_usd']:+.2f} | "
                  f"win={stats['win_rate']:.0%} | "
                  f"PF={stats['profit_factor']:.2f} | "
                  f"Sharpe={stats['sharpe']:.2f}")
        else:
            print("0 trades")

        # Offset equity by realized PnL of previous pairs (shared equity)
        if all_equity and equity:
            offset = all_equity[-1].equity
            for pt in equity:
                pt.equity    += offset
                pt.realized  += offset
        all_trades.extend(trades)
        all_equity.extend(equity)

    if not all_trades:
        print("\n  No trades found — check data files or signal conditions.")
        return

    # Combined stats
    combined_stats = compute_stats(all_trades, all_equity)

    print(f"\n{'─'*60}")
    print("  COMBINED RESULTS")
    print(f"  Trades:         {combined_stats['n_trades']}")
    print(f"  Win rate:       {combined_stats['win_rate']:.1%}")
    print(f"  Total P&L:      ${combined_stats['total_pnl_usd']:+.2f}")
    print(f"  Profit factor:  {combined_stats['profit_factor']:.2f}x")
    print(f"  Sharpe (ann.):  {combined_stats['sharpe']:.2f}")
    print(f"  Max drawdown:   ${combined_stats['max_dd_usd']:.2f}")
    print(f"  Exit breakdown: {combined_stats['exit_reasons']}")

    # Per-pair breakdown
    print(f"\n  Per-pair breakdown:")
    for pair, s in per_pair_stats.items():
        if s:
            print(f"    {pair:<12} {s['n_trades']:>3} trades | "
                  f"${s['total_pnl_usd']:+.2f} | win={s['win_rate']:.0%} | "
                  f"PF={s['profit_factor']:.2f}")

    # Save outputs
    generate_html(all_trades, all_equity, combined_stats, pairs,
                  args.notional, output_path)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump({"combined": combined_stats, "per_pair": per_pair_stats,
                   "config": {"pairs": pairs, "notional": args.notional,
                              "stop_pct": STOP_PCT, "target_pct": TARGET_PCT,
                              "pair_min_votes": PAIR_MIN_VOTES}}, f, indent=2)
    print(f"  Stats saved  → {json_path}")


if __name__ == "__main__":
    main()
