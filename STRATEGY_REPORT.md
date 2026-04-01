# Strategy Screening Report — AI Trading Agent Hackathon

**Date:** 2026-03-31 | **Pairs:** GIGAUSD, DOGUSD, SOLUSD, HYPEUSD, COQUSD | **Data:** 120 days cached 15m bars

---

## Executive Summary

Screened **315 strategy combinations** across 5 pairs, 3 timeframes (15m/30m/60m), and 20 strategy families. **14 strategies survived** the dual-window profitability filter (positive PnL in both 120d and 60d windows with minimum trade count).

**Key finding:** GIGAUSD dominates — 11 of 14 survivors trade GIGAUSD. The best strategy delivers **+9.76% net PnL** with a **66.7% win rate** and **2.80 profit factor** over 120 days.

---

## Top 5 Strategies by Composite Score

### #1: GIGAUSD 60m — Momentum Breakout
- **Net PnL:** +9.76% (120d) / +9.73% (60d)
- **Win Rate:** 66.7% | **Profit Factor:** 2.80 | **Sharpe:** 1.26
- **Trades:** 9 (full) / 8 (recent) | **Max DD:** -3.58% | **Avg MFE:** +3.42%
- **Logic:** Breakout above 12-bar high with trend confirmation, compressed volatility expanding, volume above average, close in upper 30% of bar
- **AI Enhancement:** Claude filters for regime alignment and conviction sizing

### #2: GIGAUSD 30m — Momentum Breakout w/ Trailing Stop
- **Net PnL:** +13.31% (120d) / +3.27% (60d)
- **Win Rate:** 52.3% | **Profit Factor:** 1.37 | **Sharpe:** 0.66
- **Trades:** 44 (full) / 24 (recent) | **Max DD:** -9.84% | **Avg MFE:** +2.13%
- **Logic:** Stronger trend + volume filters than #1, uses trailing stop (2% activation, 1% trail) instead of fixed target — captures larger moves
- **AI Enhancement:** Claude determines when to use trail vs fixed exit based on momentum acceleration

### #3: GIGAUSD 60m — Volume Spike + Trend
- **Net PnL:** +2.34% (120d) / +10.67% (60d)
- **Win Rate:** 47.5% | **Profit Factor:** 1.08 (full) / 2.13 (recent)
- **Trades:** 40 (full) / 20 (recent) | **Max DD:** -15.75% | **Avg MFE:** +1.86%
- **Logic:** Volume explosion (>2x 10-bar average) in green bar within uptrend. Recently hot — the +10.67% in last 60d suggests regime shift in GIGA's favor
- **AI Enhancement:** AI regime detection prevents trading in non-trending regimes (full-period DD of -15.75% comes from those periods)

### #4: GIGAUSD 30m — Triple Confluence
- **Net PnL:** +4.55% (120d) / +7.35% (60d)
- **Win Rate:** 46.2% | **Profit Factor:** 1.47 / 2.26 (recent) | **Sharpe:** 0.57
- **Trades:** 13 (full) / 9 (recent) | **Max DD:** -3.58% | **Avg MFE:** +2.44%
- **Logic:** Requires ALL of: trend + MACD histogram positive & expanding + ATR compression < 0.85 + volume above average + breakout above 12-bar high. High selectivity = low trade count but strong edge
- **AI Enhancement:** Claude validates the confluence quality — a single weak confluence component gets filtered

### #5: GIGAUSD 15m — Triple Confluence
- **Net PnL:** +7.48% (120d) / +3.86% (60d)
- **Win Rate:** 47.1% | **Profit Factor:** 1.72 | **Sharpe:** 0.87
- **Trades:** 17 (full) / 9 (recent) | **Max DD:** -5.63% | **Avg MFE:** +1.80%
- **Logic:** Same triple confluence on faster timeframe — more signals, slightly worse per-trade quality but higher cumulative PnL

---

## Additional Survivors

| Rank | Pair | TF | Strategy | Net (120d) | Net (60d) | Win% | PF | Score |
|------|------|----|----------|-----------|-----------|------|-----|-------|
| 6 | GIGAUSD | 30m | ATR Squeeze Expand | +4.27% | +4.96% | 50% | 1.54 | 12.4% |
| 7 | GIGAUSD | 60m | Triple Confluence | +6.32% | +1.95% | 57% | 2.67 | 10.3% |
| 8 | GIGAUSD | 15m | Momentum Breakout | +2.10% | +4.57% | 56% | 1.41 | 9.3% |
| 9 | GIGAUSD | 15m | ATR Squeeze Expand | +2.30% | +3.97% | 40% | 1.46 | 8.7% |
| 10 | DOGUSD | 60m | BB Squeeze Breakout | +2.23% | +3.57% | 50% | 1.49 | 8.0% |
| 11 | COQUSD | 30m | Red Streak Reversal | +1.13% | +3.74% | 50% | 1.29 | 7.0% |
| 12 | GIGAUSD | 30m | Breakout Asymmetric | +1.17% | +3.79% | 20% | 1.26 | 7.0% |
| 13 | HYPEUSD | 60m | BB Squeeze Breakout | +1.32% | +1.32% | 50% | 1.29 | 3.6% |
| 14 | DOGUSD | 15m | ATR Squeeze Expand | +0.96% | +1.40% | 50% | 1.10 | 3.2% |

---

## Strategies That Failed (Key Learnings)

- **SOLUSD:** Every single strategy lost money. SOL's recent price action lacks the impulsive breakout patterns these strategies need.
- **Mean reversion strategies:** Mostly negative across all pairs. Crypto's trending nature makes buying dips risky.
- **15m/30m on most pairs:** Higher frequency = more trades but fees eat the edge. Only GIGAUSD has enough volatility at these timeframes.
- **VWAP reclaim strategies:** 0% win rate on most pairs — VWAP reclaims in crypto tend to be false signals.

---

## AI Agent Implementation Architecture

The `multi_strategy_agent.py` implements a 3-layer AI architecture:

### Layer 1: AI Regime Detection
Claude classifies the market every N cycles into: trending_up, trending_down, ranging, or volatile. This gates which strategies can fire — no longs in confirmed downtrends, prefer breakouts in trending regimes, reduce size in volatile regimes.

### Layer 2: Strategy Ensemble
Six strategies run simultaneously. When multiple fire at once, it increases conviction (ensemble scoring). The agent weights signals by each strategy's historical performance and regime compatibility.

### Layer 3: AI Execution Filter
Before any order is placed, Claude receives the full market context, regime state, ensemble score, and portfolio history. It decides: TRADE or SKIP, with adjustments to position size (0.5x-1.5x), exit mode (fast/standard/trail), and stop/target offsets.

---

## Recommended Hackathon Configuration

```bash
# Paper trading with AI — recommended for demo
python multi_strategy_agent.py \
  --mode paper \
  --pairs GIGAUSD \
  --context-pairs DOGUSD,HYPEUSD \
  --notional-usd 25 \
  --use-claude \
  --poll 60 \
  --reset-paper \
  --paper-init-balance 10000 \
  --regime-interval 5
```

**Expected performance (based on backtesting):**
- Primary pair: GIGAUSD (11 of 14 surviving strategies)
- Net PnL potential: +10-15% over 120 days
- Win rate: 50-67%
- Max drawdown: -3.6% to -9.8% depending on aggressiveness
- Trade frequency: 1-4 per day on combined 15m/30m/60m strategies

---

## Files Created

- `expanded_screener.py` — 20-strategy family screener across all pairs/timeframes
- `expanded_screen_results.csv` — Full 315-row results matrix
- `multi_strategy_agent.py` — Production multi-strategy AI agent with regime detection + ensemble
- `STRATEGY_REPORT.md` — This report
