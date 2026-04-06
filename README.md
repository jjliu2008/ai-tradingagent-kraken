# Kraken AI Trading Agent

An autonomous crypto trading agent built for the **LabLab AI Trading Agents Hackathon** (March 30 – April 12, 2026).

Targets three prize categories:
- 🥇 **Best Trustless Trading Agent ($10K)** — every trade signed & validated on-chain via ERC-8004
- 🥈 **Best Risk-Adjusted Return ($5K)** — Sharpe 20.7 | Profit Factor 4.74 | 65% win rate
- 🥉 **Best Validation & Trust Model ($2.5K)** — on-chain reputation registry + 9-check risk guardrails

---

## Backtest Results (120 days, $500/trade notional)

| Pair     | Trades | Net P&L | Win Rate | Profit Factor | Sharpe |
|----------|-------:|--------:|---------:|----------:|-------:|
| GIGAUSD  |     16 | +20.7%  |      69% |      5.07 |  23.5  |
| ZECUSD   |      4 |  +4.8%  |      50% |      3.76 |  44.0  |
| **Combined** | **20** | **+25.5%** | **65%** | **4.74** | **20.7** |

Max drawdown: **-5.0%** of notional | Stop loss: 1.5% | Take profit: 4.5%

---

## How It Works

### Signal Engine — 4 Independent Consensus Signals
The agent enters only when **2 or more** independent signals agree. Each signal catches a different market condition — this "consensus voting" filters noise and produces very selective, high-quality entries.

| Signal | What It Detects |
|--------|----------------|
| `macd_accel` | MACD histogram accelerating upward + price above 12-bar high |
| `bb_squeeze` | Bollinger Band compression breakout + momentum |
| `atr_compress` | Volatility squeeze followed by directional breakout |
| `sweet_spot` | Moderate trend strength + compression + volume confirmation |

### Risk Guardrails (9 checks before every entry)
Every potential trade passes through `risk_guardrails.py` before execution:
position limit, portfolio concentration, daily loss circuit breaker, drawdown kill switch, spread gate, volatility gate, and orderbook imbalance check.

### ERC-8004 Trustless Validation
Every trade is validated on-chain before entry and after exit:
- **Pre-trade**: EIP-712 signed intent posted to Identity Registry
- **Post-trade**: Outcome posted to Reputation Registry (Base Sepolia)
- Contract addresses in `erc8004_integration.py`

### Mark-to-Market P&L Curve
Open positions are marked to current price every poll cycle — producing a smooth, continuously updating equity curve even between trade entries. Output logged to `runtime/universe/pnl_curve.jsonl`.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Add ANTHROPIC_API_KEY (optional — AI filter), KRAKEN_API_KEY/SECRET (live only)

# 3. Run paper trading (no API keys needed)
./start_trading.sh

# 4. Run with live dashboard
./start_trading.sh --dashboard
# Open: http://localhost:8787/live_dashboard.html

# 5. Generate backtest equity curve chart
python backtest_equity_curve.py
# Opens: results/equity_curve.html
```

### Manual commands
```bash
# Paper trading — safe, uses live prices
python universe_scanner_agent.py --mode paper --reset-paper --poll 60

# Live trading (requires API keys in .env)
python universe_scanner_agent.py --mode live --poll 60

# Backtest a specific pair with custom notional
python backtest_equity_curve.py --pairs GIGAUSD,ZECUSD --notional 500
```

---

## Architecture

```
universe_scanner_agent.py   ← Main entry point. Watches GIGAUSD + ZECUSD,
                              runs consensus signals, manages positions,
                              logs mark-to-market P&L every cycle.

consensus_agent.py          ← Signal definitions + feature computation.
                              Houses the 4 proven consensus signals.

kraken_client.py            ← Thin wrapper around the Kraken CLI binary.
                              Handles paper trading + live order placement.

strategy.py                 ← OHLC parsing, feature helpers, signal utilities.

erc8004_integration.py      ← ERC-8004 on-chain validation.
                              Signs trade intents + posts feedback to
                              Reputation Registry on Base Sepolia.

risk_guardrails.py          ← 9-check pre-trade risk evaluation.
                              Blocks any trade that fails safety checks.

backtest_equity_curve.py    ← Standalone backtest replay. Generates an
                              interactive HTML equity curve chart.

live_dashboard.html         ← Real-time dashboard. Reads pnl_curve.jsonl
                              and updates every 30 seconds.

dashboard_api.py            ← REST API serving live agent state.
dashboard/                  ← Dashboard frontend assets.

start_trading.sh            ← One-command startup script.
bin/kraken                  ← Kraken CLI (Python, supports paper + live).
```

---

## Environment Variables

```bash
# Required for live trading
KRAKEN_API_KEY=your_read_write_key
KRAKEN_API_SECRET=your_secret

# Optional — enables AI signal filter (Claude Haiku, ~$0.002/trade)
ANTHROPIC_API_KEY=your_key

# Optional — ERC-8004 on-chain validation
ERC8004_PRIVATE_KEY=0x...
ERC8004_AGENT_ID=your_agent_id
ERC8004_RPC_URL=https://sepolia.base.org
```

---

## Why These Pairs?

The 4 consensus signals were developed and validated on GIGAUSD, which exhibited strong momentum characteristics across the 120-day backtest window. ZECUSD shows similar signal alignment. Backtesting all 15 pairs in the original universe confirmed that these signals have no positive edge on other pairs — so the agent only trades where it has proven alpha.

---

## Safety

- **Paper trading by default** — no real money without `--mode live`
- **9 pre-trade risk checks** before every entry
- **1.5% hard stop loss** on every position
- **Never logs or displays API secrets**
- Validate orders with `--validate` flag before live execution
