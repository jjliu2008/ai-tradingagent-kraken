# Kraken AI Trading Agent

Kraken AI Trading Agent is a research-driven crypto trading system built for the LabLab AI Trading Agents Hackathon.

Instead of relying on one fixed strategy, it runs a daily research pipeline that screens pairs, diagnoses which strategies survive across older and recent market windows, ranks new candidates by real portfolio impact, and feeds a live research registry used by an intraday execution agent. The result is an automated trading workflow that is adaptive, inspectable, and designed to be safer to operate.

## What We Demo

The hackathon demo focuses on four connected pieces:

- A daily `n8n` research workflow that calls Python services to screen, diagnose, rank, and propose pair promotions
- An intraday 15-minute execution workflow that trades the active research book in paper mode
- A live dashboard that reads runtime artifacts and shows PnL and system state
- A trust and validation layer with risk guardrails, proposal artifacts, approval workflow plumbing, and ERC-8004 integration

## Current Demo Book

The current live research registry is:

- `GIGAUSD`
- `ZECUSD`
- `FHEUSD`
- `KERNELUSD`
- `HOUSEUSD`
- `BABYUSD`

Current demo-book backtest on uniform 15-minute cached data:

| Metric | Value |
|---|---:|
| Full 120d net | `+38.70%` |
| Older 60d net | `+3.98%` |
| Recent 60d net | `+34.72%` |
| Max drawdown | `-4.19%` |
| Trades | `27` |

These numbers come from the current `FHEUSD`-included book comparison artifact in `results/latest/fhe_vs_current_backtest.json`.

## Why This Project Is Different

Most trading bots are static. This one is structured like a research and execution loop:

- Daily research searches for pair-specific strategies that survive both weak and strong sub-periods
- Portfolio ranking measures whether a new pair actually improves the current book instead of looking good in isolation
- Intraday execution trades only the currently validated registry, with suppression logic during weak conditions
- Proposal artifacts and approval workflow structure make promotions auditable instead of opaque
- ERC-8004 integration adds a trustless validation path for trading actions and reputation logging

## Demo Flow

If you want to reproduce the hackathon demo, this is the shortest path:

### 1. Start the local research server

```bash
python research_server.py
```

This exposes the local HTTP endpoints used by `n8n` on `http://127.0.0.1:5680`.

### 2. Start n8n

```bash
cmd /c "npx n8n"
```

Then open the `n8n` UI on `http://localhost:5678`.

### 3. Run the workflows

The two key workflows are:

- `Daily Research Pipeline`
- `Intraday Agent - 15m Bar Close`

The daily workflow:

1. runs the pair screener
2. runs diagnostics
3. runs discovery
4. runs portfolio ranking / proposal generation
5. writes fresh artifacts into `results/latest/`

The intraday workflow:

1. refreshes suppression state
2. runs one paper-mode research-agent cycle
3. updates runtime logs used by the dashboard

### 4. Open the dashboard

Serve the repo root or use your existing dashboard workflow, then open:

```text
http://localhost:8787/live_dashboard.html
```

The dashboard reads runtime artifacts such as:

- `runtime/universe/pnl_curve.jsonl`
- `runtime/universe/events.jsonl`
- `results/suppression_state.json`

### 5. Show the latest research artifacts

The most useful files to open during the demo are:

- `results/latest/older60_candidates.json`
- `results/latest/core_candidates.json`
- `results/latest/proposal.json`
- `results/latest/portfolio_candidate_rankings.json`

## How It Works

### Daily Research Pipeline

The research loop is built from several Python modules:

- `older60_pair_screener.py`
  Screens cached Kraken pairs across many strategy family combinations and scores them on full 120d, older 60d, and recent 60d behavior.

- `segment_diagnostics.py`
  Extracts per-pair diagnostics and behavior notes for the screened candidates.

- `pattern_guided_discovery.py`
  Classifies screened candidates into discovery outputs used by the proposal layer.

- `portfolio_candidate_ranker.py`
  Compares non-active candidates against the current live book and ranks them by actual add/replace portfolio impact.

- `registry_proposal.py`
  Writes versioned proposal artifacts showing what the system wants to promote and why.

### Intraday Execution Agent

The intraday agent is implemented in `universe_scanner_agent.py`.

It:

- reads the active research registry
- applies pair-specific strategy configuration from `research_pair_registry.py`
- filters tradability through Kraken market conditions
- respects suppression state from `suppression_state.py`
- runs in paper mode or live mode
- logs runtime events and mark-to-market PnL

### Suppression and Weak-Regime Control

The project explicitly treats weak regimes as a first-class problem.

Instead of pretending one strategy works in all conditions, the system includes:

- pair-level suppression
- portfolio-level defensive states
- runtime notional reduction or new-entry blocking
- research focused on improving the older-60d weakness rather than hiding it

### Trustless Validation

The repo also includes ERC-8004 integration in `erc8004_integration.py`.

This is the trust layer for:

- pre-trade intent signing
- post-trade feedback / reputation logging
- building a more inspectable and accountable trading agent

## Repository Map

Core demo files:

- `research_server.py` - local HTTP backend used by `n8n`
- `research_pair_registry.py` - active and shadow pair definitions
- `universe_scanner_agent.py` - intraday execution agent
- `suppression_state.py` - suppression state machine
- `portfolio_candidate_ranker.py` - portfolio-aware candidate ranking
- `live_dashboard.html` - runtime dashboard
- `dashboard/` - dashboard frontend assets
- `results/latest/` - freshest research artifacts
- `runtime/universe/` - runtime event and PnL logs

## Quick Start

### Demo path

```bash
python research_server.py
cmd /c "npx n8n"
```

Then run the workflows manually in `n8n` and open the dashboard.

### Standalone local agent

```bash
./start_trading.sh --dashboard
```

Or run the agent directly:

```bash
python universe_scanner_agent.py ^
  --mode paper ^
  --reset-paper ^
  --strategy-mode research ^
  --use-research-universe ^
  --max-pos 6 ^
  --poll 60 ^
  --interval 15
```

## Technologies Used

- Python
- pandas
- NumPy
- Flask
- n8n
- HTML / CSS / JavaScript
- pytest
- Kraken API / CLI integration
- ERC-8004
- Base Sepolia
- Anthropic Claude API

## Hackathon Fit

This project is aimed at the following themes:

- AI agents
- algorithmic trading / fintech
- trustless trading infrastructure
- validation and risk systems
- workflow automation

The strongest hackathon framing is:

- Best Trustless Trading Agent
- Best Validation and Trust Model
- Best Risk-Adjusted Return

## Safety and Scope

- Paper trading is the default demo path
- Strategy logic and research artifacts are stored locally and are inspectable
- Promotions are intended to be auditable through proposal artifacts and approval workflows
- This project is an experimental research system, not financial advice
