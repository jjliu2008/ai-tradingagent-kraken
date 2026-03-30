# Kraken AI Trading Agent — Claude Code Context

This is an AI trading agent built for the LabLab AI Trading Agents Hackathon.

## Project Overview

An autonomous trading agent that uses the Kraken CLI and Claude API to analyze markets and execute trades on Kraken exchange. Uses paper trading for safe strategy iteration.

## Kraken CLI Integration

The `kraken` binary is installed at `~/.cargo/bin/kraken`. Always invoke with `-o json` and redirect stderr:

```bash
kraken <command> [args...] -o json 2>/dev/null
```

**Authentication**: Set env vars `KRAKEN_API_KEY` and `KRAKEN_API_SECRET` (not needed for paper trading or public market data).

**Paper trading** (no API key needed — uses live prices):
```bash
kraken paper init --balance 10000 -o json
kraken paper buy XBTUSD 0.01 -o json
kraken paper sell XBTUSD 0.01 -o json
kraken paper status -o json
```

**Market data** (public, no auth):
```bash
kraken market ticker XBTUSD -o json
kraken market ohlc XBTUSD --interval 60 -o json
kraken market orderbook XBTUSD -o json
```

**Account** (requires API key):
```bash
kraken account balance -o json
kraken account positions -o json
```

## Safety Rules

1. Never place real orders without explicit user confirmation
2. Use `--validate` to dry-run orders before executing
3. Default to paper trading for all strategy development
4. Never log or display API secrets

## Project Structure

```
agent.py          — Main trading agent entry point
strategy.py       — Trading strategy logic
kraken_client.py  — Kraken CLI wrapper functions
requirements.txt  — Python dependencies
.env.example      — Environment variable template
```

## Environment Variables

```
KRAKEN_API_KEY=your_read_only_key
KRAKEN_API_SECRET=your_secret
ANTHROPIC_API_KEY=your_claude_key
```

## Running

```bash
# Paper trading (safe, no API key needed)
python agent.py --mode paper

# Live trading (requires API keys)
python agent.py --mode live
```
