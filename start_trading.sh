#!/usr/bin/env bash
# ============================================================
# Research Registry Trader - Quick Start
# Runs paper trading on the current live research registry.
# Current core: GIGA, ZEC, FHE, KERNEL, HOUSE, BABY
#
# Usage:
#   ./start_trading.sh             # paper trading (default)
#   ./start_trading.sh --live      # REAL trading (needs API keys)
#   ./start_trading.sh --dashboard # start + open live dashboard
# ============================================================

set -e
cd "$(dirname "$0")"

MODE="paper"
OPEN_DASH=0

for arg in "$@"; do
  case "$arg" in
    --live) MODE="live" ;;
    --dashboard) OPEN_DASH=1 ;;
  esac
done

echo ""
echo "============================================================"
echo "  Research Registry Trading Agent"
echo "  Pairs: GIGA,ZEC,FHE,KERNEL,HOUSE,BABY | Mode: $MODE"
echo "============================================================"
echo ""

if [ "$OPEN_DASH" -eq 1 ]; then
  echo "Starting live dashboard server on http://localhost:8787 ..."
  python -m http.server 8787 --quiet &
  DASH_PID=$!
  sleep 1
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://localhost:8787/live_dashboard.html" &
  elif command -v open >/dev/null 2>&1; then
    open "http://localhost:8787/live_dashboard.html" &
  else
    echo "Open: http://localhost:8787/live_dashboard.html"
  fi
fi

echo "Starting trading agent (Ctrl+C to stop)..."
echo ""

if [ "$MODE" = "live" ]; then
  echo "WARNING: LIVE MODE - real orders will be placed."
  read -r -p "Type YES to confirm: " CONFIRM
  if [ "$CONFIRM" != "YES" ]; then
    echo "Aborted."
    exit 1
  fi
  python universe_scanner_agent.py \
    --mode live \
    --strategy-mode research \
    --use-research-universe \
    --notional 75 \
    --max-pos 6 \
    --poll 60 \
    --interval 15
else
  python universe_scanner_agent.py \
    --mode paper \
    --reset-paper \
    --paper-balance 10000 \
    --strategy-mode research \
    --use-research-universe \
    --notional 75 \
    --max-pos 6 \
    --poll 60 \
    --interval 15
fi
