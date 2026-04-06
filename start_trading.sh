#!/usr/bin/env bash
# ============================================================
#  Universe Scanner — Quick Start
#  Runs paper trading on GIGAUSD + ZECUSD (proven signal pairs)
#  Shows live mark-to-market P&L every 60 seconds.
#
#  Usage:
#    ./start_trading.sh             # paper trading (default)
#    ./start_trading.sh --live      # REAL trading (needs API keys)
#    ./start_trading.sh --dashboard # start + open live dashboard
# ============================================================

set -e
cd "$(dirname "$0")"

MODE="paper"
OPEN_DASH=0

for arg in "$@"; do
  case $arg in
    --live)     MODE="live" ;;
    --dashboard) OPEN_DASH=1 ;;
  esac
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         Universe Scanner Trading Agent                  ║"
echo "║  Pairs: GIGAUSD + ZECUSD  |  Mode: $MODE               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Run equity curve backtest first to confirm signals are active
echo "▶ Running backtest to confirm signal quality..."
python backtest_equity_curve.py --pairs GIGAUSD,ZECUSD --notional 500 \
  --output results/equity_curve.html 2>&1

echo ""
echo "▶ Backtest chart → results/equity_curve.html"
echo ""

# Open dashboard in background if requested
if [ "$OPEN_DASH" -eq 1 ]; then
  echo "▶ Starting live dashboard server on http://localhost:8787 ..."
  python -m http.server 8787 --quiet &
  DASH_PID=$!
  sleep 1
  # Try to open browser
  if command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:8787/live_dashboard.html" &
  elif command -v open &>/dev/null; then
    open "http://localhost:8787/live_dashboard.html" &
  else
    echo "  Open: http://localhost:8787/live_dashboard.html"
  fi
fi

echo "▶ Starting trading agent (Ctrl+C to stop)..."
echo ""

if [ "$MODE" = "live" ]; then
  echo "⚠️  WARNING: LIVE MODE — real orders will be placed!"
  read -p "   Type YES to confirm: " CONFIRM
  if [ "$CONFIRM" != "YES" ]; then
    echo "Aborted."
    exit 1
  fi
  python universe_scanner_agent.py \
    --mode live \
    --universe GIGAUSD,ZECUSD \
    --notional 150 \
    --poll 60 \
    --interval 60 \
    --min-score 2
else
  python universe_scanner_agent.py \
    --mode paper \
    --reset-paper \
    --paper-balance 10000 \
    --universe GIGAUSD,ZECUSD \
    --notional 150 \
    --poll 60 \
    --interval 60 \
    --min-score 2
fi
