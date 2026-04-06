#!/usr/bin/env bash
# pi_setup.sh — Run this once on the Raspberry Pi after copying the project
set -e

echo "=== Kraken Trading Agent — Pi Setup ==="
echo ""

# Python check
PYVER=$(python3 --version 2>&1)
echo "[1/6] Python: $PYVER"

# Ensure pip
echo "[2/6] Ensuring pip is available..."
if ! python3 -m pip --version &>/dev/null; then
  sudo apt-get install -y python3-pip python3-full
fi

# Install dependencies (Pi may not have pandas/numpy wheels, so may take a while)
echo "[3/6] Installing Python dependencies (may take 2-5 min on Pi)..."
pip3 install -r requirements.txt --break-system-packages --quiet

echo "[4/6] Fixing permissions and line endings..."
chmod +x bin/kraken start_trading.sh
sed -i 's/\r//' bin/kraken start_trading.sh universe_scanner_agent.py

echo "[5/6] Creating runtime directories..."
mkdir -p runtime/universe results logs

echo "[6/6] Verifying Kraken CLI..."
if python3 bin/kraken --help &>/dev/null; then
  echo "  OK — kraken CLI works"
else
  echo "  WARN — kraken CLI returned non-zero (may be fine)"
fi

# Check .env
if [ -f .env ]; then
  if grep -q "ANTHROPIC_API_KEY=sk-" .env; then
    echo ""
    echo "✓ .env found with Anthropic key"
  else
    echo ""
    echo "⚠  .env exists but ANTHROPIC_API_KEY may not be set."
    echo "   Edit .env and add: ANTHROPIC_API_KEY=sk-ant-..."
  fi
else
  echo ""
  echo "⚠  No .env file found. Creating from example..."
  cp .env.example .env
  echo "   Edit .env and add your ANTHROPIC_API_KEY before starting."
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Start paper trading:"
echo "  nohup python3 universe_scanner_agent.py --mode paper --reset-paper --universe GIGAUSD,ZECUSD --notional 150 --poll 60 --interval 60 --min-score 2 > logs/agent.log 2>&1 &"
echo ""
echo "Watch the logs:"
echo "  tail -f logs/agent.log"
echo ""
echo "Watch P&L:"
echo "  watch -n 30 'tail -1 runtime/universe/pnl_curve.jsonl | python3 -c \"import sys,json; d=json.load(sys.stdin); print(f\\\"P&L: \\\${d[\\\"total_pnl_usd\\\"]:+.2f} | Open: {d[\\\"n_open\\\"]} pos | Trades: {len(d.get(\\\"recent_trades\\\",[]))}\\\")\"'"
