# Pi Runbook

This repo now supports persistent runtime logs and restart-safe state so you can verify paper behavior on the Pi before risking real capital.

## 1. Prepare the Pi

Install Python dependencies and make sure the Kraken CLI binary is available through `KRAKEN_BIN` or on `PATH`.

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export KRAKEN_BIN=/path/to/kraken
```

For paper mode, do not add live Kraken credentials.

## 2. Paper Trade First

Reset the paper account, enable persistent logs, and resume-safe state:

```bash
python agent.py \
  --mode paper \
  --pairs GIGAUSD \
  --context-pairs DOGUSD,HYPEUSD \
  --notional-usd 25 \
  --poll 60 \
  --resume-state \
  --reset-paper \
  --paper-init-balance 10000 \
  --log-dir runtime/paper \
  --state-file runtime/paper/state.json
```

What to verify in paper mode:

- `runtime/paper/events.jsonl` shows `entry_placed`, `entry_filled`, `exit_placed`, `exit_filled`, `order_canceled`, and `error` events.
- `runtime/paper/state.json` updates while the agent runs.
- Maker entries time out and cancel cleanly when they do not fill.
- Restarting the process with `--resume-state` restores pending orders and open trades.

Useful checks:

```bash
tail -f runtime/paper/events.jsonl
python -m json.tool runtime/paper/state.json | sed -n '1,120p'
```

## 3. Restart Test on the Pi

While paper mode is running:

1. Stop the process with `Ctrl+C`.
2. Start the same command again with `--resume-state`.
3. Confirm the startup log includes a `state_restored` event.
4. Confirm pending orders and any open trade continue from `runtime/paper/state.json`.

## 4. Live Validation Before Real Orders

Use Kraken validation first. This does not place a real order.

```bash
python agent.py \
  --mode live \
  --pairs GIGAUSD \
  --context-pairs DOGUSD,HYPEUSD \
  --notional-usd 5 \
  --poll 60 \
  --cycles 2 \
  --validate-live-orders \
  --log-dir runtime/live-validate \
  --state-file runtime/live-validate/state.json
```

Check `runtime/live-validate/events.jsonl` for `entry_validation` or `exit_validation`.

## 5. Smallest Real Live Test

Only do this after paper logs, fills, cancels, and restart behavior look correct.

Use the smallest acceptable notional for the pair you are trading:

```bash
python agent.py \
  --mode live \
  --pairs GIGAUSD \
  --context-pairs DOGUSD,HYPEUSD \
  --notional-usd 5 \
  --poll 60 \
  --resume-state \
  --log-dir runtime/live \
  --state-file runtime/live/state.json
```

What to verify before scaling up:

- The live event log shows a maker order placement instead of an immediate taker fill.
- Fills and cancels appear as expected in `events.jsonl`.
- A restart restores state without duplicating orders.
- Real fills, exits, and PnL look consistent with Kraken account activity.

## 6. Scale Up Slowly

Only increase `--notional-usd` after a few real trades have behaved as expected end to end.

Suggested progression:

1. Validation only
2. Smallest acceptable live notional
3. A few more live trades at the same size
4. Increase size gradually after logs and exchange activity match
