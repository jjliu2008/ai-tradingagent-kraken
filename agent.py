"""
Kraken AI Trading Agent — Overextension Short Strategy
======================================================
Adapted from the ES futures live_candidate_v1:
  "Short when price is stretched high in its range AND far above VWAP
   AND buying momentum has cooled enough to suggest the upside is exhausted."

Architecture:
  - strategy.py  →  pure signal logic (rule-based, deterministic)
  - agent.py     →  Claude reasons about each signal before acting

Run:
  python agent.py --pair XBTUSD --mode paper --poll 60
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import anthropic
from dotenv import load_dotenv

import kraken_client as kraken
import strategy as strat

load_dotenv()

TRADE_VOLUME = 0.005   # BTC per trade (≈$330 at ~$67k — small for demo)


# ---------------------------------------------------------------------------
# Claude reasoning layer
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a crypto trading agent running the "overextension short" strategy.

STRATEGY LOGIC (adapted from ES futures live_candidate_v1):
- Short when price is in the TOP 15% of its session range (range_pos >= 0.85)
- AND price is >= 0.25% above session VWAP (buying exhaustion zone)
- AND net buying momentum has COOLED (sv_ratio30 <= 0.50)
- Enter 1 bar after signal (delayed), exit fast if no progress within 3 bars
- Max hold: 30 minutes. Fail-fast: exit if no favorable progress AND buying resumes AND OBI bullish

YOUR ROLE:
When a signal fires, you receive the market state and decide: TRADE or SKIP.
Weigh the mechanical signal against qualitative context (momentum character, OBI skew, etc.).
Be concise — one paragraph of reasoning + a final verdict: TRADE or SKIP.

In SKIP cases, briefly state what would need to change for the signal to be actionable.
In TRADE cases, confirm the key criteria are all aligned.
"""


def ask_claude(
    client: anthropic.Anthropic,
    signal: strat.Signal,
    market_state: str,
    portfolio: dict,
) -> tuple[bool, str]:
    """
    Ask Claude whether to trade on the current signal.
    Returns (should_trade: bool, reasoning: str).
    """
    prompt = f"""\
SIGNAL DETECTED — SHORT {signal.price:,.2f}

{signal.describe()}

MARKET STATE:
{market_state}

PORTFOLIO:
  Current value: ${portfolio.get('current_value', 0):,.2f}
  Starting:      ${portfolio.get('starting_balance', 0):,.2f}
  Unrealized PnL: {portfolio.get('unrealized_pnl_pct', 0):.2%}
  Open orders:   {portfolio.get('open_orders', 0)}

Trade {TRADE_VOLUME} BTC short (sell now, buy back on fade)?
Reply with TRADE or SKIP and your brief reasoning.
"""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    should_trade = "TRADE" in text.upper() and "SKIP" not in text.upper()
    return should_trade, text


def ask_claude_exit(
    client: anthropic.Anthropic,
    trade: strat.Trade,
    exit_reason: str,
    market_state: str,
    current_price: float,
) -> tuple[bool, str]:
    """
    Notify Claude of a pending exit. Always returns True — exit is mechanical,
    Claude's role is to log the reasoning.
    """
    bars_held = (trade.exit_bar or 0) - trade.entry_bar
    pnl = trade.side * (current_price - trade.entry_price) / trade.entry_price
    prompt = f"""\
EXIT triggered: {exit_reason}

Entry: ${trade.entry_price:,.2f}  |  Current: ${current_price:,.2f}
PnL: {pnl:.3%}  |  Best MFE: {trade.best_pct:.3%}  |  Bars held: {bars_held}

MARKET STATE:
{market_state}

Explain in 1-2 sentences why the trade worked or failed.
"""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return True, response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run(pair: str = "XBTUSD", poll_seconds: int = 60, mode: str = "paper"):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    strategy = strat.OverextensionStrategy()

    # Pending entry: signal fires on bar N, entry executes on bar N+1
    pending_signal: Optional[strat.Signal] = None
    trade_log: list[dict] = []
    bar_count = 0

    print(f"\n{'='*60}")
    print(f" Overextension Short Agent | {pair} | mode={mode}")
    print(f"{'='*60}")
    print(" Adapted from ES live_candidate_v1 (short overextension / failed continuation)")
    print(f" Signal: top-15%-of-range AND +0.25%-above-VWAP AND SV-cooled")
    print(f" Entry:  delayed 1 bar | Fail-fast: {strat.FAIL_FAST_BARS} bars | Max hold: {strat.HOLD_BARS} bars")
    print(f"{'='*60}\n")

    while True:
        try:
            # --- Fetch market data ---
            raw_ohlc = kraken.fetch_ohlc(pair)
            raw_ob = kraken.fetch_orderbook(pair)
            portfolio = kraken.paper_status()

            obi, spread_pct = strat.compute_obi(raw_ob)
            df_raw = strat.parse_ohlc(raw_ohlc)

            # Use last 120 bars (2 hours) as session window
            session_len = min(120, len(df_raw))
            session_start = len(df_raw) - session_len
            df = strat.compute_features(df_raw, session_start_bar=session_start)

            current_bar = len(df) - 1
            current_price = float(df.iloc[-1]["close"])
            market_state = strategy.describe_state(df)

            bar_count += 1
            print(f"\n[Bar {bar_count}] {time.strftime('%H:%M:%S')} | ${current_price:,.2f}")

            # --- Execute pending entry (delayed 1 bar) ---
            if pending_signal is not None:
                print(f"  → Entering SHORT @ ${current_price:,.2f} (1-bar delay)")
                if mode == "paper":
                    result = kraken.paper_sell(pair, TRADE_VOLUME)
                    actual_price = float(result.get("price", current_price))
                else:
                    actual_price = current_price  # live mode placeholder
                strategy.open_trade(current_bar, actual_price, TRADE_VOLUME)
                print(f"  ✓ Trade opened: SHORT {TRADE_VOLUME} {pair} @ ${actual_price:,.2f}")
                pending_signal = None

            # --- Check exit for open trade ---
            if strategy.trade and strategy.trade.is_open:
                exit_reason = strategy.check_exit(df, obi)
                if exit_reason:
                    print(f"  → Exit triggered: {exit_reason}")
                    if mode == "paper":
                        result = kraken.paper_buy(pair, TRADE_VOLUME)
                        exit_price = float(result.get("price", current_price))
                    else:
                        exit_price = current_price
                    closed = strategy.close_trade(current_bar, exit_price, exit_reason)
                    _, reasoning = ask_claude_exit(client, closed, exit_reason, market_state, exit_price)
                    pnl_pct = closed.pnl_pct() or 0.0
                    print(f"  ✓ Trade closed @ ${exit_price:,.2f} | PnL={pnl_pct:.3%}")
                    print(f"  Claude: {reasoning}")
                    trade_log.append({
                        "entry": closed.entry_price,
                        "exit": closed.exit_price,
                        "pnl_pct": pnl_pct,
                        "reason": exit_reason,
                        "bars": closed.bars_held(current_bar),
                    })
                else:
                    bars = strategy.trade.bars_held(current_bar)
                    unrealized = strategy.trade.side * (current_price - strategy.trade.entry_price) / strategy.trade.entry_price
                    print(f"  Holding SHORT | bars={bars} | PnL={unrealized:.3%} | MFE={strategy.trade.best_pct:.3%}")

            # --- Detect new signal (only if flat) ---
            elif pending_signal is None:
                signal = strategy.detect(df, obi=obi, spread_pct=spread_pct, session_start_bar=session_start)
                if signal:
                    strategy.record_signal(signal.bar_idx)
                    print(f"\n  *** {signal.describe()}")
                    print(f"  Market state:\n{market_state}")
                    should_trade, reasoning = ask_claude(client, signal, market_state, portfolio)
                    print(f"\n  Claude: {reasoning}")
                    if should_trade:
                        print(f"  → Signal APPROVED. Entering next bar...")
                        pending_signal = signal
                    else:
                        print(f"  → Signal SKIPPED.")
                else:
                    print(f"  No signal. {market_state.splitlines()[0]}")

        except KeyboardInterrupt:
            print("\n\nAgent stopped by user.")
            break
        except Exception as e:
            print(f"  [ERROR] {e}")

        # Print trade log summary every 10 bars
        if bar_count % 10 == 0 and trade_log:
            pnls = [t["pnl_pct"] for t in trade_log]
            print(f"\n  --- Trade log ({len(pnls)} trades) ---")
            print(f"  Win rate: {sum(1 for p in pnls if p > 0) / len(pnls):.0%}")
            print(f"  Avg PnL:  {sum(pnls) / len(pnls):.3%}")

        time.sleep(poll_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overextension Short Agent")
    parser.add_argument("--pair", default="XBTUSD")
    parser.add_argument("--mode", default="paper", choices=["paper", "live"])
    parser.add_argument("--poll", type=int, default=60, help="Seconds between polls")
    args = parser.parse_args()

    if args.mode == "live":
        confirm = input("WARNING: Live mode uses real money. Type 'yes' to confirm: ")
        if confirm.lower() != "yes":
            print("Aborted.")
            exit(0)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Add it to your .env file.")
        exit(1)

    run(pair=args.pair, poll_seconds=args.poll, mode=args.mode)
