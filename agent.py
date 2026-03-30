from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import anthropic
import pandas as pd
from dotenv import load_dotenv

import kraken_client as kraken
import strategy as strat

load_dotenv()

DEFAULT_PAIRS = ["GIGAUSD"]
DEFAULT_CONTEXT_PAIRS = ["DOGUSD", "HYPEUSD"]
DEFAULT_NOTIONAL_USD = 600.0
DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")


SYSTEM_PROMPT = """\
You are the execution filter for a deterministic crypto hourly breakout long strategy.

The rule engine already found a candidate. Your job is to accept or reject it, set size, and choose
whether the exit policy should be "fast" or "standard". Do not invent new trades.

Historical note from local screening:
- The strongest baseline was GIGAUSD on 60m bars.
- Winners tended to have an existing uptrend, compressed volatility, a close above the prior 12-bar high, strong 6-bar momentum, above-average volume, and a close near the bar high.
- Weak breakouts usually had poor close quality, no real volume expansion, or broad risk-off behavior in related meme pairs.

Respond with exactly one JSON object and no markdown:
{
  "action": "TRADE or SKIP",
  "confidence": 0.0 to 1.0,
  "size_mult": 0.0 to 1.5,
  "exit_mode": "fast or standard",
  "reason_tags": ["short_snake_case_tags"]
}

Rules:
- If action is SKIP, size_mult must be 0.
- Use TRADE only when breakout quality, trend context, and fee hurdle are acceptable.
- Prefer SKIP on weak trend, loose closes, weak volume, or broad risk-off context.
- Prefer fast exits for stretched or lower-conviction breakouts.
- Keep reason_tags short and factual.
"""


EXIT_REVIEW_PROMPT = """\
You are reviewing the outcome of a deterministic crypto breakout trade.
Reply in one short sentence explaining whether the exit was appropriate.
"""


@dataclass
class AIDecision:
    action: str
    confidence: float
    size_mult: float
    exit_mode: str
    reason_tags: list[str]
    raw_text: str = ""

    @property
    def should_trade(self) -> bool:
        return self.action == "TRADE" and self.size_mult > 0


def _extract_fill_price(payload: dict, fallback: float) -> float:
    for key in ("price", "avg_price", "fill_price"):
        value = payload.get(key)
        if isinstance(value, (int, float, str)):
            try:
                return float(value)
            except ValueError:
                pass
    return fallback


def _extract_ticker_price(payload: dict, fallback: float) -> float:
    pair_key = next(iter(payload))
    pair_payload = payload[pair_key]
    for key in ("c", "a", "b"):
        value = pair_payload.get(key)
        if isinstance(value, list) and value:
            try:
                return float(value[0])
            except (TypeError, ValueError):
                continue
    return fallback


def _extract_json_object(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text}")
    return match.group(0)


def parse_ai_decision(text: str) -> AIDecision:
    payload = json.loads(_extract_json_object(text))
    action = str(payload.get("action", "SKIP")).upper()
    if action not in {"TRADE", "SKIP"}:
        action = "SKIP"

    confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
    size_mult = max(0.0, min(1.5, float(payload.get("size_mult", 0.0))))
    exit_mode = str(payload.get("exit_mode", "standard")).lower()
    if exit_mode not in {"fast", "standard"}:
        exit_mode = "standard"
    if action == "SKIP":
        size_mult = 0.0

    reason_tags = payload.get("reason_tags", [])
    if not isinstance(reason_tags, list):
        reason_tags = []
    reason_tags = [str(tag) for tag in reason_tags[:5]]

    return AIDecision(
        action=action,
        confidence=confidence,
        size_mult=size_mult,
        exit_mode=exit_mode,
        reason_tags=reason_tags,
        raw_text=text,
    )


def default_decision() -> AIDecision:
    return AIDecision(
        action="TRADE",
        confidence=0.0,
        size_mult=1.0,
        exit_mode="standard",
        reason_tags=["rules_only"],
    )


def build_config(args: argparse.Namespace) -> strat.StrategyConfig:
    return strat.StrategyConfig(
        breakout_window=args.breakout_window,
        min_trend_strength=args.min_trend_strength,
        min_momentum_medium=args.min_momentum_medium,
        max_compression_ratio=args.max_compression_ratio,
        min_close_location=args.min_close_location,
        min_volume_ratio=args.min_volume_ratio,
        min_stop_pct=args.stop_pct,
        target_pct=args.target_pct,
        max_hold_bars=args.hold_bars,
        fast_max_hold_bars=args.fast_hold_bars,
    )


def compute_order_size(price: float, notional_usd: float) -> float:
    if price <= 0 or notional_usd <= 0:
        return 0.0
    return round(notional_usd / price, 8)


def session_label(timestamp: int) -> str:
    hour = pd.Timestamp(timestamp, unit="s", tz="UTC").tz_convert("America/New_York").hour
    if hour < 8:
        return "Asia"
    if hour < 14:
        return "Europe"
    if hour < 21:
        return "US"
    return "Late"


def build_pair_snapshot(
    pair: str,
    df: pd.DataFrame,
    obi: float,
    spread_pct: float,
    interval_minutes: int,
    live_price: float,
) -> dict:
    last = df.iloc[-1]
    return {
        "pair": pair,
        "bar_ts": int(last["ts"]),
        "session": session_label(int(last["ts"])),
        "interval_minutes": interval_minutes,
        "price": round(float(last["close"]), 6),
        "live_price": round(live_price, 6),
        "trend_strength": round(float(last.get("trend_strength", 0.0)), 6),
        "momentum_short": round(float(last.get("momentum_short", 0.0)), 6),
        "momentum_medium": round(float(last.get("momentum_medium", 0.0)), 6),
        "distance_from_vwap": round(float(last.get("distance_from_vwap", 0.0)), 6),
        "breakout_level": round(float(last.get("breakout_level", 0.0)), 6),
        "breakout_pct": round(float(last.get("breakout_pct", 0.0)), 6),
        "compression_ratio": round(float(last.get("compression_ratio", 0.0)), 6),
        "volume_ratio": round(float(last.get("volume_ratio", 0.0)), 4),
        "range_position": round(float(last.get("range_position", 0.0)), 4),
        "close_location": round(float(last.get("close_location", 0.0)), 4),
        "obi": round(obi, 4),
        "spread_pct": round(spread_pct, 6),
    }


def build_ai_snapshot(
    signal: strat.Signal,
    pair_contexts: dict[str, dict],
    portfolio: dict,
    fee_hurdle_pct: float,
    interval_minutes: int,
) -> dict:
    return {
        "strategy": {
            "name": "hourly_breakout_long",
            "interval_minutes": interval_minutes,
            "fee_hurdle_pct": round(fee_hurdle_pct, 6),
        },
        "candidate": {
            "pair": signal.pair,
            "price": round(signal.price, 6),
            "signal_type": signal.signal_type,
            "score": round(signal.score, 4),
            "trend_strength": round(signal.trend_strength, 6),
            "momentum_short": round(signal.momentum_short, 6),
            "momentum_medium": round(signal.momentum_medium, 6),
            "impulse_pct": round(signal.impulse_pct, 6),
            "breakout_pct": round(signal.reclaim_pct, 6),
            "breakout_level": round(signal.trigger_level, 6),
            "support_touch_pct": round(signal.support_touch_pct, 6),
            "distance_from_vwap": round(signal.distance_from_vwap, 6),
            "compression_ratio": round(signal.compression_ratio, 6),
            "close_location": round(signal.close_location, 6),
            "volume_ratio": round(signal.volume_ratio, 4),
            "atr_pct": round(signal.atr_pct, 6),
            "support_price": round(signal.support_price, 6),
        },
        "market_context": pair_contexts,
        "portfolio": {
            "current_value": round(float(portfolio.get("current_value", 0.0)), 2),
            "starting_balance": round(float(portfolio.get("starting_balance", 0.0)), 2),
            "unrealized_pnl_pct": round(float(portfolio.get("unrealized_pnl_pct", 0.0)), 6),
            "open_orders": int(portfolio.get("open_orders", 0)),
            "fee_rate": round(float(portfolio.get("fee_rate", 0.0)), 6),
        },
    }


def ask_claude_signal(client: anthropic.Anthropic, snapshot: dict) -> AIDecision:
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=250,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": json.dumps(snapshot, indent=2),
            }
        ],
    )
    text = response.content[0].text.strip()
    return parse_ai_decision(text)


def ask_claude_exit_review(
    client: anthropic.Anthropic,
    pair: str,
    trade: strat.Trade,
    exit_reason: str,
    pair_snapshot: dict,
) -> str:
    payload = {
        "pair": pair,
        "exit_reason": exit_reason,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "pnl_pct": trade.realized_pnl_pct(),
        "best_pct": trade.best_pct,
        "exit_mode": trade.exit_mode,
        "pair_snapshot": pair_snapshot,
    }
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=120,
        temperature=0,
        system=EXIT_REVIEW_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
    )
    return response.content[0].text.strip()


def run(
    trade_pairs: list[str],
    context_pairs: list[str],
    poll_seconds: int,
    mode: str,
    max_positions: int,
    notional_usd: float,
    use_claude: bool,
    interval_minutes: int,
    slippage_pct: float,
    fee_pct: float,
    cycles: int,
    config: strat.StrategyConfig,
) -> None:
    client: Optional[anthropic.Anthropic] = None
    if use_claude:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required when --use-claude is enabled.")
        client = anthropic.Anthropic(api_key=api_key)

    strategies = {pair: strat.HourlyBreakoutStrategy(pair=pair, config=config) for pair in trade_pairs}
    last_closed_bar_ts: dict[str, Optional[int]] = {pair: None for pair in trade_pairs}
    market_pairs = list(dict.fromkeys(trade_pairs + context_pairs))
    trade_log: list[dict] = []
    cycle = 0

    print(f"\n{'=' * 78}")
    print(f" Hourly Breakout Agent | mode={mode} | interval={interval_minutes}m")
    print(f"{'=' * 78}")
    print(f" Trade pairs: {', '.join(trade_pairs)}")
    print(f" Context pairs: {', '.join(context_pairs) if context_pairs else 'none'}")
    print(f" Max positions: {max_positions} | Notional/trade: ${notional_usd:,.2f}")
    print(
        f" Trend>={config.min_trend_strength:.2%} | "
        f"Mom6>={config.min_momentum_medium:.2%} | "
        f"Compress<={config.max_compression_ratio:.2f} | "
        f"Vol>={config.min_volume_ratio:.2f}x | "
        f"Hold={config.max_hold_bars} bars"
    )
    print(f" Claude filter: {'on' if client else 'off'}")
    print(f"{'=' * 78}\n")

    while True:
        cycle += 1
        try:
            portfolio = kraken.paper_status() if mode == "paper" else {}
            fee_hurdle_pct = 2 * (float(portfolio.get("fee_rate", fee_pct)) + slippage_pct)
            pair_contexts: dict[str, dict] = {}
            pair_frames: dict[str, pd.DataFrame] = {}
            live_prices: dict[str, float] = {}
            candidate_signals: list[tuple[str, strat.Signal]] = []
            processed_any = False

            for pair in market_pairs:
                raw_ohlc = kraken.fetch_ohlc(pair, interval=interval_minutes)
                raw_ob = kraken.fetch_orderbook(pair)
                raw_ticker = kraken.fetch_ticker(pair)

                df_raw = strat.parse_ohlc(raw_ohlc)
                if len(df_raw) <= 1:
                    continue
                df = strat.compute_features(df_raw.iloc[:-1].reset_index(drop=True), config=config)
                obi, spread_pct = strat.compute_orderbook_features(raw_ob)
                live_price = _extract_ticker_price(raw_ticker, float(df.iloc[-1]["close"]))
                pair_frames[pair] = df
                live_prices[pair] = live_price
                pair_contexts[pair] = build_pair_snapshot(
                    pair,
                    df,
                    obi,
                    spread_pct,
                    interval_minutes,
                    live_price,
                )

            for pair, strategy in strategies.items():
                df = pair_frames.get(pair)
                if df is None or df.empty:
                    continue

                last_ts = int(df.iloc[-1]["ts"])
                current_bar = len(df) - 1
                current_price = live_prices.get(pair, float(df.iloc[-1]["close"]))
                pair_snapshot = pair_contexts[pair]

                if strategy.trade and strategy.trade.is_open:
                    exit_reason = strategy.check_live_exit_price(current_price)
                    if exit_reason:
                        if mode == "paper":
                            result = kraken.paper_sell(pair, strategy.trade.size)
                            exit_price = _extract_fill_price(result, current_price)
                        else:
                            exit_price = current_price
                        closed = strategy.close_trade(current_bar, last_ts, exit_price, exit_reason)
                        note = ""
                        if client is not None:
                            note = ask_claude_exit_review(client, pair, closed, exit_reason, pair_snapshot)
                        pnl_pct = closed.realized_pnl_pct() or 0.0
                        trade_log.append(
                            {
                                "pair": pair,
                                "entry": closed.entry_price,
                                "exit": exit_price,
                                "pnl_pct": pnl_pct,
                                "reason": exit_reason,
                            }
                        )
                        print(
                            f"  EXIT {pair} | {exit_reason} | exit={exit_price:.2f} | pnl={pnl_pct:.3%}"
                        )
                        if note:
                            print(f"  Claude: {note}")
                        processed_any = True
                        continue

                if last_closed_bar_ts[pair] == last_ts:
                    continue

                processed_any = True
                last_closed_bar_ts[pair] = last_ts

                print(
                    f"[Cycle {cycle}] {strategies[pair].describe_state(df)} | "
                    f"live={current_price:.6f} | obi={pair_snapshot['obi']:.3f} | "
                    f"spread={pair_snapshot['spread_pct']:.4%}"
                )

                if strategy.trade and strategy.trade.is_open:
                    exit_reason = strategy.check_exit(df)
                    if exit_reason:
                        if mode == "paper":
                            result = kraken.paper_sell(pair, strategy.trade.size)
                            exit_price = _extract_fill_price(result, current_price)
                        else:
                            exit_price = current_price
                        closed = strategy.close_trade(current_bar, last_ts, exit_price, exit_reason)
                        note = ""
                        if client is not None:
                            note = ask_claude_exit_review(client, pair, closed, exit_reason, pair_snapshot)
                        pnl_pct = closed.realized_pnl_pct() or 0.0
                        trade_log.append(
                            {
                                "pair": pair,
                                "entry": closed.entry_price,
                                "exit": exit_price,
                                "pnl_pct": pnl_pct,
                                "reason": exit_reason,
                            }
                        )
                        print(
                            f"  EXIT {pair} | {exit_reason} | exit={exit_price:.6f} | pnl={pnl_pct:.3%}"
                        )
                        if note:
                            print(f"  Claude: {note}")
                    continue

                signal = strategy.detect(df)
                if signal is not None:
                    candidate_signals.append((pair, signal))
                    print(f"  Candidate: {signal.describe()}")

            if not processed_any:
                print(f"[Cycle {cycle}] Waiting for next closed {interval_minutes}m bar...")
            else:
                open_positions = sum(
                    1 for strategy in strategies.values() if strategy.trade and strategy.trade.is_open
                )
                if candidate_signals and open_positions < max_positions:
                    slots = max_positions - open_positions
                    ranked = sorted(candidate_signals, key=lambda item: item[1].score, reverse=True)
                    for pair, signal in ranked[:slots]:
                        decision = default_decision()
                        if client is not None:
                            snapshot = build_ai_snapshot(
                                signal=signal,
                                pair_contexts=pair_contexts,
                                portfolio=portfolio,
                                fee_hurdle_pct=fee_hurdle_pct,
                                interval_minutes=interval_minutes,
                            )
                            try:
                                decision = ask_claude_signal(client, snapshot)
                            except Exception as exc:
                                print(f"  AI parse failure for {pair}: {exc}")
                                decision = AIDecision(
                                    action="SKIP",
                                    confidence=0.0,
                                    size_mult=0.0,
                                    exit_mode="fast",
                                    reason_tags=["ai_error"],
                                )

                        print(
                            f"  Decision {pair}: action={decision.action} | "
                            f"size_mult={decision.size_mult:.2f} | exit_mode={decision.exit_mode} | "
                            f"tags={','.join(decision.reason_tags) if decision.reason_tags else 'none'}"
                        )

                        if not decision.should_trade:
                            continue

                        size = compute_order_size(signal.price, notional_usd * decision.size_mult)
                        if size <= 0:
                            print(f"  SKIP {pair} | invalid order size")
                            continue

                        if mode == "paper":
                            result = kraken.paper_buy(pair, size)
                            entry_price = _extract_fill_price(result, signal.price)
                        else:
                            entry_price = signal.price

                        strategies[pair].open_trade(
                            signal,
                            size=size,
                            entry_price=entry_price,
                            exit_mode=decision.exit_mode,
                            ai_confidence=decision.confidence,
                        )
                        print(
                            f"  BUY  {pair} | size={size:.8f} | entry={entry_price:.6f} | "
                            f"score={signal.score:.2f}"
                        )

            if cycle % 10 == 0 and trade_log:
                pnls = [trade["pnl_pct"] for trade in trade_log]
                win_rate = sum(1 for pnl in pnls if pnl > 0) / len(pnls)
                print(
                    f"\nSummary after {cycle} cycles | trades={len(pnls)} | "
                    f"win_rate={win_rate:.1%} | avg_pnl={sum(pnls) / len(pnls):.3%}\n"
                )

        except KeyboardInterrupt:
            print("\nAgent stopped by user.")
            break
        except Exception as exc:
            print(f"[ERROR] {exc}")

        if cycles and cycle >= cycles:
            print("Reached requested cycle limit.")
            break

        time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Kraken hourly breakout agent")
    parser.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    parser.add_argument("--context-pairs", default=",".join(DEFAULT_CONTEXT_PAIRS))
    parser.add_argument("--mode", default="paper", choices=["paper", "live"])
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--cycles", type=int, default=0, help="0 means run forever")
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument("--notional-usd", type=float, default=DEFAULT_NOTIONAL_USD)
    parser.add_argument("--use-claude", action="store_true")
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--fee-pct", type=float, default=0.0026)
    parser.add_argument("--breakout-window", type=int, default=strat.DEFAULT_CONFIG.breakout_window)
    parser.add_argument("--min-trend-strength", type=float, default=strat.DEFAULT_CONFIG.min_trend_strength)
    parser.add_argument("--min-momentum-medium", type=float, default=strat.DEFAULT_CONFIG.min_momentum_medium)
    parser.add_argument("--min-volume-ratio", type=float, default=strat.DEFAULT_CONFIG.min_volume_ratio)
    parser.add_argument("--max-compression-ratio", type=float, default=strat.DEFAULT_CONFIG.max_compression_ratio)
    parser.add_argument("--min-close-location", type=float, default=strat.DEFAULT_CONFIG.min_close_location)
    parser.add_argument("--stop-pct", type=float, default=strat.DEFAULT_CONFIG.min_stop_pct)
    parser.add_argument("--target-pct", type=float, default=strat.DEFAULT_CONFIG.target_pct)
    parser.add_argument("--hold-bars", type=int, default=strat.DEFAULT_CONFIG.max_hold_bars)
    parser.add_argument("--fast-hold-bars", type=int, default=strat.DEFAULT_CONFIG.fast_max_hold_bars)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trade_pairs = [pair.strip() for pair in args.pairs.split(",") if pair.strip()]
    context_pairs = [pair.strip() for pair in args.context_pairs.split(",") if pair.strip()]
    config = build_config(args)

    if args.mode == "live":
        confirm = input("WARNING: live mode can place real orders. Type 'yes' to continue: ")
        if confirm.strip().lower() != "yes":
            raise SystemExit(0)

    run(
        trade_pairs=trade_pairs,
        context_pairs=context_pairs,
        poll_seconds=args.poll,
        mode=args.mode,
        max_positions=args.max_positions,
        notional_usd=args.notional_usd,
        use_claude=args.use_claude,
        interval_minutes=args.interval,
        slippage_pct=args.slippage_pct,
        fee_pct=args.fee_pct,
        cycles=args.cycles,
        config=config,
    )
