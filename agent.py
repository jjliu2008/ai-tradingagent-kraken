from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
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
DEFAULT_LOG_DIR = "runtime"
DEFAULT_STATE_FILE = "runtime/agent_state.json"
STOPLIKE_EXIT_REASONS = {"STOP_LOSS", "TREND_LOST"}


SYSTEM_PROMPT = """\
You are the execution filter for a deterministic crypto 15-minute momentum-confluence long strategy.

The rule engine already found a candidate. Your job is to accept or reject it, set size, and choose
whether the exit policy should be "fast" or "standard". Do not invent new trades.

Historical note from local screening:
- The live baseline is GIGAUSD on a 15m execution frame.
- The current default construction is `tc15_tighter_volume_cap`, a 15m triple-confluence breakout that skips the most extreme volume spikes.
- Higher-quality trades usually had trend already up, MACD histogram rising, compressed volatility, above-average volume, and a close near the bar high.
- Weak trades usually had loose closes, weak volume, failed follow-through, or were already too extended relative to VWAP.

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
You are reviewing the outcome of a deterministic crypto momentum-confluence trade.
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


@dataclass
class PendingOrder:
    pair: str
    order_id: str
    purpose: str
    side: str
    size: float
    price: float
    created_at: float
    signal: Optional[strat.Signal] = None
    exit_reason: str = ""
    exit_mode: str = "standard"
    ai_confidence: float = 0.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def append_event(log_path: Path, event_type: str, **payload: object) -> None:
    _ensure_parent(log_path)
    record = {
        "ts": _utc_now_iso(),
        "event": event_type,
        **payload,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")


def _serialize_signal(signal: strat.Signal | None) -> dict | None:
    if signal is None:
        return None
    return asdict(signal)


def _deserialize_signal(payload: dict | None) -> Optional[strat.Signal]:
    if not payload:
        return None
    return strat.Signal(**payload)


def _serialize_trade(trade: strat.Trade | None) -> dict | None:
    if trade is None:
        return None
    return asdict(trade)


def _deserialize_trade(payload: dict | None) -> Optional[strat.Trade]:
    if not payload:
        return None
    return strat.Trade(**payload)


def _serialize_pending_order(order: PendingOrder) -> dict:
    payload = asdict(order)
    payload["signal"] = _serialize_signal(order.signal)
    return payload


def _deserialize_pending_order(payload: dict) -> PendingOrder:
    order_payload = dict(payload)
    order_payload["signal"] = _deserialize_signal(order_payload.get("signal"))
    return PendingOrder(**order_payload)


def load_runtime_state(
    state_path: Path,
    strategies: dict[str, strat.HourlyBreakoutStrategy],
    pending_orders: dict[str, PendingOrder],
    last_closed_bar_ts: dict[str, Optional[int]],
) -> tuple[int, list[dict]]:
    if not state_path.exists():
        return 0, []

    with state_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    for pair, strategy_state in payload.get("strategies", {}).items():
        strategy = strategies.get(pair)
        if strategy is None:
            continue
        strategy.last_signal_bar = int(strategy_state.get("last_signal_bar", strategy.last_signal_bar))
        strategy.trade = _deserialize_trade(strategy_state.get("trade"))

    pending_orders.clear()
    for pair, order_payload in payload.get("pending_orders", {}).items():
        if pair in strategies:
            pending_orders[pair] = _deserialize_pending_order(order_payload)

    for pair, value in payload.get("last_closed_bar_ts", {}).items():
        if pair in last_closed_bar_ts:
            last_closed_bar_ts[pair] = value

    return int(payload.get("cycle", 0)), list(payload.get("trade_log", []))


def save_runtime_state(
    state_path: Path,
    cycle: int,
    strategies: dict[str, strat.HourlyBreakoutStrategy],
    pending_orders: dict[str, PendingOrder],
    last_closed_bar_ts: dict[str, Optional[int]],
    trade_log: list[dict],
) -> None:
    payload = {
        "saved_at": _utc_now_iso(),
        "cycle": cycle,
        "strategies": {
            pair: {
                "last_signal_bar": strategy.last_signal_bar,
                "trade": _serialize_trade(strategy.trade),
            }
            for pair, strategy in strategies.items()
        },
        "pending_orders": {
            pair: _serialize_pending_order(order)
            for pair, order in pending_orders.items()
        },
        "last_closed_bar_ts": last_closed_bar_ts,
        "trade_log": trade_log,
    }
    _ensure_parent(state_path)
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


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


def _format_price(price: float) -> str:
    return f"{price:.6f}" if abs(price) < 1 else f"{price:.2f}"


def _extract_order_id(payload: dict) -> str:
    for key in ("order_id", "id"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("txid", "txids"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            return str(value[0])
        if isinstance(value, str) and value:
            return value
    raise ValueError(f"No order identifier found in payload: {payload}")


def _extract_best_bid_ask(orderbook: dict) -> tuple[float, float]:
    pair_key = next(iter(orderbook))
    payload = orderbook[pair_key]
    best_bid = float(payload["bids"][0][0]) if payload.get("bids") else 0.0
    best_ask = float(payload["asks"][0][0]) if payload.get("asks") else 0.0
    return best_bid, best_ask


def _normalize_open_orders(mode: str, payload: dict) -> dict[str, dict]:
    if mode == "paper":
        orders = payload.get("open_orders", [])
        return {
            str(order.get("id") or order.get("order_id")): order
            for order in orders
            if order.get("id") or order.get("order_id")
        }

    for key in ("open", "open_orders", "orders"):
        value = payload.get(key)
        if isinstance(value, dict):
            return {str(order_id): order for order_id, order in value.items() if isinstance(order, dict)}
        if isinstance(value, list):
            return {
                str(order.get("id") or order.get("order_id") or order.get("txid")): order
                for order in value
                if isinstance(order, dict) and (order.get("id") or order.get("order_id") or order.get("txid"))
            }

    return {str(order_id): order for order_id, order in payload.items() if isinstance(order, dict)}


def _normalize_paper_fills(payload: dict) -> dict[str, dict]:
    fills: dict[str, dict] = {}
    for trade in payload.get("trades", []):
        order_id = trade.get("order_id") or trade.get("id")
        if order_id:
            fills[str(order_id)] = trade
    return fills


def _normalize_query_orders(payload: dict) -> dict[str, dict]:
    for key in ("orders", "result"):
        value = payload.get(key)
        if isinstance(value, dict):
            return {str(order_id): order for order_id, order in value.items() if isinstance(order, dict)}
    return {str(order_id): order for order_id, order in payload.items() if isinstance(order, dict)}


def _extract_query_fill(order_info: dict) -> tuple[bool, float | None]:
    status = str(order_info.get("status", "")).lower()
    vol_exec = order_info.get("vol_exec") or order_info.get("filled")
    try:
        filled_qty = float(vol_exec) if vol_exec is not None else 0.0
    except (TypeError, ValueError):
        filled_qty = 0.0

    price_fields = [
        order_info.get("avg_price"),
        order_info.get("avgprc"),
        order_info.get("price"),
    ]
    descr = order_info.get("descr")
    if isinstance(descr, dict):
        price_fields.append(descr.get("price"))

    fill_price = None
    for value in price_fields:
        try:
            if value is not None:
                fill_price = float(value)
                break
        except (TypeError, ValueError):
            continue

    filled = status in {"closed", "filled"} or filled_qty > 0
    return filled, fill_price


def _extract_fill_timestamp(payload: dict, fallback: int) -> int:
    for key in ("time", "closetm", "closed_at"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            if isinstance(value, (int, float)):
                return int(float(value))
            return int(pd.Timestamp(value).timestamp())
        except Exception:
            continue
    return fallback


def _maker_entry_price(best_bid: float, best_ask: float, fallback: float) -> float:
    if best_bid > 0:
        return best_bid
    if best_ask > 0:
        return best_ask * 0.999
    return fallback


def _maker_exit_price(best_bid: float, best_ask: float, fallback: float) -> float:
    if best_ask > 0:
        return best_ask
    if best_bid > 0:
        return best_bid * 1.001
    return fallback


def place_maker_limit(
    mode: str,
    side: str,
    pair: str,
    size: float,
    price: float,
    validate: bool = False,
) -> dict:
    if mode == "paper":
        if side == "buy":
            return kraken.paper_buy(pair, size, order_type="limit", price=price)
        return kraken.paper_sell(pair, size, order_type="limit", price=price)
    if side == "buy":
        return kraken.order_buy(pair, size, order_type="limit", price=price, post_only=True, validate=validate)
    return kraken.order_sell(pair, size, order_type="limit", price=price, post_only=True, validate=validate)


def place_market_exit(mode: str, pair: str, size: float, validate: bool = False) -> dict:
    if mode == "paper":
        return kraken.paper_sell(pair, size, order_type="market")
    return kraken.order_sell(pair, size, order_type="market", validate=validate)


def cancel_pending_order(mode: str, order_id: str) -> dict:
    if mode == "paper":
        return kraken.paper_cancel(order_id)
    return kraken.order_cancel(order_id)


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
    construction: str,
) -> dict:
    return {
        "strategy": {
            "name": construction,
            "interval_minutes": interval_minutes,
            "fee_hurdle_pct": round(fee_hurdle_pct, 6),
        },
        "candidate": {
            "pair": signal.pair,
            "price": round(signal.price, 6),
            "signal_type": signal.signal_type,
            "component_tags": list(signal.component_tags),
            "score": round(signal.score, 4),
            "trend_strength": round(signal.trend_strength, 6),
            "gate_trend_strength": round(signal.gate_trend_strength, 6),
            "gate_momentum_medium": round(signal.gate_momentum_medium, 6),
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
    maker_entry_timeout_sec: int,
    maker_exit_timeout_sec: int,
    config: strat.StrategyConfig,
    log_dir: Path,
    state_path: Path,
    resume_state: bool,
    paper_init_balance: float,
    reset_paper: bool,
    validate_live_orders: bool,
    construction: str,
) -> None:
    if interval_minutes != strat.MASTER_INTERVAL_MINUTES:
        raise RuntimeError(
            f"The ensemble runner requires --interval {strat.MASTER_INTERVAL_MINUTES}."
        )
    if construction not in strat.ensemble_construction_names():
        supported = ", ".join(strat.ensemble_construction_names())
        raise RuntimeError(f"Unknown construction '{construction}'. Supported: {supported}")

    client: Optional[anthropic.Anthropic] = None
    if use_claude:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required when --use-claude is enabled.")
        client = anthropic.Anthropic(api_key=api_key)

    if mode == "live":
        missing = [name for name in ("KRAKEN_API_KEY", "KRAKEN_API_SECRET") if not os.environ.get(name)]
        if missing:
            raise RuntimeError(f"Missing required live trading credentials: {', '.join(missing)}")

    event_log_path = log_dir / "events.jsonl"
    summary_log_path = log_dir / "summary.jsonl"

    strategies = {
        pair: strat.TrendGateEnsembleStrategy(pair=pair, config=config, construction=construction)
        for pair in trade_pairs
    }
    last_closed_bar_ts: dict[str, Optional[int]] = {pair: None for pair in trade_pairs}
    market_pairs = list(dict.fromkeys(trade_pairs + context_pairs))
    pending_orders: dict[str, PendingOrder] = {}
    cycle, trade_log = (0, [])

    if mode == "paper" and reset_paper:
        paper_status = kraken.paper_init(balance=paper_init_balance)
        append_event(
            event_log_path,
            "paper_init",
            balance=paper_init_balance,
            result=paper_status,
        )

    if resume_state:
        cycle, trade_log = load_runtime_state(state_path, strategies, pending_orders, last_closed_bar_ts)
        append_event(
            event_log_path,
            "state_restored",
            state_file=str(state_path),
            restored_cycle=cycle,
            restored_pending_orders=len(pending_orders),
            restored_open_trades=sum(
                1 for strategy in strategies.values() if strategy.trade and strategy.trade.is_open
            ),
        )

    if validate_live_orders and (
        pending_orders
        or any(strategy.trade and strategy.trade.is_open for strategy in strategies.values())
    ):
        raise RuntimeError("Validate-only mode requires no restored open trades or pending orders.")

    def persist_state() -> None:
        save_runtime_state(
            state_path=state_path,
            cycle=cycle,
            strategies=strategies,
            pending_orders=pending_orders,
            last_closed_bar_ts=last_closed_bar_ts,
            trade_log=trade_log,
        )

    print(f"\n{'=' * 78}")
    print(f" Ensemble Agent | mode={mode} | interval={interval_minutes}m")
    print(f"{'=' * 78}")
    print(f" Trade pairs: {', '.join(trade_pairs)}")
    print(f" Context pairs: {', '.join(context_pairs) if context_pairs else 'none'}")
    print(f" Max positions: {max_positions} | Notional/trade: ${notional_usd:,.2f}")
    print(
        f" Maker entry timeout: {maker_entry_timeout_sec}s | "
        f"Maker exit timeout: {maker_exit_timeout_sec}s"
    )
    print(
        f" Entry model: {construction} | "
        f"Stop={config.min_stop_pct:.2%} | Target={config.target_pct:.2%} | "
        f"Hold={config.max_hold_bars} bars"
    )
    print(f" Claude filter: {'on' if client else 'off'}")
    print(f" Event log: {event_log_path}")
    print(f" State file: {state_path} | Resume: {'on' if resume_state else 'off'}")
    print(f" Live validate-only: {'on' if validate_live_orders else 'off'}")
    print(f"{'=' * 78}\n")

    append_event(
        event_log_path,
        "agent_started",
        mode=mode,
        trade_pairs=trade_pairs,
        context_pairs=context_pairs,
        poll_seconds=poll_seconds,
        interval_minutes=interval_minutes,
        construction=construction,
        max_positions=max_positions,
        notional_usd=notional_usd,
        validate_live_orders=validate_live_orders,
        state_file=str(state_path),
        resumed=resume_state,
    )
    persist_state()

    while True:
        cycle += 1
        try:
            portfolio = kraken.paper_status() if mode == "paper" else {}
            fee_hurdle_pct = 2 * (float(portfolio.get("fee_rate", fee_pct)) + slippage_pct)
            pair_contexts: dict[str, dict] = {}
            pair_frames: dict[str, pd.DataFrame] = {}
            live_prices: dict[str, float] = {}
            best_quotes: dict[str, tuple[float, float]] = {}
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
                best_quotes[pair] = _extract_best_bid_ask(raw_ob)
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

            open_order_map: dict[str, dict] = {}
            paper_fills: dict[str, dict] = {}
            live_query_orders: dict[str, dict] = {}
            if pending_orders:
                if mode == "paper":
                    open_order_map = _normalize_open_orders(mode, kraken.paper_orders())
                    paper_fills = _normalize_paper_fills(kraken.paper_history())
                else:
                    open_order_map = _normalize_open_orders(mode, kraken.open_orders())
                    live_query_orders = _normalize_query_orders(
                        kraken.query_orders([pending.order_id for pending in pending_orders.values()])
                    )

            for pair, strategy in strategies.items():
                df = pair_frames.get(pair)
                if df is None or df.empty:
                    continue

                last_ts = int(df.iloc[-1]["ts"])
                current_bar = len(df) - 1
                current_price = live_prices.get(pair, float(df.iloc[-1]["close"]))
                pair_snapshot = pair_contexts[pair]
                best_bid, best_ask = best_quotes.get(pair, (0.0, 0.0))
                pending = pending_orders.get(pair)

                if pending is not None:
                    processed_any = True
                    live_exit_reason = (
                        strategy.check_live_exit_price(current_price)
                        if strategy.trade and strategy.trade.is_open
                        else None
                    )
                    age_seconds = int(time.time() - pending.created_at)
                    open_info = open_order_map.get(pending.order_id)

                    if open_info is not None:
                        print(
                            f"[Cycle {cycle}] Pending {pending.purpose} {pending.side} {pair} | "
                            f"id={pending.order_id} | limit={_format_price(pending.price)} | age={age_seconds}s"
                        )
                        append_event(
                            event_log_path,
                            "pending_open",
                            cycle=cycle,
                            pair=pair,
                            purpose=pending.purpose,
                            side=pending.side,
                            order_id=pending.order_id,
                            limit_price=pending.price,
                            age_seconds=age_seconds,
                        )
                        if (
                            pending.purpose == "exit"
                            and live_exit_reason in STOPLIKE_EXIT_REASONS
                            and strategy.trade is not None
                        ):
                            cancel_pending_order(mode, pending.order_id)
                            result = place_market_exit(mode, pair, strategy.trade.size)
                            exit_price = _extract_fill_price(result, current_price)
                            closed = strategy.close_trade(
                                current_bar,
                                int(time.time()),
                                exit_price,
                                live_exit_reason,
                            )
                            note = ""
                            if client is not None:
                                note = ask_claude_exit_review(client, pair, closed, live_exit_reason, pair_snapshot)
                            pnl_pct = closed.realized_pnl_pct() or 0.0
                            trade_log.append(
                                {
                                    "pair": pair,
                                    "entry": closed.entry_price,
                                    "exit": exit_price,
                                    "pnl_pct": pnl_pct,
                                    "reason": live_exit_reason,
                                }
                            )
                            pending_orders.pop(pair, None)
                            append_event(
                                event_log_path,
                                "exit_market_fallback",
                                cycle=cycle,
                                pair=pair,
                                order_id=pending.order_id,
                                reason=live_exit_reason,
                                exit_price=exit_price,
                                pnl_pct=pnl_pct,
                            )
                            persist_state()
                            print(
                                f"  EXIT {pair} | {live_exit_reason} market fallback | "
                                f"exit={_format_price(exit_price)} | pnl={pnl_pct:.3%}"
                            )
                            if note:
                                print(f"  Claude: {note}")
                            continue

                        timeout_sec = (
                            maker_entry_timeout_sec if pending.purpose == "entry" else maker_exit_timeout_sec
                        )
                        if age_seconds >= timeout_sec:
                            cancel_pending_order(mode, pending.order_id)
                            pending_orders.pop(pair, None)
                            append_event(
                                event_log_path,
                                "order_canceled",
                                cycle=cycle,
                                pair=pair,
                                purpose=pending.purpose,
                                side=pending.side,
                                order_id=pending.order_id,
                                age_seconds=age_seconds,
                                reason="timeout",
                            )
                            if pending.purpose == "exit" and strategy.trade is not None:
                                result = place_market_exit(mode, pair, strategy.trade.size)
                                exit_price = _extract_fill_price(result, current_price)
                                closed = strategy.close_trade(
                                    current_bar,
                                    int(time.time()),
                                    exit_price,
                                    pending.exit_reason,
                                )
                                note = ""
                                if client is not None:
                                    note = ask_claude_exit_review(client, pair, closed, pending.exit_reason, pair_snapshot)
                                pnl_pct = closed.realized_pnl_pct() or 0.0
                                trade_log.append(
                                    {
                                        "pair": pair,
                                        "entry": closed.entry_price,
                                        "exit": exit_price,
                                        "pnl_pct": pnl_pct,
                                        "reason": pending.exit_reason,
                                    }
                                )
                                print(
                                    f"  EXIT {pair} | {pending.exit_reason} market timeout | "
                                    f"exit={_format_price(exit_price)} | pnl={pnl_pct:.3%}"
                                )
                                append_event(
                                    event_log_path,
                                    "exit_market_timeout",
                                    cycle=cycle,
                                    pair=pair,
                                    order_id=pending.order_id,
                                    reason=pending.exit_reason,
                                    exit_price=exit_price,
                                    pnl_pct=pnl_pct,
                                )
                                if note:
                                    print(f"  Claude: {note}")
                            else:
                                print(f"  CANCEL {pair} | stale maker entry {pending.order_id}")
                            persist_state()
                            continue

                        continue

                    fill_price: float | None = None
                    fill_ts = int(time.time())
                    filled = False

                    if mode == "paper":
                        fill_info = paper_fills.get(pending.order_id)
                        if fill_info is not None:
                            filled = True
                            fill_price = _extract_fill_price(fill_info, pending.price)
                            fill_ts = _extract_fill_timestamp(fill_info, fill_ts)
                    else:
                        query_info = live_query_orders.get(pending.order_id)
                        if query_info is not None:
                            filled, fill_price = _extract_query_fill(query_info)
                            fill_ts = _extract_fill_timestamp(query_info, fill_ts)

                    pending_orders.pop(pair, None)

                    if filled:
                        if pending.purpose == "entry" and pending.signal is not None:
                            strategy.open_trade(
                                pending.signal,
                                size=pending.size,
                                entry_price=fill_price or pending.price,
                                exit_mode=pending.exit_mode,
                                ai_confidence=pending.ai_confidence,
                                entry_bar=current_bar,
                                entry_ts=fill_ts,
                            )
                            print(
                                f"  ENTRY FILLED {pair} | id={pending.order_id} | "
                                f"price={_format_price(fill_price or pending.price)} | size={pending.size:.8f}"
                            )
                            append_event(
                                event_log_path,
                                "entry_filled",
                                cycle=cycle,
                                pair=pair,
                                order_id=pending.order_id,
                                fill_price=fill_price or pending.price,
                                size=pending.size,
                            )
                            persist_state()
                        elif pending.purpose == "exit" and strategy.trade is not None:
                            exit_price = fill_price or current_price
                            closed = strategy.close_trade(current_bar, fill_ts, exit_price, pending.exit_reason)
                            note = ""
                            if client is not None:
                                note = ask_claude_exit_review(client, pair, closed, pending.exit_reason, pair_snapshot)
                            pnl_pct = closed.realized_pnl_pct() or 0.0
                            trade_log.append(
                                {
                                    "pair": pair,
                                    "entry": closed.entry_price,
                                    "exit": exit_price,
                                    "pnl_pct": pnl_pct,
                                    "reason": pending.exit_reason,
                                }
                            )
                            print(
                                f"  EXIT FILLED {pair} | {pending.exit_reason} | "
                                f"price={_format_price(exit_price)} | pnl={pnl_pct:.3%}"
                            )
                            append_event(
                                event_log_path,
                                "exit_filled",
                                cycle=cycle,
                                pair=pair,
                                order_id=pending.order_id,
                                reason=pending.exit_reason,
                                exit_price=exit_price,
                                pnl_pct=pnl_pct,
                            )
                            persist_state()
                            if note:
                                print(f"  Claude: {note}")
                    else:
                        print(f"  CLEAR {pair} | maker order {pending.order_id} no longer open and not filled")
                        append_event(
                            event_log_path,
                            "order_cleared",
                            cycle=cycle,
                            pair=pair,
                            order_id=pending.order_id,
                            purpose=pending.purpose,
                        )
                        persist_state()
                    continue

                if strategy.trade and strategy.trade.is_open:
                    exit_reason = strategy.check_live_exit_price(current_price)
                    if exit_reason:
                        if exit_reason in STOPLIKE_EXIT_REASONS:
                            if mode == "live" and validate_live_orders:
                                result = place_market_exit(mode, pair, strategy.trade.size, validate=True)
                                append_event(
                                    event_log_path,
                                    "exit_validation",
                                    cycle=cycle,
                                    pair=pair,
                                    side="sell",
                                    order_type="market",
                                    size=strategy.trade.size,
                                    reason=exit_reason,
                                    result=result,
                                )
                                print(f"  VALIDATE EXIT {pair} | {exit_reason} market")
                                continue
                            result = place_market_exit(mode, pair, strategy.trade.size)
                            exit_price = _extract_fill_price(result, current_price)
                            closed = strategy.close_trade(current_bar, int(time.time()), exit_price, exit_reason)
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
                                f"  EXIT {pair} | {exit_reason} market | "
                                f"exit={_format_price(exit_price)} | pnl={pnl_pct:.3%}"
                            )
                            append_event(
                                event_log_path,
                                "exit_market",
                                cycle=cycle,
                                pair=pair,
                                reason=exit_reason,
                                exit_price=exit_price,
                                pnl_pct=pnl_pct,
                            )
                            persist_state()
                            if note:
                                print(f"  Claude: {note}")
                        else:
                            limit_price = _maker_exit_price(best_bid, best_ask, current_price)
                            result = place_maker_limit(
                                mode,
                                "sell",
                                pair,
                                strategy.trade.size,
                                limit_price,
                                validate=mode == "live" and validate_live_orders,
                            )
                            if mode == "live" and validate_live_orders:
                                append_event(
                                    event_log_path,
                                    "exit_validation",
                                    cycle=cycle,
                                    pair=pair,
                                    side="sell",
                                    order_type="limit",
                                    size=strategy.trade.size,
                                    limit_price=limit_price,
                                    reason=exit_reason,
                                    result=result,
                                )
                                print(
                                    f"  VALIDATE EXIT {pair} | {exit_reason} | "
                                    f"limit={_format_price(limit_price)}"
                                )
                                continue
                            order_id = _extract_order_id(result)
                            pending_orders[pair] = PendingOrder(
                                pair=pair,
                                order_id=order_id,
                                purpose="exit",
                                side="sell",
                                size=strategy.trade.size,
                                price=limit_price,
                                created_at=time.time(),
                                exit_reason=exit_reason,
                            )
                            print(
                                f"  PLACE EXIT {pair} | {exit_reason} | "
                                f"id={order_id} | limit={_format_price(limit_price)}"
                            )
                            append_event(
                                event_log_path,
                                "exit_placed",
                                cycle=cycle,
                                pair=pair,
                                order_id=order_id,
                                limit_price=limit_price,
                                size=strategy.trade.size,
                                reason=exit_reason,
                            )
                            persist_state()
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
                        if exit_reason in STOPLIKE_EXIT_REASONS:
                            if mode == "live" and validate_live_orders:
                                result = place_market_exit(mode, pair, strategy.trade.size, validate=True)
                                append_event(
                                    event_log_path,
                                    "exit_validation",
                                    cycle=cycle,
                                    pair=pair,
                                    side="sell",
                                    order_type="market",
                                    size=strategy.trade.size,
                                    reason=exit_reason,
                                    result=result,
                                )
                                print(f"  VALIDATE EXIT {pair} | {exit_reason} market")
                                continue
                            result = place_market_exit(mode, pair, strategy.trade.size)
                            exit_price = _extract_fill_price(result, current_price)
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
                                f"  EXIT {pair} | {exit_reason} market | "
                                f"exit={_format_price(exit_price)} | pnl={pnl_pct:.3%}"
                            )
                            append_event(
                                event_log_path,
                                "exit_market",
                                cycle=cycle,
                                pair=pair,
                                reason=exit_reason,
                                exit_price=exit_price,
                                pnl_pct=pnl_pct,
                            )
                            persist_state()
                            if note:
                                print(f"  Claude: {note}")
                        else:
                            limit_price = _maker_exit_price(best_bid, best_ask, current_price)
                            result = place_maker_limit(
                                mode,
                                "sell",
                                pair,
                                strategy.trade.size,
                                limit_price,
                                validate=mode == "live" and validate_live_orders,
                            )
                            if mode == "live" and validate_live_orders:
                                append_event(
                                    event_log_path,
                                    "exit_validation",
                                    cycle=cycle,
                                    pair=pair,
                                    side="sell",
                                    order_type="limit",
                                    size=strategy.trade.size,
                                    limit_price=limit_price,
                                    reason=exit_reason,
                                    result=result,
                                )
                                print(
                                    f"  VALIDATE EXIT {pair} | {exit_reason} | "
                                    f"limit={_format_price(limit_price)}"
                                )
                                continue
                            order_id = _extract_order_id(result)
                            pending_orders[pair] = PendingOrder(
                                pair=pair,
                                order_id=order_id,
                                purpose="exit",
                                side="sell",
                                size=strategy.trade.size,
                                price=limit_price,
                                created_at=time.time(),
                                exit_reason=exit_reason,
                            )
                            print(
                                f"  PLACE EXIT {pair} | {exit_reason} | "
                                f"id={order_id} | limit={_format_price(limit_price)}"
                            )
                            append_event(
                                event_log_path,
                                "exit_placed",
                                cycle=cycle,
                                pair=pair,
                                order_id=order_id,
                                limit_price=limit_price,
                                size=strategy.trade.size,
                                reason=exit_reason,
                            )
                            persist_state()
                    continue

                signal = strategy.detect(df)
                if signal is not None:
                    candidate_signals.append((pair, signal))
                    print(f"  Candidate: {signal.describe()}")

            if not processed_any:
                print(f"[Cycle {cycle}] Waiting for next closed {interval_minutes}m bar...")
            else:
                open_positions = sum(
                    1
                    for pair_name, strategy in strategies.items()
                    if (strategy.trade and strategy.trade.is_open)
                    or (
                        pair_name in pending_orders
                        and pending_orders[pair_name].purpose == "entry"
                    )
                )
                if candidate_signals and open_positions < max_positions:
                    slots = max_positions - open_positions
                    ranked = sorted(candidate_signals, key=lambda item: item[1].score, reverse=True)
                    for pair, signal in ranked[:slots]:
                        if pair in pending_orders:
                            continue
                        decision = default_decision()
                        if client is not None:
                            snapshot = build_ai_snapshot(
                                signal=signal,
                                pair_contexts=pair_contexts,
                                portfolio=portfolio,
                                fee_hurdle_pct=fee_hurdle_pct,
                                interval_minutes=interval_minutes,
                                construction=construction,
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
                        append_event(
                            event_log_path,
                            "decision",
                            cycle=cycle,
                            pair=pair,
                            action=decision.action,
                            confidence=decision.confidence,
                            size_mult=decision.size_mult,
                            exit_mode=decision.exit_mode,
                            reason_tags=decision.reason_tags,
                            signal_score=signal.score,
                            signal_type=signal.signal_type,
                            component_tags=list(signal.component_tags),
                            gate_trend_strength=signal.gate_trend_strength,
                        )

                        if not decision.should_trade:
                            continue

                        size = compute_order_size(signal.price, notional_usd * decision.size_mult)
                        if size <= 0:
                            print(f"  SKIP {pair} | invalid order size")
                            continue

                        best_bid, best_ask = best_quotes.get(pair, (0.0, 0.0))
                        entry_price = _maker_entry_price(best_bid, best_ask, signal.price)
                        result = place_maker_limit(
                            mode,
                            "buy",
                            pair,
                            size,
                            entry_price,
                            validate=mode == "live" and validate_live_orders,
                        )
                        if mode == "live" and validate_live_orders:
                            append_event(
                                event_log_path,
                                "entry_validation",
                                cycle=cycle,
                                pair=pair,
                                side="buy",
                                order_type="limit",
                                size=size,
                                limit_price=entry_price,
                                signal_score=signal.score,
                                result=result,
                            )
                            print(
                                f"  VALIDATE ENTRY {pair} | size={size:.8f} | "
                                f"limit={_format_price(entry_price)} | score={signal.score:.2f}"
                            )
                            continue
                        order_id = _extract_order_id(result)
                        pending_orders[pair] = PendingOrder(
                            pair=pair,
                            order_id=order_id,
                            purpose="entry",
                            side="buy",
                            size=size,
                            price=entry_price,
                            created_at=time.time(),
                            signal=signal,
                            exit_mode=decision.exit_mode,
                            ai_confidence=decision.confidence,
                        )
                        print(
                            f"  PLACE ENTRY {pair} | id={order_id} | size={size:.8f} | "
                            f"limit={_format_price(entry_price)} | score={signal.score:.2f}"
                        )
                        append_event(
                            event_log_path,
                            "entry_placed",
                            cycle=cycle,
                            pair=pair,
                            order_id=order_id,
                            limit_price=entry_price,
                            size=size,
                            signal_score=signal.score,
                            exit_mode=decision.exit_mode,
                            signal_type=signal.signal_type,
                            component_tags=list(signal.component_tags),
                            gate_trend_strength=signal.gate_trend_strength,
                        )
                        persist_state()

            if cycle % 10 == 0 and trade_log:
                pnls = [trade["pnl_pct"] for trade in trade_log]
                win_rate = sum(1 for pnl in pnls if pnl > 0) / len(pnls)
                print(
                    f"\nSummary after {cycle} cycles | trades={len(pnls)} | "
                    f"win_rate={win_rate:.1%} | avg_pnl={sum(pnls) / len(pnls):.3%}\n"
                )
                append_event(
                    summary_log_path,
                    "rolling_summary",
                    cycle=cycle,
                    trades=len(pnls),
                    win_rate=win_rate,
                    avg_pnl=sum(pnls) / len(pnls),
                )
            persist_state()

        except KeyboardInterrupt:
            print("\nAgent stopped by user.")
            append_event(event_log_path, "agent_stopped", cycle=cycle, reason="keyboard_interrupt")
            persist_state()
            break
        except Exception as exc:
            print(f"[ERROR] {exc}")
            append_event(event_log_path, "error", cycle=cycle, error=str(exc))
            persist_state()

        if cycles and cycle >= cycles:
            print("Reached requested cycle limit.")
            break

        time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Kraken ensemble agent")
    parser.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    parser.add_argument("--context-pairs", default=",".join(DEFAULT_CONTEXT_PAIRS))
    parser.add_argument("--mode", default="paper", choices=["paper", "live"])
    parser.add_argument("--interval", type=int, default=strat.MASTER_INTERVAL_MINUTES, help="Base execution interval. The trend-gate ensemble requires 15.")
    parser.add_argument("--construction", default=strat.DEFAULT_ENSEMBLE_CONSTRUCTION)
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--cycles", type=int, default=0, help="0 means run forever")
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument("--notional-usd", type=float, default=DEFAULT_NOTIONAL_USD)
    parser.add_argument("--use-claude", action="store_true")
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--fee-pct", type=float, default=0.0026)
    parser.add_argument("--maker-entry-timeout-sec", type=int, default=1200)
    parser.add_argument("--maker-exit-timeout-sec", type=int, default=300)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE)
    parser.add_argument("--resume-state", action="store_true")
    parser.add_argument("--paper-init-balance", type=float, default=10000.0)
    parser.add_argument("--reset-paper", action="store_true")
    parser.add_argument("--validate-live-orders", action="store_true")
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
    log_dir = Path(args.log_dir)
    state_path = Path(args.state_file)

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
        maker_entry_timeout_sec=args.maker_entry_timeout_sec,
        maker_exit_timeout_sec=args.maker_exit_timeout_sec,
        config=config,
        log_dir=log_dir,
        state_path=state_path,
        resume_state=args.resume_state,
        paper_init_balance=args.paper_init_balance,
        reset_paper=args.reset_paper,
        validate_live_orders=args.validate_live_orders,
        construction=args.construction,
    )
