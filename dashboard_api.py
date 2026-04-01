from __future__ import annotations

import argparse
import json
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware


DEFAULT_LOG_DIR = Path(os.environ.get("DASHBOARD_LOG_DIR", "runtime/paper"))
DEFAULT_STATE_FILE = Path(os.environ.get("DASHBOARD_STATE_FILE", "runtime/paper/state.json"))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_events(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    bucket: deque[str] | list[str]
    if limit and limit > 0:
        bucket = deque(maxlen=limit)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    bucket.append(line)
        lines = list(bucket)
    else:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _latest_event(events: list[dict[str, Any]], event_name: str) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("event") == event_name:
            return event
    return None


def _session_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for idx in range(len(events) - 1, -1, -1):
        if events[idx].get("event") == "agent_started":
            return events[idx:]
    return events


def _trade_log(state: dict[str, Any]) -> list[dict[str, Any]]:
    trades = state.get("trade_log")
    return trades if isinstance(trades, list) else []


def _starting_balance(events: list[dict[str, Any]]) -> float:
    paper_init = _latest_event(events, "paper_init")
    if paper_init:
        result = paper_init.get("result") or {}
        balance = _safe_float(result.get("starting_balance"), 0.0)
        if balance > 0:
            return balance
    agent_started = _latest_event(events, "agent_started")
    if agent_started:
        notional = _safe_float(agent_started.get("notional_usd"), 0.0)
        if notional > 0:
            return 10000.0
    return 0.0


def _equity_curve(starting_balance: float, trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    equity = starting_balance if starting_balance > 0 else 0.0
    points: list[dict[str, Any]] = [{"ts": None, "equity": round(equity, 2)}]
    for trade in trades:
        pnl_pct = _safe_float(trade.get("pnl_pct"), 0.0)
        if equity > 0:
            equity *= 1.0 + pnl_pct
        points.append(
            {
                "ts": trade.get("exit_ts"),
                "equity": round(equity, 2),
                "pnl_pct": pnl_pct,
                "pair": trade.get("pair"),
                "reason": trade.get("reason"),
            }
        )
    return points


def _open_positions(state: dict[str, Any]) -> list[dict[str, Any]]:
    positions: list[dict[str, Any]] = []
    for pair, payload in (state.get("strategies") or {}).items():
        trade = (payload or {}).get("trade")
        if trade:
            positions.append(
                {
                    "pair": pair,
                    "entry_price": trade.get("entry_price"),
                    "size": trade.get("size"),
                    "signal_score": trade.get("signal_score"),
                    "stop_price": trade.get("stop_price"),
                    "target_price": trade.get("target_price"),
                    "exit_mode": trade.get("exit_mode"),
                    "entry_ts": trade.get("entry_ts"),
                }
            )
    return positions


def _candidate_session_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    detected = 0
    rejected = 0
    rejection_breakdown: dict[str, int] = {}
    for event in events:
        event_name = event.get("event")
        if event_name == "candidate_detected":
            detected += 1
        elif event_name == "candidate_rejected":
            rejected += 1
            reason = str(event.get("reason") or "unknown")
            rejection_breakdown[reason] = rejection_breakdown.get(reason, 0) + 1
    return {
        "detected": detected,
        "rejected": rejected,
        "rejection_breakdown": rejection_breakdown,
    }


def _build_status(state: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    trades = _trade_log(state)
    starting_balance = _starting_balance(events)
    curve = _equity_curve(starting_balance, trades)
    current_value = curve[-1]["equity"] if curve else starting_balance
    open_positions = _open_positions(state)
    agent_started = _latest_event(events, "agent_started") or {}
    session_events = _session_events(events)
    market_watch = _latest_event(session_events, "market_watch") or {}
    no_trade = _latest_event(session_events, "no_trade_summary") or {}
    session_stats = _candidate_session_stats(session_events)
    return {
        "generated_at": _utc_now_iso(),
        "mode": agent_started.get("mode"),
        "construction": agent_started.get("construction"),
        "interval_minutes": agent_started.get("interval_minutes"),
        "trade_pairs": agent_started.get("trade_pairs") or [],
        "context_pairs": agent_started.get("context_pairs") or [],
        "cycle": state.get("cycle", 0),
        "saved_at": state.get("saved_at"),
        "starting_balance": round(starting_balance, 2),
        "current_value": round(_safe_float(current_value, starting_balance), 2),
        "realized_pnl_pct": round(((current_value / starting_balance) - 1.0) if starting_balance > 0 else 0.0, 6),
        "open_positions": open_positions,
        "open_position_count": len(open_positions),
        "pending_orders": list((state.get("pending_orders") or {}).values()),
        "uptime_started_at": agent_started.get("ts"),
        "equity_curve": curve,
        "monitored_pairs": market_watch.get("monitored_pairs") or agent_started.get("trade_pairs") or [],
        "latest_market_watch": market_watch,
        "top_candidate": market_watch.get("top_candidate"),
        "last_no_trade_reason": no_trade.get("reason"),
        "last_no_trade_summary": no_trade.get("summary"),
        "candidate_stats": session_stats,
    }


def _build_trades(state: dict[str, Any], events: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    trades = list(reversed(_trade_log(state)))[:limit]
    exit_events = [event for event in events if str(event.get("event", "")).startswith("exit_")]
    for trade in trades:
        trade["ai_review"] = None
        pair = trade.get("pair")
        reason = trade.get("reason")
        for event in reversed(exit_events):
            if event.get("pair") == pair and event.get("reason") == reason and event.get("pnl_pct") == trade.get("pnl_pct"):
                trade["event_type"] = event.get("event")
                break
    return trades


def _build_risk(events: list[dict[str, Any]]) -> dict[str, Any]:
    latest = _latest_event(events, "risk_guardrails")
    started = _latest_event(events, "agent_started") or {}
    config = started.get("risk_config") or {}
    checks = latest.get("checks") if latest else []
    utilization: list[dict[str, Any]] = []
    for check in checks or []:
        value = check.get("value")
        limit = check.get("limit")
        ratio = None
        if isinstance(value, (int, float)) and isinstance(limit, (int, float)) and limit not in (0, 0.0):
            ratio = abs(float(value)) / abs(float(limit))
        utilization.append(
            {
                "name": check.get("name"),
                "passed": check.get("passed"),
                "reason": check.get("reason"),
                "value": value,
                "limit": limit,
                "utilization": round(ratio, 4) if ratio is not None else None,
            }
        )
    return {
        "config": config,
        "latest": latest,
        "utilization": utilization,
    }


def _build_decisions(events: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    risk_events: dict[tuple[Any, Any], dict[str, Any]] = {}
    for event in events:
        if event.get("event") == "risk_guardrails":
            risk_events[(event.get("cycle"), event.get("pair"))] = event

    decisions: list[dict[str, Any]] = []
    for event in reversed(events):
        if event.get("event") != "decision":
            continue
        key = (event.get("cycle"), event.get("pair"))
        risk_event = risk_events.get(key)
        decisions.append(
            {
                "ts": event.get("ts"),
                "cycle": event.get("cycle"),
                "pair": event.get("pair"),
                "action": event.get("action"),
                "confidence": event.get("confidence"),
                "size_mult": event.get("size_mult"),
                "exit_mode": event.get("exit_mode"),
                "reason_tags": event.get("reason_tags") or [],
                "signal_score": event.get("signal_score"),
                "guardrail_allowed": risk_event.get("allowed") if risk_event else None,
                "guardrail_summary": risk_event.get("summary") if risk_event else None,
                "guardrail_size_mult": risk_event.get("approved_size_mult") if risk_event else None,
            }
        )
        if len(decisions) >= limit:
            break
    return decisions


def _build_monitoring(events: list[dict[str, Any]]) -> dict[str, Any]:
    session_events = _session_events(events)
    market_watch = _latest_event(session_events, "market_watch") or {}
    no_trade = _latest_event(session_events, "no_trade_summary") or {}
    return {
        "market_watch": market_watch,
        "no_trade": no_trade,
        "candidate_stats": _candidate_session_stats(session_events),
    }


def create_app(log_dir: Path, state_file: Path) -> FastAPI:
    app = FastAPI(title="Kraken Agent Dashboard API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    events_file = log_dir / "events.jsonl"

    @app.get("/status")
    def status() -> dict[str, Any]:
        events = _load_events(events_file)
        state = _load_state(state_file)
        return _build_status(state, events)

    @app.get("/trades")
    def trades(limit: int = Query(25, ge=1, le=200)) -> dict[str, Any]:
        events = _load_events(events_file)
        state = _load_state(state_file)
        return {"items": _build_trades(state, events, limit)}

    @app.get("/risk")
    def risk_view() -> dict[str, Any]:
        events = _load_events(events_file)
        return _build_risk(events)

    @app.get("/decisions")
    def decisions(limit: int = Query(25, ge=1, le=200)) -> dict[str, Any]:
        events = _load_events(events_file)
        return {"items": _build_decisions(events, limit)}

    @app.get("/events")
    def events(limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
        return {"items": _load_events(events_file, limit)}

    @app.get("/monitoring")
    def monitoring() -> dict[str, Any]:
        events = _load_events(events_file)
        return _build_monitoring(events)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the lightweight dashboard API")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    app = create_app(Path(args.log_dir), Path(args.state_file))
    uvicorn.run(app, host=args.host, port=args.port)
