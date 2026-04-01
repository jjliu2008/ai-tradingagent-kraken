from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RiskConfig:
    max_position_notional_usd: float = 750.0
    max_portfolio_concentration: float = 0.35
    daily_loss_limit_pct: float = 0.03
    max_drawdown_pct: float = 0.08
    max_spread_pct: float = 0.004
    max_atr_pct: float = 0.08
    atr_soft_cap_pct: float = 0.035
    min_obi: float = 0.45
    max_open_entry_orders: int = 2
    volatility_size_floor: float = 0.35
    spread_soft_cap_pct: float = 0.0025


@dataclass(frozen=True)
class RiskCheck:
    name: str
    passed: bool
    reason: str
    value: float | int | None = None
    limit: float | int | None = None


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    approved_size_mult: float
    summary: str
    checks: list[RiskCheck]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checks"] = [asdict(check) for check in self.checks]
        return payload


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _portfolio_value(portfolio: dict[str, Any]) -> float:
    for key in ("current_value", "equity", "balance", "total_value", "starting_balance"):
        value = _safe_float(portfolio.get(key), 0.0)
        if value > 0:
            return value
    return 0.0


def _starting_balance(portfolio: dict[str, Any]) -> float:
    value = _safe_float(portfolio.get("starting_balance"), 0.0)
    if value > 0:
        return value
    return _portfolio_value(portfolio)


def _open_position_count(strategies: dict[str, Any]) -> int:
    return sum(1 for strategy in strategies.values() if getattr(strategy, "trade", None) and strategy.trade.is_open)


def _pending_entry_count(pending_orders: dict[str, Any]) -> int:
    return sum(1 for order in pending_orders.values() if getattr(order, "purpose", "") == "entry")


def _realized_daily_pnl_pct(trade_log: list[dict[str, Any]], now_ts: int) -> float:
    cutoff = now_ts - 24 * 60 * 60
    total = 0.0
    for trade in trade_log:
        exit_ts = trade.get("exit_ts")
        if exit_ts is None:
            continue
        if int(exit_ts) >= cutoff:
            total += _safe_float(trade.get("pnl_pct"), 0.0)
    return total


def _max_drawdown_pct(trade_log: list[dict[str, Any]]) -> float:
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for trade in trade_log:
        pnl = _safe_float(trade.get("pnl_pct"), 0.0)
        equity *= 1.0 + pnl
        peak = max(peak, equity)
        if peak > 0:
            worst = min(worst, (equity - peak) / peak)
    return worst


def evaluate_entry(
    pair: str,
    pair_snapshot: dict[str, Any],
    portfolio: dict[str, Any],
    trade_log: list[dict[str, Any]],
    strategies: dict[str, Any],
    pending_orders: dict[str, Any],
    proposed_notional_usd: float,
    requested_size_mult: float,
    max_positions: int,
    now_ts: int,
    config: RiskConfig,
) -> RiskDecision:
    current_value = _portfolio_value(portfolio)
    starting_balance = _starting_balance(portfolio)
    open_positions = _open_position_count(strategies)
    open_entry_orders = _pending_entry_count(pending_orders)
    concentration = proposed_notional_usd / current_value if current_value > 0 else 0.0
    daily_pnl_pct = _realized_daily_pnl_pct(trade_log, now_ts)
    drawdown_pct = _max_drawdown_pct(trade_log)
    spread_pct = _safe_float(pair_snapshot.get("spread_pct"), 0.0)
    atr_pct = _safe_float(pair_snapshot.get("atr_pct"), 0.0)
    obi = _safe_float(pair_snapshot.get("obi"), 0.5)

    checks: list[RiskCheck] = [
        RiskCheck(
            name="position_limit",
            passed=proposed_notional_usd <= config.max_position_notional_usd,
            reason="position size within per-trade cap" if proposed_notional_usd <= config.max_position_notional_usd else "position size above per-trade cap",
            value=round(proposed_notional_usd, 2),
            limit=round(config.max_position_notional_usd, 2),
        ),
        RiskCheck(
            name="portfolio_slots",
            passed=open_positions < max_positions,
            reason="position slot available" if open_positions < max_positions else "max open positions reached",
            value=open_positions,
            limit=max_positions,
        ),
        RiskCheck(
            name="pending_entry_limit",
            passed=open_entry_orders < config.max_open_entry_orders,
            reason="pending entry count within cap" if open_entry_orders < config.max_open_entry_orders else "too many pending entries",
            value=open_entry_orders,
            limit=config.max_open_entry_orders,
        ),
        RiskCheck(
            name="concentration_limit",
            passed=concentration <= config.max_portfolio_concentration,
            reason="portfolio concentration within cap" if concentration <= config.max_portfolio_concentration else "portfolio concentration too high",
            value=round(concentration, 6),
            limit=round(config.max_portfolio_concentration, 6),
        ),
        RiskCheck(
            name="daily_loss_circuit",
            passed=daily_pnl_pct > -config.daily_loss_limit_pct,
            reason="daily loss circuit not triggered" if daily_pnl_pct > -config.daily_loss_limit_pct else "daily loss circuit triggered",
            value=round(daily_pnl_pct, 6),
            limit=round(-config.daily_loss_limit_pct, 6),
        ),
        RiskCheck(
            name="drawdown_kill_switch",
            passed=drawdown_pct > -config.max_drawdown_pct,
            reason="drawdown below kill threshold" if drawdown_pct > -config.max_drawdown_pct else "drawdown kill switch triggered",
            value=round(drawdown_pct, 6),
            limit=round(-config.max_drawdown_pct, 6),
        ),
        RiskCheck(
            name="spread_gate",
            passed=spread_pct <= config.max_spread_pct,
            reason="spread within cap" if spread_pct <= config.max_spread_pct else "spread too wide",
            value=round(spread_pct, 6),
            limit=round(config.max_spread_pct, 6),
        ),
        RiskCheck(
            name="volatility_gate",
            passed=atr_pct <= config.max_atr_pct,
            reason="volatility within cap" if atr_pct <= config.max_atr_pct else "volatility too high",
            value=round(atr_pct, 6),
            limit=round(config.max_atr_pct, 6),
        ),
        RiskCheck(
            name="orderbook_imbalance",
            passed=obi >= config.min_obi,
            reason="bid support acceptable" if obi >= config.min_obi else "orderbook imbalance too weak",
            value=round(obi, 6),
            limit=round(config.min_obi, 6),
        ),
    ]

    approved_size_mult = requested_size_mult
    if atr_pct > config.atr_soft_cap_pct and atr_pct > 0:
        vol_cap = max(config.volatility_size_floor, config.atr_soft_cap_pct / atr_pct)
        approved_size_mult = min(approved_size_mult, vol_cap)
    if spread_pct > config.spread_soft_cap_pct and spread_pct > 0:
        spread_cap = max(config.volatility_size_floor, config.spread_soft_cap_pct / spread_pct)
        approved_size_mult = min(approved_size_mult, spread_cap)
    if current_value > 0 and proposed_notional_usd > 0:
        concentration_cap = max(
            0.0,
            min(1.0, (config.max_portfolio_concentration * current_value) / proposed_notional_usd),
        )
        approved_size_mult = min(approved_size_mult, concentration_cap)
    approved_size_mult = max(0.0, min(requested_size_mult, approved_size_mult))

    allowed = all(check.passed for check in checks) and approved_size_mult > 0
    failed = [check.reason for check in checks if not check.passed]
    if not allowed:
        summary = failed[0] if failed else "risk check blocked trade"
    elif approved_size_mult < requested_size_mult:
        summary = "trade allowed with size cap"
    else:
        summary = "all guardrails passed"

    return RiskDecision(
        allowed=allowed,
        approved_size_mult=approved_size_mult,
        summary=summary,
        checks=checks,
    )


def portfolio_risk_snapshot(
    portfolio: dict[str, Any],
    trade_log: list[dict[str, Any]],
    strategies: dict[str, Any],
    pending_orders: dict[str, Any],
    now_ts: int,
) -> dict[str, Any]:
    return {
        "current_value": round(_portfolio_value(portfolio), 2),
        "starting_balance": round(_starting_balance(portfolio), 2),
        "open_positions": _open_position_count(strategies),
        "pending_entry_orders": _pending_entry_count(pending_orders),
        "daily_realized_pnl_pct": round(_realized_daily_pnl_pct(trade_log, now_ts), 6),
        "max_drawdown_pct": round(_max_drawdown_pct(trade_log), 6),
    }
