from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    ema_fast_span: int = 8
    ema_slow_span: int = 21
    atr_window: int = 14
    compression_window: int = 6
    compression_baseline_window: int = 24
    impulse_window: int = 12
    trigger_window: int = 2
    breakout_window: int = 12
    min_trend_strength: float = 0.0020
    min_momentum_medium: float = 0.0150
    min_impulse_pct: float = 0.0000
    min_pullback_depth: float = 0.0000
    max_pullback_depth: float = 0.0300
    support_vwap_low: float = -0.0040
    support_vwap_high: float = 0.0020
    max_extension_after_reclaim: float = 0.0055
    max_compression_ratio: float = 0.90
    min_close_location: float = 0.70
    min_volume_ratio: float = 1.00
    max_volume_ratio: float = 10.00
    deep_pullback_min_vwap: float = -0.0120
    deep_pullback_max_vwap: float = -0.0030
    stop_atr_multiple: float = 1.10
    swing_stop_buffer_pct: float = 0.0015
    min_stop_pct: float = 0.0150
    reward_to_risk: float = 0.0
    target_pct: float = 0.0450
    trailing_activation_pct: float = 0.0200
    trailing_stop_pct: float = 0.0075
    max_hold_bars: int = 5
    fast_max_hold_bars: int = 3
    min_hold_bars: int = 1
    no_progress_bars: int = 2
    no_progress_min_pct: float = 0.0015
    fast_no_progress_bars: int = 1
    fast_no_progress_min_pct: float = 0.0008
    cooldown_bars: int = 1

    @property
    def warmup_bars(self) -> int:
        return max(
            self.ema_slow_span + 2,
            self.atr_window + 2,
            self.compression_baseline_window + 2,
            self.impulse_window + 2,
            self.breakout_window + 2,
        )


DEFAULT_CONFIG = StrategyConfig()


@dataclass
class Signal:
    pair: str
    bar_idx: int
    timestamp: int
    price: float
    trigger_level: float
    support_price: float
    trend_strength: float
    momentum_short: float
    momentum_medium: float
    impulse_pct: float
    pullback_depth_pct: float
    support_touch_pct: float
    distance_from_vwap: float
    compression_ratio: float
    reclaim_pct: float
    close_location: float
    volume_ratio: float
    atr_pct: float
    score: float
    side: int = 1
    signal_type: str = "pullback_trend"

    def describe(self) -> str:
        return (
            f"{self.pair} hourly-breakout long | price=${self.price:,.6f} | "
            f"score={self.score:.2f} | trend={self.trend_strength:.3%} | "
            f"mom6={self.momentum_medium:.3%} | breakout={self.reclaim_pct:.3%} | "
            f"vol={self.volume_ratio:.2f}x | close_loc={self.close_location:.2f} | "
            f"compress={self.compression_ratio:.2f}"
        )


@dataclass
class Trade:
    pair: str
    entry_bar: int
    entry_ts: int
    entry_price: float
    size: float
    stop_price: float
    target_price: Optional[float]
    signal_score: float
    support_price: float
    exit_mode: str = "standard"
    ai_confidence: float = 0.0
    side: int = 1
    exit_bar: Optional[int] = None
    exit_ts: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    best_price: float = 0.0
    best_pct: float = 0.0

    def __post_init__(self) -> None:
        self.best_price = self.entry_price

    @property
    def is_open(self) -> bool:
        return self.exit_bar is None

    def bars_held(self, current_bar: int) -> int:
        return current_bar - self.entry_bar

    def current_pnl_pct(self, price: float) -> float:
        return self.side * (price - self.entry_price) / self.entry_price

    def realized_pnl_pct(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        return self.current_pnl_pct(self.exit_price)

    def update_best(self, current_price: float) -> None:
        self.best_price = max(self.best_price, current_price)
        self.best_pct = max(self.best_pct, self.current_pnl_pct(current_price))


def parse_ohlc(raw: dict) -> pd.DataFrame:
    pair_key = next(k for k in raw if k != "last")
    bars = raw[pair_key]
    df = pd.DataFrame(
        bars,
        columns=["ts", "open", "high", "low", "close", "vwap_k", "volume", "count"],
    )
    for col in ["open", "high", "low", "close", "vwap_k", "volume"]:
        df[col] = df[col].astype(float)
    df["ts"] = df["ts"].astype(int)
    return df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)


def _compute_session_vwap(out: pd.DataFrame) -> pd.Series:
    session_keys = pd.to_datetime(out["ts"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    cum_pv = (out["close"] * out["volume"]).groupby(session_keys).cumsum()
    cum_vol = out["volume"].groupby(session_keys).cumsum().replace(0, np.nan)
    return cum_pv / cum_vol


def compute_orderbook_features(orderbook: dict) -> tuple[float, float]:
    pair_key = next(iter(orderbook))
    payload = orderbook[pair_key]
    bids = [(float(price), float(size)) for price, size, *_ in payload["bids"][:5]]
    asks = [(float(price), float(size)) for price, size, *_ in payload["asks"][:5]]

    bid_vol = sum(size for _, size in bids)
    ask_vol = sum(size for _, size in asks)
    total = bid_vol + ask_vol
    obi = bid_vol / total if total > 0 else 0.5

    best_bid = bids[0][0] if bids else 0.0
    best_ask = asks[0][0] if asks else 0.0
    mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.0
    spread_pct = (best_ask - best_bid) / mid if mid else 0.0
    return obi, spread_pct


def compute_features(df: pd.DataFrame, config: StrategyConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)

    out["ema_fast"] = out["close"].ewm(span=config.ema_fast_span, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=config.ema_slow_span, adjust=False).mean()
    out["trend_strength"] = (out["ema_fast"] - out["ema_slow"]) / out["ema_slow"]

    out["momentum_1"] = out["close"].pct_change(1)
    out["momentum_short"] = out["close"].pct_change(3)
    out["momentum_medium"] = out["close"].pct_change(6)
    out["momentum_long"] = out["close"].pct_change(12)

    prev_close = out["close"].shift(1)
    tr_components = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    out["true_range"] = tr_components.max(axis=1)
    out["atr"] = out["true_range"].rolling(config.atr_window, min_periods=3).mean()
    out["atr_pct"] = out["atr"] / out["close"]

    out["atr_short"] = out["true_range"].rolling(config.compression_window, min_periods=3).mean()
    out["atr_long"] = out["true_range"].rolling(
        config.compression_baseline_window,
        min_periods=6,
    ).mean()
    out["compression_ratio"] = out["atr_short"] / out["atr_long"]

    out["volume_ma"] = out["volume"].rolling(20, min_periods=5).mean().shift(1)
    out["volume_ratio"] = out["volume"] / out["volume_ma"].replace(0, np.nan)

    out["session_vwap"] = _compute_session_vwap(out)
    out["distance_from_vwap"] = (out["close"] - out["session_vwap"]) / out["session_vwap"]
    out["low_vs_vwap"] = (out["low"] - out["session_vwap"]) / out["session_vwap"]
    out["support_touch_pct"] = out["low_vs_vwap"].rolling(5, min_periods=1).min()

    session_keys = pd.to_datetime(out["ts"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    out["session_high"] = out["high"].groupby(session_keys).cummax()
    out["session_low"] = out["low"].groupby(session_keys).cummin()
    session_range = (out["session_high"] - out["session_low"]).replace(0, np.nan)
    out["range_position"] = (out["close"] - out["session_low"]) / session_range

    out["swing_high"] = out["high"].rolling(config.impulse_window).max().shift(1)
    out["swing_low"] = out["low"].rolling(config.impulse_window).min().shift(1)
    out["impulse_pct"] = (out["swing_high"] - out["swing_low"]) / out["swing_low"]
    out["pullback_depth_pct"] = (out["swing_high"] - out["low"]) / out["swing_high"]

    out["trigger_level"] = out["high"].rolling(config.trigger_window).max().shift(1)
    out["breakout_level"] = out["high"].rolling(config.breakout_window).max().shift(1)
    out["reclaim_pct"] = (out["close"] - out["trigger_level"]) / out["trigger_level"]
    out["breakout_pct"] = (out["close"] - out["breakout_level"]) / out["breakout_level"]

    bar_range = (out["high"] - out["low"]).replace(0, np.nan)
    out["close_location"] = (out["close"] - out["low"]) / bar_range
    out["support_price"] = pd.concat([out["session_vwap"], out["ema_slow"]], axis=1).min(axis=1)

    return out


class HourlyBreakoutStrategy:
    def __init__(self, pair: str, config: StrategyConfig = DEFAULT_CONFIG):
        self.pair = pair
        self.config = config
        self.last_signal_bar = -config.cooldown_bars - 1
        self.trade: Optional[Trade] = None

    def detect(self, df: pd.DataFrame) -> Optional[Signal]:
        if len(df) < self.config.warmup_bars:
            return None

        bar_idx = len(df) - 1
        if self.trade and self.trade.is_open:
            return None
        if bar_idx - self.last_signal_bar <= self.config.cooldown_bars:
            return None

        last = df.iloc[-1]
        required = [
            "trend_strength",
            "momentum_short",
            "momentum_medium",
            "impulse_pct",
            "pullback_depth_pct",
            "support_touch_pct",
            "distance_from_vwap",
            "compression_ratio",
            "reclaim_pct",
            "close_location",
            "volume_ratio",
            "atr_pct",
            "breakout_level",
            "support_price",
        ]
        if any(np.isnan(float(last.get(col, np.nan))) for col in required):
            return None

        price = float(last["close"])
        trend_strength = float(last["trend_strength"])
        momentum_short = float(last["momentum_short"])
        momentum_medium = float(last["momentum_medium"])
        impulse_pct = float(last["impulse_pct"])
        pullback_depth_pct = float(last["pullback_depth_pct"])
        support_touch_pct = float(last["support_touch_pct"])
        distance_from_vwap = float(last["distance_from_vwap"])
        compression_ratio = float(last["compression_ratio"])
        reclaim_pct = float(last["reclaim_pct"])
        close_location = float(last["close_location"])
        volume_ratio = float(last["volume_ratio"])
        atr_pct = float(last["atr_pct"])
        breakout_level = float(last["breakout_level"])
        breakout_pct = float(last["breakout_pct"])
        support_price = float(last["ema_fast"])

        if trend_strength < self.config.min_trend_strength:
            return None
        if float(last["ema_fast"]) <= float(last["ema_slow"]):
            return None
        if compression_ratio > self.config.max_compression_ratio:
            return None
        if close_location < self.config.min_close_location:
            return None
        if volume_ratio < self.config.min_volume_ratio:
            return None
        if momentum_medium < self.config.min_momentum_medium:
            return None
        if price <= breakout_level:
            return None

        score = (
            100.0 * trend_strength
            + 40.0 * max(0.0, breakout_pct)
            + 12.0 * max(0.0, momentum_medium - self.config.min_momentum_medium)
            + 10.0 * max(0.0, volume_ratio - self.config.min_volume_ratio)
            + 8.0 * max(0.0, close_location - self.config.min_close_location)
            + 6.0 * max(0.0, self.config.max_compression_ratio - compression_ratio)
        )

        return Signal(
            pair=self.pair,
            bar_idx=bar_idx,
            timestamp=int(last["ts"]),
            price=price,
            trigger_level=breakout_level,
            support_price=support_price,
            trend_strength=trend_strength,
            momentum_short=momentum_short,
            momentum_medium=momentum_medium,
            impulse_pct=impulse_pct,
            pullback_depth_pct=pullback_depth_pct,
            support_touch_pct=support_touch_pct,
            distance_from_vwap=distance_from_vwap,
            compression_ratio=compression_ratio,
            reclaim_pct=breakout_pct,
            close_location=close_location,
            volume_ratio=volume_ratio,
            atr_pct=atr_pct,
            score=score,
            signal_type="hourly_breakout_long",
        )

    def record_signal(self, bar_idx: int) -> None:
        self.last_signal_bar = bar_idx

    def open_trade(
        self,
        signal: Signal,
        size: float,
        entry_price: Optional[float] = None,
        exit_mode: str = "standard",
        ai_confidence: float = 0.0,
        entry_bar: Optional[int] = None,
        entry_ts: Optional[int] = None,
    ) -> Trade:
        fill_price = signal.price if entry_price is None else entry_price
        stop_distance = fill_price * self.config.min_stop_pct

        self.trade = Trade(
            pair=self.pair,
            entry_bar=signal.bar_idx if entry_bar is None else entry_bar,
            entry_ts=signal.timestamp if entry_ts is None else entry_ts,
            entry_price=fill_price,
            size=size,
            stop_price=fill_price - stop_distance,
            target_price=fill_price * (1 + self.config.target_pct),
            signal_score=signal.score,
            support_price=signal.support_price,
            exit_mode=exit_mode,
            ai_confidence=ai_confidence,
        )
        self.record_signal(signal.bar_idx)
        return self.trade

    def check_exit(self, df: pd.DataFrame) -> Optional[str]:
        if self.trade is None or not self.trade.is_open:
            return None

        current_bar = len(df) - 1
        last = df.iloc[-1]
        current_price = float(last["close"])
        self.trade.update_best(current_price)

        if current_price <= self.trade.stop_price:
            return "STOP_LOSS"
        if self.trade.target_price is not None and current_price >= self.trade.target_price:
            return "TAKE_PROFIT"

        max_hold = (
            self.config.fast_max_hold_bars
            if self.trade.exit_mode == "fast"
            else self.config.max_hold_bars
        )
        if self.trade.bars_held(current_bar) >= max_hold:
            return "TIME_LIMIT"

        if (
            self.trade.bars_held(current_bar) >= self.config.min_hold_bars
            and current_price < float(last["ema_fast"])
            and float(last["momentum_medium"]) < 0
        ):
            return "TREND_LOST"

        return None

    def check_live_exit_price(self, current_price: float) -> Optional[str]:
        if self.trade is None or not self.trade.is_open:
            return None
        self.trade.update_best(current_price)
        if current_price <= self.trade.stop_price:
            return "STOP_LOSS"
        if self.trade.target_price is not None and current_price >= self.trade.target_price:
            return "TAKE_PROFIT"
        return None

    def close_trade(self, bar_idx: int, timestamp: int, price: float, reason: str) -> Trade:
        if self.trade is None:
            raise RuntimeError("No open trade to close")
        self.trade.exit_bar = bar_idx
        self.trade.exit_ts = timestamp
        self.trade.exit_price = price
        self.trade.exit_reason = reason
        return self.trade

    def describe_state(self, df: pd.DataFrame) -> str:
        if df.empty:
            return f"{self.pair}: no data"

        last = df.iloc[-1]
        fields = [
            f"{self.pair} price=${float(last['close']):,.6f}",
            f"trend={float(last.get('trend_strength', 0.0)):.3%}",
            f"mom_med={float(last.get('momentum_medium', 0.0)):.3%}",
            f"breakout={float(last.get('breakout_pct', 0.0)):.3%}",
            f"compress={float(last.get('compression_ratio', 0.0)):.2f}",
            f"vol={float(last.get('volume_ratio', 0.0)):.2f}x",
            f"close_loc={float(last.get('close_location', 0.0)):.2f}",
        ]
        if self.trade and self.trade.is_open:
            pnl = self.trade.current_pnl_pct(float(last["close"]))
            fields.append(f"open_pnl={pnl:.3%}")
        return " | ".join(fields)


PullbackTrendStrategy = HourlyBreakoutStrategy
