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
    min_trend_strength: float = 0.0015
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
    max_hold_bars: int = 12
    fast_max_hold_bars: int = 8
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
MASTER_INTERVAL_MINUTES = 15
ENSEMBLE_LOOKBACK_BARS = 320
DEFAULT_ENSEMBLE_CONSTRUCTION = "tc15_tighter_volume_cap"


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
    component_tags: tuple[str, ...] = ()
    gate_trend_strength: float = 0.0
    gate_momentum_medium: float = 0.0

    def describe(self) -> str:
        components = ",".join(self.component_tags) if self.component_tags else "none"
        label = self.signal_type.replace("_long", "").replace("_", " ")
        return (
            f"{self.pair} {label} | price=${self.price:,.6f} | "
            f"score={self.score:.2f} | trend={self.trend_strength:.3%} | "
            f"gate60={self.gate_trend_strength:.3%} | comps={components} | "
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


def _compute_macd_hist(series: pd.Series) -> pd.Series:
    ema_fast = series.ewm(span=12, adjust=False).mean()
    ema_slow = series.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line - signal_line


def add_ensemble_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["macd_hist"] = _compute_macd_hist(out["close"])
    out["roll_high_12"] = out["high"].rolling(12).max().shift(1)
    return out


def resample_ohlcv(df: pd.DataFrame, interval: int) -> pd.DataFrame:
    raw_cols = ["ts", "open", "high", "low", "close", "vwap_k", "volume", "count"]
    if interval == MASTER_INTERVAL_MINUTES:
        return df[raw_cols].copy().reset_index(drop=True)

    frame = df[raw_cols].copy()
    frame["dt"] = pd.to_datetime(frame["ts"], unit="s", utc=True)
    frame = frame.set_index("dt")
    frame["pv"] = frame["vwap_k"] * frame["volume"]

    agg = (
        frame.resample(f"{interval}min", label="left", closed="left", origin="epoch")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "pv": "sum",
                "volume": "sum",
                "count": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
    )
    agg["vwap_k"] = agg["pv"] / agg["volume"].replace(0, np.nan)
    agg = agg.dropna(subset=["vwap_k"]).reset_index()
    agg["ts"] = agg["dt"].map(lambda value: int(value.timestamp()))
    return agg[raw_cols].reset_index(drop=True)


def _prepare_ensemble_frame(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    raw_cols = ["ts", "open", "high", "low", "close", "vwap_k", "volume", "count"]
    return add_ensemble_features(compute_features(df[raw_cols].copy().reset_index(drop=True), config=config))


def _map_signal_to_master(
    master_ts: pd.Series,
    signal_ts: pd.Series,
    interval_minutes: int,
) -> pd.Series:
    actionable_ts = signal_ts.astype(int) + (interval_minutes - MASTER_INTERVAL_MINUTES) * 60
    return master_ts.astype(int).isin(actionable_ts.astype(int)).astype(bool)


def _build_component_masks(df: pd.DataFrame, interval_minutes: int) -> dict[str, pd.Series]:
    trend_up = df["ema_fast"] > df["ema_slow"]

    if interval_minutes == 60:
        return {
            "mb60": (
                (df["trend_strength"] > 0.0020)
                & trend_up
                & (df["close"] > df["roll_high_12"])
                & (df["momentum_medium"] > 0.015)
                & (df["volume_ratio"] > 1.0)
                & (df["close_location"] > 0.70)
                & (df["compression_ratio"] < 0.90)
            ).fillna(False),
            "vst60": (
                trend_up
                & (df["volume_ratio"] > 2.0)
                & (df["close"] > df["open"])
                & (df["close_location"] > 0.70)
                & (df["trend_strength"] > 0.0015)
                & (((df["close"] - df["open"]) / df["open"]) > 0.005)
            ).fillna(False),
        }

    if interval_minutes == MASTER_INTERVAL_MINUTES:
        return {
            "tc15": (
                trend_up
                & (df["trend_strength"] > 0.0025)
                & (df["macd_hist"] > 0)
                & (df["macd_hist"] > df["macd_hist"].shift(1))
                & (df["compression_ratio"] < 0.85)
                & (df["volume_ratio"] > 1.1)
                & (df["close_location"] > 0.65)
                & (df["close"] > df["roll_high_12"])
            ).fillna(False)
        }

    if interval_minutes == 30:
        return {
            "mbt30": (
                (df["trend_strength"] > 0.0025)
                & trend_up
                & (df["close"] > df["roll_high_12"])
                & (df["momentum_medium"] > 0.020)
                & (df["volume_ratio"] > 1.2)
                & (df["close_location"] > 0.70)
            ).fillna(False),
            "tc30": (
                trend_up
                & (df["trend_strength"] > 0.0025)
                & (df["macd_hist"] > 0)
                & (df["macd_hist"] > df["macd_hist"].shift(1))
                & (df["compression_ratio"] < 0.85)
                & (df["volume_ratio"] > 1.1)
                & (df["close_location"] > 0.65)
                & (df["close"] > df["roll_high_12"])
            ).fillna(False),
            "atr30": (
                (df["compression_ratio"] < 0.75)
                & trend_up
                & (df["close"] > df["roll_high_12"])
                & (df["momentum_medium"] > 0.010)
                & (df["volume_ratio"] > 1.0)
            ).fillna(False),
        }

    return {}


def ensemble_construction_names() -> tuple[str, ...]:
    return (
        "trend_gate",
        "baseline_mb60",
        "tc15_only",
        "tc15_tighter_volume_cap",
        "tc15_cap_or_mb60",
        "tc30_only",
        "vst60_only",
        "atr30_only",
        "baseline_or_tc15",
        "baseline_or_tc30",
        "baseline_or_atr30",
        "baseline_or_vst60",
        "tc15_or_tc30",
        "tc15_or_atr30",
        "tc15_or_vst60",
        "core_union_no_mbt30",
        "consensus_pos2",
        "tc15_strong60",
        "tc15_very60",
        "vst60_strong60",
        "union_strong60",
        "union_closehi",
        "union_volhi",
        "baseline_or_tc15_strong60",
    )


def _construction_mask(frame: pd.DataFrame, construction: str) -> pd.Series:
    signals = {
        "mb60": frame["signal_mb60"].fillna(False),
        "mbt30": frame["signal_mbt30"].fillna(False),
        "vst60": frame["signal_vst60"].fillna(False),
        "tc30": frame["signal_tc30"].fillna(False),
        "tc15": frame["signal_tc15"].fillna(False),
        "atr30": frame["signal_atr30"].fillna(False),
    }
    strong_60 = frame["gate_is_open"].fillna(False)
    very_strong_60 = frame["gate_very_strong_60"].fillna(False)
    close_high = (frame["close_location"] > 0.75).fillna(False)
    volume_high = (frame["volume_ratio"] > 1.2).fillna(False)

    constructions = {
        "trend_gate": (signals["mbt30"] | signals["tc30"] | signals["atr30"] | signals["tc15"]) & strong_60,
        "baseline_mb60": signals["mb60"],
        "tc15_only": signals["tc15"],
        "tc15_tighter_volume_cap": signals["tc15"] & (frame["volume_ratio"] <= 4.5).fillna(False),
        "tc15_cap_or_mb60": (signals["tc15"] & (frame["volume_ratio"] <= 4.5).fillna(False)) | signals["mb60"],
        "tc30_only": signals["tc30"],
        "vst60_only": signals["vst60"],
        "atr30_only": signals["atr30"],
        "baseline_or_tc15": signals["mb60"] | signals["tc15"],
        "baseline_or_tc30": signals["mb60"] | signals["tc30"],
        "baseline_or_atr30": signals["mb60"] | signals["atr30"],
        "baseline_or_vst60": signals["mb60"] | signals["vst60"],
        "tc15_or_tc30": signals["tc15"] | signals["tc30"],
        "tc15_or_atr30": signals["tc15"] | signals["atr30"],
        "tc15_or_vst60": signals["tc15"] | signals["vst60"],
        "core_union_no_mbt30": (
            signals["mb60"] | signals["tc15"] | signals["tc30"] | signals["atr30"] | signals["vst60"]
        ),
        "consensus_pos2": (
            signals["mb60"].astype(int)
            + signals["tc15"].astype(int)
            + signals["tc30"].astype(int)
            + signals["atr30"].astype(int)
            + signals["vst60"].astype(int)
        ) >= 2,
        "tc15_strong60": signals["tc15"] & strong_60,
        "tc15_very60": signals["tc15"] & very_strong_60,
        "vst60_strong60": signals["vst60"] & strong_60,
        "union_strong60": (
            signals["mb60"] | signals["tc15"] | signals["tc30"] | signals["atr30"] | signals["vst60"]
        ) & strong_60,
        "union_closehi": (
            signals["mb60"] | signals["tc15"] | signals["tc30"] | signals["atr30"] | signals["vst60"]
        ) & close_high,
        "union_volhi": (
            signals["mb60"] | signals["tc15"] | signals["tc30"] | signals["atr30"] | signals["vst60"]
        ) & volume_high,
        "baseline_or_tc15_strong60": (signals["mb60"] | signals["tc15"]) & strong_60,
    }
    if construction not in constructions:
        supported = ", ".join(ensemble_construction_names())
        raise ValueError(f"Unknown ensemble construction '{construction}'. Supported: {supported}")
    return constructions[construction].fillna(False)


def build_ensemble_frame(
    df: pd.DataFrame,
    construction: str = DEFAULT_ENSEMBLE_CONSTRUCTION,
    config: StrategyConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    raw = df[["ts", "open", "high", "low", "close", "vwap_k", "volume", "count"]].copy().reset_index(drop=True)
    master = _prepare_ensemble_frame(raw, config=config)
    frame30 = _prepare_ensemble_frame(resample_ohlcv(raw, 30), config=config)
    frame60 = _prepare_ensemble_frame(resample_ohlcv(raw, 60), config=config)
    if len(frame30) < 24 or len(frame60) < 24:
        return pd.DataFrame()

    master_ts = master["ts"].astype(int)
    mask15 = _build_component_masks(master, MASTER_INTERVAL_MINUTES)
    mask30 = _build_component_masks(frame30, 30)
    mask60 = _build_component_masks(frame60, 60)
    signal_mb60 = _map_signal_to_master(master_ts, frame60.loc[mask60["mb60"], "ts"], 60)
    signal_tc15 = mask15["tc15"].fillna(False)
    signal_mbt30 = _map_signal_to_master(master_ts, frame30.loc[mask30["mbt30"], "ts"], 30)
    signal_vst60 = _map_signal_to_master(master_ts, frame60.loc[mask60["vst60"], "ts"], 60)
    signal_tc30 = _map_signal_to_master(master_ts, frame30.loc[mask30["tc30"], "ts"], 30)
    signal_atr30 = _map_signal_to_master(master_ts, frame30.loc[mask30["atr30"], "ts"], 30)

    gate60 = frame60[["ts", "ema_fast", "ema_slow", "trend_strength", "momentum_medium"]].copy()
    gate60["effective_ts"] = gate60["ts"].astype(int) + (60 - MASTER_INTERVAL_MINUTES) * 60
    reg60 = (
        gate60.set_index("effective_ts")[["ema_fast", "ema_slow", "trend_strength", "momentum_medium"]]
        .reindex(master_ts)
        .ffill()
        .reset_index(drop=True)
    )

    gate_mask = (
        (reg60["ema_fast"] > reg60["ema_slow"])
        & (reg60["trend_strength"] > 0.0015)
    ).fillna(False)
    component_count = (
        signal_mb60.astype(int)
        + signal_tc15.astype(int)
        + signal_mbt30.astype(int)
        + signal_vst60.astype(int)
        + signal_tc30.astype(int)
        + signal_atr30.astype(int)
    )
    entry_signal = _construction_mask(
        pd.DataFrame(
            {
                "signal_mb60": signal_mb60.to_numpy(dtype=bool),
                "signal_tc15": signal_tc15.to_numpy(dtype=bool),
                "signal_mbt30": signal_mbt30.to_numpy(dtype=bool),
                "signal_vst60": signal_vst60.to_numpy(dtype=bool),
                "signal_tc30": signal_tc30.to_numpy(dtype=bool),
                "signal_atr30": signal_atr30.to_numpy(dtype=bool),
                "gate_is_open": gate_mask.to_numpy(dtype=bool),
                "gate_very_strong_60": (
                    (reg60["ema_fast"] > reg60["ema_slow"])
                    & (reg60["trend_strength"] > 0.0030)
                ).fillna(False).to_numpy(dtype=bool),
                "close_location": master["close_location"].to_numpy(dtype=float),
                "volume_ratio": master["volume_ratio"].to_numpy(dtype=float),
            }
        ),
        construction=construction,
    )
    score = (
        24.0 * component_count
        + 120.0 * reg60["trend_strength"].clip(lower=0.0)
        + 40.0 * master["breakout_pct"].clip(lower=0.0)
        + 20.0 * master["momentum_medium"].clip(lower=0.0)
        + 10.0 * (master["volume_ratio"] - 1.0).clip(lower=0.0)
        + 8.0 * (master["close_location"] - 0.60).clip(lower=0.0)
        + 6.0 * (0.90 - master["compression_ratio"]).clip(lower=0.0)
    )

    out = master.copy()
    out["construction"] = construction
    out["signal_mb60"] = signal_mb60.to_numpy(dtype=bool)
    out["signal_tc15"] = signal_tc15.to_numpy(dtype=bool)
    out["signal_mbt30"] = signal_mbt30.to_numpy(dtype=bool)
    out["signal_vst60"] = signal_vst60.to_numpy(dtype=bool)
    out["signal_tc30"] = signal_tc30.to_numpy(dtype=bool)
    out["signal_atr30"] = signal_atr30.to_numpy(dtype=bool)
    out["gate_ema_fast_60"] = reg60["ema_fast"].to_numpy()
    out["gate_ema_slow_60"] = reg60["ema_slow"].to_numpy()
    out["gate_trend_strength_60"] = reg60["trend_strength"].to_numpy()
    out["gate_momentum_medium_60"] = reg60["momentum_medium"].to_numpy()
    out["gate_is_open"] = gate_mask.to_numpy(dtype=bool)
    out["gate_very_strong_60"] = (
        (reg60["ema_fast"] > reg60["ema_slow"])
        & (reg60["trend_strength"] > 0.0030)
    ).fillna(False).to_numpy(dtype=bool)
    out["component_count"] = component_count.to_numpy(dtype=int)
    out["entry_signal"] = entry_signal.to_numpy(dtype=bool)
    out["signal_score"] = score.to_numpy(dtype=float)
    return out


def build_trend_gate_frame(df: pd.DataFrame, config: StrategyConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    return build_ensemble_frame(df, construction="trend_gate", config=config)


def build_trend_gate_signal(
    pair: str,
    frame: pd.DataFrame,
    row_idx: int,
    bar_idx: int | None = None,
) -> Optional[Signal]:
    return build_ensemble_signal(pair, frame, row_idx, construction="trend_gate", bar_idx=bar_idx)


def build_ensemble_signal(
    pair: str,
    frame: pd.DataFrame,
    row_idx: int,
    construction: str = DEFAULT_ENSEMBLE_CONSTRUCTION,
    bar_idx: int | None = None,
) -> Optional[Signal]:
    if frame.empty or row_idx < 0 or row_idx >= len(frame):
        return None

    row = frame.iloc[row_idx]
    required = [
        "trend_strength",
        "momentum_short",
        "momentum_medium",
        "impulse_pct",
        "pullback_depth_pct",
        "support_touch_pct",
        "distance_from_vwap",
        "compression_ratio",
        "close_location",
        "volume_ratio",
        "atr_pct",
        "breakout_level",
        "support_price",
        "gate_trend_strength_60",
        "gate_momentum_medium_60",
        "signal_score",
    ]
    if any(np.isnan(float(row.get(col, np.nan))) for col in required):
        return None

    component_tags = tuple(
        tag
        for tag, column in (
            ("mb60", "signal_mb60"),
            ("tc15", "signal_tc15"),
            ("mbt30", "signal_mbt30"),
            ("vst60", "signal_vst60"),
            ("tc30", "signal_tc30"),
            ("atr30", "signal_atr30"),
        )
        if bool(row.get(column, False))
    )

    return Signal(
        pair=pair,
        bar_idx=row_idx if bar_idx is None else bar_idx,
        timestamp=int(row["ts"]),
        price=float(row["close"]),
        trigger_level=float(row["breakout_level"]),
        support_price=float(row["ema_fast"]),
        trend_strength=float(row["trend_strength"]),
        momentum_short=float(row["momentum_short"]),
        momentum_medium=float(row["momentum_medium"]),
        impulse_pct=float(row["impulse_pct"]),
        pullback_depth_pct=float(row["pullback_depth_pct"]),
        support_touch_pct=float(row["support_touch_pct"]),
        distance_from_vwap=float(row["distance_from_vwap"]),
        compression_ratio=float(row["compression_ratio"]),
        reclaim_pct=float(row["breakout_pct"]),
        close_location=float(row["close_location"]),
        volume_ratio=float(row["volume_ratio"]),
        atr_pct=float(row["atr_pct"]),
        score=float(row["signal_score"]),
        signal_type=f"{construction}_long",
        component_tags=component_tags,
        gate_trend_strength=float(row["gate_trend_strength_60"]),
        gate_momentum_medium=float(row["gate_momentum_medium_60"]),
    )


class TrendGateEnsembleStrategy:
    def __init__(
        self,
        pair: str,
        config: StrategyConfig = DEFAULT_CONFIG,
        construction: str = DEFAULT_ENSEMBLE_CONSTRUCTION,
    ):
        self.pair = pair
        self.config = config
        self.construction = construction
        self.last_signal_bar = -config.cooldown_bars - 1
        self.trade: Optional[Trade] = None
        self.min_master_bars = max(96, config.warmup_bars)

    def detect(self, df: pd.DataFrame) -> Optional[Signal]:
        if len(df) < self.min_master_bars:
            return None

        full_bar_idx = len(df) - 1
        base = df.tail(ENSEMBLE_LOOKBACK_BARS).reset_index(drop=True)
        if self.trade and self.trade.is_open:
            return None
        if full_bar_idx - self.last_signal_bar <= self.config.cooldown_bars:
            return None

        frame = build_ensemble_frame(base, construction=self.construction, config=self.config)
        if frame.empty or not bool(frame.iloc[-1].get("entry_signal", False)):
            return None

        return build_ensemble_signal(
            self.pair,
            frame,
            row_idx=len(frame) - 1,
            construction=self.construction,
            bar_idx=full_bar_idx,
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


HourlyBreakoutStrategy = TrendGateEnsembleStrategy
PullbackTrendStrategy = TrendGateEnsembleStrategy
