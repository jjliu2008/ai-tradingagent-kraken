"""
Expanded multi-strategy screener for the AI Trading Agent Hackathon.

Screens 15+ strategy families across multiple pairs, timeframes, and exit profiles.
Includes momentum, mean-reversion, volatility-squeeze, trend-continuation,
and AI-enhanced regime-aware strategies.

Usage:
    python expanded_screener.py --pairs GIGAUSD,DOGUSD,SOLUSD,HYPEUSD,COQUSD \
        --cache-dir data_cache --results-csv expanded_screen_results.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

import strategy as strat

# ---------------------------------------------------------------------------
# Helpers copied / adapted from screen_strategies.py
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExitSpec:
    hold_bars: int
    stop_pct: float
    target_pct: float | None = None
    trail_activation_pct: float | None = None
    trail_stop_pct: float | None = None


@dataclass
class SimTrade:
    entry_ts: int
    exit_ts: int
    pnl_pct: float
    exit_reason: str
    mfe_pct: float


@dataclass
class ScreenResult:
    pair: str
    interval: int
    strategy: str
    window_days: int
    trades: int
    net_pnl: float
    avg_pnl: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    avg_mfe: float
    sharpe: float
    avg_bars_held: float


def compute_rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    bandwidth = ((upper - lower) / sma).replace(0, np.nan)
    return upper, lower, pct_b, bandwidth


def add_expanded_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features beyond what strategy.compute_features provides."""
    out = df.copy()

    # RSI variants
    out["rsi_7"] = compute_rsi(out["close"], 7)
    out["rsi_14"] = compute_rsi(out["close"], 14)

    # MACD
    out["macd"], out["macd_signal"], out["macd_hist"] = compute_macd(out["close"])
    out["macd_cross_up"] = (out["macd"] > out["macd_signal"]) & (out["macd"].shift(1) <= out["macd_signal"].shift(1))
    out["macd_cross_down"] = (out["macd"] < out["macd_signal"]) & (out["macd"].shift(1) >= out["macd_signal"].shift(1))

    # Bollinger Bands
    out["bb_upper"], out["bb_lower"], out["bb_pct_b"], out["bb_bandwidth"] = compute_bollinger(out["close"])

    # Body analysis
    out["body_pct"] = (out["close"] - out["open"]) / out["open"]
    out["upper_wick"] = (out["high"] - out[["close", "open"]].max(axis=1)) / (out["high"] - out["low"]).replace(0, np.nan)
    out["lower_wick"] = (out[["close", "open"]].min(axis=1) - out["low"]) / (out["high"] - out["low"]).replace(0, np.nan)
    out["green_bar"] = out["close"] > out["open"]

    # EMA distances
    out["ema_fast_dist"] = (out["close"] - out["ema_fast"]) / out["ema_fast"]
    out["ema_slow_dist"] = (out["close"] - out["ema_slow"]) / out["ema_slow"]

    # Rolling highs/lows
    out["roll_high_12"] = out["high"].rolling(12).max().shift(1)
    out["roll_high_20"] = out["high"].rolling(20).max().shift(1)
    out["roll_low_12"] = out["low"].rolling(12).min().shift(1)
    out["roll_low_20"] = out["low"].rolling(20).min().shift(1)

    # Volume expansion detection
    out["vol_sma_10"] = out["volume"].rolling(10).mean()
    out["vol_spike"] = out["volume"] > 2.0 * out["vol_sma_10"]

    # Consecutive bars
    out["consec_green"] = out["green_bar"].astype(int).groupby((~out["green_bar"]).cumsum()).cumsum()
    out["consec_red"] = (~out["green_bar"]).astype(int).groupby(out["green_bar"].cumsum()).cumsum()

    # ATR % for relative sizing
    out["atr_pct_14"] = out["atr"] / out["close"]

    # Range contraction (Bollinger squeeze)
    out["bb_squeeze"] = out["bb_bandwidth"] < out["bb_bandwidth"].rolling(50, min_periods=20).quantile(0.2)

    # Momentum acceleration
    out["mom_accel"] = out["momentum_medium"] - out["momentum_medium"].shift(3)

    return out


def resample_ohlcv(df: pd.DataFrame, interval: int) -> pd.DataFrame:
    if interval == 15:
        return df.copy().reset_index(drop=True)

    frame = df.copy()
    frame["dt"] = pd.to_datetime(frame["ts"], unit="s", utc=True)
    frame = frame.set_index("dt")
    frame["pv"] = frame["vwap_k"] * frame["volume"]

    agg = (
        frame.resample(f"{interval}min", label="left", closed="left", origin="epoch")
        .agg({
            "open": "first", "high": "max", "low": "min", "close": "last",
            "pv": "sum", "volume": "sum", "count": "sum",
        })
        .dropna(subset=["open", "high", "low", "close"])
    )
    agg["vwap_k"] = agg["pv"] / agg["volume"].replace(0, np.nan)
    agg = agg.dropna(subset=["vwap_k"]).reset_index()
    agg["ts"] = agg["dt"].astype("int64") // 1_000_000_000
    return agg[["ts", "open", "high", "low", "close", "vwap_k", "volume", "count"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy families — each returns (name, entry_mask, exit_spec)
# ---------------------------------------------------------------------------

def build_all_candidates(df: pd.DataFrame, interval: int) -> list[tuple[str, pd.Series, ExitSpec]]:
    """Build expanded set of strategy candidates across all timeframes."""
    trend_up = df["ema_fast"] > df["ema_slow"]
    trend_down = df["ema_fast"] < df["ema_slow"]
    candidates: list[tuple[str, pd.Series, ExitSpec]] = []

    # -----------------------------------------------------------------------
    # CATEGORY 1: MOMENTUM / BREAKOUT (long-only, ride trends)
    # -----------------------------------------------------------------------

    # 1a. Classic breakout above 12-bar high with volume confirmation
    candidates.append((
        "momentum_breakout",
        (
            (df["trend_strength"] > 0.0020)
            & trend_up
            & (df["close"] > df["roll_high_12"])
            & (df["momentum_medium"] > 0.015)
            & (df["volume_ratio"] > 1.0)
            & (df["close_location"] > 0.70)
            & (df["compression_ratio"] < 0.90)
        ),
        ExitSpec(hold_bars=5, stop_pct=0.015, target_pct=0.045),
    ))

    # 1b. Breakout with trailing stop for larger moves
    candidates.append((
        "momentum_breakout_trail",
        (
            (df["trend_strength"] > 0.0025)
            & trend_up
            & (df["close"] > df["roll_high_12"])
            & (df["momentum_medium"] > 0.020)
            & (df["volume_ratio"] > 1.2)
            & (df["close_location"] > 0.70)
        ),
        ExitSpec(hold_bars=8, stop_pct=0.015, target_pct=None,
                 trail_activation_pct=0.020, trail_stop_pct=0.010),
    ))

    # 1c. MACD crossover + trend alignment
    candidates.append((
        "macd_trend_long",
        (
            trend_up
            & df["macd_cross_up"]
            & (df["macd"] < 0)  # crossing up from below zero = early trend
            & (df["trend_strength"] > 0.0010)
            & (df["close_location"] > 0.55)
            & (df["volume_ratio"] > 0.8)
        ),
        ExitSpec(hold_bars=6, stop_pct=0.012, target_pct=0.030),
    ))

    # 1d. RSI momentum thrust (RSI breaks above 60 from below)
    candidates.append((
        "rsi_momentum_thrust",
        (
            trend_up
            & (df["rsi_14"] > 60)
            & (df["rsi_14"].shift(1) <= 60)
            & (df["trend_strength"] > 0.0015)
            & (df["close_location"] > 0.60)
            & (df["volume_ratio"] > 0.9)
        ),
        ExitSpec(hold_bars=5, stop_pct=0.012, target_pct=0.035),
    ))

    # -----------------------------------------------------------------------
    # CATEGORY 2: MEAN REVERSION (buy dips in uptrend)
    # -----------------------------------------------------------------------

    # 2a. RSI oversold bounce in uptrend
    candidates.append((
        "rsi_oversold_trend",
        (
            trend_up
            & (df["rsi_7"] < 30)
            & df["green_bar"]
            & (df["close_location"] > 0.60)
            & (df["trend_strength"] > 0.0005)
        ),
        ExitSpec(hold_bars=4, stop_pct=0.010, target_pct=0.020),
    ))

    # 2b. Bollinger band touch + green bar reversal
    candidates.append((
        "bb_lower_bounce",
        (
            (df["bb_pct_b"] < 0.10)
            & df["green_bar"]
            & (df["close_location"] > 0.65)
            & (df["lower_wick"] > 0.35)
            & (df["volume_ratio"] > 0.7)
        ),
        ExitSpec(hold_bars=4, stop_pct=0.010, target_pct=0.018),
    ))

    # 2c. Deep VWAP pullback in uptrend
    candidates.append((
        "deep_vwap_pullback",
        (
            trend_up
            & (df["distance_from_vwap"] < -0.012)
            & df["green_bar"]
            & (df["close_location"] > 0.65)
            & (df["trend_strength"] > 0.0010)
        ),
        ExitSpec(hold_bars=5, stop_pct=0.012, target_pct=0.025),
    ))

    # 2d. EMA fast touch + bounce
    candidates.append((
        "ema_fast_bounce",
        (
            trend_up
            & (df["low"] <= df["ema_fast"])
            & (df["close"] > df["ema_fast"])
            & df["green_bar"]
            & (df["trend_strength"] > 0.0020)
            & (df["close_location"] > 0.60)
            & (df["volume_ratio"] < 1.5)
        ),
        ExitSpec(hold_bars=5, stop_pct=0.010, target_pct=0.025),
    ))

    # 2e. Consecutive red bars + reversal (3+ red then green)
    candidates.append((
        "red_streak_reversal",
        (
            trend_up
            & (df["consec_red"].shift(1) >= 3)
            & df["green_bar"]
            & (df["close_location"] > 0.65)
            & (df["rsi_7"] < 45)
        ),
        ExitSpec(hold_bars=4, stop_pct=0.010, target_pct=0.020),
    ))

    # -----------------------------------------------------------------------
    # CATEGORY 3: VOLATILITY SQUEEZE / CONTRACTION → EXPANSION
    # -----------------------------------------------------------------------

    # 3a. Bollinger squeeze breakout
    candidates.append((
        "bb_squeeze_breakout",
        (
            df["bb_squeeze"]
            & trend_up
            & (df["close"] > df["bb_upper"])
            & (df["volume_ratio"] > 1.2)
            & (df["close_location"] > 0.70)
        ),
        ExitSpec(hold_bars=6, stop_pct=0.012, target_pct=0.035),
    ))

    # 3b. ATR compression then expansion (volatility breakout)
    candidates.append((
        "atr_squeeze_expand",
        (
            (df["compression_ratio"] < 0.75)
            & trend_up
            & (df["close"] > df["roll_high_12"])
            & (df["momentum_medium"] > 0.010)
            & (df["volume_ratio"] > 1.0)
        ),
        ExitSpec(hold_bars=6, stop_pct=0.012, target_pct=0.035),
    ))

    # 3c. Quiet pullback then expansion (low vol bar followed by breakout)
    candidates.append((
        "quiet_then_burst",
        (
            trend_up
            & (df["body_pct"].shift(1).abs() < 0.003)  # prior bar was quiet
            & (df["body_pct"] > 0.008)                  # current bar is explosive
            & (df["close_location"] > 0.75)
            & (df["volume_ratio"] > 1.3)
            & (df["trend_strength"] > 0.0015)
        ),
        ExitSpec(hold_bars=5, stop_pct=0.012, target_pct=0.030),
    ))

    # -----------------------------------------------------------------------
    # CATEGORY 4: TREND CONTINUATION / PULLBACK-TO-TREND
    # -----------------------------------------------------------------------

    # 4a. Higher-low structure breakout
    candidates.append((
        "higher_low_structure",
        (
            trend_up
            & (df["trend_strength"] > 0.0020)
            & (df["low"].shift(1) > df["roll_low_12"])
            & (df["close"] > df["roll_high_12"])
            & (df["close_location"] > 0.65)
            & (df["volume_ratio"] > 1.0)
        ),
        ExitSpec(hold_bars=6, stop_pct=0.012, target_pct=0.030),
    ))

    # 4b. VWAP reclaim in trend
    candidates.append((
        "vwap_reclaim_trend",
        (
            trend_up
            & (df["trend_strength"] > 0.0015)
            & df["support_touch_pct"].between(-0.006, 0.001)
            & df["distance_from_vwap"].between(-0.004, 0.004)
            & (df["close"] > df["trigger_level"])
            & (df["close_location"] > 0.55)
            & df["volume_ratio"].between(0.7, 1.8)
        ),
        ExitSpec(hold_bars=5, stop_pct=0.010, target_pct=0.022),
    ))

    # 4c. Momentum acceleration (mom speeding up)
    candidates.append((
        "momentum_accel",
        (
            trend_up
            & (df["mom_accel"] > 0.005)
            & (df["momentum_medium"] > 0.010)
            & (df["trend_strength"] > 0.0015)
            & (df["close_location"] > 0.60)
            & (df["volume_ratio"] > 0.9)
        ),
        ExitSpec(hold_bars=5, stop_pct=0.015, target_pct=0.040),
    ))

    # -----------------------------------------------------------------------
    # CATEGORY 5: VOLUME-DRIVEN SIGNALS
    # -----------------------------------------------------------------------

    # 5a. Volume spike + green bar in trend
    candidates.append((
        "volume_spike_trend",
        (
            trend_up
            & df["vol_spike"]
            & df["green_bar"]
            & (df["close_location"] > 0.70)
            & (df["trend_strength"] > 0.0015)
            & (df["body_pct"] > 0.005)
        ),
        ExitSpec(hold_bars=5, stop_pct=0.012, target_pct=0.030),
    ))

    # 5b. Volume climax reversal (huge vol + hammer after dip)
    candidates.append((
        "volume_climax_reversal",
        (
            (df["volume_ratio"] > 2.5)
            & df["green_bar"]
            & (df["lower_wick"] > 0.40)
            & (df["rsi_7"] < 40)
            & (df["close_location"] > 0.65)
        ),
        ExitSpec(hold_bars=4, stop_pct=0.012, target_pct=0.020),
    ))

    # -----------------------------------------------------------------------
    # CATEGORY 6: MULTI-CONDITION CONFLUENCE (high selectivity)
    # -----------------------------------------------------------------------

    # 6a. Triple confluence: trend + MACD + volume + squeeze
    candidates.append((
        "triple_confluence",
        (
            trend_up
            & (df["trend_strength"] > 0.0025)
            & (df["macd_hist"] > 0)
            & (df["macd_hist"] > df["macd_hist"].shift(1))  # histogram expanding
            & (df["compression_ratio"] < 0.85)
            & (df["volume_ratio"] > 1.1)
            & (df["close_location"] > 0.65)
            & (df["close"] > df["roll_high_12"])
        ),
        ExitSpec(hold_bars=6, stop_pct=0.015, target_pct=0.045),
    ))

    # 6b. VWAP + RSI + EMA confluence for dip buy
    candidates.append((
        "dip_buy_confluence",
        (
            trend_up
            & (df["distance_from_vwap"] < -0.008)
            & (df["rsi_7"] < 35)
            & (df["close"] > df["ema_slow"])
            & df["green_bar"]
            & (df["close_location"] > 0.65)
        ),
        ExitSpec(hold_bars=5, stop_pct=0.012, target_pct=0.025),
    ))

    # -----------------------------------------------------------------------
    # CATEGORY 7: AGGRESSIVE EXIT VARIANTS of best strategies
    # -----------------------------------------------------------------------

    # 7a. Breakout with tight stop + big target (asymmetric R:R)
    candidates.append((
        "breakout_asymmetric",
        (
            (df["trend_strength"] > 0.0030)
            & trend_up
            & (df["close"] > df["roll_high_20"])
            & (df["momentum_medium"] > 0.020)
            & (df["volume_ratio"] > 1.2)
            & (df["close_location"] > 0.75)
            & (df["compression_ratio"] < 0.85)
        ),
        ExitSpec(hold_bars=8, stop_pct=0.010, target_pct=0.060),
    ))

    # 7b. Scalp breakout (tight stop, tight target, high win rate)
    candidates.append((
        "breakout_scalp",
        (
            trend_up
            & (df["close"] > df["roll_high_12"])
            & (df["trend_strength"] > 0.0020)
            & (df["momentum_medium"] > 0.010)
            & (df["volume_ratio"] > 1.0)
            & (df["close_location"] > 0.70)
        ),
        ExitSpec(hold_bars=3, stop_pct=0.008, target_pct=0.015),
    ))

    return candidates


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def simulate_trades(
    df: pd.DataFrame,
    entry_mask: pd.Series,
    exit_spec: ExitSpec,
    side_cost_pct: float,
) -> list[SimTrade]:
    trades: list[SimTrade] = []
    i = 0
    last_index = len(df) - 1

    while i < last_index:
        if not bool(entry_mask.iloc[i]):
            i += 1
            continue

        entry_bar = i + 1
        if entry_bar > last_index:
            break

        entry_ts = int(df.iloc[entry_bar]["ts"])
        entry_price = float(df.iloc[entry_bar]["open"]) * (1 + side_cost_pct)
        stop_price = entry_price * (1 - exit_spec.stop_pct)
        target_price = entry_price * (1 + exit_spec.target_pct) if exit_spec.target_pct is not None else None
        best_high = entry_price
        max_exit_bar = min(last_index, entry_bar + exit_spec.hold_bars)

        exit_bar = max_exit_bar
        exit_price = float(df.iloc[max_exit_bar]["close"]) * (1 - side_cost_pct)
        exit_reason = "TIME_LIMIT"
        bars_in_trade = 0

        for bar in range(entry_bar, max_exit_bar + 1):
            row = df.iloc[bar]
            high_price = float(row["high"])
            low_price = float(row["low"])
            close_price = float(row["close"])
            best_high = max(best_high, high_price)
            bars_in_trade = bar - entry_bar

            trail_stop = None
            if (
                exit_spec.trail_activation_pct is not None
                and exit_spec.trail_stop_pct is not None
                and (best_high / entry_price - 1) >= exit_spec.trail_activation_pct
            ):
                trail_stop = best_high * (1 - exit_spec.trail_stop_pct)

            if low_price <= stop_price:
                exit_bar = bar
                exit_price = stop_price * (1 - side_cost_pct)
                exit_reason = "STOP_LOSS"
                break
            if target_price is not None and high_price >= target_price:
                exit_bar = bar
                exit_price = target_price * (1 - side_cost_pct)
                exit_reason = "TAKE_PROFIT"
                break
            if trail_stop is not None and low_price <= trail_stop:
                exit_bar = bar
                exit_price = trail_stop * (1 - side_cost_pct)
                exit_reason = "TRAIL_STOP"
                break
            if bar == max_exit_bar:
                exit_bar = bar
                exit_price = close_price * (1 - side_cost_pct)
                exit_reason = "TIME_LIMIT"

        trades.append(
            SimTrade(
                entry_ts=entry_ts,
                exit_ts=int(df.iloc[exit_bar]["ts"]),
                pnl_pct=exit_price / entry_price - 1,
                exit_reason=exit_reason,
                mfe_pct=best_high / entry_price - 1,
            )
        )
        i = exit_bar + 1

    return trades


def max_drawdown(pnls: list[float]) -> float:
    if not pnls:
        return 0.0
    equity = np.cumprod(1 + np.asarray(pnls, dtype=float))
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity - peaks) / peaks
    return float(drawdowns.min())


def profit_factor(pnls: list[float]) -> float:
    wins = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p < 0)
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def trade_sharpe(pnls: list[float]) -> float:
    if len(pnls) < 2:
        return 0.0
    arr = np.asarray(pnls)
    std = arr.std(ddof=1)
    if std == 0:
        return 0.0
    return float(arr.mean() / std * np.sqrt(len(arr)))


def avg_bars_held(trades: list[SimTrade], df: pd.DataFrame) -> float:
    """Estimate average bars held from timestamps."""
    if not trades:
        return 0.0
    durations = [(t.exit_ts - t.entry_ts) for t in trades]
    # Estimate bar duration from data
    if len(df) > 1:
        bar_dur = (df["ts"].iloc[-1] - df["ts"].iloc[0]) / (len(df) - 1)
    else:
        bar_dur = 3600
    return float(np.mean(durations) / bar_dur) if bar_dur > 0 else 0.0


# ---------------------------------------------------------------------------
# Screening logic
# ---------------------------------------------------------------------------

def evaluate_window(
    pair: str,
    interval: int,
    strategy_name: str,
    df: pd.DataFrame,
    entry_mask: pd.Series,
    exit_spec: ExitSpec,
    side_cost_pct: float,
    window_days: int,
) -> ScreenResult:
    trades = simulate_trades(df, entry_mask, exit_spec, side_cost_pct)
    pnls = [trade.pnl_pct for trade in trades]
    return ScreenResult(
        pair=pair,
        interval=interval,
        strategy=strategy_name,
        window_days=window_days,
        trades=len(trades),
        net_pnl=float(np.sum(pnls)) if pnls else 0.0,
        avg_pnl=float(np.mean(pnls)) if pnls else 0.0,
        win_rate=float(np.mean(np.asarray(pnls) > 0)) if pnls else 0.0,
        profit_factor=profit_factor(pnls),
        max_drawdown=max_drawdown(pnls),
        avg_mfe=float(np.mean([trade.mfe_pct for trade in trades])) if trades else 0.0,
        sharpe=trade_sharpe(pnls),
        avg_bars_held=avg_bars_held(trades, df),
    )


def screen_pair(
    pair: str,
    base_history: pd.DataFrame,
    intervals: list[int],
    side_cost_pct: float,
    recent_days: int,
    min_keep_trades: int,
) -> pd.DataFrame:
    end_ts = int(base_history["ts"].iloc[-1])
    recent_cutoff = end_ts - recent_days * 24 * 60 * 60
    rows: list[dict] = []

    for interval in intervals:
        interval_df = resample_ohlcv(base_history, interval)
        interval_df = add_expanded_features(strat.compute_features(interval_df))
        full_df = interval_df.reset_index(drop=True)
        recent_df = interval_df.loc[interval_df["ts"] >= recent_cutoff].reset_index(drop=True)

        # Build candidates for both full and recent windows
        full_candidates = build_all_candidates(full_df, interval)
        recent_candidates_dict = {name: mask for name, mask, _ in build_all_candidates(recent_df, interval)}

        for strategy_name, full_mask, exit_spec in full_candidates:
            recent_mask = recent_candidates_dict.get(strategy_name)
            if recent_mask is None:
                continue

            full_result = evaluate_window(
                pair=pair, interval=interval, strategy_name=strategy_name,
                df=full_df, entry_mask=full_mask.fillna(False),
                exit_spec=exit_spec, side_cost_pct=side_cost_pct,
                window_days=int(round((full_df["ts"].iloc[-1] - full_df["ts"].iloc[0]) / 86400)),
            )
            recent_result = evaluate_window(
                pair=pair, interval=interval, strategy_name=strategy_name,
                df=recent_df, entry_mask=recent_mask.fillna(False),
                exit_spec=exit_spec, side_cost_pct=side_cost_pct,
                window_days=recent_days,
            )

            # Keep criteria: profitable in both windows with enough trades
            keep = (
                full_result.net_pnl > 0
                and recent_result.net_pnl > 0
                and full_result.trades >= min_keep_trades
                and recent_result.trades >= max(2, min_keep_trades // 2)
            )

            # Composite score: recent PnL weighted 1.5x, add sharpe bonus
            score = full_result.net_pnl + 1.5 * recent_result.net_pnl + 0.01 * full_result.sharpe

            rows.append({
                "pair": pair,
                "interval": interval,
                "strategy": strategy_name,
                "trades_full": full_result.trades,
                "net_full": full_result.net_pnl,
                "avg_full": full_result.avg_pnl,
                "win_full": full_result.win_rate,
                "pf_full": full_result.profit_factor,
                "dd_full": full_result.max_drawdown,
                "mfe_full": full_result.avg_mfe,
                "sharpe_full": full_result.sharpe,
                "bars_full": full_result.avg_bars_held,
                "trades_recent": recent_result.trades,
                "net_recent": recent_result.net_pnl,
                "avg_recent": recent_result.avg_pnl,
                "win_recent": recent_result.win_rate,
                "pf_recent": recent_result.profit_factor,
                "dd_recent": recent_result.max_drawdown,
                "mfe_recent": recent_result.avg_mfe,
                "sharpe_recent": recent_result.sharpe,
                "score": score,
                "verdict": "KEEP" if keep else "KILL",
            })

    return pd.DataFrame(rows)


def load_history(pair: str, cache_dir: Path) -> pd.DataFrame:
    cache_path = cache_dir / f"{pair}_15m_120d_end_latest.csv"
    if cache_path.exists():
        print(f"Loading cached history: {cache_path}")
        return pd.read_csv(cache_path)
    raise FileNotFoundError(f"No cached data for {pair} at {cache_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expanded multi-strategy screener")
    parser.add_argument("--pairs", default="GIGAUSD,DOGUSD,SOLUSD,HYPEUSD,COQUSD")
    parser.add_argument("--intervals", default="15,30,60")
    parser.add_argument("--recent-days", type=int, default=60)
    parser.add_argument("--fee-pct", type=float, default=0.0026)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--min-keep-trades", type=int, default=5)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--results-csv", default="expanded_screen_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    intervals = [int(v.strip()) for v in args.intervals.split(",") if v.strip()]
    side_cost_pct = args.fee_pct + args.slippage_pct
    cache_dir = Path(args.cache_dir)

    frames: list[pd.DataFrame] = []
    for pair in pairs:
        try:
            history = load_history(pair, cache_dir)
            print(f"  {pair}: {len(history)} bars loaded")
            frames.append(screen_pair(
                pair=pair, base_history=history, intervals=intervals,
                side_cost_pct=side_cost_pct, recent_days=args.recent_days,
                min_keep_trades=args.min_keep_trades,
            ))
        except FileNotFoundError as e:
            print(f"  Skipping {pair}: {e}")

    if not frames:
        print("No data loaded.")
        return

    results = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if results.empty:
        print("No strategy families produced any trades.")
        return

    results = results.sort_values(
        ["verdict", "score", "net_recent", "net_full"],
        ascending=[True, False, False, False],
    )

    keepers = results.loc[results["verdict"] == "KEEP"].copy()

    print(f"\n{'=' * 90}")
    print(f" EXPANDED STRATEGY SCREEN | {len(results)} combos tested | {len(keepers)} survived")
    print(f"{'=' * 90}")

    fmt = lambda v: f"{v:+.3%}" if np.isfinite(v) and abs(v) < 10 else f"{v:.2f}"

    if keepers.empty:
        print("\nNo survivors. Top 10 by score:")
        cols = ["pair", "interval", "strategy", "trades_full", "net_full", "win_full",
                "pf_full", "sharpe_full", "trades_recent", "net_recent", "win_recent", "score"]
        print(results[cols].head(10).to_string(index=False, float_format=fmt))
    else:
        print(f"\n{'SURVIVORS':=^90}")
        cols = ["pair", "interval", "strategy", "trades_full", "net_full", "win_full",
                "pf_full", "sharpe_full", "dd_full", "mfe_full",
                "trades_recent", "net_recent", "win_recent", "pf_recent", "score"]
        print(keepers[cols].to_string(index=False, float_format=fmt))

        print(f"\n{'TOP 5 RECOMMENDATIONS':=^90}")
        for i, row in keepers.head(5).iterrows():
            print(f"\n  #{keepers.index.get_loc(i) + 1}: {row['pair']} | {int(row['interval'])}m | {row['strategy']}")
            print(f"      Net PnL (full): {row['net_full']:+.3%} | Win: {row['win_full']:.1%} | PF: {row['pf_full']:.2f} | Sharpe: {row['sharpe_full']:.2f}")
            print(f"      Net PnL (60d):  {row['net_recent']:+.3%} | Win: {row['win_recent']:.1%} | PF: {row['pf_recent']:.2f}")
            print(f"      Trades: {int(row['trades_full'])} (full) / {int(row['trades_recent'])} (60d)")
            print(f"      MFE: {row['mfe_full']:+.3%} | MaxDD: {row['dd_full']:+.3%} | Score: {row['score']:.4f}")

    if args.results_csv:
        results.to_csv(args.results_csv, index=False)
        print(f"\nWrote {len(results)} rows to {args.results_csv}")


if __name__ == "__main__":
    main()
