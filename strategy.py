"""
Overextension short strategy for crypto — adapted from ES live_candidate_v1.

Original (ES futures, 1-second bars):
  Signal:  range_pos >= 0.85  AND  vwap_dev_ticks >= 50  (top-of-range + far above VWAP)
  Cluster: first signal only per 15-minute cluster
  Entry:   delayed 1 bar (next second)
  Gates:   no power-hour  |  spread <= 1 tick  |  signedvol_sum30s <= 250 (or <= 100 at cash open)
  Exit:    fail-fast in 3 minutes if [no +1-tick progress AND buy-flow > thresh AND OBI bullish]
           else hold up to 30 minutes

Crypto adaptations (1-minute bars):
  - VWAP dev expressed as % instead of ticks (0.25% ≈ ES 50-tick threshold at ~5000)
  - Signed volume: OHLC-approximated (volume × sign(close - open)) per bar
  - Spread: % of mid instead of ticks
  - No power-hour exclusion (24/7 market); use a "first hour" gate instead
  - Cluster gap: 15 bars (minutes) | Fail-fast window: 3 bars (minutes)
  - Signed vol sum: 30 bars = 30 minutes (same concept as 30s at 1s bars)
  - SV gate normalized as a rolling ratio (avoids absolute-volume calibration to BTC)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

RANGE_POS_MIN = 0.85          # price must be in top 15% of session range
VWAP_DEV_PCT_MIN = 0.0025     # 0.25% above session VWAP  (ES equiv: 50 ticks / ~5000)
SPREAD_PCT_MAX = 0.0002       # 0.02% spread gate         (ES equiv: 1 tick / ~5000)

SV_WINDOW = 30                # rolling window (bars) for signed-volume sum
SV_RATIO_MAX = 0.50           # net-buy ratio gate: cooldown required below this
SV_OPEN_RATIO_MAX = 0.20      # stricter gate for "first-hour" bars (like cash_open)
FIRST_HOUR_BARS = 60          # bars treated as session open (first 60 minutes)

CLUSTER_GAP_BARS = 15         # min bars between independent signal clusters
HOLD_BARS = 30                # max hold time in bars before forced exit

FAIL_FAST_BARS = 3            # bars in which fail-fast can trigger
FAIL_FAST_MIN_PROGRESS = 0.001 # 0.1% favorable MFE needed to avoid fail-fast
FAIL_FAST_ADVERSE_OBI = 0.03   # OBI delta threshold (bid-heavy against a short)
FAIL_FAST_ADVERSE_SV_RATIO = 0.0  # net-buy ratio that counts as adverse for short


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Bar:
    """Single OHLCV bar with computed features."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    vwap_kraken: float   # exchange-reported VWAP (used for reference only)
    volume: float

    # computed
    mid: float = 0.0
    signed_vol: float = 0.0
    sv_sum30: float = 0.0
    sv_ratio30: float = 0.0
    session_vwap: float = 0.0
    vwap_dev_pct: float = 0.0
    session_high: float = 0.0
    session_low: float = 0.0
    range_pos: float = 0.0


@dataclass
class Signal:
    bar_idx: int
    price: float
    range_pos: float
    vwap_dev_pct: float
    sv_ratio30: float
    spread_pct: float
    obi: float
    side: int = -1   # always short (-1)

    def describe(self) -> str:
        return (
            f"SHORT signal at ${self.price:,.2f} | "
            f"range_pos={self.range_pos:.2f} | "
            f"vwap_dev={self.vwap_dev_pct:.3%} | "
            f"sv_ratio={self.sv_ratio30:.2f} | "
            f"spread={self.spread_pct:.4%} | "
            f"obi={self.obi:.3f}"
        )


@dataclass
class Trade:
    entry_bar: int
    entry_price: float
    side: int = -1
    volume: float = 0.0
    exit_bar: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    # MFE tracking
    best_pct: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.exit_bar is None

    def bars_held(self, current_bar: int) -> int:
        return current_bar - self.entry_bar

    def pnl_pct(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        return self.side * (self.exit_price - self.entry_price) / self.entry_price

    def update_mfe(self, current_price: float) -> None:
        favorable = self.side * (self.entry_price - current_price) / self.entry_price
        self.best_pct = max(self.best_pct, favorable)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def parse_ohlc(raw: dict) -> pd.DataFrame:
    """
    Parse Kraken CLI ohlc JSON response.

    raw: {'XXBTZUSD': [[ts, o, h, l, c, vwap, vol, count], ...], 'last': ...}
    Returns DataFrame sorted oldest→newest.
    """
    pair_key = next(k for k in raw if k != "last")
    bars = raw[pair_key]
    df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "vwap_k", "volume", "count"])
    for col in ["open", "high", "low", "close", "vwap_k", "volume"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def compute_features(df: pd.DataFrame, session_start_bar: int = 0) -> pd.DataFrame:
    """
    Add strategy features to an OHLC DataFrame.

    session_start_bar: index of the first bar of the current session window.
                       Features reset from this point (VWAP, range, etc.).
    """
    out = df.copy()

    # Mid price
    out["mid"] = out["close"]

    # Signed volume: positive = net buying (close > open), negative = net selling
    out["signed_vol"] = out["volume"] * np.sign(out["close"] - out["open"])

    # Rolling 30-bar sums and ratio (session-scoped)
    total_vol_sum = out["volume"].rolling(SV_WINDOW, min_periods=5).sum()
    sv_sum = out["signed_vol"].rolling(SV_WINDOW, min_periods=5).sum()
    out["sv_sum30"] = sv_sum
    out["sv_ratio30"] = sv_sum / (total_vol_sum + 1e-12)

    # Session VWAP — cumulative from session_start_bar
    session = out.iloc[session_start_bar:].copy()
    cum_pv = (session["close"] * session["volume"]).cumsum()
    cum_vol = session["volume"].cumsum()
    session_vwap = cum_pv / cum_vol.replace(0, np.nan)
    out.loc[session_start_bar:, "session_vwap"] = session_vwap.values
    out["session_vwap"] = out["session_vwap"].ffill()

    # VWAP deviation %
    out["vwap_dev_pct"] = (out["close"] - out["session_vwap"]) / out["session_vwap"]

    # Session range — cumulative from session_start_bar
    session_hi = out.iloc[session_start_bar:]["high"].cummax()
    session_lo = out.iloc[session_start_bar:]["low"].cummin()
    out.loc[session_start_bar:, "session_high"] = session_hi.values
    out.loc[session_start_bar:, "session_low"] = session_lo.values
    out[["session_high", "session_low"]] = out[["session_high", "session_low"]].ffill()

    session_range = (out["session_high"] - out["session_low"]).replace(0, np.nan)
    out["range_pos"] = (out["close"] - out["session_low"]) / session_range

    return out


def compute_obi(orderbook: dict) -> tuple[float, float]:
    """
    Compute order book imbalance and spread % from raw orderbook response.

    Returns (obi, spread_pct) where obi = bid_vol / (bid_vol + ask_vol) for top 5 levels.
    """
    pair_key = next(iter(orderbook))
    ob = orderbook[pair_key]
    bids = [(float(p), float(v)) for p, v, *_ in ob["bids"][:5]]
    asks = [(float(p), float(v)) for p, v, *_ in ob["asks"][:5]]
    bid_vol = sum(v for _, v in bids)
    ask_vol = sum(v for _, v in asks)
    total = bid_vol + ask_vol
    obi = bid_vol / total if total > 0 else 0.5

    best_bid = bids[0][0] if bids else 0.0
    best_ask = asks[0][0] if asks else 0.0
    mid = (best_bid + best_ask) / 2
    spread_pct = (best_ask - best_bid) / mid if mid > 0 else 0.0

    return obi, spread_pct


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------

class OverextensionStrategy:
    """
    Stateful wrapper around the overextension-short logic.

    Maintains cluster cooldown state and current open trade.
    """

    def __init__(self):
        self.last_signal_bar: int = -CLUSTER_GAP_BARS - 1
        self.trade: Optional[Trade] = None

    def detect(
        self,
        df: pd.DataFrame,
        obi: float,
        spread_pct: float,
        session_start_bar: int = 0,
    ) -> Optional[Signal]:
        """
        Run signal detection on the latest bar of df.

        Returns Signal if all entry conditions are met, else None.
        df must already have features computed via compute_features().
        """
        if len(df) < SV_WINDOW + 1:
            return None

        bar_idx = len(df) - 1

        # Cluster cooldown
        if bar_idx - self.last_signal_bar < CLUSTER_GAP_BARS:
            return None

        last = df.iloc[-1]
        range_pos = float(last.get("range_pos", 0.0))
        vwap_dev_pct = float(last.get("vwap_dev_pct", 0.0))
        sv_ratio30 = float(last.get("sv_ratio30", 0.0))

        # Core overextension conditions
        if range_pos < RANGE_POS_MIN:
            return None
        if vwap_dev_pct < VWAP_DEV_PCT_MIN:
            return None

        # Spread gate
        if spread_pct > SPREAD_PCT_MAX:
            return None

        # Signed-volume cooling gate
        bars_from_session_start = bar_idx - session_start_bar
        is_first_hour = bars_from_session_start < FIRST_HOUR_BARS
        sv_max = SV_OPEN_RATIO_MAX if is_first_hour else SV_RATIO_MAX
        if sv_ratio30 > sv_max:
            return None  # buying still too aggressive — wait for it to cool

        return Signal(
            bar_idx=bar_idx,
            price=float(last["close"]),
            range_pos=range_pos,
            vwap_dev_pct=vwap_dev_pct,
            sv_ratio30=sv_ratio30,
            spread_pct=spread_pct,
            obi=obi,
        )

    def record_signal(self, bar_idx: int) -> None:
        """Call after a signal is detected to set cluster cooldown."""
        self.last_signal_bar = bar_idx

    def open_trade(self, bar_idx: int, price: float, volume: float) -> Trade:
        """Record a new open trade (entered 1 bar after signal)."""
        self.trade = Trade(
            entry_bar=bar_idx,
            entry_price=price,
            side=-1,
            volume=volume,
        )
        return self.trade

    def check_exit(self, df: pd.DataFrame, obi: float) -> Optional[str]:
        """
        Check whether the open trade should exit.

        Returns exit-reason string if exit is warranted, else None.
        """
        if self.trade is None or not self.trade.is_open:
            return None

        current_bar = len(df) - 1
        bars_held = self.trade.bars_held(current_bar)
        current_price = float(df.iloc[-1]["close"])

        # Update MFE
        self.trade.update_mfe(current_price)

        # Max hold
        if bars_held >= HOLD_BARS:
            return "TIME_30M"

        # Fail-fast window: first FAIL_FAST_BARS after entry
        if bars_held <= FAIL_FAST_BARS:
            no_progress = self.trade.best_pct < FAIL_FAST_MIN_PROGRESS

            # For a short: adverse flow = net buying resuming (sv_ratio > threshold)
            sv_ratio = float(df.iloc[-1].get("sv_ratio30", 0.0))
            adverse_flow = sv_ratio > FAIL_FAST_ADVERSE_SV_RATIO

            # Adverse OBI for short: bid-heavy (OBI > 0.5 + threshold)
            adverse_obi = obi > (0.5 + FAIL_FAST_ADVERSE_OBI)

            if no_progress and adverse_flow and adverse_obi:
                return "FAIL_FAST"

        return None

    def close_trade(self, bar_idx: int, price: float, reason: str) -> Trade:
        """Record trade exit."""
        if self.trade is None:
            raise RuntimeError("No open trade to close")
        self.trade.exit_bar = bar_idx
        self.trade.exit_price = price
        self.trade.exit_reason = reason
        return self.trade

    def describe_state(self, df: pd.DataFrame) -> str:
        """Return a human-readable summary of current market state for the last bar."""
        if df.empty:
            return "No data."
        last = df.iloc[-1]
        lines = [
            f"Price:     ${float(last['close']):,.2f}",
            f"Range pos: {float(last.get('range_pos', 0)):.2f}  (signal >= {RANGE_POS_MIN})",
            f"VWAP dev:  {float(last.get('vwap_dev_pct', 0)):.3%}  (signal >= {VWAP_DEV_PCT_MIN:.3%})",
            f"SV ratio:  {float(last.get('sv_ratio30', 0)):.2f}  (gate <= {SV_RATIO_MAX})",
        ]
        if self.trade and self.trade.is_open:
            pnl = self.trade.side * (float(last["close"]) - self.trade.entry_price) / self.trade.entry_price
            lines.append(f"Position:  SHORT @ ${self.trade.entry_price:,.2f}  |  PnL {pnl:.3%}  |  MFE {self.trade.best_pct:.3%}")
        return "\n".join(lines)
