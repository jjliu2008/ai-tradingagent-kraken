# tests/test_older60_backtest.py
from __future__ import annotations

import types
import pandas as pd
import pytest


def _make_fake_trade(entry_ts: int, pnl_pct: float, pair: str = "FAKEUSD"):
    """Return a minimal BacktestTrade-like namespace."""
    t = types.SimpleNamespace()
    t.pair = pair
    t.entry_ts = entry_ts
    t.pnl_pct = pnl_pct
    t.construction = "tc15_only"
    t.exit_ts = entry_ts + 900
    t.exit_reason = "TAKE_PROFIT"
    t.mfe_pct = pnl_pct
    t.bars_held = 1
    t.signal_score = 50.0
    t.trend_strength = 0.01
    t.pullback_depth_pct = 0.0
    t.distance_from_vwap = 0.01
    t.compression_ratio = 0.8
    t.entry_price = 1.0
    t.exit_price = 1.0 + pnl_pct
    t.entry_bar = 0
    t.exit_bar = 1
    return t


def test_summarize_splits_older_and_recent():
    """_summarize_with_split returns correct older60 / recent60 buckets."""
    from general_portfolio_backtest import _summarize_with_split

    now_ts = 1_800_000_000
    split_ts = now_ts - 60 * 24 * 60 * 60

    old_trade = _make_fake_trade(split_ts - 1000, 0.05)
    new_trade = _make_fake_trade(split_ts + 1000, -0.02)

    result = _summarize_with_split([old_trade, new_trade], split_ts)

    assert result["older60_trades"] == 1
    assert result["recent60_trades"] == 1
    assert abs(result["older60_net_pct"] - 0.05) < 1e-9
    assert abs(result["recent60_net_pct"] - (-0.02)) < 1e-9


def test_build_equity_curve_uses_exit_order_and_preserves_end_value():
    from general_portfolio_backtest import _build_equity_curve

    trade_a = _make_fake_trade(1_700_000_100, 0.05, pair="AAAUSD")
    trade_a.exit_ts = 1_700_000_500
    trade_b = _make_fake_trade(1_700_000_200, -0.02, pair="BBBUSD")
    trade_b.exit_ts = 1_700_000_400

    curve = _build_equity_curve([trade_a, trade_b], start_ts=1_700_000_000, end_ts=1_700_001_000)

    assert curve[0]["ts"] == 1_700_000_000
    assert curve[0]["net_pct"] == 0.0
    assert curve[1]["ts"] == 1_700_000_400
    assert abs(curve[1]["net_pct"] - (-0.02)) < 1e-9
    assert curve[2]["ts"] == 1_700_000_500
    assert abs(curve[2]["net_pct"] - 0.03) < 1e-9
    assert curve[-1]["ts"] == 1_700_001_000
    assert abs(curve[-1]["net_pct"] - 0.03) < 1e-9


def test_run_portfolio_backtest_summary_keys():
    """run_portfolio_backtest summary includes older60 and recent60 keys."""
    import general_portfolio_backtest as gpb

    # Patch the internal _load_uniform_history to avoid file I/O
    original = gpb._load_uniform_history

    def _mock_load(pair, interval, history_days, refresh_live, trade_count, trade_pause_sec):
        import numpy as np
        n = 200
        ts = list(range(1_700_000_000, 1_700_000_000 + n * 900, 900))
        df = pd.DataFrame({
            "ts": ts, "open": [1.0] * n, "high": [1.05] * n,
            "low": [0.95] * n, "close": [1.0] * n,
            "volume": [1000.0] * n, "vwap_k": [1.0] * n, "count": [1] * n,
        })
        from pathlib import Path
        return df, Path("dummy.csv")

    gpb._load_uniform_history = _mock_load
    try:
        _, summary = gpb.run_portfolio_backtest(
            pairs=[], interval=15, history_days=120,
            refresh_live=False, trade_count=0, trade_pause_sec=0,
        )
        assert "older60" in summary, "summary missing 'older60'"
        assert "recent60" in summary, "summary missing 'recent60'"
        assert "split_ts" in summary, "summary missing 'split_ts'"
        assert "equity_curve" in summary, "summary missing 'equity_curve'"
        assert "recent_trades" in summary, "summary missing 'recent_trades'"
        assert "pair_plans" in summary, "summary missing 'pair_plans'"
    finally:
        gpb._load_uniform_history = original
