"""
Thin wrapper around the kraken CLI binary.
Returns raw JSON dicts — callers are responsible for parsing.
"""
from __future__ import annotations
import json
import subprocess


def _run(args: list[str]) -> dict:
    result = subprocess.run(
        ["kraken"] + args + ["-o", "json"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        try:
            err = json.loads(result.stdout)
        except Exception:
            err = {"error": result.stderr or result.stdout}
        raise RuntimeError(f"kraken {' '.join(args)} failed: {err}")
    return json.loads(result.stdout)


# --- Market data (public) ---

def fetch_ohlc(pair: str, interval: int = 1) -> dict:
    """Returns raw OHLC dict: {'XXBTZUSD': [[ts,o,h,l,c,vwap,vol,count],...], 'last': ...}"""
    return _run(["ohlc", pair, "--interval", str(interval)])


def fetch_orderbook(pair: str) -> dict:
    """Returns raw orderbook: {'XXBTZUSD': {'bids': [[price,vol,ts],...], 'asks': [...]}}"""
    return _run(["orderbook", pair])


def fetch_ticker(pair: str) -> dict:
    """Returns raw ticker dict with 'a','b','c','h','l','o','p','t','v' keys."""
    return _run(["ticker", pair])


# --- Paper trading ---

def paper_init(balance: float = 10000.0) -> dict:
    return _run(["paper", "init", "--balance", str(balance)])


def paper_buy(pair: str, volume: float) -> dict:
    return _run(["paper", "buy", pair, str(volume)])


def paper_sell(pair: str, volume: float) -> dict:
    return _run(["paper", "sell", pair, str(volume)])


def paper_status() -> dict:
    return _run(["paper", "status"])


# --- Account (requires API key) ---

def balance() -> dict:
    return _run(["balance"])


def positions() -> dict:
    return _run(["positions"])
