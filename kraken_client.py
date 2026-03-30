"""Thin wrapper around the kraken CLI binary."""
import json
import subprocess


def _run(args: list[str]) -> dict:
    """Run a kraken CLI command and return parsed JSON output."""
    result = subprocess.run(
        ["kraken"] + args + ["-o", "json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        try:
            error = json.loads(result.stdout)
        except Exception:
            error = {"error": result.stderr or result.stdout}
        raise RuntimeError(f"kraken command failed: {error}")
    return json.loads(result.stdout)


# --- Paper trading ---

def paper_init(balance: float = 10000.0) -> dict:
    return _run(["paper", "init", "--balance", str(balance)])


def paper_buy(pair: str, volume: float) -> dict:
    return _run(["paper", "buy", pair, str(volume)])


def paper_sell(pair: str, volume: float) -> dict:
    return _run(["paper", "sell", pair, str(volume)])


def paper_status() -> dict:
    return _run(["paper", "status"])


# --- Market data (public, no auth required) ---

def ticker(pair: str) -> dict:
    return _run(["market", "ticker", pair])


def ohlc(pair: str, interval: int = 60) -> dict:
    return _run(["market", "ohlc", pair, "--interval", str(interval)])


def orderbook(pair: str) -> dict:
    return _run(["market", "orderbook", pair])


# --- Account (requires API key) ---

def balance() -> dict:
    return _run(["account", "balance"])


def positions() -> dict:
    return _run(["account", "positions"])
