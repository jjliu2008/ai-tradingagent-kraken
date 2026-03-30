from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path


def _candidate_binaries() -> list[str]:
    candidates: list[str] = []
    env_bin = os.environ.get("KRAKEN_BIN")
    if env_bin:
        candidates.append(env_bin)

    which_bin = shutil.which("kraken") or shutil.which("kraken.exe")
    if which_bin:
        candidates.append(which_bin)

    home = Path.home()
    candidates.extend(
        [
            str(home / ".cargo" / "bin" / "kraken.exe"),
            str(home / ".cargo" / "bin" / "kraken"),
        ]
    )
    return candidates


def _resolve_binary() -> str:
    for candidate in _candidate_binaries():
        if Path(candidate).exists():
            return candidate
    raise RuntimeError(
        "kraken CLI binary not found. Set KRAKEN_BIN or install it under ~/.cargo/bin."
    )


KRAKEN_BIN = _resolve_binary()
PUBLIC_API_URL = os.environ.get("KRAKEN_PUBLIC_API_URL", "https://api.kraken.com/0/public")


def _run(args: list[str]) -> dict:
    result = subprocess.run(
        [KRAKEN_BIN] + args + ["-o", "json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        try:
            payload = json.loads(result.stdout) if result.stdout else {}
        except json.JSONDecodeError:
            payload = {"stderr": result.stderr.strip(), "stdout": result.stdout.strip()}
        raise RuntimeError(f"kraken {' '.join(args)} failed: {payload}")
    return json.loads(result.stdout)


def _public_get(endpoint: str, params: dict[str, str | int]) -> dict:
    query = urllib.parse.urlencode({key: str(value) for key, value in params.items()})
    url = f"{PUBLIC_API_URL}/{endpoint}?{query}"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("error"):
        raise RuntimeError(f"kraken public {endpoint} failed: {payload['error']}")
    return payload["result"]


def fetch_ohlc(pair: str, interval: int = 1, since: int | None = None) -> dict:
    args = ["ohlc", pair, "--interval", str(interval)]
    if since is not None:
        args += ["--since", str(since)]
    return _run(args)


def fetch_trades(pair: str, since: int | None = None, count: int = 1000) -> dict:
    params: dict[str, str | int] = {"pair": pair, "count": count}
    if since is not None:
        params["since"] = since
    return _public_get("Trades", params)


def fetch_orderbook(pair: str) -> dict:
    return _run(["orderbook", pair])


def fetch_ticker(pair: str) -> dict:
    return _run(["ticker", pair])


def paper_init(balance: float = 10000.0) -> dict:
    return _run(["paper", "init", "--balance", str(balance)])


def paper_buy(pair: str, volume: float) -> dict:
    return _run(["paper", "buy", pair, f"{volume:.8f}"])


def paper_sell(pair: str, volume: float) -> dict:
    return _run(["paper", "sell", pair, f"{volume:.8f}"])


def paper_balance() -> dict:
    return _run(["paper", "balance"])


def paper_status() -> dict:
    return _run(["paper", "status"])


def balance() -> dict:
    return _run(["balance"])


def positions() -> dict:
    return _run(["positions"])
