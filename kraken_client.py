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
    repo_bin = Path(__file__).resolve().parent / "bin" / "kraken"
    candidates.append(str(repo_bin))
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


def _format_decimal(value: float, places: int = 10) -> str:
    text = f"{value:.{places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


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


def paper_buy(pair: str, volume: float, order_type: str = "market", price: float | None = None) -> dict:
    args = ["paper", "buy", pair, f"{volume:.8f}", "--type", order_type]
    if price is not None:
        args += ["--price", _format_decimal(price)]
    return _run(args)


def paper_sell(pair: str, volume: float, order_type: str = "market", price: float | None = None) -> dict:
    args = ["paper", "sell", pair, f"{volume:.8f}", "--type", order_type]
    if price is not None:
        args += ["--price", _format_decimal(price)]
    return _run(args)


def paper_orders() -> dict:
    return _run(["paper", "orders"])


def paper_cancel(order_id: str) -> dict:
    return _run(["paper", "cancel", order_id])


def paper_history() -> dict:
    return _run(["paper", "history"])


def paper_balance() -> dict:
    return _run(["paper", "balance"])


def paper_status() -> dict:
    return _run(["paper", "status"])


def balance() -> dict:
    return _run(["balance"])


def positions() -> dict:
    return _run(["positions"])


def order_buy(
    pair: str,
    volume: float,
    order_type: str = "market",
    price: float | None = None,
    post_only: bool = False,
    client_order_id: str | None = None,
    validate: bool = False,
) -> dict:
    args = ["order", "buy", pair, f"{volume:.8f}", "--type", order_type]
    if price is not None:
        args += ["--price", _format_decimal(price)]
    if post_only:
        args += ["--oflags", "post"]
    if client_order_id:
        args += ["--cl-ord-id", client_order_id]
    if validate:
        args += ["--validate"]
    return _run(args)


def order_sell(
    pair: str,
    volume: float,
    order_type: str = "market",
    price: float | None = None,
    post_only: bool = False,
    client_order_id: str | None = None,
    validate: bool = False,
) -> dict:
    args = ["order", "sell", pair, f"{volume:.8f}", "--type", order_type]
    if price is not None:
        args += ["--price", _format_decimal(price)]
    if post_only:
        args += ["--oflags", "post"]
    if client_order_id:
        args += ["--cl-ord-id", client_order_id]
    if validate:
        args += ["--validate"]
    return _run(args)


def order_cancel(order_id: str) -> dict:
    return _run(["order", "cancel", order_id])


def open_orders() -> dict:
    return _run(["open-orders"])


def query_orders(order_ids: list[str] | tuple[str, ...] | str) -> dict:
    if isinstance(order_ids, str):
        args = [order_ids]
    else:
        args = [",".join(order_ids)]
    return _run(["query-orders"] + args)
