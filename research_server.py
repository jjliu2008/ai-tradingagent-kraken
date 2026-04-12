"""
research_server.py — Local HTTP server for n8n to call instead of Code nodes.

Runs on port 5680 by default.
Start with: python research_server.py

Endpoints:
  POST /run-screener       { "run_id": "..." }
  POST /run-diagnostics    { "run_id": "..." }
  POST /run-discovery      { "run_id": "..." }
  POST /run-proposal       { "run_id": "..." }
  POST /run-suppression    {}
  GET  /health
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from flask import Flask, jsonify, request

app = Flask(__name__)
BASE_DIR = Path(__file__).parent
PYTHON = sys.executable
DEFAULT_PORT = 5680


def get_server_port() -> int:
    try:
        return int(os.environ.get("RESEARCH_SERVER_PORT", str(DEFAULT_PORT)))
    except ValueError:
        return DEFAULT_PORT


def build_agent_command() -> list[str]:
    return [
        PYTHON,
        "universe_scanner_agent.py",
        "--mode",
        "paper",
        "--strategy-mode",
        "research",
        "--use-research-universe",
        "--cycles",
        "1",
        "--poll",
        "1",
        "--interval",
        "15",
    ]


def _run(cmd: list[str], timeout: int = 900) -> dict:
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
            env=env,
        )
        return {
            "ok": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": "Timeout expired", "returncode": -1}
    except Exception as e:
        return {"ok": False, "stdout": "", "stderr": str(e), "returncode": -1}


@app.get("/health")
def health():
    return jsonify({"ok": True, "server": "research_server"})


@app.post("/run-screener")
def run_screener():
    run_id = request.json.get("run_id", "") if request.is_json else ""
    cmd = [PYTHON, "older60_pair_screener.py"]
    if run_id:
        cmd += ["--run-id", run_id]
    result = _run(cmd, timeout=900)
    return jsonify({**result, "run_id": run_id}), 200 if result["ok"] else 500


@app.post("/run-diagnostics")
def run_diagnostics():
    run_id = request.json.get("run_id", "") if request.is_json else ""
    cmd = [PYTHON, "segment_diagnostics.py"]
    if run_id:
        cmd += ["--run-id", run_id]
    result = _run(cmd, timeout=300)
    return jsonify({**result, "run_id": run_id}), 200 if result["ok"] else 500


@app.post("/run-discovery")
def run_discovery():
    run_id = request.json.get("run_id", "") if request.is_json else ""
    cmd = [PYTHON, "pattern_guided_discovery.py"]
    if run_id:
        cmd += ["--run-id", run_id]
    result = _run(cmd, timeout=300)
    return jsonify({**result, "run_id": run_id}), 200 if result["ok"] else 500


@app.post("/run-proposal")
def run_proposal():
    run_id = request.json.get("run_id", "") if request.is_json else ""
    cmd = [PYTHON, "registry_proposal.py"]
    if run_id:
        cmd += ["--run-id", run_id]
    result = _run(cmd, timeout=300)
    return jsonify({**result, "run_id": run_id}), 200 if result["ok"] else 500


@app.post("/run-suppression")
def run_suppression():
    result = _run([PYTHON, "suppression_state.py"], timeout=120)
    return jsonify(result), 200 if result["ok"] else 500


@app.post("/run-agent")
def run_agent():
    result = _run(build_agent_command(), timeout=300)
    return jsonify(result), 200 if result["ok"] else 500


if __name__ == "__main__":
    port = get_server_port()
    print(f"Research server starting on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
