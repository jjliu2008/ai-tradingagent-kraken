from __future__ import annotations

import importlib
from pathlib import Path


def test_default_server_port_is_5680(monkeypatch):
    monkeypatch.delenv("RESEARCH_SERVER_PORT", raising=False)
    import research_server

    module = importlib.reload(research_server)

    assert module.get_server_port() == 5680


def test_server_port_can_be_overridden(monkeypatch):
    monkeypatch.setenv("RESEARCH_SERVER_PORT", "5801")
    import research_server

    module = importlib.reload(research_server)

    assert module.get_server_port() == 5801


def test_build_agent_command_uses_single_research_cycle():
    import research_server

    module = importlib.reload(research_server)
    cmd = module.build_agent_command()

    assert cmd[0].endswith("python.exe") or cmd[0].endswith("python")
    assert cmd[1] == "universe_scanner_agent.py"
    assert "--mode" in cmd and "paper" in cmd
    assert "--strategy-mode" in cmd and "research" in cmd
    assert "--use-research-universe" in cmd
    assert "--cycles" in cmd and "1" in cmd
    assert "--poll" in cmd and "1" in cmd
    assert "--interval" in cmd and "15" in cmd


def test_run_handles_non_ascii_stdout(tmp_path: Path):
    import research_server

    module = importlib.reload(research_server)
    script = tmp_path / "emit_unicode.py"
    script.write_text("print('ok → done')\n", encoding="utf-8")

    result = module._run([module.PYTHON, str(script)])

    assert result["ok"] is True
    assert result["returncode"] == 0
    assert "ok → done" in result["stdout"]
