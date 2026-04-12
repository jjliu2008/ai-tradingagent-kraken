from __future__ import annotations

import json


def test_load_state_appends_missing_registry_actives(tmp_path, monkeypatch):
    import roster_manager as rm

    state_path = tmp_path / "roster_state.json"
    state_path.write_text(
        json.dumps(
            {
                "active": ["GIGAUSD", "ZECUSD", "KERNELUSD", "HOUSEUSD", "BABYUSD"],
                "rotations_today": 0,
                "last_rotation_date": "2026-04-12",
                "last_bar_ts": 1775955600,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        rm,
        "active_pairs",
        lambda: ("GIGAUSD", "ZECUSD", "FHEUSD", "KERNELUSD", "HOUSEUSD", "BABYUSD"),
    )

    roster = rm.RosterManager(max_pos=7, state_file=state_path)

    assert roster.active == ["GIGAUSD", "ZECUSD", "KERNELUSD", "HOUSEUSD", "BABYUSD", "FHEUSD"]
