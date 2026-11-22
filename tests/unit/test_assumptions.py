"""A1-A6 register: all six declared, A1-A4 encoded, A5/A6 team decisions."""
from __future__ import annotations

from pathlib import Path

from src.report.assumptions import (ASSUMPTIONS, IDS, SIGNED_ON, as_dicts,
                                    by_id, encoded_ids, open_team_decisions)

ROOT = Path(__file__).resolve().parents[2]


def test_all_six_assumptions_are_declared_once():
    assert IDS == ("A1", "A2", "A3", "A4", "A5", "A6")
    assert len(ASSUMPTIONS) == 6


def test_a1_to_a4_are_encoded_and_carry_the_signoff():
    assert encoded_ids() == ("A1", "A2", "A3", "A4")
    for assumption in ASSUMPTIONS[:4]:
        assert assumption.status == "encoded"
        assert assumption.as_dict()["signed_on"] == SIGNED_ON
    assert by_id("A2").statement.count("release week") == 1
    assert "alpha" in by_id("A2").statement
    assert "smallest" in by_id("A3").statement


def test_a5_and_a6_stay_open_team_decisions():
    assert open_team_decisions() == ("A5", "A6")
    assert by_id("A5").status == "stated_limitation"
    assert by_id("A6").status == "scenario_parameter"
    assert "lead time" in by_id("A6").statement
    assert "shortage" in by_id("A6").statement
    assert by_id("A6").as_dict()["signed_by"] is None


def test_every_enforced_in_path_exists():
    for assumption in ASSUMPTIONS:
        assert (ROOT / assumption.enforced_in).exists(), assumption.id


def test_register_serialises_without_losing_status():
    payload = as_dicts()
    assert [row["id"] for row in payload] == list(IDS)
    assert {row["status"] for row in payload} == {
        "encoded", "stated_limitation", "scenario_parameter"}