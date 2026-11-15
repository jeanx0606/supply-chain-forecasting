"""Documentation contract: runbook, commands, generated register, drift detection."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

import codes.check_docs as check_docs

pytestmark = pytest.mark.docs

ROOT = Path(__file__).resolve().parents[2]


def test_both_mandated_files_exist():
    assert Path(check_docs.RUNBOOK).exists()
    assert Path(check_docs.REGISTER).exists()


def test_sections_r1_to_r13_in_order():
    text = Path(check_docs.RUNBOOK).read_text(encoding="utf-8")
    assert check_docs.headings(text) == [f"R{i}" for i in range(1, 14)]


def test_five_commands_quoted_verbatim():
    text = Path(check_docs.RUNBOOK).read_text(encoding="utf-8")
    for command in check_docs.REQUIRED_COMMANDS:
        assert command in text, command


def test_version_block_matches_register():
    register = yaml.safe_load(Path(check_docs.REGISTER).read_text(encoding="utf-8"))
    text = Path(check_docs.RUNBOOK).read_text(encoding="utf-8")
    assert f"register_version: {register['register_version']}" in text


def test_r9_block_equals_generated_table():
    register = yaml.safe_load(Path(check_docs.REGISTER).read_text(encoding="utf-8"))
    text = Path(check_docs.RUNBOOK).read_text(encoding="utf-8")
    assert check_docs.extract_block(text) == check_docs.render_register_table(register)


def test_register_has_nineteen_complete_entries():
    register = yaml.safe_load(Path(check_docs.REGISTER).read_text(encoding="utf-8"))
    entries = register["entries"]
    required = {"rep_id", "decision", "current_default", "seam", "location",
                "replacement_entails", "confirm_with", "status", "affected_outputs"}
    assert len(entries) == 19
    assert register["register_version"] == 3
    assert all(required <= set(entry) for entry in entries)
    by_rep = {entry["rep_id"]: entry for entry in entries}
    defaults = {rep_id: entry["current_default"] for rep_id, entry in by_rep.items()}
    assert defaults["REP-04"].startswith("lifecycle_membership")
    assert defaults["REP-10"].startswith("normal_approx")
    assert defaults["REP-17"].startswith("clip_at_zero")
    assert by_rep["REP-05"]["status"] == "Confirmed"
    assert by_rep["REP-05"]["confirmed_by"] == "Rue"
    assert by_rep["REP-05"]["confirmed_on"]
    assert by_rep["REP-17"]["status"] == "Provisional"
    assert by_rep["REP-18"]["status"] == "Provisional"
    assert by_rep["REP-19"]["status"] == "Provisional"
    assert by_rep["REP-18"]["confirm_with"].startswith("Linda")
    assert by_rep["REP-19"]["confirm_with"].startswith("Linda")


def test_documentation_contract_is_green():
    assert check_docs.check() == []


def test_induced_divergence_is_detected(tmp_path, monkeypatch):
    runbook = tmp_path / "RUN_AND_REPLACEABILITY.md"
    register = tmp_path / "replaceability.yaml"
    shutil.copy(check_docs.RUNBOOK, runbook)
    shutil.copy(check_docs.REGISTER, register)

    monkeypatch.setattr(check_docs, "RUNBOOK", runbook)
    monkeypatch.setattr(check_docs, "REGISTER", register)

    payload = yaml.safe_load(register.read_text(encoding="utf-8"))
    payload["entries"][9]["current_default"] = "none (edited without confirmation)"
    register.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    problems = check_docs.check()
    assert problems
    assert any("R9" in problem for problem in problems)