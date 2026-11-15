"""Decision packet: options only, never a decision or a date (TC8, D29)."""
from __future__ import annotations

import shutil
from pathlib import Path

from src.report.decision_packet import (
    PENDING,
    build_packet,
    packet_is_fresh,
    write_packet,
)

ROOT = Path(__file__).resolve().parents[2]
REGISTER = ROOT / "config" / "replaceability.yaml"


def test_packet_covers_every_pending_entry_and_records_no_decision(tmp_path):
    out = tmp_path / "DECISION_PACKET.md"
    write_packet(REGISTER, out)
    text = out.read_text(encoding="utf-8")
    for rep_id in PENDING:
        assert rep_id in text
    assert "no decision and no date" in text
    assert "confirmed_on:" not in text
    assert "REP-05" not in text  # confirmed entries are not pending


def test_packet_is_fresh_and_becomes_stale(tmp_path):
    register = tmp_path / "replaceability.yaml"
    shutil.copy(REGISTER, register)
    out = tmp_path / "DECISION_PACKET.md"
    write_packet(register, out)
    assert packet_is_fresh(register, out) is True

    register.write_text(register.read_text(encoding="utf-8") + "\n# edited\n",
                        encoding="utf-8")
    assert packet_is_fresh(register, out) is False


def test_packet_carries_the_register_digest(tmp_path):
    from src.report.decision_packet import register_sha256

    out = tmp_path / "DECISION_PACKET.md"
    write_packet(REGISTER, out)
    assert register_sha256(REGISTER) in build_packet(REGISTER)
    assert register_sha256(REGISTER) in out.read_text(encoding="utf-8")