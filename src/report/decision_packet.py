"""Decision packet generated from the register: options, never decisions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

PENDING = ("REP-04", "REP-10", "REP-17", "REP-18", "REP-19")


def register_sha256(register_path: Path) -> str:
    return hashlib.sha256(Path(register_path).read_bytes()).hexdigest()


def load_register(register_path: Path) -> dict:
    with open(register_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {"entries": []}


def build_packet(register_path: Path) -> str:
    register = load_register(register_path)
    digest = register_sha256(register_path)
    by_rep = {entry.get("rep_id"): entry for entry in register.get("entries", [])}
    lines = [
        "# Decision Packet — provisional seams",
        "",
        f"- register_version: {register.get('register_version')}",
        f"- register_sha256: {digest}",
        "- This packet records options only. It records **no decision and no date**.",
        "- The prototype cannot decide these; they belong to the named confirmer.",
        "",
        "| REP | Decision | Current default | Options | Confirmer | Affected outputs | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    pending = [rep for rep in PENDING if rep in by_rep]
    for rep_id in pending:
        entry = by_rep[rep_id]
        options = entry.get("options") or []
        options_text = "; ".join(str(o) for o in options) if options else "not declared"
        lines.append(
            f"| {rep_id} | {entry.get('decision')} | {entry.get('current_default')} | "
            f"{options_text} | {entry.get('confirm_with')} | "
            f"{', '.join(entry.get('affected_outputs', []))} | {entry.get('status')} |")

    lines += ["", "## What changes when a decision is made", ""]
    for rep_id in pending:
        entry = by_rep[rep_id]
        lines.append(f"### {rep_id}")
        lines.append(f"- Configuration keys: {', '.join(entry.get('config_keys', [])) or entry.get('seam')}")
        lines.append(f"- Options: {'; '.join(entry.get('options', [])) or 'not declared'}")
        lines.append(f"- Gates affected: {', '.join(entry.get('affected_gates', [])) or 'not declared'}")
        lines.append(f"- Replacement: {entry.get('replacement_entails')}")
        lines.append("")

    lines += [
        "## Exact update steps (performed only after a dated decision)",
        "",
        "1. Set `status: Confirmed`, `confirmed_by` and `confirmed_on` in `config/replaceability.yaml`.",
        "2. Change the affected default in `config/settings.yaml` if the decision differs.",
        "3. Run `python -m codes.check_docs --fix` to regenerate R9.",
        "4. Rerun the affected gates and the full suite.",
        "5. Regenerate this packet: `python -m codes.decision_packet`.",
        "",
        f"<!-- register_sha256: {digest} -->",
    ]
    return "\n".join(lines) + "\n"


def write_packet(register_path: Path, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_packet(register_path), encoding="utf-8")
    return out_path


def packet_is_fresh(register_path: Path, packet_path: Path) -> bool:
    if not packet_path.exists():
        return False
    return register_sha256(register_path) in packet_path.read_text(encoding="utf-8")


def packet_entries(packet_path: Path) -> list:
    text = packet_path.read_text(encoding="utf-8")
    return [rep for rep in PENDING if rep in text]