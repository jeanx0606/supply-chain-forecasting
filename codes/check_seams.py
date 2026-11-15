"""Seam audit: one marker occurrence, one owner, one register entry per seam."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

from src.seams.catalogue import ENTRIES

ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / "config" / "replaceability.yaml"
MARKER_RE = re.compile(r"SEAM-[A-Z0-9]+(?:-[A-Z0-9]+)*")


def source_files() -> list:
    return sorted((ROOT / "src").rglob("*.py"))


def scan() -> dict:
    hits: dict = {}
    for path in source_files():
        text = path.read_text(encoding="utf-8")
        for marker in MARKER_RE.findall(text):
            hits.setdefault(marker, []).append(str(path.relative_to(ROOT)).replace("\\", "/"))
    return hits


def check() -> list:
    problems = []
    hits = scan()
    expected = {entry.marker: entry for entry in ENTRIES}

    for marker, entry in expected.items():
        locations = hits.get(marker, [])
        if len(locations) == 0:
            problems.append(f"{entry.rep_id}: marker {marker} missing from {entry.owner}")
        elif len(locations) > 1:
            problems.append(f"{entry.rep_id}: marker {marker} occurs {len(locations)} times "
                            f"({sorted(set(locations))})")
        elif locations[0] != entry.owner:
            problems.append(f"{entry.rep_id}: marker {marker} lives in {locations[0]}, "
                            f"catalogue says {entry.owner}")

    for marker in hits:
        if marker not in expected:
            problems.append(f"unregistered marker {marker} in {sorted(set(hits[marker]))}")

    rep_ids = [entry.rep_id for entry in ENTRIES]
    if len(set(rep_ids)) != len(rep_ids):
        problems.append("duplicate REP ids in the catalogue")

    if not REGISTER.exists():
        problems.append(f"replaceability register missing: {REGISTER}")
        return problems
    with open(REGISTER, "r", encoding="utf-8") as handle:
        register = yaml.safe_load(handle) or {}
    reg_by_rep = {entry.get("rep_id"): entry for entry in register.get("entries", [])}
    for entry in ENTRIES:
        record = reg_by_rep.get(entry.rep_id)
        if record is None:
            problems.append(f"{entry.rep_id}: no entry in config/replaceability.yaml")
        elif record.get("seam") != entry.marker:
            problems.append(f"{entry.rep_id}: register seam {record.get('seam')} "
                            f"!= source marker {entry.marker}")
        else:
            confirmer = str(record.get("confirm_with", "")).split(" (")[0].strip()
            if confirmer != entry.confirmer:
                problems.append(f"{entry.rep_id}: confirmer '{confirmer}' in the "
                                f"register != '{entry.confirmer}' in the catalogue")
    for rep_id in reg_by_rep:
        if rep_id not in {entry.rep_id for entry in ENTRIES}:
            problems.append(f"{rep_id}: register entry has no seam in the catalogue")
    return problems


def main() -> int:
    problems = check()
    if problems:
        print("seam audit FAILED")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"seam audit OK: {len(ENTRIES)} seams, one marker each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())