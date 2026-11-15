"""Provisional markers attached at construction, readable headlessly (Tier C C5)."""
from __future__ import annotations

from pathlib import Path

import yaml

from ..seams.protocols import ProvisionalRef

DEFAULT_REGISTER = "config/replaceability.yaml"


def load_register(path: str | Path) -> dict:
    location = Path(path)
    if not location.exists():
        raise FileNotFoundError(f"replaceability register not found: {location}")
    with open(location, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {"entries": []}


def register_version(register: dict) -> int:
    return int(register.get("register_version", 0))


def _role(confirm_with: str) -> str:
    text = str(confirm_with or "team")
    return text.split(" (")[0].strip()


def provisional_refs(register: dict, rep_ids=None, confirmed=()) -> tuple:
    entries = register.get("entries", [])
    wanted = set(rep_ids) if rep_ids is not None else None
    refs = []
    for entry in entries:
        rep_id = entry.get("rep_id")
        if wanted is not None and rep_id not in wanted:
            continue
        status = str(entry.get("status", "Provisional"))
        if status == "Confirmed" or rep_id in confirmed:
            continue
        refs.append(ProvisionalRef(rep_id=rep_id,
                                   confirmer_role=_role(entry.get("confirm_with")),
                                   decision=str(entry.get("decision", ""))))
    return tuple(refs)


def provisional_for_figures(register: dict) -> tuple:
    """Every REP whose figures can reach the bundle."""
    return provisional_refs(register)


def render_provisional(ref: ProvisionalRef) -> str:
    return f"PROVISIONAL - {ref.rep_id} - {ref.confirmer_role}"