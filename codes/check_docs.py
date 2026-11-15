"""Documentation contract: runbook sections, quoted commands and generated register."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

from src.report.decision_packet import packet_is_fresh
from src.seams.catalogue import ENTRIES

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs" / "RUN_AND_REPLACEABILITY.md"
REGISTER = ROOT / "config" / "replaceability.yaml"

REQUIRED_COMMANDS = (
    "pip install -r requirements.txt",
    "python -m codes.sample_data",
    "python -m codes.run_pipeline",
    "python -m pytest",
    "streamlit run app.py",
)

R13_COMMANDS = (
    "python -m codes.preflight",
    "python -m codes.decision_packet",
    "python -m codes.chapter_report",
    "python -m codes.check_units",
)

R9_BEGIN = "<!-- R9:BEGIN -->"
R9_END = "<!-- R9:END -->"


def headings(text: str) -> list:
    return re.findall(r"^##\s+(R\d+)\.", text, flags=re.MULTILINE)


def section(text: str, code: str) -> str:
    match = re.search(rf"^##\s+{code}\..*?(?=^##\s+R\d+\.|\Z)", text,
                      flags=re.MULTILINE | re.DOTALL)
    return match.group(0) if match else ""


def table_rows(block: str) -> int:
    return len(re.findall(r"^\|\s*\d+\s*\|", block, flags=re.MULTILINE))


def render_register_table(register: dict) -> str:
    lines = ["| REP | Decision | Current default | Seam | Confirmer | Status |",
             "|---|---|---|---|---|---|"]
    for entry in register.get("entries", []):
        decision = str(entry.get("decision", "")).replace("|", "/")
        default = str(entry.get("current_default", "")).replace("|", "/")
        lines.append(
            f"| {entry.get('rep_id')} | {decision} | {default} | "
            f"{entry.get('seam')} | {entry.get('confirm_with')} | {entry.get('status')} |")
    return "\n".join(lines)


def extract_block(text: str) -> str | None:
    if R9_BEGIN not in text or R9_END not in text:
        return None
    start = text.index(R9_BEGIN) + len(R9_BEGIN)
    end = text.index(R9_END)
    return text[start:end].strip()


def check() -> list:
    problems = []
    if not RUNBOOK.exists():
        return [f"runbook missing: {RUNBOOK}"]
    if not REGISTER.exists():
        return [f"register missing: {REGISTER}"]

    text = RUNBOOK.read_text(encoding="utf-8")
    with open(REGISTER, "r", encoding="utf-8") as handle:
        register = yaml.safe_load(handle)

    found = headings(text)
    expected = [f"R{i}" for i in range(1, 14)]
    if found != expected:
        problems.append(f"runbook headings {found} != {expected}")

    for command in REQUIRED_COMMANDS:
        if command not in text:
            problems.append(f"runbook does not quote the command: {command}")

    version_line = f"register_version: {register.get('register_version')}"
    if version_line not in text:
        problems.append(f"runbook version block missing '{version_line}'")

    entries = register.get("entries", [])
    if len(entries) != len(ENTRIES):
        problems.append(f"register holds {len(entries)} entries, "
                        f"catalogue declares {len(ENTRIES)}")
    by_rep = {entry.get("rep_id"): entry for entry in entries}
    rep05 = by_rep.get("REP-05") or {}
    if rep05.get("status") != "Confirmed":
        problems.append("REP-05 must be Confirmed once the dataset owner has decided")
    if not rep05.get("confirmed_by") or not rep05.get("confirmed_on"):
        problems.append("REP-05 Confirmed requires confirmed_by and confirmed_on")
    for rep_id in ("REP-17", "REP-18", "REP-19"):
        if rep_id not in by_rep:
            problems.append(f"{rep_id} is missing from the register")

    for command in R13_COMMANDS:
        if command not in text:
            problems.append(f"runbook does not quote the R13 command: {command}")
    if "chapter_pack" not in section(text, "R6"):
        problems.append("R6 must state the chapter pack path")

    packet = ROOT / "docs" / "DECISION_PACKET.md"
    if not packet_is_fresh(REGISTER, packet):
        problems.append("decision packet is missing or stale "
                        "(run python -m codes.decision_packet)")

    r5 = section(text, "R5")
    for token in ("stfore", "53", "optional"):
        if token.lower() not in r5.lower():
            problems.append(f"R5 must explain '{token}'")

    if table_rows(section(text, "R8")) < 16:
        problems.append("R8 must hold at least 16 troubleshooting rows")

    r11 = section(text, "R11").lower()
    for token in ("raw sales units", "53", "negative"):
        if token not in r11:
            problems.append(f"R11 must state '{token}'")

    generated = render_register_table(register)
    block = extract_block(text)
    if block is None:
        problems.append("runbook is missing the generated R9 block markers")
    elif block != generated:
        problems.append("R9 table diverges from config/replaceability.yaml "
                        "(regenerate it; do not hand-edit)")
    return problems


def regenerate(runbook: Path = RUNBOOK, register_path: Path = REGISTER) -> None:
    """Rewrite the R9 block from the register; the block is never hand-edited."""
    with open(register_path, "r", encoding="utf-8") as handle:
        register = yaml.safe_load(handle)
    text = runbook.read_text(encoding="utf-8")
    start = text.index(R9_BEGIN) + len(R9_BEGIN)
    end = text.index(R9_END)
    replacement = f"\n{render_register_table(register)}\n"
    runbook.write_text(text[:start] + replacement + text[end:], encoding="utf-8")


def main() -> int:
    if "--fix" in sys.argv[1:]:
        regenerate()
        print("R9 block regenerated from config/replaceability.yaml")
        return 0
    problems = check()
    if problems:
        print("documentation contract FAILED")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("documentation contract OK: R1-R13, commands, packet, register table and version agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())