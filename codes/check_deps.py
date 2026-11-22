"""Dependency audit: no declared-unused and no imported-undeclared package (D17)."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PACKAGE_ALIASES = {
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "pytest": "pytest",
    "pandas": "pandas",
    "numpy": "numpy",
    "streamlit": "streamlit",
    "altair": "altair",
}

OPTIONAL = {"lightgbm": "install when model.backend=lightgbm or the "
                         "lightgbm_horizon candidate is enabled"}

SCAN_DIRS = ("src", "ui", "codes", "tests")


def declared() -> set:
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    packages = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split(">")[0].split("=")[0].split("<")[0].strip()
        packages.add(name)
    return packages


def imported() -> tuple[set, set]:
    third_party, optional = set(), set()
    stdlib = set(sys.stdlib_module_names)
    for folder in SCAN_DIRS:
        for path in (ROOT / folder).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module.split(".")[0]]
                for name in names:
                    if name in ("src", "ui", "codes", "tests") or name in stdlib:
                        continue
                    package = PACKAGE_ALIASES.get(name, name)
                    if package in OPTIONAL:
                        optional.add(package)
                    else:
                        third_party.add(package)
    return third_party, optional


def check() -> list:
    problems = []
    declared_packages = declared()
    used, optional = imported()

    for package in sorted(declared_packages - used - optional):
        problems.append(f"declared but never imported: {package}")
    for package in sorted(used - declared_packages):
        problems.append(f"imported but not declared in requirements.txt: {package}")
    for package in sorted(optional - declared_packages):
        note = OPTIONAL[package]
        print(f"[deps] optional backend referenced but not declared: {package} ({note})")
    return problems


def main() -> int:
    problems = check()
    if problems:
        print("dependency audit FAILED")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"dependency audit OK: {sorted(declared())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())