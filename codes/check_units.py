"""AST scan: the scale constant must not appear in business logic (C10)."""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_VALUE = 53.0
DEFAULT_TARGETS = ("src", "ui", "app.py")


class _Visitor(ast.NodeVisitor):
    def __init__(self):
        self.functions = []
        self.findings = []

    def _enter(self, node, kind="function"):
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    def visit_FunctionDef(self, node):  # noqa: N802
        self._enter(node)

    def visit_AsyncFunctionDef(self, node):  # noqa: N802
        self._enter(node)

    def visit_Constant(self, node):  # noqa: N802
        if isinstance(node.value, bool):
            return
        if isinstance(node.value, (int, float)) and float(node.value) == FORBIDDEN_VALUE:
            self.findings.append((node.lineno, self.functions[-1] if self.functions else "<module>"))


def scan_file(path: Path) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    visitor = _Visitor()
    visitor.visit(tree)
    return visitor.findings


def targets(base: Path) -> list:
    files = []
    for name in DEFAULT_TARGETS:
        path = base / name
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
    return files


def check(base: Path = ROOT, single: Path | None = None) -> list:
    problems = []
    files = [single] if single else targets(base)
    for path in files:
        for line, function in scan_file(path):
            problems.append(f"{path}:{line}: forbidden scale constant in '{function}'")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Forbid the scale constant in business logic")
    parser.add_argument("--path", default=None)
    args = parser.parse_args(argv)
    single = Path(args.path) if args.path else None
    problems = check(single=single)
    if problems:
        print("unit scan FAILED")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("unit scan OK: no scale constant in business logic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())