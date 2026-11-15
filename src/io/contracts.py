"""Non-raising contract predicates shared by the loader and the preflight (D26)."""
from __future__ import annotations

import pandas as pd


def violations(frame: pd.DataFrame, contract) -> list:
    missing = [field for field in contract.required if field not in frame.columns]
    problems = []
    if missing:
        problems.append(f"missing required fields: {missing}")
    if contract.week_columns:
        numeric = [c for c in frame.columns if str(c).strip().isdigit()]
        if len(numeric) < contract.week_columns:
            problems.append(
                f"expected {contract.week_columns} numeric week columns, observed {len(numeric)}")
    return problems


def header_problems(header: list, n_weeks: int) -> list:
    numeric = [c for c in header if str(c).strip().isdigit()]
    problems = []
    if len(numeric) < n_weeks:
        problems.append(f"expected {n_weeks} numeric week columns, observed {len(numeric)}")
    return problems


def header_shape(header: list, n_weeks: int) -> str:
    numeric = sorted(int(str(c).strip()) for c in header if str(c).strip().isdigit())
    if not numeric:
        return "no numeric week columns"
    return f"{len(numeric)} week columns: {numeric[0]}..{numeric[-1]} (expected {n_weeks})"