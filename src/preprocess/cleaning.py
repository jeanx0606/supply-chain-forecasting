"""Cleaning: undated exclusion, fill-before-cast normalisation, and the sign policy.

D23: weekly sales contain negative cells. The prototype applies exactly one
declared policy, counts affected cells *before* the policy runs, and reports both.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..errors import ConfigError, StageError
from ..io.schema import CATEGORICAL, KEY_COLUMNS
from ..seams.protocols import SeamDescription

SIGN_NOTE = ("the sign policy is a declared interpretation, not imputation "
             "and not a data correction")


@dataclass(frozen=True)
class SignOutcome:
    policy_name: str
    negative_cells: int
    negative_cells_by_week: dict
    reinterpreted: bool
    note: str = SIGN_NOTE


@dataclass
class CleanResult:
    frame: pd.DataFrame
    excluded: dict
    duplicate_count: int
    sign_outcome: SignOutcome | None = None


class NegativeSalesPolicyBase:
    SEAM_ID = "SEAM-NEGATIVE-SALES"
    implementation = "clip_at_zero"

    def describe(self) -> SeamDescription:
        return SeamDescription("negative_sales", "REP-17", self.implementation)

    def apply(self, values: pd.Series) -> pd.Series:
        raise NotImplementedError


class ClipAtZeroPolicy(NegativeSalesPolicyBase):
    implementation = "clip_at_zero"

    def apply(self, values: pd.Series) -> pd.Series:
        return values.clip(lower=0.0)


class RetainAsReturnsPolicy(NegativeSalesPolicyBase):
    implementation = "retain_as_returns"

    def apply(self, values: pd.Series) -> pd.Series:
        return values


def make_sign_policy(name) -> NegativeSalesPolicyBase:
    value = getattr(name, "value", name)
    if value == "clip_at_zero":
        return ClipAtZeroPolicy()
    if value == "retain_as_returns":
        return RetainAsReturnsPolicy()
    raise ConfigError(
        "cleaning.negative_sales_policy",
        "unregistered selector value (capability 5.2 configuration control)",
        observed=value, expected=["clip_at_zero", "retain_as_returns"])


def count_negatives(frame: pd.DataFrame, n_weeks: int = 12,
                    prefix: str = "q") -> tuple[int, dict]:
    total = 0
    by_week = {}
    for n in range(1, n_weeks + 1):
        column = f"{prefix}{n}"
        if column in frame.columns:
            count = int((pd.to_numeric(frame[column], errors="coerce") < 0).sum())
            if count:
                by_week[n] = count
                total += count
    return total, by_week


def _normalise_text(series: pd.Series) -> pd.Series:
    values = series.astype("object")
    values = values.where(values.notna(), "unknown")
    values = values.astype(str).str.strip()
    return values.mask(values.isin(["", "nan", "None", "<NA>", "NaN"]), "unknown")


def clean_sales(frame: pd.DataFrame, n_weeks: int = 12, prefix: str = "q",
                *, policy: NegativeSalesPolicyBase) -> CleanResult:
    """Apply one declared sign policy once, after coercion and before attributes."""
    df = frame.copy()
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    before = len(df)
    df = df[df["release_date"].notna()].copy()
    undated = before - len(df)
    if df.empty:
        raise StageError("clean", "no dated rows remain after cleaning",
                         next_command="python -m codes.sample_data --overwrite")

    for n in range(1, n_weeks + 1):
        column = f"{prefix}{n}"
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    negative_cells, negative_by_week = count_negatives(df, n_weeks, prefix)
    for n in range(1, n_weeks + 1):
        column = f"{prefix}{n}"
        if column in df.columns:
            df[column] = policy.apply(df[column])

    for column in CATEGORICAL:
        if column in df.columns:
            df[column] = _normalise_text(df[column])

    duplicate_mask = df.duplicated(subset=list(KEY_COLUMNS), keep="first")
    duplicate_count = int(duplicate_mask.sum())
    df = df[~duplicate_mask].reset_index(drop=True)

    excluded = {"undated_rows": int(undated), "duplicate_rows": duplicate_count}
    outcome = SignOutcome(
        policy_name=policy.describe().implementation,
        negative_cells=negative_cells,
        negative_cells_by_week=negative_by_week,
        reinterpreted=policy.describe().implementation == "retain_as_returns",
    )
    return CleanResult(df, excluded, duplicate_count, outcome)


def quality_report(frame: pd.DataFrame, excluded: dict, duplicate_count: int,
                   monday_violations: int, restock_policy: str,
                   restock_state: dict, extra_exclusions: dict | None = None,
                   *, negative_sales_policy: str = "clip_at_zero",
                   negative_cells: int = 0,
                   negative_cells_by_week: dict | None = None,
                   unit: str = "raw_sales_units",
                   applied_scale_factor=None) -> dict:
    excluded_all = dict(excluded)
    if extra_exclusions:
        for key, value in extra_exclusions.items():
            excluded_all[key] = excluded_all.get(key, 0) + int(value)

    n_items = int(frame["external_code"].nunique())
    n_stores = int(frame["retail"].nunique())
    pairs = max(n_items * n_stores, 1)
    return {
        "n_rows": int(len(frame)),
        "n_items": n_items,
        "n_stores": n_stores,
        "date_min": str(pd.Timestamp(frame["release_date"].min()).date()),
        "date_max": str(pd.Timestamp(frame["release_date"].max()).date()),
        "duplicate_count": int(duplicate_count),
        "pair_coverage": round(len(frame) / pairs, 4),
        "excluded": excluded_all,
        "monday_violations": int(monday_violations),
        "weekly_only_marker": "weekly",
        "restock_policy": restock_policy,
        "restock_date_max_state": dict(restock_state),
        "negative_sales_policy": negative_sales_policy,
        "negative_cells": int(negative_cells),
        "negative_cells_by_week": dict(negative_cells_by_week or {}),
        "unit": unit,
        "applied_scale_factor": applied_scale_factor,
    }