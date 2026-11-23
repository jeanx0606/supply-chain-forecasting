"""Reconcile Rue's cleaned pair contract against the signed Step 1 figures.

The guarded raw-file test in ``tests/unit/test_expected_counts.py`` needs the five
source files. When only the cleaned contract (one row per item-store pair with the
reconstructed stock series) is available, part of the same reconciliation can still
run. This module states exactly which signed figures the cleaned contract settles
and which ones it cannot: the A4 pair count needs the raw restock event offsets, and
the 3,386 negative cells are a raw-sales fact that cleaning has already clipped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..errors import ContractError
from .expected_counts import (A2_FIRST_STOCKED_BEFORE_RELEASE_SHARE, AW19_PAIRS,
                              EXPECTED_A4_PAIRS,
                              EXPECTED_A4_PAIRS_REQUIRING_A3,
                              EXPECTED_NEGATIVE_CELLS,
                              EXPECTED_NEGATIVE_PAIRS,
                              EXPECTED_NEGATIVE_PAIRS_WITH_A4,
                              EXPECTED_NEGATIVE_RATE_WITH_A4,
                              EXPECTED_PAIR_TOTAL, EXPECTED_SEASON_SPLIT,
                              EXPECTED_SPLIT_COUNTS, EXPECTED_STOCKOUT_PAIRS,
                              EXPECTED_STOCKOUT_RATE, LATE_AW19_PAIRS,
                              MEDIAN_SELL_THROUGH, ZERO_SALES_SHARE,
                              ZERO_SALES_WITH_ZERO_STOCK_SHARE, late_aw19_counts)

REPRODUCED = "reproduced"
MISMATCH = "mismatch"
NEEDS_RAW_EVENTS = "needs_raw_events"
NEEDS_RAW_SALES = "needs_raw_sales"

RATE_TOLERANCE = 5e-4
NEXT_COMMAND = "python -m codes.reconcile_real_data"

A4_NOTE = ("requires the raw restock event offsets; the cleaned contract stores the "
           "implied arrivals only, so A4 membership cannot be recovered from it")
NEGATIVE_CELLS_NOTE = ("a raw-sales.csv fact; the cleaned contract is already clipped "
                       "and carries 0 negative cells by construction")
ZERO_SALES_NOTE = ("a pre-cleaning fact: cleaning clips 3,386 negatives to zero, so the "
                   "cleaned contract reports a slightly higher zero share")
FIRST_STOCKED_NOTE = ("requires raw restock event timing; the cleaned contract only "
                      "aggregates opening stock, where 99.9% of pairs are positive, so "
                      "78% must mean a first restock strictly before the release week")


@dataclass(frozen=True)
class Figure:
    name: str
    declared: object
    measured: object
    status: str
    note: str = ""

    @property
    def settled(self) -> bool:
        return self.status in (REPRODUCED, MISMATCH)

    def as_dict(self) -> dict:
        return {"name": self.name, "declared": self.declared,
                "measured": self.measured, "status": self.status,
                "note": self.note}


@dataclass(frozen=True)
class Reconciliation:
    figures: tuple = ()
    structural: tuple = ()
    pair_count: int = 0

    @property
    def ok(self) -> bool:
        return not any(item.status == MISMATCH
                       for item in tuple(self.figures) + tuple(self.structural))

    @property
    def discrepancies(self) -> tuple:
        return tuple(item for item in tuple(self.figures) + tuple(self.structural)
                     if item.status == MISMATCH)

    @property
    def unsettled(self) -> tuple:
        return tuple(item for item in self.figures
                     if item.status in (NEEDS_RAW_EVENTS, NEEDS_RAW_SALES))

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "pair_count": int(self.pair_count),
            "settled": sum(1 for item in self.figures if item.settled),
            "unsettled": [item.name for item in self.unsettled],
            "figures": [item.as_dict() for item in self.figures],
            "structural": [item.as_dict() for item in self.structural],
        }

    def markdown(self) -> str:
        lines = ["# Cleaned-pairs reconciliation", "",
                 f"- pairs: {self.pair_count}",
                 f"- result: **{'PASS' if self.ok else 'FAIL'}**",
                 "- source: Rue's cleaned pair contract (one row per item-store pair)",
                 "", "| Figure | Declared | Measured | Status | Note |",
                 "|---|---|---|---|---|"]
        for item in self.figures:
            lines.append(f"| {item.name} | {_show(item.declared)} | {_show(item.measured)} "
                         f"| {item.status} | {item.note.replace('|', '/')} |")
        if self.structural:
            lines += ["", "## Structural checks", "",
                      "| Check | Measured | Status |", "|---|---|---|"]
            for item in self.structural:
                lines.append(f"| {item.name} | {_show(item.measured)} | {item.status} |")
        if self.unsettled:
            lines += ["", "## Declared but not settled by this file", ""]
            lines += [f"- `{item.name}` = {_show(item.declared)} - {item.note}"
                      for item in self.unsettled]
        return "\n".join(lines) + "\n"


def _show(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _figure(name, declared, measured, tolerance=0) -> Figure:
    if measured is None:
        return Figure(name, declared, None, NEEDS_RAW_EVENTS, A4_NOTE)
    if isinstance(declared, float) or isinstance(measured, float):
        settled = abs(float(declared) - float(measured)) <= tolerance
    else:
        settled = int(declared) == int(measured)
    return Figure(name, declared, measured, REPRODUCED if settled else MISMATCH)


def _require(frame: pd.DataFrame) -> None:
    sales = [str(week) for week in range(12)]
    stock = [f"stock_{week}" for week in range(12)]
    required = ["split", "season", "release_date", "init_stock", "extra_stock",
                "restock"] + sales + stock
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ContractError(
            "cleaned_pairs.csv", "required columns missing from the pair contract",
            missing_fields=missing, expected="one row per item-store pair, weeks 0-11",
            next_command=NEXT_COMMAND)


def reconcile_cleaned_pairs(frame) -> Reconciliation:
    """Compare the cleaned pair contract with the signed Step 1 figures."""
    frame = pd.DataFrame(frame)
    frame.columns = [str(column) for column in frame.columns]
    _require(frame)

    pairs = int(len(frame))
    sales = frame[[str(week) for week in range(12)]].to_numpy(dtype=float)
    stock = frame[[f"stock_{week}" for week in range(12)]].to_numpy(dtype=float)
    initial = frame["init_stock"].to_numpy(dtype=float)
    opening_extra = frame["extra_stock"].to_numpy(dtype=float)

    implied = np.zeros_like(stock)
    implied[:, 1:] = np.clip(stock[:, 1:] - stock[:, :-1] + sales[:, 1:], 0, None)
    balance = initial[:, None] + np.cumsum(implied, axis=1) - np.cumsum(sales, axis=1)
    negative_before_a3 = int((balance.min(axis=1) < -1e-6).sum())
    zero_or_below = int((stock.min(axis=1) <= 0).sum())
    negative_cells = int((sales < 0).sum())
    a3_pairs = int((opening_extra > 0).sum())

    zero_sales = sales == 0
    zero_sales_share = float(zero_sales.mean()) if pairs else 0.0
    zero_with_zero_stock = (float((zero_sales & (stock <= 0)).sum() / zero_sales.sum())
                            if zero_sales.any() else None)
    final_supply = (initial + opening_extra)[:, None] + np.cumsum(implied, axis=1)
    supplied = final_supply[:, -1]
    sell_through = np.divide(sales.sum(axis=1), supplied,
                             out=np.full_like(supplied, np.nan), where=supplied > 0)
    valid = np.isfinite(sell_through)
    median_sell_through = float(np.median(sell_through[valid])) if valid.any() else None

    labels = frame["split"].astype(str).str.lower()
    counts = {name: int(labels.str.startswith(prefix).sum())
              for name, prefix in (("train", "tr"), ("validation", "va"),
                                   ("test", "te"))}
    seasons = frame["season"].astype(object).where(
        frame["season"].notna(), "unknown").astype(str)
    season_counts = {str(name): int(count) for name, count in seasons.value_counts().items()}
    late = late_aw19_counts(frame)

    sell_through_figure = (
        _figure("median_sell_through", MEDIAN_SELL_THROUGH, median_sell_through,
                tolerance=0.005) if median_sell_through is not None else
        Figure("median_sell_through", MEDIAN_SELL_THROUGH, None, NEEDS_RAW_EVENTS,
               "no pair has positive supplied units"))
    zero_stock_figure = (
        _figure("zero_sales_with_zero_stock_share",
                ZERO_SALES_WITH_ZERO_STOCK_SHARE, zero_with_zero_stock,
                tolerance=0.005) if zero_with_zero_stock is not None else
        Figure("zero_sales_with_zero_stock_share", ZERO_SALES_WITH_ZERO_STOCK_SHARE,
               None, NEEDS_RAW_SALES, "no zero-sale week in this file"))

    figures = [
        _figure("pair_total", EXPECTED_PAIR_TOTAL, pairs),
        _figure("split.train", EXPECTED_SPLIT_COUNTS["train"], counts["train"]),
        _figure("split.validation", EXPECTED_SPLIT_COUNTS["validation"], counts["validation"]),
        _figure("split.test", EXPECTED_SPLIT_COUNTS["test"], counts["test"]),
    ]
    for part, names in EXPECTED_SEASON_SPLIT.items():
        observed = sum(season_counts.get(season, 0) for season in names)
        figures.append(_figure(f"season.{part}", EXPECTED_SPLIT_COUNTS[part], observed))
    figures += [
        _figure("aw19_pairs", AW19_PAIRS, late["aw19"]),
        _figure("late_aw19_pairs", LATE_AW19_PAIRS, late["late"]),
        _figure("stockout_pairs", EXPECTED_STOCKOUT_PAIRS, zero_or_below),
        _figure("stockout_rate", EXPECTED_STOCKOUT_RATE, zero_or_below / pairs if pairs else 0.0,
                tolerance=RATE_TOLERANCE),
        _figure("a3_pairs", EXPECTED_NEGATIVE_PAIRS_WITH_A4, a3_pairs),
        _figure("negative_pairs_with_a4", EXPECTED_NEGATIVE_PAIRS_WITH_A4,
                negative_before_a3),
        Figure("negative_pairs_without_a4", EXPECTED_NEGATIVE_PAIRS, None,
               NEEDS_RAW_EVENTS, A4_NOTE),
        Figure("a4_pairs", EXPECTED_A4_PAIRS, None, NEEDS_RAW_EVENTS, A4_NOTE),
        Figure("a4_pairs_requiring_a3", EXPECTED_A4_PAIRS_REQUIRING_A3, None,
               NEEDS_RAW_EVENTS, A4_NOTE),
        Figure("negative_cells", EXPECTED_NEGATIVE_CELLS, negative_cells,
               NEEDS_RAW_SALES, NEGATIVE_CELLS_NOTE),
        _figure("negative_rate_with_a4", EXPECTED_NEGATIVE_RATE_WITH_A4,
                negative_before_a3 / pairs if pairs else 0.0, tolerance=RATE_TOLERANCE),
        sell_through_figure,
        zero_stock_figure,
        Figure("zero_sales_share", ZERO_SALES_SHARE, zero_sales_share,
               NEEDS_RAW_SALES, ZERO_SALES_NOTE),
        Figure("a2_first_stocked_before_release_share",
               A2_FIRST_STOCKED_BEFORE_RELEASE_SHARE, None, NEEDS_RAW_EVENTS,
               FIRST_STOCKED_NOTE),
    ]

    structural = [
        _figure("split_partition_covers_all_pairs", pairs, sum(counts.values())),
        _figure("a3_pairs_equal_negative_pairs_before_a3", a3_pairs, negative_before_a3),
    ]
    return Reconciliation(tuple(figures), tuple(structural), pairs)