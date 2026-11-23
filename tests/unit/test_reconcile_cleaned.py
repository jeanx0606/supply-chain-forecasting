"""Cleaned-pairs reconciliation: derivations, mismatch detection, guarded real file."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.errors import ContractError
from src.preprocess.reconcile import (MISMATCH, NEEDS_RAW_EVENTS,
                                      NEEDS_RAW_SALES, REPRODUCED,
                                      reconcile_cleaned_pairs)

ROOT = Path(__file__).resolve().parents[2]


def pair(external_code, retail, season, release_date, initial, extra, split,
         stock_value: float, sales_value: float = 0.0) -> dict:
    row = {"external_code": external_code, "retail": retail, "season": season,
           "release_date": release_date, "init_stock": initial,
           "extra_stock": extra, "restock": initial + extra, "split": split}
    for week in range(12):
        row[str(week)] = sales_value
        row[f"stock_{week}"] = stock_value
    return row


def synthetic_frame() -> pd.DataFrame:
    return pd.DataFrame([
        pair(1, 10, "SS17", "2016-11-28", 5.0, 0.0, "train", 5.0),
        pair(2, 11, "AW18", "2018-08-06", 0.0, 2.0, "train", 0.0),
        pair(3, 12, "AW19", "2019-09-01", 0.0, 0.0, "test", 0.0),
        pair(4, 13, "SS19", "2019-02-04", 0.0, 1.0, "validation", 0.0),
    ])


def by_name(report) -> dict:
    return {item.name: item for item in report.figures}


def structural_by_name(report) -> dict:
    return {item.name: item for item in report.structural}


def test_declared_figures_are_measured_and_mismatches_reported():
    report = reconcile_cleaned_pairs(synthetic_frame())
    figures = by_name(report)
    assert report.pair_count == 4
    assert figures["pair_total"].measured == 4
    assert figures["pair_total"].status == MISMATCH
    assert figures["split.train"].measured == 2
    assert figures["split.validation"].measured == 1
    assert figures["split.test"].measured == 1
    assert figures["stockout_pairs"].measured == 3
    assert figures["a3_pairs"].measured == 2
    assert figures["negative_pairs_with_a4"].measured == 0
    assert report.ok is False


def test_late_aw19_is_counted_from_release_date():
    report = reconcile_cleaned_pairs(synthetic_frame())
    figures = by_name(report)
    assert figures["aw19_pairs"].measured == 1
    assert figures["late_aw19_pairs"].measured == 1


def test_negative_reconstruction_is_measured_before_a3():
    frame = pd.DataFrame([
        pair(9, 90, "SS17", "2016-11-28", 0.0, 0.0, "train", 0.0, sales_value=1.0),
        pair(10, 91, "SS17", "2016-11-28", 5.0, 0.0, "train", 5.0, sales_value=0.0),
    ])
    report = reconcile_cleaned_pairs(frame)
    figures = by_name(report)
    assert figures["negative_pairs_with_a4"].measured == 1
    assert figures["a3_pairs"].measured == 0
    structural = structural_by_name(report)
    assert structural["a3_pairs_equal_negative_pairs_before_a3"].status == MISMATCH


def test_a4_and_raw_negative_cells_are_declared_not_settled():
    report = reconcile_cleaned_pairs(synthetic_frame())
    figures = by_name(report)
    for name in ("a4_pairs", "a4_pairs_requiring_a3", "negative_pairs_without_a4"):
        assert figures[name].status == NEEDS_RAW_EVENTS
        assert figures[name].measured is None
        assert "restock event offsets" in figures[name].note
    assert figures["negative_cells"].status == NEEDS_RAW_SALES
    assert figures["negative_cells"].measured == 0
    assert figures["zero_sales_share"].status == NEEDS_RAW_SALES
    assert figures["a2_first_stocked_before_release_share"].status == NEEDS_RAW_EVENTS
    assert "before the release week" in figures["a2_first_stocked_before_release_share"].note
    assert {item.name for item in report.unsettled} == {
        "a4_pairs", "a4_pairs_requiring_a3", "negative_pairs_without_a4",
        "negative_cells", "zero_sales_share",
        "a2_first_stocked_before_release_share"}


def test_structural_split_coverage_is_reproduced():
    report = reconcile_cleaned_pairs(synthetic_frame())
    structural = structural_by_name(report)
    assert structural["split_partition_covers_all_pairs"].status == REPRODUCED
    assert structural["a3_pairs_equal_negative_pairs_before_a3"].status == MISMATCH


def test_missing_columns_raise_a_contract_error():
    frame = synthetic_frame().drop(columns=["stock_3"])
    with pytest.raises(ContractError) as error:
        reconcile_cleaned_pairs(frame)
    assert "stock_3" in str(error.value)


def test_report_serialises_and_renders():
    report = reconcile_cleaned_pairs(synthetic_frame())
    payload = report.as_dict()
    assert payload["pair_count"] == 4
    assert payload["ok"] is False
    assert "a4_pairs" in payload["unsettled"]
    text = report.markdown()
    assert "# Cleaned-pairs reconciliation" in text
    assert "needs_raw_events" in text


def _real_pairs():
    candidates = [os.environ.get("VISUELLE_CLEANED_PAIRS"),
                  ROOT / "data" / "cleaned" / "cleaned_pairs.csv"]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


REAL_PAIRS = _real_pairs()


@pytest.mark.skipif(REAL_PAIRS is None, reason="cleaned pairs absent")
def test_real_cleaned_pairs_reconcile_every_settleable_figure():
    report = reconcile_cleaned_pairs(pd.read_csv(REAL_PAIRS))
    figures = by_name(report)
    assert report.pair_count == 106850
    assert report.ok is True
    for name in ("pair_total", "split.train", "split.validation", "split.test",
                 "season.train", "season.validation", "season.test",
                 "aw19_pairs", "late_aw19_pairs", "stockout_pairs",
                 "stockout_rate", "a3_pairs", "negative_pairs_with_a4",
                 "negative_rate_with_a4", "median_sell_through",
                 "zero_sales_with_zero_stock_share"):
        assert figures[name].status == REPRODUCED, (name, figures[name])
    assert figures["a4_pairs"].status == NEEDS_RAW_EVENTS
    assert figures["a2_first_stocked_before_release_share"].status == NEEDS_RAW_EVENTS
    assert figures["negative_cells"].status == NEEDS_RAW_SALES
    assert figures["zero_sales_share"].status == NEEDS_RAW_SALES
    structural = structural_by_name(report)
    assert structural["split_partition_covers_all_pairs"].status == REPRODUCED