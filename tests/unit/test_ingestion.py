"""Ingestion contracts and canonical weekly labels (D10, DFX-C1)."""
from __future__ import annotations

import os

import pandas as pd
import pytest

from src.errors import ContractError
from src.io.loader import load_price, load_sales


def test_canonical_week_labels(tmp_path, sample_dataset):
    from pathlib import Path

    sales, stats = load_sales(Path(sample_dataset["files"]["sales.csv"]), 12, "q")
    for n in range(1, 13):
        assert f"q{n}" in sales.columns
    assert stats.rows_read == len(sales)

    price, _ = load_price(Path(sample_dataset["files"]["price_discount_series.csv"]), 12)
    for n in range(1, 13):
        assert f"disc_w{n}" in price.columns


def test_missing_release_date_aborts_naming_file_and_field(tmp_path):
    frame = pd.DataFrame({"external_code": [1], "retail": [2],
                          **{n: [1.0] for n in range(12)}})
    path = tmp_path / "sales.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ContractError) as info:
        load_sales(path, 12, "q")
    assert "release_date" in str(info.value)
    assert "sales.csv" in str(info.value)


def test_surplus_week_columns_are_ignored_and_counted(tmp_path):
    base = pd.DataFrame({
        "external_code": [1], "retail": [2], "release_date": ["2019-01-07"],
        **{n: [float(n)] for n in range(14)},
    })
    path = tmp_path / "sales.csv"
    base.to_csv(path, index=False)
    frame, stats = load_sales(path, 12, "q")
    assert stats.ignored_columns == 2
    assert "q12" in frame.columns and "q13" not in frame.columns


def test_too_few_week_columns_abort_with_expected_vs_observed(tmp_path):
    base = pd.DataFrame({
        "external_code": [1], "retail": [2], "release_date": ["2019-01-07"],
        **{n: [float(n)] for n in range(8)},
    })
    path = tmp_path / "sales.csv"
    base.to_csv(path, index=False)
    with pytest.raises(ContractError) as info:
        load_sales(path, 12, "q")
    assert "not enough numeric week columns" in str(info.value)
    assert "observed 8" in str(info.value)


def test_duplicate_item_store_counted(tmp_path):
    base = pd.DataFrame({
        "external_code": [1, 1], "retail": [2, 2], "release_date": ["2019-01-07"] * 2,
        **{n: [1.0, 2.0] for n in range(12)},
    })
    path = tmp_path / "sales.csv"
    base.to_csv(path, index=False)
    _, stats = load_sales(path, 12, "q")
    assert stats.duplicates_detected == 2


def test_raw_file_is_never_modified(sample_dataset):
    from pathlib import Path

    path = Path(sample_dataset["files"]["sales.csv"])
    before = (path.stat().st_mtime_ns, path.stat().st_size, os.path.getsize(path))
    load_sales(path, 12, "q")
    after = (path.stat().st_mtime_ns, path.stat().st_size, os.path.getsize(path))
    assert before == after