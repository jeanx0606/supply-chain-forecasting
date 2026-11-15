"""Sample generator: deterministic, contract-shaped, refuses to clobber (A2)."""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from src.errors import StageError
from src.sample import MARKER_NAME, is_sample_dataset, make_sample_data


def checksums(folder):
    return {path.name: hashlib.sha1(path.read_bytes()).hexdigest()
            for path in sorted(folder.iterdir()) if path.is_file()}


def test_same_seed_produces_identical_files(tmp_path):
    first = make_sample_data(tmp_path / "a", n_items=20, n_stores=6, seed=11, overwrite=True)
    second = make_sample_data(tmp_path / "b", n_items=20, n_stores=6, seed=11, overwrite=True)
    left = checksums(tmp_path / "a")
    right = checksums(tmp_path / "b")
    assert left == right
    assert len(first["files"]) == 5


def test_different_seed_differs(tmp_path):
    make_sample_data(tmp_path / "a", n_items=20, n_stores=6, seed=11, overwrite=True)
    make_sample_data(tmp_path / "b", n_items=20, n_stores=6, seed=12, overwrite=True)
    assert checksums(tmp_path / "a")["sales.csv"] != checksums(tmp_path / "b")["sales.csv"]


def test_contract_shape_and_marker(tmp_path):
    out = make_sample_data(tmp_path / "raw", n_items=12, n_stores=5, seed=42, overwrite=True)
    sales = pd.read_csv(tmp_path / "raw" / "sales.csv")
    for week in range(12):
        assert str(week) in sales.columns
    release = pd.to_datetime(sales["release_date"])
    assert (release.dt.weekday == 0).all()
    assert release.min() >= pd.Timestamp("2016-11-28")
    assert release.max() <= pd.Timestamp("2019-12-30")
    coverage = len(sales) / (sales["external_code"].nunique() * sales["retail"].nunique())
    assert coverage < 1.0
    weather = pd.read_csv(tmp_path / "raw" / "vis2_weather_data.csv")
    assert "/" in str(weather["date"].iloc[0])
    marker = json.loads((tmp_path / "raw" / MARKER_NAME).read_text(encoding="utf-8"))
    assert marker["self_test_only"] is True
    assert is_sample_dataset(tmp_path / "raw") is True
    assert out["notice"].startswith("[sample]")


def test_refuses_to_overwrite_without_flag(tmp_path):
    make_sample_data(tmp_path / "raw", n_items=5, n_stores=3, seed=1, overwrite=True)
    with pytest.raises(StageError) as info:
        make_sample_data(tmp_path / "raw", n_items=5, n_stores=3, seed=1)
    assert "--overwrite" in str(info.value)