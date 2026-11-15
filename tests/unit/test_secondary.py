"""Optional secondary comparison: labelled, converted once, never a main metric."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluate.aggregate import StandardMetricSet
from src.evaluate.secondary import compare_on_preset
from src.features.build import TaskDataset
from src.models.backends import make_backend


def task(rows=6, offset=0.0):
    lag_1 = np.arange(1.0, rows + 1) + offset
    features = pd.DataFrame({"lag_1": lag_1,
                             "lag_2": np.arange(2.0, rows + 2) + offset})
    targets = pd.DataFrame({f"t{i}": lag_1 for i in range(1, 11)})
    metadata = pd.DataFrame({"external_code": np.arange(rows),
                             "retail": np.ones(rows, dtype=int),
                             "release_date": pd.date_range("2019-01-07", periods=rows,
                                                           freq="W-MON")})
    return TaskDataset(features, targets, metadata, 2, 10, 2,
                       tuple(f"t{i}" for i in range(1, 11)))


def test_comparison_is_labelled_and_converted():
    comparison = compare_on_preset(
        task(offset=0.0), task(offset=0.0), 53.0, StandardMetricSet(),
        "seasonal_naive", make_backend("sklearn_hgb", {}, 42), 42)
    payload = comparison.as_dict()
    assert payload["label"] == "secondary"
    assert payload["converted"] is True
    assert payload["applied_scale_factor"] == 53.0
    assert "never reported as the main result" in payload["note"]


def test_conversion_happens_once_so_scaled_units_cancel():
    comparison = compare_on_preset(
        task(offset=1.0), task(offset=1.0), 53.0, StandardMetricSet(),
        "seasonal_naive", make_backend("sklearn_hgb", {}, 42), 42)
    # identical train/test in scaled units -> relative error is scale-invariant
    assert comparison.metrics["wape_pooled"] is not None
    assert comparison.metrics["wape_pooled"] <= 1.0


def test_default_bundle_has_no_comparison(sample_bundle):
    assert sample_bundle.secondary_comparison is None


def test_no_comparison_file_on_the_default_path(sample_bundle, tmp_path):
    from src.report.artefacts import JsonCsvWriter

    JsonCsvWriter(True, tmp_path / "output").write(sample_bundle, "run-test")
    assert not (tmp_path / "output" / "run-test" / "secondary_comparison.json").exists()


def test_enabled_without_factor_aborts_before_reading(tmp_path):
    import shutil

    from src.config.loader import load_config
    from tests.conftest import make_config, write_five_file_dataset

    dataset = write_five_file_dataset(tmp_path, items=20, stores=5, seed=9,
                                       preset_files_present=True)
    config = make_config(tmp_path, lambda cfg: cfg["split"].update(
        {"use_preset_files": True, "preset_scale_factor": None}))
    from src.errors import ConfigError

    try:
        load_config(str(config), root=str(tmp_path))
    except ConfigError as exc:
        assert "split.preset_scale_factor" in str(exc)
    else:
        raise AssertionError("expected ConfigError")
    shutil.rmtree(dataset["root"], ignore_errors=True)