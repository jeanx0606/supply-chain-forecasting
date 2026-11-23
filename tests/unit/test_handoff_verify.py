"""Handoff verifier: statuses, parity oracle, and the guarded real-artefact check."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from src.handoff.forecast import load_manifest
from src.handoff.published import PUBLISHED_TABLE_7_1
from src.handoff.verify import FAIL, INFO, PASS, SKIPPED, verify_handoff

from .test_handoff_forecast import SIGMA, FakeBooster, make_frame, write_manifest

ROOT = Path(__file__).resolve().parents[2]
ORACLE = '''
import numpy as np
from src.handoff.forecast import build_features


def predict(model, df):
    matrix = build_features(df, model.feature_name(),
                            getattr(model, "pandas_categorical", None))
    raw = np.asarray(model.predict(matrix), dtype=float).reshape(10, len(df)).T
    return np.clip(raw, 0, None)
'''
ORACLE_ZEROS = '''
import numpy as np


def predict(model, df):
    return np.zeros((len(df), 10))
'''


def pairs_with_split(rows: int = 4):
    frame = make_frame(rows)
    frame["split"] = ["validation"] * rows
    frame.loc[frame.index[:1], "split"] = "train"
    frame.loc[frame.index[-2:], "split"] = "test"
    return frame


def write_pairs(tmp_path, frame):
    path = tmp_path / "cleaned_pairs.csv"
    frame.to_csv(path, index=False)
    return path


def write_oracle(directory, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "predict_forecast.py"
    path.write_text(body, encoding="utf-8")
    return directory


def statuses(report) -> dict:
    return {check.name: check.status for check in report.checks}


def test_checks_run_on_a_supplied_pair_file(tmp_path):
    write_manifest(tmp_path)
    manifest = load_manifest(tmp_path)
    pairs = write_pairs(tmp_path, pairs_with_split())
    report = verify_handoff(tmp_path, pairs_path=pairs, booster=FakeBooster(),
                            manifest=manifest)
    observed = statuses(report)
    assert observed["manifest"] == PASS
    assert observed["feature_contract"] == PASS
    assert observed["sigma_contract"] == PASS
    assert observed["prediction_shape"] == PASS
    assert observed["non_negative"] == PASS
    assert observed["no_future_sales"] == PASS
    assert observed["uses_observed_sales"] == PASS
    assert observed["uses_future_discounts"] == INFO
    assert observed["parity_with_reference"] == SKIPPED
    assert observed["replenishment_target"] == PASS
    assert observed["replenishment_rounding"] == SKIPPED
    assert observed["sigma_provenance"] == INFO
    assert observed["test_metrics"] == INFO
    assert observed["published_table"] == SKIPPED
    assert report.ok is True
    assert report.metrics["LightGBM"]["n_pairs"] == 2
    assert report.metrics["LightGBM"]["cum_err"] is not None


def test_checks_are_skipped_without_an_input_file(tmp_path):
    write_manifest(tmp_path)
    manifest = load_manifest(tmp_path)
    report = verify_handoff(tmp_path, pairs_path=None, booster=FakeBooster(),
                            manifest=manifest)
    observed = statuses(report)
    for name in ("prediction_shape", "no_future_sales", "parity_with_reference",
                 "test_metrics"):
        assert observed[name] == SKIPPED
    assert report.ok is True


def test_parity_passes_against_a_matching_oracle(tmp_path):
    write_manifest(tmp_path)
    manifest = load_manifest(tmp_path)
    pairs = write_pairs(tmp_path, pairs_with_split())
    oracle = write_oracle(tmp_path / "oracle", ORACLE)
    report = verify_handoff(tmp_path, pairs_path=pairs, reference_dir=oracle,
                            booster=FakeBooster(), manifest=manifest)
    observed = statuses(report)
    assert observed["parity_with_reference"] == PASS
    assert report.parity["max_abs_difference"] == pytest.approx(0.0, abs=1e-9)
    assert report.ok is True


def test_parity_fails_against_a_diverging_oracle(tmp_path):
    write_manifest(tmp_path)
    manifest = load_manifest(tmp_path)
    pairs = write_pairs(tmp_path, pairs_with_split())
    oracle = write_oracle(tmp_path / "oracle", ORACLE_ZEROS)
    report = verify_handoff(tmp_path, pairs_path=pairs, reference_dir=oracle,
                            booster=FakeBooster(), manifest=manifest)
    assert statuses(report)["parity_with_reference"] == FAIL
    assert report.ok is False


def test_published_table_passes_within_tolerance(tmp_path):
    write_manifest(tmp_path)
    manifest = load_manifest(tmp_path)
    pairs = write_pairs(tmp_path, pairs_with_split())
    published = {"source": "test table", "tolerance": 5.0,
                 "rows": {"LightGBM": {"wape": 0.0, "mae": 0.0, "rmse": 0.0,
                                       "bias": 0.0, "cumulative_error": 0.0}}}
    report = verify_handoff(tmp_path, pairs_path=pairs, booster=FakeBooster(),
                            manifest=manifest, published=published)
    assert statuses(report)["published_table"] == PASS
    assert report.ok is True


def test_published_table_fails_on_a_mismatch(tmp_path):
    write_manifest(tmp_path)
    manifest = load_manifest(tmp_path)
    pairs = write_pairs(tmp_path, pairs_with_split())
    published = {"source": "test table", "tolerance": 1e-9,
                 "rows": {"LightGBM": {"wape": 0.0, "mae": 0.0, "rmse": 0.0,
                                       "bias": 0.0, "cumulative_error": 0.0}}}
    report = verify_handoff(tmp_path, pairs_path=pairs, booster=FakeBooster(),
                            manifest=manifest, published=published)
    assert statuses(report)["published_table"] == FAIL
    assert report.ok is False


def test_report_renders_markdown(tmp_path):
    write_manifest(tmp_path)
    manifest = load_manifest(tmp_path)
    report = verify_handoff(tmp_path, pairs_path=None, booster=FakeBooster(),
                            manifest=manifest)
    text = report.markdown()
    assert "# Handoff verification" in text
    assert "simulation under stated assumptions" in text


def _handoff_dir():
    candidates = [os.environ.get("VISUELLE_HANDOFF_DIR"), ROOT / "data" / "handoff"]
    for candidate in candidates:
        if candidate and (Path(candidate) / "prototype_params.json").exists():
            return Path(candidate)
    return None


REAL_HANDOFF = _handoff_dir()


@pytest.mark.skipif(REAL_HANDOFF is None, reason="handoff artefacts absent")
def test_real_handoff_reproduces_the_reference_predictor():
    pairs = Path(os.environ.get("VISUELLE_CLEANED_PAIRS",
                                ROOT / "data" / "cleaned" / "cleaned_pairs.csv"))
    if not pairs.exists():
        pytest.skip("cleaned pairs absent")
    report = verify_handoff(REAL_HANDOFF, pairs_path=pairs,
                            reference_dir=REAL_HANDOFF, sample_rows=1000,
                            published=PUBLISHED_TABLE_7_1)
    observed = statuses(report)
    assert observed["feature_contract"] == PASS
    assert observed["parity_with_reference"] == PASS
    assert report.parity["max_abs_difference"] <= 0.001
    assert observed["no_future_sales"] == PASS
    assert observed["published_table"] == PASS
    assert report.manifest["sigma_e"] == SIGMA
    assert report.ok is True