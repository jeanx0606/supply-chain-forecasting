"""Read-only preflight: named failures, one data class, no pipeline import (A10)."""
from __future__ import annotations

from pathlib import Path

from src.config.loader import load_config
from src.io.preflight import classify_dataset, preflight_exit_code, run_preflight

ROOT = Path(__file__).resolve().parents[2]


def test_sample_dataset_reports_a_ready_sample(tmp_path, sample_dataset):
    cfg = load_config(sample_dataset["config"], root=sample_dataset["root"])
    report = run_preflight(cfg)
    assert report["data_class"]["data_class"] == "sample"
    assert report["readiness"] == "ready"
    assert preflight_exit_code(report) == 0
    assert report["negative_cells"] == 0
    assert report["n_items"] > 0 and report["n_stores"] > 0
    assert 0.0 <= report["pair_coverage"] <= 1.0
    assert {entry["file"] for entry in report["optional_files"]}


def test_missing_mandatory_file_names_the_file(tmp_path, sample_dataset):
    import shutil

    from tests.conftest import write_five_file_dataset

    dataset = write_five_file_dataset(tmp_path, items=20, stores=5, seed=3)
    Path(dataset["files"]["sales.csv"]).unlink()
    cfg = load_config(dataset["config"], root=dataset["root"])
    report = run_preflight(cfg)
    assert any("sales.csv" in problem for problem in report["problems"])
    assert report["readiness"].startswith("blocked")
    assert preflight_exit_code(report) == 1


def test_data_classification(tmp_path, sample_dataset):
    import shutil

    work = tmp_path / "raw"
    shutil.copytree(sample_dataset["raw"], work)
    assert classify_dataset(work)["data_class"] == "sample"
    (work / "_sample_marker.json").unlink()
    assert classify_dataset(work)["data_class"] == "real"
    assert classify_dataset(tmp_path / "empty")["data_class"] == "absent"


def test_preflight_never_imports_the_pipeline():
    text = (ROOT / "src" / "io" / "preflight.py").read_text(encoding="utf-8")
    assert "src.pipeline" not in text
    assert "from ..pipeline" not in text
    assert "import src.pipeline" not in text

def test_modified_file_beside_the_marker_is_mixed(tmp_path):
    from tests.conftest import write_five_file_dataset

    dataset = write_five_file_dataset(tmp_path, items=20, stores=5, seed=21)
    work = Path(dataset["raw"])
    sales = work / "sales.csv"
    text = sales.read_text(encoding="utf-8")
    sales.write_text(text.replace("2019", "2018", 1), encoding="utf-8")
    result = classify_dataset(work)
    assert result["data_class"] == "mixed"
    assert "modified" in result["reason"] or "disagree" in result["reason"]
