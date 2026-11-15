"""Artefacts: every documented file parses; provisional refs travel headlessly (D22)."""
from __future__ import annotations

import copy
import json

from src.report.artefacts import JsonCsvWriter, NullWriter


def test_all_documented_artefacts_exist_and_parse(sample_bundle, tmp_path):
    writer = JsonCsvWriter(True, tmp_path / "output")
    location = writer.write(sample_bundle, "run-test")
    folder = tmp_path / "output" / "run-test"
    assert location == str(folder)
    for name in ("bundle.json", "quality.csv", "metrics.csv", "inventory.csv",
                 "replenishment.json", "run_meta.json"):
        path = folder / name
        assert path.exists(), name
        if name.endswith(".json"):
            json.loads(path.read_text(encoding="utf-8"))


def test_bundle_json_carries_provisional_refs_and_decision_weeks(sample_bundle, tmp_path):
    JsonCsvWriter(True, tmp_path / "output").write(sample_bundle, "run-test")
    payload = json.loads((tmp_path / "output" / "run-test" / "bundle.json")
                         .read_text(encoding="utf-8"))
    assert payload["provisional"]
    assert payload["provisional"][0]["rep_id"].startswith("REP-")
    assert payload["provisional"][0]["confirmer_role"]
    assert payload["replenishment"]["decision_week"] == \
        payload["replenishment"]["forecast_origin_week"]


def test_disabled_writer_writes_nothing(sample_bundle, tmp_path):
    assert NullWriter().write(sample_bundle, "run") is None
    assert not (tmp_path / "output").exists()


def test_second_write_replaces_atomically(sample_bundle, tmp_path):
    writer = JsonCsvWriter(True, tmp_path / "output")
    writer.write(sample_bundle, "run-test")
    writer.write(sample_bundle, "run-test")
    assert (tmp_path / "output" / "run-test" / "bundle.json").exists()
    assert not list((tmp_path / "output").glob(".staging-*"))

def test_register_version_and_unit_columns(sample_bundle, tmp_path):
    import csv

    from src.report.artefacts import JsonCsvWriter

    JsonCsvWriter(True, tmp_path / "output").write(sample_bundle, "run-test")
    folder = tmp_path / "output" / "run-test"
    run_meta = json.loads((folder / "run_meta.json").read_text(encoding="utf-8"))
    assert run_meta["register_version"] == 3
    assert run_meta["unit"] == "raw_sales_units"
    assert run_meta["applied_scale_factor"] is None
    assert run_meta["sign_policy"] == "clip_at_zero"

    with open(folder / "metrics.csv", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    for column in ("unit", "applied_scale_factor", "sign_policy"):
        assert column in header

    with open(folder / "quality.csv", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    for column in ("negative_sales_policy", "negative_cells", "unit",
                   "applied_scale_factor"):
        assert column in header


def test_missing_register_version_fails_instead_of_defaulting(sample_bundle, tmp_path):
    from src.report.artefacts import JsonCsvWriter

    broken = copy.copy(sample_bundle)
    broken.register_version = None
    writer = JsonCsvWriter(True, tmp_path / "output")
    try:
        writer.write(broken, "run-broken")
    except KeyError:
        pass
    else:
        raise AssertionError("missing register_version must fail the writer")
    assert not (tmp_path / "output" / "run-broken").exists()


def test_rep17_is_provisional_and_rep05_is_confirmed(sample_bundle):
    rep_ids = {ref.rep_id for ref in sample_bundle.provisional}
    assert "REP-17" in rep_ids
    assert "REP-05" not in rep_ids
