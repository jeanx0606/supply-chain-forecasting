"""Chapter pack: traceable figures, raw units, deterministic regeneration (A11, B28)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.report.chapter import build_pack, write_pack


def test_every_figure_traces_to_a_bundle_field(sample_bundle):
    bundle = sample_bundle.to_dict()
    pack = build_pack(bundle)
    by_id = {figure["figure"]: figure for figure in pack["figures"]}
    assert by_id["quality.n_rows"]["value"] == bundle["quality"]["n_rows"]
    assert by_id["replenishment.recommended_qty"]["value"] == \
        bundle["replenishment"]["recommended_qty"]
    assert by_id["selection.chosen_model"]["value"] == bundle["selection"]["chosen_model"]
    assert by_id["unit_provenance.unit"]["value"] == "raw_sales_units"
    assert all(figure["reps"] for figure in pack["figures"])


def test_metric_figures_carry_the_test_partition(sample_bundle):
    pack = build_pack(sample_bundle.to_dict())
    metric_figures = [f for f in pack["figures"] if f["figure"].startswith("metric.")]
    assert metric_figures
    assert all(f["partition"].startswith("test") for f in metric_figures)


def test_regeneration_is_byte_identical(sample_bundle, tmp_path):
    bundle = sample_bundle.to_dict()
    first = Path(write_pack(bundle, tmp_path, "run-chapter"))
    snapshot = {p.name: p.read_bytes() for p in first.iterdir()}
    second = Path(write_pack(bundle, tmp_path, "run-chapter"))
    assert first == second
    assert {p.name: p.read_bytes() for p in second.iterdir()} == snapshot


def test_pack_lives_outside_the_output_tree(sample_bundle, tmp_path):
    path = Path(write_pack(sample_bundle.to_dict(), tmp_path / "reports", "run"))
    assert "chapter_pack" in path.parts
    assert "output" not in path.parts


def test_pack_labels_provisional_reps(sample_bundle, tmp_path):
    path = Path(write_pack(sample_bundle.to_dict(), tmp_path, "run"))
    summary = (path / "summary.md").read_text(encoding="utf-8")
    assert "REP-18" in summary or "REP-19" in summary
    assert "unit" in summary.lower()


def test_bias_figures_carry_raw_units_not_ratio(sample_bundle):
    bundle = sample_bundle.to_dict()
    pack = build_pack(bundle)
    name = bundle["main_model"]
    by_key = {(figure["figure"], figure["partition"]): figure
              for figure in pack["figures"]}
    for metric_id in ("forecast_bias", "forecast_bias_item_level"):
        figure = by_key[(f"metric.{metric_id}", f"test:{name}")]
        assert figure["unit"] == "raw_sales_units"
        assert figure["value"] == pytest.approx(
            bundle["metrics"][name][metric_id], abs=5e-5)
    assert by_key[(f"metric.mae", f"test:{name}")]["unit"] == "raw_sales_units"
    assert by_key[(f"metric.wape_item_level", f"test:{name}")]["unit"] == "ratio"


def test_sigma_figures_cover_the_chosen_model_and_every_candidate(sample_bundle):
    bundle = sample_bundle.to_dict()
    pack = build_pack(bundle)
    by_key = {(figure["figure"], figure["partition"]): figure
              for figure in pack["figures"]}
    main = by_key[("model.sigma_e", "validation")]
    assert main["value"] == pytest.approx(bundle["error_scale"]["sigma_e"], abs=5e-5)
    for name in bundle["error_scale_by_model"]:
        figure = by_key[("model.sigma_e", f"validation:{name}")]
        assert figure["value"] == pytest.approx(
            bundle["error_scale_by_model"][name]["sigma_e"], abs=5e-5)
        assert figure["reps"]


def test_reconstruction_and_assumption_figures_trace_to_the_bundle(sample_bundle):
    bundle = sample_bundle.to_dict()
    pack = build_pack(bundle)
    by_id = {figure["figure"]: figure for figure in pack["figures"]}
    demonstration = bundle["demonstration"]
    assert by_id["inventory.initial_stock_base"]["value"] == pytest.approx(
        demonstration["initial_stock"]["base_value"], abs=5e-5)
    assert by_id["inventory.initial_stock_correction"]["value"] == pytest.approx(
        demonstration["initial_stock"]["correction"], abs=5e-5)
    assert by_id["inventory.would_go_negative"]["value"] == \
        demonstration["reconstruction"]["would_go_negative"]
    assert by_id["reconstruction.alpha"]["value"] == pytest.approx(1.0)
    assert by_id["assumptions.count"]["value"] == 6
    assert by_id["assumptions.open_decisions"]["value"] == "A5,A6"
    assert by_id["replenishment.protection_weeks"]["value"] == 2
    assert by_id["replenishment.simulation_label"]["value"] == \
        "simulation under stated assumptions"
    assert all(figure["reps"] for figure in pack["figures"])


def test_error_slice_figures_trace_to_the_bundle(sample_bundle):
    bundle = sample_bundle.to_dict()
    pack = build_pack(bundle)
    payload = bundle["error_slices"]
    by_key = {(figure["figure"], figure["partition"]): figure
              for figure in pack["figures"]}
    units = {"wape": "ratio", "mae": "raw_sales_units",
             "rmse": "raw_sales_units", "bias": "raw_sales_units"}
    for row in payload["slices"]:
        for metric, unit in units.items():
            figure = by_key[(f"error_slice.{metric}",
                             f"test:{row['slice']}:{row['group']}")]
            assert figure["unit"] == unit
            assert figure["reps"]
            if row[metric] is not None:
                assert figure["value"] == pytest.approx(row[metric], abs=5e-5)
    assert by_key[("error_slice.reported_count", "")]["value"] == \
        payload["reported_count"]
    assert by_key[("error_slice.min_slice_size", "")]["value"] == 10


def test_split_figures_name_the_partition(sample_bundle):
    bundle = sample_bundle.to_dict()
    pack = build_pack(bundle)
    by_id = {figure["figure"]: figure for figure in pack["figures"]}
    assert by_id["split.method"]["value"] == "chronological"
    assert by_id["split.method"]["value"] == bundle["split"]["method"]
    assert by_id["split.counts.train"]["value"] == bundle["split"]["counts"]["train"]
    assert by_id["split.counts.test"]["value"] == bundle["split"]["counts"]["test"]
    assert by_id["split.preset.statement"]["value"] == \
        bundle["split"]["preset"]["statement"]
    assert by_id["split.preset.statement"]["reps"] == ["REP-05"]


def test_scenario_summary_figures_state_the_main_setting(sample_bundle):
    bundle = sample_bundle.to_dict()
    pack = build_pack(bundle)
    by_id = {figure["figure"]: figure for figure in pack["figures"]}
    assert by_id["scenario.count"]["value"] == 6
    assert by_id["scenario.service_level"]["value"] == pytest.approx(0.95)
    assert "7.7" in by_id["scenario.cost_index_status"]["value"]
    assert by_id["scenario.unavailable"]["value"] == "none"
    for figure_id in ("scenario.count", "scenario.service_level",
                      "scenario.cost_index_status", "scenario.unavailable"):
        assert by_id[figure_id]["reps"]


def test_scenario_rows_become_labelled_figures(sample_bundle):
    bundle = sample_bundle.to_dict()
    pack = build_pack(bundle)
    rows = bundle["scenarios"]["rows"]
    by_key = {(figure["figure"], figure["partition"]): figure
              for figure in pack["figures"]}
    units = {"fill_rate": "ratio", "stockout_rate": "ratio",
             "overstock_ratio": "ratio", "turnover": "ratio",
             "weeks_of_cover": "week"}
    for row in rows:
        for metric, unit in units.items():
            figure = by_key[(f"scenario.{metric}", f"test:{row['scenario']}")]
            assert figure["unit"] == unit
            assert figure["reps"]
            if row[metric] is None:
                assert figure["value"] is None
            else:
                assert figure["value"] == pytest.approx(row[metric], abs=5e-5)