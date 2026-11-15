"""Chapter pack: traceable figures, raw units, deterministic regeneration (A11, B28)."""
from __future__ import annotations

from pathlib import Path

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