"""End-to-end: bundle completeness, stage order, determinism, optional channel."""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pytest

from src.config.loader import load_config
from src.errors import StageError
from src.pipeline import STAGES, run

pytestmark = pytest.mark.integration


def test_bundle_has_every_mandatory_member(sample_bundle):
    sample_bundle.validate_complete()
    assert sample_bundle.metrics
    assert sample_bundle.replenishment is not None
    assert sample_bundle.risk is not None
    assert sample_bundle.quality.n_rows > 0
    assert sample_bundle.provisional
    assert sample_bundle.demonstration.key is not None


def test_stage_order_is_structural():
    names = [stage.split(" ", 1)[0] for stage in STAGES]
    assert names[0] == "S0"
    assert names.index("S1") < names.index("S9")
    assert names.index("S8") < names.index("S9") < names.index("S11")
    assert names.index("S11") < names.index("S12") < names.index("S13")


def test_incomplete_bundle_is_rejected(sample_bundle):
    broken = copy.copy(sample_bundle)
    broken.metrics = {}
    with pytest.raises(StageError) as info:
        broken.validate_complete()
    assert "metrics" in str(info.value)


def test_decision_week_invariant_is_enforced(sample_bundle):
    broken = copy.copy(sample_bundle)
    recommendation = copy.copy(sample_bundle.replenishment)
    object.__setattr__(recommendation, "forecast_origin_week", 99)
    broken.replenishment = recommendation
    with pytest.raises(StageError) as info:
        broken.validate_complete()
    assert "decision_week" in str(info.value)


def test_two_runs_produce_identical_figures(sample_dataset):
    cfg = load_config(sample_dataset["config"], root=sample_dataset["root"])
    first = run(cfg, verbose=False)
    second = run(cfg, verbose=False)
    assert first.metrics.keys() == second.metrics.keys()
    for name in first.metrics:
        assert first.metrics[name].wape_pooled == second.metrics[name].wape_pooled
        assert first.metrics[name].wape_item_level == second.metrics[name].wape_item_level
    assert first.replenishment.recommended_qty == second.replenishment.recommended_qty
    assert first.seed == second.seed


def test_weather_absence_does_not_change_record_count(tmp_path, sample_bundle):
    from tests.conftest import write_five_file_dataset

    dataset = write_five_file_dataset(tmp_path, items=40, stores=8, seed=42)
    Path(dataset["files"]["vis2_weather_data.csv"]).unlink()
    cfg = load_config(dataset["config"], root=dataset["root"])
    bundle = run(cfg, verbose=False)
    weather = [s for s in bundle.channels if s.name == "weather"][0]
    assert weather.value == "skipped"
    assert weather.reason == "file_absent"
    assert bundle.quality.n_rows == sample_bundle.quality.n_rows

def test_unit_provenance_and_register_version(sample_bundle):
    assert sample_bundle.register_version == 3
    provenance = sample_bundle.unit_provenance.as_dict()
    assert provenance["unit"] == "raw_sales_units"
    assert provenance["applied_scale_factor"] is None
    assert provenance["sign_policy"] == "clip_at_zero"
    assert provenance["register_version"] == 3
    assert sample_bundle.split.preset["statement"]


def test_preset_files_on_disk_do_not_change_the_main_path(tmp_path, sample_bundle):
    from tests.conftest import write_five_file_dataset

    dataset = write_five_file_dataset(tmp_path, items=40, stores=8, seed=42,
                                       preset_files_present=True)
    cfg = load_config(dataset["config"], root=dataset["root"])
    bundle = run(cfg, verbose=False)
    assert bundle.split.preset["present"] is True
    assert bundle.split.preset["used"] is False
    assert bundle.split.counts == sample_bundle.split.counts
    assert bundle.split.cut_train_end == sample_bundle.split.cut_train_end
    assert bundle.quality.n_rows == sample_bundle.quality.n_rows
    for name in bundle.metrics:
        assert bundle.metrics[name].wape_item_level ==             sample_bundle.metrics[name].wape_item_level
    assert bundle.secondary_comparison is None


def test_sign_policy_drill_changes_figures_but_not_the_count(tmp_path):
    from tests.conftest import make_config, write_five_file_dataset

    import dataclasses

    from src.config.schema import CleaningSection, NegativeSalesPolicyName

    runs = {}
    for policy_name in ("clip_at_zero", "retain_as_returns"):
        root = tmp_path / policy_name
        dataset = write_five_file_dataset(root, items=40, stores=8, seed=42,
                                          negative_weeks=12)
        cfg = load_config(dataset["config"], root=dataset["root"])
        cfg = dataclasses.replace(
            cfg, cleaning=CleaningSection(NegativeSalesPolicyName(policy_name)))
        runs[policy_name] = run(cfg, verbose=False)

    clip, retain = runs["clip_at_zero"], runs["retain_as_returns"]
    assert clip.quality.negative_cells == retain.quality.negative_cells == 12
    assert clip.split.counts == retain.split.counts
    moved = (clip.metrics["seasonal_naive"].wape_item_level
             != retain.metrics["seasonal_naive"].wape_item_level) or (
        clip.replenishment.reorder_point != retain.replenishment.reorder_point) or (
        clip.demonstration.sellout_week != retain.demonstration.sellout_week)
    assert moved
    assert clip.unit_provenance.sign_policy == "clip_at_zero"
    assert retain.unit_provenance.sign_policy == "retain_as_returns"


def test_secondary_comparison_is_labelled_converted_and_isolated(tmp_path):
    import dataclasses

    import pandas as pd

    from tests.conftest import write_five_file_dataset

    dataset = write_five_file_dataset(tmp_path, items=40, stores=8, seed=42,
                                       preset_files_present=True)
    raw = Path(dataset["raw"])
    sales = pd.read_csv(raw / "sales.csv")
    week_columns = [str(week) for week in range(12)]
    sales[week_columns] = sales[week_columns] / 53.0
    sales.to_csv(raw / "stfore_train.csv", index=False)
    sales.to_csv(raw / "stfore_test.csv", index=False)

    cfg = load_config(dataset["config"], root=dataset["root"])
    cfg = dataclasses.replace(
        cfg, split=dataclasses.replace(
            cfg.split, use_preset_files=True, preset_scale_factor=53.0,
            preset_files=("stfore_train.csv", "stfore_test.csv")))
    bundle = run(cfg, verbose=False)

    secondary = bundle.secondary_comparison
    assert secondary is not None
    assert secondary["label"] == "secondary"
    assert secondary["converted"] is True
    assert secondary["applied_scale_factor"] == 53.0
    assert bundle.split.preset["used"] is True
    assert "secondary" not in bundle.metrics

    from src.report.artefacts import JsonCsvWriter
    JsonCsvWriter(True, tmp_path / "output").write(bundle, "run-secondary")
    assert (tmp_path / "output" / "run-secondary" /
            "secondary_comparison.json").exists()
