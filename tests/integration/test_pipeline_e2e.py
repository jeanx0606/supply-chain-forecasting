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


def test_bundle_carries_the_scenario_ladder(sample_bundle):
    scenarios = sample_bundle.scenarios
    assert scenarios is not None
    names = [row["scenario"] for row in scenarios["rows"]]
    assert names[:2] == ["R0", "R0b"]
    assert set(names[2:]) == {"R1", "R2", "R3", "R4"}
    settings = scenarios["settings"]
    assert settings["lead_time_weeks"] == 1
    assert settings["order_cycle_weeks"] == 1
    assert settings["protection_weeks"] == 2
    assert settings["simulation_label"] == "simulation under stated assumptions"
    assert scenarios["unavailable"] == []
    for row in scenarios["rows"]:
        assert row["cost_index"] is None
        if row["fill_rate"] is not None:
            assert 0.0 <= row["fill_rate"] <= 1.0
        if row["stockout_rate"] is not None:
            assert 0.0 <= row["stockout_rate"] <= 1.0


def test_season_split_configuration_runs_end_to_end(sample_dataset):
    import dataclasses

    from src.config.schema import SplitPolicyName

    cfg = load_config(sample_dataset["config"], root=sample_dataset["root"])
    cfg = dataclasses.replace(
        cfg, split=dataclasses.replace(cfg.split, policy=SplitPolicyName.SEASON))
    bundle = run(cfg, verbose=False)
    assert bundle.split.method == "season"
    assert bundle.split.counts["train"] > 0
    assert bundle.split.counts["validation"] > 0
    assert bundle.split.counts["test"] > 0
    assert bundle.scenarios["settings"]["forecast_origin_week"] == 2


def test_bundle_publishes_the_a3_audit(sample_bundle):
    audit = sample_bundle.reconstruction_audit
    assert audit["reported_a3_configuration"] == "on"
    assert audit["definition"].startswith("stock <= 0 in weeks 0-11")
    assert audit["declared"]["pairs_negative_excluding_a4"] == 1842
    assert audit["declared"]["pairs_negative_including_a4"] == 1953
    assert audit["declared"]["a4_pairs"] == 112
    assert audit["declared"]["a4_pairs_requiring_a3"] == 111
    assert audit["a4_pairs"] >= audit["a4_pairs_requiring_a3"]


def test_bundle_carries_error_slices(sample_bundle):
    payload = sample_bundle.error_slices
    assert payload is not None
    assert payload["available"] is True
    assert payload["test_partition"] == "test"
    assert "not proxy" in payload["definition"]
    assert payload["slices"]
    assert payload["min_slice_size"] == 10
    assert all({"n", "wape", "mae", "rmse", "bias", "status"} <= set(row)
               for row in payload["slices"])


def test_bundle_without_error_slices_is_rejected(sample_bundle):
    broken = copy.copy(sample_bundle)
    broken.error_slices = None
    with pytest.raises(StageError) as info:
        broken.validate_complete()
    assert "error_slices" in str(info.value)


def test_bundle_without_the_ladder_is_rejected(sample_bundle):
    broken = copy.copy(sample_bundle)
    broken.scenarios = None
    with pytest.raises(StageError) as info:
        broken.validate_complete()
    assert "scenarios" in str(info.value)


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

    import pandas as pd

    from src.config.schema import CleaningSection, NegativeSalesPolicyName

    runs = {}
    for policy_name in ("clip_at_zero", "retain_as_returns"):
        root = tmp_path / policy_name
        dataset = write_five_file_dataset(root, items=40, stores=8, seed=42)
        sales = pd.read_csv(dataset["files"]["sales.csv"])
        target = sales["release_date"].idxmax()
        for week in range(12):
            sales.loc[target, str(week)] = -2.0
        sales.to_csv(dataset["files"]["sales.csv"], index=False)
        cfg = load_config(dataset["config"], root=dataset["root"])
        cfg = dataclasses.replace(
            cfg, cleaning=CleaningSection(NegativeSalesPolicyName(policy_name)))
        runs[policy_name] = run(cfg, verbose=False)

    clip, retain = runs["clip_at_zero"], runs["retain_as_returns"]
    assert clip.quality.negative_cells == retain.quality.negative_cells == 12
    assert clip.split.counts == retain.split.counts
    moved = (clip.metrics["seasonal_naive"].wape_item_level
             != retain.metrics["seasonal_naive"].wape_item_level) or (
        clip.error_scale["sigma_e"] != retain.error_scale["sigma_e"]) or (
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
