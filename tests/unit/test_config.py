"""Configuration: positive load and five named negative cases (B18, D18)."""
from __future__ import annotations

import pytest

from src.config.audit import ConfigLedger, check_unconsumed
from src.config.loader import effective_config_dict, load_config
from src.errors import ConfigError


def test_valid_load_has_sixteen_sections(tmp_path):
    from tests.conftest import make_config, ROOT

    path = make_config(tmp_path)
    cfg = load_config(str(path), root=str(tmp_path))
    assert cfg.spine.input_weeks == 2
    assert cfg.spine.output_weeks == 10
    assert cfg.model.backend.value == "sklearn_hgb"
    assert cfg.truncation.restock_policy.value == "lifecycle_membership"
    filled = [name for name in ("data", "run", "time", "spine", "split", "model",
                                "baseline", "metrics", "channels", "weather",
                                "inventory", "truncation", "replenish", "demo",
                                "tasks", "output") if getattr(cfg, name, None) is not None]
    assert len(filled) == 16


def test_window_overflow_names_the_arithmetic(tmp_path):
    from tests.conftest import make_config

    path = make_config(tmp_path, lambda cfg: cfg["spine"].update(
        {"input_weeks": 4, "output_weeks": 10}))
    with pytest.raises(ConfigError) as info:
        load_config(str(path), root=str(tmp_path))
    assert "input(4) + output(10) > n_weeks(12)" in str(info.value)


def test_ratio_sum_rejected(tmp_path):
    from tests.conftest import make_config

    path = make_config(tmp_path, lambda cfg: cfg["split"].update(
        {"train_ratio": 0.80, "val_ratio": 0.15, "test_ratio": 0.15}))
    with pytest.raises(ConfigError) as info:
        load_config(str(path), root=str(tmp_path))
    assert "split.train_ratio" in str(info.value)
    assert "observed=1.1" in str(info.value)


def test_unknown_enum_value_rejected(tmp_path):
    from tests.conftest import make_config

    path = make_config(tmp_path, lambda cfg: cfg["model"].update({"backend": "xgboost"}))
    with pytest.raises(ConfigError) as info:
        load_config(str(path), root=str(tmp_path))
    assert "model.backend" in str(info.value)
    assert "unregistered selector value" in str(info.value)


def test_unconsumed_key_aborts():
    ledger = ConfigLedger(declared={"channels.gtrends", "data.n_weeks"})
    ledger.consume("data.n_weeks")
    assert check_unconsumed(ledger.declared, ledger.consumed) == ["channels.gtrends"]
    with pytest.raises(ConfigError) as info:
        ledger.assert_consumed()
    assert "declared but not consumed" in str(info.value)
    assert "observed=['channels.gtrends']" in str(info.value)


def test_service_level_range(tmp_path):
    from tests.conftest import make_config

    path = make_config(tmp_path, lambda cfg: cfg["replenish"].update(
        {"service_level": 1.2}))
    with pytest.raises(ConfigError) as info:
        load_config(str(path), root=str(tmp_path))
    assert "replenish.service_level" in str(info.value)
    assert "outside (0, 1)" in str(info.value)


def test_path_escape_rejected(tmp_path):
    from tests.conftest import make_config

    path = make_config(tmp_path, lambda cfg: cfg["output"].update({"dir": "../outside"}))
    with pytest.raises(ConfigError) as info:
        load_config(str(path), root=str(tmp_path))
    assert "path escapes the project root" in str(info.value)

# --- v1.3: sign policy and preset-factor keys -----------------------------

def test_cleaning_section_and_preset_defaults(tmp_path):
    from src.config.schema import AppConfig, CONFIG_SECTIONS
    from tests.conftest import make_config

    path = make_config(tmp_path)
    cfg = load_config(str(path), root=str(tmp_path))
    assert len(AppConfig.__dataclass_fields__) == 21
    assert len(CONFIG_SECTIONS) == 20
    assert "cleaning" in CONFIG_SECTIONS
    assert {"features", "report", "preflight"} <= set(CONFIG_SECTIONS)
    assert cfg.cleaning.negative_sales_policy.value == "clip_at_zero"
    assert cfg.split.preset_scale_factor is None
    assert cfg.split.preset_files == ("stfore_train.csv", "stfore_test.csv")
    assert cfg.model.candidates == ("seasonal_naive", "moving_average",
                                    "category_curve", "attribute_knn",
                                    "lag_regression", "lightgbm_horizon",
                                    "gradient_boosting")
    assert cfg.inventory.initial_stock_method.value == "restocked_to_release"
    assert cfg.inventory.opening_stock_alpha == 1.0
    assert cfg.inventory.negative_correction is True
    assert cfg.inventory.floor_at_zero is True
    assert cfg.model.qualification.on_none.value == "retained_default"
    assert cfg.features.price_discount_interaction is True
    assert cfg.features.weather.value == "as_supplied"
    assert cfg.preflight.negative_cell_reference == 3386
    effective = effective_config_dict(cfg)
    assert effective["cleaning"]["negative_sales_policy"] == "clip_at_zero"
    assert effective["split"]["preset_scale_factor"] is None
    assert effective["split"]["preset_files"] == ["stfore_train.csv", "stfore_test.csv"]


def test_unknown_sign_policy_names_key_and_capability(tmp_path):
    from tests.conftest import make_config

    path = make_config(tmp_path, lambda cfg: cfg["cleaning"].update(
        {"negative_sales_policy": "median"}))
    with pytest.raises(ConfigError) as info:
        load_config(str(path), root=str(tmp_path))
    message = str(info.value)
    assert "cleaning.negative_sales_policy" in message
    assert "clip_at_zero" in message and "retain_as_returns" in message
    assert "capability 5.2" in message


def test_scaled_read_requires_an_explicit_factor(tmp_path):
    from tests.conftest import make_config

    path = make_config(tmp_path, lambda cfg: cfg["split"].update(
        {"use_preset_files": True, "preset_scale_factor": None}))
    with pytest.raises(ConfigError) as info:
        load_config(str(path), root=str(tmp_path))
    message = str(info.value)
    assert "split.preset_scale_factor" in message
    assert "a scaled read requires an explicitly configured factor" in message
    assert "REP-05" in message


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True])
def test_factor_must_be_positive_and_finite(tmp_path, bad):
    from tests.conftest import make_config

    path = make_config(tmp_path, lambda cfg: cfg["split"].update(
        {"preset_scale_factor": bad}))
    with pytest.raises(ConfigError) as info:
        load_config(str(path), root=str(tmp_path))
    message = str(info.value)
    assert "split.preset_scale_factor" in message
    assert "positive finite" in message


def test_factor_with_switch_off_loads_as_inactive(tmp_path):
    from tests.conftest import make_config
    from src.preprocess.split import PresetSplitPolicy

    path = make_config(tmp_path, lambda cfg: cfg["split"].update(
        {"preset_scale_factor": 53.0}))
    cfg = load_config(str(path), root=str(tmp_path))
    assert cfg.split.preset_scale_factor == 53.0
    policy = PresetSplitPolicy(cfg.split.use_preset_files, cfg.split.preset_files,
                               cfg.split.preset_scale_factor)
    assert policy.scale_factor_state()["state"] == "inactive_parameter"


def test_duplicate_preset_names_rejected(tmp_path):
    from tests.conftest import make_config

    path = make_config(tmp_path, lambda cfg: cfg["split"].update(
        {"preset_files": ["same.csv", "same.csv"]}))
    with pytest.raises(ConfigError) as info:
        load_config(str(path), root=str(tmp_path))
    assert "preset file names must be distinct" in str(info.value)


# --- v1.4: selection and feature validation ------------------------------

def test_unknown_candidate_names_the_registered_set(tmp_path):
    from tests.conftest import make_config

    path = make_config(tmp_path, lambda cfg: cfg["model"].update(
        {"candidates": ["seasonal_naive", "xgboost"]}))
    with pytest.raises(ConfigError) as info:
        load_config(str(path), root=str(tmp_path))
    message = str(info.value)
    assert "model.candidates" in message and "xgboost" in message
    assert "gradient_boosting" in message


def test_qualification_reference_must_be_a_candidate(tmp_path):
    from tests.conftest import make_config

    path = make_config(tmp_path, lambda cfg: cfg["model"].update(
        {"candidates": ["moving_average", "gradient_boosting"]}))
    with pytest.raises(ConfigError) as info:
        load_config(str(path), root=str(tmp_path))
    assert "seasonal_naive" in str(info.value)


def test_selected_model_must_be_a_candidate(tmp_path):
    from tests.conftest import make_config

    path = make_config(tmp_path, lambda cfg: cfg["model"].update(
        {"candidates": ["seasonal_naive", "moving_average"],
         "selected": "gradient_boosting"}))
    with pytest.raises(ConfigError) as info:
        load_config(str(path), root=str(tmp_path))
    assert "model.selected" in str(info.value)


def test_selection_metric_must_equal_the_primary_metric(tmp_path):
    from tests.conftest import make_config

    path = make_config(tmp_path, lambda cfg: cfg["model"].update(
        {"selection_metric": "mae"}))
    with pytest.raises(ConfigError) as info:
        load_config(str(path), root=str(tmp_path))
    message = str(info.value)
    assert "model.selection_metric" in message and "metrics.primary" in message


def test_lag_depth_beyond_the_input_window_is_rejected(tmp_path):
    from tests.conftest import make_config

    path = make_config(tmp_path, lambda cfg: cfg["features"].update({"lags": 3}))
    with pytest.raises(ConfigError) as info:
        load_config(str(path), root=str(tmp_path))
    message = str(info.value)
    assert "features.lags" in message and "target week" in message


def test_report_dir_inside_output_is_rejected(tmp_path):
    from tests.conftest import make_config

    path = make_config(tmp_path, lambda cfg: cfg["report"].update(
        {"dir": "output/reports"}))
    with pytest.raises(ConfigError) as info:
        load_config(str(path), root=str(tmp_path))
    message = str(info.value)
    assert "report.dir" in message and "output.dir" in message


def test_price_discount_interaction_must_be_a_boolean(tmp_path):
    from tests.conftest import make_config

    path = make_config(tmp_path, lambda cfg: cfg["features"].update(
        {"price_discount_interaction": "true"}))
    with pytest.raises(ConfigError) as info:
        load_config(str(path), root=str(tmp_path))
    assert "features.price_discount_interaction" in str(info.value)
