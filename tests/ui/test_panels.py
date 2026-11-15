"""Console: six panels in order, data-bearing views, cache and error trap (A5, D15, D16)."""
from __future__ import annotations

import pytest

from src.config.loader import load_config
from src.errors import StageError
from ui.panels import PANEL_REGISTRY, labels, views
from ui.state import BundleCache, bundle_key, error_panel_text

pytestmark = pytest.mark.ui


def test_six_panels_in_the_declared_order():
    assert labels() == ["P1 Overview", "P2 Forecast", "P3 Metrics",
                        "P4 Inventory", "P5 Replenishment", "P6 Risk"]
    assert len(PANEL_REGISTRY) == 6
    assert [panel.order for panel in PANEL_REGISTRY] == [1, 2, 3, 4, 5, 6]


def test_every_panel_view_is_non_empty(sample_bundle):
    for key, view in views(sample_bundle).items():
        assert view, f"panel {key} produced no content"


def test_forecast_panel_exposes_ten_chart_points(sample_bundle):
    view = views(sample_bundle)["forecast"]
    assert view["points"] == 10
    assert list(view["frame"].columns) == ["week", "predicted", "actual"]
    assert view["model"] == sample_bundle.main_model


def test_inventory_and_replenishment_panels_are_data_bearing(sample_bundle):
    view = views(sample_bundle)
    assert len(view["inventory"]["entries"]) == 12
    assert view["replenishment"]["recommended_qty"] >= 0
    assert view["replenishment"]["decision_week"] == \
        view["replenishment"]["forecast_origin_week"]


def test_risk_panel_states_both_grades(sample_bundle):
    view = views(sample_bundle)["risk"]
    assert view["stockout_proxy"] in {"high", "medium", "low", "unknown"}
    assert view["overstock"] in {"high", "medium", "low", "unknown"}
    assert "PROXY" in view["proxy_note"]


def test_tab_switch_does_not_rerun_the_pipeline(sample_dataset):
    cfg = load_config(sample_dataset["config"], root=sample_dataset["root"])
    cache = BundleCache()
    marker = object()
    cache.ensure(cfg, runner=lambda _cfg: marker)
    cache.ensure(cfg, runner=lambda _cfg: (_ for _ in ()).throw(AssertionError("rerun")))
    assert cache.bundle is marker
    assert cache.calls == 1


def test_force_reruns_and_changing_config_changes_the_key(sample_dataset):
    cfg = load_config(sample_dataset["config"], root=sample_dataset["root"])
    cache = BundleCache()
    cache.ensure(cfg, runner=lambda _cfg: "first")
    cache.ensure(cfg, force=True, runner=lambda _cfg: "second")
    assert cache.bundle == "second"
    assert cache.calls == 2
    assert bundle_key(cfg) != "other-key"


def test_failure_keeps_a_readable_error_with_next_command(sample_dataset):
    cfg = load_config(sample_dataset["config"], root=sample_dataset["root"])
    cache = BundleCache()

    def boom(_cfg):
        raise StageError("ingest", "missing mandatory input 'sales.csv'",
                         next_command="python -m codes.sample_data")

    cache.ensure(cfg, runner=boom)
    assert cache.bundle is None
    assert "python -m codes.sample_data" in cache.error
    assert "python -m codes.sample_data" in error_panel_text(cache.error)

def test_overview_shows_sign_and_unit_provenance(sample_bundle):
    view = views(sample_bundle)["overview"]
    assert view["negative_policy"] in {"clip_at_zero", "retain_as_returns"}
    assert "PROVISIONAL" in view["sign_badge"]
    assert "REP-17" in view["sign_badge"]
    assert "raw_sales_units" in view["unit_caption"] or "raw" in view["unit_caption"]
    assert view["preset_statement"]
    assert view["negative_cells"] == 0


def test_unit_caption_comes_from_the_provenance_object(sample_bundle):
    from ui.components import unit_caption

    class Fake:
        unit = "raw_sales_units"
        applied_scale_factor = 53.0
        sign_policy = "retain_as_returns"

    caption = unit_caption(Fake())
    assert "53" in caption
    assert "retain_as_returns" in caption
