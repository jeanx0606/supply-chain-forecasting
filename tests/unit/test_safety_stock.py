"""Safety stock: both resolutions consume the service level (D4, REP-10)."""
from __future__ import annotations

import pytest

from src.config.loader import load_config
from src.errors import ConfigError
from src.replenish.safety import DemandContext, make_safety_stock
from src.seams.factory import build_seams
from tests.conftest import make_config


def context(service_level=0.95):
    return DemandContext(demand_slice=(5.0,) * 10, lead_time_weeks=1,
                         order_cycle_weeks=1, service_level=service_level,
                         variability=3.0)


def test_higher_service_level_never_lowers_the_buffer():
    low = make_safety_stock("normal_approx", 0.95).compute(context())
    high = make_safety_stock("normal_approx", 0.99).compute(context())
    assert high.quantity >= low.quantity > 0


def test_buffer_uses_the_protection_period_lead_plus_cycle():
    from statistics import NormalDist

    ctx = DemandContext(demand_slice=(5.0,) * 10, lead_time_weeks=2,
                        order_cycle_weeks=1, service_level=0.95,
                        variability=3.0)
    result = make_safety_stock("normal_approx", 0.95).compute(ctx)
    expected = NormalDist().inv_cdf(0.95) * 3.0 * (3.0 ** 0.5)
    assert result.quantity == pytest.approx(expected)
    assert result.protection_weeks == 3
    assert "P=3" in result.scope_note


def test_zero_method_declares_service_level_out_of_scope():
    result = make_safety_stock("none", 0.95).compute(context())
    assert result.quantity == 0.0
    assert result.service_level_applied is False
    assert "OUT OF SCOPE" in result.scope_note


def test_both_methods_hold_the_service_level():
    for name in ("normal_approx", "none"):
        assert make_safety_stock(name, 0.95).service_level == 0.95


def test_switching_method_is_one_yaml_line(tmp_path):
    normal = make_config(tmp_path / "a", lambda cfg: cfg["replenish"].update(
        {"safety_stock_method": "normal_approx"}))
    none = make_config(tmp_path / "b", lambda cfg: cfg["replenish"].update(
        {"safety_stock_method": "none"}))
    first = build_seams(load_config(str(normal), root=str(tmp_path / "a"))).safety_stock
    second = build_seams(load_config(str(none), root=str(tmp_path / "b"))).safety_stock
    assert first.method == "normal_approx"
    assert second.method == "none"
    assert first.describe().rep_id == second.describe().rep_id == "REP-10"


def test_invalid_service_level_rejected():
    with pytest.raises(ConfigError):
        make_safety_stock("normal_approx", 0.0)
    with pytest.raises(ConfigError):
        make_safety_stock("mystery", 0.95)