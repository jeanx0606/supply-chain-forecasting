"""A1-A4 reconstruction: signed definitions, alpha-dependent A3, auditable raw path."""
from __future__ import annotations

import pandas as pd
import pytest

from src.errors import ConfigError
from src.preprocess.inventory import (
    ReconstructionPolicy,
    audit_reconstruction,
    build_trajectory,
    make_initial_stock_strategy,
)


def row(value=5.0, weeks=12):
    return pd.Series({f"q{n}": float(value) for n in range(1, weeks + 1)})


def policy(alpha=1.0, correction=True, floor=True):
    return ReconstructionPolicy(alpha=alpha, negative_correction=correction,
                                floor_at_zero=floor)


def frame(values):
    return pd.DataFrame({
        "external_code": list(range(1, len(values) + 1)),
        "retail": [1] * len(values),
        **{f"q{n}": [float(v) for v in values] for n in range(1, 13)},
    })


def test_a2_applies_alpha_to_restocked_to_release_units():
    strategy = make_initial_stock_strategy("restocked_to_release", alpha=1.5)
    estimate = strategy.estimate(row(), {}, opening=100.0)
    assert estimate.value == pytest.approx(150.0)
    assert estimate.base_value == pytest.approx(100.0)
    assert "alpha 1.5" in estimate.assumption_note


def test_a4_pairs_start_at_zero_with_an_explicit_note():
    strategy = make_initial_stock_strategy("restocked_to_release", alpha=1.0)
    estimate = strategy.estimate(row(), {}, opening=0.0)
    assert estimate.value == 0.0
    assert "A4" in estimate.assumption_note


def test_alpha_must_be_positive():
    with pytest.raises(ConfigError) as info:
        make_initial_stock_strategy("restocked_to_release", alpha=0.0)
    assert "inventory.opening_stock_alpha" in str(info.value)


def test_a3_adds_the_smallest_opening_stock_that_keeps_the_path_non_negative():
    strategy = make_initial_stock_strategy("restocked_to_release", alpha=1.0)
    trajectory = build_trajectory(row(5.0), {}, strategy, n_weeks=12,
                                  opening=10.0, policy=policy())
    assert trajectory.would_go_negative is True
    assert trajectory.min_raw_stock == pytest.approx(-50.0)
    assert trajectory.negative_weeks == 10
    assert trajectory.opening_stock_correction == pytest.approx(50.0)
    assert trajectory.initial_stock.value == pytest.approx(60.0)
    assert trajectory.initial_stock.correction == pytest.approx(50.0)
    assert all(entry.reconstructed_stock >= 0 for entry in trajectory.entries)
    assert min(entry.reconstructed_stock for entry in trajectory.entries) == 0.0


def test_a3_correction_is_recomputed_per_alpha_not_scaled():
    strategy_half = make_initial_stock_strategy("restocked_to_release", alpha=0.5)
    strategy_full = make_initial_stock_strategy("restocked_to_release", alpha=1.0)
    half = build_trajectory(row(5.0), {}, strategy_half, n_weeks=12,
                            opening=10.0, policy=policy(alpha=0.5))
    full = build_trajectory(row(5.0), {}, strategy_full, n_weeks=12,
                            opening=10.0, policy=policy(alpha=1.0))
    assert half.initial_stock.value == pytest.approx(60.0)
    assert full.initial_stock.value == pytest.approx(60.0)
    assert half.opening_stock_correction == pytest.approx(55.0)
    assert full.opening_stock_correction == pytest.approx(50.0)


def test_a1_floor_publishes_zero_but_keeps_the_raw_negative_evidence():
    strategy = make_initial_stock_strategy("restocked_to_release", alpha=1.0)
    trajectory = build_trajectory(row(50.0), {}, strategy, n_weeks=10,
                                  opening=0.0,
                                  policy=policy(correction=False, floor=True))
    assert trajectory.would_go_negative is True
    assert trajectory.min_raw_stock == pytest.approx(-500.0)
    assert all(entry.reconstructed_stock == 0.0 for entry in trajectory.entries)
    assert trajectory.opening_stock_correction == 0.0


def test_legacy_policy_preserves_the_unfloored_path():
    strategy = make_initial_stock_strategy("lead_time_demand", 1, 1.0)
    trajectory = build_trajectory(row(50.0), {}, strategy, n_weeks=10,
                                  policy=ReconstructionPolicy(
                                      negative_correction=False,
                                      floor_at_zero=False))
    assert trajectory.entries[-1].reconstructed_stock < 0
    assert trajectory.opening_stock_correction == 0.0


def test_reconstruction_identity_when_no_correction_or_floor_applies():
    strategy = make_initial_stock_strategy("restocked_to_release", alpha=1.0)
    trajectory = build_trajectory(row(2.0), {3: 4.0}, strategy, n_weeks=6,
                                  opening=20.0,
                                  policy=policy(correction=False, floor=True))
    previous = 20.0
    for entry in trajectory.entries:
        previous = previous + entry.restock - entry.sales
        assert entry.reconstructed_stock == pytest.approx(previous)


def test_audit_counts_stockout_and_negative_pairs_before_a3():
    data = frame([5.0, 10.0, 100.0])
    strategy = make_initial_stock_strategy("restocked_to_release", alpha=1.0)
    opening = {(1, 1): 0.0, (2, 1): 10.0, (3, 1): 100.0}
    audit = audit_reconstruction(data, {}, opening, strategy, n_weeks=12,
                                 audit_weeks=11)
    assert audit.pairs == 3
    assert audit.opening_zero_pairs == 1
    assert audit.negative_pairs == 3
    assert audit.negative_pairs_excluding_a4 == 2
    assert audit.zero_or_below_pairs == 3
    assert audit.correction_total > 0