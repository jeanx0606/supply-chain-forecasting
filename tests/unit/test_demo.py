"""Demonstration selector: deterministic, key-ordered, never iloc[0] (D5, REP-11)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.errors import StageError
from src.replenish.demo import (
    DemoCandidate,
    MaxHorizonDemandSelector,
    make_demo_selector,
)

ROOT = Path(__file__).resolve().parents[2]


def candidates():
    return [
        DemoCandidate((5, 2), 100.0, 0),
        DemoCandidate((1, 3), 200.0, 1),
        DemoCandidate((1, 1), 200.0, 2),
        DemoCandidate((9, 9), 50.0, 3),
    ]


def test_max_horizon_demand_selected():
    chosen = MaxHorizonDemandSelector().select(candidates())
    assert chosen.horizon_demand == 200.0


def test_ties_broken_by_key_ascending():
    chosen = MaxHorizonDemandSelector().select(candidates())
    assert chosen.key == (1, 1)


def test_empty_candidates_abort():
    with pytest.raises(StageError):
        MaxHorizonDemandSelector().select([])


def test_pinned_selector_respects_configuration():
    selector = make_demo_selector("pinned", 5, 2)
    assert selector.select(candidates()).key == (5, 2)


def test_pipeline_never_uses_row_order():
    text = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
    assert "iloc[0]" not in text