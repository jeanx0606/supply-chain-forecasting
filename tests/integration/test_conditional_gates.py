"""Conditional gates: B27/B29 are never green while real files are absent (T18.6/T18.7)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.sample import is_sample_dataset

pytestmark = pytest.mark.integration

RAW = Path("data/raw")
REAL_DATA_ABSENT = is_sample_dataset(RAW) or not (RAW / "sales.csv").exists()


def test_conditional_member_states_the_precondition(sample_bundle):
    conditional = sample_bundle.conditional
    assert conditional["data_class"] == "sample"
    assert conditional["b27_real_data"] == "not evaluated - precondition absent"
    assert conditional["b29_secondary"] == "not evaluated - precondition absent"
    for value in conditional.values():
        assert value != "green"
        assert value != "passing"


@pytest.mark.skipif(REAL_DATA_ABSENT, reason="real data absent")
def test_b27_is_evaluated_only_on_real_data(sample_bundle):
    assert sample_bundle.conditional["b27_real_data"] != "not evaluated - precondition absent"


@pytest.mark.skipif(not (RAW / "stfore_train.csv").exists(), reason="real data absent")
def test_b29_is_evaluated_only_when_preset_files_exist(sample_bundle):
    assert sample_bundle.conditional["b29_secondary"] != "not evaluated - precondition absent"