"""C10 unit contract: raw units, one conversion, and the forbidden constant."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.errors import ConfigError, StageError
from src.io.unit_contract import (
    RAW_UNIT,
    SCALED_UNIT,
    assert_raw_unit,
    convert_scaled_to_raw,
    declaration,
    explicit_factor,
    stamp,
    unit_of,
)


def test_stamp_and_unit_of():
    frame = pd.DataFrame({"q1": [1.0]})
    assert unit_of(frame) == RAW_UNIT
    assert unit_of(stamp(frame)) == RAW_UNIT
    stamp(frame, SCALED_UNIT)
    assert unit_of(frame) == SCALED_UNIT


def test_conversion_requires_a_declared_factor():
    with pytest.raises(ConfigError) as info:
        convert_scaled_to_raw([1.0], None)
    message = str(info.value)
    assert "split.preset_scale_factor" in message
    assert "REP-05" in message


def test_conversion_multiplies_exactly_once():
    scaled = np.array([1.0, 2.0, 0.47])
    raw = convert_scaled_to_raw(scaled, 53)
    assert np.allclose(raw, scaled * 53)
    assert np.allclose(raw / 53, scaled)


def test_invalid_factors_rejected():
    for value in (0, -1, float("nan"), float("inf"), True):
        with pytest.raises(ConfigError):
            explicit_factor(value)


def test_scaled_frame_refused_at_the_boundary():
    frame = pd.DataFrame({"q1": [1.0]})
    stamp(frame, SCALED_UNIT)
    with pytest.raises(StageError) as info:
        assert_raw_unit(frame, "features")
    assert "C10" in str(info.value)


def test_raw_values_are_ingested_unchanged(tmp_path):
    from src.io.loader import load_sales

    probe = 12.0
    row = {"external_code": 1, "retail": 2, "release_date": "2019-01-07"}
    for week in range(12):
        row[str(week)] = probe + week
    path = tmp_path / "sales.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    frame, _ = load_sales(path, 12, "q")
    assert frame["q1"].iloc[0] == probe
    assert frame["q12"].iloc[0] == probe + 11
    assert unit_of(frame) == RAW_UNIT
    assert frame["q1"].min() == frame["q1"].max() == probe


def test_declaration_reports_no_factor_on_the_main_path():
    assert declaration().as_dict() == {
        "unit": RAW_UNIT, "applied_scale_factor": None,
        "scale_factor_source": "none"}


def test_no_forbidden_constant_in_business_logic():
    from codes.check_units import check

    assert check() == []


def test_scanner_catches_a_literal_with_file_line_and_function(tmp_path):
    from codes.check_units import check

    bad = tmp_path / "bad.py"
    bad.write_text("def scale(values):\n    return values * 53\n", encoding="utf-8")
    problems = check(single=bad)
    assert problems
    text = problems[0]
    assert "bad.py:2" in text.replace("\\", "/")
    assert "scale" in text