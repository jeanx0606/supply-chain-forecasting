"""The one sanctioned scale conversion. The constant is declared, never inferred."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import ConfigError, StageError

RAW_UNIT = "raw_sales_units"
SCALED_UNIT = "scaled_sales_units"
UNIT_ATTR = "unit"
FACTOR_KEY = "split.preset_scale_factor"


@dataclass(frozen=True)
class UnitDeclaration:
    unit: str = RAW_UNIT
    applied_scale_factor: float | None = None
    scale_factor_source: str = "none"

    def as_dict(self) -> dict:
        return {"unit": self.unit,
                "applied_scale_factor": self.applied_scale_factor,
                "scale_factor_source": self.scale_factor_source}


def explicit_factor(factor, key: str = FACTOR_KEY) -> float:
    """A factor must be explicitly configured; it is never derived from the data."""
    if factor is None:
        raise ConfigError(
            key, "a scaled read requires an explicitly configured factor",
            observed=None, expected="a positive number declared under REP-05")
    if isinstance(factor, bool) or not isinstance(factor, (int, float, np.integer, np.floating)):
        raise ConfigError(key, "factor must be a positive finite number",
                          observed=factor, expected="a finite number > 0")
    value = float(factor)
    if value <= 0 or not np.isfinite(value):
        raise ConfigError(key, "factor must be a positive finite number",
                          observed=factor, expected="a finite number > 0")
    return value


def convert_scaled_to_raw(values, factor):
    """Multiply by the declared factor exactly once. No other function may do this."""
    declared = explicit_factor(factor)
    return np.asarray(values, dtype=float) * declared


def stamp(frame, unit: str = RAW_UNIT):
    frame.attrs[UNIT_ATTR] = unit
    return frame


def unit_of(frame) -> str:
    return str(frame.attrs.get(UNIT_ATTR, RAW_UNIT))


def assert_raw_unit(frame, stage: str = "features") -> None:
    unit = unit_of(frame)
    if unit != RAW_UNIT:
        raise StageError(
            stage,
            f"C10: received a frame in '{unit}'; every feature and target must be in "
            f"{RAW_UNIT} - convert with convert_scaled_to_raw before building the task")


def declaration(scale_factor: float | None = None, source: str = "none") -> UnitDeclaration:
    if scale_factor is None:
        return UnitDeclaration(RAW_UNIT, None, source)
    return UnitDeclaration(RAW_UNIT, float(scale_factor), source)