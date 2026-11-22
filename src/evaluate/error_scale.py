"""Forecast-error scale for safety stock, measured on the validation season only.

Section 7.4.4 asks the forecasting interface for a weekly forecast-error standard
deviation, and 7.6.3 consumes it as sigma_e. The value must come from the
validation partition so the test season stays sealed (selection.py seals it).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import StageError

PARTITION = "validation"
UNIT = "raw_sales_units_per_week"


def _paired(y_true, y_pred) -> tuple:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.ndim == 1:
        actual = actual[:, None]
    if predicted.ndim == 1:
        predicted = predicted[:, None]
    if actual.shape != predicted.shape:
        raise StageError(
            "evaluation",
            f"sigma_e needs aligned predictions: actual {actual.shape} "
            f"vs predicted {predicted.shape}",
            next_command="python -m codes.run_pipeline")
    return actual, predicted


def sigma_e(y_true, y_pred) -> float:
    """Standard deviation of weekly errors (actual - forecast), ddof=1."""
    actual, predicted = _paired(y_true, y_pred)
    errors = (actual - predicted).ravel()
    if errors.size < 2:
        return 0.0
    return float(np.std(errors, ddof=1))


@dataclass(frozen=True)
class ErrorScale:
    sigma_e: float
    n_observations: int
    horizon_weeks: int
    partition: str = PARTITION
    definition: str = ("standard deviation of weekly forecast errors "
                       "(actual - forecast), ddof=1, measured on validation")
    unit: str = UNIT

    def as_dict(self) -> dict:
        return {
            "sigma_e": float(self.sigma_e),
            "n_observations": int(self.n_observations),
            "horizon_weeks": int(self.horizon_weeks),
            "partition": self.partition,
            "definition": self.definition,
            "unit": self.unit,
        }


def measure(y_true, y_pred) -> ErrorScale:
    actual, predicted = _paired(y_true, y_pred)
    return ErrorScale(sigma_e=sigma_e(actual, predicted),
                      n_observations=int(actual.size),
                      horizon_weeks=int(actual.shape[1]))