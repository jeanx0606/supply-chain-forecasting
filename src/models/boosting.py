"""Boosting forecaster built from a backend ensemble (D17)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster


class GradientBoostingForecaster(Forecaster):
    name = "gradient_boosting"

    def __init__(self, backend, seed: int = 42):
        self.backend = backend
        self.seed = int(seed)
        self.ensemble_ = None
        self.columns_ = None

    def fit(self, X, y, **kwargs):
        frame = pd.DataFrame(X)
        self.columns_ = list(frame.columns)
        targets = np.asarray(y, dtype=float)
        horizon = targets.shape[1] if targets.ndim > 1 else 1
        self.ensemble_ = self.backend.make_ensemble(horizon)
        self.ensemble_.fit(frame.to_numpy(dtype=float), targets)
        return self

    def predict(self, X) -> np.ndarray:
        frame = pd.DataFrame(X).reindex(columns=self.columns_, fill_value=0.0)
        return self._clip(self.ensemble_.predict(frame.to_numpy(dtype=float)))