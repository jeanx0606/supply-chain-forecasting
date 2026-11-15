"""Forecaster contract shared by both tracks."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Forecaster(ABC):
    name = "base"

    @abstractmethod
    def fit(self, X, y, **kwargs): ...

    @abstractmethod
    def predict(self, X) -> np.ndarray: ...

    @staticmethod
    def _clip(values) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(array, 0.0, None)