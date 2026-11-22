"""One LightGBM model for all ten horizon weeks, with the horizon week as a feature.

Section 7.4.4 names this model. It is deliberately separate from the per-horizon
sklearn ensemble, and the two are not interchangeable: the repo feature matrix
still differs from Linda's script (no Trends terms, no explicit opening-stock
column), so any comparison between them must be stated as a deviation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..errors import StageError
from .base import Forecaster

TRAIN_CAP_QUANTILE = 0.999
HORIZON_FEATURE = "horizon_week"


class LightGBMHorizon(Forecaster):
    name = "lightgbm_horizon"
    requires_backend = False

    def __init__(self, params: dict | None = None, seed: int = 42,
                 objective: str = "poisson",
                 train_cap_quantile: float = TRAIN_CAP_QUANTILE):
        self.params = dict(params or {})
        self.seed = int(seed)
        self.objective = str(objective)
        self.train_cap_quantile = float(train_cap_quantile)
        self.model_ = None
        self.columns_ = None
        self.horizon_ = None
        self.cap_ = None
        self.training_rows_ = 0

    def _import(self):
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise StageError(
                "fit",
                "backend 'lightgbm' unavailable on this runtime - install it or "
                "drop the lightgbm_horizon candidate",
                next_command="pip install lightgbm",
            ) from exc
        return lgb

    def _stack(self, frame: pd.DataFrame) -> pd.DataFrame:
        blocks = []
        for horizon in range(1, self.horizon_ + 1):
            block = frame.reindex(columns=self.columns_, fill_value=0.0).copy()
            block[HORIZON_FEATURE] = float(horizon)
            blocks.append(block)
        return pd.concat(blocks, ignore_index=True) if blocks else frame.iloc[:0]

    def fit(self, X, y, **kwargs):
        lgb = self._import()
        frame = pd.DataFrame(X)
        self.columns_ = list(frame.columns)
        targets = np.asarray(y, dtype=float)
        if targets.ndim == 1:
            targets = targets[:, None]
        self.horizon_ = int(targets.shape[1])
        cleaned = np.clip(np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0),
                          0.0, None)
        self.cap_ = np.quantile(cleaned, self.train_cap_quantile, axis=0)
        capped = np.minimum(cleaned, self.cap_)
        stacked_x = self._stack(frame)
        stacked_y = capped.T.reshape(-1)

        params = {
            "objective": self.objective,
            "random_state": self.seed,
            "verbose": -1,
            "n_estimators": int(self.params.get(
                "n_estimators", self.params.get("max_iter", 300))),
            "learning_rate": float(self.params.get("learning_rate", 0.05)),
            "num_leaves": int(self.params.get(
                "num_leaves", self.params.get("max_leaf_nodes", 31))),
        }
        self.model_ = lgb.LGBMRegressor(**params)
        self.model_.fit(stacked_x.to_numpy(dtype=float), stacked_y)
        self.training_rows_ = int(stacked_x.shape[0])
        return self

    def predict(self, X) -> np.ndarray:
        frame = pd.DataFrame(X)
        if self.model_ is None:
            raise StageError("fit", "lightgbm_horizon cannot predict before fit",
                             next_command="python -m codes.run_pipeline")
        raw = np.asarray(self.model_.predict(
            self._stack(frame).to_numpy(dtype=float)), dtype=float)
        matrix = raw.reshape(self.horizon_, len(frame)).T
        return self._clip(matrix)