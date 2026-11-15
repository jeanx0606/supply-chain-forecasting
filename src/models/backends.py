"""Boosting backends behind one adapter seam (D17, REP-14)."""
from __future__ import annotations

import numpy as np

from ..errors import ConfigError, StageError
from ..seams.protocols import SeamDescription


class HorizonEnsemble:
    """One regressor per horizon; predictions clipped at zero by the caller."""

    def __init__(self, factory, horizon: int):
        self.factory = factory
        self.horizon = int(horizon)
        self.models = []

    def fit(self, X, y):
        values = np.asarray(X, dtype=float)
        targets = np.asarray(y, dtype=float)
        if targets.ndim == 1:
            targets = targets[:, None]
        if targets.shape[1] != self.horizon:
            raise StageError("fit", f"horizon mismatch: expected {self.horizon}, "
                                    f"observed {targets.shape[1]}")
        self.models = []
        for h in range(self.horizon):
            model = self.factory()
            model.fit(values, targets[:, h])
            self.models.append(model)
        return self

    def predict(self, X) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        columns = [np.asarray(model.predict(values), dtype=float) for model in self.models]
        return np.column_stack(columns)


class BackendBase:
    SEAM_ID = "SEAM-MODEL-BACKEND"


class SklearnHGBBackend(BackendBase):
    def __init__(self, params: dict | None = None, seed: int = 42):
        self.params = dict(params or {})
        self.seed = int(seed)

    def describe(self) -> SeamDescription:
        return SeamDescription("model_backend", "REP-14", "sklearn_hgb")

    def make_ensemble(self, horizon: int) -> HorizonEnsemble:
        from sklearn.ensemble import HistGradientBoostingRegressor

        params = {
            "max_iter": int(self.params.get("max_iter", 120)),
            "learning_rate": float(self.params.get("learning_rate", 0.08)),
            "max_leaf_nodes": int(self.params.get("max_leaf_nodes", 15)),
            "random_state": self.seed,
        }

        def factory():
            return HistGradientBoostingRegressor(**params)

        return HorizonEnsemble(factory, horizon)


class LightGBMBackend(BackendBase):
    def __init__(self, params: dict | None = None, seed: int = 42):
        self.params = dict(params or {})
        self.seed = int(seed)

    def describe(self) -> SeamDescription:
        return SeamDescription("model_backend", "REP-14", "lightgbm")

    def make_ensemble(self, horizon: int) -> HorizonEnsemble:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise StageError(
                "fit",
                "backend 'lightgbm' unavailable on this runtime - install it or set "
                "model.backend=sklearn_hgb",
                next_command="pip install lightgbm  # or set model.backend: sklearn_hgb",
            ) from exc

        params = {
            "n_estimators": int(self.params.get("max_iter", 120)),
            "learning_rate": float(self.params.get("learning_rate", 0.08)),
            "num_leaves": int(self.params.get("max_leaf_nodes", 15)),
            "random_state": self.seed,
            "verbose": -1,
        }

        def factory():
            return lgb.LGBMRegressor(**params)

        return HorizonEnsemble(factory, horizon)


def make_backend(name, params: dict | None = None, seed: int = 42):
    value = getattr(name, "value", name)
    if value == "sklearn_hgb":
        return SklearnHGBBackend(params, seed)
    if value == "lightgbm":
        return LightGBMBackend(params, seed)
    raise ConfigError("model.backend", "unregistered backend", observed=value,
                      expected=["sklearn_hgb", "lightgbm"])