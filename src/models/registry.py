"""Model registry and the model-selection seam (D19, REP-02)."""
from __future__ import annotations

from ..errors import ConfigError, RegistryError
from ..seams.protocols import SeamDescription
from .baseline import BASELINE_CLASSES, BaselineDefinition
from .boosting import GradientBoostingForecaster

REGISTRY = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


register(GradientBoostingForecaster)


def available_models() -> tuple:
    return tuple(sorted(set(REGISTRY) | set(BASELINE_CLASSES)))


def get_model(name: str, backend=None, seed: int = 42, reference="last_input"):
    if name in BASELINE_CLASSES:
        model = BaselineDefinition(reference).make(name)
        if seed is not None:
            setattr(model, "seed", int(seed))
        return model
    if name in REGISTRY:
        if backend is None:
            raise ConfigError("model.backend", "backend is required for boosting models",
                              observed=None, expected="a ModelBackend")
        return REGISTRY[name](backend=backend, seed=seed)
    raise RegistryError(name, available_models())


class ModelSelection:
    SEAM_ID = "SEAM-MODEL-SELECT"

    def __init__(self, selected: str, baseline: str):
        self.selected = selected
        self.baseline = baseline

    def describe(self) -> SeamDescription:
        return SeamDescription("model_select", "REP-02", self.selected)

    def available(self) -> tuple:
        return available_models()

    def main(self) -> str:
        if self.selected not in available_models():
            raise RegistryError(self.selected, available_models())
        return self.selected

    def require_main(self, produced: dict) -> str:
        if self.selected not in produced:
            raise ConfigError(
                "model.selected",
                "configured main model produced no predictions",
                observed=sorted(produced), expected=self.selected)
        return self.selected