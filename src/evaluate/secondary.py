"""Optional secondary comparison on the benchmark partition, converted to raw units."""
from __future__ import annotations

from dataclasses import dataclass

from ..io.unit_contract import convert_scaled_to_raw
from ..models.registry import get_model


@dataclass(frozen=True)
class SecondaryComparison:
    applied_scale_factor: float
    converted: bool
    metrics: dict
    n_train: int
    n_test: int
    label: str = "secondary"
    note: str = ("official partition, single declared factor, converted to raw units; "
                 "never reported as the main result")

    def as_dict(self) -> dict:
        return {"label": self.label,
                "applied_scale_factor": self.applied_scale_factor,
                "converted": self.converted,
                "metrics": self.metrics,
                "n_train": self.n_train,
                "n_test": self.n_test,
                "note": self.note}


def compare_on_preset(train_task, test_task, factor: float, metric_seam,
                      model_name: str, backend, seed: int,
                      reference="last_input") -> SecondaryComparison:
    """Fit on the preset train file, predict the preset test file, convert once."""
    factor = float(factor)
    model = get_model(model_name, backend=backend, seed=seed, reference=reference)
    model.fit(train_task.features, train_task.targets.to_numpy())
    scaled_prediction = model.predict(test_task.features)

    raw_prediction = convert_scaled_to_raw(scaled_prediction, factor)
    raw_actual = convert_scaled_to_raw(test_task.targets.to_numpy(), factor)

    record = metric_seam.evaluate(
        raw_actual, raw_prediction, test_task.metadata,
        f"{model_name}:secondary")
    metrics = {"wape_item_level": record.wape_item_level,
               "wape_pooled": record.wape_pooled,
               "mae": record.mae,
               "rmse": record.rmse}
    return SecondaryComparison(
        applied_scale_factor=factor,
        converted=True,
        metrics=metrics,
        n_train=int(len(train_task.features)),
        n_test=int(len(test_task.features)),
    )