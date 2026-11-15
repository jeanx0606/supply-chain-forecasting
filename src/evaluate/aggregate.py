"""Evaluation with item-level aggregation as the primary figure (REP-03, C2)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from ..seams.protocols import SeamDescription
from .metrics import mae, rmse, wape

DEFAULT_SUPPORTING = ("wape_pooled", "mae", "rmse",
                      "wape_store_weighted", "wape_small_stores")


@dataclass(frozen=True)
class MetricRecord:
    model: str
    wape_item_level: float | None
    wape_pooled: float | None
    mae: float | None
    rmse: float | None
    wape_store_weighted: float | None
    wape_small_stores: float | None
    n_small_stores: int | None
    n_evaluated_records: int
    n_evaluated_series: int
    cut_train_end: str
    cut_validation_end: str
    warnings: tuple = ()
    provisional: tuple = field(default_factory=tuple)
    unit: str = "raw_sales_units"
    applied_scale_factor: float | None = None
    sign_policy: str = "clip_at_zero"
    partition: str = "test"

    def as_dict(self) -> dict:
        return asdict(self)

    def get(self, key, default=None):
        return getattr(self, key, default)


class StandardMetricSet:
    SEAM_ID = "SEAM-METRIC-SET"

    def __init__(self, primary="wape_item_level", supporting=DEFAULT_SUPPORTING):
        self.primary = getattr(primary, "value", primary)
        self.supporting = tuple(getattr(m, "value", m) for m in supporting)

    def describe(self) -> SeamDescription:
        return SeamDescription("metric_set", "REP-03", self.primary)

    def evaluate(self, y_true, y_pred, meta: pd.DataFrame | None,
                 model_name: str, cut_train_end: str = "", cut_validation_end: str = "",
                 provisional=(), unit: str = "raw_sales_units",
                 applied_scale_factor: float | None = None,
                 sign_policy: str = "clip_at_zero",
                 partition: str = "test") -> MetricRecord:
        actual = np.asarray(y_true, dtype=float)
        predicted = np.asarray(y_pred, dtype=float)
        if actual.ndim == 1:
            actual = actual[:, None]
        if predicted.ndim == 1:
            predicted = predicted[:, None]

        record = {
            "model": model_name,
            "wape_item_level": None,
            "wape_pooled": wape(actual, predicted),
            "mae": mae(actual, predicted),
            "rmse": rmse(actual, predicted),
            "wape_store_weighted": None,
            "wape_small_stores": None,
            "n_small_stores": None,
            "n_evaluated_records": int(actual.shape[0]),
            "n_evaluated_series": int(actual.shape[0]),
            "cut_train_end": cut_train_end,
            "cut_validation_end": cut_validation_end,
            "warnings": (),
            "provisional": tuple(provisional),
            "unit": unit,
            "applied_scale_factor": applied_scale_factor,
            "sign_policy": sign_policy,
            "partition": partition,
        }

        if meta is None:
            return MetricRecord(**record)
        frame = meta.reset_index(drop=True)
        if len(frame) != actual.shape[0]:
            record["warnings"] = ("metadata misaligned: aggregates other than pooled omitted",)
            return MetricRecord(**record)

        frame = frame.assign(_row=range(len(frame)))
        if "external_code" in frame.columns:
            groups = frame.groupby("external_code", observed=True)["_row"].apply(list)
            rows = [idx for idx in groups if idx]
            if rows:
                item_actual = np.array([actual[idx].sum(axis=0) for idx in rows])
                item_pred = np.array([predicted[idx].sum(axis=0) for idx in rows])
                record["wape_item_level"] = wape(item_actual, item_pred)
                record["n_evaluated_series"] = int(len(rows))

        if "retail" in frame.columns:
            scores, weights = [], []
            for _, group in frame.groupby("retail", observed=True):
                idx = group["_row"].tolist()
                if not idx:
                    continue
                score = wape(actual[idx], predicted[idx])
                if score is None:
                    continue
                scores.append(score)
                weights.append(float(np.abs(actual[idx]).sum()))
            if weights and sum(weights) > 0:
                record["wape_store_weighted"] = float(
                    np.average(scores, weights=weights))

            sizes = frame.groupby("retail")["external_code"].nunique()
            if len(sizes) >= 4:
                threshold = sizes.quantile(0.25)
                small = sizes[sizes <= threshold].index
                mask = frame["retail"].isin(small).to_numpy()
                if mask.any():
                    record["wape_small_stores"] = wape(actual[mask], predicted[mask])
                    record["n_small_stores"] = int(len(small))
        return MetricRecord(**record)


def evaluate_split(seam, y_true, y_pred, meta, model_name, cut_train_end,
                   cut_validation_end, provisional=(), unit="raw_sales_units",
                   applied_scale_factor=None, sign_policy="clip_at_zero",
                   partition="test") -> MetricRecord:
    return seam.evaluate(y_true, y_pred, meta, model_name, cut_train_end,
                         cut_validation_end, provisional=provisional, unit=unit,
                         applied_scale_factor=applied_scale_factor,
                         sign_policy=sign_policy, partition=partition)