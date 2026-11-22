"""Forecast-error slices for §7.7: where the method should not be trusted.

Computed from observed test sales, so these are diagnostics, not proxy measures.
Every group reports n, WAPE, MAE, RMSE and bias; groups below the minimum size are
kept but marked, and only reported groups are ranked.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..errors import StageError
from .metrics import bias, mae, rmse, wape

MIN_SLICE_SIZE = 10
TOP_N = 10
DEFINITION = ("Forecast-error slices computed from observed test sales; they are "
              "diagnostics of the forecast, not proxy stock-out measures.")


@dataclass(frozen=True)
class SliceResult:
    slice_name: str
    group: str
    basis: str
    n: int
    wape: float | None
    mae: float | None
    rmse: float | None
    bias: float | None
    status: str

    def as_dict(self) -> dict:
        return {"slice": self.slice_name, "group": self.group,
                "basis": self.basis, "n": int(self.n),
                "wape": self.wape, "mae": self.mae, "rmse": self.rmse,
                "bias": self.bias, "status": self.status}


def _coerce(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 2 or predicted.ndim != 2:
        raise StageError("evaluation", "error slices need a 2-D forecast matrix",
                         next_command="python -m codes.run_pipeline")
    if actual.shape != predicted.shape:
        raise StageError(
            "evaluation",
            f"error slices need aligned forecasts: actual {actual.shape} vs "
            f"predicted {predicted.shape}",
            next_command="python -m codes.run_pipeline")
    return actual, predicted


def _result(name, group, basis, a_flat, p_flat, min_size) -> SliceResult:
    n = int(a_flat.size)
    status = "reported" if n >= int(min_size) else "below_min_size"
    return SliceResult(name, group, basis, n, wape(a_flat, p_flat),
                       mae(a_flat, p_flat), rmse(a_flat, p_flat),
                       bias(a_flat, p_flat), status)


def _category_groups(features):
    columns = [c for c in features.columns if str(c).startswith("category_")]
    if not columns:
        return None
    active = features[columns].to_numpy(dtype=float) > 0.5
    groups = []
    for row in active:
        found = np.flatnonzero(row)
        groups.append(str(columns[found[0]]).replace("category_", "", 1)
                      if found.size else "unknown")
    return groups


def _price_groups(raw_test, bands=4):
    if "price" not in raw_test.columns:
        return None
    prices = pd.to_numeric(raw_test["price"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(prices)
    if finite.sum() < bands or np.unique(prices[finite]).size < 2:
        return None
    try:
        cut = pd.qcut(prices, bands, labels=[f"Q{i + 1}" for i in range(bands)],
                      duplicates="drop")
    except ValueError:
        return None
    return list(cut.astype(str))


def _volume_groups(raw_test):
    if not {"q1", "q2"} <= set(raw_test.columns):
        return None
    observed = (pd.to_numeric(raw_test["q1"], errors="coerce").fillna(0.0)
                + pd.to_numeric(raw_test["q2"], errors="coerce").fillna(0.0))
    values = observed.to_numpy(dtype=float)
    if values.size == 0:
        return None
    threshold = float(np.quantile(values, 0.25))
    return ["low_volume" if value <= threshold else "rest" for value in values]


def _discount_cells(raw_test, horizon):
    columns = [f"disc_w{3 + j}" for j in range(horizon)]
    if not all(column in raw_test.columns for column in columns):
        return None
    return raw_test[columns].to_numpy(dtype=float)


def _row_slices(name, groups, actual, predicted, min_size, results):
    for group in sorted(set(groups)):
        mask = np.array([value == group for value in groups])
        if not mask.any():
            continue
        results.append(_result(name, str(group), "rows", actual[mask].ravel(),
                               predicted[mask].ravel(), min_size))


def error_slices(actual, predicted, features, raw_test, meta_test, *,
                 min_size: int = MIN_SLICE_SIZE, top_n: int = TOP_N) -> dict:
    actual, predicted = _coerce(actual, predicted)
    if len(features) != actual.shape[0] or len(raw_test) != actual.shape[0]:
        raise StageError("evaluation", "error slices need one row per forecast row",
                         next_command="python -m codes.run_pipeline")
    horizon = actual.shape[1]
    results = []
    unavailable = []

    categories = _category_groups(features)
    if categories is None:
        unavailable.append({"slice": "category", "reason": "no category_ columns"})
    else:
        _row_slices("category", categories, actual, predicted, min_size, results)

    prices = _price_groups(raw_test)
    if prices is None:
        unavailable.append({"slice": "price_band", "reason": "price not usable"})
    else:
        _row_slices("price_band", prices, actual, predicted, min_size, results)

    volumes = _volume_groups(raw_test)
    if volumes is None:
        unavailable.append({"slice": "volume_band", "reason": "q1/q2 not present"})
    else:
        _row_slices("volume_band", volumes, actual, predicted, min_size, results)

    discount = _discount_cells(raw_test, horizon)
    if discount is None:
        unavailable.append({"slice": "discount_period",
                            "reason": "disc_w3..disc_w12 not present"})
    else:
        is_discounted = discount > 0
        for group, mask in (("any_discount", is_discounted),
                            ("no_discount", ~is_discounted)):
            if not mask.any():
                continue
            results.append(_result("discount_period", group, "cells",
                                   actual[mask], predicted[mask], min_size))

    reported = [row for row in results if row.status == "reported"]
    ranked = sorted((row for row in reported if row.wape is not None),
                    key=lambda row: row.wape, reverse=True)
    worst_groups = [{"slice": row.slice_name, "group": row.group,
                     "n": row.n, "wape": row.wape, "mae": row.mae,
                     "bias": row.bias}
                    for row in ranked[:5]]

    absolute = np.abs(actual - predicted)
    row_total = absolute.sum(axis=1)
    cumulative = np.abs(predicted.sum(axis=1) - actual.sum(axis=1))
    keys = meta_test.reset_index(drop=True)
    top_rows = []
    for index in np.argsort(-row_total, kind="stable")[:int(top_n)]:
        top_rows.append({"external_code": int(keys.loc[index, "external_code"]),
                         "retail": int(keys.loc[index, "retail"]),
                         "total_abs_error": float(row_total[index]),
                         "cumulative_error": float(cumulative[index]),
                         "category": categories[index] if categories else None,
                         "price_band": prices[index] if prices else None,
                         "volume_band": volumes[index] if volumes else None})
    top_cumulative = []
    for index in np.argsort(-cumulative, kind="stable")[:int(top_n)]:
        top_cumulative.append({
            "external_code": int(keys.loc[index, "external_code"]),
            "retail": int(keys.loc[index, "retail"]),
            "cumulative_error": float(cumulative[index]),
            "total_abs_error": float(row_total[index])})

    return {"available": True, "definition": DEFINITION,
            "test_partition": "test", "min_slice_size": int(min_size),
            "n_rows": int(actual.shape[0]), "horizon_weeks": int(horizon),
            "slices": [row.as_dict() for row in results],
            "reported_count": len(reported),
            "unavailable": unavailable,
            "worst_groups": worst_groups,
            "top_error_rows": top_rows,
            "top_cumulative_errors": top_cumulative}