"""Feature and target matrices built from the declared closed specification.

Every column comes from an input week; no mask and no partition statistic is
accepted here (C9, 5.8.1 r14c).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..errors import StageError
from ..io.unit_contract import assert_raw_unit
from .spec import FeaturePlan, FeatureSetRecord, FeatureSpec


@dataclass(frozen=True)
class TaskDataset:
    features: pd.DataFrame
    targets: pd.DataFrame
    metadata: pd.DataFrame
    origin_week: int
    horizon: int
    input_weeks: int
    labels: tuple
    feature_set: FeatureSetRecord | None = None


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _attribute_frame(frame: pd.DataFrame, plan: FeaturePlan) -> pd.DataFrame:
    names = [c for family in plan.families if family.name == "attributes"
             for c in family.columns]
    if not names:
        return pd.DataFrame(index=frame.index)
    out = pd.DataFrame(index=frame.index)
    for column in ("season", "category", "color", "fabric"):
        if column in frame.columns:
            values = frame[column].astype("object").where(
                frame[column].notna(), "unknown").astype(str)
            dummies = pd.get_dummies(values, prefix=column, dtype=float)
            for name in dummies.columns:
                if name in names:
                    out[name] = dummies[name].to_numpy()
    return out


def build_features(frame: pd.DataFrame, input_weeks: int, origin_week: int,
                   prefix: str = "q", spec: FeatureSpec | None = None,
                   plan: FeaturePlan | None = None) -> tuple[pd.DataFrame, FeaturePlan]:
    spec = spec or FeatureSpec()
    plan = plan or spec.plan(frame, input_weeks, origin_week, prefix)

    feats = pd.DataFrame(index=frame.index)

    depth = plan.lag_depth
    for lag in range(1, depth + 1):
        column = f"{prefix}{origin_week - depth + lag}"
        feats[f"lag_{lag}"] = _numeric(frame, column).to_numpy()

    if any(f.name == "discount" and f.status == "present" for f in plan.families):
        for lag in range(1, depth + 1):
            column = f"disc_w{origin_week - depth + lag}"
            feats[f"discount_lag_{lag}"] = _numeric(frame, column).to_numpy()

    if any(f.name == "price" for f in plan.families):
        feats["price_rel"] = pd.to_numeric(
            frame["price"], errors="coerce").fillna(0.0).to_numpy()

    if plan.interaction and "discount_lag_1" in feats.columns:
        for lag in range(1, depth + 1):
            feats[f"price_x_discount_{lag}"] = (
                feats["price_rel"].to_numpy() * feats[f"discount_lag_{lag}"].to_numpy())

    if (any(f.name == "aggregates" and f.status == "present" for f in plan.families)
            and "external_code" in frame.columns and "retail" in frame.columns):
        input_columns = [f"{prefix}{n}" for n in
                         range(origin_week - depth + 1, origin_week + 1)]
        row_mean = frame[input_columns].apply(
            pd.to_numeric, errors="coerce").fillna(0.0).mean(axis=1)
        work = pd.DataFrame({"external_code": frame["external_code"].to_numpy(),
                             "retail": frame["retail"].to_numpy(),
                             "_row_mean": row_mean.to_numpy()}, index=frame.index)
        declared = {family.name: family.columns for family in plan.families}
        reduced = set(declared.get("aggregates", ()))
        if "item_mean_input" in reduced:
            feats["item_mean_input"] = (
                work.groupby("external_code", observed=True)["_row_mean"]
                .transform("mean").to_numpy())
        if "store_mean_input" in reduced:
            feats["store_mean_input"] = (
                work.groupby("retail", observed=True)["_row_mean"]
                .transform("mean").to_numpy())

    if "store_size" in [c for f in plan.families if f.name == "store_scale"
                        for c in f.columns]:
        sizes = frame.groupby("retail")["external_code"].transform("nunique").astype(float)
        feats["store_size"] = sizes.to_numpy()

    attributes = _attribute_frame(frame, plan)
    if not attributes.empty:
        feats = feats.join(attributes)

    if plan.weather_source == "as_supplied":
        for column in [c for f in plan.families if f.name == "weather" for c in f.columns]:
            feats[column] = pd.to_numeric(
                frame[column], errors="coerce").fillna(0.0).to_numpy()

    ordered = [c for c in plan.columns if c in feats.columns]
    feats = feats.reindex(columns=ordered)
    spec.assert_closed(feats, plan)
    return feats.fillna(0.0).astype(float), plan


def build_targets(frame: pd.DataFrame, output_weeks: int, origin_week: int,
                  n_weeks: int = 12, prefix: str = "q") -> pd.DataFrame:
    out_columns = [f"{prefix}{n}" for n in range(origin_week + 1,
                                                min(origin_week + output_weeks, n_weeks) + 1)]
    if len(out_columns) < output_weeks:
        raise StageError(
            "features",
            f"insufficient target weeks: needed {output_weeks}, found {len(out_columns)}",
            next_command="python -m codes.run_pipeline")
    targets = frame[out_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    targets.columns = [f"t{i + 1}" for i in range(len(out_columns))]
    return targets


def build_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in ("external_code", "retail", "release_date") if c in frame.columns]
    return frame[columns].reset_index(drop=True)


def build_task(frame: pd.DataFrame, input_weeks: int, output_weeks: int,
               n_weeks: int = 12, origin_week: int | None = None,
               prefix: str = "q", spec: FeatureSpec | None = None) -> TaskDataset:
    assert_raw_unit(frame, "features")
    origin = int(origin_week if origin_week is not None else input_weeks)
    spec = spec or FeatureSpec()
    features, plan = build_features(frame, input_weeks, origin, prefix, spec)
    targets = build_targets(frame, output_weeks, origin, n_weeks, prefix)
    return TaskDataset(
        features=features.reset_index(drop=True),
        targets=targets.reset_index(drop=True),
        metadata=build_metadata(frame),
        origin_week=origin,
        horizon=targets.shape[1],
        input_weeks=input_weeks,
        labels=tuple(targets.columns),
        feature_set=spec.record(features, plan),
    )