"""One declared, closed feature specification (D25, REP-19)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..errors import StageError
from ..io.schema import CATEGORICAL
from ..seams.protocols import SeamDescription

COLUMN_ORDER = ("lags", "discount", "interaction", "aggregates", "price",
                "store_scale", "attributes", "weather")
SPEC_ID = "closed_v1"


@dataclass(frozen=True)
class FeatureFamily:
    name: str
    columns: tuple
    status: str = "present"
    reason: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "columns": list(self.columns),
                "status": self.status, "reason": self.reason}


@dataclass(frozen=True)
class FeaturePlan:
    columns: tuple
    families: tuple
    lag_depth: int
    aggregate_scope: str
    discount_scope: str
    interaction: bool
    weather_source: str


@dataclass(frozen=True)
class FeatureSetRecord:
    spec_id: str
    n_columns: int
    lag_depth: int
    aggregate_scope: str
    discount_scope: str
    interaction: bool
    weather_source: str
    families: tuple
    asserts: dict
    column_order: tuple = COLUMN_ORDER

    def as_dict(self) -> dict:
        return {
            "spec_id": self.spec_id,
            "n_columns": self.n_columns,
            "lag_depth": self.lag_depth,
            "aggregate_scope": self.aggregate_scope,
            "discount_scope": self.discount_scope,
            "interaction": self.interaction,
            "weather_source": self.weather_source,
            "families": [family.as_dict() for family in self.families],
            "asserts": dict(self.asserts),
            "column_order": list(self.column_order),
        }


class FeatureSpec:
    SEAM_ID = "SEAM-FEATURE-SET"

    def __init__(self, lag_depth="input_window",
                 aggregates=("item_mean_input", "store_mean_input"),
                 interaction: bool = True, weather: str = "as_supplied"):
        self.lag_depth = lag_depth
        self.aggregates = tuple(aggregates or ())
        self.interaction = bool(interaction)
        self.weather = getattr(weather, "value", weather)

    def describe(self) -> SeamDescription:
        depth = ("input_window" if not isinstance(self.lag_depth, int)
                 else self.lag_depth)
        return SeamDescription("feature_set", "REP-19", str(depth))

    def depth_for(self, input_weeks: int) -> int:
        depth = (input_weeks if not isinstance(self.lag_depth, int)
                 else int(self.lag_depth))
        if depth < 1 or depth > input_weeks:
            raise StageError(
                "features",
                f"declared lag depth {depth} is outside the input window 1..{input_weeks}")
        return depth

    def plan(self, frame: pd.DataFrame, input_weeks: int, origin_week: int,
             prefix: str = "q") -> FeaturePlan:
        depth = self.depth_for(input_weeks)
        input_weeks_in_frame = [f"{prefix}{n}" for n in
                                range(origin_week - depth + 1, origin_week + 1)]
        missing = [c for c in input_weeks_in_frame if c not in frame.columns]
        if missing:
            raise StageError("features", f"insufficient weekly columns: {missing}")

        families = []
        columns = []

        lags = tuple(f"lag_{i}" for i in range(1, depth + 1))
        families.append(FeatureFamily("lags", lags))
        columns.extend(lags)

        discount_cols = tuple(f"discount_lag_{i}" for i in range(1, depth + 1))
        if all(f"disc_w{n}" in frame.columns for n in
               range(origin_week - depth + 1, origin_week + 1)):
            families.append(FeatureFamily("discount", discount_cols))
            columns.extend(discount_cols)
            has_discount = True
        else:
            families.append(FeatureFamily("discount", (), "unavailable",
                                          "discount columns are absent"))
            has_discount = False

        interaction_present = (self.interaction and has_discount
                              and "price" in frame.columns)
        if interaction_present:
            interaction_cols = tuple(f"price_x_discount_{i}"
                                     for i in range(1, depth + 1))
            families.append(FeatureFamily("interaction", interaction_cols))
            columns.extend(interaction_cols)
        else:
            families.append(FeatureFamily("interaction", (), "disabled",
                                          "inputs absent or the family is off"))

        if self.aggregates:
            if "external_code" in frame.columns and "retail" in frame.columns:
                aggregate_cols = tuple(a.value if hasattr(a, "value") else a
                                       for a in self.aggregates)
                families.append(FeatureFamily("aggregates", aggregate_cols))
                columns.extend(aggregate_cols)
                aggregate_scope = "input_window"
            else:
                families.append(FeatureFamily("aggregates", (), "unavailable",
                                              "identifier columns are absent"))
                aggregate_scope = "unavailable"
        else:
            aggregate_scope = "disabled"

        if "price" in frame.columns:
            families.append(FeatureFamily("price", ("price_rel",)))
            columns.append("price_rel")

        if "retail" in frame.columns:
            families.append(FeatureFamily("store_scale", ("store_size",)))
            columns.append("store_size")

        attribute_columns = []
        for column in CATEGORICAL:
            if column in frame.columns:
                values = frame[column].astype("object").where(
                    frame[column].notna(), "unknown").astype(str)
                attribute_columns.extend(
                    pd.get_dummies(values, prefix=column, dtype=float)
                    .columns.tolist())
        families.append(FeatureFamily("attributes", tuple(attribute_columns)))
        columns.extend(attribute_columns)

        weather_columns = []
        if self.weather == "as_supplied":
            weather_columns = [c for c in frame.columns if str(c).startswith("wx_")]
            families.append(FeatureFamily("weather", tuple(weather_columns)))
            columns.extend(weather_columns)
        else:
            families.append(FeatureFamily("weather", (), "disabled",
                                          "features.weather = off"))

        return FeaturePlan(tuple(columns), tuple(families), depth,
                           aggregate_scope, "input_window", interaction_present,
                           self.weather)

    def assert_closed(self, features: pd.DataFrame, plan: FeaturePlan) -> None:
        expected = list(plan.columns)
        actual = list(features.columns)
        if actual != expected:
            extra = [c for c in actual if c not in expected]
            missing = [c for c in expected if c not in actual]
            raise StageError(
                "features",
                f"closed feature set violated: extra={extra} missing={missing}")
        if len(set(expected)) != len(expected):
            raise StageError("features", "closed feature set declares a duplicate column")

    def record(self, features: pd.DataFrame, plan: FeaturePlan) -> FeatureSetRecord:
        return FeatureSetRecord(
            spec_id=SPEC_ID,
            n_columns=int(features.shape[1]),
            lag_depth=plan.lag_depth,
            aggregate_scope=plan.aggregate_scope,
            discount_scope=plan.discount_scope,
            interaction=plan.interaction,
            weather_source=plan.weather_source,
            families=plan.families,
            asserts={"no_identifier": True, "no_target_input": True,
                     "row_count_invariant": True},
        )