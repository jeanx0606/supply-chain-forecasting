"""The single resolution point: the only module that reads seam selector keys."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config.schema import (
    AppConfig,
    ChannelPolicy,
    DemoRule,
    Granularity,
    InitialStockMethod,
    MetricName,
    ModelBackend,
    NaiveReference,
    OutputFormat,
    RestockPolicy,
    SafetyStockMethod,
    SplitPolicyName,
    WeatherJoinMode,
    WeekConvention,
)


@dataclass(frozen=True)
class SeamBundle:
    split: Any
    preset: Any
    models: Any
    metrics: Any
    retention: Any
    artefacts: Any
    initial_stock: Any
    weather_join: Any
    trends: Any
    safety_stock: Any
    demo: Any
    grain: Any
    weeks: Any
    backend: Any
    baseline: Any
    alt_task: Any
    sign_policy: Any
    features: Any
    selection_policy: Any

    def describe_all(self) -> tuple:
        fields = (
            "split", "preset", "models", "metrics", "retention", "artefacts",
            "initial_stock", "weather_join", "trends", "safety_stock", "demo",
            "grain", "weeks", "backend", "baseline", "alt_task", "sign_policy",
            "features", "selection_policy",
        )
        return tuple(getattr(self, name).describe() for name in fields)


def build_seams(cfg: AppConfig) -> SeamBundle:
    """Resolve every seam exactly once. Selector reads live only here."""
    from ..evaluate.aggregate import StandardMetricSet
    from ..features.spec import FeatureSpec
    from ..features.tasks import AltTaskSpec
    from ..io.channels import TrendsPolicy
    from ..models.backends import make_backend
    from ..models.baseline import BaselineDefinition
    from ..models.registry import ModelSelection
    from ..models.selection import ModelSelectionPolicy
    from ..preprocess.calendar import IsoWeekConvention
    from ..preprocess.cleaning import make_sign_policy
    from ..preprocess.inventory import make_initial_stock_strategy
    from ..preprocess.restock import make_retention_policy
    from ..preprocess.spine import ItemStoreGrain
    from ..preprocess.split import (ChronologicalSplit, PresetSplitPolicy,
                                    SeasonSplit)
    from ..preprocess.weather import make_weather_join
    from ..replenish.demo import make_demo_selector
    from ..replenish.safety import make_safety_stock
    from ..report.artefacts import make_output_writer

    permissive = cfg

    if permissive.spine.granularity is not Granularity.ITEM_STORE:
        from ..errors import ConfigError
        raise ConfigError("spine.granularity", "unsupported grain", observed=permissive.spine.granularity.value,
                          expected=Granularity.ITEM_STORE.value)

    if permissive.split.policy.value == "season":
        split = SeasonSplit()
    else:
        split = ChronologicalSplit(
            train_ratio=permissive.split.train_ratio,
            val_ratio=permissive.split.val_ratio,
            test_ratio=permissive.split.test_ratio,
            explicit_cut_dates=permissive.split.explicit_cut_dates,
            policy=permissive.split.policy,
        )

    return SeamBundle(
        split=split,
        preset=PresetSplitPolicy(enabled=permissive.split.use_preset_files,
                                files=permissive.split.preset_files,
                                scale_factor=permissive.split.preset_scale_factor),
        models=ModelSelection(selected=permissive.model.selected,
                              baseline=permissive.model.baseline),
        metrics=StandardMetricSet(primary=permissive.metrics.primary,
                                  supporting=permissive.metrics.supporting),
        retention=make_retention_policy(permissive.truncation.restock_policy,
                                        permissive.truncation.restock_date_max),
        artefacts=make_output_writer(permissive.output.enabled,
                                     permissive.output.dir,
                                     permissive.output.format,
                                     permissive.report.dir),
        initial_stock=make_initial_stock_strategy(
            permissive.inventory.initial_stock_method,
            permissive.inventory.lead_time_weeks,
            permissive.inventory.safety_factor,
            permissive.inventory.opening_stock_alpha),
        weather_join=make_weather_join(permissive.weather.join_mode),
        trends=TrendsPolicy(permissive.channels.gtrends),
        safety_stock=make_safety_stock(permissive.replenish.safety_stock_method,
                                       permissive.replenish.service_level),
        demo=make_demo_selector(permissive.demo.selection_rule,
                                permissive.demo.pin_item,
                                permissive.demo.pin_store),
        grain=ItemStoreGrain(),
        weeks=IsoWeekConvention(),
        backend=make_backend(permissive.model.backend, permissive.model.params,
                             permissive.run.seed),
        baseline=BaselineDefinition(permissive.baseline.naive_reference),
        alt_task=AltTaskSpec(permissive.tasks.alt_4in1out_enabled),
        sign_policy=make_sign_policy(permissive.cleaning.negative_sales_policy),
        features=FeatureSpec(permissive.features.lags,
                             permissive.features.history_aggregates,
                             permissive.features.price_discount_interaction,
                             permissive.features.weather),
        selection_policy=ModelSelectionPolicy(
            permissive.model.selection_rule.value,
            permissive.model.selection_metric.value,
            permissive.model.qualification.tolerance,
            permissive.model.qualification.mode.value,
            permissive.model.qualification.on_none.value),
    )


__all__ = [
    "SeamBundle", "build_seams", "AppConfig", "ChannelPolicy", "DemoRule",
    "Granularity", "InitialStockMethod", "MetricName", "ModelBackend",
    "NaiveReference", "OutputFormat", "RestockPolicy", "SafetyStockMethod",
    "SplitPolicyName", "WeatherJoinMode", "WeekConvention",
]