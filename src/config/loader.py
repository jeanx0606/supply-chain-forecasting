"""Typed configuration loader with fail-fast validation (S0)."""
from __future__ import annotations

import math
from pathlib import Path

import yaml

from ..errors import ConfigError
from .schema import (
    AppConfig,
    BaselineSection,
    ChannelPolicy,
    ChannelsSection,
    CleaningSection,
    DataSection,
    DemoRule,
    DemoSection,
    FileNames,
    Granularity,
    InitialStockMethod,
    InventorySection,
    KNOWN_MODELS,
    MetricsSection,
    MetricName,
    ModelBackend,
    AggregateName,
    FeatureSection,
    LagSpec,
    ModelSection,
    NaiveReference,
    NegativeSalesPolicyName,
    PreflightSection,
    QualificationMode,
    QualificationOnNone,
    QualificationSection,
    ReportSection,
    SelectionRuleName,
    WeatherFeaturePolicy,
    OutputFormat,
    OutputSection,
    ReplenishPolicy,
    ReplenishSection,
    RestockPolicy,
    RunSection,
    SafetyStockMethod,
    SplitPolicyName,
    SplitSection,
    SpineSection,
    TasksSection,
    TimeSection,
    TruncationSection,
    WeatherJoinMode,
    WeatherSection,
    WeekConvention,
)

DEFAULT_NEXT = "python -m codes.run_pipeline"


def _enum(enum_cls, value, key, hint=None):
    try:
        return enum_cls(value)
    except ValueError as exc:
        rule = "unregistered selector value"
        if hint:
            rule = f"{rule} ({hint})"
        raise ConfigError(
            key, rule, observed=value,
            expected=[m.value for m in enum_cls], next_command=DEFAULT_NEXT) from exc


def _inside(root: Path, path: Path, key: str) -> Path:
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ConfigError(
            key, "path escapes the project root", observed=str(resolved),
            expected=f"a path inside {root}", next_command=DEFAULT_NEXT) from exc
    return resolved


def load_config(path: str = "config/settings.yaml", root: str | Path = ".") -> AppConfig:
    root_path = Path(root).resolve()
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = root_path / cfg_path
    if not cfg_path.exists():
        raise ConfigError("config", "settings file not found", observed=str(cfg_path),
                          expected="config/settings.yaml", next_command=DEFAULT_NEXT)

    with open(cfg_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    for section in ("data", "run", "time", "spine", "split", "model", "baseline",
                    "metrics", "channels", "cleaning", "features", "report",
                    "preflight", "weather", "inventory", "truncation",
                    "replenish", "demo", "tasks", "output"):
        if section not in raw:
            raise ConfigError(section, "missing configuration section",
                              expected=f"section '{section}' in settings.yaml")

    data = raw["data"]
    files = data.get("files", {})
    n_weeks = int(data.get("n_weeks", 12))
    if n_weeks < 1:
        raise ConfigError("data.n_weeks", "must be a positive integer", observed=n_weeks)

    spine = raw["spine"]
    input_weeks = int(spine.get("input_weeks", 2))
    output_weeks = int(spine.get("output_weeks", 10))
    if input_weeks < 1 or output_weeks < 1:
        raise ConfigError("spine.input_weeks", "windows must be positive",
                          observed=(input_weeks, output_weeks))
    if input_weeks + output_weeks > n_weeks:
        raise ConfigError(
            "spine.input_weeks",
            f"input({input_weeks}) + output({output_weeks}) > n_weeks({n_weeks})",
            observed=input_weeks + output_weeks, expected=f"<= {n_weeks}")

    split = raw["split"]
    ratios = (float(split.get("train_ratio", 0.7)),
              float(split.get("val_ratio", 0.15)),
              float(split.get("test_ratio", 0.15)))
    if abs(sum(ratios) - 1.0) > 0.001:
        raise ConfigError("split.train_ratio", "ratios must sum to 1.0 +/- 0.001",
                          observed=round(sum(ratios), 6), expected="1.0 +/- 0.001")
    if any(r <= 0 for r in ratios):
        raise ConfigError("split.train_ratio", "every ratio must be positive",
                          observed=ratios, expected="all > 0")

    use_preset_files = bool(split.get("use_preset_files", False))
    factor_raw = split.get("preset_scale_factor")
    preset_scale_factor = None
    if factor_raw is not None:
        if isinstance(factor_raw, bool) or not isinstance(factor_raw, (int, float)):
            raise ConfigError(
                "split.preset_scale_factor",
                "factor must be a positive finite number",
                observed=factor_raw, expected="a number > 0")
        preset_scale_factor = float(factor_raw)
        if not math.isfinite(preset_scale_factor) or preset_scale_factor <= 0:
            raise ConfigError(
                "split.preset_scale_factor",
                "factor must be a positive finite number",
                observed=factor_raw, expected="a finite number > 0")
    if use_preset_files and preset_scale_factor is None:
        raise ConfigError(
            "split.preset_scale_factor",
            "a scaled read requires an explicitly configured factor",
            observed=None, expected="a positive number declared under REP-05")

    preset_files_raw = split.get("preset_files")
    preset_files = None
    if preset_files_raw is not None:
        if (not isinstance(preset_files_raw, (list, tuple))
                or len(preset_files_raw) != 2):
            raise ConfigError(
                "split.preset_files", "preset file names must be a pair",
                observed=preset_files_raw, expected="two file names")
        preset_files = tuple(str(name) for name in preset_files_raw)
        if preset_files[0] == preset_files[1]:
            raise ConfigError(
                "split.preset_files", "preset file names must be distinct",
                observed=preset_files, expected="two different file names")

    model = raw["model"]
    selected = str(model.get("selected", "gradient_boosting"))
    baseline = str(model.get("baseline", "seasonal_naive"))
    for key, value in (("model.selected", selected), ("model.baseline", baseline)):
        if value not in KNOWN_MODELS:
            raise ConfigError(key, "model not registered", observed=value,
                              expected=list(KNOWN_MODELS))

    candidates = tuple(str(name) for name in model.get("candidates", KNOWN_MODELS))
    unknown = [name for name in candidates if name not in KNOWN_MODELS]
    if unknown:
        raise ConfigError("model.candidates", "model not registered",
                          observed=unknown[0], expected=list(KNOWN_MODELS))
    if len(set(candidates)) != len(candidates):
        raise ConfigError("model.candidates", "candidate names must be distinct",
                          observed=list(candidates), expected="unique names")
    if "seasonal_naive" not in candidates:
        raise ConfigError("model.candidates",
                          "qualification reference is missing from the candidate set",
                          observed=list(candidates),
                          expected="'seasonal_naive' must be a candidate")
    for key, value in (("model.selected", selected), ("model.baseline", baseline)):
        if value not in candidates:
            raise ConfigError(key, "model is not among the declared candidates",
                              observed=value, expected=list(candidates))

    selection_rule = _enum(SelectionRuleName,
                           model.get("selection_rule", "min_primary_metric"),
                           "model.selection_rule")
    qualification_raw = model.get("qualification", {})
    qualification = QualificationSection(
        mode=_enum(QualificationMode,
                   qualification_raw.get("mode", "must_not_be_worse_than_naive"),
                   "model.qualification.mode"),
        tolerance=float(qualification_raw.get("tolerance", 0.0)),
        on_none=_enum(QualificationOnNone,
                      qualification_raw.get("on_none", "retained_default"),
                      "model.qualification.on_none"),
    )
    if (not math.isfinite(qualification.tolerance)
            or qualification.tolerance < 0):
        raise ConfigError("model.qualification.tolerance",
                          "tolerance must be a non-negative finite number",
                          observed=qualification_raw.get("tolerance"),
                          expected=">= 0 and finite")

    metrics = raw["metrics"]
    supporting = tuple(metrics.get("supporting", ()))
    primary = _enum(MetricName, metrics.get("primary", "wape_item_level"),
                    "metrics.primary")
    selection_metric = _enum(MetricName,
                             model.get("selection_metric", primary.value),
                             "model.selection_metric")
    if selection_metric is not primary:
        raise ConfigError("model.selection_metric",
                          "must equal metrics.primary (one metric decision, REP-03)",
                          observed=selection_metric.value,
                          expected=primary.value)

    features_raw = raw["features"]
    raw_lags = features_raw.get("lags", "input_window")
    if isinstance(raw_lags, int) and not isinstance(raw_lags, bool):
        if raw_lags < 1 or raw_lags > input_weeks:
            raise ConfigError(
                "features.lags",
                "a depth beyond the input window would read a target week",
                observed=raw_lags, expected=f"1..{input_weeks} (spine.input_weeks)")
        lags = raw_lags
    else:
        lags = _enum(LagSpec, raw_lags, "features.lags")
    aggregates = tuple(
        _enum(AggregateName, name, "features.history_aggregates")
        for name in features_raw.get("history_aggregates", ()))
    if len(set(aggregates)) != len(aggregates):
        raise ConfigError("features.history_aggregates",
                          "aggregate names must be distinct",
                          observed=[a.value for a in aggregates],
                          expected="{item_mean_input, store_mean_input}")
    interaction = features_raw.get("price_discount_interaction", True)
    if not isinstance(interaction, bool):
        raise ConfigError("features.price_discount_interaction",
                          "must be a YAML boolean", observed=interaction,
                          expected=[True, False])
    weather_feature = _enum(WeatherFeaturePolicy,
                            features_raw.get("weather", "as_supplied"),
                            "features.weather")

    report_raw = raw["report"]
    report_dir = _inside(root_path, Path(report_raw.get("dir", "reports")),
                         "report.dir")
    report_packet = _inside(root_path,
                            Path(report_raw.get("packet_path",
                                                "docs/DECISION_PACKET.md")),
                            "report.packet_path")
    preflight_raw = raw["preflight"]
    negative_reference = int(preflight_raw.get("negative_cell_reference", 3386))
    if negative_reference < 0:
        raise ConfigError("preflight.negative_cell_reference",
                          "must be non-negative", observed=negative_reference)

    replenish = raw["replenish"]
    service_level = float(replenish.get("service_level", 0.95))
    if not 0.0 < service_level < 1.0:
        raise ConfigError("replenish.service_level", "outside (0, 1)",
                          observed=service_level, expected="(0, 1)")
    high_weeks = int(replenish.get("risk_high_weeks", 1))
    medium_weeks = int(replenish.get("risk_medium_weeks", 2))
    if high_weeks < 1 or medium_weeks < 1:
        raise ConfigError("replenish.risk_high_weeks", "threshold weeks must be >= 1",
                          observed=(high_weeks, medium_weeks), expected=">= 1")
    if medium_weeks < high_weeks:
        raise ConfigError(
            "replenish.risk_medium_weeks",
            "medium horizon must not be shorter than the high horizon",
            observed=medium_weeks, expected=f">= replenish.risk_high_weeks ({high_weeks})")
    high_ratio = float(replenish.get("overstock_high_ratio", 2.0))
    medium_ratio = float(replenish.get("overstock_medium_ratio", 1.2))
    if medium_ratio >= high_ratio:
        raise ConfigError(
            "replenish.overstock_medium_ratio",
            "overstock_medium_ratio must be lower than overstock_high_ratio",
            observed=(medium_ratio, high_ratio), expected="medium < high")
    order_cycle = int(replenish.get("order_cycle_weeks", 1))
    if order_cycle < 1:
        raise ConfigError("replenish.order_cycle_weeks", "must be >= 1", observed=order_cycle)

    output = raw["output"]
    output_dir = _inside(root_path, Path(output.get("dir", "output")), "output.dir")
    if report_dir == output_dir or output_dir in report_dir.parents:
        raise ConfigError("report.dir",
                          "report.dir must not be inside output.dir",
                          observed=str(report_dir), expected=f"outside {output_dir}")
    raw_dir = _inside(root_path, Path(data.get("raw_dir", "data/raw")), "data.raw_dir")

    cfg = AppConfig(
        root=root_path,
        data=DataSection(
            raw_dir=raw_dir,
            files=FileNames(
                sales=str(files.get("sales", "sales.csv")),
                restocks=str(files.get("restocks", "restocks.csv")),
                price_discount=str(files.get("price_discount", "price_discount_series.csv")),
                gtrends=str(files.get("gtrends", "vis2_gtrends_data.csv")),
                weather=str(files.get("weather", "vis2_weather_data.csv")),
            ),
            n_weeks=n_weeks,
            sales_week_prefix=str(data.get("sales_week_prefix", "q")),
        ),
        run=RunSection(
            seed=int(raw["run"].get("seed", 42)),
            verbose=bool(raw["run"].get("verbose", True)),
        ),
        time=TimeSection(
            week_convention=_enum(WeekConvention, raw["time"].get("week_convention", "iso8601"),
                                  "time.week_convention"),
        ),
        spine=SpineSection(
            granularity=_enum(Granularity, spine.get("granularity", "item_store"),
                              "spine.granularity"),
            input_weeks=input_weeks,
            output_weeks=output_weeks,
        ),
        split=SplitSection(
            policy=_enum(SplitPolicyName, split.get("policy", "chronological"), "split.policy"),
            train_ratio=ratios[0],
            val_ratio=ratios[1],
            test_ratio=ratios[2],
            explicit_cut_dates=tuple(split["explicit_cut_dates"])
            if split.get("explicit_cut_dates") else None,
            use_preset_files=use_preset_files,
            preset_scale_factor=preset_scale_factor,
            preset_files=preset_files,
        ),
        model=ModelSection(
            baseline=baseline,
            selected=selected,
            backend=_enum(ModelBackend, model.get("backend", "sklearn_hgb"), "model.backend"),
            candidates=candidates,
            selection_rule=selection_rule,
            selection_metric=selection_metric,
            qualification=qualification,
            params=dict(model.get("params", {})),
        ),
        baseline=BaselineSection(
            naive_reference=_enum(NaiveReference, raw["baseline"].get(
                "naive_reference", "last_input"), "baseline.naive_reference"),
        ),
        metrics=MetricsSection(
            primary=primary,
            supporting=tuple(_enum(MetricName, m, "metrics.supporting") for m in supporting),
        ),
        channels=ChannelsSection(
            weather=_enum(ChannelPolicy, raw["channels"].get("weather", "consume"),
                          "channels.weather"),
            gtrends=_enum(ChannelPolicy, raw["channels"].get("gtrends", "deferred"),
                          "channels.gtrends"),
        ),
        cleaning=CleaningSection(
            negative_sales_policy=_enum(
                NegativeSalesPolicyName,
                raw["cleaning"].get("negative_sales_policy", "clip_at_zero"),
                "cleaning.negative_sales_policy",
                hint="capability 5.2 configuration control"),
        ),
        features=FeatureSection(
            lags=lags, history_aggregates=aggregates,
            price_discount_interaction=interaction,
            weather=weather_feature,
        ),
        report=ReportSection(
            enabled=bool(report_raw.get("enabled", True)),
            dir=report_dir,
            format=str(report_raw.get("format", "json+markdown")),
            packet_path=report_packet,
        ),
        preflight=PreflightSection(negative_cell_reference=negative_reference),
        weather=WeatherSection(
            max_interp_days=int(raw["weather"].get("max_interp_days", 30)),
            join_mode=_enum(WeatherJoinMode, raw["weather"].get(
                "join_mode", "pooled_representative"), "weather.join_mode"),
            agg=dict(raw["weather"].get("agg", {})),
        ),
        inventory=InventorySection(
            initial_stock_method=_enum(
                InitialStockMethod, raw["inventory"].get(
                    "initial_stock_method", "restocked_to_release"),
                "inventory.initial_stock_method"),
            lead_time_weeks=int(raw["inventory"].get("lead_time_weeks", 1)),
            safety_factor=float(raw["inventory"].get("safety_factor", 1.0)),
            opening_stock_alpha=float(raw["inventory"].get("opening_stock_alpha", 1.0)),
            negative_correction=bool(raw["inventory"].get("negative_correction", True)),
            floor_at_zero=bool(raw["inventory"].get("floor_at_zero", True)),
        ),
        truncation=TruncationSection(
            restock_policy=_enum(RestockPolicy, raw["truncation"].get(
                "restock_policy", "lifecycle_membership"), "truncation.restock_policy"),
            restock_date_max=str(raw["truncation"].get("restock_date_max", "2019-12-31")),
        ),
        replenish=ReplenishSection(
            policy=_enum(ReplenishPolicy, replenish.get("policy", "sS"), "replenish.policy"),
            safety_stock_method=_enum(
                SafetyStockMethod, replenish.get("safety_stock_method", "normal_approx"),
                "replenish.safety_stock_method"),
            service_level=service_level,
            order_cycle_weeks=order_cycle,
            risk_high_weeks=high_weeks,
            risk_medium_weeks=medium_weeks,
            overstock_high_ratio=high_ratio,
            overstock_medium_ratio=medium_ratio,
        ),
        demo=DemoSection(
            selection_rule=_enum(DemoRule, raw["demo"].get(
                "selection_rule", "max_horizon_demand"), "demo.selection_rule"),
            pin_item=raw["demo"].get("pin_item"),
            pin_store=raw["demo"].get("pin_store"),
        ),
        tasks=TasksSection(
            alt_4in1out_enabled=bool(
                raw["tasks"].get("alt_4in1out", {}).get("enabled", True)),
        ),
        output=OutputSection(
            enabled=bool(output.get("enabled", True)),
            dir=output_dir,
            format=_enum(OutputFormat, output.get("format", "json+csv"), "output.format"),
        ),
    )
    _validate_inventory(cfg)
    return cfg


def _validate_inventory(cfg: AppConfig) -> None:
    if cfg.inventory.lead_time_weeks < 1:
        raise ConfigError("inventory.lead_time_weeks", "must be >= 1",
                          observed=cfg.inventory.lead_time_weeks)
    if cfg.inventory.safety_factor <= 0:
        raise ConfigError("inventory.safety_factor", "must be > 0",
                          observed=cfg.inventory.safety_factor)
    if cfg.inventory.opening_stock_alpha <= 0:
        raise ConfigError("inventory.opening_stock_alpha", "must be > 0",
                          observed=cfg.inventory.opening_stock_alpha)


def effective_config_dict(cfg: AppConfig) -> dict:
    """Serialisable view used by the bundle and the artefacts."""
    return {
        "root": str(cfg.root),
        "data": {
            "raw_dir": str(cfg.data.raw_dir),
            "files": cfg.data.files.__dict__,
            "n_weeks": cfg.data.n_weeks,
            "sales_week_prefix": cfg.data.sales_week_prefix,
        },
        "run": {"seed": cfg.run.seed, "verbose": cfg.run.verbose},
        "time": {"week_convention": cfg.time.week_convention.value},
        "spine": {
            "granularity": cfg.spine.granularity.value,
            "input_weeks": cfg.spine.input_weeks,
            "output_weeks": cfg.spine.output_weeks,
        },
        "split": {
            "policy": cfg.split.policy.value,
            "train_ratio": cfg.split.train_ratio,
            "val_ratio": cfg.split.val_ratio,
            "test_ratio": cfg.split.test_ratio,
            "explicit_cut_dates": cfg.split.explicit_cut_dates,
            "use_preset_files": cfg.split.use_preset_files,
            "preset_scale_factor": cfg.split.preset_scale_factor,
            "preset_files": list(cfg.split.preset_files)
            if cfg.split.preset_files else None,
        },
        "model": {
            "baseline": cfg.model.baseline,
            "selected": cfg.model.selected,
            "backend": cfg.model.backend.value,
            "candidates": list(cfg.model.candidates),
            "selection_rule": cfg.model.selection_rule.value,
            "selection_metric": cfg.model.selection_metric.value,
            "qualification": {
                "mode": cfg.model.qualification.mode.value,
                "tolerance": cfg.model.qualification.tolerance,
                "on_none": cfg.model.qualification.on_none.value,
            },
            "params": cfg.model.params,
        },
        "baseline": {"naive_reference": cfg.baseline.naive_reference.value},
        "metrics": {
            "primary": cfg.metrics.primary.value,
            "supporting": [m.value for m in cfg.metrics.supporting],
        },
        "channels": {
            "weather": cfg.channels.weather.value,
            "gtrends": cfg.channels.gtrends.value,
        },
        "cleaning": {
            "negative_sales_policy": cfg.cleaning.negative_sales_policy.value,
        },
        "features": {
            "lags": getattr(cfg.features.lags, "value", cfg.features.lags),
            "history_aggregates": [a.value for a in cfg.features.history_aggregates],
            "price_discount_interaction": cfg.features.price_discount_interaction,
            "weather": cfg.features.weather.value,
        },
        "report": {
            "enabled": cfg.report.enabled,
            "dir": str(cfg.report.dir),
            "format": cfg.report.format,
            "packet_path": str(cfg.report.packet_path),
        },
        "preflight": {
            "negative_cell_reference": cfg.preflight.negative_cell_reference,
        },
        "weather": {
            "max_interp_days": cfg.weather.max_interp_days,
            "join_mode": cfg.weather.join_mode.value,
            "agg": cfg.weather.agg,
        },
        "inventory": {
            "initial_stock_method": cfg.inventory.initial_stock_method.value,
            "lead_time_weeks": cfg.inventory.lead_time_weeks,
            "safety_factor": cfg.inventory.safety_factor,
            "opening_stock_alpha": cfg.inventory.opening_stock_alpha,
            "negative_correction": cfg.inventory.negative_correction,
            "floor_at_zero": cfg.inventory.floor_at_zero,
        },
        "truncation": {
            "restock_policy": cfg.truncation.restock_policy.value,
            "restock_date_max": cfg.truncation.restock_date_max,
        },
        "replenish": {
            "policy": cfg.replenish.policy.value,
            "safety_stock_method": cfg.replenish.safety_stock_method.value,
            "service_level": cfg.replenish.service_level,
            "order_cycle_weeks": cfg.replenish.order_cycle_weeks,
            "risk_high_weeks": cfg.replenish.risk_high_weeks,
            "risk_medium_weeks": cfg.replenish.risk_medium_weeks,
            "overstock_high_ratio": cfg.replenish.overstock_high_ratio,
            "overstock_medium_ratio": cfg.replenish.overstock_medium_ratio,
        },
        "demo": {
            "selection_rule": cfg.demo.selection_rule.value,
            "pin_item": cfg.demo.pin_item,
            "pin_store": cfg.demo.pin_store,
        },
        "tasks": {"alt_4in1out": {"enabled": cfg.tasks.alt_4in1out_enabled}},
        "output": {
            "enabled": cfg.output.enabled,
            "dir": str(cfg.output.dir),
            "format": cfg.output.format.value,
        },
    }