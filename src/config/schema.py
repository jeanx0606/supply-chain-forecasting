"""Frozen configuration surface. No stage ever reads a raw dict (DFX-M1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path


class WeekConvention(str, Enum):
    ISO8601 = "iso8601"


class Granularity(str, Enum):
    ITEM_STORE = "item_store"


class ModelBackend(str, Enum):
    SKLEARN_HGB = "sklearn_hgb"
    LIGHTGBM = "lightgbm"


class SplitPolicyName(str, Enum):
    CHRONOLOGICAL = "chronological"
    EXPLICIT = "explicit"
    SEASON = "season"


class NaiveReference(str, Enum):
    LAST_INPUT = "last_input"
    MEAN_INPUT = "mean_input"


class MetricName(str, Enum):
    WAPE_ITEM_LEVEL = "wape_item_level"
    WAPE_POOLED = "wape_pooled"
    MAE = "mae"
    RMSE = "rmse"
    WAPE_STORE_WEIGHTED = "wape_store_weighted"
    WAPE_SMALL_STORES = "wape_small_stores"


class ChannelPolicy(str, Enum):
    CONSUME = "consume"
    DEFERRED = "deferred"
    SKIP = "skip"


class WeatherJoinMode(str, Enum):
    POOLED_REPRESENTATIVE = "pooled_representative"
    OMITTED = "omitted"
    MAPPED = "mapped"


class InitialStockMethod(str, Enum):
    RESTOCKED_TO_RELEASE = "restocked_to_release"
    LEAD_TIME_DEMAND = "lead_time_demand"
    FIRST_RESTOCK = "first_restock"
    ROLLING_MINIMUM = "rolling_minimum"


class RestockPolicy(str, Enum):
    LIFECYCLE_MEMBERSHIP = "lifecycle_membership"
    CALENDAR_CUTOFF = "calendar_cutoff"


class ReplenishPolicy(str, Enum):
    SS = "sS"


class SafetyStockMethod(str, Enum):
    NORMAL_APPROX = "normal_approx"
    NONE = "none"


class NegativeSalesPolicyName(StrEnum):
    CLIP_AT_ZERO = "clip_at_zero"
    RETAIN_AS_RETURNS = "retain_as_returns"


class SelectionRuleName(StrEnum):
    MIN_PRIMARY_METRIC = "min_primary_metric"


class QualificationMode(StrEnum):
    MUST_NOT_BE_WORSE_THAN_NAIVE = "must_not_be_worse_than_naive"
    STRICTLY_BETTER = "strictly_better"


class QualificationOnNone(StrEnum):
    RETAINED_DEFAULT = "retained_default"
    LEAST_BAD_PROVISIONAL = "least_bad_provisional"
    FAIL_RUN = "fail_run"


class LagSpec(StrEnum):
    INPUT_WINDOW = "input_window"


class AggregateName(StrEnum):
    ITEM_MEAN_INPUT = "item_mean_input"
    STORE_MEAN_INPUT = "store_mean_input"


class WeatherFeaturePolicy(StrEnum):
    AS_SUPPLIED = "as_supplied"
    OFF = "off"


class DemoRule(str, Enum):
    MAX_HORIZON_DEMAND = "max_horizon_demand"
    PINNED = "pinned"


class OutputFormat(str, Enum):
    JSON_CSV = "json+csv"


KNOWN_MODELS = ("attribute_knn", "category_curve", "gradient_boosting",
                "lag_regression", "lightgbm_horizon", "moving_average",
                "seasonal_naive")


@dataclass(frozen=True)
class FileNames:
    sales: str
    restocks: str
    price_discount: str
    gtrends: str
    weather: str


@dataclass(frozen=True)
class CleaningSection:
    negative_sales_policy: NegativeSalesPolicyName


@dataclass(frozen=True)
class QualificationSection:
    mode: QualificationMode
    tolerance: float
    on_none: QualificationOnNone


@dataclass(frozen=True)
class FeatureSection:
    lags: "int | LagSpec"
    history_aggregates: tuple
    price_discount_interaction: bool
    weather: WeatherFeaturePolicy


@dataclass(frozen=True)
class ReportSection:
    enabled: bool
    dir: Path
    format: str
    packet_path: Path


@dataclass(frozen=True)
class PreflightSection:
    negative_cell_reference: int


@dataclass(frozen=True)
class DataSection:
    raw_dir: Path
    files: FileNames
    n_weeks: int
    sales_week_prefix: str


@dataclass(frozen=True)
class RunSection:
    seed: int
    verbose: bool


@dataclass(frozen=True)
class TimeSection:
    week_convention: WeekConvention


@dataclass(frozen=True)
class SpineSection:
    granularity: Granularity
    input_weeks: int
    output_weeks: int


@dataclass(frozen=True)
class SplitSection:
    policy: SplitPolicyName
    train_ratio: float
    val_ratio: float
    test_ratio: float
    explicit_cut_dates: tuple[str, str] | None
    use_preset_files: bool
    preset_scale_factor: float | None
    preset_files: tuple[str, str] | None


@dataclass(frozen=True)
class ModelSection:
    baseline: str
    selected: str
    backend: ModelBackend
    candidates: tuple
    selection_rule: SelectionRuleName
    selection_metric: "MetricName"
    qualification: QualificationSection
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BaselineSection:
    naive_reference: NaiveReference


@dataclass(frozen=True)
class MetricsSection:
    primary: MetricName
    supporting: tuple[MetricName, ...]


@dataclass(frozen=True)
class ChannelsSection:
    weather: ChannelPolicy
    gtrends: ChannelPolicy


@dataclass(frozen=True)
class WeatherSection:
    max_interp_days: int
    join_mode: WeatherJoinMode
    agg: dict


@dataclass(frozen=True)
class InventorySection:
    initial_stock_method: InitialStockMethod
    lead_time_weeks: int
    safety_factor: float
    opening_stock_alpha: float = 1.0
    negative_correction: bool = True
    floor_at_zero: bool = True


@dataclass(frozen=True)
class TruncationSection:
    restock_policy: RestockPolicy
    restock_date_max: str


@dataclass(frozen=True)
class ReplenishSection:
    policy: ReplenishPolicy
    safety_stock_method: SafetyStockMethod
    service_level: float
    order_cycle_weeks: int
    risk_high_weeks: int
    risk_medium_weeks: int
    overstock_high_ratio: float
    overstock_medium_ratio: float


@dataclass(frozen=True)
class DemoSection:
    selection_rule: DemoRule
    pin_item: int | None
    pin_store: int | None


@dataclass(frozen=True)
class TasksSection:
    alt_4in1out_enabled: bool


@dataclass(frozen=True)
class OutputSection:
    enabled: bool
    dir: Path
    format: OutputFormat


@dataclass(frozen=True)
class AppConfig:
    root: Path
    data: DataSection
    run: RunSection
    time: TimeSection
    spine: SpineSection
    split: SplitSection
    model: ModelSection
    baseline: BaselineSection
    metrics: MetricsSection
    channels: ChannelsSection
    cleaning: CleaningSection
    features: FeatureSection
    report: ReportSection
    preflight: PreflightSection
    weather: WeatherSection
    inventory: InventorySection
    truncation: TruncationSection
    replenish: ReplenishSection
    demo: DemoSection
    tasks: TasksSection
    output: OutputSection


CONFIG_SECTIONS = (
    "data", "run", "time", "spine", "split", "model", "baseline",
    "metrics", "channels", "cleaning", "features", "report", "preflight",
    "weather", "inventory", "truncation", "replenish", "demo", "tasks", "output",
)