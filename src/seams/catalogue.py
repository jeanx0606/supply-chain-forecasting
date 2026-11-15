"""Seam catalogue: seam code -> owner module -> config key -> REP id -> default.

The marker literal is *not* stored here; ``codes/check_seams.py`` derives it,
so a naive grep of ``SEAM-`` yields exactly one hit per seam (DFX-O5).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeamEntry:
    code: str
    owner: str
    config_key: str
    rep_id: str
    default: str
    confirmer: str

    @property
    def marker(self) -> str:
        return "SEAM-" + self.code.upper().replace("_", "-")


ENTRIES: tuple[SeamEntry, ...] = (
    SeamEntry("split_cut", "src/preprocess/split.py", "split.policy",
              "REP-01", "chronological", "Linda"),
    SeamEntry("model_select", "src/models/registry.py", "model.selected",
              "REP-02", "gradient_boosting", "Linda"),
    SeamEntry("metric_set", "src/evaluate/aggregate.py", "metrics.primary",
              "REP-03", "wape_item_level", "Linda"),
    SeamEntry("restock_trunc", "src/preprocess/restock.py", "truncation.restock_policy",
              "REP-04", "lifecycle_membership", "Rue"),
    SeamEntry("preset_files", "src/preprocess/split.py", "split.use_preset_files",
              "REP-05", "false", "Rue"),
    SeamEntry("output_artefact", "src/report/artefacts.py", "output.format",
              "REP-06", "json+csv", "Ashley"),
    SeamEntry("initial_stock", "src/preprocess/inventory.py",
              "inventory.initial_stock_method", "REP-07", "lead_time_demand", "team"),
    SeamEntry("weather_join", "src/preprocess/weather.py", "weather.join_mode",
              "REP-08", "pooled_representative", "Rue"),
    SeamEntry("trends", "src/io/channels.py", "channels.gtrends",
              "REP-09", "deferred", "Ashley"),
    SeamEntry("safety_stock", "src/replenish/safety.py",
              "replenish.safety_stock_method", "REP-10", "normal_approx", "Linda"),
    SeamEntry("demo_select", "src/replenish/demo.py", "demo.selection_rule",
              "REP-11", "max_horizon_demand", "Ashley"),
    SeamEntry("grain", "src/preprocess/spine.py", "spine.granularity",
              "REP-12", "item_store", "Linda"),
    SeamEntry("week_convention", "src/preprocess/calendar.py", "time.week_convention",
              "REP-13", "iso8601", "Rue"),
    SeamEntry("model_backend", "src/models/backends.py", "model.backend",
              "REP-14", "sklearn_hgb", "Ashley"),
    SeamEntry("baseline_def", "src/models/baseline.py", "baseline.naive_reference",
              "REP-15", "last_input", "Linda"),
    SeamEntry("alt_task", "src/features/tasks.py", "tasks.alt_4in1out.enabled",
              "REP-16", "true", "Ashley"),
    SeamEntry("negative_sales", "src/preprocess/cleaning.py",
              "cleaning.negative_sales_policy", "REP-17", "clip_at_zero", "Rue"),
    SeamEntry("feature_set", "src/features/spec.py", "features.lags",
              "REP-19", "input_window", "Linda"),
    SeamEntry("model_selection", "src/models/selection.py", "model.selection_rule",
              "REP-18", "min_primary_metric", "Linda"),
)

BY_CODE = {entry.code: entry for entry in ENTRIES}
BY_REP = {entry.rep_id: entry for entry in ENTRIES}
BY_OWNER = {}
for _entry in ENTRIES:
    BY_OWNER.setdefault(_entry.owner, []).append(_entry)


def get(code: str) -> SeamEntry:
    return BY_CODE[code]


def markers() -> dict[str, str]:
    return {entry.code: entry.marker for entry in ENTRIES}