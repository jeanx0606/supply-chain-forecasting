"""Configuration consumption ledger: a declared-but-unread key aborts the run."""
from __future__ import annotations

from ..errors import ConfigError

# Grows monotonically as stages land; exhaustive at S13 (DFX-M2).
KEY_OWNERS = {
    "data.raw_dir": "S1 ingest",
    "data.files.sales": "S1 ingest",
    "data.files.restocks": "S1 ingest",
    "data.files.price_discount": "S1 ingest",
    "data.files.gtrends": "S1 channel registry",
    "data.files.weather": "S1 channel registry",
    "data.n_weeks": "S3 weekly index",
    "data.sales_week_prefix": "S1 header normalisation",
    "run.seed": "S9 ensembles",
    "run.verbose": "runtime logging",
    "time.week_convention": "S3/S5/S6 week math",
    "spine.granularity": "S4 grain adapter",
    "spine.input_weeks": "S7 task construction",
    "spine.output_weeks": "S7 task construction",
    "split.policy": "S8 split policy",
    "split.train_ratio": "S8 split policy",
    "split.val_ratio": "S8 split policy",
    "split.test_ratio": "S8 split policy",
    "split.explicit_cut_dates": "S8 split policy",
    "split.use_preset_files": "S8 preset policy",
    "split.preset_scale_factor": "S8 preset policy",
    "split.preset_files": "S1 preset partition probe",
    "model.baseline": "S9 model selection",
    "model.selected": "S9 model selection",
    "model.backend": "S9 backend adapter",
    "model.params": "S9 boosting backend",
    "baseline.naive_reference": "S9 baseline definition",
    "metrics.primary": "S10 metric set",
    "metrics.supporting": "S10 metric set",
    "channels.weather": "S6 weather channel",
    "channels.gtrends": "S1 channel registry",
    "cleaning.negative_sales_policy": "S2 sign policy",
    "weather.max_interp_days": "S6 weather interpolation",
    "weather.join_mode": "S6 weather join adapter",
    "weather.agg": "S6 weather aggregation",
    "inventory.initial_stock_method": "S11 initial-stock strategy",
    "inventory.lead_time_weeks": "S11 initial-stock strategy",
    "inventory.safety_factor": "S11 initial-stock strategy",
    "inventory.opening_stock_alpha": "S11 initial-stock strategy",
    "inventory.negative_correction": "S11 reconstruction policy",
    "inventory.floor_at_zero": "S11 reconstruction policy",
    "truncation.restock_policy": "S5 retention policy",
    "truncation.restock_date_max": "S5 retention policy",
    "replenish.policy": "S12 recommendation",
    "replenish.safety_stock_method": "S12 safety-stock adapter",
    "replenish.service_level": "S12 safety-stock adapter",
    "replenish.order_cycle_weeks": "S12 recommendation",
    "replenish.risk_high_weeks": "S12 risk grading",
    "replenish.risk_medium_weeks": "S12 risk grading",
    "replenish.overstock_high_ratio": "S12 risk grading",
    "replenish.overstock_medium_ratio": "S12 risk grading",
    "demo.selection_rule": "S12 demonstration selector",
    "demo.pin_item": "S12 demonstration selector",
    "demo.pin_store": "S12 demonstration selector",
    "tasks.alt_4in1out.enabled": "S7 alternative task",
    "model.candidates": "S9 candidate tournament",
    "model.selection_rule": "S9 candidate tournament",
    "model.selection_metric": "S9 candidate tournament",
    "model.qualification.mode": "S9 qualification",
    "model.qualification.tolerance": "S9 qualification",
    "model.qualification.on_none": "S9 no-qualification path",
    "features.lags": "S7 feature specification",
    "features.history_aggregates": "S7 feature specification",
    "features.price_discount_interaction": "S7 feature specification",
    "features.weather": "S7 feature specification",
    "output.enabled": "S13 artefact writer",
    "output.dir": "S13 artefact writer",
    "output.format": "S13 artefact writer",
}

DECLARED_KEYS = frozenset(KEY_OWNERS)

OFFLINE_KEYS = frozenset({
    "preflight.negative_cell_reference",
    "report.dir", "report.enabled", "report.format", "report.packet_path",
})


class ConfigLedger:
    """Records which configuration keys the run actually consumed."""

    def __init__(self, declared=None):
        if declared is None:
            declared = DECLARED_KEYS - OFFLINE_KEYS
        self.declared = frozenset(declared)
        self.consumed: set[str] = set()

    @classmethod
    def default_ledger(cls):
        return cls(DECLARED_KEYS - OFFLINE_KEYS)

    def consume(self, *keys: str) -> None:
        for key in keys:
            self.consumed.add(key)

    def waive(self, *keys: str) -> None:
        """Declare keys legitimately inactive for this run (e.g. skipped channel)."""
        self.declared = frozenset(self.declared - set(keys))

    def missing(self) -> list[str]:
        return sorted(self.declared - self.consumed)

    def assert_consumed(self, next_command: str = "python -m codes.run_pipeline") -> None:
        missing = self.missing()
        if missing:
            key = missing[0]
            raise ConfigError(
                key, "declared but not consumed",
                observed=missing, expected=KEY_OWNERS.get(key, "owning capability"),
                next_command=next_command)


def check_unconsumed(declared, consumed) -> list[str]:
    return sorted(set(declared) - set(consumed))