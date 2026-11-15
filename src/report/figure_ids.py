"""One figure vocabulary shared by the decision packet and the chapter pack."""
from __future__ import annotations

FIGURE_REPS = {
    "quality.n_rows": ("REP-12",),
    "quality.negative_cells": ("REP-17",),
    "quality.pair_coverage": ("REP-05",),
    "split.counts": ("REP-01", "REP-05"),
    "feature_set.n_columns": ("REP-19",),
    "selection.chosen_model": ("REP-18",),
    "selection.outcome": ("REP-18",),
    "metric.wape_item_level": ("REP-02", "REP-03", "REP-18", "REP-19"),
    "metric.wape_pooled": ("REP-03", "REP-18", "REP-19"),
    "metric.mae": ("REP-03", "REP-18", "REP-19"),
    "metric.rmse": ("REP-03", "REP-18", "REP-19"),
    "replenishment.reorder_point": ("REP-07", "REP-10", "REP-17"),
    "replenishment.target_level": ("REP-04", "REP-10", "REP-17"),
    "replenishment.recommended_qty": ("REP-04", "REP-07", "REP-10", "REP-17"),
    "inventory.sellout_week": ("REP-07", "REP-11", "REP-17"),
    "risk.stockout_proxy": ("REP-10", "REP-11"),
    "risk.overstock": ("REP-10", "REP-11"),
    "unit_provenance.unit": ("REP-05",),
    "unit_provenance.sign_policy": ("REP-17",),
}


def reps_for(figure_id: str) -> tuple:
    return FIGURE_REPS.get(figure_id, ())


def label(figure_id: str) -> str:
    return f"{figure_id} [{', '.join(reps_for(figure_id)) or 'REP-06'}]"