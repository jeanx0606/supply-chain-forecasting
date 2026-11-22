"""Chapter figure pack: read one bundle, never recompute (D27, A11, B28)."""
from __future__ import annotations

import csv
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from .figure_ids import reps_for

PACK_DIRNAME = "chapter_pack"


def rounded(value, digits: int = 4):
    if value is None:
        return None
    return float(Decimal(str(value)).quantize(
        Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def bundle_digest(bundle: dict) -> str:
    payload = json.dumps(bundle, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _figure(figure_id: str, value, unit: str, partition: str = "") -> dict:
    return {"figure": figure_id, "value": value, "unit": unit,
            "reps": list(reps_for(figure_id)), "partition": partition}


def build_pack(bundle: dict) -> dict:
    unit = (bundle.get("unit_provenance") or {}).get("unit", "raw_sales_units")
    sign = (bundle.get("unit_provenance") or {}).get("sign_policy", "")
    metrics = bundle.get("metrics", {})
    selected = bundle.get("main_model")
    main = metrics.get(selected, {})
    quality = bundle.get("quality", {})
    replenishment = bundle.get("replenishment", {})
    inventory = bundle.get("demonstration", {})
    risk = bundle.get("risk", {})
    selection = bundle.get("selection", {})
    feature_set = bundle.get("feature_set", {})
    initial = inventory.get("initial_stock") or {}
    reconstruction = inventory.get("reconstruction") or {}
    reconstruction_policy = reconstruction.get("policy") or {}
    replenish_assumptions = replenishment.get("assumptions") or {}
    error_scale = bundle.get("error_scale") or {}
    scale_units = {"mae": unit, "rmse": unit, "forecast_bias": unit,
                   "forecast_bias_item_level": unit}
    assumptions = bundle.get("assumptions") or []
    scenario_payload = bundle.get("scenarios") or {}
    scenario_rows = scenario_payload.get("rows") or []
    scenario_settings = scenario_payload.get("settings") or {}
    scenario_missing = [row.get("scenario") for row in
                        scenario_payload.get("unavailable") or []]
    slice_payload = bundle.get("error_slices") or {}
    slice_rows = slice_payload.get("slices") or []
    worst_groups = slice_payload.get("worst_groups") or []
    worst_group = (f"{worst_groups[0].get('slice')}:{worst_groups[0].get('group')}"
                   if worst_groups else "n/a")
    split_payload = bundle.get("split", {})
    split_counts = split_payload.get("counts") or {}
    split_preset = split_payload.get("preset") or {}

    figures = [
        _figure("quality.n_rows", quality.get("n_rows"), "count"),
        _figure("quality.negative_cells", quality.get("negative_cells"), "count"),
        _figure("quality.pair_coverage", rounded(quality.get("pair_coverage")), "ratio"),
        _figure("split.method", split_payload.get("method"), "name"),
        _figure("split.counts.train", split_counts.get("train"), "count"),
        _figure("split.counts.validation", split_counts.get("validation"), "count"),
        _figure("split.counts.test", split_counts.get("test"), "count"),
        _figure("split.preset.statement", split_preset.get("statement"), "name"),
        _figure("feature_set.n_columns", feature_set.get("n_columns"), "count"),
        _figure("selection.chosen_model", selection.get("chosen_model"), "name"),
        _figure("selection.outcome", selection.get("outcome"), "name"),
        _figure("model.sigma_e", rounded(error_scale.get("sigma_e")),
                error_scale.get("unit", unit),
                partition=error_scale.get("partition", "validation")),
        _figure("assumptions.count", len(assumptions), "count"),
        _figure("assumptions.open_decisions",
                ",".join(row.get("id", "") for row in assumptions
                         if row.get("status") != "encoded"), "name"),
        _figure("scenario.count", len(scenario_rows), "count"),
        _figure("scenario.service_level",
                rounded(scenario_settings.get("service_level"), 2), "ratio"),
        _figure("scenario.cost_index_status",
                scenario_settings.get("cost_index_reason"), "name"),
        _figure("scenario.unavailable", ",".join(scenario_missing) or "none", "name"),
        _figure("error_slice.reported_count", slice_payload.get("reported_count"),
                "count"),
        _figure("error_slice.min_slice_size",
                slice_payload.get("min_slice_size"), "count"),
        _figure("error_slice.worst_group", worst_group, "name"),
        _figure("error_slice.unavailable",
                ",".join(row.get("slice", "") for row in
                         slice_payload.get("unavailable") or []), "name"),
        _figure("replenishment.reorder_point",
                rounded(replenishment.get("reorder_point")), unit),
        _figure("replenishment.target_level",
                rounded(replenishment.get("target_level")), unit),
        _figure("replenishment.recommended_qty",
                replenishment.get("recommended_qty"), unit),
        _figure("replenishment.protection_weeks",
                replenish_assumptions.get("protection_weeks"), "week"),
        _figure("replenishment.simulation_label",
                replenish_assumptions.get("simulation_label"), "name"),
        _figure("inventory.initial_stock", rounded(initial.get("value")), unit),
        _figure("inventory.initial_stock_base", rounded(initial.get("base_value")), unit),
        _figure("inventory.initial_stock_correction",
                rounded(initial.get("correction")), unit),
        _figure("inventory.would_go_negative",
                reconstruction.get("would_go_negative"), "flag"),
        _figure("inventory.min_raw_stock", rounded(reconstruction.get("min_raw_stock")),
                unit),
        _figure("inventory.negative_weeks", reconstruction.get("negative_weeks"), "week"),
        _figure("reconstruction.alpha",
                rounded(reconstruction_policy.get("alpha"), 2), "ratio"),
        _figure("reconstruction.floor_at_zero",
                reconstruction_policy.get("a1_floor_at_zero"), "flag"),
        _figure("reconstruction.a3_negative_correction",
                reconstruction_policy.get("a3_negative_correction"), "flag"),
        _figure("inventory.sellout_week", inventory.get("sellout_week"), "week"),
        _figure("risk.stockout_proxy", risk.get("stockout_proxy"), "grade"),
        _figure("risk.overstock", risk.get("overstock"), "grade"),
        _figure("unit_provenance.unit", unit, "name"),
        _figure("unit_provenance.sign_policy", sign, "name"),
    ]
    for name, record in metrics.items():
        for metric_id in ("wape_item_level", "wape_pooled", "mae", "rmse",
                          "forecast_bias", "forecast_bias_item_level"):
            figures.append(_figure(
                f"metric.{metric_id}", rounded(record.get(metric_id)),
                scale_units.get(metric_id, "ratio"),
                partition=f"test:{name}"))
    for name, record in (bundle.get("error_scale_by_model") or {}).items():
        figures.append(_figure(
            "model.sigma_e", rounded(record.get("sigma_e")),
            record.get("unit", unit), partition=f"validation:{name}"))
    for row in scenario_rows:
        partition = f"test:{row.get('scenario')}"
        for figure_id, key, figure_unit in (
                ("scenario.fill_rate", "fill_rate", "ratio"),
                ("scenario.stockout_rate", "stockout_rate", "ratio"),
                ("scenario.overstock_ratio", "overstock_ratio", "ratio"),
                ("scenario.turnover", "turnover", "ratio"),
                ("scenario.weeks_of_cover", "weeks_of_cover", "week")):
            figures.append(_figure(figure_id, rounded(row.get(key)),
                                   figure_unit, partition=partition))
    for row in slice_rows:
        partition = f"test:{row.get('slice')}:{row.get('group')}"
        for figure_id, key, figure_unit in (
                ("error_slice.wape", "wape", "ratio"),
                ("error_slice.mae", "mae", unit),
                ("error_slice.rmse", "rmse", unit),
                ("error_slice.bias", "bias", unit)):
            figures.append(_figure(figure_id, rounded(row.get(key)),
                                   figure_unit, partition=partition))

    return {
        "label": "chapter figure pack",
        "note": "every figure traces to a bundle field; nothing is recomputed",
        "unit": unit,
        "sign_policy": sign,
        "main_model": selected,
        "main_model_status": bundle.get("main_model_status"),
        "provisional": bundle.get("provisional", []),
        "figures": figures,
        "trace": {f["figure"] + (f":{f['partition']}" if f["partition"] else ""):
                  f"bundle field value for {f['figure']}" for f in figures},
    }


def write_pack(bundle: dict, report_dir: Path | str, run_id: str) -> str:
    report_dir = Path(report_dir)
    pack_dir = report_dir / PACK_DIRNAME / f"{run_id}-{bundle_digest(bundle)}"
    pack_dir.mkdir(parents=True, exist_ok=True)

    pack = build_pack(bundle)
    with open(pack_dir / "figures.json", "w", encoding="utf-8") as handle:
        json.dump(pack, handle, indent=2, default=str, sort_keys=True)

    metric_rows = []
    for figure in pack["figures"]:
        metric_rows.append({key: figure[key] for key in
                            ("figure", "value", "unit", "partition")})
    with open(pack_dir / "figures.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["figure", "value", "unit", "partition"])
        writer.writeheader()
        writer.writerows(metric_rows)

    lines = ["# Chapter Figure Pack", "",
             f"- main model: {pack['main_model']} ({pack['main_model_status']})",
             f"- unit: {pack['unit']}",
             f"- sign policy: {pack['sign_policy']}",
             "- every figure is labelled with its unit and responsible REPs",
             "- provisional REPs: " + ", ".join(
                 ref.get("rep_id", "") for ref in pack["provisional"]),
             "", "| Figure | Value | Unit | REPs |", "|---|---|---|---|"]
    for figure in pack["figures"]:
        lines.append(f"| {figure['figure']} | {figure['value']} | {figure['unit']} | "
                     f"{', '.join(figure['reps'])} |")
    (pack_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(pack_dir)