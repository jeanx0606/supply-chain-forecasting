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

    figures = [
        _figure("quality.n_rows", quality.get("n_rows"), "count"),
        _figure("quality.negative_cells", quality.get("negative_cells"), "count"),
        _figure("quality.pair_coverage", rounded(quality.get("pair_coverage")), "ratio"),
        _figure("split.counts", bundle.get("split", {}).get("counts"), "count"),
        _figure("feature_set.n_columns", feature_set.get("n_columns"), "count"),
        _figure("selection.chosen_model", selection.get("chosen_model"), "name"),
        _figure("selection.outcome", selection.get("outcome"), "name"),
        _figure("replenishment.reorder_point",
                rounded(replenishment.get("reorder_point")), unit),
        _figure("replenishment.target_level",
                rounded(replenishment.get("target_level")), unit),
        _figure("replenishment.recommended_qty",
                replenishment.get("recommended_qty"), unit),
        _figure("inventory.sellout_week", inventory.get("sellout_week"), "week"),
        _figure("risk.stockout_proxy", risk.get("stockout_proxy"), "grade"),
        _figure("risk.overstock", risk.get("overstock"), "grade"),
        _figure("unit_provenance.unit", unit, "name"),
        _figure("unit_provenance.sign_policy", sign, "name"),
    ]
    for name, record in metrics.items():
        for metric_id in ("wape_item_level", "wape_pooled", "mae", "rmse"):
            figures.append(_figure(
                f"metric.{metric_id}", rounded(record.get(metric_id)), "ratio",
                partition=f"test:{name}"))

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