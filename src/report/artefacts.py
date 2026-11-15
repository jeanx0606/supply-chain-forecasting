"""Artefact writer with atomic staging and rename (D22, REP-06, O9)."""
from __future__ import annotations

import csv
import json
import shutil
import uuid
from pathlib import Path

from ..seams.protocols import SeamDescription


class JsonCsvWriter:
    SEAM_ID = "SEAM-OUTPUT-ARTEFACT"
    implementation = "json+csv"

    def __init__(self, enabled: bool = True, out_dir: Path | str = "output",
                 report_dir: Path | str | None = None):
        self.enabled = bool(enabled)
        self.out_dir = Path(out_dir)
        self.report_dir = Path(report_dir) if report_dir is not None else None

    def write_pack(self, bundle, run_id: str) -> str | None:
        from .chapter import write_pack

        if not self.enabled or self.report_dir is None:
            return None
        return write_pack(bundle.to_dict(), self.report_dir, run_id)

    def describe(self) -> SeamDescription:
        return SeamDescription("output_artefact", "REP-06", self.implementation)

    def _write_json(self, path: Path, payload) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

    def _write_csv(self, path: Path, rows: list, fieldnames: list) -> None:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})

    def write(self, bundle, run_id: str) -> str | None:
        if not self.enabled:
            return None
        staging = self.out_dir / f".staging-{uuid.uuid4().hex}"
        target = self.out_dir / run_id
        staging.mkdir(parents=True, exist_ok=True)
        try:
            payload = bundle.to_dict()
            if payload.get("register_version") is None:
                raise KeyError("register_version")
            self._write_json(staging / "bundle.json", payload)
            self._write_json(staging / "replenishment.json", payload["replenishment"])
            if payload.get("secondary_comparison"):
                self._write_json(staging / "secondary_comparison.json",
                                 payload["secondary_comparison"])

            quality = payload["quality"]
            self._write_csv(staging / "quality.csv", [quality], list(quality.keys()))

            metric_rows = list(payload["metrics"].values())
            metric_fields = ["model", "wape_item_level", "wape_pooled", "mae",
                             "rmse", "wape_store_weighted", "wape_small_stores",
                             "n_small_stores", "n_evaluated_records",
                             "n_evaluated_series", "unit", "applied_scale_factor",
                             "sign_policy", "partition"]
            self._write_csv(staging / "metrics.csv", metric_rows, metric_fields)

            inventory_rows = payload["demonstration"]["entries"]
            self._write_csv(staging / "inventory.csv", inventory_rows,
                            ["week_no", "restock", "sales", "reconstructed_stock"])

            provenance = payload.get("unit_provenance") or {}
            selection = payload.get("selection") or {}
            feature_set = payload.get("feature_set") or {}
            self._write_json(staging / "run_meta.json", {
                "run_id": run_id,
                "register_version": payload["register_version"],
                "seed": payload.get("seed"),
                "self_test": payload.get("self_test"),
                "n_evaluated_series": payload.get("n_evaluated_series"),
                "unit": provenance.get("unit"),
                "applied_scale_factor": provenance.get("applied_scale_factor"),
                "sign_policy": provenance.get("sign_policy"),
                "candidate_count": selection.get("candidate_count"),
                "selection_rule": selection.get("rule_id"),
                "main_model_status": selection.get("main_model_status"),
                "feature_set_id": feature_set.get("spec_id"),
            })

            if target.exists():
                shutil.rmtree(target)
            staging.replace(target)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return str(target)


class NullWriter:
    implementation = "disabled"

    def write_pack(self, bundle, run_id: str) -> str | None:
        return None

    def describe(self) -> SeamDescription:
        return SeamDescription("output_artefact", "REP-06", "disabled")

    def write(self, bundle, run_id: str) -> str | None:
        return None


def make_output_writer(enabled: bool, out_dir, fmt,
                       report_dir=None) -> JsonCsvWriter | NullWriter:
    if not enabled:
        return NullWriter()
    return JsonCsvWriter(True, out_dir, report_dir)