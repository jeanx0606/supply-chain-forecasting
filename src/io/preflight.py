"""Read-only data preflight: names the file, the field and the next command (A10).

This module never imports the pipeline and never fits anything.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from ..preprocess.cleaning import count_negatives
from ..preprocess.time_index import monday_violations
from ..sample import MARKER_NAME
from .contracts import header_problems, header_shape, violations
from .schema import (
    PRICE_CONTRACT,
    RESTOCK_CONTRACT,
    SALES_CONTRACT,
    TRENDS_CONTRACT,
    WEATHER_CONTRACT,
)

MANDATORY = (("sales", SALES_CONTRACT), ("restocks", RESTOCK_CONTRACT),
             ("price_discount", PRICE_CONTRACT))
OPTIONAL = (("weather", WEATHER_CONTRACT), ("gtrends", TRENDS_CONTRACT))
NEXT_AFTER_FIX = "python -m codes.run_pipeline"


def marker_hash(marker: dict) -> str:
    encoded = json.dumps(marker, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def classify_dataset(raw_dir: Path) -> dict:
    """sample / real / mixed / absent, from the marker and the five expected files."""
    marker_path = raw_dir / MARKER_NAME
    present_files = [name for name in
                     ("sales.csv", "restocks.csv", "price_discount_series.csv",
                      "vis2_weather_data.csv", "vis2_gtrends_data.csv")
                     if (raw_dir / name).exists()]
    if marker_path.exists() and present_files:
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            marker = {"unreadable": True}
        declared = marker.get("files") if isinstance(marker, dict) else None
        if declared:
            mismatched = []
            for name, digest in declared.items():
                candidate = raw_dir / name
                if not candidate.exists():
                    mismatched.append(f"{name}: missing")
                    continue
                if hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
                    mismatched.append(f"{name}: modified")
            if mismatched:
                return {"data_class": "mixed", "marker": str(marker_path),
                        "marker_sha256": marker_hash(marker),
                        "reason": "marker hashes disagree: " + "; ".join(mismatched)}
            return {"data_class": "sample", "marker": str(marker_path),
                    "marker_sha256": marker_hash(marker),
                    "reason": "marker hashes match every file"}
        return {"data_class": "sample" if len(present_files) >= 5 else "mixed",
                "marker": str(marker_path),
                "marker_sha256": marker_hash(marker),
                "reason": "self-test marker present (no per-file hashes)"}
    if present_files:
        return {"data_class": "real", "marker": None, "marker_sha256": None,
                "reason": "no sample marker"}
    return {"data_class": "absent", "marker": None, "marker_sha256": None,
            "reason": "no raw files"}


def _read(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return None, f"unreadable: {exc}"
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    return frame, None


def run_preflight(cfg) -> dict:
    raw = cfg.data.raw_dir
    files = cfg.data.files
    report = {
        "data_class": classify_dataset(raw),
        "raw_dir": str(raw),
        "files": {},
        "optional_files": [],
        "readiness": "unknown",
        "problems": [],
    }

    for label, contract, name in (
            ("sales", SALES_CONTRACT, files.sales),
            ("restocks", RESTOCK_CONTRACT, files.restocks),
            ("price_discount", PRICE_CONTRACT, files.price_discount),
            ("weather", WEATHER_CONTRACT, files.weather),
            ("gtrends", TRENDS_CONTRACT, files.gtrends)):
        path = raw / name
        required = label in ("sales", "restocks", "price_discount")
        entry = {"present": path.exists(), "required": required, "file": name}
        if not path.exists():
            entry["status"] = "absent"
            if not required:
                report["optional_files"].append({"file": name, "status": "absent"})
                report["files"][label] = entry
                continue
            report["problems"].append(f"missing mandatory input '{name}'")
            report["files"][label] = entry
            continue

        frame, error = _read(path)
        if error:
            entry["status"] = error
            report["problems"].append(f"{name}: {error}")
            report["files"][label] = entry
            continue

        contract_problems = violations(frame, contract)
        if contract.week_columns:
            entry["header_shape"] = header_shape(list(frame.columns),
                                                 contract.week_columns)
            contract_problems += header_problems(list(frame.columns),
                                                 contract.week_columns)
        if contract_problems:
            entry["status"] = "; ".join(contract_problems)
            if required:
                report["problems"].append(f"{name}: {contract_problems[0]}")
        else:
            entry["status"] = "ok"
        entry["rows"] = int(len(frame))
        report["files"][label] = entry
        if not required:
            report["optional_files"].append({"file": name, "status": entry["status"]})

    sales_entry = report["files"].get("sales", {})
    if sales_entry.get("status") == "ok":
        sales = pd.read_csv(raw / files.sales)
        sales = sales.loc[:, ~sales.columns.astype(str).str.startswith("Unnamed")]
        release = pd.to_datetime(sales.get("release_date"), errors="coerce")
        week_columns = [c for c in sales.columns if str(c).strip().isdigit()]
        week_columns = sorted(week_columns, key=lambda c: int(str(c).strip()))
        renamed = sales[week_columns].rename(
            columns={c: f"q{int(str(c).strip()) + 1}" for c in week_columns})
        negatives, by_week = count_negatives(renamed, len(week_columns))
        n_items = int(pd.to_numeric(sales["external_code"], errors="coerce").nunique())
        n_stores = int(pd.to_numeric(sales["retail"], errors="coerce").nunique())
        report.update({
            "n_rows": int(len(sales)),
            "n_items": n_items,
            "n_stores": n_stores,
            "pair_coverage": round(len(sales) / max(n_items * n_stores, 1), 4),
            "negative_cells": int(negatives),
            "negative_cells_by_week": by_week,
            "date_min": str(release.min().date()) if release.notna().any() else None,
            "date_max": str(release.max().date()) if release.notna().any() else None,
            "monday_violations": monday_violations(release),
            "negative_reference": int(cfg.preflight.negative_cell_reference),
        })

    preset_files = cfg.split.preset_files or ()
    preset_present = all((raw / name).exists() for name in preset_files) if preset_files else False
    report["conditional"] = {
        "b27_real_data": ("not evaluated - precondition absent"
                          if report["data_class"]["data_class"] != "real"
                          else "ready"),
        "b29_secondary": ("ready" if preset_present else
                          "not evaluated - precondition absent"),
        "preset_files_present": bool(preset_present),
    }
    report["readiness"] = ("ready" if not report["problems"]
                           else f"blocked: {report['problems'][0]}")
    return report


def preflight_exit_code(report: dict) -> int:
    if report["problems"]:
        return 1
    if report["data_class"]["data_class"] == "absent":
        return 1
    return 0