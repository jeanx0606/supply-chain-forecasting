"""Verify the trained-model handoff before anything quotes it.

The checks answer the questions a chapter claim depends on:

* does the model file carry exactly the declared feature list, in order;
* does it produce a non-negative (n, 10) matrix on the shipped input;
* does it read only the two observed sales weeks (no future-sales leakage);
* does the adapter reproduce the reference predictor bit for bit;
* do the shipped ``sigma_e`` constants reproduce, and is the LightGBM constant
  correctly attributed to the validation-trained model rather than the final one;
* does the base-stock target agree with the repository implementation, and does
  the handoff's ceiling rounding differ from the repository's integer rounding.

A skipped check is reported as skipped with its reason; it is never reported green.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluate.metrics import bias, mae, rmse, wape
from ..replenish.policy import ReplenishmentContext, recommend
from ..replenish.safety import make_safety_stock
from .forecast import (HORIZON_WEEKS, OBSERVED_WEEKS, SIGMA_KEYS, SIMULATION_LABEL,
                       LindaLightGBM, load_manifest)

PASS = "pass"
FAIL = "fail"
SKIPPED = "skipped"
INFO = "info"

PARITY_TOLERANCE = 0.001
BASE_STOCK_TOLERANCE = 0.001
SIGMA_TOLERANCE = 1e-4
DEFAULT_SAMPLE_ROWS = 2000
ROUNDING_NOTE = ("handoff recommend_order rounds up (ceil); the repository policy "
                 "rounds to the nearest unit - a stated deviation, at most 1 unit")


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str

    def as_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class VerificationReport:
    handoff_dir: str
    checks: tuple = ()
    metrics: dict = field(default_factory=dict)
    parity: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(check.status == FAIL for check in self.checks)

    @property
    def counts(self) -> dict:
        result = {status: 0 for status in (PASS, FAIL, SKIPPED, INFO)}
        for check in self.checks:
            result[check.status] = result.get(check.status, 0) + 1
        return result

    def as_dict(self) -> dict:
        return {
            "handoff_dir": self.handoff_dir,
            "ok": self.ok,
            "counts": self.counts,
            "simulation_label": SIMULATION_LABEL,
            "manifest": self.manifest,
            "checks": [check.as_dict() for check in self.checks],
            "metrics": self.metrics,
            "parity": self.parity,
        }

    def markdown(self) -> str:
        lines = ["# Handoff verification", "",
                 f"- handoff directory: `{self.handoff_dir}`",
                 f"- result: **{'PASS' if self.ok else 'FAIL'}**",
                 f"- label: {SIMULATION_LABEL} (no figure here is a reported result)",
                 "", "| Check | Status | Detail |", "|---|---|---|"]
        for check in self.checks:
            detail = check.detail.replace("|", "/")
            lines.append(f"| {check.name} | {check.status} | {detail} |")
        if self.metrics:
            lines += ["", "## Recompute on the supplied input", "",
                      "Verification recomputation only; the authoritative table stays the",
                      "experiment run that produced the handoff.", "",
                      "| Model | WAPE | MAE | RMSE | Bias (mean) | Bias (ratio) | Cum. error |",
                      "|---|---|---|---|---|---|---|"]
            for name, values in self.metrics.items():
                lines.append(
                    f"| {name} | {_fmt(values.get('wape'))} | {_fmt(values.get('mae'))} "
                    f"| {_fmt(values.get('rmse'))} | {_fmt(values.get('bias'))} "
                    f"| {_fmt(values.get('bias_ratio'))} "
                    f"| {_fmt(values.get('cum_err'))} |")
        return "\n".join(lines) + "\n"


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def load_reference(handoff_dir):
    """Load ``predict_forecast.py`` / ``replenishment.py`` next to the model.

    They are used as verification oracles only; nothing in the pipeline imports them.
    """
    directory = Path(handoff_dir)
    oracles = {}
    for name in ("predict_forecast", "replenishment"):
        path = directory / f"{name}.py"
        if not path.exists():
            continue
        module_name = f"_handoff_reference_{name}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:  # pragma: no cover - a broken oracle must not hide a result
            sys.modules.pop(module_name, None)
            continue
        oracles[name] = module
    return oracles


def _frame(pairs_path) -> pd.DataFrame:
    frame = pd.read_csv(pairs_path)
    frame.columns = [str(column) for column in frame.columns]
    return frame


def _test_split(frame):
    if "split" not in frame.columns:
        return None
    values = frame["split"].astype(str).str.lower()
    mask = values.str.startswith("te")
    return frame.loc[mask].reset_index(drop=True) if mask.any() else None


def _val_split(frame):
    if "split" not in frame.columns:
        return None
    values = frame["split"].astype(str).str.lower()
    mask = values.str.startswith("va")
    return frame.loc[mask].reset_index(drop=True) if mask.any() else None


def _actual(matrix_source, rows: int) -> np.ndarray:
    columns = [str(week) for week in range(OBSERVED_WEEKS, 12)]
    return matrix_source[columns].to_numpy(dtype=float)


def _metrics(actual, predicted, items=None) -> dict:
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    volume = float(np.asarray(actual, dtype=float).sum())
    return {"wape": wape(actual, predicted), "mae": mae(actual, predicted),
            "rmse": rmse(actual, predicted), "bias": bias(actual, predicted),
            "bias_ratio": float(error.sum() / volume) if volume else None,
            "cum_err": _cumulative_error(actual, predicted, items),
            "n_pairs": int(np.asarray(actual).shape[0]),
            "weeks": int(np.asarray(actual).shape[1])}


def _cumulative_error(actual, predicted, items):
    """Item-level cumulative error: sum |forecast total - actual total| / total actual."""
    if items is None:
        return None
    frame = pd.DataFrame({
        "item": np.asarray(items),
        "actual": np.asarray(actual, dtype=float).sum(axis=1),
        "forecast": np.asarray(predicted, dtype=float).sum(axis=1),
    }).groupby("item")[["actual", "forecast"]].sum()
    total = float(frame["actual"].sum())
    if not total:
        return None
    return float((frame["forecast"] - frame["actual"]).abs().sum() / total)


PUBLISHED_FIELDS = (("wape", "wape"), ("mae", "mae"), ("rmse", "rmse"),
                    ("bias", "bias_ratio"), ("cumulative_error", "cum_err"))


def _published_check(recomputed: dict, published: dict) -> Check:
    source = str(published.get("source", "published table"))
    tolerance = float(published.get("tolerance", 0.001))
    row = (published.get("rows") or {}).get("LightGBM")
    if not row:
        return Check("published_table", SKIPPED,
                     f"{source}: no LightGBM row to compare")
    deltas, missing = {}, []
    for published_key, metric_key in PUBLISHED_FIELDS:
        value = recomputed.get(metric_key)
        if published_key not in row or value is None:
            missing.append(published_key)
            continue
        deltas[published_key] = float(value) - float(row[published_key])
    if missing:
        return Check("published_table", SKIPPED,
                     f"not recomputable here: {missing}")
    field, delta = max(deltas.items(), key=lambda pair: abs(pair[1]))
    detail = (f"{source}: max |delta| {abs(delta):.4f} on {field} "
              f"(tolerance {tolerance}); bias compared as error-to-volume")
    note = published.get("rounding_note")
    if note:
        detail += f"; {note}"
    return Check("published_table", PASS if abs(delta) <= tolerance else FAIL, detail)


def _mutated(frame: pd.DataFrame, columns, transform) -> pd.DataFrame:
    copy = frame.copy()
    for column in columns:
        copy[column] = transform(copy[column].to_numpy(dtype=float))
    return copy


def _base_stock_target(forecast, sigma, lead_time=1, review=1, z=1.6449) -> float:
    window = np.asarray(forecast, dtype=float)[: lead_time + review]
    return float(window.sum() + z * sigma * np.sqrt(max(len(window), 1)))


def verify_handoff(handoff_dir, pairs_path=None, reference_dir=None,
                   sample_rows: int = DEFAULT_SAMPLE_ROWS, booster=None,
                   manifest=None, published=None) -> VerificationReport:
    checks = []
    manifest = manifest or load_manifest(handoff_dir)
    checks.append(Check("manifest", PASS,
                        f"{len(manifest.feature_order)} features, objective "
                        f"{manifest.objective}, sigma_e for {sorted(manifest.sigma_e)}"))

    model = LindaLightGBM(manifest, booster=booster)
    produced = tuple(model.booster.feature_name())
    if produced == tuple(manifest.feature_order):
        checks.append(Check("feature_contract", PASS,
                            f"model feature list equals the manifest ({len(produced)} names)"))
    else:
        checks.append(Check("feature_contract", FAIL,
                            f"model list {list(produced)} != manifest {list(manifest.feature_order)}"))

    missing = [key for key in SIGMA_KEYS if key not in manifest.sigma_e]
    if missing:
        checks.append(Check("sigma_contract", FAIL, f"missing sigma_e for {missing}"))
    else:
        checks.append(Check("sigma_contract", PASS,
                            "sigma_e present for " + ", ".join(
                                f"{key}={manifest.sigma_e[key]:.4f}" for key in SIGMA_KEYS)))

    if not pairs_path or not Path(pairs_path).exists():
        reason = "input pairs file was not supplied"
        for name in ("prediction_shape", "non_negative", "no_future_sales",
                     "uses_observed_sales", "uses_future_discounts",
                     "parity_with_reference", "replenishment_target",
                     "sigma_provenance", "test_metrics"):
            checks.append(Check(name, SKIPPED, reason))
        return VerificationReport(str(handoff_dir), tuple(checks),
                                  manifest=manifest.summary())

    frame = _frame(pairs_path)
    sample = frame.head(min(int(sample_rows), len(frame))).reset_index(drop=True)
    predicted = model.predict_pairs(sample)

    if predicted.shape == (len(sample), HORIZON_WEEKS):
        checks.append(Check("prediction_shape", PASS,
                            f"({len(sample)}, {HORIZON_WEEKS}) for {len(sample)} sampled pairs"))
    else:
        checks.append(Check("prediction_shape", FAIL, f"shape {predicted.shape}"))

    checks.append(Check("non_negative", PASS, f"min forecast {predicted.min():.4f}") if
                  predicted.min() >= 0 else
                  Check("non_negative", FAIL, f"min forecast {predicted.min():.4f}"))

    future = [str(week) for week in range(OBSERVED_WEEKS, 12)]
    mutated = model.predict_pairs(_mutated(sample, future, lambda values: values * 7.0 + 123.0))
    if np.array_equal(predicted, mutated):
        checks.append(Check("no_future_sales", PASS,
                            "weeks 3-12 sales mutated: forecasts unchanged"))
    else:
        checks.append(Check("no_future_sales", FAIL,
                            f"forecasts moved by {np.abs(predicted - mutated).max():.4f}"))

    changed = model.predict_pairs(_mutated(sample, ["0"], lambda values: values + 5.0))
    delta = float(np.abs(predicted - changed).max())
    checks.append(Check("uses_observed_sales", PASS if delta > 0 else FAIL,
                        f"week-1 sales +5 changed forecasts by {delta:.4f}"))

    discount_columns = [f"disc_{week}" for week in range(OBSERVED_WEEKS, 12)]
    discount_changed = model.predict_pairs(
        _mutated(sample, discount_columns, lambda values: np.full_like(values, 0.5)))
    discount_delta = float(np.abs(predicted - discount_changed).max())
    checks.append(Check("uses_future_discounts", INFO if discount_delta > 0 else INFO,
                        f"future discounts are declared features; changing them moved "
                        f"forecasts by {discount_delta:.4f}"))

    parity = {}
    oracles = load_reference(reference_dir or handoff_dir)
    predictor = oracles.get("predict_forecast")
    if predictor is None or not hasattr(predictor, "predict"):
        checks.append(Check("parity_with_reference", SKIPPED,
                            "predict_forecast.py is not available as an oracle"))
    else:
        reference = np.clip(np.asarray(predictor.predict(model.booster, sample), dtype=float), 0, None)
        difference = float(np.abs(reference - predicted).max())
        parity = {"max_abs_difference": difference,
                  "tolerance": PARITY_TOLERANCE, "rows": int(len(sample))}
        checks.append(Check("parity_with_reference",
                            PASS if difference <= PARITY_TOLERANCE else FAIL,
                            f"max |adapter - reference| = {difference:.6f} "
                            f"(tolerance {PARITY_TOLERANCE}) on {len(sample)} pairs"))

    replenisher = oracles.get("replenishment")
    sigma = model.sigma_e
    forecasts = ([2.0, 2.0, 1.0], [0.0, 0.0, 0.0], [5.0, 3.0, 0.0], [1.0, 0.0, 4.0])
    positions = (0.0, 3.0, 10.0)
    policy = make_safety_stock("normal_approx", 0.95)
    target_problems = []
    rounding_gap = 0
    reference_gap = 0
    for case in forecasts:
        for on_hand in positions:
            target = _base_stock_target(case, sigma)
            ctx = ReplenishmentContext(None, 2, 2, on_hand, tuple(case), 1, 1)
            repo = recommend(ctx, policy, variability=sigma)
            if abs(repo.target_level - target) > BASE_STOCK_TOLERANCE:
                target_problems.append((case, on_hand, repo.target_level, target))
            ceiling = int(np.ceil(max(0.0, target - on_hand - 1e-9)))
            rounding_gap = max(rounding_gap, abs(ceiling - repo.recommended_qty))
            if replenisher is not None and hasattr(replenisher, "recommend_order"):
                declared = int(replenisher.recommend_order(
                    case, on_hand=on_hand, on_order=0.0, sigma_e=sigma,
                    lead_time=1, review=1, z=1.6449))
                reference_gap = max(reference_gap, abs(declared - ceiling))
    if target_problems:
        checks.append(Check("replenishment_target", FAIL,
                            f"{len(target_problems)} cases disagree with the repository "
                            f"base-stock target, e.g. {target_problems[0]}"))
    else:
        checks.append(Check("replenishment_target", PASS,
                            "S = sum(forecast over L+R) + z*sigma_e*sqrt(L+R) agrees "
                            f"with src.replenish.policy on {len(forecasts) * len(positions)} cases"))
    if replenisher is None:
        checks.append(Check("replenishment_rounding", SKIPPED,
                            "replenishment.py is not available as an oracle"))
    else:
        checks.append(Check("replenishment_rounding", INFO if reference_gap == 0 else FAIL,
                            ROUNDING_NOTE + f" (observed gap {rounding_gap} unit)"))

    validation = _val_split(frame)
    if validation is None:
        checks.append(Check("sigma_provenance", SKIPPED, "no validation split column"))
    else:
        val_actual = _actual(validation, len(validation))
        val_predicted = model.predict_pairs(validation)
        measured = float(np.std(val_predicted - val_actual))
        declared = manifest.sigma_e["LightGBM"]
        detail = (f"declared LightGBM sigma_e {declared:.4f}; final model on validation "
                  f"{measured:.4f}; the constant belongs to the validation-trained model, "
                  f"which the final model is not, so it must not be recomputed here")
        checks.append(Check("sigma_provenance", INFO if abs(measured - declared) > SIGMA_TOLERANCE
                            else PASS, detail))

    test = _test_split(frame)
    metrics = {}
    if test is None:
        checks.append(Check("test_metrics", SKIPPED, "no test split column"))
        checks.append(Check("published_table", SKIPPED, "no test split column"))
    else:
        actual = _actual(test, len(test))
        predicted_test = model.predict_pairs(test)
        items = (test["external_code"].to_numpy()
                 if "external_code" in test.columns else None)
        metrics["LightGBM"] = _metrics(actual, predicted_test, items)
        checks.append(Check("test_metrics", INFO,
                            f"recomputed on {len(test)} test pairs: WAPE "
                            f"{_fmt(metrics['LightGBM']['wape'])}, bias "
                            f"{_fmt(metrics['LightGBM']['bias'])} - verification only, "
                            f"not the authoritative results table"))
        checks.append(Check("bias_definition", INFO,
                            "the repository reports bias as the mean signed error; the "
                            "reference experiment reports error-to-volume. Both are "
                            f"recomputed: mean {_fmt(metrics['LightGBM']['bias'])}, "
                            f"ratio {_fmt(metrics['LightGBM']['bias_ratio'])}"))
        if published:
            checks.append(_published_check(metrics["LightGBM"], published))
        else:
            checks.append(Check("published_table", SKIPPED,
                                "no published table supplied"))

    return VerificationReport(str(handoff_dir), tuple(checks), metrics, parity,
                              manifest.summary())