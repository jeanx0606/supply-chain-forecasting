"""Candidate selection on the validation partition only (D24, REP-18).

The public ``choose`` function takes numbers, never arrays: a test partition value
cannot reach the decision by construction (5.9.1 r13).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..errors import StageError
from ..seams.protocols import SeamDescription

REFERENCE_MODEL = "seasonal_naive"
RULE_ID = "min_primary_metric"


@dataclass(frozen=True)
class CandidateFigure:
    candidate: str
    value: float | None
    n_records: int
    available: bool = True
    reason: str = ""

    def as_dict(self) -> dict:
        return {"candidate": self.candidate, "value": self.value,
                "n_records": self.n_records, "available": self.available,
                "reason": self.reason}


@dataclass(frozen=True)
class Qualification:
    mode: str
    tolerance: float
    reference: str
    reference_value: float | None

    @property
    def passes(self) -> bool:
        return self.reference_value is not None


@dataclass(frozen=True)
class SelectionDecision:
    rule_id: str
    chosen_model: str
    outcome: str
    figures: tuple
    reference: CandidateFigure
    tie: tuple = ()
    on_none: str = "retained_default"
    note: str = ""
    fit_span: dict = field(default_factory=dict)
    provisional: tuple = ()

    def as_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "chosen_model": self.chosen_model,
            "outcome": self.outcome,
            "figures": [f.as_dict() for f in self.figures],
            "reference": self.reference.as_dict(),
            "tie": list(self.tie),
            "on_none": self.on_none,
            "note": self.note,
            "fit_span": dict(self.fit_span),
            "partition": "validation (selection)",
        }


def qualifies(value: float | None, reference: float | None, mode: str,
              tolerance: float = 0.0) -> bool:
    if value is None or reference is None:
        return False
    if mode == "strictly_better":
        return value < reference - tolerance
    if mode == "must_not_be_worse_than_naive":
        return value <= reference + tolerance
    raise StageError("selection", f"unknown qualification mode '{mode}'")


def choose(figures, rule: str = RULE_ID, tolerance: float = 0.0,
           reference_name: str = REFERENCE_MODEL, mode: str = "must_not_be_worse_than_naive",
           on_none: str = "retained_default", default_model: str = "gradient_boosting",
           candidates=None) -> SelectionDecision:
    """Choose from numbers only. No X, y, mask or array may enter this function."""
    if rule != RULE_ID:
        raise StageError("selection", f"unknown selection rule '{rule}'")
    items = list(figures)
    by_name = {f.candidate: f for f in items}
    reference = by_name.get(reference_name)
    if reference is None or not reference.available:
        raise StageError("selection",
                         f"qualification reference '{reference_name}' has no figure")

    available = [f for f in items if f.available and f.value is not None]
    if not available:
        raise StageError("selection", "no candidate produced a validation figure")

    best_value = min(f.value for f in available)
    tied = sorted([f.candidate for f in available if f.value == best_value])
    order = list(candidates or [f.candidate for f in items])
    tied.sort(key=lambda name: order.index(name) if name in order else len(order))
    best = next(f for f in available if f.candidate == tied[0])

    qualified = qualifies(best.value, reference.value, mode, tolerance)
    if qualified:
        chosen = best.candidate
        outcome = "qualified"
        note = ""
    else:
        outcome = "no candidate qualified"
        if on_none == "fail_run":
            raise StageError(
                "selection",
                f"no candidate qualified against {reference_name} "
                f"(best {best.value} vs {reference.value})")
        if on_none == "least_bad_provisional":
            others = [f for f in available if f.candidate != reference_name]
            chosen = (min(others, key=lambda f: f.value).candidate
                      if others else default_model)
            note = "least-bad non-reference candidate promoted provisionally"
        else:
            chosen = default_model
            note = ("no candidate qualified; the configured default is retained and "
                    "nothing is promoted")

    return SelectionDecision(
        rule_id=RULE_ID, chosen_model=chosen, outcome=outcome,
        figures=tuple(items), reference=reference, tie=tuple(tied) if len(tied) > 1 else (),
        on_none=on_none, note=note,
    )


class ModelSelectionPolicy:
    """Resolved selection seam: numbers in, decision out."""

    SEAM_ID = "SEAM-MODEL-SELECTION"

    def __init__(self, rule: str = RULE_ID, metric: str = "wape_item_level",
                 tolerance: float = 0.0, mode: str = "must_not_be_worse_than_naive",
                 on_none: str = "retained_default"):
        self.rule = rule
        self.metric = metric
        self.tolerance = float(tolerance)
        self.mode = mode
        self.on_none = on_none

    def describe(self) -> SeamDescription:
        return SeamDescription("model_selection", "REP-18", self.rule)

    def choose(self, figures, candidates, default_model) -> SelectionDecision:
        return choose(figures, self.rule, self.tolerance, REFERENCE_MODEL,
                      self.mode, self.on_none, default_model, candidates)

    def score_validation(self, predictions, y_val, meta_val, metric_seam, model):
        record = metric_seam.evaluate(
            y_val, predictions, meta_val, model, partition="validation (selection)")
        return CandidateFigure(candidate=model, value=getattr(record, self.metric),
                               n_records=int(len(y_val)))


class SelectionGuard:
    """Refuses a test read before the decision is sealed, and a second read."""

    def __init__(self):
        self.sealed = False
        self.reads = []

    def seal(self):
        self.sealed = True

    def evaluate_test(self, model):
        if not self.sealed:
            raise StageError("selection",
                             "test partition read before the selection was sealed")
        if model in self.reads:
            raise StageError("selection", f"test partition read twice for '{model}'")
        self.reads.append(model)
        return model