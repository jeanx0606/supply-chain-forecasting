"""Inventory trajectory reconstruction; I0 is assumed, the trajectory is not (C3).

Assumptions A1-A4 (Chapter 7 §7.6.2, signed by Rue):
A1  stock_t = opening + cumulative restocks - cumulative sales, floored at 0.
A2  opening = units restocked up to and including the release week, times alpha.
A3  if the unfloored trajectory would go negative, add the smallest extra opening
    stock that keeps it >= 0 (recomputed for every alpha).
A4  a pair with no restock on or before the release week starts at zero until the
    first observed restock, which A2 already produces.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..errors import ConfigError
from ..seams.protocols import SeamDescription


@dataclass(frozen=True)
class InitialStockEstimate:
    value: float
    strategy_name: str
    assumption_note: str
    rep_id: str = "REP-07"
    base_value: float | None = None
    correction: float = 0.0


@dataclass(frozen=True)
class WeeklyInventoryEntry:
    week_no: int
    restock: float
    sales: float
    reconstructed_stock: float


@dataclass(frozen=True)
class InventoryTrajectory:
    entries: tuple
    initial_stock: InitialStockEstimate
    sellout_week: int | None
    remaining_weeks_after_sellout: int | None
    key: tuple | None = None
    reconstruction_marker: str = field(default="reconstructed", init=False)
    opening_stock_correction: float = 0.0
    min_raw_stock: float = 0.0
    negative_weeks: int = 0
    would_go_negative: bool = False
    policy: dict = field(default_factory=dict)

    def at(self, week_no: int) -> WeeklyInventoryEntry:
        for entry in self.entries:
            if entry.week_no == int(week_no):
                return entry
        raise KeyError(f"week {week_no} outside the reconstructed horizon")

    def as_rows(self) -> list:
        return [
            {"week_no": e.week_no, "restock": e.restock, "sales": e.sales,
             "reconstructed_stock": e.reconstructed_stock}
            for e in self.entries
        ]


@dataclass(frozen=True)
class ReconstructionPolicy:
    """A1/A3 switches plus the A2 alpha; every one of them is config-driven."""

    alpha: float = 1.0
    negative_correction: bool = True
    floor_at_zero: bool = True

    def as_dict(self) -> dict:
        return {"alpha": float(self.alpha),
                "a1_floor_at_zero": bool(self.floor_at_zero),
                "a3_negative_correction": bool(self.negative_correction)}


LEGACY_POLICY = ReconstructionPolicy(alpha=1.0, negative_correction=False,
                                     floor_at_zero=False)


def _q(row, n: int) -> float:
    name = f"q{n}"
    try:
        value = row.get(name, 0.0)
    except AttributeError:
        values = row._asdict() if hasattr(row, "_asdict") else {}
        value = values.get(name, 0.0)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if value != value:
        return 0.0
    return value


def _walk(initial: float, events: dict, sales_row, n_weeks: int) -> list:
    stock = float(initial)
    path = []
    for n in range(1, n_weeks + 1):
        stock = stock + float(events.get(n, 0.0)) - _q(sales_row, n)
        path.append(stock)
    return path


class InitialStockStrategy:
    SEAM_ID = "SEAM-INITIAL-STOCK"
    name = "base"

    def describe(self) -> SeamDescription:
        return SeamDescription("initial_stock", "REP-07", self.name)

    def estimate(self, sales_row, events, opening: float = 0.0) -> InitialStockEstimate:
        raise NotImplementedError


class RestockedToReleaseStrategy(InitialStockStrategy):
    """A2: units restocked up to and including the release week, times alpha."""

    name = "restocked_to_release"

    def __init__(self, alpha: float = 1.0):
        value = float(alpha)
        if value <= 0 or value != value:
            raise ConfigError("inventory.opening_stock_alpha", "must be > 0",
                              observed=alpha, expected="a positive number")
        self.alpha = value

    def estimate(self, sales_row, events, opening: float = 0.0) -> InitialStockEstimate:
        base = float(opening)
        value = round(base * self.alpha, 6)
        if base <= 0:
            note = ("no restock on or before the release week; per A4 the pair starts "
                    f"at zero until its first observed restock (alpha {self.alpha:g})")
        else:
            note = (f"I0 = {base:g} units restocked up to and including the release "
                    f"week x alpha {self.alpha:g} (A2)")
        return InitialStockEstimate(value, self.name, note,
                                    base_value=base, correction=0.0)

    def __call__(self, sales_row, events, opening: float = 0.0):
        return self.estimate(sales_row, events, opening).value


class LeadTimeDemandStrategy(InitialStockStrategy):
    """Legacy sales-based estimate; kept selectable, superseded by A2."""

    name = "lead_time_demand"

    def __init__(self, lead_time_weeks: int = 1, safety_factor: float = 1.0):
        self.lead_time_weeks = max(1, int(lead_time_weeks))
        self.safety_factor = float(safety_factor)

    def estimate(self, sales_row, events, opening: float = 0.0) -> InitialStockEstimate:
        base = sum(_q(sales_row, n) for n in range(1, self.lead_time_weeks + 1))
        value = round(base * self.safety_factor, 6)
        return InitialStockEstimate(
            value=value, strategy_name=self.name,
            assumption_note=(f"I0 assumed as the first {self.lead_time_weeks} week(s) "
                             f"of demand times safety factor {self.safety_factor}; "
                             "absolute level uncertain; superseded by A2"),
            base_value=base)

    def __call__(self, sales_row, events, opening: float = 0.0):
        return self.estimate(sales_row, events, opening).value


class FirstRestockStrategy(InitialStockStrategy):
    name = "first_restock"

    def estimate(self, sales_row, events, opening: float = 0.0) -> InitialStockEstimate:
        if not events:
            return InitialStockEstimate(
                0.0, self.name,
                "initial stock assumed equal to the first restock; no restock event exists")
        first_week = min(events)
        return InitialStockEstimate(
            float(events[first_week]), self.name,
            "initial stock assumed equal to the first restock shipment")

    def __call__(self, sales_row, events, opening: float = 0.0):
        return self.estimate(sales_row, events, opening).value


class RollingMinimumStrategy(InitialStockStrategy):
    name = "rolling_minimum"

    def estimate(self, sales_row, events, opening: float = 0.0) -> InitialStockEstimate:
        values = [_q(sales_row, n) for n in range(1, 13)]
        nonzero = [value for value in values if value > 0]
        base = (min(nonzero) / 2.0) if nonzero else 0.0
        return InitialStockEstimate(
            float(base), self.name,
            "initial stock assumed as half the smallest non-zero weekly sale")

    def __call__(self, sales_row, events, opening: float = 0.0):
        return self.estimate(sales_row, events, opening).value


VALID_METHODS = ("restocked_to_release", "lead_time_demand", "first_restock",
                 "rolling_minimum")


def make_initial_stock_strategy(method, lead_time_weeks: int = 1,
                                safety_factor: float = 1.0,
                                alpha: float = 1.0) -> InitialStockStrategy:
    name = getattr(method, "value", method)
    if name == "restocked_to_release":
        return RestockedToReleaseStrategy(alpha)
    if name == "lead_time_demand":
        return LeadTimeDemandStrategy(lead_time_weeks, safety_factor)
    if name == "first_restock":
        return FirstRestockStrategy()
    if name == "rolling_minimum":
        return RollingMinimumStrategy()
    raise ConfigError("inventory.initial_stock_method", "unknown strategy",
                      observed=name, expected=list(VALID_METHODS))


def build_trajectory(sales_row, events: dict, strategy: InitialStockStrategy,
                     n_weeks: int = 12, key=None, opening: float = 0.0,
                     policy: ReconstructionPolicy | None = None) -> InventoryTrajectory:
    policy = policy or LEGACY_POLICY

    raw_estimate = strategy.estimate(sales_row, events, opening)
    base_initial = float(raw_estimate.value)
    raw_path = _walk(base_initial, events, sales_row, n_weeks)
    min_raw = min([base_initial] + raw_path)
    deficit = max(0.0, -min_raw)
    correction = deficit if policy.negative_correction else 0.0
    initial = base_initial + correction

    estimate = InitialStockEstimate(
        value=round(initial, 6),
        strategy_name=raw_estimate.strategy_name,
        assumption_note=_note_with_policy(raw_estimate.assumption_note, policy,
                                          correction),
        rep_id=raw_estimate.rep_id,
        base_value=raw_estimate.base_value,
        correction=round(correction, 6))

    path = _walk(initial, events, sales_row, n_weeks)
    entries = []
    sellout_week = None
    for n, (stock, raw) in enumerate(zip(path, raw_path), start=1):
        published = stock
        if policy.floor_at_zero and published < 0:
            published = 0.0
        entries.append(WeeklyInventoryEntry(n, float(events.get(n, 0.0)),
                                            _q(sales_row, n), round(published, 6)))
        if sellout_week is None and stock <= 0:
            sellout_week = n
    remaining = (n_weeks - sellout_week) if sellout_week is not None else None
    return InventoryTrajectory(
        entries=tuple(entries), initial_stock=estimate,
        sellout_week=sellout_week, remaining_weeks_after_sellout=remaining,
        key=key, opening_stock_correction=round(correction, 6),
        min_raw_stock=round(min_raw, 6),
        negative_weeks=int(sum(1 for value in raw_path if value < 0)),
        would_go_negative=bool(min_raw < 0), policy=policy.as_dict())


def _note_with_policy(note: str, policy: ReconstructionPolicy,
                      correction: float) -> str:
    parts = [note]
    if correction > 0:
        parts.append(f"A3 added {correction:g} units of opening stock to keep the "
                     "reconstruction at or above zero")
    elif policy.negative_correction:
        parts.append("A3 found no negative week, so no extra opening stock was added")
    else:
        parts.append("A3 disabled: the unfloored trajectory may be negative")
    parts.append(f"A1 floor-at-zero {'on' if policy.floor_at_zero else 'off'}")
    return "; ".join(parts)


@dataclass(frozen=True)
class ReconstructionAudit:
    pairs: int
    opening_zero_pairs: int
    a4_pairs: int
    zero_or_below_pairs: int
    negative_pairs: int
    negative_pairs_excluding_a4: int
    a4_pairs_requiring_a3: int
    correction_total: float
    min_raw_stock: float

    def as_dict(self) -> dict:
        return {"pairs": int(self.pairs),
                "a4_pairs": int(self.a4_pairs),
                "a4_opening_zero_pairs": int(self.opening_zero_pairs),
                "a4_pairs_requiring_a3": int(self.a4_pairs_requiring_a3),
                "zero_or_below_pairs": int(self.zero_or_below_pairs),
                "negative_pairs": int(self.negative_pairs),
                "negative_pairs_including_a4": int(self.negative_pairs),
                "negative_pairs_excluding_a4": int(self.negative_pairs_excluding_a4),
                "a3_correction_total": float(self.correction_total),
                "min_raw_stock": float(self.min_raw_stock)}


def audit_reconstruction(frame, events_by_key: dict, opening_by_key: dict,
                         strategy: InitialStockStrategy, n_weeks: int = 12,
                         audit_weeks: int = 11,
                         first_offset_by_key: dict | None = None) -> ReconstructionAudit:
    """Count stock-out and negative pairs before any A3 correction (Ch. 5/7 figures)."""
    first_offset_by_key = first_offset_by_key or {}
    pairs = 0
    opening_zero = 0
    a4 = 0
    zero_or_below = 0
    negative = 0
    negative_no_a4 = 0
    a4_requiring = 0
    correction_total = 0.0
    worst = 0.0
    for row in frame.itertuples(index=False):
        key = (int(row.external_code), int(row.retail))
        opening = float(opening_by_key.get(key, 0.0))
        estimate = strategy.estimate(row, events_by_key.get(key, {}), opening)
        path = _walk(float(estimate.value), events_by_key.get(key, {}), row, n_weeks)
        window = path[: audit_weeks + 1]
        pairs += 1
        is_a4 = first_offset_by_key.get(key, 0) > 0
        if opening <= 0:
            opening_zero += 1
        if is_a4:
            a4 += 1
        if any(value <= 0 for value in window):
            zero_or_below += 1
        if any(value < 0 for value in window):
            negative += 1
            if is_a4:
                a4_requiring += 1
            else:
                negative_no_a4 += 1
        correction_total += max(0.0, -min(window))
        worst = min(worst, min(window))
    return ReconstructionAudit(pairs, opening_zero, a4, zero_or_below, negative,
                               negative_no_a4, a4_requiring,
                               round(correction_total, 6), round(worst, 6))


def initial_stock_value(strategy: InitialStockStrategy, sales_row, events,
                        opening: float = 0.0) -> float:
    return float(strategy.estimate(sales_row, events, opening).value)