"""A1-A6 register: the assumption, its status and where the code enforces it.

Chapters 5 and 6 can point here instead of restating §7.6.2. A1-A4 were signed
against the cleaned data by Rue on 2026-11-21; A5 is a stated limitation and A6
declares scenario parameters that the dataset does not contain, so those two
remain team decisions rather than data facts.
"""
from __future__ import annotations

from dataclasses import dataclass

SIGNED_BY = "Rue"
SIGNED_ON = "2026-11-21"


@dataclass(frozen=True)
class Assumption:
    id: str
    statement: str
    status: str
    enforced_in: str

    def as_dict(self) -> dict:
        return {"id": self.id, "statement": self.statement,
                "status": self.status, "enforced_in": self.enforced_in,
                "signed_by": SIGNED_BY if self.status == "encoded" else None,
                "signed_on": SIGNED_ON if self.status == "encoded" else None}


ASSUMPTIONS = (
    Assumption("A1",
               "stock at week t = opening + cumulative restocks - cumulative "
               "sales, floored at zero",
               "encoded",
               "src/preprocess/inventory.py"),
    Assumption("A2",
               "opening stock = units restocked up to and including the release "
               "week, times alpha (0.5 / 1.0 / 1.5, main 1.0)",
               "encoded",
               "src/preprocess/inventory.py"),
    Assumption("A3",
               "when the unfloored trajectory would go negative, add the smallest "
               "extra opening stock that keeps it at or above zero; recomputed for "
               "every alpha",
               "encoded",
               "src/preprocess/inventory.py"),
    Assumption("A4",
               "a pair with no restock on or before the release week starts at "
               "zero until its first observed restock",
               "encoded",
               "src/preprocess/inventory.py"),
    Assumption("A5",
               "no returns, transfers or hidden stock; sales stand in for demand",
               "stated_limitation",
               "src/replenish/risk.py"),
    Assumption("A6",
               "lead time, holding cost and shortage cost are scenario parameters; "
               "no order dates, lead times or costs exist in the data",
               "scenario_parameter",
               "src/replenish/policy.py"),
)

IDS = tuple(assumption.id for assumption in ASSUMPTIONS)


def by_id(assumption_id: str) -> Assumption:
    for assumption in ASSUMPTIONS:
        if assumption.id == assumption_id:
            return assumption
    raise KeyError(assumption_id)


def as_dicts() -> list:
    return [assumption.as_dict() for assumption in ASSUMPTIONS]


def encoded_ids() -> tuple:
    return tuple(a.id for a in ASSUMPTIONS if a.status == "encoded")


def open_team_decisions() -> tuple:
    return tuple(a.id for a in ASSUMPTIONS if a.status != "encoded")