"""The alternative 4-in/1-out rolling task, reachable from one switch (D8, REP-16)."""
from __future__ import annotations

import pandas as pd

from ..seams.protocols import SeamDescription
from .build import TaskDataset, build_task


ALT_INPUT_WEEKS = 4
ALT_OUTPUT_WEEKS = 1


class AltTaskSpec:
    SEAM_ID = "SEAM-ALT-TASK"

    def __init__(self, enabled: bool = True, input_weeks: int = ALT_INPUT_WEEKS,
                 output_weeks: int = ALT_OUTPUT_WEEKS):
        self.enabled = bool(enabled)
        self.input_weeks = int(input_weeks)
        self.output_weeks = int(output_weeks)

    def describe(self) -> SeamDescription:
        return SeamDescription("alt_task", "REP-16", str(self.enabled).lower())

    def build(self, frame: pd.DataFrame, n_weeks: int = 12, spec=None) -> tuple:
        if not self.enabled:
            return ()
        return tuple(build_alt_task(frame, self.input_weeks, self.output_weeks,
                                    n_weeks, spec=spec))


def build_alt_task(frame: pd.DataFrame, input_weeks: int = 4,
                   output_weeks: int = 1, n_weeks: int = 12, spec=None) -> list:
    points = []
    for origin in range(input_weeks, n_weeks - output_weeks + 1):
        points.append(build_task(frame, input_weeks, output_weeks, n_weeks,
                                 origin_week=origin, spec=spec))
    return points


def alt_task_summary(points: tuple | list) -> list:
    return [{"origin_week": p.origin_week, "labels": p.labels,
             "n_rows": len(p.features)} for p in points]