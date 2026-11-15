"""Chronological partitioning plus an explicit statement about the preset partition."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..errors import ConfigError, StageError
from ..seams.protocols import SeamDescription


@dataclass(frozen=True)
class SplitAssignment:
    train: pd.Series
    validation: pd.Series
    test: pd.Series
    cut_train_end: str
    cut_validation_end: str
    method: str
    counts: dict


@dataclass(frozen=True)
class PresetProvenance:
    present: bool
    used: bool
    statement: str
    applied_scale_factor: float | None
    factor_state: str

    def as_dict(self) -> dict:
        return {"present": self.present, "used": self.used,
                "statement": self.statement,
                "applied_scale_factor": self.applied_scale_factor,
                "factor_state": self.factor_state}


class ChronologicalSplit:
    SEAM_ID = "SEAM-SPLIT-CUT"

    def __init__(self, train_ratio: float = 0.70, val_ratio: float = 0.15,
                 test_ratio: float = 0.15, explicit_cut_dates=None,
                 policy="chronological"):
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0, atol=0.001):
            raise ConfigError(
                "split.train_ratio", "ratios must sum to 1.0 +/- 0.001",
                observed=round(total, 6), expected="1.0 +/- 0.001")
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.explicit_cut_dates = explicit_cut_dates
        self.policy = policy

    def describe(self) -> SeamDescription:
        return SeamDescription("split_cut", "REP-01", str(self.policy))

    def resolve(self, release_dates: pd.Series) -> SplitAssignment:
        dates = pd.to_datetime(release_dates)
        uniq = np.sort(dates.dropna().unique())
        if len(uniq) < 3:
            raise StageError("split", "need at least 3 distinct release dates",
                             next_command="python -m codes.sample_data --overwrite")

        if self.explicit_cut_dates:
            cut_train = pd.Timestamp(self.explicit_cut_dates[0])
            cut_val = pd.Timestamp(self.explicit_cut_dates[1])
        else:
            i_train = int(np.floor(len(uniq) * self.train_ratio)) - 1
            i_val = int(np.floor(len(uniq) * (self.train_ratio + self.val_ratio))) - 1
            i_train = max(0, min(i_train, len(uniq) - 3))
            i_val = max(i_train + 1, min(i_val, len(uniq) - 2))
            cut_train = pd.Timestamp(uniq[i_train])
            cut_val = pd.Timestamp(uniq[i_val])

        train = (dates <= cut_train).fillna(False).astype(bool)
        validation = ((dates > cut_train) & (dates <= cut_val)).fillna(False).astype(bool)
        test = (dates > cut_val).fillna(False).astype(bool)

        overlap = (train.astype(int) + validation.astype(int) + test.astype(int))
        if (overlap > 1).any():
            raise StageError("split", "partitions overlap")
        if not test.any():
            raise StageError("split", "test set is empty - adjust split ratios")
        if not validation.any():
            raise StageError("split", "validation set is empty - adjust split ratios")

        return SplitAssignment(
            train=train,
            validation=validation,
            test=test,
            cut_train_end=str(pd.Timestamp(cut_train).date()),
            cut_validation_end=str(pd.Timestamp(cut_val).date()),
            method=str(self.policy),
            counts={"train": int(train.sum()), "validation": int(validation.sum()),
                    "test": int(test.sum())},
        )


class PresetSplitPolicy:
    """Benchmark partition: optional, secondary, never a substitute (REP-05)."""

    SEAM_ID = "SEAM-PRESET-FILES"

    def __init__(self, enabled: bool = False, files=None, scale_factor=None):
        self.enabled = bool(enabled)
        self.files = tuple(files) if files else None
        self.scale_factor = None if scale_factor is None else float(scale_factor)

    def describe(self) -> SeamDescription:
        implementation = "secondary_only" if self.enabled else "disabled"
        return SeamDescription("preset_files", "REP-05", implementation)

    def available(self) -> bool:
        return bool(self.enabled and self.files)

    def require_scale_factor(self) -> float:
        if self.scale_factor is None:
            raise ConfigError(
                "split.preset_scale_factor",
                "a scaled read requires an explicitly configured factor (REP-05)",
                observed=None,
                expected="a positive number declared under REP-05")
        return float(self.scale_factor)

    def scale_factor_state(self) -> dict:
        if self.scale_factor is None:
            return {"state": "absent", "value": None,
                    "parameter": "split.preset_scale_factor"}
        return {"state": "applied" if self.enabled else "inactive_parameter",
                "value": self.scale_factor,
                "parameter": "split.preset_scale_factor"}

    def require(self) -> None:
        if self.enabled:
            self.require_scale_factor()

    def provenance(self, present: bool) -> PresetProvenance:
        state = self.scale_factor_state()["state"]
        if not present:
            statement = ("preset split files are not present; the chronological "
                         "split is the only partition")
            return PresetProvenance(False, False, statement, self.scale_factor, state)
        if not self.enabled:
            statement = ("preset split files are present but deliberately not used; "
                         "the main experiment uses the chronological split")
            return PresetProvenance(True, False, statement, self.scale_factor, state)
        statement = ("preset split is used only as a labelled secondary comparison "
                     "in raw units")
        return PresetProvenance(True, True, statement, self.scale_factor, state)