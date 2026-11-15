"""Grain adapter: item-store is the only registered grain (D7, REP-12)."""
from __future__ import annotations

import pandas as pd

from ..errors import ConfigError, StageError
from ..seams.protocols import SeamDescription

KEY_COLUMNS = ("external_code", "retail")


class ItemStoreGrain:
    SEAM_ID = "SEAM-GRAIN"

    def describe(self) -> SeamDescription:
        return SeamDescription("grain", "REP-12", "item_store")

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in KEY_COLUMNS if c not in frame.columns]
        if missing:
            raise StageError("spine", f"missing key columns: {missing}")
        duplicates = frame.duplicated(subset=list(KEY_COLUMNS), keep=False)
        if bool(duplicates.any()):
            raise StageError(
                "spine", f"duplicate item-store keys survived cleaning: "
                         f"{int(duplicates.sum())} rows")
        return frame.sort_values(["release_date", *KEY_COLUMNS], kind="stable").reset_index(drop=True)

    def supported(self) -> tuple[str, ...]:
        return ("item_store",)


def make_grain(name: str) -> ItemStoreGrain:
    if name != "item_store":
        raise ConfigError("spine.granularity", "unsupported grain", observed=name,
                          expected=ItemStoreGrain().supported())
    return ItemStoreGrain()