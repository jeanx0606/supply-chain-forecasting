"""Grain adapter: item-store only, other values rejected by name (D7, REP-12)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.errors import ConfigError, StageError
from src.preprocess.spine import ItemStoreGrain, make_grain


def frame():
    return pd.DataFrame({
        "external_code": [2, 1], "retail": [10, 10],
        "release_date": pd.to_datetime(["2019-06-03", "2019-01-07"]),
    })


def test_item_store_build_sorts_by_release_date():
    built = ItemStoreGrain().build(frame())
    assert list(built["external_code"]) == [1, 2]
    assert built.index.tolist() == [0, 1]


def test_duplicate_keys_rejected():
    duplicated = pd.concat([frame(), frame().head(1)], ignore_index=True)
    with pytest.raises(StageError) as info:
        ItemStoreGrain().build(duplicated)
    assert "duplicate item-store keys" in str(info.value)


def test_unsupported_grain_rejected_naming_supported_value():
    with pytest.raises(ConfigError) as info:
        make_grain("item")
    assert "spine.granularity" in str(info.value)
    assert "item_store" in str(info.value)


def test_seam_identity():
    assert ItemStoreGrain().describe().rep_id == "REP-12"
    assert ItemStoreGrain().describe().implementation == "item_store"