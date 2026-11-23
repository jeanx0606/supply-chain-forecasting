"""Confirmed real-dataset facts from Step 1 - declarations, not results.

Rue signed these on 2026-11-21 against the cleaned data. They cannot be observed
on the sample dataset and must not be reported as measured output. The guarded
test at the bottom of ``tests/unit/test_expected_counts.py`` reconciles all of
them once the real five-file dataset is placed in ``data/raw``.

One arithmetic note stays visible rather than smoothed: of the 112 A4 pairs, **111
also required the A3 correction**, so 1,842 non-A4 negative pairs + 111 = 1,953.
The 112th A4 pair never went negative, which is why the A4-inclusive count is 1,953
and not 1,954. ``EXPECTED_A4_PAIRS_REQUIRING_A3`` records the 111 explicitly.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

EXPECTED_PAIR_TOTAL = 106850
EXPECTED_SPLIT_COUNTS = {"train": 62523, "validation": 19684, "test": 24643}
EXPECTED_SEASON_SPLIT = {
    "train": ("SS17", "AW17", "SS18", "AW18"),
    "validation": ("SS19",),
    "test": ("AW19",),
}
EXPECTED_NEGATIVE_CELLS = 3386

EXPECTED_STOCKOUT_PAIRS = 7930
EXPECTED_STOCKOUT_RATE = 0.0742
EXPECTED_NEGATIVE_PAIRS = 1842
EXPECTED_NEGATIVE_RATE = 0.017
EXPECTED_NEGATIVE_PAIRS_WITH_A4 = 1953
EXPECTED_NEGATIVE_RATE_WITH_A4 = 0.018
EXPECTED_A4_PAIRS = 112
EXPECTED_A4_PAIRS_REQUIRING_A3 = 111

LAST_SS19_RELEASE = "2019-08-05"
LATE_AW19_PAIRS = 19134
AW19_PAIRS = 24643
LATE_AW19_SHARE = 0.776

# Chapter 5 §5.2 / §5.3 and Chapter 6 §6.9. The zero-sales share is a pre-cleaning
# fact (cleaning clips 3,386 negatives to zero, which raises the cleaned share).
ZERO_SALES_SHARE = 0.373
MEDIAN_SELL_THROUGH = 0.71
ZERO_SALES_WITH_ZERO_STOCK_SHARE = 0.03
A2_FIRST_STOCKED_BEFORE_RELEASE_SHARE = 0.78

ATTRIBUTE_DIMENSIONS = {"category": 27, "color": 10, "fabric": 59}
TRENDS_TOTAL_TERMS = 96
TRENDS_DROPPED_TERMS = ("polyviscous", "muslin cotton or silk",
                        "fluid polyviscous")
TRENDS_REMAINING_TERMS = 93
TRENDS_FEATURES_PER_WEEK = ("category", "color", "fabric")
TRENDS_WEEKS = (-4, -1, 0, 1)
TRENDS_N_FEATURES = 12
TRENDS_FABRIC_SERIES = 56
TRENDS_PAIRS_WITHOUT_FABRIC = 3840
TRENDS_PAIRS_WITHOUT_FABRIC_SHARE = 0.036

SAMPLE_MARKER = "_sample_marker.json"


def split_total(counts) -> int:
    return int(sum(int(counts.get(name, 0)) for name in EXPECTED_SPLIT_COUNTS))


def discrepancies(counts) -> list:
    problems = []
    for name, expected in EXPECTED_SPLIT_COUNTS.items():
        observed = int(counts.get(name, 0))
        if observed != expected:
            problems.append(f"{name}: expected {expected}, observed {observed}")
    total = split_total(counts)
    if total != EXPECTED_PAIR_TOTAL:
        problems.append(f"total: expected {EXPECTED_PAIR_TOTAL}, observed {total}")
    return problems


def season_counts(frame) -> dict:
    if "season" not in frame.columns:
        return {}
    values = frame["season"].astype("object").where(
        frame["season"].notna(), "unknown").astype(str)
    return {str(name): int(count) for name, count in values.value_counts().items()}


def season_discrepancies(frame) -> list:
    counts = season_counts(frame)
    problems = []
    for part, seasons in EXPECTED_SEASON_SPLIT.items():
        observed = sum(counts.get(season, 0) for season in seasons)
        expected = EXPECTED_SPLIT_COUNTS[part]
        if observed != expected:
            problems.append(f"{part} seasons {seasons}: expected {expected}, "
                            f"observed {observed}")
    return problems


def expected_seasons() -> tuple:
    return tuple(season for seasons in EXPECTED_SEASON_SPLIT.values()
                 for season in seasons)


def late_aw19_counts(frame) -> dict:
    if "season" not in frame.columns or "release_date" not in frame.columns:
        return {"aw19": 0, "late": 0}
    dates = pd.to_datetime(frame["release_date"], errors="coerce")
    aw19 = frame["season"].astype("object").astype(str) == "AW19"
    late = aw19 & (dates > pd.Timestamp(LAST_SS19_RELEASE))
    return {"aw19": int(aw19.sum()), "late": int(late.sum())}


def late_aw19_discrepancies(frame) -> list:
    counts = late_aw19_counts(frame)
    problems = []
    if counts["aw19"] != AW19_PAIRS:
        problems.append(f"AW19 pairs: expected {AW19_PAIRS}, "
                        f"observed {counts['aw19']}")
    if counts["late"] != LATE_AW19_PAIRS:
        problems.append(f"late-AW19 pairs: expected {LATE_AW19_PAIRS}, "
                        f"observed {counts['late']}")
    return problems


def trends_feature_count() -> int:
    return len(TRENDS_FEATURES_PER_WEEK) * len(TRENDS_WEEKS)


def a4_inclusive_negative_pairs() -> int:
    return EXPECTED_NEGATIVE_PAIRS + EXPECTED_A4_PAIRS_REQUIRING_A3


def is_real_dataset(raw_dir) -> bool:
    raw = Path(raw_dir)
    return (raw / "sales.csv").exists() and not (raw / SAMPLE_MARKER).exists()