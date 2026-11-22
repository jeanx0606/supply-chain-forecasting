"""Step 1 facts: internally consistent, and reconciled against the real dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.preprocess.expected_counts import (
    ATTRIBUTE_DIMENSIONS,
    AW19_PAIRS,
    EXPECTED_A4_PAIRS_REQUIRING_A3,
    EXPECTED_A4_PAIRS,
    EXPECTED_NEGATIVE_CELLS,
    EXPECTED_NEGATIVE_PAIRS,
    EXPECTED_NEGATIVE_PAIRS_WITH_A4,
    EXPECTED_NEGATIVE_RATE,
    EXPECTED_NEGATIVE_RATE_WITH_A4,
    EXPECTED_PAIR_TOTAL,
    EXPECTED_SEASON_SPLIT,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_STOCKOUT_PAIRS,
    EXPECTED_STOCKOUT_RATE,
    LATE_AW19_PAIRS,
    LATE_AW19_SHARE,
    TRENDS_DROPPED_TERMS,
    TRENDS_FABRIC_SERIES,
    TRENDS_N_FEATURES,
    TRENDS_PAIRS_WITHOUT_FABRIC,
    TRENDS_PAIRS_WITHOUT_FABRIC_SHARE,
    TRENDS_REMAINING_TERMS,
    TRENDS_TOTAL_TERMS,
    TRENDS_WEEKS,
    a4_inclusive_negative_pairs,
    discrepancies,
    expected_seasons,
    is_real_dataset,
    late_aw19_counts,
    late_aw19_discrepancies,
    season_discrepancies,
    split_total,
    trends_feature_count,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def test_confirmed_counts_add_up():
    assert split_total(EXPECTED_SPLIT_COUNTS) == EXPECTED_PAIR_TOTAL
    assert sum(EXPECTED_SPLIT_COUNTS.values()) == 106850
    assert EXPECTED_SPLIT_COUNTS["validation"] == 19684
    assert EXPECTED_NEGATIVE_CELLS == 3386


def test_season_split_covers_every_season_once():
    seasons = expected_seasons()
    assert len(seasons) == len(set(seasons)) == 6
    assert set(seasons) == {"SS17", "AW17", "SS18", "AW18", "SS19", "AW19"}
    assert len(EXPECTED_SEASON_SPLIT["test"]) == 1


def test_stockout_and_negative_counts_are_recorded():
    assert EXPECTED_STOCKOUT_PAIRS == 7930
    assert EXPECTED_STOCKOUT_RATE == pytest.approx(7930 / 106850, abs=5e-4)
    assert EXPECTED_NEGATIVE_PAIRS == 1842
    assert EXPECTED_A4_PAIRS == 112
    assert EXPECTED_NEGATIVE_PAIRS_WITH_A4 == 1953
    assert EXPECTED_A4_PAIRS_REQUIRING_A3 == 111
    assert a4_inclusive_negative_pairs() == EXPECTED_NEGATIVE_PAIRS_WITH_A4
    assert EXPECTED_NEGATIVE_PAIRS + EXPECTED_A4_PAIRS_REQUIRING_A3 == 1953
    assert EXPECTED_A4_PAIRS == 112


def test_negative_rates_match_the_recorded_percentages():
    assert EXPECTED_NEGATIVE_RATE == pytest.approx(1842 / 106850, abs=5e-4)
    assert EXPECTED_NEGATIVE_RATE_WITH_A4 == pytest.approx(1953 / 106850, abs=5e-4)


def test_late_aw19_share_and_helper():
    assert LATE_AW19_PAIRS / AW19_PAIRS == pytest.approx(LATE_AW19_SHARE, abs=5e-4)
    frame = pd.DataFrame({
        "season": ["AW19", "AW19", "SS19"],
        "release_date": ["2019-08-06", "2019-08-05", "2019-08-06"],
    })
    assert late_aw19_counts(frame) == {"aw19": 2, "late": 1}


def test_trends_facts_match_the_recorded_statement():
    assert len(TRENDS_DROPPED_TERMS) == 3
    assert TRENDS_TOTAL_TERMS - len(TRENDS_DROPPED_TERMS) == TRENDS_REMAINING_TERMS
    assert TRENDS_REMAINING_TERMS == 93
    assert TRENDS_WEEKS == (-4, -1, 0, 1)
    assert trends_feature_count() == TRENDS_N_FEATURES == 12
    assert TRENDS_FABRIC_SERIES < ATTRIBUTE_DIMENSIONS["fabric"]
    assert TRENDS_PAIRS_WITHOUT_FABRIC / 106850 == pytest.approx(
        TRENDS_PAIRS_WITHOUT_FABRIC_SHARE, abs=5e-4)


def test_dimension_counts_are_recorded():
    assert ATTRIBUTE_DIMENSIONS == {"category": 27, "color": 10, "fabric": 59}


def test_discrepancies_are_reported_per_partition_and_total():
    assert discrepancies(EXPECTED_SPLIT_COUNTS) == []
    wrong = dict(EXPECTED_SPLIT_COUNTS, test=EXPECTED_SPLIT_COUNTS["test"] - 1)
    problems = discrepancies(wrong)
    assert any("test" in problem for problem in problems)
    assert any("total" in problem for problem in problems)


def test_sample_marker_is_not_mistaken_for_real_data():
    assert is_real_dataset(RAW) is False


@pytest.mark.skipif(not is_real_dataset(RAW), reason="real data absent")
def test_real_dataset_reconciles_every_signed_figure():
    from src.io.loader import load_restocks, load_sales
    from src.preprocess.cleaning import clean_sales, make_sign_policy
    from src.preprocess.inventory import (audit_reconstruction,
                                          make_initial_stock_strategy)
    from src.preprocess.restock import make_retention_policy
    from src.preprocess.split import SeasonSplit
    from src.preprocess.time_index import attach_week_index

    sales, _ = load_sales(RAW / "sales.csv", 12, "q")
    cleaned = clean_sales(sales, 12, "q", policy=make_sign_policy("clip_at_zero"))
    frame = attach_week_index(cleaned.frame, 12)
    restocks, _ = load_restocks(RAW / "restocks.csv")

    release_map = {
        (int(row.external_code), int(row.retail)): row.release_date.date()
        for row in frame[["external_code", "retail", "release_date"]].itertuples(
            index=False)
    }
    outcome = make_retention_policy("lifecycle_membership", "2019-12-31").retain(
        restocks, release_map, 12)
    audit = audit_reconstruction(
        frame, outcome.events_by_key, outcome.opening_by_key,
        make_initial_stock_strategy("restocked_to_release", alpha=1.0))

    pairs = frame.drop_duplicates(subset=["external_code", "retail"])
    assert len(pairs) == EXPECTED_PAIR_TOTAL
    assert season_discrepancies(pairs) == []
    assert late_aw19_discrepancies(pairs) == []
    season_split = SeasonSplit().resolve(frame["release_date"], frame["season"])
    assert season_split.counts == EXPECTED_SPLIT_COUNTS
    assert audit.pairs == EXPECTED_PAIR_TOTAL
    assert audit.opening_zero_pairs == EXPECTED_A4_PAIRS
    assert audit.zero_or_below_pairs == EXPECTED_STOCKOUT_PAIRS
    assert audit.negative_pairs == EXPECTED_NEGATIVE_PAIRS_WITH_A4
    assert audit.negative_pairs_excluding_a4 == EXPECTED_NEGATIVE_PAIRS
    assert audit.a4_pairs_requiring_a3 == EXPECTED_A4_PAIRS_REQUIRING_A3