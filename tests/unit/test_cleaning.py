"""Cleaning: undated exclusion, fill-before-cast and quality counters (D9)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.errors import StageError
from src.preprocess.cleaning import make_sign_policy, clean_sales as _clean, quality_report


def clean_sales(frame, n_weeks=12, policy="clip_at_zero"):
    return _clean(frame, n_weeks, "q", policy=make_sign_policy(policy))


def make_frame(n=10, undated=0, missing_fabric=0, zero_week=3):
    rows = []
    for i in range(n):
        row = {
            "external_code": 100 + i,
            "retail": 1,
            "release_date": "2019-01-07" if i >= undated else "",
            "season": "SS19",
            "category": "tops",
            "color": "black",
            "fabric": "" if i < missing_fabric else "cotton",
        }
        for week in range(1, 13):
            row[f"q{week}"] = 0.0 if week == zero_week and i == 0 else 4.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_undated_rows_excluded_and_counted():
    result = clean_sales(make_frame(n=50, undated=12), 12)
    assert len(result.frame) == 38
    assert result.excluded["undated_rows"] == 12
    assert result.excluded["duplicate_rows"] == 0


def test_zero_sales_are_valid_and_retained():
    result = clean_sales(make_frame(n=5), 12)
    assert (result.frame["q3"] == 0.0).sum() == 1


def test_missing_category_becomes_literal_unknown():
    result = clean_sales(make_frame(n=6, missing_fabric=2), 12)
    values = set(result.frame["fabric"])
    assert "unknown" in values
    assert not ({"nan", "None", "<NA>", "NaN"} & values)


def test_duplicates_removed_and_counted():
    frame = make_frame(n=4)
    duplicated = pd.concat([frame, frame.head(2)], ignore_index=True)
    result = clean_sales(duplicated, 12)
    assert len(result.frame) == 4
    assert result.duplicate_count == 2


def test_all_undated_aborts():
    with pytest.raises(StageError) as info:
        clean_sales(make_frame(n=3, undated=3), 12)
    assert "no dated rows remain" in str(info.value)


def test_quality_report_shape_and_coverage():
    result = clean_sales(make_frame(n=10, undated=2), 12)
    report = quality_report(result.frame, result.excluded, result.duplicate_count,
                            0, "lifecycle_membership", {"state": "inactive_parameter"})
    for key in ("n_rows", "n_items", "n_stores", "date_min", "date_max",
                "duplicate_count", "pair_coverage", "excluded",
                "monday_violations", "weekly_only_marker", "restock_policy"):
        assert key in report
    assert 0.0 <= report["pair_coverage"] <= 1.0
    assert report["weekly_only_marker"] == "weekly"
    assert report["restock_policy"] == "lifecycle_membership"

# --- v1.3: sign policy (D23 / REP-17) ------------------------------------

def signed_frame(q4=-2.0, q7=None):
    frame = make_frame(n=2)
    frame.loc[0, "q4"] = q4
    if q7 is not None:
        frame.loc[0, "q7"] = q7
    return frame


def test_clip_and_retain_share_the_count_but_not_the_value():
    clip = clean_sales(signed_frame())
    retain = clean_sales(signed_frame(), policy="retain_as_returns")
    assert clip.frame.loc[0, "q4"] == 0.0
    assert retain.frame.loc[0, "q4"] == -2.0
    assert clip.sign_outcome.negative_cells == retain.sign_outcome.negative_cells == 1
    assert clip.sign_outcome.policy_name == "clip_at_zero"
    assert retain.sign_outcome.policy_name == "retain_as_returns"
    assert retain.sign_outcome.reinterpreted is True


def test_other_weeks_unchanged_and_count_by_week_is_sorted():
    result = clean_sales(signed_frame(q4=-2.0, q7=-5.0))
    assert result.sign_outcome.negative_cells == 2
    assert list(result.sign_outcome.negative_cells_by_week) == [4, 7]
    assert result.frame.loc[0, "q1"] == 4.0
    assert result.frame.loc[0, "q2"] == 4.0


def test_missing_week_is_zero_and_not_counted_as_negative():
    frame = make_frame(n=1)
    frame.loc[0, "q5"] = float("nan")
    result = clean_sales(frame)
    assert result.frame.loc[0, "q5"] == 0.0
    assert 5 not in result.sign_outcome.negative_cells_by_week


def test_sign_note_is_an_interpretation_not_imputation():
    result = clean_sales(signed_frame())
    assert "not imputation" in result.sign_outcome.note
    assert "not a data correction" in result.sign_outcome.note


def test_negative_restock_is_still_clamped():
    import pandas as pd
    from src.preprocess.restock import RestockRetentionPolicy

    events = pd.DataFrame([(1, 2, 11, 2019, -5.0)],
                          columns=["external_code", "retail", "week", "year", "qty"])
    from datetime import date
    outcome = RestockRetentionPolicy(n_weeks=12).retain(
        events, {(1, 2): date(2019, 3, 4)})
    assert outcome.events_by_key[(1, 2)] == {2: 0.0}
