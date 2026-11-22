"""Chronological split: exhaustive, exclusive, date-group preserving (C9)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.errors import ConfigError, StageError
from src.preprocess.split import ChronologicalSplit, PresetSplitPolicy, SeasonSplit


def splitter(**kwargs):
    defaults = dict(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    defaults.update(kwargs)
    return ChronologicalSplit(**defaults)


def test_masks_are_exhaustive_and_exclusive():
    dates = pd.Series(pd.date_range("2019-01-07", periods=20, freq="W-MON"))
    assignment = splitter().resolve(dates)
    total = (assignment.train.astype(int) + assignment.validation.astype(int)
             + assignment.test.astype(int))
    assert (total == 1).all()


def test_same_release_date_stays_together():
    dates = pd.Series(
        ["2018-01-01"] * 30 + ["2018-06-04"] * 10 + ["2018-12-03"] * 10 + ["2019-06-03"] * 10)
    assignment = splitter().resolve(dates)
    for mask in (assignment.train, assignment.validation, assignment.test):
        same = mask[dates == "2018-01-01"]
        assert same.nunique() == 1


def test_masks_are_index_aligned_not_positional():
    dates = pd.Series(pd.date_range("2019-01-07", periods=20, freq="W-MON"),
                      index=range(100, 120))
    assignment = splitter().resolve(dates)
    assert assignment.test.index.equals(dates.index)


def test_too_few_dates_aborts():
    with pytest.raises(StageError):
        splitter().resolve(pd.Series(pd.to_datetime(["2019-01-07", "2019-01-14"])))


def test_empty_test_partition_aborts():
    dates = pd.Series(pd.date_range("2019-01-07", periods=20, freq="W-MON"))
    with pytest.raises(StageError) as info:
        splitter(explicit_cut_dates=("2020-01-01", "2020-06-01")).resolve(dates)
    assert "test set" in str(info.value)


def test_cut_dates_and_counts_recorded():
    dates = pd.Series(pd.date_range("2019-01-07", periods=30, freq="W-MON"))
    assignment = splitter().resolve(dates)
    assert assignment.cut_train_end < assignment.cut_validation_end
    assert sum(assignment.counts.values()) == 30
    assert assignment.method == "chronological"


def test_preset_policy_rejects_missing_factor():
    from src.errors import ConfigError

    with pytest.raises(ConfigError) as info:
        PresetSplitPolicy(enabled=True, files=("a.csv", "b.csv")).require()
    assert "split.preset_scale_factor" in str(info.value)
    assert "REP-05" in str(info.value)
    assert PresetSplitPolicy(enabled=False).available() is False


def test_preset_provenance_states_are_explicit():
    off = PresetSplitPolicy(enabled=False, files=("a.csv", "b.csv"))
    absent = off.provenance(False)
    assert absent.present is False and absent.used is False
    assert "not present" in absent.statement

    present = off.provenance(True)
    assert present.present is True and present.used is False
    assert "deliberately not used" in present.statement

    on = PresetSplitPolicy(enabled=True, files=("a.csv", "b.csv"), scale_factor=53.0)
    used = on.provenance(True)
    assert used.used is True
    assert "secondary" in used.statement and "raw units" in used.statement
    assert on.scale_factor_state()["state"] == "applied"


def test_factor_declared_while_switch_off_is_inactive():
    policy = PresetSplitPolicy(enabled=False, files=("a.csv", "b.csv"), scale_factor=53.0)
    assert policy.scale_factor_state()["state"] == "inactive_parameter"


def test_presence_on_disk_never_changes_the_partition():
    dates = pd.Series(pd.date_range("2019-01-07", periods=20, freq="W-MON"))
    off = PresetSplitPolicy(enabled=False, files=("a.csv", "b.csv"))
    assert off.provenance(False).as_dict()["present"] is False
    assert off.provenance(True).as_dict()["present"] is True
    first = splitter().resolve(dates)
    second = splitter().resolve(dates)
    assert first.counts == second.counts
    assert first.cut_train_end == second.cut_train_end


def test_season_split_maps_the_confirmed_partitions():
    seasons = pd.Series(["SS17", "AW17", "SS18", "AW18", "SS19", "AW18",
                         "AW19", "AW19"])
    dates = pd.Series(pd.date_range("2017-01-02", periods=8, freq="W-MON"))
    assignment = SeasonSplit().resolve(dates, seasons)
    assert assignment.method == "season"
    assert assignment.counts == {"train": 5, "validation": 1, "test": 2}
    assert assignment.train.tolist() == [True, True, True, True, False,
                                         True, False, False]
    assert sum(assignment.counts.values()) == 8


def test_season_split_needs_the_season_column():
    dates = pd.Series(pd.date_range("2017-01-02", periods=3, freq="W-MON"))
    with pytest.raises(ConfigError) as info:
        SeasonSplit().resolve(dates)
    assert "season column" in str(info.value)


def test_season_split_rejects_an_unknown_label():
    seasons = pd.Series(["SS17", "AW17", "ZZ20"])
    dates = pd.Series(pd.date_range("2017-01-02", periods=3, freq="W-MON"))
    with pytest.raises(ConfigError) as info:
        SeasonSplit().resolve(dates, seasons)
    assert "unknown season label" in str(info.value)


def test_season_split_refuses_an_empty_partition():
    seasons = pd.Series(["AW19", "AW19", "AW19"])
    dates = pd.Series(pd.date_range("2019-08-05", periods=3, freq="W-MON"))
    with pytest.raises(StageError) as info:
        SeasonSplit().resolve(dates, seasons)
    assert "train partition is empty" in str(info.value)


def test_chronological_method_is_a_plain_label_from_the_enum():
    from src.config.schema import SplitPolicyName
    from src.preprocess.split import ChronologicalSplit as Policy

    dates = pd.Series(pd.date_range("2019-01-07", periods=20, freq="W-MON"))
    assignment = Policy(0.7, 0.15, 0.15, policy=SplitPolicyName.CHRONOLOGICAL).resolve(dates)
    assert assignment.method == "chronological"