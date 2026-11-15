"""Channel registry: every declared channel carries a status (D6, REP-09)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.config.loader import load_config
from src.io.channels import ChannelState, TrendsPolicy, probe_channels


@pytest.fixture()
def cfg(tmp_path, sample_dataset):
    root = Path(sample_dataset["root"])
    raw = Path(sample_dataset["raw"])
    work = tmp_path / "work"
    shutil.copytree(raw, work / "raw")
    from tests.conftest import make_config

    config_path = make_config(work, lambda c: c["data"].update({"raw_dir": "raw"}))
    return load_config(str(config_path), root=str(work)), work / "raw"


def test_every_declared_channel_has_a_status(cfg):
    config, _ = cfg
    registry = probe_channels(config)
    statuses = registry.all()
    assert len(statuses) == 5
    assert {status.name for status in statuses} == {
        "sales", "restocks", "price_discount", "gtrends", "weather"}


def test_absent_weather_is_skipped_not_fatal(cfg):
    config, raw = cfg
    (raw / config.data.files.weather).unlink()
    registry = probe_channels(config)
    status = registry.status("weather")
    assert status.value == ChannelState.SKIPPED.value
    assert status.reason == "file_absent"


def test_trends_deferred_reports_out_of_scope(cfg):
    config, _ = cfg
    status = probe_channels(config).status("gtrends")
    assert status.value == ChannelState.DEFERRED.value
    assert status.reason == "out_of_scope"


def test_trends_consume_and_malformed(cfg):
    config, raw = cfg
    policy = TrendsPolicy("consume")
    assert policy.status(True, False).value == ChannelState.CONSUMED.value
    assert policy.status(True, True).value == ChannelState.SKIPPED.value
    assert policy.status(True, True).reason == "malformed"
    assert policy.status(False, False).value == ChannelState.SKIPPED.value


def test_trends_skip_policy(cfg):
    assert TrendsPolicy("skip").status(True, False).value == ChannelState.SKIPPED.value