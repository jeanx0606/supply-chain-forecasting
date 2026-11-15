"""Failure paths: a missing mandatory file aborts before any fitting (A7)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config.loader import load_config
from src.errors import PrototypeError
from src.pipeline import run
from tests.conftest import write_five_file_dataset

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("filename,label", [
    ("sales.csv", "sales"),
    ("restocks.csv", "restocks"),
    ("price_discount_series.csv", "price_discount"),
])
def test_missing_mandatory_file_aborts_with_next_command(tmp_path, filename, label):
    dataset = write_five_file_dataset(tmp_path, items=20, stores=5, seed=7)
    Path(dataset["files"][filename]).unlink()
    cfg = load_config(dataset["config"], root=dataset["root"])

    with pytest.raises(PrototypeError) as info:
        run(cfg, verbose=False)

    message = str(info.value)
    assert filename in message
    assert "python -m codes.sample_data" in message
    assert not (tmp_path / "output" / "run-42").exists()


def test_missing_config_aborts_with_parameter(tmp_path):
    from src.errors import ConfigError

    with pytest.raises(ConfigError) as info:
        load_config(str(tmp_path / "nope.yaml"), root=str(tmp_path))
    assert "settings file not found" in str(info.value)