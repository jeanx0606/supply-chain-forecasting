"""Shared fixtures: a five-file dataset with switchable defect knobs."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_CONFIG = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))


def make_config(root, mutate=None, name="settings.yaml") -> Path:
    """Write a settings file under root/config with optional mutations."""
    import copy

    config = copy.deepcopy(BASE_CONFIG)
    if mutate:
        mutate(config)
    target = Path(root) / "config" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    register_source = ROOT / "config" / "replaceability.yaml"
    register_target = target.parent / "replaceability.yaml"
    if register_source.exists() and not register_target.exists():
        register_target.write_text(register_source.read_text(encoding="utf-8"),
                                   encoding="utf-8")
    return target


def write_five_file_dataset(tmp_path, items: int = 40, stores: int = 8, seed: int = 42,
                            duplicates: int = 0, undated_rows: int = 0,
                            non_monday_dates: int = 0, missing_fabric: int = 0,
                            year_boundary_restock: bool = False,
                            late_2019_release: bool = False,
                            localities: int = 20,
                            negative_weeks: int = 0,
                            preset_files_present: bool = False) -> dict:
    from src.sample import make_sample_data

    tmp_path = Path(tmp_path)
    raw = tmp_path / "raw"
    make_sample_data(raw, items, stores, seed, overwrite=True)

    sales = pd.read_csv(raw / "sales.csv")
    if negative_weeks:
        for week in range(negative_weeks):
            sales.loc[0, str(week)] = -2.0
    if late_2019_release:
        index = sales.index[: max(1, len(sales) // 10)]
        sales.loc[index, "release_date"] = "2019-12-02"
    if duplicates:
        extra = sales.head(duplicates).copy()
        sales = pd.concat([sales, extra], ignore_index=True)
    if undated_rows:
        sales.loc[sales.index[:undated_rows], "release_date"] = ""
    if non_monday_dates:
        index = sales.index[:non_monday_dates]
        shifted = pd.to_datetime(sales.loc[index, "release_date"]) + pd.Timedelta(days=1)
        sales.loc[index, "release_date"] = shifted.dt.strftime("%Y-%m-%d")
    if missing_fabric:
        sales.loc[sales.index[:missing_fabric], "fabric"] = ""
    sales.to_csv(raw / "sales.csv", index=False)

    if year_boundary_restock or late_2019_release:
        restocks = pd.read_csv(raw / "restocks.csv")
        keys = sales[["external_code", "retail"]].drop_duplicates().head(3)
        extra_rows = [{"external_code": int(row.external_code),
                       "retail": int(row.retail), "week": week, "year": 2020, "qty": 12}
                      for row in keys.itertuples(index=False) for week in range(1, 5)]
        pd.concat([restocks, pd.DataFrame(extra_rows)], ignore_index=True).to_csv(
            raw / "restocks.csv", index=False)

    if preset_files_present:
        sales.to_csv(raw / "stfore_train.csv", index=False)
        sales.to_csv(raw / "stfore_test.csv", index=False)

    if localities and localities != 20:
        weather = pd.read_csv(raw / "vis2_weather_data.csv")
        keep = sorted(weather["locality"].unique())[:localities]
        weather[weather["locality"].isin(keep)].to_csv(
            raw / "vis2_weather_data.csv", index=False)

    config_path = make_config(tmp_path, lambda cfg: cfg["data"].update(
        {"raw_dir": "raw"}), name="settings.yaml")

    return {
        "root": str(tmp_path),
        "raw": str(raw),
        "config": str(config_path),
        "files": {name: str(raw / name) for name in
                  ("sales.csv", "restocks.csv", "price_discount_series.csv",
                   "vis2_gtrends_data.csv", "vis2_weather_data.csv")},
    }


@pytest.fixture(scope="session")
def sample_dataset(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dataset")
    return write_five_file_dataset(tmp, items=40, stores=8)


@pytest.fixture(scope="session")
def sample_bundle(sample_dataset):
    from src.config.loader import load_config
    from src.pipeline import run

    cfg = load_config(sample_dataset["config"], root=sample_dataset["root"])
    return run(cfg, verbose=False)