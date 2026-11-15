"""Deterministic synthetic dataset shaped like VISUELLE 2.0. Self-test only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import StageError

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


SAMPLE_NOTICE = ("[sample] SAMPLE DATA - SELF-TEST ONLY. "
                 "Figures derived from it must never be reported as results.")
WEEKS_IN_YEAR = 52
MARKER_NAME = "_sample_marker.json"
FILES = ("sales.csv", "restocks.csv", "price_discount_series.csv",
         "vis2_gtrends_data.csv", "vis2_weather_data.csv")


def make_sample_data(out_dir: str | Path = "data/raw", n_items: int = 200,
                     n_stores: int = 20, seed: int = 42,
                     overwrite: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    existing = [out / name for name in FILES if (out / name).exists()]
    if existing and not overwrite:
        raise StageError(
            "sample", f"refusing to overwrite {len(existing)} existing raw files",
            next_command="python -m codes.sample_data --overwrite")

    release_dates = pd.date_range("2016-11-28", "2019-12-30", freq="W-MON")
    items = np.arange(1, n_items + 1)
    stores = np.arange(1, n_stores + 1)
    seasons = ["SS17", "AW17", "SS18", "AW18", "SS19", "AW19"]
    categories = ["tops", "trousers", "skirts", "shorts", "dresses"]
    colors = ["black", "white", "blue", "red", "beige"]
    fabrics = ["cotton", "knit", "denim", "viscose"]

    sales_rows = []
    price_rows = []
    for item in items:
        release = release_dates[rng.integers(0, len(release_dates))]
        carried = min(n_stores, int(rng.integers(1, 6)))
        item_stores = rng.choice(stores, size=carried, replace=False)
        base = float(rng.uniform(3, 40))
        for store in np.sort(item_stores):
            weekly = np.maximum(
                0.0, rng.poisson(base, 12) - rng.integers(0, 4, 12)).astype(float)
            row = {
                "Unnamed: 0": len(sales_rows),
                "external_code": int(item),
                "retail": int(store),
                "season": str(rng.choice(seasons)),
                "category": str(rng.choice(categories)),
                "color": str(rng.choice(colors)),
                "fabric": str(rng.choice(fabrics)),
                "image_path": f"img/{item}.jpg",
                "release_date": pd.Timestamp(release).strftime("%Y-%m-%d"),
                "restock": int(weekly.sum() * rng.uniform(0.3, 0.8)),
            }
            for week in range(12):
                row[week] = float(weekly[week])
            sales_rows.append(row)

            price_row = {"external_code": int(item), "retail": int(store),
                         "price": round(float(rng.uniform(0.05, 0.95)), 3)}
            for week in range(12):
                price_row[week] = round(
                    0.0 if rng.random() < 0.55 else float(rng.uniform(0.05, 0.8)), 3)
            price_rows.append(price_row)

    sales = pd.DataFrame(sales_rows)
    sales.to_csv(out / "sales.csv", index=False)

    restock_rows = []
    for row in sales_rows:
        for _ in range(int(rng.integers(1, 5))):
            restock_rows.append({
                "external_code": row["external_code"],
                "retail": row["retail"],
                "week": int(rng.integers(1, WEEKS_IN_YEAR + 1)),
                "year": int(rng.integers(2017, 2021)),
                "qty": int(rng.integers(1, 60)),
            })
    pd.DataFrame(restock_rows).to_csv(out / "restocks.csv", index=False)

    pd.DataFrame(price_rows).drop_duplicates(
        subset=["external_code", "retail"]).to_csv(
        out / "price_discount_series.csv", index=False)

    trend_dates = pd.date_range("2015-10-05", "2019-12-16", freq="W-MON")
    trends = {"date": trend_dates.strftime("%Y-%m-%d")}
    for term in range(20):
        trends[f"term_{term}"] = np.round(rng.uniform(0, 100, len(trend_dates)), 2)
    pd.DataFrame(trends).to_csv(out / "vis2_gtrends_data.csv", index=False)

    weather_dates = pd.date_range("2016-01-04", "2019-12-30", freq="D")
    weather_frames = []
    fields = ["avg_temp", "min_temp", "max_temp", "dew_point", "humidity",
              "visibility", "avg_wind", "max_wind", "gust", "sea_pressure",
              "avg_pressure", "rain"]
    for locality in range(1, 21):
        frame = pd.DataFrame({
            "locality": locality,
            "date": weather_dates.strftime("%d/%m/%Y"),
        })
        for field in fields:
            frame[field] = np.round(rng.normal(15, 5, len(weather_dates)), 2)
        weather_frames.append(frame)
    pd.concat(weather_frames, ignore_index=True).to_csv(
        out / "vis2_weather_data.csv", index=False)

    hashes = {
        name: _sha256(out / name) for name in FILES
    }
    (out / MARKER_NAME).write_text(json.dumps({
        "generator": "codes.sample_data",
        "seed": int(seed),
        "items": int(n_items),
        "stores": int(n_stores),
        "self_test_only": True,
        "notice": SAMPLE_NOTICE,
        "files": hashes,
    }, indent=2), encoding="utf-8")

    return {
        "out_dir": str(out),
        "files": [str(out / name) for name in FILES],
        "sample_marker": str(out / MARKER_NAME),
        "notice": SAMPLE_NOTICE,
    }


def is_sample_dataset(raw_dir: str | Path) -> bool:
    return (Path(raw_dir) / MARKER_NAME).exists()