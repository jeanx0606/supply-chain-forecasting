"""Contract-checked ingestion with canonical weekly labels (D10, DFX-C1)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..errors import ContractError
from .unit_contract import RAW_UNIT, stamp, unit_of
from .schema import (
    CATEGORICAL,
    TRENDS_CONTRACT,
    PRICE_CONTRACT,
    RESTOCK_CONTRACT,
    SALES_CONTRACT,
    WEATHER_CONTRACT,
    ChannelContract,
)


@dataclass(frozen=True)
class ParsedFrameStats:
    rows_read: int
    unparseable_dates: int
    ignored_columns: int
    duplicates_detected: int


def _check_contract(frame: pd.DataFrame, contract: ChannelContract, file_name: str) -> None:
    missing = [c for c in contract.required if c not in frame.columns]
    if missing:
        raise ContractError(file_name, "missing required fields", missing_fields=missing,
                            expected=list(contract.required))


def _numeric_week_columns(frame: pd.DataFrame, n_weeks: int, file_name: str) -> tuple[list, int]:
    numeric = [c for c in frame.columns if str(c).strip().isdigit()]
    numeric = sorted(numeric, key=lambda c: int(str(c).strip()))
    if len(numeric) < n_weeks:
        raise ContractError(
            file_name, "not enough numeric week columns",
            missing_fields=[f"{n_weeks} weekly columns 0..{n_weeks - 1}"],
            expected=f"{n_weeks} numeric week columns, observed {len(numeric)}")
    used = numeric[:n_weeks]
    ignored = len(numeric) - len(used)
    return used, ignored


def _normalise_week_headers(frame: pd.DataFrame, n_weeks: int, prefix: str,
                            file_name: str) -> tuple[pd.DataFrame, int]:
    used, ignored = _numeric_week_columns(frame, n_weeks, file_name)
    rename = {c: f"{prefix}{int(str(c).strip()) + 1}" for c in used}
    return frame.rename(columns=rename), ignored


def load_sales(path: Path, n_weeks: int = 12, prefix: str = "q") -> tuple[pd.DataFrame, ParsedFrameStats]:
    frame = pd.read_csv(path)
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    _check_contract(frame, SALES_CONTRACT, path.name)
    frame, ignored = _normalise_week_headers(frame, n_weeks, prefix, path.name)
    release = pd.to_datetime(frame["release_date"], errors="coerce")
    unparseable = int(release.isna().sum())
    frame["release_date"] = release
    for column in ("external_code", "retail"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["external_code"].isna().any() or frame["retail"].isna().any():
        raise ContractError(path.name, "identifier column contains non-numeric values",
                            missing_fields=["external_code", "retail"])
    duplicates = int(frame.duplicated(subset=["external_code", "retail"], keep=False).sum())
    stamp(frame, RAW_UNIT)
    return frame, ParsedFrameStats(len(frame), unparseable, ignored, duplicates)


def load_price(path: Path, n_weeks: int = 12) -> tuple[pd.DataFrame, ParsedFrameStats]:
    frame = pd.read_csv(path)
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    _check_contract(frame, PRICE_CONTRACT, path.name)
    frame, ignored = _normalise_week_headers(frame, n_weeks, "disc_w", path.name)
    for column in ("external_code", "retail"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    duplicates = int(frame.duplicated(subset=["external_code", "retail"], keep=False).sum())
    stamp(frame, RAW_UNIT)
    return frame, ParsedFrameStats(len(frame), 0, ignored, duplicates)


def load_restocks(path: Path) -> tuple[pd.DataFrame, ParsedFrameStats]:
    frame = pd.read_csv(path)
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    _check_contract(frame, RESTOCK_CONTRACT, path.name)
    for column in ("external_code", "retail", "week", "year", "qty"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    bad = int(frame[["external_code", "retail", "week", "year", "qty"]].isna().any(axis=1).sum())
    stamp(frame, RAW_UNIT)
    return frame, ParsedFrameStats(len(frame), bad, 0, 0)


def load_gtrends(path: Path) -> tuple[pd.DataFrame, ParsedFrameStats]:
    frame = pd.read_csv(path)
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    _check_contract(frame, TRENDS_CONTRACT, path.name)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    unparseable = int(frame["date"].isna().sum())
    stamp(frame, RAW_UNIT)
    return frame, ParsedFrameStats(len(frame), unparseable, 0, 0)


def load_weather(path: Path) -> tuple[pd.DataFrame, ParsedFrameStats]:
    frame = pd.read_csv(path)
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    _check_contract(frame, WEATHER_CONTRACT, path.name)
    frame["date"] = pd.to_datetime(frame["date"], format="%d/%m/%Y", errors="coerce")
    frame["locality"] = pd.to_numeric(frame["locality"], errors="coerce")
    unparseable = int(frame["date"].isna().sum())
    stamp(frame, RAW_UNIT)
    return frame, ParsedFrameStats(len(frame), unparseable, 0, 0)


def canonicalise_categories(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in CATEGORICAL:
        if column in out.columns:
            values = out[column].astype("object")
            values = values.where(values.notna(), "unknown")
            values = values.astype(str).str.strip()
            values = values.mask(values.isin(["", "nan", "None", "<NA>", "NaN"]), "unknown")
            out[column] = values
    return out