"""Anomaly palette and two Styler helpers (dark red for white-on-red legibility)."""
from __future__ import annotations

import pandas as pd

ANOMALY_FILL = "#8C2F2F"
ANOMALY_TEXT = "#FFFFFF"
ANOMALY_CSS = f"background-color: {ANOMALY_FILL}; color: {ANOMALY_TEXT};"


def highlight_rows(frame: pd.DataFrame, mask: pd.Series):
    """Fill the whole row for a duplicate key (or any flagged row)."""
    flags = pd.Series(mask, index=frame.index).fillna(False).astype(bool)

    def _style(row):
        return [ANOMALY_CSS if bool(flags.loc[row.name]) else "" for _ in row]

    return frame.style.apply(_style, axis=1)


def highlight_cells(frame: pd.DataFrame, cells):
    """Fill only the anomalous cell: wrong total, bad count, low coverage, price."""
    styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
    for index, column in cells:
        if column in styles.columns and index in styles.index:
            styles.loc[index, column] = ANOMALY_CSS

    def _style(_frame):
        return styles

    return frame.style.apply(_style, axis=None)


def anomaly_legend() -> str:
    return "red fill = anomalous value: incorrect total, duplicate key, date count, coverage or price"