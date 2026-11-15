"""Anomaly palette and focused cell/row highlighting."""
from __future__ import annotations

import re

import pandas as pd

from ui.style import ANOMALY_FILL, ANOMALY_TEXT, anomaly_legend, highlight_cells, highlight_rows


def frame():
    return pd.DataFrame({"name": ["a", "b", "c"], "total": [10, 20, 30],
                         "price": [0.1, 0.2, 5.0]})


def styled_selector_count(html: str) -> int:
    """Count how many cells a rule carrying the anomaly fill applies to."""
    total = 0
    for selector_block in re.findall(r"([^{}]+)\{[^}]*" + re.escape(ANOMALY_FILL), html):
        total += len([part for part in selector_block.split(",") if part.strip()])
    return total


def test_palette_is_dark_red_with_white_text():
    assert ANOMALY_FILL == "#8C2F2F"
    assert ANOMALY_TEXT == "#FFFFFF"


def test_highlight_rows_fills_every_cell_of_flagged_row():
    html = highlight_rows(frame(), pd.Series([False, True, False])).to_html()
    assert styled_selector_count(html) == 3


def test_highlight_cells_fills_only_the_anomalous_cell():
    html = highlight_cells(frame(), [(1, "total")]).to_html()
    assert styled_selector_count(html) == 1


def test_legend_mentions_the_four_anomaly_kinds():
    text = anomaly_legend()
    for word in ("total", "duplicate", "coverage", "price"):
        assert word in text