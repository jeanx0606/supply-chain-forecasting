"""P3 Metrics panel: one block per model, item-level primary (C2)."""
from __future__ import annotations

import streamlit as st

from ui.components import constraint_caption, fmt_metric


def build_view(bundle) -> dict:
    blocks = []
    for name, record in bundle.metrics.items():
        blocks.append({
            "model": name,
            "wape_item_level": record.wape_item_level,
            "wape_pooled": record.wape_pooled,
            "mae": record.mae,
            "rmse": record.rmse,
            "wape_store_weighted": record.wape_store_weighted,
            "wape_small_stores": record.wape_small_stores,
            "n_small_stores": record.n_small_stores,
            "n_evaluated_records": record.n_evaluated_records,
            "n_evaluated_series": record.n_evaluated_series,
        })
    return {"blocks": blocks, "primary": bundle.effective_config["metrics"]["primary"]}


def render(bundle) -> None:
    view = build_view(bundle)
    for block in view["blocks"]:
        st.subheader(f"{block['model']}  -  primary: {view['primary']}")
        first, second, third, fourth = st.columns(4)
        first.metric("WAPE (item-level)", fmt_metric(block["wape_item_level"]))
        second.metric("WAPE (pooled)", fmt_metric(block["wape_pooled"]))
        third.metric("MAE", fmt_metric(block["mae"]))
        fourth.metric("RMSE", fmt_metric(block["rmse"]))
        if block["wape_small_stores"] is not None:
            st.caption(f"Small-store subset: WAPE={fmt_metric(block['wape_small_stores'])} "
                       f"(n={block['n_small_stores']}) [constraint C2]")
        st.caption(f"evaluated records={block['n_evaluated_records']}, "
                   f"item series={block['n_evaluated_series']}")
    st.caption("MAPE is not used as a primary metric: demand is sparse and zero-inflated.")
    constraint_caption("C2")