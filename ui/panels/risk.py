"""P6 Risk panel: both proxy grades side by side, thresholds stated (C4)."""
from __future__ import annotations

import streamlit as st

from ui.components import constraint_caption


def build_view(bundle) -> dict:
    risk = bundle.risk
    return {
        "stockout_proxy": risk.stockout_proxy,
        "overstock": risk.overstock,
        "demand_slice_total": risk.demand_slice_total,
        "inventory_snapshot": risk.inventory_snapshot,
        "thresholds": risk.thresholds,
        "proxy_note": risk.proxy_note,
    }


def render(bundle) -> None:
    view = build_view(bundle)
    left, right = st.columns(2)
    left.metric("Stock-out risk (proxy)", view["stockout_proxy"].upper())
    right.metric("Overstock risk", view["overstock"].upper())
    st.write(f"snapshot inventory **{view['inventory_snapshot']:.2f}** against forecast "
             f"demand **{view['demand_slice_total']:.2f}**")
    st.write(f"thresholds: {view['thresholds']}")
    st.error("Stock-out risk is a PROXY inferred from reconstructed inventory - "
             "no stock-out column exists in the data. [constraint C4]")
    constraint_caption("C4")