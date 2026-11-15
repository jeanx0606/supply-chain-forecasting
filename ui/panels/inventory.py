"""P4 Inventory panel: reconstructed trajectory, assumed I0, sell-out onset (C3)."""
from __future__ import annotations

import streamlit as st

from ui.components import constraint_caption


def build_view(bundle) -> dict:
    trajectory = bundle.demonstration
    rows = trajectory.as_rows()
    return {
        "key": trajectory.key,
        "entries": rows,
        "initial_stock": trajectory.initial_stock.value,
        "strategy": trajectory.initial_stock.strategy_name,
        "sellout_week": trajectory.sellout_week,
        "remaining": trajectory.remaining_weeks_after_sellout,
        "marker": trajectory.reconstruction_marker,
        "decision_week": bundle.replenishment.decision_week,
    }


def render(bundle) -> None:
    view = build_view(bundle)
    st.write(f"Demonstration item-store: **{view['key']}** | strategy: "
             f"**{view['strategy']}**")
    st.line_chart({"reconstructed stock": [r["reconstructed_stock"]
                                           for r in view["entries"]]})
    first, second, third = st.columns(3)
    first.metric("Assumed I0", f"{view['initial_stock']:.1f}")
    second.metric("Sell-out onset", view["sellout_week"] or "none")
    third.metric("Weeks after sell-out", view["remaining"] or "n/a")
    st.caption(f"snapshot read at decision week {view['decision_week']} "
               f"(marker: {view['marker']})")
    st.warning("Reconstructed inventory: derived from restock minus sales, not an "
               "observed stock series. The absolute level is uncertain; only the "
               "trajectory is observable. [constraint C3]")
    constraint_caption("C3")