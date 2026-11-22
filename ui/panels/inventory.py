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
        "initial_base": trajectory.initial_stock.base_value,
        "correction": trajectory.initial_stock.correction,
        "strategy": trajectory.initial_stock.strategy_name,
        "sellout_week": trajectory.sellout_week,
        "remaining": trajectory.remaining_weeks_after_sellout,
        "marker": trajectory.reconstruction_marker,
        "decision_week": bundle.replenishment.decision_week,
        "policy": trajectory.policy,
        "would_go_negative": trajectory.would_go_negative,
        "min_raw": trajectory.min_raw_stock,
        "negative_weeks": trajectory.negative_weeks,
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
    policy = view["policy"] or {}
    st.caption(
        f"opening stock {view['initial_base']:.1f} x alpha "
        f"{policy.get('alpha', 1.0):g} = {view['initial_stock']:.1f}; "
        f"A3 correction {view['correction']:+.1f}; A1 floor-at-zero "
        f"{'on' if policy.get('a1_floor_at_zero') else 'off'}")
    if view["would_go_negative"]:
        st.caption(f"pre-A3 path minimum {view['min_raw']:.1f} over "
                   f"{view['negative_weeks']} negative week(s); the A3 correction is "
                   f"assumption-driven, so the final level is uncertain")
    st.warning("Reconstructed inventory: derived from restock minus sales, not an "
               "observed stock series. The absolute level is uncertain; only the "
               "trajectory is observable. [constraint C3]")
    constraint_caption("C3")