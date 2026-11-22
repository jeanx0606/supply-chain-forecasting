"""P5 Replenishment panel: (s, S), baseline comparison, assumption status (C5)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.components import constraint_caption
from ui.style import highlight_cells


def build_view(bundle) -> dict:
    recommendation = bundle.replenishment
    baseline = bundle.replenishment_baseline
    assumptions = recommendation.assumptions
    return {
        "reorder_point": recommendation.reorder_point,
        "target_level": recommendation.target_level,
        "recommended_qty": recommendation.recommended_qty,
        "trigger": recommendation.trigger,
        "decision_week": recommendation.decision_week,
        "forecast_origin_week": recommendation.forecast_origin_week,
        "inventory_snapshot": recommendation.inventory_snapshot,
        "lead_time_weeks": assumptions.lead_time_weeks,
        "order_cycle_weeks": assumptions.order_cycle_weeks,
        "service_level": assumptions.service_level,
        "service_level_applied": assumptions.service_level_applied,
        "scope_note": assumptions.scope_note,
        "assumed_statement": assumptions.assumed_statement,
        "protection_weeks": assumptions.protection_weeks,
        "z_value": assumptions.z_value,
        "simulation_label": assumptions.simulation_label,
        "baseline_target": baseline.target_level,
        "baseline_qty": baseline.recommended_qty,
        "baseline_weeks": baseline.target_weeks,
        "caption": recommendation.quantity_vs_timing_caption,
        "scenarios": bundle.scenarios,
    }


def assumption_rows(view: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"parameter": "lead_time_weeks", "value": view["lead_time_weeks"],
         "status": "assumed"},
        {"parameter": "order_cycle_weeks", "value": view["order_cycle_weeks"],
         "status": "assumed"},
        {"parameter": "service_level", "value": view["service_level"],
         "status": "applied" if view["service_level_applied"] else "out of scope"},
        {"parameter": "protection_period_P", "value": view["protection_weeks"],
         "status": "L + R"},
        {"parameter": "simulation_label", "value": view["simulation_label"],
         "status": "label"},
    ])


def render(bundle) -> None:
    view = build_view(bundle)
    st.subheader("Recommended (s, S)")
    first, second, third, fourth = st.columns(4)
    first.metric("Reorder point s", f"{view['reorder_point']:.2f}")
    second.metric("Target level S", f"{view['target_level']:.2f}")
    third.metric("Recommended qty", f"{view['recommended_qty']:,}")
    fourth.metric("Trigger", "yes" if view["trigger"] else "no")
    st.write(f"Decision week **{view['decision_week']}** = forecast origin "
             f"**{view['forecast_origin_week']}** | inventory snapshot "
             f"**{view['inventory_snapshot']:.2f}**")

    rows = assumption_rows(view)
    anomalies = []
    if not view["service_level_applied"]:
        anomalies.append((2, "status"))
    st.dataframe(highlight_cells(rows, anomalies), hide_index=True,
                 width="stretch")
    st.caption(view["scope_note"])

    st.subheader("Baseline rule (comparison)")
    st.write(f"Fixed cycle / fixed level over {view['baseline_weeks']} weeks: "
             f"target {view['baseline_target']:.2f}, quantity {view['baseline_qty']:,}")
    st.caption(view["caption"])
    st.caption(f"All replenishment figures are a {view['simulation_label']}; "
               f"buffer uses sigma_e (validation-season forecast-error standard "
               f"deviation) over P = L + R = {view['protection_weeks']} week(s), "
               f"z = {view.get('z_value')}")

    scenarios = view.get("scenarios") or {}
    scenario_rows = scenarios.get("rows") or []
    if scenario_rows:
        settings = scenarios.get("settings") or {}
        st.subheader("Scenario ladder (main setting)")
        table = pd.DataFrame([
            {"scenario": row["scenario"],
             "forecast": row["forecast_source"],
             "fill rate": row["fill_rate"],
             "stock-out rate": row["stockout_rate"],
             "overstock ratio": row["overstock_ratio"]}
            for row in scenario_rows])
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(f"L={settings.get('lead_time_weeks')}, "
                   f"R={settings.get('order_cycle_weeks')}, "
                   f"z={settings.get('service_level')}, "
                   f"P={settings.get('protection_weeks')}; cost index: "
                   f"{scenarios.get('cost_index_reason')}")
    constraint_caption("C5")