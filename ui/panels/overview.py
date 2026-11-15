"""P1 Overview panel."""
from __future__ import annotations

import streamlit as st

from ui.components import constraint_caption


def build_view(bundle) -> dict:
    quality = bundle.quality
    provisional = getattr(bundle, "provisional", ())
    rep17 = next((ref for ref in provisional if ref.rep_id == "REP-17"), None)
    provenance = getattr(bundle, "unit_provenance", None)
    preset = (bundle.split.preset or {}) if hasattr(bundle.split, "preset") else {}
    return {
        "items": quality.n_items,
        "stores": quality.n_stores,
        "rows": quality.n_rows,
        "coverage": quality.pair_coverage,
        "date_min": quality.date_min,
        "date_max": quality.date_max,
        "duplicate_count": quality.duplicate_count,
        "excluded": quality.excluded,
        "monday_violations": quality.monday_violations,
        "restock_policy": quality.restock_policy,
        "negative_cells": quality.negative_cells,
        "negative_policy": quality.negative_sales_policy,
        "negative_by_week": quality.negative_cells_by_week,
        "unit": quality.unit,
        "applied_scale_factor": quality.applied_scale_factor,
        "unit_caption": unit_caption_for(provenance, quality),
        "preset_statement": preset.get("statement", ""),
        "sign_badge": (f"PROVISIONAL · REP-17 · {rep17.confirmer_role}"
                       if rep17 else "REP-17 confirmed"),
        "chosen_model": getattr(bundle, "main_model", None),
        "selection_outcome": (getattr(bundle, "selection", None) or {}).get("outcome"),
        "main_model_status": getattr(bundle, "main_model_status", "configured"),
        "conditional": getattr(bundle, "conditional", {}) or {},
        "self_test": bundle.self_test,
    }


def unit_caption_for(provenance, quality) -> str:
    if provenance is None:
        return "unit: unavailable"
    factor = getattr(provenance, "applied_scale_factor", None)
    factor_text = ("no scale factor applied" if factor is None
                   else f"scale factor {factor} applied once")
    return (f"All quantities are in {getattr(provenance, 'unit', 'unknown units')}; "
            f"{factor_text}. Sign policy: {getattr(provenance, 'sign_policy', 'unknown')}.")


def render(bundle) -> None:
    view = build_view(bundle)
    first, second, third, fourth = st.columns(4)
    first.metric("Items", f"{view['items']:,}")
    second.metric("Stores", f"{view['stores']:,}")
    third.metric("Item-store coverage", f"{view['coverage'] * 100:.1f}%")
    fourth.metric("Records", f"{view['rows']:,}")
    st.write(f"Release range: **{view['date_min']} -> {view['date_max']}**")
    st.write(f"Duplicate keys removed: **{view['duplicate_count']}** | "
             f"Monday violations: **{view['monday_violations']}**")
    st.write(f"Restock retention policy: **{view['restock_policy']}**")

    negative, policy, unit = st.columns(3)
    negative.metric("Negative weekly cells", f"{view['negative_cells']:,}")
    policy.metric("Sign policy", view["negative_policy"])
    unit.metric("Reported unit", view["unit"].replace("raw_sales_units", "raw sales units"))
    st.caption(f"{view['sign_badge']} · {view['preset_statement']}")
    st.caption(f"Selection: {view['chosen_model']} - {view['selection_outcome']} "
               f"({view['main_model_status']}); conditional gates: "
               f"{view['conditional'].get('b27_real_data', 'n/a')} / "
               f"{view['conditional'].get('b29_secondary', 'n/a')}")
    st.caption(view["unit_caption"])
    if view["negative_cells"]:
        with st.expander("Negative cells by week"):
            st.json(view["negative_by_week"])

    with st.expander("Exclusion counters by reason"):
        st.json(view["excluded"])
    constraint_caption("C1")
    constraint_caption("C10")
    constraint_caption("C11")