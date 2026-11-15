"""P2 Forecast panel: predicted vs actual for the demonstration item-store (D15)."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from ui.components import constraint_caption


def forecast_frame(bundle) -> pd.DataFrame:
    forecast = bundle.demo_forecast or {}
    weeks = forecast.get("weeks", [])
    predicted = forecast.get("predicted", [])
    actual = forecast.get("actual", [])
    if not weeks:
        return pd.DataFrame(columns=["week", "predicted", "actual"])
    return pd.DataFrame({"week": weeks, "predicted": predicted, "actual": actual})


def build_view(bundle) -> dict:
    frame = forecast_frame(bundle)
    key = bundle.demonstration.key
    return {
        "model": bundle.main_model,
        "item": key[0] if key else None,
        "store": key[1] if key else None,
        "points": int(len(frame)),
        "frame": frame,
        "horizon_labels": list((bundle.demo_forecast or {}).get("labels", [])),
    }


def render(bundle) -> None:
    view = build_view(bundle)
    st.write(f"Main model: **{view['model']}** | demonstration item-store: "
             f"**{view['item']} / {view['store']}**")
    if view["points"] == 0:
        st.warning("no forecast series is available for this run")
        return
    melted = view["frame"].melt(id_vars="week", var_name="series", value_name="demand")
    chart = (alt.Chart(melted)
             .mark_line(point=True)
             .encode(x=alt.X("week:O", title="lifecycle week"),
                     y=alt.Y("demand:Q", title="weekly demand"),
                     color=alt.Color("series:N", title="series"))
             .properties(height=320))
    st.altair_chart(chart, width="stretch")
    st.caption("2-in / 10-out main task. Weekly step only. [constraint C1]")
    constraint_caption("C1")