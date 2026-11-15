"""Thin console shell: page chrome, sidebar, error trap and the tab loop."""
from __future__ import annotations

import streamlit as st

from src.config.loader import load_config
from ui.components import (
    assumption_banner,
    no_monetary_notice,
    render_channel_strip,
    render_provisional,
    self_test_notice,
)
from ui.panels import PANEL_REGISTRY
from ui.state import BundleCache, error_panel_text

st.set_page_config(page_title="Demand Forecasting & Replenishment",
                   layout="wide")
st.title("Demand Forecasting & Inventory Replenishment - Prototype")
st.caption("VISUELLE 2.0 (Nuna Lie) - proof of concept - external data, "
           "not SHEIN internal data")

cfg = load_config("config/settings.yaml", root=".")
cache = st.session_state.setdefault("cache", BundleCache())
force = st.sidebar.button("Run pipeline")
bundle = cache.ensure(cfg, force=force)

if cache.error:
    st.error(error_panel_text(cache.error))
    st.stop()

self_test_notice(bundle.self_test)
st.sidebar.header("Configuration")
st.sidebar.write(f"Main model: **{bundle.main_model}**")
st.sidebar.write(f"Available: {', '.join(bundle.available_models)}")
st.sidebar.write(f"Split cuts: {bundle.split.as_dict()}")
st.sidebar.write(f"Demonstration key: {bundle.demonstration.key} "
                 f"({bundle.demo_rule})")
assumption_banner()
st.sidebar.subheader("Provisional figures")
render_provisional(bundle.provisional)
st.sidebar.subheader("Channels")
render_channel_strip(bundle.channels)
no_monetary_notice()

tabs = st.tabs([panel.label for panel in PANEL_REGISTRY])
for tab, panel in zip(tabs, PANEL_REGISTRY):
    with tab:
        panel.render(bundle)