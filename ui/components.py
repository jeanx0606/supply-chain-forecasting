"""Shared console components: assumption banner, provisional marks, constraint captions."""
from __future__ import annotations

import pandas as pd
import streamlit as st

CONSTRAINTS = {
    "C1": "C1 - no daily data exists; every output is weekly.",
    "C2": "C2 - item-store coverage is partial; item-level aggregates are the main figure.",
    "C3": "C3 - inventory is RECONSTRUCTED from restock minus sales, not observed; "
          "I0 is assumed and only the trajectory is observable.",
    "C4": "C4 - stock-out is NOT measurable; every stock-out figure is a PROXY.",
    "C5": "C5 - replenishment quantity has a real benchmark; timing does not "
          "(no order dates, lead times or costs).",
    "C10": "C10 - every reported quantity is in raw sales units; no scale factor "
           "is applied on the main path.",
    "C11": "C11 - negative weekly sales follow one declared policy; the policy is "
           "an interpretation, not a data correction.",
}


def fmt_metric(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def assumption_banner() -> None:
    st.warning("Lead time, service level and order cycle are ASSUMED values, "
               "not measured from the data (C5).")


def provisional_text(refs) -> list:
    return [f"PROVISIONAL - {ref.rep_id} - {ref.confirmer_role}" for ref in refs]


def render_provisional(refs) -> None:
    for line in provisional_text(refs):
        st.caption(line)


def constraint_caption(code: str) -> None:
    st.caption(CONSTRAINTS.get(code, code))


def channel_frame(channels) -> pd.DataFrame:
    rows = []
    for status in channels:
        rows.append({"channel": status.name, "state": status.value,
                     "reason": status.reason, "file": status.file})
    return pd.DataFrame(rows)


def render_channel_strip(channels) -> None:
    st.subheader("Channel status")
    st.dataframe(channel_frame(channels), hide_index=True, width="stretch")
    st.caption("No declared channel may be silently absent: consumed, deferred or skipped.")


def self_test_notice(self_test: bool) -> None:
    if self_test:
        st.error("SELF-TEST DATA - figures are not results and must not be quoted.")


def no_monetary_notice() -> None:
    st.caption("Prices are normalised; no monetary value is computed anywhere (C6).")


def unit_caption(provenance) -> str:
    """One sentence derived from the published provenance object (never a literal)."""
    if provenance is None:
        return "unit: unavailable"
    factor = getattr(provenance, "applied_scale_factor", None)
    factor_text = ("no scale factor applied" if factor is None
                   else f"scale factor {factor} applied once")
    return (f"All quantities are in {getattr(provenance, 'unit', 'unknown units')}; "
            f"{factor_text}. Sign policy: {getattr(provenance, 'sign_policy', 'unknown')}.")
