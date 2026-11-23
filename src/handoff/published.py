"""Published result rows used as verification oracles.

These are numbers the chapter prints, not numbers this repository produced. Each
row is compared against a fresh recomputation so a quoted claim can be traced back
to the mechanism that made it. Nothing here is imported by the pipeline.
"""
from __future__ import annotations

CHAPTER7_TABLE_7_1_SOURCE = (
    "Chapter 7, Table 7.1 - Forecast accuracy, test season AW19 "
    "(Chapter_7_Methodology_and_Experimental_Results_Report)")

# Bias in the chapter is error-to-volume (sum of errors / sum of actual), not the
# mean signed error the repository reports; the verifier maps it to ``bias_ratio``.
PUBLISHED_TABLE_7_1 = {
    "source": CHAPTER7_TABLE_7_1_SOURCE,
    "tolerance": 0.001,
    "verified_models": ("LightGBM",),
    "rows": {
        "B0": {"wape": 1.274, "mae": 1.340, "rmse": 1.916,
               "bias": 0.358, "cumulative_error": 0.521},
        "B1": {"wape": 1.198, "mae": 1.260, "rmse": 1.692,
               "bias": 0.524, "cumulative_error": 0.591},
        "B2": {"wape": 0.919, "mae": 0.967, "rmse": 1.385,
               "bias": 0.058, "cumulative_error": 0.292},
        "B3": {"wape": 0.953, "mae": 1.003, "rmse": 1.481,
               "bias": 0.077, "cumulative_error": 0.295},
        "LightGBM": {"wape": 0.823, "mae": 0.866, "rmse": 1.303,
                     "bias": -0.061, "cumulative_error": 0.227},
    },
    "rounding_note": ("two cells sit on a 4-dp half boundary: bias -0.0605 is "
                      "printed as -0.061 and cumulative 0.2265 as 0.227"),
}

CHAPTER7_TABLE_7_2_SOURCE = (
    "Chapter 7, Table 7.2 - Simulation results, test season "
    "(lead time 1 week, 95% service level)")

# Recorded for traceability and for the main-setting ladder, not recomputed by the
# handoff verifier (the repo ladder runs on sample data and is a stated deviation).
PUBLISHED_TABLE_7_2 = {
    "source": CHAPTER7_TABLE_7_2_SOURCE,
    "main_setting": {"lead_time_weeks": 1, "review_weeks": 1, "service_level": 0.95},
    "cost_index": "lost + leftover units per 100 units of demand, equal 1:1 costs",
    "rows": {
        "R0": {"fill_rate": 1.000, "overstock": 0.301, "cost_index": 56.2},
        "R0b": {"fill_rate": 0.454, "overstock": 0.046, "cost_index": 58.2},
        "R1": {"fill_rate": 0.950, "overstock": 0.352, "cost_index": 72.7},
        "R2": {"fill_rate": 0.925, "overstock": 0.270, "cost_index": 52.8},
        "R3": {"fill_rate": 0.924, "overstock": 0.254, "cost_index": 49.3},
        "R4": {"fill_rate": 0.976, "overstock": 0.028, "cost_index": 6.1},
    },
    "note": "R0 fill rate is 100% by construction (replayed restocks)",
}