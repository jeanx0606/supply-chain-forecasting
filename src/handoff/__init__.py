"""Adapter and verifier for the trained-model handoff.

The handoff ships a trained LightGBM model (one model, weeks 3-12 with the
horizon week as a feature), a parameter manifest and a reference predictor.
Nothing here is imported by the sample pipeline: the quoted model belongs to
Linda's experiment, and the repository run stays a stated deviation until the
model default is re-based. The package exists so the handoff can be called and
verified without importing a teammate's script and without switching a default.
"""
from .forecast import (CATEGORICAL_COLUMNS, HORIZON_WEEKS, MODEL_FILENAME,
                       N_WEEKS, OBSERVED_SALES_COLUMNS, OBSERVED_WEEKS,
                       PARAMS_FILENAME, SIMULATION_LABEL, HandoffManifest,
                       LindaLightGBM, build_features, load_manifest)

__all__ = [
    "CATEGORICAL_COLUMNS",
    "HORIZON_WEEKS",
    "MODEL_FILENAME",
    "N_WEEKS",
    "OBSERVED_SALES_COLUMNS",
    "OBSERVED_WEEKS",
    "PARAMS_FILENAME",
    "SIMULATION_LABEL",
    "HandoffManifest",
    "LindaLightGBM",
    "build_features",
    "load_manifest",
]