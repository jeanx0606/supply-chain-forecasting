"""Outputs and config must not imply image features (Chapter 6 §6.3)."""
from __future__ import annotations

import json
from pathlib import Path

from src.report.chapter import build_pack

ROOT = Path(__file__).resolve().parents[2]


def test_no_output_or_config_label_mentions_image(sample_bundle):
    bundle = sample_bundle.to_dict()
    payload = json.dumps({"bundle": bundle, "pack": build_pack(bundle)},
                         default=str).lower()
    assert "image" not in payload
    settings = (ROOT / "config" / "settings.yaml").read_text(encoding="utf-8").lower()
    assert "image" not in settings
    for path in (ROOT / "ui").rglob("*.py"):
        assert "image" not in path.read_text(encoding="utf-8").lower(), path.name