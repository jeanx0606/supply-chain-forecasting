"""Coverage map: every capability has at least one test module (D13)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CAPABILITY_TESTS = {
    "5.1 raw ingestion": "tests/unit/test_ingestion.py",
    "5.2 configuration": "tests/unit/test_config.py",
    "5.3 weekly index": "tests/unit/test_time_index.py",
    "5.3 week convention": "tests/unit/test_calendar.py",
    "5.4 weather aggregation": "tests/unit/test_weather.py",
    "5.5 cleaning": "tests/unit/test_cleaning.py",
    "5.6 inventory": "tests/unit/test_inventory.py",
    "5.6 restock attribution": "tests/unit/test_restock_retention.py",
    "5.7 split": "tests/unit/test_split.py",
    "5.8 features": "tests/unit/test_features.py",
    "5.8 grain": "tests/unit/test_spine.py",
    "5.9 models": "tests/unit/test_models.py",
    "5.9 baseline leakage": "tests/unit/test_baseline_no_leakage.py",
    "5.10 metrics": "tests/unit/test_metrics.py",
    "5.11 policy": "tests/unit/test_policy.py",
    "5.11 safety stock": "tests/unit/test_safety_stock.py",
    "5.12 risk": "tests/unit/test_risk.py",
    "5.12 demo selection": "tests/unit/test_demo.py",
    "5.13 orchestration": "tests/integration/test_pipeline_e2e.py",
    "5.14 sample data": "tests/unit/test_sample_determinism.py",
    "5.15 console": "tests/ui/test_panels.py",
    "5.1 channels": "tests/unit/test_channels.py",
    "5.13 artefacts": "tests/unit/test_artefacts.py",
    "5.16 documentation": "tests/docs/test_documentation_contract.py",
    "5.5 sign policy": "tests/unit/test_cleaning.py",
    "C10 unit contract": "tests/unit/test_unit_contract.py",
    "5.17 preflight": "tests/unit/test_preflight.py",
    "5.18 decision packet": "tests/unit/test_decision_packet.py",
    "5.19 chapter pack": "tests/unit/test_chapter_pack.py",
}


def test_every_capability_maps_to_an_existing_test_module():
    missing = [capability for capability, path in CAPABILITY_TESTS.items()
               if not (ROOT / path).exists()]
    assert missing == []


def test_sellout_assertions_use_the_first_crossing_only():
    offenders = []
    own_file = Path(__file__).resolve()
    for path in (ROOT / "tests").rglob("*.py"):
        if path.resolve() == own_file:
            continue
        text = path.read_text(encoding="utf-8")
        if "in find_sellout" in text or "assert 3 in" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []