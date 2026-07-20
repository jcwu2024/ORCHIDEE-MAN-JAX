from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/dev"))

from audit_hydrol_thermosoil_production import (  # noqa: E402
    ENTRY_SPECS,
    _build_fragment,
)


def test_lane_fragment_covers_all_hydrol_thermosoil_entries_with_strict_contracts():
    fragment = _build_fragment()

    assert fragment["schema_version"] == 1
    assert {record["ledger_entry"] for record in fragment["records"]} == set(ENTRY_SPECS)
    for record in fragment["records"]:
        assert record["status"] == "verified"
        assert record["oracle_family"]
        assert record["oracle_case_ids"]
        assert record["contracts"]["inputs"]
        assert record["contracts"]["outputs"]
        assert record["contracts"]["state_writeback"]
        assert record["production_owner"]["file_sha256"]
        assert record["production_owner"]["span_sha256"]
        assert record["callsites"]


def test_checked_in_lane_assets_are_passing_and_match_generated_contract():
    fragment_path = "docs/source_audits/production_families/hydrol_thermosoil.yaml"
    comparison_path = "outputs/reference_mode/production/hydrol_thermosoil/comparison.json"
    with open(fragment_path, encoding="utf-8") as handle:
        checked_in = yaml.safe_load(handle)
    with open(comparison_path, encoding="utf-8") as handle:
        comparison = json.load(handle)

    assert checked_in == _build_fragment()
    assert comparison["status"] == "passed"
    assert comparison["production_execution"] is True
    assert comparison["comparisons"]
    assert all(item["passed"] is True for item in comparison["comparisons"])
