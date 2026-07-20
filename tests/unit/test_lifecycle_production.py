from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.dev.audit_lifecycle_production import COMPARISON, ENTRIES, FRAGMENT  # noqa: E402


def test_lifecycle_production_asset_is_passing_real_execution() -> None:
    result = json.loads(COMPARISON.read_text(encoding="utf-8"))
    assert result["status"] == "passed"
    assert result["production_execution"] is True
    assert result["execution"]["trace_inputs"] is False
    assert result["execution"]["restart_netcdf_roundtrip"] is True
    assert {item["name"] for item in result["comparisons"]} == set(ENTRIES)
    for item in result["comparisons"]:
        assert item["passed"] is True
        assert item["stage3_oracle"]["passed"] is True
        assert item["checks"]
        assert all(check["passed"] is True for check in item["checks"])


def test_lifecycle_manifest_has_exact_lane_entries() -> None:
    document = yaml.safe_load(FRAGMENT.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    records = document["records"]
    assert {record["ledger_entry"] for record in records} == set(ENTRIES)
    assert all(record["status"] == "verified" for record in records)
    assert all(record["comparison_asset"] == COMPARISON.relative_to(ROOT).as_posix() for record in records)
