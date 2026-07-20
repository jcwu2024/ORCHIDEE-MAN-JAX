from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "outputs/reference_mode/production/stomate/comparison.json"
MANIFEST = ROOT / "docs/source_audits/production_families/stomate.yaml"


def test_stomate_production_evidence_is_complete_and_passing():
    evidence = json.loads(ASSET.read_text(encoding="utf-8"))
    assert evidence["status"] == "passed"
    assert evidence["production_execution"] is True
    assert evidence["execution"]["half_hour_steps_per_day"] == 48
    assert evidence["execution"]["trace_inputs"] is False
    assert len(evidence["comparisons"]) == 15
    assert all(item["passed"] for item in evidence["comparisons"])
    assert all(item["checks"] and all(check["passed"] for check in item["checks"]) for item in evidence["comparisons"])


def test_stomate_production_manifest_matches_evidence_entries():
    evidence = json.loads(ASSET.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    evidence_entries = {item["ledger_entry"] for item in evidence["comparisons"]}
    manifest_entries = {item["ledger_entry"] for item in manifest["records"]}
    assert manifest_entries == evidence_entries
    assert len(manifest_entries) == 15
