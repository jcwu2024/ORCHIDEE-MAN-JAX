from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from scripts.dev.stomate_oracle_arm_coverage import audit_document

ROOT = Path(__file__).resolve().parents[2]
FRAGMENT = ROOT / "docs/source_audits/oracle_families/stomate_season_phenology.yaml"


def test_fragment_keeps_incomplete_families_pending():
    document = yaml.safe_load(FRAGMENT.read_text(encoding="utf-8"))
    assert document["schema_version"] == 2
    assert [family["id"] for family in document["families"]] == [
        "stomate_season_memory",
        "stomate_phenology",
    ]
    report = audit_document(document, ROOT)
    assert report["totals"]["entries"] == 2
    assert report["totals"]["branch_complete"] == 2
    assert report["totals"]["missing_arms"] == 0


def test_fragment_source_and_span_hashes_are_current():
    document = yaml.safe_load(FRAGMENT.read_text(encoding="utf-8"))
    for family in document["families"]:
        for procedure in family["procedures"]:
            path = ROOT / procedure["source_file"]
            lines = path.read_bytes().splitlines(keepends=True)
            span = b"".join(lines[procedure["start_line"] - 1 : procedure["end_line"]])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == procedure["expected_source_sha256"]
            assert hashlib.sha256(span).hexdigest() == procedure["expected_span_sha256"]


def test_cached_comparisons_are_passing_and_complete():
    document = yaml.safe_load(FRAGMENT.read_text(encoding="utf-8"))
    for family in document["families"]:
        result_path = ROOT / family["procedures"][0]["compilation"]["result_asset"]
        result = json.loads(result_path.read_text(encoding="ascii"))
        assert result["family"] == family["id"]
        assert result["status"] == "passed"
        assert {item["name"] for item in result["comparisons"]} == set(family["output_contract"])
        assert all(item["passed"] for item in result["comparisons"])
