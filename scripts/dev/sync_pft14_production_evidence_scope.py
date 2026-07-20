from __future__ import annotations

import json

import yaml

from audit_pft14_production_coverage import ROOT


COVERAGE = ROOT / "outputs/reference_mode/pft14_production_coverage.json"
SCOPE = ROOT / "docs/source_audits/pft14_evidence_scope.yaml"


def main() -> int:
    coverage = json.loads(COVERAGE.read_text(encoding="utf-8"))
    if coverage.get("complete") is not True:
        raise RuntimeError("strict production coverage is not complete")
    passed = {
        record["ledger_entry"]: record
        for record in coverage.get("records", [])
        if record.get("passed") is True
    }
    if len(passed) != coverage.get("reachable_entries"):
        raise RuntimeError("not every reachable entry has passing production evidence")

    scope = yaml.safe_load(SCOPE.read_text(encoding="utf-8"))
    for entry_id, record in passed.items():
        asset = str(record["comparison_asset"]).replace("\\", "/")
        scope["entries"][entry_id] = {
            "kind": "production_verified",
            "scope": "executed production-path A/B against the branch-complete Fortran Oracle",
            "evidence": asset,
        }
    SCOPE.write_text(
        yaml.safe_dump(scope, sort_keys=False, allow_unicode=False, width=120),
        encoding="utf-8",
    )
    print(f"WROTE {SCOPE} production_entries={len(passed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
