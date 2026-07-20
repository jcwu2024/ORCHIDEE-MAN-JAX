from __future__ import annotations

import json

import yaml

from extract_fortran_micro_oracle import DEFAULT_MANIFEST, ROOT, load_manifest


COVERAGE = ROOT / "outputs/reference_mode/fortran_oracle_coverage.json"
SCOPE = ROOT / "docs/source_audits/pft14_evidence_scope.yaml"


def main() -> int:
    coverage = json.loads(COVERAGE.read_text(encoding="utf-8"))
    if not coverage.get("complete"):
        raise RuntimeError("strict Fortran Oracle coverage is not complete")
    records = coverage.get("records", [])
    passed = {record["ledger_entry"]: record for record in records if record.get("passed")}
    if len(passed) != coverage.get("reachable_entries"):
        raise RuntimeError("not every reachable entry has a passing strict record")

    manifest = load_manifest(DEFAULT_MANIFEST)
    families = {family["id"]: family for family in manifest["families"]}
    scope = yaml.safe_load(SCOPE.read_text(encoding="utf-8"))
    for entry_id, record in passed.items():
        family_id = record["family"]
        family = families[family_id]
        result_asset = family.get(
            "result_asset",
            f"outputs/reference_mode/micro_oracles/{family_id}/comparison.json",
        )
        result_path = ROOT / result_asset
        result = json.loads(result_path.read_text(encoding="ascii"))
        if result.get("status") != "passed" or not result.get("comparisons"):
            raise RuntimeError(f"{family_id}: result asset is not passing")
        scope["entries"][entry_id] = {
            "kind": "fortran_micro_oracle",
            "scope": f"branch-complete source-extracted Fortran Oracle family {family_id}",
            "evidence": result_asset.replace("\\", "/"),
        }
    SCOPE.write_text(
        yaml.safe_dump(scope, sort_keys=False, allow_unicode=False, width=120),
        encoding="utf-8",
    )
    print(f"WROTE {SCOPE} oracle_entries={len(passed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
