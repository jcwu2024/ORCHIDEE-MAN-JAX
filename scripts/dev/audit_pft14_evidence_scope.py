from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "docs" / "source_audits" / "pft14_reachable_ledger.yaml"
DEFAULT_SCOPE = ROOT / "docs" / "source_audits" / "pft14_evidence_scope.yaml"
GENERAL_ORACLE_KINDS = {"fortran_micro_oracle", "production_verified"}


def validate_scope(ledger: dict, scope: dict) -> list[str]:
    errors: list[str] = []
    ledger_ids = {entry["id"] for entry in ledger.get("entries", [])}
    scope_entries = scope.get("entries", {})
    allowed = set(scope.get("allowed_kinds", []))
    if set(scope_entries) != ledger_ids:
        missing = sorted(ledger_ids - set(scope_entries))
        extra = sorted(set(scope_entries) - ledger_ids)
        if missing:
            errors.append(f"missing evidence scope entries: {', '.join(missing)}")
        if extra:
            errors.append(f"unknown evidence scope entries: {', '.join(extra)}")
    for entry_id, evidence in scope_entries.items():
        if not isinstance(evidence, dict):
            errors.append(f"{entry_id}: evidence must be a mapping")
            continue
        kind = evidence.get("kind")
        if kind not in allowed:
            errors.append(f"{entry_id}: invalid evidence kind {kind!r}")
        if not str(evidence.get("scope", "")).strip():
            errors.append(f"{entry_id}: evidence scope is empty")
        if kind == "point_trace_anchor" and kind in GENERAL_ORACLE_KINDS:
            errors.append(f"{entry_id}: point trace cannot be a general oracle")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit evidence kind and scope for every PFT14 ledger entry.")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE)
    args = parser.parse_args(argv)
    ledger = yaml.safe_load(args.ledger.read_text(encoding="utf-8"))
    scope = yaml.safe_load(args.scope.read_text(encoding="utf-8"))
    errors = validate_scope(ledger, scope)
    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    counts: dict[str, int] = {}
    for evidence in scope["entries"].values():
        counts[evidence["kind"]] = counts.get(evidence["kind"], 0) + 1
    print(f"OK entries={len(scope['entries'])} kinds={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
