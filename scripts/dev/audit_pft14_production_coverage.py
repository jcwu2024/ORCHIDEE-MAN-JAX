from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "docs/source_audits/pft14_reachable_ledger.yaml"
DEFAULT_ORACLE_COVERAGE = ROOT / "outputs/reference_mode/fortran_oracle_coverage.json"
DEFAULT_FRAGMENT_DIR = ROOT / "docs/source_audits/production_families"
DEFAULT_OUTPUT = ROOT / "outputs/reference_mode/pft14_production_coverage.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _span_bytes(path: Path, start_line: int, end_line: int) -> bytes:
    lines = path.read_bytes().splitlines(keepends=True)
    if start_line < 1 or end_line < start_line or end_line > len(lines):
        raise ValueError(f"invalid line span {start_line}-{end_line} for {path}")
    return b"".join(lines[start_line - 1 : end_line])


def load_records(fragment_dir: Path = DEFAULT_FRAGMENT_DIR) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not fragment_dir.is_dir():
        return records
    for path in sorted(fragment_dir.glob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if document.get("schema_version") != 1:
            raise ValueError(f"{path}: schema_version must be 1")
        fragment_records = document.get("records")
        if not isinstance(fragment_records, list):
            raise ValueError(f"{path}: records must be a list")
        for record in fragment_records:
            if not isinstance(record, dict):
                raise ValueError(f"{path}: each record must be a mapping")
            records.append({**record, "fragment": path.relative_to(ROOT).as_posix()})
    return records


def _comparison_passed(path: Path) -> bool:
    if not path.is_file():
        return False
    result = json.loads(path.read_text(encoding="utf-8"))
    comparisons = result.get("comparisons")
    return bool(
        result.get("status") == "passed"
        and result.get("production_execution") is True
        and isinstance(comparisons, list)
        and comparisons
        and all(item.get("passed") is True for item in comparisons)
    )


def _validate_span(span: Any, label: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(span, dict):
        return [f"{label}: must be a mapping"]
    required = ("file", "symbol", "start_line", "end_line", "file_sha256", "span_sha256")
    missing = [field for field in required if span.get(field) in (None, "")]
    if missing:
        return [f"{label}: missing {', '.join(missing)}"]
    path = ROOT / str(span["file"])
    if not path.is_file():
        return [f"{label}: file does not exist: {span['file']}"]
    source = path.read_bytes()
    if _sha256(source) != span["file_sha256"]:
        errors.append(f"{label}: file hash drift")
    try:
        extracted = _span_bytes(path, int(span["start_line"]), int(span["end_line"]))
    except (TypeError, ValueError) as exc:
        errors.append(f"{label}: {exc}")
    else:
        if _sha256(extracted) != span["span_sha256"]:
            errors.append(f"{label}: span hash drift")
    return errors


def build_report(
    records: list[dict[str, Any]],
    ledger: dict[str, Any],
    oracle_coverage: dict[str, Any],
) -> dict[str, Any]:
    reachable = {
        entry["id"]
        for entry in ledger.get("entries", [])
        if entry.get("reachability") in {"active", "conditional"}
    }
    oracle_by_entry = {
        record["ledger_entry"]: record
        for record in oracle_coverage.get("records", [])
        if record.get("passed") is True
    }
    report_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record in records:
        entry_id = str(record.get("ledger_entry", ""))
        errors: list[str] = []
        if not entry_id:
            errors.append("ledger_entry is required")
        elif entry_id in seen:
            duplicates.add(entry_id)
        seen.add(entry_id)
        oracle = oracle_by_entry.get(entry_id)
        if entry_id not in reachable:
            errors.append("entry is not active/conditional in the ledger")
        if oracle is None:
            errors.append("entry lacks passing Stage 3 Oracle evidence")
        elif record.get("oracle_family") != oracle.get("family"):
            errors.append("oracle_family does not match strict Oracle coverage")
        if record.get("status") != "verified":
            errors.append("status must be verified")
        errors.extend(_validate_span(record.get("production_owner"), f"{entry_id}.production_owner"))
        callsites = record.get("callsites")
        if not isinstance(callsites, list) or not callsites:
            errors.append("callsites must be a non-empty list")
        else:
            for index, callsite in enumerate(callsites):
                errors.extend(_validate_span(callsite, f"{entry_id}.callsites[{index}]"))
        contracts = record.get("contracts")
        if not isinstance(contracts, dict):
            errors.append("contracts must be a mapping")
        else:
            for field in ("inputs", "outputs"):
                if not isinstance(contracts.get(field), list) or not contracts[field]:
                    errors.append(f"contracts.{field} must be a non-empty list")
            state = contracts.get("state_writeback")
            if not isinstance(state, list):
                errors.append("contracts.state_writeback must be a list")
            elif not state and not str(contracts.get("no_state_writeback_reason", "")).strip():
                errors.append("empty state_writeback requires no_state_writeback_reason")
        cases = record.get("oracle_case_ids")
        if not isinstance(cases, list) or not cases:
            errors.append("oracle_case_ids must be a non-empty list")
        asset = record.get("comparison_asset")
        if not isinstance(asset, str) or not asset:
            errors.append("comparison_asset is required")
        elif not _comparison_passed(ROOT / asset):
            errors.append("comparison_asset is missing or not a passing production execution")
        report_records.append(
            {
                "ledger_entry": entry_id,
                "fragment": record.get("fragment"),
                "oracle_family": record.get("oracle_family"),
                "comparison_asset": asset,
                "errors": errors,
                "passed": not errors,
            }
        )
    passed = {record["ledger_entry"] for record in report_records if record["passed"]}
    missing = reachable - passed
    complete = not duplicates and not missing and len(passed) == len(reachable)
    return {
        "schema_version": 1,
        "reachable_entries": len(reachable),
        "passed_entries": len(passed),
        "complete": complete,
        "duplicate_entries": sorted(duplicates),
        "missing_entries": sorted(missing),
        "records": report_records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit PFT14 production-path evidence.")
    parser.add_argument("--fragment-dir", type=Path, default=DEFAULT_FRAGMENT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--oracle-coverage", type=Path, default=DEFAULT_ORACLE_COVERAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(
        load_records(args.fragment_dir),
        yaml.safe_load(args.ledger.read_text(encoding="utf-8")),
        json.loads(args.oracle_coverage.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {args.output.resolve()} passed={report['passed_entries']}/{report['reachable_entries']}")
    return int(args.require_complete and not report["complete"])


if __name__ == "__main__":
    raise SystemExit(main())
