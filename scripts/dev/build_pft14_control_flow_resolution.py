from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = ROOT / "outputs" / "reference_mode" / "fortran_control_flow_inventory.json"
DEFAULT_PROCESS_LEDGER = ROOT / "docs" / "source_audits" / "pft14_reachable_ledger.yaml"
DEFAULT_EVIDENCE_SCOPE = ROOT / "docs" / "source_audits" / "pft14_evidence_scope.yaml"
DEFAULT_PROCESS_EXTENSIONS = (
    ROOT / "docs" / "source_audits" / "pft14_process_inventory_extensions.yaml"
)
DEFAULT_SOURCE_OWNER_GLOB = "pft14_source_owners_*.yaml"
DEFAULT_OUTPUT = ROOT / "outputs" / "reference_mode" / "pft14_control_flow_resolution.json"


def _structural_resolution(source: str) -> str | None:
    lowered = source.lower()
    if re.search(r"\bif\s*\(\s*(?:l_error\b|ier\s*/=|ierr\s*/=)", lowered):
        return "valid_input_error_guard"
    if any(name in lowered for name in ("printlev", "long_print", "printlev_loc")):
        return "diagnostic_only"
    return None


def _line_numbers(value: str | int) -> set[int]:
    result: set[int] = set()
    for item in str(value).split(","):
        bounds = [int(part) for part in item.strip().split("-")]
        result.update(range(bounds[0], bounds[-1] + 1))
    return result


def build_resolution(
    inventory: dict,
    process_ledger: dict,
    evidence_scope: dict,
    process_extensions: dict | None = None,
    source_owners: dict | None = None,
) -> dict:
    procedures = {item["id"]: item for item in inventory["procedures"]}
    branches = {item["id"]: item for item in inventory["branches"]}
    reachable = set(inventory["reachable_branch_ids"])
    process_index: dict[tuple[str, str], list[tuple[dict, set[int]]]] = {}
    for entry in process_ledger["entries"]:
        key = (
            str(entry["fortran_file"]).replace("\\", "/"),
            str(entry["fortran_subroutine"]).lower(),
        )
        process_index.setdefault(key, []).append((entry, _line_numbers(entry["fortran_lines"])))
    extension_index = {
        (
            str(entry["fortran_file"]).replace("\\", "/"),
            str(entry["fortran_subroutine"]).lower(),
        ): entry
        for entry in (process_extensions or {}).get("entries", [])
    }
    source_owner_index: dict[tuple[str, str], list[tuple[dict, set[int]]]] = {}
    for entry in (source_owners or {}).get("entries", []):
        key = (
            str(entry["fortran_file"]).replace("\\", "/"),
            str(entry["fortran_subroutine"]).lower(),
        )
        source_owner_index.setdefault(key, []).append(
            (entry, _line_numbers(entry["fortran_lines"]))
        )

    rows = []
    for branch_id in sorted(reachable):
        branch = branches[branch_id]
        procedure = procedures.get(branch["procedure_id"])
        matches = []
        if procedure is not None:
            key = (branch["file"], procedure["name"].lower())
            matches = [entry for entry, lines in process_index.get(key, []) if branch["line"] in lines]
        process_ids = [entry["id"] for entry in matches]
        evidence_kinds = [
            evidence_scope["entries"][entry_id]["kind"]
            for entry_id in process_ids
            if entry_id in evidence_scope["entries"]
        ]
        extension = None
        owner_matches: list[dict] = []
        if procedure is not None:
            procedure_key = (branch["file"], procedure["name"].lower())
            extension = extension_index.get(procedure_key)
            owner_matches = [
                entry
                for entry, lines in source_owner_index.get(procedure_key, [])
                if branch["line"] in lines
            ]
        resolution = "process_span_mapped" if process_ids else _structural_resolution(branch["source"])
        if resolution is None and owner_matches:
            resolution = (
                "source_owner_mapped"
                if any(entry["status"] == "implemented" for entry in owner_matches)
                else "source_owner_open_gap"
            )
        if resolution is None and extension is not None:
            resolution = (
                "process_inventory_open_gap"
                if extension.get("gap")
                else "process_inventory_classified"
            )
        rows.append(
            {
                "control_flow_id": branch_id,
                "file": branch["file"],
                "procedure": None if procedure is None else procedure["name"],
                "line": branch["line"],
                "kind": branch["kind"],
                "source": branch["source"],
                "process_ledger_ids": process_ids,
                "evidence_kinds": evidence_kinds,
                "process_inventory_id": None if extension is None else extension["id"],
                "process_inventory_category": (
                    None if extension is None else extension["category"]
                ),
                "process_inventory_gap": None if extension is None else extension.get("gap"),
                "source_owner_ids": [entry["id"] for entry in owner_matches],
                "source_owner_statuses": [entry["status"] for entry in owner_matches],
                "jax_owners": [
                    entry["jax_owner"] for entry in owner_matches if entry.get("jax_owner")
                ],
                "resolution": resolution or "unmapped",
            }
        )
    resolution_counts = Counter(row["resolution"] for row in rows)
    unmapped_by_procedure = Counter(
        (row["file"], row["procedure"]) for row in rows if row["resolution"] == "unmapped"
    )
    open_count = resolution_counts.get("unmapped", 0) + resolution_counts.get(
        "process_inventory_open_gap", 0
    ) + resolution_counts.get("source_owner_open_gap", 0)
    return {
        "schema_version": 1,
        "reachable_denominator": len(rows),
        "resolution_counts": dict(resolution_counts),
        "open_denominator": open_count,
        "closed": open_count == 0,
        "rows": rows,
        "largest_unmapped_procedures": [
            {"file": file, "procedure": procedure, "branches": count}
            for (file, procedure), count in unmapped_by_procedure.most_common(100)
        ],
        "note": (
            "process_span_mapped and process_inventory_classified are inventory progress only. "
            "source_owner_mapped binds an exact source span to an implemented JAX owner. "
            "process_inventory_open_gap and source_owner_open_gap remain open. valid_input_error_guard and "
            "diagnostic_only are source-text classifications outside valid scientific state. "
            "None of these establish a Fortran numerical oracle for process branches."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Map every reachable Fortran branch ID to current process evidence.")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--process-ledger", type=Path, default=DEFAULT_PROCESS_LEDGER)
    parser.add_argument("--evidence-scope", type=Path, default=DEFAULT_EVIDENCE_SCOPE)
    parser.add_argument("--process-extensions", type=Path, default=DEFAULT_PROCESS_EXTENSIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    source_owner_entries = []
    for path in sorted((ROOT / "docs" / "source_audits").glob(DEFAULT_SOURCE_OWNER_GLOB)):
        source_owner_entries.extend(yaml.safe_load(path.read_text(encoding="utf-8")).get("entries", []))
    result = build_resolution(
        json.loads(args.inventory.read_text(encoding="utf-8")),
        yaml.safe_load(args.process_ledger.read_text(encoding="utf-8")),
        yaml.safe_load(args.evidence_scope.read_text(encoding="utf-8")),
        yaml.safe_load(args.process_extensions.read_text(encoding="utf-8")),
        {"entries": source_owner_entries},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"WROTE {args.output.resolve()} denominator={result['reachable_denominator']} "
        f"resolution={result['resolution_counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
