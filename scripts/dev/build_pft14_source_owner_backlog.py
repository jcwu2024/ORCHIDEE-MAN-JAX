from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DISPOSITION = ROOT / "outputs" / "reference_mode" / "pft14_arm_disposition.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "reference_mode" / "pft14_source_owner_backlog.json"
SCIENTIFIC_DISPOSITIONS = {"pft14_active", "pft14_conditional", "paper_static_active"}


def build_backlog(disposition: dict[str, Any]) -> dict[str, Any]:
    groups: dict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)
    scientific_records = [
        record
        for record in disposition.get("records", [])
        if record.get("disposition") in SCIENTIFIC_DISPOSITIONS
    ]
    missing = [
        record for record in scientific_records if not record.get("scientific_source_owner_present")
    ]
    for record in missing:
        groups[(record["file"], record.get("procedure"))].append(record)

    rows: list[dict[str, Any]] = []
    for (file, procedure), records in groups.items():
        branch_ids = {record["branch_id"] for record in records}
        lines = sorted({int(record["line"]) for record in records})
        rows.append(
            {
                "file": file,
                "procedure": procedure,
                "missing_owner_arms": len(records),
                "branch_sites": len(branch_ids),
                "line_min": min(lines),
                "line_max": max(lines),
                "disposition_counts": dict(Counter(record["disposition"] for record in records)),
                "example_conditions": list(
                    dict.fromkeys(str(record["source"]) for record in records)
                )[:5],
            }
        )
    rows.sort(key=lambda row: (-row["missing_owner_arms"], row["file"], str(row["procedure"])))
    return {
        "schema_version": 1,
        "scientific_reachable_arms": len(scientific_records),
        "scientific_arms_with_source_owner": len(scientific_records) - len(missing),
        "scientific_arms_missing_source_owner": len(missing),
        "procedures_missing_source_owner": len(rows),
        "closed": not missing,
        "rows": rows,
        "note": (
            "This backlog follows reachability classification. A source owner is only process "
            "mapping, not a Fortran numerical oracle or production verification."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Group PFT14 scientific arms missing process owners.")
    parser.add_argument("--disposition", type=Path, default=DEFAULT_DISPOSITION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = build_backlog(json.loads(args.disposition.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"WROTE {args.output.resolve()} missing={result['scientific_arms_missing_source_owner']} "
        f"procedures={result['procedures_missing_source_owner']} closed={result['closed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
