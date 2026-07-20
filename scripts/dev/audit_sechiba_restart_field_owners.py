from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.restart_io import (  # noqa: E402
    DEFAULT_SECHIBA_RESTART_SCHEMA,
    PHYSICAL_RESTART_VARIABLES,
)
from scripts.dev.scan_stomate_io_restart_fields import (  # noqa: E402
    scan_stomate_io_restart_fields,
)


DEFAULT_SOURCE_ROOT = ROOT / "fortran_source" / "ORCHIDEE" / "src_sechiba"
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "reference_mode" / "sechiba_restart_field_owners.json"
)
SUBROUTINE_RE = re.compile(r"^\s*SUBROUTINE\s+([A-Za-z0-9_]+)", re.IGNORECASE)


def _subroutine_at(source: Path, line_number: int) -> str:
    owner = "<module>"
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines()[
        :line_number
    ]:
        match = SUBROUTINE_RE.match(line)
        if match:
            owner = match.group(1)
    return owner


def build_audit(
    schema_path: Path = DEFAULT_SECHIBA_RESTART_SCHEMA,
    source_root: Path = DEFAULT_SOURCE_ROOT,
) -> dict[str, object]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    fields = set(schema["variables"]) - PHYSICAL_RESTART_VARIABLES
    owners: dict[str, list[dict[str, object]]] = {}
    for source in sorted(source_root.glob("*.f90")):
        for record in scan_stomate_io_restart_fields(source)["records"]:
            field = str(record["name"])
            if record["call"] != "restput_p" or field not in fields:
                continue
            owners.setdefault(field, []).append(
                {
                    "file": source.relative_to(ROOT).as_posix(),
                    "subroutine": _subroutine_at(source, int(record["line"])),
                    "line": int(record["line"]),
                }
            )
    missing = sorted(fields - set(owners))
    return {
        "schema_version": 1,
        "scope": "all scientific variables in the paper SECHIBA restart schema",
        "closed": not missing and len(fields) == 93,
        "counts": {
            "scientific_fields": len(fields),
            "fields_with_fortran_writer": len(owners),
        },
        "missing_fortran_writer": missing,
        "owners": {field: owners[field] for field in sorted(owners)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SECHIBA_RESTART_SCHEMA)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-closed", action="store_true")
    args = parser.parse_args()
    audit = build_audit(args.schema, args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(
        f"WROTE {args.output.resolve()} closed={audit['closed']} "
        f"counts={audit['counts']} missing={audit['missing_fortran_writer']}"
    )
    return int(args.require_closed and not audit["closed"])


if __name__ == "__main__":
    raise SystemExit(main())
