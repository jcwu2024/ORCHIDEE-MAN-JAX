from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "fortran_source" / "ORCHIDEE" / "src_stomate" / "stomate_io.f90"

VAR_ASSIGN_RE = re.compile(r"\bvar_name\s*=\s*'([^']+)'", re.IGNORECASE)
CALL_RE = re.compile(r"\bCALL\s+(restget_p|restput_p)\b", re.IGNORECASE)
LITERAL_RE = re.compile(r"\bCALL\s+(?:restget_p|restput_p)\s*\([^,]+,\s*'([^']+)'", re.IGNORECASE)


def scan_stomate_io_restart_fields(source: Path) -> dict[str, object]:
    """Scan Fortran restart read/write calls in ``stomate_io.f90``.

    This is a ledger helper, not a Fortran parser. It handles the two restart
    call styles used by this source file: direct string field names and
    preceding ``var_name = 'field'`` assignments.
    """

    last_var_name: str | None = None
    records: list[dict[str, object]] = []
    for lineno, line in enumerate(source.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        assigned = VAR_ASSIGN_RE.search(line)
        if assigned:
            last_var_name = assigned.group(1)

        call = CALL_RE.search(line)
        if not call:
            continue
        literal = LITERAL_RE.search(line)
        name = literal.group(1) if literal else last_var_name
        if name is None:
            name = "<unresolved>"
        records.append(
            {
                "line": lineno,
                "call": call.group(1).lower(),
                "name": name,
                "text": line.strip(),
            }
        )
    return {
        "source": str(source.relative_to(ROOT) if source.is_relative_to(ROOT) else source),
        "records": records,
        "summary": {
            "records": len(records),
            "unique_call_field_pairs": len({(record["call"], record["name"]) for record in records}),
            "unique_fields": len({record["name"] for record in records}),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    scan = scan_stomate_io_restart_fields(args.source)
    text = json.dumps(scan, indent=2, sort_keys=False)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
