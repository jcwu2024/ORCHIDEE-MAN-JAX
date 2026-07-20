"""Generate the local Stage 4 STOMATE restart/data oracle interface manifest.

This is deliberately an ABI and fixture specification generator.  It does not
compile Fortran, access a server, execute JAX, or encode any STOMATE formula.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STOMATE_IO = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90"
STOMATE_DATA = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_data.f90"
DEFAULT_OUTPUT = ROOT / "docs/source_audits/stage4_restart_data_oracle_interface.json"


def _procedure_span(lines: list[str], name: str) -> tuple[int, int]:
    start = next(i for i, line in enumerate(lines) if re.match(rf"\s*SUBROUTINE\s+{name}\b", line, re.I))
    end = next(
        i for i in range(start + 1, len(lines))
        if re.match(rf"\s*END\s+SUBROUTINE\s+{name}\b", lines[i], re.I)
    )
    return start, end


def _arguments(lines: list[str], start: int) -> list[str]:
    text = ""
    for line in lines[start:]:
        text += re.sub(r"!.*$", "", line).replace("&", " ") + " "
        if ")" in text:
            break
    inside = text[text.index("(") + 1 : text.index(")")]
    return [item.strip() for item in inside.split(",") if item.strip()]


def _declarations(lines: list[str], start: int, end: int) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for offset, raw in enumerate(lines[start:end], start=start + 1):
        line = re.sub(r"!.*$", "", raw).strip()
        if "::" not in line or "INTENT" not in line.upper():
            continue
        left, right = line.split("::", 1)
        intent = re.search(r"INTENT\s*\(\s*(INOUT|IN|OUT)\s*\)", left, re.I)
        if not intent:
            continue
        kind = re.match(r"\s*(INTEGER|REAL|LOGICAL)\b", left, re.I)
        dims = re.search(r"DIMENSION\s*\(([^)]*)\)", left, re.I)
        if not kind:
            continue
        # All source declarations in these procedures are one formal per line.
        for name in re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", right):
            result[name.lower()] = {
                "name": name,
                "fortran_type": kind.group(1).upper(),
                "intent": intent.group(1).upper(),
                "dimensions": [] if dims is None else [part.strip() for part in dims.group(1).split(",")],
                "declaration_line": offset,
            }
    return result


def _interface(path: Path, procedure: str) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start, end = _procedure_span(lines, procedure)
    arguments = _arguments(lines, start)
    declarations = _declarations(lines, start, end)
    fields = []
    for ordinal, argument in enumerate(arguments, start=1):
        record = declarations.get(argument.lower())
        if record is None:
            raise ValueError(f"{path}:{procedure}: missing declaration for {argument}")
        fields.append({"ordinal": ordinal, **record})
    groups: OrderedDict[tuple[str, str, tuple[str, ...]], list[dict[str, object]]] = OrderedDict()
    for field in fields:
        key = (str(field["intent"]), str(field["fortran_type"]), tuple(field["dimensions"]))
        groups.setdefault(key, []).append(field)
    return {
        "source": str(path.relative_to(ROOT)).replace("\\", "/"),
        "procedure": procedure,
        "line_span": [start + 1, end + 1],
        "argument_count": len(fields),
        "arguments": fields,
        "argument_groups": [
            {
                "id": f"{intent.lower()}_{kind.lower()}_{'_'.join(dims) if dims else 'scalar'}",
                "intent": intent,
                "fortran_type": kind,
                "dimensions": list(dims),
                "arguments": [field["name"] for field in members],
                "ordinals": [field["ordinal"] for field in members],
            }
            for (intent, kind, dims), members in groups.items()
        ],
    }


def build_manifest() -> dict[str, object]:
    readstart = _interface(STOMATE_IO, "readstart")
    writerestart = _interface(STOMATE_IO, "writerestart")
    data = _interface(STOMATE_DATA, "data")
    writer_fields = {field["name"].lower(): field for field in writerestart["arguments"]}
    read_inputs = {"npts", "index", "lalo", "resolution", "t2m"}

    def fixture_fields(interface: dict[str, object], *, cold: bool) -> list[dict[str, object]]:
        records = []
        for field in interface["arguments"]:
            name = str(field["name"])
            if interface["procedure"] == "readstart" and name.lower() in read_inputs:
                action = "provide fixed call input"
            elif interface["procedure"] == "readstart":
                action = "observe source fallback output" if cold else "observe restored output"
            elif interface["procedure"] == "writerestart":
                action = "provide recorded literal" if not cold else "provide captured cold output or fixed call input"
            else:
                action = "provide fixed initializer input"
            records.append(
                {
                    "argument": name,
                    "ordinal": field["ordinal"],
                    "intent": field["intent"],
                    "fortran_type": field["fortran_type"],
                    "dimensions": field["dimensions"],
                    "action": action,
                }
            )
        return records
    return {
        "schema_version": 1,
        "scope": "local source-driven Stage 4 restart/data oracle harness specification only",
        "non_claims": [
            "No JAX owner closure is claimed.",
            "No numerical process formula, NetCDF layout formula, or server execution is specified.",
            "The data call is included only as the source initializer ABI required before restart fixtures.",
        ],
        "source_sha256": {
            str(STOMATE_IO.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(STOMATE_IO.read_bytes()).hexdigest(),
            str(STOMATE_DATA.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(STOMATE_DATA.read_bytes()).hexdigest(),
        },
        "interfaces": {"readstart": readstart, "writerestart": writerestart, "data": data},
        "read_write_intent_differences": [
            {
                "argument": field["name"],
                "readstart_intent": field["intent"],
                "writerestart_intent": writer_fields[field["name"].lower()]["intent"],
            }
            for field in readstart["arguments"]
            if field["name"].lower() in writer_fields
            and field["intent"] != writer_fields[field["name"].lower()]["intent"]
        ],
        "fixture_protocol": {
            "dimensions": {"npts": 1, "index": [1], "lalo": [[45.0, 2.0]], "resolution": [[1.0, 1.0]]},
            "cold_case": {
                "id": "cold_missing_restart_fields",
                "restart_file": "minimal physical template with every STOMATE restget_p variable absent",
                "call_order": ["data", "readstart", "writerestart"],
                "deterministic_fields": ["npts", "index", "lalo", "resolution", "t2m"],
                "expected_evidence": "capture every readstart output and every writerestart restput_p name/value; source fallback behavior is observed, never reconstructed",
                "fixture_fields": {
                    "data": fixture_fields(data, cold=True),
                    "readstart": fixture_fields(readstart, cold=True),
                    "writerestart": fixture_fields(writerestart, cold=True),
                },
            },
            "restart_case": {
                "id": "restart_distinct_typed_payloads",
                "restart_file": "template containing each source restget_p variable with a non-sentinel typed payload",
                "call_order": ["data", "readstart", "writerestart"],
                "deterministic_fields": ["npts", "index", "lalo", "resolution", "t2m"],
                "payload_rule": "assign a stable recorded literal per source restart variable and axis coordinate; record literals in the generated harness input manifest, not in model code",
                "expected_evidence": "capture readstart outputs and writerestart restput_p calls, including logical-to-real encodings and label-selected names",
                "fixture_fields": {
                    "data": fixture_fields(data, cold=False),
                    "readstart": fixture_fields(readstart, cold=False),
                    "writerestart": fixture_fields(writerestart, cold=False),
                },
            },
        },
        "existing_jax_boundaries_for_comparison_only": [
            "jax_orchidee.stomate.reference: StomateRestartEntryState, StomateRestartSeasonState, StomateDailyAccumulatorState, StomateOkPcRestartGasState, StomateReadstartRemainderState",
            "jax_orchidee.stomate.restart_io: read_stomate_readstart_states_from_template, write_stomate_full_writerestart_states_from_template",
            "jax_orchidee.stomate.finalize: StomateRestartWriteRequest",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {args.output} readstart={payload['interfaces']['readstart']['argument_count']} writerestart={payload['interfaces']['writerestart']['argument_count']}")


if __name__ == "__main__":
    main()
