from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = ROOT / "fortran_source" / "ORCHIDEE"
DEFAULT_OUTPUT = ROOT / "outputs" / "reference_mode" / "fortran_control_flow_inventory.json"
DEFAULT_DISABLED_EDGES = ROOT / "docs" / "source_audits" / "paper_static_disabled_call_edges.yaml"
SOURCE_DIRECTORIES = (
    "src_driver",
    "src_global",
    "src_parallel",
    "src_parameters",
    "src_sechiba",
    "src_stomate",
    "src_sticslai",
)

PROCEDURE_RE = re.compile(
    r"^\s*(?:(?:pure|elemental|recursive|impure)\s+)*"
    r"(program|subroutine|function)\s+([a-z][a-z0-9_]*)\b",
    re.IGNORECASE,
)
END_PROCEDURE_RE = re.compile(
    r"^\s*end\s+(?:program|subroutine|function)\b", re.IGNORECASE
)
CALL_RE = re.compile(r"\bcall\s+([a-z][a-z0-9_]*)\b", re.IGNORECASE)
INTERFACE_RE = re.compile(r"^\s*interface\s+([a-z][a-z0-9_]*)\b", re.IGNORECASE)
END_INTERFACE_RE = re.compile(r"^\s*end\s+interface\b", re.IGNORECASE)
MODULE_PROCEDURE_RE = re.compile(r"^\s*module\s+procedure\s+(.+)$", re.IGNORECASE)
BRANCH_PATTERNS = (
    ("else_if", re.compile(r"^\s*else\s*if\s*\(", re.IGNORECASE)),
    ("if", re.compile(r"^\s*(?:[a-z][a-z0-9_]*\s*:\s*)?if\s*\(", re.IGNORECASE)),
    ("elsewhere", re.compile(r"^\s*elsewhere\b", re.IGNORECASE)),
    ("where", re.compile(r"^\s*(?:[a-z][a-z0-9_]*\s*:\s*)?where\s*\(", re.IGNORECASE)),
    ("select_case", re.compile(r"^\s*select\s+case\s*\(", re.IGNORECASE)),
    ("case", re.compile(r"^\s*case(?:\s+default|\s*\()", re.IGNORECASE)),
)
FORTRAN_INTRINSIC_CALLS = {
    "cpu_time",
    "date_and_time",
    "flush",
    "random_number",
    "random_seed",
    "system",
    "system_clock",
}
IOIPSL_CALL_PREFIXES = (
    "flin",
    "flio",
    "getin",
    "hist",
    "ioconf_",
    "ioget_",
    "ipsl",
    "rest",
)
IOIPSL_CALLS = {"itau2ymds", "ju2ymds", "strlowercase", "tlen2itau", "ymds2ju"}
DYNAMIC_DISPATCH_PATTERNS = (
    ("procedure_pointer", re.compile(r"\bprocedure\s*\([^)]*\)\s*,?\s*pointer\b", re.IGNORECASE)),
    ("type_bound_call", re.compile(r"\bcall\s+[a-z][a-z0-9_]*\s*%\s*[a-z][a-z0-9_]*", re.IGNORECASE)),
    ("overloaded_operator_interface", re.compile(r"^\s*interface\s+(?:operator|assignment)\s*\(", re.IGNORECASE)),
)


@dataclass(frozen=True)
class Procedure:
    id: str
    name: str
    kind: str
    file: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class Branch:
    id: str
    kind: str
    file: str
    line: int
    procedure_id: str | None
    source: str


@dataclass(frozen=True)
class ControlFlowArm:
    id: str
    branch_id: str
    arm: str
    file: str
    line: int
    procedure_id: str | None
    source: str


@dataclass(frozen=True)
class Call:
    caller_id: str | None
    callee_name: str
    file: str
    line: int


@dataclass(frozen=True)
class FunctionReference:
    caller_id: str
    function_name: str
    file: str
    line: int


def _control_flow_arms(branch: Branch) -> list[ControlFlowArm]:
    if branch.kind in {"if", "else_if", "where"}:
        arm_names = ("true", "false")
    elif branch.kind == "case":
        arm_names = ("case",)
    else:
        # ELSEWHERE and SELECT CASE need block-level matching to identify their
        # exact predecessor. Retain them conservatively as fallthrough arms.
        arm_names = ("fallthrough",)
    return [
        ControlFlowArm(
            id=f"{branch.id}:{arm}",
            branch_id=branch.id,
            arm=arm,
            file=branch.file,
            line=branch.line,
            procedure_id=branch.procedure_id,
            source=branch.source,
        )
        for arm in arm_names
    ]


def _reachable_dynamic_dispatch_sites(
    procedures: list[Procedure], reachable_procedure_ids: set[str], project_root: Path
) -> list[dict[str, object]]:
    sites: list[dict[str, object]] = []
    lines_by_file: dict[str, list[str]] = {}
    for procedure in procedures:
        if procedure.id not in reachable_procedure_ids:
            continue
        lines = lines_by_file.setdefault(
            procedure.file,
            (project_root / procedure.file).read_text(encoding="utf-8", errors="replace").splitlines(),
        )
        for line_number in range(procedure.start_line, procedure.end_line + 1):
            source = _code_without_comment(lines[line_number - 1]).strip()
            for kind, pattern in DYNAMIC_DISPATCH_PATTERNS:
                if pattern.search(source):
                    sites.append(
                        {
                            "file": procedure.file,
                            "procedure_id": procedure.id,
                            "line": line_number,
                            "kind": kind,
                            "source": source,
                        }
                    )
    return sites


def _code_without_comment(line: str) -> str:
    """Remove a free-form Fortran comment while preserving quoted exclamation marks."""
    quote: str | None = None
    result: list[str] = []
    index = 0
    while index < len(line):
        char = line[index]
        if quote is not None:
            result.append(char)
            if char == quote:
                if index + 1 < len(line) and line[index + 1] == quote:
                    result.append(line[index + 1])
                    index += 1
                else:
                    quote = None
        elif char in {"'", '"'}:
            quote = char
            result.append(char)
        elif char == "!":
            break
        else:
            result.append(char)
        index += 1
    return "".join(result)


def _code_without_strings(line: str) -> str:
    """Mask quoted text so call-like words in messages are not parsed as code."""
    quote: str | None = None
    result: list[str] = []
    index = 0
    while index < len(line):
        char = line[index]
        if quote is not None:
            result.append(" ")
            if char == quote:
                if index + 1 < len(line) and line[index + 1] == quote:
                    result.append(" ")
                    index += 1
                else:
                    quote = None
        elif char in {"'", '"'}:
            quote = char
            result.append(" ")
        else:
            result.append(char)
        index += 1
    return "".join(result)


def _logical_lines(lines: list[str]):
    """Yield continuation-joined free-form Fortran statements and source spans."""
    pending = ""
    start_line = 0
    for line_number, raw_line in enumerate(lines, start=1):
        code = _code_without_comment(raw_line).rstrip()
        if not code.strip() and not pending:
            continue
        if not pending:
            start_line = line_number
        fragment = code.lstrip()
        if pending and fragment.startswith("&"):
            fragment = fragment[1:].lstrip()
        continued = fragment.endswith("&")
        if continued:
            fragment = fragment[:-1].rstrip()
        pending = f"{pending} {fragment}".strip()
        if not continued:
            yield start_line, line_number, pending
            pending = ""
    if pending:
        yield start_line, len(lines), pending


def _generic_interfaces(lines: list[str]) -> dict[str, list[str]]:
    interfaces: dict[str, list[str]] = {}
    current: str | None = None
    for _, _, code in _logical_lines(lines):
        if END_INTERFACE_RE.match(code):
            current = None
            continue
        match = INTERFACE_RE.match(code)
        if match:
            current = match.group(1).lower()
            interfaces.setdefault(current, [])
            continue
        procedure_match = MODULE_PROCEDURE_RE.match(code)
        if current is not None and procedure_match:
            names = re.findall(r"[a-z][a-z0-9_]*", procedure_match.group(1), re.IGNORECASE)
            interfaces[current].extend(name.lower() for name in names)
    return interfaces


def _external_provider(name: str) -> str | None:
    if name in FORTRAN_INTRINSIC_CALLS:
        return "fortran_intrinsic"
    if name.startswith("mpi_"):
        return "MPI"
    if name.startswith("xios_"):
        return "XIOS"
    if name in IOIPSL_CALLS or name.startswith(IOIPSL_CALL_PREFIXES):
        return "IOIPSL"
    return None


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def scan_source(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    *,
    project_root: Path = ROOT,
    entry_names: tuple[str, ...] = ("driver",),
    disabled_call_edges: set[tuple[str, int, str]] | None = None,
) -> dict[str, object]:
    disabled_call_edges = disabled_call_edges or set()
    files = sorted(
        path
        for directory in SOURCE_DIRECTORIES
        for path in (source_root / directory).rglob("*.f90")
    )
    procedures: list[Procedure] = []
    branches: list[Branch] = []
    calls: list[Call] = []
    statements_by_procedure: dict[str, list[tuple[int, str, str]]] = {}
    generic_interfaces: dict[str, list[str]] = {}

    for path in files:
        relative = _relative(path, project_root)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for generic_name, specific_names in _generic_interfaces(lines).items():
            generic_interfaces.setdefault(generic_name, []).extend(specific_names)
        current: dict[str, object] | None = None
        for line_number, raw_line in enumerate(lines, start=1):
            code = _code_without_comment(raw_line)
            if not code.strip():
                continue

            procedure_match = PROCEDURE_RE.match(code)
            if procedure_match:
                if current is not None:
                    procedures.append(Procedure(end_line=line_number - 1, **current))
                kind, name = (value.lower() for value in procedure_match.groups())
                procedure_id = f"{relative}:{line_number}:{name}"
                current = {
                    "id": procedure_id,
                    "name": name,
                    "kind": kind,
                    "file": relative,
                    "start_line": line_number,
                }
            current_id = None if current is None else str(current["id"])
            if current_id is not None:
                statements_by_procedure.setdefault(current_id, []).append(
                    (line_number, relative, code)
                )

            for call_match in CALL_RE.finditer(_code_without_strings(code)):
                calls.append(
                    Call(
                        caller_id=current_id,
                        callee_name=call_match.group(1).lower(),
                        file=relative,
                        line=line_number,
                    )
                )

            for kind, pattern in BRANCH_PATTERNS:
                if pattern.match(code):
                    branches.append(
                        Branch(
                            id=f"{relative}:{line_number}:{kind}",
                            kind=kind,
                            file=relative,
                            line=line_number,
                            procedure_id=current_id,
                            source=" ".join(code.strip().split()),
                        )
                    )
                    break

            if current is not None and END_PROCEDURE_RE.match(code):
                procedures.append(Procedure(end_line=line_number, **current))
                current = None

        if current is not None:
            procedures.append(Procedure(end_line=len(lines), **current))

    names: dict[str, list[str]] = {}
    for procedure in procedures:
        names.setdefault(procedure.name, []).append(procedure.id)
    generic_interfaces = {
        name: sorted(set(specific_names))
        for name, specific_names in generic_interfaces.items()
    }
    function_names = {
        procedure.name for procedure in procedures if procedure.kind == "function"
    }
    callable_function_names = set(function_names)
    callable_function_names.update(
        generic_name
        for generic_name, specific_names in generic_interfaces.items()
        if any(name in function_names for name in specific_names)
    )
    function_pattern = (
        re.compile(
            r"\b(" + "|".join(sorted(map(re.escape, callable_function_names), key=len, reverse=True)) + r")\s*\(",
            re.IGNORECASE,
        )
        if callable_function_names
        else None
    )
    function_references: list[FunctionReference] = []
    if function_pattern is not None:
        for caller_id, statements in statements_by_procedure.items():
            caller_name = next(
                procedure.name for procedure in procedures if procedure.id == caller_id
            )
            for line_number, relative, code in statements:
                masked = _code_without_strings(code)
                if PROCEDURE_RE.match(masked):
                    continue
                for match in function_pattern.finditer(masked):
                    function_name = match.group(1).lower()
                    if function_name != caller_name:
                        function_references.append(
                            FunctionReference(
                                caller_id=caller_id,
                                function_name=function_name,
                                file=relative,
                                line=line_number,
                            )
                        )
    unresolved = sorted(
        {
            call.callee_name
            for call in calls
            if call.callee_name not in names and call.callee_name not in generic_interfaces
        }
    )
    duplicate_names = {name: ids for name, ids in names.items() if len(ids) > 1}
    calls_by_caller: dict[str, list[Call]] = {}
    for call in calls:
        if call.caller_id is not None:
            calls_by_caller.setdefault(call.caller_id, []).append(call)
    function_references_by_caller: dict[str, list[FunctionReference]] = {}
    for reference in function_references:
        function_references_by_caller.setdefault(reference.caller_id, []).append(reference)
    reachable_procedure_ids = {
        procedure_id
        for entry_name in entry_names
        for procedure_id in names.get(entry_name.lower(), [])
    }
    frontier = list(reachable_procedure_ids)
    reachable_unresolved_calls: set[str] = set()
    applied_disabled_edges: set[tuple[str, int, str]] = set()
    while frontier:
        caller_id = frontier.pop()
        for call in calls_by_caller.get(caller_id, []):
            edge = (call.file, call.line, call.callee_name)
            if edge in disabled_call_edges:
                applied_disabled_edges.add(edge)
                continue
            candidate_names = generic_interfaces.get(call.callee_name, [call.callee_name])
            candidate_ids = [
                procedure_id
                for candidate_name in candidate_names
                for procedure_id in names.get(candidate_name, [])
            ]
            if not candidate_ids:
                reachable_unresolved_calls.add(call.callee_name)
                continue
            for candidate_id in candidate_ids:
                if candidate_id not in reachable_procedure_ids:
                    reachable_procedure_ids.add(candidate_id)
                    frontier.append(candidate_id)
        for reference in function_references_by_caller.get(caller_id, []):
            candidate_names = generic_interfaces.get(
                reference.function_name, [reference.function_name]
            )
            for candidate_name in candidate_names:
                for candidate_id in names.get(candidate_name, []):
                    if candidate_id not in reachable_procedure_ids:
                        reachable_procedure_ids.add(candidate_id)
                        frontier.append(candidate_id)
    reachable_branches = [
        branch for branch in branches if branch.procedure_id in reachable_procedure_ids
    ]
    control_flow_arms = [
        arm for branch in branches for arm in _control_flow_arms(branch)
    ]
    target_reachable_arm_ids = [
        arm.id
        for branch in reachable_branches
        for arm in _control_flow_arms(branch)
    ]
    reachable_external_calls = {
        name: _external_provider(name)
        for name in sorted(reachable_unresolved_calls)
        if _external_provider(name) is not None
    }
    reachable_unresolved_project_calls = sorted(
        name for name in reachable_unresolved_calls if name not in reachable_external_calls
    )
    reachable_dynamic_dispatch_sites = _reachable_dynamic_dispatch_sites(
        procedures, reachable_procedure_ids, project_root
    )
    call_graph_limitations = []
    if reachable_dynamic_dispatch_sites:
        call_graph_limitations.append(
            "Reachable dynamic-dispatch or overloaded-operator sites require explicit target resolution."
        )
    return {
        "schema_version": 2,
        "source_root": _relative(source_root, project_root),
        "source_directories": list(SOURCE_DIRECTORIES),
        "counts": {
            "files": len(files),
            "procedures": len(procedures),
            "branches": len(branches),
            "control_flow_arms": len(control_flow_arms),
            "calls": len(calls),
            "function_references": len(function_references),
            "unresolved_callee_names": len(unresolved),
            "duplicate_procedure_names": len(duplicate_names),
            "reachable_procedures": len(reachable_procedure_ids),
            "reachable_branches": len(reachable_branches),
            "target_reachable_arms": len(target_reachable_arm_ids),
            "reachable_unresolved_callee_names": len(reachable_unresolved_calls),
            "reachable_external_callee_names": len(reachable_external_calls),
            "reachable_unresolved_project_callee_names": len(reachable_unresolved_project_calls),
        },
        "procedures": [asdict(record) for record in procedures],
        "branches": [asdict(record) for record in branches],
        "control_flow_arms": [asdict(record) for record in control_flow_arms],
        "calls": [asdict(record) for record in calls],
        "function_references": [asdict(record) for record in function_references],
        "unresolved_callee_names": unresolved,
        "duplicate_procedure_names": duplicate_names,
        "generic_interfaces": generic_interfaces,
        "entry_names": [name.lower() for name in entry_names],
        "reachable_procedure_ids": sorted(reachable_procedure_ids),
        "reachable_branch_ids": [branch.id for branch in reachable_branches],
        "target_reachable_arm_ids": target_reachable_arm_ids,
        "reachable_unresolved_callee_names": sorted(reachable_unresolved_calls),
        "reachable_external_calls": reachable_external_calls,
        "reachable_unresolved_project_callee_names": reachable_unresolved_project_calls,
        "reachable_dynamic_dispatch_sites": reachable_dynamic_dispatch_sites,
        "call_graph_capability_notes": [
            "Explicit CALL statements, declared function references, and generic MODULE PROCEDURE interfaces are resolved.",
            "Procedure pointers, type-bound CALL dispatch, and overloaded operator interfaces are scanned in reachable procedures.",
        ],
        "call_graph_limitations": call_graph_limitations,
        "disabled_call_edges": [
            {"file": file, "line": line, "callee": callee}
            for file, line, callee in sorted(disabled_call_edges)
        ],
        "applied_disabled_call_edges": [
            {"file": file, "line": line, "callee": callee}
            for file, line, callee in sorted(applied_disabled_edges)
        ],
        "unmatched_disabled_call_edges": [
            {"file": file, "line": line, "callee": callee}
            for file, line, callee in sorted(disabled_call_edges - applied_disabled_edges)
        ],
    }


def load_disabled_call_edges(path: Path = DEFAULT_DISABLED_EDGES) -> set[tuple[str, int, str]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    run_def_path = ROOT / document["run_def"]
    run_def_values: dict[str, str] = {}
    for raw_line in run_def_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        run_def_values[key.strip().upper()] = value.strip().lower()
    for entry in document.get("edges", []):
        for assertion in str(entry.get("config", "")).split(";"):
            key, separator, expected = assertion.strip().partition("=")
            if not separator:
                raise ValueError(f"invalid disabled-edge config assertion: {assertion!r}")
            actual = run_def_values.get(key.strip().upper())
            if actual != expected.strip().lower():
                raise ValueError(
                    f"disabled-edge config mismatch for {key.strip()}: expected {expected.strip()!r}, got {actual!r}"
                )
    return {
        (str(entry["file"]).replace("\\", "/"), int(entry["line"]), str(entry["callee"]).lower())
        for entry in document.get("edges", [])
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory Fortran procedures, calls, and branch sites.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--entry", action="append", dest="entry_names")
    parser.add_argument("--disabled-edges", type=Path, default=DEFAULT_DISABLED_EDGES)
    parser.add_argument("--no-static-pruning", action="store_true")
    args = parser.parse_args(argv)
    inventory = scan_source(
        args.source_root,
        entry_names=tuple(args.entry_names or ("driver",)),
        disabled_call_edges=(
            set() if args.no_static_pruning else load_disabled_call_edges(args.disabled_edges)
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    counts = inventory["counts"]
    print(
        f"WROTE {args.output.resolve()} "
        f"files={counts['files']} procedures={counts['procedures']} "
        f"branches={counts['branches']} calls={counts['calls']} "
        f"arms={counts['control_flow_arms']} "
        f"reachable_procedures={counts['reachable_procedures']} "
        f"reachable_branches={counts['reachable_branches']} "
        f"target_reachable_arms={counts['target_reachable_arms']} "
        f"reachable_external={counts['reachable_external_callee_names']} "
        f"reachable_project_unresolved={counts['reachable_unresolved_project_callee_names']}"
    )
    if inventory["unmatched_disabled_call_edges"]:
        print(f"UNMATCHED_DISABLED_EDGES {len(inventory['unmatched_disabled_call_edges'])}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
