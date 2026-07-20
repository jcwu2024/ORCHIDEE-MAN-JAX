from __future__ import annotations

import json
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Set
from pathlib import Path
from typing import Any

from extract_fortran_micro_oracle import extract_procedure_bytes
from fortran_gcov import (
    locate_generated_span,
    parse_gcov,
    parse_gcov_line_counts,
    select_arm_branch,
)
from fortran_oracle_common import (
    COMPILE_FLAGS,
    DEFAULT_COMPILER,
    ROOT,
    compile_fortran,
    compiler_environment,
)


GCOV = Path(r"C:\msys64\ucrt64\bin\gcov.exe")
CONTRACT_CLASSES = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
SOURCE_PROOFS = ROOT / "outputs/reference_mode/pft14_source_proof_evidence.json"


def run_extracted_owner_coverage(
    *,
    family: str,
    source_file: Path,
    procedures: Set[str],
    compose: Callable[[Path], object],
    run_numerical_oracle: Callable[[Path, Path], dict[str, Any]],
    output_dir: Path,
    compiler: Path = DEFAULT_COMPILER,
    gcov: Path = GCOV,
    condition_terminal_lines: Mapping[int, int] | None = None,
    execute_harness: Callable[[Path, Path, Path], None] | None = None,
    evidence_routes: Set[str] = frozenset({"fortran_transition_oracle_and_production"}),
    arm_branch_indices: Mapping[str, int] | None = None,
    arm_execution_lines: Mapping[str, int] | None = None,
    fatal_boundary_witness_lines: Mapping[str, int] | None = None,
    coverage_extra_flags: tuple[str, ...] = ("--coverage",),
    coverage_base_flags: tuple[str, ...] = COMPILE_FLAGS,
) -> dict[str, Any]:
    comparison = run_numerical_oracle(output_dir, compiler)
    if comparison.get("status") != "passed":
        raise RuntimeError(f"{family} numerical comparison did not pass")

    contracts = json.loads(CONTRACT_CLASSES.read_text(encoding="utf-8"))
    source_proofs = json.loads(SOURCE_PROOFS.read_text(encoding="utf-8"))
    passed_source_proofs = {
        str(record["arm_id"])
        for record in source_proofs.get("records", [])
        if record.get("passed") is True
    }
    source_key = source_file.relative_to(ROOT).as_posix()
    owner_regions = {
        str(record["base_region_id"]): record
        for record in contracts.get("owner_regions", [])
        if record.get("fortran_file") == source_key
        and record.get("fortran_procedure") in procedures
        and record.get("evidence_route") in evidence_routes
    }
    target_arms = [
        record
        for record in contracts.get("arm_classifications", [])
        if str(record.get("base_region_id")) in owner_regions
        and record.get("evidence_route") in evidence_routes
    ]
    if not owner_regions:
        raise RuntimeError(f"{family} has no target owner regions")

    terminal_lines = dict(condition_terminal_lines or {})
    explicit_branch_indices = dict(arm_branch_indices or {})
    explicit_execution_lines = dict(arm_execution_lines or {})
    explicit_fatal_lines = dict(fatal_boundary_witness_lines or {})
    with tempfile.TemporaryDirectory(prefix=f"orchidee_{family}_gcov_") as temporary:
        build = Path(temporary)
        generated_source = build / "oracle.f90"
        compose(generated_source)
        executable = build / "oracle_gcov.exe"
        compile_fortran(
            generated_source,
            executable,
            compiler,
            extra_flags=coverage_extra_flags,
            base_flags=coverage_base_flags,
        )
        if execute_harness is None:
            subprocess.run(
                [str(executable), str(build / "fortran_outputs.csv")],
                cwd=build,
                env=compiler_environment(compiler),
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            execute_harness(executable, build, compiler)
        notes = next(build.glob("*.gcno"))
        subprocess.run(
            [str(gcov), "-b", "-c", notes.name],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
        raw_branches = parse_gcov(build / f"{generated_source.name}.gcov")
        raw_line_counts = parse_gcov_line_counts(
            build / f"{generated_source.name}.gcov"
        )
        generated_bytes = generated_source.read_bytes()
        spans = {
            procedure: extract_procedure_bytes(source_file, procedure)
            for procedure in procedures
        }
        generated_spans = {
            procedure: (
                locate_generated_span(generated_bytes, span.span_bytes),
                int(span.start_line),
                span.span_sha256,
            )
            for procedure, span in spans.items()
        }

        arm_records: list[dict[str, Any]] = []
        for arm in target_arms:
            procedure = str(arm["fortran_procedure"])
            generated_start, original_start, span_sha = generated_spans[procedure]
            original_line = int(arm["line"])
            generated_line = generated_start + original_line - original_start
            terminal_original_line = terminal_lines.get(original_line, original_line)
            terminal_generated_line = (
                generated_start + terminal_original_line - original_start
            )
            branch_rows = raw_branches.get(terminal_generated_line, [])
            arm_id = str(arm["arm_id"])
            witness_original_line = explicit_execution_lines.get(arm_id)
            witness_generated_line = (
                None
                if witness_original_line is None
                else generated_start + witness_original_line - original_start
            )
            witness_count = (
                None
                if witness_generated_line is None
                else raw_line_counts.get(witness_generated_line, 0)
            )
            fatal_original_line = explicit_fatal_lines.get(arm_id)
            fatal_generated_line = (
                None
                if fatal_original_line is None
                else generated_start + fatal_original_line - original_start
            )
            fatal_count = (
                None
                if fatal_generated_line is None
                else raw_line_counts.get(fatal_generated_line, 0)
            )
            if witness_original_line is not None or fatal_original_line is not None:
                branch = None
            elif arm_id in explicit_branch_indices:
                branch = next(
                    (
                        row
                        for row in branch_rows
                        if row["branch_index"] == explicit_branch_indices[arm_id]
                    ),
                    None,
                )
            else:
                branch = select_arm_branch(arm_id, branch_rows)
            gcov_passed = bool(
                witness_count > 0
                if witness_count is not None
                else branch and branch["taken"] > 0
            )
            fatal_passed = bool(fatal_count is not None and fatal_count > 0)
            source_proof_passed = arm_id in passed_source_proofs
            arm_records.append(
                {
                    "arm_id": arm["arm_id"],
                    "base_region_id": arm["base_region_id"],
                    "procedure": procedure,
                    "original_line": original_line,
                    "generated_line": generated_line,
                    "condition_terminal_original_line": terminal_original_line,
                    "condition_terminal_generated_line": terminal_generated_line,
                    "gcov_branch_index": (
                        -1 if branch is None else branch["branch_index"]
                    ),
                    "gcov_taken": 0 if branch is None else branch["taken"],
                    "execution_witness_original_line": witness_original_line,
                    "execution_witness_generated_line": witness_generated_line,
                    "execution_witness_count": witness_count,
                    "fatal_boundary_witness_original_line": fatal_original_line,
                    "fatal_boundary_witness_generated_line": fatal_generated_line,
                    "fatal_boundary_witness_count": fatal_count,
                    "gcov_passed": gcov_passed,
                    "fatal_boundary_passed": fatal_passed,
                    "source_proof_passed": source_proof_passed,
                    "procedure_span_sha256": span_sha,
                    "passed": gcov_passed or fatal_passed or source_proof_passed,
                    "raw_gcov_branches": branch_rows,
                }
            )

    missing_hits = sorted(
        record["arm_id"] for record in arm_records if not record["passed"]
    )
    coverage = {
        "schema_version": 1,
        "family": family,
        "branch_complete": not missing_hits,
        "required_arm_count": len(arm_records),
        "covered_arm_count": sum(record["passed"] for record in arm_records),
        "missing_arm_ids": missing_hits,
        "arms": arm_records,
        "compiler": str(compiler),
        "gcov": str(gcov),
        "mapping_policy": (
            "GNU Fortran -O0 --coverage final true/false edge pair. Compound "
            "conditions use the explicitly declared terminal physical line."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "branch_coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="ascii"
    )

    coverage_by_region: dict[str, set[str]] = defaultdict(set)
    fatal_by_region: dict[str, set[str]] = defaultdict(set)
    for record in arm_records:
        if record["gcov_passed"]:
            coverage_by_region[str(record["base_region_id"])].add(str(record["arm_id"]))
        if record["fatal_boundary_passed"]:
            fatal_by_region[str(record["base_region_id"])].add(str(record["arm_id"]))
    contract_by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in contracts.get("arm_classifications", []):
        if str(record.get("base_region_id")) in owner_regions:
            contract_by_region[str(record["base_region_id"])].append(record)

    evidence_records: list[dict[str, Any]] = []
    for region_id, owner in sorted(owner_regions.items()):
        required = {str(record["arm_id"]) for record in contract_by_region[region_id]}
        numerical = coverage_by_region.get(region_id, set())
        fatal_backed = fatal_by_region.get(region_id, set())
        source_backed = required & passed_source_proofs
        covered = numerical | fatal_backed | source_backed
        evidence_records.append(
            {
                "owner_region_id": owner["owner_region_id"],
                "base_region_id": region_id,
                "fortran_procedure": owner["fortran_procedure"],
                "jax_owners": owner["jax_owners"],
                "comparison_asset": (
                    f"outputs/reference_mode/micro_oracles/{family}/comparison.json"
                ),
                "branch_coverage_asset": (
                    f"outputs/reference_mode/micro_oracles/{family}/branch_coverage.json"
                ),
                "required_arm_ids": sorted(required),
                "gcov_arm_ids": sorted(numerical),
                "fatal_boundary_arm_ids": sorted(fatal_backed),
                "source_proof_arm_ids": sorted(source_backed),
                "covered_arm_ids": sorted(covered),
                "missing_arm_ids": sorted(required - covered),
                "passed": bool(required and required == covered),
            }
        )
    owner_evidence = {
        "schema_version": 1,
        "family": family,
        "complete": bool(evidence_records)
        and all(record["passed"] for record in evidence_records),
        "records": evidence_records,
    }
    (output_dir / "owner_region_evidence.json").write_text(
        json.dumps(owner_evidence, indent=2) + "\n", encoding="ascii"
    )
    if not coverage["branch_complete"] or not owner_evidence["complete"]:
        owner_missing = [
            record["missing_arm_ids"]
            for record in evidence_records
            if not record["passed"]
        ]
        raise RuntimeError(
            f"{family} owner coverage incomplete: branch={missing_hits}, "
            f"owner={owner_missing}"
        )
    return {"branch_coverage": coverage, "owner_evidence": owner_evidence}
