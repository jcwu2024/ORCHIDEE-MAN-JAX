from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_qsat_micro_oracle import (  # noqa: E402
    COMPILE_FLAGS,
    DEFAULT_COMPILER,
    ROOT,
    compose_harness,
)
from run_qsat_micro_oracle import _write_fortran_input, run_oracle  # noqa: E402
from fortran_gcov import (  # noqa: E402
    locate_generated_span,
    parse_gcov,
    select_arm_branch,
)


DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles/qsat_moisture"
DEFAULT_CONTRACT_CLASSES = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
DEFAULT_SOURCE_PROOFS = ROOT / "outputs/reference_mode/pft14_source_proof_evidence.json"
PROCEDURES = {"qsatcalc", "dev_qsatcalc", "qsfrict_init"}
GCOV = Path(r"C:\msys64\ucrt64\bin\gcov.exe")
def _compiler_environment(compiler: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PATH"] = str(compiler.parent) + os.pathsep + environment.get("PATH", "")
    return environment


def _generated_spans(source: Path) -> dict[str, tuple[int, int, str]]:
    from build_qsat_micro_oracle import _load_extractor

    extractor = _load_extractor()
    extracted = extractor.validate_and_extract(extractor.load_manifest(), root=ROOT)
    source_bytes = source.read_bytes()
    result: dict[str, tuple[int, int, str]] = {}
    for entry, span in extracted:
        procedure = str(entry["procedure"])
        if procedure not in PROCEDURES:
            continue
        generated_start = locate_generated_span(source_bytes, span.span_bytes)
        result[procedure] = (generated_start, int(entry["start_line"]), span.span_sha256)
    missing = PROCEDURES - set(result)
    if missing:
        raise RuntimeError(f"generated harness lacks procedure spans: {sorted(missing)}")
    return result


def run_coverage(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    compiler: Path = DEFAULT_COMPILER,
    gcov: Path = GCOV,
    contract_classes_path: Path = DEFAULT_CONTRACT_CLASSES,
    source_proofs_path: Path = DEFAULT_SOURCE_PROOFS,
) -> dict[str, Any]:
    comparison = run_oracle(output_dir=output_dir, compiler=compiler)
    if comparison["status"] != "passed":
        raise RuntimeError("QSAT numerical comparison did not pass")
    contracts = json.loads(contract_classes_path.read_text(encoding="utf-8"))
    source_proofs = json.loads(source_proofs_path.read_text(encoding="utf-8"))
    passed_source_proofs = {
        str(record["arm_id"])
        for record in source_proofs.get("records", [])
        if record.get("passed") is True
    }
    target_arms = [
        record
        for record in contracts.get("arm_classifications", [])
        if record.get("fortran_file", "").endswith("/qsat_moisture.f90")
        and record.get("fortran_procedure") in PROCEDURES
        and record.get("evidence_route")
        == "fortran_transition_oracle_and_production"
    ]
    owner_regions = {
        str(record["base_region_id"]): record
        for record in contracts.get("owner_regions", [])
        if record.get("fortran_file", "").endswith("/qsat_moisture.f90")
        and record.get("fortran_procedure") in PROCEDURES
    }

    with tempfile.TemporaryDirectory(prefix="orchidee_qsat_gcov_") as temporary:
        build_dir = Path(temporary)
        source, _ = compose_harness(build_dir)
        executable = build_dir / "qsat_gcov.exe"
        command = [
            str(compiler),
            *COMPILE_FLAGS,
            "--coverage",
            source.name,
            "-o",
            executable.name,
        ]
        subprocess.run(
            command,
            cwd=build_dir,
            env=_compiler_environment(compiler),
            text=True,
            capture_output=True,
            check=True,
        )
        input_path = build_dir / "oracle_input.dat"
        _write_fortran_input(input_path)
        run_output = build_dir / "run_output"
        run_output.mkdir()
        subprocess.run(
            [str(executable), str(input_path), str(run_output)],
            cwd=build_dir,
            env=_compiler_environment(compiler),
            text=True,
            capture_output=True,
            check=True,
        )
        notes = next(build_dir.glob("*qsat_micro_oracle_harness.gcno"))
        subprocess.run(
            [str(gcov), "-b", "-c", notes.name],
            cwd=build_dir,
            env=_compiler_environment(compiler),
            text=True,
            capture_output=True,
            check=True,
        )
        gcov_path = build_dir / f"{source.name}.gcov"
        raw_branches = parse_gcov(gcov_path)
        spans = _generated_spans(source)

        arm_records: list[dict[str, Any]] = []
        for arm in target_arms:
            procedure = str(arm["fortran_procedure"])
            generated_start, original_start, span_sha = spans[procedure]
            generated_line = generated_start + int(arm["line"]) - original_start
            branch_rows = raw_branches.get(generated_line, [])
            branch = select_arm_branch(str(arm["arm_id"]), branch_rows)
            branch_index = -1 if branch is None else int(branch["branch_index"])
            taken = 0 if branch is None else int(branch["taken"])
            arm_records.append(
                {
                    "arm_id": arm["arm_id"],
                    "base_region_id": arm["base_region_id"],
                    "procedure": procedure,
                    "original_line": arm["line"],
                    "generated_line": generated_line,
                    "gcov_branch_index": branch_index,
                    "gcov_taken": taken,
                    "procedure_span_sha256": span_sha,
                    "passed": taken > 0,
                }
            )

    missing_hits = sorted(record["arm_id"] for record in arm_records if not record["passed"])
    coverage = {
        "schema_version": 1,
        "family": "qsat_moisture",
        "mapping_policy": (
            "At -O0 GNU Fortran emits IF condition true/false as gcov branches 0/1. "
            "Array WHERE may prepend bounds/loop checks under -fcheck=all; its final two "
            "branches are the mask true/false edges. Counts are taken from a "
            "source-extracted --coverage compile unit."
        ),
        "branch_complete": not missing_hits,
        "required_arm_count": len(arm_records),
        "covered_arm_count": sum(record["passed"] for record in arm_records),
        "missing_arm_ids": missing_hits,
        "arms": arm_records,
        "compiler": str(compiler),
        "gcov": str(gcov),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "branch_coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="ascii"
    )

    evidence_records: list[dict[str, Any]] = []
    coverage_by_region: dict[str, set[str]] = defaultdict(set)
    for record in arm_records:
        if record["passed"]:
            coverage_by_region[str(record["base_region_id"])].add(str(record["arm_id"]))
    contract_by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in contracts.get("arm_classifications", []):
        if str(record.get("base_region_id")) in owner_regions:
            contract_by_region[str(record["base_region_id"])].append(record)
    for region_id, owner in sorted(owner_regions.items()):
        required = {str(record["arm_id"]) for record in contract_by_region[region_id]}
        numerical = coverage_by_region.get(region_id, set())
        source_backed = required & passed_source_proofs
        covered = numerical | source_backed
        evidence_records.append(
            {
                "owner_region_id": owner["owner_region_id"],
                "base_region_id": region_id,
                "fortran_procedure": owner["fortran_procedure"],
                "jax_owners": owner["jax_owners"],
                "comparison_asset": "outputs/reference_mode/micro_oracles/qsat_moisture/comparison.json",
                "branch_coverage_asset": "outputs/reference_mode/micro_oracles/qsat_moisture/branch_coverage.json",
                "required_arm_ids": sorted(required),
                "gcov_arm_ids": sorted(numerical),
                "source_proof_arm_ids": sorted(source_backed),
                "covered_arm_ids": sorted(covered),
                "missing_arm_ids": sorted(required - covered),
                "passed": bool(required and required == covered),
            }
        )
    owner_evidence = {
        "schema_version": 1,
        "family": "qsat_moisture",
        "complete": bool(evidence_records)
        and all(record["passed"] for record in evidence_records),
        "records": evidence_records,
    }
    (output_dir / "owner_region_evidence.json").write_text(
        json.dumps(owner_evidence, indent=2) + "\n", encoding="ascii"
    )
    if not coverage["branch_complete"] or not owner_evidence["complete"]:
        raise RuntimeError(
            f"QSAT owner coverage incomplete: branch={missing_hits}, "
            f"owner={[r['missing_arm_ids'] for r in evidence_records if not r['passed']]}"
        )
    return {"branch_coverage": coverage, "owner_evidence": owner_evidence}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run QSAT gcov owner-region evidence.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    parser.add_argument("--gcov", type=Path, default=GCOV)
    args = parser.parse_args(argv)
    try:
        result = run_coverage(
            output_dir=args.output_dir.resolve(),
            compiler=args.compiler.resolve(),
            gcov=args.gcov.resolve(),
        )
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(
        json.dumps(
            {
                "status": "passed",
                "covered_arms": result["branch_coverage"]["covered_arm_count"],
                "owner_regions": len(result["owner_evidence"]["records"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
