from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from fortran_gcov import (  # noqa: E402
    locate_generated_span,
    parse_gcov,
    select_arm_branch,
)
from fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    ROOT,
    compile_fortran,
    compiler_environment,
)
from oracle_lane_surface_enerbil_main import (  # noqa: E402
    ENERBIL,
    FAMILY,
    _compose,
    run_oracle,
)


PROCEDURES = {
    "enerbil_begin",
    "enerbil_surftemp",
    "enerbil_flux",
    "enerbil_evapveg",
}
GCOV = Path(r"C:\msys64\ucrt64\bin\gcov.exe")
DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
DEFAULT_CONTRACT_CLASSES = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
DEFAULT_SOURCE_PROOFS = ROOT / "outputs/reference_mode/pft14_source_proof_evidence.json"
# This compound IF continues onto line 1562. GNU gcov emits the terminal
# true/false edge pair on the terminal physical line, not the IF line.
CONDITION_TERMINAL_LINES = {1561: 1562}


def run_coverage(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    compiler: Path = DEFAULT_COMPILER,
    gcov: Path = GCOV,
) -> dict[str, Any]:
    comparison = run_oracle(output_dir=output_dir, compiler=compiler)
    if comparison["status"] != "passed":
        raise RuntimeError(f"{FAMILY} numerical comparison did not pass")

    contracts = json.loads(DEFAULT_CONTRACT_CLASSES.read_text(encoding="utf-8"))
    source_proofs = json.loads(DEFAULT_SOURCE_PROOFS.read_text(encoding="utf-8"))
    passed_source_proofs = {
        str(record["arm_id"])
        for record in source_proofs.get("records", [])
        if record.get("passed") is True
    }
    owner_regions = {
        str(record["base_region_id"]): record
        for record in contracts.get("owner_regions", [])
        if record.get("fortran_file", "").endswith("/enerbil.f90")
        and record.get("fortran_procedure") in PROCEDURES
        and record.get("evidence_route")
        == "fortran_transition_oracle_and_production"
    }
    target_arms = [
        record
        for record in contracts.get("arm_classifications", [])
        if str(record.get("base_region_id")) in owner_regions
        and record.get("evidence_route")
        == "fortran_transition_oracle_and_production"
    ]

    with tempfile.TemporaryDirectory(prefix="orchidee_enerbil_main_gcov_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        _compose(source)
        executable = build / "oracle_gcov.exe"
        compile_fortran(source, executable, compiler, extra_flags=("--coverage",))
        run_output = build / "fortran_outputs.csv"
        subprocess.run(
            [str(executable), str(run_output)],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
        notes = next(build.glob("*.gcno"))
        subprocess.run(
            [str(gcov), "-b", "-c", notes.name],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
        raw_branches = parse_gcov(build / f"{source.name}.gcov")
        source_bytes = source.read_bytes()
        spans = {
            procedure: extract_procedure_bytes(ENERBIL, procedure)
            for procedure in PROCEDURES
        }
        generated_spans = {
            procedure: (
                locate_generated_span(source_bytes, span.span_bytes),
                int(span.start_line),
                span.span_sha256,
            )
            for procedure, span in spans.items()
        }

        arm_records: list[dict[str, Any]] = []
        for arm in target_arms:
            procedure = str(arm["fortran_procedure"])
            generated_start, original_start, span_sha = generated_spans[procedure]
            generated_line = generated_start + int(arm["line"]) - original_start
            terminal_original_line = CONDITION_TERMINAL_LINES.get(
                int(arm["line"]), int(arm["line"])
            )
            terminal_generated_line = (
                generated_start + terminal_original_line - original_start
            )
            branch_rows = raw_branches.get(terminal_generated_line, [])
            branch = select_arm_branch(str(arm["arm_id"]), branch_rows)
            arm_records.append(
                {
                    "arm_id": arm["arm_id"],
                    "base_region_id": arm["base_region_id"],
                    "procedure": procedure,
                    "original_line": arm["line"],
                    "generated_line": generated_line,
                    "condition_terminal_original_line": terminal_original_line,
                    "condition_terminal_generated_line": terminal_generated_line,
                    "gcov_branch_index": (
                        -1 if branch is None else branch["branch_index"]
                    ),
                    "gcov_taken": 0 if branch is None else branch["taken"],
                    "procedure_span_sha256": span_sha,
                    "passed": bool(branch and branch["taken"] > 0),
                    "raw_gcov_branches": branch_rows,
                }
            )

    missing_hits = sorted(record["arm_id"] for record in arm_records if not record["passed"])
    coverage = {
        "schema_version": 1,
        "family": FAMILY,
        "branch_complete": not missing_hits,
        "required_arm_count": len(arm_records),
        "covered_arm_count": sum(record["passed"] for record in arm_records),
        "missing_arm_ids": missing_hits,
        "arms": arm_records,
        "compiler": str(compiler),
        "gcov": str(gcov),
        "mapping_policy": (
            "Simple IF arms use the final two gcov branches on the physical IF "
            "line. The line-1561 compound condition continues through line 1562, "
            "so its terminal true/false edge pair is read from line 1562."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "branch_coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="ascii"
    )

    coverage_by_region: dict[str, set[str]] = defaultdict(set)
    for record in arm_records:
        if record["passed"]:
            coverage_by_region[str(record["base_region_id"])].add(str(record["arm_id"]))
    contract_by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in contracts.get("arm_classifications", []):
        if str(record.get("base_region_id")) in owner_regions:
            contract_by_region[str(record["base_region_id"])].append(record)

    evidence_records: list[dict[str, Any]] = []
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
                "comparison_asset": (
                    f"outputs/reference_mode/micro_oracles/{FAMILY}/comparison.json"
                ),
                "branch_coverage_asset": (
                    f"outputs/reference_mode/micro_oracles/{FAMILY}/branch_coverage.json"
                ),
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
        "family": FAMILY,
        "complete": bool(evidence_records)
        and all(record["passed"] for record in evidence_records),
        "records": evidence_records,
    }
    (output_dir / "owner_region_evidence.json").write_text(
        json.dumps(owner_evidence, indent=2) + "\n", encoding="ascii"
    )
    if not coverage["branch_complete"] or not owner_evidence["complete"]:
        raise RuntimeError(
            f"{FAMILY} owner coverage incomplete: branch={missing_hits}, "
            f"owner={[r['missing_arm_ids'] for r in evidence_records if not r['passed']]}"
        )
    return {"branch_coverage": coverage, "owner_evidence": owner_evidence}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ENERBIL main gcov owner evidence.")
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
