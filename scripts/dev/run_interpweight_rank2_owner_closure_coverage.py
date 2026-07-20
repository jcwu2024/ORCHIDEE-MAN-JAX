from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_pft14_owner_region_evidence import build_report  # noqa: E402
from fortran_oracle_common import DEFAULT_COMPILER, ROOT, compiler_environment  # noqa: E402
from fortran_owner_coverage import (  # noqa: E402
    CONTRACT_CLASSES,
    GCOV,
    SOURCE_PROOFS,
    run_extracted_owner_coverage,
)
from oracle_lane_interpweight_rank2_owner_closure import (  # noqa: E402
    CONTRACT_MODES,
    FAMILY,
    OWNER_IDS,
    SOURCE,
    compose as compose_oracle,
    run_oracle,
)

DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
PROCEDURES = frozenset({"interpweight_2d", "interpweight_2dcont"})

# SELECT CASE declarations do not have stable GNU branch rows. Each case is
# bound to the first original statement that can only execute in that arm.
ARM_EXECUTION_LINES = {
    f"fortran_source/ORCHIDEE/src_global/interpweight.f90:{case_line}:case:case": witness
    for case_line, witness in (
        (580, 581),
        (593, 594),
        (612, 613),
        (720, 721),
        (722, 723),
        (724, 725),
        (754, 756),
        (757, 759),
        (760, 762),
        (1954, 1955),
        (1967, 1968),
        (1981, 1982),
        (2085, 2086),
        (2087, 2088),
        (2089, 2090),
        (2118, 2120),
        (2121, 2123),
        (2124, 2126),
    )
}

ARM_BRANCH_INDICES = {
    f"fortran_source/ORCHIDEE/src_global/interpweight.f90:{line}:if:{arm}": index
    for line, true_index, false_index in (
        (878, 6, 7),
        (881, 0, 1),
        (882, 0, 1),
        (885, 0, 1),
        (886, 0, 1),
        (887, 0, 1),
        (889, 0, 1),
        (892, 0, 1),
        (2213, 6, 7),
        (2218, 0, 1),
        (2219, 0, 1),
        (2222, 0, 1),
        (2223, 0, 1),
        (2224, 0, 1),
        (2226, 0, 1),
        (2227, 0, 1),
        (2228, 0, 1),
        (2229, 0, 1),
    )
    for arm, index in (("true", true_index), ("false", false_index))
}

FATAL_BOUNDARY_WITNESSES = {
    f"fortran_source/ORCHIDEE/src_global/interpweight.f90:{line}:{kind}": witness
    for line, kind, witness in (
        (719, "select_case:fallthrough", 660),
        (753, "select_case:fallthrough", 660),
        (885, "if:false", 856),
        (886, "if:true", 856),
        (887, "if:true", 856),
        (2084, "select_case:fallthrough", 2025),
        (2117, "select_case:fallthrough", 2025),
        (2222, "if:false", 2191),
        (2223, "if:true", 2191),
        (2224, "if:true", 2191),
        (2227, "if:true", 2191),
        (2228, "if:true", 2191),
    )
}


def compose(path: Path) -> object:
    return compose_oracle(path, path.parent / "extracted_original_bytes")


def execute_harness(executable: Path, build: Path, compiler: Path) -> None:
    environment = compiler_environment(compiler)
    subprocess.run(
        [str(executable), "valid", str(build / "valid.csv")],
        cwd=build,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    for mode in ("coverage_success", "coverage_nonroot"):
        subprocess.run(
            [str(executable), mode],
            cwd=build,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
    for mode in CONTRACT_MODES:
        completed = subprocess.run(
            [str(executable), mode],
            cwd=build,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        diagnostic = "IPSLERR" if "default" in mode else "is not allocated"
        if completed.returncode == 0 or diagnostic.lower() not in (
            completed.stdout + completed.stderr
        ).lower():
            raise RuntimeError(f"contract mode {mode!r} lacked its fatal witness")


def _run_target_audit(output_dir: Path, owner_evidence: dict) -> dict:
    contracts = json.loads(CONTRACT_CLASSES.read_text(encoding="utf-8"))
    target_ids = set(OWNER_IDS.values())
    owners = [
        record
        for record in contracts["owner_regions"]
        if record["owner_region_id"] in target_ids
    ]
    if {record["owner_region_id"] for record in owners} != target_ids:
        raise RuntimeError("canonical target-owner projection is incomplete")
    projection = {
        "schema_version": contracts["schema_version"],
        "source_asset": CONTRACT_CLASSES.relative_to(ROOT).as_posix(),
        "owner_regions": owners,
    }
    projection_path = output_dir / "target_contract_classes.json"
    projection_path.write_text(json.dumps(projection, indent=2) + "\n", encoding="ascii")

    evidence_root = output_dir / "formal_audit_evidence"
    family_dir = evidence_root / FAMILY
    shutil.rmtree(evidence_root, ignore_errors=True)
    family_dir.mkdir(parents=True)
    staged_path = family_dir / "owner_region_evidence.json"
    staged_path.write_text(json.dumps(owner_evidence, indent=2) + "\n", encoding="ascii")
    source_proofs = json.loads(SOURCE_PROOFS.read_text(encoding="utf-8"))
    report = build_report(
        projection,
        source_proofs,
        [(staged_path.relative_to(ROOT).as_posix(), owner_evidence)],
    )
    report_path = output_dir / "formal_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    if not report["complete"] or report["counts"]["passed_owner_regions"] != 2:
        raise RuntimeError("formal target-owner audit rejected mapped evidence")
    return report


def run_coverage(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    compiler: Path = DEFAULT_COMPILER,
    gcov: Path = GCOV,
) -> dict[str, object]:
    result = run_extracted_owner_coverage(
        family=FAMILY,
        source_file=SOURCE,
        procedures=PROCEDURES,
        compose=compose,
        run_numerical_oracle=run_oracle,
        output_dir=output_dir,
        compiler=compiler,
        gcov=gcov,
        execute_harness=execute_harness,
        arm_execution_lines=ARM_EXECUTION_LINES,
        arm_branch_indices=ARM_BRANCH_INDICES,
        fatal_boundary_witness_lines=FATAL_BOUNDARY_WITNESSES,
        coverage_extra_flags=("--coverage", "-cpp"),
    )
    evidence = result["owner_evidence"]
    comparison = json.loads((output_dir / "comparison.json").read_text(encoding="ascii"))
    evidence["fatal_and_undefined_contracts"] = comparison[
        "fatal_and_undefined_contracts"
    ]
    (output_dir / "owner_region_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="ascii"
    )
    audit = _run_target_audit(output_dir, evidence)
    return {**result, "formal_audit": audit}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    parser.add_argument("--gcov", type=Path, default=GCOV)
    arguments = parser.parse_args(argv)
    try:
        result = run_coverage(
            arguments.output_dir.resolve(),
            arguments.compiler.resolve(),
            arguments.gcov.resolve(),
        )
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        print(f"FAIL: {error}")
        return 1
    print(
        json.dumps(
            {
                "status": "passed",
                "covered_arms": result["branch_coverage"]["covered_arm_count"],
                "owner_regions": len(result["owner_evidence"]["records"]),
                "audit_complete": result["formal_audit"]["complete"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
