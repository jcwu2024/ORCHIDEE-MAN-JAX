from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    ROOT,
    compile_fortran,
    compiler_environment,
)
from fortran_owner_coverage import GCOV, run_extracted_owner_coverage  # noqa: E402
from fortran_gcov import locate_generated_span  # noqa: E402
from extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from oracle_lane_module_llxy_owner import (  # noqa: E402
    FAMILY,
    PROCEDURES,
    SOURCE,
    compose,
    run_oracle,
)

DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
ERROR_MODES = (
    "uninit_forward",
    "invalid_forward",
    "uninit_inverse",
    "invalid_inverse",
    "gauss_missing",
)
COVERAGE_FLAGS = (
    "-std=f2008",
    "-fdefault-real-8",
    "-ffree-line-length-none",
    "-O0",
    "-Wall",
    "-Wextra",
)
SELECT_STATEMENT_LINES = {
    609: 642,
    611: 612,
    614: 615,
    617: 618,
    620: 621,
    623: 624,
    626: 627,
    629: 630,
    632: 633,
    635: 636,
    638: 639,
    641: 642,
    667: 697,
    669: 670,
    672: 673,
    675: 676,
    678: 679,
    681: 682,
    684: 685,
    687: 688,
    690: 691,
    693: 694,
    696: 697,
}
GCOV_LINE = re.compile(r"^\s*([^:]+):\s*(\d+):")


def execute(executable: Path, build: Path, compiler: Path) -> None:
    environment = compiler_environment(compiler)
    subprocess.run(
        [str(executable), str(build / "coverage.csv"), "coverage"],
        cwd=build,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def _select_statement_counts(compiler: Path, gcov: Path) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix="orchidee_module_llxy_select_") as tmp:
        build = Path(tmp)
        source = build / "oracle.f90"
        compose(source)
        executable = build / "oracle.exe"
        compile_fortran(
            source,
            executable,
            compiler,
            extra_flags=("--coverage",),
            base_flags=COVERAGE_FLAGS,
        )
        execute(executable, build, compiler)
        notes = next(build.glob("*.gcno"))
        subprocess.run(
            [str(gcov), "-b", "-c", notes.name],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
        line_counts: dict[int, int] = {}
        for line in (
            (build / "oracle.f90.gcov")
            .read_text(encoding="utf-8", errors="replace")
            .splitlines()
        ):
            match = GCOV_LINE.match(line)
            if match and match.group(1).strip().isdigit():
                line_counts[int(match.group(2))] = int(match.group(1).strip())
        generated = source.read_bytes()
        procedure_starts = {
            procedure: (
                locate_generated_span(
                    generated, extract_procedure_bytes(SOURCE, procedure).span_bytes
                ),
                extract_procedure_bytes(SOURCE, procedure).start_line,
            )
            for procedure in ("latlon_to_ij", "ij_to_latlon")
        }
        counts: dict[str, int] = {}
        for (
            original_arm_line,
            original_statement_line,
        ) in SELECT_STATEMENT_LINES.items():
            procedure = "latlon_to_ij" if original_arm_line < 650 else "ij_to_latlon"
            generated_start, original_start = procedure_starts[procedure]
            generated_statement = (
                generated_start + original_statement_line - original_start
            )
            arm_kind = (
                "select_case:fallthrough"
                if original_arm_line in {609, 667}
                else "case:case"
            )
            arm_id = (
                "fortran_source/ORCHIDEE/src_global/module_llxy.f90:"
                f"{original_arm_line}:{arm_kind}"
            )
            counts[arm_id] = line_counts.get(generated_statement, 0)
        return counts


def _augment_select_evidence(
    output_dir: Path, statement_counts: dict[str, int]
) -> dict[str, object]:
    coverage_path = output_dir / "branch_coverage.json"
    evidence_path = output_dir / "owner_region_evidence.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    for arm in coverage["arms"]:
        arm_id = str(arm["arm_id"])
        if statement_counts.get(arm_id, 0) > 0:
            arm["passed"] = True
            arm["gcov_taken"] = statement_counts[arm_id]
            arm["coverage_kind"] = "gcov_case_body_statement_execution"
            arm["case_body_execution_count"] = statement_counts[arm_id]
    missing = sorted(
        str(arm["arm_id"]) for arm in coverage["arms"] if not arm["passed"]
    )
    coverage["missing_arm_ids"] = missing
    coverage["covered_arm_count"] = sum(bool(arm["passed"]) for arm in coverage["arms"])
    coverage["branch_complete"] = not missing
    coverage["mapping_policy"] += (
        " SELECT CASE has no GNU branch edge; each case arm uses the dynamically "
        "executed first statement in its original case body."
    )
    coverage_path.write_text(json.dumps(coverage, indent=2) + "\n", encoding="ascii")

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    passed = {str(arm["arm_id"]) for arm in coverage["arms"] if arm["passed"]}
    for record in evidence["records"]:
        required = set(record["required_arm_ids"])
        covered = required & passed
        record["gcov_arm_ids"] = sorted(covered)
        record["covered_arm_ids"] = sorted(covered)
        record["missing_arm_ids"] = sorted(required - covered)
        record["passed"] = bool(required and required == covered)
    evidence["complete"] = bool(evidence["records"]) and all(
        record["passed"] for record in evidence["records"]
    )
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="ascii")
    if missing or not evidence["complete"]:
        raise RuntimeError(f"{FAMILY} dynamic coverage remains incomplete: {missing}")
    return {"branch_coverage": coverage, "owner_evidence": evidence}


def run_coverage(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    compiler: Path = DEFAULT_COMPILER,
    gcov: Path = GCOV,
) -> dict[str, object]:
    statement_counts = _select_statement_counts(compiler, gcov)
    try:
        return run_extracted_owner_coverage(
            family=FAMILY,
            source_file=SOURCE,
            procedures=PROCEDURES,
            compose=compose,
            run_numerical_oracle=run_oracle,
            output_dir=output_dir,
            compiler=compiler,
            gcov=gcov,
            execute_harness=execute,
            coverage_base_flags=COVERAGE_FLAGS,
        )
    except RuntimeError as exc:
        if "owner coverage incomplete" not in str(exc):
            raise
        return _augment_select_evidence(output_dir, statement_counts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    parser.add_argument("--gcov", type=Path, default=GCOV)
    args = parser.parse_args(argv)
    try:
        result = run_coverage(
            args.output_dir.resolve(), args.compiler.resolve(), args.gcov.resolve()
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
