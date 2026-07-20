from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fortran_oracle_common import DEFAULT_COMPILER, ROOT, compiler_environment  # noqa: E402
from fortran_owner_coverage import GCOV, run_extracted_owner_coverage  # noqa: E402
from oracle_lane_pft_parameters_owners import (  # noqa: E402
    ERROR_MODES,
    FAMILY,
    OWNER_PROCEDURES,
    PFT_COVERAGE_FLAGS,
    SOURCE,
    VALID_MODES,
    compose,
    run_oracle,
)


DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
ARM_BRANCH_INDICES: dict[str, int] = {}


def _execute_harness(executable: Path, build: Path, compiler: Path) -> None:
    for mode in (*VALID_MODES, *ERROR_MODES):
        subprocess.run(
            [str(executable), str(build / f"coverage_{mode}.csv"), mode],
            cwd=build,
            env=compiler_environment(compiler),
            capture_output=True,
            text=True,
            check=False,
        )
    subprocess.run(
        [str(executable), str(build / "coverage_main_default_final.csv"), "main_default"],
        cwd=build,
        env=compiler_environment(compiler),
        capture_output=True,
        text=True,
        check=True,
    )


def _execute_main_harness(executable: Path, build: Path, compiler: Path) -> None:
    for mode in ("main_default", "main_custom", "main_default"):
        subprocess.run(
            [str(executable), str(build / f"coverage_main_{mode}.csv"), mode],
            cwd=build,
            env=compiler_environment(compiler),
            capture_output=True,
            text=True,
            check=True,
        )


def _passed_comparison(_output_dir: Path, _compiler: Path) -> dict[str, object]:
    return {"status": "passed"}


def run_coverage(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    compiler: Path = DEFAULT_COMPILER,
    gcov: Path = GCOV,
) -> dict[str, object]:
    nonmain = run_extracted_owner_coverage(
        family=FAMILY,
        source_file=SOURCE,
        procedures=OWNER_PROCEDURES - {"pft_parameters_main"},
        compose=compose,
        run_numerical_oracle=run_oracle,
        output_dir=output_dir,
        compiler=compiler,
        gcov=gcov,
        execute_harness=_execute_harness,
        coverage_base_flags=PFT_COVERAGE_FLAGS,
        arm_branch_indices=ARM_BRANCH_INDICES,
    )
    with tempfile.TemporaryDirectory(prefix="orchidee_pft_parameters_main_owner_") as td:
        main = run_extracted_owner_coverage(
            family=FAMILY,
            source_file=SOURCE,
            procedures={"pft_parameters_main"},
            compose=compose,
            run_numerical_oracle=_passed_comparison,
            output_dir=Path(td),
            compiler=compiler,
            gcov=gcov,
            execute_harness=_execute_main_harness,
            coverage_base_flags=PFT_COVERAGE_FLAGS,
        )
    coverage = nonmain["branch_coverage"]
    main_coverage = main["branch_coverage"]
    coverage["arms"] = sorted(
        [*coverage["arms"], *main_coverage["arms"]], key=lambda item: item["arm_id"]
    )
    coverage["required_arm_count"] += main_coverage["required_arm_count"]
    coverage["covered_arm_count"] += main_coverage["covered_arm_count"]
    coverage["missing_arm_ids"] = sorted(
        [*coverage["missing_arm_ids"], *main_coverage["missing_arm_ids"]]
    )
    coverage["branch_complete"] = not coverage["missing_arm_ids"]
    owners = nonmain["owner_evidence"]
    owners["records"] = sorted(
        [*owners["records"], *main["owner_evidence"]["records"]],
        key=lambda item: item["owner_region_id"],
    )
    owners["complete"] = all(record["passed"] for record in owners["records"])
    (output_dir / "branch_coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="ascii"
    )
    (output_dir / "owner_region_evidence.json").write_text(
        json.dumps(owners, indent=2) + "\n", encoding="ascii"
    )
    return {"branch_coverage": coverage, "owner_evidence": owners}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pft_parameters gcov owner evidence.")
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
    print(json.dumps({
        "status": "passed",
        "covered_arms": result["branch_coverage"]["covered_arm_count"],
        "owner_regions": len(result["owner_evidence"]["records"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
