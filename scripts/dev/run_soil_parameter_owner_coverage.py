from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from fortran_oracle_common import DEFAULT_COMPILER, ROOT, compiler_environment
from fortran_owner_coverage import GCOV, run_extracted_owner_coverage
from oracle_lane_soil_parameter_owner import (
    COVERAGE_FLAGS, ERROR_MODES, FAMILY, PROCEDURE, SOURCE, VALID_MODES, compose,
    run_oracle,
)


DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY


def _execute(executable: Path, build: Path, compiler: Path) -> None:
    for mode in (*VALID_MODES, *ERROR_MODES):
        subprocess.run(
            [str(executable), str(build / f"coverage_{mode}.csv"), mode], cwd=build,
            env=compiler_environment(compiler), capture_output=True, text=True, check=False,
        )


def run_coverage(output_dir: Path = DEFAULT_OUTPUT_DIR, compiler: Path = DEFAULT_COMPILER, gcov: Path = GCOV) -> dict[str, object]:
    return run_extracted_owner_coverage(
        family=FAMILY, source_file=SOURCE, procedures={PROCEDURE}, compose=compose,
        run_numerical_oracle=run_oracle, output_dir=output_dir, compiler=compiler,
        gcov=gcov, execute_harness=_execute, coverage_base_flags=COVERAGE_FLAGS,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run config_soil_parameters gcov evidence.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    parser.add_argument("--gcov", type=Path, default=GCOV)
    args = parser.parse_args(argv)
    try:
        result = run_coverage(args.output_dir.resolve(), args.compiler.resolve(), args.gcov.resolve())
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(json.dumps({"status": "passed", "covered_arms": result["branch_coverage"]["covered_arm_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
