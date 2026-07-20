from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fortran_oracle_common import (  # noqa: E402
    COMPILE_FLAGS,
    DEFAULT_COMPILER,
    ROOT,
    compiler_environment,
)
from fortran_owner_coverage import GCOV, run_extracted_owner_coverage  # noqa: E402
from oracle_lane_interpweight_helpers import (  # noqa: E402
    FAMILY,
    PROCEDURES,
    SOURCE,
    compose,
    run_oracle,
)

DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
FATAL_MODES = (
    "mask1_sum",
    "mask1_bad",
    "mask2_bad",
    "mask3_bad",
    "mask4_bad",
    "frac1_zero",
    "frac1_over",
    "frac1_bad",
    "frac2_zero",
    "frac2_over",
    "frac2_bad",
    "frac4_dims",
    "frac4_zero",
    "frac4_bad",
    "interp_default_zero",
    "interp_slope_zero",
    "interp_bad",
    "val_bad",
)

# GNU emits no gcov edge on SELECT/CASE declaration lines.  Each declaration
# is mapped to the first executable statement that uniquely proves that arm.
CASE_TERMINAL_LINES = {
    2940: 2958,
    2941: 2942,
    2943: 2949,
    2951: 2953,
    3010: 3045,
    3011: 3012,
    3013: 3020,
    3023: 3026,
    3029: 3035,
    3097: 3130,
    3098: 3099,
    3100: 3107,
    3110: 3113,
    3116: 3120,
    3181: 3218,
    3182: 3183,
    3184: 3191,
    3194: 3197,
    3200: 3207,
    3288: 3343,
    3289: 3291,
    3413: 3470,
    3414: 3416,
    3715: 3775,
    3716: 3718,
    3848: 3933,
    3849: 3851,
    3890: 3892,
    4929: 4954,
    4930: 4931,
    4935: 4936,
    4940: 4941,
    4945: 4946,
    4950: 4954,
}
CASE_BRANCH_INDICES = {
    f"fortran_source/ORCHIDEE/src_global/interpweight.f90:{line}:{kind}": 0
    for line, kind in (
        (2940, "select_case:fallthrough"),
        (2941, "case:case"),
        (2943, "case:case"),
        (2951, "case:case"),
        (3010, "select_case:fallthrough"),
        (3011, "case:case"),
        (3013, "case:case"),
        (3023, "case:case"),
        (3029, "case:case"),
        (3097, "select_case:fallthrough"),
        (3098, "case:case"),
        (3100, "case:case"),
        (3110, "case:case"),
        (3116, "case:case"),
        (3181, "select_case:fallthrough"),
        (3182, "case:case"),
        (3184, "case:case"),
        (3194, "case:case"),
        (3200, "case:case"),
        (3288, "select_case:fallthrough"),
        (3289, "case:case"),
        (3413, "select_case:fallthrough"),
        (3414, "case:case"),
        (3715, "select_case:fallthrough"),
        (3716, "case:case"),
        (3848, "select_case:fallthrough"),
        (3849, "case:case"),
        (3890, "case:case"),
        (4929, "select_case:fallthrough"),
        (4930, "case:case"),
        (4935, "case:case"),
        (4940, "case:case"),
        (4945, "case:case"),
        (4950, "case:case"),
    )
}


def execute_harness(executable: Path, build: Path, compiler: Path) -> None:
    environment = compiler_environment(compiler)
    subprocess.run(
        [str(executable), str(build / "normal.csv")],
        cwd=build,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    for mode in FATAL_MODES:
        completed = subprocess.run(
            [str(executable), str(build / f"{mode}.csv"), mode],
            cwd=build,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0 or "ERROR STOP ipslerr_p" not in completed.stderr:
            raise RuntimeError(f"fatal case {mode!r} did not reach source error exit")


def run_coverage(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    compiler: Path = DEFAULT_COMPILER,
    gcov: Path = GCOV,
) -> dict[str, object]:
    return run_extracted_owner_coverage(
        family=FAMILY,
        source_file=SOURCE,
        procedures=PROCEDURES,
        compose=compose,
        run_numerical_oracle=run_oracle,
        output_dir=output_dir,
        compiler=compiler,
        gcov=gcov,
        execute_harness=execute_harness,
        condition_terminal_lines=CASE_TERMINAL_LINES,
        arm_branch_indices=CASE_BRANCH_INDICES,
        coverage_base_flags=tuple(
            flag for flag in COMPILE_FLAGS if not flag.startswith("-fcheck=")
        ),
    )


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
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
