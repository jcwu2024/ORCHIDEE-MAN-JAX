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
from oracle_lane_interregxy_owners import (  # noqa: E402
    FAMILY,
    PROCEDURES,
    SOURCE,
    compose,
    run_oracle,
)

DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY


def execute(executable: Path, build: Path, compiler: Path) -> None:
    env = compiler_environment(compiler)
    subprocess.run(
        [str(executable), str(build / "out.csv")],
        cwd=build,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    failed = subprocess.run(
        [str(executable), str(build / "fatal.csv"), "findpoints_capacity"],
        cwd=build,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if (
        "interregxy_findpoints" not in failed.stderr
        and "interregxy_findpoints" not in failed.stdout
    ):
        raise RuntimeError("findpoints_capacity missed expected fatal contract")


def run_coverage(output_dir=DEFAULT_OUTPUT_DIR, compiler=DEFAULT_COMPILER, gcov=GCOV):
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
        # GNU Fortran associates this vector-expression IF with the second
        # physical continuation line. Branch 22 is fallthrough (true), 23
        # leaves the block (false), as shown by the generated gcov report.
        condition_terminal_lines={125: 126, 281: 282},
        arm_branch_indices={
            "fortran_source/ORCHIDEE/src_global/interregxy.f90:125:if:true": 22,
            "fortran_source/ORCHIDEE/src_global/interregxy.f90:125:if:false": 23,
            "fortran_source/ORCHIDEE/src_global/interregxy.f90:281:if:true": 22,
            "fortran_source/ORCHIDEE/src_global/interregxy.f90:281:if:false": 23,
        },
        coverage_extra_flags=("--coverage",),
        coverage_base_flags=tuple(
            flag for flag in COMPILE_FLAGS if not flag.startswith("-fcheck=")
        ),
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    p.add_argument("--gcov", type=Path, default=GCOV)
    a = p.parse_args(argv)
    try:
        r = run_coverage(a.output_dir.resolve(), a.compiler.resolve(), a.gcov.resolve())
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(
        json.dumps(
            {
                "status": "passed",
                "covered_arms": r["branch_coverage"]["covered_arm_count"],
                "owner_regions": len(r["owner_evidence"]["records"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
