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
    DEFAULT_COMPILER,
    ROOT,
    compiler_environment,
)
from fortran_owner_coverage import GCOV, run_extracted_owner_coverage  # noqa: E402
from oracle_lane_solar_owners import (  # noqa: E402
    FAMILY,
    MODES,
    PROCEDURES,
    SOURCE,
    compose,
    run_oracle,
)

DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY


def execute_harness(executable: Path, build: Path, compiler: Path) -> None:
    for mode in MODES:
        subprocess.run(
            [str(executable), str(build / f"coverage_{mode}.csv"), mode],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )


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
        arm_branch_indices={
            "fortran_source/ORCHIDEE/src_global/solar.f90:129:if:true": 0,
            "fortran_source/ORCHIDEE/src_global/solar.f90:129:if:false": 1,
            "fortran_source/ORCHIDEE/src_global/solar.f90:130:if:true": 0,
            "fortran_source/ORCHIDEE/src_global/solar.f90:130:if:false": 1,
            "fortran_source/ORCHIDEE/src_global/solar.f90:238:if:true": 6,
            "fortran_source/ORCHIDEE/src_global/solar.f90:238:if:false": 7,
            "fortran_source/ORCHIDEE/src_global/solar.f90:263:if:true": 6,
            "fortran_source/ORCHIDEE/src_global/solar.f90:263:if:false": 7,
            "fortran_source/ORCHIDEE/src_global/solar.f90:264:if:true": 6,
            "fortran_source/ORCHIDEE/src_global/solar.f90:264:if:false": 7,
            "fortran_source/ORCHIDEE/src_global/solar.f90:549:if:true": 6,
            "fortran_source/ORCHIDEE/src_global/solar.f90:549:if:false": 7,
        },
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
