from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fortran_oracle_common import DEFAULT_COMPILER, ROOT, compiler_environment  # noqa: E402
from fortran_owner_coverage import GCOV, run_extracted_owner_coverage  # noqa: E402
from oracle_lane_haversine_grid_owners import (  # noqa: E402
    FAMILY,
    PROCEDURES,
    SOURCE,
    compose,
    run_oracle,
)

DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY


def execute(executable: Path, build: Path, compiler: Path) -> None:
    environment = compiler_environment(compiler)
    subprocess.run(
        [str(executable), str(build / "out.csv")],
        cwd=build,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    for mode in ("missing_longitude", "missing_latitude", "invalid_projection"):
        invalid = subprocess.run(
            [str(executable), str(build / f"{mode}.csv"), mode],
            cwd=build,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if "ERROR STOP ipslerr" not in invalid.stderr:
            raise RuntimeError(f"{mode} missed ipslerr fatal contract")


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
        execute_harness=execute,
        arm_branch_indices={
            f"fortran_source/ORCHIDEE/src_global/haversine.f90:{line}:if:true": 0
            for line in (*range(239, 247), *range(371, 379))
        }
        | {
            f"fortran_source/ORCHIDEE/src_global/haversine.f90:{line}:if:false": 1
            for line in (*range(239, 247), *range(371, 379))
        },
    )


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
