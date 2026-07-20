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
from fortran_owner_coverage import GCOV, run_extracted_owner_coverage  # noqa:E402
from oracle_lane_hydrol_state_updates import PROCEDURES, SOURCE, compose, run_oracle  # noqa:E402

FAMILY = "hydrol_state_updates"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY


def _execute(executable, build, compiler):
    subprocess.run(
        [str(executable)],
        cwd=build,
        env=compiler_environment(compiler),
        check=True,
        capture_output=True,
        text=True,
    )


def run_coverage(output_dir=DEFAULT_OUTPUT_DIR, compiler=DEFAULT_COMPILER, gcov=GCOV):
    return run_extracted_owner_coverage(
        family=FAMILY,
        source_file=SOURCE,
        procedures=set(PROCEDURES),
        compose=compose,
        run_numerical_oracle=run_oracle,
        output_dir=output_dir,
        compiler=compiler,
        gcov=gcov,
        execute_harness=_execute,
        coverage_base_flags=tuple(x for x in COMPILE_FLAGS if x != "-fcheck=all"),
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    p.add_argument("--gcov", type=Path, default=GCOV)
    a = p.parse_args(argv)
    try:
        r = run_coverage(a.output_dir.resolve(), a.compiler.resolve(), a.gcov.resolve())
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as e:
        print(f"FAIL: {e}")
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
