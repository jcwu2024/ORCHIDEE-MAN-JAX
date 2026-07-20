from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from fortran_oracle_common import DEFAULT_COMPILER, ROOT, compiler_environment  # noqa:E402
from fortran_owner_coverage import GCOV, run_extracted_owner_coverage  # noqa:E402
from oracle_lane_lpj_crown_owner import FAMILY, PROCEDURES, SOURCE, compose, run_oracle  # noqa:E402

DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY


def _execute(exe: Path, build: Path, compiler: Path) -> None:
    env = compiler_environment(compiler)
    subprocess.run(
        [str(exe), str(build / "normal.csv"), "normal"],
        cwd=build,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    fatal = subprocess.run(
        [str(exe), str(build / "fatal.csv"), "fatal"],
        cwd=build,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if fatal.returncode == 0 or "ipslerr_p" not in fatal.stderr:
        raise RuntimeError("crown coherence case missed fatal exit")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    parser.add_argument("--gcov", type=Path, default=GCOV)
    a = parser.parse_args(argv)
    try:
        r = run_extracted_owner_coverage(
            family=FAMILY,
            source_file=SOURCE,
            procedures=PROCEDURES,
            compose=compose,
            run_numerical_oracle=run_oracle,
            output_dir=a.output_dir.resolve(),
            compiler=a.compiler.resolve(),
            gcov=a.gcov.resolve(),
            execute_harness=_execute,
        )
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
