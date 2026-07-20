from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from fortran_oracle_common import DEFAULT_COMPILER  # noqa: E402
from oracle_lane_slowproc_main_owners import GCOV, OUTPUT_DIR, run_coverage  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    parser.add_argument("--gcov", type=Path, default=GCOV)
    args = parser.parse_args(argv)
    try:
        result = run_coverage(args.output_dir.resolve(), args.compiler.resolve(), args.gcov.resolve())
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(json.dumps({"status": "passed", "covered_arms": result["branch_coverage"]["covered_arm_count"], "owner_regions": len(result["owner_evidence"]["records"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
