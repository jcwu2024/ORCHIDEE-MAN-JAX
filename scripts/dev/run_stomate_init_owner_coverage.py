from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fortran_oracle_common import DEFAULT_COMPILER, ROOT  # noqa: E402
from oracle_lane_stomate_init_owner import FAMILY, run_oracle  # noqa: E402

DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run STOMATE init owner evidence.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    args = parser.parse_args(argv)
    try:
        result = run_oracle(args.output_dir.resolve(), args.compiler.resolve())
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    owners = json.loads((args.output_dir.resolve() / "owner_region_evidence.json").read_text(encoding="ascii"))
    coverage = json.loads((args.output_dir.resolve() / "branch_coverage.json").read_text(encoding="ascii"))
    print(
        json.dumps(
            {
                "status": result["status"],
                "covered_arms": coverage["covered_arm_count"],
                "owner_regions": len(owners["records"]),
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "passed" and owners["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
