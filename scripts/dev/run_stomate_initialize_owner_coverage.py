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
from oracle_lane_stomate_initialize_owner import FAMILY, run_oracle  # noqa: E402

DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY


def run_coverage(output_dir: Path = DEFAULT_OUTPUT_DIR, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    result = run_oracle(output_dir, compiler)
    if result.get("status") != "passed":
        raise RuntimeError(f"{FAMILY} numerical comparison did not pass")
    owners = json.loads((output_dir / "owner_region_evidence.json").read_text(encoding="ascii"))
    if owners.get("complete") is not True:
        raise RuntimeError(f"{FAMILY} owner evidence is incomplete")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    args = parser.parse_args(argv)
    try:
        result = run_coverage(args.output_dir.resolve(), args.compiler.resolve())
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(json.dumps({"status": result["status"], "owner_regions": 2}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
