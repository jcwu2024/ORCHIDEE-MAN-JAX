from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
from fortran_oracle_common import ROOT
from oracle_lane_hydrol_lifecycle_owners import FAMILY, run

DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    a = p.parse_args(argv)
    try:
        r = run(a.output_dir.resolve())
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as e:
        print(f"FAIL: {e}")
        return 1
    print(
        json.dumps(
            {"status": "passed", "owner_regions": len(r["owner_evidence"]["records"])},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
