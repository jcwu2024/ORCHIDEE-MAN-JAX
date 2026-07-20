from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.domain import load_case_config  # noqa: E402
from jax_orchidee.driver.paper_mosaic import (  # noqa: E402
    discover_paper_mosaic_cases,
    materialize_paper_mosaic_case_run_defs,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize complete runtime used_run.def files for independent paper mosaic cases."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "orchidee_man_250919.yaml")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--base-used-run-def",
        type=Path,
        default=ROOT / "outputs" / "server_1961_trace_full_20260623" / "run" / "used_run.def",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "paper_mosaic_materialized_run_defs",
    )
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_case_config(args.config)
    mosaic = config.get("paper_mosaic", {})
    root = args.root if args.root is not None else Path(mosaic["local_reference_root"])
    cases = discover_paper_mosaic_cases(root)
    rows = materialize_paper_mosaic_case_run_defs(
        root,
        base_used_run_def=args.base_used_run_def,
        output_root=args.output_root,
        max_cases=args.max_cases,
        dry_run=args.dry_run,
    )
    payload = {
        "root": str(root),
        "base_used_run_def": str(args.base_used_run_def),
        "output_root": str(args.output_root),
        "expected_landpoint_cases": mosaic.get("expected_landpoint_cases"),
        "discovered_cases": len(cases),
        "materialized_cases": len(rows),
        "dry_run": bool(args.dry_run),
        "rows": rows,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
