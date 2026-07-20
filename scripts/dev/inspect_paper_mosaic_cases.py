from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.domain import load_case_config  # noqa: E402
from jax_orchidee.driver.paper_mosaic import (  # noqa: E402
    build_paper_mosaic_manifest,
    build_paper_mosaic_runtime_tasks,
    build_paper_mosaic_year_tasks,
    discover_paper_mosaic_cases,
    summarize_paper_mosaic_manifest,
    summarize_paper_mosaic_runtime_tasks,
    summarize_paper_mosaic_year_tasks,
)


def _parse_years(text: str) -> tuple[int, ...]:
    if ":" in text:
        start, stop = text.split(":", 1)
        return tuple(range(int(start), int(stop) + 1))
    return tuple(int(part) for part in text.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect independent single-landpoint paper mosaic case directories."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "orchidee_man_250919.yaml")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--years", default="1961:2010")
    parser.add_argument("--check-history-shape", action="store_true")
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--emit-year-tasks", action="store_true")
    parser.add_argument("--emit-runtime-tasks", action="store_true")
    parser.add_argument(
        "--materialized-run-def-root",
        type=Path,
        default=ROOT / "outputs" / "paper_mosaic_materialized_run_defs",
    )
    args = parser.parse_args()

    config = load_case_config(args.config)
    mosaic = config.get("paper_mosaic", {})
    root = args.root if args.root is not None else Path(mosaic["local_reference_root"])
    years = _parse_years(args.years)
    cases = discover_paper_mosaic_cases(root)
    manifest = build_paper_mosaic_manifest(
        root,
        years=years,
        check_history_shape=args.check_history_shape,
        max_cases=args.max_cases,
    )
    summary = summarize_paper_mosaic_manifest(
        manifest,
        discovered_cases=len(cases),
        expected_landpoint_cases=mosaic.get("expected_landpoint_cases"),
    )

    payload = {
        "root": str(root),
        **summary,
        "shown_cases": len(manifest),
        "years": [years[0], years[-1]] if years else [],
        "execution_unit": mosaic.get("execution_unit"),
        "routing_scope": "true nbp_glo>1 routing is outside the independent single-landpoint mosaic",
        "cases": [row.to_json_row() for row in manifest],
    }
    if args.emit_year_tasks:
        tasks = build_paper_mosaic_year_tasks(root, years=years, max_cases=args.max_cases)
        payload["year_task_summary"] = summarize_paper_mosaic_year_tasks(tasks)
        payload["year_tasks"] = [task.to_json_row() for task in tasks]
    if args.emit_runtime_tasks:
        runtime_tasks = build_paper_mosaic_runtime_tasks(
            root,
            years=years,
            materialized_run_def_root=args.materialized_run_def_root,
            max_cases=args.max_cases,
        )
        payload["materialized_run_def_root"] = str(args.materialized_run_def_root)
        payload["runtime_task_summary"] = summarize_paper_mosaic_runtime_tasks(runtime_tasks)
        payload["runtime_tasks"] = [task.to_json_row() for task in runtime_tasks]
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
