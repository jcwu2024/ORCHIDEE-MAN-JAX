from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.reference_layout import (  # noqa: E402
    inventory_paper_references,
    write_reference_manifest,
)


def _zip_repr(value) -> str | None:
    if value is None:
        return None
    return f"{value.archive}!{value.member}"


def _row(reference) -> dict[str, object]:
    return {
        "landpoint_id": reference.landpoint_id,
        "layout": reference.layout,
        "iteration_id": reference.iteration_id,
        "param_set": reference.param_set,
        "output_dir": None if reference.output_dir is None else str(reference.output_dir),
        "history_count": len(reference.histories),
        "years_available": list(reference.years_available),
        "has_driver_start": reference.driver_start is not None,
        "has_sechiba_start": reference.sechiba_start is not None,
        "has_stomate_start": reference.stomate_start is not None,
        "has_driver_restart": reference.driver_restart is not None,
        "has_sechiba_restart": reference.sechiba_restart is not None,
        "has_stomate_restart": reference.stomate_restart is not None,
        "run_def": None if reference.run_def is None else str(reference.run_def),
        "used_run_def": None if reference.used_run_def is None else str(reference.used_run_def),
        "modelout_csv": None if reference.modelout_csv is None else str(reference.modelout_csv),
        "modelout_csv_zip": _zip_repr(reference.modelout_csv_zip),
        "job_script": None if reference.job_script is None else str(reference.job_script),
        "job_script_zip": _zip_repr(reference.job_script_zip),
        "generated_run_def": None if reference.generated_run_def is None else str(reference.generated_run_def),
        "generated_run_def_zip": _zip_repr(reference.generated_run_def_zip),
        "has_minimum_annual_truth": reference.has_minimum_annual_truth,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory local ORCHIDEE-MAN paper reference truth assets.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--landpoint-id", action="append", default=None)
    parser.add_argument("--write-manifest", type=Path, default=None)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    references = inventory_paper_references(args.root)
    if args.landpoint_id:
        selected = set(args.landpoint_id)
        references = tuple(reference for reference in references if reference.landpoint_id in selected)
    payload = {
        "root": str(args.root),
        "count": len(references),
        "complete_count": sum(1 for reference in references if reference.has_minimum_annual_truth),
    }
    if args.summary_only:
        payload["complete_landpoints"] = [
            reference.landpoint_id for reference in references if reference.has_minimum_annual_truth
        ]
    else:
        payload["landpoints"] = [_row(reference) for reference in references]
    if args.write_manifest is not None:
        payload["manifest"] = str(write_reference_manifest(args.root, args.write_manifest))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
