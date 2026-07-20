from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.orchestration import (  # noqa: E402
    paper_1961_driver_cold_start_first_step_coverage,
    prepare_paper_1961_driver_context,
)
from jax_orchidee.driver.reference_layout import resolve_paper_landpoint_reference  # noqa: E402


def _delta(actual, expected) -> dict[str, object]:
    left = np.asarray(actual).reshape(-1).astype(np.float64)
    right = np.asarray(expected).reshape(-1).astype(np.float64)
    shape_match = left.shape == right.shape
    max_abs = float(np.max(np.abs(left - right))) if shape_match and left.size else 0.0
    return {
        "actual": left.tolist(),
        "reference": right.tolist(),
        "shape_match": shape_match,
        "max_abs_error": max_abs,
        "passed": shape_match and max_abs <= 1.0e-12,
    }


def _audit_point(landpoint_id: str) -> dict[str, object]:
    reference = resolve_paper_landpoint_reference(ROOT, landpoint_id)
    if reference.output_dir is None or reference.used_run_def is None or reference.sechiba_start is None:
        raise FileNotFoundError(f"incomplete static truth package for {landpoint_id}")
    context = prepare_paper_1961_driver_context(
        ROOT / "configs" / "orchidee_man_250919.yaml",
        used_run_def_path=reference.used_run_def,
        reference_run_dir=reference.output_dir,
    )
    coverage = paper_1961_driver_cold_start_first_step_coverage(
        context.config_path,
        prepared_context=context,
    )
    slow = coverage.initialized_payloads["slowproc_init_pft14"]
    with xr.open_dataset(reference.sechiba_start, decode_times=False) as dataset:
        comparisons = {
            "njsc": _delta(slow.njsc, dataset["njsc"].values[0]),
            "clayfraction": _delta(slow.clayfraction, dataset["clay_frac"].values[0]),
            "sandfraction": _delta(slow.sandfraction, dataset["sand_frac"].values[0]),
            "veget_max": _delta(slow.veget_max, dataset["veget_max"].values[0].transpose(1, 2, 0)),
            "soil_ph": _delta(slow.soil_ph, dataset["soil_ph"].values[0]),
            "poor_soils": _delta(slow.poor_soils, dataset["poor_soils"].values[0]),
        }
    return {
        "landpoint_id": landpoint_id,
        "passed": all(row["passed"] for row in comparisons.values()),
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit invariant point-static fields against archived Fortran state.")
    parser.add_argument("--landpoint-id", action="append", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "acceptance" / "landpoint_static_state_bindings.json",
    )
    args = parser.parse_args()
    points = [_audit_point(landpoint_id) for landpoint_id in args.landpoint_id]
    result = {
        "status": "passed" if all(point["passed"] for point in points) else "failed",
        "scope": "Time-invariant static fields only; archived start files are 2009 year-end dynamic state.",
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
