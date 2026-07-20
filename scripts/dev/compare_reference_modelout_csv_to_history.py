from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.reference_layout import (  # noqa: E402
    inventory_paper_references,
    resolve_paper_landpoint_reference,
)
from jax_orchidee.stomate.modelout import (  # noqa: E402
    compute_modelout_from_fields,
    select_history_point_fields,
)
from jax_orchidee.stomate.reference import (  # noqa: E402
    pack_history_modelout_fields,
    read_paper_modelout_csv,
)


def _scalar(value) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(()))


def _delta(jax_value: float, reference_value: float) -> dict[str, float | bool]:
    abs_error = abs(jax_value - reference_value)
    float32_history = float(np.float32(jax_value))
    float32_csv = float(np.float32(reference_value))
    float32_abs_error = abs(float32_history - float32_csv)
    return {
        "history_formula": float(jax_value),
        "csv": float(reference_value),
        "abs_error": float(abs_error),
        "rel_error": float(abs_error / max(abs(reference_value), 1.0)),
        "float32_history_formula": float32_history,
        "float32_csv": float32_csv,
        "float32_abs_error": float(float32_abs_error),
        "float32_equal": bool(float32_abs_error == 0.0),
    }


def _compare_landpoint(root: Path, landpoint_id: str) -> dict[str, object]:
    reference = resolve_paper_landpoint_reference(root, landpoint_id)
    result: dict[str, object] = {
        "landpoint_id": landpoint_id,
        "layout": reference.layout,
        "output_dir": None if reference.output_dir is None else str(reference.output_dir),
        "history_count": len(reference.histories),
        "has_minimum_annual_truth": reference.has_minimum_annual_truth,
        "rows": [],
        "ok": False,
        "max_abs_error": None,
    }
    if not reference.has_minimum_annual_truth:
        result["notes"] = "missing full NetCDF output package or modelout CSV"
        return result

    histories = {path.name: path for path in reference.histories}
    max_abs_error = 0.0
    rows = []
    for csv_row in read_paper_modelout_csv(root=root, landpoint_id=landpoint_id):
        history_path = histories.get(f"stomate_history_{csv_row.target_year}.nc")
        if history_path is None:
            rows.append(
                {
                    "Igrid": csv_row.i_grid,
                    "age": csv_row.age,
                    "target_year": csv_row.target_year,
                    "ok": False,
                    "notes": "target history file is missing",
                }
            )
            continue
        fields = select_history_point_fields(pack_history_modelout_fields(history_path))
        modelout = compute_modelout_from_fields(fields)
        values = {
            "AGB_model": _scalar(modelout.AGB_model),
            "BGB_model": _scalar(modelout.BGB_model),
            "GPP_model": _scalar(modelout.GPP_model),
            "NPP_model": _scalar(modelout.NPP_model),
        }
        deltas = {
            name: _delta(values[name], reference_value)
            for name, reference_value in csv_row.values.items()
        }
        row_max = max(float(delta["abs_error"]) for delta in deltas.values())
        row_float32_max = max(float(delta["float32_abs_error"]) for delta in deltas.values())
        max_abs_error = max(max_abs_error, row_max)
        rows.append(
            {
                "Igrid": csv_row.i_grid,
                "age": csv_row.age,
                "target_year": csv_row.target_year,
                "history_path": str(history_path),
                "ok": row_float32_max == 0.0,
                "max_abs_error": row_max,
                "max_float32_abs_error": row_float32_max,
                "delta": deltas,
                "provenance": (
                    "fortran_run_scripts/paper_250919/c2.4_Model_run_functions_sensitivity.py "
                    "uses age -> stomate_history_(1961 + age - 1).nc, reads PFT14 [0,13,0,0], "
                    "and writes pandas CSV decimals from NetCDF float32-derived values"
                ),
            }
        )
    result["rows"] = rows
    result["max_abs_error"] = max_abs_error
    result["ok"] = bool(rows) and all(row.get("ok", False) for row in rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare downloaded paper modelout CSV targets against matching Fortran history NetCDF fields."
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--landpoint-id", action="append", default=None)
    parser.add_argument("--complete-only", action="store_true", default=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "reference_modelout_csv_history_compare.json")
    args = parser.parse_args()

    if args.landpoint_id:
        landpoint_ids = tuple(args.landpoint_id)
    else:
        references = inventory_paper_references(args.root)
        if args.complete_only:
            references = tuple(reference for reference in references if reference.has_minimum_annual_truth)
        landpoint_ids = tuple(reference.landpoint_id for reference in references)
    comparisons = [_compare_landpoint(args.root, landpoint_id) for landpoint_id in landpoint_ids]
    payload = {
        "root": str(args.root),
        "landpoint_count": len(comparisons),
        "ok_count": sum(1 for item in comparisons if item["ok"]),
        "max_abs_error": max((item["max_abs_error"] or 0.0) for item in comparisons) if comparisons else None,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ok_count"] == payload["landpoint_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
