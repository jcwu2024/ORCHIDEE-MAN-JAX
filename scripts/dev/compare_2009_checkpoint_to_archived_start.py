from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.reference_layout import resolve_paper_landpoint_reference  # noqa: E402
from jax_orchidee.stomate.reference import (  # noqa: E402
    read_stomate_restart_entry_state,
    read_stomate_restart_season_state,
)


PFT_AXIS_BY_ENTRY_FIELD = {
    "biomass": 1,
    "resp_maint_part": 1,
    "gpp_daily": 1,
    "npp_daily": 1,
    "turnover_daily": 1,
    "resp_maint": 1,
    "resp_growth": 1,
    "leaf_age": 1,
    "leaf_frac": 1,
    "age": 1,
    "sla_calc": 1,
    "pft_present": 1,
    "ind": 1,
    "adapted": 1,
    "regenerate": 1,
    "npp_longterm": 1,
    "lm_lastyearmax": 1,
    "turnover_time": 1,
    "turnover_longterm": 1,
    "senescence": 1,
    "when_growthinit": 1,
    "co2_to_bm": 1,
    "veget_lastlight": 1,
    "everywhere": 1,
    "need_adjacent": 1,
    "litterpart": 1,
    "dead_leaves": 1,
    "carbon": 2,
    "litter": 2,
    "lignin_struc": 1,
    "fuel_1hr": 1,
    "fuel_10hr": 1,
    "fuel_100hr": 1,
    "fuel_1000hr": 1,
    "bm_to_litter": 1,
    "rip_time": 1,
    "assim_param": 1,
    "altmax": 1,
    "fixed_cryoturbation_depth": 1,
    "litter_above": 2,
    "litter_below": 2,
    "carbon_32l": 2,
    "DOC": 1,
    "interception_storage": 1,
    "lignin_struc_above": 1,
    "lignin_struc_below": 1,
    "deepC_a": 2,
    "deepC_s": 2,
    "deepC_p": 2,
    "soilc_total": 2,
}


def _metrics(actual, reference) -> dict[str, object]:
    left = np.asarray(actual)
    right = np.asarray(reference)
    if left.shape != right.shape:
        return {"passed_shape": False, "actual_shape": list(left.shape), "reference_shape": list(right.shape)}
    if left.dtype.kind in "biu" or right.dtype.kind in "biu":
        return {"passed_shape": True, "exact": bool(np.array_equal(left, right))}
    left = left.astype(np.float64)
    right = right.astype(np.float64)
    delta = np.abs(left - right)
    scale = np.maximum(np.abs(right), 1.0)
    return {
        "passed_shape": True,
        "max_abs_error": float(np.max(delta)) if delta.size else 0.0,
        "max_normalized_error": float(np.max(delta / scale)) if delta.size else 0.0,
        "rmse": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
    }


def _audit_point(landpoint_id: str, checkpoint: Path) -> dict[str, object]:
    reference = resolve_paper_landpoint_reference(ROOT, landpoint_id)
    if reference.stomate_start is None:
        raise FileNotFoundError(f"missing archived 2009 STOMATE start for {landpoint_id}")
    with checkpoint.open("rb") as handle:
        packet = pickle.load(handle)["state"]
    state = packet.fields_by_component["slowproc_stomate_previous_step_state"]
    entry = read_stomate_restart_entry_state(reference.stomate_start)
    season = read_stomate_restart_season_state(reference.stomate_start)
    entry_fields = sorted(set(state).intersection(entry._fields))
    season_fields = sorted(set(state).intersection(season._fields))
    comparisons = {
        "entry": {name: _metrics(state[name], getattr(entry, name)) for name in entry_fields},
        "season": {name: _metrics(state[name], getattr(season, name)) for name in season_fields},
    }
    pft14 = {
        name: _metrics(
            np.take(np.asarray(state[name]), 13, axis=PFT_AXIS_BY_ENTRY_FIELD[name]),
            np.take(np.asarray(getattr(entry, name)), 13, axis=PFT_AXIS_BY_ENTRY_FIELD[name]),
        )
        for name in entry_fields
        if name in PFT_AXIS_BY_ENTRY_FIELD
    }
    ranked = []
    for group, fields in comparisons.items():
        for name, metrics in fields.items():
            ranked.append(
                {
                    "group": group,
                    "field": name,
                    "max_normalized_error": float(metrics.get("max_normalized_error", 0.0)),
                    "max_abs_error": float(metrics.get("max_abs_error", 0.0)),
                    "exact": metrics.get("exact"),
                }
            )
    ranked.sort(key=lambda row: (row["max_normalized_error"], row["max_abs_error"]), reverse=True)
    pft14_ranked = [
        {
            "field": name,
            "max_normalized_error": float(metrics.get("max_normalized_error", 0.0)),
            "max_abs_error": float(metrics.get("max_abs_error", 0.0)),
            "exact": metrics.get("exact"),
        }
        for name, metrics in pft14.items()
    ]
    pft14_ranked.sort(key=lambda row: (row["max_normalized_error"], row["max_abs_error"]), reverse=True)
    return {
        "landpoint_id": landpoint_id,
        "checkpoint": str(checkpoint),
        "reference": str(reference.stomate_start),
        "entry_field_count": len(entry_fields),
        "season_field_count": len(season_fields),
        "largest_differences": ranked[:20],
        "pft14_largest_differences": pft14_ranked[:20],
        "pft14_comparisons": pft14,
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare JAX 2009 state checkpoints with archived Fortran start files.")
    parser.add_argument("--point", action="append", nargs=2, metavar=("LANDPOINT", "CHECKPOINT"), required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "acceptance" / "checkpoint_2009_vs_archived_start.json",
    )
    args = parser.parse_args()
    points = [_audit_point(landpoint_id, Path(checkpoint)) for landpoint_id, checkpoint in args.point]
    result = {
        "scope": "JAX 2009 year-end state versus Fortran 2009 year-end state archived as the 2010 start file.",
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({point["landpoint_id"]: point["pft14_largest_differences"][:12] for point in points}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
