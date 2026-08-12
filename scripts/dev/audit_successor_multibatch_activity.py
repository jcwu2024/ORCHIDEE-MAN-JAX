"""Audit process activity behind successor multibatch gradient measurements."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_architecture import (
    PFT_TRAIT_BARE_SOIL,
)
from scripts.hpc.successor_small_fit_prepared import PreparedUpdateArtifact


def _finite_summary(value: Any, defined: Any | None = None) -> Mapping[str, Any]:
    array = np.asarray(value, dtype=np.float64)
    valid = np.isfinite(array)
    if defined is not None:
        valid &= np.asarray(defined, dtype=np.bool_)
    selected = array[valid]
    if selected.size == 0:
        return {
            "count": 0,
            "nonzero_count": 0,
            "absolute_mean": 0.0,
            "rms": 0.0,
            "absolute_maximum": 0.0,
        }
    return {
        "count": int(selected.size),
        "nonzero_count": int(np.count_nonzero(selected)),
        "absolute_mean": float(np.mean(np.abs(selected))),
        "rms": float(np.sqrt(np.mean(selected * selected))),
        "absolute_maximum": float(np.max(np.abs(selected))),
    }


def _load_artifact(path: Path, compression: str) -> PreparedUpdateArtifact:
    if compression == "gzip":
        with gzip.open(path, "rb") as handle:
            artifact = pickle.load(handle)
    elif compression == "none":
        with path.open("rb") as handle:
            artifact = pickle.load(handle)
    else:
        raise ValueError(f"unsupported prepared-artifact compression: {compression}")
    if not isinstance(artifact, PreparedUpdateArtifact):
        raise TypeError(f"unexpected prepared-artifact type at {path}")
    return artifact


def _arm_measurements(measurements: list[Mapping[str, Any]], update: int) -> Mapping[str, Any]:
    selected = [item for item in measurements if int(item["update"]) == update]
    return {
        str(item["arm"]): {
            "anchor_loss": float(item["anchor_loss"]),
            "anchor_gradient_norm": float(item["anchor_gradient_norm"]),
            "component_losses": {str(name): float(value) for name, value in item["component_losses"].items()},
            "component_gradient_norms": {
                str(name): float(value) for name, value in item["component_gradient_norms"].items()
            },
        }
        for item in selected
    }


def _audit_artifact(
    artifact: PreparedUpdateArtifact,
    measurements: list[Mapping[str, Any]],
) -> Mapping[str, Any]:
    entry = artifact.schedule_entry
    batch = artifact.batch
    inputs = batch.rollout_inputs
    fraction = np.asarray(inputs.pft.pft_fraction, dtype=np.float64)
    traits = np.asarray(inputs.pft.pft_traits, dtype=np.float64)
    active = np.asarray(inputs.pft.active_pft_mask, dtype=np.bool_)
    if fraction.shape != active.shape or traits.shape[:-1] != fraction.shape:
        raise ValueError("prepared rollout PFT arrays have incompatible shapes")
    bare = traits[..., PFT_TRAIT_BARE_SOIL] > 0.5
    vegetation = active & ~bare & (fraction > 0.0)
    vegetation_fraction = np.sum(np.where(vegetation, fraction, 0.0), axis=-1)

    target_summaries = {
        str(name): _finite_summary(target.value, target.defined) for name, target in batch.rollout_targets.items()
    }
    active_target_fields = sum(int(summary["nonzero_count"] > 0) for summary in target_summaries.values())
    total_defined = sum(int(summary["count"]) for summary in target_summaries.values())
    total_nonzero = sum(int(summary["nonzero_count"]) for summary in target_summaries.values())
    top_targets = sorted(
        target_summaries.items(),
        key=lambda item: float(item[1]["rms"]),
        reverse=True,
    )[:12]

    day_index = np.asarray(inputs.day_index)
    expected_state = np.asarray(batch.expected_next_state, dtype=np.float64)
    start_state = np.asarray(inputs.continuous_state, dtype=np.float64)
    expected_delta = expected_state - start_state
    expected_delta_defined = (
        np.asarray(batch.expected_next_state_defined, dtype=np.bool_)
        & np.isfinite(start_state)
        & (np.abs(start_state) < 0.5e20)
    )
    update = int(entry["update"])
    return {
        "update": update,
        "landpoint_id": str(entry["landpoint_id"]),
        "year": int(entry["year"]),
        "anchor_starts": [int(value) for value in entry["anchor_starts"]],
        "rollout_starts": [int(value) for value in entry["rollout_starts"]],
        "rollout_day_index_ranges": [[int(window[0]), int(window[-1])] for window in day_index],
        "vegetation_fraction": {
            "minimum": float(np.min(vegetation_fraction)),
            "median": float(np.median(vegetation_fraction)),
            "maximum": float(np.max(vegetation_fraction)),
            "positive_day_count": int(np.count_nonzero(vegetation_fraction > 0.0)),
            "day_count": int(vegetation_fraction.size),
        },
        "teacher_fast_gpp": _finite_summary(
            batch.expected_fast_gpp,
            batch.expected_fast_gpp_defined,
        ),
        "teacher_process_targets": {
            "field_count": len(target_summaries),
            "active_field_count": active_target_fields,
            "defined_element_count": total_defined,
            "nonzero_element_count": total_nonzero,
            "top_fields_by_rms": [{"field": name, **summary} for name, summary in top_targets],
        },
        "teacher_expected_state_delta": _finite_summary(
            expected_delta,
            expected_delta_defined,
        ),
        "measurements": _arm_measurements(measurements, update),
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    manifest_path = args.prepared_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    probe = json.loads(args.probe_report.read_text(encoding="utf-8"))
    measurements = list(probe["measurements"])
    records = []
    for entry in manifest["entries"]:
        path = manifest_path.parent / str(entry["path"])
        artifact = _load_artifact(path, str(entry.get("compression", "none")))
        records.append(_audit_artifact(artifact, measurements))

    payload = {
        "schema_version": "successor_multibatch_activity_audit_v1",
        "status": "passed",
        "prepared_manifest": str(manifest_path),
        "probe_report": str(args.probe_report.resolve()),
        "batch_count": len(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--probe-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    payload = run(_parser().parse_args())
    for record in payload["records"]:
        gradients = {
            arm: min(values["component_gradient_norms"].values()) for arm, values in record["measurements"].items()
        }
        vegetation = record["vegetation_fraction"]
        print(
            f"update={record['update']:02d} point={record['landpoint_id']} "
            f"year={record['year']} vegetation_median={vegetation['median']:.6g} "
            f"gpp_rms={record['teacher_fast_gpp']['rms']:.6g} "
            f"minimum_component_gradient={min(gradients.values()):.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
