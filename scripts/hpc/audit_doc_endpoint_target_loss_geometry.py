"""Audit v4 DOC endpoint target and loss geometry without training."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.hpc.prepare_successor_small_fit_batches import _initialize_cpu_backend
from scripts.hpc.run_strong_baseline_one_point_qualification import _prepare_point_year
from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    FAST_STATISTICS_JSON_SHA256,
    _load_extended_runtime,
)
from scripts.hpc.run_successor_process_head_canonical_gate import (
    EXPECTED,
    FROZEN_LANDPOINT,
    _atomic_json,
)

INCOMING_FIELD = "inventory.carbon.DOC.incoming_amount"
OUTGOING_FIELD = "inventory.carbon.DOC.outgoing_fraction"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _field_index(schema: Any, name: str) -> int:
    matches = tuple(index for index, field in enumerate(schema.fields) if field.name == name)
    if len(matches) != 1:
        raise ValueError(f"DOC geometry audit requires exactly one field {name!r}")
    return matches[0]


def _day_start_doc(dataset: Mapping[str, Any], layout: Any, field_index: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(layout.local_state_indices[field_index], dtype=np.int32)
    if indices.size == 0:
        raise ValueError("DOC geometry audit lacks the day-start inventory owner")
    mapped = indices >= 0
    safe = np.maximum(indices, 0)
    state = np.asarray(dataset["assembled_inputs"].state.other, dtype=np.float64)
    defined = np.asarray(dataset["assembled_inputs"].state_defined.other, dtype=np.bool_)
    values = state[:, safe]
    admitted = mapped[None, :] & defined[:, safe] & np.isfinite(values)
    return values, admitted


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    from research.daily_coarse_graining.daily_operator.coordinate_coupled_doc_attention import (
        CoordinateCoupledDocTransitionConfig,
    )
    from research.daily_coarse_graining.daily_operator.doc_target_diagnostics import (
        inventory_transition_geometry_statistics,
    )
    from research.daily_coarse_graining.daily_operator.hierarchical_coordinate_attention import (
        HierarchicalCoordinateAttentionConfig,
    )
    from research.daily_coarse_graining.daily_operator.mechanistic_output_contract import (
        build_mechanistic_learned_schema,
        load_mechanistic_output_contract,
    )
    from research.daily_coarse_graining.daily_operator.mechanistic_process_head import (
        build_mechanistic_output_layout,
    )
    from research.daily_coarse_graining.daily_operator.strong_baseline_contract import (
        load_strong_baseline_contract,
    )

    runtime = _load_extended_runtime(args, backend_initializer=_initialize_cpu_backend)
    architecture_contract = load_strong_baseline_contract()
    mechanism_contract = load_mechanistic_output_contract()
    learned_schema = build_mechanistic_learned_schema(runtime["schema"], mechanism_contract)
    layout = build_mechanistic_output_layout(
        adapter=runtime["adapter"],
        complete_schema=runtime["schema"],
        learned_schema=learned_schema,
        owner_contract=runtime["owner_contract"],
        architecture_contract=architecture_contract,
        mechanism_contract=mechanism_contract,
        inputs=runtime["probe_assembled"],
    )
    dataset = _prepare_point_year(runtime)
    if dataset["sample_count"] != 364:
        raise ValueError("DOC geometry audit requires the frozen 364-day point-year")

    incoming_index = _field_index(learned_schema, INCOMING_FIELD)
    outgoing_index = _field_index(learned_schema, OUTGOING_FIELD)
    incoming_field = learned_schema.fields[incoming_index]
    outgoing_field = learned_schema.fields[outgoing_index]
    if incoming_field.shape != outgoing_field.shape:
        raise ValueError("DOC geometry endpoint shapes drifted")
    incoming = dataset["targets"][INCOMING_FIELD]
    outgoing = dataset["targets"][OUTGOING_FIELD]
    day_start, day_start_defined = _day_start_doc(dataset, layout, incoming_index)
    expected_shape = np.asarray(incoming.value).shape
    day_start = day_start.reshape(expected_shape)
    day_start_defined = day_start_defined.reshape(expected_shape)

    geometry = inventory_transition_geometry_statistics(
        incoming.value,
        incoming.defined,
        outgoing.value,
        outgoing.defined,
        day_start,
        day_start_defined,
        incoming_output_mean=runtime["output_normalization"].mean[INCOMING_FIELD],
        incoming_output_scale=runtime["output_normalization"].scale[INCOMING_FIELD],
        outgoing_output_mean=runtime["output_normalization"].mean[OUTGOING_FIELD],
        outgoing_output_scale=runtime["output_normalization"].scale[OUTGOING_FIELD],
        incoming_loss_scale=runtime["loss_scales"][INCOMING_FIELD],
        outgoing_loss_scale=runtime["loss_scales"][OUTGOING_FIELD],
        current_minimum_inventory_ratio=(
            CoordinateCoupledDocTransitionConfig().minimum_inventory_ratio
        ),
        separate_minimum_nonnegative_ratio=(
            HierarchicalCoordinateAttentionConfig().minimum_nonnegative_ratio
        ),
    )
    integrity_passed = (
        geometry["declared_count"] == geometry["valid_count"]
        and geometry["partition"]["simultaneously_active_count"] == 0
        and geometry["current_joint_signed_day_end"][
            "finite_inverse_maximum_reconstruction_error"
        ]
        <= 1.0e-10
    )
    report = {
        "schema_version": "doc_endpoint_target_loss_geometry_audit_v1",
        "status": "passed" if integrity_passed else "failed",
        "training_updates": 0,
        "checkpoint_written": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "identity": {
            "git_head": _git_head(),
            "dataset_manifest_sha256": EXPECTED["dataset_manifest"],
            "fast_owner_statistics_json_sha256": FAST_STATISTICS_JSON_SHA256,
        },
        "data": {
            "landpoint_id": FROZEN_LANDPOINT,
            "year": 1961,
            "spatial_split": "train",
            "temporal_split": "train",
            "day_count": int(dataset["sample_count"]),
            "day_index_minimum": int(np.min(dataset["day_indices"])),
            "day_index_maximum": int(np.max(dataset["day_indices"])),
        },
        "fields": {
            "incoming": {
                "name": INCOMING_FIELD,
                "label": learned_schema.labels[incoming_field.label_index],
                "constraint": incoming_field.constraint,
                "value_representation": incoming_field.value_representation,
            },
            "outgoing": {
                "name": OUTGOING_FIELD,
                "label": learned_schema.labels[outgoing_field.label_index],
                "constraint": outgoing_field.constraint,
                "value_representation": outgoing_field.value_representation,
            },
        },
        "geometry": geometry,
        "decision": {
            "current_joint_conditioning_passed": geometry[
                "current_joint_signed_day_end"
            ]["conditioning_passed"],
            "architecture_promoted": False,
            "next_gate": "review_and_freeze_revised_doc_endpoint_mechanism",
        },
    }
    _atomic_json(args.output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
