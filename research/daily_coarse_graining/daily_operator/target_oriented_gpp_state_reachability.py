"""Field-grouped GPP-to-next-state dependency inventory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np

from research.daily_coarse_graining.daily_markov_contract import DailyMarkovContract

from .target_oriented_coupled_gpp_carbon_feedback import (
    CARBON_COMPONENT,
    CarbonFeedbackStateLayout,
)
from .target_oriented_source_shaped_maintenance import contract_sha256

SCHEMA_VERSION = "target_oriented_gpp_state_reachability_v1"
FROZEN_CONTRACT_SHA256 = "1bf2bbfc13c2b6b4b04d323b1a95917e3b33d501dd249f7b2fd36084307c82af"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_gpp_state_reachability_v1.json"
)


@dataclass(frozen=True)
class GPPStateReachabilityContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str


class GPPStateReachabilityLayout(NamedTuple):
    state_indices: np.ndarray
    field_index: np.ndarray
    field_names: tuple[str, ...]
    field_state_starts: tuple[int, ...]
    field_state_stops: tuple[int, ...]
    field_shapes: tuple[tuple[int, ...], ...]
    field_axis_names: tuple[tuple[str, ...], ...]


def load_gpp_state_reachability_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> GPPStateReachabilityContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_after_real_v6_admission_for_zero_training_execution"
    ):
        raise ValueError("GPP state-reachability contract schema drift")
    if payload.get("contract_sha256") != observed or observed != FROZEN_CONTRACT_SHA256:
        raise ValueError("GPP state-reachability contract is not frozen v1")
    data = payload["data"]
    boundary = payload["differentiated_boundary"]
    if (
        int(data["case_count"]) != 13
        or len(data["ordinary_rows"]) != 12
        or len(data["restart_rows"]) != 1
        or boundary["input_shape"] != [1, 2]
        or boundary["compact_pft_indices"] != [0, 13]
        or int(boundary["output_coordinate_count"]) != 2_386
        or int(boundary["output_field_count"]) != 107
    ):
        raise ValueError("GPP state-reachability frozen inventory drift")
    return GPPStateReachabilityContract(payload, source, observed)


def build_gpp_state_reachability_layout(
    markov_contract: DailyMarkovContract,
    carbon_layout: CarbonFeedbackStateLayout,
) -> GPPStateReachabilityLayout:
    """Attach each selected carbon coordinate to its unique Markov field."""

    leaves_by_key = {
        leaf.key: leaf
        for leaf in markov_contract.state_leaves
        if not leaf.discrete and leaf.component == CARBON_COMPONENT
    }
    fields = []
    for field_index, key in enumerate(carbon_layout.field_names):
        leaf = leaves_by_key.get(key)
        if leaf is None or leaf.start is None or leaf.stop is None:
            raise ValueError(f"GPP reachability field has no continuous owner: {key}")
        selected = np.flatnonzero(carbon_layout.field_index == field_index)
        indices = np.asarray(carbon_layout.state_indices[selected], dtype=np.int64)
        expected = np.arange(int(leaf.start), int(leaf.stop), dtype=np.int64)
        if not np.array_equal(indices, expected):
            raise ValueError(f"GPP reachability field slice drift: {key}")
        fields.append(leaf)
    if len(fields) != len(leaves_by_key) or len(fields) != 107:
        raise ValueError("GPP reachability field inventory is incomplete")
    return GPPStateReachabilityLayout(
        state_indices=np.asarray(carbon_layout.state_indices, dtype=np.int64),
        field_index=np.asarray(carbon_layout.field_index, dtype=np.int64),
        field_names=tuple(leaf.key for leaf in fields),
        field_state_starts=tuple(int(leaf.start) for leaf in fields),
        field_state_stops=tuple(int(leaf.stop) for leaf in fields),
        field_shapes=tuple(tuple(int(value) for value in leaf.shape) for leaf in fields),
        field_axis_names=tuple(tuple(str(value) for value in leaf.axis_names) for leaf in fields),
    )


def summarize_gpp_state_jacobians(
    jacobians: Any,
    *,
    layout: GPPStateReachabilityLayout,
    case_ids: Sequence[str],
    state_scale: Any,
    gpp_scale: Any,
    normalized_report_threshold: float,
) -> Mapping[str, Any]:
    """Return the exact-nonzero union and field-grouped sensitivity report."""

    values = np.asarray(jacobians, dtype=np.float64)
    coordinate_count = int(layout.state_indices.size)
    expected_shape = (len(case_ids), coordinate_count, 2)
    if values.shape != expected_shape:
        raise ValueError(f"GPP reachability Jacobian shape is {values.shape}; expected {expected_shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("GPP reachability Jacobian contains nonfinite values")
    state = np.asarray(state_scale, dtype=np.float64).reshape(-1)
    gpp = np.asarray(gpp_scale, dtype=np.float64).reshape(-1)
    if (
        state.shape != (coordinate_count,)
        or gpp.shape != (2,)
        or not np.all(np.isfinite(state))
        or not np.all(np.isfinite(gpp))
        or np.any(state <= 0.0)
        or np.any(gpp <= 0.0)
    ):
        raise ValueError("GPP reachability scales are invalid")
    threshold = float(normalized_report_threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("GPP reachability reporting threshold must be positive")

    absolute = np.abs(values)
    normalized = absolute * gpp[None, None, :] / state[None, :, None]
    bare = absolute[:, :, 0]
    pft14 = absolute[:, :, 1]
    exact = pft14 != 0.0
    union = np.any(exact, axis=0)
    normalized_union = np.max(normalized[:, :, 1], axis=0) >= threshold

    per_case = []
    for index, case_id in enumerate(case_ids):
        per_case.append(
            {
                "case_id": str(case_id),
                "reachable_coordinate_count": int(np.count_nonzero(exact[index])),
                "reachable_field_count": int(np.unique(layout.field_index[exact[index]]).size),
                "bare_soil_maximum_absolute_derivative": float(np.max(bare[index])),
                "pft14_maximum_absolute_derivative": float(np.max(pft14[index])),
                "pft14_l2_norm": float(np.linalg.norm(values[index, :, 1])),
                "maximum_normalized_sensitivity": float(np.max(normalized[index, :, 1])),
            }
        )

    fields = []
    for field, name in enumerate(layout.field_names):
        selected = layout.field_index == field
        local_union = union[selected]
        if not np.any(local_union):
            continue
        local_state_indices = layout.state_indices[selected]
        fields.append(
            {
                "field": name,
                "state_start": layout.field_state_starts[field],
                "state_stop": layout.field_state_stops[field],
                "shape": list(layout.field_shapes[field]),
                "axis_names": list(layout.field_axis_names[field]),
                "coordinate_count": int(np.count_nonzero(selected)),
                "reachable_coordinate_count": int(np.count_nonzero(local_union)),
                "reachable_flat_indices": np.flatnonzero(local_union).astype(int).tolist(),
                "reachable_global_state_indices": local_state_indices[local_union].astype(int).tolist(),
                "case_reachable_coordinate_counts": {
                    str(case_id): int(np.count_nonzero(exact[index, selected]))
                    for index, case_id in enumerate(case_ids)
                },
                "maximum_absolute_derivative": float(np.max(pft14[:, selected])),
                "maximum_normalized_sensitivity": float(np.max(normalized[:, selected, 1])),
            }
        )

    return {
        "jacobian_shape": list(values.shape),
        "all_finite": True,
        "bare_soil_maximum_absolute_derivative": float(np.max(bare)),
        "pft14_maximum_absolute_derivative": float(np.max(pft14)),
        "union_reachable_coordinate_count": int(np.count_nonzero(union)),
        "union_reachable_field_count": len(fields),
        "normalized_report_threshold": threshold,
        "normalized_threshold_coordinate_count": int(np.count_nonzero(normalized_union)),
        "exact_nonzero_below_normalized_threshold_count": int(np.count_nonzero(union & ~normalized_union)),
        "per_case": per_case,
        "reachable_fields": fields,
    }


__all__ = [
    "GPPStateReachabilityContract",
    "GPPStateReachabilityLayout",
    "build_gpp_state_reachability_layout",
    "load_gpp_state_reachability_contract",
    "summarize_gpp_state_jacobians",
]
