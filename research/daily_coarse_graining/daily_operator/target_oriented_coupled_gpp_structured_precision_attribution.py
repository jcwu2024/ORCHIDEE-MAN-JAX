"""Pure diagnostics for structured-GPP parameter precision attribution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

SCHEMA_VERSION = "target_oriented_coupled_gpp_structured_precision_attribution_result_v1"
STRUCTURED_GPP_PARAMETER_COUNT = 12_879_363
STRUCTURED_GPP_GROUPS = (
    "forcing_input",
    "forcing_field_embedding",
    "forcing_position_embedding",
    "branch_embedding",
    "parent_hidden_projection",
    "local_input",
    "pft_query_projection",
    "forcing_blocks",
    "potential_blocks",
    "limitation_blocks",
    "correction_blocks",
    "potential_output",
    "limitation_output",
    "correction_output",
)
INTERPOLATION_FRACTIONS = tuple(index / 16.0 for index in range(17))
SHARP_ADJACENT_FRACTION_MINIMUM = 0.75
SMOOTHED_ADJACENT_FRACTION_MAXIMUM = 0.25
COMPARABLE_ENDPOINT_RATIO_MINIMUM = 0.75
COMPARABLE_ENDPOINT_RATIO_MAXIMUM = 1.25
INDIVIDUALLY_MATERIAL_FRACTION_MINIMUM = 0.10
MINIMUM_EFFECT = 1.0e-12


def _named_array_leaves(tree: Any, prefix: tuple[str, ...] = ()) -> list[tuple[str, np.ndarray]]:
    if isinstance(tree, Mapping):
        output = []
        for key, value in tree.items():
            output.extend(_named_array_leaves(value, (*prefix, str(key))))
        return output
    if isinstance(tree, (tuple, list)):
        output = []
        for index, value in enumerate(tree):
            output.extend(_named_array_leaves(value, (*prefix, str(index))))
        return output
    path = "/".join(prefix) if prefix else "<root>"
    value = np.asarray(tree)
    if value.dtype == np.dtype("O"):
        raise TypeError(f"parameter leaf is not an array: {path}")
    return [(path, value)]


def _matched_leaves(*trees: Any) -> list[tuple[str, tuple[np.ndarray, ...]]]:
    flattened = [_named_array_leaves(tree) for tree in trees]
    paths = [tuple(path for path, _value in leaves) for leaves in flattened]
    if not paths or any(value != paths[0] for value in paths[1:]):
        raise ValueError("parameter tree structure drift")
    output = []
    for leaf_index, path in enumerate(paths[0]):
        values = tuple(leaves[leaf_index][1] for leaves in flattened)
        if any(value.shape != values[0].shape for value in values[1:]):
            raise ValueError(f"parameter leaf shape drift: {path}")
        output.append((path, values))
    return output


def _ordered_float32(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.dtype(np.float32) or not np.all(np.isfinite(array)):
        raise ValueError("ULP ordering requires finite float32 values")
    bits = array.view(np.uint32).astype(np.uint64)
    negative = (bits & np.uint64(0x80000000)) != 0
    return np.where(
        negative,
        np.uint64(0x100000000) - bits,
        bits + np.uint64(0x80000000),
    )


def _delta_leaf_record(first: np.ndarray, second: np.ndarray) -> Mapping[str, Any]:
    if first.dtype != np.dtype(np.float32) or second.dtype != np.dtype(np.float32):
        raise ValueError("structured-GPP parameter leaves must be float32")
    finite = np.isfinite(first) & np.isfinite(second)
    if not np.all(finite):
        raise ValueError("structured-GPP parameter leaves must be finite")
    changed = first != second
    delta = second.astype(np.float64) - first.astype(np.float64)
    ulp = np.abs(_ordered_float32(second).astype(np.int64) - _ordered_float32(first).astype(np.int64))
    changed_ulp = ulp[changed]
    changed_count = int(np.sum(changed, dtype=np.int64))
    one_ulp = int(np.sum(changed_ulp == 1, dtype=np.int64))
    two_ulp = int(np.sum(changed_ulp == 2, dtype=np.int64))
    more_ulp = int(np.sum(changed_ulp > 2, dtype=np.int64))
    count = int(first.size)
    return {
        "shape": list(first.shape),
        "dtype": str(first.dtype),
        "parameter_count": count,
        "changed_coordinate_count": changed_count,
        "unchanged_coordinate_count": count - changed_count,
        "changed_fraction": changed_count / count if count else 0.0,
        "delta_l2": float(np.linalg.norm(delta.reshape(-1))),
        "maximum_absolute_change": (float(np.max(np.abs(delta[changed]))) if changed_count else 0.0),
        "sign_crossing_count": int(np.sum(changed & (np.signbit(first) != np.signbit(second)), dtype=np.int64)),
        "one_ulp_changed_count": one_ulp,
        "two_ulp_changed_count": two_ulp,
        "more_than_two_ulp_changed_count": more_ulp,
        "one_ulp_fraction_of_changed": one_ulp / changed_count if changed_count else None,
        "maximum_ulp_distance": int(np.max(changed_ulp)) if changed_count else 0,
        "ulp_count_conserved": one_ulp + two_ulp + more_ulp == changed_count,
    }


def _aggregate_delta_summaries(
    summaries: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    count = sum(int(record["parameter_count"]) for record in summaries)
    changed = sum(int(record["changed_coordinate_count"]) for record in summaries)
    one_ulp = sum(int(record["one_ulp_changed_count"]) for record in summaries)
    two_ulp = sum(int(record["two_ulp_changed_count"]) for record in summaries)
    more_ulp = sum(int(record["more_than_two_ulp_changed_count"]) for record in summaries)
    return {
        "leaf_count": len(summaries),
        "parameter_count": count,
        "changed_coordinate_count": changed,
        "unchanged_coordinate_count": count - changed,
        "changed_fraction": changed / count if count else 0.0,
        "delta_l2": float(np.sqrt(sum(float(record["delta_l2"]) ** 2 for record in summaries))),
        "maximum_absolute_change": max(
            (float(record["maximum_absolute_change"]) for record in summaries),
            default=0.0,
        ),
        "sign_crossing_count": sum(int(record["sign_crossing_count"]) for record in summaries),
        "one_ulp_changed_count": one_ulp,
        "two_ulp_changed_count": two_ulp,
        "more_than_two_ulp_changed_count": more_ulp,
        "one_ulp_fraction_of_changed": one_ulp / changed if changed else None,
        "maximum_ulp_distance": max(
            (int(record["maximum_ulp_distance"]) for record in summaries),
            default=0,
        ),
        "ulp_count_conserved": bool(
            all(bool(record["ulp_count_conserved"]) for record in summaries) and one_ulp + two_ulp + more_ulp == changed
        ),
    }


def float32_delta_inventory(
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
    *,
    groups: Sequence[str] = STRUCTURED_GPP_GROUPS,
) -> Mapping[str, Any]:
    """Return exact leaf, group, and whole-tree float32 endpoint deltas."""

    groups = tuple(groups)
    expected = frozenset(groups)
    if len(groups) != len(expected) or frozenset(control) != expected or frozenset(treatment) != expected:
        raise ValueError("structured-GPP top-level parameter registry drift")
    group_records = {}
    group_summaries = []
    for group in groups:
        leaves = {
            path: _delta_leaf_record(first, second)
            for path, (first, second) in _matched_leaves(control[group], treatment[group])
        }
        summary = _aggregate_delta_summaries(tuple(leaves.values()))
        group_records[group] = {"summary": summary, "leaves": leaves}
        group_summaries.append(summary)
    return {
        "summary": _aggregate_delta_summaries(group_summaries),
        "groups": group_records,
    }


def _membership_leaf_record(
    control: np.ndarray,
    treatment: np.ndarray,
    realized: np.ndarray,
) -> Mapping[str, Any]:
    if any(value.dtype != np.dtype(np.float32) for value in (control, treatment, realized)):
        raise ValueError("float32 interpolation membership requires float32 leaves")
    nonfinite = int(np.sum(~np.isfinite(realized), dtype=np.int64))
    changed = control != treatment
    unchanged = ~changed
    equal_control = changed & (realized == control)
    equal_treatment = changed & (realized == treatment)
    intermediate = changed & ~equal_control & ~equal_treatment
    changed_count = int(np.sum(changed, dtype=np.int64))
    control_count = int(np.sum(equal_control, dtype=np.int64))
    treatment_count = int(np.sum(equal_treatment, dtype=np.int64))
    intermediate_count = int(np.sum(intermediate, dtype=np.int64))
    unchanged_count = int(np.sum(unchanged, dtype=np.int64))
    unchanged_preserved = int(np.sum(unchanged & (realized == control), dtype=np.int64))
    return {
        "parameter_count": int(control.size),
        "endpoint_changed_coordinate_count": changed_count,
        "equals_control_changed_count": control_count,
        "equals_treatment_changed_count": treatment_count,
        "intermediate_changed_count": intermediate_count,
        "unchanged_coordinate_count": unchanged_count,
        "unchanged_preserved_count": unchanged_preserved,
        "realized_nonfinite_count": nonfinite,
        "changed_membership_conserved": (control_count + treatment_count + intermediate_count == changed_count),
    }


def _aggregate_membership_summaries(
    summaries: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    names = (
        "parameter_count",
        "endpoint_changed_coordinate_count",
        "equals_control_changed_count",
        "equals_treatment_changed_count",
        "intermediate_changed_count",
        "unchanged_coordinate_count",
        "unchanged_preserved_count",
        "realized_nonfinite_count",
    )
    output = {name: sum(int(record[name]) for record in summaries) for name in names}
    changed = output["endpoint_changed_coordinate_count"]
    output["equals_control_fraction_of_changed"] = output["equals_control_changed_count"] / changed if changed else None
    output["equals_treatment_fraction_of_changed"] = (
        output["equals_treatment_changed_count"] / changed if changed else None
    )
    output["intermediate_fraction_of_changed"] = output["intermediate_changed_count"] / changed if changed else None
    output["changed_membership_conserved"] = bool(
        all(bool(record["changed_membership_conserved"]) for record in summaries)
        and output["equals_control_changed_count"]
        + output["equals_treatment_changed_count"]
        + output["intermediate_changed_count"]
        == changed
    )
    output["unchanged_preserved"] = output["unchanged_preserved_count"] == output["unchanged_coordinate_count"]
    return output


def float32_interpolation_membership(
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
    realized: Mapping[str, Any],
    *,
    groups: Sequence[str] = STRUCTURED_GPP_GROUPS,
) -> Mapping[str, Any]:
    """Count where a realized float32 interpolation lies between endpoints."""

    groups = tuple(groups)
    expected = frozenset(groups)
    if (
        len(groups) != len(expected)
        or frozenset(control) != expected
        or frozenset(treatment) != expected
        or frozenset(realized) != expected
    ):
        raise ValueError("structured-GPP top-level parameter registry drift")
    records = {}
    summaries = []
    for group in groups:
        leaves = tuple(
            _membership_leaf_record(first, second, current)
            for _path, (first, second, current) in _matched_leaves(control[group], treatment[group], realized[group])
        )
        summary = _aggregate_membership_summaries(leaves)
        records[group] = summary
        summaries.append(summary)
    return {
        "summary": _aggregate_membership_summaries(summaries),
        "groups": records,
    }


def summarize_gpp_path(
    fractions: Sequence[float],
    gpp: Sequence[float],
) -> Mapping[str, Any]:
    """Measure how much of an endpoint GPP effect occurs in one grid interval."""

    fraction = np.asarray(tuple(fractions), dtype=np.float64)
    values = np.asarray(tuple(gpp), dtype=np.float64)
    if (
        fraction.ndim != 1
        or values.shape != fraction.shape
        or values.size < 2
        or not np.all(np.isfinite(fraction))
        or not np.all(np.isfinite(values))
        or fraction[0] != 0.0
        or fraction[-1] != 1.0
        or np.any(np.diff(fraction) <= 0.0)
    ):
        raise ValueError("precision path must be finite, ordered, and span zero to one")
    jump = float(values[-1] - values[0])
    if abs(jump) <= MINIMUM_EFFECT:
        raise ValueError("precision path requires a nonzero endpoint GPP effect")
    adjacent = np.diff(values)
    absolute = np.abs(adjacent)
    selected = int(np.argmax(absolute))
    return {
        "endpoint_gpp_start": float(values[0]),
        "endpoint_gpp_end": float(values[-1]),
        "endpoint_gpp_jump": jump,
        "maximum_absolute_adjacent_change": float(absolute[selected]),
        "maximum_adjacent_fraction_of_endpoint_jump": float(absolute[selected] / abs(jump)),
        "maximum_adjacent_interval": [
            float(fraction[selected]),
            float(fraction[selected + 1]),
        ],
        "total_variation_to_endpoint_ratio": float(np.sum(absolute) / abs(jump)),
    }


def classify_precision_paths(
    *,
    fractions: Sequence[float],
    production_gpp: Sequence[float],
    float64_weight_gpp: Sequence[float],
) -> Mapping[str, Any]:
    """Apply the frozen sharpness and endpoint-comparability classification."""

    production = summarize_gpp_path(fractions, production_gpp)
    diagnostic = summarize_gpp_path(fractions, float64_weight_gpp)
    first = float(production["endpoint_gpp_jump"])
    second = float(diagnostic["endpoint_gpp_jump"])
    ratio = abs(second) / abs(first)
    same_sign = bool(np.signbit(first) == np.signbit(second))
    comparable = bool(same_sign and COMPARABLE_ENDPOINT_RATIO_MINIMUM <= ratio <= COMPARABLE_ENDPOINT_RATIO_MAXIMUM)
    production_sharp = bool(
        float(production["maximum_adjacent_fraction_of_endpoint_jump"]) >= SHARP_ADJACENT_FRACTION_MINIMUM
    )
    diagnostic_sharp = bool(
        float(diagnostic["maximum_adjacent_fraction_of_endpoint_jump"]) >= SHARP_ADJACENT_FRACTION_MINIMUM
    )
    diagnostic_smoothed = bool(
        float(diagnostic["maximum_adjacent_fraction_of_endpoint_jump"]) <= SMOOTHED_ADJACENT_FRACTION_MAXIMUM
    )
    if comparable and production_sharp and diagnostic_smoothed:
        classification = "float32_weight_quantization_dominant_for_sampled_jump"
    elif comparable and production_sharp and diagnostic_sharp:
        classification = "structured_gpp_conditioning_persists_with_float64_weights"
    else:
        classification = "mixed_or_inconclusive_precision_effect"
    return {
        "classification": classification,
        "production_float32": production,
        "float64_weight_interpolation": diagnostic,
        "endpoint_jump_magnitude_ratio": ratio,
        "endpoint_jump_same_sign": same_sign,
        "endpoint_jumps_comparable": comparable,
        "production_path_sharp": production_sharp,
        "float64_weight_path_sharp": diagnostic_sharp,
        "float64_weight_path_smoothed": diagnostic_smoothed,
        "thresholds": {
            "sharp_adjacent_fraction_minimum": SHARP_ADJACENT_FRACTION_MINIMUM,
            "smoothed_adjacent_fraction_maximum": SMOOTHED_ADJACENT_FRACTION_MAXIMUM,
            "comparable_endpoint_ratio_minimum": COMPARABLE_ENDPOINT_RATIO_MINIMUM,
            "comparable_endpoint_ratio_maximum": COMPARABLE_ENDPOINT_RATIO_MAXIMUM,
        },
    }


def classify_subtree_controls(
    *,
    control_gpp: float,
    treatment_gpp: float,
    treatment_group_only_gpp: Mapping[str, float],
    treatment_without_group_gpp: Mapping[str, float],
    groups: Sequence[str] = STRUCTURED_GPP_GROUPS,
) -> Mapping[str, Any]:
    """Report top-level subtree sufficiency and leave-one-out necessity."""

    groups = tuple(groups)
    expected = frozenset(groups)
    if (
        len(groups) != len(expected)
        or frozenset(treatment_group_only_gpp) != expected
        or frozenset(treatment_without_group_gpp) != expected
    ):
        raise ValueError("structured-GPP subtree control registry drift")
    finite = np.asarray(
        (
            control_gpp,
            treatment_gpp,
            *treatment_group_only_gpp.values(),
            *treatment_without_group_gpp.values(),
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(finite)):
        raise ValueError("structured-GPP subtree GPP values must be finite")
    full_effect = float(treatment_gpp - control_gpp)
    if abs(full_effect) <= MINIMUM_EFFECT:
        raise ValueError("structured-GPP subtree controls require a nonzero effect")
    records = {}
    selected = []
    for group in groups:
        only = float(treatment_group_only_gpp[group])
        without = float(treatment_without_group_gpp[group])
        single = only - float(control_gpp)
        residual = without - float(control_gpp)
        sufficiency = single / full_effect
        necessity = 1.0 - abs(residual) / abs(full_effect)
        material = bool(max(abs(sufficiency), abs(necessity)) >= INDIVIDUALLY_MATERIAL_FRACTION_MINIMUM)
        if material:
            selected.append(group)
        records[group] = {
            "treatment_group_only_gpp": only,
            "treatment_without_group_gpp": without,
            "single_group_effect": single,
            "residual_effect_without_group": residual,
            "signed_sufficiency_fraction": sufficiency,
            "necessity_capture_fraction": necessity,
            "individually_material": material,
        }
    return {
        "full_parameter_effect": full_effect,
        "selected_groups": selected,
        "groups": records,
        "material_fraction_minimum": INDIVIDUALLY_MATERIAL_FRACTION_MINIMUM,
    }


__all__ = [
    "COMPARABLE_ENDPOINT_RATIO_MAXIMUM",
    "COMPARABLE_ENDPOINT_RATIO_MINIMUM",
    "INDIVIDUALLY_MATERIAL_FRACTION_MINIMUM",
    "INTERPOLATION_FRACTIONS",
    "MINIMUM_EFFECT",
    "SCHEMA_VERSION",
    "SHARP_ADJACENT_FRACTION_MINIMUM",
    "SMOOTHED_ADJACENT_FRACTION_MAXIMUM",
    "STRUCTURED_GPP_GROUPS",
    "STRUCTURED_GPP_PARAMETER_COUNT",
    "classify_precision_paths",
    "classify_subtree_controls",
    "float32_delta_inventory",
    "float32_interpolation_membership",
    "summarize_gpp_path",
]
