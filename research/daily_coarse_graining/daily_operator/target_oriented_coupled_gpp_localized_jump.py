"""Classify the first material day in a localized coupled-GPP jump."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

MATERIAL_NORMALIZED_STATE_CHANGE = 1.0e-3


def _relative_change(before: float, after: float) -> float:
    first = float(before)
    second = float(after)
    if not np.isfinite(first) or first <= 0.0 or not np.isfinite(second) or second < 0.0:
        raise ValueError("localized-jump score is invalid")
    return (second - first) / first


def select_largest_positive_window(
    records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Select the matched window with the largest treatment loss increase."""

    values = tuple(records)
    if not values:
        raise ValueError("localized-jump window selection requires records")
    indices = tuple(int(value["window_index"]) for value in values)
    if len(set(indices)) != len(indices):
        raise ValueError("localized-jump window registry is not unique")

    normalized = []
    for value in values:
        control = float(value["control_score"])
        treatment = float(value["treatment_score"])
        normalized.append(
            dict(
                value,
                window_index=int(value["window_index"]),
                control_score=control,
                treatment_score=treatment,
                absolute_displacement=treatment - control,
                relative_displacement=_relative_change(control, treatment),
            )
        )
    displacements = np.asarray(
        [value["absolute_displacement"] for value in normalized],
        dtype=np.float64,
    )
    selected_position = int(np.argmax(displacements))
    selected = normalized[selected_position]
    positive = bool(float(selected["absolute_displacement"]) > 0.0)
    return {
        "classification": ("largest_positive_window_selected" if positive else "no_positive_window_displacement"),
        "selected_window_index": int(selected["window_index"]),
        "selected_absolute_displacement": float(selected["absolute_displacement"]),
        "selected_relative_displacement": float(selected["relative_displacement"]),
        "control_mean": float(np.mean([value["control_score"] for value in normalized])),
        "treatment_mean": float(np.mean([value["treatment_score"] for value in normalized])),
        "records": normalized,
    }


def classify_material_days(
    records: Sequence[Mapping[str, Any]],
    *,
    normalized_state_threshold: float = MATERIAL_NORMALIZED_STATE_CHANGE,
) -> Mapping[str, Any]:
    """Return the first day with a discrete, mask, or material state change."""

    values = tuple(records)
    if not values:
        raise ValueError("localized-jump day classification requires records")
    days = tuple(int(value["day_index"]) for value in values)
    if len(set(days)) != len(days) or tuple(sorted(days)) != days:
        raise ValueError("localized-jump days must be unique and ordered")
    threshold = float(normalized_state_threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("localized-jump state threshold must be positive")

    normalized = []
    for value in values:
        discrete = int(value["discrete_mismatch_count"])
        defined = int(value["defined_mask_mismatch_count"])
        state = float(value["maximum_normalized_state_change"])
        if discrete < 0 or defined < 0 or not np.isfinite(state) or state < 0.0:
            raise ValueError("localized-jump day record is invalid")
        reasons = []
        if discrete:
            reasons.append("discrete_state")
        if defined:
            reasons.append("defined_mask")
        if state >= threshold:
            reasons.append("continuous_state")
        normalized.append(dict(value, material=bool(reasons), material_reasons=reasons))

    material = [value for value in normalized if value["material"]]
    first = material[0] if material else None
    if first is None:
        classification = "no_material_state_or_branch_change"
    elif "discrete_state" in first["material_reasons"]:
        classification = "discrete_state_change_first"
    elif "defined_mask" in first["material_reasons"]:
        classification = "defined_mask_change_first"
    else:
        classification = "continuous_state_change_first"
    return {
        "classification": classification,
        "first_material_day_index": None if first is None else int(first["day_index"]),
        "first_material_reasons": [] if first is None else list(first["material_reasons"]),
        "normalized_state_threshold": threshold,
        "records": normalized,
    }


__all__ = [
    "MATERIAL_NORMALIZED_STATE_CHANGE",
    "classify_material_days",
    "select_largest_positive_window",
]
