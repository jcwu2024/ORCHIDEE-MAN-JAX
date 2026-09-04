"""Classify exact-zero versus smallest-step coupled-GPP evaluations."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

BASELINE_REPRODUCTION_RTOL = 1.0e-6
BASELINE_REPRODUCTION_ATOL = 1.0e-10
MATERIAL_RELATIVE_CHANGE = 1.0e-3
LOCALIZED_POSITIVE_DELTA_FRACTION = 0.9


def _relative_change(before: float, after: float) -> float:
    first = float(before)
    second = float(after)
    if not np.isfinite(first) or first <= 0.0 or not np.isfinite(second) or second < 0.0:
        raise ValueError("zero-displacement score is invalid")
    return (second - first) / first


def classify_zero_displacement(
    records: Sequence[Mapping[str, Any]],
    *,
    reproduction_rtol: float = BASELINE_REPRODUCTION_RTOL,
    reproduction_atol: float = BASELINE_REPRODUCTION_ATOL,
    material_relative_change: float = MATERIAL_RELATIVE_CHANGE,
    localized_positive_delta_fraction: float = LOCALIZED_POSITIVE_DELTA_FRACTION,
) -> Mapping[str, Any]:
    """Separate a baseline-path mismatch from a localized nonzero-step jump."""

    values = tuple(records)
    if not values:
        raise ValueError("zero-displacement classification requires records")
    updates = tuple(int(value["artifact_update"]) for value in values)
    if len(set(updates)) != len(updates):
        raise ValueError("zero-displacement artifact registry is not unique")

    normalized = []
    for value in values:
        baseline = float(value["gradient_baseline"])
        exact_zero = float(value["exact_zero_forward"])
        smallest = float(value["smallest_nonzero_forward"])
        zero_matches = bool(
            np.isclose(
                exact_zero,
                baseline,
                rtol=float(reproduction_rtol),
                atol=float(reproduction_atol),
            )
        )
        normalized.append(
            {
                "artifact_update": int(value["artifact_update"]),
                "gradient_baseline": baseline,
                "exact_zero_forward": exact_zero,
                "smallest_nonzero_forward": smallest,
                "exact_zero_matches_gradient_baseline": zero_matches,
                "exact_zero_vs_gradient_baseline_relative_change": _relative_change(baseline, exact_zero),
                "smallest_nonzero_vs_exact_zero_relative_change": _relative_change(exact_zero, smallest),
                "smallest_nonzero_minus_exact_zero": smallest - exact_zero,
            }
        )

    baseline_mean = float(np.mean([value["gradient_baseline"] for value in normalized]))
    zero_mean = float(np.mean([value["exact_zero_forward"] for value in normalized]))
    smallest_mean = float(np.mean([value["smallest_nonzero_forward"] for value in normalized]))
    positive_deltas = np.asarray(
        [max(0.0, float(value["smallest_nonzero_minus_exact_zero"])) for value in normalized],
        dtype=np.float64,
    )
    positive_total = float(np.sum(positive_deltas))
    dominant_index = int(np.argmax(positive_deltas)) if positive_total > 0.0 else int(
        np.argmax(
            np.abs(
                np.asarray(
                    [value["smallest_nonzero_minus_exact_zero"] for value in normalized],
                    dtype=np.float64,
                )
            )
        )
    )
    dominant_fraction = float(positive_deltas[dominant_index] / positive_total) if positive_total > 0.0 else 0.0
    exact_zero_reproduces_baseline = all(
        bool(value["exact_zero_matches_gradient_baseline"]) for value in normalized
    )
    smallest_relative_change = _relative_change(zero_mean, smallest_mean)

    if not exact_zero_reproduces_baseline:
        classification = "gradient_baseline_not_reproduced_by_exact_zero_forward"
        next_gate = "compare_gradient_wrapper_forward_with_pure_forward_on_mismatching_artifacts"
    elif smallest_relative_change > float(material_relative_change):
        if dominant_fraction >= float(localized_positive_delta_fraction):
            classification = "localized_nonzero_step_discontinuity"
            next_gate = "trace_first_rollout_branch_change_for_dominant_artifact"
        else:
            classification = "distributed_nonzero_step_discontinuity"
            next_gate = "trace_rollout_branch_changes_across_affected_artifacts"
    else:
        classification = "no_material_nonzero_step_discontinuity"
        next_gate = "audit_aggregate_scale_record_construction"

    return {
        "classification": classification,
        "next_gate": next_gate,
        "exact_zero_reproduces_gradient_baseline": exact_zero_reproduces_baseline,
        "artifact_count": len(normalized),
        "gradient_baseline_mean": baseline_mean,
        "exact_zero_forward_mean": zero_mean,
        "smallest_nonzero_forward_mean": smallest_mean,
        "exact_zero_vs_gradient_baseline_relative_change": _relative_change(baseline_mean, zero_mean),
        "smallest_nonzero_vs_exact_zero_relative_change": smallest_relative_change,
        "dominant_positive_delta_artifact_update": normalized[dominant_index]["artifact_update"],
        "dominant_positive_delta_fraction": dominant_fraction,
        "thresholds": {
            "baseline_reproduction_rtol": float(reproduction_rtol),
            "baseline_reproduction_atol": float(reproduction_atol),
            "material_relative_change": float(material_relative_change),
            "localized_positive_delta_fraction": float(localized_positive_delta_fraction),
        },
        "records": normalized,
    }


__all__ = [
    "BASELINE_REPRODUCTION_ATOL",
    "BASELINE_REPRODUCTION_RTOL",
    "LOCALIZED_POSITIVE_DELTA_FRACTION",
    "MATERIAL_RELATIVE_CHANGE",
    "classify_zero_displacement",
]
