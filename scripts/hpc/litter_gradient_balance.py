"""Choose one diagnostic litter weight from fixed TRAIN gradient geometry.

This balances parameter-space gradient norms, not physical accuracy. It is
not a production stopping rule or a claim that competing objectives agree.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

COMPONENTS = ("daily_old", "litter", "T7", "T30")
ORIGINAL_WEIGHTS = np.asarray([1.0, 0.0, 0.5, 0.5])


def select_litter_weight(
    gram_matrices: Sequence[Any],
    *,
    target_median_ratio: float = 0.5,
    maximum_ratio: float = 1.0,
    maximum_weight: float = 1e-3,
) -> dict[str, Any]:
    """Select a constant coefficient before any optimizer update.

    For r_i=||g_litter||/||g_old_mixed||, choose
    min(target/median(r), cap/max(r), maximum_weight). Two fixed TRAIN
    calibration batches are required. Vector cross-terms are included in
    the old mixed norm; summing component norms would be incorrect.
    """
    settings = (target_median_ratio, maximum_ratio, maximum_weight)
    if not all(np.isfinite(value) and value > 0 for value in settings):
        raise ValueError("gradient balance settings must be finite and positive")
    if target_median_ratio > maximum_ratio:
        raise ValueError("target ratio must not exceed the per-batch cap")
    if len(gram_matrices) != 2:
        raise ValueError("two declared TRAIN calibration batches are required")
    grams = []
    ratios = []
    old_norms = []
    litter_norms = []
    for gram in gram_matrices:
        matrix = np.asarray(gram, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("finite four-component Gram matrix required")
        if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
            raise ValueError("gradient Gram matrix must be symmetric")
        old_squared = float(ORIGINAL_WEIGHTS @ matrix @ ORIGINAL_WEIGHTS)
        litter_squared = float(matrix[1, 1])
        if old_squared <= 0 or litter_squared <= 0 or np.any(np.diag(matrix) < 0):
            raise ValueError("old mixed and litter gradients must have resolved positive norms")
        old_norm = float(np.sqrt(old_squared))
        litter_norm = float(np.sqrt(litter_squared))
        grams.append(matrix)
        old_norms.append(old_norm)
        litter_norms.append(litter_norm)
        ratios.append(litter_norm / old_norm)

    # Select from ratios themselves, not their reciprocals: reciprocal and
    # median do not commute for an even number of calibration batches.
    selected = float(min(
        target_median_ratio / np.median(ratios),
        maximum_ratio / max(ratios),
        maximum_weight,
    ))
    weights = ORIGINAL_WEIGHTS.copy()
    weights[1] = selected
    panels = []
    for matrix, old_norm, litter_norm in zip(grams, old_norms, litter_norms, strict=True):
        cross = float(ORIGINAL_WEIGHTS @ matrix[:, 1])
        mixed_norm = float(np.sqrt(max(float(weights @ matrix @ weights), 0.0)))
        panels.append({
            "component_norms": dict(zip(COMPONENTS, np.sqrt(np.diag(matrix)).tolist(), strict=True)),
            "old_mixed_norm": old_norm,
            "weighted_litter_norm": selected * litter_norm,
            "weighted_litter_to_old_mixed_ratio": selected * litter_norm / old_norm,
            "litter_old_mixed_cosine": cross / (old_norm * litter_norm),
            "candidate_mixed_norm": mixed_norm,
            "candidate_old_mixed_cosine": (
                (old_norm**2 + selected * cross) / (old_norm * mixed_norm)
                if mixed_norm > 0 else None
            ),
        })
    return {
        "rule": "min_target_median_norm_ratio_per_batch_cap_and_weight_ceiling",
        "selected_weight": selected,
        "target_median_ratio": float(target_median_ratio),
        "maximum_ratio": float(maximum_ratio),
        "maximum_weight": float(maximum_weight),
        "components": list(COMPONENTS),
        "original_weights": ORIGINAL_WEIGHTS.tolist(),
        "candidate_weights": weights.tolist(),
        "panels": panels,
        "selection_data": "two_fixed_TRAIN_batches_at_identical_source_parameters",
        "scientific_acceptance": False,
    }
