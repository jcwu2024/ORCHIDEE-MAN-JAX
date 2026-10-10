"""Compact evidence for genuine full-state self-replay tolerance failures.

This helper only describes NumPy arrays already preserved by the caller. It
does not change the algorithm, comparison tolerance, or acceptance decision.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

RTOL = 1e-5
ATOL = 1e-6


def state_replay_execution_metadata(*, execution_architecture, execution_identity):
    """Keep a snapshot's executed architecture and validated model identity explicit."""
    if not isinstance(execution_architecture, str) or not execution_architecture:
        raise ValueError("state replay execution architecture must be explicit")
    if not isinstance(execution_identity, Mapping) or not execution_identity:
        raise ValueError("state replay execution identity must be explicit")
    return {"identity": dict(execution_identity),
            "execution_architecture": execution_architecture,
            "execution_scope": "same_architecture_full_initial_state_AD_replay"}


def _number(value):
    value = float(value)
    if np.isnan(value):
        return "NaN"
    if np.isposinf(value):
        return "+Infinity"
    if np.isneginf(value):
        return "-Infinity"
    return value


def _context(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _context(child) for key, child in value.items()}
    array = np.asarray(value)
    if array.ndim == 0:
        return _number(array)
    if array.size <= 16:
        return [_context(child) for child in array]
    finite = np.isfinite(array)
    return {"shape": list(array.shape), "dtype": str(array.dtype), "coordinates": int(array.size),
            "finite_coordinates": int(np.count_nonzero(finite)),
            "minimum_finite": _number(np.min(array[finite])) if np.any(finite) else None,
            "maximum_finite": _number(np.max(array[finite])) if np.any(finite) else None}


def _scale_arrays(value, shape):
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): _scale_arrays(child, shape) for key, child in value.items()}
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("state scales must be numeric scalars or arrays")
    try:
        return np.broadcast_to(array, shape)
    except ValueError as error:
        raise ValueError("state scale shape cannot map to state coordinates") from error


def _scale_at(value, index):
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {key: _scale_at(child, index) for key, child in value.items()}
    return _number(value[index])


def summarize_state_replay(actual, reference, *, input_state, state_scales=None,
                           loss_scales=None, max_failed_rows=32):
    """Describe fixed-tolerance failures; scales are context, never rescaling.

    ``state_scales`` accepts broadcastable numeric arrays/scalars or nested
    mappings of those arrays. Each displayed failed coordinate gets its own
    scale value. ``loss_scales`` is caller-provided context: small arrays and
    mappings retain values, while large arrays receive compact metadata.
    ``failed_rows`` is capped; global counts always include every coordinate.
    Nonfinite values are explicit strings so strict JSON serialization works.
    """
    actual, reference, initial = map(np.asarray, (actual, reference, input_state))
    if actual.shape != reference.shape or actual.shape != initial.shape:
        raise ValueError("actual/reference/input-state shapes must match")
    if not np.issubdtype(actual.dtype, np.floating) or not np.issubdtype(reference.dtype, np.floating):
        raise ValueError("state gradients must be floating arrays")
    if not isinstance(max_failed_rows, int) or isinstance(max_failed_rows, bool) or max_failed_rows < 0:
        raise ValueError("max_failed_rows must be a nonnegative integer")
    scales = _scale_arrays(state_scales, actual.shape)
    finite_actual, finite_reference = np.isfinite(actual), np.isfinite(reference)
    finite = finite_actual & finite_reference
    with np.errstate(invalid="ignore", over="ignore"):
        close = np.isclose(actual, reference, rtol=RTOL, atol=ATOL)
        failed = ~finite | ~close
        absolute_error = np.abs(actual - reference)
        tolerance = ATOL + RTOL * np.abs(reference)
    indices = np.flatnonzero(failed)
    rows = []
    for flat in indices[:max_failed_rows]:
        index = tuple(int(value) for value in np.unravel_index(int(flat), actual.shape))
        row = {"flat_index": int(flat), "index": list(index),
               "actual": _number(actual[index]), "reference": _number(reference[index]),
               "actual_finite": bool(finite_actual[index]), "reference_finite": bool(finite_reference[index]),
               "absolute_error": _number(absolute_error[index]), "effective_tolerance": _number(tolerance[index]),
               "tolerance_excess": _number(absolute_error[index] - tolerance[index]) if finite[index] else None,
               "input_state": _number(initial[index]), "input_state_finite": bool(np.isfinite(initial[index])),
               "state_scales": _scale_at(scales, index),
               "reason": "tolerance_exceeded" if finite[index] else "nonfinite_gradient"}
        if actual.ndim == 2:
            row.update(sample_row=index[0], state_coordinate=index[1])
        elif actual.ndim == 1:
            row["state_coordinate"] = index[0]
        rows.append(row)
    finite_errors = absolute_error[finite]
    return {"passed": not bool(indices.size), "rtol": RTOL, "atol": ATOL,
            "comparison": "finite actual/reference AND np.isclose(actual, reference); reference sets relative tolerance",
            "shape": list(actual.shape), "actual_dtype": str(actual.dtype), "reference_dtype": str(reference.dtype),
            "gradient_dtypes_match": bool(actual.dtype == reference.dtype),
            "input_state_dtype": str(initial.dtype), "coordinates": int(actual.size),
            "failed_coordinates": int(indices.size), "finite_tolerance_failed_coordinates": int(np.count_nonzero(failed & finite)),
            "actual_nonfinite_coordinates": int(np.count_nonzero(~finite_actual)),
            "reference_nonfinite_coordinates": int(np.count_nonzero(~finite_reference)),
            "input_state_nonfinite_coordinates": int(np.count_nonzero(~np.isfinite(initial))),
            "maximum_absolute_finite_error": _number(np.max(finite_errors)) if finite_errors.size else None,
            "failed_rows": rows, "failed_rows_omitted": int(max(0, indices.size - len(rows))),
            "state_scale_context": _context(state_scales), "loss_scale_context": _context(loss_scales),
            "full_arrays_preserved_by": "caller", "cause_attribution": None}
