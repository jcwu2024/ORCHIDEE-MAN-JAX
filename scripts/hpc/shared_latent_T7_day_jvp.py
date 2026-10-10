"""Same-precision first-day state-JVP comparison for the canonical T7 step."""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np


BATCH_SIZE = 4
SAMPLE_INDEX = 3
HORIZON = 7
RTOL = 1.0e-5
ATOL = 1.0e-6


class FirstDayInputs(NamedTuple):
    initial_continuous: Any
    initial_discrete: Any
    day: Any


class FloatStepOutput(NamedTuple):
    next_continuous: Any
    observations: Any


class DayJVPResult(NamedTuple):
    primal: FloatStepOutput
    tangent: FloatStepOutput
    auxiliary: Any


def _dtype(value: Any) -> np.dtype:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        dtype = np.asarray(value).dtype
    return np.dtype(dtype)


def _require_state_direction(state: Any, direction: Any, *, batched: bool) -> None:
    state_shape = tuple(getattr(state, "shape", np.shape(state)))
    direction_shape = tuple(getattr(direction, "shape", np.shape(direction)))
    expected_rank = 2 if batched else 1
    if len(state_shape) != expected_rank or state_shape != direction_shape:
        raise ValueError("state/direction shape differs from the one-day JVP contract")
    if _dtype(state) != np.dtype("float64") or _dtype(direction) != np.dtype("float64"):
        raise ValueError("one-day state and direction must both be FP64")
    if batched and state_shape[0] != BATCH_SIZE:
        raise ValueError(f"B4 one-day JVP requires exactly {BATCH_SIZE} cases")


def first_day_inputs(jax: Any, case: Any, *, batched: bool) -> FirstDayInputs:
    """Select day zero from the existing five-part B1 case or stacked B4 case.

    ``case.days`` has a leading day axis for B1 and ``[batch, day, ...]``
    axes for B4. All five prepared case arguments remain otherwise untouched;
    targets and loss context are not inputs to a daily transition.
    """
    if len(case) != 5:
        raise ValueError("a prepared recurrent case must retain its five-part contract")
    state, discrete = case[0], case[1]
    days = case.days if hasattr(case, "days") else case[2]
    day_axis = 1 if batched else 0
    day_count = None

    def take_first(value: Any) -> Any:
        nonlocal day_count
        shape = tuple(getattr(value, "shape", np.shape(value)))
        if batched and (not shape or shape[0] != BATCH_SIZE):
            raise ValueError("B4 prepared day leaf is missing its four-case batch axis")
        if len(shape) <= day_axis:
            raise ValueError("prepared day leaf is missing its day axis")
        if day_count is None:
            day_count = shape[day_axis]
        elif shape[day_axis] != day_count:
            raise ValueError("prepared day leaves disagree on their day-axis length")
        if shape[day_axis] != HORIZON:
            raise ValueError(f"prepared case requires a {HORIZON}-day input axis")
        return value[(slice(None), 0) + (slice(None),) * (len(shape) - 2)] if batched else value[0]

    first_day = jax.tree_util.tree_map(take_first, days)
    return FirstDayInputs(state, discrete, first_day)


def _split_observations(jax: Any, observations: Any) -> tuple[Any, Any]:
    if not isinstance(observations, dict):
        raise TypeError("canonical daily step observations must be a dict")
    leaves, treedef = jax.tree_util.tree_flatten(observations)
    floating, auxiliary = [], []
    for value in leaves:
        if _dtype(value).kind == "f":
            floating.append(value)
            auxiliary.append(None)
        else:
            floating.append(None)
            auxiliary.append(value)
    return treedef.unflatten(floating), treedef.unflatten(auxiliary)


def route_functions(jax: Any, step: Any):
    """Return same-precision B1 and four-case vmapped one-day JVP functions.

    B1 arguments are ``(parameters, state, discrete, first_day, direction)``.
    B4 arguments replace state/discrete/day/direction with their four-case
    batched versions. The daily transition itself is never averaged or scaled.
    Both functions return ``DayJVPResult``. Only next continuous state and
    floating observation leaves are differentiated; next discrete state and
    the remaining observations are returned through ``jax.jvp(has_aux=True)``.
    """

    def one(parameters, state, discrete, day):
        next_continuous, next_discrete, observations = step(parameters, state, discrete, day)
        floating_observations, auxiliary_observations = _split_observations(jax, observations)
        return FloatStepOutput(next_continuous, floating_observations), (next_discrete, auxiliary_observations)

    def b1(parameters, state, discrete, day, direction):
        _require_state_direction(state, direction, batched=False)
        primal, tangent, auxiliary = jax.jvp(
            lambda value: one(parameters, value, discrete, day),
            (state,),
            (direction,),
            has_aux=True,
        )
        return DayJVPResult(primal, tangent, auxiliary)

    def b4(parameters, states, discretes, days, directions):
        _require_state_direction(states, directions, batched=True)

        def all_cases(values):
            return jax.vmap(one, in_axes=(None, 0, 0, 0))(parameters, values, discretes, days)

        primal, tangent, auxiliary = jax.jvp(
            all_cases,
            (states,),
            (directions,),
            has_aux=True,
        )
        return DayJVPResult(primal, tangent, auxiliary)

    return b1, b4


def _select_sample(jax: Any, tree: Any, sample_index: int, batch_size: int) -> Any:
    if (not isinstance(sample_index, (int, np.integer)) or isinstance(sample_index, (bool, np.bool_))
            or not 0 <= int(sample_index) < batch_size):
        raise ValueError("sample index is outside the B4 batch")
    def select(value: Any) -> Any:
        shape = tuple(getattr(value, "shape", np.shape(value)))
        if _dtype(value).kind in "biufc":
            if not shape or shape[0] != batch_size:
                raise ValueError("numeric B4 result leaf is missing its four-case batch axis")
            return value[int(sample_index)]
        return value

    return jax.tree_util.tree_map(select, tree)


def _float_tree_comparison(jax: Any, actual: Any, reference: Any, *, prefix: str,
                           rtol: float, atol: float) -> list[dict[str, Any]]:
    actual_paths, actual_tree = jax.tree_util.tree_flatten_with_path(actual)
    reference_paths, reference_tree = jax.tree_util.tree_flatten_with_path(reference)
    if actual_tree != reference_tree or [jax.tree_util.keystr(path) for path, _ in actual_paths] != [
            jax.tree_util.keystr(path) for path, _ in reference_paths]:
        raise ValueError(prefix + " pytree structure differs")
    records = []
    for (path, actual_value), (_, reference_value) in zip(actual_paths, reference_paths, strict=True):
        actual_array, reference_array = np.asarray(actual_value), np.asarray(reference_value)
        if actual_array.shape != reference_array.shape:
            raise ValueError(prefix + " leaf shape differs at " + jax.tree_util.keystr(path))
        if actual_array.dtype != reference_array.dtype or actual_array.dtype.kind != "f":
            raise ValueError(prefix + " leaf dtype differs or is not floating at " + jax.tree_util.keystr(path))
        finite = np.isfinite(actual_array) & np.isfinite(reference_array)
        difference = np.zeros(actual_array.shape, dtype=np.float64)
        difference[finite] = np.abs(actual_array[finite] - reference_array[finite])
        allowance = atol + rtol * np.abs(reference_array.astype(np.float64, copy=False))
        failed = ~finite | (difference > allowance)
        ratio = np.zeros(actual_array.shape, dtype=np.float64)
        positive = allowance > 0
        ratio[positive] = difference[positive] / allowance[positive]
        ratio[~positive & (difference > 0)] = np.inf
        if np.any(~finite):
            ratio[~finite] = np.inf
        flat_worst = int(np.argmax(ratio)) if ratio.size else 0
        worst_index = ([int(index) for index in np.unravel_index(flat_worst, ratio.shape)]
                       if ratio.size and ratio.shape else [])
        finite_differences = difference[finite]
        path_text = prefix + jax.tree_util.keystr(path)
        record = {
            "path": path_text,
            "shape": list(actual_array.shape),
            "dtype": str(actual_array.dtype),
            "failed_coordinates": int(np.count_nonzero(failed)),
            "nonfinite_coordinates": int(np.count_nonzero(~finite)),
            "max_absolute_error": float(np.max(finite_differences, initial=0.0)),
            "max_error_to_allowance": float(np.max(ratio, initial=0.0)),
            "worst_position": worst_index,
            "passed": bool(not np.any(failed)),
        }
        if path_text.endswith("['budgets']"):
            record["finite_gate"] = bool(np.all(np.isfinite(actual_array))
                and np.all(np.isfinite(reference_array)))
        records.append(record)
    return records


def _auxiliary_comparison(jax: Any, actual: Any, reference: Any) -> list[dict[str, Any]]:
    actual_paths, actual_tree = jax.tree_util.tree_flatten_with_path(actual)
    reference_paths, reference_tree = jax.tree_util.tree_flatten_with_path(reference)
    if actual_tree != reference_tree or [jax.tree_util.keystr(path) for path, _ in actual_paths] != [
            jax.tree_util.keystr(path) for path, _ in reference_paths]:
        raise ValueError("non-floating auxiliary pytree structure differs")
    records = []
    for (path, actual_value), (_, reference_value) in zip(actual_paths, reference_paths, strict=True):
        actual_array, reference_array = np.asarray(actual_value), np.asarray(reference_value)
        if actual_array.shape != reference_array.shape or actual_array.dtype != reference_array.dtype:
            raise ValueError("non-floating auxiliary shape/dtype differs at " + jax.tree_util.keystr(path))
        equal = bool(np.array_equal(actual_array, reference_array, equal_nan=True))
        record = {"path": jax.tree_util.keystr(path), "shape": list(actual_array.shape),
                  "dtype": str(actual_array.dtype), "exact_match": equal}
        if jax.tree_util.keystr(path).endswith("['learned_output_nonfinite']"):
            record["zero_gate"] = bool(np.all(actual_array == 0) and np.all(reference_array == 0))
            record["actual_values"] = actual_array.tolist()
            record["reference_values"] = reference_array.tolist()
        if jax.tree_util.keystr(path).endswith("['budgets']"):
            record["finite_gate"] = bool(np.all(np.isfinite(actual_array)) and np.all(np.isfinite(reference_array)))
        records.append(record)
    return records


def compare_results(jax: Any, actual: DayJVPResult, reference: DayJVPResult, *,
                    sample_index: int | None = None, batch_size: int = BATCH_SIZE,
                    rtol: float = RTOL, atol: float = ATOL) -> dict[str, Any]:
    """Strictly compare full state/float outputs, tangents and exact auxiliaries.

    Pass ``sample_index=3`` when comparing a B1 result with the B4 result. The
    return value retains all per-leaf failures; only ABI/shape/dtype mismatches
    raise. Nonfinite numbers are recorded as numerical failures.
    """
    if not isinstance(actual, DayJVPResult) or not isinstance(reference, DayJVPResult):
        raise TypeError("day JVP results must use the DayJVPResult contract")
    if sample_index is not None:
        reference = _select_sample(jax, reference, sample_index, batch_size)
    records = []
    for label, field in (("primal", "primal"), ("tangent", "tangent")):
        actual_payload = getattr(actual, field)
        reference_payload = getattr(reference, field)
        records.extend(_float_tree_comparison(jax, actual_payload.next_continuous,
            reference_payload.next_continuous, prefix=label + ".next_continuous", rtol=rtol, atol=atol))
        records.extend(_float_tree_comparison(jax, actual_payload.observations,
            reference_payload.observations, prefix=label + ".observations", rtol=rtol, atol=atol))
    auxiliary = _auxiliary_comparison(jax, actual.auxiliary, reference.auxiliary)
    return {
        "passed": all(row["passed"] for row in records)
            and all(row["exact_match"] for row in auxiliary)
            and all(row.get("zero_gate", True) for row in auxiliary)
            and all(row.get("finite_gate", True) for row in auxiliary),
        "rtol": float(rtol),
        "atol": float(atol),
        "float_leaf_count": len(records),
        "failed_float_coordinates": sum(row["failed_coordinates"] for row in records),
        "nonfinite_float_coordinates": sum(row["nonfinite_coordinates"] for row in records),
        "float_leaves": records,
        "auxiliary_leaf_count": len(auxiliary),
        "auxiliary_mismatches": sum(not row["exact_match"] for row in auxiliary),
        "auxiliary_leaves": auxiliary,
        "state_output_accounting": (
            "next_continuous is compared once; canonical daily observations exclude the rollout-added state key"
        ),
        "scope": "single_transition_only_not_T7_or_loss_acceptance",
    }
