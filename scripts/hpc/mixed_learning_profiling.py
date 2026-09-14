"""Component timing for one discarded mixed-learning update.

This module only replays the production update.  It does not own data
selection, differentiation, optimizer construction, or parameter retention.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.hpc import probe_feedback_loss_calibration as calibration
from scripts.hpc import probe_feedback_loss_updates as admission


SCHEMA_VERSION = "mixed_learning_component_profile_v1"
PARITY_RTOL = 1e-5
PARITY_ATOL = 1e-6
MICROBATCH_SIZE = 8
DAILY_BATCH_SIZE = 64
_HORIZONS = (7, 30)


def _recurrent_value(values: Mapping[Any, Any], horizon: int, *, name: str) -> Any:
    matches = [key for key in (f"T{horizon}", horizon) if key in values]
    if len(matches) != 1:
        raise ValueError(f"{name} must contain exactly one T{horizon} entry")
    return values[matches[0]]


def _parameter_inventory(jax: Any, parameters: Any) -> dict[str, Any]:
    leaves = jax.tree_util.tree_leaves(parameters)
    by_dtype: dict[str, dict[str, int]] = {}
    for leaf in leaves:
        dtype = np.dtype(leaf.dtype)
        row = by_dtype.setdefault(
            dtype.name, {"leaf_count": 0, "element_count": 0, "byte_count": 0}
        )
        row["leaf_count"] += 1
        row["element_count"] += int(leaf.size)
        row["byte_count"] += int(leaf.size) * int(dtype.itemsize)
    return {
        "leaf_count": len(leaves),
        "element_count": sum(row["element_count"] for row in by_dtype.values()),
        "byte_count": sum(row["byte_count"] for row in by_dtype.values()),
        "by_dtype": dict(sorted(by_dtype.items())),
    }


def _host_hard_map(jax: Any, hard: Mapping[str, Any], *, name: str) -> dict[str, int]:
    if not isinstance(hard, Mapping) or not hard:
        raise ValueError(f"{name} must return a nonempty hard-count map")
    host = jax.device_get(hard)
    result: dict[str, int] = {}
    for key, value in host.items():
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError(f"{name} hard count {key!r} is not scalar")
        scalar = float(array.reshape(()))
        if not math.isfinite(scalar) or scalar < 0 or scalar != int(scalar):
            raise ValueError(f"{name} hard count {key!r} is invalid")
        result[str(key)] = int(scalar)
    return result


def _summary(samples: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in samples]
    return {
        "samples_seconds": values,
        "median_seconds": float(np.median(values)),
        "mean_seconds": float(np.mean(values)),
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
    }


def _numeric_record(record: Mapping[str, Any]) -> dict[str, float]:
    return {
        "daily_loss": float(record["losses"]["daily"]),
        "T7_loss": float(record["losses"]["T7"]),
        "T30_loss": float(record["losses"]["T30"]),
        "mixed_loss": float(record["mixed_loss"]),
        "daily_gradient_norm": float(record["gradient_norms"]["daily"]),
        "T7_gradient_norm": float(record["gradient_norms"]["T7"]),
        "T30_gradient_norm": float(record["gradient_norms"]["T30"]),
        "combined_gradient_norm": float(record["combined_gradient_norm"]),
        "clip": float(record["clip"]),
    }


def _parity(baseline: Sequence[Mapping[str, Any]], profiled: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reference = _numeric_record(baseline[0])
    comparisons = []
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    passed = True
    for mode, records in (("baseline", baseline), ("instrumented", profiled)):
        for repeat, record in enumerate(records, start=1):
            observed = _numeric_record(record)
            row_passed = True
            for key, expected in reference.items():
                actual = observed[key]
                absolute = abs(actual - expected)
                relative = absolute / max(abs(actual), abs(expected), PARITY_ATOL)
                maximum_absolute_error = max(maximum_absolute_error, absolute)
                maximum_relative_error = max(maximum_relative_error, relative)
                row_passed = bool(
                    row_passed
                    and np.isclose(actual, expected, rtol=PARITY_RTOL, atol=PARITY_ATOL)
                )
            row_passed = bool(
                row_passed
                and record["all_hard_counts_zero"]
                and record["combined_finite"]
                and record["proposal_finite"]
            )
            passed = bool(passed and row_passed)
            comparisons.append({"mode": mode, "repeat": repeat, "passed": row_passed})
    return {
        "rtol": PARITY_RTOL,
        "atol": PARITY_ATOL,
        "passed": passed,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
        "comparisons": comparisons,
        "reference": reference,
    }


def profile_components(
    *,
    jax,
    jnp,
    parameters,
    optimizer,
    sample,
    daily_rows,
    recurrent_rows,
    window_pool,
    daily_executable,
    recurrent_executables,
    coefficients,
    context,
    forward,
    validate_recurrent,
    update,
    add,
    finish,
    combine,
    finite,
    norm_fn,
    repeats=3,
    trace_dir=None,
    emit=None,
    checked_recurrent_executables=None,
    checked_counter_names=(),
):
    """Replay and time the fixed daily64 plus first T7/T30 mixed update.

    ``parameters`` and ``optimizer`` are never replaced or returned.  Every
    optimizer proposal is diagnostic and discarded.
    """

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    rows = admission._rows(daily_rows)
    if len(rows) != DAILY_BATCH_SIZE:
        raise ValueError("mixed profiling requires exactly the fixed daily64 rows")
    recurrent = {
        horizon: _recurrent_value(recurrent_rows, horizon, name="recurrent_rows")
        for horizon in _HORIZONS
    }
    executables = {
        horizon: _recurrent_value(
            recurrent_executables, horizon, name="recurrent_executables"
        )
        for horizon in _HORIZONS
    }
    checked = (None if checked_recurrent_executables is None else {
        horizon: _recurrent_value(checked_recurrent_executables, horizon, name='checked executables')
        for horizon in _HORIZONS
    })
    notify: Callable[..., None] = emit if emit is not None else lambda stage, **info: None
    inventory = _parameter_inventory(jax, parameters)

    def timed(timings: dict[str, float] | None, label: str, operation: Callable[[], Any]):
        if timings is None:
            return operation()
        began = time.perf_counter()
        result = operation()
        jax.block_until_ready(result)
        elapsed = time.perf_counter() - began
        timings[label] = timings.get(label, 0.0) + elapsed
        return result

    def replay(*, instrumented: bool) -> tuple[dict[str, Any], dict[str, float]]:
        timings: dict[str, float] | None = {} if instrumented else None
        daily_total = jax.tree_util.tree_map(jnp.zeros_like, parameters)
        daily_loss = 0.0
        all_hard_counts_zero = True

        for start in range(0, DAILY_BATCH_SIZE, MICROBATCH_SIZE):
            selected = rows[start : start + MICROBATCH_SIZE]
            if instrumented:
                batch = timed(timings, "daily_read_pack", lambda selected=selected: sample(selected))
                batch = timed(timings, "daily_transfer", lambda batch=batch: jax.device_put(batch))
            else:
                batch = sample(selected)
            outputs = timed(
                timings,
                "daily_backward",
                lambda batch=batch: daily_executable(parameters, batch, coefficients),
            )
            (value, auxiliary), gradient = outputs

            def check_daily():
                hard = calibration._hard_summary(
                    jax.device_get(auxiliary["hard"]), batch_size=MICROBATCH_SIZE
                )
                if not np.isfinite(float(value)) or not hard["all_zero"]:
                    raise ValueError("daily profiling replay failed physical/finite gates")
                return hard

            hard = timed(timings, "daily_checks", check_daily)
            all_hard_counts_zero = bool(all_hard_counts_zero and hard["all_zero"])
            daily_total = timed(
                timings,
                "daily_accumulate",
                lambda daily_total=daily_total, gradient=gradient: add(
                    daily_total, gradient, MICROBATCH_SIZE
                ),
            )
            daily_loss += float(value) / (DAILY_BATCH_SIZE / MICROBATCH_SIZE)
            del batch, outputs, value, auxiliary, gradient, hard

        daily_mean = timed(
            timings,
            "daily_accumulate",
            lambda daily_total=daily_total: finish(daily_total, DAILY_BATCH_SIZE),
        )
        daily_norm = timed(timings, "daily_accumulate", lambda: norm_fn(daily_mean))
        component_norms = {"daily": float(daily_norm)}
        losses = {"daily": daily_loss}
        mixed = daily_mean
        del daily_total, daily_mean, daily_norm

        for horizon in _HORIZONS:
            key = f"T{horizon}"
            if instrumented:
                case = timed(
                    timings,
                    key + "_read_pack_transfer",
                    lambda horizon=horizon: jax.device_put(window_pool.case(jax, recurrent[horizon])),
                )
            else:
                case = window_pool.case(jax, recurrent[horizon])
            initial, discrete, days, targets, expected_discrete = case
            if checked is None:
                rollout = timed(
                    timings,
                    key + "_forward",
                    lambda: forward(parameters, initial, discrete, days),
                )
                observations = rollout[1]
                if not instrumented:
                    jax.block_until_ready(observations)

                def check_recurrent():
                    hard = _host_hard_map(
                        jax, validate_recurrent(case, observations), name=key + " validation",
                    )
                    if any(hard.values()):
                        raise ValueError(key + " profiling replay failed physical/finite gates")
                    return hard

                recurrent_hard = timed(timings, key + "_validity_loss_checks", check_recurrent)
                del rollout, observations
                value, gradient = timed(
                    timings, key + "_backward",
                    lambda horizon=horizon: executables[horizon](parameters, *case[:4], context),
                )
            else:
                from scripts.hpc.checked_recurrent_execution import decode_counts

                (value, counts), gradient = timed(
                    timings, key + '_checked_backward',
                    lambda horizon=horizon: checked[horizon](parameters, *case[:4], context, case[4]),
                )

                def check_compact():
                    loss_host, counts_host = jax.device_get((value, counts))
                    hard = decode_counts(checked_counter_names, counts_host)
                    if not np.isfinite(loss_host) or any(hard.values()):
                        raise ValueError(key + ' checked replay failed physical/finite gates')
                    return hard

                recurrent_hard = timed(timings, key + '_compact_checks', check_compact)
                del counts
            all_hard_counts_zero = bool(
                all_hard_counts_zero and not any(recurrent_hard.values())
            )
            del recurrent_hard
            if not np.isfinite(float(value)):
                raise ValueError(key + " profiling loss is nonfinite")

            if instrumented:
                def norm_and_merge():
                    component_norm = norm_fn(gradient)
                    merged = combine(mixed, gradient, 1.0, 0.5)
                    return component_norm, merged

                recurrent_norm, next_mixed = timed(
                    timings, key + "_norm_merge", norm_and_merge
                )
            else:
                recurrent_norm = norm_fn(gradient)
                float(recurrent_norm)
                next_mixed = combine(mixed, gradient, 1.0, 0.5)
                jax.block_until_ready(next_mixed)
            losses[key] = float(value)
            component_norms[key] = float(recurrent_norm)
            mixed = next_mixed
            del (
                case,
                initial,
                discrete,
                days,
                targets,
                expected_discrete,
                value,
                gradient,
                recurrent_norm,
                next_mixed,
            )

        combined_finite = bool(
            timed(timings, "combined_finite", lambda: finite(mixed))
        )
        if not combined_finite or not all(
            np.isfinite(value) and value > 0 for value in component_norms.values()
        ):
            raise ValueError("profiled component/mixed gradient is nonfinite or empty")
        proposed, proposed_optimizer, combined_norm, clip = timed(
            timings, "adamw", lambda: update(parameters, mixed, optimizer)
        )
        if not instrumented:
            jax.block_until_ready((proposed, proposed_optimizer, combined_norm, clip))
        proposal_finite = bool(
            timed(
                timings,
                "proposed_finite",
                lambda: finite((proposed, proposed_optimizer)),
            )
        )
        combined_norm_value, clip_value = float(combined_norm), float(clip)
        if (
            not proposal_finite
            or not np.isfinite(combined_norm_value)
            or not 0 < clip_value <= 1
        ):
            raise ValueError("profiled AdamW proposal is invalid")
        record = {
            "losses": losses,
            "mixed_loss": losses["daily"] + 0.5 * (losses["T7"] + losses["T30"]),
            "gradient_norms": component_norms,
            "combined_gradient_norm": combined_norm_value,
            "clip": clip_value,
            "all_hard_counts_zero": all_hard_counts_zero,
            "combined_finite": combined_finite,
            "proposal_finite": proposal_finite,
        }
        del mixed, proposed, proposed_optimizer, combined_norm, clip
        return record, timings or {}

    notify("profiling_warmup", discarded=True)
    warmup_record, _ = replay(instrumented=False)
    del warmup_record

    notify("profiling_instrumented_warmup", discarded=True)
    warmup_record, warmup_timings = replay(instrumented=True)
    del warmup_record, warmup_timings

    baseline_records: list[dict[str, Any]] = []
    baseline_seconds: list[float] = []
    for repeat in range(repeats):
        notify("profiling_baseline", repeat=repeat + 1, repeats=repeats)
        began = time.perf_counter()
        record, _ = replay(instrumented=False)
        jax.block_until_ready(record)
        baseline_seconds.append(time.perf_counter() - began)
        baseline_records.append(record)

    profiled_records: list[dict[str, Any]] = []
    profiled_seconds: list[float] = []
    component_samples: dict[str, list[float]] = {}
    for repeat in range(repeats):
        notify("profiling_instrumented", repeat=repeat + 1, repeats=repeats)
        began = time.perf_counter()
        record, component_times = replay(instrumented=True)
        jax.block_until_ready(record)
        profiled_seconds.append(time.perf_counter() - began)
        profiled_records.append(record)
        for key, value in component_times.items():
            component_samples.setdefault(key, []).append(value)

    baseline_timing = _summary(baseline_seconds)
    profiled_timing = _summary(profiled_seconds)
    component_timing = {key: _summary(values) for key, values in sorted(component_samples.items())}
    unattributed = [
        overall - sum(values[index] for values in component_samples.values())
        for index, overall in enumerate(profiled_seconds)
    ]
    profile_median = profiled_timing["median_seconds"]
    for row in component_timing.values():
        row["fraction_of_profile_median"] = row["median_seconds"] / profile_median
    baseline_median = baseline_timing["median_seconds"]
    inflation = profile_median / baseline_median
    parity = _parity(baseline_records, profiled_records)
    if not parity["passed"]:
        raise ValueError("instrumented replay differs from the production-like baseline")

    trace = {"requested": trace_dir is not None, "status": "not_requested"}
    if trace_dir is not None:
        destination = Path(trace_dir).resolve()
        trace = {"requested": True, "status": "running", "directory": str(destination)}
        notify("profiling_trace", directory=str(destination))
        try:
            destination.mkdir(parents=True, exist_ok=True)
            with jax.profiler.trace(str(destination)):
                trace_record, _ = replay(instrumented=True)
                jax.block_until_ready(trace_record)
            trace["status"] = "completed"
        except Exception as error:  # Trace availability is diagnostic, not an update gate.
            trace.update(status="failed", error=f"{type(error).__name__}: {error}")

    result = {
        "schema_version": SCHEMA_VERSION,
        "recurrent_execution": 'compact_checked' if checked is not None else 'separate_forward_checks',
        "status": "completed",
        "repeats": repeats,
        "daily_batch_size": DAILY_BATCH_SIZE,
        "microbatch_size": MICROBATCH_SIZE,
        "parameters": inventory,
        "baseline": {**baseline_timing, "records": baseline_records},
        "instrumented": {
            "overall": {**profiled_timing, "records": profiled_records},
            "components": component_timing,
            "unattributed": _summary(unattributed),
        },
        "profile_to_baseline_inflation": {
            "ratio": inflation,
            "fraction": inflation - 1.0,
        },
        "parity": parity,
        "trace": trace,
        "retained_parameter_or_optimizer_state": False,
    }
    notify(
        "profiling_completed",
        baseline_median_seconds=baseline_median,
        profile_median_seconds=profile_median,
        profile_to_baseline_ratio=inflation,
    )
    return result


__all__ = ["profile_components"]
