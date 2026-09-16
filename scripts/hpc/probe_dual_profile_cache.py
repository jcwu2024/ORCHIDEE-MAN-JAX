"""Synthetic two-GPU persistent-cache regression for the mixed profiler.

This probe deliberately exercises the production ``profile_dual_device``
boundary with tiny arrays.  It does not read Teacher data and its result is
diagnostic rather than real-model acceptance evidence.
"""
from __future__ import annotations

import argparse
import gc
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from scripts.hpc.check_daily_training_runtime import _atomic_json
from scripts.hpc.probe_dual_device_runtime import parser as runtime_parser


class _WindowPool:
    """Build the exact five-field case tuple consumed by the profiler."""

    def case(self, jax: Any, rows: Any) -> tuple[Any, ...]:
        del jax
        days = np.asarray(rows, dtype=np.float32)
        horizon = int(days.shape[0])
        if days.shape != (horizon, 2):
            raise ValueError("synthetic recurrent rows must have shape (horizon, 2)")
        initial = np.asarray([0.2, -0.1], dtype=np.float32)
        discrete = np.asarray(0, dtype=np.int32)
        targets = np.zeros((horizon, 2), dtype=np.float32)
        expected_discrete = np.asarray(0, dtype=np.int32)
        return initial, discrete, days, targets, expected_discrete


class _PerDeviceLower:
    """Dispatch AOT lowering to device-distinct toy raw functions.

    The fixture intentionally avoids making the raw backward itself the cache
    regression: identical synthetic raw functions on both GPUs can collide in
    old JAX versions before the validator path is reached.
    """

    def __init__(self, jax: Any, functions: dict[int, Any]) -> None:
        self._jax = jax
        self._functions = functions

    def _function(self, parameters: Any) -> Any:
        leaves = self._jax.tree_util.tree_leaves(parameters)
        if not leaves or not hasattr(leaves[0], "devices"):
            raise ValueError("synthetic checked backward parameters are not device arrays")
        devices = tuple(leaves[0].devices())
        if len(devices) != 1:
            raise ValueError("synthetic checked backward parameters must be single-device")
        try:
            return self._functions[devices[0].id]
        except KeyError as error:
            raise ValueError(f"no synthetic raw backward for device {devices[0]}") from error

    def lower(self, parameters: Any, *args: Any) -> Any:
        return self._function(parameters).lower(parameters, *args)


def _fixture(jax: Any, jnp: Any, devices: list[Any]) -> dict[str, Any]:
    primary, secondary = devices
    rng = np.random.default_rng(20260916)
    parameters = {
        "scale": rng.normal(0.15, 0.02, (2,)).astype(np.float32),
        "bias": np.asarray([0.05, -0.03], dtype=np.float32),
    }
    coefficients = np.asarray(0.1, dtype=np.float32)

    def sample(rows: Any) -> Any:
        values = np.asarray(rows, dtype=np.float32)
        if values.shape != (8, 2):
            raise ValueError("synthetic daily sample must be one microbatch of shape (8, 2)")
        return jnp.asarray(values)

    def daily_objective(params: Any, batch: Any, coefficient: Any) -> tuple[Any, dict[str, Any]]:
        prediction = batch * params["scale"] + params["bias"]
        value = jnp.mean(jnp.square(prediction - coefficient))
        hard = {"nonfinite": jnp.zeros((batch.shape[0],), dtype=jnp.int32)}
        return value, {"hard": hard}

    daily_backward = jax.jit(jax.value_and_grad(daily_objective, has_aux=True))

    def recurrent_objective(
        params: Any,
        initial: Any,
        discrete: Any,
        days: Any,
        targets: Any,
        context: Any,
    ) -> tuple[Any, tuple[Any, Any]]:
        del discrete
        observations = initial + jnp.cumsum(days * params["scale"], axis=0) + params["bias"]
        value = jnp.mean(jnp.square(observations - targets)) + jnp.asarray(context) * 0.0
        science_counts = jnp.zeros((1,), dtype=jnp.int64)
        return value, (observations, science_counts)

    raw_functions: dict[int, Any] = {}
    for device in (primary, secondary):
        name = f"toy_checked_backward_gpu{device.id}"

        def named_objective(*args: Any, _objective=recurrent_objective, _name=name) -> Any:
            del _name
            return _objective(*args)

        named_objective.__name__ = name
        named_objective.__qualname__ = name
        raw_functions[device.id] = jax.jit(
            jax.value_and_grad(named_objective, has_aux=True)
        )

    # Keep this validator identical across devices.  The captured value is
    # primary-resident on purpose: it is the condition that exposed the old
    # unscoped helper/norm cache collision.  The production wrapper scopes its
    # validator compilation independently of the general helper hook.
    captured = jax.device_put(np.asarray([2.0, 3.0], dtype=np.float32), primary)
    initial_reference = jax.device_put(
        np.asarray([0.2, -0.1], dtype=np.float32), primary
    )

    def validator(
        initial: Any,
        observations: Any,
        targets: Any,
        expected_discrete: Any,
        templates: Any,
        counts: Any,
    ) -> Any:
        finite = (
            jnp.all(jnp.isfinite(initial))
            & jnp.all(jnp.isfinite(observations))
            & jnp.all(jnp.isfinite(targets))
        )
        # Use every argument in the valid-path check so placement and changed
        # initial/expected values cannot disappear as dead validator inputs.
        unchanged = jnp.all(expected_discrete == expected_discrete) & jnp.all(
            templates == templates
        )
        initial_matches = jnp.allclose(initial, initial_reference)
        expected_matches = jnp.all(expected_discrete == 0)
        captured_finite = jnp.all(jnp.isfinite(captured))
        count_shape = counts.shape[0] == 1
        invalid = ~(
            finite
            & unchanged
            & initial_matches
            & expected_matches
            & captured_finite
            & count_shape
        )
        return jnp.asarray([invalid], dtype=jnp.int64)

    recurrent_rows = {
        horizon: (np.arange(horizon * 2, dtype=np.float32).reshape(horizon, 2) + 1.0) * 0.01
        for horizon in (7, 30)
    }
    return {
        "parameters": parameters,
        "optimizer": None,
        "sample": sample,
        "daily_rows": np.arange(64 * 2, dtype=np.float32).reshape(64, 2) * 0.01,
        "recurrent_rows": recurrent_rows,
        "window_pool": _WindowPool(),
        "daily_backward": daily_backward,
        "checked_backward": _PerDeviceLower(jax, raw_functions),
        "checked_validator": jax.jit(validator),
        "checked_counter_names": ("nonfinite",),
        "coefficients": coefficients,
        "context": np.asarray(0.0, dtype=np.float32),
        "primary": primary,
    }


def _clear_in_memory_caches(jax: Any) -> None:
    clear = getattr(jax, "clear_caches", None)
    if clear is None:
        raise RuntimeError("this JAX runtime has no in-memory cache clearing API")
    clear()
    gc.collect()


def _profile(jax: Any, jnp: Any, fixture: dict[str, Any], output: Path, events: list[str]) -> dict[str, Any]:
    from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        initialize_target_oriented_adamw,
    )
    from scripts.hpc.mixed_learning_dual_device import profile_dual_device

    parameters = jax.device_put(fixture["parameters"], fixture["primary"])
    optimizer = initialize_target_oriented_adamw(parameters)
    update = make_update(
        {
            "optimizer": {
                "learning_rate": 1e-4,
                "clip_norm": 0.01,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
                "weight_decay": 0.0,
            }
        }
    )
    return profile_dual_device(
        jax=jax,
        jnp=jnp,
        parameters=parameters,
        optimizer=optimizer,
        sample=fixture["sample"],
        daily_rows=fixture["daily_rows"],
        recurrent_rows=fixture["recurrent_rows"],
        window_pool=fixture["window_pool"],
        daily_backward=fixture["daily_backward"],
        checked_backward=fixture["checked_backward"],
        checked_validator=fixture["checked_validator"],
        checked_counter_names=fixture["checked_counter_names"],
        coefficients=fixture["coefficients"],
        context=fixture["context"],
        update=update,
        repeats=3,
        emit=lambda stage, **info: events.append(stage),
        output=output,
        compiled_memory_limit_gib=1.0,
    )


def _diagnostic_profile(result: dict[str, Any]) -> dict[str, Any]:
    """Retain correctness evidence without claiming tiny-fixture speed."""

    return {
        "schema_version": result["schema_version"],
        "status": result["status"],
        "numerical_comparison": result["numerical_comparison"],
        "timing_parity": result["timing_parity"],
        "compilation": result["compilation"],
        "devices": result["devices"],
        "same_data_and_weights": result["same_data_and_weights"],
        "includes_updated_weight_transfer": result["includes_updated_weight_transfer"],
        "includes_host_reads_and_checks": result["includes_host_reads_and_checks"],
        "timing_claimed": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    began = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(
        tempfile.mkdtemp(prefix="dual-profile-cache-", dir=str(args.output.parent))
    )
    # This must happen before the first lowering/compile in the production
    # profiler.  Keep the cache on disk between runs; only JAX memory caches
    # are cleared when checking persistent-cache reuse.
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_enable_x64", True)
    devices = jax.local_devices(backend=args.backend)
    if len(devices) != 2 or any(device.platform != "gpu" for device in devices):
        raise ValueError("dual profile cache probe requires exactly two visible GPUs")

    fixture = _fixture(jax, jnp, devices)
    from scripts.hpc import mixed_learning_dual_device as profile_module

    events: list[str] = []
    legacy_error: str | None = None
    legacy_exception: Exception | None = None
    original_helper = getattr(profile_module, "device_helper_cache", None)
    if original_helper is None:
        raise RuntimeError("production checked_recurrent_execution lacks device_helper_cache")
    try:
        # The raw fixture is deliberately device-distinct, so this control
        # isolates the old unscoped norm/helper path.  The validator wrapper's
        # own device scope remains active and is restored below.
        profile_module.device_helper_cache = lambda device: nullcontext()
        try:
            _profile(jax, jnp, fixture, args.output, events)
        except Exception as error:
            legacy_exception = error
            legacy_error = f"{type(error).__name__}: {error}"
    finally:
        profile_module.device_helper_cache = original_helper

    _clear_in_memory_caches(jax)
    expected_fragments = ("Buffer passed to Execute()", "replica is assigned")
    legacy_reproduced = bool(
        legacy_error is not None
        and all(fragment in legacy_error for fragment in expected_fragments)
        and "dual_T7_gpu1_validator_compiled" in events
    )
    if legacy_exception is not None and not legacy_reproduced:
        report = {
            "schema_version": "dual_profile_cache_probe_v1",
            "status": "failed",
            "failure": "legacy control raised an unexpected exception",
            "legacy_control_error": legacy_error,
            "legacy_control_events": events,
            "cache_dir": str(cache_dir),
            "teacher_payload_reads": 0,
            "real_model_acceptance": False,
        }
        _atomic_json(args.output, report)
        raise RuntimeError(report["failure"]) from legacy_exception
    if args.require_reproduction and not legacy_reproduced:
        report = {
            "schema_version": "dual_profile_cache_probe_v1",
            "status": "failed",
            "failure": "legacy cache control did not reproduce an error",
            "legacy_control_error": legacy_error,
            "cache_dir": str(cache_dir),
            "teacher_payload_reads": 0,
            "real_model_acceptance": False,
        }
        _atomic_json(args.output, report)
        raise RuntimeError(report["failure"])

    runs: list[dict[str, Any]] = []
    try:
        for index in range(2):
            _clear_in_memory_caches(jax)
            runs.append(_diagnostic_profile(_profile(jax, jnp, fixture, args.output, events)))
    except Exception as error:
        report = {
            "schema_version": "dual_profile_cache_probe_v1",
            "status": "failed",
            "failure": f"{type(error).__name__}: {error}",
            "completed_profile_runs": len(runs),
            "legacy_control_error": legacy_error,
            "cache_dir": str(cache_dir),
            "teacher_payload_reads": 0,
            "real_model_acceptance": False,
        }
        _atomic_json(args.output, report)
        raise

    report = {
        "schema_version": "dual_profile_cache_probe_v1",
        "status": "passed",
        "jax_version": jax.__version__,
        "devices": [str(device) for device in devices],
        "cache_dir": str(cache_dir),
        "persistent_cache_reused": True,
        "in_memory_caches_cleared_between_profiles": True,
        "legacy_control_error": legacy_error,
        "legacy_control_reproduced": legacy_reproduced,
        "profile_runs": runs,
        "profile_runs_required": 2,
        "profile_repeats": 3,
        "timing_claimed": False,
        "toy_raw_function_names": [
            f"toy_checked_backward_gpu{device.id}" for device in devices
        ],
        "events": events,
        "elapsed_seconds": time.perf_counter() - began,
        "teacher_payload_reads": 0,
        "real_model_acceptance": False,
    }
    _atomic_json(args.output, report)
    print(report, flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = runtime_parser()
    result.description = __doc__
    result.add_argument(
        "--require-reproduction",
        action="store_true",
        help="fail unless the unscoped legacy control raises during profiling",
    )
    return result


if __name__ == "__main__":
    run(parser().parse_args())
