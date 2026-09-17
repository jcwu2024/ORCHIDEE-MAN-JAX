"""Bounded continuity regression for the production dual-device trainer.

This synthetic probe exercises the same callback, merge, clipping, AdamW,
peer-synchronization, and full-state checkpoint boundaries as the production
runner.  It is a runtime regression only: it reads no Teacher data and is not
real-model acceptance evidence.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Any

import numpy as np

from scripts.hpc.check_daily_training_runtime import _atomic_json, _tree_error
from scripts.hpc.mixed_learning_dual_device import ContinuousDualDeviceTrainer
from scripts.hpc.probe_dual_device_runtime import parser as runtime_parser

METHOD = {
    "schema_version": "tiny_dual_continuity_method_v1",
    "updates": 3,
    "daily_batch_size": 64,
    "daily_microbatch_size": 8,
}
SOURCES = {"daily": {"identity": "tiny-daily-v1"}, "recurrent": {"identity": "tiny-recurrent-v1"}}
SCHEDULE = {"schema_version": "tiny-schedule-v1", "sha256": "tiny-schedule"}
FRESH = {"seed": 20260918, "parameter_count": 4}


def _tree_copy(jax: Any, value: Any) -> Any:
    return jax.tree_util.tree_map(lambda leaf: np.array(leaf, copy=True), jax.device_get(value))


def _tree_max_error(jax: Any, actual: Any, expected: Any) -> float:
    return _tree_error(jax, actual, expected, rtol=2e-5, atol=2e-6)


def _tree_digest(jax: Any, value: Any) -> tuple[Any, ...]:
    return tuple(
        np.asarray(leaf).copy()
        for leaf in jax.tree_util.tree_leaves(jax.device_get(value))
    )


def _digest_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return len(left) == len(right) and all(np.array_equal(a, b) for a, b in zip(left, right, strict=True))


def _device_of(jax: Any, parameters: Any) -> Any:
    leaves = jax.tree_util.tree_leaves(parameters)
    devices = tuple(leaves[0].devices())
    if len(devices) != 1:
        raise ValueError("tiny callback parameters must be single-device arrays")
    return devices[0]


def _fixture(jax: Any, jnp: Any, devices: list[Any]) -> dict[str, Any]:
    primary, secondary = devices
    rng = np.random.default_rng(20260918)
    initial = {
        "scale": rng.normal(0.15, 0.02, (2,)).astype(np.float32),
        "bias": np.asarray([0.05, -0.03], dtype=np.float32),
    }
    coefficient = np.asarray(0.1, dtype=np.float32)

    def daily_objective(parameters: Any, rows: Any, target: Any) -> tuple[Any, Any]:
        prediction = rows * parameters["scale"] + parameters["bias"]
        hard = jnp.sum(~jnp.isfinite(prediction) | ~jnp.isfinite(target))
        loss = jnp.mean(jnp.square(prediction - target))
        return loss, hard

    def recurrent_objective(parameters: Any, days: Any, target: Any) -> tuple[Any, Any]:
        observations = jnp.asarray([0.2, -0.1], dtype=days.dtype)
        observations = observations + jnp.cumsum(days * parameters["scale"], axis=0) + parameters["bias"]
        hard = jnp.sum(~jnp.isfinite(observations) | ~jnp.isfinite(target))
        loss = jnp.mean(jnp.square(observations - target))
        return loss, hard

    daily_executables: dict[int, Any] = {}
    recurrent_executables: dict[int, dict[int, Any]] = {}
    daily_batch = jnp.asarray(np.zeros((8, 2), dtype=np.float32))
    recurrent_batch = {
        horizon: jnp.asarray(np.arange(horizon * 2, dtype=np.float32).reshape(horizon, 2) * 0.01)
        for horizon in (7, 30)
    }
    target_batch = jnp.asarray(np.full((8, 2), 0.1, dtype=np.float32))
    target_recurrent = {
        horizon: jnp.asarray(np.full((horizon, 2), 0.2, dtype=np.float32))
        for horizon in (7, 30)
    }
    # Raw model AOT compilation is intentionally outside device_helper_cache.
    for device in devices:
        daily_name = f"tiny_daily_backward_gpu{device.id}"
        recurrent_name = f"tiny_recurrent_backward_gpu{device.id}"

        def named_daily(parameters: Any, rows: Any, target: Any, *, _name=daily_name) -> tuple[Any, Any]:
            del _name
            return daily_objective(parameters, rows, target)

        def named_recurrent(parameters: Any, days: Any, target: Any, *, _name=recurrent_name) -> tuple[Any, Any]:
            del _name
            return recurrent_objective(parameters, days, target)

        named_daily.__name__ = daily_name
        named_daily.__qualname__ = daily_name
        named_recurrent.__name__ = recurrent_name
        named_recurrent.__qualname__ = recurrent_name
        daily_raw = jax.jit(jax.value_and_grad(named_daily, has_aux=True))
        recurrent_raw = jax.jit(jax.value_and_grad(named_recurrent, has_aux=True))
        with jax.default_device(device):
            parameters = jax.device_put(initial, device)
            daily_executables[device.id] = daily_raw.lower(
                parameters, daily_batch, target_batch
            ).compile()
            recurrent_executables[device.id] = {
                horizon: recurrent_raw.lower(
                    parameters, recurrent_batch[horizon], target_recurrent[horizon]
                ).compile()
                for horizon in (7, 30)
            }

    def daily_branch(parameters: Any, payload: dict[str, Any]) -> dict[str, Any]:
        device = _device_of(jax, parameters)
        started = time.perf_counter()
        executable = daily_executables[device.id]
        total = jax.tree_util.tree_map(jnp.zeros_like, parameters)
        loss = 0.0
        hard_total = 0
        rows = np.asarray(payload["rows"], dtype=np.float32)
        target = np.asarray(payload["target"], dtype=np.float32)
        if rows.shape != (64, 2) or target.shape != (64, 2):
            raise ValueError("tiny daily payload must be 64 examples of width 2")
        for start in range(0, 64, 8):
            batch = jax.device_put(rows[start : start + 8], device)
            targets = jax.device_put(target[start : start + 8], device)
            (value, hard), gradient = executable(parameters, batch, targets)
            value, hard = jax.device_get((value, hard))
            if not np.isfinite(value) or int(hard) != 0:
                raise ValueError("tiny daily physical/finite gate failed")
            total = jax.tree_util.tree_map(lambda old, new: old + new, total, gradient)
            loss += float(value) / 8.0
            hard_total += int(hard)
        gradient = jax.tree_util.tree_map(lambda value: value / 64.0, total)
        norm = float(np.sqrt(sum(np.sum(np.square(np.asarray(leaf, dtype=np.float64))) for leaf in jax.device_get(jax.tree_util.tree_leaves(gradient)))))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("tiny daily gradient norm is invalid")
        return {
            "gradient": gradient,
            "loss": loss,
            "norm": norm,
            "hard": {"physical": hard_total},
            "seconds": time.perf_counter() - started,
        }

    peer_observations: list[np.ndarray] = []

    def recurrent_branch(parameters: Any, payload: dict[int, Any]) -> tuple[dict[int, dict[str, Any]], float]:
        device = _device_of(jax, parameters)
        started = time.perf_counter()
        peer_observations.append(np.asarray(jax.device_get(parameters["scale"]), dtype=np.float32).copy())
        results: dict[int, dict[str, Any]] = {}
        for horizon in (7, 30):
            rows = np.asarray(payload[horizon]["rows"], dtype=np.float32)
            target = np.asarray(payload[horizon]["target"], dtype=np.float32)
            if rows.shape != (horizon, 2) or target.shape != (horizon, 2):
                raise ValueError("tiny recurrent payload horizon shape drift")
            days = jax.device_put(rows, device)
            targets = jax.device_put(target, device)
            (value, hard), gradient = recurrent_executables[device.id][horizon](parameters, days, targets)
            value, hard = jax.device_get((value, hard))
            if not np.isfinite(value) or int(hard) != 0:
                raise ValueError(f"tiny T{horizon} physical/finite gate failed")
            norm = float(np.sqrt(sum(np.sum(np.square(np.asarray(leaf, dtype=np.float64))) for leaf in jax.device_get(jax.tree_util.tree_leaves(gradient)))))
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError(f"tiny T{horizon} gradient norm is invalid")
            results[horizon] = {
                "gradient": gradient,
                "loss": float(value),
                "norm": norm,
                "hard": {"physical": int(hard)},
                "seconds": time.perf_counter() - started,
            }
        return results, time.perf_counter() - started

    return {
        "parameters": initial,
        "primary": primary,
        "secondary": secondary,
        "daily_branch": daily_branch,
        "recurrent_branch": recurrent_branch,
        "peer_observations": peer_observations,
        "daily_payload": {
            "rows": np.arange(128, dtype=np.float32).reshape(64, 2) * 0.01,
            "target": np.full((64, 2), 0.1, dtype=np.float32),
        },
        "recurrent_payload": {
            horizon: {
                "rows": np.arange(horizon * 2, dtype=np.float32).reshape(horizon, 2) * 0.01,
                "target": np.full((horizon, 2), 0.2, dtype=np.float32),
            }
            for horizon in (7, 30)
        },
        "coefficient": coefficient,
    }


def _make_optimizer(jax: Any, parameters: Any) -> Any:
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        initialize_target_oriented_adamw,
    )

    return initialize_target_oriented_adamw(parameters)


def _make_runtime(jax: Any, jnp: Any, fixture: dict[str, Any], parameters: Any, optimizer: Any):
    from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
    from research.daily_coarse_graining.daily_operator.mixed_objective import make_mixed_gradient

    protocol = {
        "optimizer": {
            "learning_rate": 1.0e-3,
            "clip_norm": 0.01,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1.0e-8,
            "weight_decay": 0.0,
        }
    }
    finite = jax.jit(lambda tree: jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in jax.tree_util.tree_leaves(tree)])))
    return ContinuousDualDeviceTrainer(
        jax=jax,
        jnp=jnp,
        parameters=parameters,
        optimizer=optimizer,
        daily_branch=fixture["daily_branch"],
        recurrent_branch=fixture["recurrent_branch"],
        combine=make_mixed_gradient(jax, jnp),
        update=make_update(protocol),
        finite=finite,
        devices=[fixture["primary"], fixture["secondary"]],
        require_gpus=False,
    )


def _payload_for_step(fixture: dict[str, Any], step: int) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Return changing deterministic windows, keeping the shape contract fixed."""

    offset = np.float32(step * 0.003)
    daily = fixture["daily_payload"]
    daily_payload = {
        "rows": np.asarray(daily["rows"], dtype=np.float32) + offset,
        "target": np.asarray(daily["target"], dtype=np.float32) + offset / 2,
    }
    recurrent_payload = {}
    for horizon, payload in fixture["recurrent_payload"].items():
        recurrent_payload[horizon] = {
            "rows": np.asarray(payload["rows"], dtype=np.float32) + offset,
            "target": np.asarray(payload["target"], dtype=np.float32) + offset / 3,
        }
    return daily_payload, recurrent_payload


def _serial_step(jax: Any, jnp: Any, fixture: dict[str, Any], parameters: Any, optimizer: Any, *, daily_payload: Any, recurrent_payload: Any | None, weights: dict[str, float], update_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
    from research.daily_coarse_graining.daily_operator.mixed_objective import make_mixed_gradient

    daily = fixture["daily_branch"](parameters, daily_payload)
    recurrent = {} if recurrent_payload is None else fixture["recurrent_branch"](parameters, recurrent_payload)[0]
    with jax.default_device(fixture["primary"]):
        mixed = daily["gradient"]
        combine = make_mixed_gradient(jax, jnp)
        for horizon in (7, 30):
            if horizon in recurrent:
                mixed = combine(mixed, recurrent[horizon]["gradient"], 1.0, weights[f"T{horizon}"])
        update = make_update({"optimizer": {"learning_rate": 1.0e-3, "clip_norm": 0.01, "beta1": 0.9, "beta2": 0.999, "epsilon": 1.0e-8, "weight_decay": 0.0}})
        proposal, next_optimizer, _, _ = update(parameters, mixed, optimizer)
        jax.block_until_ready((proposal, next_optimizer))
    return {"gradient": mixed, "parameters": proposal, "optimizer": next_optimizer}, {"optimizer_step": update_index}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.mixed_training_state import (
        load_mixed_training_checkpoint,
        save_mixed_training_checkpoint,
    )

    began = time.perf_counter()
    jax.config.update("jax_enable_x64", True)
    cache_dir = args.output.parent / f".dual-training-continuity-cache-{os.getpid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    devices = jax.local_devices(backend=args.backend)
    if len(devices) != 2:
        raise ValueError("continuity probe requires exactly two visible devices")
    fixture = _fixture(jax, jnp, devices)
    weights = {"T7": 0.5, "T30": 0.5}
    initial_parameters = jax.device_put(fixture["parameters"], fixture["primary"])
    initial_optimizer = _make_optimizer(jax, initial_parameters)

    # Compare every full proposal with a sequential execution using identical callbacks.
    serial_parameters, serial_optimizer = initial_parameters, initial_optimizer
    dual = _make_runtime(jax, jnp, fixture, initial_parameters, initial_optimizer)
    step_comparisons = []
    checkpoint_path = args.output.with_suffix(".checkpoint.pkl")
    try:
        for index in range(1, 4):
            daily_payload, recurrent_payload = _payload_for_step(fixture, index)
            serial_values, _ = _serial_step(
                jax, jnp, fixture, serial_parameters, serial_optimizer,
                daily_payload=daily_payload, recurrent_payload=recurrent_payload,
                weights=weights, update_index=index,
            )
            serial_parameters, serial_optimizer = serial_values["parameters"], serial_values["optimizer"]
            record, values = dual.step(
                daily_payload=daily_payload, recurrent_payload=recurrent_payload,
                weights=weights, expected_optimizer_step=index,
            )
            step_comparisons.append({
                "step": index,
                "gradient_max_abs_error": _tree_max_error(jax, values["gradient"], serial_values["gradient"]),
                "parameter_max_abs_error": _tree_max_error(jax, values["parameters"], serial_values["parameters"]),
                "optimizer_max_abs_error": _tree_max_error(jax, values["optimizer"], serial_values["optimizer"]),
                "clip": record["clip"],
            })
            if index == 2:
                save_mixed_training_checkpoint(
                    checkpoint_path,
                    jax=jax,
                    parameters=values["parameters"],
                    optimizer_state=values["optimizer"],
                    completed_updates=2,
                    daily_sampler_cursor={"epoch": 0, "batch_offset": 2},
                    recurrent_schedule_cursor=2,
                    method=METHOD,
                    source_identities=SOURCES,
                    schedule_identity=SCHEDULE,
                    fresh_initialization_identity=FRESH,
                )
        uninterrupted_final = {"parameters": dual.parameters, "optimizer": dual.optimizer}
    finally:
        dual.close()

    restored = load_mixed_training_checkpoint(
        checkpoint_path,
        expected_method=METHOD,
        expected_source_identities=SOURCES,
        expected_schedule_identity=SCHEDULE,
        expected_fresh_initialization_identity=FRESH,
    )
    resumed = _make_runtime(jax, jnp, fixture, restored["parameters"], restored["optimizer_state"])
    try:
        resumed_record, resumed_values = resumed.step(
            daily_payload=_payload_for_step(fixture, 3)[0],
            recurrent_payload=_payload_for_step(fixture, 3)[1],
            weights=weights, expected_optimizer_step=3,
        )
        resume_error = {
            "parameter_max_abs_error": _tree_max_error(jax, resumed_values["parameters"], uninterrupted_final["parameters"]),
            "optimizer_max_abs_error": _tree_max_error(jax, resumed_values["optimizer"], uninterrupted_final["optimizer"]),
            "optimizer_step": resumed_record["optimizer_step"],
        }
    finally:
        resumed.close()

    # Daily-only persistence must update the peer before a subsequent mixed step.
    phase = _make_runtime(jax, jnp, fixture, jax.device_put(fixture["parameters"], fixture["primary"]), _make_optimizer(jax, jax.device_put(fixture["parameters"], fixture["primary"])))
    try:
        phase_daily_1, _ = _payload_for_step(fixture, 1)
        phase_daily_2, phase_recurrent_2 = _payload_for_step(fixture, 2)
        phase.step(daily_payload=phase_daily_1, expected_optimizer_step=1)
        daily_peer_error = _tree_max_error(jax, phase.parameters, phase.peer_parameters)
        before_mixed = np.asarray(jax.device_get(phase.parameters["scale"]), dtype=np.float32).copy()
        fixture["peer_observations"].clear()
        phase.step(daily_payload=phase_daily_2, recurrent_payload=phase_recurrent_2, weights=weights, expected_optimizer_step=2)
        callback_peer_error = float(np.max(np.abs(fixture["peer_observations"][0] - before_mixed)))
    finally:
        phase.close()

    # A physical failure must reject the update without changing either state tree.
    failure = _make_runtime(jax, jnp, fixture, jax.device_put(fixture["parameters"], fixture["primary"]), _make_optimizer(jax, jax.device_put(fixture["parameters"], fixture["primary"])))
    before_parameters = _tree_digest(jax, failure.parameters)
    before_optimizer = _tree_digest(jax, failure.optimizer)
    invalid_payload = {key: np.array(value, copy=True) for key, value in _payload_for_step(fixture, 1)[0].items()}
    invalid_payload["target"][0, 0] = np.nan
    rejected = False
    rejection_message = ""
    try:
        try:
            failure.step(daily_payload=invalid_payload, expected_optimizer_step=1)
        except ValueError as error:
            rejected, rejection_message = True, str(error)
        if not rejected or not _digest_equal(before_parameters, _tree_digest(jax, failure.parameters)) or not _digest_equal(before_optimizer, _tree_digest(jax, failure.optimizer)):
            raise RuntimeError("failed physical proposal changed trainer state")
    finally:
        failure.close()
    if not all(0 < row["clip"] < 1 for row in step_comparisons):
        raise RuntimeError("tiny continuity probe did not exercise active clipping")
    if daily_peer_error > 2e-5 or callback_peer_error > 2e-5:
        raise RuntimeError("tiny continuity probe failed peer parameter synchronization")

    report = {
        "schema_version": "dual_training_continuity_probe_v1",
        "status": "passed",
        "jax_version": jax.__version__,
        "devices": [str(device) for device in devices],
        "cache_dir": str(cache_dir),
        "updates": 3,
        "daily_batch_size": 64,
        "daily_microbatch_size": 8,
        "horizons": [7, 30],
        "serial_dual_comparison": step_comparisons,
        "checkpoint": {"saved_after_update": 2, "resumed_update": 3, **resume_error},
        "daily_only_peer_sync_max_abs_error": daily_peer_error,
        "mixed_callback_peer_sync_max_abs_error": callback_peer_error,
        "rejected_failure": {"state_unchanged": True, "message": rejection_message},
        "active_clipping": all(0 < row["clip"] < 1 for row in step_comparisons),
        "teacher_payload_reads": 0,
        "real_model_acceptance": False,
        "elapsed_seconds": time.perf_counter() - began,
    }
    _atomic_json(args.output, report)
    print(report, flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = runtime_parser()
    result.description = __doc__
    return result


if __name__ == "__main__":
    run(parser().parse_args())
