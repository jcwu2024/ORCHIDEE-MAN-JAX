"""Tiny runtime gate for public daily AD, accumulation, and AdamW plumbing.

Only synthetic arrays are used. Passing this gate is not real-model or
scientific acceptance.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "daily_training_runtime_check_v1"
HORIZONS = (1, 2, 3, 7)
DTYPES = ("float32", "float64")


def _atomic_json(path: Path, payload: dict) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _tree_error(jax, actual, expected, *, rtol: float, atol: float) -> float:
    left, left_tree = jax.tree_util.tree_flatten(actual)
    right, right_tree = jax.tree_util.tree_flatten(expected)
    if left_tree != right_tree or len(left) != len(right):
        raise RuntimeError("runtime result tree structures differ")
    maximum = 0.0
    for observed, reference in zip(left, right, strict=True):
        observed = np.asarray(observed, dtype=np.float64)
        reference = np.asarray(reference, dtype=np.float64)
        if observed.shape != reference.shape:
            raise RuntimeError("runtime result leaf shapes differ")
        if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(reference)):
            raise RuntimeError("runtime result contains a nonfinite value")
        maximum = max(
            maximum,
            float(np.max(np.abs(observed - reference), initial=0.0)),
        )
        if not np.allclose(observed, reference, rtol=rtol, atol=atol):
            raise RuntimeError("runtime numerical comparison failed")
    return maximum


def _all_finite(jax, tree) -> bool:
    return all(
        bool(np.all(np.isfinite(np.asarray(value))))
        for value in jax.tree_util.tree_leaves(tree)
    )


def _step(jax, jnp):
    def step(parameters, state, discrete, day):
        dtype = state.dtype
        gate = jnp.where(
            discrete["gate"], jnp.asarray(0.025, dtype), jnp.asarray(-0.015, dtype)
        )
        fraction = jax.nn.sigmoid(
            parameters["fraction_gain"] * state
            + day["fraction_bias"]
            + jnp.asarray(0.01, dtype) * discrete["phase"].astype(dtype)
        )
        following = jnp.tanh(
            parameters["matrix"] @ state
            + parameters["bias"]
            + day["forcing"]
            + jnp.asarray(0.12, dtype) * (fraction - jnp.asarray(0.5, dtype))
            + gate
        )
        observations = {
            "valid": day["valid"] & jnp.isfinite(following),
            "fraction": fraction,
            "flux": fraction
            * (following + parameters["output_scale"] * day["forcing"]),
            # Ignored by the loss: their cotangents are scalar float0 arrays.
            "scalar_gate": discrete["gate"],
            "scalar_phase": discrete["phase"],
        }
        next_discrete = {
            "gate": jnp.logical_xor(discrete["gate"], day["toggle"]),
            "phase": discrete["phase"] + day["phase_increment"],
        }
        return following, next_discrete, observations

    return step


def _loss(jnp):
    def loss(parameters, initial, observations, targets, context):
        valid = observations["valid"]
        count = jnp.maximum(jnp.sum(valid), 1)
        flux = jnp.where(valid, observations["flux"] - targets["flux"], 0.0)
        fraction = jnp.where(
            valid, observations["fraction"] - targets["fraction"], 0.0
        )
        state = jnp.where(
            valid, observations["state"] - targets["state"], 0.0
        )
        return (
            context[0] * jnp.sum(flux * flux) / count
            + context[1] * jnp.sum(fraction * fraction) / count
            + context[2] * jnp.sum(state * state) / count
            + context[3]
            * jnp.sum((observations["state"][-1] - targets["terminal"]) ** 2)
            + context[4] * jnp.sum(parameters["regularizer"] ** 2)
            + context[5] * jnp.sum((initial - targets["initial"]) ** 2)
        )

    return loss


def _case(jnp, dtype_name: str, horizon: int):
    dtype = jnp.dtype(dtype_name)
    index = jnp.arange(horizon, dtype=dtype)
    integer = jnp.arange(horizon)
    parameters = {
        "bias": jnp.asarray([0.02, -0.04], dtype),
        "fraction_gain": jnp.asarray([0.45, -0.30], dtype),
        "matrix": jnp.asarray([[0.30, -0.12], [0.08, 0.27]], dtype),
        "output_scale": jnp.asarray(0.15, dtype),
        "regularizer": jnp.asarray([0.11, -0.07], dtype),
    }
    initial = jnp.asarray([0.10, -0.20], dtype)
    discrete = {
        "gate": jnp.asarray(False),
        "phase": jnp.asarray(1, dtype=jnp.int32),
    }
    days = {
        "valid": jnp.stack((integer % 4 != 2, integer % 5 != 3), axis=1),
        "forcing": jnp.stack(
            (0.12 + 0.025 * index, -0.08 + 0.017 * index), axis=1
        ),
        "fraction_bias": jnp.stack(
            (-0.10 + 0.013 * index, 0.06 - 0.009 * index), axis=1
        ),
        "phase_increment": (1 + integer % 2).astype(jnp.int32),
        "toggle": integer % 3 == 0,
    }
    targets = {
        "flux": jnp.stack(
            (0.03 + 0.01 * index, -0.02 + 0.004 * index), axis=1
        ),
        "fraction": jnp.stack(
            (0.48 + 0.002 * index, 0.52 - 0.003 * index), axis=1
        ),
        "state": jnp.stack(
            (0.04 + 0.006 * index, -0.05 + 0.003 * index), axis=1
        ),
        "initial": jnp.asarray([-0.03, 0.04], dtype),
        "terminal": jnp.asarray([0.08, -0.06], dtype),
    }
    context = jnp.asarray([0.70, 0.15, 0.08, 0.35, 0.12, 0.09], dtype)
    return parameters, initial, discrete, days, targets, context


def _changed_values(jnp, arguments):
    parameters, initial, discrete, days, targets, context = arguments
    dtype = initial.dtype
    targets = dict(
        targets,
        flux=targets["flux"] + jnp.asarray([0.025, -0.015], dtype),
        fraction=targets["fraction"] + jnp.asarray([0.01, -0.008], dtype),
        terminal=targets["terminal"] + jnp.asarray([-0.02, 0.015], dtype),
    )
    context = context.at[0].multiply(1.2).at[2].multiply(0.8)
    return parameters, initial, discrete, days, targets, context


def _float0_check(jax, component, arguments) -> bool:
    parameters, initial, discrete, days, targets, context = arguments
    _, observations = component.rollout(parameters, initial, discrete, days)
    _, (_, _, cotangents) = component._loss_partials(
        parameters, initial, observations, targets, context
    )
    daily = component._slice(cotangents, 0)
    float0 = np.dtype(jax.dtypes.float0)
    return all(
        isinstance(daily[name], np.ndarray)
        and daily[name].shape == ()
        and np.dtype(daily[name].dtype) == float0
        for name in ("scalar_gate", "scalar_phase")
    )


def _rollout_errors(jax, actual, expected, *, rtol, atol):
    return (
        _tree_error(jax, actual[0], expected[0], rtol=rtol, atol=atol),
        _tree_error(jax, actual[1][0], expected[1][0], rtol=rtol, atol=atol),
        _tree_error(jax, actual[1][1], expected[1][1], rtol=rtol, atol=atol),
    )


def _fraction_check(jnp, component, arguments, *, dtype_name, rtol, atol):
    parameters, initial, discrete, days, _, _ = arguments
    _, baseline = component.rollout(parameters, initial, discrete, days)
    forcing = days["forcing"].at[0].add(
        jnp.asarray([0.30, -0.20], dtype=initial.dtype)
    )
    _, changed = component.rollout(parameters, initial, discrete, dict(days, forcing=forcing))
    baseline = np.asarray(baseline["fraction"])
    changed = np.asarray(changed["fraction"])
    checks = {
        "nonconstant": not np.allclose(
            baseline[1:], baseline[:-1], rtol=rtol, atol=atol
        ),
        "day_zero_forcing_independent": np.allclose(
            changed[0], baseline[0], rtol=rtol, atol=atol
        ),
        "day_one_state_coupled": not np.allclose(
            changed[1], baseline[1], rtol=rtol, atol=atol
        ),
    }
    if not all(checks.values()):
        raise RuntimeError("synthetic PFT fraction is not state-coupled")
    return {"dtype": dtype_name, **{name: bool(value) for name, value in checks.items()}, "passed": True}


def _rollout_runtime_check(jax, jnp) -> dict:
    from research.daily_coarse_graining.daily_operator.rollout_loss import (
        make_rollout_loss,
    )

    comparisons, float0_rows, fraction_rows, reuse_rows = [], [], [], []
    for dtype_name in DTYPES:
        component = make_rollout_loss(jax, jnp, _step(jax, jnp), _loss(jnp))
        direct = jax.jit(jax.value_and_grad(component.loss, argnums=(0, 1)))
        rtol, atol = (
            (3.0e-10, 3.0e-11)
            if dtype_name == "float64"
            else (5.0e-5, 5.0e-6)
        )
        for horizon in HORIZONS:
            arguments = _case(jnp, dtype_name, horizon)
            if str(arguments[1].dtype) != dtype_name:
                raise RuntimeError(f"JAX did not enable requested {dtype_name}")
            expected = direct(*arguments)
            actual = component.streamed_value_and_grad(*arguments)
            value_error, parameter_error, initial_error = _rollout_errors(
                jax, actual, expected, rtol=rtol, atol=atol
            )
            compile_count = int(component.reverse_compile_count)
            if compile_count != 1:
                raise RuntimeError("one-day reverse executable was not reused")
            comparisons.append(
                {
                    "dtype": dtype_name,
                    "horizon": horizon,
                    "value_max_abs_error": value_error,
                    "parameter_gradient_max_abs_error": parameter_error,
                    "initial_gradient_max_abs_error": initial_error,
                    "reverse_compile_count": compile_count,
                    "passed": True,
                }
            )
            if horizon == 1:
                if not _float0_check(jax, component, arguments):
                    raise RuntimeError("scalar float0 cotangents lost array form")
                float0_rows.append({"dtype": dtype_name, "horizon": 1, "passed": True})
            if horizon == 7:
                horizon_seven = arguments

        arguments = horizon_seven
        fraction_rows.append(
            _fraction_check(
                jnp,
                component,
                arguments,
                dtype_name=dtype_name,
                rtol=rtol,
                atol=atol,
            )
        )

        direct_executable = direct.lower(*arguments).compile()
        baseline_direct = direct_executable(*arguments)
        changed_arguments = _changed_values(jnp, arguments)
        changed_direct = direct_executable(*changed_arguments)
        before = int(component.reverse_compile_count)
        changed_streamed = component.streamed_value_and_grad(*changed_arguments)
        after = int(component.reverse_compile_count)
        value_error, parameter_error, initial_error = _rollout_errors(
            jax, changed_streamed, changed_direct, rtol=rtol, atol=atol
        )
        loss_change = abs(float(changed_direct[0]) - float(baseline_direct[0]))
        if before != 1 or after != 1 or loss_change <= atol:
            raise RuntimeError("same-shape target/context cache reuse failed")
        reuse_rows.append(
            {
                "dtype": dtype_name,
                "horizon": 7,
                "loss_change": loss_change,
                "changed_value_max_abs_error": value_error,
                "changed_parameter_gradient_max_abs_error": parameter_error,
                "changed_initial_gradient_max_abs_error": initial_error,
                "reverse_compile_count_before": before,
                "reverse_compile_count_after": after,
                "passed": True,
            }
        )

    return {
        "horizons": list(HORIZONS),
        "dtypes": list(DTYPES),
        "direct_streamed_cases": comparisons,
        "scalar_float0_aot": float0_rows,
        "fraction_state_coupling": fraction_rows,
        "same_shape_target_context_reuse": reuse_rows,
    }


def _microbatch_optimizer_check(jax, jnp) -> dict:
    from research.daily_coarse_graining.daily_operator.fresh_start_training import (
        make_update,
    )
    from research.daily_coarse_graining.daily_operator.one_day_training_state import (
        make_gradient_accumulator,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        initialize_target_oriented_adamw,
    )

    parameters = {
        "weight": jnp.asarray([0.4, -0.3], dtype=jnp.float32),
        "bias": jnp.asarray(0.2, dtype=jnp.float32),
    }
    features = jnp.asarray(
        [[1.0, 2.0], [3.0, -1.0], [0.5, 4.0], [-2.0, 1.0], [5.0, 3.0]],
        dtype=jnp.float32,
    )
    targets = jnp.asarray([2.0, -1.0, 3.0, 0.5, -4.0], dtype=jnp.float32)

    def mean_loss(values, x, y):
        prediction = x @ values["weight"] + values["bias"]
        return jnp.mean((prediction - y) ** 2)

    gradient = jax.jit(jax.grad(mean_loss))
    direct_gradient = gradient(parameters, features, targets)
    counts = (2, 3)
    microgradients = (
        gradient(parameters, features[:2], targets[:2]),
        gradient(parameters, features[2:], targets[2:]),
    )
    add, finish = make_gradient_accumulator(jax, jnp)
    total = jax.tree_util.tree_map(jnp.zeros_like, parameters)
    for mean_microgradient, count in zip(microgradients, counts, strict=True):
        total = add(total, mean_microgradient, count)
    total_count = sum(counts)
    accumulated_gradient = finish(total, total_count)
    gradient_error = _tree_error(
        jax,
        accumulated_gradient,
        direct_gradient,
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    protocol = {
        "optimizer": {
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1.0e-8,
            "weight_decay": 1.0e-3,
            "learning_rate": 1.0e-2,
            "clip_norm": 0.25,
        }
    }
    optimizer = initialize_target_oriented_adamw(parameters)
    update = make_update(protocol)
    direct_result = update(parameters, direct_gradient, optimizer)
    accumulated_result = update(parameters, accumulated_gradient, optimizer)
    update_error = _tree_error(
        jax,
        accumulated_result,
        direct_result,
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    updated, state, norm, scale = accumulated_result
    finite = _all_finite(jax, (updated, state, norm, scale))
    changed = any(
        not np.array_equal(np.asarray(before), np.asarray(after))
        for before, after in zip(
            jax.tree_util.tree_leaves(parameters),
            jax.tree_util.tree_leaves(updated),
            strict=True,
        )
    )
    step = int(np.asarray(state.step))
    passed = finite and changed and step == 1 and total_count == 5
    if not passed:
        raise RuntimeError("weighted microbatch AdamW runtime check failed")
    return {
        "microbatch_counts": list(counts),
        "direct_batch_count": 5,
        "gradient_max_abs_error": gradient_error,
        "update_max_abs_error": update_error,
        "gradient_norm": float(norm),
        "clip_scale": float(scale),
        "optimizer_step": step,
        "parameters_and_optimizer_finite": bool(finite),
        "parameters_changed": bool(changed),
        "passed": True,
    }


def _base_report(status: str, passed: bool) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "passed": passed,
        "scope": "synthetic_runtime_interfaces_only",
        "real_model_checked": False,
        "teacher_data_read": False,
        "scientific_acceptance": False,
        "real_training_updates": 0,
    }


def check_runtime(jax, jnp) -> dict:
    """Exercise small synthetic versions of the public training interfaces."""
    jax.config.update("jax_enable_x64", True)
    return {
        **_base_report("passed", True),
        "jax_version": str(jax.__version__),
        "numpy_version": str(np.__version__),
        "rollout": _rollout_runtime_check(jax, jnp),
        "microbatch_adamw": _microbatch_optimizer_check(jax, jnp),
    }


def _initialize_runtime(backend: str):
    if backend == "gpu":
        from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend

        jax, devices = initialize_gpu_backend()
    elif backend == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        import jax

        devices = tuple(jax.devices("cpu"))
    else:
        raise ValueError(f"unsupported backend: {backend}")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    if not devices:
        raise RuntimeError(f"{backend} backend has no devices")
    return jax, jnp, devices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--output", type=Path, required=True)
    arguments, _ = parser.parse_known_args(argv)
    started = time.perf_counter()
    paid_observation = None
    expected_uuid = os.environ.get('ORCJAX_AFFINE_EXPECTED_GPU_UUID')
    if expected_uuid is not None:
        paid_observation = dict(requested_backend=arguments.backend, expected_gpu_uuid=expected_uuid,
            environment={key: os.environ.get(key) for key in
                ('ORCJAX_AFFINE_EXPECTED_GPU_UUID', 'CUDA_VISIBLE_DEVICES', 'JAX_PLATFORMS',
                 'JAX_ENABLE_X64', 'XLA_PYTHON_CLIENT_PREALLOCATE', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
    try:
        jax, jnp, devices = _initialize_runtime(arguments.backend)
        binding = None
        if expected_uuid is not None:
            from scripts.hpc.shared_affine_cuda_identity import verify_expected_gpu
            if arguments.backend != 'gpu':
                raise ValueError('paid physical GPU binding requires GPU backend')
            paid_observation.update(jax_version=str(jax.__version__),
                actual_backend=str(jax.default_backend()),
                devices=[dict(device=str(device), platform=getattr(device, 'platform', None),
                    local_hardware_id=getattr(device, 'local_hardware_id', None)) for device in devices],
                affine_gpu_identity={})
            binding = verify_expected_gpu(devices, expected_uuid,
                observation=paid_observation['affine_gpu_identity'])
        report = check_runtime(jax, jnp)
        if binding is not None:
            report['affine_gpu_identity'] = binding
        report.update(
            requested_backend=arguments.backend,
            actual_backend=str(jax.default_backend()),
            devices=[str(device) for device in devices],
            elapsed_seconds=time.perf_counter() - started,
        )
        result = 0
    except Exception as error:  # Write a machine-readable failure before exiting.
        cause = getattr(error, 'cause', error)
        report = {
            **_base_report("failed", False),
            "requested_backend": arguments.backend,
            "error_type": type(cause).__name__,
            "error": str(cause),
            "elapsed_seconds": time.perf_counter() - started,
        }
        result = 1
    if paid_observation is not None:
        report['paid_startup_observation'] = paid_observation
    _atomic_json(arguments.output, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
