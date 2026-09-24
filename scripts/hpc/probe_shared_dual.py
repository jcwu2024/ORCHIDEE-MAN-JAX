"""Matched scalar32 shared update on one versus two devices; no retained training."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from scripts.hpc.checked_recurrent_execution import device_helper_cache
from scripts.hpc.probe_shared_execution import compare_trees
from scripts.hpc.run_shared_learning import (
    _check_gradient_component,
    _method_identity,
    execute_learning_update,
    load_learning_checkpoint,
    make_tree_operations,
    memory_sizes,
    persist_tree,
    shared_config,
)
from scripts.hpc.run_shared_learning import parser as learning_parser
from scripts.hpc.shared_daily_microbatch import make_daily_accumulator


def compile_evaluator_on_device(
    *,
    jax: Any,
    objective: Callable[..., Any],
    arguments: Sequence[Any],
    device: Any,
) -> Callable[..., Any]:
    """Compile one value-and-gradient executable for an explicit device."""

    with jax.default_device(device), device_helper_cache(device):
        local_arguments = jax.device_put(tuple(arguments), device)
        backward = jax.jit(jax.value_and_grad(objective, has_aux=True))
        return backward.lower(*local_arguments).compile()


def make_scalar_daily_branch(
    *,
    jax: Any,
    device: Any,
    evaluator: Callable[..., Any],
    operations: Mapping[str, Callable[..., Any]],
    expected_count: int,
) -> Callable[[Any, Sequence[Any]], Mapping[str, Any]]:
    """Return the unchanged scalar-row mean-gradient branch on one device."""

    count = int(expected_count)

    def branch(parameters: Any, samples: Sequence[Any]) -> Mapping[str, Any]:
        if len(samples) != count:
            raise ValueError("dual daily row count differs from the declared mean")
        began = time.perf_counter()
        with jax.default_device(device), device_helper_cache(device):
            total = operations["zero"](parameters)
            values = []
            hard: dict[str, int] = {}
            for row, sample in enumerate(samples):
                (value, auxiliary), gradient = evaluator(parameters, sample)
                _check_gradient_component(
                    name="dual_daily",
                    value=value,
                    auxiliary=auxiliary,
                    gradient=gradient,
                    finite=operations["finite"],
                    row=row,
                )
                total = operations["add"](total, gradient)
                values.append(float(value))
                for name, item in auxiliary["hard"].items():
                    hard[name] = hard.get(name, 0) + int(np.asarray(item))
            gradient = operations["daily_mean"](total)
            norm = float(np.asarray(operations["norm"](gradient)))
        return {
            "gradient": gradient,
            "loss": float(np.mean(values, dtype=np.float64)),
            "norm": norm,
            "hard": hard,
            "seconds": time.perf_counter() - began,
        }

    return branch


def make_microbatch_daily_branch(
    *,
    jax: Any,
    jnp: Any,
    device: Any,
    evaluator: Callable[..., Any],
    operations: Mapping[str, Callable[..., Any]],
    microbatch_size: int,
    expected_count: int,
) -> Callable[[Any, Any], Mapping[str, Any]]:
    """Return the exact equal-row mean-gradient branch using bounded chunks."""

    accumulate = make_daily_accumulator(
        jax=jax,
        jnp=jnp,
        microbatch_size=microbatch_size,
        evaluator=evaluator,
        expected_count=expected_count,
    )

    def branch(parameters: Any, samples: Any) -> Mapping[str, Any]:
        began = time.perf_counter()
        with jax.default_device(device), device_helper_cache(device):
            local_parameters = jax.device_put(parameters, device)
            local_samples = jax.device_put(samples, device)
            (value, auxiliary), gradient = accumulate(local_parameters, local_samples)
            hard = {
                name: int(np.asarray(item))
                for name, item in auxiliary["hard"].items()
            }
            norm = float(np.asarray(operations["norm"](gradient)))
            _check_gradient_component(
                name="dual_daily",
                value=value,
                auxiliary=auxiliary,
                gradient=gradient,
                finite=operations["finite"],
            )
            if not np.isfinite(norm):
                raise ValueError("dual_daily_numerical_or_physical: nonfinite gradient norm")
        return {
            "gradient": gradient,
            "loss": float(np.asarray(value)),
            "norm": norm,
            "hard": hard,
            "seconds": time.perf_counter() - began,
        }

    return branch


def make_recurrent_branch(
    *,
    jax: Any,
    device: Any,
    evaluators: Mapping[int, Callable[..., Any]],
    operations: Mapping[str, Callable[..., Any]],
) -> Callable[[Any, Mapping[int, Sequence[Any]]], tuple[Mapping[int, Any], float]]:
    """Return the unchanged sequential T7/T30 branch on one device."""

    if set(evaluators) != {7, 30}:
        raise ValueError("dual recurrent evaluators must be exactly T7 and T30")

    def branch(parameters: Any, cases: Mapping[int, Sequence[Any]]) -> tuple[Mapping[int, Any], float]:
        if set(cases) != {7, 30}:
            raise ValueError("dual recurrent cases must be exactly T7 and T30")
        began = time.perf_counter()
        results = {}
        with jax.default_device(device), device_helper_cache(device):
            for horizon in (7, 30):
                started = time.perf_counter()
                (value, auxiliary), gradient = evaluators[horizon](parameters, *cases[horizon])
                _check_gradient_component(
                    name=f"dual_T{horizon}",
                    value=value,
                    auxiliary=auxiliary,
                    gradient=gradient,
                    finite=operations["finite"],
                )
                results[horizon] = {
                    "gradient": gradient,
                    "loss": float(value),
                    "norm": float(np.asarray(operations["norm"](gradient))),
                    "hard": {
                        name: int(np.asarray(item)) for name, item in auxiliary["hard"].items()
                    },
                    "seconds": time.perf_counter() - started,
                }
        return results, time.perf_counter() - began

    return branch


def make_component_combiner(jax: Any) -> Callable[[Any, Any, Any], Any]:
    """Add the already-mean daily gradient to T7 and T30 exactly once."""

    return jax.jit(
        lambda daily, seven, thirty: jax.tree_util.tree_map(
            lambda one, two, three: one + two + three, daily, seven, thirty
        )
    )


def _any_hard(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_any_hard(item) for item in value.values())
    return bool(value)


def dual_cost_comparison(serial_seconds: float, dual_seconds: float, *, cpu_cores: int = 8) -> Any:
    """Report wall, GPU-hour, and Explore1000 charge ratios."""

    if not (
        np.isfinite(serial_seconds)
        and np.isfinite(dual_seconds)
        and serial_seconds > 0.0
        and dual_seconds > 0.0
    ):
        raise ValueError("dual cost comparison requires positive finite timings")
    speedup = serial_seconds / dual_seconds
    serial_rate = 2.20 + 0.07 * cpu_cores
    dual_rate = 4.40 + 0.07 * cpu_cores
    return {
        "speedup": speedup,
        "time_reduction": 1.0 - 1.0 / speedup,
        "gpu_hour_ratio": 2.0 / speedup,
        "estimated_total_charge_ratio": dual_rate / serial_rate / speedup,
        "serial_yuan_per_hour": serial_rate,
        "dual_yuan_per_hour": dual_rate,
        "assumed_cpu_cores_per_run": cpu_cores,
    }


class SharedDualExecutor:
    """Overlap scalar daily and recurrent branches, then update on device 0."""

    def __init__(
        self,
        *,
        jax: Any,
        parameters: Any,
        optimizer: Any,
        daily_branch: Callable[..., Any],
        recurrent_branch: Callable[..., Any],
        combine: Callable[..., Any],
        update: Callable[..., Any],
        finite: Callable[[Any], Any],
        devices: Sequence[Any] | None = None,
        require_gpus: bool = True,
    ) -> None:
        selected = tuple(jax.local_devices() if devices is None else devices)
        if len(selected) != 2:
            raise ValueError("shared dual execution requires exactly two devices")
        if require_gpus and any(device.platform != "gpu" for device in selected):
            raise ValueError("shared dual execution requires exactly two visible GPUs")
        self.jax = jax
        self.primary, self.secondary = selected
        self.daily_branch = daily_branch
        self.recurrent_branch = recurrent_branch
        self.combine = combine
        self.update = update
        self.finite = finite
        with jax.default_device(self.primary), device_helper_cache(self.primary):
            self.parameters, self.optimizer = jax.device_put(
                (parameters, optimizer), self.primary
            )
            jax.block_until_ready((self.parameters, self.optimizer))
        with jax.default_device(self.secondary), device_helper_cache(self.secondary):
            self.peer_parameters = jax.device_put(self.parameters, self.secondary)
            jax.block_until_ready(self.peer_parameters)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shared_recurrent")

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> "SharedDualExecutor":
        return self

    def __exit__(self, *_arguments: Any) -> None:
        self.close()

    def step(
        self,
        *,
        daily_payload: Sequence[Any],
        recurrent_payload: Mapping[int, Sequence[Any]],
        persist: bool,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        if self._executor is None:
            raise RuntimeError("shared dual executor is closed")
        began = time.perf_counter()
        future = self._executor.submit(
            self._run_recurrent, self.peer_parameters, recurrent_payload
        )
        try:
            daily = self._run_daily(self.parameters, daily_payload)
        except BaseException:
            try:
                future.result()
            except BaseException:
                pass
            raise
        recurrent, recurrent_seconds = future.result()
        branch_seconds = time.perf_counter() - began
        hard = {"daily": daily["hard"]}
        hard.update({f"T{horizon}": recurrent[horizon]["hard"] for horizon in (7, 30)})
        if _any_hard(hard):
            raise ValueError("shared dual branch returned nonzero physical counters")

        merge_started = time.perf_counter()
        with self.jax.default_device(self.primary), device_helper_cache(self.primary):
            seven = self.jax.device_put(recurrent[7]["gradient"], self.primary)
            thirty = self.jax.device_put(recurrent[30]["gradient"], self.primary)
            mixed = self.combine(daily["gradient"], seven, thirty)
            self.jax.block_until_ready(mixed)
            losses = {
                "daily": float(daily["loss"]),
                "T7": float(recurrent[7]["loss"]),
                "T30": float(recurrent[30]["loss"]),
            }
            norms = {
                "daily": float(daily["norm"]),
                "T7": float(recurrent[7]["norm"]),
                "T30": float(recurrent[30]["norm"]),
            }
            if (
                not all(np.isfinite(value) and value >= 0.0 for value in norms.values())
                or not all(np.isfinite(value) for value in losses.values())
                or not bool(np.asarray(self.finite(mixed)))
            ):
                raise ValueError("shared dual component or mixed gradient is nonfinite")
            proposal, next_optimizer, gradient_norm, clip_scale = self.update(
                self.parameters, mixed, self.optimizer
            )
            self.jax.block_until_ready(
                (proposal, next_optimizer, gradient_norm, clip_scale)
            )
            expected_step = int(np.asarray(self.optimizer.step)) + 1
            if (
                not bool(np.asarray(self.finite((proposal, next_optimizer))))
                or not np.isfinite(float(gradient_norm))
                or not 0.0 < float(clip_scale) <= 1.0
                or int(np.asarray(next_optimizer.step)) != expected_step
            ):
                raise ValueError("shared dual AdamW proposal is invalid")
        merge_seconds = time.perf_counter() - merge_started

        sync_started = time.perf_counter()
        with self.jax.default_device(self.secondary), device_helper_cache(self.secondary):
            peer_proposal = self.jax.device_put(proposal, self.secondary)
            self.jax.block_until_ready(peer_proposal)
        sync_seconds = time.perf_counter() - sync_started
        if persist:
            self.parameters, self.optimizer = proposal, next_optimizer
            self.peer_parameters = peer_proposal
        record = {
            "loss": {**losses, "joint": sum(losses.values())},
            "gradient_norms": norms,
            "optimizer_gradient_norm": float(gradient_norm),
            "clip_scale": float(clip_scale),
            "optimizer_step": int(np.asarray(next_optimizer.step)),
            "hard": hard,
            "all_hard_counts_zero": True,
            "timing_seconds": {
                "daily_scalar32": float(daily["seconds"]),
                "recurrent_total": float(recurrent_seconds),
                "branches_elapsed": branch_seconds,
                "merge_checks_optimizer": merge_seconds,
                "peer_parameter_sync": sync_seconds,
                "total": time.perf_counter() - began,
            },
            "synchronized_parameter_bytes": sum(
                int(leaf.size) * int(np.dtype(leaf.dtype).itemsize)
                for leaf in self.jax.tree_util.tree_leaves(peer_proposal)
            ),
        }
        return record, {
            "gradient": mixed,
            "parameters": proposal,
            "optimizer": next_optimizer,
            "peer_parameters": peer_proposal,
        }

    def _run_daily(self, parameters: Any, payload: Any) -> Any:
        with self.jax.default_device(self.primary), device_helper_cache(self.primary):
            return self.daily_branch(parameters, payload)

    def _run_recurrent(self, parameters: Any, payload: Any) -> Any:
        with self.jax.default_device(self.secondary), device_helper_cache(self.secondary):
            return self.recurrent_branch(parameters, payload)


def _compiled(
    *,
    jax: Any,
    objective: Callable[..., Any],
    arguments: Sequence[Any],
    device: Any,
    name: str,
    memory_limit: int,
    emit: Callable[..., None],
) -> tuple[Callable[..., Any], Mapping[str, Any]]:
    emit("compile_gradient_graph", component=name, device=str(device))
    began = time.perf_counter()
    executable = compile_evaluator_on_device(
        jax=jax, objective=objective, arguments=arguments, device=device
    )
    metadata = {
        "compile_seconds": time.perf_counter() - began,
        "memory": memory_sizes(executable),
        "device": str(device),
    }
    if metadata["memory"]["estimated_live_bytes"] > memory_limit:
        raise MemoryError(f"{name} exceeds the compiled-memory limit")
    emit("gradient_graph_compiled", component=name, compiled=metadata)
    return executable, metadata


def _device_payload(jax: Any, device: Any, value: Any) -> Any:
    with jax.default_device(device), device_helper_cache(device):
        result = jax.device_put(value, device)
        jax.block_until_ready(result)
        return result


def _artifact_payload(jax: Any, head: Any, artifact: Any, daily_count: int) -> tuple[Any, Any]:
    from research.daily_coarse_graining.daily_operator.shared_recurrent_training import (
        prepare_shared_window,
    )

    one_day = artifact.batch.one_day
    if int(np.shape(one_day.inputs.continuous_state)[0]) != int(daily_count):
        raise ValueError("dual artifact daily row count differs from the declared mean")
    samples = [
        jax.tree_util.tree_map(lambda value, index=index: value[index], one_day)
        for index in range(daily_count)
    ]
    recurrent = {
        7: prepare_shared_window(head, artifact.batch.seven_day, window_index=0, horizon=7),
        30: prepare_shared_window(head, artifact.batch.thirty_day, window_index=0, horizon=30),
    }
    return samples, recurrent


def _serial_mixed(
    *,
    parameters: Any,
    daily_payload: Any,
    recurrent_payload: Any,
    daily_branch: Callable[..., Any],
    recurrent_branch: Callable[..., Any],
    combine: Callable[..., Any],
) -> tuple[Any, Mapping[str, Any]]:
    daily = daily_branch(parameters, daily_payload)
    recurrent, recurrent_seconds = recurrent_branch(parameters, recurrent_payload)
    mixed = combine(daily["gradient"], recurrent[7]["gradient"], recurrent[30]["gradient"])
    return mixed, {"daily": daily, "recurrent": recurrent, "recurrent_seconds": recurrent_seconds}


def run(args: Any) -> Mapping[str, Any]:
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
    from scripts.hpc.shared_learning_plan import load_learning_configuration

    began = time.perf_counter()
    report: dict[str, Any] = {
        "status": "running",
        "retained_training_updates": 0,
        "scientific_acceptance": False,
        "source_checkpoint": str(args.resume_checkpoint),
        "source_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "host": platform.node(),
        "python_version": platform.python_version(),
    }

    def emit(stage: str, **values: Any) -> None:
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - began, **values)
        _atomic_json(args.output, report)
        print(f"shared_dual stage={stage} seconds={report['elapsed_seconds']:.1f}", flush=True)

    try:
        config = json.loads(args.benchmark_protocol.read_text(encoding="utf-8"))
        if config["schema_version"] != "shared_dual_execution_v1":
            raise ValueError("unsupported shared dual benchmark")
        if args.resume_checkpoint is None:
            raise ValueError("trained source checkpoint required")
        loaded = load_learning_configuration(
            args.protocol,
            args.joint_admission_report,
            args.prepared_manifest,
            args.validation_manifest,
        )
        saved = load_learning_checkpoint(
            args.resume_checkpoint, expected_method=_method_identity(loaded)
        )
        if int(saved["completed_updates"]) != int(config["source_update"]):
            raise ValueError("dual benchmark requires the declared update100 source")
        emit("source_verified", configuration=config, source_update=saved["completed_updates"])
        if args.prepare_only:
            emit("completed", status="passed", scope="checkpoint_and_metadata_only")
            return report

        from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
        from research.daily_coarse_graining.daily_operator.shared_persistent_outputs import (
            build_shared_persistent_coordinates,
        )
        from research.daily_coarse_graining.daily_operator.shared_process_head import (
            build_shared_process_head,
        )
        from research.daily_coarse_graining.daily_operator.shared_recurrent_training import (
            make_shared_joint_daily_objective,
            make_shared_recurrent_objective,
        )
        from research.daily_coarse_graining.daily_operator.shared_training import (
            derive_shared_state_change_scales,
        )
        from scripts.hpc.shared_transformer_runtime import load_shared_runtime

        artifact_ids = tuple(int(value) for value in config["artifact_ids"])
        if len(artifact_ids) != int(config["continuous_updates"]) or len(artifact_ids) != 2:
            raise ValueError("dual benchmark requires exactly two changing artifacts")
        emit("load_runtime")
        runtime, parent, template, artifacts, _metadata = load_shared_runtime(args, artifact_ids)
        jax = runtime["jax"]
        import jax.numpy as jnp

        devices = tuple(jax.local_devices())
        if len(devices) != 2 or any(device.platform != "gpu" for device in devices):
            raise ValueError("real shared dual benchmark requires exactly two visible GPUs")
        primary, secondary = devices
        report.update(jax_version=jax.__version__, devices=[str(device) for device in devices])

        daily_protocol = loaded["daily_protocol"]
        coordinates = build_shared_persistent_coordinates(
            parent.learned_schema,
            parent.contract,
            runtime["base_statistics"].arrays["state_delta"],
        )
        head = build_shared_process_head(
            parent,
            template,
            config=shared_config(daily_protocol["architecture"]),
            thermal_transform=daily_protocol["architecture"]["thermal_transform"],
            persistent_coordinates=coordinates,
            source_bare_cimean=True,
        )
        state_scales = derive_shared_state_change_scales(runtime, head)
        daily_objective = make_shared_joint_daily_objective(
            runtime, head, template, state_scales, loaded["admission"]["scales"]
        )
        _, recurrent_objective = make_shared_recurrent_objective(
            head, template, state_scales, loaded["admission"]["scales"]
        )
        parameters, optimizer = _device_payload(
            jax, primary, (saved["parameters"], saved["optimizer"])
        )
        del saved
        host_payloads = [
            _artifact_payload(jax, head, artifact, int(config["daily_count"]))
            for artifact in artifacts
        ]
        primary_payloads = [_device_payload(jax, primary, payload) for payload in host_payloads]
        memory_limit = int(config["compiled_memory_limit_gib"]) * 2**30
        compiles: dict[str, Any] = {}

        daily_evaluator, compiles["daily_gpu0"] = _compiled(
            jax=jax,
            objective=daily_objective,
            arguments=(parameters, primary_payloads[0][0][0]),
            device=primary,
            name="daily_gpu0",
            memory_limit=memory_limit,
            emit=emit,
        )
        recurrent_primary = {}
        for horizon in (7, 30):
            recurrent_primary[horizon], compiles[f"T{horizon}_gpu0"] = _compiled(
                jax=jax,
                objective=recurrent_objective,
                arguments=(parameters, *primary_payloads[0][1][horizon]),
                device=primary,
                name=f"T{horizon}_gpu0",
                memory_limit=memory_limit,
                emit=emit,
            )
        primary_operations = make_tree_operations(jax, jnp, int(config["daily_count"]))
        with jax.default_device(primary), device_helper_cache(primary):
            combine = make_component_combiner(jax)
        update = make_update(daily_protocol)
        daily_primary = make_scalar_daily_branch(
            jax=jax,
            device=primary,
            evaluator=daily_evaluator,
            operations=primary_operations,
            expected_count=int(config["daily_count"]),
        )
        recurrent_primary_branch = make_recurrent_branch(
            jax=jax,
            device=primary,
            evaluators=recurrent_primary,
            operations=primary_operations,
        )
        serial_evaluators = {
            "daily": daily_evaluator,
            "T7": recurrent_primary[7],
            "T30": recurrent_primary[30],
        }

        def serial_once(current_parameters: Any, current_optimizer: Any, payload: Any) -> Any:
            with jax.default_device(primary), device_helper_cache(primary):
                return execute_learning_update(
                    jax=jax,
                    parameters=current_parameters,
                    optimizer=current_optimizer,
                    daily_samples=payload[0],
                    seven_case=payload[1][7],
                    thirty_case=payload[1][30],
                    evaluators=serial_evaluators,
                    operations=primary_operations,
                    update=update,
                )

        emit("serial_reference_measurement")
        serial_result = serial_once(parameters, optimizer, primary_payloads[0])
        jax.block_until_ready(serial_result[:2])
        serial_gradient, serial_components = _serial_mixed(
            parameters=parameters,
            daily_payload=primary_payloads[0][0],
            recurrent_payload=primary_payloads[0][1],
            daily_branch=daily_primary,
            recurrent_branch=recurrent_primary_branch,
            combine=combine,
        )
        reference_host = jax.device_get(
            {
                "gradient": serial_gradient,
                "parameters": serial_result[0],
                "optimizer": serial_result[1],
            }
        )
        reference_report = jax.device_get(serial_result[2])
        reference_path = args.output.with_suffix(".serial-reference.pkl")
        persist_tree(
            reference_path,
            jax,
            {
                **reference_host,
                "update": reference_report,
                "config": config,
                "source_checkpoint": str(args.resume_checkpoint),
            },
        )
        report["serial_reference"] = {
            "saved": str(reference_path),
            "process_memory": primary.memory_stats(),
        }
        del serial_result, serial_gradient, serial_components
        emit("serial_reference_saved")

        peer_parameters = _device_payload(jax, secondary, parameters)
        secondary_payloads = [
            (payload[0], _device_payload(jax, secondary, host_payloads[index][1]))
            for index, payload in enumerate(primary_payloads)
        ]
        recurrent_secondary = {}
        for horizon in (7, 30):
            recurrent_secondary[horizon], compiles[f"T{horizon}_gpu1"] = _compiled(
                jax=jax,
                objective=recurrent_objective,
                arguments=(peer_parameters, *secondary_payloads[0][1][horizon]),
                device=secondary,
                name=f"T{horizon}_gpu1",
                memory_limit=memory_limit,
                emit=emit,
            )
        secondary_operations = make_tree_operations(jax, jnp, int(config["daily_count"]))
        recurrent_secondary_branch = make_recurrent_branch(
            jax=jax,
            device=secondary,
            evaluators=recurrent_secondary,
            operations=secondary_operations,
        )
        comparisons = {
            "rtol": float(config["comparison_rtol"]),
            "atol": float(config["comparison_atol"]),
        }

        with SharedDualExecutor(
            jax=jax,
            parameters=parameters,
            optimizer=optimizer,
            daily_branch=daily_primary,
            recurrent_branch=recurrent_secondary_branch,
            combine=combine,
            update=update,
            finite=primary_operations["finite"],
            devices=devices,
        ) as dual:
            for _ in range(int(config["warmups"])):
                serial_warm = serial_once(parameters, optimizer, primary_payloads[0])
                jax.block_until_ready(serial_warm[:2])
                dual_warm_record, dual_warm = dual.step(
                    daily_payload=primary_payloads[0][0],
                    recurrent_payload=secondary_payloads[0][1],
                    persist=False,
                )
                del serial_warm, dual_warm_record, dual_warm
            serial_rows, dual_rows = [], []
            timed_serial_result = None
            dual_values = None
            dual_record = None
            for repeat in range(int(config["timed_replays"])):
                order = ("serial", "dual") if repeat % 2 == 0 else ("dual", "serial")
                for mode in order:
                    started = time.perf_counter()
                    if mode == "serial":
                        timed_serial_result = serial_once(
                            parameters, optimizer, primary_payloads[0]
                        )
                        jax.block_until_ready(timed_serial_result[:2])
                        serial_rows.append(
                            {
                                "wall_seconds": time.perf_counter() - started,
                                **timed_serial_result[2],
                            }
                        )
                    else:
                        dual_record, dual_values = dual.step(
                            daily_payload=primary_payloads[0][0],
                            recurrent_payload=secondary_payloads[0][1],
                            persist=False,
                        )
                        dual_rows.append(
                            {"wall_seconds": time.perf_counter() - started, **dual_record}
                        )
            serial_median = float(
                np.median([row["wall_seconds"] for row in serial_rows])
            )
            dual_median = float(np.median([row["wall_seconds"] for row in dual_rows]))
            report["serial_reference"]["timing"] = {
                "replays": serial_rows,
                "median_seconds": serial_median,
                "interleaved_after_both_devices_compiled": True,
            }
            fixed_comparisons = {
                name: compare_trees(jax, dual_values[name], reference_host[name], **comparisons)
                for name in ("gradient", "parameters", "optimizer")
            }
            fixed_comparisons["peer_parameters"] = compare_trees(
                jax, dual_values["peer_parameters"], reference_host["parameters"], **comparisons
            )
            joint_scalar_pass = bool(
                np.isclose(
                    dual_record["loss"]["joint"],
                    reference_report["loss"]["joint"],
                    **comparisons,
                )
            )
            fixed_pass = all(item["passed"] for item in fixed_comparisons.values()) and joint_scalar_pass
            report["fixed_source"] = {
                "status": "passed" if fixed_pass else "rejected",
                "comparisons": fixed_comparisons,
                "joint_scalar_pass": joint_scalar_pass,
                "timing": {
                    "replays": dual_rows,
                    "median_seconds": dual_median,
                    "interleaved_after_both_devices_compiled": True,
                },
                "cost": dual_cost_comparison(serial_median, dual_median),
            }
            if not fixed_pass:
                raise ValueError("dual fixed-source update differs from serial reference")
            del dual_values, timed_serial_result

        serial_parameters, serial_optimizer = parameters, optimizer
        changing = []
        with SharedDualExecutor(
            jax=jax,
            parameters=parameters,
            optimizer=optimizer,
            daily_branch=daily_primary,
            recurrent_branch=recurrent_secondary_branch,
            combine=combine,
            update=update,
            finite=primary_operations["finite"],
            devices=devices,
        ) as continuous_dual:
            for index, identifier in enumerate(artifact_ids):
                serial_mixed, _components = _serial_mixed(
                    parameters=serial_parameters,
                    daily_payload=primary_payloads[index][0],
                    recurrent_payload=primary_payloads[index][1],
                    daily_branch=daily_primary,
                    recurrent_branch=recurrent_primary_branch,
                    combine=combine,
                )
                serial_parameters, serial_optimizer, serial_update = serial_once(
                    serial_parameters, serial_optimizer, primary_payloads[index]
                )
                dual_update, dual_values = continuous_dual.step(
                    daily_payload=primary_payloads[index][0],
                    recurrent_payload=secondary_payloads[index][1],
                    persist=True,
                )
                step_comparisons = {
                    "gradient": compare_trees(
                        jax, dual_values["gradient"], serial_mixed, **comparisons
                    ),
                    "parameters": compare_trees(
                        jax, dual_values["parameters"], serial_parameters, **comparisons
                    ),
                    "optimizer": compare_trees(
                        jax, dual_values["optimizer"], serial_optimizer, **comparisons
                    ),
                    "peer_parameters": compare_trees(
                        jax, dual_values["peer_parameters"], serial_parameters, **comparisons
                    ),
                }
                scalar_pass = bool(
                    np.isclose(
                        dual_update["loss"]["joint"],
                        serial_update["loss"]["joint"],
                        **comparisons,
                    )
                )
                step_pass = all(item["passed"] for item in step_comparisons.values()) and scalar_pass
                changing.append(
                    {
                        "update": index + 1,
                        "artifact_id": identifier,
                        "status": "passed" if step_pass else "rejected",
                        "comparisons": step_comparisons,
                        "joint_scalar_pass": scalar_pass,
                        "serial_optimizer_step": int(np.asarray(serial_optimizer.step)),
                        "dual_optimizer_step": dual_update["optimizer_step"],
                    }
                )
                report["changing_data"] = changing
                emit(
                    "changing_update_compared",
                    changing_update=index + 1,
                    changing_update_status=changing[-1]["status"],
                )
                if not step_pass:
                    raise ValueError("dual changing-data update differs from serial reference")
                del serial_mixed, dual_values, _components
        report.update(
            compilation=compiles,
            changing_data=changing,
            memory={str(device): device.memory_stats() for device in devices},
        )
        emit(
            "completed",
            status="passed",
            scope="bounded_shared_scalar32_dual_execution_only",
            retained_training_updates=0,
            includes_recurrent_gradient_transfer=True,
            includes_peer_parameter_sync=True,
            next_gate="integrate_only_after_measured_cost_and_speed_decision",
        )
        return report
    except Exception as error:
        emit("error", status="error", error_type=type(error).__name__, error=str(error))
        raise


def parser() -> Any:
    result = learning_parser()
    result.description = __doc__
    result.add_argument("--benchmark-protocol", type=Path, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
