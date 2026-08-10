"""Train the matched successor arms under the frozen bounded-fit contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import (
    admit_prepared_manifest_for_checkpoint_fit,
    load_successor_checkpoint_handoff,
    write_successor_checkpoint_set,
)
from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    EXPECTED_PARAMETER_COUNT,
    _head,
    _load_extended_runtime,
)
from scripts.hpc.run_successor_process_head_canonical_gate import (
    _atomic_json,
    _retained_adapter,
)
from scripts.hpc.run_successor_small_fit_probe import (
    _plan_context,
    _reference_for_entry,
    _teacher_replay_gate_with_transition,
    _TeacherOutputHead,
)
from scripts.hpc.successor_small_fit_prepared import (
    load_prepared_artifact,
    load_prepared_manifest,
    sha256_file,
)


class ArmUpdateResult(NamedTuple):
    parameters: Any
    optimizer: Any
    loss: Any
    anchor: Any
    components: Any
    gradient_norm: Any
    nonfinite_gradient_values: Any
    hard_counts: Any
    update_applied: Any


class ArmEvaluation(NamedTuple):
    loss: Any
    anchor: Any
    components: Any
    hard_counts: Any


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _trace_static_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _trace_static_jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if hasattr(value, "_asdict"):
        return {
            "__type__": type(value).__qualname__,
            "fields": _trace_static_jsonable(value._asdict()),
        }
    if isinstance(value, (tuple, list)):
        return [_trace_static_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if callable(value):
        return {
            "__callable__": (
                f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', type(value).__qualname__)}"
            )
        }
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    array = np.asarray(value)
    if array.dtype.kind == "O":
        raise TypeError(f"unsupported retained trace-static value {type(value)!r}")
    return {
        "__array__": True,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _retained_tail_trace_signature(adapter: Any) -> str:
    """Hash controls that genuinely select a retained-owner executable."""

    owner = adapter.static_owner
    metadata = owner.metadata
    payload = {
        "config_path": str(owner.config_path.resolve()),
        "contract_sha256": owner.contract.sha256,
        "mineral_imin": int(owner.mineral_imin),
        "mineral_imax": int(owner.mineral_imax),
        "daily_carbon_dispatch": _trace_static_jsonable(owner.daily_carbon_dispatch),
        "metadata": _trace_static_jsonable(
            {
                "nstm": metadata.nstm,
                "is_tree": metadata.is_tree,
                "is_peat": metadata.is_peat,
                "npts": metadata.npts,
            }
        ),
        "season_provenance": _trace_static_jsonable(owner.season_provenance),
        "lifecycle_start_tstep": int(owner.lifecycle_start_tstep),
        "result_day_index": int(owner.result_day_index),
        "include_modelout_diagnostics": bool(owner.include_modelout_diagnostics),
        "fast_owner_materializer": type(adapter.fast_owner_materializer).__qualname__,
    }
    return _canonical_sha256(payload)


def _tree_shape_signature(jax: Any, value: Any) -> tuple[tuple[tuple[int, ...], str], ...]:
    return tuple((tuple(np.shape(leaf)), str(np.asarray(leaf).dtype)) for leaf in jax.tree_util.tree_leaves(value))


def _hard_counts_host(value: Any) -> Mapping[str, int]:
    return {name: int(np.asarray(item)) for name, item in value._asdict().items()}


def _component_record(names: tuple[str, ...], values: Any) -> Mapping[str, float]:
    return {name: float(value) for name, value in zip(names, np.asarray(values), strict=True)}


def _device_memory_report(devices: Any) -> list[Mapping[str, Any]]:
    records = []
    for device in devices:
        stats = device.memory_stats() or {}
        records.append(
            {
                "device": str(device),
                "bytes_in_use": int(stats.get("bytes_in_use", 0)),
                "peak_bytes_in_use": int(stats.get("peak_bytes_in_use", 0)),
                "bytes_limit": int(stats.get("bytes_limit", 0)),
            }
        )
    return records


def _validate_update_count(updates: int, maximum: int) -> int:
    if not 2 <= int(updates) <= int(maximum):
        raise ValueError(f"updates must lie in [2, {maximum}]")
    return int(updates)


def _make_replay_transition(runtime: Mapping[str, Any], adapter_template: Any) -> Any:
    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator
    from research.daily_coarse_graining.daily_operator.successor_architecture import (
        SOURCE_ROUTED_ATTENTION,
    )

    jax = runtime["jax"]
    replay_head = _TeacherOutputHead(_head(runtime, SOURCE_ROUTED_ATTENTION))

    @jax.jit
    def replay(parameters: Any, inputs: Any, day_inputs: Any) -> Any:
        retained_tail = replace(adapter_template, day_inputs=day_inputs)
        transition = bind_daily_operator(
            input_assembler=runtime["adapter"].input_assembler,
            process_head=replay_head,
            retained_tail=retained_tail,
        )
        return transition(parameters, inputs)

    return replay


def _make_arm_steps(
    runtime: Mapping[str, Any],
    arm: str,
    adapter_template: Any,
    coefficients: Mapping[str, float],
    long_scales: Any,
    runner_contract: Any,
) -> tuple[Any, Any]:
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        successor_adamw_update,
        successor_anchor_batch_loss,
        successor_rollout_component_losses,
        successor_rollout_hard_gate_counts,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        LONG_HORIZON_COMPONENTS,
    )

    jax = runtime["jax"]
    head = _head(runtime, arm)
    coefficient_vector = jnp.asarray(
        tuple(coefficients[name] for name in LONG_HORIZON_COMPONENTS),
        dtype=jnp.float64,
    )

    def objective(parameters: Any, batch: Any, day_inputs: Any, *, rematerialize: bool):
        retained_tail = replace(adapter_template, day_inputs=day_inputs)
        transition = bind_daily_operator(
            input_assembler=runtime["adapter"].input_assembler,
            process_head=head,
            retained_tail=retained_tail,
        )
        anchor = successor_anchor_batch_loss(
            parameters,
            batch.anchor_inputs,
            batch.anchor_targets,
            head=head,
            schema=runtime["schema"],
            scales=runtime["loss_scales"],
        )
        components, trace = successor_rollout_component_losses(
            parameters,
            batch,
            transition=transition,
            markov_contract=runtime["contract"],
            flux_scales=long_scales.fluxes,
            stock_scales=long_scales.stocks,
            rematerialize=rematerialize,
        )
        hard_counts = successor_rollout_hard_gate_counts(
            batch,
            trace,
            markov_contract=runtime["contract"],
            budget_atol=float(runner_contract.payload["gates"]["budget_absolute_tolerance"]),
            stock_atol=float(runner_contract.payload["gates"]["stock_negative_absolute_tolerance"]),
        )
        return anchor + jnp.sum(coefficient_vector * components), (
            anchor,
            components,
            hard_counts,
        )

    @jax.jit
    def train_step(
        parameters: Any,
        optimizer: Any,
        batch: Any,
        day_inputs: Any,
        learning_rate: Any,
    ) -> ArmUpdateResult:
        (loss, auxiliary), gradients = jax.value_and_grad(
            lambda values: objective(values, batch, day_inputs, rematerialize=True),
            has_aux=True,
        )(parameters)
        anchor, components, hard_counts = auxiliary
        proposed_parameters, proposed_optimizer, gradient_norm = successor_adamw_update(
            parameters,
            gradients,
            optimizer,
            learning_rate,
            runner_contract,
        )
        nonfinite = sum(
            (jnp.sum(~jnp.isfinite(jnp.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(gradients)),
            jnp.asarray(0, dtype=jnp.int64),
        )
        hard_total = sum(
            (jnp.asarray(leaf) for leaf in jax.tree_util.tree_leaves(hard_counts)),
            jnp.asarray(0, dtype=jnp.int64),
        )
        applied = jnp.isfinite(loss) & (nonfinite == 0) & (hard_total == 0)
        next_parameters = jax.tree_util.tree_map(
            lambda proposed, current: jnp.where(applied, proposed, current),
            proposed_parameters,
            parameters,
        )
        next_optimizer = jax.tree_util.tree_map(
            lambda proposed, current: jnp.where(applied, proposed, current),
            proposed_optimizer,
            optimizer,
        )
        return ArmUpdateResult(
            next_parameters,
            next_optimizer,
            loss,
            anchor,
            components,
            gradient_norm,
            nonfinite,
            hard_counts,
            applied,
        )

    @jax.jit
    def evaluate(parameters: Any, batch: Any, day_inputs: Any) -> ArmEvaluation:
        loss, auxiliary = objective(
            parameters,
            batch,
            day_inputs,
            rematerialize=False,
        )
        anchor, components, hard_counts = auxiliary
        return ArmEvaluation(loss, anchor, components, hard_counts)

    return train_step, evaluate


def _prepare_adapter(
    runtime: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> tuple[Any, Any, Mapping[str, float]]:
    from jax_orchidee.driver import orchestration as teacher
    from research.daily_coarse_graining.daily_markov_contract import load_markov_shard

    timings = {}
    reference = _reference_for_entry(runtime["index"], entry)
    paths = _plan_context(
        runtime["manifest"],
        landpoint_id=str(entry["landpoint_id"]),
        year=int(entry["year"]),
    )
    started = time.perf_counter()
    context = teacher.prepare_paper_1961_driver_context(
        paths["config"],
        used_run_def_path=paths["run_def"],
        reference_run_dir=paths["reference_run_dir"],
    )
    parent = load_markov_shard(reference.parent.path, contract=runtime["contract"])
    sample = parent.sample(int(entry["rollout_starts"][0]))
    adapter = _retained_adapter(
        runtime | {"context_paths": paths},
        context,
        sample,
        lifecycle_start_tstep=48,
        result_day_index=2,
        fast_owner_materializer=runtime["materializer"],
    )
    timings["retained_runtime_seconds"] = time.perf_counter() - started
    return reference, adapter, timings


def _prepare_update(
    runtime: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> tuple[Any, Any, Any, Mapping[str, float]]:
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        prepare_successor_small_fit_batch,
    )

    reference = _reference_for_entry(runtime["index"], entry)
    started = time.perf_counter()
    batch = prepare_successor_small_fit_batch(
        reference,
        entry,
        adapter=runtime["adapter"],
        schema=runtime["schema"],
        owner_contract=runtime["owner_contract"],
    )
    batch_seconds = time.perf_counter() - started
    retained_reference, adapter, timings = _prepare_adapter(runtime, entry)
    if retained_reference != reference:
        raise ValueError("successor batch and retained adapter reference drift")
    return (
        reference,
        batch,
        adapter,
        {
            "batch_preparation_seconds": batch_seconds,
            **timings,
        },
    )


def _git_head() -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        text=True,
    ).strip()


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"small-fit output already exists: {args.output}")
    run_started = time.perf_counter()
    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]

    from research.daily_coarse_graining.daily_operator.successor_architecture import (
        SOURCE_ROUTED_ATTENTION,
        SOURCE_ROUTED_GRAPH_CONTROL,
        initialize_successor_parameters,
        successor_parameter_count,
    )
    from research.daily_coarse_graining.daily_operator.successor_frozen_coefficients import (
        load_frozen_successor_coefficients,
    )
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        build_successor_small_fit_schedule,
        initialize_successor_adamw,
        load_successor_small_fit_contract,
        successor_learning_rate,
        verify_successor_small_fit_schedule,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        LONG_HORIZON_COMPONENTS,
        derive_long_horizon_scales,
    )

    runner_contract = load_successor_small_fit_contract(args.runner_contract)
    checkpoint_handoff = load_successor_checkpoint_handoff(args.checkpoint_handoff)
    maximum_updates = int(runner_contract.sampling["maximum_updates_per_arm"])
    requested_updates = _validate_update_count(args.updates, maximum_updates)
    production_fit = requested_updates == maximum_updates
    if not production_fit:
        raise ValueError("the checkpoint-handoff runner requires the full 500-update production fit")
    identity = runner_contract.payload["identity"]
    if (
        identity["successor_candidate_contract_sha256"] != runtime["successor_contract"].sha256
        or identity["fast_owner_contract_sha256"] != runtime["owner_contract"].contract_sha256
    ):
        raise ValueError("small-fit runner and model contracts differ")

    references = runtime["index"].select(
        spatial_split="train",
        temporal_split="train",
    )
    schedule = build_successor_small_fit_schedule(
        references,
        runner_contract,
        updates=requested_updates,
    )
    verify_successor_small_fit_schedule(schedule, references, runner_contract)
    git_head = _git_head()
    producer = checkpoint_handoff.producer
    prepared_manifest = load_prepared_manifest(
        args.prepared_manifest,
        runner_contract_sha256=str(producer["producer_runner_contract_sha256"]),
        schedule_sha256=schedule["schedule_sha256"],
        updates=requested_updates,
        source_git_head=str(producer["producer_git_head"]),
    )
    prepared_manifest_sha256 = sha256_file(args.prepared_manifest)
    artifact_admission = admit_prepared_manifest_for_checkpoint_fit(
        checkpoint_handoff,
        args.prepared_manifest,
        prepared_manifest,
        runner_contract_sha256=runner_contract.sha256,
        schedule_sha256=schedule["schedule_sha256"],
        consumer_git_head=git_head,
    )
    objective_contract = runner_contract.payload["objective"]
    coefficient_load_started = time.perf_counter()
    frozen_coefficients = load_frozen_successor_coefficients(
        args.coefficient_contract,
        source_report_path=args.coefficient_source_report,
        runner_contract_sha256=runner_contract.sha256,
        schedule_sha256=schedule["schedule_sha256"],
        successor_contract_sha256=runtime["successor_contract"].sha256,
        fast_owner_contract_sha256=runtime["owner_contract"].contract_sha256,
        first_prepared_artifact_sha256=str(prepared_manifest["entries"][0]["sha256"]),
        components=LONG_HORIZON_COMPONENTS,
        gradient_ratio_targets=objective_contract["gradient_ratio_targets"],
        coefficient_minimum=float(objective_contract["coefficient_minimum"]),
        coefficient_maximum=float(objective_contract["coefficient_maximum"]),
    )
    coefficient_contract_load_seconds = time.perf_counter() - coefficient_load_started
    coefficients = dict(frozen_coefficients.coefficients)
    execution_identity = dict(identity) | dict(artifact_admission) | {
        "git_head": git_head,
        "runner_contract_sha256": runner_contract.sha256,
        "schedule_sha256": schedule["schedule_sha256"],
        "prepared_manifest_sha256": prepared_manifest_sha256,
        "frozen_coefficient_contract_sha256": frozen_coefficients.sha256,
        "frozen_coefficient_source_report_sha256": frozen_coefficients.payload["source_evidence"][
            "runner_report_sha256"
        ],
    }
    first_entry = schedule["entries"][0]
    first_load_started = time.perf_counter()
    first_artifact = load_prepared_artifact(
        args.prepared_manifest,
        prepared_manifest["entries"][0],
        schedule_entry=first_entry,
    )
    first_load_seconds = time.perf_counter() - first_load_started
    _reference, first_adapter, first_adapter_timing = _prepare_adapter(runtime, first_entry)
    first_signature = _retained_tail_trace_signature(first_adapter)
    if first_artifact.trace_signature != first_signature:
        raise ValueError("prepared successor trace differs from live retained owner")
    first_batch = first_artifact.batch
    first_adapter = replace(first_adapter, day_inputs=first_artifact.day_inputs)
    replay_steps = {first_signature: _make_replay_transition(runtime, first_adapter)}
    replay_step = replay_steps[first_signature]
    replay_started = time.perf_counter()
    first_replay = _teacher_replay_gate_with_transition(
        runtime,
        first_batch,
        first_adapter,
        lambda parameters, inputs: replay_step(
            parameters,
            inputs,
            first_adapter.day_inputs,
        ),
    )
    first_replay_seconds = time.perf_counter() - replay_started
    if first_replay["status"] != "passed":
        report = {
            "schema_version": "successor_small_fit_contract_runner_v1",
            "status": "failed",
            "failed_stage": "first_annual_shard_teacher_replay",
            "checkpoint_written": False,
            "training_updates_per_arm": 0,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "identity": execution_identity,
            "annual_shard_teacher_replay": first_replay,
        }
        _atomic_json(args.output, report)
        return report

    assembled = runtime["adapter"].input_assembler.assemble(
        jax.tree_util.tree_map(lambda value: value[0], first_batch.anchor_inputs)
    )
    arms = (SOURCE_ROUTED_ATTENTION, SOURCE_ROUTED_GRAPH_CONTROL)
    parameters = {
        arm: initialize_successor_parameters(
            jax.random.PRNGKey(int(runner_contract.sampling["seed"])),
            schema=runtime["schema"],
            layout=runtime["layout"],
            inputs=assembled,
            arm=arm,
        )
        for arm in arms
    }
    counts = {arm: successor_parameter_count(value) for arm, value in parameters.items()}
    if set(counts.values()) != {EXPECTED_PARAMETER_COUNT}:
        raise ValueError(f"small-fit parameter count drift: {counts}")
    optimizer = {arm: initialize_successor_adamw(parameters[arm]) for arm in arms}
    long_scales = derive_long_horizon_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )

    step_cache: dict[tuple[str, str], tuple[Any, Any]] = {}
    first_batches = {first_signature: (first_batch, first_adapter)}
    records = []
    initial_metrics = {}
    completed_updates = {arm: 0 for arm in arms}
    total_started = time.perf_counter()
    memory_before = _device_memory_report(runtime["devices"])
    prepared_first = (
        first_batch,
        first_adapter,
        {
            "prepared_artifact_load_seconds": first_load_seconds,
            "prepared_artifact_size_bytes": int(prepared_manifest["entries"][0]["size_bytes"]),
            **first_adapter_timing,
        },
    )
    prepared_artifact_load_seconds = first_load_seconds

    for update, entry in enumerate(schedule["entries"]):
        if update == 0:
            batch, adapter, preparation = prepared_first
        else:
            load_started = time.perf_counter()
            artifact = load_prepared_artifact(
                args.prepared_manifest,
                prepared_manifest["entries"][update],
                schedule_entry=entry,
            )
            load_seconds = time.perf_counter() - load_started
            prepared_artifact_load_seconds += load_seconds
            if artifact.trace_signature != first_signature:
                raise ValueError("prepared successor trace identity changed within schedule")
            batch = artifact.batch
            adapter = replace(first_adapter, day_inputs=artifact.day_inputs)
            preparation = {
                "prepared_artifact_load_seconds": load_seconds,
                "prepared_artifact_size_bytes": int(prepared_manifest["entries"][update]["size_bytes"]),
                "retained_runtime_seconds": 0.0,
            }
        signature = _retained_tail_trace_signature(adapter)
        first_batches.setdefault(signature, (batch, adapter))
        if signature not in replay_steps:
            replay_steps[signature] = _make_replay_transition(runtime, adapter)
        replay_step = replay_steps[signature]
        replay_cache_before = int(replay_step._cache_size())
        replay_started = time.perf_counter()
        replay = (
            first_replay
            if update == 0
            else _teacher_replay_gate_with_transition(
                runtime,
                batch,
                adapter,
                lambda values, inputs, step=replay_step, day_inputs=adapter.day_inputs: step(
                    values,
                    inputs,
                    day_inputs,
                ),
            )
        )
        replay_seconds = first_replay_seconds if update == 0 else time.perf_counter() - replay_started
        replay_cache_after = int(replay_step._cache_size())
        if replay["status"] != "passed":
            report = {
                "schema_version": "successor_small_fit_contract_runner_v1",
                "status": "failed",
                "failed_stage": "annual_shard_teacher_replay",
                "failed_update": update + 1,
                "failed_schedule_entry": entry,
                "checkpoint_written": False,
                "completed_updates_per_arm": completed_updates,
                "teacher_data_regenerated": False,
                "historical_v5_used": False,
                "test_split_used": False,
                "identity": execution_identity,
                "teacher_replay": replay,
                "completed_update_records": records,
                "total_wall_seconds": time.perf_counter() - run_started,
                "next_gate": "repair_annual_shard_retained_tail_runtime",
            }
            _atomic_json(args.output, report)
            return report

        arm_records = {}
        rate = successor_learning_rate(runner_contract, update + 1)
        for arm in arms:
            key = (arm, signature)
            if key not in step_cache:
                step_cache[key] = _make_arm_steps(
                    runtime,
                    arm,
                    adapter,
                    coefficients,
                    long_scales,
                    runner_contract,
                )
            train_step, _evaluate = step_cache[key]
            cache_before = int(train_step._cache_size())
            started = time.perf_counter()
            result = train_step(
                parameters[arm],
                optimizer[arm],
                batch,
                adapter.day_inputs,
                np.asarray(rate, dtype=np.float64),
            )
            jax.block_until_ready(result)
            seconds = time.perf_counter() - started
            cache_after = int(train_step._cache_size())
            applied = bool(np.asarray(result.update_applied))
            hard_counts = _hard_counts_host(result.hard_counts)
            arm_record = {
                "loss": float(np.asarray(result.loss)),
                "one_day_anchor": float(np.asarray(result.anchor)),
                "long_horizon_components": _component_record(
                    LONG_HORIZON_COMPONENTS,
                    result.components,
                ),
                "gradient_norm": float(np.asarray(result.gradient_norm)),
                "nonfinite_gradient_values": int(np.asarray(result.nonfinite_gradient_values)),
                "hard_gate_counts": hard_counts,
                "update_applied": applied,
                "seconds": seconds,
                "jit_cache_size_before": cache_before,
                "jit_cache_size_after": cache_after,
                "new_jit_executables": cache_after - cache_before,
            }
            arm_records[arm] = arm_record
            if update == 0:
                initial_metrics[arm] = {
                    "total": arm_record["loss"],
                    "one_day_anchor": arm_record["one_day_anchor"],
                    "long_horizon_components": arm_record["long_horizon_components"],
                }
            if not applied:
                report = {
                    "schema_version": "successor_small_fit_contract_runner_v1",
                    "status": "failed",
                    "failed_stage": "optimizer_update",
                    "failed_update": update + 1,
                    "failed_arm": arm,
                    "failed_schedule_entry": entry,
                    "failed_arm_record": arm_record,
                    "checkpoint_written": False,
                    "completed_updates_per_arm": completed_updates,
                    "teacher_data_regenerated": False,
                    "historical_v5_used": False,
                    "test_split_used": False,
                    "identity": execution_identity,
                    "completed_update_records": records,
                    "total_wall_seconds": time.perf_counter() - run_started,
                    "next_gate": "repair_contract_runner",
                }
                _atomic_json(args.output, report)
                return report
            parameters[arm] = result.parameters
            optimizer[arm] = result.optimizer
            completed_updates[arm] += 1

        record = {
            "update": update + 1,
            "landpoint_id": str(entry["landpoint_id"]),
            "year": int(entry["year"]),
            "trace_signature": signature,
            "learning_rate": rate,
            "preparation": dict(preparation),
            "teacher_replay": {
                "status": replay["status"],
                "seconds": replay_seconds,
                "jit_cache_size_before": replay_cache_before,
                "jit_cache_size_after": replay_cache_after,
                "new_jit_executables": replay_cache_after - replay_cache_before,
            },
            "arms": arm_records,
        }
        records.append(record)
        print(
            json.dumps(
                {
                    "event": "successor_small_fit_update",
                    "update": update + 1,
                    "updates": requested_updates,
                    "landpoint_id": entry["landpoint_id"],
                    "year": int(entry["year"]),
                    "attention_loss": arm_records[SOURCE_ROUTED_ATTENTION]["loss"],
                    "control_loss": arm_records[SOURCE_ROUTED_GRAPH_CONTROL]["loss"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    final_metrics = {}
    final_evaluation_seconds = {}
    final_evaluation_compile_reuse = {}
    for arm in arms:
        train_step, _evaluate = step_cache[(arm, first_signature)]
        cache_before = int(train_step._cache_size())
        started = time.perf_counter()
        evaluation = train_step(
            parameters[arm],
            optimizer[arm],
            first_batch,
            first_adapter.day_inputs,
            np.asarray(0.0, dtype=np.float64),
        )
        jax.block_until_ready(evaluation)
        final_evaluation_seconds[arm] = time.perf_counter() - started
        cache_after = int(train_step._cache_size())
        hard_counts = _hard_counts_host(evaluation.hard_counts)
        if not bool(np.asarray(evaluation.update_applied)):
            raise ValueError(f"final zero-rate evaluation failed hard gates for {arm}")
        final_evaluation_compile_reuse[arm] = {
            "jit_cache_size_before": cache_before,
            "jit_cache_size_after": cache_after,
            "new_jit_executables": cache_after - cache_before,
            "optimizer_result_discarded": True,
            "learning_rate": 0.0,
        }
        final_metrics[arm] = {
            "total": float(np.asarray(evaluation.loss)),
            "one_day_anchor": float(np.asarray(evaluation.anchor)),
            "long_horizon_components": _component_record(
                LONG_HORIZON_COMPONENTS,
                evaluation.components,
            ),
            "hard_gate_counts": hard_counts,
        }

    coefficient_array = np.asarray(
        [coefficients[name] for name in LONG_HORIZON_COMPONENTS],
        dtype=np.float64,
    )
    reductions = {}
    promotion_passed = True
    for arm in arms:
        before_components = np.asarray(
            [initial_metrics[arm]["long_horizon_components"][name] for name in LONG_HORIZON_COMPONENTS],
            dtype=np.float64,
        )
        after_components = np.asarray(
            [final_metrics[arm]["long_horizon_components"][name] for name in LONG_HORIZON_COMPONENTS],
            dtype=np.float64,
        )
        before_long = float(np.sum(coefficient_array * before_components))
        after_long = float(np.sum(coefficient_array * after_components))
        one_day_reduction = 1.0 - (
            final_metrics[arm]["one_day_anchor"] / max(initial_metrics[arm]["one_day_anchor"], 1.0e-30)
        )
        long_reduction = 1.0 - after_long / max(before_long, 1.0e-30)
        reductions[arm] = {
            "one_day_loss_reduction": one_day_reduction,
            "weighted_long_horizon_loss_reduction": long_reduction,
            "initial_weighted_long_horizon_loss": before_long,
            "final_weighted_long_horizon_loss": after_long,
        }
        if requested_updates == maximum_updates:
            promotion_passed = promotion_passed and (
                one_day_reduction >= float(runner_contract.payload["gates"]["minimum_one_day_loss_reduction"])
                and long_reduction >= float(runner_contract.payload["gates"]["minimum_long_horizon_loss_reduction"])
                and all(value == 0 for value in final_metrics[arm]["hard_gate_counts"].values())
            )

    reuse_records = records[1:]
    reuse_passed = len(first_batches) == 1 and all(
        record["trace_signature"] == first_signature
        and record["teacher_replay"]["new_jit_executables"] == 0
        and record["teacher_replay"]["jit_cache_size_after"] == 1
        and all(
            arm_record["new_jit_executables"] == 0 and arm_record["jit_cache_size_after"] == 1
            for arm_record in record["arms"].values()
        )
        for record in reuse_records
    )
    passed = reuse_passed and (promotion_passed if production_fit else True)
    checkpoint_set = None
    checkpoint_failure = None
    if production_fit and passed:
        try:
            checkpoint_set = write_successor_checkpoint_set(
                args.checkpoint_dir,
                identity=execution_identity,
                parameters={arm: jax.device_get(parameters[arm]) for arm in arms},
                optimizers={arm: jax.device_get(optimizer[arm]) for arm in arms},
                completed_updates=completed_updates,
                parameter_counts=counts,
                final_metrics=final_metrics,
                loss_reductions=reductions,
                expected_arms=arms,
                required_updates=maximum_updates,
                promotion_passed=promotion_passed,
            )
        except Exception as error:
            passed = False
            checkpoint_failure = f"{type(error).__name__}: {error}"
    report = {
        "schema_version": "successor_small_fit_contract_runner_v1",
        "status": "passed" if passed else "failed",
        "run_kind": "production_bounded_fit" if production_fit else "bounded_reuse_probe",
        "checkpoint_written": checkpoint_set is not None,
        "checkpoint_set": checkpoint_set,
        "checkpoint_publication_failure": checkpoint_failure,
        "failed_stage": "checkpoint_publication" if checkpoint_failure is not None else None,
        "training_updates_per_arm": requested_updates,
        "completed_updates_per_arm": completed_updates,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "test_split_used": False,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in runtime["devices"]],
        "identity": execution_identity,
        "parameter_count_per_arm": counts,
        "matched_coefficients_shared_by_both_arms": True,
        "coefficient_preflight_recomputed": False,
        "coefficient_preflight_seconds": 0.0,
        "coefficient_contract_load_seconds": coefficient_contract_load_seconds,
        "prepared_batches_used": True,
        "prepared_manifest": str(args.prepared_manifest.resolve()),
        "prepared_artifact_count": len(prepared_manifest["entries"]),
        "prepared_artifact_total_size_bytes": int(prepared_manifest["total_size_bytes"]),
        "prepared_artifact_load_seconds": prepared_artifact_load_seconds,
        "live_retained_owner_build_count": 1,
        "coefficient_preflight": dict(frozen_coefficients.source_preflight),
        "trace_signature_count": len(first_batches),
        "dynamic_retained_input_shape_signatures": {
            signature: _tree_shape_signature(jax, adapter.day_inputs)
            for signature, (_batch, adapter) in first_batches.items()
        },
        "update_to_update_compile_reuse_passed": reuse_passed,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "loss_reductions": reductions,
        "promotion_gates_evaluated": production_fit,
        "promotion_gates_passed": promotion_passed if production_fit else None,
        "timing": {
            "total_wall_seconds": time.perf_counter() - run_started,
            "training_loop_seconds": time.perf_counter() - total_started,
            "final_evaluation_seconds": final_evaluation_seconds,
            "final_evaluation_reuses_train_step": True,
            "final_evaluation_compile_reuse": final_evaluation_compile_reuse,
        },
        "device_memory_before_training": memory_before,
        "device_memory_after_training": _device_memory_report(runtime["devices"]),
        "updates": records,
        "boundary": {
            "binding": runner_contract.payload["retained_tail"]["binding"],
            "native_forcing_records_per_day": 5,
            "reconstructs_48_forcing_records": False,
            "executes_48_step_state_scan": False,
            "landpoint_year_arrays_are_dynamic_jit_inputs": True,
        },
        "next_gate": (
            "run_7_30_day_screen_in_a_separately_authorized_lifecycle"
            if production_fit and passed
            else "repair_checkpoint_publication"
            if checkpoint_failure is not None
            else "freeze_paid_resource_envelope"
            if passed
            else "repair_contract_runner"
        ),
    }
    _atomic_json(args.output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--runner-contract", type=Path, required=True)
    parser.add_argument("--checkpoint-handoff", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--coefficient-contract", type=Path, required=True)
    parser.add_argument("--coefficient-source-report", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
