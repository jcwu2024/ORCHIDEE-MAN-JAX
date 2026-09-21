"""Bounded fresh shared-Transformer daily/T7/T30 GPU admission.

This is a technical admission only.  It reuses accepted TRAIN calibration,
starts from the declared fresh seed, differentiates only through the public
native-checkpointed objectives, and performs exactly two joint AdamW updates.
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from scripts.hpc.prepare_shared_recurrent_admission import (
    load_protocol_and_calibration,
    selected_window_metadata,
)
from scripts.hpc.run_shared_transformer_admission import (
    ROOT,
    hard_ok,
    memory_sizes,
    persist_tree,
    shared_config,
)

FORWARD_RTOL = 2.0e-5
FORWARD_ATOL = 2.0e-6


class AdmissionFailure(ValueError):
    """A bounded numerical or physical admission gate failed."""


@dataclass(frozen=True)
class Component:
    objective: Callable[..., Any]
    arguments: tuple[Any, ...]
    direction_check: bool = False


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_prerequisites(args: Any) -> dict[str, Any]:
    """Validate all reusable evidence without importing JAX or loading arrays."""
    protocol, daily, calibration, artifact_ids = load_protocol_and_calibration(
        args.protocol, args.calibration_report, args.prepared_manifest
    )
    if int(protocol.get("complete_joint_updates", -1)) != 2:
        raise ValueError("bounded recurrent admission requires exactly two joint updates")
    daily_admission = _read_json(args.daily_admission_report)
    required = (
        daily_admission.get("status") == "passed"
        and daily_admission.get("classification")
        == "one_day_two_update_technical_admission_only"
        and int(daily_admission.get("completed_updates", -1)) == 2
        and daily_admission.get("protocol") == daily
        and daily_admission.get("checkpoint_inherited") is False
        and int(daily_admission.get("parameter_count", -1))
        == int(daily["architecture"]["parameter_count"])
        and int(daily_admission.get("trainable_parameter_count", -1))
        == int(daily["architecture"]["parameter_count"])
    )
    if not required:
        raise ValueError("daily admission prerequisite is not the passed fresh full-network v3 gate")
    for name in ("calibration", "scales"):
        if daily_admission.get(name) != calibration.get(name):
            raise ValueError(f"daily admission and calibration reports have different {name}")
    daily_identity = daily_admission.get("data", {}).get("prepared_manifest_identity")
    calibration_identity = calibration.get("data", {}).get("prepared_manifest_identity")
    if daily_identity != calibration_identity:
        raise ValueError("daily admission and calibration reports use different prepared data")

    cpu = None
    if getattr(args, "cpu_preparation_report", None) is not None:
        cpu = _read_json(args.cpu_preparation_report)
        if (
            cpu.get("status") != "passed"
            or cpu.get("classification") != "CPU_preparation_only"
            or int(cpu.get("completed_updates", -1)) != 0
            or cpu.get("protocol") != protocol
            or cpu.get("daily_protocol") != daily
            or tuple(cpu.get("artifact_ids", ())) != tuple(artifact_ids)
            or cpu.get("scales") != calibration.get("scales")
            or cpu.get("data", {}).get("prepared_manifest_identity") != daily_identity
            or int(cpu.get("parameter_count", -1)) != int(daily["architecture"]["parameter_count"])
            or len(cpu.get("windows", ())) != 4
        ):
            raise ValueError("CPU recurrent preparation prerequisite differs from this admission")

    identity = {
        "protocol": protocol,
        "daily_protocol": daily,
        "artifact_ids": list(artifact_ids),
        "calibration": calibration["calibration"],
        "scales": calibration["scales"],
        "prepared_manifest_identity": daily_identity,
    }
    return {
        "protocol": protocol,
        "daily_protocol": daily,
        "calibration_report": calibration,
        "daily_admission_report": daily_admission,
        "cpu_preparation_report": cpu,
        "artifact_ids": tuple(artifact_ids),
        "identity": identity,
    }


def sum_component_gradients(jax: Any, gradients: Sequence[Any]) -> Any:
    if len(gradients) != 3:
        raise ValueError("joint admission requires exactly daily, T7 and T30 gradients")
    return jax.tree_util.tree_map(lambda x, y, z: x + y + z, *gradients)


def _tree_compatible(jax: Any, value: Any, shapes: Any, name: str) -> None:
    leaves, definition = jax.tree_util.tree_flatten(value)
    references, expected = jax.tree_util.tree_flatten(shapes)
    if definition != expected or len(leaves) != len(references):
        raise ValueError(f"recovered {name} tree differs from the current architecture")
    for leaf, reference in zip(leaves, references, strict=True):
        if np.shape(leaf) != reference.shape or np.asarray(leaf).dtype != reference.dtype:
            raise ValueError(f"recovered {name} array shape/dtype differs")


def _load_pickle(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_resume(args: Any, evidence_identity: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    path = Path(args.resume_report).resolve()
    if path == Path(args.output).resolve():
        raise ValueError("recovery must preserve the original report")
    prior = _read_json(path)
    if prior.get("evidence_identity") != evidence_identity:
        raise ValueError("recovery evidence or sample identity changed")
    if prior.get("reason") or prior.get("status") == "rejected":
        raise ValueError("recovery cannot bypass a rejected admission gate")
    completed = int(prior.get("completed_updates", -1))
    if completed not in (0, 1, 2):
        raise ValueError("recovery report has an invalid completed-update count")
    payload = None
    artifact = prior.get("gradient_artifact")
    if artifact and Path(artifact).is_file():
        candidate = _load_pickle(Path(artifact))
        if candidate.get("evidence_identity") == evidence_identity:
            payload = candidate
    return prior, payload


def _forward(
    jax: Any,
    component: Component,
    cache: dict[int, Callable[..., Any]] | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    key = id(component.objective)
    if cache is None:
        function = jax.jit(component.objective)
    else:
        function = cache.get(key)
        if function is None:
            function = jax.jit(component.objective)
            cache[key] = function
    result = function(*component.arguments)
    return jax.block_until_ready(result)


def _finite_forward(value: Any, auxiliary: Mapping[str, Any]) -> bool:
    return bool(np.isfinite(float(value)) and hard_ok(auxiliary))


def _gradient_groups(gradient: Any) -> Mapping[str, Any]:
    return gradient if isinstance(gradient, Mapping) else {"all": gradient}


def execute_bounded_joint(
    *,
    jax: Any,
    parameters: Any,
    optimizer: Any,
    update: Callable[..., Any],
    batches: Sequence[Mapping[str, Component]],
    replay: Mapping[str, Component],
    protocol: Mapping[str, Any],
    evidence_identity: Mapping[str, Any],
    output: Path,
    emit: Callable[..., None],
    numeric_report: Callable[[Any, Any], Mapping[str, Any]],
    start_update: int = 0,
    recovered_gradients: Mapping[str, Any] | None = None,
    previous_updates: Sequence[Mapping[str, Any]] = (),
    previous_initial: Mapping[str, Any] | None = None,
) -> tuple[Any, Any, list[dict[str, Any]], Mapping[str, Any]]:
    """Run the small fixed admission orchestration around supplied objectives."""
    if len(batches) != 2 or any(tuple(batch) != ("daily", "T7", "T30") for batch in batches):
        raise ValueError("joint admission batches must be two ordered daily/T7/T30 triples")
    limit = int(protocol["gradient_memory_limit_gib"]) * 2**30
    gradient_path = output.with_suffix(".gradients.pkl")
    checkpoint_path = output.with_suffix(".checkpoint.pkl")
    updates: list[dict[str, Any]] = [dict(row) for row in previous_updates]
    if len(updates) != start_update:
        raise ValueError("recovery update history differs from its checkpoint")

    initial = dict(previous_initial or {})
    forward_functions: dict[int, Callable[..., Any]] = {}
    gradient_functions: dict[int, Callable[..., Any]] = {}
    compiled_gradients: dict[str, Any] = {}
    if start_update == 0:
        emit("initial_recurrent_forwards")
        for label, component in replay.items():
            if ".T" not in label:
                continue
            value, auxiliary = _forward(jax, component, forward_functions)
            if not _finite_forward(value, auxiliary):
                emit("forward_rejected", component=label, value=value, hard=auxiliary.get("hard", {}))
                raise AdmissionFailure("initial_recurrent_forward_physical")
            initial[label] = (value, auxiliary)
            emit("initial_recurrent_forward_completed", initial_recurrent=initial)
        for label, component in replay.items():
            if ".T" in label:
                continue
            value, auxiliary = _forward(jax, component, forward_functions)
            if not _finite_forward(value, auxiliary):
                emit("forward_rejected", component=label, value=value, hard=auxiliary.get("hard", {}))
                raise AdmissionFailure("initial_daily_forward_physical")
            initial[label] = (value, auxiliary)
        emit("initial_forwards_completed", initial=initial)

    seed_payload = recovered_gradients or {}
    for update_index in range(start_update, int(protocol["complete_joint_updates"])):
        batch = batches[update_index]
        saved_components = {}
        if int(seed_payload.get("base_completed_updates", -1)) == update_index:
            saved_components = dict(seed_payload.get("components", {}))
        component_results: dict[str, Any] = {}
        gradients = []
        values = []
        before_components = {}
        for name, declared_component in batch.items():
            component = Component(
                declared_component.objective,
                (parameters, *declared_component.arguments[1:]),
                declared_component.direction_check,
            )
            matching_value, matching_auxiliary = _forward(jax, component, forward_functions)
            if not _finite_forward(matching_value, matching_auxiliary):
                emit(
                    "forward_rejected",
                    component=name,
                    update=update_index + 1,
                    value=matching_value,
                    hard=matching_auxiliary.get("hard", {}),
                )
                raise AdmissionFailure(f"{name}_forward_physical")
            before_components[name] = (matching_value, matching_auxiliary)
            saved = saved_components.get(name)
            if saved is None:
                emit("compile_component_gradient", component=name, update=update_index + 1)
                gradient_function = gradient_functions.setdefault(
                    id(component.objective),
                    jax.jit(jax.value_and_grad(component.objective, has_aux=True)),
                )
                compile_started = time.perf_counter()
                compiled = compiled_gradients.get(name)
                compiled_reused = compiled is not None
                if compiled is None:
                    compiled = gradient_function.lower(*component.arguments).compile()
                    compiled_gradients[name] = compiled
                compiled_memory = memory_sizes(compiled)
                emit(
                    "component_gradient_compiled",
                    component=name,
                    update=update_index + 1,
                    gradient_compile_or_cache_seconds=time.perf_counter() - compile_started,
                    compiled_gradient_memory=compiled_memory,
                    gradient_compiled_reused=compiled_reused,
                )
                if compiled_memory["estimated_live_bytes"] > limit:
                    raise AdmissionFailure(f"{name}_compiled_gradient_memory")
                (value, auxiliary), gradient = compiled(*component.arguments)
                jax.block_until_ready(gradient)
                if not np.isclose(
                    float(value), float(matching_value), rtol=FORWARD_RTOL, atol=FORWARD_ATOL
                ):
                    raise AdmissionFailure(f"{name}_gradient_primal_mismatch")
                report = dict(numeric_report(jax, gradient))
                if not report["all_finite"] or report["l2_norm"] <= 0.0 or not hard_ok(auxiliary):
                    raise AdmissionFailure(f"{name}_full_parameter_gradient")
                saved = {
                    "value": float(value),
                    "gradient": gradient,
                    "gradient_report": report,
                    "compiled_memory": compiled_memory,
                    "device_memory": jax.devices()[0].memory_stats(),
                }
                component_results[name] = saved
                persist_tree(
                    gradient_path,
                    jax,
                    {
                        "schema_version": "shared_recurrent_component_gradients_v1",
                        "evidence_identity": evidence_identity,
                        "base_completed_updates": update_index,
                        "parameters": parameters,
                        "components": component_results,
                    },
                )
                emit(
                    "component_gradient_preserved",
                    gradient_artifact=str(gradient_path),
                    component=name,
                    update=update_index + 1,
                    component_gradients={key: {k: v for k, v in row.items() if k != "gradient"}
                                         for key, row in component_results.items()},
                )
            else:
                gradient = jax.device_put(saved["gradient"])
                if not np.isclose(
                    float(saved["value"]), float(matching_value), rtol=FORWARD_RTOL, atol=FORWARD_ATOL
                ):
                    raise ValueError(f"recovered {name} gradient belongs to different parameters/data")
                component_results[name] = {**saved, "gradient": gradient}

            gradient = component_results[name]["gradient"]
            if component.direction_check and update_index == 0:
                direction_record = component_results[name].get("direction")
                if direction_record is None:
                    norm = float(component_results[name]["gradient_report"]["l2_norm"])
                    direction = jax.tree_util.tree_map(lambda value: value / norm, gradient)

                    def tangent_function(*arguments):
                        current, tangent, *rest = arguments
                        return jax.jvp(
                            lambda candidate: component.objective(candidate, *rest)[0],
                            (current,),
                            (tangent,),
                        )[1]

                    emit("compile_component_direction", component=name, update=update_index + 1)
                    direction_started = time.perf_counter()
                    tangent = jax.jit(tangent_function)
                    tangent_compiled = tangent.lower(parameters, direction, *component.arguments[1:]).compile()
                    tangent_memory = memory_sizes(tangent_compiled)
                    emit(
                        "component_direction_compiled",
                        component=name,
                        update=update_index + 1,
                        direction_compile_seconds=time.perf_counter() - direction_started,
                        compiled_direction_memory=tangent_memory,
                    )
                    if tangent_memory["estimated_live_bytes"] > limit:
                        raise AdmissionFailure(f"{name}_compiled_direction_memory")
                    jvp_value = float(tangent_compiled(parameters, direction, *component.arguments[1:]))
                    reverse = float(sum(
                        np.vdot(np.asarray(g, dtype=np.float64), np.asarray(d, dtype=np.float64))
                        for g, d in zip(
                            jax.tree_util.tree_leaves(gradient),
                            jax.tree_util.tree_leaves(direction),
                            strict=True,
                        )
                    ))
                    passed = bool(np.isfinite(jvp_value) and np.isclose(
                        jvp_value,
                        reverse,
                        rtol=float(protocol["direction_rtol"]),
                        atol=float(protocol["direction_atol"]),
                    ))
                    direction_record = {
                        "jvp": jvp_value,
                        "reverse_projection": reverse,
                        "direction": "normalized_full_parameter_gradient",
                        "compiled_memory": tangent_memory,
                        "passed": passed,
                    }
                    component_results[name]["direction"] = direction_record
                    persist_tree(
                        gradient_path,
                        jax,
                        {
                            "schema_version": "shared_recurrent_component_gradients_v1",
                            "evidence_identity": evidence_identity,
                            "base_completed_updates": update_index,
                            "parameters": parameters,
                            "components": component_results,
                        },
                    )
                    del direction, tangent, tangent_compiled
                if not direction_record["passed"]:
                    raise AdmissionFailure(f"{name}_jvp_reverse_agreement")
                emit("component_direction_completed", component=name, derivative=direction_record)
            gradients.append(gradient)
            values.append(float(component_results[name]["value"]))

        combined = sum_component_gradients(jax, gradients)
        combined_report = dict(numeric_report(jax, combined))
        group_reports = {
            name: dict(numeric_report(jax, group))
            for name, group in _gradient_groups(combined).items()
        }
        if (
            not combined_report["all_finite"]
            or combined_report["l2_norm"] <= 0.0
            or not all(row["all_finite"] and row["l2_norm"] > 0.0 for row in group_reports.values())
        ):
            raise AdmissionFailure("combined_gradient_group_participation")
        before = float(sum(values))
        parameters, optimizer, norm, clip = update(parameters, combined, optimizer)
        jax.block_until_ready(parameters)
        after_components = {}
        for name, component in batch.items():
            after_value, after_auxiliary = _forward(
                jax,
                Component(component.objective, (parameters, *component.arguments[1:])),
                forward_functions,
            )
            if not _finite_forward(after_value, after_auxiliary):
                emit(
                    "forward_rejected",
                    component=name,
                    update=update_index + 1,
                    value=after_value,
                    hard=after_auxiliary.get("hard", {}),
                )
                raise AdmissionFailure("updated_forward_physical")
            after_components[name] = (after_value, after_auxiliary)
        row = {
            "update": update_index + 1,
            "loss_before": before,
            "loss_after": float(sum(float(value) for value, _aux in after_components.values())),
            "component_loss_before": dict(zip(batch, values, strict=True)),
            "gradient": combined_report,
            "gradient_groups": group_reports,
            "component_gradients": {
                name: {key: value for key, value in result.items() if key != "gradient"}
                for name, result in component_results.items()
            },
            "gradient_norm": norm,
            "clip_scale": clip,
            "forward_only_science_metrics_before": before_components,
            "forward_only_science_metrics_after": after_components,
        }
        updates.append(row)
        persist_tree(
            checkpoint_path,
            jax,
            {
                "schema_version": "shared_recurrent_admission_checkpoint_v1",
                "evidence_identity": evidence_identity,
                "parameters": parameters,
                "optimizer": optimizer,
                "completed_updates": update_index + 1,
            },
        )
        emit(
            "optimizer_update_completed",
            completed_updates=update_index + 1,
            updates=updates,
            checkpoint=str(checkpoint_path),
        )
        seed_payload = {}
        del gradients, combined

    restored = _load_pickle(checkpoint_path)
    parameters = jax.device_put(restored["parameters"])
    restored_optimizer = jax.device_put(restored["optimizer"])
    optimizer_equal = all(
        np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(
            jax.tree_util.tree_leaves(optimizer),
            jax.tree_util.tree_leaves(restored_optimizer),
            strict=True,
        )
    )
    final = {}
    for label, component in replay.items():
        value, auxiliary = _forward(
            jax,
            Component(component.objective, (parameters, *component.arguments[1:])),
            forward_functions,
        )
        if not _finite_forward(value, auxiliary):
            emit("forward_rejected", component=label, value=value, hard=auxiliary.get("hard", {}))
            raise AdmissionFailure("final_cross_sample_physical")
        final[label] = (value, auxiliary)
    if not optimizer_equal:
        raise AdmissionFailure("checkpoint_optimizer_roundtrip")
    return parameters, restored_optimizer, updates, {
        "checkpoint": str(checkpoint_path),
        "checkpoint_optimizer_roundtrip_equal": optimizer_equal,
        "initial": initial,
        "final": final,
    }


def run(args: Any) -> Mapping[str, Any]:
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json

    started = time.perf_counter()
    report: dict[str, Any] = {
        "status": "running",
        "classification": "shared_daily_T7_T30_technical_admission_only",
        "completed_updates": 0,
        "scientific_acceptance": False,
        "checkpoint_inherited": False,
        "source_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }

    def emit(stage: str, **values: Any) -> None:
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **values)
        _atomic_json(args.output, report)
        print(f"shared_recurrent_admission stage={stage} seconds={report['elapsed_seconds']:.1f}", flush=True)

    try:
        emit("validate_prerequisites")
        evidence = load_prerequisites(args)
        report.update(
            protocol=evidence["protocol"],
            daily_protocol=evidence["daily_protocol"],
            evidence_identity=evidence["identity"],
            artifact_ids=evidence["artifact_ids"],
            calibration_source=str(args.calibration_report),
            daily_admission_source=str(args.daily_admission_report),
            cpu_preparation_source=(str(args.cpu_preparation_report)
                                    if args.cpu_preparation_report else None),
            scales=evidence["calibration_report"]["scales"],
        )
        if args.validate_only:
            emit(
                "completed",
                status="passed",
                classification="shared_recurrent_prerequisites_validation_only",
                model_loaded=False,
                numerical_gradient_checked=False,
            )
            return report

        prior = gradient_payload = None
        if args.resume_report:
            prior, gradient_payload = load_resume(args, evidence["identity"])
            report.update(
                admission_resumed=True,
                resumed_report=str(args.resume_report),
                resumed_source_git_head=prior.get("source_git_head"),
            )

        from scripts.hpc.shared_transformer_runtime import load_shared_runtime

        emit("load_existing_train_artifacts")
        runtime, parent, template, artifacts, metadata = load_shared_runtime(
            args, evidence["artifact_ids"]
        )
        jax = runtime["jax"]
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
            prepare_shared_window,
        )
        from research.daily_coarse_graining.daily_operator.shared_training import (
            derive_shared_state_change_scales,
        )
        from research.daily_coarse_graining.daily_operator.target_oriented_training import (
            initialize_target_oriented_adamw,
        )
        from scripts.hpc.run_successor_fast_owner_gpu_probe import _tree_numeric_report

        daily = evidence["daily_protocol"]
        coordinates = build_shared_persistent_coordinates(
            parent.learned_schema,
            parent.contract,
            runtime["base_statistics"].arrays["state_delta"],
        )
        head = build_shared_process_head(
            parent,
            template,
            config=shared_config(daily["architecture"]),
            thermal_transform=daily["architecture"]["thermal_transform"],
            persistent_coordinates=coordinates,
            source_bare_cimean=True,
        )
        state_scales = derive_shared_state_change_scales(runtime, head)
        daily_objective = make_shared_joint_daily_objective(
            runtime, head, template, state_scales, evidence["calibration_report"]["scales"]
        )
        _rollout, recurrent_objective = make_shared_recurrent_objective(
            head, template, state_scales, evidence["calibration_report"]["scales"]
        )
        samples = [
            jax.tree_util.tree_map(lambda value: value[0], artifact.batch.one_day)
            for artifact in artifacts
        ]
        cases = []
        windows = []
        for artifact in artifacts:
            artifact_cases = {}
            for horizon, component_name in ((7, "seven_day"), (30, "thirty_day")):
                artifact_cases[horizon] = prepare_shared_window(
                    head,
                    getattr(artifact.batch, component_name),
                    window_index=0,
                    horizon=horizon,
                )
                windows.append(selected_window_metadata(artifact, component_name, horizon))
            cases.append(artifact_cases)
        if evidence["cpu_preparation_report"] is not None:
            expected = [
                {key: row[key] for key in ("artifact_id", "component", "horizon", "window_index",
                                            "landpoint_id", "year", "start_row")}
                for row in evidence["cpu_preparation_report"]["windows"]
            ]
            if windows != expected:
                raise ValueError("actual recurrent windows differ from passed CPU preparation")
        emit("actual_windows_prepared", data=metadata, windows=windows)

        def initializer():
            return head.initialize(runtime["probe_assembled"], seed=daily["seed"])

        shapes = jax.eval_shape(initializer)
        counts = {
            name: sum(int(np.prod(value.shape)) for value in jax.tree_util.tree_leaves(group))
            for name, group in shapes.items()
        }
        parameter_count = sum(counts.values())
        if parameter_count != int(daily["architecture"]["parameter_count"]):
            raise ValueError("shared recurrent parameter tree differs from the daily protocol")
        report.update(
            parameter_count=parameter_count,
            trainable_parameter_count=parameter_count,
            parameter_groups=counts,
            initialization_seed=daily["seed"],
            initialization="fresh_no_checkpoint_weights",
            ad_boundary="public_native_JAX_checkpointed_daily_scan_only",
        )

        start_update = 0
        previous_updates = ()
        if prior is None:
            emit("initialize_fresh_parameters")
            parameters = jax.jit(initializer)()
            optimizer = initialize_target_oriented_adamw(parameters)
        elif int(prior["completed_updates"]) > 0:
            checkpoint = _load_pickle(Path(prior["checkpoint"]))
            if checkpoint.get("evidence_identity") != evidence["identity"]:
                raise ValueError("recovery checkpoint evidence identity changed")
            if int(checkpoint.get("completed_updates", -1)) != int(prior["completed_updates"]):
                raise ValueError("recovery checkpoint and report completed-update counts differ")
            start_update = int(checkpoint["completed_updates"])
            parameters = checkpoint["parameters"]
            optimizer = checkpoint["optimizer"]
            _tree_compatible(jax, parameters, shapes, "parameters")
            parameters, optimizer = jax.device_put((parameters, optimizer))
            previous_updates = tuple(prior.get("updates", ()))
            report.update(
                completed_updates=start_update,
                recovery_checkpoint_reused=True,
                recovery_checkpoint_source=str(prior["checkpoint"]),
            )
            persist_tree(
                args.output.with_suffix(".checkpoint.pkl"),
                jax,
                {
                    "schema_version": "shared_recurrent_admission_checkpoint_v1",
                    "evidence_identity": evidence["identity"],
                    "parameters": parameters,
                    "optimizer": optimizer,
                    "completed_updates": start_update,
                },
            )
        elif gradient_payload is not None:
            parameters = gradient_payload["parameters"]
            _tree_compatible(jax, parameters, shapes, "parameters")
            parameters = jax.device_put(parameters)
            optimizer = initialize_target_oriented_adamw(parameters)
            report["partial_component_gradients_reused"] = sorted(
                gradient_payload.get("components", {})
            )
        else:
            raise ValueError("zero-update recovery report has no compatible saved component gradient")

        batches = []
        replay = {}
        for index, (sample, artifact_cases) in enumerate(zip(samples, cases, strict=True)):
            identifier = int(artifacts[index].schedule_entry["update"])
            batch = {
                "daily": Component(daily_objective, (parameters, sample)),
                "T7": Component(recurrent_objective, (parameters, *artifact_cases[7]), True),
                "T30": Component(recurrent_objective, (parameters, *artifact_cases[30]), True),
            }
            batches.append(batch)
            replay[f"artifact{identifier}.daily"] = batch["daily"]
            replay[f"artifact{identifier}.T7"] = batch["T7"]
            replay[f"artifact{identifier}.T30"] = batch["T30"]

        _parameters, _optimizer, updates, final = execute_bounded_joint(
            jax=jax,
            parameters=parameters,
            optimizer=optimizer,
            update=make_update(daily),
            batches=batches,
            replay=replay,
            protocol=evidence["protocol"],
            evidence_identity=evidence["identity"],
            output=args.output,
            emit=emit,
            numeric_report=_tree_numeric_report,
            start_update=start_update,
            recovered_gradients=gradient_payload,
            previous_updates=previous_updates,
            previous_initial=(prior.get("initial") if prior is not None else None),
        )
        emit(
            "completed",
            status="passed",
            classification="shared_daily_T7_T30_two_update_technical_admission_only",
            completed_updates=2,
            updates=updates,
            **final,
            device_memory=runtime["devices"][0].memory_stats(),
            science_metrics_scope="forward_diagnostics_only_not_accuracy_acceptance",
            next_gate="bounded_joint_learning_then_independent_scientific_panels",
        )
        return report
    except AdmissionFailure as error:
        emit("stopped", status="rejected", reason=str(error))
        return report
    except Exception as error:
        emit("exception", status="error", error_type=type(error).__name__, error=str(error))
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in (
        "dataset-manifest",
        "acceptance-report",
        "base-statistics",
        "process-statistics",
        "fast-statistics",
        "prepared-manifest",
        "calibration-report",
        "daily-admission-report",
        "output",
    ):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--cpu-preparation-report", type=Path)
    result.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "manifests/coarse_graining/shared_recurrent_admission_v1.json",
    )
    result.add_argument("--resume-report", type=Path)
    result.add_argument(
        "--validate-only",
        action="store_true",
        help="validate protocols and evidence identities without loading JAX/model/data arrays",
    )
    return result


if __name__ == "__main__":
    outcome = run(parser().parse_args())
    raise SystemExit(0 if outcome["status"] == "passed" else 1)
