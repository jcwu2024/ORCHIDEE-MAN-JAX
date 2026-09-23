"""Run bounded fresh shared daily/T7/T30 learning methods."""
from __future__ import annotations

import argparse
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from scripts.hpc.run_shared_recurrent_admission import (
    ROOT,
    memory_sizes,
    persist_tree,
    shared_config,
)


class LearningFailure(ValueError):
    """A declared numerical, physical, or resource gate failed."""

    def __init__(self, reason: str, **details: Any):
        super().__init__(reason)
        self.reason = reason
        self.details = details


def checkpoint_path(output: Path, update: int | None = None) -> Path:
    if update is None:
        return output.with_suffix(".checkpoint.pkl")
    return output.with_name(f"{output.stem}.update-{update:06d}.checkpoint.pkl")


def _tree_compatible(jax: Any, value: Any, shapes: Any, name: str) -> None:
    leaves, definition = jax.tree_util.tree_flatten(value)
    references, expected = jax.tree_util.tree_flatten(shapes)
    if definition != expected or len(leaves) != len(references):
        raise ValueError(f"checkpoint {name} tree differs from the declared method")
    for leaf, reference in zip(leaves, references, strict=True):
        if np.shape(leaf) != reference.shape or np.asarray(leaf).dtype != reference.dtype:
            raise ValueError(f"checkpoint {name} shape/dtype differs from the declared method")


def save_learning_checkpoint(
    path: Path,
    *,
    jax: Any,
    parameters: Any,
    optimizer: Any,
    completed_updates: int,
    method: Mapping[str, Any],
    updates: Sequence[Mapping[str, Any]],
    monitors: Sequence[Mapping[str, Any]],
) -> None:
    persist_tree(
        path,
        jax,
        {
            "schema_version": "shared_learning_checkpoint_v1",
            "method": method,
            "parameters": parameters,
            "optimizer": optimizer,
            "completed_updates": int(completed_updates),
            "cursor": {"next_artifact_index": int(completed_updates)},
            "updates": list(updates),
            "monitors": list(monitors),
        },
    )


def load_learning_checkpoint(
    path: Path,
    *,
    expected_method: Mapping[str, Any],
    jax: Any | None = None,
    parameter_shapes: Any | None = None,
    optimizer_shapes: Any | None = None,
) -> Mapping[str, Any]:
    with Path(path).open("rb") as handle:
        saved = pickle.load(handle)
    completed = int(saved.get("completed_updates", -1))
    if (
        saved.get("schema_version") != "shared_learning_checkpoint_v1"
        or saved.get("method") != expected_method
        or saved.get("cursor") != {"next_artifact_index": completed}
        or not 0 <= completed <= int(expected_method["configuration"]["max_updates"])
    ):
        raise ValueError("checkpoint method, data, or cursor differs from this learning run")
    optimizer_step = int(np.asarray(saved["optimizer"].step))
    if optimizer_step != completed:
        raise ValueError("checkpoint AdamW step and learning cursor differ")
    if len(saved.get("updates", ())) != completed:
        raise ValueError("checkpoint compact update history and cursor differ")
    if jax is not None:
        _tree_compatible(jax, saved["parameters"], parameter_shapes, "parameters")
        _tree_compatible(jax, saved["optimizer"], optimizer_shapes, "optimizer")
    return saved


def make_tree_operations(jax: Any, jnp: Any, daily_count: int) -> Mapping[str, Callable[..., Any]]:
    """Small cached device operations; no process loss is reconstructed here."""

    add = jax.jit(lambda left, right: jax.tree_util.tree_map(lambda x, y: x + y, left, right))
    zero = jax.jit(lambda tree: jax.tree_util.tree_map(jnp.zeros_like, tree))
    daily_mean = jax.jit(
        lambda tree: jax.tree_util.tree_map(lambda value: value / daily_count, tree)
    )
    combine = jax.jit(
        lambda daily, seven, thirty: jax.tree_util.tree_map(
            lambda d, s, t: d / daily_count + s + t, daily, seven, thirty
        )
    )
    combine_daily_mean = jax.jit(
        lambda daily, seven, thirty: jax.tree_util.tree_map(
            lambda d, s, t: d + s + t, daily, seven, thirty
        )
    )

    def finite(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in leaves]))

    def norm(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        squared = sum(
            (jnp.sum(jnp.square(jnp.asarray(value, dtype=jnp.float32))) for value in leaves),
            start=jnp.asarray(0.0, dtype=jnp.float32),
        )
        return jnp.sqrt(squared)

    return {
        "daily_count": daily_count,
        "add": add,
        "zero": zero,
        "daily_mean": daily_mean,
        "combine": combine,
        "combine_daily_mean": combine_daily_mean,
        "finite": jax.jit(finite),
        "norm": jax.jit(norm),
    }


def _hard_pass(auxiliary: Mapping[str, Any]) -> bool:
    return all(int(np.asarray(value)) == 0 for value in auxiliary["hard"].values())


def _check_gradient_component(
    *,
    name: str,
    value: Any,
    auxiliary: Mapping[str, Any],
    gradient: Any,
    finite: Callable[[Any], Any],
    row: int | None = None,
) -> None:
    finite_value = bool(np.isfinite(float(value)))
    finite_gradient = bool(np.asarray(finite(gradient)))
    if not finite_value or not finite_gradient or not _hard_pass(auxiliary):
        raise LearningFailure(
            f"{name}_numerical_or_physical",
            component=name,
            row=row,
            value=float(value),
            finite_gradient=finite_gradient,
            hard={key: int(np.asarray(count)) for key, count in auxiliary["hard"].items()},
        )


def _mean_terms(rows: Sequence[Mapping[str, float]]) -> Mapping[str, float]:
    keys = sorted(set.intersection(*(set(row) for row in rows))) if rows else []
    return {key: float(np.mean([row[key] for row in rows], dtype=np.float64)) for key in keys}


def execute_learning_update(
    *,
    jax: Any,
    parameters: Any,
    optimizer: Any,
    daily_samples: Any,
    seven_case: Sequence[Any],
    thirty_case: Sequence[Any],
    evaluators: Mapping[str, Callable[..., Any]],
    operations: Mapping[str, Callable[..., Any]],
    update: Callable[..., Any],
    daily_accumulator: Callable[[Any, Any], Any] | None = None,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Combine the declared daily mean with T7/T30, then update once."""
    if daily_accumulator is None and len(daily_samples) != int(operations["daily_count"]):
        raise ValueError("learning update daily row count differs from the declared mean")
    began = time.perf_counter()
    daily_values = []
    daily_terms = []
    daily_started = time.perf_counter()
    if daily_accumulator is None:
        daily_total = operations["zero"](parameters)
        for row_index, sample in enumerate(daily_samples):
            (value, auxiliary), gradient = evaluators["daily"](parameters, sample)
            _check_gradient_component(
                name="daily",
                value=value,
                auxiliary=auxiliary,
                gradient=gradient,
                finite=operations["finite"],
                row=row_index,
            )
            daily_total = operations["add"](daily_total, gradient)
            daily_values.append(float(value))
            daily_terms.append(
                {
                    key: float(auxiliary[key])
                    for key in ("process_day", "flux", "stock", "joint_daily_contribution")
                    if key in auxiliary
                }
            )
    else:
        from scripts.hpc.shared_daily_microbatch import MicrobatchFailure

        try:
            (value, auxiliary), daily_total = daily_accumulator(parameters, daily_samples)
        except MicrobatchFailure as error:
            raise LearningFailure(
                error.reason,
                component="daily",
                microbatch_index=error.microbatch_index,
                **error.details,
            ) from error
        _check_gradient_component(
            name="daily",
            value=value,
            auxiliary=auxiliary,
            gradient=daily_total,
            finite=operations["finite"],
        )
        daily_values.append(float(value))
        daily_terms.append(
            {
                key: float(auxiliary[key])
                for key in ("process_day", "flux", "stock", "joint_daily_contribution")
                if key in auxiliary
            }
        )
    daily_seconds = time.perf_counter() - daily_started

    recurrent_results = {}
    recurrent_gradients = {}
    recurrent_seconds = {}
    for name, case in (("T7", seven_case), ("T30", thirty_case)):
        started = time.perf_counter()
        (value, auxiliary), gradient = evaluators[name](parameters, *case)
        _check_gradient_component(
            name=name,
            value=value,
            auxiliary=auxiliary,
            gradient=gradient,
            finite=operations["finite"],
        )
        recurrent_results[name] = {
            "loss": float(value),
            **{
                key: float(auxiliary[key])
                for key in ("flux", "stock", "living_stock", "nonliving_stock")
                if key in auxiliary
            },
        }
        recurrent_gradients[name] = gradient
        recurrent_seconds[name] = time.perf_counter() - started

    combine = (
        operations["combine"]
        if daily_accumulator is None else operations["combine_daily_mean"]
    )
    combined = combine(
        daily_total, recurrent_gradients["T7"], recurrent_gradients["T30"]
    )
    if not bool(np.asarray(operations["finite"](combined))):
        raise LearningFailure("combined_gradient_nonfinite")
    component_norms = {
        "daily_mean": float(
            operations["norm"](
                operations["daily_mean"](daily_total)
                if daily_accumulator is None else daily_total
            )
        ),
        "T7": float(operations["norm"](recurrent_gradients["T7"])),
        "T30": float(operations["norm"](recurrent_gradients["T30"])),
        "combined": float(operations["norm"](combined)),
    }
    update_started = time.perf_counter()
    parameters, optimizer, gradient_norm, clip_scale = update(parameters, combined, optimizer)
    jax.block_until_ready(parameters)
    if (
        not bool(np.asarray(operations["finite"](parameters)))
        or not bool(np.asarray(operations["finite"](optimizer)))
        or not np.isfinite(float(gradient_norm))
        or not np.isfinite(float(clip_scale))
    ):
        raise LearningFailure("optimizer_update_nonfinite")
    update_seconds = time.perf_counter() - update_started
    return parameters, optimizer, {
        "loss": {
            "daily_mean": float(np.mean(daily_values, dtype=np.float64)),
            "daily_terms_mean": _mean_terms(daily_terms),
            "T7": recurrent_results["T7"],
            "T30": recurrent_results["T30"],
            "joint": float(np.mean(daily_values, dtype=np.float64))
            + recurrent_results["T7"]["loss"]
            + recurrent_results["T30"]["loss"],
        },
        "gradient_norms": component_norms,
        "optimizer_gradient_norm": gradient_norm,
        "clip_scale": clip_scale,
        "timing_seconds": {
            (
                "daily_32_sequential"
                if daily_accumulator is None else "daily_32_microbatch_4"
            ): daily_seconds,
            "T7": recurrent_seconds["T7"],
            "T30": recurrent_seconds["T30"],
            "optimizer": update_seconds,
            "total": time.perf_counter() - began,
        },
    }


def compile_gradient_evaluators(
    *,
    jax: Any,
    parameters: Any,
    daily_objective: Callable[..., Any],
    daily_sample: Any,
    recurrent_objective: Callable[..., Any],
    seven_case: Sequence[Any],
    thirty_case: Sequence[Any],
    memory_limit_bytes: int,
    emit: Callable[..., None],
    daily_microbatch_size: int | None = None,
) -> tuple[Mapping[str, Callable[..., Any]], Mapping[str, Any]]:
    """Compile one stable reverse executable per admitted component shape."""
    daily_backward = jax.jit(jax.value_and_grad(daily_objective, has_aux=True))
    recurrent_backward = jax.jit(jax.value_and_grad(recurrent_objective, has_aux=True))
    definitions = (
        ("daily", daily_backward, (parameters, daily_sample)),
        ("T7", recurrent_backward, (parameters, *seven_case)),
        ("T30", recurrent_backward, (parameters, *thirty_case)),
    )
    evaluators = {}
    metadata = {}
    for name, function, arguments in definitions:
        emit("compile_gradient_graph", component=name)
        started = time.perf_counter()
        if name == "daily" and daily_microbatch_size is not None:
            import jax.numpy as jnp

            from scripts.hpc.shared_daily_microbatch import (
                compile_microbatch_evaluator,
                make_microbatch_objective,
            )

            if daily_microbatch_size != 4:
                raise ValueError("shared learning supports only four-row daily microbatches")
            microbatch_objective = make_microbatch_objective(jax, jnp, daily_objective)
            compiled = compile_microbatch_evaluator(
                jax=jax,
                parameters=parameters,
                microbatch_objective=microbatch_objective,
                microbatch_example=daily_sample,
            )
        else:
            compiled = function.lower(*arguments).compile()
        memory = memory_sizes(compiled)
        metadata[name] = {
            "compile_seconds": time.perf_counter() - started,
            "memory": memory,
            "cache_policy": "single_stable_shape_executable_reused_all_updates",
        }
        emit("gradient_graph_compiled", component=name, compiled_gradient_graphs=metadata)
        if memory["estimated_live_bytes"] > memory_limit_bytes:
            raise LearningFailure(
                "compiled_gradient_memory", component=name, compiled_memory=memory
            )
        evaluators[name] = compiled
    return evaluators, metadata


def _method_identity(loaded: Mapping[str, Any]) -> Mapping[str, Any]:
    admission = loaded["admission"]
    return {
        "schema_version": (
            "shared_learning_method_identity_v2"
            if loaded["config"]["schema_version"] == "shared_learning_v2"
            else "shared_learning_method_identity_v1"
        ),
        "configuration": loaded["config"],
        "daily_protocol": loaded["daily_protocol"],
        "joint_protocol": admission["protocol"],
        "training_plan": loaded["plan"],
        "scales": admission["scales"],
        "validation_panel_sha256": loaded["validation_panel"]["panel_sha256"],
        "initialization": "fresh_declared_seed_no_exploratory_weights",
    }


def run(args: Any) -> Mapping[str, Any]:
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
    from scripts.hpc.shared_learning_plan import load_learning_configuration

    started = time.perf_counter()
    report: dict[str, Any] = {
        "status": "running",
        "classification": "bounded_shared_joint_learning_not_scientific_acceptance",
        "completed_updates": 0,
        "updates": [],
        "monitors": [],
        "checkpoints": [],
        "checkpoint_inherited": False,
        "scientific_acceptance": False,
        "source_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }

    def emit(stage: str, **values: Any) -> None:
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **values)
        _atomic_json(args.output, report)
        print(
            f"shared_learning stage={stage} updates={report['completed_updates']} "
            f"seconds={report['elapsed_seconds']:.1f}",
            flush=True,
        )

    try:
        emit("load_learning_configuration")
        loaded = load_learning_configuration(
            args.protocol,
            args.joint_admission_report,
            args.prepared_manifest,
            args.validation_manifest,
        )
        method = _method_identity(loaded)
        config = loaded["config"]
        microbatch_size = (
            int(config["daily_microbatch_size"])
            if config["schema_version"] == "shared_learning_v2"
            else None
        )
        plan = loaded["plan"]
        daily_protocol = loaded["daily_protocol"]
        report.update(
            method=method,
            training_plan=plan,
            validation_panel=loaded["validation_panel"],
            joint_admission_source=str(args.joint_admission_report),
            initialization_seed=int(daily_protocol["seed"]),
            optimizer=daily_protocol["optimizer"],
            parameter_count=int(daily_protocol["architecture"]["parameter_count"]),
            trainable_parameter_count=int(daily_protocol["architecture"]["parameter_count"]),
            gradient_composition=(
                "mean_32_joint_daily_contributions_plus_T7_plus_T30_no_extra_averaging"
            ),
        )
        if args.prepare_only:
            emit(
                "completed",
                status="passed",
                classification="shared_learning_metadata_preparation_only",
                model_loaded=False,
                numerical_gradient_checked=False,
            )
            return report

        from scripts.hpc.shared_transformer_runtime import (
            _load_prepared_artifacts,
            load_shared_runtime,
            validate_shared_artifact_metadata,
        )

        training_ids = tuple(int(value) for value in plan["training_artifact_ids"])
        emit("bootstrap_runtime", bootstrap_artifact_id=training_ids[0])
        runtime, parent, template, bootstrap_artifacts, metadata = load_shared_runtime(
            args, training_ids[:1]
        )
        jax = runtime["jax"]
        import jax.numpy as jnp

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
        science_scales = loaded["admission"]["scales"]
        daily_objective = make_shared_joint_daily_objective(
            runtime, head, template, state_scales, science_scales
        )
        _rollout, recurrent_objective = make_shared_recurrent_objective(
            head, template, state_scales, science_scales
        )

        def initializer():
            return head.initialize(runtime["probe_assembled"], seed=daily_protocol["seed"])

        parameter_shapes = jax.eval_shape(initializer)
        parameter_count = sum(
            int(np.prod(value.shape)) for value in jax.tree_util.tree_leaves(parameter_shapes)
        )
        if parameter_count != int(daily_protocol["architecture"]["parameter_count"]):
            raise ValueError("shared learning parameter tree differs from the admitted architecture")
        optimizer_shapes = jax.eval_shape(initialize_target_oriented_adamw, parameter_shapes)

        if args.resume_checkpoint is None:
            emit("initialize_fresh_parameters")
            parameters = jax.jit(initializer)()
            optimizer = initialize_target_oriented_adamw(parameters)
        else:
            saved = load_learning_checkpoint(
                args.resume_checkpoint,
                expected_method=method,
                jax=jax,
                parameter_shapes=parameter_shapes,
                optimizer_shapes=optimizer_shapes,
            )
            parameters, optimizer = jax.device_put((saved["parameters"], saved["optimizer"]))
            report.update(
                completed_updates=int(saved["completed_updates"]),
                updates=list(saved["updates"]),
                monitors=list(saved["monitors"]),
                resumed_from=str(args.resume_checkpoint),
                resume_same_declared_fresh_run=True,
                checkpoint=str(args.resume_checkpoint),
            )
        operations = make_tree_operations(jax, jnp, int(config["daily_rows_per_update"]))
        if not bool(np.asarray(operations["finite"](parameters))) or not bool(
            np.asarray(operations["finite"](optimizer))
        ):
            raise LearningFailure("initial_or_resumed_tree_nonfinite")

        rolling_checkpoint = checkpoint_path(args.output)

        def save_checkpoint(*, retained: bool) -> None:
            update_number = int(report["completed_updates"])
            path = checkpoint_path(args.output, update_number) if retained else rolling_checkpoint
            save_learning_checkpoint(
                path,
                jax=jax,
                parameters=parameters,
                optimizer=optimizer,
                completed_updates=update_number,
                method=method,
                updates=report["updates"],
                monitors=report["monitors"],
            )
            record = {
                "update": update_number,
                "path": str(path),
                "retained_milestone": retained,
            }
            report["checkpoints"] = [
                item
                for item in report["checkpoints"]
                if not (
                    item["update"] == update_number
                    and item["retained_milestone"] == retained
                )
            ] + [record]
            report["checkpoint"] = str(path)
            if not retained:
                report["rolling_checkpoint"] = str(path)
            emit("checkpoint_completed", checkpoint_record=record)

        # A durable declared initialization exists before monitor/gradient compilation.
        if report["completed_updates"] == 0 and args.resume_checkpoint is None:
            save_checkpoint(retained=True)

        from scripts.hpc.fresh_method_validation_pool import SPLITS, FreshMethodValidationPool
        from scripts.hpc.shared_learning_monitor import (
            SharedFitPool,
            build_shared_learning_monitor,
        )

        monitor = build_shared_learning_monitor(
            runtime,
            head,
            template,
            state_scales,
            science_scales,
            memory_limit_bytes=int(config["gradient_memory_limit_gib"]) * 2**30,
        )
        fit_pool = SharedFitPool(
            runtime, head, args.prepared_manifest, plan["fit_artifact_ids"]
        )
        validation_pool = FreshMethodValidationPool(args.validation_manifest)

        def run_monitor(update_number: int) -> None:
            existing = next(
                (
                    row
                    for row in report["monitors"]
                    if int(row["update"]) == update_number
                ),
                None,
            )
            if existing is not None and all(
                bool(panel["physical_gate_passed"])
                for panel in existing["panels"].values()
            ):
                return
            if existing is not None:
                raise LearningFailure(
                    "saved_milestone_monitor_physical", update=update_number
                )
            emit("milestone_monitor_started", monitor_update=update_number)
            panels = {
                "fixed_train_fit": monitor(
                    parameters,
                    fit_pool,
                    fit_pool.rows,
                    panel_name="fixed_train_fit",
                    split_role="fixed_train_fit",
                )
            }
            for name in SPLITS:
                rows = [row for row in validation_pool.rows if row["slice_id"] == name]
                panels[f"development_{name}"] = monitor(
                    parameters,
                    validation_pool,
                    rows,
                    panel_name=f"development_{name}",
                    split_role="held_out_development",
                )
            record = {"update": update_number, "panels": panels}
            report["monitors"].append(record)
            emit("milestone_monitor_completed", monitor_update=update_number)
            if not all(bool(panel["physical_gate_passed"]) for panel in panels.values()):
                raise LearningFailure("milestone_monitor_physical", update=update_number)
            # The pre-monitor checkpoint protects the weights. Republish the
            # same state after success so completed panel evidence is durable.
            save_checkpoint(retained=False)
            save_checkpoint(retained=True)

        if int(report["completed_updates"]) in config["milestones"]:
            run_monitor(int(report["completed_updates"]))

        def load_training_artifact(identifier: int) -> Any:
            if identifier == training_ids[0]:
                artifact = bootstrap_artifacts[0]
            else:
                artifacts, manifest = _load_prepared_artifacts(
                    args.prepared_manifest, (identifier,)
                )
                if manifest["identity"] != plan["prepared_manifest_identity"]:
                    raise ValueError("streamed artifact manifest identity changed")
                artifact = artifacts[0]
            validate_shared_artifact_metadata(runtime["index"], (identifier,), (artifact,))
            return artifact

        def artifact_cases(artifact: Any) -> tuple[Any, Any, Any]:
            one_day = artifact.batch.one_day
            row_count = int(np.shape(one_day.inputs.continuous_state)[0])
            if row_count != int(config["daily_rows_per_update"]):
                raise ValueError("training artifact does not contain the declared 32 daily rows")
            samples = (
                one_day
                if microbatch_size is not None
                else [
                    jax.tree_util.tree_map(lambda value, row=row: value[row], one_day)
                    for row in range(row_count)
                ]
            )
            seven = prepare_shared_window(
                head,
                artifact.batch.seven_day,
                window_index=int(config["recurrent_window_index"]),
                horizon=7,
            )
            thirty = prepare_shared_window(
                head,
                artifact.batch.thirty_day,
                window_index=int(config["recurrent_window_index"]),
                horizon=30,
            )
            return samples, seven, thirty

        completed = int(report["completed_updates"])
        evaluators = None
        if completed < int(config["max_updates"]):
            first_artifact = load_training_artifact(training_ids[completed])
            bootstrap_artifacts = ()
            first_samples, first_seven, first_thirty = artifact_cases(first_artifact)
            daily_example = (
                jax.tree_util.tree_map(lambda value: value[:microbatch_size], first_samples)
                if microbatch_size is not None
                else first_samples[0]
            )
            evaluators, graph_metadata = compile_gradient_evaluators(
                jax=jax,
                parameters=parameters,
                daily_objective=daily_objective,
                daily_sample=daily_example,
                recurrent_objective=recurrent_objective,
                seven_case=first_seven,
                thirty_case=first_thirty,
                memory_limit_bytes=int(config["gradient_memory_limit_gib"]) * 2**30,
                emit=emit,
                daily_microbatch_size=microbatch_size,
            )
            report["compiled_gradient_graphs"] = graph_metadata
        daily_accumulator = None
        if microbatch_size is not None and evaluators is not None:
            from scripts.hpc.shared_daily_microbatch import make_daily_accumulator

            daily_accumulator = make_daily_accumulator(
                jax=jax,
                jnp=jnp,
                microbatch_size=microbatch_size,
                evaluator=evaluators["daily"],
                expected_count=int(config["daily_rows_per_update"]),
            )
        update_function = make_update(daily_protocol)
        for index in range(completed, int(config["max_updates"])):
            identifier = training_ids[index]
            began = time.perf_counter()
            if index == completed:
                artifact = first_artifact
                daily_samples, seven_case, thirty_case = (
                    first_samples,
                    first_seven,
                    first_thirty,
                )
            else:
                artifact = load_training_artifact(identifier)
                daily_samples, seven_case, thirty_case = artifact_cases(artifact)
            report["attempted_update"] = index + 1
            report["attempted_artifact_id"] = identifier
            parameters, optimizer, row = execute_learning_update(
                jax=jax,
                parameters=parameters,
                optimizer=optimizer,
                daily_samples=daily_samples,
                seven_case=seven_case,
                thirty_case=thirty_case,
                evaluators=evaluators,
                operations=operations,
                update=update_function,
                daily_accumulator=daily_accumulator,
            )
            row = {
                "update": index + 1,
                "artifact_id": identifier,
                **row,
                "wall_seconds": time.perf_counter() - began,
                "device_memory": runtime["devices"][0].memory_stats(),
            }
            report["updates"].append(row)
            report["completed_updates"] = index + 1
            emit("optimizer_update_completed")
            del artifact, daily_samples, seven_case, thirty_case

            due_interval = (index + 1) % int(config["checkpoint_interval"]) == 0
            due_milestone = index + 1 in config["milestones"]
            if due_interval or due_milestone:
                save_checkpoint(retained=False)
            if due_milestone:
                save_checkpoint(retained=True)
                run_monitor(index + 1)

        emit(
            "completed",
            status="passed",
            classification=(
                "bounded_shared_split_500_update_learning_completed"
                if config["schema_version"] == "shared_learning_v2"
                else "bounded_shared_joint_100_update_learning_completed"
            ),
            final_checkpoint=report["checkpoint"],
            checkpoint_selection=config["checkpoint_selection"],
            device_memory=runtime["devices"][0].memory_stats(),
            scientific_acceptance=False,
            next_gate="interpret_declared_fit_and_held_out_development_panels",
        )
        return report
    except LearningFailure as error:
        emit("stopped", status="rejected", reason=error.reason, failure_details=error.details)
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
        "joint-admission-report",
        "validation-manifest",
        "output",
    ):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "manifests/coarse_graining/shared_learning_v1.json",
    )
    result.add_argument("--prepare-only", action="store_true")
    result.add_argument("--resume-checkpoint", type=Path)
    return result


if __name__ == "__main__":
    outcome = run(parser().parse_args())
    raise SystemExit(0 if outcome["status"] == "passed" else 1)
