"""Calibrate and run matched litter-weight mixed-gradient arms on one FREE GPU.

The runner composes the admitted daily objective, checked recurrent native AD,
shared mixed-gradient merge, and AdamW update.  It introduces no alternate
adjoint, cotangent construction, or scientific process implementation.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.hpc import litter_gradient_balance as balance
from scripts.hpc import probe_feedback_loss_calibration as calibration
from scripts.hpc import probe_feedback_loss_updates as admission
from scripts.hpc import probe_litter_change_admission as daily_admission
from scripts.hpc import probe_mixed_carbon_endpoints as endpoints
from scripts.hpc import run_fresh_start_one_day as prototype
from scripts.hpc.run_mixed_learning import configure_execution, validate_configuration

SCHEMA_VERSION = "litter_mixed_admission_result_v1"
PROTOCOL_SCHEMA_VERSION = "litter_mixed_admission_v1"
GRADIENT_BUNDLE_SCHEMA_VERSION = "litter_mixed_gradient_bundle_v1"
EXPECTED_PARAMETER_COUNT = 109_464_375
COMPONENTS = balance.COMPONENTS
ARMS = ("old", "selected")
PANEL_NAMES = ("fixed_training_daily", "excluded_train_daily")
PANEL_SIZE = 8


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain a JSON object")
    return value


def _validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the bounded geometry without coupling to the earlier daily-only trial."""

    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("litter mixed-admission protocol schema drift")
    source_update = int(protocol.get("source_update", -1))
    calibration_indices = tuple(int(value) for value in protocol.get("calibration_batch_indices", ()))
    update_indices = tuple(int(value) for value in protocol.get("update_batch_indices", ()))
    updates = int(protocol.get("updates_per_arm", -1))
    admission_updates = int(protocol.get("admission_updates", -1))
    if source_update != 4000 or calibration_indices != (4000, 4001):
        raise ValueError("litter mixed-admission calibration source geometry drift")
    if updates < 2 or admission_updates != 2:
        raise ValueError("litter mixed admission requires update 2 as its admission milestone")
    if update_indices != tuple(range(source_update, source_update + updates)):
        raise ValueError("litter mixed-admission update batches must be successive from source update")
    if protocol.get("daily_batch") != 64 or protocol.get("microbatch") != 8:
        raise ValueError("litter mixed-admission daily batch geometry drift")
    if protocol.get("outer_weights") != {"daily": 1.0, "T7": 0.5, "T30": 0.5}:
        raise ValueError("litter mixed-admission outer weights drift")
    old = tuple(float(value) for value in protocol.get("daily_coefficients_old", ()))
    litter = tuple(float(value) for value in protocol.get("daily_coefficients_litter_only", ()))
    if old != (1.0, 0.1, 0.1, 0.0) or litter != (0.0, 0.0, 0.0, 1.0):
        raise ValueError("litter mixed-admission daily component coefficients drift")
    selection = dict(protocol.get("weight_selection", {}))
    if set(selection) != {"target_median_ratio", "maximum_ratio", "maximum_weight"}:
        raise ValueError("litter mixed-admission weight-selection keys drift")
    # Exercise the shared selector's scalar validation without inventing a second policy.
    if not all(np.isfinite(float(value)) and float(value) > 0 for value in selection.values()):
        raise ValueError("litter mixed-admission weight-selection values must be positive and finite")
    litter_configuration = protocol.get("litter_carbon_stock_change")
    if not isinstance(litter_configuration, Mapping) or dict(litter_configuration) != {
        "weight": 0.1,
        "penalty": "pseudo_huber",
        "scale_source_pft_index": 13,
        "minimum_scale_gC_m2_day": 0.01,
    }:
        raise ValueError("litter mixed-admission loss configuration drift")
    if protocol.get("scientific_acceptance") is not False:
        raise ValueError("bounded mixed admission must not claim scientific acceptance")
    return {
        "source_update": source_update,
        "calibration_indices": calibration_indices,
        "update_indices": update_indices,
        "updates_per_arm": updates,
        "admission_updates": admission_updates,
        "old_coefficients": old,
        "litter_coefficients": litter,
        "weight_selection": {key: float(value) for key, value in selection.items()},
        "litter_carbon_stock_change": dict(litter_configuration),
    }


def _select_batches(plan: Mapping[str, Any], indices: Sequence[int]) -> tuple[Any, ...]:
    batches = tuple(plan["batches"][int(index)] for index in indices)
    for expected, batch in zip(indices, batches, strict=True):
        if int(batch.batch_index) != int(expected) or int(batch.draw_count) != 64:
            raise ValueError("selected mixed-admission daily batch differs from protocol")
        if len(batch.microbatches) != 8 or any(int(value.draw_count) != 8 for value in batch.microbatches):
            raise ValueError("selected mixed-admission batch lacks eight microbatch-8 draws")
    return batches


def _artifact_paths(output: Path, arm: str | None = None, update: int | None = None) -> Path:
    output = Path(output)
    if arm is None:
        if update is None:
            raise ValueError("calibration artifact requires a batch index")
        suffix = f"calibration-{int(update):04d}.pkl"
    else:
        if arm not in ARMS or update is None or int(update) <= 0:
            raise ValueError("arm checkpoint identity is invalid")
        suffix = f"{arm}-update-{int(update):06d}.checkpoint.pkl"
    return output.with_name(f"{output.stem}.{suffix}")


def _json_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")))


def _gradient_bundle_identity(
    *,
    source_checkpoint: Path,
    source_update: int,
    source_checkpoint_identity: Mapping[str, Any],
    source_git_head: str,
    runner_git_head: str,
    plan_identities: Mapping[str, Any],
    batch: Any,
    recurrent_rows: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    return _json_identity({
        "source_checkpoint_path": str(Path(source_checkpoint).resolve()),
        "source_update": int(source_update),
        "source_checkpoint_identity": dict(source_checkpoint_identity),
        "source_git_head": str(source_git_head),
        "runner_git_head": str(runner_git_head),
        "plan_identities": dict(plan_identities),
        "batch_index": int(batch.batch_index),
        "daily_batch": batch.as_metadata(),
        "recurrent_rows": dict(recurrent_rows),
        "protocol": dict(protocol),
        "components": list(COMPONENTS),
    })


def _save_gradient_bundle(path: Path, *, identity: Mapping[str, Any], gradients: Mapping[str, Any],
                          gram: Mapping[str, Any] | None, losses: Mapping[str, float],
                          hard_counts: Mapping[str, Any]) -> None:
    from research.daily_coarse_graining.daily_operator.one_day_training_state import _atomic_pickle

    completed = tuple(gradients)
    if not completed or completed != COMPONENTS[: len(completed)]:
        raise ValueError("gradient bundle components must be a nonempty declared prefix")
    if tuple(losses) != completed or tuple(hard_counts) != completed:
        raise ValueError("gradient bundle loss/hard evidence differs from completed components")
    if completed != COMPONENTS and gram is not None:
        raise ValueError("partial gradient bundle cannot claim a complete Gram matrix")
    _atomic_pickle(Path(path).resolve(), {
        "schema_version": GRADIENT_BUNDLE_SCHEMA_VERSION,
        "identity": _json_identity(identity),
        "completed_components": list(completed),
        "gradients": dict(gradients),
        "gram": None if gram is None else dict(gram),
        "losses": {name: float(losses[name]) for name in completed},
        "hard_counts": dict(hard_counts),
    })


def _load_gradient_bundle(path: Path, *, expected_identity: Mapping[str, Any],
                          require_complete: bool = True) -> dict[str, Any]:
    with Path(path).resolve().open("rb") as handle:
        payload = pickle.load(handle)
    expected_keys = {
        "schema_version", "identity", "completed_components", "gradients", "gram", "losses",
        "hard_counts",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("litter mixed gradient-bundle structure drift")
    if payload["schema_version"] != GRADIENT_BUNDLE_SCHEMA_VERSION:
        raise ValueError("litter mixed gradient-bundle schema drift")
    if payload["identity"] != _json_identity(expected_identity):
        raise ValueError("litter mixed gradient bundle belongs to different source metadata")
    completed = tuple(payload["completed_components"])
    if (
        not completed
        or completed != COMPONENTS[: len(completed)]
        or tuple(payload["gradients"]) != completed
        or tuple(payload["losses"]) != completed
        or tuple(payload["hard_counts"]) != completed
    ):
        raise ValueError("litter mixed gradient-bundle components drift")
    if require_complete and completed != COMPONENTS:
        raise ValueError("litter mixed gradient bundle is incomplete")
    if completed == COMPONENTS:
        if require_complete and payload["gram"] is None:
            raise ValueError("complete litter mixed gradient bundle has pending Gram geometry")
        if payload["gram"] is not None:
            matrix = np.asarray(payload["gram"].get("matrix"), dtype=np.float64)
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                raise ValueError("litter mixed gradient-bundle Gram matrix is invalid")
    elif payload["gram"] is not None:
        raise ValueError("partial litter mixed gradient bundle contains a Gram claim")
    return payload


def _tree_gram_report(jax: Any, gradients: Mapping[str, Any], gram_function: Any) -> dict[str, Any]:
    if tuple(gradients) != COMPONENTS:
        raise ValueError("Gram inputs must use the declared component order")
    matrix = np.asarray(jax.device_get(gram_function(tuple(gradients.values()))), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise FloatingPointError("full-parameter litter mixed Gram matrix is invalid")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise FloatingPointError("full-parameter litter mixed Gram matrix is asymmetric")
    diagonal = np.diag(matrix)
    if np.any(diagonal < 0):
        raise FloatingPointError("litter mixed component Gram diagonal is negative")
    norms = np.sqrt(diagonal)
    denominator = np.outer(norms, norms)
    cosine = np.divide(
        matrix,
        denominator,
        out=np.full_like(matrix, np.nan),
        where=denominator > 0,
    )
    return {
        "components": list(COMPONENTS),
        "matrix": matrix.tolist(),
        "raw_norms": dict(zip(COMPONENTS, norms.tolist(), strict=True)),
        "cosine_matrix": [
            [None if not np.isfinite(value) else float(value) for value in row]
            for row in cosine
        ],
        "all_finite": True,
    }


def _combine_component_gradients(combine: Any, gradients: Mapping[str, Any], *, litter_weight: float) -> Any:
    if set(gradients) != set(COMPONENTS):
        raise ValueError("mixed update requires all four ordered component gradients")
    if not np.isfinite(litter_weight) or litter_weight < 0:
        raise ValueError("mixed update litter weight must be finite and nonnegative")
    mixed = combine(gradients["daily_old"], gradients["litter"], 1.0, litter_weight)
    mixed = combine(mixed, gradients["T7"], 1.0, 0.5)
    return combine(mixed, gradients["T30"], 1.0, 0.5)


def _matched_start_record(source_checkpoint: Path, source_update: int,
                          optimizer_steps: Mapping[str, int]) -> dict[str, Any]:
    steps = {str(name): int(value) for name, value in optimizer_steps.items()}
    if set(steps) != set(ARMS) or any(value != int(source_update) for value in steps.values()):
        raise ValueError("paired arms did not start from the same source AdamW step")
    return {
        "source_checkpoint_path": str(Path(source_checkpoint).resolve()),
        "source_update": int(source_update),
        "arm_optimizer_steps": steps,
        "exact_same_parameter_payload": True,
        "exact_same_optimizer_payload": True,
    }


def _select_weight(gram_matrices: Sequence[Any], admitted: Mapping[str, Any]) -> dict[str, Any]:
    result = balance.select_litter_weight(
        gram_matrices, **dict(admitted["weight_selection"])
    )
    if result.get("components") != list(COMPONENTS):
        raise ValueError("shared litter gradient-balance component registry drift")
    selected = float(result.get("selected_weight", np.nan))
    if not np.isfinite(selected) or not 0.0 < selected <= admitted["weight_selection"]["maximum_weight"]:
        raise ValueError("shared litter gradient-balance selection is outside protocol")
    return result


def _checkpoint_method(*, source_checkpoint: Path, source_update: int, selected_weight: float,
                       arm: str, protocol: Mapping[str, Any], source_git_head: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("unknown litter mixed-admission arm")
    return {
        "implementation": "litter_mixed_admission_diagnostic_v1",
        "source_checkpoint": str(Path(source_checkpoint).resolve()),
        "source_update": int(source_update),
        "selected_weight": float(selected_weight),
        "arm": arm,
        "protocol": dict(protocol),
        "source_git_head": str(source_git_head),
        "claim": "bounded_diagnostic_not_formal_mixed_training_checkpoint",
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    from research.daily_coarse_graining.daily_operator.canonical_rollout import make_canonical_step
    from research.daily_coarse_graining.daily_operator.fresh_start_training import make_objective, make_update
    from research.daily_coarse_graining.daily_operator.mixed_objective import make_mixed_gradient
    from research.daily_coarse_graining.daily_operator.mixed_training_state import load_mixed_training_checkpoint
    from research.daily_coarse_graining.daily_operator.one_day_training_state import (
        load_training_checkpoint,
        make_gradient_accumulator,
        save_training_checkpoint,
    )
    from research.daily_coarse_graining.daily_operator.prepared_one_day import PreparedOneDayPool
    from research.daily_coarse_graining.daily_operator.recurrent_science_loss import bind_recurrent_science_loss
    from research.daily_coarse_graining.daily_operator.rollout_loss import make_rollout_loss
    from scripts.hpc.check_daily_training_runtime import _atomic_json
    from scripts.hpc.checked_recurrent_execution import bind_checked_backward, decode_counts, wrap_separate_checked
    from scripts.hpc.mixed_learning_monitor import build_mixed_rollout_monitor
    from scripts.hpc.mixed_learning_plan import load_plan
    from scripts.hpc.probe_one_day_batch import _compiled_memory, _device_memory
    from scripts.hpc.trained_screen_common import PreparedWindowPool, WindowPool, json_safe

    started = time.perf_counter()
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "stage": "created",
        "claim_scope": "bounded_matched_litter_weight_full_mixed_admission_and_continuation",
        "source_git_head": current_head,
        "scientific_acceptance": False,
        "calibration": [],
        "arms": {},
        "checkpoints": [],
    }
    output_owned = not args.output.exists()

    def emit(stage: str, **values: Any) -> None:
        report.update(values, stage=stage, elapsed_seconds=time.perf_counter() - started)
        _atomic_json(args.output, json_safe(report))
        print(
            f"litter_mixed_admission stage={stage} arms={len(report['arms'])} "
            f"seconds={report['elapsed_seconds']:.1f}",
            flush=True,
        )

    try:
        if args.shape_only or args.admission_only or args.cpu_forward_only:
            raise ValueError("litter mixed admission requires complete GPU execution")
        configuration = _read_json(args.mixed_protocol, name="mixed protocol")
        model_protocol = _read_json(args.protocol, name="model protocol")
        protocol = _read_json(args.mixed_admission_protocol, name="mixed admission protocol")
        admitted = _validate_protocol(protocol)
        if int(args.expected_update) != admitted["source_update"]:
            raise ValueError("CLI expected update differs from mixed-admission protocol")
        daily_admission._validate_model_protocol(model_protocol)
        validate_configuration(configuration, model_protocol)
        configured_protocol = (args.mixed_protocol.parent / configuration["model_protocol"]).resolve()
        if configured_protocol != args.protocol.resolve():
            raise ValueError("CLI and mixed model protocols differ")
        execution_policy = configure_execution(args)
        if execution_policy is None or not args.checked_recurrent or not args.separate_checked_validation:
            raise ValueError("litter mixed admission requires admitted split checked recurrent execution")
        plan = load_plan(args, configuration)
        calibration_batches = _select_batches(plan, admitted["calibration_indices"])
        update_batches = _select_batches(plan, admitted["update_indices"])
        schedule = plan["schedule"]
        if len(schedule["update_rows"]) <= admitted["update_indices"][-1]:
            raise ValueError("recurrent schedule does not cover mixed-admission updates")
        training_report = _read_json(args.training_report, name="training report")
        checkpoint_identity = endpoints._checkpoint_identity(
            training_report,
            configuration=configuration,
            model_protocol=model_protocol,
            execution_policy=execution_policy,
            plan=plan,
            expected_update=admitted["source_update"],
        )
        report.update(
            protocol=protocol,
            protocol_path=str(args.mixed_admission_protocol.resolve()),
            source_checkpoint_path=str(args.checkpoint.resolve()),
            source_training_report_path=str(args.training_report.resolve()),
            execution_policy=execution_policy,
            plan_identities=plan["identities"],
            selected_batches=[batch.as_metadata() for batch in update_batches],
            component_order=list(COMPONENTS),
        )
        if args.output.is_file():
            previous = _read_json(args.output, name="incomplete mixed-admission report")
            compatible = (
                previous.get("schema_version") == SCHEMA_VERSION
                and previous.get("protocol") == protocol
                and Path(previous.get("source_checkpoint_path", "")).resolve()
                == args.checkpoint.resolve()
                and previous.get("source_git_head") == current_head
                and previous.get("plan_identities") == plan["identities"]
            )
            if not compatible:
                raise ValueError("existing mixed-admission report belongs to a different run")
            if previous.get("status") == "completed":
                raise FileExistsError("completed mixed-admission output already exists")
            preserved_arms = dict(previous.get("arms", {}))
            preserved_checkpoints = list(previous.get("checkpoints", ()))
            preserved_monitors = dict(previous.get("development_monitors", {}))
            current_metadata = dict(report)
            report = previous
            report.update(current_metadata)
            report["arms"] = preserved_arms
            report["checkpoints"] = preserved_checkpoints
            if preserved_monitors:
                report["development_monitors"] = preserved_monitors
            report["prior_elapsed_seconds"] = (
                float(previous.get("prior_elapsed_seconds", 0.0))
                + float(previous.get("elapsed_seconds", 0.0))
            )
            report["calibration"] = []
            report["resumed_incomplete_report"] = True
            output_owned = True
        emit("metadata_validated")

        runtime, head, template, scales, fresh_parameters, _, _ = prototype.bootstrap(args, model_protocol)
        jax = runtime["jax"]
        import jax.numpy as jnp

        devices = tuple(jax.local_devices())
        if len(devices) != 1 or devices[0].platform != "gpu":
            raise ValueError("litter mixed admission requires exactly one visible FREE GPU")
        device = devices[0]
        if sum(int(value.size) for value in jax.tree_util.tree_leaves(fresh_parameters)) != EXPECTED_PARAMETER_COUNT:
            raise ValueError("actual full-network parameter count differs from admission contract")
        del fresh_parameters
        saved = load_mixed_training_checkpoint(args.checkpoint, **checkpoint_identity)
        source_optimizer_step = int(np.asarray(saved["optimizer_state"].step))
        if (
            int(saved["completed_updates"]) != admitted["source_update"]
            or source_optimizer_step != admitted["source_update"]
            or saved["daily_sampler_cursor"] != {"epoch": 0, "batch_offset": admitted["source_update"]}
        ):
            raise ValueError("source parameter/AdamW state is not exact update 4000")
        source_parameters = saved["parameters"]
        source_optimizer = saved["optimizer_state"]
        source_checkpoint_record = {
            "completed_updates": int(saved["completed_updates"]),
            "optimizer_step": source_optimizer_step,
            "daily_sampler_cursor": saved["daily_sampler_cursor"],
            "recurrent_schedule_cursor": int(saved["recurrent_schedule_cursor"]),
            "method": saved["method"],
            "source_identities": saved["source_identities"],
        }
        del saved

        candidate_protocol = daily_admission._candidate_model_protocol(model_protocol, protocol)
        graph_protocol = calibration._loss_protocol(
            candidate_protocol,
            canopy_source_pft_index=calibration._canopy_source_pft_index(runtime, head),
        )
        objective = make_objective(runtime, head, template, scales, graph_protocol)
        base_weights = calibration._base_weights(graph_protocol)
        batch_objective = calibration._make_batch_objective(jax, jnp, objective, base_weights)
        daily_evaluator = jax.jit(jax.value_and_grad(batch_objective, argnums=0, has_aux=True))
        daily_forward = jax.jit(batch_objective)
        add_gradient, mean_gradient = make_gradient_accumulator(jax, jnp)
        combine = make_mixed_gradient(jax, jnp)
        optimizer_update = make_update(model_protocol)
        finite = jax.jit(lambda tree: jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(value)) for value in jax.tree_util.tree_leaves(tree)
        ])))
        norm = jax.jit(lambda tree: jnp.sqrt(sum(
            jnp.sum(jnp.square(value.astype(jnp.float64)))
            for value in jax.tree_util.tree_leaves(tree)
        )))
        gram_function = jax.jit(lambda trees: jnp.stack([
            jnp.stack([admission._tree_dot(jax, jnp, left, right) for right in trees])
            for left in trees
        ]))
        coefficients = {
            "old": jax.device_put(jnp.asarray(admitted["old_coefficients"], dtype=jnp.float64), device),
            "litter": jax.device_put(jnp.asarray(admitted["litter_coefficients"], dtype=jnp.float64), device),
        }
        daily_pool = PreparedOneDayPool(plan["daily_source"], cache_size=5)
        recurrent_pool = PreparedWindowPool(schedule["source_manifest"], cache_size=2)
        development_pool = WindowPool(args.selection)
        if daily_pool.identity != plan["catalogue"]["source_identity"] or recurrent_pool.identity != schedule["source_identity"]:
            raise ValueError("daily/recurrent readers differ from validated source identities")

        def sample(rows: Sequence[Any]) -> Any:
            metadata = admission._rows(rows)
            return daily_pool.batch([(int(row["artifact_id"]), int(row["batch_row"])) for row in metadata])

        source_device_parameters = jax.device_put(source_parameters, device)
        first_samples = jax.device_put(sample(calibration_batches[0].microbatches[0].rows), device)
        emit("compile_daily_gradient")
        began = time.perf_counter()
        daily_executable = daily_evaluator.lower(
            source_device_parameters, first_samples, coefficients["old"]
        ).compile()
        daily_memory = _compiled_memory(daily_executable)
        if daily_memory["estimated_live_bytes"] > configuration["compiled_memory_limit_gib"] * 1024**3:
            raise MemoryError("daily gradient exceeds compiled-memory envelope")
        daily_compile_seconds = time.perf_counter() - began
        report["compilation"] = {
            "daily": {"seconds": daily_compile_seconds, "memory": daily_memory}
        }
        emit("daily_gradient_compiled", memory=_device_memory(device))

        recurrent_loss, recurrent_scales = bind_recurrent_science_loss(
            runtime, head, configuration["recurrent_weights"]
        )
        rollout = make_rollout_loss(jax, jnp, make_canonical_step(head, template), recurrent_loss)
        checked_backward, checked_names, validator = bind_checked_backward(
            jax, jnp, rollout, recurrent_loss, runtime, head, scales, model_protocol,
            separate_validation=True,
        )
        if validator is None:
            raise ValueError("split checked recurrent validator was not constructed")
        context = {"stock_weight": jnp.asarray(1.0, dtype=jnp.float64)}
        recurrent_executables: dict[int, Any] = {}
        report.update(
            parameter_count=EXPECTED_PARAMETER_COUNT,
            device=str(device),
            recurrent_scales=recurrent_scales,
        )

        def daily_components(parameters: Any, batch: Any, *,
                             names: Sequence[str] = ("daily_old", "litter")) -> tuple[dict[str, Any], dict[str, Any]]:
            names = tuple(names)
            if not names or any(name not in ("daily_old", "litter", "daily_selected") for name in names):
                raise ValueError("daily component request is invalid")
            totals = {
                name: jax.tree_util.tree_map(jnp.zeros_like, parameters)
                for name in names
            }
            losses = {name: 0.0 for name in names}
            terms = {name: np.zeros((4,), dtype=np.float64) for name in names}
            hard_totals = {name: {} for name in names}
            for micro_index, microbatch in enumerate(batch.microbatches):
                samples = jax.device_put(sample(microbatch.rows), device)
                for name in names:
                    coefficient_name = {
                        "daily_old": "old",
                        "litter": "litter",
                        "daily_selected": "selected",
                    }[name]
                    (value_auxiliary, gradient) = daily_executable(
                        parameters, samples, coefficients[coefficient_name]
                    )
                    value, auxiliary = value_auxiliary
                    jax.block_until_ready((value, auxiliary, gradient))
                    hard = calibration._hard_summary(
                        jax.device_get(auxiliary["hard"]), batch_size=microbatch.draw_count
                    )
                    litter_auxiliary = jax.device_get(auxiliary["litter_stock_change"])
                    litter_invalid = int(np.sum(np.asarray(litter_auxiliary["hard_invalid_count"])))
                    litter_empty = int(np.sum(np.asarray(litter_auxiliary["hard_empty_count"])))
                    if not np.isfinite(float(value)) or not hard["all_zero"] or litter_invalid or litter_empty:
                        raise ValueError(
                            f"daily {name} physical/finite gate failed at batch "
                            f"{batch.batch_index} microbatch {micro_index}"
                        )
                    totals[name] = add_gradient(totals[name], gradient, microbatch.draw_count)
                    losses[name] += float(value) * microbatch.draw_count
                    terms[name] += np.mean(
                        np.asarray(jax.device_get(auxiliary["terms"]), dtype=np.float64), axis=0
                    ) * microbatch.draw_count
                    for key, count in hard["totals"].items():
                        hard_totals[name][key] = hard_totals[name].get(key, 0) + int(count)
                    del auxiliary, gradient
                del samples
            result = {
                name: mean_gradient(totals[name], batch.draw_count)
                for name in names
            }
            return result, {
                "losses": {name: losses[name] / batch.draw_count for name in names},
                "terms": {name: (terms[name] / batch.draw_count).tolist() for name in names},
                "hard": hard_totals,
            }

        def recurrent_component(parameters: Any, batch_index: int, horizon: int) -> tuple[Any, float, dict[str, int], Mapping[str, Any]]:
            row = schedule["update_rows"][batch_index][f"T{horizon}"]
            case = recurrent_pool.case(jax, row)
            if horizon not in recurrent_executables:
                emit(f"compile_T{horizon}", calibration_batch_index=batch_index)
                compile_started = time.perf_counter()
                raw = checked_backward.lower(parameters, *case[:4], context).compile()
                memory = _compiled_memory(raw)
                if memory["estimated_live_bytes"] > configuration["compiled_memory_limit_gib"] * 1024**3:
                    raise MemoryError(f"T{horizon} checked gradient exceeds compiled-memory envelope")
                recurrent_executables[horizon] = wrap_separate_checked(jax, raw, validator)
                report["compilation"][f"T{horizon}"] = {
                    "seconds": time.perf_counter() - compile_started,
                    "memory": memory,
                }
                del raw
                emit(f"compiled_T{horizon}", calibration_batch_index=batch_index)
            (value, counts), gradient = recurrent_executables[horizon](
                parameters, *case[:4], context, case[4]
            )
            value, counts = jax.device_get((value, counts))
            hard = decode_counts(checked_names, counts)
            if not np.isfinite(float(value)) or any(hard.values()):
                raise ValueError(f"checked T{horizon} physical/finite gate failed at batch {batch_index}")
            return gradient, float(value), hard, row

        first_bundle: dict[str, Any] | None = None
        grams = []
        for batch in calibration_batches:
            batch_index = int(batch.batch_index)
            recurrent_rows = {
                f"T{horizon}": schedule["update_rows"][batch_index][f"T{horizon}"]
                for horizon in (7, 30)
            }
            identity = _gradient_bundle_identity(
                source_checkpoint=args.checkpoint,
                source_update=admitted["source_update"],
                source_checkpoint_identity=source_checkpoint_record,
                source_git_head=training_report["source_git_head"],
                runner_git_head=current_head,
                plan_identities=plan["identities"],
                batch=batch,
                recurrent_rows=recurrent_rows,
                protocol=protocol,
            )
            path = _artifact_paths(args.output, update=batch_index)
            reused_components: list[str] = []
            if path.is_file():
                bundle = _load_gradient_bundle(
                    path, expected_identity=identity, require_complete=False
                )
                host_gradients = dict(bundle["gradients"])
                losses = dict(bundle["losses"])
                hard_counts = dict(bundle["hard_counts"])
                reused_components = list(bundle["completed_components"])
                gram = bundle["gram"]
            else:
                host_gradients, losses, hard_counts = {}, {}, {}
                gram = None
            for name in COMPONENTS[len(host_gradients):]:
                emit(
                    "calibration_gradient",
                    calibration_batch_index=batch_index,
                    calibration_component=name,
                )
                if name in ("daily_old", "litter"):
                    values, record = daily_components(
                        source_device_parameters, batch, names=(name,)
                    )
                    gradient = values[name]
                    value = record["losses"][name]
                    hard = record["hard"][name]
                else:
                    horizon = int(name[1:])
                    gradient, value, hard, _ = recurrent_component(
                        source_device_parameters, batch_index, horizon
                    )
                host_gradients[name] = jax.device_get(gradient)
                losses[name] = float(value)
                hard_counts[name] = hard
                _save_gradient_bundle(
                    path,
                    identity=identity,
                    gradients=host_gradients,
                    gram=gram,
                    losses=losses,
                    hard_counts=hard_counts,
                )
                del gradient
                gc.collect()
                emit(
                    "calibration_component_saved",
                    calibration_batch_index=batch_index,
                    calibration_component=name,
                )
            if gram is None:
                device_gradients = jax.device_put(host_gradients, device)
                ordered = {component: device_gradients[component] for component in COMPONENTS}
                gram = _tree_gram_report(jax, ordered, gram_function)
                _save_gradient_bundle(
                    path,
                    identity=identity,
                    gradients=host_gradients,
                    gram=gram,
                    losses=losses,
                    hard_counts=hard_counts,
                )
                del ordered, device_gradients
                emit("calibration_gram_saved", calibration_batch_index=batch_index)
            bundle = _load_gradient_bundle(path, expected_identity=identity)
            del host_gradients
            gc.collect()
            grams.append(bundle["gram"]["matrix"])
            report["calibration"].append({
                "batch_index": batch_index,
                "artifact": str(path.resolve()),
                "artifact_bytes": path.stat().st_size,
                "reused_components": reused_components,
                "gram": bundle["gram"],
                "losses": bundle["losses"],
                "hard_counts": bundle["hard_counts"],
            })
            if batch_index == admitted["calibration_indices"][0]:
                first_bundle = bundle
            else:
                del bundle
            emit("calibration_batch_completed", calibration_batch_index=batch_index)
        if first_bundle is None:
            raise RuntimeError("first calibration gradient bundle is unavailable")
        selection = _select_weight(grams, admitted)
        selected_weight = float(selection["selected_weight"])
        report["weight_selection"] = selection
        coefficients["selected"] = jax.device_put(
            jnp.asarray((*admitted["old_coefficients"][:3], selected_weight), dtype=jnp.float64),
            device,
        )
        emit("weight_selected", selected_weight=selected_weight)

        panel_rows = daily_admission._panel_rows(plan)
        panel_coefficients = {
            "old": coefficients["old"],
            "selected": coefficients["selected"],
        }
        start_steps = {name: source_optimizer_step for name in ARMS}
        report["matched_start"] = _matched_start_record(
            args.checkpoint, admitted["source_update"], start_steps
        )
        for arm in ARMS:
            prior_arm = report.get("arms", {}).get(arm, {})
            report["arms"][arm] = {
                "status": "pending",
                "litter_weight": 0.0 if arm == "old" else selected_weight,
                "source_update": admitted["source_update"],
                "updates": list(prior_arm.get("updates", ())),
                "initial_daily_panels": dict(prior_arm.get("initial_daily_panels", {})),
                "final_daily_panels": dict(prior_arm.get("final_daily_panels", {})),
                "checkpoints": list(prior_arm.get("checkpoints", ())),
            }
            for panel_name in PANEL_NAMES:
                if panel_name not in report["arms"][arm]["initial_daily_panels"]:
                    report["arms"][arm]["initial_daily_panels"][panel_name] = daily_admission._panel_record(
                        source_device_parameters,
                        panel_rows[panel_name],
                        daily_pool,
                        daily_forward,
                        panel_coefficients[arm],
                        jax,
                    )
        initial_daily_clean = all(
            panel["hard_counts"]["all_zero"]
            and panel["litter_stock_change"]["hard_invalid_count"] == 0
            and panel["litter_stock_change"]["hard_empty_count"] == 0
            for arm in ARMS
            for panel in report["arms"][arm]["initial_daily_panels"].values()
        )
        report["initial_daily_physical_gate_passed"] = bool(initial_daily_clean)
        if not initial_daily_clean:
            raise ValueError("initial daily physical monitor gate failed")
        emit("initial_daily_panels_completed")

        checkpoint_methods = {
            arm: _checkpoint_method(
                source_checkpoint=args.checkpoint,
                source_update=admitted["source_update"],
                selected_weight=selected_weight,
                arm=arm,
                protocol=protocol,
                source_git_head=current_head,
            )
            for arm in ARMS
        }
        milestone = admitted["admission_updates"]
        for arm in ARMS:
            arm_report = report["arms"][arm]
            final_path = _artifact_paths(args.output, arm=arm, update=admitted["updates_per_arm"])
            milestone_path = _artifact_paths(args.output, arm=arm, update=milestone)
            recovery_path = final_path if final_path.is_file() else milestone_path if milestone_path.is_file() else None
            if recovery_path is None:
                parameters, optimizer = jax.device_put((source_parameters, source_optimizer), device)
                completed = 0
            else:
                recovered = load_training_checkpoint(
                    recovery_path, expected_method=checkpoint_methods[arm]
                )
                completed = int(recovered["completed_updates"])
                expected_cursor = {
                    "epoch": 0,
                    "batch_offset": admitted["source_update"] + completed,
                }
                if completed not in (milestone, admitted["updates_per_arm"]) or recovered["cursor"] != expected_cursor:
                    raise ValueError("litter mixed arm recovery checkpoint progress drift")
                parameters, optimizer = jax.device_put(
                    (recovered["parameters"], recovered["optimizer_state"]), device
                )
                arm_report["resumed_from"] = str(recovery_path.resolve())
                arm_report["recovered_completed_updates"] = completed
                del recovered
            if int(np.asarray(jax.device_get(optimizer.step))) != admitted["source_update"] + completed:
                raise ValueError("litter mixed arm AdamW recovery step drift")
            arm_report.update(status="running", completed_updates=completed)
            retained_updates = [
                row for row in arm_report["updates"]
                if int(row.get("update_in_arm", -1)) <= completed
            ]
            if retained_updates and [int(row["update_in_arm"]) for row in retained_updates] != list(
                range(1, completed + 1)
            ):
                raise ValueError("incomplete report has nonsequential recovered arm history")
            arm_report["updates"] = retained_updates
            if completed and not retained_updates:
                arm_report["recovered_history"] = {
                    "through_update": completed,
                    "details_unavailable_after_checkpoint_report_interruption": True,
                }

            for offset in range(completed, admitted["updates_per_arm"]):
                batch = update_batches[offset]
                batch_index = int(batch.batch_index)
                update_started = time.perf_counter()
                if offset == 0:
                    components = jax.device_put(first_bundle["gradients"], device)
                    losses = dict(first_bundle["losses"])
                    hard_counts = dict(first_bundle["hard_counts"])
                    gram = dict(first_bundle["gram"])
                    reused_calibration_gradient = True
                    arm_weight = 0.0 if arm == "old" else selected_weight
                    daily_loss = losses["daily_old"] + arm_weight * losses["litter"]
                    mixed = _combine_component_gradients(
                        combine, components, litter_weight=arm_weight
                    )
                else:
                    daily_name = "daily_selected" if arm == "selected" else "daily_old"
                    daily_values, daily_record = daily_components(
                        parameters, batch, names=(daily_name,)
                    )
                    daily_gradient = daily_values[daily_name]
                    daily_loss = float(daily_record["losses"][daily_name])
                    losses = {
                        "daily_weighted": daily_loss,
                        "daily_component_means": dict(zip(
                            daily_admission.LOSS_NAMES,
                            daily_record["terms"][daily_name],
                            strict=True,
                        )),
                    }
                    hard_counts = {"daily": daily_record["hard"]}
                    recurrent_gradients = {}
                    for horizon in (7, 30):
                        gradient, value, hard, _ = recurrent_component(parameters, batch_index, horizon)
                        recurrent_gradients[f"T{horizon}"] = gradient
                        losses[f"T{horizon}"] = value
                        hard_counts[f"T{horizon}"] = hard
                    gram = {
                        "components": ["daily_weighted", "T7", "T30"],
                        "raw_norms": {
                            "daily_weighted": float(norm(daily_gradient)),
                            "T7": float(norm(recurrent_gradients["T7"])),
                            "T30": float(norm(recurrent_gradients["T30"])),
                        },
                        "calibration_only_full_gram": True,
                    }
                    mixed = combine(daily_gradient, recurrent_gradients["T7"], 1.0, 0.5)
                    mixed = combine(mixed, recurrent_gradients["T30"], 1.0, 0.5)
                    components = {"daily": daily_gradient, **recurrent_gradients}
                    reused_calibration_gradient = False
                jax.block_until_ready(mixed)
                if not bool(finite(mixed)):
                    raise ValueError("combined litter mixed gradient is nonfinite")
                proposed, proposed_optimizer, gradient_norm, clip = optimizer_update(
                    parameters, mixed, optimizer
                )
                jax.block_until_ready((proposed, proposed_optimizer, gradient_norm, clip))
                expected_step = admitted["source_update"] + offset + 1
                if (
                    not bool(finite((proposed, proposed_optimizer)))
                    or not np.isfinite(float(gradient_norm))
                    or not 0.0 < float(clip) <= 1.0
                    or int(np.asarray(jax.device_get(proposed_optimizer.step))) != expected_step
                ):
                    raise ValueError("complete litter mixed AdamW proposal is invalid")
                parameters, optimizer = proposed, proposed_optimizer
                mixed_loss = daily_loss + 0.5 * losses["T7"] + 0.5 * losses["T30"]
                arm_report["updates"].append({
                    "update_in_arm": offset + 1,
                    "source_absolute_update": expected_step,
                    "daily_batch_index": batch_index,
                    "reused_frozen_source_gradient": reused_calibration_gradient,
                    "losses": losses,
                    "mixed_loss": mixed_loss,
                    "component_geometry": gram,
                    "combined_gradient_norm": float(gradient_norm),
                    "clip": float(clip),
                    "hard_counts": hard_counts,
                    "optimizer_step": expected_step,
                    "seconds": time.perf_counter() - update_started,
                })
                arm_report["completed_updates"] = offset + 1
                del mixed, components, proposed, proposed_optimizer
                emit(f"{arm}_update_completed", active_arm=arm, active_update=offset + 1)

                if offset + 1 in (milestone, admitted["updates_per_arm"]):
                    checkpoint_path = _artifact_paths(args.output, arm=arm, update=offset + 1)
                    save_training_checkpoint(
                        checkpoint_path,
                        jax=jax,
                        parameters=parameters,
                        optimizer_state=optimizer,
                        completed_updates=offset + 1,
                        cursor={
                            "epoch": 0,
                            "batch_offset": admitted["source_update"] + offset + 1,
                        },
                        method=checkpoint_methods[arm],
                    )
                    checkpoint_record = {
                        "arm": arm,
                        "update": offset + 1,
                        "path": str(checkpoint_path.resolve()),
                        "bytes": checkpoint_path.stat().st_size,
                    }
                    arm_report["checkpoints"].append(checkpoint_record)
                    report["checkpoints"].append(checkpoint_record)
                    emit(f"{arm}_checkpoint_saved", active_arm=arm, active_update=offset + 1)
            arm_report["status"] = "trained"
            arm_report["final_optimizer_step"] = int(np.asarray(jax.device_get(optimizer.step)))
            del parameters, optimizer
            gc.collect()
            emit(f"{arm}_training_completed", active_arm=None, active_update=None)

        # All durable update milestones exist before any post-training monitor.
        del first_bundle, source_device_parameters
        del daily_components, recurrent_component
        recurrent_executables.clear()
        daily_evaluator = daily_executable = checked_backward = validator = rollout = None
        gram_function = None
        gc.collect()
        monitor = build_mixed_rollout_monitor(
            runtime,
            head,
            template,
            scales,
            model_protocol,
            compiled_memory_limit_bytes=configuration["compiled_memory_limit_gib"] * 1024**3,
        )
        report.setdefault("development_monitors", {})
        if "initial" not in report["development_monitors"]:
            report["development_monitors"]["initial"] = monitor(
                jax.device_put(source_parameters, device),
                development_pool,
                development_pool.selection["windows"],
                panel_name="initial_source_update_4000",
                development=True,
                apply_science_gates=True,
            )
        initial_recurrent_clean = (
            report["development_monitors"]["initial"]["physical_gate_passed"]
            and report["development_monitors"]["initial"]["metric_population_complete"]
        )
        report["initial_development_monitor_gate"] = {
            "physical_gate_passed": bool(
                report["development_monitors"]["initial"]["physical_gate_passed"]
            ),
            "metric_population_complete": bool(
                report["development_monitors"]["initial"]["metric_population_complete"]
            ),
            "science_accuracy_gate_required": False,
        }
        if not initial_recurrent_clean:
            raise ValueError("initial recurrent physical or metric-population monitor gate failed")
        emit("initial_development_monitor_completed")
        for arm in ARMS:
            checkpoint_path = _artifact_paths(
                args.output, arm=arm, update=admitted["updates_per_arm"]
            )
            restored = load_training_checkpoint(
                checkpoint_path, expected_method=checkpoint_methods[arm]
            )
            parameters = jax.device_put(restored["parameters"], device)
            for panel_name in PANEL_NAMES:
                if panel_name not in report["arms"][arm]["final_daily_panels"]:
                    report["arms"][arm]["final_daily_panels"][panel_name] = daily_admission._panel_record(
                        parameters,
                        panel_rows[panel_name],
                        daily_pool,
                        daily_forward,
                        panel_coefficients[arm],
                        jax,
                    )
            if arm not in report["development_monitors"]:
                report["development_monitors"][arm] = monitor(
                    parameters,
                    development_pool,
                    development_pool.selection["windows"],
                    panel_name=f"{arm}_after_{admitted['updates_per_arm']}_updates",
                    development=True,
                    apply_science_gates=True,
                )
            daily_clean = all(
                panel["hard_counts"]["all_zero"]
                and panel["litter_stock_change"]["hard_invalid_count"] == 0
                and panel["litter_stock_change"]["hard_empty_count"] == 0
                for panel in report["arms"][arm]["final_daily_panels"].values()
            )
            recurrent_clean = (
                report["development_monitors"][arm]["physical_gate_passed"]
                and report["development_monitors"][arm]["metric_population_complete"]
            )
            report["arms"][arm]["final_monitor_gate"] = {
                "daily_physical_gate_passed": bool(daily_clean),
                "recurrent_physical_gate_passed": bool(
                    report["development_monitors"][arm]["physical_gate_passed"]
                ),
                "recurrent_metric_population_complete": bool(
                    report["development_monitors"][arm]["metric_population_complete"]
                ),
                "science_accuracy_gate_required": False,
            }
            if not daily_clean or not recurrent_clean:
                raise ValueError(f"{arm} final physical or metric-population monitor gate failed")
            report["arms"][arm]["status"] = "completed"
            del restored, parameters
            gc.collect()
            emit(f"{arm}_final_monitors_completed")

        admission_complete = all(
            any(row["update"] == milestone for row in report["arms"][arm]["checkpoints"])
            or _artifact_paths(args.output, arm=arm, update=milestone).is_file()
            for arm in ARMS
        )
        continuation_complete = all(
            report["arms"][arm]["completed_updates"] == admitted["updates_per_arm"]
            for arm in ARMS
        )
        monitor_gates_passed = bool(
            initial_daily_clean
            and initial_recurrent_clean
            and all(
                gate["daily_physical_gate_passed"]
                and gate["recurrent_physical_gate_passed"]
                and gate["recurrent_metric_population_complete"]
                for gate in (
                    report["arms"][arm]["final_monitor_gate"] for arm in ARMS
                )
            )
        )
        full_admission = bool(admission_complete and continuation_complete and monitor_gates_passed)
        emit(
            "completed",
            status="completed",
            classification="bounded_matched_litter_weight_full_mixed_admission_and_continuation",
            technical_admission_passed=bool(admission_complete and monitor_gates_passed),
            bounded_continuation_completed=bool(continuation_complete),
            full_mixed_admission=full_admission,
            monitor_gates_passed=monitor_gates_passed,
            scientific_acceptance=False,
            final_memory=_device_memory(device),
            daily_reader=daily_pool.statistics(),
            active_arm=None,
            active_update=None,
        )
    except Exception as error:
        if not output_owned:
            raise
        if not args.output.exists():
            report["output_creation_failed_before_first_emit"] = True
        emit("failed", status="failed", error=f"{type(error).__name__}: {error}")
        raise
    return report


def parser() -> argparse.ArgumentParser:
    result = endpoints.parser()
    result.description = __doc__
    result.add_argument("--mixed-admission-protocol", type=Path, required=True)
    result.set_defaults(
        compare_checked_recurrent=False,
        checked_reference_report=None,
        dual_device_profile=False,
        dual_device_training=False,
    )
    return result


if __name__ == "__main__":
    result = run(parser().parse_args())
    if result.get("status") != "completed" or result.get("technical_admission_passed") is not True:
        raise SystemExit(2)
