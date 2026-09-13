"""Run two train-only AdamW updates for the calibrated feedback objective.

This is a bounded technical admission from fresh seed 20260911.  It consumes
exactly the first two epoch-0 batch-32 draws, accumulates batch-4 gradients,
and writes no checkpoint or reusable model state.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.hpc import probe_feedback_loss_calibration as calibration
from scripts.hpc import run_fresh_start_one_day as prototype

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "feedback_loss_update_admission_result_v1"
CONFIG_SCHEMA_VERSION = "feedback_loss_update_admission_v1"
UPDATE_COUNT = 2
PARENT_BATCH_SIZE = 32
MICROBATCH_SIZE = 4
FIXED_PANEL_SIZE = 64
DIRECTION_SEEDS = (20260913, 20260914)
DIRECTION_RTOL = 0.01
DIRECTION_ATOL = 1e-6
MEMORY_LIMIT_BYTES = 24 * 1024**3


def _read_json(path: Path, *, name: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain a JSON object")
    return value


def _coefficients(configuration: Mapping[str, Any]) -> tuple[float, float, float]:
    if configuration.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("feedback update admission configuration schema drift")
    values = configuration.get("coefficients")
    if not isinstance(values, dict) or set(values) != set(calibration.LOSS_NAMES):
        raise ValueError("feedback coefficients must contain exactly base, canopy and litterpart")
    result = tuple(float(values[name]) for name in calibration.LOSS_NAMES)
    if not all(np.isfinite(value) and value >= 0.0 for value in result):
        raise ValueError("feedback coefficients must be explicitly supplied finite nonnegative values")
    if result[0] != 1.0:
        raise ValueError("the base-loss coefficient must remain exactly 1")
    return result


def _calibration_complete(report: Mapping[str, Any]) -> bool:
    batches = report.get("batches")
    return bool(
        report.get("schema_version") == calibration.SCHEMA_VERSION
        and report.get("status") == "completed"
        and report.get("training_updates") == 0
        and report.get("coefficient_selected") is None
        and report.get("all_reported_losses_finite") is True
        and report.get("all_gradient_values_finite") is True
        and report.get("all_hard_counts_zero") is True
        and isinstance(batches, list)
        and len(batches) == calibration.MICROBATCH_COUNT
        and all(isinstance(batch, dict) and batch.get("completed") is True for batch in batches)
    )


def _validate_inputs(
    pool_protocol: Mapping[str, Any],
    model_protocol: Mapping[str, Any],
    calibration_report: Mapping[str, Any],
    *,
    shape_only: bool,
) -> None:
    calibration._validate_protocols(pool_protocol, model_protocol)
    if not shape_only and not _calibration_complete(calibration_report):
        raise ValueError("completed finite zero-hard-count calibration is required before updates")


def _make_finite_evaluator(jax: Any, jnp: Any) -> Any:
    return jax.jit(
        lambda tree: jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in jax.tree_util.tree_leaves(tree)]))
    )


def _random_unit_direction(jax: Any, jnp: Any, parameters: Any, seed: int) -> Any:
    leaves, structure = jax.tree_util.tree_flatten(parameters)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    directions = [
        jnp.where(jax.random.bernoulli(key, shape=value.shape), 1.0, -1.0).astype(value.dtype)
        for key, value in zip(keys, leaves, strict=True)
    ]
    element_count = sum(int(value.size) for value in leaves)
    scale = jnp.asarray(np.sqrt(element_count), dtype=jnp.float64)
    directions = [value / scale.astype(value.dtype) for value in directions]
    return jax.tree_util.tree_unflatten(structure, directions)


def _tree_dot(jax: Any, jnp: Any, left: Any, right: Any) -> Any:
    return sum(
        jnp.vdot(a.astype(jnp.float64), b.astype(jnp.float64))
        for a, b in zip(
            jax.tree_util.tree_leaves(left),
            jax.tree_util.tree_leaves(right),
            strict=True,
        )
    )


def _directional_record(primal: Any, reverse: Any, forward: Any, *, seed: int) -> dict[str, Any]:
    primal_value = float(primal)
    reverse_value = float(reverse)
    forward_value = float(forward)
    return {
        "seed": seed,
        "primal": primal_value,
        "reverse_projection": reverse_value,
        "jvp_projection": forward_value,
        "absolute_error": abs(reverse_value - forward_value),
        "relative_tolerance": DIRECTION_RTOL,
        "absolute_tolerance": DIRECTION_ATOL,
        "resolved_projection": max(abs(reverse_value), abs(forward_value)) > 1e-5,
        "passed": bool(
            np.isfinite(primal_value)
            and np.isfinite(reverse_value)
            and np.isfinite(forward_value)
            and np.isclose(
                reverse_value,
                forward_value,
                rtol=DIRECTION_RTOL,
                atol=DIRECTION_ATOL,
            )
        ),
    }


def _rows(rows: Sequence[Any]) -> list[dict[str, Any]]:
    return [row.as_metadata() if hasattr(row, "as_metadata") else dict(row) for row in rows]


def _concatenate_auxiliary(jax: Any, batches: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not batches:
        raise ValueError("cannot summarize an empty fixed panel")
    host = tuple(jax.device_get(batch) for batch in batches)
    return jax.tree_util.tree_map(lambda *values: np.concatenate(values, axis=0), *host)


def _panel_record(
    parameters: Any,
    rows: Sequence[Mapping[str, Any]],
    pool: Any,
    forward: Any,
    coefficients: Any,
    jax: Any,
) -> Mapping[str, Any]:
    if len(rows) != FIXED_PANEL_SIZE:
        raise ValueError(f"fixed train panel must contain exactly {FIXED_PANEL_SIZE} rows")
    auxiliaries = []
    losses = []
    began = time.perf_counter()
    for start in range(0, len(rows), MICROBATCH_SIZE):
        selected = rows[start : start + MICROBATCH_SIZE]
        samples = pool.batch([(int(row["artifact_id"]), int(row["batch_row"])) for row in selected])
        loss, auxiliary = forward(parameters, samples, coefficients)
        jax.block_until_ready((loss, auxiliary))
        losses.append(float(loss))
        auxiliaries.append(auxiliary)
    combined = _concatenate_auxiliary(jax, auxiliaries)
    hard = calibration._hard_summary(combined["hard"], batch_size=len(rows))
    diagnostics = calibration._diagnostics(jax, combined, batch_size=len(rows))
    return {
        "row_count": len(rows),
        "rows": _rows(rows),
        "mean_combined_loss": float(np.mean(losses)),
        "mean_component_losses": {
            name: float(np.mean(np.asarray(combined["terms"])[:, index]))
            for index, name in enumerate(calibration.LOSS_NAMES)
        },
        "hard_counts": hard,
        "diagnostics": diagnostics,
        "seconds": time.perf_counter() - began,
    }


def run(args: Any) -> Mapping[str, Any]:
    from research.daily_coarse_graining.daily_operator.fresh_start_training import (
        make_objective,
        make_update,
    )
    from research.daily_coarse_graining.daily_operator.one_day_training_state import (
        make_gradient_accumulator,
    )
    from research.daily_coarse_graining.daily_operator.prepared_one_day import PreparedOneDayPool
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        initialize_target_oriented_adamw,
    )
    from scripts.hpc.probe_one_day_batch import _compiled_memory, _device_memory, _resolved_reference
    from scripts.hpc.run_fresh_start_one_day_pool import pool_plan
    from scripts.hpc.run_successor_fast_owner_gpu_probe import _tree_numeric_report
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json

    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "claim_scope": "two_update_train_only_technical_admission",
        "completed_updates": 0,
        "checkpoint_inherited": False,
        "checkpoint_written": False,
        "scientific_acceptance_claim": False,
        "long_horizon_acceptance": False,
        "updates": [],
        "monitors": [],
        "source_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }

    def emit(stage: str, **values: Any) -> None:
        report.update(values, stage=stage, elapsed_seconds=time.perf_counter() - started)
        _atomic_json(args.output, report)
        print(
            f"feedback_loss_updates stage={stage} updates={report['completed_updates']} "
            f"seconds={report['elapsed_seconds']:.1f}",
            flush=True,
        )

    try:
        if args.admission_only or args.cpu_forward_only:
            raise ValueError("feedback update admission supports full execution or --shape-only")
        if args.output.exists():
            raise FileExistsError("feedback update admission output already exists")
        pool_configuration = _read_json(args.pool_protocol, name="pool protocol")
        model_protocol = _read_json(args.protocol, name="model protocol")
        admission_configuration = _read_json(args.admission_config, name="admission configuration")
        calibration_report = _read_json(args.calibration_report, name="calibration report")
        coefficient_values = _coefficients(admission_configuration)
        _validate_inputs(
            pool_configuration,
            model_protocol,
            calibration_report,
            shape_only=args.shape_only,
        )
        configured_protocol = _resolved_reference(
            pool_configuration["model_protocol"], relative_to=args.pool_protocol.resolve()
        )
        if configured_protocol != args.protocol.resolve():
            raise ValueError("pool and CLI model protocols differ")
        calibration_batches = calibration_report.get("batches")
        if not isinstance(calibration_batches, list):
            calibration_batches = ()
        report.update(
            coefficients=dict(zip(calibration.LOSS_NAMES, coefficient_values, strict=True)),
            coefficient_source=(
                "explicit_synthetic_shape_only" if args.shape_only else "externally_selected_after_calibration"
            ),
            calibration_complete=_calibration_complete(calibration_report),
            admission_configuration=admission_configuration,
            admission_config_path=str(args.admission_config.resolve()),
            calibration_report_facts={
                "schema_version": calibration_report.get("schema_version"),
                "status": calibration_report.get("status"),
                "training_updates": calibration_report.get("training_updates"),
                "all_reported_losses_finite": calibration_report.get("all_reported_losses_finite"),
                "all_gradient_values_finite": calibration_report.get("all_gradient_values_finite"),
                "all_hard_counts_zero": calibration_report.get("all_hard_counts_zero"),
                "completed_batch_count": sum(
                    int(isinstance(batch, dict) and batch.get("completed") is True) for batch in calibration_batches
                ),
            },
            calibration_report_path=str(args.calibration_report.resolve()),
            pool_protocol_path=str(args.pool_protocol.resolve()),
            protocol_path=str(args.protocol.resolve()),
            training_updates=UPDATE_COUNT,
            parent_batch_size=PARENT_BATCH_SIZE,
            microbatch_size=MICROBATCH_SIZE,
            gradient_accumulation_microbatches=PARENT_BATCH_SIZE // MICROBATCH_SIZE,
            fixed_train_panel_size=FIXED_PANEL_SIZE,
            timing_protocol={
                "initial_monitor": "cold_forward_then_hot_batch4_calls",
                "gradient_compile": "explicit_lower_and_compile",
                "gradient_first_execution": "first_execution_after_explicit_compile",
                "training_microbatches": "hot_reuse_of_verified_batch4_gradient_executable",
                "update_monitors": "hot_reuse_of_batch4_forward_executable",
            },
        )

        emit("read_catalogue")
        catalogue_report = _read_json(args.catalogue_report, name="catalogue report")
        if catalogue_report.get("status") != "passed":
            raise ValueError("one-day catalogue report is not accepted")
        if catalogue_report.get("schema_version") != "prepared_one_day_catalogue_v1":
            raise ValueError("one-day catalogue report schema drift")
        catalogue_path = _resolved_reference(catalogue_report["catalogue"], relative_to=args.catalogue_report.resolve())
        with gzip.open(catalogue_path, "rt", encoding="utf-8") as handle:
            catalogue = json.load(handle)
        sampler, _ = pool_plan(catalogue, pool_configuration)
        parent_batches = tuple(itertools.islice(sampler.iter_epoch(0), UPDATE_COUNT))
        if len(parent_batches) != UPDATE_COUNT or any(
            batch.batch_index != index or batch.draw_count != PARENT_BATCH_SIZE
            for index, batch in enumerate(parent_batches)
        ):
            raise ValueError("epoch 0 does not begin with the required two complete batch-32 draws")
        if any(
            len(batch.microbatches) != PARENT_BATCH_SIZE // MICROBATCH_SIZE
            or any(micro.draw_count != MICROBATCH_SIZE for micro in batch.microbatches)
            for batch in parent_batches
        ):
            raise ValueError("the first two parent batches are not eight complete batch-4 microbatches")
        monitor_rows = tuple(catalogue.get("monitor_rows", ()))
        if len(monitor_rows) != FIXED_PANEL_SIZE:
            raise ValueError("catalogue fixed train monitor panel is not 64 rows")
        report.update(
            catalogue_path=str(catalogue_path),
            catalogue_source_identity=catalogue.get("source_identity"),
            selected_training_batches=[batch.as_metadata() for batch in parent_batches],
            fixed_train_panel_rows=_rows(monitor_rows),
        )

        emit("bootstrap")
        runtime, head, template, scales, parameters, _, _ = prototype.bootstrap(args, model_protocol)
        jax = runtime["jax"]
        import jax.numpy as jnp

        parameter_count = sum(int(value.size) for value in jax.tree_util.tree_leaves(parameters))
        if parameter_count != calibration.EXPECTED_PARAMETER_COUNT:
            raise ValueError("fresh neural parameter count differs from the admission contract")
        canopy_source_pft_index = calibration._canopy_source_pft_index(runtime, head)
        graph_protocol = calibration._loss_protocol(model_protocol, canopy_source_pft_index=canopy_source_pft_index)
        objective = make_objective(runtime, head, template, scales, graph_protocol)
        batch_objective = calibration._make_batch_objective(
            jax, jnp, objective, calibration._base_weights(graph_protocol)
        )
        scalar_loss = calibration._make_batch_loss(jax, jnp, objective, calibration._base_weights(graph_protocol))
        evaluator = jax.jit(jax.value_and_grad(batch_objective, argnums=0, has_aux=True))
        forward = jax.jit(batch_objective)
        update = make_update(model_protocol)
        add_gradient, mean_gradient = make_gradient_accumulator(jax, jnp)
        finite = _make_finite_evaluator(jax, jnp)
        coefficients = jnp.asarray(coefficient_values, dtype=jnp.float64)
        source_manifest = _resolved_reference(catalogue["source_manifest"], relative_to=catalogue_path)
        pool = PreparedOneDayPool(source_manifest, cache_size=pool_configuration["sampling"]["max_artifacts_per_batch"])
        if pool.identity != catalogue.get("source_identity"):
            raise ValueError("catalogue and prepared-pool source identities differ")
        first_microbatch = parent_batches[0].microbatches[0]
        first_samples = pool.batch([(row.artifact_id, row.batch_row) for row in first_microbatch.rows])
        optimizer = initialize_target_oriented_adamw(parameters)
        report.update(
            parameter_count=parameter_count,
            trainable_parameter_count=parameter_count,
            initialization_seed=calibration.INITIALIZATION_SEED,
            optimizer=model_protocol["optimizer"],
            parameter_finite_initial=bool(finite(parameters)),
            optimizer_finite_initial=bool(finite(optimizer)),
            reader=pool.statistics(),
            device=str(runtime["devices"][0]),
            memory_after_bootstrap=_device_memory(runtime["devices"][0]),
        )
        if not report["parameter_finite_initial"] or not report["optimizer_finite_initial"]:
            raise ValueError("fresh parameters or optimizer state are nonfinite")

        if args.shape_only:
            emit("trace_gradient_jvp_optimizer_and_monitor")
            (_, _), gradient_shape = jax.eval_shape(evaluator, parameters, first_samples, coefficients)
            direction_shape = jax.tree_util.tree_map(jnp.zeros_like, parameters)
            jax.eval_shape(
                lambda p, d, samples, weights: jax.jvp(
                    lambda current: scalar_loss(current, samples, weights), (p,), (d,)
                ),
                parameters,
                direction_shape,
                first_samples,
                coefficients,
            )
            jax.eval_shape(update, parameters, gradient_shape, optimizer)
            jax.eval_shape(forward, parameters, first_samples, coefficients)
            emit(
                "completed",
                status="completed",
                classification="explicit_synthetic_coefficient_shapes_only_no_science_execution",
                optimizer_updates_executed=0,
                scientific_acceptance_claim=False,
            )
            return report

        emit("initial_fixed_train_panel")
        initial_monitor = _panel_record(parameters, monitor_rows, pool, forward, coefficients, jax)
        report["monitors"].append({"update": 0, "panel": initial_monitor})
        if not initial_monitor["hard_counts"]["all_zero"]:
            emit("stopped", status="rejected", reason="initial_monitor_hard_counts")
            return report

        emit("compile_combined_gradient")
        compile_started = time.perf_counter()
        compiled_gradient = evaluator.lower(parameters, first_samples, coefficients).compile()
        gradient_compile_seconds = time.perf_counter() - compile_started
        gradient_memory = _compiled_memory(compiled_gradient)
        emit(
            "combined_gradient_compiled",
            gradient_compile_seconds=gradient_compile_seconds,
            gradient_compiled_memory=gradient_memory,
            memory_after_gradient_compile=_device_memory(runtime["devices"][0]),
        )
        if gradient_memory["estimated_live_bytes"] > MEMORY_LIMIT_BYTES:
            emit("stopped", status="resource_limited", reason="combined_gradient_memory")
            return report

        gradient_started = time.perf_counter()
        (first_value_and_auxiliary, first_gradient) = compiled_gradient(parameters, first_samples, coefficients)
        first_loss, first_auxiliary = first_value_and_auxiliary
        jax.block_until_ready((first_loss, first_auxiliary, first_gradient))
        gradient_execution_seconds = time.perf_counter() - gradient_started
        gradient_numeric = _tree_numeric_report(jax, first_gradient)
        first_hard = calibration._hard_summary(jax.device_get(first_auxiliary["hard"]), batch_size=MICROBATCH_SIZE)
        report.update(
            first_microbatch_combined_loss=float(first_loss),
            first_microbatch_gradient=gradient_numeric,
            first_microbatch_hard_counts=first_hard,
            gradient_first_execution_seconds=gradient_execution_seconds,
        )
        if not gradient_numeric["all_finite"] or gradient_numeric["l2_norm"] <= 0.0:
            emit("stopped", status="rejected", reason="combined_gradient_invalid")
            return report
        if not first_hard["all_zero"]:
            emit("stopped", status="rejected", reason="directional_batch_hard_counts")
            return report

        def tangent(current: Any, direction: Any, samples: Any, weights: Any) -> Any:
            return jax.jvp(
                lambda candidate: scalar_loss(candidate, samples, weights),
                (current,),
                (direction,),
            )

        reverse_references = []
        dot = jax.jit(lambda left, right: _tree_dot(jax, jnp, left, right))
        for seed in DIRECTION_SEEDS:
            reverse_started = time.perf_counter()
            direction = _random_unit_direction(jax, jnp, parameters, seed)
            reverse_projection = dot(first_gradient, direction)
            jax.block_until_ready(reverse_projection)
            reverse_references.append(
                {
                    "seed": seed,
                    "reverse_projection": float(reverse_projection),
                    "seconds": time.perf_counter() - reverse_started,
                }
            )
            emit("reverse_projection_completed", reverse_projection_references=reverse_references)
            del direction, reverse_projection

        emit("compile_combined_jvp")
        first_direction = _random_unit_direction(jax, jnp, parameters, DIRECTION_SEEDS[0])
        jvp_compile_started = time.perf_counter()
        compiled_jvp = jax.jit(tangent).lower(parameters, first_direction, first_samples, coefficients).compile()
        jvp_compile_seconds = time.perf_counter() - jvp_compile_started
        jvp_memory = _compiled_memory(compiled_jvp)
        emit(
            "combined_jvp_compiled",
            jvp_compile_seconds=jvp_compile_seconds,
            jvp_compiled_memory=jvp_memory,
            memory_after_jvp_compile=_device_memory(runtime["devices"][0]),
        )
        if jvp_memory["estimated_live_bytes"] > MEMORY_LIMIT_BYTES:
            emit("stopped", status="resource_limited", reason="combined_jvp_memory")
            return report

        directional = []
        for index, seed in enumerate(DIRECTION_SEEDS):
            direction = first_direction if index == 0 else _random_unit_direction(jax, jnp, parameters, seed)
            began = time.perf_counter()
            primal, forward_projection = compiled_jvp(parameters, direction, first_samples, coefficients)
            jax.block_until_ready((primal, forward_projection))
            row = _directional_record(
                primal,
                reverse_references[index]["reverse_projection"],
                forward_projection,
                seed=seed,
            )
            row["seconds"] = time.perf_counter() - began
            row["primal_matches_reverse_evaluation"] = bool(
                np.isclose(float(primal), float(first_loss), rtol=1e-5, atol=1e-6)
            )
            row["passed"] = bool(row["passed"] and row["primal_matches_reverse_evaluation"])
            directional.append(row)
            emit("directional_projection_completed", directional_checks=directional)
            del direction
        report["bounded_directional_gate_passed"] = all(row["passed"] for row in directional)
        if not report["bounded_directional_gate_passed"]:
            emit("stopped", status="rejected", reason="bounded_directional_agreement")
            return report
        del first_direction, first_gradient, first_auxiliary

        for update_index, parent_batch in enumerate(parent_batches, start=1):
            began = time.perf_counter()
            total_gradient = jax.tree_util.tree_map(jnp.zeros_like, parameters)
            weighted_loss = 0.0
            micro_records = []
            for microbatch in parent_batch.microbatches:
                samples = pool.batch([(row.artifact_id, row.batch_row) for row in microbatch.rows])
                micro_started = time.perf_counter()
                (loss_and_auxiliary, micro_gradient) = compiled_gradient(parameters, samples, coefficients)
                loss, auxiliary = loss_and_auxiliary
                jax.block_until_ready((loss, auxiliary, micro_gradient))
                hard = calibration._hard_summary(jax.device_get(auxiliary["hard"]), batch_size=microbatch.draw_count)
                record = {
                    "rows": _rows(microbatch.rows),
                    "loss": float(loss),
                    "hard_counts": hard,
                    "seconds": time.perf_counter() - micro_started,
                }
                micro_records.append(record)
                if not np.isfinite(record["loss"]) or not hard["all_zero"]:
                    emit(
                        "stopped",
                        status="rejected",
                        reason="training_microbatch_forward",
                        attempted_update=update_index,
                        failing_microbatch=record,
                    )
                    return report
                total_gradient = add_gradient(total_gradient, micro_gradient, microbatch.draw_count)
                weighted_loss += record["loss"] * microbatch.draw_count
            gradient = mean_gradient(total_gradient, parent_batch.draw_count)
            gradient_report = _tree_numeric_report(jax, gradient)
            if not gradient_report["all_finite"] or gradient_report["l2_norm"] <= 0.0:
                emit("stopped", status="rejected", reason="accumulated_gradient")
                return report
            update_started = time.perf_counter()
            proposed_parameters, proposed_optimizer, norm, clip_scale = update(parameters, gradient, optimizer)
            jax.block_until_ready((proposed_parameters, proposed_optimizer, norm, clip_scale))
            optimizer_seconds = time.perf_counter() - update_started
            finite_parameters = bool(finite(proposed_parameters))
            finite_optimizer = bool(finite(proposed_optimizer))
            if (
                not np.isfinite(float(norm))
                or not np.isfinite(float(clip_scale))
                or not 0.0 < float(clip_scale) <= 1.0
                or not finite_parameters
                or not finite_optimizer
            ):
                emit("stopped", status="rejected", reason="optimizer_proposal")
                return report
            parameters, optimizer = proposed_parameters, proposed_optimizer
            report["completed_updates"] = update_index
            update_record = {
                "update": update_index,
                "epoch": 0,
                "batch_index": parent_batch.batch_index,
                "rows": _rows(parent_batch.rows),
                "mean_combined_loss": weighted_loss / parent_batch.draw_count,
                "gradient": gradient_report,
                "gradient_norm_before_clip": float(norm),
                "clip_threshold": float(model_protocol["optimizer"]["clip_norm"]),
                "clip_scale": float(clip_scale),
                "clipped": float(clip_scale) < 1.0,
                "parameters_all_finite": finite_parameters,
                "optimizer_all_finite": finite_optimizer,
                "microbatches": micro_records,
                "optimizer_seconds": optimizer_seconds,
                "total_seconds": time.perf_counter() - began,
                "memory": _device_memory(runtime["devices"][0]),
            }
            report["updates"].append(update_record)
            emit("fixed_train_panel", active_monitor_update=update_index)
            monitor = _panel_record(parameters, monitor_rows, pool, forward, coefficients, jax)
            report["monitors"].append({"update": update_index, "panel": monitor})
            if not monitor["hard_counts"]["all_zero"]:
                emit("stopped", status="rejected", reason="monitor_hard_counts")
                return report
            emit(
                "update_completed",
                active_monitor_update=None,
                reader=pool.statistics(),
                memory=_device_memory(runtime["devices"][0]),
            )

        all_hard_zero = all(monitor["panel"]["hard_counts"]["all_zero"] for monitor in report["monitors"]) and all(
            micro["hard_counts"]["all_zero"]
            for update_record in report["updates"]
            for micro in update_record["microbatches"]
        )
        emit(
            "completed",
            status="completed",
            classification="two_update_train_only_technical_execution_complete",
            technical_admission_passed=bool(
                report["completed_updates"] == UPDATE_COUNT
                and report["bounded_directional_gate_passed"]
                and all_hard_zero
                and bool(finite((parameters, optimizer)))
            ),
            all_hard_counts_zero=all_hard_zero,
            parameters_all_finite=bool(finite(parameters)),
            optimizer_all_finite=bool(finite(optimizer)),
            clipped_update_count=sum(int(record["clipped"]) for record in report["updates"]),
            checkpoint_written=False,
            reader=pool.statistics(),
            final_memory=_device_memory(runtime["devices"][0]),
        )
    except Exception as error:
        emit("failed", status="failed", error=f"{type(error).__name__}: {error}")
        raise
    return report


def parser() -> argparse.ArgumentParser:
    result = prototype.parser()
    result.description = __doc__
    result.set_defaults(protocol=ROOT / "manifests/coarse_graining/fresh_start_one_day_probe_v3.json")
    result.add_argument(
        "--pool-protocol",
        type=Path,
        default=ROOT / "manifests/coarse_graining/fresh_start_one_day_pool_v1.json",
    )
    result.add_argument("--catalogue-report", type=Path, required=True)
    result.add_argument("--admission-config", type=Path, required=True)
    result.add_argument("--calibration-report", type=Path, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
