"""Run the bounded paired daily-only litter-change admission on a FREE GPU.

This runner adds no scientific process logic.  It composes the canonical
one-day objective, native JAX AD, and AdamW implementation around the exact
retained mixed-learning update-4000 state.  Both two-update arms consume the
same daily batches and independently restart from that state.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.hpc import probe_feedback_loss_calibration as calibration
from scripts.hpc import probe_feedback_loss_updates as admission
from scripts.hpc import probe_mixed_carbon_endpoints as endpoints
from scripts.hpc import run_fresh_start_one_day as prototype
from scripts.hpc.run_mixed_learning import configure_execution, validate_configuration

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "litter_change_admission_result_v1"
CONFIG_SCHEMA_VERSION = "litter_change_admission_v1"
EXPECTED_PARAMETER_COUNT = 109_464_375
EXPECTED_UPDATE = 4000
LOSS_NAMES = (
    "base",
    "canopy_stock_change",
    "litterpart_feedback",
    "litter_stock_change",
)
ARM_NAMES = ("old", "candidate")
PANEL_NAMES = ("fixed_training_daily", "excluded_train_daily")
PANEL_SIZE = 8
PHYSICAL_FLUX_NAMES = ("gpp", "npp", "autotrophic_respiration")


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain a JSON object")
    return value


def _validate_admission_protocol(configuration: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "updates_per_arm",
        "batch_indices",
        "daily_batch",
        "microbatch",
        "direction_seeds",
        "coefficients_old",
        "coefficients_candidate",
        "litter_carbon_stock_change",
    }
    if set(configuration) != expected_keys:
        raise ValueError("litter-change admission protocol keys drift")
    if configuration.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("litter-change admission protocol schema drift")
    if configuration.get("updates_per_arm") != 2:
        raise ValueError("litter-change admission requires two updates per arm")
    if configuration.get("batch_indices") != [4000, 4001]:
        raise ValueError("litter-change admission batch indices drift")
    if configuration.get("daily_batch") != 64 or configuration.get("microbatch") != 8:
        raise ValueError("litter-change admission daily batch geometry drift")
    if configuration.get("direction_seeds") != [20260920, 20260921]:
        raise ValueError("litter-change admission direction seeds drift")

    old = tuple(float(value) for value in configuration.get("coefficients_old", ()))
    candidate = tuple(float(value) for value in configuration.get("coefficients_candidate", ()))
    if old != (1.0, 0.1, 0.1, 0.0):
        raise ValueError("old-arm coefficients drift")
    if candidate != (1.0, 0.1, 0.1, 0.1):
        raise ValueError("candidate-arm coefficients drift")
    if not all(np.isfinite(value) and value >= 0.0 for value in (*old, *candidate)):
        raise ValueError("litter-change coefficients must be finite and nonnegative")

    litter = configuration.get("litter_carbon_stock_change")
    if not isinstance(litter, Mapping) or set(litter) != {
        "weight",
        "penalty",
        "scale_source_pft_index",
        "minimum_scale_gC_m2_day",
    }:
        raise ValueError("litter stock-change configuration keys drift")
    if (
        float(litter["weight"]) != 0.1
        or litter["penalty"] != "pseudo_huber"
        or int(litter["scale_source_pft_index"]) != 13
        or float(litter["minimum_scale_gC_m2_day"]) != 0.01
    ):
        raise ValueError("litter stock-change configuration differs from the admitted trial")
    return {
        "old": old,
        "candidate": candidate,
        "direction_seeds": tuple(int(value) for value in configuration["direction_seeds"]),
        "batch_indices": tuple(int(value) for value in configuration["batch_indices"]),
        "litter_carbon_stock_change": dict(litter),
    }


def _candidate_model_protocol(
    model_protocol: Mapping[str, Any], admission_protocol: Mapping[str, Any]
) -> dict[str, Any]:
    if "litter_carbon_stock_change" in model_protocol:
        raise ValueError("the original model protocol already declares litter stock-change supervision")
    candidate = copy.deepcopy(dict(model_protocol))
    candidate["litter_carbon_stock_change"] = copy.deepcopy(
        admission_protocol["litter_carbon_stock_change"]
    )
    if dict(model_protocol) == candidate:
        raise ValueError("candidate model protocol did not change")
    return candidate


def _validate_model_protocol(model_protocol: Mapping[str, Any]) -> None:
    optimizer = model_protocol.get("optimizer", {})
    if (
        model_protocol.get("parameter_count") != EXPECTED_PARAMETER_COUNT
        or model_protocol.get("trainable_scope") != "all_neural_parameters"
        or float(optimizer.get("learning_rate", np.nan)) != 1e-5
        or float(optimizer.get("clip_norm", np.nan)) != 1.0
    ):
        raise ValueError("model parameter or AdamW contract differs from source-4000 training")
    canopy = model_protocol.get("canopy_carbon_stock_change", {})
    if (
        float(canopy.get("weight", np.nan)) != 0.1
        or int(canopy.get("scale_source_pft_index", -1)) != 13
        or canopy.get("penalty") != "pseudo_huber"
        or float(model_protocol.get("loss", {}).get("litterpart_feedback_mse", np.nan)) != 0.1
    ):
        raise ValueError("existing physical daily-loss geometry differs from the paired trial")


def _select_training_batches(
    plan: Mapping[str, Any], configuration: Mapping[str, Any]
) -> tuple[Any, ...]:
    indices = tuple(int(value) for value in configuration["batch_indices"])
    batches = tuple(plan["batches"][index] for index in indices)
    batch_size = int(configuration["daily_batch"])
    microbatch_size = int(configuration["microbatch"])
    for expected_index, batch in zip(indices, batches, strict=True):
        if int(batch.batch_index) != expected_index or int(batch.draw_count) != batch_size:
            raise ValueError("selected daily batch differs from the admission protocol")
        if (
            len(batch.microbatches) != batch_size // microbatch_size
            or any(int(micro.draw_count) != microbatch_size for micro in batch.microbatches)
        ):
            raise ValueError("selected daily batch does not contain complete microbatch-8 draws")
    return batches


def _panel_rows(plan: Mapping[str, Any]) -> dict[str, tuple[dict[str, Any], ...]]:
    panels = {
        "fixed_training_daily": tuple(admission._rows(plan["batches"][0].rows[:PANEL_SIZE])),
        "excluded_train_daily": tuple(admission._rows(plan["monitor_rows"][:PANEL_SIZE])),
    }
    if any(len(rows) != PANEL_SIZE for rows in panels.values()):
        raise ValueError("litter-change admission panels must each contain eight TRAIN rows")
    return panels


def _finite_statistics(values: Any, valid: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    mask = np.broadcast_to(np.asarray(valid, dtype=np.bool_), array.shape)
    nonfinite_valid = int(np.count_nonzero(mask & ~np.isfinite(array)))
    selected = array[mask & np.isfinite(array)]
    if selected.size == 0:
        return {
            "count": 0,
            "nonfinite_valid": nonfinite_valid,
            "mean": None,
            "rmse": None,
        }
    return {
        "count": int(selected.size),
        "nonfinite_valid": nonfinite_valid,
        "mean": float(np.mean(selected)),
        "rmse": float(np.sqrt(np.mean(selected * selected))),
    }


def _constant_value(value: Any, *, name: str) -> float | int:
    array = np.asarray(value)
    if array.size < 1 or not np.all(array == array.reshape(-1)[0]):
        raise ValueError(f"panel litter scale field {name!r} is not constant")
    scalar = array.reshape(-1)[0]
    return int(scalar) if np.issubdtype(array.dtype, np.integer) else float(scalar)


def _litter_panel_summary(litter: Mapping[str, Any]) -> dict[str, Any]:
    owners = litter.get("per_owner")
    if not isinstance(owners, Mapping) or set(owners) != {"aboveground", "belowground"}:
        raise ValueError("litter panel owner registry drift")
    result: dict[str, Any] = {
        "unit": "gC_m-2_day-1",
        "weight": _constant_value(litter["weight"], name="weight"),
        "scale_source_pft_index": _constant_value(
            litter["scale_source_pft_index"], name="scale_source_pft_index"
        ),
        "scale_compact_pft_index": _constant_value(
            litter["scale_compact_pft_index"], name="scale_compact_pft_index"
        ),
        "minimum_scale_gC_m2_day": _constant_value(
            litter["minimum_scale_gC_m2_day"], name="minimum_scale_gC_m2_day"
        ),
        "hard_invalid_count": int(np.sum(np.asarray(litter["hard_invalid_count"]))),
        "hard_empty_count": int(np.sum(np.asarray(litter["hard_empty_count"]))),
        "per_owner": {},
    }
    for name in ("aboveground", "belowground"):
        owner = owners[name]
        valid = np.asarray(owner["valid"], dtype=np.bool_)
        result["per_owner"][name] = {
            "loss_mean": float(np.mean(np.asarray(owner["loss"], dtype=np.float64))),
            "raw_scale_gC_m2_day": _constant_value(owner["raw_scale"], name=f"{name}.raw_scale"),
            "effective_scale_gC_m2_day": _constant_value(
                owner["effective_scale"], name=f"{name}.effective_scale"
            ),
            "scale_coordinate_count": _constant_value(
                owner["scale_coordinate_count"], name=f"{name}.scale_coordinate_count"
            ),
            "valid_count": int(np.sum(np.asarray(owner["valid_count"]))),
            "expected_count": int(np.sum(np.asarray(owner["expected_count"]))),
            "invalid_prediction_count": int(
                np.sum(np.asarray(owner["invalid_prediction_count"]))
            ),
            "invalid_target_count": int(np.sum(np.asarray(owner["invalid_target_count"]))),
            "predicted_delta": np.asarray(owner["predicted_delta"], dtype=np.float64).tolist(),
            "teacher_delta": np.asarray(owner["teacher_delta"], dtype=np.float64).tolist(),
            "physical_residual": np.asarray(
                owner["physical_residual"], dtype=np.float64
            ).tolist(),
            "predicted_delta_statistics": _finite_statistics(owner["predicted_delta"], valid),
            "teacher_delta_statistics": _finite_statistics(owner["teacher_delta"], valid),
            "physical_residual_statistics": _finite_statistics(owner["physical_residual"], valid),
        }
    return result


def _physical_flux_summary(jax: Any, physical: Mapping[str, Any], *, batch_size: int) -> dict[str, Any]:
    from research.daily_coarse_graining.daily_operator.one_day_monitoring import summarize_monitor

    host = jax.device_get(physical)
    rows = tuple(
        jax.tree_util.tree_map(lambda value, index=index: np.asarray(value)[index], host)
        for index in range(batch_size)
    )
    summary = summarize_monitor(rows)["fluxes"]
    fields = summary["fields"]
    missing = [name for name in PHYSICAL_FLUX_NAMES if name not in fields]
    if missing:
        raise ValueError(f"daily panel lacks required physical fluxes: {missing!r}")
    return {
        "unit": summary["unit"],
        "fields": {name: fields[name] for name in PHYSICAL_FLUX_NAMES},
    }


def _panel_record(
    parameters: Any,
    rows: Sequence[Mapping[str, Any]],
    pool: Any,
    forward: Any,
    coefficients: Any,
    jax: Any,
) -> dict[str, Any]:
    if len(rows) != PANEL_SIZE:
        raise ValueError("daily admission panel must contain exactly eight rows")
    began = time.perf_counter()
    samples = pool.batch(
        [(int(row["artifact_id"]), int(row["batch_row"])) for row in rows]
    )
    loss, auxiliary = forward(parameters, samples, coefficients)
    jax.block_until_ready((loss, auxiliary))
    host = jax.device_get(auxiliary)
    terms = np.asarray(host["terms"], dtype=np.float64)
    if terms.shape != (PANEL_SIZE, len(LOSS_NAMES)):
        raise ValueError("daily panel loss-term shape drift")
    return {
        "row_count": PANEL_SIZE,
        "rows": [dict(row) for row in rows],
        "combined_loss": float(loss),
        "mean_component_losses": {
            name: float(np.mean(terms[:, index])) for index, name in enumerate(LOSS_NAMES)
        },
        "hard_counts": calibration._hard_summary(host["hard"], batch_size=PANEL_SIZE),
        "physical_fluxes": _physical_flux_summary(
            jax, auxiliary["physical"], batch_size=PANEL_SIZE
        ),
        "litter_stock_change": _litter_panel_summary(host["litter_stock_change"]),
        "seconds": time.perf_counter() - began,
    }


def _tree_numeric_function(jax: Any, jnp: Any) -> Any:
    def summarize(tree: Any) -> Mapping[str, Any]:
        leaves = jax.tree_util.tree_leaves(tree)
        return {
            "all_finite": jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in leaves])),
            "squared_norm": sum(
                jnp.sum(jnp.square(value.astype(jnp.float64))) for value in leaves
            ),
            "maximum_absolute_value": jnp.max(
                jnp.stack([jnp.max(jnp.abs(value)) for value in leaves])
            ),
            "nonzero_leaf_count": sum(jnp.any(value != 0) for value in leaves),
        }

    return jax.jit(summarize)


def _tree_numeric_report(jax: Any, tree: Any, compiled: Any) -> dict[str, Any]:
    host = jax.device_get(compiled(tree))
    leaves = jax.tree_util.tree_leaves(tree)
    return {
        "all_finite": bool(host["all_finite"]),
        "element_count": sum(int(value.size) for value in leaves),
        "leaf_count": len(leaves),
        "nonzero_leaf_count": int(host["nonzero_leaf_count"]),
        "l2_norm": float(np.sqrt(float(host["squared_norm"]))),
        "maximum_absolute_value": float(host["maximum_absolute_value"]),
    }


def _merge_hard_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = {name for record in records for name in record["totals"]}
    totals = {name: sum(int(record["totals"].get(name, 0)) for record in records) for name in names}
    return {"totals": totals, "all_zero": all(value == 0 for value in totals.values())}


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    from research.daily_coarse_graining.daily_operator.fresh_start_training import (
        make_objective,
        make_update,
    )
    from research.daily_coarse_graining.daily_operator.mixed_training_state import (
        load_mixed_training_checkpoint,
    )
    from research.daily_coarse_graining.daily_operator.one_day_training_state import (
        make_gradient_accumulator,
    )
    from research.daily_coarse_graining.daily_operator.prepared_one_day import PreparedOneDayPool
    from scripts.hpc.check_daily_training_runtime import _atomic_json
    from scripts.hpc.mixed_learning_plan import load_plan
    from scripts.hpc.probe_one_day_batch import _compiled_memory, _device_memory
    from scripts.hpc.trained_screen_common import json_safe

    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "stage": "created",
        "claim_scope": "bounded_paired_two_update_daily_only_litter_change_admission",
        "scientific_acceptance": False,
        "full_mixed_admission": False,
        "checkpoint_written": False,
        "arms": {},
        "source_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }

    def emit(stage: str, **values: Any) -> None:
        report.update(values, stage=stage, elapsed_seconds=time.perf_counter() - started)
        _atomic_json(args.output, json_safe(report))
        print(
            f"litter_change_admission stage={stage} "
            f"arms={len(report['arms'])} seconds={report['elapsed_seconds']:.1f}",
            flush=True,
        )

    def reject(reason: str, **values: Any) -> Mapping[str, Any]:
        emit("rejected", status="rejected", reason=reason, **values)
        return report

    try:
        if args.output.exists():
            raise FileExistsError("litter-change admission output already exists")
        if args.shape_only or args.admission_only or args.cpu_forward_only:
            raise ValueError("litter-change admission requires complete GPU execution")
        configuration = _read_json(args.mixed_protocol, name="mixed protocol")
        model_protocol = _read_json(args.protocol, name="model protocol")
        admission_protocol = _read_json(args.admission_protocol, name="admission protocol")
        admitted = _validate_admission_protocol(admission_protocol)
        _validate_model_protocol(model_protocol)
        validate_configuration(configuration, model_protocol)
        configured_protocol = (
            args.mixed_protocol.parent / configuration["model_protocol"]
        ).resolve()
        if configured_protocol != args.protocol.resolve():
            raise ValueError("CLI and mixed model protocols differ")
        execution_policy = configure_execution(args)
        if execution_policy is None:
            raise ValueError("litter-change admission requires the admitted execution policy")
        plan = load_plan(args, configuration)
        batches = _select_training_batches(plan, admission_protocol)
        panels = _panel_rows(plan)
        training_report = _read_json(args.training_report, name="training report")
        identity = endpoints._checkpoint_identity(
            training_report,
            configuration=configuration,
            model_protocol=model_protocol,
            execution_policy=execution_policy,
            plan=plan,
            expected_update=args.expected_update,
        )
        candidate_protocol = _candidate_model_protocol(model_protocol, admission_protocol)
        report.update(
            admission_protocol=admission_protocol,
            admission_protocol_path=str(args.admission_protocol.resolve()),
            mixed_protocol_path=str(args.mixed_protocol.resolve()),
            model_protocol_path=str(args.protocol.resolve()),
            training_report_path=str(args.training_report.resolve()),
            source_checkpoint_path=str(args.checkpoint.resolve()),
            expected_source_update=args.expected_update,
            execution_policy=execution_policy,
            loss_names=list(LOSS_NAMES),
            arm_coefficients={name: list(admitted[name]) for name in ARM_NAMES},
            candidate_model_change={
                "only_added_key": "litter_carbon_stock_change",
                "value": candidate_protocol["litter_carbon_stock_change"],
            },
            selected_training_batches=[batch.as_metadata() for batch in batches],
            panel_rows={name: list(rows) for name, rows in panels.items()},
            panel_classification="TRAIN_development_panels_not_holdout",
        )
        emit("metadata_validated")

        runtime, head, template, scales, fresh_parameters, _, _ = prototype.bootstrap(
            args, model_protocol
        )
        jax = runtime["jax"]
        import jax.numpy as jnp

        devices = tuple(jax.local_devices())
        if not devices or devices[0].platform != "gpu":
            raise ValueError("litter-change admission requires a visible FREE GPU")
        device = devices[0]
        actual_parameter_count = sum(
            int(value.size) for value in jax.tree_util.tree_leaves(fresh_parameters)
        )
        if actual_parameter_count != EXPECTED_PARAMETER_COUNT:
            raise ValueError("actual full-network parameter count differs from the admission contract")
        del fresh_parameters
        gc.collect()

        saved = load_mixed_training_checkpoint(args.checkpoint, **identity)
        optimizer_step = int(np.asarray(saved["optimizer_state"].step))
        if (
            int(saved["completed_updates"]) != args.expected_update
            or saved["daily_sampler_cursor"] != {"epoch": 0, "batch_offset": args.expected_update}
            or optimizer_step != args.expected_update
        ):
            raise ValueError("source checkpoint progress or AdamW step differs from update 4000")
        source_parameters = saved["parameters"]
        source_optimizer = saved["optimizer_state"]
        report.update(
            checkpoint_inherited=True,
            checkpoint_identity={
                "completed_updates": int(saved["completed_updates"]),
                "daily_sampler_cursor": saved["daily_sampler_cursor"],
                "recurrent_schedule_cursor": int(saved["recurrent_schedule_cursor"]),
                "optimizer_step": optimizer_step,
                "method": saved["method"],
                "source_identities": saved["source_identities"],
            },
            parameter_count=actual_parameter_count,
            trainable_parameter_count=actual_parameter_count,
            optimizer=model_protocol["optimizer"],
            device=str(device),
            memory_after_checkpoint_load=_device_memory(device),
        )
        del saved

        graph_protocol = calibration._loss_protocol(
            candidate_protocol,
            canopy_source_pft_index=calibration._canopy_source_pft_index(runtime, head),
        )
        objective = make_objective(runtime, head, template, scales, graph_protocol)
        base_weights = calibration._base_weights(graph_protocol)
        batch_objective = calibration._make_batch_objective(
            jax, jnp, objective, base_weights
        )
        scalar_loss = calibration._make_batch_loss(jax, jnp, objective, base_weights)
        evaluator = jax.jit(jax.value_and_grad(batch_objective, argnums=0, has_aux=True))
        forward = jax.jit(batch_objective)
        update = make_update(model_protocol)
        add_gradient, mean_gradient = make_gradient_accumulator(jax, jnp)
        tree_numeric = _tree_numeric_function(jax, jnp)
        coefficients = {
            name: jax.device_put(jnp.asarray(admitted[name], dtype=jnp.float64), device)
            for name in ARM_NAMES
        }
        litter_only_coefficients = jax.device_put(
            jnp.asarray((0.0, 0.0, 0.0, 1.0), dtype=jnp.float64), device
        )
        pool = PreparedOneDayPool(plan["daily_source"], cache_size=5)
        if pool.identity != plan["catalogue"].get("source_identity"):
            raise ValueError("daily reader and validated source identity differ")

        def sample(rows: Sequence[Any]) -> Any:
            metadata = admission._rows(rows)
            return pool.batch(
                [(int(row["artifact_id"]), int(row["batch_row"])) for row in metadata]
            )

        first_microbatch = batches[0].microbatches[0]
        first_samples = jax.device_put(sample(first_microbatch.rows), device)
        gate_parameters = jax.device_put(source_parameters, device)
        emit("compile_candidate_native_gradient")
        compile_started = time.perf_counter()
        compiled_gradient = evaluator.lower(
            gate_parameters, first_samples, coefficients["candidate"]
        ).compile()
        report.update(
            gradient_compile_seconds=time.perf_counter() - compile_started,
            gradient_compiled_memory=_compiled_memory(compiled_gradient),
            memory_after_gradient_compile=_device_memory(device),
        )
        emit("candidate_native_gradient")
        gate_started = time.perf_counter()
        (gate_value_auxiliary, gate_gradient) = compiled_gradient(
            gate_parameters, first_samples, coefficients["candidate"]
        )
        gate_loss, gate_auxiliary = gate_value_auxiliary
        jax.block_until_ready((gate_loss, gate_auxiliary, gate_gradient))
        gate_numeric = _tree_numeric_report(jax, gate_gradient, tree_numeric)
        gate_hard = calibration._hard_summary(
            jax.device_get(gate_auxiliary["hard"]), batch_size=8
        )
        gate_litter = _litter_panel_summary(
            jax.device_get(gate_auxiliary["litter_stock_change"])
        )
        report["candidate_native_gradient_gate"] = {
            "loss": float(gate_loss),
            "gradient": gate_numeric,
            "hard_counts": gate_hard,
            "litter_stock_change": gate_litter,
            "seconds": time.perf_counter() - gate_started,
        }
        emit("litter_only_gradient")
        (litter_value_auxiliary, litter_gradient) = compiled_gradient(
            gate_parameters, first_samples, litter_only_coefficients
        )
        litter_loss, litter_auxiliary = litter_value_auxiliary
        jax.block_until_ready((litter_loss, litter_auxiliary, litter_gradient))
        litter_numeric = _tree_numeric_report(jax, litter_gradient, tree_numeric)
        report["litter_only_gradient_gate"] = {
            "loss": float(litter_loss),
            "gradient": litter_numeric,
            "passed": bool(
                np.isfinite(float(litter_loss))
                and litter_numeric["all_finite"]
                and litter_numeric["l2_norm"] > 0.0
            ),
        }
        if (
            not np.isfinite(float(gate_loss))
            or not gate_numeric["all_finite"]
            or gate_numeric["l2_norm"] <= 0.0
        ):
            return reject("candidate_native_gradient_invalid")
        if not gate_hard["all_zero"] or gate_litter["hard_invalid_count"] or gate_litter["hard_empty_count"]:
            return reject("candidate_directional_batch_physical_counts")
        if not report["litter_only_gradient_gate"]["passed"]:
            return reject("litter_only_gradient_zero_or_nonfinite")

        def tangent(current: Any, direction: Any, samples: Any, weights: Any) -> Any:
            return jax.jvp(
                lambda candidate: scalar_loss(candidate, samples, weights),
                (current,),
                (direction,),
            )

        dot = jax.jit(lambda left, right: admission._tree_dot(jax, jnp, left, right))
        directions = [
            admission._random_unit_direction(jax, jnp, gate_parameters, seed)
            for seed in admitted["direction_seeds"]
        ]
        reverse_projections = []
        for seed, direction in zip(admitted["direction_seeds"], directions, strict=True):
            value = dot(gate_gradient, direction)
            jax.block_until_ready(value)
            reverse_projections.append(float(value))
            emit(
                "reverse_projection_completed",
                reverse_projection_count=len(reverse_projections),
            )

        emit("compile_candidate_jvp")
        jvp_compile_started = time.perf_counter()
        compiled_jvp = jax.jit(tangent).lower(
            gate_parameters,
            directions[0],
            first_samples,
            coefficients["candidate"],
        ).compile()
        report.update(
            jvp_compile_seconds=time.perf_counter() - jvp_compile_started,
            jvp_compiled_memory=_compiled_memory(compiled_jvp),
            memory_after_jvp_compile=_device_memory(device),
        )
        directional = []
        for seed, direction, reverse_projection in zip(
            admitted["direction_seeds"], directions, reverse_projections, strict=True
        ):
            directional_started = time.perf_counter()
            primal, forward_projection = compiled_jvp(
                gate_parameters,
                direction,
                first_samples,
                coefficients["candidate"],
            )
            jax.block_until_ready((primal, forward_projection))
            record = admission._directional_record(
                primal, reverse_projection, forward_projection, seed=seed
            )
            record["seconds"] = time.perf_counter() - directional_started
            record["primal_matches_reverse_evaluation"] = bool(
                np.isclose(float(primal), float(gate_loss), rtol=1e-5, atol=1e-6)
            )
            record["passed"] = bool(
                record["passed"]
                and record["resolved_projection"]
                and record["primal_matches_reverse_evaluation"]
            )
            directional.append(record)
            emit("directional_check_completed", directional_checks=directional)
        report["bounded_directional_gate_passed"] = all(
            record["passed"] for record in directional
        )
        if not report["bounded_directional_gate_passed"]:
            return reject("candidate_bounded_directional_agreement")
        del direction, directions, reverse_projections, gate_gradient, litter_gradient
        del gate_value_auxiliary, litter_value_auxiliary
        del gate_auxiliary, litter_auxiliary, gate_parameters
        gc.collect()

        for arm_name in ARM_NAMES:
            arm_started = time.perf_counter()
            arm_parameters, arm_optimizer = jax.device_put(
                (source_parameters, source_optimizer), device
            )
            jax.block_until_ready((arm_parameters, arm_optimizer))
            if int(np.asarray(jax.device_get(arm_optimizer.step))) != args.expected_update:
                raise ValueError("paired arm did not restart from AdamW step 4000")
            arm_report: dict[str, Any] = {
                "status": "running",
                "coefficients": list(admitted[arm_name]),
                "source_update": args.expected_update,
                "source_optimizer_step": args.expected_update,
                "completed_updates": 0,
                "updates": [],
                "initial_panels": {},
                "final_panels": {},
            }
            report["arms"][arm_name] = arm_report
            emit(f"{arm_name}_initial_panels", active_arm=arm_name)
            for panel_name in PANEL_NAMES:
                arm_report["initial_panels"][panel_name] = _panel_record(
                    arm_parameters,
                    panels[panel_name],
                    pool,
                    forward,
                    coefficients[arm_name],
                    jax,
                )
                emit(
                    f"{arm_name}_initial_{panel_name}",
                    active_arm=arm_name,
                    active_panel=panel_name,
                )
            if any(
                not panel["hard_counts"]["all_zero"]
                or panel["litter_stock_change"]["hard_invalid_count"]
                or panel["litter_stock_change"]["hard_empty_count"]
                for panel in arm_report["initial_panels"].values()
            ):
                arm_report["status"] = "rejected"
                return reject("initial_panel_physical_counts", active_arm=arm_name)

            for update_offset, parent_batch in enumerate(batches, start=1):
                update_started = time.perf_counter()
                total_gradient = jax.tree_util.tree_map(jnp.zeros_like, arm_parameters)
                weighted_loss = 0.0
                weighted_terms = np.zeros((len(LOSS_NAMES),), dtype=np.float64)
                micro_records = []
                hard_records = []
                for micro_index, microbatch in enumerate(parent_batch.microbatches):
                    samples = jax.device_put(sample(microbatch.rows), device)
                    micro_started = time.perf_counter()
                    (loss_auxiliary, micro_gradient) = compiled_gradient(
                        arm_parameters, samples, coefficients[arm_name]
                    )
                    loss, auxiliary = loss_auxiliary
                    jax.block_until_ready((loss, auxiliary, micro_gradient))
                    host_auxiliary = jax.device_get(auxiliary)
                    hard = calibration._hard_summary(
                        host_auxiliary["hard"], batch_size=microbatch.draw_count
                    )
                    litter = _litter_panel_summary(host_auxiliary["litter_stock_change"])
                    term_means = np.mean(
                        np.asarray(host_auxiliary["terms"], dtype=np.float64), axis=0
                    )
                    record = {
                        "microbatch": micro_index,
                        "rows": admission._rows(microbatch.rows),
                        "loss": float(loss),
                        "mean_component_losses": dict(
                            zip(LOSS_NAMES, term_means.tolist(), strict=True)
                        ),
                        "hard_counts": hard,
                        "litter_hard_invalid_count": litter["hard_invalid_count"],
                        "litter_hard_empty_count": litter["hard_empty_count"],
                        "seconds": time.perf_counter() - micro_started,
                    }
                    micro_records.append(record)
                    hard_records.append(hard)
                    arm_report["active_update"] = update_offset
                    arm_report["active_microbatch"] = micro_index
                    arm_report["last_microbatch"] = record
                    emit(
                        f"{arm_name}_training_microbatch",
                        active_arm=arm_name,
                        active_update=update_offset,
                        active_microbatch=micro_index,
                    )
                    if (
                        not np.isfinite(record["loss"])
                        or not hard["all_zero"]
                        or litter["hard_invalid_count"]
                        or litter["hard_empty_count"]
                    ):
                        arm_report["status"] = "rejected"
                        return reject(
                            "training_microbatch_physical_or_finite_gate",
                            active_arm=arm_name,
                            active_update=update_offset,
                            active_microbatch=micro_index,
                        )
                    total_gradient = add_gradient(
                        total_gradient, micro_gradient, microbatch.draw_count
                    )
                    weighted_loss += float(loss) * microbatch.draw_count
                    weighted_terms += term_means * microbatch.draw_count
                    del samples, auxiliary, micro_gradient

                gradient = mean_gradient(total_gradient, parent_batch.draw_count)
                del total_gradient
                gradient_report = _tree_numeric_report(jax, gradient, tree_numeric)
                if not gradient_report["all_finite"] or gradient_report["l2_norm"] <= 0.0:
                    arm_report["status"] = "rejected"
                    return reject(
                        "accumulated_gradient_invalid",
                        active_arm=arm_name,
                        active_update=update_offset,
                    )
                optimizer_started = time.perf_counter()
                proposed_parameters, proposed_optimizer, norm, clip_scale = update(
                    arm_parameters, gradient, arm_optimizer
                )
                jax.block_until_ready(
                    (proposed_parameters, proposed_optimizer, norm, clip_scale)
                )
                proposal_report = _tree_numeric_report(
                    jax, (proposed_parameters, proposed_optimizer), tree_numeric
                )
                expected_step = args.expected_update + update_offset
                observed_step = int(np.asarray(jax.device_get(proposed_optimizer.step)))
                if (
                    not proposal_report["all_finite"]
                    or not np.isfinite(float(norm))
                    or not np.isfinite(float(clip_scale))
                    or not 0.0 < float(clip_scale) <= 1.0
                    or observed_step != expected_step
                ):
                    arm_report["status"] = "rejected"
                    return reject(
                        "optimizer_proposal_invalid",
                        active_arm=arm_name,
                        active_update=update_offset,
                    )
                arm_parameters, arm_optimizer = proposed_parameters, proposed_optimizer
                del gradient, proposed_parameters, proposed_optimizer
                merged_hard = _merge_hard_counts(hard_records)
                update_record = {
                    "update_in_arm": update_offset,
                    "source_absolute_update": args.expected_update + update_offset,
                    "daily_batch_index": int(parent_batch.batch_index),
                    "mean_loss": weighted_loss / parent_batch.draw_count,
                    "mean_component_losses": dict(
                        zip(
                            LOSS_NAMES,
                            (weighted_terms / parent_batch.draw_count).tolist(),
                            strict=True,
                        )
                    ),
                    "gradient": gradient_report,
                    "gradient_norm_before_clip": float(norm),
                    "clip_threshold": float(model_protocol["optimizer"]["clip_norm"]),
                    "clip_scale": float(clip_scale),
                    "hard_counts": merged_hard,
                    "optimizer_step": observed_step,
                    "optimizer_seconds": time.perf_counter() - optimizer_started,
                    "total_seconds": time.perf_counter() - update_started,
                    "microbatches": micro_records,
                    "memory": _device_memory(device),
                }
                arm_report["updates"].append(update_record)
                arm_report["completed_updates"] = update_offset
                arm_report.pop("active_microbatch", None)
                arm_report.pop("last_microbatch", None)
                emit(
                    f"{arm_name}_update_completed",
                    active_arm=arm_name,
                    active_update=update_offset,
                )

            emit(f"{arm_name}_final_panels", active_arm=arm_name)
            for panel_name in PANEL_NAMES:
                arm_report["final_panels"][panel_name] = _panel_record(
                    arm_parameters,
                    panels[panel_name],
                    pool,
                    forward,
                    coefficients[arm_name],
                    jax,
                )
                emit(
                    f"{arm_name}_final_{panel_name}",
                    active_arm=arm_name,
                    active_panel=panel_name,
                )
            final_panels_clean = all(
                panel["hard_counts"]["all_zero"]
                and panel["litter_stock_change"]["hard_invalid_count"] == 0
                and panel["litter_stock_change"]["hard_empty_count"] == 0
                for panel in arm_report["final_panels"].values()
            )
            if not final_panels_clean:
                arm_report["status"] = "rejected"
                return reject("final_panel_physical_counts", active_arm=arm_name)
            arm_report.update(
                status="completed",
                final_optimizer_step=int(np.asarray(jax.device_get(arm_optimizer.step))),
                elapsed_seconds=time.perf_counter() - arm_started,
            )
            arm_report.pop("active_update", None)
            del arm_parameters, arm_optimizer
            gc.collect()
            emit(f"{arm_name}_completed", active_arm=None, active_panel=None)

        all_updates_complete = all(
            arm["status"] == "completed"
            and arm["completed_updates"] == admission_protocol["updates_per_arm"]
            for arm in report["arms"].values()
        )
        emit(
            "completed",
            status="completed",
            classification="bounded_paired_daily_only_litter_change_technical_admission",
            technical_admission_passed=bool(
                all_updates_complete
                and report["bounded_directional_gate_passed"]
                and report["litter_only_gradient_gate"]["passed"]
            ),
            active_arm=None,
            active_update=None,
            active_microbatch=None,
            active_panel=None,
            reader=pool.statistics(),
            final_memory=_device_memory(device),
            checkpoint_written=False,
            scientific_acceptance=False,
            full_mixed_admission=False,
        )
    except Exception as error:
        if isinstance(error, FileExistsError) and args.output.exists():
            raise
        if not args.output.exists():
            report["output_creation_failed_before_first_emit"] = True
        emit("failed", status="failed", error=f"{type(error).__name__}: {error}")
        raise
    return report


def parser() -> argparse.ArgumentParser:
    result = prototype.parser()
    result.description = __doc__
    for name in (
        "mixed-protocol",
        "catalogue-report",
        "recurrent-schedule",
        "selection",
        "execution-policy",
        "training-report",
        "checkpoint",
        "admission-protocol",
    ):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--expected-update", type=int, default=EXPECTED_UPDATE, choices=(EXPECTED_UPDATE,))
    result.add_argument("--samples-per-panel", type=int, default=PANEL_SIZE, choices=(PANEL_SIZE,))
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
