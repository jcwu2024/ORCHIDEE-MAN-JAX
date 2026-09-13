"""Report train-only gradient geometry for two proposed feedback losses.

This is a fresh-initialization, zero-update diagnostic.  It selects the first
four batch-4 microbatches from epoch 0 of the declared one-day pool and reuses
one coefficient-dynamic native JAX reverse graph for the base, canopy-change,
and litterpart-feedback loss directions.
"""

from __future__ import annotations

import copy
import gzip
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.hpc import run_fresh_start_one_day as prototype

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "feedback_loss_gradient_calibration_v1"
INITIALIZATION_SEED = 20260911
SAMPLING_SEED = 20260912
EXPECTED_PARAMETER_COUNT = 109_464_375
MICROBATCH_SIZE = 4
MICROBATCH_COUNT = 4
MEMORY_LIMIT_BYTES = 24 * 1024**3
LOSS_NAMES = ("base", "canopy_stock_change", "litterpart_feedback")
COEFFICIENT_CANDIDATES = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
CANOPY_STOCK_KEY = "slowproc_stomate_previous_step_state.interception_storage"


def _validate_protocols(configuration: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    sampling = configuration.get("sampling", {})
    expected_sampling = {
        "seed": SAMPLING_SEED,
        "batch_size": 32,
        "microbatch_size": MICROBATCH_SIZE,
        "max_artifacts_per_batch": 4,
        "epochs": 1,
    }
    for name, expected in expected_sampling.items():
        if sampling.get(name) != expected:
            raise ValueError(f"one-day pool sampling {name!r} differs from the calibration contract")
    if sampling.get("scope") != "prepared_one_day_extension_1001_3000":
        raise ValueError("feedback calibration requires the declared broad prepared train pool")
    if protocol.get("schema_version") != "fresh_start_one_day_probe_v3":
        raise ValueError("feedback calibration requires fresh-start protocol v3")
    if int(protocol.get("seed", -1)) != INITIALIZATION_SEED:
        raise ValueError("feedback calibration requires fresh initialization seed 20260911")
    if int(protocol.get("parameter_count", -1)) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("feedback calibration requires the declared full neural parameter count")
    if protocol.get("trainable_scope") != "all_neural_parameters":
        raise ValueError("feedback calibration requires all neural parameters to be trainable")


def _loss_protocol(protocol: Mapping[str, Any], *, canopy_source_pft_index: int) -> dict[str, Any]:
    """Enable both unweighted terms while preserving the existing base loss."""

    result = copy.deepcopy(protocol)
    result["canopy_carbon_stock_change"] = {
        "weight": 1.0,
        "scale_source_pft_index": int(canopy_source_pft_index),
    }
    result["loss"]["litterpart_feedback_mse"] = 1.0
    return result


def _canopy_source_pft_index(runtime: Mapping[str, Any], head: Any) -> int:
    """Resolve the sole active vegetation source PFT through the compact contract."""

    from research.daily_coarse_graining.daily_operator.parameter_routing import (
        PFT_TRAIT_BARE_SOIL,
    )

    leaves = tuple(leaf for leaf in head.contract.state_leaves if leaf.key == CANOPY_STOCK_KEY)
    if len(leaves) != 1:
        raise ValueError("canopy stock owner is not unique")
    source_indices = tuple(int(value) for value in leaves[0].selected_pft_indices)
    assembled = runtime["probe_assembled"]
    active = np.asarray(assembled.pft.active_pft_mask, dtype=np.bool_)
    traits = np.asarray(assembled.pft.pft_traits, dtype=np.float64)
    if active.shape != (len(source_indices),) or traits.shape[0] != len(source_indices):
        raise ValueError("compact PFT contract and bootstrap PFT inputs differ")
    bare = traits[:, PFT_TRAIT_BARE_SOIL] > 0.5
    vegetation = tuple(
        source for source, eligible in zip(source_indices, active & ~bare, strict=True) if eligible
    )
    if vegetation != (13,):
        raise ValueError(f"calibration requires the PFT14-only vegetation contract, got {vegetation!r}")
    return vegetation[0]


def _make_evaluator(
    jax: Any, jnp: Any, objective: Any, base_weights: Mapping[str, float]
) -> Any:
    """Differentiate only parameters; the data and three coefficients are dynamic."""

    def one(parameters: Any, sample: Any, coefficients: Any) -> tuple[Any, Any]:
        _, auxiliary = objective(parameters, sample)
        values = auxiliary
        terms = jnp.stack(
            (
                base_weights["independent_daily_flux_mse"] * values["flux_loss"]
                + base_weights["one_day_state_anchor"] * values["state_loss"]
                + base_weights["necessary_process_supervision"] * values["process_loss"],
                values["canopy_stock_change"]["loss"],
                values["litterpart_feedback"]["loss"],
            )
        )
        selected_auxiliary = {
            "terms": terms,
            "hard": values["hard"],
            "physical": values["physical"],
            "canopy_stock_change": values["canopy_stock_change"],
            "litterpart_feedback": values["litterpart_feedback"],
        }
        return jnp.vdot(coefficients, terms), selected_auxiliary

    def batch(parameters: Any, samples: Any, coefficients: Any) -> tuple[Any, Any]:
        losses, auxiliary = jax.vmap(one, in_axes=(None, 0, None))(
            parameters, samples, coefficients
        )
        return jnp.mean(losses), auxiliary

    return jax.jit(jax.value_and_grad(batch, argnums=0, has_aux=True))


def _base_weights(protocol: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: float(protocol["loss"][name])
        for name in (
            "independent_daily_flux_mse",
            "one_day_state_anchor",
            "necessary_process_supervision",
        )
    }


def _hard_summary(hard: Mapping[str, Any], *, batch_size: int) -> dict[str, Any]:
    values = {name: np.asarray(value) for name, value in hard.items()}
    if any(value.shape != (batch_size,) for value in values.values()):
        raise ValueError("vmapped hard-count outputs do not have the expected batch axis")
    per_sample = tuple(
        {name: int(value[index]) for name, value in values.items()}
        for index in range(batch_size)
    )
    totals = {name: int(sum(row[name] for row in per_sample)) for name in values}
    return {
        "per_sample": per_sample,
        "totals": totals,
        "all_zero": all(value == 0 for value in totals.values()),
    }


def _gradient_reports_and_geometry(
    jax: Any, gradients: Sequence[Any], names: Sequence[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure complete trees and pairwise geometry without retaining host copies."""

    leaves = tuple(tuple(jax.tree_util.tree_leaves(value)) for value in gradients)
    if len(leaves) != len(names) or not leaves or any(len(row) != len(leaves[0]) for row in leaves):
        raise ValueError("gradient tree registries differ across loss groups")
    statistics = {
        name: {
            "all_finite": True,
            "element_count": 0,
            "leaf_count": len(leaves[index]),
            "nonzero_leaf_count": 0,
            "squared_norm": 0.0,
            "maximum_absolute_value": 0.0,
        }
        for index, name in enumerate(names)
    }
    pair_dots = {(left, right): 0.0 for left in range(len(names)) for right in range(left + 1, len(names))}
    for leaf_values in zip(*leaves, strict=True):
        arrays = tuple(np.asarray(value, dtype=np.float64) for value in leaf_values)
        shape = arrays[0].shape
        if any(array.shape != shape for array in arrays[1:]):
            raise ValueError("gradient leaf shapes differ across loss groups")
        for index, (name, array) in enumerate(zip(names, arrays, strict=True)):
            row = statistics[name]
            row["all_finite"] = bool(row["all_finite"] and np.all(np.isfinite(array)))
            row["element_count"] += int(array.size)
            row["nonzero_leaf_count"] += int(bool(np.any(array != 0.0)))
            row["squared_norm"] += float(np.sum(array * array, dtype=np.float64))
            if array.size:
                row["maximum_absolute_value"] = max(
                    row["maximum_absolute_value"], float(np.max(np.abs(array)))
                )
        for (left, right) in pair_dots:
            pair_dots[(left, right)] += float(
                np.sum(arrays[left] * arrays[right], dtype=np.float64)
            )
    reports: dict[str, Any] = {}
    for name in names:
        row = statistics[name]
        reports[name] = {
            key: value for key, value in row.items() if key != "squared_norm"
        }
        reports[name]["l2_norm"] = math.sqrt(row["squared_norm"])
    cross = {}
    for (left, right), dot in pair_dots.items():
        left_name, right_name = names[left], names[right]
        denominator = reports[left_name]["l2_norm"] * reports[right_name]["l2_norm"]
        cross[f"{left_name}__{right_name}"] = {
            "dot": dot,
            "cosine": dot / denominator if denominator > 0.0 else None,
        }
    return reports, cross


def _coefficient_ratios(base_norm: float, supplemental_norm: float) -> list[dict[str, float]]:
    denominator = max(float(base_norm), 1e-30)
    return [
        {
            "coefficient": coefficient,
            "q": coefficient * float(supplemental_norm) / denominator,
        }
        for coefficient in COEFFICIENT_CANDIDATES
    ]


def _candidate_q_summary(batches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for term in LOSS_NAMES[1:]:
        records = []
        for candidate_index, coefficient in enumerate(COEFFICIENT_CANDIDATES):
            values = [
                float(batch["candidate_q"][term][candidate_index]["q"]) for batch in batches
            ]
            records.append(
                {
                    "coefficient": coefficient,
                    "batch_q": values,
                    "minimum_q": float(np.min(values)),
                    "median_q": float(np.median(values)),
                    "maximum_q": float(np.max(values)),
                }
            )
        result[term] = records
    return result


def _finite_statistics(values: Any, valid: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    mask = np.broadcast_to(np.asarray(valid, dtype=np.bool_), array.shape)
    nonfinite_valid = int(np.count_nonzero(mask & ~np.isfinite(array)))
    selected = array[mask & np.isfinite(array)]
    if selected.size == 0:
        return {"count": 0, "nonfinite_valid": nonfinite_valid, "mean": None, "rmse": None}
    return {
        "count": int(selected.size),
        "nonfinite_valid": nonfinite_valid,
        "mean": float(np.mean(selected)),
        "rmse": float(np.sqrt(np.mean(selected * selected))),
    }


def _diagnostics(jax: Any, auxiliary: Mapping[str, Any], *, batch_size: int) -> dict[str, Any]:
    from research.daily_coarse_graining.daily_operator.one_day_monitoring import summarize_monitor

    host = jax.device_get(auxiliary)
    physical = tuple(
        jax.tree_util.tree_map(lambda value, index=index: np.asarray(value)[index], host["physical"])
        for index in range(batch_size)
    )
    canopy = host["canopy_stock_change"]
    canopy_valid = np.asarray(canopy["row_valid"], dtype=np.bool_) & np.asarray(
        canopy["scale_valid"], dtype=np.bool_
    )
    litter = host["litterpart_feedback"]
    litter_valid = np.asarray(litter["valid"], dtype=np.bool_)
    return {
        "physical": summarize_monitor(physical),
        "canopy_stock_change": {
            "unit": "gC_m-2_day-1",
            "scale": np.asarray(canopy["scale"], dtype=np.float64).tolist(),
            "scale_source_pft_index": np.asarray(
                canopy["scale_source_pft_index"], dtype=np.int64
            ).tolist(),
            "row_valid_count": int(np.count_nonzero(canopy_valid)),
            "active_slot_count": int(np.sum(np.asarray(canopy["active_slot_count"]))),
            "bare_active_slot_count": int(np.sum(np.asarray(canopy["bare_active_slot_count"]))),
            "teacher_delta": _finite_statistics(canopy["teacher_delta"], canopy_valid),
            "predicted_delta": _finite_statistics(canopy["predicted_delta"], canopy_valid),
            "physical_residual": _finite_statistics(canopy["physical_residual"], canopy_valid),
            "normalized_residual": _finite_statistics(canopy["normalized_residual"], canopy_valid),
            "bare_teacher_delta": _finite_statistics(canopy["bare_teacher_delta"], canopy_valid),
            "bare_predicted_delta": _finite_statistics(canopy["bare_predicted_delta"], canopy_valid),
            "bare_physical_residual": _finite_statistics(
                canopy["bare_physical_residual"], canopy_valid
            ),
        },
        "litterpart_feedback": {
            "unit": "litterpart_state_units",
            "valid_coordinate_count": int(np.count_nonzero(litter_valid)),
            "expected_coordinate_count": int(np.sum(np.asarray(litter["expected_count"]))),
            "invalid_coordinate_count": int(np.sum(np.asarray(litter["invalid_count"]))),
            "physical_residual": _finite_statistics(litter["physical_residual"], litter_valid),
            "normalized_residual": _finite_statistics(litter["normalized_residual"], litter_valid),
            "predicted": _finite_statistics(litter["predicted"], litter_valid),
            "teacher": _finite_statistics(litter["teacher"], litter_valid),
        },
    }


def _row_metadata(rows: Sequence[Any]) -> list[dict[str, Any]]:
    return [row.as_metadata() for row in rows]


def run(args: Any) -> Mapping[str, Any]:
    from scripts.hpc.probe_one_day_batch import _compiled_memory, _device_memory, _resolved_reference
    from scripts.hpc.run_fresh_start_one_day_pool import pool_plan
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json

    started = time.perf_counter()
    report: dict[str, Any] = {
        "status": "running",
        "schema_version": SCHEMA_VERSION,
        "claim_scope": "train_only_zero_update_gradient_geometry_report",
        "training_updates": 0,
        "optimizer_created": False,
        "checkpoint_inherited": False,
        "checkpoint_written": False,
        "coefficient_selected": None,
        "candidate_coefficients": list(COEFFICIENT_CANDIDATES),
        "compiled_memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "source_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }

    def emit(stage: str, **values: Any) -> None:
        report.update(values, stage=stage, elapsed_seconds=time.perf_counter() - started)
        _atomic_json(args.output, report)
        print(f"feedback_loss_calibration stage={stage} seconds={report['elapsed_seconds']:.1f}", flush=True)

    try:
        if args.admission_only or args.cpu_forward_only:
            raise ValueError("feedback calibration supports full execution or --shape-only")
        configuration = json.loads(args.pool_protocol.read_text())
        protocol = json.loads(args.protocol.read_text())
        _validate_protocols(configuration, protocol)
        configured_protocol = _resolved_reference(
            configuration["model_protocol"], relative_to=args.pool_protocol.resolve()
        )
        if configured_protocol != args.protocol.resolve():
            raise ValueError("pool and CLI model protocols differ")
        report.update(pool_configuration=configuration, base_model_protocol=protocol)
        emit("read_catalogue")
        catalogue_report = json.loads(args.catalogue_report.read_text())
        if catalogue_report.get("status") != "passed":
            raise ValueError("one-day catalogue report is not accepted")
        if catalogue_report.get("schema_version") != "prepared_one_day_catalogue_v1":
            raise ValueError("one-day catalogue report schema drift")
        catalogue_path = _resolved_reference(
            catalogue_report["catalogue"], relative_to=args.catalogue_report.resolve()
        )
        with gzip.open(catalogue_path, "rt", encoding="utf-8") as handle:
            catalogue = json.load(handle)
        sampler, _ = pool_plan(catalogue, configuration)
        first_batch = next(sampler.iter_epoch(0))
        microbatches = tuple(first_batch.microbatches[:MICROBATCH_COUNT])
        if len(microbatches) != MICROBATCH_COUNT or any(
            batch.draw_count != MICROBATCH_SIZE for batch in microbatches
        ):
            raise ValueError("epoch-0 pool does not begin with four complete batch-4 microbatches")
        selected_rows = tuple(row for batch in microbatches for row in batch.rows)
        report.update(
            sampling={
                "seed": SAMPLING_SEED,
                "epoch": 0,
                "parent_batch_index": first_batch.batch_index,
                "parent_batch_size": first_batch.draw_count,
                "microbatch_size": MICROBATCH_SIZE,
                "microbatch_count": MICROBATCH_COUNT,
                "row_count": len(selected_rows),
                "rows": _row_metadata(selected_rows),
            },
            protocol_path=str(args.protocol.resolve()),
            pool_protocol_path=str(args.pool_protocol.resolve()),
            catalogue_report_path=str(args.catalogue_report.resolve()),
            catalogue_schema_version=catalogue.get("schema_version"),
            catalogue_source_identity=catalogue.get("source_identity"),
        )

        emit("bootstrap")
        runtime, head, template, scales, parameters, _, _ = prototype.bootstrap(args, protocol)
        jax = runtime["jax"]
        import jax.numpy as jnp

        from research.daily_coarse_graining.daily_operator.fresh_start_training import make_objective
        from research.daily_coarse_graining.daily_operator.prepared_one_day import PreparedOneDayPool

        parameter_count = sum(int(value.size) for value in jax.tree_util.tree_leaves(parameters))
        if parameter_count != EXPECTED_PARAMETER_COUNT:
            raise ValueError("fresh neural parameter count differs from the calibration contract")
        canopy_source_pft_index = _canopy_source_pft_index(runtime, head)
        graph_protocol = _loss_protocol(protocol, canopy_source_pft_index=canopy_source_pft_index)
        report["graph_model_protocol"] = graph_protocol
        objective = make_objective(runtime, head, template, scales, graph_protocol)
        evaluator = _make_evaluator(
            jax, jnp, objective, _base_weights(graph_protocol)
        )
        source_manifest = _resolved_reference(catalogue["source_manifest"], relative_to=catalogue_path)
        pool = PreparedOneDayPool(
            source_manifest, cache_size=configuration["sampling"]["max_artifacts_per_batch"]
        )
        if pool.identity != catalogue.get("source_identity"):
            raise ValueError("catalogue and prepared-pool source identities differ")
        batches = tuple(
            pool.batch([(row.artifact_id, row.batch_row) for row in batch.rows])
            for batch in microbatches
        )
        directions = tuple(np.eye(len(LOSS_NAMES), dtype=np.float64))
        report.update(
            initialization_seed=INITIALIZATION_SEED,
            parameter_count=parameter_count,
            trainable_scope="all_neural_parameters",
            canopy_scale_source_pft_index=canopy_source_pft_index,
            graph_loss_weights={
                "canopy_stock_change": 1.0,
                "litterpart_feedback": 1.0,
            },
            coefficient_directions={
                name: direction.tolist() for name, direction in zip(LOSS_NAMES, directions, strict=True)
            },
            reader=pool.statistics(),
            device=str(runtime["devices"][0]),
            memory_after_bootstrap=_device_memory(runtime["devices"][0]),
        )

        first_direction = jnp.asarray(directions[0], dtype=jnp.float64)
        if args.shape_only:
            emit("trace_objective_and_backward")
            jax.eval_shape(evaluator, parameters, batches[0], first_direction)
            emit(
                "completed",
                status="completed",
                classification="actual_objective_and_parameter_backward_shapes_only",
                report_only=True,
                coefficient_selected=None,
            )
            return report

        emit("compile_objective_and_backward")
        compile_started = time.perf_counter()
        compiled = evaluator.lower(parameters, batches[0], first_direction).compile()
        compile_seconds = time.perf_counter() - compile_started
        memory = _compiled_memory(compiled)
        emit(
            "compiled",
            compile_seconds=compile_seconds,
            compiled_memory=memory,
            memory_after_compile=_device_memory(runtime["devices"][0]),
        )
        if memory["estimated_live_bytes"] > MEMORY_LIMIT_BYTES:
            emit(
                "stopped",
                status="resource_limited",
                reason="compiled_gradient_memory_exceeds_24_gib",
            )
            return report

        report["batches"] = []
        for batch_index, (selection, samples) in enumerate(zip(microbatches, batches, strict=True)):
            emit("evaluate_batch", active_batch=batch_index)
            gradients = []
            batch_record: dict[str, Any] = {
                "batch_index": batch_index,
                "rows": _row_metadata(selection.rows),
                "groups": [],
                "completed": False,
            }
            report["batches"].append(batch_record)
            for name, direction in zip(LOSS_NAMES, directions, strict=True):
                began = time.perf_counter()
                (loss_and_auxiliary, gradient) = compiled(
                    parameters, samples, jnp.asarray(direction, dtype=jnp.float64)
                )
                loss, auxiliary = loss_and_auxiliary
                jax.block_until_ready((loss, auxiliary, gradient))
                gradient_report, _ = _gradient_reports_and_geometry(jax, (gradient,), (name,))
                batch_record["groups"].append(
                    {
                        "name": name,
                        "loss": float(loss),
                        "individual_term_losses": np.asarray(
                            auxiliary["terms"][:, LOSS_NAMES.index(name)], dtype=np.float64
                        ).tolist(),
                        "hard_counts": _hard_summary(
                            jax.device_get(auxiliary["hard"]), batch_size=selection.draw_count
                        ),
                        "gradient": gradient_report[name],
                        "seconds": time.perf_counter() - began,
                    }
                )
                gradients.append(gradient)
                emit(
                    "gradient_completed",
                    active_batch=batch_index,
                    active_group=name,
                    memory_after_gradient=_device_memory(runtime["devices"][0]),
                )
            gradient_reports, cross = _gradient_reports_and_geometry(jax, gradients, LOSS_NAMES)
            for evaluation in batch_record["groups"]:
                evaluation["gradient"] = gradient_reports[evaluation["name"]]
            batch_record.update(
                gradient_cross_geometry=cross,
                candidate_q={
                    "canopy_stock_change": _coefficient_ratios(
                        gradient_reports["base"]["l2_norm"],
                        gradient_reports["canopy_stock_change"]["l2_norm"],
                    ),
                    "litterpart_feedback": _coefficient_ratios(
                        gradient_reports["base"]["l2_norm"],
                        gradient_reports["litterpart_feedback"]["l2_norm"],
                    ),
                },
                diagnostics=_diagnostics(jax, auxiliary, batch_size=selection.draw_count),
                memory=_device_memory(runtime["devices"][0]),
                completed=True,
            )
            del gradients, auxiliary, gradient
            emit("batch_completed", active_batch=None, active_group=None, reader=pool.statistics())

        all_groups = [group for batch in report["batches"] for group in batch["groups"]]
        emit(
            "completed",
            status="completed",
            classification="report_only_train_feedback_loss_gradient_calibration_no_selection",
            report_only=True,
            coefficient_selected=None,
            candidate_q_summary=_candidate_q_summary(report["batches"]),
            all_reported_losses_finite=all(np.isfinite(group["loss"]) for group in all_groups),
            all_gradient_values_finite=all(group["gradient"]["all_finite"] for group in all_groups),
            all_hard_counts_zero=all(group["hard_counts"]["all_zero"] for group in all_groups),
            reader=pool.statistics(),
            final_memory=_device_memory(runtime["devices"][0]),
        )
    except Exception as error:
        emit("failed", status="failed", error=f"{type(error).__name__}: {error}")
        raise
    return report


def parser() -> Any:
    result = prototype.parser()
    result.description = __doc__
    result.set_defaults(
        protocol=ROOT / "manifests/coarse_graining/fresh_start_one_day_probe_v3.json"
    )
    result.add_argument(
        "--pool-protocol",
        type=Path,
        default=ROOT / "manifests/coarse_graining/fresh_start_one_day_pool_v1.json",
    )
    result.add_argument("--catalogue-report", type=Path, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
