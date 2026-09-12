"""Fresh batch-4 one-day gradient/resource equivalence; no training."""

from __future__ import annotations

import gzip
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.hpc import run_fresh_start_one_day as training

ROOT = Path(__file__).resolve().parents[2]
BATCH_SIZE = 4
MEMORY_LIMIT_BYTES = 18 * 1024**3
EXPECTED_SEED = 20260911
EXPECTED_PARAMETER_COUNT = 109_464_375
EXPECTED_ROWS = (
    (1400, 14, "279.0-095.0", 1994, 71),
    (1066, 10, "249.0-113.0", 2000, 163),
    (1678, 31, "009.0-087.0", 1984, 207),
    (2216, 2, "119.0-107.0", 1993, 74),
)


def _resolved_reference(value: str, *, relative_to: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidates = (
        ROOT / path,
        relative_to.parent / path,
        relative_to.parent / path.name,
    )
    return next((candidate.resolve() for candidate in candidates if candidate.exists()), candidates[0].resolve())


def catalogue_selection(
    report_path: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Path, tuple[Mapping[str, Any], ...]]:
    """Read the accepted catalogue and return its first four exact row records."""

    report_path = report_path.resolve()
    report = json.loads(report_path.read_text())
    if report.get("status") != "passed":
        raise ValueError("one-day catalogue report is not accepted")
    if report.get("schema_version") != "prepared_one_day_catalogue_v1":
        raise ValueError("one-day catalogue report schema drift")
    catalogue_path = _resolved_reference(report["catalogue"], relative_to=report_path)
    with gzip.open(catalogue_path, "rt", encoding="utf-8") as handle:
        catalogue = json.load(handle)
    source_manifest = _resolved_reference(
        catalogue["source_manifest"], relative_to=catalogue_path
    )
    rows = tuple(report.get("first_batch", {}).get("rows", ()))
    if len(rows) < BATCH_SIZE:
        raise ValueError("one-day catalogue first batch contains fewer than four rows")
    selected = rows[:BATCH_SIZE]
    if any(not isinstance(row, Mapping) for row in selected):
        raise TypeError("one-day catalogue rows must be mappings")
    for row in selected:
        artifact_id, batch_row = row.get("artifact_id"), row.get("batch_row")
        if isinstance(artifact_id, bool) or not isinstance(artifact_id, int):
            raise TypeError("one-day catalogue artifact IDs must be integers")
        if isinstance(batch_row, bool) or not isinstance(batch_row, int):
            raise TypeError("one-day catalogue batch rows must be integers")
    observed = tuple(
        (
            row["artifact_id"],
            row["batch_row"],
            row.get("landpoint_id"),
            row.get("year"),
            row.get("start_row"),
        )
        for row in selected
    )
    if observed != EXPECTED_ROWS:
        raise ValueError("one-day catalogue first four rows differ from the frozen probe")
    return report, catalogue, source_manifest, selected


def gradient_evaluators(jax: Any, jnp: Any, objective: Any) -> tuple[Any, Any]:
    """Build single-row and vmapped mean-loss gradients with dynamic data args."""

    def one(parameters: Any, sample: Any) -> tuple[Any, Any]:
        loss, auxiliary = objective(parameters, sample)
        return loss, auxiliary["hard"]

    def batch(parameters: Any, samples: Any) -> tuple[Any, Any]:
        losses, hard = jax.vmap(one, in_axes=(None, 0))(parameters, samples)
        return jnp.mean(losses), {"individual_losses": losses, "hard": hard}

    single = jax.jit(jax.value_and_grad(one, has_aux=True))
    vmapped = jax.jit(jax.value_and_grad(batch, has_aux=True))
    return single, vmapped


def _compiled_memory(compiled: Any) -> Mapping[str, int]:
    analysis = compiled.memory_analysis()
    names = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
    )
    sizes = {name: int(getattr(analysis, name, 0) or 0) for name in names}
    sizes["estimated_live_bytes"] = max(
        0,
        sizes["argument_size_in_bytes"]
        + sizes["output_size_in_bytes"]
        + sizes["temp_size_in_bytes"]
        - sizes["alias_size_in_bytes"],
    )
    return sizes


def _device_memory(device: Any) -> Mapping[str, Any]:
    try:
        statistics = device.memory_stats() or {}
    except Exception as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}
    selected = {
        name: int(value) if isinstance(value, (int, np.integer)) else value
        for name, value in statistics.items()
        if name
        in {
            "bytes_in_use",
            "peak_bytes_in_use",
            "pool_bytes",
            "peak_pool_bytes",
            "bytes_limit",
            "num_allocs",
        }
    }
    return {"available": bool(selected), **selected}


def _hard_rows(hard: Mapping[str, Any]) -> tuple[Mapping[str, int], ...]:
    arrays = {name: np.asarray(value) for name, value in hard.items()}
    lengths = {int(value.shape[0]) for value in arrays.values() if value.ndim > 0}
    if lengths != {BATCH_SIZE}:
        raise ValueError("vmapped physical-hard outputs do not have a batch-4 axis")
    return tuple(
        {name: int(value[index]) for name, value in arrays.items()}
        for index in range(BATCH_SIZE)
    )


def _hard_summary(rows: tuple[Mapping[str, int], ...]) -> Mapping[str, Any]:
    names = tuple(rows[0])
    if any(tuple(row) != names for row in rows):
        raise ValueError("physical-hard registries differ across samples")
    totals = {name: sum(row[name] for row in rows) for name in names}
    return {
        "per_sample": rows,
        "totals": totals,
        "all_zero": all(value == 0 for value in totals.values()),
    }


def _single_hard_row(hard: Mapping[str, Any]) -> Mapping[str, int]:
    return {name: int(np.asarray(value)) for name, value in hard.items()}


def _add_gradients(jax: Any, left: Any, right: Any) -> Any:
    return jax.tree_util.tree_map(lambda a, b: a + b, left, right)


def _scale_gradient(jax: Any, gradient: Any, scale: float) -> Any:
    return jax.tree_util.tree_map(lambda value: value * scale, gradient)


def run(args: Any) -> Mapping[str, Any]:
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json

    started = time.perf_counter()
    report: dict[str, Any] = {
        "status": "running",
        "schema_version": "fresh_one_day_batch4_probe_v1",
        "training_updates": 0,
        "batch_size": BATCH_SIZE,
        "checkpoint_inherited": False,
        "compiled_memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "claim_scope": "batching_and_resource_equivalence_only_no_fd_certification",
    }

    def emit(stage: str, **values: Any) -> None:
        report.update(values, stage=stage, elapsed_seconds=time.perf_counter() - started)
        _atomic_json(args.output, report)
        print(f"one_day_batch stage={stage} seconds={report['elapsed_seconds']:.1f}", flush=True)

    try:
        if args.admission_only or args.cpu_forward_only:
            raise ValueError("one-day batch probe supports only full execution or --shape-only")
        emit("read_catalogue")
        catalogue_report, catalogue, source_manifest, selected_rows = catalogue_selection(
            args.catalogue_report
        )
        protocol = json.loads(args.protocol.read_text())
        if protocol.get("schema_version") != "fresh_start_one_day_probe_v3":
            raise ValueError("one-day batch probe requires fresh-start protocol v3")
        if int(protocol.get("seed", -1)) != EXPECTED_SEED:
            raise ValueError("one-day batch probe requires the frozen fresh-start seed")
        if int(protocol.get("parameter_count", -1)) != EXPECTED_PARAMETER_COUNT:
            raise ValueError("one-day batch probe requires the frozen full parameter count")
        if protocol.get("trainable_scope") != "all_neural_parameters":
            raise ValueError("one-day batch probe requires the full fresh neural parameter tree")
        catalogue_path = _resolved_reference(
            catalogue_report["catalogue"], relative_to=args.catalogue_report.resolve()
        )
        report.update(
            source_git_head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            protocol=str(args.protocol.resolve()),
            catalogue_report=str(args.catalogue_report.resolve()),
            catalogue=str(catalogue_path),
            catalogue_source_manifest=str(source_manifest),
            catalogue_source_identity=catalogue.get("source_identity"),
            first_four_rows=[dict(row) for row in selected_rows],
        )

        emit("bootstrap")
        runtime, head, template, scales, parameters, _samples, _metadata = training.bootstrap(
            args, protocol
        )
        jax = runtime["jax"]
        import jax.numpy as jnp

        from research.daily_coarse_graining.daily_operator.fresh_start_training import make_objective
        from research.daily_coarse_graining.daily_operator.prepared_one_day import PreparedOneDayPool
        from scripts.hpc.probe_public_rollout import gradient_comparison
        from scripts.hpc.run_successor_fast_owner_gpu_probe import _tree_numeric_report

        parameter_count = sum(int(value.size) for value in jax.tree_util.tree_leaves(parameters))
        if parameter_count != int(protocol["parameter_count"]):
            raise ValueError("fresh parameter count differs from protocol")

        pool = PreparedOneDayPool(source_manifest, cache_size=BATCH_SIZE)
        if pool.identity != catalogue.get("source_identity"):
            raise ValueError("catalogue and prepared-pool source identities differ")
        row_keys = tuple(
            (int(row["artifact_id"]), int(row["batch_row"])) for row in selected_rows
        )
        samples = pool.batch(row_keys)
        day_indices = np.asarray(samples.inputs.day_index)
        if day_indices.shape[0] != BATCH_SIZE:
            raise ValueError("materialized one-day payload does not have four rows")
        row_metadata = []
        for index, row in enumerate(selected_rows):
            prepared_day_index = int(np.asarray(day_indices[index]).item())
            declared = row.get("prepared_day_index")
            if declared is not None and int(declared) != prepared_day_index:
                raise ValueError("catalogue and payload prepared-day indices differ")
            row_metadata.append({**row, "prepared_day_index": prepared_day_index})
        report.update(
            parameter_count=parameter_count,
            first_four_rows=row_metadata,
            reader=pool.statistics(),
            device=str(runtime["devices"][0]),
            memory_after_bootstrap=_device_memory(runtime["devices"][0]),
        )

        objective = make_objective(runtime, head, template, scales, protocol)
        single_evaluator, batch_evaluator = gradient_evaluators(jax, jnp, objective)
        first_sample = jax.tree_util.tree_map(lambda value: value[0], samples)
        if args.shape_only:
            emit("trace_gradients")
            jax.eval_shape(single_evaluator, parameters, first_sample)
            jax.eval_shape(batch_evaluator, parameters, samples)
            emit(
                "completed",
                status="passed",
                classification="fresh_batch4_and_single_gradient_shapes_only",
                finite_difference_certified=False,
            )
            return report

        emit("compile_gradients")
        compile_started = time.perf_counter()
        compiled_batch = batch_evaluator.lower(parameters, samples).compile()
        batch_compile_seconds = time.perf_counter() - compile_started
        compile_started = time.perf_counter()
        compiled_single = single_evaluator.lower(parameters, first_sample).compile()
        single_compile_seconds = time.perf_counter() - compile_started
        compiled_memory = {
            "batch4": _compiled_memory(compiled_batch),
            "single": _compiled_memory(compiled_single),
        }
        emit(
            "compiled_memory",
            compile_seconds={"batch4": batch_compile_seconds, "single": single_compile_seconds},
            compiled_memory=compiled_memory,
            compiled_memory_limit_gib=18,
            memory_after_compile=_device_memory(runtime["devices"][0]),
        )
        if max(value["estimated_live_bytes"] for value in compiled_memory.values()) > MEMORY_LIMIT_BYTES:
            emit(
                "stopped",
                status="resource_limited",
                reason="compiled_gradient_memory_exceeds_18_gib",
            )
            return report

        emit("batch_cold_evaluation")
        evaluation_started = time.perf_counter()
        (cold_loss, cold_auxiliary), cold_gradient = compiled_batch(parameters, samples)
        jax.block_until_ready((cold_loss, cold_auxiliary, cold_gradient))
        cold_seconds = time.perf_counter() - evaluation_started
        cold_loss_value = float(cold_loss)
        cold_hard = _hard_rows(cold_auxiliary["hard"])
        del cold_gradient, cold_auxiliary

        emit("batch_hot_evaluation")
        evaluation_started = time.perf_counter()
        (batch_loss, batch_auxiliary), batch_gradient = compiled_batch(parameters, samples)
        jax.block_until_ready((batch_loss, batch_auxiliary, batch_gradient))
        hot_seconds = time.perf_counter() - evaluation_started
        batch_loss_value = float(batch_loss)
        batch_individual_losses = np.asarray(
            batch_auxiliary["individual_losses"], dtype=np.float64
        )
        batch_hard_rows = _hard_rows(batch_auxiliary["hard"])
        memory_after_batch = _device_memory(runtime["devices"][0])

        emit("single_gradients")
        single_losses = []
        single_hard_rows = []
        single_gradient_sum = None
        single_seconds = []
        for index in range(BATCH_SIZE):
            sample = jax.tree_util.tree_map(lambda value: value[index], samples)
            evaluation_started = time.perf_counter()
            (loss, hard), gradient = compiled_single(parameters, sample)
            jax.block_until_ready((loss, hard, gradient))
            single_seconds.append(time.perf_counter() - evaluation_started)
            single_losses.append(float(loss))
            single_hard_rows.append(_single_hard_row(hard))
            if single_gradient_sum is None:
                single_gradient_sum = gradient
            else:
                single_gradient_sum = _add_gradients(jax, single_gradient_sum, gradient)
                jax.block_until_ready(single_gradient_sum)
            del gradient
        mean_single_gradient = _scale_gradient(
            jax, single_gradient_sum, 1.0 / BATCH_SIZE
        )
        jax.block_until_ready(mean_single_gradient)
        del single_gradient_sum

        batch_numeric = _tree_numeric_report(jax, batch_gradient)
        mean_single_numeric = _tree_numeric_report(jax, mean_single_gradient)
        comparison = gradient_comparison(jax, batch_gradient, mean_single_gradient)
        norm_difference = abs(batch_numeric["l2_norm"] - mean_single_numeric["l2_norm"])
        comparison.update(
            batch_l2_norm=batch_numeric["l2_norm"],
            mean_single_l2_norm=mean_single_numeric["l2_norm"],
            global_norm_absolute_difference=norm_difference,
            global_norm_relative_difference=norm_difference
            / max(mean_single_numeric["l2_norm"], np.finfo(np.float64).tiny),
        )
        hard_summary = _hard_summary(batch_hard_rows)
        single_hard_summary = _hard_summary(tuple(single_hard_rows))
        individual_losses_agree = bool(
            np.allclose(batch_individual_losses, single_losses, atol=1e-6, rtol=1e-5)
        )
        mean_loss_agrees = bool(
            np.isclose(batch_loss_value, np.mean(single_losses), atol=1e-6, rtol=1e-5)
        )
        cold_hot_agree = bool(
            np.isclose(cold_loss_value, batch_loss_value, atol=1e-8, rtol=1e-10)
            and cold_hard == batch_hard_rows
        )
        hard_agrees = bool(batch_hard_rows == tuple(single_hard_rows))
        passed = bool(
            cold_hot_agree
            and individual_losses_agree
            and mean_loss_agrees
            and hard_agrees
            and hard_summary["all_zero"]
            and single_hard_summary["all_zero"]
            and batch_numeric["all_finite"]
            and mean_single_numeric["all_finite"]
            and batch_numeric["l2_norm"] > 0.0
            and mean_single_numeric["l2_norm"] > 0.0
            and comparison["passed"]
        )
        emit(
            "completed",
            status="passed" if passed else "rejected",
            classification="fresh_batch4_gradient_resource_equivalence_no_training",
            passed=passed,
            finite_difference_certified=False,
            losses={
                "batch_mean": batch_loss_value,
                "single": single_losses,
                "individual_losses_agree": individual_losses_agree,
                "mean_loss_agrees": mean_loss_agrees,
                "cold_hot_agree": cold_hot_agree,
            },
            physical_hard={
                "batch4": hard_summary,
                "single": single_hard_summary,
                "per_sample_agrees": hard_agrees,
            },
            gradients={
                "batch4": batch_numeric,
                "mean_single": mean_single_numeric,
                "comparison": comparison,
            },
            evaluation_seconds={
                "batch4_cold": cold_seconds,
                "batch4_hot": hot_seconds,
                "single": single_seconds,
                "single_total": sum(single_seconds),
            },
            actual_memory={
                "after_batch4": memory_after_batch,
                "after_comparison": _device_memory(runtime["devices"][0]),
            },
            reader=pool.statistics(),
        )
    except Exception as error:
        emit("failed", status="failed", error=f"{type(error).__name__}: {error}")
        raise
    return report


def parser() -> Any:
    result = training.parser()
    result.description = __doc__
    result.set_defaults(
        protocol=ROOT / "manifests/coarse_graining/fresh_start_one_day_probe_v3.json"
    )
    result.add_argument(
        "--catalogue-report",
        type=Path,
        required=True,
        help="accepted one-day catalogue report; its catalogue selects the prepared source manifest",
    )
    return result


if __name__ == "__main__":
    run(parser().parse_args())
