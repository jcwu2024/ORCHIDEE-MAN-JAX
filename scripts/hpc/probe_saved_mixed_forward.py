"""Replay two saved mixed-learning checkpoints through the same daily forward.

This is supplemental saved-model evidence after an interrupted runner.  It
performs no training, gradient evaluation, optimizer update, or recurrent
rollout, and cannot reconstruct the missing original-process live-versus-disk
comparison.
"""
from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.hpc import probe_feedback_loss_calibration as calibration
from scripts.hpc import probe_feedback_loss_updates as admission
from scripts.hpc import run_fresh_start_one_day as prototype
from scripts.hpc.run_mixed_learning import configure_execution, validate_configuration

SCHEMA_VERSION = "saved_mixed_forward_replay_v1"
EXPECTED_UPDATES = 3
RTOL = 1.0e-5
ATOL = 1.0e-6


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain a JSON object")
    return value


def _checkpoint_path(report_path: Path, explicit: Path | None) -> Path:
    return (explicit if explicit is not None else
            report_path.with_suffix(".recovery.checkpoint.pkl")).resolve()


def _method_from_report(
    report: Mapping[str, Any],
    *,
    configuration: Mapping[str, Any],
    model_protocol: Mapping[str, Any],
    execution_policy: Mapping[str, Any],
) -> dict[str, Any]:
    method = {
        "implementation": "mixed_learning_v1",
        "configuration": dict(configuration),
        "model_protocol": dict(model_protocol),
        "scope": "integration_only",
        "integration_updates": EXPECTED_UPDATES,
        "source_git_head": report["source_git_head"],
        "execution_policy": dict(execution_policy),
        "recurrent_execution": execution_policy["recurrent_execution"],
    }
    if report.get("learning_schedule") is not None:
        method["learning_schedule"] = report["learning_schedule"]
    if report.get("device_execution") is not None:
        method["device_execution"] = report["device_execution"]
    return method


def _checkpoint_identity(
    report: Mapping[str, Any],
    *,
    name: str,
    configuration: Mapping[str, Any],
    model_protocol: Mapping[str, Any],
    execution_policy: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    if (report.get("configuration") != configuration
            or report.get("model_protocol") != model_protocol
            or report.get("execution_policy") != execution_policy):
        raise ValueError(f"{name} report configuration or execution method differs")
    if report.get("data_plan") != plan["summary"]:
        raise ValueError(f"{name} report data plan differs from the validated inputs")
    if (report.get("scope") != "integration_only"
            or report.get("completed_updates") != EXPECTED_UPDATES):
        raise ValueError(f"{name} report is not the bounded three-update integration run")
    if [row.get("update") for row in report.get("updates", [])] != [1, 2, 3]:
        raise ValueError(f"{name} report does not contain the three ordered update records")
    if EXPECTED_UPDATES not in [row.get("update") for row in report.get("monitors", [])]:
        raise ValueError(f"{name} report lacks its final saved-model monitor")
    fresh = {
        "seed": model_protocol["seed"],
        "initialization": model_protocol["initialization"],
        "parameter_count": model_protocol["parameter_count"],
        "inherited_weights": False,
    }
    return {
        "expected_method": _method_from_report(
            report,
            configuration=configuration,
            model_protocol=model_protocol,
            execution_policy=execution_policy,
        ),
        "expected_source_identities": plan["identities"],
        "expected_schedule_identity": plan["identities"],
        "expected_fresh_initialization_identity": fresh,
        "maximum_updates": configuration["updates"],
    }


def _input_identity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "row_count": len(rows),
        "rows": rows,
        "metadata_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _evaluate_checkpoint(
    *,
    name: str,
    path: Path,
    identity: Mapping[str, Any],
    jax: Any,
    device: Any,
    forward: Any,
    batch: Any,
    coefficients: Any,
    input_sha256: str,
    progress: Any,
) -> dict[str, Any]:
    from research.daily_coarse_graining.daily_operator.mixed_training_state import (
        load_mixed_training_checkpoint,
    )

    progress(f"load_{name}")
    state = load_mixed_training_checkpoint(path, **identity)
    if (state["completed_updates"] != EXPECTED_UPDATES
            or state["daily_sampler_cursor"] != {
                "epoch": 0, "batch_offset": EXPECTED_UPDATES}
            or state["recurrent_schedule_cursor"] != EXPECTED_UPDATES):
        raise ValueError(f"{name} checkpoint is not the saved three-update state")
    checkpoint_record = {
        "path": str(path),
        "completed_updates": state["completed_updates"],
        "daily_sampler_cursor": state["daily_sampler_cursor"],
        "recurrent_schedule_cursor": state["recurrent_schedule_cursor"],
    }
    host_parameters = state["parameters"]
    del state
    gc.collect()

    progress(f"forward_{name}")
    parameters = jax.device_put(host_parameters, device)
    del host_parameters
    gc.collect()
    loss, auxiliary = forward(parameters, batch, coefficients)
    selected = jax.device_get({
        "loss": loss,
        "terms": auxiliary["terms"],
        "hard": auxiliary["hard"],
    })
    del parameters, loss, auxiliary
    gc.collect()

    terms = np.asarray(selected["terms"], dtype=np.float64)
    hard = calibration._hard_summary(selected["hard"], batch_size=terms.shape[0])
    loss_value = float(selected["loss"])
    if terms.shape != (8, 3):
        raise ValueError(f"{name} daily forward terms have an unexpected shape")
    if not np.isfinite(loss_value) or not np.all(np.isfinite(terms)):
        raise ValueError(f"{name} daily forward produced non-finite values")
    return {
        "checkpoint": checkpoint_record,
        "input_metadata_sha256": input_sha256,
        "combined_loss": loss_value,
        "per_sample_terms": terms.tolist(),
        "mean_terms": np.mean(terms, axis=0).tolist(),
        "physical_counts": hard,
    }


def _compare_forward_records(
    candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_terms = np.asarray(candidate["per_sample_terms"], dtype=np.float64)
    reference_terms = np.asarray(reference["per_sample_terms"], dtype=np.float64)
    same_input = candidate["input_metadata_sha256"] == reference["input_metadata_sha256"]
    loss_close = bool(np.isclose(
        candidate["combined_loss"], reference["combined_loss"], rtol=RTOL, atol=ATOL,
    ))
    same_term_shape = candidate_terms.shape == reference_terms.shape
    terms_close = bool(same_term_shape and np.allclose(
        candidate_terms, reference_terms, rtol=RTOL, atol=ATOL))
    counts_equal = candidate["physical_counts"] == reference["physical_counts"]
    candidate_counts_zero = candidate["physical_counts"].get("all_zero") is True
    reference_counts_zero = reference["physical_counts"].get("all_zero") is True
    return {
        "relative_tolerance": RTOL,
        "absolute_tolerance": ATOL,
        "same_input": same_input,
        "combined_loss_close": loss_close,
        "combined_loss_absolute_error": abs(
            float(candidate["combined_loss"]) - float(reference["combined_loss"])
        ),
        "per_sample_terms_close": terms_close,
        "per_sample_terms_same_shape": same_term_shape,
        "per_sample_terms_max_absolute_error": (
            float(np.max(np.abs(candidate_terms - reference_terms)))
            if same_term_shape else None
        ),
        "physical_counts_equal": counts_equal,
        "candidate_physical_counts_all_zero": candidate_counts_zero,
        "reference_physical_counts_all_zero": reference_counts_zero,
        "passed": bool(same_input and loss_close and terms_close and counts_equal
                       and candidate_counts_zero and reference_counts_zero),
    }


def run(args: Any) -> dict[str, Any]:
    from research.daily_coarse_graining.daily_operator.fresh_start_training import make_objective
    from research.daily_coarse_graining.daily_operator.prepared_one_day import PreparedOneDayPool
    from scripts.hpc.check_daily_training_runtime import _atomic_json
    from scripts.hpc.mixed_learning_plan import load_plan
    from scripts.hpc.mixed_learning_dual_device import DUAL_DEVICE_TRAINING_LAYOUT
    from scripts.hpc.trained_screen_common import json_safe

    started = time.perf_counter()

    def progress(stage: str) -> None:
        print(f"saved_mixed_forward stage={stage} seconds={time.perf_counter()-started:.1f}",
              flush=True)

    progress("metadata")
    execution_policy = configure_execution(args)
    if execution_policy is None:
        raise ValueError("saved forward replay requires the admitted execution policy")
    configuration = _read_json(args.mixed_protocol, name="mixed protocol")
    model_protocol = _read_json(args.protocol, name="model protocol")
    validate_configuration(configuration, model_protocol)
    if (args.mixed_protocol.parent / configuration["model_protocol"]).resolve() != args.protocol.resolve():
        raise ValueError("CLI and mixed model protocols differ")
    plan = load_plan(args, configuration)
    candidate_report = _read_json(args.training_report, name="training report")
    reference_report = _read_json(
        args.reference_training_report, name="reference training report")
    if not candidate_report.get("resumed_from"):
        raise ValueError("training report is not the recovered new-process run")
    if (reference_report.get("status") != "completed"
            or reference_report.get("checkpoint_forward_replay") is not True):
        raise ValueError("reference report lacks its completed persisted-forward replay")
    if (candidate_report.get("device_execution") != DUAL_DEVICE_TRAINING_LAYOUT
            or reference_report.get("device_execution") != DUAL_DEVICE_TRAINING_LAYOUT):
        raise ValueError("saved forward replay requires the admitted dual-device run pair")
    for key in ("configuration", "model_protocol", "source_git_head",
                "execution_policy", "data_plan", "device_execution"):
        if candidate_report.get(key) != reference_report.get(key):
            raise ValueError(f"candidate and reference report identity differs: {key}")

    candidate_identity = _checkpoint_identity(
        candidate_report,
        name="candidate",
        configuration=configuration,
        model_protocol=model_protocol,
        execution_policy=execution_policy,
        plan=plan,
    )
    reference_identity = _checkpoint_identity(
        reference_report,
        name="reference",
        configuration=configuration,
        model_protocol=model_protocol,
        execution_policy=execution_policy,
        plan=plan,
    )
    candidate_checkpoint = _checkpoint_path(args.training_report, args.training_checkpoint)
    reference_checkpoint = _checkpoint_path(
        args.reference_training_report, args.reference_training_checkpoint)
    if candidate_checkpoint == reference_checkpoint:
        raise ValueError("candidate and reference must be distinct saved checkpoints")

    fixed_rows = plan["batches"][0].rows[:8]
    row_metadata = admission._rows(fixed_rows)
    input_identity = _input_identity(row_metadata)
    if input_identity["row_count"] != 8:
        raise ValueError("saved replay requires exactly the fixed first eight daily rows")

    progress("bootstrap")
    runtime, head, template, scales, fresh_parameters, _, _ = prototype.bootstrap(
        args, model_protocol)
    jax = runtime["jax"]
    import jax.numpy as jnp

    devices = tuple(jax.local_devices())
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("saved forward replay requires exactly one visible GPU")
    device = devices[0]
    del fresh_parameters
    gc.collect()

    graph = calibration._loss_protocol(
        model_protocol,
        canopy_source_pft_index=calibration._canopy_source_pft_index(runtime, head),
    )
    objective = make_objective(runtime, head, template, scales, graph)
    forward = jax.jit(calibration._make_batch_objective(
        jax, jnp, objective, calibration._base_weights(graph)))
    coefficients = jnp.asarray(
        (1.0, model_protocol["canopy_carbon_stock_change"]["weight"],
         model_protocol["loss"]["litterpart_feedback_mse"]),
        dtype=jnp.float64,
    )
    daily_pool = PreparedOneDayPool(plan["daily_source"], cache_size=5)
    if daily_pool.identity != plan["catalogue"]["source_identity"]:
        raise ValueError("daily reader and validated source identity differ")
    host_batch = daily_pool.batch([
        (int(row["artifact_id"]), int(row["batch_row"])) for row in row_metadata
    ])
    batch = jax.device_put(host_batch, device)
    del host_batch

    reference = _evaluate_checkpoint(
        name="reference",
        path=reference_checkpoint,
        identity=reference_identity,
        jax=jax,
        device=device,
        forward=forward,
        batch=batch,
        coefficients=coefficients,
        input_sha256=input_identity["metadata_sha256"],
        progress=progress,
    )
    candidate = _evaluate_checkpoint(
        name="candidate",
        path=candidate_checkpoint,
        identity=candidate_identity,
        jax=jax,
        device=device,
        forward=forward,
        batch=batch,
        coefficients=coefficients,
        input_sha256=input_identity["metadata_sha256"],
        progress=progress,
    )
    comparison = _compare_forward_records(candidate, reference)
    result = json_safe({
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if comparison["passed"] else "failed",
        "scope": "supplemental_saved_model_daily_forward_parity_only",
        "candidate_training_report": str(args.training_report.resolve()),
        "reference_training_report": str(args.reference_training_report.resolve()),
        "device": str(device),
        "input": input_identity,
        "candidate": candidate,
        "reference": reference,
        "comparison": comparison,
        "new_optimizer_updates": 0,
        "gradient_evaluations": 0,
        "checkpoint_inherited_for_training": False,
        "scientific_acceptance": False,
        "original_timeout_unchanged": True,
        "original_process_live_vs_disk_replay_observed": False,
        "missing_original_evidence": (
            "the interrupted original process did not persist its live-versus-disk "
            "daily forward replay or completion marker"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    })
    _atomic_json(args.output, result)
    progress("completed" if comparison["passed"] else "failed")
    print(json.dumps(result, separators=(",", ":"), allow_nan=False), flush=True)
    if not comparison["passed"]:
        raise ValueError("saved candidate and reference daily forwards differ")
    return result


def parser():
    result = prototype.parser()
    for name in ("mixed-protocol", "catalogue-report", "recurrent-schedule", "selection",
                 "execution-policy", "training-report", "reference-training-report"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--training-checkpoint", type=Path,
                        help="defaults to TRAINING_REPORT with .recovery.checkpoint.pkl")
    result.add_argument("--reference-training-checkpoint", type=Path,
                        help="defaults to REFERENCE_TRAINING_REPORT with .recovery.checkpoint.pkl")
    result.set_defaults(compare_checked_recurrent=False)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
