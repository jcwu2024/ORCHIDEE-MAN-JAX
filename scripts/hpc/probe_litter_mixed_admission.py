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
CONTINUATION_SCHEMA_VERSION = "litter_mixed_continuation_result_v1"
CONTINUATION_PROTOCOL_SCHEMA_VERSION = "litter_mixed_continuation_v1"
GRADIENT_BUNDLE_SCHEMA_VERSION = "litter_mixed_gradient_bundle_v1"
EXPECTED_PARAMETER_COUNT = 109_464_375
EXPECTED_SELECTED_WEIGHT = 5.221716892123007e-5
EXPECTED_SOURCE_ADMISSION_GIT_HEAD = "1391a75b1fc7ee322c7a73d4877f9f53a22c3a18"
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


def _validate_continuation_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the fixed selected20-to-selected100 continuation geometry."""

    if protocol.get("schema_version") != CONTINUATION_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("litter mixed-continuation protocol schema drift")
    expected = {
        "source_relative_update": 20,
        "target_relative_update": 100,
        "source_absolute_update": 4020,
        "milestones": [40, 60, 80, 100],
        "recovery_interval": 5,
        "daily_batch": 64,
        "microbatch": 8,
        "outer_weights": {"daily": 1.0, "T7": 0.5, "T30": 0.5},
        "scientific_stop": {
            "consecutive_monitors": 2,
            "rmse_ratio": 2.0,
            "minimum_absolute_increase_gC_m2": {
                "flux": 1.0,
                "biomass": 1.0,
                "canopy": 5.0,
                "nonliving": 10.0,
            },
        },
        "scientific_acceptance": False,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"litter mixed-continuation {key} drift")
    if float(protocol.get("selected_weight", np.nan)) != EXPECTED_SELECTED_WEIGHT:
        raise ValueError("litter mixed-continuation selected weight drift")
    return {
        **expected,
        "milestones": tuple(expected["milestones"]),
        "selected_weight": EXPECTED_SELECTED_WEIGHT,
        "update_indices": tuple(range(4020, 4100)),
    }


def _source_admission_method(report: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the exact method stored by the completed selected20 checkpoint."""

    if report.get("schema_version") != SCHEMA_VERSION or report.get("status") != "completed":
        raise ValueError("source admission report is not a completed mixed admission")
    if (
        report.get("technical_admission_passed") is not True
        or report.get("bounded_continuation_completed") is not True
        or report.get("full_mixed_admission") is not True
        or report.get("monitor_gates_passed") is not True
        or report.get("scientific_acceptance") is not False
    ):
        raise ValueError("source admission report lacks completed technical gates")
    if report.get("source_git_head") != EXPECTED_SOURCE_ADMISSION_GIT_HEAD:
        raise ValueError("source admission code revision drift")
    source_protocol = report.get("protocol")
    if not isinstance(source_protocol, Mapping):
        raise ValueError("source admission protocol is unavailable")
    admitted = _validate_protocol(source_protocol)
    selected_weight = float(report.get("selected_weight", np.nan))
    selection = report.get("weight_selection")
    selected_arm = report.get("arms", {}).get("selected")
    if (
        selected_weight != EXPECTED_SELECTED_WEIGHT
        or not isinstance(selection, Mapping)
        or float(selection.get("selected_weight", np.nan)) != selected_weight
        or not isinstance(selected_arm, Mapping)
        or float(selected_arm.get("litter_weight", np.nan)) != selected_weight
    ):
        raise ValueError("source admission selected weight drift")
    if (
        selected_arm.get("status") != "completed"
        or int(selected_arm.get("completed_updates", -1)) != 20
        or int(selected_arm.get("final_optimizer_step", -1)) != 4020
        or admitted["source_update"] != 4000
        or admitted["updates_per_arm"] != 20
    ):
        raise ValueError("source admission selected20 progress drift")
    monitor = report.get("development_monitors", {}).get("selected")
    panels = selected_arm.get("final_daily_panels", {})
    if (
        not isinstance(monitor, Mapping)
        or monitor.get("physical_gate_passed") is not True
        or monitor.get("metric_population_complete") is not True
        or set(panels) != set(PANEL_NAMES)
    ):
        raise ValueError("source admission selected20 monitor evidence is incomplete")
    for panel in panels.values():
        if (
            panel.get("hard_counts", {}).get("all_zero") is not True
            or int(panel.get("litter_stock_change", {}).get("hard_invalid_count", -1)) != 0
            or int(panel.get("litter_stock_change", {}).get("hard_empty_count", -1)) != 0
        ):
            raise ValueError("source admission selected20 daily physical gate failed")
    return _checkpoint_method(
        source_checkpoint=Path(str(report["source_checkpoint_path"])),
        source_update=admitted["source_update"],
        selected_weight=selected_weight,
        arm="selected",
        protocol=source_protocol,
        source_git_head=str(report["source_git_head"]),
    )


def _continuation_checkpoint_method(
    *, source_checkpoint: Path, source_method: Mapping[str, Any],
    protocol: Mapping[str, Any], runner_git_head: str,
) -> dict[str, Any]:
    return {
        "implementation": "litter_mixed_selected_continuation_v1",
        "parent_checkpoint": str(Path(source_checkpoint).resolve()),
        "parent_method": dict(source_method),
        "protocol": dict(protocol),
        "runner_git_head": str(runner_git_head),
        "claim": "bounded_selected_only_diagnostic_continuation_not_final_model",
    }


def _continuation_artifact_path(
    output: Path, *, relative_update: int | None = None, recovery: bool = False,
) -> Path:
    if recovery == (relative_update is not None):
        raise ValueError("choose exactly one continuation checkpoint identity")
    suffix = (
        "selected-continuation-recovery.checkpoint.pkl"
        if recovery
        else f"selected-continuation-update-{int(relative_update):06d}.checkpoint.pkl"
    )
    return Path(output).with_name(f"{Path(output).stem}.{suffix}")


def _align_continuation_history(
    updates: Sequence[Mapping[str, Any]],
    monitors: Sequence[Mapping[str, Any]],
    *,
    completed: int,
    milestones: Sequence[int],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], int | None]:
    """Align durable report evidence to a recovered checkpoint.

    A milestone checkpoint may exist before its monitor finishes. That exact
    milestone is returned as due; a missing earlier monitor is invalid because
    training must never have advanced past an unevaluated milestone.
    """

    retained_updates = [
        row for row in updates if int(row.get("relative_update", -1)) <= completed
    ]
    if [int(row.get("relative_update", -1)) for row in retained_updates] != list(
        range(21, completed + 1)
    ):
        raise ValueError("continuation report history is not aligned with recovery checkpoint")
    retained_monitors = [
        row for row in monitors if int(row.get("relative_update", -1)) <= completed
    ]
    monitor_updates = [
        int(row.get("relative_update", -1)) for row in retained_monitors
    ]
    if len(monitor_updates) != len(set(monitor_updates)):
        raise ValueError("continuation milestone-monitor history is duplicated")
    if any(milestone < completed and milestone not in monitor_updates for milestone in milestones):
        raise ValueError("continuation advanced past an unevaluated milestone")
    due = completed if completed in milestones and completed not in monitor_updates else None
    return retained_updates, retained_monitors, due


def _resume_continuation_report(
    previous: Mapping[str, Any], current_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Refresh immutable metadata without discarding durable progress lists."""

    result = dict(previous)
    result.update(current_metadata)
    for key in ("updates", "milestone_monitors", "milestone_checkpoints"):
        result[key] = list(previous.get(key, ()))
    if previous.get("recovery_checkpoint") is not None:
        result["recovery_checkpoint"] = dict(previous["recovery_checkpoint"])
    result["prior_elapsed_seconds"] = (
        float(previous.get("prior_elapsed_seconds", 0.0))
        + float(previous.get("elapsed_seconds", 0.0))
    )
    result["resumed_incomplete_report"] = True
    return result


def _run_admission(args: argparse.Namespace) -> Mapping[str, Any]:
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
        if args.mixed_admission_protocol is None:
            raise ValueError("mixed admission protocol is required for the admission branch")
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


def _run_continuation(args: argparse.Namespace) -> Mapping[str, Any]:
    """Continue only the admitted selected arm without recalibration or replay."""

    from research.daily_coarse_graining.daily_operator.canonical_rollout import make_canonical_step
    from research.daily_coarse_graining.daily_operator.fresh_start_training import make_objective, make_update
    from research.daily_coarse_graining.daily_operator.mixed_objective import make_mixed_gradient
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
    from scripts.hpc.litter_continuation_monitor import should_stop, summarize_progress
    from scripts.hpc.mixed_learning_monitor import build_mixed_rollout_monitor
    from scripts.hpc.mixed_learning_plan import load_plan
    from scripts.hpc.probe_one_day_batch import _compiled_memory, _device_memory
    from scripts.hpc.trained_screen_common import PreparedWindowPool, WindowPool, json_safe

    started = time.perf_counter()
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    report: dict[str, Any] = {
        "schema_version": CONTINUATION_SCHEMA_VERSION,
        "status": "running",
        "stage": "created",
        "claim_scope": "bounded_selected_only_litter_mixed_continuation",
        "source_git_head": current_head,
        "scientific_acceptance": False,
        "updates": [],
        "milestone_monitors": [],
        "milestone_checkpoints": [],
    }
    output_owned = not args.output.exists()

    def emit(stage: str, **values: Any) -> None:
        report.update(values, stage=stage, elapsed_seconds=time.perf_counter() - started)
        flush = (
            stage != "update_completed"
            or int(report.get("completed_relative_update", 20)) % 5 == 0
        )
        if flush:
            _atomic_json(args.output, json_safe(report))
        print(
            f"litter_mixed_continuation stage={stage} "
            f"relative_update={report.get('completed_relative_update', 20)} "
            f"seconds={report['elapsed_seconds']:.1f}",
            flush=True,
        )

    try:
        if args.shape_only or args.admission_only or args.cpu_forward_only:
            raise ValueError("litter mixed continuation requires complete GPU execution")
        configuration = _read_json(args.mixed_protocol, name="mixed protocol")
        model_protocol = _read_json(args.protocol, name="model protocol")
        protocol = _read_json(args.continuation_protocol, name="continuation protocol")
        admitted = _validate_continuation_protocol(protocol)
        source_admission = _read_json(
            args.source_admission_report, name="source admission report"
        )
        source_method = _source_admission_method(source_admission)
        if args.mixed_admission_protocol is not None:
            supplied_source_protocol = _read_json(
                args.mixed_admission_protocol, name="mixed admission protocol"
            )
            if supplied_source_protocol != source_admission["protocol"]:
                raise ValueError("supplied mixed admission protocol differs from source report")
        if int(args.expected_update) != 4000:
            raise ValueError("continuation retains expected-update 4000 for its source report")
        daily_admission._validate_model_protocol(model_protocol)
        validate_configuration(configuration, model_protocol)
        optimizer_contract = model_protocol.get("optimizer", {})
        if (
            optimizer_contract.get("name") != "AdamW"
            or float(optimizer_contract.get("learning_rate", np.nan)) != 1e-5
            or float(optimizer_contract.get("clip_norm", np.nan)) != 1.0
            or int(model_protocol.get("parameter_count", -1)) != EXPECTED_PARAMETER_COUNT
        ):
            raise ValueError("continuation model/AdamW contract drift")
        configured_protocol = (args.mixed_protocol.parent / configuration["model_protocol"]).resolve()
        if configured_protocol != args.protocol.resolve():
            raise ValueError("CLI and mixed model protocols differ")
        execution_policy = configure_execution(args)
        if execution_policy is None or not args.checked_recurrent or not args.separate_checked_validation:
            raise ValueError("litter mixed continuation requires admitted split checked recurrent execution")
        plan = load_plan(args, configuration)
        update_batches = _select_batches(plan, admitted["update_indices"])
        schedule = plan["schedule"]
        if len(schedule["update_rows"]) <= admitted["update_indices"][-1]:
            raise ValueError("recurrent schedule does not cover continuation updates")
        training_report = _read_json(args.training_report, name="training report")
        source_identity = endpoints._checkpoint_identity(
            training_report,
            configuration=configuration,
            model_protocol=model_protocol,
            execution_policy=execution_policy,
            plan=plan,
            expected_update=4000,
        )
        source_rows = [
            row for row in training_report.get("checkpoints", ())
            if int(row.get("update", -1)) == 4000
        ]
        if (
            len(source_rows) != 1
            or Path(str(source_rows[0].get("path", ""))).resolve()
            != Path(str(source_admission["source_checkpoint_path"])).resolve()
            or Path(str(source_admission.get("source_training_report_path", ""))).resolve()
            != args.training_report.resolve()
        ):
            raise ValueError("source4000 training metadata differs from completed admission")
        del source_identity

        source_checkpoint_rows = [
            row for row in source_admission["arms"]["selected"].get("checkpoints", ())
            if int(row.get("update", -1)) == admitted["source_relative_update"]
        ]
        if (
            len(source_checkpoint_rows) != 1
            or Path(str(source_checkpoint_rows[0].get("path", ""))).resolve()
            != args.checkpoint.resolve()
        ):
            raise ValueError("--checkpoint is not the completed selected20 checkpoint")
        checkpoint_method = _continuation_checkpoint_method(
            source_checkpoint=args.checkpoint,
            source_method=source_method,
            protocol=protocol,
            runner_git_head=current_head,
        )
        baseline_monitor = source_admission["development_monitors"]["selected"]
        baseline_daily_panels = source_admission["arms"]["selected"]["final_daily_panels"]
        report.update(
            protocol=protocol,
            protocol_path=str(args.continuation_protocol.resolve()),
            source_admission_report_path=str(args.source_admission_report.resolve()),
            source_training_report_path=str(args.training_report.resolve()),
            source_checkpoint_path=str(args.checkpoint.resolve()),
            source_checkpoint_method=source_method,
            checkpoint_method=checkpoint_method,
            execution_policy=execution_policy,
            plan_identities=plan["identities"],
            selected_batches=[batch.as_metadata() for batch in update_batches],
            selected_weight=admitted["selected_weight"],
            baseline_selected20={
                "development_monitor": baseline_monitor,
                "daily_panels": baseline_daily_panels,
            },
            completed_relative_update=admitted["source_relative_update"],
            completed_absolute_update=admitted["source_absolute_update"],
        )
        if args.output.is_file():
            previous = _read_json(args.output, name="incomplete continuation report")
            compatible = (
                previous.get("schema_version") == CONTINUATION_SCHEMA_VERSION
                and previous.get("protocol") == protocol
                and Path(str(previous.get("source_checkpoint_path", ""))).resolve()
                == args.checkpoint.resolve()
                and previous.get("source_git_head") == current_head
                and previous.get("plan_identities") == plan["identities"]
                and previous.get("checkpoint_method") == checkpoint_method
            )
            if not compatible:
                raise ValueError("existing continuation report belongs to a different run")
            if previous.get("status") in ("completed", "stopped_science_guard"):
                raise FileExistsError("terminal litter continuation output already exists")
            current_metadata = dict(report)
            report = _resume_continuation_report(previous, current_metadata)
            output_owned = True
        emit("metadata_validated")

        runtime, head, template, scales, fresh_parameters, _, _ = prototype.bootstrap(
            args, model_protocol
        )
        jax = runtime["jax"]
        import jax.numpy as jnp

        devices = tuple(jax.local_devices())
        if len(devices) != 1 or devices[0].platform != "gpu":
            raise ValueError("litter mixed continuation requires exactly one visible FREE GPU")
        device = devices[0]
        if sum(int(value.size) for value in jax.tree_util.tree_leaves(fresh_parameters)) != EXPECTED_PARAMETER_COUNT:
            raise ValueError("actual full-network parameter count differs from continuation contract")
        del fresh_parameters

        recovery_path = _continuation_artifact_path(args.output, recovery=True)
        if recovery_path.is_file():
            saved = load_training_checkpoint(recovery_path, expected_method=checkpoint_method)
            resumed_from = recovery_path
        else:
            saved = load_training_checkpoint(args.checkpoint, expected_method=source_method)
            resumed_from = args.checkpoint
        completed = int(saved["completed_updates"])
        if (
            completed < admitted["source_relative_update"]
            or completed > admitted["target_relative_update"]
            or (completed - admitted["source_relative_update"]) % admitted["recovery_interval"] != 0
            or saved["cursor"] != {"epoch": 0, "batch_offset": 4000 + completed}
            or int(np.asarray(saved["optimizer_state"].step)) != 4000 + completed
        ):
            raise ValueError("continuation recovery checkpoint progress drift")
        parameters_host = saved["parameters"]
        optimizer_host = saved["optimizer_state"]
        del saved
        report["resumed_from"] = str(Path(resumed_from).resolve())
        report["completed_relative_update"] = completed
        report["completed_absolute_update"] = 4000 + completed
        aligned_updates, aligned_monitors, due_monitor = _align_continuation_history(
            report["updates"],
            report["milestone_monitors"],
            completed=completed,
            milestones=admitted["milestones"],
        )
        report["updates"] = aligned_updates
        report["milestone_monitors"] = aligned_monitors
        report["milestone_checkpoints"] = [
            row for row in report["milestone_checkpoints"]
            if int(row.get("relative_update", -1)) <= completed
        ]
        emit("checkpoint_restored")

        candidate_protocol = daily_admission._candidate_model_protocol(
            model_protocol, source_admission["protocol"]
        )
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
        selected_coefficients = jax.device_put(
            jnp.asarray((1.0, 0.1, 0.1, admitted["selected_weight"]), dtype=jnp.float64),
            device,
        )
        daily_pool = PreparedOneDayPool(plan["daily_source"], cache_size=5)
        recurrent_pool = PreparedWindowPool(schedule["source_manifest"], cache_size=2)
        development_pool = WindowPool(args.selection)
        if daily_pool.identity != plan["catalogue"]["source_identity"] or recurrent_pool.identity != schedule["source_identity"]:
            raise ValueError("daily/recurrent readers differ from validated source identities")
        panel_rows = daily_admission._panel_rows(plan)
        baseline_selections = [row["selection"] for row in baseline_monitor["windows"]]
        if development_pool.selection["windows"] != baseline_selections:
            raise ValueError("continuation development windows differ from selected20 baseline")
        if any(
            list(panel_rows[name]) != baseline_daily_panels[name]["rows"]
            for name in PANEL_NAMES
        ):
            raise ValueError("continuation daily panels differ from selected20 baseline")

        def sample(rows: Sequence[Any]) -> Any:
            metadata = admission._rows(rows)
            return daily_pool.batch([
                (int(row["artifact_id"]), int(row["batch_row"])) for row in metadata
            ])

        parameters, optimizer = jax.device_put((parameters_host, optimizer_host), device)
        del parameters_host, optimizer_host
        first_samples = jax.device_put(sample(update_batches[0].microbatches[0].rows), device)
        emit("compile_daily_gradient")
        began = time.perf_counter()
        daily_executable = daily_evaluator.lower(
            parameters, first_samples, selected_coefficients
        ).compile()
        daily_memory = _compiled_memory(daily_executable)
        if daily_memory["estimated_live_bytes"] > configuration["compiled_memory_limit_gib"] * 1024**3:
            raise MemoryError("daily gradient exceeds compiled-memory envelope")
        report["compilation"] = {
            "daily": {"seconds": time.perf_counter() - began, "memory": daily_memory}
        }
        emit("daily_gradient_compiled", memory=_device_memory(device))

        recurrent_loss, recurrent_scales = bind_recurrent_science_loss(
            runtime, head, configuration["recurrent_weights"]
        )
        rollout = make_rollout_loss(
            jax, jnp, make_canonical_step(head, template), recurrent_loss
        )
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

        def daily_component(current_parameters: Any, batch: Any) -> tuple[Any, dict[str, Any]]:
            total = jax.tree_util.tree_map(jnp.zeros_like, current_parameters)
            loss = 0.0
            terms = np.zeros((4,), dtype=np.float64)
            hard_totals: dict[str, int] = {}
            for micro_index, microbatch in enumerate(batch.microbatches):
                samples = jax.device_put(sample(microbatch.rows), device)
                (value_auxiliary, gradient) = daily_executable(
                    current_parameters, samples, selected_coefficients
                )
                value, auxiliary = value_auxiliary
                jax.block_until_ready((value, auxiliary, gradient))
                hard = calibration._hard_summary(
                    jax.device_get(auxiliary["hard"]), batch_size=microbatch.draw_count
                )
                litter_auxiliary = jax.device_get(auxiliary["litter_stock_change"])
                if (
                    not np.isfinite(float(value))
                    or not hard["all_zero"]
                    or int(np.sum(np.asarray(litter_auxiliary["hard_invalid_count"])))
                    or int(np.sum(np.asarray(litter_auxiliary["hard_empty_count"])))
                ):
                    raise ValueError(
                        f"daily selected physical/finite gate failed at batch "
                        f"{batch.batch_index} microbatch {micro_index}"
                    )
                total = add_gradient(total, gradient, microbatch.draw_count)
                loss += float(value) * microbatch.draw_count
                terms += np.mean(
                    np.asarray(jax.device_get(auxiliary["terms"]), dtype=np.float64), axis=0
                ) * microbatch.draw_count
                for key, count in hard["totals"].items():
                    hard_totals[key] = hard_totals.get(key, 0) + int(count)
                del samples, auxiliary, gradient
            return mean_gradient(total, batch.draw_count), {
                "loss": loss / batch.draw_count,
                "terms": (terms / batch.draw_count).tolist(),
                "hard": hard_totals,
            }

        def recurrent_component(
            current_parameters: Any, batch_index: int, horizon: int
        ) -> tuple[Any, float, dict[str, int]]:
            row = schedule["update_rows"][batch_index][f"T{horizon}"]
            case = recurrent_pool.case(jax, row)
            if horizon not in recurrent_executables:
                emit(f"compile_T{horizon}", active_batch_index=batch_index)
                compile_started = time.perf_counter()
                raw = checked_backward.lower(current_parameters, *case[:4], context).compile()
                memory = _compiled_memory(raw)
                if memory["estimated_live_bytes"] > configuration["compiled_memory_limit_gib"] * 1024**3:
                    raise MemoryError(f"T{horizon} checked gradient exceeds compiled-memory envelope")
                recurrent_executables[horizon] = wrap_separate_checked(jax, raw, validator)
                report["compilation"][f"T{horizon}"] = {
                    "seconds": time.perf_counter() - compile_started,
                    "memory": memory,
                }
                del raw
                emit(f"compiled_T{horizon}", active_batch_index=batch_index)
            (value, counts), gradient = recurrent_executables[horizon](
                current_parameters, *case[:4], context, case[4]
            )
            value, counts = jax.device_get((value, counts))
            hard = decode_counts(checked_names, counts)
            if not np.isfinite(float(value)) or any(hard.values()):
                raise ValueError(
                    f"checked T{horizon} physical/finite gate failed at batch {batch_index}"
                )
            return gradient, float(value), hard

        monitor = build_mixed_rollout_monitor(
            runtime,
            head,
            template,
            scales,
            model_protocol,
            compiled_memory_limit_bytes=configuration["compiled_memory_limit_gib"] * 1024**3,
        )

        def save_milestone_checkpoint(relative_update: int) -> None:
            milestone_path = _continuation_artifact_path(
                args.output, relative_update=relative_update
            )
            existing = [
                row for row in report["milestone_checkpoints"]
                if int(row.get("relative_update", -1)) == relative_update
            ]
            if len(existing) > 1:
                raise ValueError("duplicate continuation milestone checkpoint records")
            if existing and milestone_path.is_file():
                return
            if milestone_path.is_file():
                persisted = load_training_checkpoint(
                    milestone_path, expected_method=checkpoint_method
                )
                if (
                    int(persisted["completed_updates"]) != relative_update
                    or persisted["cursor"] != {
                        "epoch": 0, "batch_offset": 4000 + relative_update
                    }
                    or int(np.asarray(persisted["optimizer_state"].step))
                    != 4000 + relative_update
                ):
                    raise ValueError("persisted milestone checkpoint progress drift")
                del persisted
            else:
                save_training_checkpoint(
                    milestone_path,
                    jax=jax,
                    parameters=parameters,
                    optimizer_state=optimizer,
                    completed_updates=relative_update,
                    cursor={"epoch": 0, "batch_offset": 4000 + relative_update},
                    method=checkpoint_method,
                )
            checkpoint_record = {
                "relative_update": relative_update,
                "absolute_update": 4000 + relative_update,
                "path": str(milestone_path.resolve()),
                "bytes": milestone_path.stat().st_size,
            }
            report["milestone_checkpoints"] = [
                row for row in report["milestone_checkpoints"]
                if int(row.get("relative_update", -1)) != relative_update
            ] + [checkpoint_record]
            emit("milestone_checkpoint_saved", active_relative_update=relative_update)

        def evaluate_milestone(relative_update: int) -> dict[str, Any]:
            existing = [
                row for row in report["milestone_monitors"]
                if int(row.get("relative_update", -1)) == relative_update
            ]
            if existing:
                return dict(existing[0]["stop_decision"])
            daily_panels = {
                panel_name: daily_admission._panel_record(
                    parameters,
                    panel_rows[panel_name],
                    daily_pool,
                    daily_forward,
                    selected_coefficients,
                    jax,
                )
                for panel_name in PANEL_NAMES
            }
            development = monitor(
                parameters,
                development_pool,
                development_pool.selection["windows"],
                panel_name=f"selected_after_{relative_update}_relative_updates",
                development=True,
                apply_science_gates=True,
            )
            daily_clean = all(
                panel["hard_counts"]["all_zero"]
                and panel["litter_stock_change"]["hard_invalid_count"] == 0
                and panel["litter_stock_change"]["hard_empty_count"] == 0
                for panel in daily_panels.values()
            )
            recurrent_clean = bool(
                development["physical_gate_passed"]
                and development["metric_population_complete"]
            )
            if not daily_clean or not recurrent_clean:
                raise ValueError(
                    f"relative update {relative_update} physical or metric-population gate failed"
                )
            progress = summarize_progress(baseline_monitor, development)
            progress["relative_update"] = relative_update
            progress["absolute_update"] = 4000 + relative_update
            history = [
                row["progress_summary"] for row in report["milestone_monitors"]
            ] + [progress]
            stop_decision = should_stop(history, admitted["scientific_stop"])
            record = {
                "relative_update": relative_update,
                "absolute_update": 4000 + relative_update,
                "daily_panels": daily_panels,
                "development_monitor": development,
                "progress_summary": progress,
                "stop_decision": stop_decision,
                "physical_gate_passed": True,
                "metric_population_complete": True,
                "science_accuracy_gate_required": False,
                "scientific_acceptance": False,
            }
            report["milestone_monitors"].append(record)
            emit(
                "milestone_monitor_completed",
                active_relative_update=relative_update,
                latest_stop_decision=stop_decision,
            )
            return stop_decision

        stopped = False
        if due_monitor is not None:
            save_milestone_checkpoint(due_monitor)
            stopped = bool(evaluate_milestone(due_monitor)["stop"])
        elif completed in admitted["milestones"]:
            completed_monitor = next(
                row for row in report["milestone_monitors"]
                if int(row["relative_update"]) == completed
            )
            stopped = bool(completed_monitor["stop_decision"]["stop"])

        for relative_update in range(completed + 1, admitted["target_relative_update"] + 1):
            if stopped:
                break
            batch = update_batches[
                relative_update - admitted["source_relative_update"] - 1
            ]
            batch_index = int(batch.batch_index)
            update_started = time.perf_counter()
            daily_gradient, daily_record = daily_component(parameters, batch)
            losses = {
                "daily_weighted": float(daily_record["loss"]),
                "daily_component_means": dict(zip(
                    daily_admission.LOSS_NAMES, daily_record["terms"], strict=True
                )),
            }
            hard_counts: dict[str, Any] = {"daily": daily_record["hard"]}
            recurrent_gradients = {}
            for horizon in (7, 30):
                gradient, value, hard = recurrent_component(
                    parameters, batch_index, horizon
                )
                recurrent_gradients[f"T{horizon}"] = gradient
                losses[f"T{horizon}"] = value
                hard_counts[f"T{horizon}"] = hard
            component_geometry = {
                "components": ["daily_weighted", "T7", "T30"],
                "raw_norms": {
                    "daily_weighted": float(norm(daily_gradient)),
                    "T7": float(norm(recurrent_gradients["T7"])),
                    "T30": float(norm(recurrent_gradients["T30"])),
                },
                "litter_weight": admitted["selected_weight"],
            }
            mixed = combine(daily_gradient, recurrent_gradients["T7"], 1.0, 0.5)
            mixed = combine(mixed, recurrent_gradients["T30"], 1.0, 0.5)
            jax.block_until_ready(mixed)
            if not bool(finite(mixed)):
                raise ValueError("combined continuation gradient is nonfinite")
            proposed, proposed_optimizer, gradient_norm, clip = optimizer_update(
                parameters, mixed, optimizer
            )
            jax.block_until_ready((proposed, proposed_optimizer, gradient_norm, clip))
            expected_step = 4000 + relative_update
            if (
                not bool(finite((proposed, proposed_optimizer)))
                or not np.isfinite(float(gradient_norm))
                or not 0.0 < float(clip) <= 1.0
                or int(np.asarray(jax.device_get(proposed_optimizer.step))) != expected_step
            ):
                raise ValueError("complete continuation AdamW proposal is invalid")
            parameters, optimizer = proposed, proposed_optimizer
            report["updates"].append({
                "relative_update": relative_update,
                "source_absolute_update": expected_step,
                "daily_batch_index": batch_index,
                "losses": losses,
                "mixed_loss": losses["daily_weighted"] + 0.5 * losses["T7"] + 0.5 * losses["T30"],
                "component_geometry": component_geometry,
                "combined_gradient_norm": float(gradient_norm),
                "clip": float(clip),
                "hard_counts": hard_counts,
                "optimizer_step": expected_step,
                "seconds": time.perf_counter() - update_started,
            })
            report["completed_relative_update"] = relative_update
            report["completed_absolute_update"] = expected_step
            del mixed, proposed, proposed_optimizer, daily_gradient, recurrent_gradients, gradient
            emit("update_completed", active_relative_update=relative_update)

            if relative_update % admitted["recovery_interval"] == 0:
                save_training_checkpoint(
                    recovery_path,
                    jax=jax,
                    parameters=parameters,
                    optimizer_state=optimizer,
                    completed_updates=relative_update,
                    cursor={"epoch": 0, "batch_offset": 4000 + relative_update},
                    method=checkpoint_method,
                )
                report["recovery_checkpoint"] = {
                    "relative_update": relative_update,
                    "absolute_update": 4000 + relative_update,
                    "path": str(recovery_path.resolve()),
                    "bytes": recovery_path.stat().st_size,
                }
                emit("recovery_checkpoint_saved", active_relative_update=relative_update)

            if relative_update in admitted["milestones"]:
                save_milestone_checkpoint(relative_update)
                gc.collect()
                stopped = bool(evaluate_milestone(relative_update)["stop"])

        final_relative_update = int(report["completed_relative_update"])
        final_monitors = [
            int(row["relative_update"]) for row in report["milestone_monitors"]
        ]
        expected_milestones = [
            value for value in admitted["milestones"] if value <= final_relative_update
        ]
        checkpoint_updates = sorted(
            int(row["relative_update"]) for row in report["milestone_checkpoints"]
        )
        recovery_record = report.get("recovery_checkpoint", {})
        if (
            final_monitors != expected_milestones
            or checkpoint_updates != expected_milestones
            or int(recovery_record.get("relative_update", -1)) != final_relative_update
            or any(not Path(row["path"]).is_file() for row in report["milestone_checkpoints"])
        ):
            raise ValueError("terminal continuation checkpoint/monitor evidence is incomplete")
        if stopped:
            terminal_status = "stopped_science_guard"
        else:
            terminal_status = "completed"
            if (
                final_relative_update != admitted["target_relative_update"]
                or final_monitors != list(admitted["milestones"])
                or len(report["updates"]) != 80
                or int(np.asarray(jax.device_get(optimizer.step))) != 4100
            ):
                raise ValueError("continuation completion evidence is incomplete")
        emit(
            terminal_status,
            status=terminal_status,
            classification="bounded_selected_only_litter_mixed_continuation",
            technical_continuation_passed=True,
            physical_monitoring_passed=True,
            bounded_continuation_completed=terminal_status == "completed",
            stopped_by_science_guard=terminal_status == "stopped_science_guard",
            scientific_acceptance=False,
            final_memory=_device_memory(device),
            daily_reader=daily_pool.statistics(),
            active_relative_update=None,
        )
    except Exception as error:
        if not output_owned:
            raise
        if not args.output.exists():
            report["output_creation_failed_before_first_emit"] = True
        emit("failed", status="failed", error=f"{type(error).__name__}: {error}")
        raise
    return report


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    continuation_requested = args.continuation_protocol is not None
    source_requested = args.source_admission_report is not None
    if continuation_requested != source_requested:
        raise ValueError(
            "continuation protocol and source admission report are required together"
        )
    return _run_continuation(args) if continuation_requested else _run_admission(args)


def parser() -> argparse.ArgumentParser:
    result = endpoints.parser()
    result.description = __doc__
    result.add_argument("--mixed-admission-protocol", type=Path)
    result.add_argument("--continuation-protocol", type=Path)
    result.add_argument("--source-admission-report", type=Path)
    result.set_defaults(
        compare_checked_recurrent=False,
        checked_reference_report=None,
        dual_device_profile=False,
        dual_device_training=False,
    )
    return result


if __name__ == "__main__":
    result = run(parser().parse_args())
    clean = (
        result.get("status") == "completed"
        and result.get("technical_admission_passed") is True
    ) or (
        result.get("status") in ("completed", "stopped_science_guard")
        and result.get("technical_continuation_passed") is True
    )
    if not clean:
        raise SystemExit(2)
