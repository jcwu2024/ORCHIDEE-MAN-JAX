"""Run bounded batch-four joint training for the structured-GPP specialist."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_checkpoint import (
    load_structured_gpp_checkpoint,
    load_structured_gpp_recovery,
    write_structured_gpp_checkpoint,
    write_structured_gpp_recovery,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_joint_batch4 import (
    mean_source_losses,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_joint_calibration import (
    DIRECT_SOURCE,
    GRADIENT_SOURCES,
    SEVEN_SOURCE,
    THIRTY_SOURCE,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_joint_training import (
    build_joint_training_schedule,
    epoch_panel_continuation_passed,
    load_structured_gpp_joint_training_contract,
    panel_source_changes,
    selected_epoch_from_panels,
    validate_joint_training_recovery_metadata,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_joint_update import (
    structured_gpp_joint_objectives,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_training import (
    CANDIDATE_PARAMETER_COUNT,
    SPECIALIST_PARAMETER_COUNT,
    replace_specialist,
    specialist_element_count,
    structured_gpp_direct_loss,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    target_oriented_adamw_update,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import _load_context, _tree_size
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import (
    _load_artifact,
    _replace_adapter,
)
from scripts.hpc.calibrate_target_oriented_structured_gpp_joint_gradients import (
    _direct_gradient,
    _horizon_gradient,
    _make_direct_steps,
    _one_day_hard,
    _retained_parameter_report,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_fast_owner_gpu_probe import PROBE_SEED
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import _head as _parent_head
from scripts.hpc.run_target_oriented_full_step_preflight import _hard_failure_count, _make_steps
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import (
    ADAPTER_PARAMETER_COUNT,
    PARENT_PARAMETER_COUNT,
    _direct_forward,
    _hard_host,
    _horizon_forward,
)

SCHEMA_VERSION = "target_oriented_structured_gpp_joint_training_result_v1"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _source_losses(direct: Any, seven: Any, thirty: Any) -> Mapping[str, float]:
    return {
        DIRECT_SOURCE: float(np.asarray(structured_gpp_direct_loss(direct))),
        SEVEN_SOURCE: float(np.asarray(seven.total)),
        THIRTY_SOURCE: float(np.asarray(thirty.total)),
    }


def _mean_gradients(jax: Any, jnp: Any, gradients: Sequence[Any]) -> Any:
    values = tuple(gradients)
    if not values:
        raise ValueError("structured GPP joint training mean requires gradients")
    scale = jnp.asarray(1.0 / len(values), dtype=jnp.float32)
    return jax.tree_util.tree_map(
        lambda *leaves: sum(
            (jnp.asarray(leaf, dtype=jnp.float32) for leaf in leaves),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        * scale,
        *values,
    )


def _combine_gradients(
    jax: Any,
    jnp: Any,
    gradients: Mapping[str, Any],
    coefficients: Mapping[str, float],
) -> Any:
    seven_weight = jnp.asarray(coefficients[SEVEN_SOURCE], dtype=jnp.float32)
    thirty_weight = jnp.asarray(coefficients[THIRTY_SOURCE], dtype=jnp.float32)
    return jax.tree_util.tree_map(
        lambda direct, seven, thirty: direct + seven_weight * seven + thirty_weight * thirty,
        gradients[DIRECT_SOURCE],
        gradients[SEVEN_SOURCE],
        gradients[THIRTY_SOURCE],
    )


def _tree_scalar_report(jax: Any, jnp: Any, tree: Any) -> Mapping[str, Any]:
    leaves = tuple(jax.tree_util.tree_leaves(tree))
    squared = sum(
        (jnp.sum(jnp.square(jnp.asarray(value)), dtype=jnp.float64) for value in leaves),
        jnp.asarray(0.0, dtype=jnp.float64),
    )
    nonfinite = sum(
        (jnp.sum(~jnp.isfinite(jnp.asarray(value)), dtype=jnp.int64) for value in leaves),
        jnp.asarray(0, dtype=jnp.int64),
    )
    nonzero_leaves = sum(
        (jnp.any(jnp.asarray(value) != 0.0).astype(jnp.int32) for value in leaves),
        jnp.asarray(0, dtype=jnp.int32),
    )
    jax.block_until_ready((squared, nonfinite, nonzero_leaves))
    return {
        "all_finite": int(np.asarray(nonfinite)) == 0,
        "element_count": sum(int(np.asarray(value).size) for value in leaves),
        "leaf_count": len(leaves),
        "nonzero_leaf_count": int(np.asarray(nonzero_leaves)),
        "l2_norm": float(np.sqrt(float(np.asarray(squared)))),
    }


def _make_proposal_check(jax: Any, jnp: Any) -> Any:
    @jax.jit
    def check(before: Any, after: Any, optimizer: Any) -> tuple[Any, Any, Any]:
        before_leaves = jax.tree_util.tree_leaves(before)
        after_leaves = jax.tree_util.tree_leaves(after)
        optimizer_leaves = jax.tree_util.tree_leaves(optimizer)
        parameter_nonfinite = sum(
            (jnp.sum(~jnp.isfinite(value), dtype=jnp.int64) for value in after_leaves),
            jnp.asarray(0, dtype=jnp.int64),
        )
        optimizer_nonfinite = sum(
            (
                jnp.sum(~jnp.isfinite(value), dtype=jnp.int64)
                for value in optimizer_leaves
            ),
            jnp.asarray(0, dtype=jnp.int64),
        )
        delta_squared = sum(
            (
                jnp.sum(
                    jnp.square(
                        jnp.asarray(after_value, dtype=jnp.float64)
                        - jnp.asarray(before_value, dtype=jnp.float64)
                    ),
                    dtype=jnp.float64,
                )
                for before_value, after_value in zip(
                    before_leaves, after_leaves, strict=True
                )
            ),
            jnp.asarray(0.0, dtype=jnp.float64),
        )
        return parameter_nonfinite, optimizer_nonfinite, delta_squared

    return check


def _validate_recovered_checkpoint_files(
    checkpoints: Sequence[Mapping[str, Any]], output_dir: Path
) -> None:
    for checkpoint in checkpoints:
        path = Path(str(checkpoint["path"])).resolve()
        if (
            path.parent != output_dir
            or not path.is_file()
            or sha256_file(path) != str(checkpoint["sha256"])
        ):
            raise ValueError("structured GPP joint recovery checkpoint file drift")


def _joint_panel(
    args: argparse.Namespace,
    context: Mapping[str, Any],
    parameters: Any,
    specialist: Any,
    steps: Mapping[str, Any],
    direct_steps: Mapping[str, Any],
    updates: tuple[int, ...],
    flux_basis: Any,
    *,
    label: str,
) -> Mapping[str, Any]:
    jax = context["runtime"]["jax"]
    jnp = context["runtime"]["jnp"]
    records = []
    for update in updates:
        started = time.perf_counter()
        artifact = _load_artifact(args, context, update)
        direct, direct_seconds = _direct_forward(jax, specialist, artifact.batch, direct_steps)
        one_hard = _one_day_hard(jax, jnp, parameters, artifact.batch, steps)
        seven, seven_hard, seven_seconds = _horizon_forward(
            jax,
            jnp,
            parameters,
            artifact.batch,
            steps,
            horizon=7,
            flux_basis=flux_basis,
        )
        thirty, thirty_hard, thirty_seconds = _horizon_forward(
            jax,
            jnp,
            parameters,
            artifact.batch,
            steps,
            horizon=30,
            flux_basis=flux_basis,
        )
        losses = _source_losses(direct, seven, thirty)
        hard = {
            "one_day": _hard_host(jax.device_get(one_hard)),
            "seven_day": _hard_host(jax.device_get(seven_hard)),
            "thirty_day": _hard_host(jax.device_get(thirty_hard)),
        }
        if (
            not all(np.isfinite(value) and value > 0.0 for value in losses.values())
            or any(_hard_failure_count(value) != 0 for value in hard.values())
        ):
            raise RuntimeError(f"structured GPP {label} panel artifact {update} is invalid")
        records.append(
            {
                "artifact_update": update,
                "runtime_year": int(context["schedule"]["entries"][update - 1]["runtime_year"]),
                "source_losses": losses,
                "hard_counts": hard,
                "timing_seconds": {
                    "direct": direct_seconds,
                    "seven_day": seven_seconds,
                    "thirty_day": thirty_seconds,
                    "total": time.perf_counter() - started,
                },
            }
        )
        print(f"structured_gpp joint_training {label}_panel={update} complete", flush=True)
    means = mean_source_losses(tuple(record["source_losses"] for record in records))
    direct_panel = {
        "updates": list(updates),
        "aggregate_direct_gpp_loss": means[DIRECT_SOURCE],
        "evaluated_values": 208 * len(updates),
        "records": [
            {
                "artifact_update": record["artifact_update"],
                "runtime_year": record["runtime_year"],
                "direct_gpp_loss": record["source_losses"][DIRECT_SOURCE],
                "evaluated_values": 208,
                "elapsed_seconds": record["timing_seconds"]["direct"],
            }
            for record in records
        ],
    }
    return {
        "label": label,
        "artifact_updates": list(updates),
        "source_mean_losses": means,
        "joint_objective": structured_gpp_joint_objectives(means, context["joint_coefficients"]),
        "direct_panel": direct_panel,
        "records": records,
        "hard_gate_passed": True,
    }


def _checkpoint_metadata(
    initial_panel: Mapping[str, Any],
    epoch_panels: Sequence[Mapping[str, Any]],
    epoch_checkpoints: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    return {
        "initial_joint_panel": initial_panel,
        "epoch_panels": list(epoch_panels),
        "epoch_checkpoints": list(epoch_checkpoints),
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    started = time.perf_counter()

    contract = load_structured_gpp_joint_training_contract(args.joint_training_contract)
    identity_contract = contract.payload["identity"]
    if sha256_file(args.batch4_preflight_report) != identity_contract["batch4_preflight_report_sha256"]:
        raise ValueError("structured GPP joint training preflight report hash drift")
    preflight = json.loads(args.batch4_preflight_report.read_text(encoding="utf-8"))
    if (
        preflight.get("status") != "passed"
        or not all(bool(value) for value in preflight["acceptance"].values())
        or preflight["next_gate"]
        != "freeze_bounded_batch4_joint_training_contract_and_request_paid_resource_authorization"
        or preflight["policy"]["optimizer_updates_persisted"] != 0
    ):
        raise ValueError("structured GPP joint training preflight is not accepted")
    if sha256_file(args.prepared_manifest) != identity_contract["prepared_manifest_sha256"]:
        raise ValueError("structured GPP joint training prepared-manifest identity drift")
    if sha256_file(args.warm_start_report) != identity_contract["warm_start_report_sha256"]:
        raise ValueError("structured GPP joint training warm-start report hash drift")
    warm_report = json.loads(args.warm_start_report.read_text(encoding="utf-8"))
    if (
        warm_report.get("status") != "passed"
        or warm_report["checkpoint"]["sha256"] != args.warm_start_checkpoint_sha256
        or args.warm_start_checkpoint_sha256 != identity_contract["warm_start_checkpoint_sha256"]
    ):
        raise ValueError("structured GPP joint training warm-start result is not accepted")

    from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_checkpoint import (
        load_distribution_aware_checkpoint,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_maintenance import (
        initialize_distribution_aware_maintenance_parameters,
        load_distribution_aware_maintenance_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_flux_basis import (
        load_target_oriented_flux_basis_amendment,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_process_head import (
        build_target_oriented_distribution_aware_maintenance_head,
        build_target_oriented_structured_gpp_head,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp import (
        initialize_structured_gpp_parameters,
        load_structured_gpp_contract,
    )

    amendment = load_target_oriented_flux_basis_amendment(args.flux_basis_amendment)
    if amendment.sha256 != identity_contract["flux_basis_amendment_sha256"]:
        raise ValueError("structured GPP joint training flux-basis identity drift")
    context = dict(
        _load_context(
            args,
            science_flux_basis=amendment.flux_basis,
            verify_validation_artifacts=False,
        )
    )
    runtime = context["runtime"]
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    target_contract = context["target_contract"]
    context["joint_coefficients"] = contract.coefficients
    if (
        context["training_contract"].sha256 != identity_contract["base_training_contract_sha256"]
        or target_contract.sha256 != identity_contract["target_contract_sha256"]
    ):
        raise ValueError("structured GPP joint training parent contract identity drift")

    distribution_contract = load_distribution_aware_maintenance_contract(
        args.distribution_contract
    )
    structured_contract = load_structured_gpp_contract(args.structured_contract)
    if structured_contract.sha256 != identity_contract["candidate_contract_sha256"]:
        raise ValueError("structured GPP joint training candidate contract identity drift")
    distribution_report = json.loads(args.distribution_training_report.read_text(encoding="utf-8"))
    if (
        distribution_report.get("status") != "passed"
        or distribution_report["checkpoint"]["sha256"]
        != args.distribution_checkpoint_sha256
        or args.distribution_checkpoint_sha256
        != identity_contract["parent_distribution_checkpoint_sha256"]
    ):
        raise ValueError("structured GPP joint training parent checkpoint identity drift")
    distribution_checkpoint = load_distribution_aware_checkpoint(
        args.distribution_checkpoint,
        expected_sha256=args.distribution_checkpoint_sha256,
        identity=distribution_report["identity"],
        completed_updates=250,
        parameter_count=ADAPTER_PARAMETER_COUNT,
        clip_threshold=float(distribution_report["training"]["clip_threshold"]),
    )
    initialized_parent = initialize_distribution_aware_maintenance_parameters(
        context["parameters"],
        jax.random.PRNGKey(PROBE_SEED + 20260828),
        schema=runtime["learned_schema"],
        target_contract=target_contract,
        distribution_contract=distribution_contract,
    )
    parent_parameters = _replace_adapter(
        initialized_parent,
        jax.device_put(distribution_checkpoint["adapter"]),
    )
    if _tree_size(jax, parent_parameters) != PARENT_PARAMETER_COUNT:
        raise ValueError("structured GPP joint training pre-specialist parent count drift")
    distribution_head = build_target_oriented_distribution_aware_maintenance_head(
        _parent_head(runtime, target_contract),
        context["template"],
        distribution_contract=distribution_contract,
    )
    head = build_target_oriented_structured_gpp_head(
        distribution_head,
        structured_contract=structured_contract,
    )
    initialized_parameters = initialize_structured_gpp_parameters(
        parent_parameters,
        jax.random.PRNGKey(PROBE_SEED + 20260829),
        schema=runtime["learned_schema"],
        target_contract=target_contract,
        structured_contract=structured_contract,
    )
    warm_checkpoint = load_structured_gpp_checkpoint(
        args.warm_start_checkpoint,
        expected_sha256=args.warm_start_checkpoint_sha256,
        identity=warm_report["identity"],
        completed_updates=250,
        parameter_count=SPECIALIST_PARAMETER_COUNT,
        clip_threshold=float(identity_contract["warm_start_clip_threshold"]),
    )
    warm_specialist, warm_optimizer = jax.device_put(
        (warm_checkpoint["specialist"], warm_checkpoint["optimizer"])
    )
    warm_records = list(warm_checkpoint["update_records"])
    if (
        _tree_size(jax, replace_specialist(initialized_parameters, warm_specialist))
        != CANDIDATE_PARAMETER_COUNT
        or specialist_element_count(warm_specialist) != SPECIALIST_PARAMETER_COUNT
        or int(np.asarray(warm_optimizer.step)) != 250
        or len(warm_records) != 250
    ):
        raise ValueError("structured GPP joint training warm endpoint drift")

    steps = _make_steps(
        runtime,
        head,
        context["template"],
        context["scales"],
        target_contract,
        science_flux_basis=amendment.flux_basis,
    )
    fixed_parameters = replace_specialist(initialized_parameters, {})
    gradient_steps = steps["make_trainable_gradient_steps"](
        fixed_parameters, replace_specialist
    )
    direct_steps = _make_direct_steps(runtime, head, initialized_parameters)
    learning_rate = float(contract.payload["optimizer"]["learning_rate"])
    optimizer_step = jax.jit(
        lambda values, derivatives, state: target_oriented_adamw_update(
            values,
            derivatives,
            state,
            learning_rate=jnp.asarray(learning_rate, dtype=jnp.float32),
            clip_norm=jnp.asarray(contract.clip_threshold, dtype=jnp.float32),
            contract=context["training_contract"],
        )
    )
    proposal_check = _make_proposal_check(jax, jnp)
    schedule = build_joint_training_schedule()
    groups = schedule.flat_groups
    maximum_joint_updates = len(groups)
    maximum_absolute_step = 250 + maximum_joint_updates
    recovery_interval = int(
        contract.payload["bounded_stage"]["recovery_interval_joint_updates"]
    )
    memory_before = _memory_report(runtime["devices"][0])
    identity = {
        "git_head": _git_head(),
        "joint_training_contract_sha256": contract.sha256,
        "batch4_preflight_report_sha256": sha256_file(args.batch4_preflight_report),
        "warm_start_report_sha256": sha256_file(args.warm_start_report),
        "warm_start_checkpoint_sha256": args.warm_start_checkpoint_sha256,
        "candidate_contract_sha256": structured_contract.sha256,
        "distribution_checkpoint_sha256": args.distribution_checkpoint_sha256,
        "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
        "base_training_contract_sha256": context["training_contract"].sha256,
        "target_contract_sha256": target_contract.sha256,
        "flux_basis_amendment_sha256": amendment.sha256,
        "materialized_schedule_sha256": schedule.sha256,
    }
    recovery_path = output_dir / "recovery.pkl"
    recovery = load_structured_gpp_recovery(
        recovery_path,
        identity=identity,
        parameter_count=SPECIALIST_PARAMETER_COUNT,
        clip_threshold=contract.clip_threshold,
        maximum_updates=maximum_absolute_step,
    )
    if recovery is None:
        specialist, optimizer = warm_specialist, warm_optimizer
        parameters = replace_specialist(initialized_parameters, specialist)
        initial_panel = _joint_panel(
            args,
            context,
            parameters,
            specialist,
            steps,
            direct_steps,
            contract.panel_updates,
            amendment.flux_basis,
            label="initial",
        )
        epoch_panels: list[Mapping[str, Any]] = []
        epoch_checkpoints: list[Mapping[str, Any]] = []
        joint_records: list[Mapping[str, Any]] = []
        all_records = warm_records
        completed_joint = 0
    else:
        specialist, optimizer = jax.device_put(
            (recovery["specialist"], recovery["optimizer"])
        )
        all_records = list(recovery["update_records"])
        joint_records = all_records[250:]
        completed_joint = int(recovery["completed_updates"]) - 250
        metadata = validate_joint_training_recovery_metadata(
            recovery.get("stage_metadata", {}),
            completed_joint_updates=completed_joint,
        )
        initial_panel = metadata["initial_joint_panel"]
        epoch_panels = list(metadata["epoch_panels"])
        epoch_checkpoints = list(metadata["epoch_checkpoints"])
        _validate_recovered_checkpoint_files(epoch_checkpoints, output_dir)
        expected_prefix = [list(group) for group in groups[:completed_joint]]
        if (
            completed_joint < 0
            or completed_joint > maximum_joint_updates
            or [record["artifact_updates"] for record in joint_records] != expected_prefix
        ):
            raise ValueError("structured GPP joint training recovery schedule drift")
        latest_recovery = {
            "path": str(recovery_path),
            "sha256": sha256_file(recovery_path),
            "completed_updates": int(recovery["completed_updates"]),
            "promotion_eligible": False,
        }
        print(f"structured_gpp joint_training resumed joint_update={completed_joint}", flush=True)
    jax.block_until_ready((specialist, optimizer))

    progress_path = output_dir / "progress.json"
    if recovery is None:
        latest_recovery = None
    stop_after_epoch = bool(
        epoch_panels and not bool(epoch_panels[-1]["continuation_gate_passed"])
    )
    for joint_update in range(completed_joint + 1, maximum_joint_updates + 1):
        if stop_after_epoch:
            break
        group = groups[joint_update - 1]
        epoch = (joint_update - 1) // 62 + 1
        epoch_group = (joint_update - 1) % 62 + 1
        absolute_step = 250 + joint_update
        parameters = replace_specialist(initialized_parameters, specialist)
        update_started = time.perf_counter()
        source_gradients = {name: [] for name in GRADIENT_SOURCES}
        artifact_records = []
        for artifact_update in group:
            artifact_started = time.perf_counter()
            artifact = _load_artifact(args, context, artifact_update)
            direct, direct_gradient, direct_forward_seconds, direct_gradient_seconds = (
                _direct_gradient(jax, jnp, specialist, artifact.batch, direct_steps)
            )
            one_hard = _one_day_hard(jax, jnp, parameters, artifact.batch, steps)
            seven, seven_hard, seven_gradient, seven_timing = _horizon_gradient(
                jax,
                jnp,
                parameters,
                specialist,
                artifact.batch,
                steps,
                gradient_steps,
                horizon=7,
                flux_basis=amendment.flux_basis,
            )
            thirty, thirty_hard, thirty_gradient, thirty_timing = _horizon_gradient(
                jax,
                jnp,
                parameters,
                specialist,
                artifact.batch,
                steps,
                gradient_steps,
                horizon=30,
                flux_basis=amendment.flux_basis,
            )
            gradients = {
                DIRECT_SOURCE: direct_gradient,
                SEVEN_SOURCE: seven_gradient,
                THIRTY_SOURCE: thirty_gradient,
            }
            for name in GRADIENT_SOURCES:
                source_gradients[name].append(gradients[name])
            hard = {
                "one_day": _hard_host(jax.device_get(one_hard)),
                "seven_day": _hard_host(jax.device_get(seven_hard)),
                "thirty_day": _hard_host(jax.device_get(thirty_hard)),
            }
            losses = _source_losses(direct, seven, thirty)
            if (
                not all(np.isfinite(value) and value > 0.0 for value in losses.values())
                or any(_hard_failure_count(value) != 0 for value in hard.values())
            ):
                raise RuntimeError(
                    f"structured GPP joint update {joint_update} artifact {artifact_update} failed"
                )
            artifact_records.append(
                {
                    "artifact_update": artifact_update,
                    "runtime_year": int(
                        context["schedule"]["entries"][artifact_update - 1]["runtime_year"]
                    ),
                    "source_losses": losses,
                    "hard_counts": hard,
                    "timing_seconds": {
                        "direct_forward": direct_forward_seconds,
                        "direct_gradient": direct_gradient_seconds,
                        "seven_day": seven_timing,
                        "thirty_day": thirty_timing,
                        "total": time.perf_counter() - artifact_started,
                    },
                }
            )
        mean_gradients = {
            name: _mean_gradients(jax, jnp, gradients)
            for name, gradients in source_gradients.items()
        }
        combined_gradient = _combine_gradients(
            jax, jnp, mean_gradients, contract.coefficients
        )
        jax.block_until_ready(combined_gradient)
        gradient_reports = {
            name: _tree_scalar_report(jax, jnp, gradient)
            for name, gradient in mean_gradients.items()
        }
        combined_report = _tree_scalar_report(jax, jnp, combined_gradient)
        if any(
            not report["all_finite"]
            or report["element_count"] != SPECIALIST_PARAMETER_COUNT
            or report["l2_norm"] <= 0.0
            or report["nonzero_leaf_count"] != report["leaf_count"]
            for report in (*gradient_reports.values(), combined_report)
        ):
            raise RuntimeError(
                f"structured GPP joint update {joint_update} aggregate gradient failed"
            )
        updated_specialist, updated_optimizer, raw_norm, clip_scale = optimizer_step(
            specialist, combined_gradient, optimizer
        )
        jax.block_until_ready(
            (updated_specialist, updated_optimizer, raw_norm, clip_scale)
        )
        parameter_nonfinite, optimizer_nonfinite, delta_squared = proposal_check(
            specialist, updated_specialist, updated_optimizer
        )
        jax.block_until_ready(
            (parameter_nonfinite, optimizer_nonfinite, delta_squared)
        )
        raw_norm_value = float(np.asarray(raw_norm))
        clip_scale_value = float(np.asarray(clip_scale))
        parameter_nonfinite_count = int(np.asarray(parameter_nonfinite))
        optimizer_nonfinite_count = int(np.asarray(optimizer_nonfinite))
        parameter_delta_l2 = float(np.sqrt(float(np.asarray(delta_squared))))
        if (
            int(np.asarray(updated_optimizer.step)) != absolute_step
            or not np.isfinite(raw_norm_value)
            or raw_norm_value <= 0.0
            or not 0.0 < clip_scale_value <= 1.0
            or parameter_nonfinite_count != 0
            or optimizer_nonfinite_count != 0
            or not np.isfinite(parameter_delta_l2)
            or parameter_delta_l2 <= 0.0
        ):
            raise RuntimeError(
                f"structured GPP joint update {joint_update} optimizer proposal failed"
            )
        proposed_parameters = replace_specialist(
            initialized_parameters, updated_specialist
        )
        probe_artifact = _load_artifact(args, context, group[0])
        post_one_hard = _hard_host(
            jax.device_get(
                _one_day_hard(
                    jax,
                    jnp,
                    proposed_parameters,
                    probe_artifact.batch,
                    steps,
                )
            )
        )
        if _hard_failure_count(post_one_hard) != 0:
            raise RuntimeError(
                f"structured GPP joint update {joint_update} post-update hard gate failed"
            )
        source_means = mean_source_losses(
            tuple(record["source_losses"] for record in artifact_records)
        )
        record = {
            "optimizer_update": absolute_step,
            "joint_update": joint_update,
            "epoch": epoch,
            "epoch_group": epoch_group,
            "artifact_updates": list(group),
            "source_mean_losses": source_means,
            "joint_objective": structured_gpp_joint_objectives(
                source_means, contract.coefficients
            ),
            "aggregate_gradients": {
                "sources": gradient_reports,
                "combined": combined_report,
            },
            "learning_rate": learning_rate,
            "raw_combined_gradient_norm": raw_norm_value,
            "clip_threshold": contract.clip_threshold,
            "clip_scale": clip_scale_value,
            "clipped": clip_scale_value < 1.0,
            "optimizer_proposal": {
                "parameter_nonfinite_count": parameter_nonfinite_count,
                "optimizer_nonfinite_count": optimizer_nonfinite_count,
                "parameter_delta_l2": parameter_delta_l2,
            },
            "artifact_records": artifact_records,
            "post_update_one_day_hard_counts": post_one_hard,
            "timing_seconds": {"total": time.perf_counter() - update_started},
        }
        specialist, optimizer = updated_specialist, updated_optimizer
        joint_records.append(record)
        all_records.append(record)
        _atomic_json(
            progress_path,
            {
                "schema_version": "target_oriented_structured_gpp_joint_training_progress_v1",
                "identity": identity,
                "completed_joint_updates": joint_update,
                "completed_absolute_optimizer_step": absolute_step,
                "epoch_panels": epoch_panels,
                "joint_updates": joint_records,
            },
        )
        print(
            f"structured_gpp joint_training update={joint_update}/{maximum_joint_updates} "
            f"step={absolute_step} epoch={epoch} group={epoch_group} "
            f"loss={record['joint_objective']['joint']:.9g} "
            f"norm={raw_norm_value:.9g} clip={clip_scale_value:.6g} "
            f"seconds={record['timing_seconds']['total']:.3f}",
            flush=True,
        )
        epoch_boundary = epoch_group == 62
        if epoch_boundary:
            panel_parameters = replace_specialist(
                initialized_parameters, specialist
            )
            panel = _joint_panel(
                args,
                context,
                panel_parameters,
                specialist,
                steps,
                direct_steps,
                contract.panel_updates,
                amendment.flux_basis,
                label=f"epoch_{epoch}",
            )
            previous_panel = initial_panel if epoch == 1 else epoch_panels[-1]["panel"]
            changes = panel_source_changes(
                previous_panel["source_mean_losses"], panel["source_mean_losses"]
            )
            continuation = epoch_panel_continuation_passed(changes)
            epoch_record = {
                "epoch": epoch,
                "absolute_optimizer_step": absolute_step,
                "source_changes_from_previous_panel": changes,
                "continuation_gate_passed": continuation,
                "panel": panel,
            }
            epoch_panels.append(epoch_record)
            if continuation:
                checkpoint_path = output_dir / f"stage-update-{absolute_step:04d}.pkl"
                stage_metadata = _checkpoint_metadata(
                    initial_panel, epoch_panels, epoch_checkpoints
                )
                if checkpoint_path.exists():
                    checkpoint_hash = sha256_file(checkpoint_path)
                    load_structured_gpp_checkpoint(
                        checkpoint_path,
                        expected_sha256=checkpoint_hash,
                        identity=identity,
                        completed_updates=absolute_step,
                        parameter_count=SPECIALIST_PARAMETER_COUNT,
                        clip_threshold=contract.clip_threshold,
                    )
                    checkpoint = {
                        "path": str(checkpoint_path),
                        "sha256": checkpoint_hash,
                        "completed_updates": absolute_step,
                        "parameter_count": SPECIALIST_PARAMETER_COUNT,
                        "promotion_eligible": False,
                    }
                else:
                    host_specialist, host_optimizer = jax.device_get(
                        (specialist, optimizer)
                    )
                    checkpoint = write_structured_gpp_checkpoint(
                        checkpoint_path,
                        identity=identity,
                        specialist=host_specialist,
                        optimizer=host_optimizer,
                        completed_updates=absolute_step,
                        parameter_count=SPECIALIST_PARAMETER_COUNT,
                        clip_threshold=contract.clip_threshold,
                        update_records=all_records,
                        initial_panel=initial_panel["direct_panel"],
                        final_panel=panel["direct_panel"],
                        stage_metadata=stage_metadata,
                    )
                epoch_checkpoints.append({"epoch": epoch} | checkpoint)
                print(
                    f"structured_gpp joint_training epoch={epoch} panel=passed "
                    f"checkpoint={absolute_step}",
                    flush=True,
                )
            else:
                stop_after_epoch = True
                print(
                    f"structured_gpp joint_training epoch={epoch} panel=failed stop=true",
                    flush=True,
                )
        if joint_update % recovery_interval == 0 or epoch_boundary:
            host_specialist, host_optimizer = jax.device_get((specialist, optimizer))
            latest_recovery = write_structured_gpp_recovery(
                recovery_path,
                identity=identity,
                specialist=host_specialist,
                optimizer=host_optimizer,
                completed_updates=absolute_step,
                parameter_count=SPECIALIST_PARAMETER_COUNT,
                clip_threshold=contract.clip_threshold,
                update_records=all_records,
                initial_panel=initial_panel["direct_panel"],
                stage_metadata=_checkpoint_metadata(
                    initial_panel, epoch_panels, epoch_checkpoints
                ),
            )
            print(
                f"structured_gpp joint_training recovery_step={absolute_step}",
                flush=True,
            )
        source_gradients.clear()
        del combined_gradient, mean_gradients, parameters, proposed_parameters

    selected_epoch = selected_epoch_from_panels(epoch_panels)
    if selected_epoch == 0:
        selected_checkpoint = {
            "epoch": 0,
            "path": str(args.warm_start_checkpoint.resolve()),
            "sha256": args.warm_start_checkpoint_sha256,
            "completed_updates": 250,
            "parameter_count": SPECIALIST_PARAMETER_COUNT,
            "promotion_eligible": False,
        }
    else:
        selected_checkpoint = epoch_checkpoints[selected_epoch - 1]
    final_parameters = replace_specialist(initialized_parameters, specialist)
    retained = _retained_parameter_report(
        jax,
        replace_specialist(initialized_parameters, warm_specialist),
        final_parameters,
    )
    checkpoint_unchanged = (
        sha256_file(args.warm_start_checkpoint) == args.warm_start_checkpoint_sha256
    )
    hard_and_gradient_passed = all(
        all(
            report["all_finite"]
            and report["element_count"] == SPECIALIST_PARAMETER_COUNT
            and report["l2_norm"] > 0.0
            and report["nonzero_leaf_count"] == report["leaf_count"]
            for report in (
                *record["aggregate_gradients"]["sources"].values(),
                record["aggregate_gradients"]["combined"],
            )
        )
        and all(
            _hard_failure_count(value) == 0
            for artifact in record["artifact_records"]
            for value in artifact["hard_counts"].values()
        )
        and _hard_failure_count(record["post_update_one_day_hard_counts"]) == 0
        for record in joint_records
    )
    execution_passed = (
        len(all_records) == int(np.asarray(optimizer.step))
        and len(joint_records) == int(np.asarray(optimizer.step)) - 250
        and hard_and_gradient_passed
        and retained["exactly_unchanged"]
        and checkpoint_unchanged
    )
    if not execution_passed:
        raise RuntimeError("structured GPP joint training execution gate failed")
    norms = np.asarray(
        [record["raw_combined_gradient_norm"] for record in joint_records],
        dtype=np.float64,
    )
    timings = np.asarray(
        [record["timing_seconds"]["total"] for record in joint_records],
        dtype=np.float64,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "identity": identity,
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "validation_values_evaluated": False,
            "sealed_test_read": False,
            "joint_updates_executed": len(joint_records),
            "artifact_presentations": 4 * len(joint_records),
            "retained_parent_parameters_trainable": False,
            "recovery_written": latest_recovery is not None,
            "failed_epoch_checkpoint_selected": False,
            "checkpoint_promotion_eligible": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "model": {
            "candidate_parameter_count": CANDIDATE_PARAMETER_COUNT,
            "trainable_specialist_parameter_count": SPECIALIST_PARAMETER_COUNT,
            "retained_parent_parameters": retained,
        },
        "training": {
            "effective_batch_size": 4,
            "maximum_epochs": 4,
            "completed_epoch_panels": len(epoch_panels),
            "selected_epoch": selected_epoch,
            "stopped_after_failed_epoch_panel": bool(
                epoch_panels and not epoch_panels[-1]["continuation_gate_passed"]
            ),
            "learning_rate": learning_rate,
            "clip_threshold": contract.clip_threshold,
            "clipped_update_count": sum(bool(record["clipped"]) for record in joint_records),
            "raw_combined_gradient_norm": {
                "minimum": float(np.min(norms)),
                "median": float(np.median(norms)),
                "maximum": float(np.max(norms)),
            },
            "update_seconds": {
                "minimum": float(np.min(timings)),
                "median": float(np.median(timings)),
                "maximum": float(np.max(timings)),
                "total": float(np.sum(timings)),
            },
            "joint_updates": joint_records,
        },
        "initial_joint_panel": initial_panel,
        "epoch_panels": epoch_panels,
        "epoch_checkpoints": epoch_checkpoints,
        "selected_checkpoint": selected_checkpoint,
        "latest_recovery": latest_recovery,
        "acceptance": {
            "execution_and_hard_gates_passed": execution_passed,
            "retained_parent_parameters_passed": retained["exactly_unchanged"],
            "warm_start_checkpoint_unchanged": checkpoint_unchanged,
            "new_checkpoint_selected": selected_epoch > 0,
        },
        "memory_before": memory_before,
        "memory_after": _memory_report(runtime["devices"][0]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": contract.payload["next_gate"][
            "on_new_checkpoint_selected" if selected_epoch > 0 else "on_no_epoch_pass"
        ],
    }
    _atomic_json(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--flux-basis-amendment", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-updates", type=int, default=1000)
    parser.add_argument("--validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--stage-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-checkpoint-sha256", required=True)
    parser.add_argument("--distribution-contract", type=Path, required=True)
    parser.add_argument("--distribution-training-report", type=Path, required=True)
    parser.add_argument("--distribution-checkpoint", type=Path, required=True)
    parser.add_argument("--distribution-checkpoint-sha256", required=True)
    parser.add_argument("--structured-contract", type=Path, required=True)
    parser.add_argument("--warm-start-report", type=Path, required=True)
    parser.add_argument("--warm-start-checkpoint", type=Path, required=True)
    parser.add_argument("--warm-start-checkpoint-sha256", required=True)
    parser.add_argument("--batch4-preflight-report", type=Path, required=True)
    parser.add_argument("--joint-training-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
