"""Attribute direct-GPP monitor regression along persistent updates 1-10."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_operator.gradient_feasibility import (
    gradient_direction_diagnostics,
)
from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_direct_generalization import (
    AUDITED_PERSISTENT_UPDATES,
    PERSISTENT_RESULT_SHA256,
    SELECTED_SCALES,
    cosine_matrix_from_gram,
    directional_attribution,
    finite_loss_changes,
    first_adjacent_regression,
    load_coupled_gpp_direct_generalization_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    DIRECT_SOURCE,
    GRADIENT_SOURCES,
    replace_coupled_gpp_trainable,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_persistent_scale_training import (
    build_persistent_scale_training_schedule,
    load_coupled_gpp_persistent_scale_training_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_training import (
    structured_gpp_direct_loss,
)
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import _load_artifact
from scripts.hpc.probe_successor_direct_trust_region import _make_combined_direction
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_gradient_balanced_smoke import _make_gradient_gram
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_coupled_gpp_direct_recalibration import (
    _make_dtype_preserving_direct_virtual_step,
    _measure_source_gradients,
)
from scripts.hpc.run_target_oriented_coupled_gpp_persistent_scale_training import (
    _parser as _persistent_parser,
)
from scripts.hpc.run_target_oriented_coupled_gpp_scale_selecting_recalibration import (
    _git_head,
)
from scripts.hpc.run_target_oriented_coupled_gpp_scale_selecting_recalibration import (
    run as _run_scale_selecting,
)
from scripts.hpc.run_target_oriented_coupled_gpp_update2_geometry import _scaled_parameters
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import (
    _direct_forward,
    _direct_gradient,
)

SCHEMA_VERSION = "target_oriented_coupled_gpp_direct_generalization_audit_result_v1"


def _load_persistent_result(
    path: Path,
    *,
    audit_contract: Any,
    persistent_contract: Any,
    schedule: Any,
) -> Mapping[str, Any]:
    if sha256_file(path) != PERSISTENT_RESULT_SHA256:
        raise ValueError("direct-GPP generalization persistent result identity drift")
    report = json.loads(path.read_text(encoding="utf-8"))
    training = report.get("training", {})
    updates = tuple(training.get("updates", ()))
    observed_groups = tuple(tuple(int(value) for value in item.get("artifact_updates", ())) for item in updates)
    observed_scales = tuple(float(item.get("selected_scale", -1.0)) for item in updates)
    milestone = report.get("milestone_monitors", ())
    if (
        report.get("schema_version") != "target_oriented_coupled_gpp_persistent_scale_training_result_v1"
        or report.get("status") != "stopped_scientific_gate"
        or report.get("classification") != "milestone_scientific_gate"
        or report.get("identity", {}).get("persistent_contract_sha256") != persistent_contract.sha256
        or report.get("identity", {}).get("prepared_manifest_sha256")
        != audit_contract.payload["identity"]["prepared_manifest_sha256"]
        or training.get("completed_updates") != len(AUDITED_PERSISTENT_UPDATES)
        or training.get("stopped_reason") != "milestone_scientific_gate"
        or observed_groups != schedule.groups[: len(AUDITED_PERSISTENT_UPDATES)]
        or observed_scales != SELECTED_SCALES
        or any(item.get("finite_step_accepted") is not True for item in updates)
        or len(milestone) != 1
        or int(milestone[0].get("persistent_update", -1)) != 10
        or report.get("selected_checkpoint") is not None
        or report.get("checkpoints") != []
    ):
        raise ValueError("direct-GPP generalization persistent result content drift")
    return report


def _direct_loss(value: Any) -> float:
    result = float(np.asarray(structured_gpp_direct_loss(value)))
    if not np.isfinite(result) or result <= 0.0:
        raise RuntimeError("direct-GPP audit encountered an invalid loss")
    return result


def _direct_panel(
    args: Any,
    context: Mapping[str, Any],
    *,
    trainable: Any,
    direct_steps: Mapping[str, Any],
    updates: Sequence[int],
    label: str,
) -> Mapping[str, Any]:
    jax = context["runtime"]["jax"]
    records = []
    for ordinal, update in enumerate(updates, start=1):
        artifact = _load_artifact(args, context, int(update))
        direct, seconds = _direct_forward(jax, trainable, artifact.batch, direct_steps)
        records.append(
            {
                "artifact_update": int(update),
                "runtime_year": int(context["schedule"]["entries"][int(update) - 1]["runtime_year"]),
                "loss": _direct_loss(direct),
                "seconds": seconds,
            }
        )
        print(
            f"coupled_gpp direct_generalization {label}={ordinal}/{len(updates)} artifact={update}",
            flush=True,
        )
    losses = tuple(float(record["loss"]) for record in records)
    return {
        "artifact_updates": [int(value) for value in updates],
        "mean_loss": float(np.mean(np.asarray(losses, dtype=np.float64))),
        "artifact_losses": records,
    }


def _mean_tree(jax: Any, jnp: Any, trees: Sequence[Any]) -> Any:
    values = tuple(trees)
    if not values:
        raise ValueError("direct-GPP gradient mean requires at least one tree")
    scale = jnp.asarray(1.0 / len(values), dtype=jnp.float32)
    return jax.tree_util.tree_map(
        lambda *leaves: (
            sum(
                (jnp.asarray(leaf, dtype=jnp.float32) for leaf in leaves),
                jnp.asarray(0.0, dtype=jnp.float32),
            )
            * scale
        ),
        *values,
    )


def _pair_summary(values: Any) -> Mapping[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("direct-GPP cosine group is invalid")
    return {
        "count": int(len(array)),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
        "negative_count": int(np.sum(array < 0.0)),
        "positive_count": int(np.sum(array > 0.0)),
    }


def _cosine_report(
    cosine: np.ndarray,
    *,
    labels: Sequence[str],
    training_count: int,
    monitor_count: int,
) -> Mapping[str, Any]:
    expected = training_count + monitor_count + 3
    if cosine.shape != (expected, expected) or len(labels) != expected:
        raise ValueError("direct-GPP cosine registry drift")
    train = np.arange(0, training_count)
    monitor = np.arange(training_count, training_count + monitor_count)
    training_mean = training_count + monitor_count
    monitor_mean = training_mean + 1
    applied = monitor_mean + 1
    train_upper = cosine[np.ix_(train, train)][np.triu_indices(training_count, k=1)]
    monitor_upper = cosine[np.ix_(monitor, monitor)][np.triu_indices(monitor_count, k=1)]
    cross = cosine[np.ix_(train, monitor)].reshape(-1)
    return {
        "labels": list(labels),
        "matrix": cosine.tolist(),
        "within_training_artifacts": _pair_summary(train_upper),
        "within_monitor_artifacts": _pair_summary(monitor_upper),
        "training_monitor_cross_artifacts": _pair_summary(cross),
        "training_artifacts_to_training_mean": _pair_summary(cosine[train, training_mean]),
        "monitor_artifacts_to_monitor_mean": _pair_summary(cosine[monitor, monitor_mean]),
        "training_artifacts_to_applied_descent": _pair_summary(cosine[train, applied]),
        "monitor_artifacts_to_applied_descent": _pair_summary(cosine[monitor, applied]),
        "training_monitor_mean_cosine": float(cosine[training_mean, monitor_mean]),
        "training_mean_applied_descent_cosine": float(cosine[training_mean, applied]),
        "monitor_mean_applied_descent_cosine": float(cosine[monitor_mean, applied]),
    }


def _replay_direction(
    args: Any,
    context: Mapping[str, Any],
    *,
    initialized_parameters: Any,
    trainable: Any,
    steps: Mapping[str, Any],
    gradient_steps: Mapping[str, Any],
    direct_steps: Mapping[str, Any],
    state_steps: Mapping[str, Any],
    layout: Any,
    group: Sequence[int],
    flux_basis: Sequence[str],
    persistent_contract: Any,
    persistent_update: int,
    expected_record: Mapping[str, Any],
    selected_scale: float,
    gram_measurement: Any,
    combine_direction: Any,
    direct_virtual_step: Any,
) -> Mapping[str, Any]:
    runtime = context["runtime"]
    jax = runtime["jax"]
    parameters = replace_coupled_gpp_trainable(initialized_parameters, trainable)
    measurement = _measure_source_gradients(
        args,
        context,
        parameters=parameters,
        trainable=trainable,
        steps=steps,
        gradient_steps=gradient_steps,
        direct_steps=direct_steps,
        state_steps=state_steps,
        layout=layout,
        group=group,
        flux_basis=flux_basis,
        macro_update=persistent_update,
    )
    ordered = tuple(measurement["source_gradients"][name] for name in GRADIENT_SOURCES)
    gram, nonfinite = gram_measurement(ordered)
    jax.block_until_ready((gram, nonfinite))
    gram_host = np.asarray(gram, dtype=np.float64)
    nonfinite_host = np.asarray(nonfinite, dtype=np.int64)
    diagnostics = gradient_direction_diagnostics(gram_host, labels=GRADIENT_SOURCES)
    selected = diagnostics["directions"]["minimum_norm_unit"]
    minimum_cosine = float(persistent_contract.payload["direction"]["minimum_source_cosine"])
    common = bool(
        np.all(nonfinite_host == 0)
        and all(
            float(selected["normalized_directional_derivatives"][name]) >= minimum_cosine for name in GRADIENT_SOURCES
        )
    )
    weights = np.asarray([selected["weights"][name] for name in GRADIENT_SOURCES], dtype=np.float64)
    common_gradient, common_norm, common_nonfinite = combine_direction(ordered, weights)
    jax.block_until_ready((common_gradient, common_norm, common_nonfinite))
    proposed, proposal_nonfinite = direct_virtual_step(
        trainable,
        common_gradient,
        common_norm,
        np.asarray(float(persistent_contract.payload["direction"]["full_parameter_step_l2"]), dtype=np.float64),
    )
    jax.block_until_ready((proposed, proposal_nonfinite))
    expected_losses = expected_record["source_mean_losses_before"]
    losses_match = all(
        np.isclose(
            float(measurement["source_mean_losses"][name]),
            float(expected_losses[name]),
            rtol=1.0e-6,
            atol=1.0e-9,
        )
        for name in GRADIENT_SOURCES
    )
    if (
        not common
        or int(np.asarray(common_nonfinite)) != 0
        or int(np.asarray(proposal_nonfinite)) != 0
        or tuple(int(value) for value in expected_record["artifact_updates"]) != tuple(int(value) for value in group)
        or float(expected_record["selected_scale"]) != float(selected_scale)
        or expected_record["finite_step_accepted"] is not True
        or not losses_match
    ):
        raise RuntimeError(f"direct-GPP persistent update {persistent_update} replay drift")
    updated = _scaled_parameters(jax, trainable, proposed, selected_scale)
    return {
        "updated_trainable": updated,
        "record": {
            "persistent_update": persistent_update,
            "artifact_updates": [int(value) for value in group],
            "selected_scale": float(selected_scale),
            "source_mean_losses_before": measurement["source_mean_losses"],
            "source_losses_match_persistent_result": losses_match,
            "minimum_source_cosine": min(
                float(selected["normalized_directional_derivatives"][name]) for name in GRADIENT_SOURCES
            ),
            "combined_gradient_norm": float(np.asarray(common_norm)),
            "parameter_delta_l2": float(selected_scale)
            * float(persistent_contract.payload["direction"]["full_parameter_step_l2"]),
        },
    }


def _artifact_gradients_and_changes(
    args: Any,
    context: Mapping[str, Any],
    *,
    before_trainable: Any,
    after_trainable: Any,
    direct_steps: Mapping[str, Any],
    updates: Sequence[int],
    label: str,
) -> Mapping[str, Any]:
    jax = context["runtime"]["jax"]
    jnp = context["runtime"]["jnp"]
    gradients = []
    records = []
    for ordinal, update in enumerate(updates, start=1):
        started = time.perf_counter()
        artifact = _load_artifact(args, context, int(update))
        before, gradient, forward_seconds, backward_seconds = _direct_gradient(
            jax,
            jnp,
            before_trainable,
            artifact.batch,
            direct_steps,
        )
        after, after_seconds = _direct_forward(jax, after_trainable, artifact.batch, direct_steps)
        gradients.append(gradient)
        records.append(
            {
                "artifact_update": int(update),
                "runtime_year": int(context["schedule"]["entries"][int(update) - 1]["runtime_year"]),
                "before_loss": _direct_loss(before),
                "after_loss": _direct_loss(after),
                "timing_seconds": {
                    "before_forward": forward_seconds,
                    "gradient": backward_seconds,
                    "after_forward": after_seconds,
                    "total": time.perf_counter() - started,
                },
            }
        )
        print(
            f"coupled_gpp direct_generalization {label}_gradient={ordinal}/{len(updates)} artifact={update}",
            flush=True,
        )
    changes = finite_loss_changes(
        [record["before_loss"] for record in records],
        [record["after_loss"] for record in records],
    )
    records = [dict(record, finite_change=change) for record, change in zip(records, changes, strict=True)]
    return {"gradients": tuple(gradients), "records": records}


def run(args: Any) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    audit_contract = load_coupled_gpp_direct_generalization_contract(args.generalization_contract)
    persistent_contract = load_coupled_gpp_persistent_scale_training_contract(args.persistent_contract)
    if persistent_contract.sha256 != audit_contract.payload["identity"]["persistent_contract_sha256"]:
        raise ValueError("direct-GPP generalization persistent contract chain drift")
    schedule = build_persistent_scale_training_schedule()
    persistent_result = _load_persistent_result(
        args.persistent_result,
        audit_contract=audit_contract,
        persistent_contract=persistent_contract,
        schedule=schedule,
    )

    reconstructed, internal = _run_scale_selecting(args, write_output=False, return_internal=True)
    if reconstructed.get("status") != "passed" or reconstructed.get("smoke", {}).get("updates_accepted") != [5, 6, 7]:
        raise RuntimeError("direct-GPP generalization update-7 reconstruction failed")
    context = internal["context"]
    runtime = context["runtime"]
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    required = {value for group in schedule.groups[: len(AUDITED_PERSISTENT_UPDATES)] for value in group} | set(
        audit_contract.monitor_updates
    )
    missing = sorted(required - set(context["prepared_records"]))
    if missing:
        raise ValueError(f"direct-GPP generalization prepared coverage drift: {missing}")

    initialized_parameters = internal["initialized_parameters"]
    fixed_parameters = replace_coupled_gpp_trainable(initialized_parameters, {})
    steps = internal["steps"]
    gradient_steps = steps["make_trainable_gradient_steps"](
        fixed_parameters,
        replace_coupled_gpp_trainable,
    )
    direct_steps = internal["direct_steps"]
    state_steps = internal["state_steps"]
    layout = internal["layout"]
    flux_basis = internal["flux_basis"]
    trainable = internal["final_trainable"]
    memory_before = _memory_report(internal["device"])
    gram_measurement = _make_gradient_gram(jax)
    combine_direction = _make_combined_direction(jax)
    direct_virtual_step = _make_dtype_preserving_direct_virtual_step(jax)

    initial_panel = _direct_panel(
        args,
        context,
        trainable=trainable,
        direct_steps=direct_steps,
        updates=audit_contract.monitor_updates,
        label="trajectory_0",
    )
    persistent_initial = float(persistent_result["initial_monitor"]["source_mean_losses"][DIRECT_SOURCE])
    rtol = float(audit_contract.payload["trajectory"]["persistent_result_loss_match_rtol"])
    atol = float(audit_contract.payload["trajectory"]["persistent_result_loss_match_atol"])
    if not np.isclose(initial_panel["mean_loss"], persistent_initial, rtol=rtol, atol=atol):
        raise RuntimeError("direct-GPP initial monitor replay drift")

    trajectory = [dict(persistent_update=0, panel=initial_panel)]
    replay_records = []
    first_state = None
    previous_loss = float(initial_panel["mean_loss"])
    expected_updates = persistent_result["training"]["updates"]
    for index, persistent_update in enumerate(AUDITED_PERSISTENT_UPDATES):
        before_trainable = trainable
        replayed = _replay_direction(
            args,
            context,
            initialized_parameters=initialized_parameters,
            trainable=trainable,
            steps=steps,
            gradient_steps=gradient_steps,
            direct_steps=direct_steps,
            state_steps=state_steps,
            layout=layout,
            group=schedule.groups[index],
            flux_basis=flux_basis,
            persistent_contract=persistent_contract,
            persistent_update=persistent_update,
            expected_record=expected_updates[index],
            selected_scale=SELECTED_SCALES[index],
            gram_measurement=gram_measurement,
            combine_direction=combine_direction,
            direct_virtual_step=direct_virtual_step,
        )
        trainable = replayed["updated_trainable"]
        panel = _direct_panel(
            args,
            context,
            trainable=trainable,
            direct_steps=direct_steps,
            updates=audit_contract.monitor_updates,
            label=f"trajectory_{persistent_update}",
        )
        current_loss = float(panel["mean_loss"])
        transition_change = (current_loss - previous_loss) / previous_loss
        record = dict(
            replayed["record"], monitor_mean_loss=current_loss, monitor_adjacent_relative_change=transition_change
        )
        replay_records.append(record)
        trajectory.append(dict(persistent_update=persistent_update, panel=panel))
        if first_state is None and first_adjacent_regression((previous_loss, current_loss)) == 1:
            first_state = {
                "persistent_update": persistent_update,
                "before_trainable": before_trainable,
                "after_trainable": trainable,
                "training_group": tuple(schedule.groups[index]),
                "training_direct_loss_before": float(replayed["record"]["source_mean_losses_before"][DIRECT_SOURCE]),
                "monitor_direct_loss_before": previous_loss,
                "monitor_direct_loss_after": current_loss,
            }
        previous_loss = current_loss
        print(
            f"coupled_gpp direct_generalization replay={persistent_update}/10 monitor={current_loss:.12g} ",
            flush=True,
        )

    trajectory_losses = tuple(float(item["panel"]["mean_loss"]) for item in trajectory)
    first_regression = first_adjacent_regression(trajectory_losses)
    persistent_final = float(persistent_result["milestone_monitors"][0]["source_mean_losses"][DIRECT_SOURCE])
    endpoint_matches = bool(np.isclose(trajectory_losses[-1], persistent_final, rtol=rtol, atol=atol))
    if first_state is None or first_regression is None or first_state["persistent_update"] != first_regression:
        raise RuntimeError("direct-GPP first adjacent regression was not retained")
    if not endpoint_matches:
        raise RuntimeError("direct-GPP update-10 monitor replay drift")

    training = _artifact_gradients_and_changes(
        args,
        context,
        before_trainable=first_state["before_trainable"],
        after_trainable=first_state["after_trainable"],
        direct_steps=direct_steps,
        updates=first_state["training_group"],
        label="training",
    )
    monitor = _artifact_gradients_and_changes(
        args,
        context,
        before_trainable=first_state["before_trainable"],
        after_trainable=first_state["after_trainable"],
        direct_steps=direct_steps,
        updates=audit_contract.monitor_updates,
        label="monitor",
    )
    training_gradients = training["gradients"]
    monitor_gradients = monitor["gradients"]
    training_mean = _mean_tree(jax, jnp, training_gradients)
    monitor_mean = _mean_tree(jax, jnp, monitor_gradients)
    applied_descent = jax.tree_util.tree_map(
        lambda before, after: before - after,
        first_state["before_trainable"],
        first_state["after_trainable"],
    )
    gradients = (*training_gradients, *monitor_gradients, training_mean, monitor_mean, applied_descent)
    labels = tuple(
        [f"training_artifact_{value}" for value in first_state["training_group"]]
        + [f"monitor_artifact_{value}" for value in audit_contract.monitor_updates]
        + ["training_mean", "monitor_mean", "applied_parameter_descent"]
    )
    gram, nonfinite = gram_measurement(gradients)
    jax.block_until_ready((gram, nonfinite))
    gram_host = np.asarray(gram, dtype=np.float64)
    nonfinite_host = np.asarray(nonfinite, dtype=np.int64)
    cosine_report = _cosine_report(
        cosine_matrix_from_gram(gram_host),
        labels=labels,
        training_count=len(training_gradients),
        monitor_count=len(monitor_gradients),
    )
    training_before = float(np.mean([item["before_loss"] for item in training["records"]]))
    training_after = float(np.mean([item["after_loss"] for item in training["records"]]))
    monitor_before = float(np.mean([item["before_loss"] for item in monitor["records"]]))
    monitor_after = float(np.mean([item["after_loss"] for item in monitor["records"]]))
    means_match = bool(
        np.isclose(training_before, first_state["training_direct_loss_before"], rtol=rtol, atol=atol)
        and np.isclose(monitor_before, first_state["monitor_direct_loss_before"], rtol=rtol, atol=atol)
        and np.isclose(monitor_after, first_state["monitor_direct_loss_after"], rtol=rtol, atol=atol)
    )
    engineering_passed = bool(
        len(replay_records) == len(AUDITED_PERSISTENT_UPDATES)
        and endpoint_matches
        and np.all(nonfinite_host == 0)
        and np.all(np.diag(gram_host) > 0.0)
        and means_match
    )
    if not engineering_passed:
        raise RuntimeError("direct-GPP attribution engineering gate failed")

    monitor_relative_change = (monitor_after - monitor_before) / monitor_before
    attribution = directional_attribution(
        training_monitor_mean_cosine=cosine_report["training_monitor_mean_cosine"],
        training_applied_descent_cosine=cosine_report["training_mean_applied_descent_cosine"],
        monitor_applied_descent_cosine=cosine_report["monitor_mean_applied_descent_cosine"],
        monitor_mean_relative_change=monitor_relative_change,
        negative_within_training_pairs=cosine_report["within_training_artifacts"]["negative_count"],
        negative_within_monitor_pairs=cosine_report["within_monitor_artifacts"]["negative_count"],
        negative_cross_pairs=cosine_report["training_monitor_cross_artifacts"]["negative_count"],
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "classification": "direct_gpp_generalization_attribution_completed",
        "identity": {
            "git_head": _git_head(),
            "audit_contract_sha256": audit_contract.sha256,
            "persistent_contract_sha256": persistent_contract.sha256,
            "persistent_result_sha256": sha256_file(args.persistent_result),
            "schedule_sha256": schedule.sha256,
        },
        "policy": {
            "training_updates_persisted": 0,
            "checkpoint_written": False,
            "persistent_recovery_read": False,
            "recovery_written": False,
            "trainable_parameters_written": False,
            "fresh_validation_values_evaluated": False,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "sealed_test_read": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "trajectory": {
            "initial_persistent_update": 0,
            "final_persistent_update": 10,
            "first_adjacent_regression_update": first_regression,
            "losses": [
                {
                    "persistent_update": int(item["persistent_update"]),
                    "mean_direct_gpp_loss": float(item["panel"]["mean_loss"]),
                    "artifact_losses": item["panel"]["artifact_losses"],
                }
                for item in trajectory
            ],
            "replay_updates": replay_records,
            "endpoint_matches_persistent_result": endpoint_matches,
        },
        "first_regression_attribution": {
            "persistent_update": first_regression,
            "training_artifact_updates": list(first_state["training_group"]),
            "monitor_artifact_updates": list(audit_contract.monitor_updates),
            "training_mean_finite_change": {
                "before": training_before,
                "after": training_after,
                "relative_change": (training_after - training_before) / training_before,
            },
            "monitor_mean_finite_change": {
                "before": monitor_before,
                "after": monitor_after,
                "relative_change": monitor_relative_change,
            },
            "training_artifacts": training["records"],
            "monitor_artifacts": monitor["records"],
            "gradient_nonfinite_counts": {label: int(nonfinite_host[index]) for index, label in enumerate(labels)},
            "gradient_l2_norms": {label: float(np.sqrt(gram_host[index, index])) for index, label in enumerate(labels)},
            "cosines": cosine_report,
            "directional_attribution": attribution,
            "mean_loss_integrity_passed": means_match,
        },
        "acceptance": {
            "all_ten_updates_replayed": len(replay_records) == len(AUDITED_PERSISTENT_UPDATES),
            "first_adjacent_regression_located": first_regression is not None,
            "all_sixteen_direct_gradients_finite_nonzero": bool(
                np.all(nonfinite_host[:16] == 0) and np.all(np.diag(gram_host)[:16] > 0.0)
            ),
            "trajectory_endpoint_matches_persistent_result": endpoint_matches,
            "engineering_gate_passed": engineering_passed,
            "scientific_model_promotion_implied": False,
        },
        "memory_before": memory_before,
        "memory_after": _memory_report(internal["device"]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": audit_contract.payload["next_gate"]["on_completed_attribution"],
    }
    _atomic_json(output, report)
    return report


def _parser() -> Any:
    parser = _persistent_parser()
    parser.add_argument("--generalization-contract", type=Path, required=True)
    parser.add_argument("--persistent-result", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
