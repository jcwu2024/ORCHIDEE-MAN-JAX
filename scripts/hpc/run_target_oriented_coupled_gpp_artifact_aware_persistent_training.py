"""Run bounded, resumable artifact-aware coupled-GPP training."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_artifact_aware_continuity import (
    load_coupled_gpp_artifact_aware_continuity_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_artifact_aware_persistence import (
    FINAL_LOGICAL_UPDATE,
    START_COMPLETED_LOGICAL_UPDATE,
    load_checkpoint,
    load_recovery,
    load_start_snapshot,
    parameter_trees_identical,
    write_checkpoint,
    write_recovery,
    write_start_snapshot,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_artifact_aware_persistent_training import (
    CONTINUITY_RESULT_SHA256,
    LOGICAL_UPDATES,
    artifact_aware_monitor_gate,
    load_coupled_gpp_artifact_aware_persistent_training_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_direct_generalization import (
    load_coupled_gpp_direct_generalization_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    RETAINED_PARAMETER_COUNT,
    TRAINABLE_PARAMETER_COUNT,
    replace_coupled_gpp_trainable,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_persistent_scale_training import (
    build_persistent_scale_training_schedule,
    load_coupled_gpp_persistent_scale_training_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    derive_target_oriented_acceptance_scales,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_artifact_aware_continuity import (
    _parser as _continuity_parser,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_artifact_aware_continuity import (
    _train_only_update,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_direct_generalization import (
    _bootstrap_update7,
    _load_persistent_result,
    _replay_direction,
)
from scripts.hpc.probe_successor_direct_trust_region import _make_combined_direction
from scripts.hpc.probe_successor_multibatch_direction import _make_flattened_unit_gradient
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_gradient_balanced_smoke import _make_gradient_gram
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_coupled_gpp_bounded_training import _fixed_parameter_report
from scripts.hpc.run_target_oriented_coupled_gpp_direct_recalibration import (
    _git_head,
    _make_dtype_preserving_direct_virtual_step,
)
from scripts.hpc.run_target_oriented_coupled_gpp_direct_recalibration import (
    run as _initialize_coupled_gpp,
)
from scripts.hpc.run_target_oriented_coupled_gpp_persistent_scale_training import (
    _evaluate_detailed_monitor,
)
from scripts.hpc.run_target_oriented_full_step_preflight import _make_steps

SCHEMA_VERSION = "target_oriented_coupled_gpp_artifact_aware_persistent_training_result_v1"


def _load_continuity_result(
    path: Path,
    *,
    continuity_contract_sha256: str,
) -> Mapping[str, Any]:
    digest = sha256_file(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        digest != CONTINUITY_RESULT_SHA256
        or report.get("schema_version") != "target_oriented_coupled_gpp_artifact_aware_continuity_smoke_result_v1"
        or report.get("status") != "passed"
        or report.get("classification") != "artifact_aware_three_update_continuity_passes_monitor"
        or report.get("identity", {}).get("continuity_contract_sha256") != continuity_contract_sha256
        or report.get("train_only", {}).get("accepted_updates") != 3
        or report.get("integrity", {}).get("engineering_gate_passed") is not True
    ):
        raise ValueError("artifact-aware persistent continuity result drift")
    return report


def _initialization_only(args: Any) -> Mapping[str, Any]:
    report, internal = _initialize_coupled_gpp(
        args,
        write_output=False,
        return_internal=True,
        initialization_only=True,
    )
    if report.get("status") != "passed":
        raise RuntimeError("artifact-aware persistent context initialization failed")
    return internal


def _reconstruct_start(
    args: Any,
    *,
    parent_contract: Any,
    parent_result: Mapping[str, Any],
    schedule: Any,
) -> tuple[Mapping[str, Any], Any, Mapping[str, Any]]:
    bootstrap = _bootstrap_update7(args, parent_contract)
    jax = bootstrap["context"]["runtime"]["jax"]
    expected = parent_result["training"]["updates"][0]
    replay = _replay_direction(
        args,
        bootstrap["context"],
        initialized_parameters=bootstrap["initialized_parameters"],
        trainable=bootstrap["trainable"],
        steps=bootstrap["steps"],
        gradient_steps=bootstrap["gradient_steps"],
        direct_steps=bootstrap["direct_steps"],
        state_steps=bootstrap["state_steps"],
        layout=bootstrap["layout"],
        group=schedule.groups[0],
        flux_basis=bootstrap["flux_basis"],
        persistent_contract=parent_contract,
        persistent_update=1,
        expected_record=expected,
        selected_scale=float(expected["selected_scale"]),
        gram_measurement=_make_gradient_gram(jax),
        combine_direction=_make_combined_direction(jax),
        direct_virtual_step=_make_dtype_preserving_direct_virtual_step(jax),
    )
    return bootstrap, replay["updated_trainable"], replay["record"]


def _prepare_bootstrap(
    args: Any,
    *,
    start_exists: bool,
    parent_contract: Any,
    parent_result: Mapping[str, Any],
    schedule: Any,
) -> tuple[Mapping[str, Any], Any | None, Mapping[str, Any] | None, str]:
    if start_exists:
        return _initialization_only(args), None, None, "identity_exact_resume"
    bootstrap, reconstructed_start, replay_record = _reconstruct_start(
        args,
        parent_contract=parent_contract,
        parent_result=parent_result,
        schedule=schedule,
    )
    return bootstrap, reconstructed_start, replay_record, "one_time_reconstruction"


def _with_monitor_steps(bootstrap: Mapping[str, Any]) -> Mapping[str, Any]:
    context = bootstrap["context"]
    runtime = context["runtime"]
    fixed_parameters = replace_coupled_gpp_trainable(bootstrap["initialized_parameters"], {})
    acceptance_scales = derive_target_oriented_acceptance_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    monitor_steps = _make_steps(
        runtime,
        bootstrap["head"],
        context["template"],
        context["scales"],
        context["target_contract"],
        acceptance_scales=acceptance_scales,
        acceptance_flux_names=("gpp",),
        science_flux_basis=bootstrap["flux_basis"],
    )
    steps = dict(bootstrap["steps"])
    steps.update(
        seven_monitor_forward=monitor_steps["seven_monitor_forward"],
        thirty_monitor_forward=monitor_steps["thirty_monitor_forward"],
    )
    return dict(
        bootstrap,
        steps=steps,
        gradient_steps=steps["make_trainable_gradient_steps"](
            fixed_parameters,
            replace_coupled_gpp_trainable,
        ),
    )


def _monitor(
    args: Any,
    bootstrap: Mapping[str, Any],
    *,
    trainable: Any,
    label: str,
) -> Mapping[str, Any]:
    parameters = replace_coupled_gpp_trainable(bootstrap["initialized_parameters"], trainable)
    return _evaluate_detailed_monitor(
        args,
        bootstrap["context"],
        parameters=parameters,
        trainable=trainable,
        steps=bootstrap["steps"],
        direct_steps=bootstrap["direct_steps"],
        state_steps=bootstrap["state_steps"],
        layout=bootstrap["layout"],
        updates=bootstrap["contract"].monitor_updates,
        flux_basis=bootstrap["flux_basis"],
        coefficients=bootstrap["coefficients"],
        label=label,
    )


def _write_progress(
    path: Path,
    *,
    identity: Mapping[str, Any],
    completed_logical_update: int,
    records: list[Mapping[str, Any]],
    milestones: list[Mapping[str, Any]],
) -> None:
    _atomic_json(
        path,
        {
            "schema_version": "target_oriented_coupled_gpp_artifact_aware_progress_v1",
            "identity": identity,
            "completed_logical_update": int(completed_logical_update),
            "selected_scales": [value["selection"]["selected_scale"] for value in records],
            "milestone_gates": [value["gate"] for value in milestones],
        },
    )


def run(args: Any) -> Mapping[str, Any]:
    started = time.perf_counter()
    output = args.output.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    contract = load_coupled_gpp_artifact_aware_persistent_training_contract(args.artifact_aware_persistent_contract)
    continuity_contract = load_coupled_gpp_artifact_aware_continuity_contract(args.continuity_contract)
    if continuity_contract.sha256 != contract.payload["identity"]["continuity_contract_sha256"]:
        raise ValueError("artifact-aware persistent continuity contract drift")
    _load_continuity_result(
        args.continuity_result,
        continuity_contract_sha256=continuity_contract.sha256,
    )
    parent_contract = load_coupled_gpp_persistent_scale_training_contract(args.persistent_contract)
    generalization_contract = load_coupled_gpp_direct_generalization_contract(args.generalization_contract)
    schedule = build_persistent_scale_training_schedule()
    parent_result = _load_persistent_result(
        args.persistent_result,
        audit_contract=generalization_contract,
        persistent_contract=parent_contract,
        schedule=schedule,
    )
    if (
        parent_contract.sha256 != contract.payload["identity"]["parent_persistent_contract_sha256"]
        or schedule.sha256 != contract.payload["identity"]["parent_persistent_schedule_sha256"]
        or sha256_file(args.persistent_result) != contract.payload["identity"]["parent_persistent_result_sha256"]
        or sha256_file(args.prepared_manifest) != contract.payload["identity"]["prepared_manifest_sha256"]
    ):
        raise ValueError("artifact-aware persistent parent identity drift")

    start_path = output_dir / "start.pkl"
    recovery_path = output_dir / "recovery.pkl"
    checkpoint_path = output_dir / f"checkpoint-update-{FINAL_LOGICAL_UPDATE:04d}.pkl"
    if recovery_path.exists() and not start_path.exists():
        raise ValueError("artifact-aware recovery exists without its immutable start")

    bootstrap, reconstructed_start, replay_record, initialization_mode = _prepare_bootstrap(
        args,
        start_exists=start_path.exists(),
        parent_contract=parent_contract,
        parent_result=parent_result,
        schedule=schedule,
    )

    bootstrap = _with_monitor_steps(bootstrap)
    bootstrap = dict(bootstrap, contract=contract)
    context = bootstrap["context"]
    jax = context["runtime"]["jax"]
    required = {value for panel in contract.train_panels for group in panel for value in group}
    required.update(contract.monitor_updates)
    missing = sorted(required - set(context["prepared_records"]))
    if missing:
        raise ValueError(f"artifact-aware persistent prepared coverage drift: {missing}")

    base_identity = {
        "git_head": _git_head(),
        "artifact_aware_persistent_contract_sha256": contract.sha256,
        "continuity_result_sha256": sha256_file(args.continuity_result),
        "parent_persistent_contract_sha256": parent_contract.sha256,
        "parent_persistent_result_sha256": sha256_file(args.persistent_result),
        "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
    }
    if not start_path.exists():
        host_start = jax.device_get(reconstructed_start)
        written = write_start_snapshot(
            start_path,
            identity=base_identity,
            trainable=host_start,
            parameter_count=TRAINABLE_PARAMETER_COUNT,
        )
        loaded_start = load_start_snapshot(
            start_path,
            identity=base_identity,
            parameter_count=TRAINABLE_PARAMETER_COUNT,
        )
        if written["sha256"] != sha256_file(start_path) or not parameter_trees_identical(
            host_start, loaded_start["trainable"]
        ):
            raise RuntimeError("artifact-aware persistent start roundtrip failed")
    else:
        loaded_start = load_start_snapshot(
            start_path,
            identity=base_identity,
            parameter_count=TRAINABLE_PARAMETER_COUNT,
        )
    start_sha256 = sha256_file(start_path)
    persistence_identity = dict(base_identity, start_snapshot_sha256=start_sha256)
    trainable = jax.device_put(loaded_start["trainable"])
    records: list[Mapping[str, Any]] = []
    milestones: list[Mapping[str, Any]] = []
    initial_monitor = None
    completed = START_COMPLETED_LOGICAL_UPDATE

    recovered = load_recovery(
        recovery_path,
        identity=persistence_identity,
        parameter_count=TRAINABLE_PARAMETER_COUNT,
    )
    if recovered is not None:
        trainable = jax.device_put(recovered["trainable"])
        records = list(recovered["update_records"])
        milestones = list(recovered["milestone_monitors"])
        initial_monitor = recovered["initial_monitor"]
        completed = int(recovered["completed_logical_update"])
        print(f"artifact-aware persistent resumed_after_update={completed}", flush=True)

    memory_before = _memory_report(bootstrap["device"])
    if initial_monitor is None:
        initial_monitor = _monitor(
            args,
            bootstrap,
            trainable=trainable,
            label="initial_exact_start_monitor",
        )
        write_recovery(
            recovery_path,
            identity=persistence_identity,
            trainable=jax.device_get(trainable),
            completed_logical_update=completed,
            parameter_count=TRAINABLE_PARAMETER_COUNT,
            update_records=records,
            initial_monitor=initial_monitor,
            milestone_monitors=milestones,
        )
        _write_progress(
            output_dir / "progress.json",
            identity=persistence_identity,
            completed_logical_update=completed,
            records=records,
            milestones=milestones,
        )

    stopped_reason = None
    if milestones and milestones[-1]["gate"]["stop_training"] is True:
        stopped_reason = (
            "final_monitor_gate"
            if int(milestones[-1]["logical_update"]) == FINAL_LOGICAL_UPDATE
            else "intermediate_monitor_gate"
        )

    flatten_gradient = _make_flattened_unit_gradient(jax)
    direct_virtual_step = _make_dtype_preserving_direct_virtual_step(jax)
    for logical_update, panel, source_registry in zip(
        contract.logical_updates,
        contract.train_panels,
        contract.source_registries,
        strict=True,
    ):
        if logical_update <= completed or stopped_reason is not None:
            continue
        measured = _train_only_update(
            args,
            context,
            contract=contract,
            initialized_parameters=bootstrap["initialized_parameters"],
            trainable=trainable,
            bootstrap=bootstrap,
            panel=panel,
            source_registry=source_registry,
            logical_update=logical_update,
            flatten_gradient=flatten_gradient,
            direct_virtual_step=direct_virtual_step,
        )
        record = measured["record"]
        if record["selection"]["accepted"] is not True:
            stopped_reason = "no_common_direction_or_scale"
            _atomic_json(
                output_dir / "rejected-proposal.json",
                {
                    "schema_version": "target_oriented_coupled_gpp_artifact_aware_rejection_v1",
                    "identity": persistence_identity,
                    "retained_through_logical_update": completed,
                    "proposal": record,
                },
            )
            break
        trainable = measured["updated_trainable"]
        records.append(record)
        completed = int(logical_update)
        print(
            f"artifact-aware persistent update={completed}/{FINAL_LOGICAL_UPDATE} "
            f"scale={record['selection']['selected_scale']} "
            f"seconds={record['timing_seconds']['total']:.3f}",
            flush=True,
        )
        if completed in contract.payload["monitor"]["milestone_logical_updates"]:
            current = _monitor(
                args,
                bootstrap,
                trainable=trainable,
                label=f"milestone_{completed}_monitor",
            )
            gate = artifact_aware_monitor_gate(
                contract,
                logical_update=completed,
                initial=initial_monitor,
                current=current,
            )
            milestones.append(dict(current, logical_update=completed, gate=gate))
            if gate["stop_training"]:
                stopped_reason = (
                    "final_monitor_gate" if completed == FINAL_LOGICAL_UPDATE else "intermediate_monitor_gate"
                )
            print(
                f"artifact-aware persistent milestone={completed} passed={gate['passed']}",
                flush=True,
            )
        write_recovery(
            recovery_path,
            identity=persistence_identity,
            trainable=jax.device_get(trainable),
            completed_logical_update=completed,
            parameter_count=TRAINABLE_PARAMETER_COUNT,
            update_records=records,
            initial_monitor=initial_monitor,
            milestone_monitors=milestones,
        )
        _write_progress(
            output_dir / "progress.json",
            identity=persistence_identity,
            completed_logical_update=completed,
            records=records,
            milestones=milestones,
        )

    final_gate_passed = bool(
        completed == FINAL_LOGICAL_UPDATE
        and milestones
        and int(milestones[-1]["logical_update"]) == FINAL_LOGICAL_UPDATE
        and milestones[-1]["gate"]["passed"] is True
    )
    checkpoint = None
    if final_gate_passed:
        host_trainable = jax.device_get(trainable)
        if checkpoint_path.exists():
            digest = sha256_file(checkpoint_path)
            loaded_checkpoint = load_checkpoint(
                checkpoint_path,
                expected_sha256=digest,
                identity=persistence_identity,
                parameter_count=TRAINABLE_PARAMETER_COUNT,
            )
            if not parameter_trees_identical(host_trainable, loaded_checkpoint["trainable"]):
                raise RuntimeError("artifact-aware persistent checkpoint parameter drift")
            checkpoint = {
                "path": str(checkpoint_path),
                "sha256": digest,
                "completed_logical_update": FINAL_LOGICAL_UPDATE,
                "parameter_count": TRAINABLE_PARAMETER_COUNT,
                "promotion_eligible": False,
            }
        else:
            checkpoint = write_checkpoint(
                checkpoint_path,
                identity=persistence_identity,
                trainable=host_trainable,
                completed_logical_update=completed,
                parameter_count=TRAINABLE_PARAMETER_COUNT,
                update_records=records,
                initial_monitor=initial_monitor,
                milestone_monitors=milestones,
            )

    final_parameters = replace_coupled_gpp_trainable(bootstrap["initialized_parameters"], trainable)
    fixed_report = _fixed_parameter_report(jax, bootstrap["initialized_parameters"], final_parameters)
    if not fixed_report["exactly_unchanged"] or int(fixed_report["element_count"]) != RETAINED_PARAMETER_COUNT:
        raise RuntimeError("artifact-aware persistent fixed-parameter boundary failed")

    if final_gate_passed:
        status = "passed"
        classification = "artifact_aware_persistent_checkpoint_selected"
        next_gate = contract.payload["next_gate"]["on_final_monitor_pass"]
    elif stopped_reason == "intermediate_monitor_gate":
        status = "failed_scientific_gate"
        classification = "artifact_aware_persistent_stopped_at_intermediate_monitor"
        next_gate = contract.payload["next_gate"]["on_intermediate_monitor_stop"]
    elif stopped_reason == "final_monitor_gate":
        status = "failed_scientific_gate"
        classification = "artifact_aware_persistent_failed_final_monitor"
        next_gate = contract.payload["next_gate"]["on_final_monitor_failure"]
    else:
        status = "failed_scientific_gate"
        classification = "artifact_aware_persistent_stopped_train_only"
        next_gate = contract.payload["next_gate"]["on_no_common_direction_or_scale"]

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "classification": classification,
        "identity": persistence_identity,
        "policy": {
            "initialization_mode": initialization_mode,
            "historical_updates_replayed_during_resume": 0,
            "persistent_update_1_replayed_during_initialization": initialization_mode == "one_time_reconstruction",
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "fresh_validation_values_evaluated": False,
            "sealed_test_read": False,
            "optimizer_state_used": False,
            "fixed_parameters_persisted": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "starting_state": {
            "completed_logical_update": START_COMPLETED_LOGICAL_UPDATE,
            "path": str(start_path),
            "sha256": start_sha256,
            "roundtrip_passed": True,
            "one_time_parent_update_1_replay": replay_record,
        },
        "training": {
            "requested_logical_updates": list(LOGICAL_UPDATES),
            "completed_logical_update": completed,
            "accepted_update_count": len(records),
            "stopped_reason": stopped_reason,
            "selected_scales": [value["selection"]["selected_scale"] for value in records],
            "updates": records,
        },
        "initial_monitor": initial_monitor,
        "milestone_monitors": milestones,
        "persistence": {
            "recovery_path": str(recovery_path),
            "recovery_sha256": sha256_file(recovery_path),
            "checkpoint": checkpoint,
        },
        "model": {
            "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
            "fixed_parameters": fixed_report,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in context["runtime"]["devices"]],
        "memory_before": memory_before,
        "memory_after": _memory_report(bootstrap["device"]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": next_gate,
    }
    _atomic_json(output, report)
    return report


def _parser() -> Any:
    parser = _continuity_parser()
    parser.add_argument("--artifact-aware-persistent-contract", type=Path, required=True)
    parser.add_argument("--continuity-result", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
