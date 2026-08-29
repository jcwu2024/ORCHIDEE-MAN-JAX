"""Run the frozen 250-update direct-GPP specialist warm start."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_checkpoint import (
    load_structured_gpp_recovery,
    write_structured_gpp_checkpoint,
    write_structured_gpp_recovery,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_training import (
    CANDIDATE_PARAMETER_COUNT,
    SPECIALIST_PARAMETER_COUNT,
    compare_structured_gpp_direct_panels,
    load_structured_gpp_warm_start_contract,
    replace_specialist,
    specialist_element_count,
    specialist_tree,
    structured_gpp_direct_loss,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    add_error_sufficient_statistics,
    initialize_target_oriented_adamw,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import _load_context, _tree_size
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import _load_artifact, _replace_adapter
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_fast_owner_gpu_probe import PROBE_SEED
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import _head as _parent_head
from scripts.hpc.run_target_oriented_full_step_preflight import _hard_failure_count, _make_steps
from scripts.hpc.run_target_oriented_source_shaped_branch_training import _make_optimizer_step
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import (
    ADAPTER_PARAMETER_COUNT,
    PARENT_PARAMETER_COUNT,
    _direct_forward,
    _direct_gradient,
    _hard_host,
    _make_direct_steps,
    _one_day_hard,
    _retained_parameter_report,
)

SCHEMA_VERSION = "target_oriented_structured_gpp_warm_start_result_v1"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _direct_panel(
    args: argparse.Namespace,
    context: Mapping[str, Any],
    specialist: Any,
    direct_steps: Mapping[str, Any],
    updates: tuple[int, ...],
    *,
    label: str,
) -> Mapping[str, Any]:
    jax = context["runtime"]["jax"]
    records = []
    aggregate = None
    for update in updates:
        artifact = _load_artifact(args, context, update)
        statistics, elapsed = _direct_forward(jax, specialist, artifact.batch, direct_steps)
        count = int(np.asarray(statistics.count))
        loss = float(np.asarray(structured_gpp_direct_loss(statistics)))
        if count != 208 or not np.isfinite(loss) or loss < 0.0:
            raise RuntimeError(f"structured GPP {label} panel artifact {update} is invalid")
        aggregate = statistics if aggregate is None else add_error_sufficient_statistics(aggregate, statistics)
        records.append(
            {
                "artifact_update": update,
                "runtime_year": int(context["schedule"]["entries"][update - 1]["runtime_year"]),
                "direct_gpp_loss": loss,
                "evaluated_values": count,
                "elapsed_seconds": elapsed,
            }
        )
        print(f"structured_gpp warm_start {label}_panel={update} loss={loss:.9g}", flush=True)
    aggregate_loss = float(np.asarray(structured_gpp_direct_loss(aggregate)))
    evaluated = int(np.asarray(aggregate.count))
    if evaluated != 208 * len(updates) or not np.isfinite(aggregate_loss) or aggregate_loss < 0.0:
        raise RuntimeError(f"structured GPP {label} aggregate is invalid")
    return {
        "updates": list(updates),
        "aggregate_direct_gpp_loss": aggregate_loss,
        "evaluated_values": evaluated,
        "records": records,
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    started = time.perf_counter()

    warm = load_structured_gpp_warm_start_contract(args.warm_start_contract)
    identity_contract = warm.payload["identity"]
    if sha256_file(args.preflight_report) != identity_contract["training_preflight_report_sha256"]:
        raise ValueError("structured GPP warm-start preflight report identity drift")
    preflight = json.loads(args.preflight_report.read_text(encoding="utf-8"))
    if (
        preflight.get("status") != "passed"
        or preflight.get("next_gate") != "freeze_bounded_250_update_direct_gpp_warm_start_contract"
        or preflight["policy"]["training_updates_persisted"] != 0
    ):
        raise ValueError("structured GPP warm-start preflight is not accepted")
    if sha256_file(args.prepared_manifest) != identity_contract["prepared_manifest_sha256"]:
        raise ValueError("structured GPP warm-start prepared-manifest identity drift")

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
        raise ValueError("structured GPP warm-start flux-basis identity drift")
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
    if (
        context["training_contract"].sha256 != identity_contract["base_training_contract_sha256"]
        or target_contract.sha256 != identity_contract["target_contract_sha256"]
    ):
        raise ValueError("structured GPP warm-start parent contract identity drift")

    distribution_contract = load_distribution_aware_maintenance_contract(args.distribution_contract)
    structured_contract = load_structured_gpp_contract(args.structured_contract)
    if structured_contract.sha256 != identity_contract["candidate_contract_sha256"]:
        raise ValueError("structured GPP warm-start candidate contract identity drift")
    distribution_report = json.loads(args.distribution_training_report.read_text(encoding="utf-8"))
    if (
        distribution_report.get("status") != "passed"
        or distribution_report["checkpoint"]["sha256"] != args.distribution_checkpoint_sha256
        or args.distribution_checkpoint_sha256 != identity_contract["parent_distribution_checkpoint_sha256"]
    ):
        raise ValueError("structured GPP warm-start distribution checkpoint identity drift")
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
        raise ValueError("structured GPP warm-start parent parameter count drift")
    distribution_head = build_target_oriented_distribution_aware_maintenance_head(
        _parent_head(runtime, target_contract),
        context["template"],
        distribution_contract=distribution_contract,
    )
    head = build_target_oriented_structured_gpp_head(
        distribution_head,
        structured_contract=structured_contract,
    )
    parameters = initialize_structured_gpp_parameters(
        parent_parameters,
        jax.random.PRNGKey(PROBE_SEED + 20260829),
        schema=runtime["learned_schema"],
        target_contract=target_contract,
        structured_contract=structured_contract,
    )
    initial_specialist = specialist_tree(parameters)
    if (
        _tree_size(jax, parameters) != CANDIDATE_PARAMETER_COUNT
        or specialist_element_count(initial_specialist) != SPECIALIST_PARAMETER_COUNT
    ):
        raise ValueError("structured GPP warm-start candidate parameter count drift")

    steps = _make_steps(
        runtime,
        head,
        context["template"],
        context["scales"],
        target_contract,
        science_flux_basis=amendment.flux_basis,
    )
    direct_steps = _make_direct_steps(runtime, head, parameters)
    clip_threshold = float(warm.optimizer["global_gradient_norm_clip"]["threshold"])
    learning_rate = float(warm.optimizer["learning_rate"])
    optimizer_step = _make_optimizer_step(jax, jnp, warm, clip_threshold)
    maximum_updates = int(warm.optimizer["maximum_updates"])
    recovery_interval = int(warm.payload["bounded_stage"]["recovery_interval_updates"])
    memory_before = _memory_report(runtime["devices"][0])

    identity = {
        "git_head": _git_head(),
        "warm_start_contract_sha256": warm.sha256,
        "candidate_contract_sha256": structured_contract.sha256,
        "preflight_report_sha256": sha256_file(args.preflight_report),
        "distribution_checkpoint_sha256": args.distribution_checkpoint_sha256,
        "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
        "base_training_contract_sha256": context["training_contract"].sha256,
        "target_contract_sha256": target_contract.sha256,
        "flux_basis_amendment_sha256": amendment.sha256,
    }
    recovery_path = output_dir / "recovery.pkl"
    recovery = load_structured_gpp_recovery(
        recovery_path,
        identity=identity,
        parameter_count=SPECIALIST_PARAMETER_COUNT,
        clip_threshold=clip_threshold,
        maximum_updates=maximum_updates,
    )
    if recovery is None:
        initial_panel = _direct_panel(
            args,
            context,
            initial_specialist,
            direct_steps,
            warm.evaluation_updates,
            label="initial",
        )
        specialist = initial_specialist
        optimizer = initialize_target_oriented_adamw(specialist)
        completed = 0
        records: list[Mapping[str, Any]] = []
        latest_recovery = None
    else:
        initial_panel = recovery["initial_direct_panel"]
        specialist, optimizer = jax.device_put((recovery["specialist"], recovery["optimizer"]))
        completed = int(recovery["completed_updates"])
        records = list(recovery["update_records"])
        latest_recovery = {
            "path": str(recovery_path),
            "sha256": sha256_file(recovery_path),
            "completed_updates": completed,
            "promotion_eligible": False,
        }
        print(f"structured_gpp warm_start resumed update={completed}", flush=True)
    jax.block_until_ready((specialist, optimizer))

    progress_path = output_dir / "progress.json"
    for optimizer_update in range(completed + 1, maximum_updates + 1):
        artifact_update = warm.training_updates[optimizer_update - 1]
        artifact = _load_artifact(args, context, artifact_update)
        update_started = time.perf_counter()
        statistics, gradient, forward_seconds, gradient_seconds = _direct_gradient(
            jax,
            jnp,
            specialist,
            artifact.batch,
            direct_steps,
        )
        loss = float(np.asarray(structured_gpp_direct_loss(statistics)))
        count = int(np.asarray(statistics.count))
        if not np.isfinite(loss) or loss <= 0.0 or count != 208:
            raise RuntimeError(f"structured GPP warm-start update {optimizer_update} objective is invalid")
        proposal = optimizer_step(
            specialist,
            gradient,
            optimizer,
            jnp.asarray(learning_rate, dtype=jnp.float32),
        )
        jax.block_until_ready(proposal)
        specialist, optimizer = proposal[0], proposal[1]
        raw_norm = float(np.asarray(proposal[2]))
        clip_scale = float(np.asarray(proposal[3]))
        nonfinite = int(np.asarray(proposal[4]))
        step_norm = float(np.asarray(proposal[5]))
        if (
            nonfinite != 0
            or not all(np.isfinite(value) for value in (raw_norm, clip_scale, step_norm))
            or raw_norm <= 0.0
            or not 0.0 < clip_scale <= 1.0
            or step_norm <= 0.0
        ):
            raise RuntimeError(f"structured GPP warm-start update {optimizer_update} proposal is invalid")
        record = {
            "optimizer_update": optimizer_update,
            "artifact_update": artifact_update,
            "runtime_year": int(context["schedule"]["entries"][artifact_update - 1]["runtime_year"]),
            "direct_gpp_loss": loss,
            "evaluated_values": count,
            "learning_rate": learning_rate,
            "raw_gradient_norm": raw_norm,
            "clip_threshold": clip_threshold,
            "clip_scale": clip_scale,
            "clipped": clip_scale < 1.0,
            "parameter_step_norm": step_norm,
            "timing_seconds": {
                "direct_forward": forward_seconds,
                "direct_gradient": gradient_seconds,
                "total": time.perf_counter() - update_started,
            },
        }
        records.append(record)
        _atomic_json(
            progress_path,
            {
                "schema_version": "target_oriented_structured_gpp_warm_start_progress_v1",
                "identity": identity,
                "completed_updates": optimizer_update,
                "updates": records,
            },
        )
        print(
            f"structured_gpp warm_start update={optimizer_update}/{maximum_updates} "
            f"artifact={artifact_update} loss={loss:.9g} norm={raw_norm:.9g} "
            f"clip={clip_scale:.6g} seconds={record['timing_seconds']['total']:.3f}",
            flush=True,
        )
        if optimizer_update % recovery_interval == 0:
            host_specialist, host_optimizer = jax.device_get((specialist, optimizer))
            latest_recovery = write_structured_gpp_recovery(
                recovery_path,
                identity=identity,
                specialist=host_specialist,
                optimizer=host_optimizer,
                completed_updates=optimizer_update,
                parameter_count=SPECIALIST_PARAMETER_COUNT,
                clip_threshold=clip_threshold,
                update_records=records,
                initial_panel=initial_panel,
            )
            print(f"structured_gpp warm_start recovery={optimizer_update}", flush=True)
        del gradient

    final_parameters = replace_specialist(parameters, specialist)
    retained = _retained_parameter_report(jax, parameters, final_parameters)
    probe_update = int(warm.payload["data"]["physical_probe_update"])
    probe_artifact = _load_artifact(args, context, probe_update)
    final_hard = _hard_host(jax.device_get(_one_day_hard(jax, jnp, final_parameters, probe_artifact.batch, steps)))
    final_panel = _direct_panel(
        args,
        context,
        specialist,
        direct_steps,
        warm.evaluation_updates,
        label="final",
    )
    comparison = compare_structured_gpp_direct_panels(initial_panel, final_panel, warm)
    execution_passed = (
        len(records) == maximum_updates
        and int(np.asarray(optimizer.step)) == maximum_updates
        and retained["exactly_unchanged"]
        and _hard_failure_count(final_hard) == 0
    )
    if not execution_passed:
        raise RuntimeError("structured GPP warm start failed its execution or physical gate")

    host_specialist, host_optimizer = jax.device_get((specialist, optimizer))
    checkpoint = write_structured_gpp_checkpoint(
        output_dir / "stage-update-0250.pkl",
        identity=identity,
        specialist=host_specialist,
        optimizer=host_optimizer,
        completed_updates=maximum_updates,
        parameter_count=SPECIALIST_PARAMETER_COUNT,
        clip_threshold=clip_threshold,
        update_records=records,
        initial_panel=initial_panel,
        final_panel=final_panel,
    )
    norms = np.asarray([record["raw_gradient_norm"] for record in records], dtype=np.float64)
    timings = np.asarray([record["timing_seconds"]["total"] for record in records], dtype=np.float64)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "identity": identity,
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "validation_values_evaluated": False,
            "sealed_test_read": False,
            "seven_or_thirty_day_gradients_evaluated": False,
            "training_updates_executed": maximum_updates,
            "training_updates_persisted": maximum_updates,
            "retained_parent_parameters_trainable": False,
            "final_checkpoint_written": True,
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
            "artifact_updates": list(warm.training_updates),
            "optimizer_update_count": maximum_updates,
            "learning_rate": learning_rate,
            "clip_threshold": clip_threshold,
            "clipped_update_count": sum(bool(record["clipped"]) for record in records),
            "raw_gradient_norm": {
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
            "updates": records,
        },
        "initial_direct_panel": initial_panel,
        "final_direct_panel": final_panel,
        "endpoint_comparison": comparison,
        "final_one_day_hard_counts": final_hard,
        "checkpoint": checkpoint,
        "latest_recovery": latest_recovery,
        "memory_before": memory_before,
        "memory_after": _memory_report(runtime["devices"][0]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": (
            warm.payload["next_gate"]["on_capability_pass"]
            if comparison["capability_gate_passed"]
            else warm.payload["next_gate"]["on_capability_failure"]
        ),
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
    parser.add_argument("--warm-start-contract", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
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
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
