"""Re-evaluate the rejected coupled-GPP update at exact zero displacement."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_artifact_aware_persistence import (
    load_recovery,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_artifact_aware_persistent_training import (
    load_coupled_gpp_artifact_aware_persistent_training_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    SEVEN_SOURCE,
    TRAINABLE_PARAMETER_COUNT,
    replace_coupled_gpp_trainable,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_zero_displacement import (
    classify_zero_displacement,
)
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import _load_artifact
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_coupled_gpp_artifact_aware_persistent_training import (
    _initialization_only,
)
from scripts.hpc.run_target_oriented_coupled_gpp_artifact_aware_persistent_training import (
    _parser as _persistent_parser,
)
from scripts.hpc.run_target_oriented_coupled_gpp_direct_recalibration import _git_head
from scripts.hpc.run_target_oriented_full_step_preflight import _hard_failure_count
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import (
    _hard_host,
    _horizon_forward,
)

SCHEMA_VERSION = "target_oriented_coupled_gpp_zero_displacement_result_v1"


def _load_parent_evidence(
    report_path: Path,
    rejected_path: Path,
    *,
    contract_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rejected = json.loads(rejected_path.read_text(encoding="utf-8"))
    proposal = rejected.get("proposal", {})
    if (
        report.get("schema_version")
        != "target_oriented_coupled_gpp_artifact_aware_persistent_training_result_v1"
        or report.get("status") != "failed_scientific_gate"
        or report.get("classification") != "artifact_aware_persistent_stopped_train_only"
        or report.get("identity", {}).get("artifact_aware_persistent_contract_sha256") != contract_sha256
        or int(report.get("training", {}).get("completed_logical_update", -1)) != 2
        or report.get("training", {}).get("stopped_reason") != "no_common_direction_or_scale"
        or rejected.get("schema_version") != "target_oriented_coupled_gpp_artifact_aware_rejection_v1"
        or rejected.get("identity") != report.get("identity")
        or int(rejected.get("retained_through_logical_update", -1)) != 2
        or int(proposal.get("logical_update", -1)) != 3
        or proposal.get("selection", {}).get("accepted") is not False
        or not proposal.get("scale_evaluations")
    ):
        raise ValueError("zero-displacement parent evidence drift")
    return report, rejected


def _historical_records(
    rejected: Mapping[str, Any],
    *,
    expected_updates: tuple[int, ...],
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    proposal = rejected["proposal"]
    baselines = proposal["baseline"]["scheduled_artifact_records"]
    baseline_updates = tuple(int(value["artifact_update"]) for value in baselines)
    smallest = min(proposal["scale_evaluations"], key=lambda value: float(value["scale"]))
    smallest_records = smallest["scheduled_evaluation"]["evaluation"]["records"]
    smallest_updates = tuple(int(value["artifact_update"]) for value in smallest_records)
    if baseline_updates != expected_updates or smallest_updates != expected_updates:
        raise ValueError("zero-displacement scheduled artifact order drift")
    smallest_by_update = {int(value["artifact_update"]): value for value in smallest_records}
    records = [
        {
            "artifact_update": int(value["artifact_update"]),
            "gradient_baseline": float(value["source_losses_before"][SEVEN_SOURCE]),
            "smallest_nonzero_forward": float(
                smallest_by_update[int(value["artifact_update"])]["source_losses"][SEVEN_SOURCE]
            ),
        }
        for value in baselines
    ]
    return records, smallest


def run(args: Any) -> Mapping[str, Any]:
    started = time.perf_counter()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    contract = load_coupled_gpp_artifact_aware_persistent_training_contract(
        args.artifact_aware_persistent_contract
    )
    parent_report, rejected = _load_parent_evidence(
        args.parent_persistent_training_report.resolve(),
        args.parent_rejected_proposal.resolve(),
        contract_sha256=contract.sha256,
    )
    recovery_digest = sha256_file(args.parent_artifact_aware_recovery)
    if recovery_digest != parent_report.get("persistence", {}).get("recovery_sha256"):
        raise ValueError("zero-displacement recovery identity drift")
    recovered = load_recovery(
        args.parent_artifact_aware_recovery,
        identity=parent_report["identity"],
        parameter_count=TRAINABLE_PARAMETER_COUNT,
    )
    if recovered is None or int(recovered["completed_logical_update"]) != 2:
        raise ValueError("zero-displacement recovery progress drift")

    logical_index = contract.logical_updates.index(3)
    scheduled_updates = tuple(int(value) for value in contract.train_panels[logical_index][0])
    historical, smallest = _historical_records(rejected, expected_updates=scheduled_updates)
    historical_by_update = {int(value["artifact_update"]): value for value in historical}

    bootstrap = _initialization_only(args)
    context = bootstrap["context"]
    runtime = context["runtime"]
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    trainable = jax.device_put(recovered["trainable"])
    parameters = replace_coupled_gpp_trainable(bootstrap["initialized_parameters"], trainable)

    exact_records = []
    for ordinal, update in enumerate(scheduled_updates, start=1):
        artifact = _load_artifact(args, context, update)
        value, hard, seconds = _horizon_forward(
            jax,
            jnp,
            parameters,
            artifact.batch,
            bootstrap["steps"],
            horizon=7,
            flux_basis=bootstrap["flux_basis"],
        )
        hard_host = _hard_host(jax.device_get(hard))
        exact_score = float(np.asarray(value.total))
        if not np.isfinite(exact_score) or exact_score < 0.0:
            raise RuntimeError(f"zero-displacement artifact {update} score is invalid")
        exact_records.append(
            historical_by_update[update]
            | {
                "exact_zero_forward": exact_score,
                "runtime_year": int(context["schedule"]["entries"][update - 1]["runtime_year"]),
                "hard_counts": hard_host,
                "hard_failure_count": _hard_failure_count(hard_host),
                "timing_seconds": seconds,
            }
        )
        print(
            f"coupled_gpp zero_displacement artifact={ordinal}/{len(scheduled_updates)} "
            f"update={update} score={exact_score:.12g}",
            flush=True,
        )

    classification = classify_zero_displacement(exact_records)
    hard_failure_count = sum(int(value["hard_failure_count"]) for value in exact_records)
    if hard_failure_count != 0:
        raise RuntimeError("zero-displacement physical hard gate failed")
    exact_metadata = {
        int(value["artifact_update"]): {
            "runtime_year": int(value["runtime_year"]),
            "hard_counts": value["hard_counts"],
            "hard_failure_count": int(value["hard_failure_count"]),
            "timing_seconds": float(value["timing_seconds"]),
        }
        for value in exact_records
    }
    comparison_records = [
        value | exact_metadata[int(value["artifact_update"])] for value in classification["records"]
    ]

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "classification": classification["classification"],
        "identity": {
            "git_head": _git_head(),
            "artifact_aware_persistent_contract_sha256": contract.sha256,
            "parent_persistent_training_report_sha256": sha256_file(args.parent_persistent_training_report),
            "parent_rejected_proposal_sha256": sha256_file(args.parent_rejected_proposal),
            "parent_recovery_sha256": recovery_digest,
        },
        "policy": {
            "logical_update_recomputed": 0,
            "gradient_sources_recomputed": 0,
            "parameter_displacements_applied": 0,
            "trainable_parameters_written": False,
            "checkpoint_written": False,
            "recovery_written": False,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "fresh_validation_values_evaluated": False,
            "sealed_test_read": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "target": {
            "logical_update": 3,
            "source": SEVEN_SOURCE,
            "artifact_updates": list(scheduled_updates),
            "exact_parameter_state": "artifact_aware_recovery_after_logical_update_2",
            "historical_smallest_nonzero_scale": float(smallest["scale"]),
            "historical_smallest_parameter_delta_l2": float(smallest["parameter_delta_l2"]),
        },
        "comparison": dict(classification, records=comparison_records),
        "integrity": {
            "engineering_gate_passed": True,
            "physical_hard_failure_count": hard_failure_count,
            "recovery_completed_logical_update": int(recovered["completed_logical_update"]),
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "memory_after": _memory_report(bootstrap["device"]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": classification["next_gate"],
    }
    _atomic_json(output, report)
    return report


def _parser() -> Any:
    parser = _persistent_parser()
    parser.add_argument("--parent-persistent-training-report", type=Path, required=True)
    parser.add_argument("--parent-rejected-proposal", type=Path, required=True)
    parser.add_argument("--parent-artifact-aware-recovery", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
