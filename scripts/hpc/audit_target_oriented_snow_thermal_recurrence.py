"""Test coupled snowheat/snowtemp recurrence without training."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_flux_basis import (
    load_target_oriented_flux_basis_amendment,
)
from research.daily_coarse_graining.daily_operator.target_oriented_snow_thermal_recurrence import (
    SNOW_THERMAL_STATE_KEYS,
    SNOW_THERMAL_TENDENCY_FIELDS,
    classify_snow_thermal_recurrence,
    load_target_oriented_snow_thermal_recurrence_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_snowheat_recurrence import (
    resolve_continuous_state_leaves,
)
from scripts.hpc.audit_target_oriented_flux_gradient_path import (
    _family_reports,
    _head_reports,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import (
    CHECKPOINT_UPDATE,
    _hard_host,
    _load_context,
    _tree_size,
)
from scripts.hpc.audit_target_oriented_snowheat_recurrence import (
    MAINTENANCE_WEIGHTS,
    _add_hard_totals,
    _state_head_norm,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import PARAMETER_COUNT
from scripts.hpc.run_target_oriented_full_step_preflight import _hard_failure_count, _slice
from scripts.hpc.target_oriented_training_prepared import load_prepared_target_artifact

SCHEMA_VERSION = "target_oriented_snow_thermal_recurrence_intervention_report_v1"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _verify_parent_record(
    report: Mapping[str, Any],
    declared: Mapping[str, Any],
) -> Mapping[str, Any]:
    record = report["records"].get(str(declared["name"]))
    if not isinstance(record, Mapping):
        raise ValueError("snow thermal recurrence parent window is missing")
    source = record["declared"]
    for name in ("horizon", "update", "runtime_year", "window_index_within_group", "start_row"):
        if int(source[name]) != int(declared[name]):
            raise ValueError(f"snow thermal recurrence parent window {name} drift")
    if str(source["landpoint_id"]) != str(declared["landpoint_id"]):
        raise ValueError("snow thermal recurrence parent landpoint drift")
    parent = record["snowheat_teacher_carry"]
    snowtemp_head = _state_head_norm(parent, SNOW_THERMAL_TENDENCY_FIELDS[1])
    expected = (
        float(declared["snowheat_parent_score"]),
        float(declared["snowheat_parent_gradient_l2_norm"]),
        float(declared["snowheat_parent_snowtemp_head_l2_norm"]),
    )
    observed = (
        float(parent["score"]),
        float(parent["gradient"]["l2_norm"]),
        snowtemp_head,
    )
    if observed != expected:
        raise ValueError("snow thermal recurrence parent measurement drift")
    return parent


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    contract = load_target_oriented_snow_thermal_recurrence_contract(args.snow_thermal_recurrence_contract)
    amendment = load_target_oriented_flux_basis_amendment(args.flux_basis_amendment)
    if sha256_file(args.snowheat_parent_report) != contract.payload["identity"]["snowheat_parent_report_sha256"]:
        raise ValueError("snow thermal recurrence parent report identity drift")
    parent_report = json.loads(args.snowheat_parent_report.read_text(encoding="utf-8"))
    context = _load_context(
        args,
        science_flux_basis=amendment.flux_basis,
        verify_validation_artifacts=False,
        diagnostic_teacher_carry_state_keys=SNOW_THERMAL_STATE_KEYS,
    )
    runtime = context["runtime"]
    jax = runtime["jax"]
    parameters = context["parameters"]
    steps = context["steps"]
    schedule = context["schedule"]
    prepared_records = context["prepared_records"]
    training_contract = context["training_contract"]
    if _tree_size(jax, parameters) != PARAMETER_COUNT:
        raise ValueError("snow thermal recurrence parameter coverage drift")
    state_leaves = resolve_continuous_state_leaves(
        runtime["contract"],
        SNOW_THERMAL_STATE_KEYS,
    )
    memory_before = _memory_report(runtime["devices"][0])

    records = {}
    hard_totals: dict[str, int] = {}
    gradient_reports = []
    audit_started = time.perf_counter()
    for declared in contract.windows:
        parent = _verify_parent_record(parent_report, declared)
        update = int(declared["update"])
        entry = schedule["entries"][update - 1]
        artifact = load_prepared_target_artifact(
            args.prepared_manifest,
            prepared_records[update],
            training_contract_sha256=training_contract.sha256,
            schedule_entry=entry,
        )
        if artifact.trace_signature != context["trace_signature"]:
            raise ValueError("snow thermal recurrence prepared trace drift")
        index = int(declared["window_index_within_group"])
        batch = _slice(artifact.batch.thirty_day, index, index + 1)
        (score, hard), gradient = steps["thirty_teacher_carry_flux_field_gradient"](
            parameters,
            batch,
            MAINTENANCE_WEIGHTS,
        )
        jax.block_until_ready((score, hard, gradient))
        _add_hard_totals(hard_totals, _hard_host(jax.device_get(hard)))
        host_gradient = jax.tree_util.tree_map(np.asarray, jax.device_get(gradient))
        reports = _family_reports(jax, host_gradient)
        reports["head_norms_by_field"] = _head_reports(
            jax,
            host_gradient,
            runtime["learned_schema"],
        )
        treatment_norm = float(reports["full"]["l2_norm"])
        snowtemp_head_norm = _state_head_norm(
            reports,
            SNOW_THERMAL_TENDENCY_FIELDS[1],
        )
        parent_norm = float(parent["gradient"]["l2_norm"])
        parent_snowtemp_norm = float(declared["snowheat_parent_snowtemp_head_l2_norm"])
        records[str(declared["name"])] = {
            "declared": dict(declared),
            "state_slices": [
                {
                    "key": leaf.key,
                    "start": int(leaf.start),
                    "stop": int(leaf.stop),
                    "shape": list(leaf.shape),
                    "axis_names": list(leaf.axis_names),
                    "source_ref": leaf.source_ref,
                }
                for leaf in state_leaves
            ],
            "snowheat_parent": {
                "score": float(parent["score"]),
                "gradient_l2_norm": parent_norm,
                "snowtemp_head_l2_norm": parent_snowtemp_norm,
            },
            "snowheat_plus_snowtemp_teacher_carry": {
                "score": float(np.asarray(score)),
                "gradient": reports["full"],
                "parameter_families": {
                    "coarse": reports["coarse"],
                    "fine": reports["fine"],
                    "fine_squared_norm_fraction": reports["fine_squared_norm_fraction"],
                },
                "head_norms_by_field": reports["head_norms_by_field"],
                "snowtemp_head_l2_norm": snowtemp_head_norm,
            },
            "parent_to_treatment_gradient_reduction_ratio": parent_norm / max(treatment_norm, 1.0e-30),
            "parent_to_treatment_snowtemp_head_reduction_ratio": parent_snowtemp_norm
            / max(snowtemp_head_norm, 1.0e-30),
        }
        gradient_reports.append(reports["full"])
        print(
            f"snow_thermal_recurrence window={declared['name']} complete",
            flush=True,
        )
        del gradient

    classification = classify_snow_thermal_recurrence(records, contract)
    gradients_pass = len(gradient_reports) == 2 and all(
        int(value["value_count"]) == PARAMETER_COUNT
        and int(value["finite_value_count"]) == PARAMETER_COUNT
        and float(value["l2_norm"]) > 0.0
        for value in gradient_reports
    )
    hard_failure_count = _hard_failure_count(hard_totals)
    engineering_pass = gradients_pass and hard_failure_count == 0
    memory_after = _memory_report(runtime["devices"][0])
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if engineering_pass else "failed",
        "action": classification["action"],
        "identity": {
            "git_head": _git_head(),
            "snow_thermal_recurrence_contract_sha256": contract.sha256,
            "snowheat_parent_report_sha256": sha256_file(args.snowheat_parent_report),
            "flux_basis_amendment_sha256": amendment.sha256,
            "stage_checkpoint_sha256": args.stage_checkpoint_sha256,
            "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
            "training_contract_sha256": training_contract.sha256,
            "target_contract_sha256": context["target_contract"].sha256,
        },
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "validation_values_evaluated": False,
            "sealed_test_read": False,
            "optimizer_invoked": False,
            "optimizer_updates_retained": 0,
            "parameter_steps": 0,
            "checkpoint_written": False,
            "recovery_written": False,
            "architecture_changed": False,
            "intervention_allowed_in_training": False,
            "intervention_allowed_in_final_inference": False,
        },
        "parameter_count": PARAMETER_COUNT,
        "gradient_count": len(gradient_reports),
        "hard_counts": hard_totals,
        "hard_failure_count": hard_failure_count,
        "state_slices": [
            {
                "key": leaf.key,
                "start": int(leaf.start),
                "stop": int(leaf.stop),
                "shape": list(leaf.shape),
                "axis_names": list(leaf.axis_names),
                "source_ref": leaf.source_ref,
            }
            for leaf in state_leaves
        ],
        "records": records,
        "classification": classification,
        "engineering": {
            "all_gradients_finite_complete_nonzero": gradients_pass,
            "all_hard_gates_zero": hard_failure_count == 0,
        },
        "timing_seconds": {
            "audit": time.perf_counter() - audit_started,
            "total": time.perf_counter() - started,
        },
        "memory_before": memory_before,
        "memory_after": memory_after,
    }
    _atomic_json(output, report)
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
    parser.add_argument("--snow-thermal-recurrence-contract", type=Path, required=True)
    parser.add_argument("--snowheat-parent-report", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-updates", type=int, default=CHECKPOINT_UPDATE)
    parser.add_argument("--validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--stage-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
