"""Test the causal role of snowheat-only cross-day recurrence without training."""

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
from research.daily_coarse_graining.daily_operator.target_oriented_flux_gradient_path import (
    gram_cosine_summary,
)
from research.daily_coarse_graining.daily_operator.target_oriented_snowheat_recurrence import (
    SNOWHEAT_STATE_KEY,
    SNOWHEAT_TENDENCY_FIELD,
    classify_snowheat_recurrence,
    load_target_oriented_snowheat_recurrence_contract,
    resolve_continuous_state_leaf,
)
from scripts.hpc.audit_target_oriented_component_resolved_geometry import _chunked_tree_gram
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
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import PARAMETER_COUNT
from scripts.hpc.run_target_oriented_full_step_preflight import _hard_failure_count, _slice
from scripts.hpc.target_oriented_training_prepared import load_prepared_target_artifact

SCHEMA_VERSION = "target_oriented_snowheat_recurrence_intervention_report_v1"
MAINTENANCE_WEIGHTS = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _relative_difference(observed: float, expected: float) -> float:
    return abs(float(observed) - float(expected)) / max(abs(float(expected)), 1.0e-30)


def _verify_source_record(
    report: Mapping[str, Any],
    declared: Mapping[str, Any],
) -> Mapping[str, Any]:
    record = report["records"].get(str(declared["name"]))
    if not isinstance(record, Mapping):
        raise ValueError("snowheat recurrence source window is missing")
    source = record["declared"]
    for name in ("horizon", "update", "runtime_year", "window_index_within_group", "start_row"):
        if int(source[name]) != int(declared[name]):
            raise ValueError(f"snowheat recurrence source window {name} drift")
    if str(source["landpoint_id"]) != str(declared["landpoint_id"]):
        raise ValueError("snowheat recurrence source landpoint drift")
    return record


def _state_head_norm(report: Mapping[str, Any], field: str) -> float:
    matches = tuple(
        record
        for record in report["head_norms_by_field"]["state_heads"]
        if record["field"] == field
    )
    if len(matches) > 1:
        raise ValueError(f"state-head attribution is duplicated for {field}")
    return float(matches[0]["l2_norm"]) if matches else 0.0


def _snowheat_head_norm(report: Mapping[str, Any]) -> float:
    return _state_head_norm(report, SNOWHEAT_TENDENCY_FIELD)


def _add_hard_totals(target: dict[str, int], value: Mapping[str, int]) -> None:
    for name, count in value.items():
        target[name] = target.get(name, 0) + int(count)


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    contract = load_target_oriented_snowheat_recurrence_contract(args.snowheat_recurrence_contract)
    amendment = load_target_oriented_flux_basis_amendment(args.flux_basis_amendment)
    if sha256_file(args.gradient_path_report) != contract.payload["identity"]["gradient_path_report_sha256"]:
        raise ValueError("snowheat recurrence gradient-path report identity drift")
    path_report = json.loads(args.gradient_path_report.read_text(encoding="utf-8"))
    context = _load_context(
        args,
        science_flux_basis=amendment.flux_basis,
        verify_validation_artifacts=False,
        diagnostic_teacher_carry_state_key=SNOWHEAT_STATE_KEY,
    )
    runtime = context["runtime"]
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    parameters = context["parameters"]
    steps = context["steps"]
    schedule = context["schedule"]
    prepared_records = context["prepared_records"]
    training_contract = context["training_contract"]
    if _tree_size(jax, parameters) != PARAMETER_COUNT:
        raise ValueError("snowheat recurrence parameter coverage drift")
    snowheat_leaf = resolve_continuous_state_leaf(runtime["contract"], SNOWHEAT_STATE_KEY)
    memory_before = _memory_report(runtime["devices"][0])
    chunk_gram = jax.jit(
        lambda matrix: jnp.matmul(
            matrix,
            matrix.T,
            precision=jax.lax.Precision.HIGHEST,
        )
    )

    loaded_artifacts = {}
    records = {}
    hard_totals: dict[str, int] = {}
    gradient_reports = []
    score_checks = []
    audit_started = time.perf_counter()
    for declared in contract.windows:
        _verify_source_record(path_report, declared)
        update = int(declared["update"])
        entry = schedule["entries"][update - 1]
        artifact = loaded_artifacts.get(update)
        if artifact is None:
            artifact = load_prepared_target_artifact(
                args.prepared_manifest,
                prepared_records[update],
                training_contract_sha256=training_contract.sha256,
                schedule_entry=entry,
            )
            if artifact.trace_signature != context["trace_signature"]:
                raise ValueError("snowheat recurrence prepared trace drift")
            loaded_artifacts[update] = artifact
        horizon = int(declared["horizon"])
        prefix = "seven" if horizon == 7 else "thirty"
        component = "seven_day" if horizon == 7 else "thirty_day"
        index = int(declared["window_index_within_group"])
        batch = _slice(getattr(artifact.batch, component), index, index + 1)

        mode_records = {}
        gradients = []
        for mode, gradient_step in (
            (
                "free_reference",
                f"{prefix}_flux_field_gradient",
            ),
            (
                "snowheat_teacher_carry",
                f"{prefix}_teacher_carry_flux_field_gradient",
            ),
        ):
            if mode == "free_reference":
                _statistics, hard = steps[f"{prefix}_forward"](parameters, batch)
                score, gradient = steps[gradient_step](
                    parameters,
                    batch,
                    MAINTENANCE_WEIGHTS,
                )
                jax.block_until_ready((_statistics, hard, score, gradient))
            else:
                (score, hard), gradient = steps[gradient_step](
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
            reports["score"] = float(np.asarray(score))
            reports["snowheat_head_l2_norm"] = _snowheat_head_norm(reports)
            mode_records[mode] = reports
            gradients.append(host_gradient)
            gradient_reports.append(reports["full"])
            print(
                f"snowheat_recurrence window={declared['name']} mode={mode} complete",
                flush=True,
            )
            del gradient

        gram = _chunked_tree_gram(jax, gradients, chunk_gram)
        free = mode_records["free_reference"]
        intervention = mode_records["snowheat_teacher_carry"]
        free_norm = float(free["full"]["l2_norm"])
        intervention_norm = float(intervention["full"]["l2_norm"])
        free_head_norm = float(free["snowheat_head_l2_norm"])
        intervention_head_norm = float(intervention["snowheat_head_l2_norm"])
        expected_score = float(declared["free_maintenance_score"])
        expected_gradient = float(declared["free_maintenance_gradient_l2_norm"])
        score_check = {
            "absolute_difference": abs(float(free["score"]) - expected_score),
            "relative_difference": _relative_difference(free["score"], expected_score),
        }
        score_checks.append(score_check)
        records[str(declared["name"])] = {
            "declared": dict(declared),
            "state_slice": {
                "key": snowheat_leaf.key,
                "start": int(snowheat_leaf.start),
                "stop": int(snowheat_leaf.stop),
                "shape": list(snowheat_leaf.shape),
                "axis_names": list(snowheat_leaf.axis_names),
                "source_ref": snowheat_leaf.source_ref,
            },
            "free_reference": {
                "score": free["score"],
                "gradient": free["full"],
                "parameter_families": {
                    "coarse": free["coarse"],
                    "fine": free["fine"],
                    "fine_squared_norm_fraction": free["fine_squared_norm_fraction"],
                },
                "head_norms_by_field": free["head_norms_by_field"],
                "snowheat_head_l2_norm": free_head_norm,
                "score_reproduction": score_check,
                "gradient_l2_norm_relative_difference": _relative_difference(
                    free_norm,
                    expected_gradient,
                ),
            },
            "snowheat_teacher_carry": {
                "score": intervention["score"],
                "gradient": intervention["full"],
                "parameter_families": {
                    "coarse": intervention["coarse"],
                    "fine": intervention["fine"],
                    "fine_squared_norm_fraction": intervention["fine_squared_norm_fraction"],
                },
                "head_norms_by_field": intervention["head_norms_by_field"],
                "snowheat_head_l2_norm": intervention_head_norm,
            },
            "mode_gradient_gram": gram_cosine_summary(
                gram,
                ("free_reference", "snowheat_teacher_carry"),
            ),
            "gradient_reduction_ratio": free_norm / max(intervention_norm, 1.0e-30),
            "snowheat_head_reduction_ratio": free_head_norm / max(intervention_head_norm, 1.0e-30),
        }
        del gradients

    classification = classify_snowheat_recurrence(records, contract)
    score_rtol = float(contract.payload["measurement"]["score_reproduction_rtol"])
    score_atol = float(contract.payload["measurement"]["score_reproduction_atol"])
    score_pass = all(
        float(check["relative_difference"]) <= score_rtol or float(check["absolute_difference"]) <= score_atol
        for check in score_checks
    )
    gradients_pass = len(gradient_reports) == 8 and all(
        int(value["value_count"]) == PARAMETER_COUNT
        and int(value["finite_value_count"]) == PARAMETER_COUNT
        and float(value["l2_norm"]) > 0.0
        for value in gradient_reports
    )
    hard_failure_count = _hard_failure_count(hard_totals)
    engineering_pass = gradients_pass and hard_failure_count == 0 and score_pass
    memory_after = _memory_report(runtime["devices"][0])
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if engineering_pass else "failed",
        "action": classification["action"],
        "identity": {
            "git_head": _git_head(),
            "snowheat_recurrence_contract_sha256": contract.sha256,
            "gradient_path_report_sha256": sha256_file(args.gradient_path_report),
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
        "state_slice": {
            "key": snowheat_leaf.key,
            "start": int(snowheat_leaf.start),
            "stop": int(snowheat_leaf.stop),
            "shape": list(snowheat_leaf.shape),
            "axis_names": list(snowheat_leaf.axis_names),
            "source_ref": snowheat_leaf.source_ref,
        },
        "score_reproduction": {
            "maximum_absolute_difference": max(float(check["absolute_difference"]) for check in score_checks),
            "maximum_relative_difference": max(float(check["relative_difference"]) for check in score_checks),
            "passed": score_pass,
        },
        "records": records,
        "classification": classification,
        "engineering": {
            "all_gradients_finite_complete_nonzero": gradients_pass,
            "all_hard_gates_zero": hard_failure_count == 0,
            "free_score_reproduction_passed": score_pass,
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
    parser.add_argument("--snowheat-recurrence-contract", type=Path, required=True)
    parser.add_argument("--gradient-path-report", type=Path, required=True)
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
