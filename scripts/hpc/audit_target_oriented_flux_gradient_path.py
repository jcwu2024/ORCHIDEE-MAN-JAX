"""Attribute extreme flux gradients without training or parameter updates."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_flux_basis import (
    load_target_oriented_flux_basis_amendment,
)
from research.daily_coarse_graining.daily_operator.target_oriented_flux_gradient_path import (
    FIELD_BASIS,
    classify_gradient_path,
    fine_parameter_family_trees,
    gram_cosine_summary,
    load_target_oriented_flux_gradient_path_contract,
)
from scripts.hpc.audit_target_oriented_component_resolved_geometry import (
    _chunked_tree_gram,
    _gradient_report,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import (
    CHECKPOINT_UPDATE,
    _audit_family_trees,
    _hard_host,
    _load_context,
    _tree_size,
)
from scripts.hpc.audit_target_oriented_flux_residual_tail import _window_identities
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import PARAMETER_COUNT
from scripts.hpc.run_target_oriented_full_step_preflight import _hard_failure_count, _slice
from scripts.hpc.target_oriented_training_prepared import load_prepared_target_artifact

SCHEMA_VERSION = "target_oriented_flux_gradient_path_attribution_report_v1"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _relative_difference(observed: float, expected: float) -> float:
    return abs(float(observed) - float(expected)) / max(abs(float(expected)), 1.0e-30)


def _family_reports(jax: Any, gradient: Mapping[str, Any]) -> Mapping[str, Any]:
    full = _gradient_report(jax, gradient)
    coarse = {
        name: _gradient_report(jax, tree)
        for name, tree in _audit_family_trees(gradient).items()
    }
    fine = {
        name: _gradient_report(jax, tree)
        for name, tree in fine_parameter_family_trees(gradient).items()
    }
    if sum(int(value["value_count"]) for value in coarse.values()) != int(full["value_count"]):
        raise ValueError("gradient-path coarse family coverage drift")
    if sum(int(value["value_count"]) for value in fine.values()) != int(full["value_count"]):
        raise ValueError("gradient-path fine family coverage drift")
    norm_squared = max(float(full["l2_norm"]) ** 2, 1.0e-300)
    fine_fraction = {
        name: float(value["l2_norm"]) ** 2 / norm_squared
        for name, value in fine.items()
    }
    return {
        "full": full,
        "coarse": coarse,
        "fine": fine,
        "fine_squared_norm_fraction": fine_fraction,
        "dominant_fine_parameter_family": max(fine_fraction, key=fine_fraction.get),
    }


def _head_reports(
    jax: Any,
    gradient: Mapping[str, Any],
    learned_schema: Any,
) -> Mapping[str, Any]:
    target = gradient["target_oriented"]
    result = {}
    for family in ("primary_heads", "state_heads"):
        trees = target[family]
        if len(trees) != len(learned_schema.fields):
            raise ValueError("gradient-path field-head registry drift")
        active = []
        for field, tree in zip(learned_schema.fields, trees, strict=True):
            report = _gradient_report(jax, tree)
            if int(report["value_count"]) > 0 and float(report["l2_norm"]) > 0.0:
                active.append({"field": field.name, **report})
        result[family] = active
    return result


def _aggregate_alignment(gram: np.ndarray) -> tuple[float, list[float]]:
    coefficients = np.asarray((0.125, 0.125, 0.125, 0.0, 0.0, 0.0), dtype=np.float64)
    aggregate_squared = float(coefficients @ gram @ coefficients)
    aggregate_norm = float(np.sqrt(max(aggregate_squared, 0.0)))
    field_norms = np.sqrt(np.maximum(np.diag(gram)[:3], 0.0))
    dots = gram[:3] @ coefficients
    alignments = np.divide(
        dots,
        field_norms * aggregate_norm,
        out=np.zeros(3, dtype=np.float64),
        where=(field_norms * aggregate_norm) > 0.0,
    )
    return aggregate_norm, alignments.tolist()


def _source_window(
    report: Mapping[str, Any],
    declared: Mapping[str, Any],
) -> Mapping[str, Any]:
    horizon = str(int(declared["horizon"]))
    matches = [
        record
        for record in report["windows"][horizon]
        if int(record["global_window_index"]) == int(declared["global_window_index"])
    ]
    if len(matches) != 1:
        raise ValueError("gradient-path source window is missing")
    record = matches[0]
    for name in ("update", "runtime_year", "window_index_within_group", "start_row"):
        if int(record[name]) != int(declared[name]):
            raise ValueError(f"gradient-path source window {name} drift")
    if str(record["landpoint_id"]) != str(declared["landpoint_id"]):
        raise ValueError("gradient-path source landpoint drift")
    return record


def _tree_weighted_sum(jax: Any, gradients: Sequence[Any], weight: float) -> Any:
    return jax.tree_util.tree_map(
        lambda *values: sum((np.asarray(value) for value in values), np.asarray(0.0)) * weight,
        *gradients,
    )


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    contract = load_target_oriented_flux_gradient_path_contract(args.gradient_path_contract)
    amendment = load_target_oriented_flux_basis_amendment(args.flux_basis_amendment)
    if sha256_file(args.residual_tail_report) != contract.payload["identity"]["residual_tail_report_sha256"]:
        raise ValueError("gradient-path residual-tail report identity drift")
    residual_report = json.loads(args.residual_tail_report.read_text(encoding="utf-8"))
    context = _load_context(
        args,
        science_flux_basis=amendment.flux_basis,
        verify_validation_artifacts=False,
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
        raise ValueError("gradient-path parameter coverage drift")
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
    hard_totals = {}
    all_gradient_reports = []
    score_checks = []
    audit_started = time.perf_counter()
    for declared in contract.windows:
        source = _source_window(residual_report, declared)
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
                raise ValueError("gradient-path prepared trace drift")
            loaded_artifacts[update] = artifact
        horizon = int(declared["horizon"])
        component = "seven_day" if horizon == 7 else "thirty_day"
        batch_value = getattr(artifact.batch, component)
        index = int(declared["window_index_within_group"])
        identities = _window_identities(entry, component)
        if identities[index] != {
            "landpoint_id": str(declared["landpoint_id"]),
            "year": int(declared["runtime_year"]),
            "start_row": int(declared["start_row"]),
        }:
            raise ValueError("gradient-path prepared window identity drift")
        batch = _slice(batch_value, index, index + 1)
        prefix = "seven" if horizon == 7 else "thirty"
        _statistics, hard = steps[f"{prefix}_forward"](parameters, batch)
        jax.block_until_ready((_statistics, hard))
        hard_host = _hard_host(jax.device_get(hard))
        for name, value in hard_host.items():
            hard_totals[name] = hard_totals.get(name, 0) + int(value)

        field_records = {}
        gradient_by_label = {}
        for field_index, field in enumerate(FIELD_BASIS):
            weights = np.zeros(len(FIELD_BASIS), dtype=np.float64)
            weights[field_index] = 1.0
            mode_records = {}
            for mode, step_name in (
                ("free", f"{prefix}_flux_field_gradient"),
                ("teacher_state", f"{prefix}_teacher_state_flux_field_gradient"),
            ):
                score, gradient = steps[step_name](parameters, batch, weights)
                jax.block_until_ready((score, gradient))
                host_gradient = jax.tree_util.tree_map(np.asarray, jax.device_get(gradient))
                reports = _family_reports(jax, host_gradient)
                reports["head_norms_by_field"] = _head_reports(
                    jax,
                    host_gradient,
                    runtime["learned_schema"],
                )
                reports["score"] = float(np.asarray(score))
                mode_records[mode] = reports
                gradient_by_label[f"{mode}:{field}"] = host_gradient
                all_gradient_reports.append(reports["full"])
                del gradient
                print(
                    f"flux_gradient_path window={declared['name']} mode={mode} field={field} complete",
                    flush=True,
                )
            expected_score = float(source["component_fields"][field]["score"])
            absolute_difference = abs(float(mode_records["free"]["score"]) - expected_score)
            relative_difference = _relative_difference(mode_records["free"]["score"], expected_score)
            score_checks.append(
                {
                    "absolute_difference": absolute_difference,
                    "relative_difference": relative_difference,
                }
            )
            field_records[field] = {
                "source_free_score": expected_score,
                "free_score": mode_records["free"]["score"],
                "free_score_absolute_difference": absolute_difference,
                "free_score_relative_difference": relative_difference,
                "free_gradient": mode_records["free"]["full"],
                "teacher_state_score": mode_records["teacher_state"]["score"],
                "teacher_state_gradient": mode_records["teacher_state"]["full"],
                "free_parameter_families": {
                    "coarse": mode_records["free"]["coarse"],
                    "fine": mode_records["free"]["fine"],
                    "fine_squared_norm_fraction": mode_records["free"]["fine_squared_norm_fraction"],
                },
                "teacher_state_parameter_families": {
                    "coarse": mode_records["teacher_state"]["coarse"],
                    "fine": mode_records["teacher_state"]["fine"],
                    "fine_squared_norm_fraction": mode_records["teacher_state"][
                        "fine_squared_norm_fraction"
                    ],
                },
                "dominant_fine_parameter_family": mode_records["free"][
                    "dominant_fine_parameter_family"
                ],
                "dominant_fine_parameter_family_squared_norm_fraction": max(
                    mode_records["free"]["fine_squared_norm_fraction"].values()
                ),
                "free_head_norms_by_field": mode_records["free"]["head_norms_by_field"],
                "teacher_state_head_norms_by_field": mode_records["teacher_state"][
                    "head_norms_by_field"
                ],
            }

        labels = [
            *(f"free:{field}" for field in FIELD_BASIS),
            *(f"teacher_state:{field}" for field in FIELD_BASIS),
        ]
        gradients = [gradient_by_label[label] for label in labels]
        gram = _chunked_tree_gram(jax, gradients, chunk_gram)
        aggregate_norm, alignments = _aggregate_alignment(gram)
        free_teacher_cosines = []
        for field_index, field in enumerate(FIELD_BASIS):
            free_norm = np.sqrt(max(float(gram[field_index, field_index]), 0.0))
            teacher_index = field_index + len(FIELD_BASIS)
            teacher_norm = np.sqrt(max(float(gram[teacher_index, teacher_index]), 0.0))
            cosine = float(gram[field_index, teacher_index]) / max(free_norm * teacher_norm, 1.0e-300)
            field_records[field]["free_alignment_to_aggregate"] = alignments[field_index]
            field_records[field]["free_teacher_state_cosine"] = cosine
            free_teacher_cosines.append(cosine)

        aggregate_gradient = _tree_weighted_sum(jax, gradients[:3], 0.125)
        aggregate_report = _gradient_report(jax, aggregate_gradient)
        aggregate_source_score = float(source["standalone_flux_score"])
        aggregate_reconstructed_score = 0.125 * sum(
            float(field_records[name]["free_score"]) for name in FIELD_BASIS
        )
        records[str(declared["name"])] = {
            "declared": dict(declared),
            "source": {
                "standalone_flux_score": aggregate_source_score,
                "gradient_l2_norm": float(source["gradient"]["l2_norm"]),
            },
            "hard_counts": hard_host,
            "fields": field_records,
            "gradient_gram": gram_cosine_summary(gram, labels),
            "free_teacher_state_same_field_cosines": dict(
                zip(FIELD_BASIS, free_teacher_cosines, strict=True)
            ),
            "reconstructed_free_aggregate": {
                **aggregate_report,
                "gram_l2_norm": aggregate_norm,
                "source_gradient_l2_norm_relative_difference": _relative_difference(
                    aggregate_report["l2_norm"],
                    float(source["gradient"]["l2_norm"]),
                ),
                "score": aggregate_reconstructed_score,
                "source_score_relative_difference": _relative_difference(
                    aggregate_reconstructed_score,
                    aggregate_source_score,
                ),
            },
        }
        del gradients, gradient_by_label, aggregate_gradient

    classification = classify_gradient_path(records, contract)
    score_rtol = float(contract.payload["measurement"]["score_reproduction_rtol"])
    score_atol = float(contract.payload["measurement"]["score_reproduction_atol"])
    score_pass = all(
        float(check["relative_difference"]) <= score_rtol
        or float(check["absolute_difference"]) <= score_atol
        for check in score_checks
    )
    gradients_pass = len(all_gradient_reports) == 24 and all(
        int(value["value_count"]) == PARAMETER_COUNT
        and int(value["finite_value_count"]) == PARAMETER_COUNT
        and float(value["l2_norm"]) > 0.0
        for value in all_gradient_reports
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
            "gradient_path_contract_sha256": contract.sha256,
            "residual_tail_report_sha256": sha256_file(args.residual_tail_report),
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
        },
        "parameter_count": PARAMETER_COUNT,
        "gradient_count": len(all_gradient_reports),
        "hard_counts": hard_totals,
        "hard_failure_count": hard_failure_count,
        "score_reproduction": {
            "maximum_field_absolute_difference": max(
                float(check["absolute_difference"]) for check in score_checks
            ),
            "maximum_field_relative_difference": max(
                float(check["relative_difference"]) for check in score_checks
            ),
            "passed": score_pass,
        },
        "records": records,
        "classification": classification,
        "engineering": {
            "all_gradients_finite_complete_nonzero": gradients_pass,
            "all_free_rollout_hard_gates_zero": hard_failure_count == 0,
            "score_reproduction_passed": score_pass,
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
    parser.add_argument("--gradient-path-contract", type=Path, required=True)
    parser.add_argument("--residual-tail-report", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-updates", type=int, default=CHECKPOINT_UPDATE)
    parser.add_argument("--validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--stage-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "action": report["action"],
                "gradient_count": report["gradient_count"],
                "hard_failure_count": report["hard_failure_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
