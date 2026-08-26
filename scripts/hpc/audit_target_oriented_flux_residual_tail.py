"""Audit window-level flux loss and gradient tails without training."""

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
from research.daily_coarse_graining.daily_operator.target_oriented_flux_residual_tail import (
    AUDIT_UPDATES,
    DERIVED_FLUX_KEYS,
    HORIZONS,
    OPTIMIZED_FLUX_KEYS,
    classify_tail_mechanism,
    contribution_summary,
    gram_direction_summary,
    influence_candidates,
    load_target_oriented_flux_residual_tail_contract,
)
from scripts.hpc.audit_target_oriented_component_resolved_geometry import (
    _chunked_tree_gram,
    _gradient_report,
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
from scripts.hpc.run_target_oriented_full_step_preflight import (
    _hard_failure_count,
    _slice,
)
from scripts.hpc.target_oriented_training_prepared import load_prepared_target_artifact

SCHEMA_VERSION = "target_oriented_flux_residual_tail_audit_report_v1"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _error_record(statistics: Any) -> Mapping[str, float]:
    count = float(np.asarray(statistics.count))
    squared = float(np.asarray(statistics.sum_squared))
    signed = float(np.asarray(statistics.sum_signed))
    mse = squared / count if count > 0.0 else 0.0
    bias = signed / count if count > 0.0 else 0.0
    return {
        "count": count,
        "normalized_mse": mse,
        "normalized_signed_bias": bias,
        "score": 0.5 * (mse + bias * bias),
    }


def _science_record(statistics: Any, flux_basis: Sequence[str]) -> Mapping[str, Any]:
    fields = {name: _error_record(statistics.fluxes[name]) for name in flux_basis}
    active = [value for value in fields.values() if value["count"] > 0.0]
    if not active:
        score = 0.0
    else:
        score = 0.25 * (
            float(np.mean([value["normalized_mse"] for value in active]))
            + float(np.mean([value["normalized_signed_bias"] ** 2 for value in active]))
        )
    return {"component_fields": fields, "forward_flux_score": score}


def _window_identities(entry: Mapping[str, Any], component: str) -> list[Mapping[str, Any]]:
    result = []
    for record in entry["components"][component]:
        for start_row in record["start_rows"]:
            result.append(
                {
                    "landpoint_id": str(record["landpoint_id"]),
                    "year": int(record["year"]),
                    "start_row": int(start_row),
                }
            )
    return result


def _pairwise_summary(gram: np.ndarray) -> Mapping[str, float]:
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    cosine = gram / np.outer(norms, norms)
    off_diagonal = cosine[np.triu_indices(cosine.shape[0], k=1)]
    return {
        "minimum": float(np.min(off_diagonal)),
        "median": float(np.median(off_diagonal)),
        "maximum": float(np.max(off_diagonal)),
        "negative_fraction": float(np.mean(off_diagonal < 0.0)),
    }


def _candidate_geometry(
    window_records: Sequence[Mapping[str, Any]],
    group_records: Sequence[Mapping[str, Any]],
    contract: Any,
) -> Mapping[str, Any]:
    scores = np.asarray(
        [float(record["standalone_flux_score"]) for record in window_records],
        dtype=np.float64,
    )
    candidates = influence_candidates(scores, contract)
    result = {}
    for name, candidate in candidates.items():
        global_weights = np.asarray(candidate["weights"], dtype=np.float64)
        group_summaries = []
        all_cosines = []
        for group in group_records:
            indices = np.asarray(group["global_window_indices"], dtype=np.int64)
            summary = gram_direction_summary(group["gram"], global_weights[indices])
            all_cosines.extend(float(value) for value in summary["constraint_cosines"])
            group_summaries.append(
                {
                    "update": int(group["update"]),
                    "runtime_year": int(group["runtime_year"]),
                    **summary,
                }
            )
        all_cosines_array = np.asarray(all_cosines, dtype=np.float64)
        result[name] = {
            "family": str(candidate["family"]),
            "tau": None if candidate["tau"] is None else float(candidate["tau"]),
            "global_weight_concentration": contribution_summary(
                global_weights,
                top_counts=contract.top_counts,
            ),
            "minimum_window_cosine": float(np.min(all_cosines_array)),
            "median_window_cosine": float(np.median(all_cosines_array)),
            "negative_window_cosine_fraction": float(np.mean(all_cosines_array < 0.0)),
            "groups": group_summaries,
        }
    return result


def _classification_action(classification: Mapping[str, Any], contract: Any) -> str:
    decisions = contract.payload["decision"]
    values = tuple(classification.values())
    if any(value["broad_groups"] for value in values):
        return str(decisions["broad_group_regime"])
    if any(value["localized_loss_tail"] and value["gradient_norm_tail"] for value in values):
        return str(decisions["localized_loss_and_gradient_tail"])
    if any(value["gradient_norm_tail"] and not value["localized_loss_tail"] for value in values):
        return str(decisions["gradient_tail_without_loss_tail"])
    if any(float(value["equal_window_minimum_cosine"]) < 0.0 for value in values):
        return str(decisions["diffuse_loss_with_conflicting_window_gradients"])
    return "no_frozen_tail_mechanism_identified"


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    contract = load_target_oriented_flux_residual_tail_contract(args.residual_tail_contract)
    amendment = load_target_oriented_flux_basis_amendment(args.flux_basis_amendment)
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
    target_contract = context["target_contract"]
    if _tree_size(jax, parameters) != PARAMETER_COUNT:
        raise ValueError("residual-tail parameter coverage drift")
    memory_before = _memory_report(runtime["devices"][0])

    chunk_gram = jax.jit(
        lambda matrix: jnp.matmul(
            matrix,
            matrix.T,
            precision=jax.lax.Precision.HIGHEST,
        )
    )
    windows: dict[str, list[Mapping[str, Any]]] = {str(horizon): [] for horizon in HORIZONS}
    groups: dict[str, list[Mapping[str, Any]]] = {str(horizon): [] for horizon in HORIZONS}
    hard_totals: dict[str, int] = {}
    audit_started = time.perf_counter()
    for update in AUDIT_UPDATES:
        entry = schedule["entries"][update - 1]
        artifact = load_prepared_target_artifact(
            args.prepared_manifest,
            prepared_records[update],
            training_contract_sha256=training_contract.sha256,
            schedule_entry=entry,
        )
        if artifact.trace_signature != context["trace_signature"]:
            raise ValueError("residual-tail prepared trace drift")
        for horizon, component, batch_value, forward_name, acceptance_name, gradient_name in (
            (
                7,
                "seven_day",
                artifact.batch.seven_day,
                "seven_forward",
                "seven_acceptance_forward",
                "seven_flux_window_gradient",
            ),
            (
                30,
                "thirty_day",
                artifact.batch.thirty_day,
                "thirty_forward",
                "thirty_acceptance_forward",
                "thirty_flux_window_gradient",
            ),
        ):
            key = str(horizon)
            identities = _window_identities(entry, component)
            batches = [_slice(batch_value, index, index + 1) for index in range(len(identities))]
            gradient_trees = []
            global_indices = []
            for index, (identity, batch) in enumerate(zip(identities, batches, strict=True)):
                science, science_hard = steps[forward_name](parameters, batch)
                acceptance, acceptance_hard = steps[acceptance_name](parameters, batch)
                loss, gradient = steps[gradient_name](parameters, batch)
                jax.block_until_ready((science, science_hard, acceptance, acceptance_hard, loss, gradient))
                science_hard_host = _hard_host(jax.device_get(science_hard))
                acceptance_hard_host = _hard_host(jax.device_get(acceptance_hard))
                if science_hard_host != acceptance_hard_host:
                    raise ValueError("residual-tail duplicate forward hard counts differ")
                if _hard_failure_count(science_hard_host) != 0:
                    raise RuntimeError("residual-tail window has a physical hard failure")
                for name, value in science_hard_host.items():
                    hard_totals[name] = hard_totals.get(name, 0) + int(value)

                science_host = jax.device_get(science)
                acceptance_host = jax.device_get(acceptance)
                science_record = _science_record(science_host, amendment.flux_basis)
                exact_loss = float(np.asarray(loss))
                forward_loss = float(science_record["forward_flux_score"])
                if not np.isfinite(exact_loss) or not np.isfinite(forward_loss):
                    raise FloatingPointError("standalone residual-tail loss is nonfinite")
                score_difference = abs(exact_loss - forward_loss)
                score_relative_difference = score_difference / max(abs(forward_loss), 1.0e-15)
                gradient_report = _gradient_report(jax, gradient)
                if (
                    int(gradient_report["value_count"]) != PARAMETER_COUNT
                    or int(gradient_report["finite_value_count"]) != PARAMETER_COUNT
                    or float(gradient_report["l2_norm"]) <= 0.0
                ):
                    raise FloatingPointError("residual-tail window gradient is invalid")
                gradient_trees.append(jax.device_get(gradient))
                del gradient

                acceptance_fields = {
                    name: _error_record(acceptance_host.fluxes[name])
                    for name in (*OPTIMIZED_FLUX_KEYS, *DERIVED_FLUX_KEYS)
                }
                global_index = len(windows[key])
                global_indices.append(global_index)
                windows[key].append(
                    {
                        "global_window_index": global_index,
                        "update": update,
                        "runtime_year": int(entry["runtime_year"]),
                        "window_index_within_group": index,
                        **identity,
                        **science_record,
                        "standalone_flux_score": exact_loss,
                        "score_reproduction_absolute_difference": score_difference,
                        "score_reproduction_relative_difference": score_relative_difference,
                        "acceptance_fields": acceptance_fields,
                        "gradient": gradient_report,
                        "sensitivity_proxy": float(gradient_report["l2_norm"]) / np.sqrt(max(exact_loss, 1.0e-30)),
                        "hard_counts": science_hard_host,
                    }
                )
                print(
                    f"flux_residual_tail update={update} horizon={horizon} window={index + 1}/{len(batches)} complete",
                    flush=True,
                )

            gram = _chunked_tree_gram(jax, gradient_trees, chunk_gram)
            equal = gram_direction_summary(gram, np.ones(len(gradient_trees), dtype=np.float64))
            groups[key].append(
                {
                    "update": update,
                    "runtime_year": int(entry["runtime_year"]),
                    "global_window_indices": global_indices,
                    "gram": gram.tolist(),
                    "pairwise_cosines": _pairwise_summary(gram),
                    "equal_window": equal,
                }
            )
            del gradient_trees
            print(
                f"flux_residual_tail update={update} horizon={horizon} gram complete",
                flush=True,
            )

    expected_counts = {"7": 64, "30": 32}
    if {key: len(value) for key, value in windows.items()} != expected_counts:
        raise ValueError("residual-tail window coverage drift")

    classification = {}
    analysis = {}
    for horizon in HORIZONS:
        key = str(horizon)
        losses = [float(record["standalone_flux_score"]) for record in windows[key]]
        norms = [float(record["gradient"]["l2_norm"]) for record in windows[key]]
        labels = [int(record["update"]) for record in windows[key]]
        current = classify_tail_mechanism(losses, norms, labels, contract)
        candidates = _candidate_geometry(windows[key], groups[key], contract)
        current["equal_window_minimum_cosine"] = float(candidates["equal_window"]["minimum_window_cosine"])
        classification[key] = current
        field_concentration = {
            name: contribution_summary(
                [float(record["acceptance_fields"][name]["score"]) for record in windows[key]],
                top_counts=contract.top_counts,
            )
            for name in (*OPTIMIZED_FLUX_KEYS, *DERIVED_FLUX_KEYS)
        }
        analysis[key] = {
            "score_reproduction": {
                "maximum_absolute_difference": max(
                    float(record["score_reproduction_absolute_difference"]) for record in windows[key]
                ),
                "maximum_relative_difference": max(
                    float(record["score_reproduction_relative_difference"]) for record in windows[key]
                ),
                "median_relative_difference": float(
                    np.median([float(record["score_reproduction_relative_difference"]) for record in windows[key]])
                ),
            },
            "window_loss_concentration": contribution_summary(
                losses,
                top_counts=contract.top_counts,
            ),
            "window_gradient_norm_concentration": contribution_summary(
                norms,
                top_counts=contract.top_counts,
            ),
            "field_score_concentration": field_concentration,
            "candidate_influence_geometry": candidates,
        }

    action = _classification_action(classification, contract)
    memory_after = _memory_report(runtime["devices"][0])
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "action": action,
        "identity": {
            "git_head": _git_head(),
            "residual_tail_contract_sha256": contract.sha256,
            "protected_virtual_step_report_sha256": contract.payload["identity"][
                "protected_virtual_step_report_sha256"
            ],
            "flux_basis_amendment_sha256": amendment.sha256,
            "stage_checkpoint": str(args.stage_checkpoint.resolve()),
            "stage_checkpoint_sha256": args.stage_checkpoint_sha256,
            "completed_updates": CHECKPOINT_UPDATE,
            "prepared_manifest": str(args.prepared_manifest.resolve()),
            "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
            "training_contract_sha256": training_contract.sha256,
            "target_contract_sha256": target_contract.sha256,
            "state_anchor_policy_sha256": context["state_anchor_policy"].sha256,
        },
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "validation_values_evaluated": False,
            "validation_artifact_values_read": False,
            "sealed_test_read": False,
            "optimizer_invoked": False,
            "optimizer_updates_retained": 0,
            "parameter_steps": 0,
            "proposed_parameters_retained": False,
            "checkpoint_written": False,
            "recovery_written": False,
            "audit_updates": list(AUDIT_UPDATES),
        },
        "parameter_count": PARAMETER_COUNT,
        "window_counts": {key: len(value) for key, value in windows.items()},
        "hard_counts": hard_totals,
        "hard_failure_count": _hard_failure_count(hard_totals),
        "windows": windows,
        "groups": groups,
        "analysis": analysis,
        "classification": classification,
        "timing_seconds": {
            "window_audit": time.perf_counter() - audit_started,
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
    parser.add_argument("--residual-tail-contract", type=Path, required=True)
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
                "window_counts": report["window_counts"],
                "hard_failure_count": report["hard_failure_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
