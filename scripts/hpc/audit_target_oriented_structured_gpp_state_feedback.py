"""Attribute structured-GPP cross-day feedback to train-only source components."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_broad_checkpoint import (
    load_broad_checkpoint,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_state_feedback_attribution import (
    classify_component_feedback,
    load_structured_gpp_state_feedback_attribution_contract,
    resolve_component_state_masks,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_training import (
    CANDIDATE_PARAMETER_COUNT,
    SPECIALIST_PARAMETER_COUNT,
    replace_specialist,
    specialist_element_count,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    add_target_oriented_acceptance_statistics,
    derive_target_oriented_acceptance_scales,
    finalize_target_oriented_acceptance_statistics,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import (
    AUDIT_FLUX_KEYS,
    _hard_host,
    _load_context,
    _tree_size,
)
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import _replace_adapter
from scripts.hpc.run_successor_fast_owner_gpu_probe import PROBE_SEED
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import _head as _parent_head
from scripts.hpc.run_target_oriented_full_step_preflight import _hard_failure_count, _make_steps, _slice
from scripts.hpc.run_target_oriented_structured_gpp_fresh_screen import _load_broad_training_report
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import (
    ADAPTER_PARAMETER_COUNT,
    PARENT_PARAMETER_COUNT,
)
from scripts.hpc.target_oriented_training_prepared import load_prepared_target_artifact

SCHEMA_VERSION = "target_oriented_structured_gpp_state_feedback_attribution_result_v1"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _groups(context: Mapping[str, Any], updates: Sequence[int]) -> list[Mapping[str, Any]]:
    groups = []
    for update in updates:
        artifact = load_prepared_target_artifact(
            context["prepared_manifest"],
            context["prepared_records"][update],
            training_contract_sha256=context["training_contract"].sha256,
            schedule_entry=context["schedule"]["entries"][update - 1],
        )
        if artifact.trace_signature != context["trace_signature"]:
            raise ValueError("structured-GPP state-feedback prepared trace drift")
        groups.append(
            {
                "update": int(update),
                "runtime_year": int(context["schedule"]["entries"][update - 1]["runtime_year"]),
                "seven_batches": [_slice(artifact.batch.seven_day, index, index + 1) for index in range(8)],
                "thirty_batches": [_slice(artifact.batch.thirty_day, index, index + 1) for index in range(4)],
            }
        )
    return groups


def _evaluate_group(
    jax: Any,
    parameters: Any,
    steps: Mapping[str, Any],
    group: Mapping[str, Any],
    target_contract: Any,
    *,
    mode: str,
    state_mask: Any | None = None,
) -> Mapping[str, Any]:
    acceptance = {}
    hard = {}
    for horizon, batches in (("7", group["seven_batches"]), ("30", group["thirty_batches"])):
        prefix = "seven" if horizon == "7" else "thirty"
        if mode == "free":
            forward = steps[f"{prefix}_acceptance_forward"]
            values = [forward(parameters, batch) for batch in batches]
        elif mode == "full_teacher":
            forward = steps[f"{prefix}_teacher_state_acceptance_forward"]
            values = [forward(parameters, batch) for batch in batches]
        elif mode == "component":
            if state_mask is None:
                raise ValueError("component state mask is required")
            forward = steps[f"{prefix}_masked_teacher_carry_acceptance_forward"]
            values = [forward(parameters, batch, state_mask) for batch in batches]
        else:
            raise ValueError(f"unknown state-feedback attribution mode: {mode}")
        jax.block_until_ready(values)
        total = values[0][0]
        hard_counts = {name: 0 for name in values[0][1]._fields}
        for index, value in enumerate(values):
            if index:
                total = add_target_oriented_acceptance_statistics(total, value[0])
            for name, count in _hard_host(jax.device_get(value[1])).items():
                hard_counts[name] += int(count)
        acceptance[horizon] = finalize_target_oriented_acceptance_statistics(
            total,
            horizon=int(horizon),
            target_contract=target_contract,
        )
        hard[horizon] = hard_counts
    return {
        "update": int(group["update"]),
        "runtime_year": int(group["runtime_year"]),
        "acceptance": acceptance,
        "hard_counts": hard,
        "hard_failure_count": sum(_hard_failure_count(value) for value in hard.values()),
    }


def _evaluate_mode(
    jax: Any,
    parameters: Any,
    steps: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
    target_contract: Any,
    *,
    mode: str,
    state_mask: Any | None = None,
) -> list[Mapping[str, Any]]:
    records = []
    for group in groups:
        started = time.perf_counter()
        record = _evaluate_group(
            jax,
            parameters,
            steps,
            group,
            target_contract,
            mode=mode,
            state_mask=state_mask,
        )
        records.append(record | {"elapsed_seconds": time.perf_counter() - started})
        print(f"state-feedback mode={mode} update={group['update']} complete", flush=True)
    return records


def _all_acceptance_finite(modes: Sequence[Sequence[Mapping[str, Any]]]) -> bool:
    for groups in modes:
        for group in groups:
            for horizon in ("7", "30"):
                for record in group["acceptance"][horizon]["keys"].values():
                    values = (
                        float(record["normalized_rmse"]),
                        float(record["normalized_signed_bias"]),
                        float(record["evaluated_values"]),
                    )
                    if not all(np.isfinite(value) for value in values) or values[2] <= 0.0:
                        return False
    return True


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    started = time.perf_counter()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    attribution = load_structured_gpp_state_feedback_attribution_contract(args.attribution_contract)
    identity = attribution.payload["identity"]
    if sha256_file(args.prepared_manifest) != identity["prepared_manifest_sha256"]:
        raise ValueError("structured-GPP state-feedback prepared-manifest hash drift")

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
    from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_broad_training import (
        load_structured_gpp_broad_training_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_fresh_screen import (
        load_structured_gpp_fresh_screen_contract,
    )

    screen = load_structured_gpp_fresh_screen_contract(args.screen_contract)
    screen_identity = screen.payload["identity"]
    broad_report = _load_broad_training_report(args.broad_training_report, screen)
    if (
        sha256_file(args.broad_training_report) != identity["broad_training_report_sha256"]
        or sha256_file(args.selected_checkpoint) != identity["selected_checkpoint_sha256"]
        or args.stage_checkpoint_sha256 != screen_identity["base_attention_checkpoint_sha256"]
        or args.distribution_checkpoint_sha256 != screen_identity["parent_distribution_checkpoint_sha256"]
    ):
        raise ValueError("structured-GPP state-feedback checkpoint identity drift")

    amendment = load_target_oriented_flux_basis_amendment(args.flux_basis_amendment)
    context = dict(
        _load_context(
            args,
            science_flux_basis=amendment.flux_basis,
            verify_validation_artifacts=False,
        )
    )
    context["prepared_manifest"] = args.prepared_manifest
    runtime = context["runtime"]
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    target_contract = context["target_contract"]
    broad_contract = load_structured_gpp_broad_training_contract(args.broad_training_contract)
    structured_contract = load_structured_gpp_contract(args.structured_contract)
    distribution_contract = load_distribution_aware_maintenance_contract(args.distribution_contract)
    if (
        amendment.sha256 != screen_identity["flux_basis_amendment_sha256"]
        or target_contract.sha256 != screen_identity["target_contract_sha256"]
        or context["training_contract"].sha256 != screen_identity["base_training_contract_sha256"]
        or broad_contract.sha256 != identity["broad_training_contract_sha256"]
        or structured_contract.sha256 != identity["candidate_contract_sha256"]
    ):
        raise ValueError("structured-GPP state-feedback contract identity drift")

    distribution_report = json.loads(args.distribution_training_report.read_text(encoding="utf-8"))
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
        raise ValueError("structured-GPP state-feedback parent parameter count drift")

    distribution_head = build_target_oriented_distribution_aware_maintenance_head(
        _parent_head(runtime, target_contract),
        context["template"],
        distribution_contract=distribution_contract,
    )
    head = build_target_oriented_structured_gpp_head(
        distribution_head,
        structured_contract=structured_contract,
    )
    initialized = initialize_structured_gpp_parameters(
        parent_parameters,
        jax.random.PRNGKey(PROBE_SEED + 20260829),
        schema=runtime["learned_schema"],
        target_contract=target_contract,
        structured_contract=structured_contract,
    )
    selected_checkpoint = load_broad_checkpoint(
        args.selected_checkpoint,
        expected_sha256=identity["selected_checkpoint_sha256"],
        identity=broad_report["identity"],
        completed_updates=int(identity["selected_checkpoint_update"]),
        parameter_count=SPECIALIST_PARAMETER_COUNT,
        clip_threshold=float(broad_report["training"]["clip_threshold"]),
    )
    specialist = jax.device_put(selected_checkpoint["specialist"])
    parameters = replace_specialist(initialized, specialist)
    if (
        specialist_element_count(specialist) != SPECIALIST_PARAMETER_COUNT
        or _tree_size(jax, parameters) != CANDIDATE_PARAMETER_COUNT
    ):
        raise ValueError("structured-GPP state-feedback candidate parameter count drift")

    acceptance_scales = derive_target_oriented_acceptance_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    steps = _make_steps(
        runtime,
        head,
        context["template"],
        context["scales"],
        target_contract,
        acceptance_scales=acceptance_scales,
        acceptance_flux_names=AUDIT_FLUX_KEYS,
        science_flux_basis=amendment.flux_basis,
        diagnostic_teacher_carry_state_mask=True,
    )
    groups = _groups(context, attribution.evaluation_updates)
    component_masks = resolve_component_state_masks(runtime["contract"], attribution.source_components)
    free_groups = _evaluate_mode(jax, parameters, steps, groups, target_contract, mode="free")
    teacher_groups = _evaluate_mode(jax, parameters, steps, groups, target_contract, mode="full_teacher")
    component_groups = {}
    for name, _source_component in attribution.source_components:
        component_groups[name] = _evaluate_mode(
            jax,
            parameters,
            steps,
            groups,
            target_contract,
            mode="component",
            state_mask=jnp.asarray(component_masks[name]["mask"]),
        )
    classification = classify_component_feedback(
        free_groups,
        teacher_groups,
        component_groups,
        attribution,
    )
    all_modes = (free_groups, teacher_groups, *component_groups.values())
    hard_failure_count = sum(int(group["hard_failure_count"]) for mode in all_modes for group in mode)
    finite = _all_acceptance_finite(all_modes)
    passed = bool(hard_failure_count == 0 and finite)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "identity": {
            "git_head": _git_head(),
            "attribution_contract_sha256": attribution.sha256,
            "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
            "broad_training_report_sha256": sha256_file(args.broad_training_report),
            "selected_checkpoint_sha256": sha256_file(args.selected_checkpoint),
        },
        "policy": {
            "data_split": "train/train",
            "fresh_validation_values_read": False,
            "sealed_test_read": False,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "optimizer_created": False,
            "training_updates": 0,
            "parameter_steps": 0,
            "checkpoint_written": False,
            "architecture_changed": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "parameter_count": CANDIDATE_PARAMETER_COUNT,
        "component_state_scope": {
            name: {
                "source_component": record["source_component"],
                "state_keys": list(record["state_keys"]),
                "state_coordinate_count": record["state_coordinate_count"],
            }
            for name, record in component_masks.items()
        },
        "ordinary_free_rollout": free_groups,
        "full_teacher_day_start": teacher_groups,
        "single_component_teacher_carry": component_groups,
        "classification": classification,
        "integrity": {
            "all_acceptance_statistics_finite_and_nonempty": finite,
            "hard_failure_count": hard_failure_count,
            "passed": passed,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": classification["action"] if passed else attribution.payload["next_gate"]["on_integrity_failure"],
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
    parser.add_argument("--broad-training-contract", type=Path, required=True)
    parser.add_argument("--broad-training-report", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--screen-contract", type=Path, required=True)
    parser.add_argument("--attribution-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
