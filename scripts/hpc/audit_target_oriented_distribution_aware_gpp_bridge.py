"""Evaluate the frozen update-250 Teacher-GPP bridge without training."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import (
    sha256_file,
)
from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_checkpoint import (
    load_distribution_aware_checkpoint,
)
from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_gpp_bridge import (
    classify_gpp_bridge,
    compare_gpp_bridge_modes,
    load_distribution_aware_gpp_bridge_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_training import (
    load_distribution_aware_bounded_training_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_flux_basis import (
    load_target_oriented_flux_basis_amendment,
)
from scripts.hpc.audit_target_oriented_component_resolved_geometry import _merge
from scripts.hpc.audit_target_oriented_flux_gradient_variance import (
    AUDIT_FLUX_KEYS,
    _hard_host,
    _load_context,
    _tree_size,
)
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import (
    _adapter_tree,
    _replace_adapter,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_fast_owner_gpu_probe import PROBE_SEED
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import _head as _parent_head
from scripts.hpc.run_target_oriented_distribution_aware_one_update_admission import (
    _retained_parameter_report,
)
from scripts.hpc.run_target_oriented_distribution_aware_training import (
    ADAPTER_PARAMETER_COUNT,
    CANDIDATE_PARAMETER_COUNT,
)
from scripts.hpc.run_target_oriented_full_step_preflight import (
    _add_hard,
    _hard_failure_count,
    _make_steps,
    _zero_hard,
)
from scripts.hpc.run_target_oriented_source_shaped_train_train_zero_panel import _groups

SCHEMA_VERSION = "target_oriented_distribution_aware_gpp_bridge_report_v1"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _load_training_report(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("distribution-aware training report identity drift")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != "target_oriented_distribution_aware_training_v1"
        or report.get("status") != "passed"
        or report.get("next_gate") != "stop_unchanged_adapter_only_continuation"
        or int(report["training"]["optimizer_update_count"]) != 250
        or report["checkpoint"]["promotion_eligible"] is not False
    ):
        raise ValueError("distribution-aware training result is not the stopped update-250 run")
    return report


def _mode_group(
    jax: Any,
    jnp: Any,
    parameters: Any,
    steps: Mapping[str, Any],
    group: Mapping[str, Any],
    target_contract: Any,
    *,
    seven_step: str,
    thirty_step: str,
) -> Mapping[str, Any]:
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        add_target_oriented_acceptance_statistics,
        finalize_target_oriented_acceptance_statistics,
    )

    values = {
        "7": [steps[seven_step](parameters, batch) for batch in group["seven_batches"]],
        "30": [
            steps[thirty_step](parameters, batch)
            for batch in group["thirty_batches"]
        ],
    }
    jax.block_until_ready(values)
    hard = _zero_hard(jnp)
    acceptance = {}
    for horizon, current in values.items():
        total = _merge(
            add_target_oriented_acceptance_statistics,
            [value[0] for value in current],
        )
        acceptance[horizon] = finalize_target_oriented_acceptance_statistics(
            total,
            horizon=int(horizon),
            target_contract=target_contract,
        )
        for value in current:
            hard = _add_hard(jax, hard, value[1])
    jax.block_until_ready(hard)
    hard_host = _hard_host(jax.device_get(hard))
    return {
        "update": int(group["update"]),
        "runtime_year": int(group["runtime_year"]),
        "acceptance": acceptance,
        "hard_counts": hard_host,
        "hard_failure_count": _hard_failure_count(hard_host),
    }


def _evaluate_panel(
    jax: Any,
    jnp: Any,
    parameters: Any,
    steps: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
    target_contract: Any,
    *,
    mode: str,
    seven_step: str,
    thirty_step: str,
) -> list[Mapping[str, Any]]:
    records = []
    for group in groups:
        started = time.perf_counter()
        record = _mode_group(
            jax,
            jnp,
            parameters,
            steps,
            group,
            target_contract,
            seven_step=seven_step,
            thirty_step=thirty_step,
        )
        records.append(record | {"elapsed_seconds": time.perf_counter() - started})
        print(f"distribution_aware gpp_bridge mode={mode} update={group['update']} complete", flush=True)
    return records


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()

    bridge = load_distribution_aware_gpp_bridge_contract(args.bridge_contract)
    identities = bridge.payload["identity"]
    bounded = load_distribution_aware_bounded_training_contract(args.bounded_contract)
    if bounded.sha256 != identities["bounded_training_contract_sha256"]:
        raise ValueError("Teacher-GPP bridge bounded-training contract drift")
    training_report = _load_training_report(
        args.training_report,
        identities["training_report_sha256"],
    )
    if (
        training_report["checkpoint"]["sha256"] != identities["candidate_checkpoint_sha256"]
        or args.candidate_checkpoint_sha256 != identities["candidate_checkpoint_sha256"]
        or args.stage_checkpoint_sha256 != identities["parent_checkpoint_sha256"]
        or training_report["identity"]["bounded_training_contract_sha256"] != bounded.sha256
    ):
        raise ValueError("Teacher-GPP bridge checkpoint boundary drift")
    free_groups = training_report["final_screen"]["candidate_groups"]

    amendment = load_target_oriented_flux_basis_amendment(args.flux_basis_amendment)
    if amendment.sha256 != identities["flux_basis_amendment_sha256"]:
        raise ValueError("Teacher-GPP bridge flux-basis identity drift")
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
    if (
        sha256_file(args.prepared_manifest) != identities["prepared_manifest_sha256"]
        or target_contract.sha256 != identities["target_contract_sha256"]
        or context["training_contract"].sha256 != bounded.payload["identity"][
            "base_training_contract_sha256"
        ]
    ):
        raise ValueError("Teacher-GPP bridge prepared data or objective identity drift")

    from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_maintenance import (
        initialize_distribution_aware_maintenance_parameters,
        load_distribution_aware_maintenance_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_process_head import (
        build_target_oriented_distribution_aware_maintenance_head,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        derive_target_oriented_acceptance_scales,
    )

    candidate = load_distribution_aware_maintenance_contract(args.candidate_contract)
    if candidate.sha256 != identities["candidate_contract_sha256"]:
        raise ValueError("Teacher-GPP bridge candidate contract drift")
    initialized = initialize_distribution_aware_maintenance_parameters(
        context["parameters"],
        jax.random.PRNGKey(PROBE_SEED + 20260828),
        schema=runtime["learned_schema"],
        target_contract=target_contract,
        distribution_contract=candidate,
    )
    checkpoint = load_distribution_aware_checkpoint(
        args.candidate_checkpoint,
        expected_sha256=args.candidate_checkpoint_sha256,
        identity=training_report["identity"],
        completed_updates=250,
        parameter_count=ADAPTER_PARAMETER_COUNT,
        clip_threshold=float(training_report["training"]["clip_threshold"]),
    )
    parameters = _replace_adapter(initialized, jax.device_put(checkpoint["adapter"]))
    retained = _retained_parameter_report(jax, initialized, parameters)
    if (
        _tree_size(jax, parameters) != CANDIDATE_PARAMETER_COUNT
        or _tree_size(jax, _adapter_tree(parameters)) != ADAPTER_PARAMETER_COUNT
        or not retained["exactly_unchanged"]
    ):
        raise ValueError("Teacher-GPP bridge parameter ownership drift")

    head = build_target_oriented_distribution_aware_maintenance_head(
        _parent_head(runtime, target_contract),
        context["template"],
        distribution_contract=candidate,
    )
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
        diagnostic_teacher_gpp_bridge=True,
    )
    groups = _groups(context, bridge.evaluation_updates)
    memory_before = _memory_report(runtime["devices"][0])
    bridge_groups = _evaluate_panel(
        jax,
        jnp,
        parameters,
        steps,
        groups,
        target_contract,
        mode="teacher_gpp_bridge",
        seven_step="seven_teacher_gpp_bridge_acceptance_forward",
        thirty_step="thirty_teacher_gpp_bridge_acceptance_forward",
    )
    teacher_groups = _evaluate_panel(
        jax,
        jnp,
        parameters,
        steps,
        groups,
        target_contract,
        mode="teacher_day_start",
        seven_step="seven_teacher_state_acceptance_forward",
        thirty_step="thirty_teacher_state_acceptance_forward",
    )
    bridge_comparison = compare_gpp_bridge_modes(free_groups, bridge_groups, bridge)
    teacher_comparison = compare_gpp_bridge_modes(free_groups, teacher_groups, bridge)
    bridge_hard = sum(int(record["hard_failure_count"]) for record in bridge_groups)
    teacher_hard = sum(int(record["hard_failure_count"]) for record in teacher_groups)
    classification = classify_gpp_bridge(
        bridge_comparison,
        hard_failure_count=bridge_hard,
        contract=bridge,
    )
    gpp_wiring_passed = all(
        bridge_comparison["aggregate_acceptance"][horizon]["keys"]["gpp"]["passed"]
        for horizon in ("7", "30")
    )
    passed = bool(
        bridge_comparison["all_finite"]
        and teacher_comparison["all_finite"]
        and bridge_hard == 0
        and teacher_hard == 0
        and gpp_wiring_passed
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "identity": {
            "git_head": _git_head(),
            "bridge_contract_sha256": bridge.sha256,
            "bounded_training_contract_sha256": bounded.sha256,
            "training_report_sha256": sha256_file(args.training_report),
            "candidate_checkpoint_sha256": sha256_file(args.candidate_checkpoint),
            "parent_checkpoint_sha256": args.stage_checkpoint_sha256,
            "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
        },
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "validation_values_evaluated": False,
            "validation_artifact_values_read": False,
            "sealed_test_read": False,
            "optimizer_created": False,
            "training_updates": 0,
            "parameter_steps": 0,
            "checkpoint_written": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
            "architecture_changed": False,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "parameter_count": CANDIDATE_PARAMETER_COUNT,
        "retained_parameters": retained,
        "ordinary_free_rollout": {
            "source": "accepted update-250 final screen",
            "groups": free_groups,
        },
        "teacher_gpp_bridge": {
            "groups": bridge_groups,
            "hard_failure_count": bridge_hard,
            "gpp_wiring_gate_passed": gpp_wiring_passed,
            "comparison_to_free": bridge_comparison,
        },
        "teacher_day_start": {
            "groups": teacher_groups,
            "hard_failure_count": teacher_hard,
            "comparison_to_free": teacher_comparison,
        },
        "classification": classification,
        "memory_before": memory_before,
        "memory_after": _memory_report(runtime["devices"][0]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": "architecture_discussion_required_after_teacher_gpp_bridge",
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
    parser.add_argument("--candidate-contract", type=Path, required=True)
    parser.add_argument("--bounded-contract", type=Path, required=True)
    parser.add_argument("--bridge-contract", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-updates", type=int, default=1000)
    parser.add_argument("--validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--stage-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-checkpoint-sha256", required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
