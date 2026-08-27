"""Evaluate the frozen four-owner Teacher-carry bridge without training."""

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
from research.daily_coarse_graining.daily_operator.target_oriented_flux_basis import (
    load_target_oriented_flux_basis_amendment,
)
from research.daily_coarse_graining.daily_operator.target_oriented_source_owner_state_bridge import (
    classify_source_owner_state_bridge,
    load_source_owner_state_bridge_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_source_shaped_checkpoint import (
    load_source_shaped_checkpoint,
)
from research.daily_coarse_graining.daily_operator.target_oriented_source_shaped_coupling import (
    carbon_bias_identity_report,
    compare_update_states,
    load_source_shaped_coupling_attribution_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_source_shaped_screen import (
    load_source_shaped_train_train_screen_contract,
)
from scripts.hpc.audit_target_oriented_component_resolved_geometry import _merge
from scripts.hpc.audit_target_oriented_flux_gradient_variance import (
    AUDIT_FLUX_KEYS,
    _hard_host,
    _load_context,
    _tree_size,
)
from scripts.hpc.audit_target_oriented_source_shaped_coupling import _load_report
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_fast_owner_gpu_probe import PROBE_SEED
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import _head as _parent_head
from scripts.hpc.run_target_oriented_full_step_preflight import (
    _add_hard,
    _hard_failure_count,
    _make_steps,
    _zero_hard,
)
from scripts.hpc.run_target_oriented_source_shaped_one_update_admission import (
    BRANCH_PARAMETER_COUNT,
    CANDIDATE_PARAMETER_COUNT,
    _replace_source_shaped_branches,
    _retained_parameter_report,
)
from scripts.hpc.run_target_oriented_source_shaped_train_train_zero_panel import _groups

SCHEMA_VERSION = "target_oriented_source_owner_state_bridge_report_v1"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _bridge_group(
    jax: Any,
    jnp: Any,
    parameters: Any,
    steps: Mapping[str, Any],
    group: Mapping[str, Any],
    target_contract: Any,
) -> Mapping[str, Any]:
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        add_target_oriented_acceptance_statistics,
        finalize_target_oriented_acceptance_statistics,
    )

    values = {
        "7": [steps["seven_teacher_carry_acceptance_forward"](parameters, batch) for batch in group["seven_batches"]],
        "30": [
            steps["thirty_teacher_carry_acceptance_forward"](parameters, batch) for batch in group["thirty_batches"]
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


def _evaluate_bridge_panel(
    jax: Any,
    jnp: Any,
    parameters: Any,
    steps: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
    target_contract: Any,
    *,
    state_name: str,
) -> list[Mapping[str, Any]]:
    records = []
    for group in groups:
        started = time.perf_counter()
        record = _bridge_group(jax, jnp, parameters, steps, group, target_contract)
        records.append(record | {"elapsed_seconds": time.perf_counter() - started})
        print(
            f"source_owner bridge state={state_name} update={group['update']} complete",
            flush=True,
        )
    return records


def _verify_source_result(
    report: Mapping[str, Any],
    bridge: Any,
) -> None:
    identities = bridge.payload["identity"]
    if (
        report.get("schema_version") != "target_oriented_source_shaped_coupling_attribution_report_v1"
        or report["identity"]["attribution_contract_sha256"] != identities["source_coupling_contract_sha256"]
        or report["identity"]["branch_checkpoint_sha256"] != identities["branch_checkpoint_sha256"]
        or report["identity"]["parent_checkpoint_sha256"] != identities["parent_checkpoint_sha256"]
    ):
        raise ValueError("source-owner bridge source-result identity drift")
    expected = list(bridge.required_reversal_keys)
    classification = report["classification"]
    for horizon in ("7", "30"):
        if classification["recurrence_required_regressed_keys"][horizon] != expected:
            raise ValueError("source-owner bridge prerequisite reversal registry drift")


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()

    bridge = load_source_owner_state_bridge_contract(args.bridge_contract)
    identities = bridge.payload["identity"]
    attribution = load_source_shaped_coupling_attribution_contract(args.attribution_contract)
    if attribution.sha256 != identities["source_coupling_contract_sha256"]:
        raise ValueError("source-owner bridge attribution contract drift")
    source_report = _load_report(
        args.source_coupling_report,
        identities["source_coupling_report_sha256"],
        "source-shaped coupling attribution",
    )
    _verify_source_result(source_report, bridge)
    screen = load_source_shaped_train_train_screen_contract(args.screen_contract)
    if screen.sha256 != identities["source_shaped_screen_contract_sha256"]:
        raise ValueError("source-owner bridge screen contract drift")
    training_report = _load_report(
        args.branch_training_report,
        identities["branch_training_report_sha256"],
        "source-shaped branch training",
    )
    if (
        training_report.get("next_gate") != "stop_and_attribute_branch_training_screen_failure"
        or training_report["checkpoint"]["sha256"] != identities["branch_checkpoint_sha256"]
        or training_report["identity"]["parent_checkpoint_sha256"] != identities["parent_checkpoint_sha256"]
        or args.stage_checkpoint_sha256 != identities["parent_checkpoint_sha256"]
    ):
        raise ValueError("source-owner bridge training-result boundary drift")

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
    device = runtime["devices"][0]
    memory_before = _memory_report(device)

    from research.daily_coarse_graining.daily_operator.target_oriented_process_head import (
        build_target_oriented_source_shaped_maintenance_head,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_source_shaped_maintenance import (
        initialize_source_shaped_maintenance_parameters,
        load_source_shaped_maintenance_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        derive_target_oriented_acceptance_scales,
    )

    source = load_source_shaped_maintenance_contract(args.source_contract)
    if source.sha256 != screen.payload["identity"]["source_shaped_contract_sha256"]:
        raise ValueError("source-owner bridge mechanism contract drift")
    update_0_parameters = initialize_source_shaped_maintenance_parameters(
        context["parameters"],
        jax.random.PRNGKey(PROBE_SEED + 20260827),
        schema=runtime["learned_schema"],
        target_contract=target_contract,
        source_contract=source,
    )
    checkpoint = load_source_shaped_checkpoint(
        args.branch_checkpoint,
        expected_sha256=identities["branch_checkpoint_sha256"],
        identity=training_report["identity"],
        completed_updates=100,
        parameter_count=BRANCH_PARAMETER_COUNT,
        clip_threshold=float(training_report["training"]["clip_threshold"]),
    )
    update_100_parameters = _replace_source_shaped_branches(
        update_0_parameters,
        jax.device_put(checkpoint["branches"]),
    )
    retained = _retained_parameter_report(jax, update_0_parameters, update_100_parameters)
    if (
        _tree_size(jax, update_0_parameters) != CANDIDATE_PARAMETER_COUNT
        or _tree_size(jax, update_100_parameters) != CANDIDATE_PARAMETER_COUNT
        or not retained["exactly_unchanged"]
    ):
        raise ValueError("source-owner bridge parameter ownership drift")

    head = build_target_oriented_source_shaped_maintenance_head(
        _parent_head(runtime, target_contract),
        context["template"],
        source_contract=source,
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
        diagnostic_teacher_carry_state_keys=bridge.state_owner_keys,
    )
    groups = _groups(context, bridge.evaluation_updates)
    bridge_update_0 = _evaluate_bridge_panel(
        jax,
        jnp,
        update_0_parameters,
        steps,
        groups,
        target_contract,
        state_name="update_0",
    )
    bridge_update_100 = _evaluate_bridge_panel(
        jax,
        jnp,
        update_100_parameters,
        steps,
        groups,
        target_contract,
        state_name="update_100",
    )

    bridge_comparison = compare_update_states(
        bridge_update_0,
        bridge_update_100,
        attribution,
    )
    free_comparison = source_report["free_rollout_update_100_vs_update_0"]
    teacher_comparison = source_report["teacher_day_start_update_100_vs_update_0"]
    identity = carbon_bias_identity_report(
        {"update_0": bridge_update_0, "update_100": bridge_update_100},
        flux_scales=acceptance_scales.fluxes,
        contract=attribution,
    )
    classification = classify_source_owner_state_bridge(
        free_comparison,
        bridge_comparison,
        teacher_comparison,
        bridge,
    )
    all_hard = sum(int(record["hard_failure_count"]) for record in (*bridge_update_0, *bridge_update_100))
    passed = bool(bridge_comparison["all_finite"] and identity["passed"] and all_hard == 0)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "identity": {
            "git_head": _git_head(),
            "bridge_contract_sha256": bridge.sha256,
            "source_coupling_contract_sha256": attribution.sha256,
            "source_coupling_report_sha256": sha256_file(args.source_coupling_report),
            "screen_contract_sha256": screen.sha256,
            "branch_training_report_sha256": sha256_file(args.branch_training_report),
            "branch_checkpoint_sha256": sha256_file(args.branch_checkpoint),
            "parent_checkpoint_sha256": args.stage_checkpoint_sha256,
            "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
        },
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "validation_values_evaluated": False,
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
        "state_owner_keys": list(bridge.state_owner_keys),
        "four_owner_teacher_carry": {
            "update_0_groups": bridge_update_0,
            "update_100_groups": bridge_update_100,
            "hard_failure_count": all_hard,
        },
        "free_rollout_update_100_vs_update_0": free_comparison,
        "four_owner_teacher_carry_update_100_vs_update_0": bridge_comparison,
        "teacher_day_start_update_100_vs_update_0": teacher_comparison,
        "carbon_bias_identities": identity,
        "classification": classification,
        "memory_before": memory_before,
        "memory_after": _memory_report(device),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": "architecture_discussion_required_after_source_owner_bridge",
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
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--screen-contract", type=Path, required=True)
    parser.add_argument("--attribution-contract", type=Path, required=True)
    parser.add_argument("--bridge-contract", type=Path, required=True)
    parser.add_argument("--source-coupling-report", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-updates", type=int, default=1000)
    parser.add_argument("--validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--stage-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-checkpoint-sha256", required=True)
    parser.add_argument("--branch-training-report", type=Path, required=True)
    parser.add_argument("--branch-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
