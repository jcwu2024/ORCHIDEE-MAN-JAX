"""Zero-update common-descent audit for the accepted Attention state."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.gradient_feasibility import (
    GRADIENT_FEASIBILITY_LABELS,
    classify_gradient_feasibility,
    gradient_direction_diagnostics,
)
from research.daily_coarse_graining.daily_operator.successor_architecture import (
    SOURCE_ROUTED_ATTENTION,
)
from research.daily_coarse_graining.daily_operator.successor_staged_checkpoint import (
    load_staged_recovery,
)
from scripts.hpc.probe_successor_batch_diversity import (
    DIAGNOSTIC_GROUPS,
    POINTS_PER_MIXED_UPDATE,
    _join_context_anchors,
    _selected_positions,
    _slice_context_batch,
    _stack_contexts,
)
from scripts.hpc.probe_successor_named_biomass_gradients import (
    _calibration_loss_vector,
    _make_context_trajectory,
)
from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    EXPECTED_PARAMETER_COUNT,
    _load_extended_runtime,
)
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_successor_small_fit import (
    _device_memory_report,
    _hard_counts_host,
    _make_arm_objectives,
    _prepare_adapter,
    _retained_tail_trace_signature,
)
from scripts.hpc.run_successor_staged_training import _training_identity
from scripts.hpc.successor_small_fit_prepared import load_prepared_artifact
from scripts.hpc.successor_staged_prepared import (
    load_staged_prepared_manifest,
    stage_update_range,
)
from scripts.hpc.successor_staged_validation_prepared import (
    load_staged_validation_prepared,
)

PARENT_UPDATE = 1000


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("contract_sha256", None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_contract(
    path: Path,
    *,
    schema_version: str,
    status: str,
) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != schema_version
        or payload.get("status") != status
        or payload.get("contract_sha256") != _canonical_sha256(payload)
    ):
        raise ValueError(f"contract drift: {schema_version}")
    return payload


def _make_measurement(
    runtime: Mapping[str, Any],
    adapter_template: Any,
    *,
    runner_contract: Any,
    long_scales: Any,
    tendency_scales: Mapping[int, Mapping[str, float]],
    flux_coefficients: Mapping[str, float],
    tendency_coefficient: float,
) -> Any:
    import jax.numpy as jnp

    jax = runtime["jax"]
    anchor_objective, _rollout_objective = _make_arm_objectives(
        runtime,
        SOURCE_ROUTED_ATTENTION,
        adapter_template,
        long_scales,
    )
    context_trajectory = _make_context_trajectory(
        runtime,
        adapter_template,
        runner_contract,
    )

    def objective(
        parameters: Any,
        anchor_batch: Any,
        context_batches: Any,
        day_inputs: Any,
    ) -> Any:
        anchor = anchor_objective(parameters, anchor_batch)
        predicted, expected, counts = jax.vmap(
            context_trajectory,
            in_axes=(None, 0, 0),
        )(parameters, context_batches, day_inputs)
        hard_counts = jax.tree_util.tree_map(
            lambda values: jnp.sum(values, axis=0),
            counts,
        )
        long_components = _calibration_loss_vector(
            jax,
            predicted,
            expected,
            flux_scales=long_scales.fluxes,
            stock_scales=long_scales.stocks,
            tendency_scales=tendency_scales,
            flux_coefficients=flux_coefficients,
        )
        losses = jnp.stack(
            (
                anchor,
                long_components[0],
                tendency_coefficient * long_components[1],
                tendency_coefficient * long_components[2],
            )
        )
        return losses, hard_counts

    def measurement(
        parameters: Any,
        seed: Any,
        anchor_batch: Any,
        context_batches: Any,
        day_inputs: Any,
    ) -> Any:
        losses, pullback, hard_counts = jax.vjp(
            lambda current: objective(
                current,
                anchor_batch,
                context_batches,
                day_inputs,
            ),
            parameters,
            has_aux=True,
        )
        return losses, pullback(seed)[0], hard_counts

    return jax.jit(measurement)


def _measurement_record(
    jax: Any,
    measurement: Any,
    parameters: Any,
    anchor_batch: Any,
    context_batches: Any,
    day_inputs: Any,
) -> Mapping[str, Any]:
    vectors = []
    nonfinite = []
    losses_host = None
    hard_host = None
    for seed in np.eye(len(GRADIENT_FEASIBILITY_LABELS), dtype=np.float64):
        losses, gradient, hard_counts = measurement(
            parameters,
            seed,
            anchor_batch,
            context_batches,
            day_inputs,
        )
        jax.block_until_ready((losses, gradient, hard_counts))
        current_losses = np.asarray(losses, dtype=np.float64)
        current_hard = _hard_counts_host(hard_counts)
        if losses_host is None:
            losses_host = current_losses
            hard_host = current_hard
        elif not np.array_equal(losses_host, current_losses) or hard_host != current_hard:
            raise ValueError("gradient feasibility seed changed objective outputs")
        leaves = tuple(
            np.asarray(value, dtype=np.float64).reshape(-1)
            for value in jax.device_get(jax.tree_util.tree_leaves(gradient))
        )
        vector = np.concatenate(leaves)
        nonfinite.append(int(np.count_nonzero(~np.isfinite(vector))))
        vectors.append(vector)
        del losses, gradient, hard_counts, leaves, vector
    if losses_host is None or hard_host is None:
        raise ValueError("gradient feasibility produced no measurements")
    gram = np.asarray(
        [
            [float(np.dot(left, right)) for right in vectors]
            for left in vectors
        ],
        dtype=np.float64,
    )
    diagnostics = gradient_direction_diagnostics(gram)
    del vectors
    return {
        "losses": {
            label: float(losses_host[index])
            for index, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
        },
        "nonfinite_gradient_values": {
            label: int(nonfinite[index])
            for index, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
        },
        "hard_gate_counts": hard_host,
        "gram_matrix": gram.tolist(),
        "diagnostics": diagnostics,
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    feasibility = _load_contract(
        args.feasibility_contract,
        schema_version="successor_gradient_feasibility_v1",
        status="frozen_zero_update_no_training_authorized",
    )
    candidate = _load_contract(
        args.candidate_coefficient_contract,
        schema_version="successor_named_biomass_candidate_coefficients_v1",
        status="accepted_zero_update_preflight_no_training_authorized",
    )
    if _sha256_file(args.rejected_ab_report) != feasibility["identity"][
        "rejected_objective_ab_report_sha256"
    ]:
        raise ValueError("gradient feasibility rejected A/B report drift")

    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]
    from research.daily_coarse_graining.daily_operator.successor_architecture import (
        initialize_successor_parameters,
        successor_parameter_count,
    )
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        STAGED_SCHEMA_VERSION,
        build_successor_small_fit_schedule,
        load_successor_small_fit_contract,
        verify_successor_small_fit_schedule,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        LONG_HORIZON_COMPONENTS,
        derive_long_horizon_scales,
    )
    from research.daily_coarse_graining.daily_operator.successor_training_assets import (
        load_named_biomass_tendency_scale_asset,
        load_successor_training_assets,
    )

    contract = load_successor_small_fit_contract(args.runner_contract)
    if (
        contract.payload["schema_version"] != STAGED_SCHEMA_VERSION
        or feasibility["identity"]["staged_training_contract_sha256"]
        != contract.sha256
        or feasibility["identity"]["candidate_coefficient_contract_sha256"]
        != candidate["contract_sha256"]
    ):
        raise ValueError("gradient feasibility training identity drift")
    arms = tuple(contract.payload["matched_arms"])
    references = runtime["index"].select(spatial_split="train", temporal_split="train")
    schedule = build_successor_small_fit_schedule(
        references,
        contract,
        updates=int(contract.sampling["maximum_updates_per_arm"]),
    )
    verify_successor_small_fit_schedule(schedule, references, contract)
    start_update, end_update = stage_update_range(
        contract.sampling["stage_milestones"],
        2000,
    )
    manifest = load_staged_prepared_manifest(
        args.prepared_manifest,
        staged_training_contract_sha256=contract.sha256,
        full_schedule_sha256=schedule["schedule_sha256"],
        start_update=start_update,
        end_update=end_update,
    )
    validation_prepared, prepared_panel = load_staged_validation_prepared(
        args.validation_prepared_manifest,
        staged_training_contract_sha256=contract.sha256,
        validation_panel_sha256=args.validation_panel_sha256,
    )
    assets = load_successor_training_assets(
        args.coefficient_contract,
        args.validation_panel,
        coefficient_report_path=args.coefficient_report,
        prepared_manifest_path=args.coefficient_prepared_manifest,
        runner_contract=contract,
        components=LONG_HORIZON_COMPONENTS,
    )
    if prepared_panel != assets.validation_panel:
        raise ValueError("gradient feasibility validation identity drift")
    scale_asset = load_named_biomass_tendency_scale_asset(
        args.tendency_scale_asset,
        args.tendency_scale_acceptance,
    )
    scale_acceptance = json.loads(
        args.tendency_scale_acceptance.read_text(encoding="utf-8")
    )
    if scale_acceptance["contract_sha256"] != feasibility["identity"][
        "scale_acceptance_contract_sha256"
    ]:
        raise ValueError("gradient feasibility scale acceptance drift")

    identity = _training_identity(runtime, contract, schedule, assets, validation_prepared)
    selected = []
    for position in _selected_positions(len(manifest["entries"])):
        record = manifest["entries"][position]
        entry = schedule["entries"][int(record["update"]) - 1]
        artifact = load_prepared_artifact(
            args.prepared_manifest,
            record,
            schedule_entry=entry,
        )
        selected.append((entry, artifact))
    if len(selected) != DIAGNOSTIC_GROUPS * POINTS_PER_MIXED_UPDATE:
        raise ValueError("gradient feasibility selection size drift")
    first_entry, first_artifact = selected[0]
    _reference, adapter_template, adapter_timing = _prepare_adapter(runtime, first_entry)
    signature = _retained_tail_trace_signature(adapter_template)
    if any(artifact.trace_signature != signature for _entry, artifact in selected):
        raise ValueError("gradient feasibility retained-tail trace drift")
    assembled = runtime["adapter"].input_assembler.assemble(
        jax.tree_util.tree_map(lambda value: value[0], first_artifact.batch.anchor_inputs)
    )
    initialized = {
        arm: initialize_successor_parameters(
            jax.random.PRNGKey(int(contract.sampling["seed"])),
            schema=runtime["schema"],
            layout=runtime["layout"],
            inputs=assembled,
            arm=arm,
        )
        for arm in arms
    }
    counts = {arm: successor_parameter_count(initialized[arm]) for arm in arms}
    if counts.get(SOURCE_ROUTED_ATTENTION) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("gradient feasibility Attention parameter count drift")
    recovery = load_staged_recovery(
        args.recovery_path,
        identity=identity,
        expected_arms=arms,
        parameter_counts=counts,
        minimum_completed_updates=PARENT_UPDATE,
        maximum_completed_updates=PARENT_UPDATE,
    )
    if recovery is None:
        raise ValueError("gradient feasibility requires update-1000 recovery")
    parameters = recovery["parameters"][SOURCE_ROUTED_ATTENTION]
    long_scales = derive_long_horizon_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    tendency_coefficient = float(
        candidate["resolved_coefficients"]["tendency_mse"]
    )
    measurement = _make_measurement(
        runtime,
        adapter_template,
        runner_contract=contract,
        long_scales=long_scales,
        tendency_scales=scale_asset.scales,
        flux_coefficients=assets.coefficients,
        tendency_coefficient=tendency_coefficient,
    )

    records = []
    memory_before = _device_memory_report(runtime["devices"])
    for group_index in range(DIAGNOSTIC_GROUPS):
        group = selected[
            group_index * POINTS_PER_MIXED_UPDATE :
            (group_index + 1) * POINTS_PER_MIXED_UPDATE
        ]
        contexts = tuple(
            _slice_context_batch(jax, artifact.batch, context_index=index)
            for index, (_entry, artifact) in enumerate(group)
        )
        anchor_batch = _join_context_anchors(jax, contexts)
        context_batch = _stack_contexts(jax, contexts)
        context_inputs = _stack_contexts(
            jax,
            tuple(artifact.day_inputs for _entry, artifact in group),
        )
        record = _measurement_record(
            jax,
            measurement,
            parameters,
            anchor_batch,
            context_batch,
            context_inputs,
        )
        records.append(
            {
                "group": group_index,
                "point_years": [
                    {
                        "update": int(entry["update"]),
                        "landpoint_id": str(entry["landpoint_id"]),
                        "year": int(entry["year"]),
                    }
                    for entry, _artifact in group
                ],
                **record,
            }
        )
        del group, contexts, anchor_batch, context_batch, context_inputs
        gc.collect()
        print(
            json.dumps(
                {
                    "event": "successor_gradient_feasibility_group",
                    "group": group_index + 1,
                    "total": DIAGNOSTIC_GROUPS,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    hard_failure_count = sum(
        sum(record["hard_gate_counts"].values()) for record in records
    )
    nonfinite_gradient_count = sum(
        sum(record["nonfinite_gradient_values"].values()) for record in records
    )
    all_losses_finite = all(
        np.all(np.isfinite(tuple(record["losses"].values()))) for record in records
    )
    geometry = classify_gradient_feasibility(
        records,
        gates=feasibility["gates"],
    )
    passed = (
        geometry["passed"]
        and hard_failure_count == 0
        and nonfinite_gradient_count == 0
        and all_losses_finite
    )
    report = {
        "schema_version": "successor_gradient_feasibility_report_v1",
        "status": "passed" if passed else "failed",
        "git_head": _git_head(),
        "identity": {
            "feasibility_contract_sha256": feasibility["contract_sha256"],
            "staged_training_contract_sha256": contract.sha256,
            "candidate_coefficient_contract_sha256": candidate["contract_sha256"],
            "tendency_scale_asset_sha256": scale_asset.payload["asset_sha256"],
            "rejected_objective_ab_report_sha256": _sha256_file(
                args.rejected_ab_report
            ),
        },
        "parent_update": PARENT_UPDATE,
        "architecture": SOURCE_ROUTED_ATTENTION,
        "parameter_count": counts[SOURCE_ROUTED_ATTENTION],
        "gradient_labels": list(GRADIENT_FEASIBILITY_LABELS),
        "component_policy": {
            "one_day_anchor_weight": 1.0,
            "npp_uses_accepted_flux_coefficients": True,
            "shared_tendency_coefficient": tendency_coefficient,
            "tendency_scales": {
                str(horizon): dict(values)
                for horizon, values in scale_asset.scales.items()
            },
        },
        "gates": {
            "all_losses_finite": all_losses_finite,
            "hard_failure_count": hard_failure_count,
            "nonfinite_gradient_value_count": nonfinite_gradient_count,
        },
        "classification": geometry,
        "records": records,
        "adapter_preparation": adapter_timing,
        "device_memory_before": memory_before,
        "device_memory_after": _device_memory_report(runtime["devices"]),
        "elapsed_seconds": time.perf_counter() - started,
        "training_updates": 0,
        "checkpoint_written": False,
        "recovery_written": False,
        "validation_values_evaluated": False,
        "sealed_test_used": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "next_gate": (
            feasibility["pass_action"] if passed else feasibility["fail_action"]
        ),
    }
    _atomic_json(args.output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--runner-contract", type=Path, required=True)
    parser.add_argument("--feasibility-contract", type=Path, required=True)
    parser.add_argument("--candidate-coefficient-contract", type=Path, required=True)
    parser.add_argument("--tendency-scale-asset", type=Path, required=True)
    parser.add_argument("--tendency-scale-acceptance", type=Path, required=True)
    parser.add_argument("--coefficient-contract", type=Path, required=True)
    parser.add_argument("--coefficient-report", type=Path, required=True)
    parser.add_argument("--coefficient-prepared-manifest", type=Path, required=True)
    parser.add_argument("--validation-panel", type=Path, required=True)
    parser.add_argument("--validation-panel-sha256", required=True)
    parser.add_argument("--validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--recovery-path", type=Path, required=True)
    parser.add_argument("--rejected-ab-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
