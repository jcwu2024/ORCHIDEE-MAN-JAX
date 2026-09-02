"""Run three zero-retention scale-selecting coupled-GPP recalibrations."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.gradient_feasibility import (
    gradient_direction_diagnostics,
)
from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_bounded_training import (
    build_coupled_gpp_training_schedule,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    GRADIENT_SOURCES,
    replace_coupled_gpp_trainable,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_scale_selecting_recalibration import (
    CONTINUED_GROUPS,
    CONTINUED_UPDATES,
    EXPECTED_UPDATE5_ACCEPTED_SCALES,
    EXPECTED_UPDATE5_SELECTED_SCALE,
    REQUIRED_CONSECUTIVE_UPDATES,
    UPDATE5_SCALE_AUDIT_RESULT_SHA256,
    load_coupled_gpp_scale_selecting_recalibration_contract,
)
from scripts.hpc.probe_successor_direct_trust_region import _make_combined_direction
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_gradient_balanced_smoke import _make_gradient_gram
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_coupled_gpp_bounded_training import _fixed_parameter_report
from scripts.hpc.run_target_oriented_coupled_gpp_direct_recalibration import (
    _git_head,
    _make_dtype_preserving_direct_virtual_step,
    _measure_source_gradients,
)
from scripts.hpc.run_target_oriented_coupled_gpp_update2_geometry import (
    _direction_record,
    _evaluate_scales,
    _scaled_parameters,
)
from scripts.hpc.run_target_oriented_coupled_gpp_update5_scale_audit import (
    _parser as _scale_audit_parser,
)
from scripts.hpc.run_target_oriented_coupled_gpp_update5_scale_audit import (
    run as _run_scale_audit,
)

SCHEMA_VERSION = "target_oriented_coupled_gpp_scale_selecting_recalibration_smoke_result_v1"


def _load_update5_scale_result(path: Path, contract: Any) -> Mapping[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    observed_sha256 = sha256_file(path)
    if (
        observed_sha256 != UPDATE5_SCALE_AUDIT_RESULT_SHA256
        or observed_sha256 != contract.payload["identity"]["update5_scale_audit_result_sha256"]
        or report.get("schema_version") != "target_oriented_coupled_gpp_update5_scale_audit_result_v1"
        or report.get("status") != "passed"
        or report.get("classification") != "bounded_direct_finite_step"
        or tuple(float(value) for value in report.get("update_5", {}).get("accepted_scales", ()))
        != EXPECTED_UPDATE5_ACCEPTED_SCALES
        or report.get("policy", {}).get("checkpoint_written") is not False
        or report.get("integrity", {}).get("parent_recovery_unchanged") is not True
    ):
        raise ValueError("coupled GPP update-5 scale result drift")
    return report


def _selected_update5_record(report: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [
        record for record in report["update_5"]["scales"] if float(record["scale"]) == EXPECTED_UPDATE5_SELECTED_SCALE
    ]
    if len(matches) != 1 or matches[0]["finite_step_accepted"] is not True:
        raise RuntimeError("coupled GPP update-5 selected scale is unavailable")
    return matches[0]


def run(
    args: Any,
    *,
    write_output: bool = True,
    return_internal: bool = False,
) -> Mapping[str, Any] | tuple[Mapping[str, Any], Mapping[str, Any]]:
    output = args.output.resolve()
    if write_output:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(output)
    started = time.perf_counter()
    contract = load_coupled_gpp_scale_selecting_recalibration_contract(args.scale_selecting_contract)
    predecessor = _load_update5_scale_result(args.scale_audit_result, contract)
    recovery_before = args.parent_recovery.stat()

    reconstructed, internal = _run_scale_audit(args, write_output=False, return_internal=True)
    if (
        reconstructed["status"] != "passed"
        or reconstructed["classification"] != "bounded_direct_finite_step"
        or tuple(float(value) for value in reconstructed["update_5"]["accepted_scales"])
        != EXPECTED_UPDATE5_ACCEPTED_SCALES
        or float(internal["selected_scale"]) != EXPECTED_UPDATE5_SELECTED_SCALE
    ):
        raise RuntimeError("coupled GPP update-5 scale selection reconstruction drift")

    context = internal["context"]
    runtime = context["runtime"]
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    schedule = build_coupled_gpp_training_schedule()
    if (
        schedule.sha256 != contract.payload["identity"]["schedule_sha256"]
        or tuple(schedule.groups[update - 1] for update in CONTINUED_UPDATES) != CONTINUED_GROUPS
    ):
        raise ValueError("coupled GPP scale-selecting schedule drift")
    required = {value for group in CONTINUED_GROUPS for value in group}
    missing = sorted(required - set(context["prepared_records"]))
    if missing:
        raise ValueError(f"coupled GPP scale-selecting prepared coverage drift: {missing}")

    initialized_parameters = internal["initialized_parameters"]
    steps = internal["steps"]
    gradient_steps = internal["gradient_steps"]
    direct_steps = internal["direct_steps"]
    state_steps = internal["state_steps"]
    layout = internal["layout"]
    flux_basis = internal["flux_basis"]
    coefficients = internal["coefficients"]
    relative_tolerance = float(contract.payload["evaluation"]["relative_source_tolerance"])
    minimum_cosine = float(contract.payload["direction"]["minimum_source_cosine"])
    parameter_step_l2 = float(contract.payload["direction"]["full_parameter_step_l2"])
    trainable = internal["selected_trainable"]
    gram_measurement = _make_gradient_gram(jax)
    combine_direction = _make_combined_direction(jax)
    direct_virtual_step = _make_dtype_preserving_direct_virtual_step(jax)

    selected_update5 = _selected_update5_record(reconstructed)
    records: list[Mapping[str, Any]] = [
        {
            "macro_update": 5,
            "artifact_updates": list(internal["group"]),
            "reconstructed_from_update5_scale_audit": True,
            "accepted_scales": list(reconstructed["update_5"]["accepted_scales"]),
            "selected_scale": EXPECTED_UPDATE5_SELECTED_SCALE,
            "selected_scale_record": selected_update5,
            "common_first_order_direction": True,
            "finite_step_accepted": True,
        }
    ]
    accepted_updates = [5]
    selected_scales = [EXPECTED_UPDATE5_SELECTED_SCALE]
    stop_classification = None

    for macro_update, group in zip(CONTINUED_UPDATES, CONTINUED_GROUPS, strict=True):
        update_started = time.perf_counter()
        parameters = replace_coupled_gpp_trainable(initialized_parameters, trainable)
        measurement = _measure_source_gradients(
            args,
            context,
            parameters=parameters,
            trainable=trainable,
            steps=steps,
            gradient_steps=gradient_steps,
            direct_steps=direct_steps,
            state_steps=state_steps,
            layout=layout,
            group=group,
            flux_basis=flux_basis,
            macro_update=macro_update,
        )
        source_gradients = measurement["source_gradients"]
        ordered_gradients = tuple(source_gradients[name] for name in GRADIENT_SOURCES)
        gram, source_nonfinite = gram_measurement(ordered_gradients)
        jax.block_until_ready((gram, source_nonfinite))
        gram_host = np.asarray(gram, dtype=np.float64)
        nonfinite_host = np.asarray(source_nonfinite, dtype=np.int64)
        diagnostics = gradient_direction_diagnostics(gram_host, labels=GRADIENT_SOURCES)
        selected_direction = diagnostics["directions"]["minimum_norm_unit"]
        common_first_order = bool(
            np.all(nonfinite_host == 0)
            and all(
                float(selected_direction["normalized_directional_derivatives"][name]) >= minimum_cosine
                for name in GRADIENT_SOURCES
            )
        )
        base_record = {
            "macro_update": macro_update,
            "artifact_updates": list(group),
            "source_mean_losses_before": measurement["source_mean_losses"],
            "source_gradients": measurement["source_reports"],
            "gradient_gram": gram_host.tolist(),
            "source_nonfinite_counts": {
                name: int(nonfinite_host[index]) for index, name in enumerate(GRADIENT_SOURCES)
            },
            "direction_diagnostics": diagnostics,
            "selected_minimum_norm_direction": selected_direction,
            "common_first_order_direction": common_first_order,
            "artifact_records": measurement["artifact_records"],
        }
        if not common_first_order:
            records.append(
                base_record
                | {
                    "finite_step_accepted": False,
                    "selected_scale": None,
                    "scale_evaluations": [],
                    "timing_seconds": {"total": time.perf_counter() - update_started},
                }
            )
            stop_classification = "no_common_first_order_direction"
            break

        weights = np.asarray([selected_direction["weights"][name] for name in GRADIENT_SOURCES], dtype=np.float64)
        common_gradient, common_norm, common_nonfinite = combine_direction(ordered_gradients, weights)
        jax.block_until_ready((common_gradient, common_norm, common_nonfinite))
        if (
            int(np.asarray(common_nonfinite)) != 0
            or not np.isfinite(float(np.asarray(common_norm)))
            or float(np.asarray(common_norm)) <= 0.0
        ):
            raise RuntimeError(f"coupled GPP scale-selecting update {macro_update} common gradient is invalid")
        proposed, proposal_nonfinite = direct_virtual_step(
            trainable,
            common_gradient,
            common_norm,
            np.asarray(parameter_step_l2, dtype=np.float64),
        )
        jax.block_until_ready((proposed, proposal_nonfinite))
        if int(np.asarray(proposal_nonfinite)) != 0:
            raise RuntimeError(f"coupled GPP scale-selecting update {macro_update} proposal is nonfinite")
        direction = _direction_record(jax, jnp, source_gradients, trainable, proposed)
        if direction["common_source_descent_direction"] is not True:
            raise RuntimeError(f"coupled GPP scale-selecting update {macro_update} realization drift")

        scale_records = []
        selected_scale = None
        for scale in contract.scales:
            current = _evaluate_scales(
                args,
                context,
                initialized_parameters=initialized_parameters,
                before_trainable=trainable,
                proposed_trainable=proposed,
                before_losses=measurement["source_mean_losses"],
                scales=(scale,),
                scale_one_evaluation=None,
                steps=steps,
                direct_steps=direct_steps,
                state_steps=state_steps,
                layout=layout,
                group=group,
                flux_basis=flux_basis,
                coefficients=coefficients,
                relative_tolerance=relative_tolerance,
                label=f"scale_selecting_update_{macro_update}",
                direction_norm=float(direction["parameter_descent_direction"]["l2_norm"]),
            )[0]
            scale_records.append(current)
            if current["finite_step_accepted"] is True:
                selected_scale = float(scale)
                break

        accepted = selected_scale is not None
        records.append(
            base_record
            | {
                "combined_gradient_norm": float(np.asarray(common_norm)),
                "direction": direction,
                "full_parameter_delta_l2": parameter_step_l2,
                "scale_evaluations": scale_records,
                "selected_scale": selected_scale,
                "finite_step_accepted": accepted,
                "timing_seconds": {"total": time.perf_counter() - update_started},
            }
        )
        print(
            f"coupled_gpp scale_selecting update={macro_update} selected_scale={selected_scale}",
            flush=True,
        )
        if not accepted:
            stop_classification = "no_accepted_finite_scale"
            break
        trainable = _scaled_parameters(jax, trainable, proposed, selected_scale)
        accepted_updates.append(macro_update)
        selected_scales.append(selected_scale)

    passed = len(accepted_updates) == REQUIRED_CONSECUTIVE_UPDATES
    classification = "three_scale_selected_recalibrations_accepted" if passed else stop_classification
    recovery_after = args.parent_recovery.stat()
    recovery_unchanged = (
        recovery_before.st_size == recovery_after.st_size and recovery_before.st_mtime_ns == recovery_after.st_mtime_ns
    )
    final_parameters = replace_coupled_gpp_trainable(initialized_parameters, trainable)
    fixed_report = _fixed_parameter_report(jax, initialized_parameters, final_parameters)
    if not recovery_unchanged or not fixed_report["exactly_unchanged"]:
        raise RuntimeError("coupled GPP scale-selecting retention boundary failed")

    next_key = (
        "on_three_updates_accepted"
        if passed
        else (
            "on_no_common_first_order_direction"
            if classification == "no_common_first_order_direction"
            else "on_no_accepted_scale"
        )
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed_scientific_gate",
        "classification": classification,
        "identity": {
            "git_head": _git_head(),
            "scale_selecting_contract_sha256": contract.sha256,
            "update5_scale_audit_contract_sha256": reconstructed["identity"]["scale_audit_contract_sha256"],
            "update5_scale_audit_result_sha256": UPDATE5_SCALE_AUDIT_RESULT_SHA256,
            "schedule_sha256": schedule.sha256,
        },
        "policy": {
            "optimizer_updates_persisted": 0,
            "parameter_proposals_persisted": 0,
            "checkpoint_written": False,
            "parent_recovery_modified": False,
            "fresh_validation_values_evaluated": False,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "sealed_test_read": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "reconstruction": {
            "predecessor_classification": predecessor["classification"],
            "current_classification": reconstructed["classification"],
            "update5_selected_scale": internal["selected_scale"],
        },
        "smoke": {
            "required_consecutive_updates": REQUIRED_CONSECUTIVE_UPDATES,
            "updates_evaluated": [int(record["macro_update"]) for record in records],
            "updates_accepted": accepted_updates,
            "selected_scales": selected_scales,
            "records": records,
        },
        "integrity": {
            "parent_recovery_unchanged": recovery_unchanged,
            "fixed_parameters": fixed_report,
            "update5_scale_audit_result_unchanged": True,
            "reconstructed_integrity": reconstructed["integrity"],
        },
        "memory_after": _memory_report(internal["device"]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": contract.payload["next_gate"][next_key],
    }
    if write_output:
        _atomic_json(output, report)
    if return_internal:
        if not passed:
            raise RuntimeError("coupled GPP scale-selecting smoke has no accepted final state")
        return report, {
            "context": context,
            "head": internal["head"],
            "initialized_parameters": initialized_parameters,
            "baseline_trainable": internal["before_trainable"],
            "final_trainable": trainable,
            "steps": steps,
            "direct_steps": direct_steps,
            "state_steps": state_steps,
            "layout": layout,
            "flux_basis": flux_basis,
            "coefficients": coefficients,
            "device": internal["device"],
        }
    return report


def _parser() -> Any:
    parser = _scale_audit_parser()
    parser.add_argument("--scale-selecting-contract", type=Path, required=True)
    parser.add_argument("--scale-audit-result", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
