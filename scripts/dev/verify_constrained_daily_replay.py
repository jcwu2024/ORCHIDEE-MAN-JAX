from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver import orchestration as teacher  # noqa: E402
from jax_orchidee.runtime import configure_jax_compilation_cache  # noqa: E402
from research.daily_coarse_graining.constrained_daily_replay import (  # noqa: E402
    audit_daily_carbon_budget,
    audit_daily_thermal_residual,
    audit_daily_water_budget,
    capture_defined_status,
    constrained_replay_boundary,
)
from research.daily_coarse_graining.replay_ceiling import (  # noqa: E402
    capture_pre_daily_stomate_record,
    compare_replay_day,
    replay_pre_daily_stomate_record,
)


def _block(value: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(name): _json_value(child) for name, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(child) for child in value]
    if isinstance(value, np.ndarray) or hasattr(value, "shape"):
        array = np.asarray(value)
        return array.item() if array.ndim == 0 else array.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _capture_case(
    *,
    name: str,
    config: Path,
    context: Any,
    previous_state: Any,
    year: int,
    day_index: int,
    start_tstep: int,
    inventory: Path,
) -> tuple[dict[str, Any], Any]:
    common = dict(
        used_run_def_path=context.run_def_path,
        prepared_context=context,
        module_jit=True,
        diffuco_local_jit=True,
        maintenance_local_jit=True,
        retain_stomate_step_results=False,
        use_static_jit_daily_carbon=True,
        prebuild_day_payloads=True,
        use_compiled_sechiba_day=True,
        capture_daily_flux_labels=True,
    )
    record = capture_pre_daily_stomate_record(
        config,
        previous_state=previous_state,
        year=year,
        day_index=day_index,
        start_tstep=start_tstep,
        **common,
    )
    _block(record.expected_result)
    boundary = constrained_replay_boundary(
        previous_state,
        record,
        inventory_path=str(inventory),
    )
    replay = replay_pre_daily_stomate_record(
        config,
        previous_state=previous_state,
        record=boundary.record,
        used_run_def_path=context.run_def_path,
        prepared_context=context,
        module_jit=True,
        diffuco_local_jit=True,
        use_static_jit_daily_carbon=True,
    )
    _block(replay)
    comparison = compare_replay_day(boundary.record, replay, atol=1.0e-10, rtol=1.0e-12)
    water = audit_daily_water_budget(
        previous_state,
        record,
        dz_mm=context.hydrol_dz_mm,
    )
    carbon = audit_daily_carbon_budget(previous_state, record)
    thermal = audit_daily_thermal_residual(record)
    defined_masks = capture_defined_status(record.daily_flux_labels)
    netrad_defined = defined_masks["energy.netrad_pft"]
    if netrad_defined.shape[1] != 14:
        raise RuntimeError("paper PFT14 replay expected the 14-slot Teacher layout")
    mask_report = {
        "pft14_energy_defined": bool(netrad_defined[:, 13].all()),
        "bare_energy_undefined": bool((~netrad_defined[:, 0]).all()),
        "vegetation_slots_energy_defined": bool(netrad_defined[:, 1:].all()),
        "inactive_vegetation_slots_present": bool(
            (
                np.asarray(previous_state.fields_by_component["slowproc_stomate_previous_step_state"]["veget_max"])[
                    :, 1:13
                ]
                <= 1.0e-8
            ).all()
        ),
        "defined_masks_explicit": True,
        "captured_tensor_count": len(defined_masks),
    }
    passed = bool(
        comparison["day_end_state"]["passed"]
        and comparison["modelout_fields"]["passed"]
        and comparison["modelout"]["passed"]
        and boundary.report["minimum_nonnegative_inventory_component"]
        >= boundary.report["inventory_numerical_zero_lower_bound"]
        and mask_report["pft14_energy_defined"]
        and mask_report["bare_energy_undefined"]
        and mask_report["vegetation_slots_energy_defined"]
        and mask_report["inactive_vegetation_slots_present"]
        and water["max_absolute_residual"] <= 2.0e-8
        and carbon["max_absolute_residual"] <= 2.0e-8
        and thermal["max_absolute_flux_side_residual"] <= 1.0e-7
    )
    return (
        {
            "name": name,
            "passed": passed,
            "year": year,
            "day_index": day_index,
            "start_tstep": start_tstep,
            "boundary": boundary.report,
            "comparison": comparison,
            "water_budget": water,
            "carbon_budget": carbon,
            "thermal_residual": thermal,
            "mask": mask_report,
        },
        record.expected_result.day_end_state,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Gate-C2 non-neural constrained replay.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/orchidee_man_250919.yaml",
    )
    parser.add_argument(
        "--run-def",
        type=Path,
        default=ROOT / "outputs/reference_mode/used_run.def",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=ROOT / "manifests/coarse_graining/daily_flux_label_inventory_v1.json",
    )
    parser.add_argument(
        "--restart-evidence",
        type=Path,
        default=ROOT / "outputs/reference_mode/stage5_lifecycle_acceptance.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/research/daily_coarse_graining/gate_c2_constrained_replay/comparison.json",
    )
    args = parser.parse_args()

    configure_jax_compilation_cache(ROOT)
    context = teacher.prepare_paper_1961_driver_context(
        args.config,
        used_run_def_path=args.run_def,
    )
    seed = teacher.paper_1961_driver_multiday_modelout_lite_run(
        args.config,
        ndays=1,
        year=1961,
        used_run_def_path=args.run_def,
        module_jit=True,
        diffuco_local_jit=True,
        compact_later_days=True,
        use_static_jit_daily_carbon=True,
        prebuild_day_payloads=True,
        use_compiled_sechiba_day=True,
    )
    if not seed.ready_for_requested_days or seed.last_day_end_state is None:
        raise RuntimeError(f"cold-start seed failed: {seed.missing_components}")

    cold_continuation, day2_state = _capture_case(
        name="cold_start_continuation_day2",
        config=args.config,
        context=context,
        previous_state=seed.last_day_end_state,
        year=1961,
        day_index=2,
        start_tstep=48,
        inventory=args.inventory,
    )
    ordinary, day3_state = _capture_case(
        name="ordinary_later_day3",
        config=args.config,
        context=context,
        previous_state=day2_state,
        year=1961,
        day_index=3,
        start_tstep=96,
        inventory=args.inventory,
    )
    restart_start = teacher.rebase_driver_state_for_year_start(day3_state)
    restart_year, _ = _capture_case(
        name="restart_year_boundary_1962_day1",
        config=args.config,
        context=context,
        previous_state=restart_start,
        year=1962,
        day_index=1,
        start_tstep=0,
        inventory=args.inventory,
    )

    restart_evidence = json.loads(args.restart_evidence.read_text(encoding="utf-8"))
    required_restart_checks = {
        "driver_file_roundtrip_exact",
        "sechiba_file_roundtrip_exact",
        "stomate_file_roundtrip_exact",
        "split_vs_memory_state_exact",
    }
    passed_restart_checks = {item["name"] for item in restart_evidence.get("checks", ()) if item.get("passed") is True}
    restart_roundtrip = {
        "evidence": str(args.restart_evidence.relative_to(ROOT)),
        "evidence_status": restart_evidence.get("status"),
        "required_checks": sorted(required_restart_checks),
        "passed": restart_evidence.get("status") == "passed" and required_restart_checks <= passed_restart_checks,
        "composition": (
            "accepted exact three-file roundtrip followed by the constrained 1962 day-1 replay case in this report"
        ),
    }
    cases = [cold_continuation, ordinary, restart_year]
    report = {
        "schema_version": "gate_c2_constrained_daily_replay_v1",
        "status": "passed" if all(case["passed"] for case in cases) and restart_roundtrip["passed"] else "failed",
        "scope": ("true-label non-neural constrained fast boundary plus exact retained daily tail"),
        "candidate_uses_native_daily_boundary": True,
        "candidate_reconstructs_48_forcing_steps": False,
        "candidate_executes_48_state_transitions": False,
        "endpoint_tendencies_are_process_fluxes": False,
        "cases": cases,
        "restart_roundtrip": restart_roundtrip,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_value(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_value(report), indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
