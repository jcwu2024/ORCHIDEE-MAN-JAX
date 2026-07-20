from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.orchestration import (  # noqa: E402
    DriverPreviousStepStatePacket,
    paper_1961_driver_cold_start_day_scaffold,
    paper_1961_driver_later_day_scaffold,
    paper_1961_driver_restart_year_start_day_result,
    prepare_paper_1961_driver_context,
)
from jax_orchidee.driver.restart_file_io import read_dim2_driver_restart  # noqa: E402
from jax_orchidee.driver.restart_bundle import (  # noqa: E402
    PaperRestartBundlePhysicalState,
    paper_restart_bundle_state_from_day_end_packet,
    read_restart_physical_state,
    write_paper_restart_start_bundle,
)
from jax_orchidee.driver.restart_state import reference_case_first_step_restart_state  # noqa: E402
from jax_orchidee.sechiba.restart_io import (  # noqa: E402
    sechiba_finalize_source_state_from_restart,
)


CONFIG = ROOT / "configs/orchidee_man_250919.yaml"
OUTPUT = ROOT / "outputs/reference_mode/stage5_lifecycle_acceptance.json"
STATE_LEDGER = ROOT / "outputs/reference_mode/pft14_state_transition_ledger.json"
RESTART_LIFECYCLE = ROOT / "outputs/reference_mode/restart_state_lifecycle.json"
RESTART_BUNDLE = ROOT / "outputs/reference_mode/paper_restart_bundle_phase.json"
PRODUCTION_LIFECYCLE = ROOT / "outputs/reference_mode/production/lifecycle/comparison.json"
PRODUCTION_HYDROL_THERMOSOIL = ROOT / (
    "outputs/reference_mode/production/hydrol_thermosoil/comparison.json"
)
PRODUCTION_STOMATE = ROOT / "outputs/reference_mode/production/stomate/comparison.json"
OWNER_EVIDENCE = ROOT / "outputs/reference_mode/pft14_owner_region_evidence.json"
ARM_EVIDENCE = ROOT / "outputs/reference_mode/pft14_arm_evidence_gap.json"


def _reference_run() -> Path:
    root = ROOT / "reference/OUT/orc_calibrate_250919_sen/arg2_1.0/001.0-071.0"
    runs = sorted(path.parent for path in root.rglob("driver_start.nc"))
    if not runs:
        raise FileNotFoundError(f"paper restart triplet not found below {root}")
    return runs[0]


def _equal(left: Any, right: Any) -> bool:
    try:
        a = np.asarray(left)
        b = np.asarray(right)
    except (TypeError, ValueError):
        return left == right
    if a.shape != b.shape:
        return False
    try:
        return bool(np.array_equal(a, b, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(a, b))


def _mapping_mismatches(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    mismatches = [f"missing_right:{name}" for name in left.keys() - right.keys()]
    mismatches.extend(f"missing_left:{name}" for name in right.keys() - left.keys())
    for name in left.keys() & right.keys():
        if isinstance(left[name], dict) and isinstance(right[name], dict):
            mismatches.extend(f"{name}.{item}" for item in _mapping_mismatches(left[name], right[name]))
        elif not _equal(left[name], right[name]):
            mismatches.append(name)
    return sorted(mismatches)


def _state_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    if hasattr(value, "_asdict"):
        return dict(value._asdict())
    raise TypeError(f"state object is not mapping-like: {type(value).__name__}")


def _leaf_count(value: Any) -> int:
    try:
        mapping = _state_mapping(value)
    except TypeError:
        return 1
    return sum(_leaf_count(item) for item in mapping.values())


def _object_mismatches(left: Any, right: Any) -> list[str]:
    return _mapping_mismatches(_state_mapping(left), _state_mapping(right))


def _state_mismatches(left: Any, right: Any) -> list[str]:
    mismatches = []
    if left.tstep != right.tstep:
        mismatches.append("tstep")
    mismatches.extend(_mapping_mismatches(left.fields_by_component, right.fields_by_component))
    return mismatches


def _packet_from_restart_readers(
    template: DriverPreviousStepStatePacket,
    *,
    driver: Any,
    sechiba: Any,
    stomate: Any,
) -> DriverPreviousStepStatePacket:
    """Rebuild persisted year-end state from the three production readers."""

    components = {
        component: dict(values)
        for component, values in template.fields_by_component.items()
    }
    driver_fields = {
        name: getattr(driver, name) for name in driver.__dataclass_fields__
    }
    # dim2_driver.f90 lines 1139-1152 restores the two persisted bands into
    # albedo(:,:,1:2) before the first intersurf call.
    driver_fields["albedo"] = np.stack(
        (driver_fields["albedo_vis"], driver_fields["albedo_nir"]), axis=1
    )
    components["driver_previous_step_state"] = driver_fields
    finalize = sechiba_finalize_source_state_from_restart(sechiba)
    components["sechiba_finalize_state"] = finalize
    for component in (
        "diffuco_previous_step_state",
        "enerbil_previous_step_state",
        "hydrol_previous_step_state",
        "thermosoil_previous_step_state",
    ):
        values = components[component]
        for name in values.keys() & finalize.keys():
            values[name] = finalize[name]

    slowproc = components["slowproc_stomate_previous_step_state"]
    for name in slowproc.keys() & finalize.keys():
        slowproc[name] = finalize[name]
    for group in (stomate.entry_state, stomate.season_state):
        slowproc.update(
            {
                name: value
                for name, value in group._asdict().items()
                if name != "provenance"
            }
        )
    slowproc["daily_accumulators"] = {
        name: value
        for name, value in stomate.daily_state._asdict().items()
        if name != "provenance"
    }
    return DriverPreviousStepStatePacket(
        tstep=template.tstep,
        fields_by_component=components,
        provenance_by_component={
            component: (
                *template.provenance_by_component.get(component, ()),
                "Stage 5 production restart readers reconstructed persisted year-handoff state",
            )
            for component in components
        },
    )


def _check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), **details}


def run(config: Path = CONFIG) -> dict[str, Any]:
    state_ledger = json.loads(STATE_LEDGER.read_text(encoding="utf-8"))
    restart_lifecycle = json.loads(RESTART_LIFECYCLE.read_text(encoding="utf-8"))
    restart_bundle = json.loads(RESTART_BUNDLE.read_text(encoding="utf-8"))
    production_lifecycle = json.loads(PRODUCTION_LIFECYCLE.read_text(encoding="utf-8"))
    production_hydrol_thermosoil = json.loads(
        PRODUCTION_HYDROL_THERMOSOIL.read_text(encoding="utf-8")
    )
    production_stomate = json.loads(PRODUCTION_STOMATE.read_text(encoding="utf-8"))
    owner_evidence = json.loads(OWNER_EVIDENCE.read_text(encoding="utf-8"))
    arm_evidence = json.loads(ARM_EVIDENCE.read_text(encoding="utf-8"))

    context = prepare_paper_1961_driver_context(config)
    day1 = paper_1961_driver_cold_start_day_scaffold(
        config, prepared_context=context, module_jit=True, diffuco_local_jit=True
    )
    if not day1.ready_for_first_day_end_state:
        raise RuntimeError(f"cold-start day failed: {day1.missing_components}")
    day2 = paper_1961_driver_later_day_scaffold(
        config,
        previous_state=day1.first_day_end_state,
        start_tstep=48,
        prepared_context=context,
        module_jit=True,
        diffuco_local_jit=True,
        retain_stomate_step_results=False,
        runtime_entry_payloads=True,
    )
    if not day2.ready_for_day_end_state:
        raise RuntimeError(f"Day2 failed: {day2.missing_components}")

    direct_1962 = paper_1961_driver_restart_year_start_day_result(
        config,
        previous_year_end_state=day2.day_end_state,
        year=1962,
        prepared_context=context,
        use_static_jit_daily_carbon=True,
        prebuild_day_payloads=True,
    )
    if not direct_1962.ready_for_day_end_state:
        raise RuntimeError(f"in-memory 1962 handoff failed: {direct_1962.missing_components}")

    template_run = _reference_run()
    base = context.first_step_restart_state
    bundle_state = paper_restart_bundle_state_from_day_end_packet(
        day2.day_end_state,
        base_stomate=base.stomate_readstart,
        kjit=96,
    )
    physical = PaperRestartBundlePhysicalState(
        driver=read_restart_physical_state(template_run / "driver_start.nc"),
        sechiba=read_restart_physical_state(template_run / "sechiba_start.nc"),
        stomate=read_restart_physical_state(template_run / "stomate_start.nc"),
    )
    with tempfile.TemporaryDirectory(prefix="orchjax_stage5_bundle_") as temporary:
        report = write_paper_restart_start_bundle(
            Path(temporary) / "1962_start",
            state=bundle_state,
            physical_state=physical,
        )
        restored = reference_case_first_step_restart_state(
            config,
            root=ROOT,
            run_dir=report.output_directory,
            stomate_filename="stomate_start.nc",
        )
        # The paper job carries the same STOMATE file under both producer and
        # next-year consumer names around the yearly rename boundary.
        shutil.copyfile(
            report.output_directory / "stomate_start.nc",
            report.output_directory / "stomate_restart.nc",
        )
        history = sorted(template_run.glob("stomate_history_*.nc"))[0]
        shutil.copyfile(history, report.output_directory / history.name)
        split_context = prepare_paper_1961_driver_context(
            config,
            reference_run_dir=report.output_directory,
        )
        restored_driver = read_dim2_driver_restart(
            report.output_directory / "driver_start.nc"
        )
        restored_packet = _packet_from_restart_readers(
            day2.day_end_state,
            driver=restored_driver,
            sechiba=restored.sechiba_restart_state,
            stomate=restored.stomate_readstart,
        )
        split_1962 = paper_1961_driver_restart_year_start_day_result(
            config,
            year=1962,
            previous_year_end_state=restored_packet,
            prepared_context=split_context,
            module_jit=True,
            diffuco_local_jit=True,
            retain_stomate_step_results=False,
            use_static_jit_daily_carbon=True,
        )

        driver_mismatches = _object_mismatches(bundle_state.driver, restored_driver)
        sechiba_mismatches = _mapping_mismatches(
            dict(bundle_state.sechiba.fields), dict(restored.sechiba_restart_state.fields)
        )
        stomate_groups = ("entry_state", "season_state", "daily_state", "gas_state", "remainder_state")
        stomate_mismatches = {
            group: _object_mismatches(
                getattr(bundle_state.stomate, group),
                getattr(restored.stomate_readstart, group),
            )
            for group in stomate_groups
        }
        stomate_mismatches = {
            group: mismatches for group, mismatches in stomate_mismatches.items() if mismatches
        }
        direct_modelout = dict(direct_1962.daily_modelout.modelout_fields)
        split_modelout = (
            {}
            if split_1962.daily_modelout is None
            else dict(split_1962.daily_modelout.modelout_fields)
        )
        modelout_mismatches = _mapping_mismatches(direct_modelout, split_modelout)
        split_state_mismatches = (
            ["split_day_end_missing"]
            if split_1962.day_end_state is None
            else _state_mismatches(direct_1962.day_end_state, split_1962.day_end_state)
        )
        checks = [
            _check(
                "source_driven_state_ledger_closed",
                state_ledger.get("closed") is True
                and state_ledger.get("all_components_closed") is True,
            ),
            _check("restart_lifecycle_181_fields_closed", restart_lifecycle.get("closed") is True, fields=restart_lifecycle.get("summary", {}).get("closed_fields")),
            _check("three_file_restart_contract_closed", restart_bundle.get("closed") is True),
            _check(
                "owner_region_evidence_260_of_260",
                owner_evidence.get("complete") is True
                and owner_evidence.get("counts", {}).get("required_owner_regions") == 260
                and owner_evidence.get("counts", {}).get("passed_owner_regions") == 260,
            ),
            _check(
                "scientific_arm_evidence_4580_of_4580",
                arm_evidence.get("complete") is True
                and arm_evidence.get("counts", {}).get("scientific_reachable_arms") == 4580
                and arm_evidence.get("counts", {}).get("unresolved_executable_evidence") == 0
                and arm_evidence.get("counts", {}).get("pending_evidence_items") == 0,
            ),
            _check(
                "hydrol_thermosoil_production_execution_passed",
                production_hydrol_thermosoil.get("status") == "passed"
                and production_hydrol_thermosoil.get("production_execution") is True,
                comparisons=len(production_hydrol_thermosoil.get("comparisons", [])),
            ),
            _check(
                "stomate_production_execution_passed",
                production_stomate.get("status") == "passed"
                and production_stomate.get("production_execution") is True,
                comparisons=len(production_stomate.get("comparisons", [])),
            ),
            _check("stage4_lifecycle_execution_passed", production_lifecycle.get("status") == "passed" and production_lifecycle.get("production_execution") is True),
            _check("cold_start_day1_complete", day1.ready_for_first_day_end_state),
            _check("model_state_day2_complete", day2.ready_for_day_end_state),
            _check("production_packet_has_13_driver_fields", len(bundle_state.driver.__dataclass_fields__) == 13),
            _check("production_packet_has_93_sechiba_fields", len(bundle_state.sechiba.fields) == 93),
            _check("three_restart_files_written", all(path.is_file() for path in report.output_paths)),
            _check(
                "driver_file_roundtrip_exact",
                not driver_mismatches,
                compared_fields=_leaf_count(bundle_state.driver),
                mismatches=driver_mismatches,
            ),
            _check(
                "sechiba_file_roundtrip_exact",
                not sechiba_mismatches,
                compared_fields=len(bundle_state.sechiba.fields),
                mismatches=sechiba_mismatches,
            ),
            _check(
                "stomate_file_roundtrip_exact",
                not stomate_mismatches,
                compared_fields=sum(
                    _leaf_count(getattr(bundle_state.stomate, group)) for group in stomate_groups
                ),
                mismatches=stomate_mismatches,
            ),
            _check(
                "split_1962_day1_complete",
                split_1962.ready_for_day_end_state,
                missing=list(split_1962.missing_components),
            ),
            _check("split_vs_memory_modelout_exact", not modelout_mismatches, mismatches=modelout_mismatches),
            _check("split_vs_memory_state_exact", not split_state_mismatches, mismatches=split_state_mismatches),
        ]

    return {
        "schema_version": 1,
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "production_execution": True,
        "scope": "cold start, cross-day state, three-file restart, and year-handoff equivalence",
        "execution": {
            "cold_start": "1961 Day1",
            "continuous_day": "1961 Day2",
            "handoff_target": "1962 Day1",
            "split_path": "production packet -> three NetCDF restart files -> production restart reader",
            "history_support": "immutable template stomate_history file; not a restart-state source",
            "trace_inputs": False,
            "server_used": False,
        },
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Stage 5 lifecycle and split-restart equivalence.")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = run(args.config)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {OUTPUT} status={result['status']} checks={len(result['checks'])}")
    return int(args.verify and result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
