from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.orchestration import (  # noqa: E402
    STOMATE_DAY_END_EXPLICIT_WRITEBACK_FIELDS,
    STOMATE_DAY_SEASON_STATE_FIELDS,
    STOMATE_DAY_STATE_PERSISTED_ENTRY_FIELDS,
    driver_year_handoff_state_gaps,
    paper_1961_driver_cold_start_day_scaffold,
    paper_1961_driver_later_day_scaffold,
    paper_1961_driver_restart_year_start_day_result,
    prepare_paper_1961_driver_context,
    rebase_driver_state_for_year_start,
)
from jax_orchidee.driver.restart_export import (  # noqa: E402
    stomate_restart_states_from_day_end_packet,
)
from jax_orchidee.stomate.modelout import (  # noqa: E402
    MODEL_OUTPUT_FIELD_NAMES,
    PAPER_MODEL_PFT14_INDEX,
    annual_history_mean_fields_from_daily_modelout,
    compute_modelout_from_fields,
    select_history_point_fields,
)
from jax_orchidee.stomate.reference import (  # noqa: E402
    read_stomate_daily_accumulator_state,
    read_stomate_restart_entry_state,
    read_stomate_restart_season_state,
)
from jax_orchidee.stomate.restart_io import (  # noqa: E402
    StomateRestartPhysicalState,
    read_stomate_readstart_states_from_template,
    write_stomate_full_writerestart_states,
)


CONFIG = ROOT / "configs/orchidee_man_250919.yaml"
LEDGER = ROOT / "docs/source_audits/pft14_reachable_ledger.yaml"
ORACLE_COVERAGE = ROOT / "outputs/reference_mode/fortran_oracle_coverage.json"
ORACLE_DIR = ROOT / "outputs/reference_mode/micro_oracles"
OUTPUT_DIR = ROOT / "outputs/reference_mode/production/lifecycle"
COMPARISON = OUTPUT_DIR / "comparison.json"
FRAGMENT = ROOT / "docs/source_audits/production_families/lifecycle.yaml"

ENTRIES = (
    "slowproc.active.daily_surface_writeback",
    "restart.active.day_and_year_handoff",
    "modelout.active.paper_history_formula",
    "sechiba.active.module_order",
)

CALLSITES = {
    "slowproc.active.daily_surface_writeback": (
        "jax_orchidee.driver.orchestration._paper_day_slowproc_surface_update",
        "jax_orchidee.driver.orchestration._paper_day_end_state_packet",
        "jax_orchidee.driver.orchestration.paper_1961_driver_cold_start_day_scaffold",
    ),
    "restart.active.day_and_year_handoff": (
        "jax_orchidee.driver.orchestration._paper_day_end_state_packet",
        "jax_orchidee.driver.orchestration.rebase_driver_state_for_year_start",
        "jax_orchidee.driver.orchestration.paper_1961_driver_restart_year_start_day_result",
        "jax_orchidee.driver.restart_export.stomate_restart_states_from_day_end_packet",
    ),
    "modelout.active.paper_history_formula": (
        "jax_orchidee.driver.orchestration._driver_modelout_from_outputs",
        "jax_orchidee.stomate.modelout.annual_history_mean_fields_from_daily_modelout",
    ),
    "sechiba.active.module_order": (
        "jax_orchidee.driver.orchestration.paper_1961_next_step_runtime_result",
        "jax_orchidee.driver.orchestration.paper_1961_driver_later_day_scaffold",
    ),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _span(dotted: str) -> dict[str, Any]:
    module, symbol = dotted.rsplit(".", 1)
    path = ROOT / (module.replace(".", "/") + ".py")
    data = path.read_bytes()
    tree = ast.parse(data.decode("utf-8"))
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == symbol
    ]
    if len(nodes) != 1:
        raise ValueError(f"expected one symbol {dotted}, found {len(nodes)}")
    node = nodes[0]
    lines = data.splitlines(keepends=True)
    return {
        "file": path.relative_to(ROOT).as_posix(),
        "symbol": symbol,
        "start_line": node.lineno,
        "end_line": node.end_lineno,
        "file_sha256": _sha256(data),
        "span_sha256": _sha256(b"".join(lines[node.lineno - 1 : node.end_lineno])),
    }


def _check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), **details}


def _equal(left: Any, right: Any) -> bool:
    try:
        a = np.asarray(left)
        b = np.asarray(right)
    except (TypeError, ValueError):
        return left == right
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


def _states_equal(expected: object, actual: object) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    for field in expected._fields:
        if field == "provenance" or field.startswith("read_input_"):
            continue
        if not _equal(getattr(expected, field), getattr(actual, field)):
            mismatches.append(field)
    return not mismatches, mismatches


def _template() -> Path:
    case = ROOT / "reference/OUT/orc_calibrate_250919_sen/arg2_1.0/001.0-071.0"
    paths = sorted(case.rglob("stomate_start.nc"))
    if not paths:
        raise FileNotFoundError(f"no stomate_start.nc below {case}")
    return paths[0]


def _physical(path: Path) -> StomateRestartPhysicalState:
    with Dataset(path) as dataset:
        return StomateRestartPhysicalState(
            nav_lon=np.asarray(dataset.variables["nav_lon"][:]),
            nav_lat=np.asarray(dataset.variables["nav_lat"][:]),
            nav_lev=np.asarray(dataset.variables["nav_lev"][:]),
            time=np.asarray(dataset.variables["time"][:]),
            time_steps=np.asarray(dataset.variables["time_steps"][:]),
        )


def _read_full_states(path: Path):
    entry = read_stomate_restart_entry_state(path)
    season = read_stomate_restart_season_state(path)
    daily = read_stomate_daily_accumulator_state(path)
    with Dataset(path) as dataset:
        return read_stomate_readstart_states_from_template(
            path,
            t2m=np.asarray(daily.t2m_daily),
            nvm=entry.age.shape[1],
            nslm=season.tsoil_month.shape[1],
            ndeep=entry.carbon_32l.shape[3],
            nsnow=int(dataset.variables["O2_snow"].shape[2]),
            nvert=int(dataset.variables["uo_0"].shape[1]),
            months_num=int(dataset.variables["fwet_series"].shape[1]),
            ncarb=entry.carbon.shape[1],
            nlitt=entry.litter.shape[1],
            nbpools=int(dataset.variables["MatrixV"].shape[1]),
        )


def _oracle_checks() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    coverage = json.loads(ORACLE_COVERAGE.read_text(encoding="utf-8"))
    by_entry = {
        record["ledger_entry"]: record
        for record in coverage["records"]
        if record.get("ledger_entry") in ENTRIES
    }
    checks: dict[str, Any] = {}
    for entry in ENTRIES:
        record = by_entry[entry]
        path = ORACLE_DIR / record["family"] / "comparison.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        checks[entry] = {
            "family": record["family"],
            "asset": path.relative_to(ROOT).as_posix(),
            "asset_sha256": _sha256(path.read_bytes()),
            "passed": record.get("passed") is True
            and result.get("status") == "passed"
            and bool(result.get("comparisons"))
            and all(item.get("passed") is True for item in result["comparisons"]),
        }
    return checks, by_entry


def _modelout_checks(day1: Any, day2: Any) -> list[dict[str, Any]]:
    daily = (dict(day1.stomate_outputs.modelout_fields), dict(day2.stomate_outputs.modelout_fields))
    checks: list[dict[str, Any]] = []
    for index, day in enumerate((day1, day2), start=1):
        expected = compute_modelout_from_fields(day.stomate_outputs.modelout_fields)
        actual = day.stomate_outputs.modelout
        checks.append(_check(f"day{index}.paper_formula", _equal(expected, actual)))

    history = annual_history_mean_fields_from_daily_modelout(daily)
    for name in MODEL_OUTPUT_FIELD_NAMES:
        manual = np.mean(np.stack([np.asarray(fields[name]) for fields in daily]), axis=0)
        actual = np.asarray(history[name])[0, :, 0, :].T
        checks.append(_check(f"history_average.{name}", _equal(manual, actual)))

    selected = select_history_point_fields(history)
    selected_modelout = compute_modelout_from_fields(selected)
    perturbed = {name: np.array(value, copy=True) for name, value in history.items()}
    for value in perturbed.values():
        value[:, 0, :, :] = 1.0e30
    selected_after_bare_change = compute_modelout_from_fields(select_history_point_fields(perturbed))
    checks.extend(
        (
            _check("pft14_index", PAPER_MODEL_PFT14_INDEX == 13),
            _check("bare_soil_mask_does_not_change_pft14", _equal(selected_modelout, selected_after_bare_change)),
            _check(
                "biomass_unit_factor_0p02",
                _equal(
                    selected_modelout.AGB_model,
                    0.02
                    * sum(
                        selected[name]
                        for name in (
                            "LEAF_M", "SAP_M_AB", "HEART_M_AB", "AGR_SAP_ST_M",
                            "AGR_HRT_ST_M", "AGR_SAP_PN_M", "AGR_HRT_PN_M",
                        )
                    ),
                ),
            ),
        )
    )
    return checks


def _restart_roundtrip(packet: Mapping[str, object]) -> list[dict[str, Any]]:
    template = _template()
    base = _read_full_states(template)
    entry, season, daily, merge = stomate_restart_states_from_day_end_packet(
        packet,
        base_entry=base.entry_state,
        base_season=base.season_state,
        base_daily=base.daily_state,
    )
    with tempfile.TemporaryDirectory(prefix="orchjax_stage4_restart_") as temporary:
        output = Path(temporary) / "stomate_restart.nc"
        report = write_stomate_full_writerestart_states(
            output,
            physical_state=_physical(template),
            entry_state=entry,
            season_state=season,
            daily_state=daily,
            gas_state=base.gas_state,
            remainder_state=base.remainder_state,
        )
        restored = _read_full_states(output)
        contracts = (
            ("entry", entry, restored.entry_state),
            ("season", season, restored.season_state),
            ("daily", daily, restored.daily_state),
            ("gas", base.gas_state, restored.gas_state),
            ("remainder", base.remainder_state, restored.remainder_state),
        )
        checks = []
        for name, expected, actual in contracts:
            passed, mismatches = _states_equal(expected, actual)
            checks.append(_check(f"restart_roundtrip.{name}", passed, mismatches=mismatches))
        checks.extend(
            (
                _check("restart_writer_field_count", len(report.written_fields) == 161, count=len(report.written_fields)),
                _check("restart_writer_unsupported_empty", report.unsupported_fields == ()),
                _check("restart_packet_entry_updates", bool(merge.entry_updates), count=len(merge.entry_updates)),
                _check("restart_packet_season_updates", bool(merge.season_updates), count=len(merge.season_updates)),
                _check("restart_packet_daily_updates", bool(merge.daily_updates), count=len(merge.daily_updates)),
            )
        )
        return checks


def run(config: Path = CONFIG) -> dict[str, Any]:
    oracle, _ = _oracle_checks()
    context = prepare_paper_1961_driver_context(config)
    day1 = paper_1961_driver_cold_start_day_scaffold(
        config,
        prepared_context=context,
        module_jit=True,
        diffuco_local_jit=True,
    )
    if not day1.ready_for_first_day_end_state:
        raise RuntimeError(f"cold-start day failed: {day1.missing_components}")
    day2 = paper_1961_driver_later_day_scaffold(
        config,
        previous_state=day1.first_day_end_state,
        start_tstep=day1.steps_per_stomate,
        prepared_context=context,
        module_jit=True,
        diffuco_local_jit=True,
        retain_stomate_step_results=False,
        runtime_entry_payloads=True,
    )
    if not day2.ready_for_day_end_state:
        raise RuntimeError(f"normal day failed: {day2.missing_components}")

    slow1 = day1.first_day_end_state.fields_by_component["slowproc_stomate_previous_step_state"]
    slow2 = day2.day_end_state.fields_by_component["slowproc_stomate_previous_step_state"]
    source1 = day1.first_day_stomate_state.fields
    persisted = (
        set(STOMATE_DAY_END_EXPLICIT_WRITEBACK_FIELDS)
        | set(STOMATE_DAY_STATE_PERSISTED_ENTRY_FIELDS)
        | set(STOMATE_DAY_SEASON_STATE_FIELDS)
    ) & set(source1)
    slowproc_checks = [
        _check("cold_start_day_completed", day1.completed_steps == day1.steps_per_stomate == 48),
        _check("normal_day_completed", len(day2.completed_entry_payloads) == day2.steps_per_stomate == 48),
        _check(
            "slow_day_boundary_reset_executed",
            day1.daily_reset_fields is not None
            and np.all(np.asarray(day1.daily_reset_fields["gpp_daily"]) == 0.0)
            and np.all(np.asarray(day1.daily_reset_fields["resp_maint_part"]) == 0.0),
        ),
        _check("day1_all_persisted_fields_written", all(_equal(slow1[name], source1[name]) for name in persisted), count=len(persisted)),
        _check("day2_consumes_day1_packet", day2.input_previous_state is day1.first_day_end_state),
        _check("day2_surface_state_present", all(name in slow2 for name in ("lai", "veget", "veget_max", "soiltile", "tot_bare_soil"))),
        _check(
            "day1_surface_mirrored_to_diffuco",
            all(
                _equal(day1.first_day_end_state.fields_by_component["diffuco_previous_step_state"][name], slow1[name])
                for name in ("lai", "veget", "veget_max", "tot_bare_soil")
            ),
        ),
    ]

    gaps = driver_year_handoff_state_gaps(day2.day_end_state)
    rebased = rebase_driver_state_for_year_start(day2.day_end_state)
    next_year = paper_1961_driver_restart_year_start_day_result(
        config,
        previous_year_end_state=day2.day_end_state,
        year=1962,
        prepared_context=context,
        use_static_jit_daily_carbon=True,
        prebuild_day_payloads=True,
    )
    restart_checks = [
        _check("year_handoff_has_no_state_gaps", gaps == (), count=len(gaps)),
        _check("year_counter_rebased", rebased.tstep == -1),
        _check("nonrestart_nroot_removed", "nroot" not in rebased.fields_by_component["hydrol_previous_step_state"]),
        _check("source_packet_not_mutated", "nroot" in day2.day_end_state.fields_by_component["hydrol_previous_step_state"]),
        _check("next_year_day1_completed", next_year.ready_for_day_end_state, missing=list(next_year.missing_components)),
        _check("next_year_forcing_counter_starts_zero", next_year.start_tstep == 0),
        _check("next_year_state_advances_48_steps", next_year.day_end_state is not None and next_year.day_end_state.tstep == 47),
    ]
    restart_checks.extend(_restart_roundtrip(slow2))

    final_entry = day2.completed_entry_payloads[-1]
    final_components = day2.previous_step_state.fields_by_component
    module_checks = [
        _check("cold_and_normal_production_steps_executed", day1.completed_steps == len(day2.completed_entry_payloads) == 48),
        _check("diffuco_output_present", all(name in final_entry for name in ("gpp", "veget", "veget_max"))),
        _check("enerbil_output_present", all(name in final_entry for name in ("temp_sol", "evapot_corr"))),
        _check("hydrol_output_present", all(name in final_entry for name in ("soil_mc", "wat_flux", "runoff_per_soil"))),
        _check("condveg_output_present", "diffuco_previous_step_state" in final_components),
        _check("thermosoil_output_present", all(name in final_entry for name in ("stempdiag", "tdeep", "hsdeep"))),
        _check(
            "all_sechiba_component_states_written",
            all(
                name in final_components
                for name in (
                    "diffuco_previous_step_state",
                    "enerbil_previous_step_state",
                    "hydrol_previous_step_state",
                    "thermosoil_previous_step_state",
                )
            ),
        ),
        _check("slowproc_day_boundary_present", day2.day_stomate_state is not None),
        _check("same_step_stomate_boundary_present", day2.ok_leak_boundary_inputs is not None),
    ]

    modelout_checks = _modelout_checks(day1, day2)
    comparisons = [
        {
            "name": "slowproc.active.daily_surface_writeback",
            "passed": all(item["passed"] for item in slowproc_checks),
            "checks": slowproc_checks,
        },
        {
            "name": "restart.active.day_and_year_handoff",
            "passed": all(item["passed"] for item in restart_checks),
            "checks": restart_checks,
        },
        {
            "name": "modelout.active.paper_history_formula",
            "passed": all(item["passed"] for item in modelout_checks),
            "checks": modelout_checks,
        },
        {
            "name": "sechiba.active.module_order",
            "passed": all(item["passed"] for item in module_checks),
            "checks": module_checks,
        },
    ]
    for item in comparisons:
        item["stage3_oracle"] = oracle[item["name"]]
        item["passed"] = item["passed"] and oracle[item["name"]]["passed"]
    return {
        "schema_version": 1,
        "status": "passed" if all(item["passed"] for item in comparisons) else "failed",
        "production_execution": True,
        "execution": {
            "config": config.relative_to(ROOT).as_posix(),
            "cold_start_day": {"year": 1961, "day": 1, "steps": 48},
            "normal_day": {"year": 1961, "day": 2, "steps": 48},
            "year_handoff_day": {"source_year": 1961, "target_year": 1962, "steps": 48},
            "restart_netcdf_roundtrip": True,
            "server_used": False,
            "trace_inputs": False,
        },
        "comparisons": comparisons,
    }


def write_manifest() -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    by_entry = {entry["id"]: entry for entry in ledger["entries"]}
    coverage = json.loads(ORACLE_COVERAGE.read_text(encoding="utf-8"))
    oracle = {
        record["ledger_entry"]: record
        for record in coverage["records"]
        if record.get("ledger_entry") in ENTRIES and record.get("passed") is True
    }
    records = []
    for entry_id in ENTRIES:
        entry = by_entry[entry_id]
        state = list(entry["same_step_outputs"])
        if entry_id == "modelout.active.paper_history_formula":
            state = []
        records.append(
            {
                "ledger_entry": entry_id,
                "oracle_family": oracle[entry_id]["family"],
                "status": "verified",
                "production_owner": _span(entry["kernel"]),
                "callsites": [_span(name) for name in CALLSITES[entry_id]],
                "contracts": {
                    "inputs": list(entry["state_inputs"]),
                    "outputs": list(entry["same_step_outputs"]),
                    "state_writeback": state,
                    **(
                        {"no_state_writeback_reason": "Modelout is diagnostic and does not modify prognostic state."}
                        if not state
                        else {}
                    ),
                },
                "oracle_case_ids": ["all_stage3_branch_cases"],
                "comparison_asset": COMPARISON.relative_to(ROOT).as_posix(),
            }
        )
    FRAGMENT.parent.mkdir(parents=True, exist_ok=True)
    FRAGMENT.write_text(
        yaml.safe_dump({"schema_version": 1, "records": records}, sort_keys=False),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Stage 4 lifecycle production wiring.")
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    result = run(args.config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if result["status"] != "passed":
        print(json.dumps(result, indent=2))
        return 1
    write_manifest()
    print(f"WROTE {COMPARISON} status=passed entries={len(ENTRIES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
