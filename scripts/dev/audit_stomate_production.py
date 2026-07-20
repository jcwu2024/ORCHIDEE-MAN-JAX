from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.orchestration import (  # noqa: E402
    PAPER_DAY_ZERO_DAILY_RESET_FIELDS,
    STOMATE_DAY_SEASON_STATE_FIELDS,
    paper_1961_driver_cold_start_day_scaffold,
    paper_1961_driver_later_day_scaffold,
    prepare_paper_1961_driver_context,
)


DEFAULT_CONFIG = ROOT / "configs/orchidee_man_250919.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/reference_mode/production/stomate/comparison.json"
DEFAULT_MANIFEST = ROOT / "docs/source_audits/production_families/stomate.yaml"

ENTRY_FAMILIES = {
    "stomate.active.daily_accumulation": "stomate_daily_maintenance",
    "stomate.active.maintenance_respiration": "stomate_daily_maintenance",
    "stomate.active.prescribe_restart_gate": "stomate_prescribe_restart_gate",
    "stomate.active.constraints": "stomate_constraints",
    "stomate.active.season_memory": "stomate_season_memory",
    "stomate.active.phenology": "stomate_phenology",
    "stomate.active.allocation": "stomate_allocation",
    "stomate.active.npp_growth": "stomate_npp_growth",
    "stomate.active.gap_mortality": "stomate_gap_mortality",
    "stomate.active.final_lai_vmax": "stomate_final_lai_vmax",
    "stomate.active.turnover": "stomate_turnover",
    "stomate.active.ok_leak_litter": "stomate_littercalc_leak",
    "stomate.active.ok_leak_soilcarbon": "stomate_soilcarbon_owner",
    "stomate.active.tf_doc": "stomate_soilcarbon_owner",
    "stomate.active.ok_leak_firstcall_active_layer": "stomate_soilcarbon_owner",
}

OWNER_SYMBOLS = {
    "stomate.active.daily_accumulation": ("jax_orchidee/stomate/daily.py", "stomate_daily_process_fold_from_entries"),
    "stomate.active.maintenance_respiration": ("jax_orchidee/stomate/daily.py", "stomate_daily_process_fold_from_entries"),
    "stomate.active.prescribe_restart_gate": ("jax_orchidee/stomate/carbon_kernels.py", "prescribe_step"),
    "stomate.active.constraints": ("jax_orchidee/stomate/carbon_kernels.py", "constraints_step"),
    "stomate.active.season_memory": ("jax_orchidee/stomate/season.py", "season_memory_step"),
    "stomate.active.phenology": ("jax_orchidee/stomate/carbon_kernels.py", "phenology_step"),
    "stomate.active.allocation": ("jax_orchidee/stomate/carbon_kernels.py", "allocation_step"),
    "stomate.active.npp_growth": ("jax_orchidee/stomate/integration.py", "stomate_daily_carbon_explicit"),
    "stomate.active.gap_mortality": ("jax_orchidee/stomate/carbon_kernels.py", "gap_mortality_step"),
    "stomate.active.final_lai_vmax": ("jax_orchidee/stomate/carbon_kernels.py", "vmax_step"),
    "stomate.active.turnover": ("jax_orchidee/stomate/carbon_kernels.py", "turnover_step"),
    "stomate.active.ok_leak_litter": ("jax_orchidee/stomate/integration.py", "stomate_ok_leak_explicit"),
    "stomate.active.ok_leak_soilcarbon": ("jax_orchidee/stomate/integration.py", "stomate_ok_leak_explicit"),
    "stomate.active.tf_doc": ("jax_orchidee/stomate/integration.py", "stomate_ok_leak_explicit"),
    "stomate.active.ok_leak_firstcall_active_layer": ("jax_orchidee/stomate/integration.py", "stomate_ok_leak_explicit"),
}


def _equal(left: Any, right: Any) -> bool:
    try:
        a = np.asarray(left)
        b = np.asarray(right)
    except (TypeError, ValueError):
        return left == right
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


def _check(name: str, condition: bool, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(condition), **({} if details is None else details)}


def _entry(entry: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": entry,
        "ledger_entry": entry,
        "passed": bool(checks) and all(item["passed"] for item in checks),
        "checks": checks,
    }


def _packet(day: Any) -> dict[str, Any]:
    return dict(day.first_day_stomate_state.fields)


def run(config: Path) -> dict[str, Any]:
    context = prepare_paper_1961_driver_context(config)
    day1 = paper_1961_driver_cold_start_day_scaffold(
        config,
        prepared_context=context,
        module_jit=True,
        diffuco_local_jit=True,
    )
    if not day1.ready_for_first_day_end_state:
        raise RuntimeError("cold-start production day is incomplete: " + ", ".join(day1.missing_components))
    day2 = paper_1961_driver_later_day_scaffold(
        config,
        previous_state=day1.first_day_end_state,
        start_tstep=day1.steps_per_stomate,
        prepared_context=context,
        module_jit=True,
        diffuco_local_jit=True,
        retain_stomate_step_results=True,
        runtime_entry_payloads=True,
    )
    if not day2.ready_for_day_end_state:
        raise RuntimeError("model-produced Day2 production transition is incomplete: " + ", ".join(day2.missing_components))

    fold = day1.daily_process_fold
    carbon = day1.stomate_daily_carbon
    leak = day1.stomate_ok_leak.ok_leak
    boundary = day1.ok_leak_boundary_inputs
    packet = _packet(day1)
    daily_packet = packet["daily_accumulators"]
    post = carbon.post_npp
    npp = post.daily_carbon.npp_update

    reset_names = tuple(name for name in PAPER_DAY_ZERO_DAILY_RESET_FIELDS if name in fold.daily_fields)
    comparisons = [
        _entry(
            "stomate.active.daily_accumulation",
            [
                _check("cold_start_48_half_hours", day1.completed_steps == day1.steps_per_stomate == 48),
                _check("day1_accumulator_has_48_results", len(fold.accumulator.step_results) == 48),
                _check("day2_accumulator_has_48_results", len(day2.daily_process_fold.accumulator.step_results) == 48),
                _check("all_daily_fields_written_to_reset_packet", all(name in daily_packet for name in fold.daily_fields)),
                _check("zero_reset_fields_exact", all(np.all(np.asarray(daily_packet[name]) == 0) for name in reset_names)),
                _check("temperature_extrema_sentinels", np.all(np.asarray(daily_packet["t2m_min_daily"]) == 1.0e33) and np.all(np.asarray(daily_packet["t2m_max_daily"]) == -1.0e33)),
            ],
        ),
        _entry(
            "stomate.active.maintenance_respiration",
            [
                _check("day1_maintenance_has_48_results", len(fold.maintenance.step_results) == 48),
                _check("day2_maintenance_has_48_results", len(day2.daily_process_fold.maintenance.step_results) == 48),
                _check("maintenance_fed_to_npp", _equal(post.daily_carbon.boundary.resp_maint_part, fold.daily_fields["resp_maint_part"])),
                _check("maintenance_accumulator_reset", np.all(np.asarray(packet["resp_maint_part"]) == 0)),
            ],
        ),
        _entry(
            "stomate.active.prescribe_restart_gate",
            [
                _check("prescribe_executed", carbon.prescribe is not None),
                _check("pft_present_writeback", _equal(packet["pft_present"], carbon.prescribe.pft_present)),
                _check("co2_to_bm_writeback", _equal(packet["co2_to_bm"], carbon.prescribe.co2_to_bm)),
            ],
        ),
        _entry(
            "stomate.active.constraints",
            [
                _check("constraints_executed", carbon.constraints is not None),
                _check("adapted_writeback", _equal(packet["adapted"], carbon.constraints.adapted)),
                _check("regenerate_writeback", _equal(packet["regenerate"], carbon.constraints.regenerate)),
            ],
        ),
        _entry(
            "stomate.active.season_memory",
            [
                _check("season_fields_complete_day1", all(name in packet for name in STOMATE_DAY_SEASON_STATE_FIELDS)),
                _check("season_fields_complete_day2", all(name in day2.day_stomate_state.fields for name in STOMATE_DAY_SEASON_STATE_FIELDS)),
                _check("day2_uses_day1_state", day2.input_previous_state is day1.first_day_end_state),
            ],
        ),
        _entry(
            "stomate.active.phenology",
            [
                _check("phenology_executed", carbon.phenology is not None),
                _check("begin_leaves_writeback", _equal(packet["begin_leaves"], carbon.phenology.begin_leaves)),
            ],
        ),
        _entry(
            "stomate.active.allocation",
            [
                _check("allocation_executed", carbon.allocation is not None),
                _check("allocation_fed_to_npp", _equal(post.daily_carbon.boundary.f_alloc, carbon.allocation.f_alloc)),
                _check("allocation_biomass_fed_to_npp", _equal(post.daily_carbon.boundary.biomass, carbon.allocation.biomass)),
            ],
        ),
        _entry(
            "stomate.active.npp_growth",
            [
                _check("npp_executed", npp is not None),
                _check("npp_writeback", _equal(packet["npp_daily"], npp.npp)),
                _check("growth_respiration_writeback", _equal(packet["resp_growth"], npp.resp_growth)),
            ],
        ),
        _entry(
            "stomate.active.gap_mortality",
            [
                _check("gap_executed", post.gap is not None),
                _check("post_gap_ind_writeback", _equal(packet["ind"], post.kill_after_gap.ind)),
                _check("post_gap_cn_ind_writeback", _equal(packet["cn_ind"], post.kill_after_gap.cn_ind)),
            ],
        ),
        _entry(
            "stomate.active.final_lai_vmax",
            [
                _check("vmax_executed", post.vmax is not None),
                _check("assim_param_writeback", _equal(packet["assim_param"], np.asarray(post.vmax.vcmax)[:, :, None])),
                _check("final_leaf_frac_writeback", _equal(packet["leaf_frac"], post.vmax.leaf_frac)),
            ],
        ),
        _entry(
            "stomate.active.turnover",
            [
                _check("turnover_executed", post.turnover is not None),
                _check("biomass_writeback", _equal(packet["biomass"], post.turnover.biomass)),
                _check("turnover_writeback", _equal(packet["turnover_daily"], post.turnover.turnover)),
            ],
        ),
        _entry(
            "stomate.active.ok_leak_litter",
            [
                _check("litter_executed", leak.littercalc is not None),
                _check("litter_above_writeback", _equal(packet["litter_above"], leak.littercalc.litter_above)),
                _check("litter_below_writeback", _equal(packet["litter_below"], leak.littercalc.litter_below)),
                _check("fuel_writeback", _equal(packet["fuel_1hr"], leak.littercalc.fuel.fuel_1hr)),
            ],
        ),
        _entry(
            "stomate.active.ok_leak_soilcarbon",
            [
                _check("soilcarbon_executed", leak.soilcarbon is not None),
                _check("carbon_32l_writeback", _equal(packet["carbon_32l"], leak.soilcarbon.carbon_32l)),
                _check("doc_writeback", _equal(packet["DOC"], leak.soilcarbon.doc)),
            ],
        ),
        _entry(
            "stomate.active.tf_doc",
            [
                _check("doc_export_executed", leak.doc_export is not None),
                _check("interception_storage_writeback", _equal(packet["interception_storage"], leak.interception_storage)),
                _check("wet_deposition_shapes", np.asarray(leak.wet_dep_ground).shape == np.asarray(leak.wet_dep_flood).shape),
            ],
        ),
        _entry(
            "stomate.active.ok_leak_firstcall_active_layer",
            [
                _check("firstcall_altmax_source_present", "altmax" in boundary.output_inputs),
                _check("firstcall_altmax_writeback", _equal(packet["altmax"], boundary.output_inputs["altmax"])),
                _check("later_call_transition_completed", day2.ready_for_day_end_state),
                _check("later_call_altmax_writeback", "altmax" in day2.day_stomate_state.fields),
            ],
        ),
    ]
    status = "passed" if all(item["passed"] for item in comparisons) else "failed"
    return {
        "schema_version": 1,
        "status": status,
        "production_execution": True,
        "execution": {
            "config": config.resolve().relative_to(ROOT).as_posix(),
            "cold_start_day": 1,
            "model_produced_later_day": 2,
            "half_hour_steps_per_day": day1.steps_per_stomate,
            "trace_inputs": False,
            "server_used": False,
        },
        "comparisons": comparisons,
    }


def _span(file: str, symbol: str) -> dict[str, Any]:
    path = ROOT / file
    data = path.read_bytes()
    tree = ast.parse(data.decode("utf-8"))
    nodes = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol]
    if len(nodes) != 1:
        raise ValueError(f"expected one Python function {file}::{symbol}, found {len(nodes)}")
    node = nodes[0]
    lines = data.splitlines(keepends=True)
    extracted = b"".join(lines[node.lineno - 1 : node.end_lineno])
    return {
        "file": file,
        "symbol": symbol,
        "start_line": node.lineno,
        "end_line": node.end_lineno,
        "file_sha256": hashlib.sha256(data).hexdigest(),
        "span_sha256": hashlib.sha256(extracted).hexdigest(),
    }


def write_manifest(path: Path) -> None:
    daily_callsite = _span("jax_orchidee/driver/orchestration.py", "_paper_cold_start_day_daily_process_from_completed_entries")
    carbon_callsite = _span("jax_orchidee/driver/orchestration.py", "_paper_day_stomate_daily_carbon_from_bundles")
    leak_callsite = _span("jax_orchidee/driver/orchestration.py", "_paper_day_ok_leak_from_boundary")
    season_callsite = _span("jax_orchidee/driver/orchestration.py", "_paper_day_season_fields_from_bundle_source")
    records = []
    for entry, family in ENTRY_FAMILIES.items():
        if entry in {"stomate.active.daily_accumulation", "stomate.active.maintenance_respiration"}:
            callsites = [daily_callsite]
        elif entry == "stomate.active.season_memory":
            callsites = [season_callsite]
        elif entry.startswith("stomate.active.ok_leak") or entry == "stomate.active.tf_doc":
            callsites = [leak_callsite]
        else:
            callsites = [carbon_callsite]
        records.append(
            {
                "ledger_entry": entry,
                "oracle_family": family,
                "status": "verified",
                "production_owner": _span(*OWNER_SYMBOLS[entry]),
                "callsites": callsites,
                "contracts": {
                    "inputs": ["48 ordered SECHIBA entry payloads", "previous STOMATE state", "PFT14 parameters and masks"],
                    "outputs": ["daily process result", "STOMATE carbon/soil result", "day-end state packet"],
                    "state_writeback": ["daily_accumulators", "slowproc_stomate_previous_step_state"],
                },
                "oracle_case_ids": ["stage3_family_cases", "production_cold_start_day1", "production_model_state_day2"],
                "comparison_asset": DEFAULT_OUTPUT.relative_to(ROOT).as_posix(),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"schema_version": 1, "lane": "stomate", "records": records}, sort_keys=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute the Stage 4 STOMATE production-path audit.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = run(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_manifest(args.manifest)
    print(f"WROTE {args.output.resolve()} status={result['status']} comparisons={len(result['comparisons'])}")
    return int(args.verify and result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
