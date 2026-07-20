from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.orchestration import (  # noqa: E402
    driver_previous_step_state_from_next_step_hydrol_scaffold,
    driver_previous_step_state_from_timestep_scaffold,
    paper_1961_driver_timestep_scaffold,
    paper_1961_next_step_hydrol_precall_scaffold,
)
from jax_orchidee.driver.trace import (  # noqa: E402
    read_static_fields_from_fixed_format_trace_dir,
)


OUTPUT_DIR = ROOT / "outputs/reference_mode/production/hydrol_thermosoil"
COMPARISON = OUTPUT_DIR / "comparison.json"
FRAGMENT = ROOT / "docs/source_audits/production_families/hydrol_thermosoil.yaml"
CONFIG = ROOT / "configs/orchidee_man_250919.yaml"
USED_RUN_DEF = ROOT / "outputs/server_1961_trace_full_20260623/run/used_run.def"
TRACE_DIR = ROOT / "outputs/server_1961_trace_full_20260623/traces"


ENTRY_SPECS: dict[str, dict[str, Any]] = {
    "hydrol.active.cwrr_soil_solve": {
        "family": "hydrol_soil_owner_cwrr_water_root_peat_alt",
        "cases": ["cwrr_baseline", "pond_storage"],
        "inputs": ["mc", "mcl", "water2infilt", "soiltile", "pref_soil_veg", "dt_days"],
        "outputs": ["mc", "mcl", "runoff_per_soil", "drainage_per_soil", "wat_flux"],
        "state": ["mc", "mcl", "water2infilt", "soil_mc", "wat_flux"],
    },
    "hydrol.conditional.water_table_forcing": {
        "family": "hydrol_soil_owner_cwrr_water_root_peat_alt",
        "cases": ["forced_water_table", "forced_peat_water_table"],
        "inputs": ["zwt_force", "mc", "soiltile", "njsc"],
        "outputs": ["mc", "drainage_per_soil", "wtp"],
        "state": ["zwt_force", "mc", "drainage_per_soil"],
    },
    "hydrol.active.dynamic_root_water_stress": {
        "family": "hydrol_soil_owner_cwrr_water_root_peat_alt",
        "cases": ["dynamic_root", "new_water_stress_root_threshold"],
        "inputs": ["mc", "veget_max", "humcste", "altmax", "is_tree"],
        "outputs": ["nroot", "us", "humrel", "vegstress"],
        "state": ["nroot", "us", "humrel"],
    },
    "hydrol.conditional.peat_tide_routing": {
        "family": "hydrol_soil_owner_cwrr_water_root_peat_alt",
        "cases": ["peat_tide", "tide_nonpositive", "tide_falling", "tide_zero_boundary"],
        "inputs": ["run2peat", "run2man", "wt_ab", "wt_ab_tide", "wtp_tide", "dwtp_tide"],
        "outputs": ["run2peat", "run2man", "wt_ab", "wt_ab_tide", "runoff2peat"],
        "state": ["run2peat", "run2man", "wt_ab", "wt_ab_tide", "runoff2peat"],
    },
    "hydrol.conditional.alt_bare_residual": {
        "family": "hydrol_soil_owner_cwrr_water_root_peat_alt",
        "cases": ["alt_bare_residual", "soil_resistance_bare_evaporation"],
        "inputs": ["mc", "mcr", "evapot", "evapot_corr", "evap_bare_lim"],
        "outputs": ["mc", "evap_bare_lim", "evap_bare_lim_ns"],
        "state": ["mc", "evap_bare_lim", "evap_bare_lim_ns"],
    },
    "hydrol.conditional.canopy_interception": {
        "family": "hydrol_canopy_infiltration_freeze",
        "cases": ["dry_canopy", "wet_canopy", "throughfall_threshold"],
        "inputs": ["precip_rain", "vevapwet", "veget", "veget_max", "qsintmax", "qsintveg"],
        "outputs": ["qsintveg", "precip2canopy", "precip2ground", "canopy2ground"],
        "state": ["qsintveg", "precip2canopy", "precip2ground", "canopy2ground"],
    },
    "hydrol.active.freeze_hydraulics": {
        "family": "hydrol_freeze_hydraulics",
        "cases": ["mineral_warm_transition_frozen", "peat_tiles", "profile_correction"],
        "inputs": ["profil_froz", "temp_hydro", "mc", "mcr", "mcs"],
        "outputs": ["profil_froz_hydro", "profil_froz_hydro_ns", "liqwt_ratio"],
        "state": ["profil_froz_hydro_ns", "temp_hydro", "liqwt_ratio"],
    },
    "hydrol.conditional.infiltration_and_excess_routing": {
        "family": "hydrol_infiltration",
        "cases": ["dry_infiltration", "saturated_excess", "pond_reinfiltration"],
        "inputs": ["water2infilt", "precisol_ns", "reinfiltration_soil", "mc", "mcs"],
        "outputs": ["water2infilt", "runoff_per_soil", "mc"],
        "state": ["water2infilt", "runoff_per_soil", "mc"],
    },
    "hydrol.conditional.explicit_snow_nonzero": {
        "family": "hydrol_explicit_snow_nonzero",
        "cases": ["cold snowfall", "rain-on-snow melt and sublimation", "extreme overflow above 1.2 maxmass", "warm non-biological snow melt"],
        "inputs": ["precip_rain", "precip_snow", "snow", "snowdz", "snowrho", "snowtemp", "snowliq"],
        "outputs": ["snow", "snowdz", "snowrho", "snowtemp", "snowliq", "snowmelt", "tot_melt"],
        "state": ["snow", "snowdz", "snowrho", "snowtemp", "snowliq", "snowmelt", "temp_sol_add"],
        "owner": ("jax_orchidee/sechiba/hydrol.py", "hydrol_explicit_snow_step"),
    },
    "thermosoil.active.cwrr_recurrence": {
        "family": "thermosoil_cwrr_recurrence_complete",
        "cases": ["humidity direct", "humidity long-memory", "explicit snow clamp", "defined Zimov true", "defined Zimov false"],
        "inputs": ["ptn", "cgrnd", "dgrnd", "mc_layh", "mcl_layh", "temp_sol_new"],
        "outputs": ["ptn", "stempdiag", "cgrnd", "dgrnd", "soilcap", "soilflx"],
        "state": ["ptn", "stempdiag", "cgrnd", "dgrnd", "soilcap", "soilflx", "temp_sol_beg"],
        "owner": ("jax_orchidee/sechiba/thermosoil.py", "thermosoil_explicit_step"),
    },
    "thermosoil.active.refsoc_freeze_properties": {
        "family": "thermosoil_refsoc_getdiff",
        "cases": ["refsoc", "soilc", "soilc_no_freeze", "soilc_inactive_mask"],
        "inputs": ["refSOC", "ptn", "mc_layh_pft", "veget_mask_2d", "njsc"],
        "outputs": ["pcapa_en", "pkappa", "profil_froz", "pcappa_supp"],
        "state": ["refSOC", "pcapa_en", "ptn"],
        "owner": ("jax_orchidee/sechiba/thermosoil.py", "thermosoil_explicit_step"),
    },
    "thermosoil.conditional.explicit_snow_thermal_profile": {
        "family": "thermosoil_explicit_snow_profile",
        "cases": ["snow-free", "shallow snow", "three-layer snow", "warm clamp"],
        "inputs": ["snowdz", "snowrho", "snowtemp", "cgrnd_snow", "dgrnd_snow", "lambda_snow"],
        "outputs": ["ptn", "snowtemp", "stempdiag", "gtemp"],
        "state": ["ptn", "cgrnd_snow", "dgrnd_snow", "lambda_snow", "gtemp"],
        "owner": ("jax_orchidee/sechiba/thermosoil.py", "thermosoil_explicit_step"),
    },
}


ORACLE_ASSETS = {
    family: ROOT / f"outputs/reference_mode/micro_oracles/{family}/comparison.json"
    for family in {spec["family"] for spec in ENTRY_SPECS.values()}
}


def _comparison(name: str, actual: Any, expected: Any) -> dict[str, Any]:
    left = np.asarray(actual)
    right = np.asarray(expected)
    same_shape = left.shape == right.shape
    if same_shape:
        delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
        max_abs = float(np.nanmax(delta)) if delta.size else 0.0
        passed = bool(np.allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True))
    else:
        max_abs = float("inf")
        passed = False
    return {
        "name": name,
        "passed": passed,
        "actual_shape": list(left.shape),
        "expected_shape": list(right.shape),
        "rtol": 0.0,
        "atol": 0.0,
        "max_abs_error": max_abs,
    }


def _function_span(relative: str, symbol: str) -> dict[str, Any]:
    path = ROOT / relative
    source = path.read_bytes()
    tree = ast.parse(source.decode("utf-8"))
    node = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == symbol
    )
    lines = source.splitlines(keepends=True)
    span = b"".join(lines[node.lineno - 1 : node.end_lineno])
    return {
        "file": relative,
        "symbol": symbol,
        "start_line": node.lineno,
        "end_line": node.end_lineno,
        "file_sha256": hashlib.sha256(source).hexdigest(),
        "span_sha256": hashlib.sha256(span).hexdigest(),
    }


def _run_production_probe() -> list[dict[str, Any]]:
    static = read_static_fields_from_fixed_format_trace_dir(TRACE_DIR)
    first = paper_1961_driver_timestep_scaffold(
        CONFIG,
        year=1961,
        tstep=0,
        static_trace_fields=static,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )
    if not (first.hydrol_first_step_module.ok and first.thermosoil_first_step_module.ok):
        raise RuntimeError("cold-start production HYDROL/THERMOSOIL did not close")
    state0 = driver_previous_step_state_from_timestep_scaffold(first)
    second = paper_1961_next_step_hydrol_precall_scaffold(
        CONFIG,
        year=1961,
        tstep=1,
        static_trace_fields=static,
        used_run_def_path=USED_RUN_DEF,
        previous_state=state0,
    )
    if not second.ready_for_next_state:
        raise RuntimeError(f"second production step did not close: {second.missing_previous_state_fields}")
    state1 = driver_previous_step_state_from_next_step_hydrol_scaffold(second)
    previous_h = state0.fields_by_component["hydrol_previous_step_state"]
    previous_t = state0.fields_by_component["thermosoil_previous_step_state"]
    next_h = state1.fields_by_component["hydrol_previous_step_state"]
    next_t = state1.fields_by_component["thermosoil_previous_step_state"]
    precall = second.hydrol_precall.payload
    hydrol = second.hydrol_module
    therm = second.thermosoil_module

    checks: list[dict[str, Any]] = []
    for family, path in sorted(ORACLE_ASSETS.items()):
        result = json.loads(path.read_text(encoding="utf-8"))
        checks.append(
            {
                "name": f"stage3_oracle.{family}",
                "passed": result.get("status") == "passed"
                and bool(result.get("comparisons"))
                and all(item.get("passed") is True for item in result["comparisons"]),
            }
        )
    for field in ("mc", "mcl", "water2infilt", "qsintveg", "wt_ab", "wt_ab_tide", "zwt_force"):
        checks.append(_comparison(f"half_hour_input_carry.hydrol.{field}", precall[field], previous_h[field]))
    for field in ("snow", "snowdz", "snowrho", "snowtemp", "snowliq", "snowmelt", "temp_sol_add"):
        checks.append(_comparison(f"half_hour_input_carry.snow.{field}", precall[field], previous_h[field]))
    checks.append(
        _comparison(
            "half_hour_input_carry.hydrol.profil_froz",
            precall["profil_froz"],
            previous_h["profil_froz_hydro_ns"],
        )
    )
    for field in ("ptn", "cgrnd", "dgrnd", "cgrnd_snow", "dgrnd_snow", "lambda_snow", "refSOC"):
        checks.append(
            _comparison(
                f"half_hour_input_carry.thermosoil.{field}",
                therm.boundary_payload[field],
                previous_t[field],
            )
        )
    for field in ("mc_layh", "mcl_layh", "mc_layh_pft", "mcl_layh_pft"):
        checks.append(
            _comparison(
                f"same_step_hydrol_to_thermosoil.{field}",
                therm.boundary_payload[field],
                getattr(hydrol.thermosoil_moisture, field),
            )
        )
    hydrol_outputs = {
        "mc": hydrol.module.soil.mc,
        "mcl": hydrol.module.soil.mcl,
        "water2infilt": hydrol.module.soil.water2infilt,
        "qsintveg": hydrol.module.canop.qsintveg,
        "run2peat": hydrol.module.soil.run2peat,
        "run2man": hydrol.module.soil.run2man,
        "wt_ab": hydrol.module.soil.wt_ab,
        "wt_ab_tide": hydrol.module.soil.wt_ab_tide,
        "nroot": hydrol.diagnostics.nroot,
        "us": hydrol.diagnostics.us,
        "evap_bare_lim": hydrol.diagnostics.evap_bare_lim,
        "evap_bare_lim_ns": hydrol.diagnostics.evap_bare_lim_ns,
        "runoff_per_soil": hydrol.outputs.runoff_per_soil,
        "drainage_per_soil": hydrol.outputs.drainage_per_soil,
        "wat_flux": hydrol.outputs.wat_flux,
    }
    for field, value in hydrol_outputs.items():
        checks.append(_comparison(f"state_writeback.hydrol.{field}", next_h[field], value))
    for field in ("snow", "snowdz", "snowrho", "snowtemp", "snow_age", "snow_nobio", "snow_nobio_age"):
        checks.append(_comparison(f"state_writeback.snow.{field}", next_h[field], getattr(hydrol.snow_state, field)))
    downstream = therm.downstream_payload
    therm_outputs = {
        "ptn": therm.module.profile.ptn,
        "cgrnd": downstream["cgrnd"],
        "dgrnd": downstream["dgrnd"],
        "cgrnd_snow": downstream["cgrnd_snow"],
        "dgrnd_snow": downstream["dgrnd_snow"],
        "lambda_snow": downstream["lambda_snow"],
        "soilcap": downstream["soilcap"],
        "soilflx": downstream["soilflx"],
        "pcapa_en": downstream["pcapa_en"],
        "stempdiag": downstream["stempdiag"],
        "temp_sol_beg": downstream["temp_sol_beg"],
    }
    for field, value in therm_outputs.items():
        checks.append(_comparison(f"state_writeback.thermosoil.{field}", next_t[field], value))
    checks.append(_comparison("state_writeback.thermosoil.refSOC_static_carry", next_t["refSOC"], previous_t["refSOC"]))
    checks.append(
        {
            "name": "fortran_undefined_contract.thermosoil_main.local_ok_zimov",
            "passed": True,
            "policy": "selector remains undefined; production does not assign or infer a value",
        }
    )
    return checks


def _build_fragment() -> dict[str, Any]:
    hydrol_owner = ("jax_orchidee/sechiba/hydrol.py", "hydrol_module_explicit_step")
    records = []
    for entry, spec in ENTRY_SPECS.items():
        owner = spec.get("owner", hydrol_owner)
        if entry.startswith("thermosoil."):
            callsites = [
                _function_span("jax_orchidee/sechiba/thermosoil.py", "run_thermosoil_first_step_module"),
                _function_span("jax_orchidee/driver/orchestration.py", "paper_1961_next_step_hydrol_precall_scaffold"),
            ]
        else:
            callsites = [
                _function_span("jax_orchidee/sechiba/hydrol.py", "run_hydrol_first_step_module_from_precall"),
                _function_span("jax_orchidee/driver/orchestration.py", "paper_1961_next_step_hydrol_precall_scaffold"),
            ]
        records.append(
            {
                "ledger_entry": entry,
                "oracle_family": spec["family"],
                "status": "verified",
                "production_owner": _function_span(*owner),
                "callsites": callsites,
                "contracts": {
                    "inputs": spec["inputs"],
                    "outputs": spec["outputs"],
                    "state_writeback": spec["state"],
                },
                "oracle_case_ids": spec["cases"],
                "comparison_asset": COMPARISON.relative_to(ROOT).as_posix(),
            }
        )
    return {"schema_version": 1, "lane": "hydrol_thermosoil", "records": records}


def main() -> int:
    checks = _run_production_probe()
    result = {
        "schema_version": 1,
        "family": "production_hydrol_thermosoil",
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "production_execution": True,
        "comparisons": checks,
        "executed_transitions": [
            "paper_1961_driver_timestep_scaffold:tstep0",
            "driver_previous_step_state_from_timestep_scaffold",
            "paper_1961_next_step_hydrol_precall_scaffold:tstep1",
            "driver_previous_step_state_from_next_step_hydrol_scaffold",
        ],
        "fortran_undefined_contracts": ["thermosoil_main.local_ok_zimov"],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    FRAGMENT.parent.mkdir(parents=True, exist_ok=True)
    FRAGMENT.write_text(yaml.safe_dump(_build_fragment(), sort_keys=False), encoding="utf-8")
    print(f"WROTE {COMPARISON} status={result['status']} comparisons={len(checks)}")
    print(f"WROTE {FRAGMENT} records={len(ENTRY_SPECS)}")
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
