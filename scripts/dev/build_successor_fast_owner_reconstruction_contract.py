#!/usr/bin/env python3
"""Build and validate the frozen successor fast-owner reconstruction contract.

This utility reads only v6 contract metadata. It does not read Teacher rows,
regenerate data, train a model, or execute the half-hour model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_CONTRACT = ROOT / "docs/source_audits/sechiba_state_field_contract.yaml"
DEFAULT_OUTPUT = ROOT / "manifests/coarse_graining/successor_fast_owner_reconstruction_v1.json"

SCHEMA_VERSION = "successor_fast_owner_reconstruction_v1"
DATASET_ID = "pft14-daily-teacher-669-1961-2010"
DATASET_MANIFEST_SHA256 = "5fc2c0691bfa53252f875f7646f3c1a5bea203c2759a6e947b26bb51dc13aa99"
MARKOV_CONTRACT_SHA256 = "6813065b51e95a2ec6663d8bc92f5336148704aea502632a05fd51d7e2e9d442"
SOURCE_CONTRACT_SHA256 = "9ec8b89a9c1ff1988d7571d0ba25bb3310520f27aaf2ea0d7fea6c942cc235bd"
EXPECTED_LEAF_COUNT = 160
EXPECTED_TARGET_WIDTH = 2855

ROLE_EXISTING = "existing_process_endpoint"
ROLE_NEURAL = "independent_neural_endpoint"
ROLE_DERIVATION = "deterministic_source_exact_derivation"
ROLE_MIRROR = "mirror_of_canonical_owner"
ROLE_CARRY = "true_guarded_or_static_carry"
ROLES = frozenset({ROLE_EXISTING, ROLE_NEURAL, ROLE_DERIVATION, ROLE_MIRROR, ROLE_CARRY})
EXPECTED_ROLE_LEAF_COUNTS = {
    ROLE_DERIVATION: 19,
    ROLE_EXISTING: 21,
    ROLE_NEURAL: 79,
    ROLE_MIRROR: 30,
    ROLE_CARRY: 11,
}
EXPECTED_ROLE_COLUMN_COUNTS = {
    ROLE_DERIVATION: 95,
    ROLE_EXISTING: 1482,
    ROLE_NEURAL: 839,
    ROLE_MIRROR: 369,
    ROLE_CARRY: 70,
}


def _keys(prefix: str, names: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"{prefix}.{name}" for name in names)


_EXISTING_PROCESS_OWNERS = {
    **{
        key: f"conservative_update.water.{key.rsplit('.', 1)[-1]}"
        for key in _keys(
            "hydrol_previous_step_state",
            ("flood_res", "mc", "mcl", "qsintveg", "snow", "snow_nobio", "snowliq"),
        )
    },
    **{
        key: f"conservative_update.thermal.{key}"
        for key in (
            "enerbil_previous_step_state.temp_sol",
            "hydrol_previous_step_state.snowheat",
            "hydrol_previous_step_state.snowtemp",
            "thermosoil_previous_step_state.ptn",
            "thermosoil_previous_step_state.stempdiag",
        )
    },
    **{
        key: f"conservative_update.carbon.{key.rsplit('.', 1)[-1]}"
        for key in _keys(
            "ok_leak",
            ("litter_above", "litter_below", "carbon_32l", "DOC", "interception_storage"),
        )
    },
    "daily_interface.gpp_daily": "named_output.daily.fast_gpp",
    "daily_interface.resp_maint_part": "named_output.daily.fast_maintenance_respiration",
    "daily_interface.tsoil_daily": "named_output.daily.temperature_context.tsoil",
    "daily_interface.tsurf_daily": "named_output.daily.temperature_context.tsurf",
}

_DERIVATION_RULES = {
    "driver_previous_step_state.albedo_nir": "band_1_of:driver_previous_step_state.albedo",
    "driver_previous_step_state.albedo_vis": "band_0_of:driver_previous_step_state.albedo",
    "driver_previous_step_state.old_eair": "final_native_forcing:CP_AIR*Tair+GRAVITY*measurement_height",
    "driver_previous_step_state.old_qair": "final_native_forcing:Qair",
    "driver_previous_step_state.old_zlev": "static_condition:measurement_heights[0]",
    "driver_previous_step_state.peqAcoef": "source_exact_constant:0",
    "driver_previous_step_state.peqBcoef": "copy:driver_previous_step_state.old_qair",
    "driver_previous_step_state.petAcoef": "source_exact_constant:0",
    "driver_previous_step_state.petBcoef": "copy:driver_previous_step_state.old_eair",
    "driver_previous_step_state.rau_old": "final_native_forcing:air_density",
    "daily_interface.flood_root_radia": "source_exact_constant:0_for_accepted_pft14_scope",
    "daily_interface.precip_daily": "native_forcing_daily_integral:Rainf+Snowf",
    "daily_interface.snowfall_daily": "native_forcing_daily_integral:Snowf",
    "daily_interface.t2m_daily": "native_forcing_daily_mean:Tair",
    "daily_interface.t2m_max_daily": "native_forcing_daily_max:Tair",
    "daily_interface.t2m_min_daily": "native_forcing_daily_min:Tair",
    "daily_interface.tmc_topgrass_daily": "source_exact_constant:0_for_pft14_no_grass",
    "ok_leak.deepC_peat": "source_exact_derived_diagnostic:sum_carbon_32l_pool_axis",
    "final_diagnostics.t2mdiag": "final_native_forcing:Tair",
}

_MIRROR_OWNERS = {
    "driver_previous_step_state.z0": "diffuco_previous_step_state.z0m",
    "diffuco_previous_step_state.evap_bare_lim": "hydrol_previous_step_state.evap_bare_lim",
    "diffuco_previous_step_state.humrel": "hydrol_previous_step_state.humrel",
    "diffuco_previous_step_state.qsintveg": "hydrol_previous_step_state.qsintveg",
    "diffuco_previous_step_state.snow": "hydrol_previous_step_state.snow",
    "diffuco_previous_step_state.snow_age": "hydrol_previous_step_state.snow_age",
    "diffuco_previous_step_state.snow_nobio": "hydrol_previous_step_state.snow_nobio",
    "diffuco_previous_step_state.snow_nobio_age": "hydrol_previous_step_state.snow_nobio_age",
    "diffuco_previous_step_state.snowdz": "hydrol_previous_step_state.snowdz",
    "diffuco_previous_step_state.snowgrain": "hydrol_previous_step_state.snowgrain",
    "diffuco_previous_step_state.snowheat": "hydrol_previous_step_state.snowheat",
    "diffuco_previous_step_state.snowrho": "hydrol_previous_step_state.snowrho",
    "diffuco_previous_step_state.snowtemp": "hydrol_previous_step_state.snowtemp",
    "diffuco_previous_step_state.temp_sol": "enerbil_previous_step_state.temp_sol",
    "enerbil_previous_step_state.albedo": "driver_previous_step_state.albedo",
    "enerbil_previous_step_state.cgrnd": "thermosoil_previous_step_state.cgrnd",
    "enerbil_previous_step_state.dgrnd": "thermosoil_previous_step_state.dgrnd",
    "enerbil_previous_step_state.qsurf": "diffuco_previous_step_state.qsurf",
    "enerbil_previous_step_state.roughness": "diffuco_previous_step_state.roughheight",
    "enerbil_previous_step_state.snowdz": "hydrol_previous_step_state.snowdz",
    "enerbil_previous_step_state.soilcap": "thermosoil_previous_step_state.soilcap",
    "enerbil_previous_step_state.soilflx": "thermosoil_previous_step_state.soilflx",
    "enerbil_previous_step_state.temp_sol_add": "hydrol_previous_step_state.temp_sol_add",
    "enerbil_previous_step_state.temp_sol_pft": "diffuco_previous_step_state.temp_sol_pft",
    "hydrol_previous_step_state.drainage": "hydrol_previous_step_state.drainage_per_soil",
    "hydrol_previous_step_state.runoff": "hydrol_previous_step_state.runoff_per_soil",
    "hydrol_previous_step_state.soil_mc": "hydrol_previous_step_state.mc",
    "thermosoil_previous_step_state.hsdeep": "thermosoil_previous_step_state.shum_ngrnd_permalong",
    "thermosoil_previous_step_state.tdeep": "thermosoil_previous_step_state.ptn",
    "thermosoil_previous_step_state.temp_sol_beg": "enerbil_previous_step_state.temp_sol",
}

_CARRY_OWNERS = {
    "diffuco_previous_step_state.control_salinity": "day_start.diffuco_previous_step_state.control_salinity",
    "diffuco_previous_step_state.frac_nobio": "day_start.slowproc_stomate_previous_step_state.frac_nobio",
    "diffuco_previous_step_state.height": "day_start.diffuco_previous_step_state.height",
    "diffuco_previous_step_state.lai": "day_start.slowproc_stomate_previous_step_state.lai",
    "diffuco_previous_step_state.soilalbedo_bg": "day_start.diffuco_previous_step_state.soilalbedo_bg",
    "diffuco_previous_step_state.veget": "day_start.slowproc_stomate_previous_step_state.veget",
    "diffuco_previous_step_state.veget_max": "day_start.slowproc_stomate_previous_step_state.veget_max",
    "hydrol_previous_step_state.free_drain_coef": "day_start.hydrol_previous_step_state.free_drain_coef",
    "hydrol_previous_step_state.zwt_force": "day_start.hydrol_previous_step_state.zwt_force",
    "thermosoil_previous_step_state.e_soil_lat": "day_start.thermosoil_previous_step_state.e_soil_lat",
    "thermosoil_previous_step_state.refSOC": "day_start.thermosoil_previous_step_state.refSOC",
}

_NEURAL_ENDPOINT_KEYS = frozenset(
    {
        *_keys("driver_previous_step_state", ("albedo", "fluxsens", "vevapp")),
        *_keys(
            "diffuco_previous_step_state",
            (
                "cimean",
                "control_inudate",
                "drysoil_frac",
                "evapot",
                "evapot_corr",
                "gpp",
                "gsmean",
                "qsurf",
                "roughheight",
                "roughheight_pft",
                "rstruct",
                "rveget",
                "temp_sol_pft",
                "z0h",
                "z0m",
            ),
        ),
        *_keys("enerbil_previous_step_state", ("soilcap_pft", "soilflx_pft")),
        *_keys(
            "hydrol_previous_step_state",
            (
                "ae_ns",
                "canopy2ground",
                "drainage_per_soil",
                "evap_bare_lim",
                "evap_bare_lim_ns",
                "fwet_new",
                "grndflux",
                "humrel",
                "humrelv",
                "nroot",
                "precip2canopy",
                "precip2ground",
                "profil_froz_hydro_ns",
                "run2man",
                "run2peat",
                "runoff2peat",
                "runoff_per_soil",
                "shumdiag_peat",
                "snow_age",
                "snow_nobio_age",
                "snowdz",
                "snowgrain",
                "snowmelt",
                "snowrho",
                "soilflxresid",
                "subsnownobio",
                "temp_hydro",
                "temp_sol_add",
                "us",
                "wat_flux",
                "water2infilt",
                "wt_ab",
                "wt_ab_tide",
            ),
        ),
        *_keys(
            "thermosoil_previous_step_state",
            (
                "cgrnd",
                "cgrnd_snow",
                "dgrnd",
                "dgrnd_snow",
                "gtemp",
                "lambda_snow",
                "pcapa_en",
                "shum_ngrnd_permalong",
                "soilcap",
                "soilflx",
            ),
        ),
        *_keys(
            "daily_interface",
            (
                "humrel_daily",
                "litterhum_daily",
                "resp_maint_radia",
                "snowmass_daily",
                "soilhum_daily",
                "wspeed_daily",
            ),
        ),
        *_keys(
            "ok_leak",
            (
                "lignin_struc_above",
                "lignin_struc_below",
                "litterpart",
                "dead_leaves",
                "fuel_1hr",
                "fuel_10hr",
                "fuel_100hr",
                "fuel_1000hr",
            ),
        ),
        "final_diagnostics.temp_sol",
        "sechiba_finalize_state.leaf_ci",
    }
)

_DIFFUCO_PROCESS_DOMAINS = {
    **{
        key: "carbon"
        for key in _keys(
            "diffuco_previous_step_state",
            (
                "cimean",
                "control_inudate",
                "gpp",
                "gsmean",
                "rstruct",
                "rveget",
            ),
        )
    },
    **{
        key: "water"
        for key in _keys(
            "diffuco_previous_step_state",
            ("drysoil_frac", "evapot", "evapot_corr", "qsurf"),
        )
    },
    **{
        key: "energy"
        for key in _keys(
            "diffuco_previous_step_state",
            ("roughheight", "roughheight_pft", "temp_sol_pft", "z0h", "z0m"),
        )
    },
}

_DRIVER_PROCESS_DOMAINS = {
    "driver_previous_step_state.albedo": "energy",
    "driver_previous_step_state.fluxsens": "energy",
    "driver_previous_step_state.vevapp": "water",
}

_SEMANTIC_COLLISION_KEYS = frozenset(
    {
        "driver_previous_step_state.fluxsens",
        "enerbil_previous_step_state.soilflx",
        "enerbil_previous_step_state.soilflx_pft",
        *_keys(
            "hydrol_previous_step_state",
            (
                "canopy2ground",
                "drainage",
                "drainage_per_soil",
                "precip2canopy",
                "precip2ground",
                "runoff",
                "runoff2peat",
                "runoff_per_soil",
                "subsnownobio",
                "wat_flux",
                "water2infilt",
            ),
        ),
    }
)

_UNIT_INTERVAL_FIELDS = frozenset(
    {
        "driver_previous_step_state.albedo",
        "diffuco_previous_step_state.control_inudate",
        "diffuco_previous_step_state.drysoil_frac",
        "hydrol_previous_step_state.fwet_new",
        "hydrol_previous_step_state.humrel",
        "hydrol_previous_step_state.humrelv",
        "daily_interface.humrel_daily",
        "daily_interface.litterhum_daily",
        "daily_interface.soilhum_daily",
        "ok_leak.lignin_struc_above",
        "ok_leak.lignin_struc_below",
        "ok_leak.litterpart",
    }
)

_NONNEGATIVE_NAME_PARTS = (
    "albedo",
    "evap_bare_lim",
    "gpp",
    "gsmean",
    "rough",
    "rstruct",
    "rveget",
    "resp_maint",
    "soilcap",
    "cgrnd",
    "dgrnd",
    "lambda_snow",
    "pcapa_en",
    "snow",
    "drainage",
    "runoff",
    "precip",
    "canopy2ground",
    "water2infilt",
    "wspeed",
    "nroot",
    "wt_ab",
    "lignin",
    "litterpart",
    "dead_leaves",
    "fuel_",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_json({key: value for key, value in payload.items() if key != "contract_sha256"})
    ).hexdigest()


def _load_json(path: str | Path) -> Mapping[str, Any]:
    if str(path) == "-":
        value = json.loads(sys.stdin.read().lstrip(chr(0xFEFF)))
    else:
        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("JSON root must be an object")
    return value


def _load_source_contract(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("SECHIBA source contract root must be an object")
    return value


def _source_ledger(source_contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {f"{item['component']}.{item['field']}": item for item in source_contract["contracts"]}


def _role_for_key(key: str) -> tuple[str, str]:
    matches = []
    if key in _EXISTING_PROCESS_OWNERS:
        matches.append((ROLE_EXISTING, _EXISTING_PROCESS_OWNERS[key]))
    if key in _NEURAL_ENDPOINT_KEYS:
        matches.append((ROLE_NEURAL, _neural_output_key(key)))
    if key in _DERIVATION_RULES:
        matches.append((ROLE_DERIVATION, _DERIVATION_RULES[key]))
    if key in _MIRROR_OWNERS:
        matches.append((ROLE_MIRROR, _MIRROR_OWNERS[key]))
    if key in _CARRY_OWNERS:
        matches.append((ROLE_CARRY, _CARRY_OWNERS[key]))
    if len(matches) != 1:
        raise ValueError(f"fast leaf {key!r} has {len(matches)} owner classifications")
    return matches[0]


def _neural_output_key(key: str) -> str:
    if key.startswith("daily_interface."):
        return f"endpoint.daily_summary.{key.rsplit('.', 1)[-1]}"
    if key.startswith("ok_leak."):
        return f"endpoint.ok_leak_aux.{key.rsplit('.', 1)[-1]}"
    return f"endpoint.fast_state.{key}"


def _process_domain(key: str) -> str:
    if key in _DRIVER_PROCESS_DOMAINS:
        return _DRIVER_PROCESS_DOMAINS[key]
    if key in _DIFFUCO_PROCESS_DOMAINS:
        return _DIFFUCO_PROCESS_DOMAINS[key]
    if key.startswith("enerbil_previous_step_state."):
        return "energy"
    if key.startswith("hydrol_previous_step_state."):
        return "water"
    if key.startswith("thermosoil_previous_step_state."):
        return "energy"
    if key == "daily_interface.resp_maint_radia":
        return "carbon"
    if key == "daily_interface.wspeed_daily":
        return "energy"
    if key.startswith("daily_interface."):
        return "water"
    if key == "final_diagnostics.temp_sol":
        return "energy"
    if key.startswith("ok_leak.") or key == "sechiba_finalize_state.leaf_ci":
        return "carbon"
    raise ValueError(f"missing neural process-domain routing for {key}")


def _loss_group(key: str) -> str:
    component = key.split(".", 1)[0]
    return {
        "driver_previous_step_state": "fast_owner.driver",
        "diffuco_previous_step_state": "fast_owner.diffuco",
        "enerbil_previous_step_state": "fast_owner.enerbil",
        "hydrol_previous_step_state": "fast_owner.hydrol",
        "thermosoil_previous_step_state": "fast_owner.thermosoil",
        "daily_interface": "daily_summary",
        "ok_leak": "ok_leak_aux",
        "final_diagnostics": "fast_owner.enerbil",
        "sechiba_finalize_state": "fast_owner.sechiba_finalize",
    }[component]


def _value_representation(key: str) -> str:
    if key.startswith(("daily_interface.", "ok_leak.")):
        return "normalized_absolute_teacher_native"
    return "normalized_day_end_minus_day_start_owner"


def _scientific_constraint(key: str, role: str) -> str:
    if key == "diffuco_previous_step_state.rveget":
        return "dynamic_defined_nonnegative"
    if key in _UNIT_INTERVAL_FIELDS:
        return "unit_interval"
    if role == ROLE_NEURAL and any(part in key for part in _NONNEGATIVE_NAME_PARTS):
        return "nonnegative"
    return "finite_teacher_native"


def _defined_policy(key: str) -> str:
    if key == "diffuco_previous_step_state.rveget":
        return "neural_dynamic_value_and_defined_logit"
    if key == "daily_interface.gpp_daily":
        return "deterministic_active_vegetation_mask"
    if key in {"daily_interface.resp_maint_part", "daily_interface.resp_maint_radia"}:
        return "deterministic_active_vegetation_and_part_mask"
    return "frozen_contract_static_mask"


def _provenance_for_key(
    key: str,
    source_by_key: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    source = source_by_key.get(key)
    if source is not None:
        return {
            name: source[name]
            for name in (
                "freshness",
                "producer_phase",
                "jax_owner",
                "source_ref",
            )
        }
    if key.startswith("daily_interface."):
        return {
            "producer_phase": "stomate_daily_fold",
            "jax_owner": "jax_orchidee.stomate.daily.stomate_first_step_daily_accumulation_from_entry",
            "source_ref": "fortran_source/ORCHIDEE/src_stomate/stomate.f90 daily accumulation",
        }
    if key.startswith("ok_leak."):
        return {
            "producer_phase": "ok_leak_half_hour_fold",
            "jax_owner": "jax_orchidee.driver.orchestration._paper_compiled_ok_leak_fold",
            "source_ref": "research.daily_coarse_graining.carbon_budget_ownership.OK_LEAK_CARBON_FIELD_OWNERSHIP",
        }
    if key == "sechiba_finalize_state.leaf_ci":
        return {
            "producer_phase": "diffuco_end",
            "jax_owner": "jax_orchidee.driver.orchestration._updated_leaf_ci",
            "source_ref": "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_trans_co2 lines 2018-2969",
        }
    if key.startswith("final_diagnostics."):
        return {
            "producer_phase": "final_half_hour_entry",
            "jax_owner": "research.daily_coarse_graining.daily_markov_contract._diagnostic_inventory",
            "source_ref": "compact_sechiba_entry_payload",
        }
    if key.startswith("driver_previous_step_state.old_") or key.endswith(".rau_old"):
        return {
            "producer_phase": "enerbil_end",
            "jax_owner": "jax_orchidee.driver.orchestration._add_enerbil_local_to_previous_fields",
            "source_ref": "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_main lines 390-598",
        }
    if key == "thermosoil_previous_step_state.e_soil_lat":
        return {
            "producer_phase": "thermosoil_guarded_ok_Ecorr_carry",
            "jax_owner": _CARRY_OWNERS[key],
            "source_ref": ("fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90 lines 463-465, 595-597, 2508-2536"),
        }
    raise ValueError(f"missing source provenance for {key}")


def build_contract(
    markov_contract: Mapping[str, Any],
    source_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if markov_contract.get("schema_version") != "daily_markov_contract_v5":
        raise ValueError("successor fast-owner contract requires daily_markov_contract_v5")
    if hashlib.sha256(_canonical_json(markov_contract)).hexdigest() != MARKOV_CONTRACT_SHA256:
        raise ValueError("embedded v6 Markov contract SHA256 drift")
    source_by_key = _source_ledger(source_contract)
    leaves = []
    for leaf in markov_contract["fast_day_target_leaves"]:
        key = str(leaf["key"])
        role, owner = _role_for_key(key)
        record = {
            "key": key,
            "family": str(leaf["family"]),
            "component": leaf.get("component"),
            "path": [str(value) for value in leaf["path"]],
            "shape": [int(value) for value in leaf["shape"]],
            "axis_names": [str(value) for value in leaf.get("axis_names", ())],
            "start": int(leaf["start"]),
            "stop": int(leaf["stop"]),
            "width": int(leaf["stop"]) - int(leaf["start"]),
            "role": role,
            "canonical_owner": owner,
            "teacher_target": f"v6.fast_day_target[{int(leaf['start'])}:{int(leaf['stop'])}]",
            "unit_policy": "teacher_native_no_conversion",
            "scientific_constraint": _scientific_constraint(key, role),
            "defined_policy": _defined_policy(key),
            "provenance": _provenance_for_key(key, source_by_key),
        }
        if role == ROLE_NEURAL:
            record["output_key"] = owner
            record["value_representation"] = _value_representation(key)
            record["proposed_process_domain"] = _process_domain(key)
            record["loss_group"] = _loss_group(key)
            record["loss_mask_policy"] = (
                "teacher_dynamic_defined_value_mask_and_eligible_slot_mask_bce"
                if key == "diffuco_previous_step_state.rveget"
                else "teacher_defined_and_frozen_contract_static_mask"
            )
            pft_axis = record["axis_names"].index("nvm") if "nvm" in record["axis_names"] else None
            record["pft_axis"] = pft_axis
            record["pft_axis_policy"] = "compact_active_pft_axis" if pft_axis is not None else "not_pft_indexed"
            if key == "diffuco_previous_step_state.rveget":
                record["defined_output_key"] = f"{owner}.defined_logit"
        if key in _SEMANTIC_COLLISION_KEYS:
            record["typed_integral_collision"] = {
                "forbidden_source": f"typed_daily_integral_with_same_short_name:{key.rsplit('.', 1)[-1]}",
                "required_source": owner,
            }
        leaves.append(record)

    counts = Counter(record["role"] for record in leaves)
    columns = Counter()
    outputs = []
    value_output_width = 0
    dynamic_defined_width = 0
    for record in leaves:
        columns[record["role"]] += int(record["width"])
        if record["role"] == ROLE_NEURAL:
            outputs.append(record["output_key"])
            value_output_width += int(record["width"])
            if "defined_output_key" in record:
                dynamic_defined_width += int(record["width"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_for_materializer_acceptance_no_decoder_change_authorized",
        "dataset": {
            "release_id": "pft14-daily-teacher-v6",
            "dataset_id": DATASET_ID,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "markov_contract_sha256": MARKOV_CONTRACT_SHA256,
            "source_contract_sha256": SOURCE_CONTRACT_SHA256,
            "historical_v5_allowed": False,
            "teacher_regeneration_allowed": False,
        },
        "boundary": {
            "binding": "bind_daily_operator(input_assembler=canonical_assembler,process_head=candidate_head,retained_tail=canonical_retained_tail_adapter)",
            "fast_leaf_count": int(markov_contract["fast_day_target_leaves"].__len__()),
            "fast_target_width": int(markov_contract["fast_day_target_width"]),
            "anonymous_fast_vector_allowed": False,
            "reconstruct_48_forcing_records": False,
            "execute_48_step_state_scan": False,
        },
        "decoder_extension": {
            "status": "not_implemented_pending_user_architecture_review",
            "value_output_count": len(outputs),
            "value_output_width": value_output_width,
            "dynamic_defined_logit_count": 1,
            "dynamic_defined_logit_width": dynamic_defined_width,
            "total_added_scalar_width": value_output_width + dynamic_defined_width,
            "output_keys": outputs,
            "candidate_control_share_identical_output_schema": True,
            "decoder_transform": "architecture_review_pending",
        },
        "role_leaf_counts": dict(sorted(counts.items())),
        "role_column_counts": dict(sorted(columns.items())),
        "acceptance_gates": {
            "contract": [
                "160_of_160_fast_leaves_classified_once",
                "no_duplicate_or_anonymous_owner",
                "typed_daily_integrals_never_materialize_instantaneous_state",
            ],
            "pre_training_materializer": [
                "all_teacher_160_of_160_with_atol_1e-8_rtol_1e-10",
                "cold_ordinary_restart_masks_and_discrete_exact",
                "rveget_defined_status_exact",
            ],
            "local_before_gpu": [
                "candidate_and_control_forward_loss_jvp_vjp_finite",
                "candidate_control_parameter_difference_at_most_0.02",
                "one_day_state_and_water_carbon_budget_checks_pass",
            ],
            "closed": [
                "training",
                "7_day_rollout",
                "30_day_rollout",
                "365_day_rollout",
                "gate_d2",
                "product_acceptance",
            ],
        },
        "leaves": leaves,
    }
    payload["contract_sha256"] = _payload_sha256(payload)
    validate_contract(payload, markov_contract=markov_contract)
    return payload


def validate_contract(
    payload: Mapping[str, Any],
    *,
    markov_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("successor fast-owner schema version drift")
    if payload.get("contract_sha256") != _payload_sha256(payload):
        raise ValueError("successor fast-owner contract SHA256 mismatch")
    dataset = payload["dataset"]
    expected_identity = {
        "dataset_id": DATASET_ID,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "markov_contract_sha256": MARKOV_CONTRACT_SHA256,
        "source_contract_sha256": SOURCE_CONTRACT_SHA256,
    }
    for name, expected in expected_identity.items():
        if dataset.get(name) != expected:
            raise ValueError(f"successor fast-owner {name} identity drift")
    leaves = list(payload["leaves"])
    if len(leaves) != EXPECTED_LEAF_COUNT:
        raise ValueError(f"expected {EXPECTED_LEAF_COUNT} fast leaves, got {len(leaves)}")
    keys = [str(record["key"]) for record in leaves]
    if len(set(keys)) != len(keys):
        raise ValueError("successor fast-owner contract contains duplicate leaves")
    cursor = 0
    outputs = []
    dynamic = []
    for record in leaves:
        if record["role"] not in ROLES:
            raise ValueError(f"unknown fast-owner role for {record['key']}")
        if int(record["start"]) != cursor or int(record["stop"]) <= cursor:
            raise ValueError(f"non-contiguous fast-owner slice for {record['key']}")
        cursor = int(record["stop"])
        if int(record["width"]) != cursor - int(record["start"]):
            raise ValueError(f"fast-owner width drift for {record['key']}")
        if record["role"] == ROLE_NEURAL:
            output = str(record["output_key"])
            if not output.startswith("endpoint.") or "anonymous" in output:
                raise ValueError(f"invalid named endpoint output for {record['key']}")
            outputs.append(output)
        if record["defined_policy"] == "neural_dynamic_value_and_defined_logit":
            dynamic.append(str(record["key"]))
        if record["role"] == ROLE_MIRROR:
            owner = str(record["canonical_owner"])
            if owner not in keys:
                raise ValueError(f"mirror owner {owner} is absent")
    if cursor != EXPECTED_TARGET_WIDTH:
        raise ValueError(f"expected target width {EXPECTED_TARGET_WIDTH}, got {cursor}")
    if len(set(outputs)) != len(outputs):
        raise ValueError("neural endpoint output keys are not unique")
    if dynamic != ["diffuco_previous_step_state.rveget"]:
        raise ValueError(f"unexpected dynamic-defined fields: {dynamic}")
    by_key = {str(record["key"]): record for record in leaves}
    for key, record in by_key.items():
        if record["role"] != ROLE_MIRROR:
            continue
        owner = str(record["canonical_owner"])
        if owner == key or by_key[owner]["role"] == ROLE_MIRROR:
            raise ValueError(f"mirror chain or cycle is forbidden for {key}")
    for key in _SEMANTIC_COLLISION_KEYS:
        record = by_key[key]
        if "typed_integral_collision" not in record:
            raise ValueError(f"typed/state collision is not explicit for {key}")
        if str(record["canonical_owner"]).startswith("typed."):
            raise ValueError(f"typed daily integral still owns fast state {key}")
    counts = Counter(str(record["role"]) for record in leaves)
    if dict(sorted(counts.items())) != dict(payload["role_leaf_counts"]):
        raise ValueError("fast-owner role counts drifted")
    if dict(counts) != EXPECTED_ROLE_LEAF_COUNTS:
        raise ValueError("fast-owner role leaf counts differ from the frozen design")
    columns = Counter()
    for record in leaves:
        columns[str(record["role"])] += int(record["width"])
    if dict(columns) != EXPECTED_ROLE_COLUMN_COUNTS:
        raise ValueError("fast-owner role column counts differ from the frozen design")
    if len(outputs) != int(payload["decoder_extension"]["value_output_count"]):
        raise ValueError("decoder value-output count drifted")
    value_width = sum(int(record["width"]) for record in leaves if record["role"] == ROLE_NEURAL)
    dynamic_width = sum(int(record["width"]) for record in leaves if "defined_output_key" in record)
    decoder = payload["decoder_extension"]
    if int(decoder["value_output_width"]) != value_width:
        raise ValueError("decoder value-output width drifted")
    if int(decoder["dynamic_defined_logit_width"]) != dynamic_width:
        raise ValueError("decoder dynamic-defined width drifted")
    if int(decoder["total_added_scalar_width"]) != value_width + dynamic_width:
        raise ValueError("decoder total added scalar width drifted")
    process_domains = Counter(
        str(record["proposed_process_domain"]) for record in leaves if record["role"] == ROLE_NEURAL
    )
    if dict(process_domains) != {"energy": 21, "water": 42, "carbon": 16}:
        raise ValueError("neural endpoint process-domain counts drifted")
    if markov_contract is not None:
        if int(markov_contract["fast_day_target_width"]) != EXPECTED_TARGET_WIDTH:
            raise ValueError("v6 fast-target width drift")
        expected = list(markov_contract["fast_day_target_leaves"])
        if len(expected) != len(leaves):
            raise ValueError("v6 fast-target leaf count drift")
        for record, leaf in zip(leaves, expected, strict=True):
            for name in ("key", "family", "component", "path", "shape", "axis_names", "start", "stop"):
                left = record[name]
                right = leaf.get(name, [] if name == "axis_names" else None)
                if left != right:
                    raise ValueError(f"v6 metadata drift for {record['key']}.{name}")
    return {
        "contract_sha256": payload["contract_sha256"],
        "leaf_count": len(leaves),
        "target_width": cursor,
        "role_leaf_counts": dict(sorted(counts.items())),
        "neural_value_output_count": len(outputs),
        "neural_value_output_width": value_width,
        "dynamic_defined_output_count": len(dynamic),
        "dynamic_defined_output_width": dynamic_width,
        "typed_collision_count": len(_SEMANTIC_COLLISION_KEYS),
    }


def _write_json(payload: Mapping[str, Any], output: str | Path) -> None:
    if str(output) == "-":
        handle: TextIO = sys.stdout
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def render_markdown(payload: Mapping[str, Any]) -> str:
    """Render the review document from the machine contract."""

    neural = [record for record in payload["leaves"] if record["role"] == ROLE_NEURAL]
    decoder = payload["decoder_extension"]
    role_rows = "\n".join(
        f"| `{role}` | {payload['role_leaf_counts'][role]} | {payload['role_column_counts'][role]} |"
        for role in (
            ROLE_EXISTING,
            ROLE_DERIVATION,
            ROLE_MIRROR,
            ROLE_CARRY,
            ROLE_NEURAL,
        )
    )
    endpoint_rows = []
    for record in neural:
        shape = " x ".join(str(value) for value in record["shape"]) or "scalar"
        axes = ", ".join(record["axis_names"]) or "scalar"
        mask = (
            "dynamic value mask + eligible-slot BCE"
            if "defined_output_key" in record
            else "Teacher-defined + static contract mask"
        )
        endpoint_rows.append(
            f"| `{record['key']}` | `{record['output_key']}` | `{shape}` | "
            f"`{axes}` | `{record['proposed_process_domain']}` | "
            f"`{record['scientific_constraint']}` | {mask} | "
            f"`[{record['start']}:{record['stop']}]` |"
        )
    collisions = "\n".join(
        f"- `{record['key']}` -> `{record['canonical_owner']}`"
        for record in payload["leaves"]
        if "typed_integral_collision" in record
    )
    added_fields = int(decoder["value_output_count"]) + int(decoder["dynamic_defined_logit_count"])
    field_parameter_increase = added_fields * 2 * 256
    proposed_label_increase = 8 * 256
    total_parameter_increase = field_parameter_increase + proposed_label_increase
    return f"""# Successor Fast-Owner Reconstruction Contract v1

Status: **schema implemented in the all-Teacher materializer; neural decoder is not implemented**.

Date: 2026-08-09.

Machine contract: `manifests/coarse_graining/successor_fast_owner_reconstruction_v1.json`

Contract SHA256: `{payload["contract_sha256"]}`

## Purpose

The existing 101 source-backed process outputs are preserved. This contract
adds the missing canonical fast-owner reconstruction layer required to rebuild
all 160 leaves and 2,855 columns of the accepted v6 fast-day boundary before
the retained exact daily tail runs.

The new endpoints do not directly predict biomass, NPP, allocation, or
mortality. They reconstruct day-end fast state and daily auxiliary values.
The retained exact processes continue to update biomass and productivity from
the named fast GPP, maintenance respiration, water, thermal, and carbon
inventory endpoints.

This design reads no v5 data, regenerates no v6 data, reconstructs no 48-step
forcing sequence, and performs no 48-step state scan.

## Frozen Data Identity

- Dataset: `{DATASET_ID}`
- Dataset manifest SHA256: `{DATASET_MANIFEST_SHA256}`
- Markov contract SHA256: `{MARKOV_CONTRACT_SHA256}`
- SECHIBA state source-contract SHA256: `{SOURCE_CONTRACT_SHA256}`

## Complete Ownership

| Role | Leaves | Scalar columns |
| --- | ---: | ---: |
{role_rows}
| **Total** | **{EXPECTED_LEAF_COUNT}** | **{EXPECTED_TARGET_WIDTH}** |

Every leaf has exactly one role. Mirrors must point directly to a non-mirror
canonical owner, so mirror chains and cycles are forbidden. The JSON contract
contains the owner, source provenance, unit policy, constraint, mask policy,
shape, axes, and Teacher slice for all 160 leaves.

## Added Neural Boundary

- {decoder["value_output_count"]} named value heads spanning {decoder["value_output_width"]} scalar values.
- One `rveget` defined-logit head spanning 2 compact-PFT scalar logits.
- Total added scalar output width: {decoder["total_added_scalar_width"]}.
- Proposed route domains: 42 water heads, 21 energy heads, 16 carbon heads.
- Candidate and matched control must use the identical added schema, masks,
  losses, normalization, and materializer.
- Fast-state endpoints use normalized day-end-minus-day-start-owner targets.
  Daily summaries and OK_LEAK auxiliaries use normalized absolute targets.
- All values remain in Teacher-native units. Constraint transforms are applied
  only after their exact decoder form is approved.

### Architecture impact requiring review

Under the current source-routed implementation, the {decoder["value_output_count"]} value heads plus the
mask head would add {added_fields} field queries. They would participate in the same
source-admitted process graph: content-dependent attention in the candidate
and fixed graph mixing in the control. They do not add cross-day memory.

The decoder is coordinate-shared, so {decoder["total_added_scalar_width"]} scalar outputs do not imply an
`{decoder["total_added_scalar_width"]} x d_model` dense layer. With `d_model=256`, {added_fields} query/output embeddings
add {field_parameter_increase:,} parameters. If the proposed eight monitoring/loss groups are
kept as separate label embeddings, the estimated total increase is
{total_parameter_increase:,} parameters, from 8,679,681 to
{8_679_681 + total_parameter_increase:,} per arm (about
{100.0 * total_parameter_increase / 8_679_681:.3f}%). This estimate is not an implementation
authorization; the query interaction, label/loss grouping, and mask decoder
must be approved first.

## New Value Heads

`unit_policy` is `teacher_native_no_conversion` for every row.

| Teacher leaf | Output key | Shape | Axes | Domain | Constraint | Mask | Teacher slice |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(endpoint_rows)}

## Typed/State Collision Exclusions

The following 14 instantaneous state endpoints must never be populated from a
same-named `typed.*` daily integral:

{collisions}

## Gates Before Training

1. Implement a non-neural materializer and prove an all-Teacher oracle reaches
   160/160 leaves at `atol=1e-8`, `rtol=1e-10`.
2. Prove cold, ordinary later-day, and restart-year exact masks and discrete
   behavior, including the dynamic `rveget` status.
3. After explicit architecture approval, extend both arms identically and run
   local forward/loss/JVP/VJP plus one-day state and water/carbon budget checks.
4. Only then reopen a bounded `gln01` real-v6 smoke and small fit. The 7/30-day,
   365-day, long-held-point, Gate D2, and product gates remain closed.
"""


def _write_text(value: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-manifest",
        required=True,
        help="Accepted v6 dataset manifest path, or '-' for stdin.",
    )
    parser.add_argument(
        "--dataset-manifest-sha256",
        help="Required accepted manifest SHA256 when --dataset-manifest is '-'.",
    )
    parser.add_argument(
        "--source-contract",
        type=Path,
        default=DEFAULT_SOURCE_CONTRACT,
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument(
        "--validate-only",
        type=Path,
        help="Validate an existing generated contract instead of building it.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.dataset_manifest == "-":
        manifest_sha256 = args.dataset_manifest_sha256
    else:
        manifest_sha256 = _sha256_file(Path(args.dataset_manifest))
    if manifest_sha256 != DATASET_MANIFEST_SHA256:
        raise ValueError("accepted v6 dataset manifest SHA256 drift")
    if _sha256_file(args.source_contract) != SOURCE_CONTRACT_SHA256:
        raise ValueError("SECHIBA state source-contract SHA256 drift")
    raw = _load_json(args.dataset_manifest)
    markov_contract = raw.get("markov_contract", raw)
    if raw.get("dataset_id", DATASET_ID) != DATASET_ID:
        raise ValueError("dataset identity drift")
    if raw.get("markov_contract_sha256", MARKOV_CONTRACT_SHA256) != MARKOV_CONTRACT_SHA256:
        raise ValueError("v6 Markov contract SHA256 drift")
    if args.validate_only is None:
        payload = build_contract(markov_contract, _load_source_contract(args.source_contract))
        _write_json(payload, args.output)
        summary = validate_contract(payload, markov_contract=markov_contract)
    else:
        payload = _load_json(args.validate_only)
        summary = validate_contract(payload, markov_contract=markov_contract)
    if args.markdown_output is not None:
        _write_text(render_markdown(payload), args.markdown_output)
    print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
