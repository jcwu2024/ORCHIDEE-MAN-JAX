from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.stomate.reference import (  # noqa: E402
    StomateDailyAccumulatorState,
    StomateRestartEntryState,
    StomateRestartSeasonState,
)
from scripts.dev.audit_stomate_annual_memory_fields import (  # noqa: E402
    ANNUAL_MEMORY_FIELDS,
)
from scripts.dev.audit_stomate_restart_state_coverage import ALIASES  # noqa: E402
from scripts.dev.scan_stomate_save_state import (  # noqa: E402
    DEFAULT_SOURCE_DIR,
    scan_fortran_save_state,
)


DEFAULT_SAVE_SCAN = ROOT / "outputs" / "reference_mode" / "stomate_save_state_scan_20260707.json"

LIFECYCLE_NAMES = {
    "check",
    "first_call_grassland_manag",
    "firstcall",
    "firstcall_alloc",
    "firstcall_constraints",
    "firstcall_establish",
    "firstcall_fire",
    "firstcall_gap",
    "firstcall_kill",
    "firstcall_litter",
    "firstcall_lpj",
    "firstcall_npp",
    "firstcall_phenology",
    "firstcall_resp",
    "firstcall_soilcarbon",
    "firstcall_vmax",
    "last_action",
    "pr_fois",
}

CROP_STICS_NAMES = {
    "box_biom",
    "box_biomrem",
    "box_durage",
    "box_lai",
    "box_lairem",
    "box_ndays",
    "box_somsenbase",
    "box_tdev",
    "box_ulai",
    "c_leafb",
    "c_reserve",
    "deltgrain",
    "deltai",
    "dltaisen",
    "dltaisenat",
    "dltags",
    "dltams",
    "dltamsen",
    "drylen",
    "durvie",
    "durvieI",
    "cyc_num",
    "cyc_num_tot",
    "efdensite",
    "evapot_daily",
    "f_crop_recycle",
    "f_rot_stom",
    "f_sen_lai",
    "fgellev",
    "fgelflo",
    "fstressgel",
    "ftempremp",
    "gdh_daily",
    "gelee",
    "gslen",
    "hist_latestset",
    "hist_sencourset",
    "histgrowthset",
    "in_cycle",
    "ircarb",
    "kreprac",
    "laisen",
    "mafeuiljaune",
    "magrain",
    "masec",
    "masecveg",
    "msneojaune",
    "namf",
    "nbfeuille",
    "nbgrains",
    "nbgraingel",
    "nbj0remp",
    "nbjhumec",
    "ndebsen",
    "nstopfeuille",
    "nstoprac",
    "nsen",
    "nsencour",
    "onarretesomcourdrp",
    "pdircarb",
    "pdbiomass",
    "pdlai",
    "pdlaisen",
    "pdmagrain",
    "pdmasec",
    "pdulai",
    "pdsfruittot",
    "phoi",
    "pgrain",
    "pgraingel",
    "plantdate",
    "plantdate_now",
    "reajust",
    "reprac",
    "repracmax",
    "repracmin",
    "rot_cmd_store",
    "R_stlevamf",
    "slai",
    "somfeuille",
    "somsenreste",
    "somtemphumec",
    "somtemprac",
    "st2m_daily",
    "st2m_max_daily",
    "st2m_min_daily",
    "stmatrec",
    "stpltlev",
    "svmax",
    "tempeff",
    "ulai",
    "urac",
    "v_dltams",
    "vitmoy",
    "wus_cm_daily",
    "wut_cm_daily",
    "when_growthinit_cut",
}

FIRE_LCC_NAMES = {
    "bafrac_deforest_accu",
    "cflux_prod10",
    "cflux_prod100",
    "cflux_prod_monthly",
    "convflux",
    "def_fuel_1000hr_remain",
    "def_fuel_100hr_remain",
    "def_fuel_10hr_remain",
    "def_fuel_1hr_remain",
    "deforest_biomass_remain",
    "deforest_litter_remain",
    "emideforest_biomass_accu",
    "emideforest_litter_accu",
    "fire_numday",
    "fire_numday_g",
    "glcc_pft",
    "harvest_above",
    "harvest_above_monthly",
    "lcc",
    "ni_acc",
    "after_snow",
    "after_wet",
}

INACTIVE_OK_PC_DYNPEAT_NAMES = {
    "acro_to_cato",
    "acro_to_cato_d",
    "carbon_acro",
    "carbon_cato",
    "carbon_save",
    "deepC_pftmean",
    "deepC_a_save",
    "deepC_p_save",
    "deepC_peat",
    "deepC_s_save",
    "delta_fsave",
    "height_acro",
    "height_cato",
    "litter_to_acro",
    "litter_to_acro_d",
    "resp_acro_anoxic_d",
    "resp_acro_anoxic",
    "resp_acro_oxic_d",
    "resp_acro_oxic",
    "resp_cato_d",
    "resp_cato",
    "tcarbon_acro",
    "tcarbon_cato",
}

CH4_OR_WETLAND_TOKENS = (
    "ch4",
    "o2_",
    "uo_",
    "uold2",
    "fwet",
    "liqwt",
    "wet1",
    "wet2",
    "wet3",
    "wet4",
)

CH4_OR_WETLAND_NAMES = {
    "carbon_surf",
    "tsurf_year",
}

SAVE_ALIASES = {
    "humrel_week": "moiavail_week",
    "humrel_month": "moiavail_month",
    "maxhumrel_lastyear": "maxmoiavail_lastyear",
    "maxhumrel_thisyear": "maxmoiavail_thisyear",
    "minhumrel_lastyear": "minmoiavail_lastyear",
    "minhumrel_thisyear": "minmoiavail_thisyear",
}

PROCESS_LOCAL_COVERED = {
    "alt",
    "alt_ind",
    "altmax_ind",
    "altmax_ind_lastyear",
    "altmax_lastyear",
    "alpha_C",
    "alpha_DOC",
    "alpha_litter_below",
    "beta_C",
    "beta_DOC",
    "beta_litter_below",
    "bioturb_location",
    "bm_to_littercalc",
    "carbon_32l_conct",
    "carbon_32l_pftmean",
    "Cmax",
    "control_moist_daily",
    "control_temp_daily",
    "cryoturb_location",
    "diff_k",
    "fpc_max",
    "fbact",
    "fbact_doc",
    "floodcarbon_input_daily",
    "limit_decomp_moisture",
    "moistfunc_below_1",
    "mu_soil",
    "prmfrst_soilc_tempctrl",
    "prmfrst_soilc_tempctrl_doc",
    "resp_growth_d",
    "resp_hetero_d",
    "resp_hetero_radia",
    "resp_maint_d",
    "resp_maint_part_radia",
    "resp_maint_radia",
    "sla_age1",
    "soilcarbon_input_daily",
    "soilcarbon_input_DOC_daily",
    "turnover_littercalc",
    "veget_cov_max",
    "veget_mask_2d",
    "rootlev",
    "xc_cryoturb",
    "xd_cryoturb",
    "xe_ALL",
    "z_root",
}

INACTIVE_CFORCING_AGGREGATE_NAMES = {
    "control_moist",
    "control_temp",
    "floodcarbon_input",
    "soilcarbon_input",
    "soilcarbon_input_DOC",
}

INACTIVE_OK_PC_OR_PERMAFROST_CFORCING_DAILY_BUFFERS = {
    "fbact_daily",
    "fbact_doc_daily",
    "hsdeep_daily",
    "pb_pa_daily",
    "prmfrst_soilc_tempctrl_daily",
    "prmfrst_soilc_tempctrl_doc_daily",
    "snow_daily",
    "snowdz_daily",
    "snowrho_daily",
    "tdeep_daily",
    "temp_sol_daily",
}

IO_METADATA_OR_INDEX = {
    "hist_id_stomate",
    "hist_id_stomate_IPCC",
    "horideep_index",
    "hori_index",
    "horip10_index",
    "horip100_index",
    "horip101_index",
    "horip11_index",
    "horipft_index",
    "horisnow_index",
    "itime",
    "lalo_global",
    "rest_id_stomate",
    "trefe",
}

VERTICAL_GRID_CACHE = {
    "z_soil",
    "zf_soil",
    "zf_soil_B",
    "zi_soil",
}

STATIC_CONFIG_NAMES = {
    "bioturbation_depth",
    "bioturbation_diff_k_in",
    "carbon_tau",
    "cryoturbation_diff_k_in",
    "cryoturbation_method",
    "DOC_tau",
    "frac_soil",
    "frozen_respiration_func",
    "litter_tau",
    "litterfrac",
    "max_cryoturb_alt",
    "min_cryoturb_alt",
    "pool_tau",
    "priming_param",
    "use_new_cryoturbation",
}

SOURCE_LOCAL_UNUSED_SNOW_GEOMETRY = {
    "heights_snow",
    "zf_snow",
    "zf_snow_nopftdim",
    "zi_snow",
    "zi_snow_nopftdim",
}

SPINUP_OR_CFORCING_NAMES = {
    "carbon_eq",
    "current_stock",
    "dt_forcesoil",
    "eps_carbon",
    "iatt",
    "iisf",
    "isf",
    "MatrixW",
    "nbp_accu",
    "nbp_flux",
    "nf_cumul",
    "nf_written",
    "nparan",
    "nforce",
    "nsfm",
    "nsft",
    "previous_stock",
    "VectorU",
}

GRASSLAND_OR_GRAZING_NAMES = {
    "compt_ugb",
    "herbivores",
    "import_yield",
    "litter_avail",
    "litter_avail_frac",
    "litter_not_avail",
    "nb_ani",
    "sr_ugb",
    "t2m_14",
    "wshtotsum",
}

TIME_METADATA_NAMES = {
    "months_num",
    "nummonth",
}

ACTIVE_LAYER_GEOMETRY_NAMES = {
    "alt",
    "alt_ind",
    "altmax_ind",
    "altmax_ind_lastyear",
    "altmax_lastyear",
    "lalo_global",
    "rootlev",
    "veget_mask_2d",
    "z_root",
}

CRYOTURBATION_COEFFICIENT_NAMES = {
    "alpha_C",
    "alpha_DOC",
    "alpha_a",
    "alpha_litter_below",
    "alpha_p",
    "alpha_s",
    "beta_C",
    "beta_DOC",
    "beta_a",
    "beta_litter_below",
    "beta_p",
    "beta_s",
    "bioturb_location",
    "Cmax",
    "cryoturb_location",
    "diff_k",
    "fc",
    "fr",
    "xc_cryoturb",
    "xd_cryoturb",
    "xe_ALL",
    "xe_a",
    "xe_p",
    "xe_s",
}

LITTER_PARAMETER_NAMES = {
    "carbon_tau",
    "frac_soil",
    "litter_tau",
    "litterfrac",
    "pool_tau",
}

PERMAFROST_CONFIG_NAMES = {
    "bioturbation_depth",
    "bioturbation_diff_k_in",
    "cond_fact",
    "cryoturbation_diff_k_in",
    "cryoturbation_method",
    "fbactratio",
    "frozen_respiration_func",
    "lhc",
    "limit_decomp_moisture",
    "max_cryoturb_alt",
    "min_cryoturb_alt",
    "O2m",
    "ok_cryoturb",
    "ok_methane",
    "oxlim",
    "use_new_cryoturbation",
}

SNOW_SOIL_GEOMETRY_NAMES = {
    "airvol_snow",
    "airvol_soil",
    "conduct_snow",
    "conduct_soil",
    "heights_snow",
    "mu_snow",
    "mu_soil",
    "zf_coeff_snow",
    "zf_snow",
    "zf_snow_nopftdim",
    "zi_coeff_snow",
    "zi_snow",
    "zi_snow_nopftdim",
    "z_thickness",
}

LOCAL_SCRATCH_NAMES = {
    "id",
    "id2",
    "id3",
    "id4",
    "tcounter",
}


def _load_save_scan(path: Path) -> dict[str, object]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return scan_fortran_save_state(DEFAULT_SOURCE_DIR)


def _jax_fields() -> set[str]:
    return (
        set(StomateRestartEntryState._fields)
        | set(StomateRestartSeasonState._fields)
        | set(StomateDailyAccumulatorState._fields)
    )


def _mapped_name(name: str) -> str:
    return SAVE_ALIASES.get(name, ALIASES.get(name, name))


def classify_save_record(
    record: dict[str, object],
    *,
    jax_fields: set[str],
    annual_names: set[str],
    annual_jax_names: set[str],
) -> tuple[str, str]:
    name = str(record["name"])
    lower = name.lower()
    mapped = _mapped_name(name)
    file_name = Path(str(record["file"])).name.lower()
    declaration = str(record.get("declaration", "")).lower()
    line = int(record["line"])

    if name in annual_names or mapped in annual_jax_names:
        return "active_paper_annual_memory_covered", mapped
    if mapped in jax_fields:
        return "jax_restart_or_daily_state_covered", mapped
    if name in LOCAL_SCRATCH_NAMES:
        return "diagnostic_io_local_scratch_state", mapped
    if file_name == "stomate_permafrost_soilcarbon.f90":
        return "inactive_ok_pc_dynpeat_state", mapped
    if name in PROCESS_LOCAL_COVERED:
        return "active_process_local_derived_state_covered", mapped
    if name in INACTIVE_CFORCING_AGGREGATE_NAMES:
        return "inactive_cforcing_or_analytic_spinup_cache", mapped
    if name in INACTIVE_OK_PC_OR_PERMAFROST_CFORCING_DAILY_BUFFERS:
        return "inactive_ok_pc_or_permafrost_cforcing_daily_buffer", mapped
    if name in IO_METADATA_OR_INDEX:
        return "io_metadata_or_parallel_index_state", mapped
    if name in VERTICAL_GRID_CACHE:
        return "static_vertical_grid_cache", mapped
    if name in STATIC_CONFIG_NAMES:
        return "static_config_or_parameter", mapped
    if name in SOURCE_LOCAL_UNUSED_SNOW_GEOMETRY:
        return "source_local_unused_snow_geometry_cache", mapped
    if lower in LIFECYCLE_NAMES or lower.startswith("firstcall"):
        return "lifecycle_or_initialization_flag", mapped
    if name.endswith("_Cforcing_g") or "cforcing" in lower or lower in {"cforcing_id", "cforcing_permafrost_id"}:
        return "inactive_cforcing_or_analytic_spinup_cache", mapped
    if name in SPINUP_OR_CFORCING_NAMES or "spinup" in lower or lower in {"matrixv", "vector_u", "global_years", "ok_equilibrium", "npp_equil", "npp_tot"}:
        return "inactive_cforcing_or_analytic_spinup_cache", mapped
    if name in INACTIVE_OK_PC_DYNPEAT_NAMES:
        return "inactive_ok_pc_dynpeat_state", mapped
    if name in CH4_OR_WETLAND_NAMES or any(token in lower for token in CH4_OR_WETLAND_TOKENS):
        return "inactive_wetland_ch4_or_wetdiag_state", mapped
    if name in GRASSLAND_OR_GRAZING_NAMES or file_name.startswith("grassland_") or "graz" in lower or "animal" in lower:
        return "inactive_grassland_management_or_grazing_state", mapped
    if name in CROP_STICS_NAMES or (file_name == "stomate.f90" and 1080 <= line <= 1280):
        return "inactive_crop_stics_state", mapped
    if name in FIRE_LCC_NAMES or "fire" in lower or "deforest" in lower:
        return "inactive_fire_lcc_or_product_branch_state", mapped
    if name in TIME_METADATA_NAMES:
        return "time_or_calendar_metadata", mapped
    if name == "N_limfert":
        return "inactive_crop_stics_state", mapped
    if name == "temp_str" or ("allocatable" not in declaration and bool(record.get("initializer"))):
        return "static_config_or_parameter", mapped
    if file_name in {"stomate_soilcarbon.f90", "stomate_permafrost_soilcarbon.f90", "stomate_litter.f90"}:
        return "pft14_leak_permafrost_save_needs_microcase_or_manual_audit", mapped
    if file_name == "stomate.f90":
        return "stomate_main_save_needs_manual_audit", mapped
    return "other_save_needs_manual_audit", mapped


def pft14_leak_microcase_group(record: dict[str, object], status: str) -> str | None:
    name = str(record["name"])
    file_name = Path(str(record["file"])).name.lower()
    if status != "pft14_leak_permafrost_save_needs_microcase_or_manual_audit":
        return None
    if file_name == "stomate.f90":
        return "stomate_main_ok_leak_daily_buffers"
    if name in LITTER_PARAMETER_NAMES:
        return "litter_static_parameters"
    if file_name == "stomate_litter.f90":
        return "litter_dynamic_or_moisture_cache"
    if name in ACTIVE_LAYER_GEOMETRY_NAMES:
        return "active_layer_geometry_and_rooting"
    if name in CRYOTURBATION_COEFFICIENT_NAMES:
        return "cryoturbation_coefficients_and_locations"
    if name in SNOW_SOIL_GEOMETRY_NAMES:
        return "snow_soil_vertical_geometry"
    if name in PERMAFROST_CONFIG_NAMES:
        return "permafrost_cryoturbation_config"
    if name in LOCAL_SCRATCH_NAMES:
        return "local_scratch_indices"
    if name in {"carbon_32l_conct", "carbon_32l_pftmean", "DOC_tau", "priming_param", "deepC_pftmean"}:
        return "soilcarbon_doc_profile_coefficients"
    return "pft14_leak_other_manual_audit"


def build_save_state_coverage_audit(save_scan_path: Path = DEFAULT_SAVE_SCAN) -> dict[str, object]:
    scan = _load_save_scan(save_scan_path)
    annual_names = {fortran_name for _, fortran_name, _ in ANNUAL_MEMORY_FIELDS}
    annual_jax_names = {jax_name for _, _, jax_name in ANNUAL_MEMORY_FIELDS}
    fields = _jax_fields()
    by_status: dict[str, list[dict[str, object]]] = {}
    records: list[dict[str, object]] = []
    for source in scan["records"]:
        status, mapped = classify_save_record(
            source,
            jax_fields=fields,
            annual_names=annual_names,
            annual_jax_names=annual_jax_names,
        )
        item = {
            "status": status,
            "name": source["name"],
            "mapped_to": mapped,
            "file": source["file"],
            "line": source["line"],
            "module": source["module"],
            "procedure": source["procedure"],
            "threadprivate": source["threadprivate"],
            "text": source["text"],
        }
        microcase_group = pft14_leak_microcase_group(source, status)
        if microcase_group is not None:
            item["microcase_group"] = microcase_group
        records.append(item)
        by_status.setdefault(status, []).append(item)

    unique_by_status: dict[str, int] = {}
    for status, items in by_status.items():
        unique_by_status[status] = len({str(item["name"]) for item in items})
    microcase_groups: dict[str, int] = {}
    for item in records:
        group = item.get("microcase_group")
        if group is not None:
            microcase_groups[str(group)] = microcase_groups.get(str(group), 0) + 1

    return {
        "save_scan": str(save_scan_path.relative_to(ROOT) if save_scan_path.is_relative_to(ROOT) else save_scan_path),
        "summary": {
            "save_records": len(records),
            "unique_save_names": len({str(record["name"]) for record in records}),
            "statuses": {status: len(items) for status, items in sorted(by_status.items())},
            "unique_names_by_status": dict(sorted(unique_by_status.items())),
            "microcase_groups": dict(sorted(microcase_groups.items())),
            "unclassified": 0,
        },
        "by_status": dict(sorted(by_status.items())),
        "records": records,
        "notes": {
            "classification_policy": (
                "Conservative source-ledger classification. Covered means a known JAX restart/daily/season "
                "field or the focused active annual-memory audit owns the state. Broad soilcarbon/permafrost "
                "SAVE variables are intentionally routed to micro-case/manual audit instead of being claimed closed."
            ),
            "paper_case_switches": {
                "GRM_ENABLE_GRAZING": "FALSE",
                "ENABLE_GRAZING": "n",
                "CH4_CALCUL": "n",
                "OK_STICS": "inactive for PFT14 paper path",
                "LAND_COVER_CHANGE": "n",
                "FIRE_DISABLE": "y",
                "STOMATE_CFORCING": "not used as active paper production path",
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-scan", type=Path, default=DEFAULT_SAVE_SCAN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    audit = build_save_state_coverage_audit(args.save_scan)
    text = json.dumps(audit, indent=2, sort_keys=False)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
