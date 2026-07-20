from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.dev.audit_stomate_save_state_coverage import build_save_state_coverage_audit  # noqa: E402


def _has(audit, status: str, name: str, mapped_to: str | None = None) -> bool:
    for record in audit["by_status"].get(status, []):
        if record["name"] == name and (mapped_to is None or record["mapped_to"] == mapped_to):
            return True
    return False


def test_stomate_save_state_coverage_audit_classifies_all_save_records():
    audit = build_save_state_coverage_audit()
    summary = audit["summary"]
    statuses = summary["statuses"]

    assert summary["save_records"] == 1094
    assert summary["unique_save_names"] == 996
    assert summary["unclassified"] == 0
    assert "stomate_main_save_needs_manual_audit" not in statuses
    assert "other_save_needs_manual_audit" not in statuses
    assert "pft14_leak_permafrost_save_needs_microcase_or_manual_audit" not in statuses
    assert "dgvm_or_light_competition_save_needs_microcase" not in statuses

    assert statuses["active_paper_annual_memory_covered"] == 49
    assert statuses["active_process_local_derived_state_covered"] == 46
    assert statuses["jax_restart_or_daily_state_covered"] == 45
    assert statuses["diagnostic_io_local_scratch_state"] == 11
    assert statuses["inactive_grassland_management_or_grazing_state"] == 354
    assert statuses["inactive_crop_stics_state"] == 161
    assert statuses["inactive_cforcing_or_analytic_spinup_cache"] == 123
    assert statuses["inactive_ok_pc_dynpeat_state"] == 125
    assert statuses["inactive_ok_pc_or_permafrost_cforcing_daily_buffer"] == 11
    assert statuses["source_local_unused_snow_geometry_cache"] == 5
    assert statuses["static_config_or_parameter"] == 43

    assert summary["microcase_groups"] == {}


def test_stomate_save_state_coverage_audit_keeps_key_fields_in_source_ledgers():
    audit = build_save_state_coverage_audit()

    assert _has(audit, "active_paper_annual_memory_covered", "humrel_week", "moiavail_week")
    assert _has(audit, "active_paper_annual_memory_covered", "maxhumrel_lastyear", "maxmoiavail_lastyear")
    assert _has(audit, "active_process_local_derived_state_covered", "veget_cov_max")
    assert _has(audit, "active_process_local_derived_state_covered", "fbact")
    assert _has(audit, "inactive_wetland_ch4_or_wetdiag_state", "carbon_surf")
    assert _has(audit, "inactive_cforcing_or_analytic_spinup_cache", "soilcarbon_input_DOC")
    assert _has(audit, "static_config_or_parameter", "litterfrac")
    assert _has(audit, "static_config_or_parameter", "carbon_tau")
    assert _has(audit, "static_config_or_parameter", "DOC_tau")
    assert _has(audit, "active_process_local_derived_state_covered", "carbon_32l_pftmean")
    assert _has(audit, "active_process_local_derived_state_covered", "altmax_lastyear")
    assert _has(audit, "active_process_local_derived_state_covered", "rootlev")
    assert _has(audit, "active_process_local_derived_state_covered", "alpha_C")
    assert _has(audit, "active_process_local_derived_state_covered", "diff_k")
    assert _has(audit, "active_process_local_derived_state_covered", "Cmax")
    assert _has(audit, "active_process_local_derived_state_covered", "moistfunc_below_1")
    assert _has(audit, "active_process_local_derived_state_covered", "fpc_max")
    assert _has(audit, "inactive_ok_pc_dynpeat_state", "deepC_pftmean")
    assert _has(audit, "inactive_ok_pc_or_permafrost_cforcing_daily_buffer", "fbact_daily")
    assert _has(audit, "source_local_unused_snow_geometry_cache", "zf_snow")
    assert _has(audit, "static_config_or_parameter", "use_new_cryoturbation")
    assert _has(audit, "inactive_fire_lcc_or_product_branch_state", "convflux")
    assert _has(audit, "inactive_ok_pc_dynpeat_state", "carbon_acro")
    assert _has(audit, "inactive_wetland_ch4_or_wetdiag_state", "tsurf_year")
    assert _has(audit, "inactive_grassland_management_or_grazing_state", "herbivores")
    assert _has(audit, "inactive_crop_stics_state", "f_crop_recycle")
    assert _has(audit, "inactive_cforcing_or_analytic_spinup_cache", "dt_forcesoil")
    assert _has(audit, "diagnostic_io_local_scratch_state", "tcounter")
    assert _has(audit, "static_config_or_parameter", "temp_str")
