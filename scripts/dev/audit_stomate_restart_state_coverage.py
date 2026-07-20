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


DEFAULT_SCAN = ROOT / "outputs" / "reference_mode" / "stomate_io_restart_field_scan_20260706.json"

ALIASES = {
    "PFTpresent": "pft_present",
    "RIP_time": "rip_time",
    "co2_to_bm_dgvm": "co2_to_bm",
    "maint_resp": "resp_maint_part",
    "lig_struc_be": "lignin_struc_below",
    "fixed_cryoturb_depth": "fixed_cryoturbation_depth",
    "freedoc": "DOC",
    "adsdoc": "DOC",
    "carbon_32l_a": "carbon_32l",
    "carbon_32l_s": "carbon_32l",
    "carbon_32l_p": "carbon_32l",
    "prod10": "prod10",
    "prod100": "prod100",
    "dt_days": "dt_days_read",
    "moiavail_daily": "humrel_daily",
    "Tseason": "tseason",
    "Tseason_length": "tseason_length",
    "Tseason_tmp": "tseason_tmp",
    "Tmin_spring_time": "tmin_spring_time",
    "maxmoistr_last": "maxmoiavail_lastyear",
    "maxmoistr_this": "maxmoiavail_thisyear",
    "minmoistr_last": "minmoiavail_lastyear",
    "minmoistr_this": "minmoiavail_thisyear",
}

INACTIVE_FIRE_LCC_GRAZING = {
    "co2_fire",
    "convflux",
    "cflux_prod10",
    "cflux_prod100",
    "fireindex",
    "firelitter",
    "flux10",
    "flux100",
    "grazed_frac",
    "import_yield",
    "nb_ani",
    "nb_grazingdays",
    "sr_ugb",
    "wshtotsum",
    "after_snow",
    "after_wet",
    "wet1day",
    "wet2day",
    "litter_not_avail",
    "t2m_14",
    "ni_acc",
}

INACTIVE_WETLAND_CH4 = {
    "CH4_snow",
    "CH4_soil",
    "O2_snow",
    "O2_soil",
    "fwet_month",
    "fwet_series",
    "liqwt_max",
    "liqwt_month",
    "tsurf_year",
    "uo_0",
    "uo_wet1",
    "uo_wet2",
    "uo_wet3",
    "uo_wet4",
    "uold2_0",
    "uold2_wet1",
    "uold2_wet2",
    "uold2_wet3",
    "uold2_wet4",
}

INACTIVE_DYNAMIC_PEAT = {
    "carbon_acro",
    "carbon_cato",
    "carbon_save",
    "deepC_a_save",
    "deepC_p_save",
    "deepC_peat",
    "deepC_s_save",
    "delta_fsave",
    "height_acro",
}

SPINUP_EQUILIBRIUM = {
    "Global_years",
    "MatrixV",
    "Vector_U",
    "current_stock",
    "nbp_flux",
    "nbp_sum",
    "ok_equilibrium",
    "previous_stock",
}

NON_LEAK_OR_DGVM_BRANCH = {
    "carbon",
    "deepC_a",
    "deepC_p",
    "deepC_s",
    "depth_deepsoil",
    "lignin_struc",
    "litter",
    "need_adjacent",
    "veget_lastlight",
}

ACTIVE_OUTPUT_DIAGNOSTIC = {
    "resp_hetero",
}


def classify_field(field: str, covered_names: set[str]) -> tuple[str, str]:
    mapped = ALIASES.get(field, field)
    if mapped in covered_names:
        return "covered_by_jax_restart_state", mapped
    if field in INACTIVE_FIRE_LCC_GRAZING:
        return "inactive_fire_lcc_grazing_branch", mapped
    if field in INACTIVE_WETLAND_CH4:
        return "inactive_wetland_ch4_branch", mapped
    if field in INACTIVE_DYNAMIC_PEAT:
        return "inactive_ok_pc_dynpeat_branch", mapped
    if field in SPINUP_EQUILIBRIUM:
        return "spinup_equilibrium_state", mapped
    if field in NON_LEAK_OR_DGVM_BRANCH:
        return "non_leak_or_dgvm_branch_state", mapped
    if field in ACTIVE_OUTPUT_DIAGNOSTIC:
        return "active_output_diagnostic_not_later_day_entry_state", mapped
    return "unclassified", mapped


def build_audit(scan_path: Path) -> dict[str, object]:
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    fields = sorted({record["name"] for record in scan["records"]})
    covered_names = set(StomateRestartEntryState._fields) | set(StomateRestartSeasonState._fields) | set(
        StomateDailyAccumulatorState._fields
    )
    by_status: dict[str, list[dict[str, str]]] = {}
    for field in fields:
        status, mapped = classify_field(field, covered_names)
        by_status.setdefault(status, []).append({"field": field, "mapped_to": mapped})
    return {
        "source_scan": str(scan_path.relative_to(ROOT) if scan_path.is_relative_to(ROOT) else scan_path),
        "summary": {
            "scan_fields": len(fields),
            "unclassified": len(by_status.get("unclassified", ())),
            **{status: len(items) for status, items in sorted(by_status.items())},
        },
        "by_status": dict(sorted(by_status.items())),
        "notes": {
            "paper_case_switches": {
                "FIRE_DISABLE": "y",
                "LAND_COVER_CHANGE": "n",
                "GRM_ENABLE_GRAZING": "FALSE",
                "CH4_CALCUL": "n",
                "OK_PC": "n",
                "DYN_PEAT": "n",
                "PERMA_PEAT": "y",
                "TF_DOC": "y",
                "STOMATE_OK_DGVM": "n",
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, default=DEFAULT_SCAN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    audit = build_audit(args.scan)
    text = json.dumps(audit, indent=2, sort_keys=False)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
