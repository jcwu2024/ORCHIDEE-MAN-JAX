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
from scripts.dev.scan_stomate_save_state import (  # noqa: E402
    DEFAULT_SOURCE_DIR,
    scan_fortran_save_state,
)


DEFAULT_RESTART_SCAN = ROOT / "outputs" / "reference_mode" / "stomate_io_restart_field_scan_20260706.json"
DEFAULT_SAVE_SCAN = ROOT / "outputs" / "reference_mode" / "stomate_save_state_scan_20260707.json"

ANNUAL_MEMORY_FIELDS = (
    ("daily_carbon_accumulators", "gpp_daily", "gpp_daily"),
    ("daily_carbon_accumulators", "npp_daily", "npp_daily"),
    ("daily_carbon_accumulators", "turnover_daily", "turnover_daily"),
    ("season_memory", "moiavail_daily", "humrel_daily"),
    ("season_memory", "moiavail_week", "moiavail_week"),
    ("season_memory", "moiavail_month", "moiavail_month"),
    ("season_memory", "precip_daily", "precip_daily"),
    ("season_memory", "t2m_daily", "t2m_daily"),
    ("season_memory", "tsoil_daily", "tsoil_daily"),
    ("season_memory", "soilhum_daily", "soilhum_daily"),
    ("season_memory", "gdd_from_growthinit", "gdd_from_growthinit"),
    ("season_memory", "gdd_midwinter", "gdd_midwinter"),
    ("season_memory", "gdd_m5_dormance", "gdd_m5_dormance"),
    ("season_memory", "ncd_dormance", "ncd_dormance"),
    ("season_memory", "ngd_minus5", "ngd_minus5"),
    ("season_memory", "maxmoistr_last", "maxmoiavail_lastyear"),
    ("season_memory", "maxmoistr_this", "maxmoiavail_thisyear"),
    ("season_memory", "minmoistr_last", "minmoiavail_lastyear"),
    ("season_memory", "minmoistr_this", "minmoiavail_thisyear"),
    ("vegetation_identity", "PFTpresent", "pft_present"),
    ("vegetation_identity", "ind", "ind"),
    ("vegetation_identity", "adapted", "adapted"),
    ("vegetation_identity", "regenerate", "regenerate"),
    ("vegetation_identity", "npp_longterm", "npp_longterm"),
    ("vegetation_identity", "lm_lastyearmax", "lm_lastyearmax"),
    ("vegetation_identity", "lm_thisyearmax", "lm_thisyearmax"),
    ("vegetation_identity", "maxfpc_lastyear", "maxfpc_lastyear"),
    ("vegetation_identity", "maxfpc_thisyear", "maxfpc_thisyear"),
    ("biomass_leaf_memory", "biomass", "biomass"),
    ("biomass_leaf_memory", "maint_resp", "resp_maint_part"),
    ("biomass_leaf_memory", "leaf_age", "leaf_age"),
    ("biomass_leaf_memory", "leaf_frac", "leaf_frac"),
    ("biomass_leaf_memory", "age", "age"),
    ("biomass_leaf_memory", "when_growthinit", "when_growthinit"),
    ("litter_soil_carbon", "litterpart", "litterpart"),
    ("litter_soil_carbon", "litter", "litter"),
    ("litter_soil_carbon", "dead_leaves", "dead_leaves"),
    ("litter_soil_carbon", "carbon", "carbon"),
    ("litter_soil_carbon", "lignin_struc", "lignin_struc"),
    ("litter_soil_carbon", "bm_to_litter", "bm_to_litter"),
    ("litter_soil_carbon", "carb_mass_total", "carb_mass_total"),
    ("product_pools", "prod10", "prod10"),
    ("product_pools", "prod100", "prod100"),
    ("product_pools", "flux10", "flux10"),
    ("product_pools", "flux100", "flux100"),
    ("leak_deep_carbon_doc", "altmax", "altmax"),
    ("leak_deep_carbon_doc", "fixed_cryoturb_depth", "fixed_cryoturbation_depth"),
    ("leak_deep_carbon_doc", "deepC_a", "deepC_a"),
    ("leak_deep_carbon_doc", "deepC_s", "deepC_s"),
    ("leak_deep_carbon_doc", "deepC_p", "deepC_p"),
    ("leak_deep_carbon_doc", "carbon_32l_a", "carbon_32l"),
    ("leak_deep_carbon_doc", "carbon_32l_s", "carbon_32l"),
    ("leak_deep_carbon_doc", "carbon_32l_p", "carbon_32l"),
    ("leak_deep_carbon_doc", "freedoc", "DOC"),
    ("leak_deep_carbon_doc", "adsdoc", "DOC"),
    ("leak_deep_carbon_doc", "interception_storage", "interception_storage"),
)


def _load_json_or_build_save_scan(path: Path) -> dict[str, object]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return scan_fortran_save_state(DEFAULT_SOURCE_DIR)


def build_annual_memory_audit(restart_scan_path: Path = DEFAULT_RESTART_SCAN, save_scan_path: Path = DEFAULT_SAVE_SCAN):
    restart_scan = json.loads(restart_scan_path.read_text(encoding="utf-8"))
    save_scan = _load_json_or_build_save_scan(save_scan_path)
    restart_by_name: dict[str, list[dict[str, object]]] = {}
    for record in restart_scan["records"]:
        restart_by_name.setdefault(record["name"], []).append(record)
    save_by_name: dict[str, list[dict[str, object]]] = {}
    for record in save_scan["records"]:
        save_by_name.setdefault(record["name"], []).append(record)

    jax_fields = (
        set(StomateRestartEntryState._fields)
        | set(StomateRestartSeasonState._fields)
        | set(StomateDailyAccumulatorState._fields)
    )
    records: list[dict[str, object]] = []
    for group, fortran_name, jax_name in ANNUAL_MEMORY_FIELDS:
        restart_sources = restart_by_name.get(fortran_name, [])
        save_sources = save_by_name.get(fortran_name, [])
        source_records = [
            {
                "kind": "restart_io",
                "file": restart_scan["source"],
                "line": source["line"],
                "call": source["call"],
            }
            for source in restart_sources
        ]
        source_records.extend(
            {
                "kind": "save_state",
                "file": source["file"],
                "line": source["line"],
                "module": source["module"],
                "procedure": source["procedure"],
            }
            for source in save_sources
        )
        records.append(
            {
                "group": group,
                "fortran_name": fortran_name,
                "jax_field": jax_name,
                "jax_field_present": jax_name in jax_fields,
                "source_present": bool(source_records),
                "sources": source_records,
            }
        )

    missing_jax = [record for record in records if not record["jax_field_present"]]
    missing_source = [record for record in records if not record["source_present"]]
    by_group: dict[str, int] = {}
    for record in records:
        by_group[record["group"]] = by_group.get(record["group"], 0) + 1
    return {
        "restart_scan": str(restart_scan_path.relative_to(ROOT) if restart_scan_path.is_relative_to(ROOT) else restart_scan_path),
        "save_scan": str(save_scan_path.relative_to(ROOT) if save_scan_path.is_relative_to(ROOT) else save_scan_path),
        "summary": {
            "annual_memory_fields": len(records),
            "missing_jax_field": len(missing_jax),
            "missing_source": len(missing_source),
            "groups": by_group,
        },
        "records": records,
        "missing_jax_field": missing_jax,
        "missing_source": missing_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restart-scan", type=Path, default=DEFAULT_RESTART_SCAN)
    parser.add_argument("--save-scan", type=Path, default=DEFAULT_SAVE_SCAN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    audit = build_annual_memory_audit(args.restart_scan, args.save_scan)
    text = json.dumps(audit, indent=2, sort_keys=False)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
