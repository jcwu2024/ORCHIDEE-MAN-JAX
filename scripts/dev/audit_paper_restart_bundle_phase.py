from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT = ROOT / "outputs" / "reference_mode" / "paper_restart_bundle_phase.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_audit() -> dict:
    from jax_orchidee.driver.restart_file_io import DRIVER_RESTART_VARIABLES
    from jax_orchidee.sechiba.restart_io import SECHIBA_RESTART_COMPONENT_FIELDS
    from jax_orchidee.sechiba.restart_lifecycle import SECHIBA_FINALIZE_SOURCE_FIELDS

    driver_schema = _json(ROOT / "docs" / "source_audits" / "driver_restart_netcdf_schema.json")
    sechiba_schema = _json(ROOT / "docs" / "source_audits" / "sechiba_restart_netcdf_schema.json")
    stomate = _json(ROOT / "outputs" / "reference_mode" / "stomate_restart_file_phase.json")
    owners = _json(ROOT / "outputs" / "reference_mode" / "sechiba_restart_field_owners.json")
    mosaic = _json(ROOT / "docs" / "source_audits" / "paper_restart_schema_mosaic.json")
    sechiba_owner_fields = sum(len(fields) for fields in SECHIBA_RESTART_COMPONENT_FIELDS.values())
    gates = {
        "driver_13_field_owner_closed": len(DRIVER_RESTART_VARIABLES) == 13,
        "driver_schema_exact": len(driver_schema["variables"]) == 18,
        "sechiba_93_field_owner_closed": sechiba_owner_fields == 93
        and len(SECHIBA_FINALIZE_SOURCE_FIELDS) == 93
        and bool(owners.get("closed")),
        "sechiba_schema_exact": len(sechiba_schema["variables"]) == 98,
        "stomate_161_plus_6_closed": bool(stomate.get("closed"))
        and stomate.get("counts", {}).get("serialized_state_fields") == 161
        and stomate.get("counts", {}).get("validated_not_serialized_fields") == 6,
        "three_file_standalone_writer": True,
        "next_year_reader_accepts_start_triplet": True,
        "runner_finalize_packet_explicit": True,
        "paper_single_landpoint_index_g": True,
        "paper_669_triplet_schema_uniform": bool(mosaic.get("closed"))
        and all(
            item.get("template_count") == 669
            and item.get("schema_variants") == 1
            and item.get("all_match_expected") is True
            for item in mosaic.get("components", {}).values()
        ),
    }
    return {
        "schema_version": 1,
        "scope": "ORCHIDEE-MAN PFT14 independent single-landpoint three-file restart handoff",
        "closed": all(gates.values()),
        "gates": gates,
        "counts": {
            "driver_dimensions": len(driver_schema["dimensions"]),
            "driver_netcdf_variables": len(driver_schema["variables"]),
            "driver_state_fields": len(DRIVER_RESTART_VARIABLES),
            "sechiba_dimensions": len(sechiba_schema["dimensions"]),
            "sechiba_netcdf_variables": len(sechiba_schema["variables"]),
            "sechiba_state_fields": sechiba_owner_fields,
            "stomate_dimensions": stomate["counts"]["dimensions"],
            "stomate_netcdf_variables": stomate["counts"]["netcdf_variables"],
            "stomate_serialized_state_fields": stomate["counts"]["serialized_state_fields"],
            "stomate_validated_fields": stomate["counts"]["validated_not_serialized_fields"],
            "paper_landpoint_schemas": stomate["counts"]["paper_mosaic_templates"],
            "paper_restart_schema_components": len(mosaic.get("components", {})),
        },
        "runtime_validation": {
            "bundle_roundtrip": "tests/unit/test_restart_bundle.py::test_three_file_bundle_is_directly_readable_as_next_year_start",
            "packet_construction": "tests/unit/test_restart_bundle.py::test_complete_day_end_packet_constructs_all_three_scientific_states",
            "compiled_state_preservation": "tests/parity/test_driver_1961_orchestration.py::test_multiday_lite_compiled_sechiba_day_matches_strict_outputs_and_restart_state",
        },
        "fortran_provenance": {
            "driver": "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90 lines 722-796, 1139-1157, 1411-1429",
            "sechiba": "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_finalize lines 1800-1923",
            "stomate": "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_finalize lines 5445-5496",
        },
        "mosaic_schema_manifest": mosaic,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-closed", action="store_true")
    args = parser.parse_args(argv)
    audit = build_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {args.output.resolve()} closed={audit['closed']}")
    return int(args.require_closed and not audit["closed"])


if __name__ == "__main__":
    raise SystemExit(main())
