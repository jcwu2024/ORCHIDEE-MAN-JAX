from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.stomate.restart_io import (  # noqa: E402
    DEFAULT_STOMATE_RESTART_SCHEMA,
    StomateRestartPhysicalState,
    stomate_writerestart_field_ledger,
    write_stomate_full_writerestart_states,
)
from scripts.dev.extract_stomate_restart_netcdf_schema import (  # noqa: E402
    extract_schema,
)


DEFAULT_OUTPUT = ROOT / "outputs" / "reference_mode" / "stomate_restart_file_phase.json"
DEFAULT_MOSAIC_ROOT = (
    ROOT / "reference" / "OUT" / "orc_calibrate_250919_sen" / "arg2_1.0"
)
DEFAULT_MOSAIC_MANIFEST = (
    ROOT / "docs" / "source_audits" / "stomate_restart_schema_mosaic.json"
)


def _structural_schema(schema: dict[str, object]) -> dict[str, object]:
    return {
        key: schema[key]
        for key in (
            "file_format",
            "dimensions",
            "physical_variables",
            "single_landpoint_scatter",
            "variables",
        )
    }


def scan_mosaic_schema(
    schema: dict[str, object], mosaic_root: Path
) -> dict[str, object]:
    templates = sorted(mosaic_root.rglob("stomate_start.nc"))
    signatures = {
        json.dumps(_structural_schema(extract_schema(path)), sort_keys=True)
        for path in templates
    }
    paths = [path.relative_to(mosaic_root).as_posix() for path in templates]
    return {
        "schema_version": 1,
        "scope": "669 independent paper landpoint STOMATE restart schemas",
        "mosaic_root": mosaic_root.relative_to(ROOT).as_posix(),
        "template_count": len(templates),
        "schema_variants": len(signatures),
        "path_list_sha256": hashlib.sha256("\n".join(paths).encode()).hexdigest(),
        "expected_structure_sha256": hashlib.sha256(
            json.dumps(_structural_schema(schema), sort_keys=True).encode()
        ).hexdigest(),
        "all_match_expected": signatures
        == {json.dumps(_structural_schema(schema), sort_keys=True)},
    }


def build_audit(
    schema_path: Path = DEFAULT_STOMATE_RESTART_SCHEMA,
    mosaic_root: Path = DEFAULT_MOSAIC_ROOT,
    *,
    verify_mosaic: bool = False,
    mosaic_manifest_path: Path = DEFAULT_MOSAIC_MANIFEST,
) -> dict[str, object]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    source_reference = ROOT / schema["source_reference"]
    schema_exact = source_reference.is_file() and extract_schema(source_reference) == schema
    if verify_mosaic:
        mosaic = scan_mosaic_schema(schema, mosaic_root)
    else:
        mosaic = json.loads(mosaic_manifest_path.read_text(encoding="utf-8"))
    expected_structure_sha256 = hashlib.sha256(
        json.dumps(_structural_schema(schema), sort_keys=True).encode()
    ).hexdigest()
    ledger = stomate_writerestart_field_ledger()
    gates = {
        "audited_schema_exact": schema_exact,
        "complete_file_definition": len(schema.get("dimensions", {})) == 26
        and len(schema.get("variables", {})) == 169,
        "physical_variables_explicit": set(schema.get("physical_variables", ()))
        == {"nav_lon", "nav_lat", "nav_lev", "time", "time_steps"},
        "paper_scatter_explicit": schema.get("single_landpoint_scatter", {}).get(
            "index_g"
        )
        == [1]
        and schema.get("single_landpoint_scatter", {}).get("grid_shape") == [1, 1],
        "standalone_writer_executable": callable(write_stomate_full_writerestart_states)
        and StomateRestartPhysicalState is not None,
        "normalized_writer_complete": len(ledger.serialized_fields) == 161
        and len(ledger.validated_not_serialized_fields) == 6,
        "physical_io_boundaries_closed": not ledger.independent_netcdf_boundaries,
        "paper_mosaic_schema_uniform": mosaic.get("template_count") == 669
        and mosaic.get("schema_variants") == 1
        and mosaic.get("all_match_expected") is True
        and mosaic.get("expected_structure_sha256")
        == expected_structure_sha256,
    }
    return {
        "schema_version": 1,
        "scope": schema["scope"],
        "closed": all(gates.values()),
        "gates": gates,
        "counts": {
            "dimensions": len(schema.get("dimensions", {})),
            "netcdf_variables": len(schema.get("variables", {})),
            "serialized_state_fields": len(ledger.serialized_fields),
            "validated_not_serialized_fields": len(
                ledger.validated_not_serialized_fields
            ),
            "paper_mosaic_templates": mosaic.get("template_count"),
            "paper_mosaic_schema_variants": mosaic.get("schema_variants"),
        },
        "source_reference": schema["source_reference"],
        "fortran_provenance": schema["fortran_provenance"],
        "mosaic_schema_manifest": mosaic,
        "runtime_validation": {
            "test": "tests/unit/test_stomate_restart_standalone.py::test_standalone_writer_roundtrips_all_writerestart_state",
            "contract": "five normalized state objects and exact NetCDF schema metadata",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=DEFAULT_STOMATE_RESTART_SCHEMA)
    parser.add_argument("--mosaic-root", type=Path, default=DEFAULT_MOSAIC_ROOT)
    parser.add_argument(
        "--mosaic-manifest", type=Path, default=DEFAULT_MOSAIC_MANIFEST
    )
    parser.add_argument("--verify-mosaic", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-closed", action="store_true")
    args = parser.parse_args()
    audit = build_audit(
        args.schema,
        args.mosaic_root,
        verify_mosaic=args.verify_mosaic,
        mosaic_manifest_path=args.mosaic_manifest,
    )
    if args.verify_mosaic:
        args.mosaic_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.mosaic_manifest.write_text(
            json.dumps(audit["mosaic_schema_manifest"], indent=2) + "\n",
            encoding="utf-8",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {args.output.resolve()} closed={audit['closed']} gates={audit['gates']}")
    return int(args.require_closed and not audit["closed"])


if __name__ == "__main__":
    raise SystemExit(main())
