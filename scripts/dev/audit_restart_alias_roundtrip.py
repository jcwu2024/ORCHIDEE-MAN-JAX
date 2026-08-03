from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.restart_bundle import (  # noqa: E402
    PaperRestartBundlePhysicalState,
    PaperRestartBundleState,
    read_restart_physical_state,
    write_paper_restart_start_bundle,
)
from jax_orchidee.driver.restart_file_io import read_dim2_driver_restart  # noqa: E402
from jax_orchidee.driver.restart_state import (  # noqa: E402
    reference_case_first_step_restart_state,
)
from scripts.dev.audit_stomate_restart_state_coverage import ALIASES, DEFAULT_SCAN  # noqa: E402


CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"
REFERENCE_ROOT = (
    ROOT
    / "reference"
    / "OUT"
    / "orc_calibrate_250919_sen"
    / "arg2_1.0"
    / "001.0-071.0"
)
VAL_EXP = 999999.0

# Axis indices are zero-based normalized JAX/NumPy indices. The alias order is
# the source order in stomate_io::readstart/writerestart.
PACK_AXIS_DECLARATIONS = {
    "carbon_32l": {
        "aliases": ("carbon_32l_a", "carbon_32l_s", "carbon_32l_p"),
        "pack_axis": 1,
        "pack_axis_name": "ncarb",
        "canonical_axes": ("npts", "ncarb", "nvm", "ndeep"),
        "alias_axes_after_reader": ("npts", "nvm", "ndeep"),
        "reader": "jax_orchidee.stomate.reference.read_restart_carbon_32l",
        "writer": "jax_orchidee.driver.restart_bundle.write_paper_restart_start_bundle",
        "packet_field": "carbon_32l",
        "fortran_provenance": {
            "file": "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90",
            "subroutine": "readstart/writerestart",
            "read_lines": "1540-1551",
            "write_lines": "2377-2386",
        },
    },
    "DOC": {
        "aliases": ("freedoc", "adsdoc"),
        "pack_axis": 3,
        "pack_axis_name": "ndoc",
        "canonical_axes": ("npts", "nvm", "ndeep", "ndoc", "npool", "nelements"),
        "alias_axes_after_reader": ("npts", "nvm", "ndeep", "npool", "nelements"),
        "reader": "jax_orchidee.stomate.reference.read_restart_doc",
        "writer": "jax_orchidee.driver.restart_bundle.write_paper_restart_start_bundle",
        "packet_field": "DOC",
        "fortran_provenance": {
            "file": "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90",
            "subroutine": "readstart/writerestart",
            "read_lines": "1553-1563",
            "write_lines": "2388-2394",
        },
    },
}


def many_to_one_alias_groups(aliases: Mapping[str, str] = ALIASES) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for source, canonical in aliases.items():
        if source != canonical:
            grouped.setdefault(canonical, []).append(source)
    return {
        canonical: tuple(sorted(sources))
        for canonical, sources in sorted(grouped.items())
        if len(sources) > 1
    }


def project_packet_aliases(packet: object) -> dict[str, np.ndarray]:
    fields = getattr(packet, "fields", packet)
    if not isinstance(fields, Mapping):
        raise TypeError("packet must be a mapping or expose a mapping-valued fields attribute")
    projected: dict[str, np.ndarray] = {}
    for canonical, declaration in PACK_AXIS_DECLARATIONS.items():
        if canonical not in fields:
            continue
        value = np.asarray(fields[canonical])
        axis = int(declaration["pack_axis"])
        aliases = tuple(declaration["aliases"])
        expected_rank = len(declaration["canonical_axes"])
        if value.ndim != expected_rank:
            raise ValueError(f"{canonical} must have rank {expected_rank}, got shape {value.shape}")
        if value.shape[axis] != len(aliases):
            raise ValueError(
                f"{canonical} axis {axis} ({declaration['pack_axis_name']}) must have "
                f"length {len(aliases)}, got shape {value.shape}"
            )
        for index, alias in enumerate(aliases):
            projected[alias] = np.take(value, index, axis=axis)
    return projected


def _reference_run() -> Path:
    candidates = sorted(path.parent for path in REFERENCE_ROOT.rglob("driver_start.nc"))
    if not candidates:
        raise FileNotFoundError(f"restart triplet not found below {REFERENCE_ROOT}")
    return candidates[0]


def _sentinel_values(shape: tuple[int, ...], base: float) -> np.ndarray:
    """Make every axis position distinguishable and include readstart sentinels."""

    values = base + np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 16.0
    flat = values.reshape(-1)
    flat[0] = VAL_EXP
    if flat.size > 1:
        flat[1] = 0.0
    if flat.size > 2:
        flat[2] = -base
    return values


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _variable_metadata(dataset: Dataset, name: str) -> dict[str, Any]:
    variable = dataset.variables[name]
    return {
        "dimensions": list(variable.dimensions),
        "shape": list(variable.shape),
        "dtype": str(variable.dtype),
        "attributes": {key: _json_value(variable.getncattr(key)) for key in variable.ncattrs()},
    }


def _comparison(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    exact = expected.shape == actual.shape and bool(np.array_equal(expected, actual, equal_nan=True))
    difference = np.abs(expected - actual) if expected.shape == actual.shape else np.asarray([np.inf])
    return {
        "passed": exact,
        "expected_shape": list(expected.shape),
        "actual_shape": list(actual.shape),
        "exact": exact,
        "rtol": 0.0,
        "atol": 0.0,
        "max_abs_error": float(np.nanmax(difference)),
    }


def _run_production_roundtrip() -> dict[str, Any]:
    run = _reference_run()
    original = reference_case_first_step_restart_state(
        CONFIG, root=ROOT, run_dir=run, stomate_filename="stomate_start.nc"
    )
    entry = original.stomate_readstart.entry_state
    carbon = np.empty_like(np.asarray(entry.carbon_32l), dtype=np.float64)
    for index in range(carbon.shape[1]):
        carbon[:, index] = _sentinel_values(carbon[:, index].shape, (index + 1) * 1000.0)
    doc = np.empty_like(np.asarray(entry.DOC), dtype=np.float64)
    for index in range(doc.shape[3]):
        doc[:, :, :, index] = _sentinel_values(
            doc[:, :, :, index].shape, (index + 1) * 10000.0
        )
    changed_entry = entry._replace(carbon_32l=carbon, DOC=doc)
    changed_stomate = replace(original.stomate_readstart, entry_state=changed_entry)
    state = PaperRestartBundleState(
        driver=read_dim2_driver_restart(run / "driver_start.nc"),
        sechiba=original.sechiba_restart_state,
        stomate=changed_stomate,
    )
    physical = PaperRestartBundlePhysicalState(
        driver=read_restart_physical_state(run / "driver_start.nc"),
        sechiba=read_restart_physical_state(run / "sechiba_start.nc"),
        stomate=read_restart_physical_state(run / "stomate_start.nc"),
    )

    with tempfile.TemporaryDirectory(prefix="orchjax_restart_alias_") as temporary:
        report = write_paper_restart_start_bundle(
            Path(temporary) / "next_year",
            state=state,
            physical_state=physical,
            pft_layout=original.pft_layout,
        )
        restored = reference_case_first_step_restart_state(
            CONFIG,
            root=ROOT,
            run_dir=report.output_directory,
            stomate_filename="stomate_start.nc",
        )
        restored_entry = restored.stomate_readstart.entry_state
        comparisons = {
            "carbon_32l": _comparison(carbon, restored_entry.carbon_32l),
            "DOC": _comparison(doc, restored_entry.DOC),
        }
        expected_aliases = project_packet_aliases({"carbon_32l": carbon, "DOC": doc})
        restored_aliases = project_packet_aliases(
            {"carbon_32l": restored_entry.carbon_32l, "DOC": restored_entry.DOC}
        )
        alias_comparisons = {
            name: _comparison(expected_aliases[name], restored_aliases[name])
            for name in expected_aliases
        }
        with Dataset(run / "stomate_start.nc") as template, Dataset(
            report.stomate.output_path
        ) as generated:
            metadata = {}
            for name in expected_aliases:
                expected_metadata = _variable_metadata(template, name)
                actual_metadata = _variable_metadata(generated, name)
                raw = generated.variables[name][:]
                metadata[name] = {
                    "passed": expected_metadata == actual_metadata,
                    "template": expected_metadata,
                    "generated": actual_metadata,
                    "masked_value_count": int(np.ma.count_masked(raw)),
                    "val_exp_count": int(np.count_nonzero(np.asarray(raw) == VAL_EXP)),
                }

    mixed = set(restored.stomate_readstart.report.mixed_sentinel_fields)
    checks = {
        "canonical_exact": all(item["passed"] for item in comparisons.values()),
        "alias_members_exact": all(item["passed"] for item in alias_comparisons.values()),
        "metadata_exact": all(item["passed"] for item in metadata.values()),
        "sentinels_preserved": all(item["val_exp_count"] > 0 for item in metadata.values()),
        "sentinels_not_netcdf_masked": all(
            item["masked_value_count"] == 0 for item in metadata.values()
        ),
        "readstart_mixed_status": {"carbon_32l", "DOC"}.issubset(mixed),
        "production_writer_covered_fields": {"carbon_32l", "DOC"}.issubset(
            report.stomate.written_fields
        ),
    }
    return {
        "passed": all(checks.values()),
        "path": "production three-file bundle writer -> NetCDF -> production restart reader",
        "writer": "jax_orchidee.driver.restart_bundle.write_paper_restart_start_bundle",
        "reader": "jax_orchidee.driver.restart_state.reference_case_first_step_restart_state",
        "checks": checks,
        "canonical_comparisons": comparisons,
        "alias_comparisons": alias_comparisons,
        "variable_metadata": metadata,
        "sentinel": {"value": VAL_EXP, "semantics": "mixed val_exp is preserved exactly"},
    }


def build_audit(scan_path: Path = DEFAULT_SCAN) -> dict[str, object]:
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    scan_records: dict[str, list[dict[str, object]]] = {}
    for record in scan["records"]:
        scan_records.setdefault(record["name"], []).append(record)

    groups = many_to_one_alias_groups()
    records: list[dict[str, object]] = []
    for canonical, aliases in groups.items():
        declaration = PACK_AXIS_DECLARATIONS.get(canonical)
        declared_aliases = set(declaration["aliases"]) if declaration is not None else set()
        axis_declared = declaration is not None and set(aliases) == declared_aliases
        record: dict[str, object] = {
            "canonical": canonical,
            "aliases": list(aliases),
            "pack_axis_declared": axis_declared,
            "source_calls": {
                alias: sorted({str(item["call"]) for item in scan_records.get(alias, ())})
                for alias in aliases
            },
        }
        if declaration is not None:
            record.update(
                {
                    "pack_members": [
                        {"alias": alias, "index": index}
                        for index, alias in enumerate(declaration["aliases"])
                    ],
                    **{key: value for key, value in declaration.items() if key != "aliases"},
                }
            )
        records.append(record)

    roundtrip = _run_production_roundtrip()
    undeclared = [record["canonical"] for record in records if not record["pack_axis_declared"]]
    closed = not undeclared and bool(roundtrip["passed"])
    return {
        "schema_version": 2,
        "source_alias_table": "scripts/dev/audit_stomate_restart_state_coverage.py::ALIASES",
        "source_scan": str(scan_path.relative_to(ROOT) if scan_path.is_relative_to(ROOT) else scan_path),
        "closed": closed,
        "summary": {
            "many_to_one_groups": len(records),
            "declared_pack_axes": len(records) - len(undeclared),
            "undeclared_pack_axes": len(undeclared),
            "real_restart_writer_roundtrips": int(roundtrip["passed"]),
            "alias_variables_compared": len(roundtrip["alias_comparisons"]),
        },
        "records": records,
        "production_roundtrip": roundtrip,
        "limitations": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, default=DEFAULT_SCAN)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "reference_mode" / "restart_alias_roundtrip.json",
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    audit = build_audit(args.scan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {args.output.resolve()} closed={audit['closed']}")
    return int(args.verify and not audit["closed"])


if __name__ == "__main__":
    raise SystemExit(main())
