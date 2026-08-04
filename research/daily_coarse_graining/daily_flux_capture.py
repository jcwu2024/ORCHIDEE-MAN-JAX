from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "daily_flux_capture_v1"

CAPTURE_LABEL_PATHS: dict[str, tuple[str, ...]] = {
    "water.canopy_precipitation_partition": ("water.precip2canopy", "water.precip2ground"),
    "water.canopy_to_ground": ("water.canopy2ground",),
    "water.wet_canopy_evaporation": ("water.vevapwet",),
    "water.transpiration": ("water.transpir", "water.rootsink"),
    "water.bare_soil_evaporation": ("water.vevapnu", "water.vevapnu_ns"),
    "water.snow_sublimation": ("water.vevapsno", "water.subsnownobio"),
    "water.flood_evaporation": ("water.vevapflo",),
    "water.snow_phase_and_melt_transfer": (
        "water.snowmelt",
        "water.snowmelt_from_maxmass",
        "water.snow_melt_mass_by_layer",
        "water.snow_refreeze_mass_by_layer",
        "water.snow_liquid_percolation_by_interface",
    ),
    "water.soil_infiltration": ("water.soil_infiltration", "water.water2infilt"),
    "water.soil_vertical_transfer": ("water.wat_flux",),
    "water.runoff": ("water.runoff", "water.runoff_per_soil", "water.runoff2peat"),
    "water.drainage": ("water.drainage", "water.drainage_per_soil"),
    "water.routing_exchange": (
        "water.returnflow",
        "water.reinfiltration",
        "water.irrigation",
        "water.floodout",
    ),
    "carbon.litter_respiration": (
        "ok_leak.litter_respiration",
        "ok_leak.litter_flood_respiration",
    ),
    "carbon.litter_to_doc": ("ok_leak.litter_to_doc", "ok_leak.floodcarbon_input"),
    "carbon.poc_gross_decomposition": (
        "ok_leak.poc_gross_decomposition",
        "ok_leak.poc_flood_gross_decomposition",
    ),
    "carbon.doc_gross_decomposition": (
        "ok_leak.doc_gross_decomposition",
        "ok_leak.doc_flood_gross_decomposition",
    ),
    "carbon.doc_external_input": (
        "ok_leak.doc_to_topsoil",
        "ok_leak.doc_to_subsoil",
        "ok_leak.doc_precip2ground",
        "ok_leak.doc_precip2canopy",
        "ok_leak.dry_dep_canopy",
    ),
    "carbon.doc_export": (
        "ok_leak.doc_run",
        "ok_leak.doc_drain",
        "ok_leak.doc_flood",
        "ok_leak.doc_run_2_peat",
    ),
    "carbon.doc_free_adsorbed_equilibration": ("ok_leak.doc_free_to_adsorbed",),
    "carbon.doc_vertical_water_transport": ("ok_leak.doc_advective_interface",),
    "carbon.doc_vertical_diffusion": ("ok_leak.doc_diffusive_interface",),
    "carbon.cryoturbation_redistribution": (
        "ok_leak.cryoturbation_carbon_transfer",
        "ok_leak.cryoturbation_doc_transfer",
        "ok_leak.cryoturbation_litter_transfer",
    ),
    "carbon.perma_peat_redistribution": ("ok_leak.perma_peat_carbon_transfer",),
    "energy.net_radiation_integral": ("energy.netrad", "energy.netrad_pft"),
    "energy.sensible_heat_integral": ("energy.fluxsens", "energy.fluxsens_pft"),
    "energy.latent_heat_integral": (
        "energy.fluxlat",
        "energy.fluxsubli",
        "energy.lareva",
        "energy.larsub",
    ),
    "energy.ground_heat_integral": (
        "energy.pgflux",
        "energy.soilflx",
        "energy.soilflx_pft",
    ),
    "energy.phase_change_integral": (
        "energy.fusion",
        "energy.snow_melt_refreeze",
        "energy.snow_liquid_excess",
    ),
}


def _flatten(value: Any, prefix: str, arrays: dict[str, np.ndarray]) -> None:
    if isinstance(value, Mapping):
        for name, child in value.items():
            _flatten(child, f"{prefix}.{name}" if prefix else str(name), arrays)
        return
    if hasattr(value, "_asdict"):
        for name, child in value._asdict().items():
            _flatten(child, f"{prefix}.{name}" if prefix else name, arrays)
        return
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        raise TypeError(f"capture leaf {prefix!r} is not numeric: {array.dtype}")
    arrays[prefix] = array


def daily_flux_capture_arrays(labels: Any) -> dict[str, np.ndarray]:
    if labels is None or not hasattr(labels, "sechiba") or not hasattr(labels, "ok_leak"):
        raise TypeError("expected one complete DriverDailyFluxLabelsV1 value")
    arrays: dict[str, np.ndarray] = {}
    _flatten(labels.sechiba.water, "water", arrays)
    _flatten(labels.sechiba.energy, "energy", arrays)
    _flatten(labels.ok_leak, "ok_leak", arrays)
    return arrays


def validate_daily_flux_capture(
    labels: Any,
    *,
    inventory_path: str | Path,
    steps_per_day: int = 48,
) -> dict[str, Any]:
    arrays = daily_flux_capture_arrays(labels)
    inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    missing_labels = {
        entry["id"]: entry["capture"]["family"]
        for entry in inventory["labels"]
        if entry["availability"] == "missing_non_identifiable"
    }
    undeclared = sorted(set(missing_labels) - set(CAPTURE_LABEL_PATHS))
    stale = sorted(set(CAPTURE_LABEL_PATHS) - set(missing_labels))
    missing_paths = {
        label: [path for path in CAPTURE_LABEL_PATHS[label] if path not in arrays] for label in missing_labels
    }
    missing_paths = {label: paths for label, paths in missing_paths.items() if paths}
    step_axis_paths = sorted(
        path for path, array in arrays.items() if array.ndim and array.shape[0] == int(steps_per_day)
    )
    nan_paths = sorted(path for path, array in arrays.items() if array.dtype.kind in "fc" and np.isnan(array).any())
    infinite_paths = sorted(
        path for path, array in arrays.items() if array.dtype.kind in "fc" and np.isinf(array).any()
    )
    ok = not (undeclared or stale or missing_paths or step_axis_paths or infinite_paths)
    report = {
        "schema_version": SCHEMA_VERSION,
        "ok": ok,
        "inventory_schema_version": inventory["schema_version"],
        "required_capture_label_count": len(missing_labels),
        "captured_array_count": len(arrays),
        "undeclared_labels": undeclared,
        "stale_labels": stale,
        "missing_paths": missing_paths,
        "step_axis_paths": step_axis_paths,
        "nan_paths_requiring_defined_mask": nan_paths,
        "infinite_paths": infinite_paths,
        "arrays": {
            path: {"shape": list(array.shape), "dtype": str(array.dtype)} for path, array in sorted(arrays.items())
        },
    }
    if not ok:
        raise ValueError(f"daily flux capture failed inventory validation: {report}")
    return report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_daily_flux_capture(
    labels: Any,
    *,
    output_dir: str | Path,
    inventory_path: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validation = validate_daily_flux_capture(labels, inventory_path=inventory_path)
    arrays = daily_flux_capture_arrays(labels)
    serialized_arrays = dict(arrays)
    for path, array in arrays.items():
        if array.dtype.kind in "fc" and np.isnan(array).any():
            serialized_arrays[f"{path}__defined"] = ~np.isnan(array)
    arrays_path = output / "daily_flux_labels.npz"
    temporary_arrays = arrays_path.with_suffix(".npz.tmp")
    with temporary_arrays.open("wb") as handle:
        np.savez_compressed(handle, **serialized_arrays)
    temporary_arrays.replace(arrays_path)
    report = {
        **validation,
        "metadata": dict(metadata or {}),
        "capture_npz": arrays_path.name,
        "capture_npz_sha256": _sha256_file(arrays_path),
    }
    report_path = output / "capture_report.json"
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_report.replace(report_path)
    return report
