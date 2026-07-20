from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.init import parse_run_def  # noqa: E402
from jax_orchidee.driver.restart import reference_restart_path  # noqa: E402
from jax_orchidee.stomate.carbon_kernels import (  # noqa: E402
    IAGRSAPPN,
    IAGRSAPST,
    IAGRHRTPN,
    IAGRHRTST,
    ICARBRES,
    IFRUIT,
    IHEARTABOVE,
    IHEARTBELOW,
    ILEAF,
    IROOT,
    ISAPABOVE,
    ISAPBELOW,
)
from jax_orchidee.stomate.reference import find_stomate_reference_files, read_restart_biomass_carbon  # noqa: E402


POOL_NAMES = {
    ILEAF: "leaf",
    ISAPABOVE: "sap_above",
    ISAPBELOW: "sap_below",
    IHEARTABOVE: "heart_above",
    IHEARTBELOW: "heart_below",
    IROOT: "root",
    IFRUIT: "fruit",
    ICARBRES: "reserve",
    IAGRSAPST: "agrsap_st",
    IAGRSAPPN: "agrsap_pn",
    IAGRHRTST: "agrhrt_st",
    IAGRHRTPN: "agrhrt_pn",
}

RUN_DEF_KEYS = (
    "DT_SECHIBA",
    "NVM",
    "NSTM",
    "SECHIBA_restart_in",
    "STOMATE_RESTART_FILEIN",
    "SECHIBA_restart_out",
    "STOMATE_RESTART_FILEOUT",
    "XIOS_ORCHIDEE_OK",
    "IMPOSE_VEG",
    "LAND_COVER_CHANGE",
    "STOMATE_OK_DGVM",
    "STOMATE_OK_STOMATE",
    "READ_LAI",
    "OK_LAIDEV__00014",
    "PFT_TO_MTC__00014",
    "PREF_SOIL_VEG__00014",
    "IS_PEAT__00014",
    "PEAT_HYDRO",
    "OK_PEAT",
    "PERMA_PEAT",
    "OK_LEAK",
    "TF_DOC",
    "RIVER_ROUTING",
    "TIDES",
    "READ_TIDE",
    "READ_SALINITY",
    "ROUGH_DYN",
    "OK_EXPLICITSNOW",
    "OK_FREEZE_CWRR",
    "TOPMODEL_NEW",
    "dyn_nroot_larix",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def _run_def_key_view(path: Path) -> dict[str, Any]:
    values = parse_run_def(path)
    return {
        "path": str(path),
        "key_count": len(values),
        "selected": {key: values.get(key) for key in RUN_DEF_KEYS if key in values},
        "missing_selected": [key for key in RUN_DEF_KEYS if key not in values],
    }


def _run_def_diff(left: Path, right: Path) -> dict[str, Any]:
    left_values = parse_run_def(left)
    right_values = parse_run_def(right)
    common = sorted(set(left_values) & set(right_values))
    selected_diffs = {
        key: {"left": left_values.get(key), "right": right_values.get(key)}
        for key in RUN_DEF_KEYS
        if left_values.get(key) != right_values.get(key)
    }
    all_diffs = [key for key in common if left_values[key] != right_values[key]]
    return {
        "left": str(left),
        "right": str(right),
        "left_key_count": len(left_values),
        "right_key_count": len(right_values),
        "common_key_count": len(common),
        "all_common_diff_count": len(all_diffs),
        "selected_diffs": selected_diffs,
        "sample_common_diffs": {
            key: {"left": left_values[key], "right": right_values[key]}
            for key in all_diffs[:40]
        },
    }


def _var_meta(path: Path, names: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    with Dataset(path) as ds:
        for name in names:
            if name not in ds.variables:
                result[name] = {"present": False}
                continue
            var = ds.variables[name]
            result[name] = {
                "present": True,
                "dimensions": tuple(var.dimensions),
                "shape": tuple(int(size) for size in var.shape),
                "units": getattr(var, "units", None),
            }
    return result


def _history_time_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    with Dataset(path) as ds:
        if "time_counter" not in ds.variables:
            return {"path": str(path), "exists": True, "time_counter_present": False}
        values = np.asarray(ds.variables["time_counter"][:], dtype=float).reshape(-1)
    return {
        "path": str(path),
        "exists": True,
        "time_counter_present": True,
        "time_counter_len": int(values.size),
        "time_counter_values": values[:10].tolist(),
        "frequency_interpretation": (
            "daily_or_subdaily" if values.size > 300 else "yearly_or_sparse_history"
        ),
    }


def _drop_time(data):
    if "time" in data.dims:
        return data.isel(time=0)
    if "time_counter" in data.dims:
        return data.isel(time_counter=0)
    return data


def _flatten_yx(data) -> np.ndarray:
    data = _drop_time(data)
    if "y" in data.dims and "x" in data.dims:
        other = tuple(dim for dim in data.dims if dim not in ("y", "x"))
        arr = np.asarray(data.transpose("y", "x", *other).values)
        return arr.reshape((-1, *arr.shape[2:]))
    return np.asarray(data.values)


def _first_pft14_from_restart(path: Path, name: str) -> dict[str, Any]:
    if not path.exists():
        return {"present": False, "reason": "file_missing"}
    with xr.open_dataset(path, decode_times=False) as ds:
        if name not in ds.variables:
            return {"present": False, "reason": "variable_missing"}
        values = _flatten_yx(ds[name])
    flat = np.asarray(values)
    if flat.ndim == 1:
        first = float(flat.reshape(-1)[0])
        return {"present": True, "first_point": first}
    if flat.shape[-1] >= 14:
        pft14_values = flat[..., 13]
    elif flat.ndim >= 2 and flat.shape[1] >= 14:
        pft14_values = flat[:, 13]
    else:
        return {"present": True, "shape": tuple(int(size) for size in flat.shape), "pft14_available": False}
    return {
        "present": True,
        "shape": tuple(int(size) for size in flat.shape),
        "pft14_first_point": float(np.asarray(pft14_values).reshape(-1)[0]),
        "pft14_sum": float(np.asarray(pft14_values).sum()),
    }


def _biomass_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"present": False, "reason": "file_missing"}
    try:
        biomass = read_restart_biomass_carbon(path)
    except Exception as exc:  # pragma: no cover - diagnostic script
        return {"present": False, "reason": f"{type(exc).__name__}: {exc}"}
    pft14 = np.asarray(biomass[0, 13, :, 0], dtype=float)
    pools = {POOL_NAMES.get(index, f"pool_{index}"): float(pft14[index]) for index in range(pft14.size)}
    return {
        "present": True,
        "shape": tuple(int(size) for size in biomass.shape),
        "pft14_first_point_total": float(pft14.sum()),
        "pft14_first_point_pools": pools,
    }


def _restart_summary(path: Path) -> dict[str, Any]:
    return {
        "file": _file_info(path),
        "variables": _var_meta(
            path,
            (
                "biomass",
                "gpp_daily",
                "npp_daily",
                "maint_resp",
                "resp_growth",
                "PFTpresent",
                "sla_calc",
                "leaf_age",
                "leaf_frac",
                "age",
            ),
        )
        if path.exists()
        else {},
        "biomass": _biomass_summary(path),
        "pft14_fields": {
            name: _first_pft14_from_restart(path, name)
            for name in ("gpp_daily", "npp_daily", "maint_resp", "resp_growth", "PFTpresent", "sla_calc", "age")
        },
    }


def _sechiba_summary(path: Path) -> dict[str, Any]:
    return {
        "file": _file_info(path),
        "variables": _var_meta(path, ("veget", "veget_max", "lai", "frac_nobio", "soiltile", "soilcap", "soilflx"))
        if path.exists()
        else {},
        "pft14_fields": {
            name: _first_pft14_from_restart(path, name) for name in ("veget", "veget_max", "lai")
        },
    }


def _metric_delta(left: dict[str, Any], right: dict[str, Any], key_path: tuple[str, ...]) -> dict[str, Any]:
    def get(data: dict[str, Any]) -> Any:
        value: Any = data
        for key in key_path:
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        return value

    left_value = get(left)
    right_value = get(right)
    if left_value is None or right_value is None:
        return {"left": left_value, "right": right_value, "delta": None}
    return {"left": left_value, "right": right_value, "delta": float(right_value) - float(left_value)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit alignment among paper reference, server trace, and current runner assets.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "orchidee_man_250919.yaml")
    parser.add_argument("--server-package", type=Path, default=ROOT / "outputs" / "server_1961_trace_full_20260623")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "dev_truth_package_alignment.json")
    args = parser.parse_args()

    stomate_files = find_stomate_reference_files(ROOT)
    reference_run_def = stomate_files.run_dir / "run.def"
    server_run_def = args.server_package / "run" / "run.def"
    server_used_run_def = args.server_package / "run" / "used_run.def"
    reference_sechiba_start = reference_restart_path(args.config, "sechiba_start.nc")
    reference_sechiba_restart = reference_restart_path(args.config, "sechiba_restart.nc")
    server_sechiba_restart = args.server_package / "netcdf" / "sechiba_restart.nc"

    reference_start = _restart_summary(stomate_files.start)
    reference_restart = _restart_summary(stomate_files.restart)
    server_restart = _restart_summary(args.server_package / "netcdf" / "stomate_restart.nc")

    summary = {
        "purpose": (
            "Determine whether local reference files, existing server traces, and the current JAX runner share "
            "the same run-definition and initial-state package before using traces as parity truth."
        ),
        "current_runner_initial_state": {
            "mode": "reference_case_first_step_restart_state",
            "driver_start": str(reference_restart_path(args.config, "driver_start.nc")),
            "sechiba_start": str(reference_sechiba_start),
            "stomate_start": str(stomate_files.start),
            "note": (
                "prepare_paper_1961_driver_context reads reference/case_001_071 start files. Existing server traces "
                "come with netcdf/*restart.nc files and are not automatically day-level truth for this runner."
            ),
        },
        "packages": {
            "reference_case": {
                "run_dir": str(stomate_files.run_dir),
                "run_def": _run_def_key_view(reference_run_def),
                "stomate_start": reference_start,
                "stomate_restart": reference_restart,
                "sechiba_start": _sechiba_summary(reference_sechiba_start),
                "sechiba_restart": _sechiba_summary(reference_sechiba_restart),
                "history_1961": _history_time_summary(stomate_files.run_dir / "stomate_history_1961.nc"),
            },
            "server_trace_package": {
                "root": str(args.server_package),
                "manifest": str(args.server_package / "MANIFEST.txt"),
                "run_def": _run_def_key_view(server_run_def),
                "used_run_def": _run_def_key_view(server_used_run_def),
                "stomate_restart": server_restart,
                "sechiba_restart": _sechiba_summary(server_sechiba_restart),
                "history_1961": _history_time_summary(args.server_package / "netcdf" / "stomate_history_1961.nc"),
            },
        },
        "run_def_alignment": {
            "reference_vs_server_run_def": _run_def_diff(reference_run_def, server_run_def),
            "reference_vs_server_used_run_def": _run_def_diff(reference_run_def, server_used_run_def),
        },
        "restart_alignment_selected_deltas": {
            "reference_start_to_server_restart": {
                "biomass_pft14_total": _metric_delta(
                    reference_start, server_restart, ("biomass", "pft14_first_point_total")
                ),
                "gpp_daily_pft14": _metric_delta(
                    reference_start, server_restart, ("pft14_fields", "gpp_daily", "pft14_first_point")
                ),
                "npp_daily_pft14": _metric_delta(
                    reference_start, server_restart, ("pft14_fields", "npp_daily", "pft14_first_point")
                ),
                "age_pft14": _metric_delta(
                    reference_start, server_restart, ("pft14_fields", "age", "pft14_first_point")
                ),
            },
            "reference_start_to_reference_restart": {
                "biomass_pft14_total": _metric_delta(
                    reference_start, reference_restart, ("biomass", "pft14_first_point_total")
                ),
                "gpp_daily_pft14": _metric_delta(
                    reference_start, reference_restart, ("pft14_fields", "gpp_daily", "pft14_first_point")
                ),
                "npp_daily_pft14": _metric_delta(
                    reference_start, reference_restart, ("pft14_fields", "npp_daily", "pft14_first_point")
                ),
                "age_pft14": _metric_delta(
                    reference_start, reference_restart, ("pft14_fields", "age", "pft14_first_point")
                ),
            },
        },
        "interpretation": {
            "history_files": (
                "Both audited stomate_history_1961.nc files are sparse/yearly if time_counter_len is 1; they are "
                "not daily trace truth for day1/day2/day3 comparisons."
            ),
            "daily_trace_use": (
                "Use server STOMATE daily trace for parity only with a runner initialized from the same server package, "
                "or regenerate trace from the reference start package."
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_jsonable(summary), indent=2, sort_keys=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
