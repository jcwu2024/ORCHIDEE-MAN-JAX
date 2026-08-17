"""Audit exact THERMOSOIL coefficient reconstruction at the daily boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_markov_contract import (
    daily_markov_contract_from_metadata,
    load_markov_shard,
)
from research.daily_coarse_graining.typed_sidecar import (
    load_coherent_typed_sidecar_index,
)

SCHEMA_VERSION = "thermal_coefficient_exact_reconstruction_panel_v1"
THERMAL_COEFFICIENT_FIELDS = (
    "cgrnd",
    "dgrnd",
    "cgrnd_snow",
    "dgrnd_snow",
    "lambda_snow",
    "pcapa_en",
    "soilcap",
    "soilcap_pft",
    "soilflx",
    "soilflx_pft",
)
SOURCE_RECONSTRUCTION_ATOL = 1.0e-10
SOURCE_RECONSTRUCTION_RTOL = 1.0e-12


def _error_summary(reconstructed: Any, target: Any) -> dict[str, Any]:
    left = np.asarray(reconstructed, dtype=np.float64)
    right = np.asarray(target, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"source reconstruction shape drift: {left.shape} != {right.shape}")
    defined = np.isfinite(right) & (np.abs(right) < 0.5e20)
    if not np.any(defined):
        raise ValueError("source reconstruction target has no defined values")
    selected_left = left[defined]
    selected_right = right[defined]
    difference = np.abs(selected_left - selected_right)
    tolerance = SOURCE_RECONSTRUCTION_ATOL + SOURCE_RECONSTRUCTION_RTOL * np.abs(selected_right)
    return {
        "shape": list(left.shape),
        "value_count": int(selected_left.size),
        "undefined_target_count": int(left.size - selected_left.size),
        "nonfinite_reconstructed_count": int(np.count_nonzero(~np.isfinite(selected_left))),
        "maximum_absolute_error": float(np.max(difference)),
        "root_mean_squared_error": float(np.sqrt(np.mean(difference * difference))),
        "exact_value_count": int(np.count_nonzero(selected_left == selected_right)),
        "within_tolerance_count": int(np.count_nonzero(difference <= tolerance)),
        "passed": bool(np.all(difference <= tolerance)),
        "atol": SOURCE_RECONSTRUCTION_ATOL,
        "rtol": SOURCE_RECONSTRUCTION_RTOL,
    }


def parse_selection(value: str) -> tuple[str, int]:
    landpoint, separator, year = value.rpartition(":")
    if not separator or not landpoint:
        raise ValueError(f"selection must be LANDPOINT:YEAR, got {value!r}")
    parsed_year = int(year)
    if parsed_year < 1900 or parsed_year > 2200:
        raise ValueError(f"selection year is outside the supported range: {parsed_year}")
    return landpoint, parsed_year


def _paper_vertical_grid():
    from jax_orchidee.sechiba.diffuco import cwrr_vertical_soil_grid_from_params

    # Frozen pft14-daily-teacher-v6 paper configuration. The stored HYDROL
    # geometry is checked against this source-derived grid before use.
    return cwrr_vertical_soil_grid_from_params(
        depth_max_h=2.0,
        depth_max_t=38.0,
        depth_topthickness=9.77517107e-4,
        depth_cstthickness=2.0,
        depth_geom=2.0,
        ratio_geom_below=1.05,
    )


def reconstruct_thermal_coefficient_fields(
    *,
    ptn: Any,
    shum_ngrnd_permalong: Any,
    ref_soc: Any,
    mc: Any,
    mcl: Any,
    veget_max: Any,
    frac_nobio: Any,
    lai: Any,
    temp_sol: Any,
    temp_sol_pft: Any,
    snow: Any,
    snow_nobio: Any,
    snowrho: Any,
    snowtemp: Any,
    snowdz: Any,
    pressure_hpa: Any,
    njsc: Any,
    hydrol_zz_mm: Any,
    hydrol_dz_mm: Any,
    pref_soil_veg: Any,
    ext_coeff_vegetfrac: Any,
    model_interval_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute end-of-step coefficients from daily-boundary primitives.

    The leading axis is a batch of independent daily endpoints. No forcing
    expansion or recurrent subdaily transition is performed.
    """

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from jax_orchidee.sechiba.condveg import condveg_frac_snow
    from jax_orchidee.sechiba.hydrol import (
        HydrolThermosoilMoistureInputs,
        hydrol_layer_thickness_from_zz_mm,
        hydrol_shumdiag,
        hydrol_soilmoist_aggregates,
    )
    from jax_orchidee.sechiba.slowproc import slowproc_veget_explicit
    from jax_orchidee.sechiba.thermosoil import (
        thermosoil_coef_explicit_snow,
        thermosoil_final_state,
        thermosoil_getdiff_explicit,
        thermosoil_humlev,
        thermosoil_soilc_tempdiff_fraction,
    )

    ptn = jnp.asarray(ptn, dtype=jnp.float64)
    mc = jnp.asarray(mc, dtype=jnp.float64)
    mcl = jnp.asarray(mcl, dtype=jnp.float64)
    veget_max = jnp.asarray(veget_max, dtype=jnp.float64)
    frac_nobio = jnp.asarray(frac_nobio, dtype=jnp.float64)
    lai = jnp.asarray(lai, dtype=jnp.float64)
    hydrol_zz_mm = jnp.asarray(hydrol_zz_mm, dtype=jnp.float64)
    hydrol_dz_mm = jnp.asarray(hydrol_dz_mm, dtype=jnp.float64)
    pref_soil_veg = jnp.asarray(pref_soil_veg, dtype=jnp.int32)
    ext_coeff_vegetfrac = jnp.asarray(ext_coeff_vegetfrac, dtype=jnp.float64)
    njsc = jnp.asarray(njsc, dtype=jnp.int32)
    if ptn.ndim != 3 or mc.ndim != 3:
        raise ValueError("ptn and mc must have batched THERMOSOIL/HYDROL shapes")
    batch, _ngrnd, nvm = ptn.shape
    if mcl.shape != mc.shape or veget_max.shape != (batch, nvm):
        raise ValueError("thermal boundary PFT or HYDROL shapes drift")
    if pref_soil_veg.shape != (nvm,) or ext_coeff_vegetfrac.shape != (nvm,):
        raise ValueError("compact PFT static inputs must match the endpoint PFT axis")
    if njsc.shape != (batch,):
        raise ValueError("njsc must provide one soil class per daily endpoint")

    grid = _paper_vertical_grid()
    grid_znh_mm = jnp.asarray(grid.znh, dtype=jnp.float64) * 1000.0
    grid_dnh_mm = jnp.asarray(grid.dnh, dtype=jnp.float64) * 1000.0
    zz_error = float(jnp.max(jnp.abs(grid_znh_mm - hydrol_zz_mm)))
    dz_error = float(jnp.max(jnp.abs(grid_dnh_mm - hydrol_dz_mm)))
    if zz_error > 1.0e-10 or dz_error > 1.0e-10:
        raise ValueError("stored v6 HYDROL geometry does not match the frozen paper thermal grid")

    vegetation = slowproc_veget_explicit(
        lai=lai,
        frac_nobio=frac_nobio,
        veget_max=veget_max,
        pref_soil_veg=pref_soil_veg,
        ext_coeff_vegetfrac=ext_coeff_vegetfrac,
        nstm=mc.shape[2],
    )
    vegtot = 1.0 - vegetation.totfrac_nobio
    moisture = hydrol_soilmoist_aggregates(
        mc=mc,
        mcl=mcl,
        soiltile=vegetation.soiltile,
        vegtot=vegtot,
        dz_mm=hydrol_dz_mm,
    )
    dh_mm = hydrol_layer_thickness_from_zz_mm(hydrol_zz_mm)
    shumdiag = hydrol_shumdiag(
        mc=mc,
        soil_wet_ns=jnp.zeros_like(mc),
        soilmoist=moisture.soilmoist,
        soiltile=vegetation.soiltile,
        dh_mm=dh_mm,
        dz_mm=hydrol_dz_mm,
        njsc=njsc,
        peat_hydro=False,
    )
    pref_zero = pref_soil_veg - 1
    mc_layh_pft = jnp.take(moisture.mc_layh_s, pref_zero, axis=2)
    mcl_layh_pft = jnp.take(moisture.mcl_layh_s, pref_zero, axis=2)
    tmc_layh_pft = jnp.broadcast_to(moisture.soilmoist[:, :, None], (batch, mc.shape[1], nvm))
    thermal_moisture = HydrolThermosoilMoistureInputs(
        shumdiag_perma=shumdiag.shumdiag_perma,
        mc_layh=moisture.mc_layh,
        mcl_layh=moisture.mcl_layh,
        tmc_layh=moisture.soilmoist,
        mc_layh_pft=mc_layh_pft,
        mcl_layh_pft=mcl_layh_pft,
        tmc_layh_pft=tmc_layh_pft,
    )
    humlev = thermosoil_humlev(
        shumdiag_perma=thermal_moisture.shumdiag_perma,
        mc_layh=thermal_moisture.mc_layh,
        mcl_layh=thermal_moisture.mcl_layh,
        tmc_layh=thermal_moisture.tmc_layh,
        mc_layh_pft=thermal_moisture.mc_layh_pft,
        mcl_layh_pft=thermal_moisture.mcl_layh_pft,
        tmc_layh_pft=thermal_moisture.tmc_layh_pft,
        znt=grid.znt,
        zlt=grid.zlt,
        dz5=grid.dz5,
    )
    zx1, _zx2 = thermosoil_soilc_tempdiff_fraction(refsoc=ref_soc, nvm=nvm)
    getdiff = thermosoil_getdiff_explicit(
        ptn=ptn,
        njsc=njsc,
        veget_max=vegetation.veget_max,
        shum_ngrnd_permalong=shum_ngrnd_permalong,
        mc_layt=humlev.mc_layt,
        tmc_layt_pft=humlev.tmc_layt_pft,
        snowrho=snowrho,
        snowtemp=snowtemp,
        pb=pressure_hpa,
        dlt=grid.dlt,
        zx1=zx1,
        ok_laidev=jnp.zeros((nvm,), dtype=jnp.bool_),
        nslm=mc.shape[1],
    )
    final = thermosoil_final_state(
        ptn=ptn,
        pkappa=getdiff.pkappa,
        veget_max=vegetation.veget_max,
        shum_ngrnd_permalong=shum_ngrnd_permalong,
    )
    snow_fraction = condveg_frac_snow(
        snow=snow,
        snow_nobio=snow_nobio,
        snowrho=snowrho,
        snowdz=snowdz,
        ok_explicitsnow=True,
    )
    coefficient = thermosoil_coef_explicit_snow(
        ptn=ptn,
        ptn_pftmean=final.ptn_pftmean,
        pcapa=getdiff.pcapa,
        pkappa=getdiff.pkappa,
        pcapa_snow=getdiff.pcapa_snow,
        pkappa_snow=getdiff.pkappa_snow,
        snowdz=snowdz,
        snowtemp=snowtemp,
        temp_sol_new=temp_sol,
        temp_sol_new_pft=temp_sol_pft,
        veget_max=vegetation.veget_max,
        ok_laidev=jnp.zeros((nvm,), dtype=jnp.bool_),
        dlt=grid.dlt,
        dz1=grid.dz1,
        zlt=grid.zlt,
        dt_sechiba=model_interval_seconds,
        lambda_thermal=grid.lambda_thermal,
        frac_snow_veg=snow_fraction.frac_snow_veg,
        frac_snow_nobio=snow_fraction.frac_snow_nobio,
        totfrac_nobio=vegetation.totfrac_nobio,
    )
    fields = {
        "cgrnd": coefficient.soil.cgrnd,
        "dgrnd": coefficient.soil.dgrnd,
        "cgrnd_snow": coefficient.cgrnd_snow,
        "dgrnd_snow": coefficient.dgrnd_snow,
        "lambda_snow": coefficient.lambda_snow,
        "pcapa_en": getdiff.pcapa_en,
        "soilcap": coefficient.soilcap,
        "soilcap_pft": coefficient.soil.soilcap_pft,
        "soilflx": coefficient.soilflx,
        "soilflx_pft": coefficient.soil.soilflx_pft,
    }
    metadata = {
        "hydrol_zz_mm_maximum_absolute_error": zz_error,
        "hydrol_dz_mm_maximum_absolute_error": dz_error,
        "forcing_records_materialized": 0,
        "subdaily_state_scan_steps": 0,
        "source_sequence": [
            "slowproc_veget_explicit",
            "hydrol_soilmoist_aggregates",
            "hydrol_shumdiag",
            "thermosoil_humlev",
            "thermosoil_getdiff_explicit",
            "thermosoil_coef_explicit_snow",
        ],
    }
    return fields, metadata


def _compact_static_pft(value: np.ndarray, selected: Sequence[int]) -> np.ndarray:
    value = np.asarray(value)
    indices = np.asarray(selected, dtype=np.int32)
    if indices.size == 0:
        return value
    return value[indices]


def reconstruct_shard_thermal_coefficients(shard: Any, contract: Any) -> dict[str, Any]:
    fast_leaves = {leaf.key: leaf for leaf in contract.fast_day_target_leaves}
    static_leaves = {leaf.name: leaf for leaf in contract.static_conditions.landpoint_static_leaves}

    def fast(name: str) -> np.ndarray:
        leaf = fast_leaves[name]
        value = shard.fast_day_target[:, leaf.start : leaf.stop].reshape((shard.days, *leaf.shape))
        if not leaf.shape or leaf.shape[0] != 1:
            raise ValueError(f"thermal reconstruction expects singleton landpoint axis: {name}")
        return np.asarray(value[:, 0], dtype=np.float64)

    def static(name: str) -> np.ndarray:
        leaf = static_leaves[name]
        return np.asarray(shard.landpoint_static[leaf.start : leaf.stop], dtype=np.float64).reshape(leaf.shape)

    pft_leaf = fast_leaves["diffuco_previous_step_state.veget_max"]
    selected = pft_leaf.selected_pft_indices
    forcing_index = {name: index for index, name in enumerate(contract.native_forcing.fields)}
    forcing = np.asarray(shard.forcing_native, dtype=np.float64)
    pressure_hpa = forcing[:, -1, forcing_index["PSurf"]] / 100.0
    njsc = np.broadcast_to(static("njsc").reshape(-1)[0], (shard.days,)).astype(np.int32)
    reconstructed, metadata = reconstruct_thermal_coefficient_fields(
        ptn=fast("thermosoil_previous_step_state.ptn"),
        shum_ngrnd_permalong=fast("thermosoil_previous_step_state.shum_ngrnd_permalong"),
        ref_soc=fast("thermosoil_previous_step_state.refSOC"),
        mc=fast("hydrol_previous_step_state.mc"),
        mcl=fast("hydrol_previous_step_state.mcl"),
        veget_max=fast("diffuco_previous_step_state.veget_max"),
        frac_nobio=fast("diffuco_previous_step_state.frac_nobio"),
        lai=fast("diffuco_previous_step_state.lai"),
        temp_sol=fast("enerbil_previous_step_state.temp_sol"),
        temp_sol_pft=fast("diffuco_previous_step_state.temp_sol_pft"),
        snow=fast("hydrol_previous_step_state.snow"),
        snow_nobio=fast("hydrol_previous_step_state.snow_nobio"),
        snowrho=fast("hydrol_previous_step_state.snowrho"),
        snowtemp=fast("hydrol_previous_step_state.snowtemp"),
        snowdz=fast("hydrol_previous_step_state.snowdz"),
        pressure_hpa=pressure_hpa,
        njsc=njsc,
        hydrol_zz_mm=static("hydrol_zz_mm"),
        hydrol_dz_mm=static("hydrol_dz_mm"),
        pref_soil_veg=_compact_static_pft(static("pref_soil_veg"), selected),
        ext_coeff_vegetfrac=_compact_static_pft(static("ext_coeff_vegetfrac"), selected),
        model_interval_seconds=contract.native_forcing.model_interval_seconds,
    )
    targets = {
        "cgrnd": fast("thermosoil_previous_step_state.cgrnd"),
        "dgrnd": fast("thermosoil_previous_step_state.dgrnd"),
        "cgrnd_snow": fast("thermosoil_previous_step_state.cgrnd_snow"),
        "dgrnd_snow": fast("thermosoil_previous_step_state.dgrnd_snow"),
        "lambda_snow": fast("thermosoil_previous_step_state.lambda_snow"),
        "pcapa_en": fast("thermosoil_previous_step_state.pcapa_en"),
        "soilcap": fast("thermosoil_previous_step_state.soilcap"),
        "soilcap_pft": fast("enerbil_previous_step_state.soilcap_pft"),
        "soilflx": fast("thermosoil_previous_step_state.soilflx"),
        "soilflx_pft": fast("enerbil_previous_step_state.soilflx_pft"),
    }
    fields = {name: _error_summary(reconstructed[name], targets[name]) for name in THERMAL_COEFFICIENT_FIELDS}
    return {
        "fields": fields,
        "passed": all(item["passed"] for item in fields.values()),
        "metadata": metadata,
    }


def summarize_panel_fields(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    if not records:
        raise ValueError("thermal reconstruction panel cannot be empty")
    return {
        name: {
            "point_year_count": len(records),
            "passed_point_year_count": sum(bool(record["thermal_coefficients"]["fields"][name]["passed"]) for record in records),
            "failed_point_year_count": sum(not bool(record["thermal_coefficients"]["fields"][name]["passed"]) for record in records),
            "value_count": sum(int(record["thermal_coefficients"]["fields"][name]["value_count"]) for record in records),
            "maximum_absolute_error": max(float(record["thermal_coefficients"]["fields"][name]["maximum_absolute_error"]) for record in records),
            "maximum_root_mean_squared_error": max(float(record["thermal_coefficients"]["fields"][name]["root_mean_squared_error"]) for record in records),
        }
        for name in THERMAL_COEFFICIENT_FIELDS
    }


def build_panel_report(*, dataset_manifest: Path, selections: Sequence[tuple[str, int]]) -> dict[str, Any]:
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    contract = daily_markov_contract_from_metadata(manifest["markov_contract"])
    index = load_coherent_typed_sidecar_index(dataset_manifest, verify_hashes=False)
    references = {(reference.parent.landpoint_id, int(reference.parent.year)): reference for reference in index.shards}
    if len(selections) != len(set(selections)):
        raise ValueError("thermal reconstruction panel selections must be unique")
    records = []
    for selection in selections:
        reference = references.get(selection)
        if reference is None:
            raise ValueError(f"v6 panel selection is missing: {selection}")
        parent = reference.parent
        if parent.spatial_split == "test" or parent.temporal_split == "test":
            raise ValueError(f"sealed test selection is forbidden: {selection}")
        shard = load_markov_shard(parent.path, contract=contract)
        records.append(
            {
                "landpoint_id": parent.landpoint_id,
                "year": int(parent.year),
                "spatial_split": parent.spatial_split,
                "temporal_split": parent.temporal_split,
                "day_count": shard.days,
                "thermal_coefficients": reconstruct_shard_thermal_coefficients(shard, contract),
            }
        )
    fields = summarize_panel_fields(records)
    failures = [name for name, field in fields.items() if field["failed_point_year_count"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "dataset_id": index.parent_index.dataset_id,
            "teacher_git_head": index.parent_index.teacher_git_head,
            "point_year_count": len(records),
        },
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_read": False,
            "typed_sidecar_values_read": False,
            "sealed_test_allowed": False,
            "forcing_records_materialized": 0,
            "subdaily_state_scan_steps": 0,
        },
        "summary": {
            "status": "exact_candidates_passed" if not failures else "exact_candidates_failed",
            "field_count": len(THERMAL_COEFFICIENT_FIELDS),
            "failure_count": len(failures),
            "failures": failures,
        },
        "fields": fields,
        "records": records,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--selection", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = build_panel_report(
        dataset_manifest=args.dataset_manifest,
        selections=tuple(parse_selection(value) for value in args.selection),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
