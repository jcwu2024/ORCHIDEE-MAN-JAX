"""Audit no-scan hydrology endpoints and the canopy-debt coordinate on v6."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jax_orchidee.parameters.pft_catalog import load_pft_catalog
from research.daily_coarse_graining.daily_markov_contract import (
    daily_markov_contract_from_metadata,
    load_markov_shard,
)
from research.daily_coarse_graining.daily_operator.training_data import (
    DEFAULT_PFT_CATALOG_PATH,
)
from research.daily_coarse_graining.markov_dataset import (
    COHERENT_DATASET_SCHEMA_VERSION,
)
from research.daily_coarse_graining.typed_sidecar import (
    load_coherent_typed_sidecar_index,
    read_typed_sidecar_shard,
)

SCHEMA_VERSION = "hydrology_canopy_boundary_audit_v1"
SOURCE_RECONSTRUCTION_ATOL = 1.0e-10
SOURCE_RECONSTRUCTION_RTOL = 1.0e-12
ORCHIDEE_UNDEFINED_THRESHOLD = 0.5e20
WET_CANOPY_FIELD = "water.vevapwet"
SPLIT_NAMES = {
    ("train", "train"): "train_train",
    ("train", "validation"): "temporal_validation",
    ("validation", "train"): "spatial_validation",
    ("validation", "validation"): "joint_validation",
}


@dataclass(frozen=True)
class ReconstructionComparison:
    reconstructed: np.ndarray
    target: np.ndarray
    defined: np.ndarray


@dataclass(frozen=True)
class CanopyDebtArrays:
    start_debt: np.ndarray
    end_storage: np.ndarray
    end_debt: np.ndarray
    capacity: np.ndarray
    wet_canopy_evaporation: np.ndarray
    defined: np.ndarray


def parse_selection(value: str) -> tuple[str, int]:
    landpoint, separator, year = value.rpartition(":")
    if not separator or not landpoint:
        raise ValueError(f"selection must be LANDPOINT:YEAR, got {value!r}")
    parsed_year = int(year)
    if parsed_year < 1900 or parsed_year > 2200:
        raise ValueError(f"selection year is outside the supported range: {parsed_year}")
    return landpoint, parsed_year


def split_name(spatial_split: str, temporal_split: str) -> str:
    if spatial_split == "test" or temporal_split == "test":
        raise ValueError("sealed test selection is forbidden")
    try:
        return SPLIT_NAMES[(spatial_split, temporal_split)]
    except KeyError as error:
        raise ValueError(
            f"unsupported non-test split pair: {(spatial_split, temporal_split)}"
        ) from error


def _series_from_state(shard: Any, contract: Any, key: str) -> np.ndarray:
    leaves = {leaf.key: leaf for leaf in contract.state_leaves}
    leaf = leaves.get(key)
    if leaf is None or leaf.discrete:
        raise ValueError(f"continuous day-start leaf is missing: {key}")
    values = shard.state_trajectory[:-1, leaf.start : leaf.stop]
    values = values.reshape((shard.days, *leaf.shape))
    if not leaf.shape or leaf.shape[0] != 1:
        raise ValueError(f"day-start leaf lacks a singleton point axis: {key}")
    return np.asarray(values[:, 0], dtype=np.float64)


def _series_from_fast_target(shard: Any, contract: Any, key: str) -> np.ndarray:
    leaves = {leaf.key: leaf for leaf in contract.fast_day_target_leaves}
    leaf = leaves.get(key)
    if leaf is None:
        raise ValueError(f"fast-day endpoint leaf is missing: {key}")
    values = shard.fast_day_target[:, leaf.start : leaf.stop]
    values = values.reshape((shard.days, *leaf.shape))
    if not leaf.shape or leaf.shape[0] != 1:
        raise ValueError(f"fast-day endpoint leaf lacks a singleton point axis: {key}")
    return np.asarray(values[:, 0], dtype=np.float64)


def _static_value(shard: Any, contract: Any, name: str) -> np.ndarray:
    matches = tuple(
        leaf
        for leaf in contract.static_conditions.landpoint_static_leaves
        if leaf.name == name
    )
    if len(matches) != 1:
        raise ValueError(f"static registry requires exactly one {name!r} leaf")
    leaf = matches[0]
    return np.asarray(
        shard.landpoint_static[leaf.start : leaf.stop], dtype=np.float64
    ).reshape(leaf.shape)


def _defined(value: Any) -> np.ndarray:
    array = np.asarray(value)
    return np.isfinite(array) & (np.abs(array) < ORCHIDEE_UNDEFINED_THRESHOLD)


def _comparison_summary(comparison: ReconstructionComparison) -> dict[str, Any]:
    reconstructed = np.asarray(comparison.reconstructed, dtype=np.float64)
    target = np.asarray(comparison.target, dtype=np.float64)
    defined = np.asarray(comparison.defined, dtype=np.bool_)
    if reconstructed.shape != target.shape or defined.shape != target.shape:
        raise ValueError("source reconstruction comparison shapes do not match")
    selected = defined & _defined(target)
    count = int(np.count_nonzero(selected))
    if count == 0:
        raise ValueError("source reconstruction comparison has no defined values")
    nonfinite = selected & ~np.isfinite(reconstructed)
    finite = selected & np.isfinite(reconstructed)
    difference = np.abs(reconstructed[finite] - target[finite])
    tolerance = (
        SOURCE_RECONSTRUCTION_ATOL
        + SOURCE_RECONSTRUCTION_RTOL * np.abs(target[finite])
    )
    squared_error_sum = float(np.sum(difference * difference))
    maximum = float(np.max(difference)) if difference.size else None
    passed = not np.any(nonfinite) and bool(np.all(difference <= tolerance))
    return {
        "shape": list(target.shape),
        "value_count": count,
        "nonfinite_reconstructed_count": int(np.count_nonzero(nonfinite)),
        "maximum_absolute_error": maximum,
        "root_mean_squared_error": (
            None if not difference.size else float(np.sqrt(squared_error_sum / count))
        ),
        "squared_error_sum": squared_error_sum,
        "within_tolerance_count": int(np.count_nonzero(difference <= tolerance)),
        "passed": passed,
        "atol": SOURCE_RECONSTRUCTION_ATOL,
        "rtol": SOURCE_RECONSTRUCTION_RTOL,
    }


def summarize_comparisons(
    comparisons: Sequence[ReconstructionComparison],
) -> dict[str, Any]:
    if not comparisons:
        raise ValueError("cannot summarize an empty reconstruction panel")
    metrics = [_comparison_summary(item) for item in comparisons]
    count = sum(int(item["value_count"]) for item in metrics)
    squared_error_sum = sum(float(item["squared_error_sum"]) for item in metrics)
    maximum_values = [
        float(item["maximum_absolute_error"])
        for item in metrics
        if item["maximum_absolute_error"] is not None
    ]
    return {
        "point_year_count": len(comparisons),
        "passed_point_year_count": sum(bool(item["passed"]) for item in metrics),
        "failed_point_year_count": sum(not bool(item["passed"]) for item in metrics),
        "value_count": count,
        "nonfinite_reconstructed_count": sum(
            int(item["nonfinite_reconstructed_count"]) for item in metrics
        ),
        "maximum_absolute_error": max(maximum_values),
        "root_mean_squared_error": float(np.sqrt(squared_error_sum / count)),
        "passed": all(bool(item["passed"]) for item in metrics),
        "atol": SOURCE_RECONSTRUCTION_ATOL,
        "rtol": SOURCE_RECONSTRUCTION_RTOL,
    }


def reconstruct_hydrology_boundary_fields(
    *,
    mc: Any,
    mcl: Any,
    start_nroot: Any,
    end_nroot: Any,
    veget: Any,
    veget_max: Any,
    soiltile: Any,
    pref_soil_veg: Any,
    dz_mm: Any,
    zz_mm: Any,
    diaglev_m: Any,
    njsc: Any,
    is_tree: Any,
    end_stempdiag: Any,
    end_snow: Any,
    end_snowdz: Any,
    evap_bare_lim_ns: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Evaluate source kernels once at each daily endpoint, without a scan."""

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from jax_orchidee.sechiba.hydrol import (
        USDA_MCR,
        hydrol_calculate_temp_hydro_profile,
        hydrol_evap_bare_limit_grid_beta,
        hydrol_soil_layer_diagnostics,
        hydrol_stress_aggregates,
        hydrol_vegupd_static_state,
        hydrol_water_stress_diagnostics,
    )

    mc = jnp.asarray(mc, dtype=jnp.float64)
    mcl = jnp.asarray(mcl, dtype=jnp.float64)
    start_nroot = jnp.asarray(start_nroot, dtype=jnp.float64)
    end_nroot = jnp.asarray(end_nroot, dtype=jnp.float64)
    veget = jnp.asarray(veget, dtype=jnp.float64)
    veget_max = jnp.asarray(veget_max, dtype=jnp.float64)
    soiltile = jnp.asarray(soiltile, dtype=jnp.float64)
    pref_soil_veg_np = np.asarray(pref_soil_veg, dtype=np.int32)
    njsc = jnp.asarray(njsc, dtype=jnp.int32)
    is_tree = jnp.asarray(is_tree, dtype=jnp.bool_)
    npts, _nslm, nstm = mc.shape
    if mcl.shape != mc.shape or soiltile.shape != (npts, nstm):
        raise ValueError("hydrology boundary soil shapes drift")
    if start_nroot.shape != end_nroot.shape or end_nroot.shape[:1] != (npts,):
        raise ValueError("hydrology boundary nroot shapes drift")
    if veget.shape != veget_max.shape or veget.shape != end_nroot.shape[:2]:
        raise ValueError("hydrology boundary vegetation axes drift")
    if pref_soil_veg_np.shape != (veget.shape[1],):
        raise ValueError("pref_soil_veg must match the full PFT axis")

    vegtot = jnp.sum(veget_max, axis=1)
    vegupd = hydrol_vegupd_static_state(
        veget=veget,
        veget_max=veget_max,
        soiltile=soiltile,
        vegtot=vegtot,
        pref_soil_veg=pref_soil_veg_np,
    )
    mcr = jnp.asarray(USDA_MCR, dtype=jnp.float64)[njsc - 1]
    inferred_under = (vegupd.mask_soiltile > 0.0) & (mc[:, 0, :] < mcr[:, None])

    def stress(seed: Any, *, dynamic: bool) -> dict[str, Any]:
        nroot_work = seed
        by_tile = []
        for tile in range(nstm):
            layer = hydrol_soil_layer_diagnostics(
                mc=mc[:, :, tile],
                mcl=mcl[:, :, tile],
                dz_mm=dz_mm,
                njsc=njsc,
                peat_hydro=False,
                soil_tile_index=tile + 1,
            )
            result = hydrol_water_stress_diagnostics(
                layer=layer,
                nroot=nroot_work,
                vegetmax_soil_tile=vegupd.vegetmax_soil[:, :, tile],
                is_under_mcr=inferred_under[:, tile],
                njsc=njsc,
                peat_hydro=False,
                soil_tile_index=tile + 1,
                dyn_nroot_larix=dynamic,
                is_tree=is_tree,
                znt=diaglev_m,
                new_watstress=False,
            )
            nroot_work = result.nroot
            by_tile.append(result)
        us = jnp.stack(tuple(item.us for item in by_tile), axis=2)
        humrelv = jnp.stack(tuple(item.humrelv for item in by_tile), axis=2)
        vegstressv = jnp.stack(tuple(item.vegstressv for item in by_tile), axis=2)
        aggregate = hydrol_stress_aggregates(
            humrelv=humrelv,
            vegstressv=vegstressv,
            veget_max=veget_max,
        )
        return {
            "nroot": nroot_work,
            "us": us,
            "humrelv": humrelv,
            "humrel": aggregate.humrel,
        }

    from_end_nroot = stress(end_nroot, dynamic=False)
    one_boundary = stress(start_nroot, dynamic=True)
    temp_hydro = hydrol_calculate_temp_hydro_profile(
        stempdiag=end_stempdiag,
        zz_mm=zz_mm,
        diaglev_m=diaglev_m,
        snow=end_snow,
        snowdz=end_snowdz,
        ok_explicitsnow=True,
    )
    evap_bare_lim = hydrol_evap_bare_limit_grid_beta(
        evap_bare_lim_ns=evap_bare_lim_ns,
        vegtot=vegtot,
        soiltile=soiltile,
    )
    fields = {
        **{
            f"from_true_end_nroot.{name}": np.asarray(value, dtype=np.float64)
            for name, value in from_end_nroot.items()
            if name != "nroot"
        },
        **{
            f"from_start_nroot_one_boundary.{name}": np.asarray(value, dtype=np.float64)
            for name, value in one_boundary.items()
        },
        "from_end_stempdiag.temp_hydro": np.asarray(temp_hydro, dtype=np.float64),
        "from_tile_beta.evap_bare_lim": np.asarray(evap_bare_lim, dtype=np.float64),
    }
    metadata = {
        "inferred_under_mcr_count": int(np.count_nonzero(np.asarray(inferred_under))),
        "vegtot_maximum_absolute_error_from_one": float(
            np.max(np.abs(np.asarray(vegtot) - 1.0))
        ),
        "forcing_records_materialized": 0,
        "subdaily_state_scan_steps": 0,
        "daily_boundary_kernel_calls_per_endpoint": 1,
    }
    return fields, metadata


def analyze_hydrology_shard(
    shard: Any,
    contract: Any,
    *,
    is_tree: np.ndarray,
    selected_pft_indices: Sequence[int],
) -> tuple[dict[str, ReconstructionComparison], dict[str, Any]]:
    start_nroot = _series_from_state(
        shard, contract, "hydrol_previous_step_state.nroot"
    )
    end_nroot = _series_from_fast_target(
        shard, contract, "hydrol_previous_step_state.nroot"
    )
    start_nroot_defined = np.all(_defined(start_nroot), axis=(1, 2))
    safe_start_nroot = np.where(
        start_nroot_defined[:, None, None], start_nroot, end_nroot
    )
    days = shard.days
    njsc = np.full(
        (days,), int(_static_value(shard, contract, "njsc").reshape(-1)[0]), dtype=np.int32
    )
    soiltile_static = _static_value(shard, contract, "soiltile")
    soiltile = np.broadcast_to(soiltile_static, (days, *soiltile_static.shape))[:, 0]
    fields, metadata = reconstruct_hydrology_boundary_fields(
        mc=_series_from_fast_target(shard, contract, "hydrol_previous_step_state.mc"),
        mcl=_series_from_fast_target(shard, contract, "hydrol_previous_step_state.mcl"),
        start_nroot=safe_start_nroot,
        end_nroot=end_nroot,
        veget=_series_from_state(
            shard, contract, "slowproc_stomate_previous_step_state.veget"
        ),
        veget_max=_series_from_state(
            shard, contract, "slowproc_stomate_previous_step_state.veget_max"
        ),
        soiltile=soiltile,
        pref_soil_veg=np.take(
            _static_value(shard, contract, "pref_soil_veg"),
            np.asarray(selected_pft_indices, dtype=np.int32),
        ),
        dz_mm=_static_value(shard, contract, "hydrol_dz_mm"),
        zz_mm=_static_value(shard, contract, "hydrol_zz_mm"),
        diaglev_m=_static_value(shard, contract, "diaglev"),
        njsc=njsc,
        is_tree=is_tree,
        end_stempdiag=_series_from_fast_target(
            shard, contract, "thermosoil_previous_step_state.stempdiag"
        ),
        end_snow=_series_from_fast_target(
            shard, contract, "hydrol_previous_step_state.snow"
        ),
        end_snowdz=_series_from_fast_target(
            shard, contract, "hydrol_previous_step_state.snowdz"
        ),
        evap_bare_lim_ns=_series_from_fast_target(
            shard, contract, "hydrol_previous_step_state.evap_bare_lim_ns"
        ),
    )
    targets = {
        "nroot": end_nroot,
        "us": _series_from_fast_target(
            shard, contract, "hydrol_previous_step_state.us"
        ),
        "humrelv": _series_from_fast_target(
            shard, contract, "hydrol_previous_step_state.humrelv"
        ),
        "humrel": _series_from_fast_target(
            shard, contract, "hydrol_previous_step_state.humrel"
        ),
        "temp_hydro": _series_from_fast_target(
            shard, contract, "hydrol_previous_step_state.temp_hydro"
        ),
        "evap_bare_lim": _series_from_fast_target(
            shard, contract, "hydrol_previous_step_state.evap_bare_lim"
        ),
    }
    comparisons = {}
    for name, reconstructed in fields.items():
        field_name = name.rsplit(".", 1)[1]
        target = targets[field_name]
        mask = _defined(target)
        if name.startswith("from_start_nroot_one_boundary"):
            mask &= start_nroot_defined.reshape(
                (days, *(1 for _ in range(target.ndim - 1)))
            )
        comparisons[name] = ReconstructionComparison(
            reconstructed=reconstructed,
            target=target,
            defined=mask,
        )
    metadata["undefined_day_start_nroot_day_count"] = int(
        np.count_nonzero(~start_nroot_defined)
    )
    return comparisons, metadata


def analyze_canopy_debt_shard(
    shard: Any,
    contract: Any,
    typed_values: Mapping[str, Any],
    typed_defined: Mapping[str, Any],
) -> CanopyDebtArrays:
    state_leaves = {leaf.key: leaf for leaf in contract.state_leaves}
    qsintveg_leaf = state_leaves["hydrol_previous_step_state.qsintveg"]
    source_pft_indices = tuple(int(index) for index in qsintveg_leaf.selected_pft_indices)
    local_pft_indices = np.asarray(
        [local for local, source in enumerate(source_pft_indices) if source != 0],
        dtype=np.int32,
    )
    vegetation_source_indices = np.asarray(
        [source for source in source_pft_indices if source != 0], dtype=np.int32
    )
    if local_pft_indices.size == 0:
        raise ValueError("canopy-debt audit requires an active vegetation PFT")
    start = np.take(
        _series_from_state(shard, contract, "hydrol_previous_step_state.qsintveg"),
        local_pft_indices,
        axis=1,
    )
    end = np.take(
        _series_from_fast_target(shard, contract, "hydrol_previous_step_state.qsintveg"),
        local_pft_indices,
        axis=1,
    )
    veget = np.take(
        _series_from_state(
            shard, contract, "slowproc_stomate_previous_step_state.veget"
        ),
        local_pft_indices,
        axis=1,
    )
    lai = np.take(
        _series_from_state(shard, contract, "slowproc_stomate_previous_step_state.lai"),
        local_pft_indices,
        axis=1,
    )
    capacity = float(_static_value(shard, contract, "sechiba_qsint")) * veget * lai
    wet = np.take(
        np.asarray(typed_values[WET_CANOPY_FIELD], dtype=np.float64),
        vegetation_source_indices,
        axis=1,
    )
    wet_defined = np.take(
        np.asarray(typed_defined[WET_CANOPY_FIELD], dtype=np.bool_),
        vegetation_source_indices,
        axis=1,
    )
    defined = _defined(start) & _defined(end) & _defined(capacity) & wet_defined
    return CanopyDebtArrays(
        start_debt=np.maximum(-start, 0.0),
        end_storage=np.maximum(end, 0.0),
        end_debt=np.maximum(-end, 0.0),
        capacity=capacity,
        wet_canopy_evaporation=wet,
        defined=defined,
    )


def summarize_canopy_debt(records: Sequence[CanopyDebtArrays]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty canopy-debt panel")

    def merged(name: str) -> np.ndarray:
        return np.concatenate(
            [np.asarray(getattr(item, name)).reshape(-1) for item in records]
        )

    defined = merged("defined").astype(np.bool_)
    start_debt = merged("start_debt")[defined]
    storage = merged("end_storage")[defined]
    debt = merged("end_debt")[defined]
    capacity = merged("capacity")[defined]
    wet = merged("wet_canopy_evaporation")[defined]
    if not debt.size:
        raise ValueError("canopy-debt panel has no defined vegetation endpoints")
    tolerance = SOURCE_RECONSTRUCTION_ATOL + SOURCE_RECONSTRUCTION_RTOL * np.abs(wet)
    debt_positive = debt > SOURCE_RECONSTRUCTION_ATOL
    storage_positive = storage > SOURCE_RECONSTRUCTION_ATOL
    zero_endpoint = ~debt_positive & ~storage_positive
    positive_debt = debt[debt_positive]
    positive_wet = wet[debt_positive]
    incoming = np.maximum(debt - start_debt, 0.0)
    positive_incoming = incoming > SOURCE_RECONSTRUCTION_ATOL
    positive_debt_scale = start_debt + np.maximum(wet, 0.0)
    debt_scale_valid = debt_positive & (
        positive_debt_scale > SOURCE_RECONSTRUCTION_ATOL
    )
    debt_ratios = debt[debt_scale_valid] / positive_debt_scale[debt_scale_valid]
    incoming_scale_valid = positive_incoming & (wet > SOURCE_RECONSTRUCTION_ATOL)
    incoming_ratios = incoming[incoming_scale_valid] / wet[incoming_scale_valid]

    def quantiles(values: np.ndarray) -> dict[str, float | None]:
        if not values.size:
            return {
                "minimum": None,
                "median": None,
                "p95": None,
                "p99": None,
                "maximum": None,
            }
        return {
            "minimum": float(np.min(values)),
            "median": float(np.quantile(values, 0.5)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "maximum": float(np.max(values)),
        }

    return {
        "point_year_count": len(records),
        "defined_vegetation_endpoint_count": int(debt.size),
        "end_storage_positive_count": int(np.count_nonzero(storage_positive)),
        "end_storage_positive_fraction": float(np.mean(storage_positive)),
        "end_zero_count": int(np.count_nonzero(zero_endpoint)),
        "end_zero_fraction": float(np.mean(zero_endpoint)),
        "end_debt_positive_count": int(np.count_nonzero(debt_positive)),
        "end_debt_positive_fraction": float(np.mean(debt_positive)),
        "start_debt_positive_fraction": float(
            np.mean(start_debt > SOURCE_RECONSTRUCTION_ATOL)
        ),
        "debt_incoming_positive_fraction": float(np.mean(positive_incoming)),
        "end_debt": quantiles(positive_debt),
        "debt_incoming": quantiles(incoming[positive_incoming]),
        "daily_wet_evaporation_when_debt_positive": quantiles(positive_wet),
        "end_debt_fraction_of_start_debt_plus_daily_wet_evaporation": quantiles(
            debt_ratios
        ),
        "debt_incoming_fraction_of_daily_wet_evaporation": quantiles(
            incoming_ratios
        ),
        "positive_debt_with_nonpositive_daily_wet_evaporation_count": int(
            np.count_nonzero(debt_positive & (wet <= 0.0))
        ),
        "debt_incoming_exceeds_daily_wet_evaporation_count": int(
            np.count_nonzero(incoming > wet + tolerance)
        ),
        "end_debt_exceeds_start_debt_plus_daily_wet_evaporation_count": int(
            np.count_nonzero(debt > start_debt + wet + tolerance)
        ),
        "storage_exceeds_capacity_count": int(
            np.count_nonzero(storage > capacity + tolerance)
        ),
        "capacity_zero_with_positive_storage_count": int(
            np.count_nonzero((capacity <= 0.0) & (storage > SOURCE_RECONSTRUCTION_ATOL))
        ),
        "start_debt_plus_daily_wet_evaporation_is_valid_end_debt_scale": bool(
            not np.any(wet < -tolerance)
            and not np.any(debt > start_debt + wet + tolerance)
        ),
        "daily_wet_evaporation_is_valid_debt_incoming_scale": bool(
            not np.any(wet < -tolerance)
            and not np.any(incoming > wet + tolerance)
        ),
    }


def build_panel_report(
    *,
    dataset_manifest: Path,
    selections: Sequence[tuple[str, int]],
) -> dict[str, Any]:
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != COHERENT_DATASET_SCHEMA_VERSION:
        raise ValueError("hydrology/canopy audit requires the coherent v6 Teacher dataset")
    contract = daily_markov_contract_from_metadata(manifest["markov_contract"])
    index = load_coherent_typed_sidecar_index(dataset_manifest, verify_hashes=False)
    references = {
        (reference.parent.landpoint_id, int(reference.parent.year)): reference
        for reference in index.shards
    }
    if len(selections) != len(set(selections)):
        raise ValueError("hydrology/canopy panel selections must be unique")

    fast_leaves = {leaf.key: leaf for leaf in contract.fast_day_target_leaves}
    selected_pft_indices = tuple(
        int(index)
        for index in fast_leaves[
            "hydrol_previous_step_state.nroot"
        ].selected_pft_indices
    )
    if not selected_pft_indices:
        raise ValueError("hydrology audit requires explicit selected PFT indices")
    catalog = load_pft_catalog(DEFAULT_PFT_CATALOG_PATH)
    by_fortran_id = {entry.fortran_pft_id: entry for entry in catalog.entries}
    is_tree = np.asarray(
        [
            bool(by_fortran_id[source_index + 1].traits["is_tree"])
            for source_index in selected_pft_indices
        ],
        dtype=np.bool_,
    )

    comparisons: dict[str, list[ReconstructionComparison]] = {}
    comparisons_by_split: dict[str, dict[str, list[ReconstructionComparison]]] = {}
    canopy_records: list[CanopyDebtArrays] = []
    canopy_by_split: dict[str, list[CanopyDebtArrays]] = {}
    records = []
    for selection in selections:
        reference = references.get(selection)
        if reference is None:
            raise ValueError(f"v6 panel selection is missing: {selection}")
        parent = reference.parent
        group = split_name(parent.spatial_split, parent.temporal_split)
        shard = load_markov_shard(parent.path, contract=contract)
        typed_values, typed_defined, typed_day_index = read_typed_sidecar_shard(
            reference.path,
            contract=index.contract,
            expected_days=reference.day_count,
        )
        if not np.array_equal(typed_day_index, shard.day_index):
            raise ValueError(f"typed/base day alignment drift: {selection}")
        hydrology, metadata = analyze_hydrology_shard(
            shard,
            contract,
            is_tree=is_tree,
            selected_pft_indices=selected_pft_indices,
        )
        canopy = analyze_canopy_debt_shard(
            shard,
            contract,
            typed_values,
            typed_defined,
        )
        canopy_records.append(canopy)
        canopy_by_split.setdefault(group, []).append(canopy)
        split_comparisons = comparisons_by_split.setdefault(group, {})
        for name, comparison in hydrology.items():
            comparisons.setdefault(name, []).append(comparison)
            split_comparisons.setdefault(name, []).append(comparison)
        records.append(
            {
                "landpoint_id": parent.landpoint_id,
                "year": int(parent.year),
                "spatial_split": parent.spatial_split,
                "temporal_split": parent.temporal_split,
                "evaluation_split": group,
                "day_count": shard.days,
                "hydrology": {
                    name: _comparison_summary(comparison)
                    for name, comparison in hydrology.items()
                },
                "hydrology_metadata": metadata,
                "canopy_debt": summarize_canopy_debt((canopy,)),
            }
        )

    hydrology_summary = {
        name: summarize_comparisons(items) for name, items in comparisons.items()
    }
    exact_fields = sorted(
        name for name, metrics in hydrology_summary.items() if bool(metrics["passed"])
    )
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
            "typed_sidecar_values_read": True,
            "sealed_test_allowed": False,
            "forcing_records_materialized": 0,
            "subdaily_state_scan_steps": 0,
        },
        "source_configuration": {
            "dyn_nroot_larix": True,
            "new_watstress": False,
            "peat_hydrology": False,
            "use_refSOC_hydrol": False,
            "ok_explicitsnow": True,
        },
        "summary": {
            "hydrology": hydrology_summary,
            "hydrology_by_split": {
                group: {
                    name: summarize_comparisons(items)
                    for name, items in fields.items()
                }
                for group, fields in comparisons_by_split.items()
            },
            "exact_daily_boundary_fields": exact_fields,
            "canopy_debt": summarize_canopy_debt(canopy_records),
            "canopy_debt_by_split": {
                group: summarize_canopy_debt(items)
                for group, items in canopy_by_split.items()
            },
        },
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
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
