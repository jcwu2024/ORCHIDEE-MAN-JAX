"""Audit fast-owner endpoint reachability and source-exact reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_markov_contract import (
    daily_markov_contract_from_metadata,
    load_markov_shard,
)
from research.daily_coarse_graining.daily_operator.fast_owner_reconstruction import (
    load_fast_owner_reconstruction_contract,
)
from research.daily_coarse_graining.daily_operator.neural_architecture import (
    NeuralOutputSchema,
    OutputFieldSpec,
)
from research.daily_coarse_graining.daily_operator.successor_fast_owner_decoder import (
    DAY_START_DELTA,
    build_successor_fast_owner_schema,
)
from research.daily_coarse_graining.daily_operator.successor_fast_owner_statistics import (
    vectorized_successor_fast_owner_targets,
)
from research.daily_coarse_graining.daily_operator.training_data import (
    DailyProcessTrainingAdapter,
)
from research.daily_coarse_graining.daily_operator.training_statistics import (
    OUTPUT_MINIMUM_SCALE,
)
from research.daily_coarse_graining.typed_sidecar import (
    load_coherent_typed_sidecar_index,
)

SCHEMA_VERSION = "fast_owner_endpoint_reachability_audit_v2"
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
SOURCE_RECONSTRUCTION_ATOL = 1.0e-10
SOURCE_RECONSTRUCTION_RTOL = 1.0e-12


def _range_masks(values: np.ndarray, constraint: str) -> tuple[np.ndarray, np.ndarray]:
    """Return endpoint range violations and infinite-raw boundary targets."""

    if constraint == "nonnegative":
        return values < 0.0, values == 0.0
    if constraint == "unit_interval":
        return (values < 0.0) | (values > 1.0), (values == 0.0) | (values == 1.0)
    if constraint == "signed_unit":
        return (values < -1.0) | (values > 1.0), (values == -1.0) | (values == 1.0)
    if constraint == "signed":
        empty = np.zeros(values.shape, dtype=np.bool_)
        return empty, empty
    raise ValueError(f"unsupported delta constraint: {constraint}")


def _softplus_inverse(value: np.ndarray) -> np.ndarray:
    return value + np.log(-np.expm1(-value))


def _logit(value: np.ndarray) -> np.ndarray:
    return np.log(value) - np.log1p(-value)


def summarize_endpoint_reachability(
    deltas: Any,
    day_start: Any,
    defined: Any,
    *,
    mean: Any,
    scale: Any,
    constraint: str,
) -> dict[str, Any]:
    """Summarize targets against the actual endpoint-constrained decoder.

    The network is supervised on a signed delta, but ``output_transform`` adds
    the day-start owner before applying an absolute-state constraint. Negative
    deltas are therefore valid whenever the resulting endpoint is valid.
    """

    delta = np.asarray(deltas, dtype=np.float64)
    base = np.asarray(day_start, dtype=np.float64)
    valid = np.asarray(defined, dtype=np.bool_) & np.isfinite(delta) & np.isfinite(base)
    endpoint = base + delta
    selected_delta = delta[valid]
    selected_base = base[valid]
    selected_endpoint = endpoint[valid]
    if selected_delta.size == 0:
        raise ValueError("endpoint reachability audit field has no defined values")

    broadcast_mean = np.broadcast_to(np.asarray(mean, dtype=np.float64), delta.shape)[valid]
    broadcast_scale = np.broadcast_to(np.asarray(scale, dtype=np.float64), delta.shape)[valid]
    broadcast_scale = np.where(
        np.isfinite(broadcast_scale) & (broadcast_scale > 0.0),
        broadcast_scale,
        1.0,
    )
    hard, boundary = _range_masks(selected_endpoint, constraint)
    interior = ~(hard | boundary)
    required_raw = np.full(selected_delta.shape, np.nan, dtype=np.float64)
    center = selected_base + broadcast_mean
    if constraint == "nonnegative":
        positive_center = np.maximum(center, broadcast_scale * 1.0e-12)
        baseline = _softplus_inverse(positive_center / broadcast_scale)
        required_raw[interior] = (
            _softplus_inverse(selected_endpoint[interior] / broadcast_scale[interior]) - baseline[interior]
        )
    elif constraint == "unit_interval":
        bounded = np.clip(center, 1.0e-6, 1.0 - 1.0e-6)
        local_slope = np.maximum(bounded * (1.0 - bounded), 1.0e-6)
        required_raw[interior] = (
            (_logit(selected_endpoint[interior]) - _logit(bounded[interior]))
            * local_slope[interior]
            / broadcast_scale[interior]
        )
    elif constraint == "signed_unit":
        bounded = np.clip(center, -1.0 + 1.0e-6, 1.0 - 1.0e-6)
        local_slope = np.maximum(1.0 - bounded * bounded, 1.0e-6)
        required_raw[interior] = (
            (np.arctanh(selected_endpoint[interior]) - np.arctanh(bounded[interior]))
            * local_slope[interior]
            / broadcast_scale[interior]
        )
    elif constraint == "signed":
        required_raw[interior] = (selected_endpoint[interior] - center[interior]) / broadcast_scale[interior]
    else:
        raise ValueError(f"unsupported endpoint constraint: {constraint}")

    finite_raw = required_raw[np.isfinite(required_raw)]
    finite_raw_absolute = np.abs(finite_raw)
    raw_summary = (
        {
            "required_raw_absolute_median": None,
            "required_raw_absolute_p99": None,
            "required_raw_absolute_maximum": None,
        }
        if finite_raw.size == 0
        else {
            "required_raw_absolute_median": float(np.median(finite_raw_absolute)),
            "required_raw_absolute_p99": float(np.quantile(finite_raw_absolute, 0.99)),
            "required_raw_absolute_maximum": float(np.max(finite_raw_absolute)),
        }
    )
    delta_quantiles = np.quantile(selected_delta, QUANTILES)
    endpoint_quantiles = np.quantile(selected_endpoint, QUANTILES)
    return {
        "defined_value_count": int(selected_delta.size),
        "negative_delta_count": int(np.count_nonzero(selected_delta < 0.0)),
        "zero_delta_count": int(np.count_nonzero(selected_delta == 0.0)),
        "positive_delta_count": int(np.count_nonzero(selected_delta > 0.0)),
        "delta_quantiles": {
            str(quantile): float(value) for quantile, value in zip(QUANTILES, delta_quantiles, strict=True)
        },
        "endpoint_quantiles": {
            str(quantile): float(value) for quantile, value in zip(QUANTILES, endpoint_quantiles, strict=True)
        },
        "hard_range_impossible_count": int(np.count_nonzero(hard)),
        "hard_range_impossible_fraction": float(np.mean(hard)),
        "finite_raw_boundary_only_count": int(np.count_nonzero(boundary & ~hard)),
        "finite_raw_boundary_only_fraction": float(np.mean(boundary & ~hard)),
        "finite_required_raw_count": int(finite_raw.size),
        **raw_summary,
        "required_raw_absolute_over_20_count": int(np.count_nonzero(finite_raw_absolute > 20.0)),
    }


def _load_global_statistics(path: Path) -> tuple[Mapping[str, Any], Mapping[str, np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    arrays_path = path.parent / str(payload["statistics_npz"])
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return payload, arrays


def _normalization_summary(
    name: str,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    count = np.asarray(arrays[f"output__{name}__count"], dtype=np.uint64)
    mean = np.asarray(arrays[f"output__{name}__mean"], dtype=np.float64)
    variance = np.asarray(arrays[f"output__{name}__variance"], dtype=np.float64)
    scale = np.asarray(arrays[f"output__{name}__scale"], dtype=np.float64)
    raw_standard_deviation = np.sqrt(np.maximum(variance, 0.0))
    fallback = (count <= 1) | ~np.isfinite(raw_standard_deviation) | (raw_standard_deviation <= OUTPUT_MINIMUM_SCALE)
    return {
        "component_count": int(scale.size),
        "global_mean_minimum": float(np.min(mean)),
        "global_mean_median": float(np.median(mean)),
        "global_mean_maximum": float(np.max(mean)),
        "global_raw_standard_deviation_minimum": float(np.min(raw_standard_deviation)),
        "global_raw_standard_deviation_median": float(np.median(raw_standard_deviation)),
        "global_raw_standard_deviation_maximum": float(np.max(raw_standard_deviation)),
        "global_applied_scale_minimum": float(np.min(scale)),
        "global_applied_scale_median": float(np.median(scale)),
        "global_applied_scale_maximum": float(np.max(scale)),
        "minimum_scale_or_low_count_fallback_components": int(np.count_nonzero(fallback)),
        "minimum_scale_or_low_count_fallback_fraction": float(np.mean(fallback)),
        "minimum_scale": OUTPUT_MINIMUM_SCALE,
    }


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


def _source_reconstruction_report(shard: Any, contract: Any) -> dict[str, Any]:
    import jax

    jax.config.update("jax_enable_x64", True)
    from jax_orchidee.sechiba.condveg import condveg_main_minimal
    from jax_orchidee.sechiba.hydrol import (
        hydrol_layer_thickness_from_zz_mm,
        hydrol_litter_grid_diagnostics,
        hydrol_litter_top_diagnostics,
    )
    from jax_orchidee.sechiba.thermosoil import (
        thermosoil_diaglev_from_ptn,
        thermosoil_final_state,
    )

    fast_leaves = {leaf.key: leaf for leaf in contract.fast_day_target_leaves}
    static_leaves = {leaf.name: leaf for leaf in contract.static_conditions.landpoint_static_leaves}

    def fast(name: str) -> np.ndarray:
        leaf = fast_leaves[name]
        value = shard.fast_day_target[:, leaf.start : leaf.stop].reshape((shard.days, *leaf.shape))
        if not leaf.shape or leaf.shape[0] != 1:
            raise ValueError(f"source reconstruction expects a singleton landpoint axis: {name}")
        return np.asarray(value[:, 0], dtype=np.float64)

    def static(name: str) -> np.ndarray:
        leaf = static_leaves[name]
        return np.asarray(shard.landpoint_static[leaf.start : leaf.stop], dtype=np.float64).reshape(leaf.shape)

    forcing_index = {name: index for index, name in enumerate(contract.native_forcing.fields)}
    forcing = np.asarray(shard.forcing_native, dtype=np.float64)
    measurement_heights = static("measurement_heights")
    frac_nobio = fast("diffuco_previous_step_state.frac_nobio")

    def reconstruct_condveg(record_index: int) -> Any:
        forcing_record = forcing[:, record_index]
        return condveg_main_minimal(
            snow=fast("hydrol_previous_step_state.snow"),
            snow_nobio=fast("hydrol_previous_step_state.snow_nobio"),
            snowrho=fast("hydrol_previous_step_state.snowrho"),
            snowdz=fast("hydrol_previous_step_state.snowdz"),
            veget=fast("diffuco_previous_step_state.veget"),
            veget_max=fast("diffuco_previous_step_state.veget_max"),
            frac_nobio=frac_nobio,
            totfrac_nobio=np.sum(frac_nobio, axis=1),
            zlev=np.full((shard.days,), measurement_heights[0], dtype=np.float64),
            height=fast("diffuco_previous_step_state.height"),
            temp_air=forcing_record[:, forcing_index["Tair"]],
            pb=forcing_record[:, forcing_index["PSurf"]] / 100.0,
            u=forcing_record[:, forcing_index["Wind_N"]],
            v=forcing_record[:, forcing_index["Wind_E"]],
            lai=fast("diffuco_previous_step_state.lai"),
            emis_scal=1.0,
            impaze=False,
            rough_dyn=True,
            ok_explicitsnow=True,
        )

    condveg = reconstruct_condveg(-1)
    condveg_fields = {
        name: _error_summary(
            getattr(condveg, name),
            fast(f"diffuco_previous_step_state.{name}"),
        )
        for name in ("z0m", "z0h", "roughheight", "roughheight_pft")
    }
    z0h_target = fast("diffuco_previous_step_state.z0h")
    z0h_forcing_record_sensitivity = []
    for record_index in range(forcing.shape[1]):
        result = _error_summary(
            reconstruct_condveg(record_index).z0h,
            z0h_target,
        )
        z0h_forcing_record_sensitivity.append(
            {
                "record_index": record_index,
                "record_time_seconds": float((record_index - 1) * contract.native_forcing.source_interval_seconds),
                **result,
            }
        )

    mc = fast("hydrol_previous_step_state.mc")
    dz_mm = static("hydrol_dz_mm")
    zz_mm = static("hydrol_zz_mm")
    dh_mm = hydrol_layer_thickness_from_zz_mm(zz_mm)
    njsc_value = int(static("njsc").reshape(-1)[0])
    njsc = np.full((shard.days,), njsc_value, dtype=np.int32)
    soiltile = np.broadcast_to(static("soiltile"), (shard.days, *static("soiltile").shape))
    soiltile = soiltile[:, 0, :]
    litter = hydrol_litter_top_diagnostics(
        mc=mc,
        dz_mm=dz_mm,
        dh_mm=dh_mm,
        njsc=njsc,
    )
    litter_grid = hydrol_litter_grid_diagnostics(
        litter=litter,
        soiltile=soiltile,
        vegtot=np.ones((shard.days,), dtype=np.float64),
        njsc=njsc,
        # k_litt shares this compound kernel but is irrelevant to drysoil_frac.
        k_lin_layer1=np.ones((50, shard.days), dtype=np.float64),
        ks=static("hydrol_cwrr_ks"),
    )
    drysoil = _error_summary(
        litter_grid.drysoil_frac,
        fast("diffuco_previous_step_state.drysoil_frac"),
    )
    ptn = fast("thermosoil_previous_step_state.ptn")
    deep_humidity = fast("thermosoil_previous_step_state.shum_ngrnd_permalong")
    thermosoil_final = thermosoil_final_state(
        ptn=ptn,
        pkappa=np.zeros_like(ptn),
        veget_max=fast("diffuco_previous_step_state.veget_max"),
        shum_ngrnd_permalong=deep_humidity,
    )
    thermosoil_fields = {
        "gtemp": _error_summary(
            thermosoil_final.gtemp,
            fast("thermosoil_previous_step_state.gtemp"),
        ),
        "tdeep": _error_summary(
            thermosoil_final.deeptemp_prof,
            fast("thermosoil_previous_step_state.tdeep"),
        ),
        "hsdeep": _error_summary(
            thermosoil_final.deephum_prof,
            fast("thermosoil_previous_step_state.hsdeep"),
        ),
        "stempdiag": _error_summary(
            thermosoil_diaglev_from_ptn(
                ptn=ptn,
                veget_max=fast("diffuco_previous_step_state.veget_max"),
                nslm=static("hydrol_zz_mm").size,
            ),
            fast("thermosoil_previous_step_state.stempdiag"),
        ),
    }
    return {
        "policy": {
            "forcing_records_materialized": 0,
            "subdaily_state_scan_steps": 0,
            "forcing_boundary": "final active native record only",
            "primitive_state": "true Teacher day-end primitives for reconstruction audit",
        },
        "condveg_roughness": {
            "source": "condveg.f90::condveg_main/condveg_z0cdrag_dyn",
            "jax_kernel": "jax_orchidee.sechiba.condveg.condveg_main_minimal",
            "fields": condveg_fields,
            "z0h_forcing_record_sensitivity": z0h_forcing_record_sensitivity,
            "passed": all(item["passed"] for item in condveg_fields.values()),
        },
        "hydrol_drysoil_fraction": {
            "source": "hydrol.f90::hydrol_diag_soil lines 9112-9155",
            "jax_kernels": [
                "jax_orchidee.sechiba.hydrol.hydrol_litter_top_diagnostics",
                "jax_orchidee.sechiba.hydrol.hydrol_litter_grid_diagnostics",
            ],
            **drysoil,
        },
        "thermosoil_final_diagnostics": {
            "source": "thermosoil.f90::thermosoil_main lines 1011-1032",
            "jax_kernels": [
                "jax_orchidee.sechiba.thermosoil.thermosoil_final_state",
                "jax_orchidee.sechiba.thermosoil.thermosoil_diaglev_from_ptn",
            ],
            "fields": thermosoil_fields,
            "passed": all(item["passed"] for item in thermosoil_fields.values()),
        },
    }


def build_report(
    *,
    dataset_manifest: Path,
    fast_statistics: Path | None,
    landpoint_id: str,
    year: int,
) -> dict[str, Any]:
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    contract = daily_markov_contract_from_metadata(manifest["markov_contract"])
    adapter = DailyProcessTrainingAdapter.from_contract(contract)
    index = load_coherent_typed_sidecar_index(dataset_manifest, verify_hashes=False)
    matches = tuple(
        reference
        for reference in index.shards
        if reference.parent.landpoint_id == landpoint_id and reference.parent.year == year
    )
    if len(matches) != 1:
        raise ValueError(f"expected one v6 shard for {landpoint_id}:{year}, found {len(matches)}")
    reference = matches[0]
    shard = load_markov_shard(reference.parent.path, contract=contract)
    owner_contract = load_fast_owner_reconstruction_contract(markov_contract=contract)
    base_schema = NeuralOutputSchema(
        fields=(
            OutputFieldSpec(
                name="audit.direct_placeholder",
                shape=(),
                label_index=0,
                constraint="signed",
                value_representation="direct",
            ),
        ),
        labels=("audit.placeholder",),
        active_pft_indices=tuple(contract.active_pft_indices),
    )
    schema = build_successor_fast_owner_schema(
        base_schema,
        markov_contract=contract,
        owner_contract=owner_contract,
    )
    targets = vectorized_successor_fast_owner_targets(
        shard,
        schema=schema,
        adapter=adapter,
        owner_contract=owner_contract,
    )
    statistics_payload = None
    statistics_arrays = None
    if fast_statistics is not None:
        statistics_payload, statistics_arrays = _load_global_statistics(fast_statistics)

    fields = {}
    for field in schema.fields:
        if field.value_representation != DAY_START_DELTA:
            continue
        target = targets[field.name]
        start = int(field.day_start_state_start)
        stop = int(field.day_start_state_stop)
        day_start = shard.state_trajectory[:-1, start:stop].reshape((shard.days, *field.shape))
        start_defined = np.isfinite(day_start) & (np.abs(day_start) < 0.5e20)
        day_start = np.where(start_defined, day_start, 0.0)
        if statistics_arrays is None:
            mean = np.zeros(field.shape, dtype=np.float64)
            scale = np.ones(field.shape, dtype=np.float64)
        else:
            mean = np.asarray(
                statistics_arrays[f"output__{field.name}__mean"],
                dtype=np.float64,
            )
            scale = np.asarray(
                statistics_arrays[f"output__{field.name}__scale"],
                dtype=np.float64,
            )
        record = {
            "constraint_applied_to_reconstructed_endpoint": field.constraint,
            "shape": list(field.shape),
            "loss_label": schema.labels[field.label_index],
            **summarize_endpoint_reachability(
                target.value,
                day_start,
                target.defined,
                mean=mean,
                scale=scale,
                constraint=field.constraint,
            ),
        }
        if statistics_arrays is not None:
            record["train_train_global_normalization"] = _normalization_summary(
                field.name,
                statistics_arrays,
            )
        fields[field.name] = record

    total_values = sum(item["defined_value_count"] for item in fields.values())
    total_hard = sum(item["hard_range_impossible_count"] for item in fields.values())
    affected = [name for name, item in fields.items() if item["hard_range_impossible_count"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "dataset_id": index.parent_index.dataset_id,
            "teacher_git_head": index.parent_index.teacher_git_head,
            "landpoint_id": landpoint_id,
            "year": year,
            "day_count": shard.days,
            "spatial_split": reference.parent.spatial_split,
            "temporal_split": reference.parent.temporal_split,
        },
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_read": False,
            "typed_sidecar_values_read": False,
            "dataset_wide_hash_scan": False,
            "hard_range_impossible": ("reconstructed absolute endpoint lies outside the decoder transform range"),
            "finite_raw_boundary_only": (
                "absolute endpoint is reachable only as raw output tends to positive or negative infinity"
            ),
            "negative_delta_is_not_a_range_failure": True,
        },
        "global_training_statistics": (
            None
            if statistics_payload is None
            else {
                "schema_version": statistics_payload.get("schema_version"),
                "sample_count": statistics_payload.get("sample_count"),
                "selection": statistics_payload.get("selection"),
            }
        ),
        "summary": {
            "delta_field_count": len(fields),
            "defined_value_count": total_values,
            "hard_range_impossible_count": total_hard,
            "hard_range_impossible_fraction": total_hard / total_values,
            "affected_field_count": len(affected),
            "affected_fields": affected,
        },
        "source_reconstruction": _source_reconstruction_report(shard, contract),
        "fields": fields,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path)
    parser.add_argument("--landpoint-id", default="001.0-071.0")
    parser.add_argument("--year", type=int, default=1961)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = build_report(
        dataset_manifest=args.dataset_manifest,
        fast_statistics=args.fast_statistics,
        landpoint_id=args.landpoint_id,
        year=args.year,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
