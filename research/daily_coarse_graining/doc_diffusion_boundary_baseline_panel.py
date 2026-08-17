"""Audit no-scan daily-boundary proxies for cumulative DOC diffusion."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_markov_contract import (
    daily_markov_contract_from_metadata,
    load_markov_shard,
)
from research.daily_coarse_graining.markov_dataset import (
    COHERENT_DATASET_SCHEMA_VERSION,
)
from research.daily_coarse_graining.typed_sidecar import (
    load_coherent_typed_sidecar_index,
    read_typed_sidecar_shard,
)

SCHEMA_VERSION = "doc_diffusion_boundary_baseline_panel_v1"
TARGET_FIELD = "ok_leak.doc_diffusive_interface"
START_DOC_KEY = "slowproc_stomate_previous_step_state.DOC"
START_TPROF_KEY = "thermosoil_previous_step_state.ptn"
END_DOC_KEY = "ok_leak.DOC"
END_TPROF_KEY = "thermosoil_previous_step_state.ptn"
PROXY_NAMES = ("day_start", "day_end", "endpoint_trapezoid")
SPLIT_NAMES = {
    ("train", "train"): "train_train",
    ("train", "validation"): "temporal_validation",
    ("validation", "train"): "spatial_validation",
    ("validation", "validation"): "joint_validation",
}
REQUIRED_SPLITS = tuple(SPLIT_NAMES.values())
DAILY_DOC_DIFFUSIVITY = 1.0e-5
BOUNDARY_BASELINE_GATE = {
    "selection_split": "train_train",
    "required_evaluation_splits": list(REQUIRED_SPLITS),
    "maximum_normalized_rmse": 1.0,
    "maximum_residual_variance_fraction": 0.75,
    "minimum_correlation": 0.70,
    "maximum_absolute_bias_over_target_rms": 0.25,
    "maximum_target_nonzero_normalized_rmse": 1.0,
}


@dataclass(frozen=True)
class ProxyArrays:
    target: np.ndarray
    defined: np.ndarray
    predictions: Mapping[str, np.ndarray]


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


def _paper_zf_soil_b() -> np.ndarray:
    from jax_orchidee.sechiba.diffuco import cwrr_vertical_soil_grid_from_params

    grid = cwrr_vertical_soil_grid_from_params(
        depth_max_h=2.0,
        depth_max_t=38.0,
        depth_topthickness=9.77517107e-4,
        depth_cstthickness=2.0,
        depth_geom=2.0,
        ratio_geom_below=1.05,
    )
    return np.concatenate((np.zeros((1,), dtype=np.float64), np.asarray(grid.zlt, dtype=np.float64)))


def _series_from_state(shard: Any, contract: Any, key: str) -> np.ndarray:
    leaves = {leaf.key: leaf for leaf in contract.state_leaves}
    leaf = leaves.get(key)
    if leaf is None or leaf.discrete:
        raise ValueError(f"continuous day-start leaf is missing: {key}")
    if not leaf.axis_names or leaf.axis_names[0] != "npts" or leaf.shape[0] != 1:
        raise ValueError(f"day-start leaf does not have a singleton point axis: {key}")
    values = shard.state_trajectory[:-1, leaf.start : leaf.stop]
    return values.reshape((shard.days, *leaf.shape))[:, 0]


def _series_from_fast_target(shard: Any, contract: Any, key: str) -> np.ndarray:
    leaves = {leaf.key: leaf for leaf in contract.fast_day_target_leaves}
    leaf = leaves.get(key)
    if leaf is None:
        raise ValueError(f"fast-day endpoint leaf is missing: {key}")
    if not leaf.axis_names or leaf.axis_names[0] != "npts" or leaf.shape[0] != 1:
        raise ValueError(f"fast-day endpoint leaf does not have a singleton point axis: {key}")
    values = shard.fast_day_target[:, leaf.start : leaf.stop]
    return values.reshape((shard.days, *leaf.shape))[:, 0]


@lru_cache(maxsize=1)
def _compiled_diffusion_flux():
    import jax

    jax.config.update("jax_enable_x64", True)
    from jax_orchidee.stomate.soilcarbon_kernels import soilcarbon_leak_doc_diffusion

    def flux(doc, tprof, zf_soil_b, dif_doc):
        return soilcarbon_leak_doc_diffusion(
            doc,
            tprof,
            zf_soil_b,
            dif_doc,
        ).doc_flux_diff

    return jax.jit(flux)


def boundary_diffusion_flux(
    doc: Any,
    tprof: Any,
    zf_soil_b: Any,
    *,
    daily_diffusivity: float = DAILY_DOC_DIFFUSIVITY,
) -> np.ndarray:
    doc = np.asarray(doc, dtype=np.float64)
    tprof = np.asarray(tprof, dtype=np.float64)
    if doc.ndim != 6 or tprof.shape != (doc.shape[0], doc.shape[2], doc.shape[1]):
        raise ValueError("DOC and temperature boundary shapes are inconsistent")
    diffusivity = np.full((doc.shape[0],), daily_diffusivity, dtype=np.float64)
    return np.asarray(
        _compiled_diffusion_flux()(doc, tprof, np.asarray(zf_soil_b, dtype=np.float64), diffusivity),
        dtype=np.float64,
    )


def compute_boundary_proxies(
    *,
    start_doc: Any,
    start_tprof: Any,
    end_doc: Any,
    end_tprof: Any,
    zf_soil_b: Any,
) -> dict[str, np.ndarray]:
    start = boundary_diffusion_flux(start_doc, start_tprof, zf_soil_b)
    end = boundary_diffusion_flux(end_doc, end_tprof, zf_soil_b)
    if start.shape != end.shape:
        raise ValueError("DOC boundary proxy shape drift")
    return {
        "day_start": start,
        "day_end": end,
        "endpoint_trapezoid": 0.5 * (start + end),
    }


def _core_metrics(prediction: Any, target: Any, selected: Any) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    selected = np.asarray(selected, dtype=np.bool_)
    selected &= np.isfinite(prediction) & np.isfinite(target) & (np.abs(target) < 0.5e20)
    count = int(np.count_nonzero(selected))
    if count == 0:
        return {
            "value_count": 0,
            "target_rms": None,
            "rmse": None,
            "normalized_rmse": None,
            "signed_bias": None,
            "absolute_bias_over_target_rms": None,
            "correlation": None,
            "residual_variance_fraction": None,
        }
    predicted = prediction[selected]
    observed = target[selected]
    residual = predicted - observed
    target_rms = float(np.sqrt(np.mean(observed * observed)))
    rmse = float(np.sqrt(np.mean(residual * residual)))
    target_variance = float(np.var(observed))
    predicted_variance = float(np.var(predicted))
    correlation = (
        float(np.corrcoef(predicted, observed)[0, 1])
        if count > 1 and target_variance > 0.0 and predicted_variance > 0.0
        else None
    )
    signed_bias = float(np.mean(residual))
    return {
        "value_count": count,
        "target_rms": target_rms,
        "rmse": rmse,
        "normalized_rmse": None if target_rms == 0.0 else rmse / target_rms,
        "signed_bias": signed_bias,
        "absolute_bias_over_target_rms": None if target_rms == 0.0 else abs(signed_bias) / target_rms,
        "correlation": correlation,
        "residual_variance_fraction": (
            None if target_variance == 0.0 else float(np.var(residual)) / target_variance
        ),
    }


def proxy_metrics(prediction: Any, target: Any, defined: Any) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    defined = np.asarray(defined, dtype=np.bool_)
    if prediction.shape != target.shape or defined.shape != target.shape:
        raise ValueError("DOC proxy, target, and mask shapes must match")
    finite_defined = defined & np.isfinite(target) & (np.abs(target) < 0.5e20)
    defined_count = int(np.count_nonzero(finite_defined))
    target_nonzero = finite_defined & (target != 0.0)
    prediction_nonzero = finite_defined & (prediction != 0.0)
    return {
        "defined": _core_metrics(prediction, target, finite_defined),
        "target_nonzero": _core_metrics(prediction, target, target_nonzero),
        "target_nonzero_fraction": (
            None if defined_count == 0 else float(np.count_nonzero(target_nonzero)) / defined_count
        ),
        "prediction_nonzero_fraction": (
            None if defined_count == 0 else float(np.count_nonzero(prediction_nonzero)) / defined_count
        ),
    }


def aggregate_proxy_arrays(records: Sequence[ProxyArrays]) -> dict[str, Any]:
    if not records:
        raise ValueError("DOC boundary proxy aggregation cannot be empty")
    target = np.concatenate([record.target.reshape(-1) for record in records])
    defined = np.concatenate([record.defined.reshape(-1) for record in records])
    return {
        name: proxy_metrics(
            np.concatenate([record.predictions[name].reshape(-1) for record in records]),
            target,
            defined,
        )
        for name in PROXY_NAMES
    }


def decide_boundary_baseline(slices: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    missing = [name for name in REQUIRED_SPLITS if name not in slices]
    if missing:
        return {
            "status": "incomplete_non_test_panel",
            "selected_proxy": None,
            "missing_splits": missing,
            "gate": dict(BOUNDARY_BASELINE_GATE),
        }
    train = slices["train_train"]
    eligible = [
        name
        for name in PROXY_NAMES
        if train[name]["defined"]["normalized_rmse"] is not None
    ]
    if not eligible:
        return {
            "status": "target_not_measurable",
            "selected_proxy": None,
            "missing_splits": [],
            "gate": dict(BOUNDARY_BASELINE_GATE),
        }
    selected_proxy = min(
        eligible,
        key=lambda name: (train[name]["defined"]["normalized_rmse"], PROXY_NAMES.index(name)),
    )
    checks = {}
    for split in REQUIRED_SPLITS:
        metrics = slices[split][selected_proxy]
        defined = metrics["defined"]
        nonzero = metrics["target_nonzero"]
        values = {
            "normalized_rmse": defined["normalized_rmse"],
            "residual_variance_fraction": defined["residual_variance_fraction"],
            "correlation": defined["correlation"],
            "absolute_bias_over_target_rms": defined["absolute_bias_over_target_rms"],
            "target_nonzero_normalized_rmse": nonzero["normalized_rmse"],
        }
        passed = (
            values["normalized_rmse"] is not None
            and values["normalized_rmse"] <= BOUNDARY_BASELINE_GATE["maximum_normalized_rmse"]
            and values["residual_variance_fraction"] is not None
            and values["residual_variance_fraction"]
            <= BOUNDARY_BASELINE_GATE["maximum_residual_variance_fraction"]
            and values["correlation"] is not None
            and values["correlation"] >= BOUNDARY_BASELINE_GATE["minimum_correlation"]
            and values["absolute_bias_over_target_rms"] is not None
            and values["absolute_bias_over_target_rms"]
            <= BOUNDARY_BASELINE_GATE["maximum_absolute_bias_over_target_rms"]
            and values["target_nonzero_normalized_rmse"] is not None
            and values["target_nonzero_normalized_rmse"]
            <= BOUNDARY_BASELINE_GATE["maximum_target_nonzero_normalized_rmse"]
        )
        checks[split] = {"passed": bool(passed), **values}
    qualified = all(check["passed"] for check in checks.values())
    return {
        "status": (
            "physical_boundary_plus_learned_residual_qualified"
            if qualified
            else "daily_cumulative_diffusion_remains_learned_primitive"
        ),
        "selected_proxy": selected_proxy,
        "missing_splits": [],
        "all_splits_passed": qualified,
        "checks": checks,
        "gate": dict(BOUNDARY_BASELINE_GATE),
    }


def analyze_shard(shard: Any, contract: Any, typed_values: Mapping[str, Any], typed_defined: Mapping[str, Any]) -> ProxyArrays:
    start_doc = _series_from_state(shard, contract, START_DOC_KEY)
    start_tprof = _series_from_state(shard, contract, START_TPROF_KEY)
    end_doc = _series_from_fast_target(shard, contract, END_DOC_KEY)
    end_tprof = _series_from_fast_target(shard, contract, END_TPROF_KEY)
    predictions = compute_boundary_proxies(
        start_doc=start_doc,
        start_tprof=start_tprof,
        end_doc=end_doc,
        end_tprof=end_tprof,
        zf_soil_b=_paper_zf_soil_b(),
    )
    target = np.asarray(typed_values[TARGET_FIELD], dtype=np.float64)
    defined = np.asarray(typed_defined[TARGET_FIELD], dtype=np.bool_)
    selected_pfts = np.asarray(contract.active_pft_indices, dtype=np.int32)
    target = np.take(target, selected_pfts, axis=1)
    defined = np.take(defined, selected_pfts, axis=1)
    if any(value.shape != target.shape for value in predictions.values()):
        raise ValueError("DOC boundary proxy does not match the compact typed target")
    return ProxyArrays(target=target, defined=defined, predictions=predictions)


def build_panel_report(*, dataset_manifest: Path, selections: Sequence[tuple[str, int]]) -> dict[str, Any]:
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != COHERENT_DATASET_SCHEMA_VERSION:
        raise ValueError("DOC boundary audit requires the coherent v6 Teacher dataset")
    contract = daily_markov_contract_from_metadata(manifest["markov_contract"])
    index = load_coherent_typed_sidecar_index(dataset_manifest, verify_hashes=False)
    references = {(reference.parent.landpoint_id, int(reference.parent.year)): reference for reference in index.shards}
    if len(selections) != len(set(selections)):
        raise ValueError("DOC boundary panel selections must be unique")

    arrays_by_split: dict[str, list[ProxyArrays]] = {name: [] for name in REQUIRED_SPLITS}
    all_arrays = []
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
        arrays = analyze_shard(shard, contract, typed_values, typed_defined)
        all_arrays.append(arrays)
        arrays_by_split[group].append(arrays)
        records.append(
            {
                "landpoint_id": parent.landpoint_id,
                "year": int(parent.year),
                "spatial_split": parent.spatial_split,
                "temporal_split": parent.temporal_split,
                "evaluation_split": group,
                "day_count": shard.days,
                "proxies": {
                    name: proxy_metrics(arrays.predictions[name], arrays.target, arrays.defined)
                    for name in PROXY_NAMES
                },
            }
        )

    slices = {
        name: aggregate_proxy_arrays(items)
        for name, items in arrays_by_split.items()
        if items
    }
    decision = decide_boundary_baseline(slices)
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
            "daily_boundary_kernel_calls_per_endpoint": 1,
        },
        "source_semantics": {
            "target": "sum of 48 source-ordered half-hour nonnegative interface flux magnitudes",
            "proxy": "one source-kernel call at a daily boundary with dif_doc=1e-5 day-1",
            "claim": "boundary explanatory baseline, not retained-exact reconstruction",
        },
        "summary": {
            "decision": decision,
            "all_selected": aggregate_proxy_arrays(all_arrays),
            "slices": slices,
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
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
