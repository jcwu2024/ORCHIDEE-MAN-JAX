"""Diagnose spatial coverage and learned use of exogenous conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import numpy as np

from research.daily_coarse_graining.canonical_daily_model import (
    CanonicalDayBatch,
    canonical_model_apply,
)
from research.daily_coarse_graining.canonical_rollout import _load_neural_checkpoint
from research.daily_coarse_graining.canonical_training import (
    fast_day_target_representation_from_contract,
    loss_weights_from_contract,
    prepare_canonical_batch,
)
from research.daily_coarse_graining.canonical_training_run import (
    load_contract_metadata,
    verify_training_acceptance,
)
from research.daily_coarse_graining.daily_markov_contract import load_markov_shard
from research.daily_coarse_graining.markov_dataset import (
    MarkovDatasetIndex,
    MarkovShardRef,
    collate_samples,
    defined_numeric_mask,
    load_dataset_index,
    load_training_statistics,
)

SCHEMA_VERSION = "spatial_conditioning_diagnostic_v1"
CONDITION_GROUPS = (
    "parameters",
    "landpoint_static",
    "annual_conditions",
    "forcing_native",
)
CONDITION_FINITE_FIELDS = {
    "parameters": "parameters_finite",
    "landpoint_static": "landpoint_static_finite",
    "annual_conditions": "annual_conditions_finite",
    "forcing_native": "forcing_finite",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _split_by_landpoint(index: MarkovDatasetIndex) -> dict[str, str]:
    result: dict[str, str] = {}
    for reference in index.shards:
        previous = result.setdefault(reference.landpoint_id, reference.spatial_split)
        if previous != reference.spatial_split:
            raise ValueError(f"spatial split drift for {reference.landpoint_id}")
    return result


def _training_references(index: MarkovDatasetIndex) -> dict[str, list[MarkovShardRef]]:
    grouped: dict[str, list[MarkovShardRef]] = defaultdict(list)
    for reference in index.select(temporal_split="train"):
        if reference.spatial_split == "test":
            continue
        grouped[reference.landpoint_id].append(reference)
    for references in grouped.values():
        references.sort(key=lambda item: item.year)
    if not grouped:
        raise ValueError("diagnostic has no unsealed temporal-training shards")
    return dict(grouped)


def _finite_summary(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape((-1, values.shape[-1]))
    finite = defined_numeric_mask(values)
    count = np.sum(finite, axis=0)
    safe = np.where(finite, values, 0.0)
    mean = np.divide(
        np.sum(safe, axis=0),
        count,
        out=np.zeros(values.shape[-1], dtype=np.float64),
        where=count > 0,
    )
    centered = np.where(finite, values - mean, 0.0)
    variance = np.divide(
        np.sum(centered * centered, axis=0),
        count,
        out=np.zeros(values.shape[-1], dtype=np.float64),
        where=count > 0,
    )
    return np.concatenate((mean, np.sqrt(np.maximum(variance, 0.0))))


def _landpoint_features(
    references: Mapping[str, Sequence[MarkovShardRef]],
) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for landpoint_id, point_references in references.items():
        forcing = []
        parameters = None
        landpoint_static = None
        for reference in point_references:
            shard = load_markov_shard(reference.path)
            forcing.append(np.asarray(shard.forcing_native))
            if parameters is None:
                parameters = np.asarray(shard.parameters, dtype=np.float64)
                landpoint_static = np.asarray(shard.landpoint_static, dtype=np.float64)
            else:
                np.testing.assert_array_equal(parameters, shard.parameters)
                np.testing.assert_array_equal(landpoint_static, shard.landpoint_static)
        result[landpoint_id] = {
            "parameters": np.asarray(parameters),
            "landpoint_static": np.asarray(landpoint_static),
            "forcing_climatology": _finite_summary(np.concatenate(forcing, axis=0)),
        }
    return result


def _normalize_feature_group(
    features: Mapping[str, np.ndarray], train_ids: Sequence[str]
) -> tuple[dict[str, np.ndarray], int]:
    ids = tuple(features)
    matrix = np.stack([np.asarray(features[name], dtype=np.float64) for name in ids])
    finite = defined_numeric_mask(matrix)
    train_rows = np.stack([matrix[ids.index(name)] for name in train_ids])
    train_finite = defined_numeric_mask(train_rows)
    fully_observed = np.all(train_finite, axis=0)
    mean = np.mean(np.where(train_finite, train_rows, 0.0), axis=0)
    scale = np.std(np.where(train_finite, train_rows, mean), axis=0)
    active = fully_observed & np.isfinite(scale) & (scale > 1.0e-12)
    if not np.any(active):
        return {name: np.zeros((0,), dtype=np.float64) for name in ids}, 0
    normalized = (matrix[:, active] - mean[active]) / scale[active]
    normalized = np.where(finite[:, active], normalized, 0.0)
    return {name: normalized[index] for index, name in enumerate(ids)}, int(np.sum(active))


def _nearest_training_distances(
    features: Mapping[str, np.ndarray],
    train_ids: Sequence[str],
    split_by_id: Mapping[str, str],
) -> dict[str, Any]:
    train_ids = tuple(train_ids)
    if len(train_ids) < 2:
        raise ValueError("coverage diagnosis requires at least two training landpoints")

    def distance(left: str, right: str) -> float:
        delta = np.asarray(features[left]) - np.asarray(features[right])
        return float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0

    train_loo = {
        name: min(distance(name, other) for other in train_ids if other != name)
        for name in train_ids
    }
    reference = np.asarray(tuple(train_loo.values()), dtype=np.float64)
    median = float(np.median(reference))
    points = {}
    for name in features:
        candidates = tuple(other for other in train_ids if other != name)
        nearest_id = min(candidates, key=lambda other: (distance(name, other), other))
        nearest = distance(name, nearest_id)
        points[name] = {
            "spatial_split": split_by_id[name],
            "nearest_train_id": nearest_id,
            "nearest_train_rms_distance": nearest,
            "relative_to_train_loo_median": nearest / median if median > 0.0 else None,
            "above_all_train_loo": bool(nearest > float(np.max(reference))),
        }
    return {
        "train_leave_one_out": train_loo,
        "train_loo_median": median,
        "train_loo_maximum": float(np.max(reference)),
        "points": points,
    }


def build_coverage_report(
    raw_features: Mapping[str, Mapping[str, np.ndarray]],
    split_by_id: Mapping[str, str],
) -> dict[str, Any]:
    train_ids = tuple(
        sorted(name for name in raw_features if split_by_id[name] == "train")
    )
    groups = {}
    normalized_groups = {}
    for group in ("parameters", "landpoint_static", "forcing_climatology"):
        normalized, active = _normalize_feature_group(
            {name: values[group] for name, values in raw_features.items()}, train_ids
        )
        normalized_groups[group] = normalized
        groups[group] = {
            "active_feature_count": active,
            **_nearest_training_distances(normalized, train_ids, split_by_id),
        }
    combined = {
        name: np.concatenate(
            [
                normalized_groups[group][name]
                / np.sqrt(max(normalized_groups[group][name].size, 1))
                for group in normalized_groups
            ]
        )
        for name in raw_features
    }
    groups["combined"] = {
        "active_feature_count": int(next(iter(combined.values())).size),
        **_nearest_training_distances(combined, train_ids, split_by_id),
    }
    validation_ids = tuple(
        name for name in raw_features if split_by_id[name] == "validation"
    )
    outside_groups = [
        group
        for group in ("parameters", "landpoint_static", "forcing_climatology")
        if any(
            groups[group]["points"][name]["above_all_train_loo"]
            for name in validation_ids
        )
    ]
    return {
        "training_landpoints": list(train_ids),
        "evaluated_landpoints": sorted(raw_features),
        "groups": groups,
        "validation_outside_training_loo": bool(outside_groups),
        "validation_outside_training_loo_groups": outside_groups,
        "validation_outside_combined_training_loo": any(
            groups["combined"]["points"][name]["above_all_train_loo"]
            for name in validation_ids
        ),
    }


def _sample_keys(references: Mapping[str, Sequence[MarkovShardRef]]) -> tuple[tuple[int, int], ...]:
    common_years = set.intersection(
        *(set(reference.year for reference in values) for values in references.values())
    )
    if not common_years:
        raise ValueError("landpoints have no common temporal-training years")
    ordered = sorted(common_years)
    selected_years = sorted(
        {ordered[index] for index in np.linspace(0, len(ordered) - 1, 4, dtype=int)}
    )
    days = (15, 60, 105, 150, 195, 240, 285, 330)
    return tuple((year, day) for year in selected_years for day in days)


def _diagnostic_batch(
    references: Mapping[str, Sequence[MarkovShardRef]],
) -> tuple[dict[str, Any], tuple[str, ...], tuple[tuple[int, int], ...]]:
    keys = _sample_keys(references)
    samples = []
    sample_landpoints = []
    by_point = {
        name: {reference.year: reference for reference in point_references}
        for name, point_references in references.items()
    }
    cache = {}
    for year, day in keys:
        for landpoint_id in sorted(references):
            reference = by_point[landpoint_id][year]
            cache_key = (landpoint_id, year)
            if cache_key not in cache:
                cache[cache_key] = load_markov_shard(reference.path)
            shard = cache[cache_key]
            matches = np.flatnonzero(np.asarray(shard.day_index) == day)
            if matches.size != 1:
                raise ValueError(f"expected one sample for {landpoint_id}:{year}:day{day}")
            samples.append(shard.sample(int(matches[0])))
            sample_landpoints.append(landpoint_id)
    return collate_samples(samples), tuple(sample_landpoints), keys


def _weighted_huber_per_row(prediction, target, finite, weights) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    finite = np.asarray(finite, dtype=bool)
    error = np.where(finite, prediction - target, 0.0)
    absolute = np.abs(error)
    loss = np.where(absolute <= 1.0, 0.5 * error * error, absolute - 0.5)
    weighted = finite * np.asarray(weights, dtype=np.float64)[None, :]
    denominator = np.maximum(np.sum(weighted, axis=1), 1.0e-12)
    return np.sum(weighted * loss, axis=1) / denominator


def _replace_condition(
    batch: CanonicalDayBatch,
    group: str,
    values: np.ndarray,
    finite: np.ndarray | None = None,
) -> CanonicalDayBatch:
    if group not in CONDITION_GROUPS:
        raise ValueError(f"unknown condition group {group!r}")
    replacements = {group: values}
    if finite is not None:
        replacements[CONDITION_FINITE_FIELDS[group]] = finite
    return batch._replace(**replacements)


def _permutation_indices(sample_landpoints: Sequence[str]) -> np.ndarray:
    sample_landpoints = np.asarray(sample_landpoints)
    result = np.empty(sample_landpoints.size, dtype=np.int64)
    for start in range(0, sample_landpoints.size, len(set(sample_landpoints))):
        stop = start + len(set(sample_landpoints))
        block = np.arange(start, stop)
        if stop > sample_landpoints.size or len(set(sample_landpoints[block])) != block.size:
            raise ValueError("diagnostic samples are not aligned by date and landpoint")
        result[block] = np.roll(block, 1)
    return result


def build_condition_use_report(
    *,
    parameters,
    model_batch: CanonicalDayBatch,
    normalized_target: np.ndarray,
    target_finite: np.ndarray,
    weights: np.ndarray,
    sample_landpoints: Sequence[str],
    split_by_id: Mapping[str, str],
) -> dict[str, Any]:
    model = jax.jit(canonical_model_apply)
    full = np.asarray(
        jax.device_get(model(parameters, model_batch).normalized_fast_day_target)
    )
    full_loss = _weighted_huber_per_row(full, normalized_target, target_finite, weights)
    permutation = _permutation_indices(sample_landpoints)
    sample_splits = np.asarray([split_by_id[name] for name in sample_landpoints])

    def summarize(loss: np.ndarray, shift: np.ndarray) -> dict[str, Any]:
        result = {}
        for split in ("train", "validation"):
            selected = sample_splits == split
            result[split] = {
                "samples": int(np.sum(selected)),
                "mean_huber_loss": float(np.mean(loss[selected])),
                "loss_change_from_full": float(np.mean(loss[selected] - full_loss[selected])),
                "prediction_shift_rmse": float(
                    np.sqrt(np.mean(shift[selected] * shift[selected]))
                ),
            }
        return result

    report = {
        "full": summarize(full_loss, np.zeros_like(full)),
        "groups": {},
    }
    for group in CONDITION_GROUPS:
        values = np.asarray(getattr(model_batch, group))
        finite = np.asarray(getattr(model_batch, CONDITION_FINITE_FIELDS[group]))
        mean_prediction = np.asarray(
            jax.device_get(
                model(
                    parameters,
                    _replace_condition(model_batch, group, np.zeros_like(values)),
                ).normalized_fast_day_target
            )
        )
        permuted_prediction = np.asarray(
            jax.device_get(
                model(
                    parameters,
                    _replace_condition(
                        model_batch,
                        group,
                        values[permutation],
                        finite[permutation],
                    ),
                ).normalized_fast_day_target
            )
        )
        mean_loss = _weighted_huber_per_row(
            mean_prediction, normalized_target, target_finite, weights
        )
        permuted_loss = _weighted_huber_per_row(
            permuted_prediction, normalized_target, target_finite, weights
        )
        report["groups"][group] = {
            "train_mean_ablation": summarize(mean_loss, mean_prediction - full)["train"],
            "validation_mean_ablation": summarize(mean_loss, mean_prediction - full)[
                "validation"
            ],
            "train_cross_point_permutation": summarize(
                permuted_loss, permuted_prediction - full
            )["train"],
            "validation_cross_point_permutation": summarize(
                permuted_loss, permuted_prediction - full
            )["validation"],
        }
    return report


def classify_diagnostic(
    coverage: Mapping[str, Any], condition_use: Mapping[str, Any]
) -> dict[str, Any]:
    static = condition_use["groups"]["landpoint_static"]
    parameter = condition_use["groups"]["parameters"]
    spatial_condition_loss_change = max(
        static["validation_cross_point_permutation"]["loss_change_from_full"],
        parameter["validation_cross_point_permutation"]["loss_change_from_full"],
    )
    condition_used = spatial_condition_loss_change > 1.0e-4
    outside = bool(coverage["validation_outside_training_loo"])
    if outside and condition_used:
        decision = "spatial_coverage_is_primary"
    elif outside:
        decision = "spatial_coverage_and_condition_use_both_insufficient"
    elif condition_used:
        decision = "coverage_not_extreme_architecture_or_representation_remains"
    else:
        decision = "condition_channel_underused"
    return {
        "decision": decision,
        "validation_outside_training_leave_one_out_envelope": outside,
        "spatial_condition_channel_detectably_used": condition_used,
        "maximum_validation_condition_permutation_loss_change": float(
            spatial_condition_loss_change
        ),
        "policy": {
            "condition_use_threshold": 1.0e-4,
            "coverage_outlier": "combined nearest-train distance exceeds every train leave-one-out distance",
        },
    }


def run_diagnostic(args: argparse.Namespace) -> Mapping[str, Any]:
    dataset = args.dataset.resolve()
    statistics_path = args.statistics.resolve()
    acceptance = args.acceptance.resolve()
    checkpoint_path = args.checkpoint.resolve()
    verify_training_acceptance(acceptance, dataset, statistics_path)
    index = load_dataset_index(dataset, verify_hashes=True)
    split_by_id = _split_by_landpoint(index)
    references = _training_references(index)
    raw_features = _landpoint_features(references)
    coverage = build_coverage_report(raw_features, split_by_id)

    raw_batch, sample_landpoints, sample_keys = _diagnostic_batch(references)
    metadata = load_contract_metadata(dataset)
    representation = fast_day_target_representation_from_contract(metadata)
    statistics = load_training_statistics(statistics_path, index=index)
    prepared = prepare_canonical_batch(raw_batch, statistics, representation)
    checkpoint, model_config = _load_neural_checkpoint(
        checkpoint_path,
        dataset_path=dataset,
        statistics_path=statistics_path,
        acceptance_path=acceptance,
    )
    if prepared.model_input.state.shape[-1] != model_config.state_width:
        raise ValueError("diagnostic state width does not match checkpoint")
    parameters = jax.tree_util.tree_map(jax.numpy.asarray, checkpoint["parameters"])
    condition_use = build_condition_use_report(
        parameters=parameters,
        model_batch=prepared.model_input,
        normalized_target=prepared.normalized_fast_day_target,
        target_finite=prepared.fast_day_target_finite,
        weights=loss_weights_from_contract(metadata),
        sample_landpoints=sample_landpoints,
        split_by_id=split_by_id,
    )
    classification = classify_diagnostic(coverage, condition_use)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "dataset": {
            "id": index.dataset_id,
            "manifest": str(dataset),
            "manifest_sha256": _sha256_file(dataset),
            "sealed_test_used": False,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256_file(checkpoint_path),
            "training_objective": checkpoint["identity"].get("training_objective"),
        },
        "sampling": {
            "keys": [{"year": year, "day": day} for year, day in sample_keys],
            "samples_per_landpoint": len(sample_keys),
            "landpoints": sorted(references),
        },
        "coverage": coverage,
        "condition_use": condition_use,
        "classification": classification,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_diagnostic(args)
    _atomic_json(args.output.resolve(), result)
    print(json.dumps(result["classification"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
