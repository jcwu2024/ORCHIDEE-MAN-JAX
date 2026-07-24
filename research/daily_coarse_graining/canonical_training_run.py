"""Streamed training entry point for the canonical v4 fast-day operator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.canonical_daily_model import (
    CanonicalModelConfig,
    CanonicalModelParameters,
    canonical_model_apply,
    canonical_one_step_loss,
    initialize_canonical_model,
    parameter_count,
)
from research.daily_coarse_graining.canonical_training import (
    CanonicalTrainingBatch,
    FastDayTargetRepresentation,
    audit_fast_day_target_representation,
    fast_day_target_representation_from_contract,
    loss_weights_from_contract,
    model_config_from_batch,
    prepare_canonical_batch,
    restore_fast_day_prediction,
    restore_fast_day_target,
)
from research.daily_coarse_graining.daily_markov_contract import load_markov_shard
from research.daily_coarse_graining.markov_dataset import (
    DATASET_SCHEMA_VERSION,
    SPLITS,
    MarkovDatasetIndex,
    TrainingStatistics,
    collate_samples,
    fit_training_statistics,
    load_dataset_index,
    load_training_statistics,
    write_training_statistics,
)

CHECKPOINT_SCHEMA_VERSION = "canonical_fast_day_checkpoint_v3"
ACCEPTANCE_SCHEMA_VERSION = "canonical_daily_dataset_acceptance_v1"


class AdamState(NamedTuple):
    step: Any
    first: Any
    second: Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_contract_metadata(manifest_path: str | Path) -> Mapping[str, Any]:
    raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    expected = str(raw["markov_contract_sha256"])
    top_level = raw.get("markov_contract")
    if top_level is not None:
        if _sha256_json(top_level) != expected:
            raise ValueError("dataset Markov contract metadata hash mismatch")
        return top_level
    contracts = [item.get("markov_contract") for item in raw.get("shards", [])]
    contracts = [item for item in contracts if item is not None]
    if not contracts:
        raise ValueError("dataset manifest does not contain Markov contract metadata")
    first = contracts[0]
    if _sha256_json(first) != expected:
        raise ValueError("dataset Markov contract metadata hash mismatch")
    if any(item != first for item in contracts[1:]):
        raise ValueError("dataset manifest contains inconsistent Markov contracts")
    return first


def fit_statistics_asset(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    chunk_rows: int = 64,
) -> Path:
    index = load_dataset_index(manifest_path)
    statistics = fit_training_statistics(index, chunk_rows=chunk_rows)
    return write_training_statistics(statistics, output_path)


def _validate_complete_dataset_manifest(
    manifest_path: Path,
    index: MarkovDatasetIndex,
) -> dict[str, Any]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"training acceptance requires {DATASET_SCHEMA_VERSION!r}"
        )
    if raw.get("status") != "complete":
        raise ValueError("training acceptance requires a complete aggregate")
    if int(raw.get("shard_count", -1)) != len(index.shards):
        raise ValueError("dataset manifest shard count mismatch")
    landpoints = {reference.landpoint_id for reference in index.shards}
    years = {reference.year for reference in index.shards}
    if int(raw.get("landpoint_count", -1)) != len(landpoints):
        raise ValueError("dataset manifest landpoint count mismatch")
    if int(raw.get("year_count", -1)) != len(years):
        raise ValueError("dataset manifest year count mismatch")
    if not raw.get("plan_sha256"):
        raise ValueError("dataset manifest is missing its generation-plan hash")

    split_pairs: dict[str, int] = {}
    spatial_splits = set()
    temporal_splits = set()
    ordered_splits = ("train", "validation", "test")
    for spatial in ordered_splits:
        for temporal in ordered_splits:
            count = len(
                index.select(
                    spatial_split=spatial,
                    temporal_split=temporal,
                )
            )
            split_pairs[f"{spatial}/{temporal}"] = count
            if count:
                spatial_splits.add(spatial)
                temporal_splits.add(temporal)
    if spatial_splits != set(SPLITS) or temporal_splits != set(SPLITS):
        raise ValueError(
            "dataset must contain train, validation, and test spatial and temporal splits"
        )
    required_training_pairs = (
        "train/train",
        "train/validation",
        "validation/train",
        "validation/validation",
    )
    missing = [name for name in required_training_pairs if not split_pairs[name]]
    if missing:
        raise ValueError(f"dataset is missing required training split pairs: {missing}")
    return {
        "schema_version": raw["schema_version"],
        "status": raw["status"],
        "provisional_teacher": bool(raw.get("provisional_teacher", False)),
        "plan_sha256": str(raw["plan_sha256"]),
        "landpoints": len(landpoints),
        "years": len(years),
        "shards": len(index.shards),
        "split_pair_shards": split_pairs,
    }


def _training_plumbing_smoke(
    index: MarkovDatasetIndex,
    statistics: TrainingStatistics,
    contract: Mapping[str, Any],
    *,
    batch_size: int,
    seed: int,
    architecture: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("acceptance smoke batch size must be positive")
    representation = fast_day_target_representation_from_contract(contract)
    raw = next(
        _iter_epoch_batches(
            index,
            spatial_split="train",
            temporal_split="train",
            batch_size=batch_size,
            seed=seed,
            drop_last=False,
        )
    )
    batch = prepare_canonical_batch(raw, statistics, representation)
    config = model_config_from_batch(batch, **dict(architecture or {}))
    parameters = initialize_canonical_model(config, seed=seed)
    weights = jnp.asarray(loss_weights_from_contract(contract))

    def loss(value):
        return canonical_one_step_loss(
            value,
            batch.model_input,
            normalized_fast_day_target=batch.normalized_fast_day_target,
            fast_day_target_finite=batch.fast_day_target_finite,
            fast_day_target_weights=weights,
            dynamic_undefined_flip_target=batch.dynamic_undefined_flip_target,
        )

    prediction = canonical_model_apply(parameters, batch.model_input)
    loss_value, gradients = jax.value_and_grad(loss)(parameters)
    prediction, loss_value, gradients = jax.device_get(
        (prediction, loss_value, gradients)
    )
    prediction_leaves = jax.tree_util.tree_leaves(prediction)
    gradient_leaves = jax.tree_util.tree_leaves(gradients)
    prediction_finite = all(np.all(np.isfinite(value)) for value in prediction_leaves)
    gradient_finite = all(np.all(np.isfinite(value)) for value in gradient_leaves)
    gradient_squared_norm = sum(
        float(np.sum(np.asarray(value, dtype=np.float64) ** 2))
        for value in gradient_leaves
    )
    gradient_norm = float(np.sqrt(gradient_squared_norm))
    passed = bool(
        prediction_finite
        and np.isfinite(loss_value)
        and gradient_finite
        and gradient_norm > 0.0
    )
    report = {
        "status": "passed" if passed else "failed",
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "batch_size": int(batch.normalized_fast_day_target.shape[0]),
        "model_config": config._asdict(),
        "parameter_count": parameter_count(parameters),
        "prediction_shapes": [list(np.asarray(value).shape) for value in prediction_leaves],
        "prediction_finite": prediction_finite,
        "loss": float(loss_value),
        "loss_finite": bool(np.isfinite(loss_value)),
        "gradient_leaves": len(gradient_leaves),
        "gradient_finite": gradient_finite,
        "gradient_norm": gradient_norm,
    }
    if not passed:
        raise ValueError(f"canonical model plumbing smoke failed: {report}")
    return report


def accept_training_dataset(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    chunk_rows: int = 64,
    batch_size: int = 8,
    seed: int = 0,
    architecture: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Validate one aggregate and emit the only asset accepted by GPU training."""

    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    index = load_dataset_index(manifest_path, verify_hashes=True)
    dataset_validation = _validate_complete_dataset_manifest(manifest_path, index)
    contract = load_contract_metadata(manifest_path)
    representation = audit_fast_day_target_representation(index, contract)
    if representation["status"] != "passed":
        raise ValueError(
            "fast-day target representation audit failed with "
            f"{representation['persistence_mismatches']} mismatches"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    statistics_path = fit_statistics_asset(
        manifest_path,
        output_dir / "training_statistics.json",
        chunk_rows=chunk_rows,
    )
    statistics = load_training_statistics(statistics_path, index=index)
    smoke = _training_plumbing_smoke(
        index,
        statistics,
        contract,
        batch_size=batch_size,
        seed=seed,
        architecture=architecture,
    )
    statistics_metadata = json.loads(statistics_path.read_text(encoding="utf-8"))
    report = {
        "schema_version": ACCEPTANCE_SCHEMA_VERSION,
        "status": "passed",
        "identity": {
            "dataset_id": index.dataset_id,
            "teacher_git_head": index.teacher_git_head,
            "markov_contract_sha256": index.contract_sha256,
        },
        "dataset_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
        },
        "dataset_validation": dataset_validation,
        "target_representation_audit": representation,
        "training_statistics": {
            "path": str(statistics_path),
            "sha256": _sha256_file(statistics_path),
            "arrays_path": str(statistics_path.with_suffix(".npz")),
            "arrays_sha256": statistics_metadata["statistics_npz_sha256"],
            "sample_count": statistics.sample_count,
        },
        "training_plumbing_smoke": smoke,
    }
    report_path = output_dir / "acceptance_report.json"
    _atomic_json(report_path, report)
    return report


def verify_training_acceptance(
    acceptance_path: str | Path,
    manifest_path: str | Path,
    statistics_path: str | Path,
) -> Mapping[str, Any]:
    acceptance_path = Path(acceptance_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    statistics_path = Path(statistics_path).resolve()
    report = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != ACCEPTANCE_SCHEMA_VERSION:
        raise ValueError("unsupported canonical dataset acceptance schema")
    if report.get("status") != "passed":
        raise ValueError("canonical dataset acceptance did not pass")
    if report.get("dataset_validation", {}).get("status") != "complete":
        raise ValueError("accepted dataset validation is not complete")
    if report.get("target_representation_audit", {}).get("status") != "passed":
        raise ValueError("accepted target representation audit did not pass")
    if report.get("training_plumbing_smoke", {}).get("status") != "passed":
        raise ValueError("accepted training plumbing smoke did not pass")
    if report.get("dataset_manifest", {}).get("sha256") != _sha256_file(manifest_path):
        raise ValueError("accepted dataset manifest hash mismatch")
    if report.get("training_statistics", {}).get("sha256") != _sha256_file(statistics_path):
        raise ValueError("accepted training statistics hash mismatch")
    index = load_dataset_index(manifest_path, verify_hashes=True)
    statistics = load_training_statistics(statistics_path, index=index)
    identity = {
        "dataset_id": index.dataset_id,
        "teacher_git_head": index.teacher_git_head,
        "markov_contract_sha256": index.contract_sha256,
    }
    if report.get("identity") != identity:
        raise ValueError("canonical dataset acceptance identity mismatch")
    if int(report["training_statistics"]["sample_count"]) != statistics.sample_count:
        raise ValueError("accepted training statistics sample count mismatch")
    return report


def _iter_epoch_batches(
    index: MarkovDatasetIndex,
    *,
    spatial_split: str,
    temporal_split: str,
    batch_size: int,
    seed: int,
    drop_last: bool,
) -> Iterator[dict[str, Any]]:
    references = list(
        index.select(
            spatial_split=spatial_split,
            temporal_split=temporal_split,
        )
    )
    if not references:
        raise ValueError(
            f"dataset has no {spatial_split}/{temporal_split} shards"
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(references)
    pending = []
    for reference in references:
        shard = load_markov_shard(reference.path)
        for day in rng.permutation(shard.days):
            pending.append(shard.sample(int(day)))
            if len(pending) == batch_size:
                yield collate_samples(pending)
                pending.clear()
    if pending and not drop_last:
        yield collate_samples(pending)


def _adam_init(parameters: CanonicalModelParameters) -> AdamState:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, parameters)
    return AdamState(jnp.asarray(0, dtype=jnp.int32), zeros, zeros)


def _adam_update(
    parameters,
    gradients,
    state: AdamState,
    *,
    learning_rate,
    beta1=0.9,
    beta2=0.999,
    epsilon=1.0e-8,
):
    step = state.step + 1
    first = jax.tree_util.tree_map(
        lambda old, grad: beta1 * old + (1.0 - beta1) * grad,
        state.first,
        gradients,
    )
    second = jax.tree_util.tree_map(
        lambda old, grad: beta2 * old + (1.0 - beta2) * grad * grad,
        state.second,
        gradients,
    )
    first_hat = jax.tree_util.tree_map(
        lambda value: value / (1.0 - beta1**step), first
    )
    second_hat = jax.tree_util.tree_map(
        lambda value: value / (1.0 - beta2**step), second
    )
    parameters = jax.tree_util.tree_map(
        lambda value, mean, variance: value
        - learning_rate * mean / (jnp.sqrt(variance) + epsilon),
        parameters,
        first_hat,
        second_hat,
    )
    return parameters, AdamState(step, first, second)


def _train_step(parameters, optimizer, batch, weights, learning_rate):
    def loss(value):
        return canonical_one_step_loss(
            value,
            batch.model_input,
            normalized_fast_day_target=batch.normalized_fast_day_target,
            fast_day_target_finite=batch.fast_day_target_finite,
            fast_day_target_weights=weights,
            dynamic_undefined_flip_target=batch.dynamic_undefined_flip_target,
        )

    loss_value, gradients = jax.value_and_grad(loss)(parameters)
    parameters, optimizer = _adam_update(
        parameters,
        gradients,
        optimizer,
        learning_rate=learning_rate,
    )
    return parameters, optimizer, loss_value


_COMPILED_TRAIN_STEP = jax.jit(_train_step)


def _family_ranges(contract: Mapping[str, Any]):
    result: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for leaf in contract["fast_day_target_leaves"]:
        result[str(leaf["family"])].append(
            (int(leaf["start"]), int(leaf["stop"]))
        )
    return dict(result)


def _evaluate(
    parameters,
    index: MarkovDatasetIndex,
    statistics: TrainingStatistics,
    contract: Mapping[str, Any],
    representation: FastDayTargetRepresentation,
    *,
    spatial_split: str,
    temporal_split: str,
    batch_size: int,
    max_batches: int | None,
) -> dict[str, Any]:
    ranges = _family_ranges(contract)
    squared = defaultdict(float)
    absolute = defaultdict(float)
    counts = defaultdict(int)
    leaf_squared = defaultdict(float)
    leaf_absolute = defaultdict(float)
    leaf_maximum = defaultdict(float)
    leaf_counts = defaultdict(int)
    leaves = tuple(contract["fast_day_target_leaves"])
    undefined_values = 0
    undefined_exact = 0
    undefined_true_positive = 0
    undefined_true_negative = 0
    undefined_false_positive = 0
    undefined_false_negative = 0
    batches = 0
    for raw in _iter_epoch_batches(
        index,
        spatial_split=spatial_split,
        temporal_split=temporal_split,
        batch_size=batch_size,
        seed=0,
        drop_last=False,
    ):
        batch = prepare_canonical_batch(raw, statistics, representation)
        model_prediction = canonical_model_apply(parameters, batch.model_input)
        predicted = np.asarray(model_prediction.normalized_fast_day_target)
        target = np.asarray(batch.normalized_fast_day_target)
        finite = np.asarray(batch.fast_day_target_finite)
        predicted_physical = restore_fast_day_prediction(
            predicted,
            np.asarray(model_prediction.dynamic_undefined_flip_logits),
            batch,
            statistics,
            representation,
        )
        target_physical = restore_fast_day_target(target, batch, statistics)
        undefined = np.asarray(batch.fast_day_target_undefined)
        expected_undefined = np.asarray(batch.fast_day_target_undefined_values)
        undefined_values += int(np.count_nonzero(undefined))
        undefined_exact += int(
            np.count_nonzero(
                undefined
                & (
                    (np.isnan(predicted_physical) & np.isnan(expected_undefined))
                    | (predicted_physical == expected_undefined)
                )
            )
        )
        predicted_dynamic = np.logical_xor(
            np.asarray(batch.persistent_dynamic_undefined),
            np.asarray(model_prediction.dynamic_undefined_flip_logits) >= 0.0,
        )
        expected_dynamic = np.asarray(batch.fast_day_target_undefined)[
            :, representation.dynamic_undefined_indices
        ]
        undefined_true_positive += int(
            np.count_nonzero(predicted_dynamic & expected_dynamic)
        )
        undefined_true_negative += int(
            np.count_nonzero(~predicted_dynamic & ~expected_dynamic)
        )
        undefined_false_positive += int(
            np.count_nonzero(predicted_dynamic & ~expected_dynamic)
        )
        undefined_false_negative += int(
            np.count_nonzero(~predicted_dynamic & expected_dynamic)
        )
        for family, slices in ranges.items():
            for start, stop in slices:
                selected = (predicted[:, start:stop] - target[:, start:stop])[
                    finite[:, start:stop]
                ]
                squared[family] += float(np.sum(selected * selected))
                absolute[family] += float(np.sum(np.abs(selected)))
                counts[family] += int(selected.size)
        for leaf in leaves:
            start = int(leaf["start"])
            stop = int(leaf["stop"])
            component = leaf.get("component") or leaf["family"]
            path = tuple(str(name) for name in leaf.get("path", ()))
            key = (
                ".".join((str(component), *path))
                if path
                else f"{component}[{start}:{stop}]"
            )
            selected = (
                predicted_physical[:, start:stop]
                - target_physical[:, start:stop]
            )[finite[:, start:stop]]
            if selected.size:
                leaf_squared[key] += float(np.sum(selected * selected))
                leaf_absolute[key] += float(np.sum(np.abs(selected)))
                leaf_maximum[key] = max(
                    leaf_maximum[key], float(np.max(np.abs(selected)))
                )
                leaf_counts[key] += int(selected.size)
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    if batches == 0:
        raise ValueError("evaluation selection produced no batches")
    return {
        "batches": batches,
        "source_undefined_persistence": {
            "values": undefined_values,
            "exact": undefined_exact,
        },
        "dynamic_undefined_classification": {
            "true_positive": undefined_true_positive,
            "true_negative": undefined_true_negative,
            "false_positive": undefined_false_positive,
            "false_negative": undefined_false_negative,
        },
        "families": {
            family: {
                "normalized_rmse": float(np.sqrt(squared[family] / counts[family])),
                "normalized_mae": absolute[family] / counts[family],
                "evaluated_values": counts[family],
            }
            for family in sorted(ranges)
            if counts[family]
        },
        "physical_leaves": {
            key: {
                "rmse": float(np.sqrt(leaf_squared[key] / leaf_counts[key])),
                "mae": leaf_absolute[key] / leaf_counts[key],
                "max_absolute_error": leaf_maximum[key],
                "evaluated_values": leaf_counts[key],
            }
            for key in sorted(leaf_counts)
        },
    }


def _validation_selection_score(validation: Mapping[str, Any]) -> float:
    """Equal-weight validation splits and process families for model selection."""

    split_scores = []
    for split in ("temporal", "spatial", "joint"):
        families = validation[split]["families"]
        values = [
            float(metrics["normalized_rmse"])
            for metrics in families.values()
        ]
        if not values or not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite or empty validation families for {split}")
        split_scores.append(float(np.mean(values)))
    return float(np.mean(split_scores))


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _checkpoint_identity(
    index: MarkovDatasetIndex,
    statistics_path: Path,
    acceptance_path: Path,
    config: CanonicalModelConfig,
) -> dict[str, Any]:
    return {
        "dataset_id": index.dataset_id,
        "teacher_git_head": index.teacher_git_head,
        "markov_contract_sha256": index.contract_sha256,
        "statistics_sha256": _sha256_file(statistics_path),
        "acceptance_sha256": _sha256_file(acceptance_path),
        "model_config": config._asdict(),
    }


def _load_checkpoint(path: Path, identity: Mapping[str, Any]):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported canonical checkpoint schema")
    if payload.get("identity") != identity:
        raise ValueError("checkpoint dataset or model identity mismatch")
    return payload


def train_experiment(
    manifest_path: str | Path,
    statistics_path: str | Path,
    output_dir: str | Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    resume: bool = True,
    max_eval_batches: int | None = None,
    architecture: Mapping[str, int] | None = None,
    acceptance_path: str | Path,
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0.0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    manifest_path = Path(manifest_path).resolve()
    statistics_path = Path(statistics_path).resolve()
    acceptance_path = Path(acceptance_path).resolve()
    output_dir = Path(output_dir).resolve()
    verify_training_acceptance(
        acceptance_path,
        manifest_path,
        statistics_path,
    )
    index = load_dataset_index(manifest_path)
    statistics = load_training_statistics(statistics_path, index=index)
    contract = load_contract_metadata(manifest_path)
    representation = fast_day_target_representation_from_contract(contract)
    representation_audit = audit_fast_day_target_representation(index, contract)
    if representation_audit["status"] != "passed":
        raise ValueError(
            "fast-day target representation audit failed with "
            f"{representation_audit['persistence_mismatches']} mismatches"
        )
    first_raw = next(
        _iter_epoch_batches(
            index,
            spatial_split="train",
            temporal_split="train",
            batch_size=batch_size,
            seed=seed,
            drop_last=True,
        )
    )
    first = prepare_canonical_batch(first_raw, statistics, representation)
    config = model_config_from_batch(first, **dict(architecture or {}))
    identity = _checkpoint_identity(
        index,
        statistics_path,
        acceptance_path,
        config,
    )
    checkpoint_path = output_dir / "checkpoint.pkl"
    best_checkpoint_path = output_dir / "best_checkpoint.pkl"
    if resume and checkpoint_path.exists():
        checkpoint = _load_checkpoint(checkpoint_path, identity)
        parameters = checkpoint["parameters"]
        optimizer = checkpoint["optimizer"]
        completed_epochs = int(checkpoint["completed_epochs"])
        history = list(checkpoint["history"])
    else:
        parameters = initialize_canonical_model(config, seed=seed)
        optimizer = _adam_init(parameters)
        completed_epochs = 0
        history = []
    weights = jnp.asarray(loss_weights_from_contract(contract))
    learning_rate_array = jnp.asarray(learning_rate, dtype=jnp.float32)
    for epoch in range(completed_epochs, epochs):
        started = time.perf_counter()
        total_loss = 0.0
        batches = 0
        for raw in _iter_epoch_batches(
            index,
            spatial_split="train",
            temporal_split="train",
            batch_size=batch_size,
            seed=seed + epoch,
            drop_last=True,
        ):
            batch: CanonicalTrainingBatch = prepare_canonical_batch(
                raw, statistics, representation
            )
            parameters, optimizer, loss = _COMPILED_TRAIN_STEP(
                parameters,
                optimizer,
                batch,
                weights,
                learning_rate_array,
            )
            total_loss += float(loss)
            batches += 1
        if not batches:
            raise ValueError("training selection is smaller than one batch")
        jax.block_until_ready(parameters)
        record = {
            "epoch": epoch + 1,
            "mean_training_loss": total_loss / batches,
            "training_batches": batches,
            "seconds": time.perf_counter() - started,
            "validation": {},
        }
        for name, spatial, temporal in (
            ("temporal", "train", "validation"),
            ("spatial", "validation", "train"),
            ("joint", "validation", "validation"),
        ):
            record["validation"][name] = _evaluate(
                parameters,
                index,
                statistics,
                contract,
                representation,
                spatial_split=spatial,
                temporal_split=temporal,
                batch_size=batch_size,
                max_batches=max_eval_batches,
            )
        record["selection_score"] = _validation_selection_score(
            record["validation"]
        )
        history.append(record)
        _atomic_pickle(
            checkpoint_path,
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "identity": identity,
                "parameters": jax.device_get(parameters),
                "optimizer": jax.device_get(optimizer),
                "completed_epochs": epoch + 1,
                "history": history,
            },
        )
        previous_best = min(
            (
                float(item["selection_score"])
                for item in history[:-1]
                if "selection_score" in item
            ),
            default=float("inf"),
        )
        if record["selection_score"] < previous_best:
            _atomic_pickle(
                best_checkpoint_path,
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "identity": identity,
                    "parameters": jax.device_get(parameters),
                    "epoch": epoch + 1,
                    "selection_score": record["selection_score"],
                },
            )
    best_record = min(history, key=lambda item: float(item["selection_score"]))
    summary = {
        "schema_version": "canonical_fast_day_training_report_v2",
        "identity": identity,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "parameter_count": parameter_count(parameters),
        "checkpoint": str(checkpoint_path),
        "best_checkpoint": str(best_checkpoint_path),
        "best_epoch": int(best_record["epoch"]),
        "best_validation_score": float(best_record["selection_score"]),
        "target_representation_audit": representation_audit,
        "history": history,
        "test_split_evaluated": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_report.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    statistics = commands.add_parser("fit-statistics")
    statistics.add_argument("--dataset", type=Path, required=True)
    statistics.add_argument("--output", type=Path, required=True)
    statistics.add_argument("--chunk-rows", type=int, default=64)
    audit = commands.add_parser("audit-target-representation")
    audit.add_argument("--dataset", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    acceptance = commands.add_parser("accept-dataset")
    acceptance.add_argument("--dataset", type=Path, required=True)
    acceptance.add_argument("--output-dir", type=Path, required=True)
    acceptance.add_argument("--chunk-rows", type=int, default=64)
    acceptance.add_argument("--batch-size", type=int, default=8)
    acceptance.add_argument("--seed", type=int, default=0)
    train = commands.add_parser("train")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--statistics", type=Path, required=True)
    train.add_argument("--acceptance", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=1.0e-3)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--max-eval-batches", type=int)
    train.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "fit-statistics":
        output = fit_statistics_asset(
            args.dataset,
            args.output,
            chunk_rows=args.chunk_rows,
        )
        print(json.dumps({"statistics": str(output)}, indent=2))
        return 0
    if args.command == "audit-target-representation":
        index = load_dataset_index(args.dataset)
        contract = load_contract_metadata(args.dataset)
        report = audit_fast_day_target_representation(index, contract)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "passed" else 1
    if args.command == "accept-dataset":
        report = accept_training_dataset(
            args.dataset,
            args.output_dir,
            chunk_rows=args.chunk_rows,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        print(json.dumps(report, indent=2))
        return 0
    summary = train_experiment(
        args.dataset,
        args.statistics,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        resume=not args.no_resume,
        max_eval_batches=args.max_eval_batches,
        acceptance_path=args.acceptance,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
