"""Streamed training entry point for the canonical v3 fast-day operator."""

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
    loss_weights_from_contract,
    model_config_from_batch,
    prepare_canonical_batch,
)
from research.daily_coarse_graining.daily_markov_contract import load_markov_shard
from research.daily_coarse_graining.markov_dataset import (
    MarkovDatasetIndex,
    TrainingStatistics,
    collate_samples,
    fit_training_statistics,
    load_dataset_index,
    load_training_statistics,
    write_training_statistics,
)

CHECKPOINT_SCHEMA_VERSION = "canonical_fast_day_checkpoint_v1"


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
    batches = 0
    for raw in _iter_epoch_batches(
        index,
        spatial_split=spatial_split,
        temporal_split=temporal_split,
        batch_size=batch_size,
        seed=0,
        drop_last=False,
    ):
        batch = prepare_canonical_batch(raw, statistics)
        predicted = np.asarray(
            canonical_model_apply(parameters, batch.model_input)
            .normalized_fast_day_target
        )
        target = np.asarray(batch.normalized_fast_day_target)
        finite = np.asarray(batch.fast_day_target_finite)
        for family, slices in ranges.items():
            for start, stop in slices:
                selected = (predicted[:, start:stop] - target[:, start:stop])[
                    finite[:, start:stop]
                ]
                squared[family] += float(np.sum(selected * selected))
                absolute[family] += float(np.sum(np.abs(selected)))
                counts[family] += int(selected.size)
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    if batches == 0:
        raise ValueError("evaluation selection produced no batches")
    return {
        "batches": batches,
        "families": {
            family: {
                "normalized_rmse": float(np.sqrt(squared[family] / counts[family])),
                "normalized_mae": absolute[family] / counts[family],
                "evaluated_values": counts[family],
            }
            for family in sorted(ranges)
            if counts[family]
        },
    }


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _checkpoint_identity(
    index: MarkovDatasetIndex,
    statistics_path: Path,
    config: CanonicalModelConfig,
) -> dict[str, Any]:
    return {
        "dataset_id": index.dataset_id,
        "teacher_git_head": index.teacher_git_head,
        "markov_contract_sha256": index.contract_sha256,
        "statistics_sha256": _sha256_file(statistics_path),
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
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0.0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    manifest_path = Path(manifest_path).resolve()
    statistics_path = Path(statistics_path).resolve()
    output_dir = Path(output_dir).resolve()
    index = load_dataset_index(manifest_path)
    statistics = load_training_statistics(statistics_path, index=index)
    contract = load_contract_metadata(manifest_path)
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
    first = prepare_canonical_batch(first_raw, statistics)
    config = model_config_from_batch(first, **dict(architecture or {}))
    identity = _checkpoint_identity(index, statistics_path, config)
    checkpoint_path = output_dir / "checkpoint.pkl"
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
                raw, statistics
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
                spatial_split=spatial,
                temporal_split=temporal,
                batch_size=batch_size,
                max_batches=max_eval_batches,
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
    summary = {
        "schema_version": "canonical_fast_day_training_report_v1",
        "identity": identity,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "parameter_count": parameter_count(parameters),
        "checkpoint": str(checkpoint_path),
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
    train = commands.add_parser("train")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--statistics", type=Path, required=True)
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
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
