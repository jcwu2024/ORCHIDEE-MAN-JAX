"""Fail-closed training foundation for the first neural process-head A/B."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, NamedTuple, Sequence

import numpy as np

from research.daily_coarse_graining.daily_markov_contract import (
    daily_markov_contract_from_metadata,
    load_markov_shard,
)
from research.daily_coarse_graining.daily_operator.neural_architecture import (
    AXIAL_CANDIDATE,
    FACTORIZED_CONTROL,
    build_process_axis_layout,
    initialize_neural_process_parameters,
    neural_process_outputs,
)
from research.daily_coarse_graining.daily_operator.training_contract import (
    FirstCandidateContract,
    load_first_candidate_contract,
)
from research.daily_coarse_graining.daily_operator.training_data import (
    DailyProcessTrainingAdapter,
)
from research.daily_coarse_graining.daily_operator.training_loss import (
    masked_equal_label_macro_mse,
)
from research.daily_coarse_graining.daily_operator.training_statistics import (
    ProcessTrainingStatistics,
    load_process_training_statistics,
    vectorized_process_targets,
)
from research.daily_coarse_graining.markov_dataset import load_training_statistics
from research.daily_coarse_graining.typed_sidecar import (
    load_coherent_typed_sidecar_index,
    read_typed_sidecar_shard,
)

PROTOCOL_SCHEMA_VERSION = "first_neural_process_training_protocol_v1"
CHECKPOINT_SCHEMA_VERSION = "first_neural_process_training_checkpoint_v1"
DEFAULT_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[2] / "manifests" / "coarse_graining" / "first_neural_process_training_v1.json"
)


def _canonical_sha256(payload: Mapping[str, Any], hash_field: str) -> str:
    canonical = dict(payload)
    canonical.pop(hash_field, None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ProcessTrainingProtocol:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def training(self) -> Mapping[str, Any]:
        return self.payload["training"]

    @property
    def preflight(self) -> Mapping[str, Any]:
        return self.payload["preflight"]


def load_process_training_protocol(
    path: str | Path = DEFAULT_PROTOCOL_PATH,
) -> ProcessTrainingProtocol:
    source_path = Path(path).resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported first-process training protocol")
    if payload.get("status") != "ready_unsubmitted":
        raise ValueError("first-process training protocol is not ready and unsubmitted")
    observed = _canonical_sha256(payload, "protocol_sha256")
    if payload.get("protocol_sha256") != observed:
        raise ValueError("first-process training protocol hash mismatch")
    sampling = payload.get("sampling", {})
    if sampling != {
        "strategy": "deterministic_shuffled_shards_and_days_without_replacement_v1",
        "spatial_split": "train",
        "temporal_split": "train",
        "same_order_for_matched_arms": True,
        "test_split_allowed": False,
    }:
        raise ValueError("first-process training sampling policy drift")
    training = payload.get("training", {})
    required = {
        "arms": [AXIAL_CANDIDATE, FACTORIZED_CONTROL],
        "seeds": [20260807, 20260808, 20260809],
        "batch_size": 16,
        "prefetch_batches": 2,
        "maximum_updates": 10000,
        "checkpoint_interval_updates": 250,
        "optimizer": "adamw",
        "learning_rate": 3.0e-4,
        "warmup_updates": 500,
        "final_learning_rate": 3.0e-5,
        "schedule": "linear_warmup_cosine_decay",
        "weight_decay": 1.0e-5,
        "global_gradient_norm_clip": 1.0,
        "rollout_loss": False,
    }
    if training != required:
        raise ValueError("first-process training budget drift")
    validation = payload.get("validation", {})
    if validation.get("selection_slices") != [
        "train/validation",
        "validation/train",
        "validation/validation",
    ]:
        raise ValueError("first-process validation slices drift")
    if int(validation.get("maximum_batches_per_slice", 0)) != 64:
        raise ValueError("first-process validation panel drift")
    if validation.get("sealed_test_evaluation") is not False:
        raise ValueError("the test split must remain sealed")
    preflight = payload.get("preflight", {})
    if preflight != {
        "arms": [AXIAL_CANDIDATE, FACTORIZED_CONTROL],
        "seed": 20260807,
        "batch_size": 16,
        "warmup_updates": 3,
        "measured_updates": 20,
        "streaming_updates": 100,
        "streaming_prefetch_batches": 2,
        "validation_batches_per_slice": 4,
        "reuse_one_real_train_batch": True,
        "checkpoint_written": False,
    }:
        raise ValueError("first-process timing preflight drift")
    post = payload.get("post_training", {})
    if post != {
        "rollout_days": [7, 30],
        "requires_both_matched_arms": True,
        "long_rollout_allowed": False,
    }:
        raise ValueError("first-process post-training gate drift")
    return ProcessTrainingProtocol(payload=payload, source_path=source_path, sha256=observed)


@dataclass(frozen=True)
class PreparedProcessTraining:
    candidate: FirstCandidateContract
    protocol: ProcessTrainingProtocol
    adapter: DailyProcessTrainingAdapter
    index: Any
    base_statistics: Any
    process_statistics: ProcessTrainingStatistics
    axis_layout: Any
    artifact_hashes: Mapping[str, str]


def prepare_process_training(
    *,
    dataset_manifest: str | Path,
    acceptance_report: str | Path,
    base_statistics: str | Path,
    process_statistics: str | Path,
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
) -> PreparedProcessTraining:
    candidate = load_first_candidate_contract()
    protocol = load_process_training_protocol(protocol_path)
    paths = {
        "dataset_manifest": Path(dataset_manifest).resolve(),
        "acceptance_report": Path(acceptance_report).resolve(),
        "base_statistics": Path(base_statistics).resolve(),
        "process_statistics": Path(process_statistics).resolve(),
    }
    base_payload = json.loads(paths["base_statistics"].read_text(encoding="utf-8"))
    base_npz = paths["base_statistics"].parent / str(base_payload["statistics_npz"])
    process_payload = json.loads(paths["process_statistics"].read_text(encoding="utf-8"))
    process_npz = paths["process_statistics"].parent / str(process_payload["statistics_npz"])
    hashes = {
        "dataset_manifest_sha256": sha256_file(paths["dataset_manifest"]),
        "acceptance_report_sha256": sha256_file(paths["acceptance_report"]),
        "base_statistics_json_sha256": sha256_file(paths["base_statistics"]),
        "base_statistics_npz_sha256": sha256_file(base_npz),
        "process_statistics_json_sha256": sha256_file(paths["process_statistics"]),
        "process_statistics_npz_sha256": sha256_file(process_npz),
    }
    data = candidate.payload["data"]
    expected_candidate = {
        "dataset_manifest_sha256": data["dataset_manifest_sha256"],
        "acceptance_report_sha256": data["acceptance_report_sha256"],
        "base_statistics_json_sha256": data["training_statistics_json_sha256"],
        "base_statistics_npz_sha256": data["training_statistics_npz_sha256"],
    }
    drift = {
        name: (hashes[name], expected) for name, expected in expected_candidate.items() if hashes[name] != expected
    }
    parent = protocol.payload["parent"]
    expected_protocol = {
        "candidate_contract_sha256": candidate.sha256,
        "process_statistics_json_sha256": hashes["process_statistics_json_sha256"],
        "process_statistics_npz_sha256": hashes["process_statistics_npz_sha256"],
    }
    drift.update(
        {
            f"protocol.{name}": (parent.get(name), expected)
            for name, expected in expected_protocol.items()
            if parent.get(name) != expected
        }
    )
    if drift:
        raise ValueError(f"first-process training artifact identity drift: {drift}")
    acceptance = json.loads(paths["acceptance_report"].read_text(encoding="utf-8"))
    if acceptance.get("status") != "passed":
        raise ValueError("v6 acceptance report is not passed")
    manifest = json.loads(paths["dataset_manifest"].read_text(encoding="utf-8"))
    contract = daily_markov_contract_from_metadata(manifest["markov_contract"])
    adapter = DailyProcessTrainingAdapter.from_contract(contract)
    index = load_coherent_typed_sidecar_index(paths["dataset_manifest"], verify_hashes=False)
    base = load_training_statistics(paths["base_statistics"], index=index.parent_index)
    process = load_process_training_statistics(
        paths["process_statistics"],
        index=index,
        adapter=adapter,
        candidate_contract=candidate,
        base_statistics=base,
    )
    return PreparedProcessTraining(
        candidate=candidate,
        protocol=protocol,
        adapter=adapter,
        index=index,
        base_statistics=base,
        process_statistics=process,
        axis_layout=build_process_axis_layout(adapter.input_assembler),
        artifact_hashes=hashes,
    )


def _stack_trees(values: Sequence[Any]) -> Any:
    import jax

    return jax.tree_util.tree_map(lambda *items: np.stack(items), *values)


def _iter_process_batches(
    prepared: PreparedProcessTraining,
    *,
    spatial_split: str,
    temporal_split: str,
    seed: int,
    batch_size: int,
    start_batch: int = 0,
    maximum_batches: int,
) -> Iterator[tuple[Any, Any]]:
    """Yield deterministic coherent batches without replacement."""

    if batch_size < 1 or start_batch < 0 or maximum_batches < 1:
        raise ValueError("invalid first-process batch request")
    references = list(
        prepared.index.select(
            spatial_split=spatial_split,
            temporal_split=temporal_split,
        )
    )
    if not references:
        raise ValueError(f"coherent v6 dataset has no {spatial_split}/{temporal_split} shards")
    order_rng = np.random.default_rng(seed)
    order_rng.shuffle(references)
    start_sample = start_batch * batch_size
    stop_sample = (start_batch + maximum_batches) * batch_size
    if stop_sample > sum(int(reference.day_count) for reference in references):
        raise ValueError("first-process batch request exceeds one no-replacement pass")
    cursor = 0
    inputs: list[Any] = []
    targets: list[Any] = []
    yielded = 0
    for position, reference in enumerate(references):
        next_cursor = cursor + int(reference.day_count)
        overlap_start = max(start_sample, cursor)
        overlap_stop = min(stop_sample, next_cursor)
        if overlap_start < overlap_stop:
            row_rng = np.random.default_rng(np.random.SeedSequence((seed, position, 0x51A7)))
            rows = row_rng.permutation(int(reference.day_count))
            selected = rows[overlap_start - cursor : overlap_stop - cursor]
            parent = load_markov_shard(reference.parent.path, contract=prepared.adapter.contract)
            values, defined, day_index = read_typed_sidecar_shard(
                reference.path,
                contract=prepared.index.contract,
                expected_days=reference.day_count,
            )
            if not np.array_equal(day_index, parent.day_index):
                raise ValueError("coherent process batch day index drift")
            vectorized_targets = vectorized_process_targets(
                parent,
                values,
                defined,
                prepared.adapter,
            )
            for row in selected:
                inputs.append(prepared.adapter.assemble_input(parent.sample(int(row))))
                targets.append(
                    {
                        name: type(target)(
                            target.value[int(row)],
                            target.defined[int(row)],
                        )
                        for name, target in vectorized_targets.items()
                    }
                )
                if len(inputs) == batch_size:
                    yield _stack_trees(inputs), _stack_trees(targets)
                    yielded += 1
                    inputs.clear()
                    targets.clear()
        cursor = next_cursor
        if cursor >= stop_sample:
            break
    if inputs or yielded != maximum_batches:
        raise ValueError("first-process deterministic sampler produced an incomplete batch plan")


def iter_process_training_batches(
    prepared: PreparedProcessTraining,
    *,
    seed: int,
    batch_size: int,
    start_batch: int = 0,
    maximum_batches: int,
) -> Iterator[tuple[Any, Any]]:
    """Yield train/train only; no caller can override this split."""

    yield from _iter_process_batches(
        prepared,
        spatial_split="train",
        temporal_split="train",
        seed=seed,
        batch_size=batch_size,
        start_batch=start_batch,
        maximum_batches=maximum_batches,
    )


def iter_process_validation_batches(
    prepared: PreparedProcessTraining,
    *,
    spatial_split: str,
    temporal_split: str,
    seed: int,
    batch_size: int,
    maximum_batches: int,
) -> Iterator[tuple[Any, Any]]:
    """Yield only one of the three declared, non-test validation slices."""

    allowed = {
        ("train", "validation"),
        ("validation", "train"),
        ("validation", "validation"),
    }
    if (spatial_split, temporal_split) not in allowed:
        raise ValueError("first-process validation split is not declared")
    yield from _iter_process_batches(
        prepared,
        spatial_split=spatial_split,
        temporal_split=temporal_split,
        seed=seed,
        batch_size=batch_size,
        maximum_batches=maximum_batches,
    )


@dataclass(frozen=True)
class _PrefetchFailure:
    error: BaseException


def prefetched_process_training_batches(
    prepared: PreparedProcessTraining,
    *,
    seed: int,
    batch_size: int,
    start_batch: int = 0,
    maximum_batches: int,
    prefetch_batches: int,
) -> Iterator[tuple[Any, Any]]:
    """Overlap bounded host preparation with GPU work without changing order."""

    if prefetch_batches < 1:
        raise ValueError("first-process prefetch must be positive")
    pending: queue.Queue[Any] = queue.Queue(maxsize=prefetch_batches)
    finished = object()

    def produce() -> None:
        try:
            for batch in iter_process_training_batches(
                prepared,
                seed=seed,
                batch_size=batch_size,
                start_batch=start_batch,
                maximum_batches=maximum_batches,
            ):
                pending.put(batch)
        except BaseException as error:
            pending.put(_PrefetchFailure(error))
        finally:
            pending.put(finished)

    worker = threading.Thread(
        target=produce,
        name="first-process-training-prefetch",
        daemon=True,
    )
    worker.start()
    while True:
        value = pending.get()
        if value is finished:
            break
        if isinstance(value, _PrefetchFailure):
            raise value.error
        yield value
    worker.join()


class AdamWState(NamedTuple):
    step: Any
    first: Any
    second: Any


def initialize_adamw(parameters: Any) -> AdamWState:
    import jax
    import jax.numpy as jnp

    zeros = jax.tree_util.tree_map(jnp.zeros_like, parameters)
    return AdamWState(jnp.asarray(0, dtype=jnp.int32), zeros, zeros)


def learning_rate_at_update(protocol: ProcessTrainingProtocol, update: int) -> float:
    training = protocol.training
    maximum = int(training["maximum_updates"])
    warmup = int(training["warmup_updates"])
    if not 1 <= update <= maximum:
        raise ValueError("training update is outside the frozen budget")
    peak = float(training["learning_rate"])
    final = float(training["final_learning_rate"])
    if update <= warmup:
        return peak * update / warmup
    progress = (update - warmup) / (maximum - warmup)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(final + (peak - final) * cosine)


def initialize_process_parameters(
    prepared: PreparedProcessTraining,
    inputs: Any,
    *,
    seed: int,
) -> Any:
    import jax

    return initialize_neural_process_parameters(
        jax.random.PRNGKey(seed),
        schema=prepared.process_statistics.schema,
        axis_layout=prepared.axis_layout,
        inputs=prepared.adapter.input_assembler.assemble(jax.tree_util.tree_map(lambda value: value[0], inputs)),
    )


def build_process_batch_loss(prepared: PreparedProcessTraining, *, arm: str):
    import jax
    import jax.numpy as jnp

    if arm not in (AXIAL_CANDIDATE, FACTORIZED_CONTROL):
        raise ValueError("unknown first-process training arm")
    statistics = prepared.process_statistics
    normalization = statistics.input_normalization()
    output_normalization = statistics.output_normalization()
    schema = statistics.schema
    labels = statistics.field_labels
    scales = statistics.loss_scales

    def one(parameters, inputs, targets):
        from research.daily_coarse_graining.daily_operator.neural_architecture import (
            assemble_neural_process_context,
        )

        assembled = prepared.adapter.input_assembler.assemble(inputs)
        context = assemble_neural_process_context(
            parameters,
            assembled,
            normalization=normalization,
            axis_layout=prepared.axis_layout,
        )
        prediction = neural_process_outputs(
            parameters,
            context,
            schema=schema,
            normalization=output_normalization,
            arm=arm,
        )
        return masked_equal_label_macro_mse(
            prediction,
            targets,
            scales=scales,
            field_labels=labels,
        ).total

    def batch_loss(parameters, inputs, targets):
        return jnp.mean(jax.vmap(lambda x, y: one(parameters, x, y))(inputs, targets))

    return batch_loss


def make_adamw_train_step(
    batch_loss: Any,
    protocol: ProcessTrainingProtocol,
):
    import jax
    import jax.numpy as jnp

    weight_decay = float(protocol.training["weight_decay"])
    clip_norm = float(protocol.training["global_gradient_norm_clip"])

    @jax.jit
    def train_step(parameters, optimizer, inputs, targets, learning_rate):
        loss, gradients = jax.value_and_grad(batch_loss)(parameters, inputs, targets)
        squared_norm = sum(jnp.sum(leaf * leaf, dtype=jnp.float64) for leaf in jax.tree_util.tree_leaves(gradients))
        gradient_norm = jnp.sqrt(squared_norm)
        clip_scale = jnp.minimum(
            1.0,
            jnp.asarray(clip_norm, dtype=jnp.float64)
            / jnp.maximum(gradient_norm, jnp.asarray(1.0e-30, dtype=jnp.float64)),
        )
        gradients = jax.tree_util.tree_map(lambda value: value * clip_scale, gradients)
        step = optimizer.step + 1
        first = jax.tree_util.tree_map(
            lambda old, grad: 0.9 * old + 0.1 * grad,
            optimizer.first,
            gradients,
        )
        second = jax.tree_util.tree_map(
            lambda old, grad: 0.999 * old + 0.001 * grad * grad,
            optimizer.second,
            gradients,
        )
        beta1_correction = 1.0 - jnp.power(0.9, step)
        beta2_correction = 1.0 - jnp.power(0.999, step)
        next_parameters = jax.tree_util.tree_map(
            lambda value, mean, variance: (
                value
                - learning_rate
                * ((mean / beta1_correction) / (jnp.sqrt(variance / beta2_correction) + 1.0e-8) + weight_decay * value)
            ),
            parameters,
            first,
            second,
        )
        return (
            next_parameters,
            AdamWState(step, first, second),
            loss,
            gradient_norm,
            clip_scale,
        )

    return train_step


def checkpoint_identity(
    prepared: PreparedProcessTraining,
    *,
    arm: str,
    seed: int,
    source_git_head: str,
) -> dict[str, Any]:
    if arm not in prepared.protocol.training["arms"]:
        raise ValueError("checkpoint arm is outside the protocol")
    if seed not in prepared.protocol.training["seeds"]:
        raise ValueError("checkpoint seed is outside the protocol")
    return {
        "source_git_head": source_git_head,
        "candidate_contract_sha256": prepared.candidate.sha256,
        "training_protocol_sha256": prepared.protocol.sha256,
        **dict(prepared.artifact_hashes),
        "arm": arm,
        "seed": seed,
    }


def write_training_checkpoint(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    update: int,
    parameters: Any,
    optimizer: AdamWState,
    history: Sequence[Mapping[str, Any]],
) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "identity": dict(identity),
        "update": int(update),
        "parameters": parameters,
        "optimizer": optimizer,
        "history": list(history),
    }
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, target)
    return target


def load_training_checkpoint(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    maximum_updates: int,
) -> Mapping[str, Any]:
    with Path(path).resolve().open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported first-process checkpoint schema")
    if payload.get("identity") != dict(identity):
        raise ValueError("first-process checkpoint identity drift")
    update = int(payload.get("update", -1))
    if not 0 <= update <= maximum_updates:
        raise ValueError("first-process checkpoint update is invalid")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def evaluate_process_validation(
    prepared: PreparedProcessTraining,
    parameters: Any,
    *,
    arm: str,
    seed: int,
    maximum_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate the fixed non-test panel used only for checkpoint selection."""

    import jax

    batch_size = int(prepared.protocol.training["batch_size"])
    frozen_maximum = int(prepared.protocol.payload["validation"]["maximum_batches_per_slice"])
    maximum_batches = frozen_maximum if maximum_batches is None else maximum_batches
    if not 1 <= maximum_batches <= frozen_maximum:
        raise ValueError("first-process validation batch request is outside the panel")
    evaluate = jax.jit(build_process_batch_loss(prepared, arm=arm))
    slices = (
        ("temporal", "train", "validation", 100_003),
        ("spatial", "validation", "train", 200_003),
        ("joint", "validation", "validation", 300_007),
    )
    result = {}
    for name, spatial, temporal, offset in slices:
        losses = []
        for inputs, targets in iter_process_validation_batches(
            prepared,
            spatial_split=spatial,
            temporal_split=temporal,
            seed=seed + offset,
            batch_size=batch_size,
            maximum_batches=maximum_batches,
        ):
            loss = evaluate(parameters, inputs, targets)
            losses.append(float(loss))
        if len(losses) != maximum_batches or not np.all(np.isfinite(losses)):
            raise ValueError(f"first-process {name} validation is incomplete or nonfinite")
        result[name] = {
            "spatial_split": spatial,
            "temporal_split": temporal,
            "batches": maximum_batches,
            "samples": maximum_batches * batch_size,
            "macro_process_loss": float(np.mean(losses)),
        }
    result["selection_score"] = float(np.mean([result[name]["macro_process_loss"] for name, *_ in slices]))
    result["test_split_evaluated"] = False
    return result


def train_first_process_arm(
    prepared: PreparedProcessTraining,
    *,
    arm: str,
    seed: int,
    source_git_head: str,
    output_dir: str | Path,
    resume: bool = True,
) -> dict[str, Any]:
    """Run one fixed-budget arm/seed; post-training physical gates remain separate."""

    import jax
    import jax.numpy as jnp

    if prepared.protocol.payload.get("status") != "ready_unsubmitted":
        raise ValueError("first-process protocol is not frozen ready for formal training")
    training = prepared.protocol.training
    if arm not in training["arms"] or seed not in training["seeds"]:
        raise ValueError("first-process arm or seed is outside the frozen protocol")
    output = Path(output_dir).resolve()
    latest_path = output / "checkpoint.pkl"
    best_path = output / "best_checkpoint.pkl"
    report_path = output / "training_report.json"
    identity = checkpoint_identity(
        prepared,
        arm=arm,
        seed=seed,
        source_git_head=source_git_head,
    )
    maximum_updates = int(training["maximum_updates"])
    checkpoint_interval = int(training["checkpoint_interval_updates"])
    batch_size = int(training["batch_size"])
    prefetch = int(training["prefetch_batches"])
    if resume and latest_path.exists():
        checkpoint = load_training_checkpoint(
            latest_path,
            identity=identity,
            maximum_updates=maximum_updates,
        )
        parameters = checkpoint["parameters"]
        optimizer = checkpoint["optimizer"]
        completed = int(checkpoint["update"])
        history = list(checkpoint["history"])
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("new first-process output directory is not empty")
        first_inputs, _first_targets = next(
            iter_process_training_batches(
                prepared,
                seed=seed,
                batch_size=batch_size,
                maximum_batches=1,
            )
        )
        parameters = initialize_process_parameters(
            prepared,
            first_inputs,
            seed=seed,
        )
        optimizer = initialize_adamw(parameters)
        completed = 0
        history = []
    if completed == maximum_updates:
        if not report_path.exists():
            raise ValueError("completed checkpoint has no training report")
        return json.loads(report_path.read_text(encoding="utf-8"))
    loss_function = build_process_batch_loss(prepared, arm=arm)
    train_step = make_adamw_train_step(loss_function, prepared.protocol)
    window_losses: list[float] = []
    window_gradient_norms: list[float] = []
    window_clip_scales: list[float] = []
    window_started = time.perf_counter()
    batches = prefetched_process_training_batches(
        prepared,
        seed=seed,
        batch_size=batch_size,
        start_batch=completed,
        maximum_batches=maximum_updates - completed,
        prefetch_batches=prefetch,
    )
    for update, (inputs, targets) in enumerate(batches, start=completed + 1):
        learning_rate = jnp.asarray(
            learning_rate_at_update(prepared.protocol, update),
            dtype=jnp.float64,
        )
        parameters, optimizer, loss, gradient_norm, clip_scale = train_step(
            parameters,
            optimizer,
            inputs,
            targets,
            learning_rate,
        )
        loss_value = float(loss)
        gradient_value = float(gradient_norm)
        clip_value = float(clip_scale)
        if not np.all(np.isfinite((loss_value, gradient_value, clip_value))):
            raise ValueError(f"first-process training became nonfinite at update {update}")
        window_losses.append(loss_value)
        window_gradient_norms.append(gradient_value)
        window_clip_scales.append(clip_value)
        if update % checkpoint_interval:
            continue
        validation_started = time.perf_counter()
        validation = evaluate_process_validation(
            prepared,
            parameters,
            arm=arm,
            seed=seed,
        )
        record = {
            "update": update,
            "mean_training_loss": float(np.mean(window_losses)),
            "maximum_gradient_norm": max(window_gradient_norms),
            "minimum_clip_scale": min(window_clip_scales),
            "learning_rate": float(learning_rate),
            "training_seconds": time.perf_counter() - window_started,
            "validation_seconds": time.perf_counter() - validation_started,
            "validation": validation,
            "selection_score": validation["selection_score"],
        }
        history.append(record)
        host_parameters = jax.device_get(parameters)
        host_optimizer = jax.device_get(optimizer)
        write_training_checkpoint(
            latest_path,
            identity=identity,
            update=update,
            parameters=host_parameters,
            optimizer=host_optimizer,
            history=history,
        )
        previous_best = min(
            (float(item["selection_score"]) for item in history[:-1]),
            default=float("inf"),
        )
        if float(record["selection_score"]) < previous_best:
            write_training_checkpoint(
                best_path,
                identity=identity,
                update=update,
                parameters=host_parameters,
                optimizer=host_optimizer,
                history=history,
            )
        progress = {
            "schema_version": "first_neural_process_training_report_v1",
            "status": "running" if update < maximum_updates else "passed",
            "identity": identity,
            "arm": arm,
            "seed": seed,
            "completed_updates": update,
            "maximum_updates": maximum_updates,
            "batch_size": batch_size,
            "prefetch_batches": prefetch,
            "checkpoint_interval_updates": checkpoint_interval,
            "history": history,
            "best_update": int(min(history, key=lambda item: float(item["selection_score"]))["update"]),
            "best_checkpoint": str(best_path),
            "best_checkpoint_sha256": (sha256_file(best_path) if best_path.exists() else None),
            "test_split_evaluated": False,
            "hard_physical_gate_status": "pending_post_training_admission",
            "rollout_status": "pending_7_and_30_day_gate",
        }
        _atomic_json(report_path, progress)
        window_losses.clear()
        window_gradient_norms.clear()
        window_clip_scales.clear()
        window_started = time.perf_counter()
    return json.loads(report_path.read_text(encoding="utf-8"))
