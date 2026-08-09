"""Train/train-only statistics for the added successor fast-owner heads."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_markov_contract import MarkovShard
from research.daily_coarse_graining.typed_sidecar import TypedSidecarIndex

from .fast_owner_reconstruction import FastOwnerReconstructionContract
from .neural_architecture import NeuralOutputNormalization, NeuralOutputSchema
from .successor_contract import SuccessorCandidateContract
from .successor_fast_owner_decoder import (
    ABSOLUTE_TARGET,
    DAY_START_DELTA,
    DEFINED_LOGIT,
    RVEGET_KEY,
    RVEGET_LOGIT_KEY,
    UNDEFINED_THRESHOLD,
    _day_start_owner_key,
    _neural_leaves,
    extend_successor_output_normalization,
    schema_field_labels,
)
from .training_data import DailyProcessTrainingAdapter, MaskedProcessTarget
from .training_statistics import (
    OUTPUT_MINIMUM_SCALE,
    MaskedMoments,
    MomentStatistics,
    ProcessTrainingStatistics,
    _load_moments,
    _merge_finalized_moments,
    _sha256_file,
)

SCHEMA_VERSION = "successor_fast_owner_training_statistics_v1"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_inventory(references: Sequence[Any]) -> tuple[int, str]:
    entries = [
        {
            "landpoint_id": reference.parent.landpoint_id,
            "year": reference.parent.year,
            "base_sha256": reference.parent.sha256,
            "day_count": reference.day_count,
        }
        for reference in references
    ]
    return len(entries), _canonical_sha256(entries)


def _added_fields(schema: NeuralOutputSchema) -> tuple[Any, ...]:
    fields = tuple(field for field in schema.fields if field.value_representation != "direct")
    if len(fields) != 80:
        raise ValueError("successor fast-owner statistics require 80 added fields")
    if sum(field.value_representation == DEFINED_LOGIT for field in fields) != 1:
        raise ValueError("successor fast-owner statistics require one defined logit")
    return fields


def _value_fields(schema: NeuralOutputSchema) -> tuple[Any, ...]:
    fields = tuple(field for field in _added_fields(schema) if field.value_representation != DEFINED_LOGIT)
    if len(fields) != 79:
        raise ValueError("successor fast-owner statistics require 79 value fields")
    return fields


def vectorized_successor_fast_owner_targets(
    shard: MarkovShard,
    *,
    schema: NeuralOutputSchema,
    adapter: DailyProcessTrainingAdapter,
    owner_contract: FastOwnerReconstructionContract,
) -> dict[str, MaskedProcessTarget]:
    """Construct all 80 added targets directly from an existing v6 shard."""

    if adapter.contract.sha256 != owner_contract.markov_contract_sha256:
        raise ValueError("fast-owner vectorized target contract drift")
    fields = {field.name: field for field in _added_fields(schema)}
    fast_leaves = {leaf.key: leaf for leaf in adapter.contract.fast_day_target_leaves}
    state_leaves = {leaf.key: leaf for leaf in adapter.contract.state_leaves}
    result: dict[str, MaskedProcessTarget] = {}
    rveget_defined: np.ndarray | None = None

    for leaf in _neural_leaves(owner_contract):
        field = fields.get(str(leaf.output_key))
        fast_leaf = fast_leaves.get(leaf.key)
        if field is None or fast_leaf is None:
            raise ValueError(f"vectorized fast-owner registry is incomplete for {leaf.key}")
        absolute = shard.fast_day_target[:, fast_leaf.start : fast_leaf.stop].reshape((shard.days, *field.shape))
        teacher_defined = np.isfinite(absolute) & (np.abs(absolute) < UNDEFINED_THRESHOLD)
        if field.value_representation == DAY_START_DELTA:
            owner_key = _day_start_owner_key(leaf)
            state_leaf = state_leaves.get(owner_key)
            if state_leaf is None or tuple(state_leaf.shape) != field.shape:
                raise ValueError(f"vectorized fast-owner state owner drift for {leaf.key}")
            day_start = shard.state_trajectory[:-1, state_leaf.start : state_leaf.stop].reshape(
                (shard.days, *field.shape)
            )
            start_defined = np.isfinite(day_start) & (np.abs(day_start) < UNDEFINED_THRESHOLD)
            base = np.where(start_defined, day_start, 0.0)
            value = np.where(teacher_defined, absolute - base, 0.0)
        elif field.value_representation == ABSOLUTE_TARGET:
            value = np.where(teacher_defined, absolute, 0.0)
        else:
            raise ValueError(f"vectorized fast-owner representation drift for {leaf.key}")
        result[field.name] = MaskedProcessTarget(value, teacher_defined)
        if leaf.key == RVEGET_KEY:
            rveget_defined = teacher_defined

    logit = fields.get(RVEGET_LOGIT_KEY)
    if rveget_defined is None or logit is None:
        raise ValueError("vectorized fast-owner targets lack rveget status")
    eligible = np.ones((shard.days, *logit.shape), dtype=np.bool_)
    result[RVEGET_LOGIT_KEY] = MaskedProcessTarget(
        rveget_defined.astype(np.float64),
        eligible,
    )
    if set(result) != set(fields):
        raise ValueError("vectorized fast-owner target registry drift")
    return result


def _loss_scales(
    outputs: Mapping[str, MomentStatistics],
    schema: NeuralOutputSchema,
) -> dict[str, np.ndarray]:
    fields = {field.name: field for field in _value_fields(schema)}
    if set(outputs) != set(fields):
        raise ValueError("fast-owner statistics output registry drift")
    return {
        name: (
            np.ones(statistics.mean.shape, dtype=np.float64)
            if fields[name].value_representation == ABSOLUTE_TARGET
            and fields[name].constraint in {"unit_interval", "signed_unit"}
            else statistics.scale.copy()
        )
        for name, statistics in outputs.items()
    }


@dataclass(frozen=True)
class SuccessorFastOwnerStatistics:
    successor_contract_sha256: str
    owner_contract_sha256: str
    dataset_id: str
    teacher_git_head: str
    markov_contract_sha256: str
    typed_contract_sha256: str
    dataset_manifest_sha256: str
    base_process_statistics_json_sha256: str
    base_process_statistics_npz_sha256: str
    selection: Mapping[str, str]
    partition_index: int
    partition_count: int
    sample_count: int
    source_shard_count: int
    source_inventory_sha256: str
    rveget_defined_count: int
    rveget_eligible_count: int
    schema: NeuralOutputSchema
    outputs: Mapping[str, MomentStatistics]
    loss_scales: Mapping[str, np.ndarray]

    def output_normalization(
        self,
        base: ProcessTrainingStatistics,
    ) -> NeuralOutputNormalization:
        return extend_successor_output_normalization(
            base.output_normalization(),
            self.schema,
            added_mean={name: value.mean for name, value in self.outputs.items()},
            added_scale={name: value.scale for name, value in self.outputs.items()},
        )

    def combined_loss_scales(
        self,
        base: ProcessTrainingStatistics,
    ) -> dict[str, np.ndarray]:
        result = dict(base.loss_scales) | dict(self.loss_scales)
        expected = {field.name for field in self.schema.fields if field.value_representation != DEFINED_LOGIT}
        if set(result) != expected:
            raise ValueError("combined successor loss-scale registry drift")
        return result


def _validate_identity(
    index: TypedSidecarIndex,
    adapter: DailyProcessTrainingAdapter,
    base: ProcessTrainingStatistics,
    successor_contract: SuccessorCandidateContract,
    owner_contract: FastOwnerReconstructionContract,
    schema: NeuralOutputSchema,
) -> None:
    data = successor_contract.payload["data"]
    expected = {
        "dataset_manifest_sha256": index.parent_manifest_sha256,
        "markov_contract_sha256": index.parent_index.contract_sha256,
        "typed_contract_sha256": index.contract.sha256,
    }
    drift = {name: (data.get(name), value) for name, value in expected.items() if data.get(name) != value}
    if drift:
        raise ValueError(f"successor fast-owner statistics data identity drift: {drift}")
    if adapter.contract.sha256 != index.parent_index.contract_sha256:
        raise ValueError("successor fast-owner statistics adapter drift")
    if owner_contract.markov_contract_sha256 != adapter.contract.sha256:
        raise ValueError("successor fast-owner owner-contract drift")
    base_identity = (
        base.dataset_id,
        base.teacher_git_head,
        base.markov_contract_sha256,
        base.typed_contract_sha256,
        base.dataset_manifest_sha256,
    )
    expected_base = (
        index.parent_index.dataset_id,
        index.parent_index.teacher_git_head,
        index.parent_index.contract_sha256,
        index.contract.sha256,
        index.parent_manifest_sha256,
    )
    if base_identity != expected_base:
        raise ValueError("base process statistics identity drift")
    if len(base.schema.fields) != 101 or tuple(schema.fields[:101]) != tuple(base.schema.fields):
        raise ValueError("successor fast-owner statistics base schema drift")
    _added_fields(schema)


def fit_successor_fast_owner_statistics(
    index: TypedSidecarIndex,
    adapter: DailyProcessTrainingAdapter,
    base: ProcessTrainingStatistics,
    successor_contract: SuccessorCandidateContract,
    owner_contract: FastOwnerReconstructionContract,
    schema: NeuralOutputSchema,
    *,
    partition_index: int = 0,
    partition_count: int = 1,
    chunk_rows: int = 64,
) -> SuccessorFastOwnerStatistics:
    """Fit one mergeable partition from accepted v6 arrays only."""

    if partition_count < 1 or not 0 <= partition_index < partition_count:
        raise ValueError("invalid successor fast-owner statistics partition")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    _validate_identity(
        index,
        adapter,
        base,
        successor_contract,
        owner_contract,
        schema,
    )
    all_references = index.select(spatial_split="train", temporal_split="train")
    expected_samples = sum(int(reference.day_count) for reference in all_references)
    if base.sample_count != expected_samples:
        raise ValueError("base process statistics sample count drift")
    references = all_references[partition_index::partition_count]
    if not references:
        raise ValueError("successor fast-owner statistics partition is empty")

    moments = {
        field.name: MaskedMoments(
            field.shape,
            pft_axis=field.pft_axis,
            pft_indices=(tuple(range(field.shape[field.pft_axis])) if field.pft_axis is not None else ()),
        )
        for field in _value_fields(schema)
    }
    sample_count = 0
    rveget_defined_count = 0
    rveget_eligible_count = 0
    for reference in references:
        from research.daily_coarse_graining.daily_markov_contract import load_markov_shard

        shard = load_markov_shard(reference.parent.path, contract=adapter.contract)
        targets = vectorized_successor_fast_owner_targets(
            shard,
            schema=schema,
            adapter=adapter,
            owner_contract=owner_contract,
        )
        for name, accumulator in moments.items():
            target = targets[name]
            accumulator.update(target.value, target.defined, chunk_rows=chunk_rows)
        status = targets[RVEGET_LOGIT_KEY]
        eligible = np.asarray(status.defined, dtype=np.bool_)
        rveget_defined_count += int(np.count_nonzero(eligible & (np.asarray(status.value) > 0.5)))
        rveget_eligible_count += int(np.count_nonzero(eligible))
        sample_count += shard.days

    outputs = {name: value.finalize(minimum_scale=OUTPUT_MINIMUM_SCALE) for name, value in moments.items()}
    source_count, source_hash = _source_inventory(references)
    data = successor_contract.payload["data"]
    return SuccessorFastOwnerStatistics(
        successor_contract_sha256=successor_contract.sha256,
        owner_contract_sha256=owner_contract.contract_sha256,
        dataset_id=index.parent_index.dataset_id,
        teacher_git_head=index.parent_index.teacher_git_head,
        markov_contract_sha256=index.parent_index.contract_sha256,
        typed_contract_sha256=index.contract.sha256,
        dataset_manifest_sha256=index.parent_manifest_sha256,
        base_process_statistics_json_sha256=str(data["training_statistics_json_sha256"]),
        base_process_statistics_npz_sha256=str(data["training_statistics_npz_sha256"]),
        selection={"spatial_split": "train", "temporal_split": "train"},
        partition_index=partition_index,
        partition_count=partition_count,
        sample_count=sample_count,
        source_shard_count=source_count,
        source_inventory_sha256=source_hash,
        rveget_defined_count=rveget_defined_count,
        rveget_eligible_count=rveget_eligible_count,
        schema=schema,
        outputs=outputs,
        loss_scales=_loss_scales(outputs, schema),
    )


def write_successor_fast_owner_statistics(
    statistics: SuccessorFastOwnerStatistics,
    path: str | Path,
) -> Path:
    metadata_path = Path(path).with_suffix(".json")
    arrays_path = metadata_path.with_suffix(".npz")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"output__{name}__{member}": np.asarray(getattr(value, member))
        for name, value in statistics.outputs.items()
        for member in ("count", "mean", "m2", "variance", "scale")
    }
    arrays.update({f"loss_scale__{name}": np.asarray(value) for name, value in statistics.loss_scales.items()})
    np.savez_compressed(arrays_path, **arrays)
    fields = {field.name: field for field in _value_fields(statistics.schema)}
    labels = schema_field_labels(statistics.schema)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "successor_contract_sha256": statistics.successor_contract_sha256,
        "owner_contract_sha256": statistics.owner_contract_sha256,
        "dataset": {
            "dataset_id": statistics.dataset_id,
            "teacher_git_head": statistics.teacher_git_head,
            "markov_contract_sha256": statistics.markov_contract_sha256,
            "typed_contract_sha256": statistics.typed_contract_sha256,
            "dataset_manifest_sha256": statistics.dataset_manifest_sha256,
            "base_process_statistics_json_sha256": statistics.base_process_statistics_json_sha256,
            "base_process_statistics_npz_sha256": statistics.base_process_statistics_npz_sha256,
        },
        "selection": dict(statistics.selection),
        "partition": {
            "index": statistics.partition_index,
            "count": statistics.partition_count,
        },
        "sample_count": statistics.sample_count,
        "source_shard_count": statistics.source_shard_count,
        "source_inventory_sha256": statistics.source_inventory_sha256,
        "rveget_status": {
            "defined_count": statistics.rveget_defined_count,
            "eligible_count": statistics.rveget_eligible_count,
        },
        "statistics_npz": arrays_path.name,
        "statistics_npz_sha256": _sha256_file(arrays_path),
        "policies": {
            "selection": "v6_train_train_only",
            "fast_state_representation": DAY_START_DELTA,
            "daily_ok_leak_representation": ABSOLUTE_TARGET,
            "undefined_day_start_base": "zero",
            "pft_moments": "shared_across_compact_active_slots",
            "minimum_scale": OUTPUT_MINIMUM_SCALE,
            "teacher_regeneration": False,
            "historical_v5": False,
        },
        "outputs": {
            name: {
                "feature_shape": list(value.mean.shape),
                "label": labels[name],
                "constraint": fields[name].constraint,
                "pft_axis": fields[name].pft_axis,
                "process_domain": fields[name].process_domain,
                "source_route": fields[name].source_route,
                "value_representation": fields[name].value_representation,
                "never_defined_columns": int(np.count_nonzero(value.count == 0)),
            }
            for name, value in statistics.outputs.items()
        },
    }
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def load_successor_fast_owner_statistics(
    path: str | Path,
    *,
    index: TypedSidecarIndex,
    adapter: DailyProcessTrainingAdapter,
    base: ProcessTrainingStatistics,
    successor_contract: SuccessorCandidateContract,
    owner_contract: FastOwnerReconstructionContract,
    schema: NeuralOutputSchema,
) -> SuccessorFastOwnerStatistics:
    metadata_path = Path(path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported successor fast-owner statistics schema")
    _validate_identity(
        index,
        adapter,
        base,
        successor_contract,
        owner_contract,
        schema,
    )
    data = payload["dataset"]
    expected_data = {
        "dataset_id": index.parent_index.dataset_id,
        "teacher_git_head": index.parent_index.teacher_git_head,
        "markov_contract_sha256": index.parent_index.contract_sha256,
        "typed_contract_sha256": index.contract.sha256,
        "dataset_manifest_sha256": index.parent_manifest_sha256,
        "base_process_statistics_json_sha256": successor_contract.payload["data"]["training_statistics_json_sha256"],
        "base_process_statistics_npz_sha256": successor_contract.payload["data"]["training_statistics_npz_sha256"],
    }
    if data != expected_data:
        raise ValueError("successor fast-owner statistics dataset identity drift")
    if payload.get("successor_contract_sha256") != successor_contract.sha256:
        raise ValueError("successor fast-owner candidate contract drift")
    if payload.get("owner_contract_sha256") != owner_contract.contract_sha256:
        raise ValueError("successor fast-owner reconstruction contract drift")
    if payload.get("selection") != {
        "spatial_split": "train",
        "temporal_split": "train",
    }:
        raise ValueError("successor fast-owner statistics are not train/train-only")
    arrays_path = metadata_path.parent / str(payload["statistics_npz"])
    if _sha256_file(arrays_path) != payload["statistics_npz_sha256"]:
        raise ValueError("successor fast-owner statistics NPZ hash mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    outputs, consumed = _load_moments(
        arrays,
        payload["outputs"],
        prefix="output",
    )
    expected_fields = {field.name: field for field in _value_fields(schema)}
    if set(outputs) != set(expected_fields):
        raise ValueError("successor fast-owner statistics output registry drift")
    for name, metadata in payload["outputs"].items():
        field = expected_fields[name]
        expected_metadata = (
            field.shape,
            schema.labels[field.label_index],
            field.constraint,
            field.pft_axis,
            field.process_domain,
            field.source_route,
            field.value_representation,
        )
        observed_metadata = (
            tuple(metadata["feature_shape"]),
            metadata["label"],
            metadata["constraint"],
            metadata["pft_axis"],
            metadata["process_domain"],
            metadata["source_route"],
            metadata["value_representation"],
        )
        if observed_metadata != expected_metadata:
            raise ValueError(f"successor fast-owner field metadata drift: {name}")
    loss_scales = {}
    expected_scales = _loss_scales(outputs, schema)
    for name, expected in expected_scales.items():
        key = f"loss_scale__{name}"
        if key not in arrays or not np.array_equal(arrays[key], expected):
            raise ValueError(f"successor fast-owner loss scale drift: {name}")
        loss_scales[name] = arrays[key]
        consumed.add(key)
    if set(arrays) != consumed:
        raise ValueError("successor fast-owner statistics NPZ contains undeclared arrays")
    partition = payload["partition"]
    status = payload["rveget_status"]
    return SuccessorFastOwnerStatistics(
        successor_contract_sha256=successor_contract.sha256,
        owner_contract_sha256=owner_contract.contract_sha256,
        dataset_id=data["dataset_id"],
        teacher_git_head=data["teacher_git_head"],
        markov_contract_sha256=data["markov_contract_sha256"],
        typed_contract_sha256=data["typed_contract_sha256"],
        dataset_manifest_sha256=data["dataset_manifest_sha256"],
        base_process_statistics_json_sha256=data["base_process_statistics_json_sha256"],
        base_process_statistics_npz_sha256=data["base_process_statistics_npz_sha256"],
        selection=payload["selection"],
        partition_index=int(partition["index"]),
        partition_count=int(partition["count"]),
        sample_count=int(payload["sample_count"]),
        source_shard_count=int(payload["source_shard_count"]),
        source_inventory_sha256=str(payload["source_inventory_sha256"]),
        rveget_defined_count=int(status["defined_count"]),
        rveget_eligible_count=int(status["eligible_count"]),
        schema=schema,
        outputs=outputs,
        loss_scales=loss_scales,
    )


def merge_successor_fast_owner_statistics(
    partials: Sequence[SuccessorFastOwnerStatistics],
    *,
    index: TypedSidecarIndex,
    adapter: DailyProcessTrainingAdapter,
    base: ProcessTrainingStatistics,
    successor_contract: SuccessorCandidateContract,
    owner_contract: FastOwnerReconstructionContract,
    schema: NeuralOutputSchema,
) -> SuccessorFastOwnerStatistics:
    if not partials:
        raise ValueError("cannot merge zero successor fast-owner statistics partitions")
    counts = {partial.partition_count for partial in partials}
    if len(counts) != 1:
        raise ValueError("successor fast-owner partition counts drift")
    partition_count = counts.pop()
    by_index = {partial.partition_index: partial for partial in partials}
    if len(by_index) != len(partials) or set(by_index) != set(range(partition_count)):
        raise ValueError("successor fast-owner partitions are duplicated or incomplete")
    _validate_identity(
        index,
        adapter,
        base,
        successor_contract,
        owner_contract,
        schema,
    )
    ordered = tuple(by_index[index_value] for index_value in range(partition_count))
    expected_identity = (
        successor_contract.sha256,
        owner_contract.contract_sha256,
        index.parent_index.dataset_id,
        index.parent_index.teacher_git_head,
        index.parent_index.contract_sha256,
        index.contract.sha256,
        index.parent_manifest_sha256,
        successor_contract.payload["data"]["training_statistics_json_sha256"],
        successor_contract.payload["data"]["training_statistics_npz_sha256"],
    )
    for partial in ordered:
        identity = (
            partial.successor_contract_sha256,
            partial.owner_contract_sha256,
            partial.dataset_id,
            partial.teacher_git_head,
            partial.markov_contract_sha256,
            partial.typed_contract_sha256,
            partial.dataset_manifest_sha256,
            partial.base_process_statistics_json_sha256,
            partial.base_process_statistics_npz_sha256,
        )
        if identity != expected_identity or partial.schema != schema:
            raise ValueError("successor fast-owner statistics merge identity drift")
    references = index.select(spatial_split="train", temporal_split="train")
    source_count, source_hash = _source_inventory(references)
    sample_count = sum(partial.sample_count for partial in ordered)
    if sample_count != sum(int(reference.day_count) for reference in references):
        raise ValueError("successor fast-owner statistics merge sample-count drift")
    if sum(partial.source_shard_count for partial in ordered) != source_count:
        raise ValueError("successor fast-owner statistics merge shard-count drift")
    outputs = {
        name: _merge_finalized_moments(
            tuple(partial.outputs[name] for partial in ordered),
            minimum_scale=OUTPUT_MINIMUM_SCALE,
        )
        for name in ordered[0].outputs
    }
    return SuccessorFastOwnerStatistics(
        successor_contract_sha256=successor_contract.sha256,
        owner_contract_sha256=owner_contract.contract_sha256,
        dataset_id=index.parent_index.dataset_id,
        teacher_git_head=index.parent_index.teacher_git_head,
        markov_contract_sha256=index.parent_index.contract_sha256,
        typed_contract_sha256=index.contract.sha256,
        dataset_manifest_sha256=index.parent_manifest_sha256,
        base_process_statistics_json_sha256=expected_identity[-2],
        base_process_statistics_npz_sha256=expected_identity[-1],
        selection={"spatial_split": "train", "temporal_split": "train"},
        partition_index=0,
        partition_count=1,
        sample_count=sample_count,
        source_shard_count=source_count,
        source_inventory_sha256=source_hash,
        rveget_defined_count=sum(partial.rveget_defined_count for partial in ordered),
        rveget_eligible_count=sum(partial.rveget_eligible_count for partial in ordered),
        schema=schema,
        outputs=outputs,
        loss_scales=_loss_scales(outputs, schema),
    )
