"""Bounded runtime and TRAIN-scale preparation for the shared Transformer.

The bootstrap is intentionally limited to the canonical v6 runtime, prepared
TRAIN artifacts, retained-process templates, and the historical parent-head
wrapper.  It never initializes neural parameters and never reads a checkpoint.
The reused bootstrap provenance is ``scripts/hpc/run_fresh_start_one_day.py``
lines 18-92.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

CALIBRATION_SELECTION_SEED = 20260921
MAX_CALIBRATION_ARTIFACTS = 16
MAX_CALIBRATION_ROWS = 512
_COMPONENT_HORIZONS = {"one_day": 1, "seven_day": 7, "thirty_day": 30}


def _canonical_metadata(record: Mapping[str, Any]) -> str:
    """Return a stable metadata-only identity for a prepared manifest row."""

    return json.dumps(dict(record), sort_keys=True, separators=(",", ":"), default=str)


def select_artifact_ids(
    records: Sequence[Mapping[str, Any]], count: int = MAX_CALIBRATION_ARTIFACTS
) -> tuple[int, ...]:
    """Select a deterministic, year-spanning metadata-only calibration subset.

    Records are ordered by runtime year and a fixed-seed hash of their manifest
    metadata.  Evenly spaced positions are then selected from that ordering.
    No artifact or Teacher array is opened by this helper.
    """

    requested = int(count)
    if requested <= 0:
        raise ValueError("shared calibration artifact count must be positive")
    if not records:
        raise ValueError("shared calibration requires prepared manifest records")

    normalized = []
    seen = set()
    for record in records:
        if not isinstance(record, Mapping) or "update" not in record or "runtime_year" not in record:
            raise ValueError("prepared manifest record lacks update/runtime_year metadata")
        update = int(record["update"])
        if update in seen:
            raise ValueError("prepared manifest update identities are not unique")
        seen.add(update)
        seeded = hashlib.sha256(
            f"{CALIBRATION_SELECTION_SEED}|{_canonical_metadata(record)}".encode("utf-8")
        ).hexdigest()
        normalized.append((int(record["runtime_year"]), seeded, update))

    ordered = sorted(normalized)
    selected_count = min(requested, len(ordered))
    if selected_count == 1:
        positions = (len(ordered) // 2,)
    else:
        positions = tuple(
            round(index * (len(ordered) - 1) / (selected_count - 1))
            for index in range(selected_count)
        )
    if len(set(positions)) != selected_count:
        raise AssertionError("internal calibration linspace produced duplicate positions")
    return tuple(ordered[position][2] for position in positions)


def _reference_identity(value: Any) -> tuple[str, int, str, str]:
    parent = value.parent
    return (
        str(parent.landpoint_id),
        int(parent.year),
        str(parent.sha256),
        str(value.sha256),
    )


def _record_identity(record: Mapping[str, Any]) -> tuple[str, int, str, str]:
    required = ("landpoint_id", "year", "parent_shard_sha256", "typed_shard_sha256")
    if any(name not in record for name in required):
        raise ValueError("prepared schedule record lacks its canonical v6 identity")
    return (
        str(record["landpoint_id"]),
        int(record["year"]),
        str(record["parent_shard_sha256"]),
        str(record["typed_shard_sha256"]),
    )


def _identity_metadata(identity: tuple[str, int, str, str]) -> dict[str, Any]:
    return {
        "landpoint_id": identity[0],
        "year": identity[1],
        "parent_shard_sha256": identity[2],
        "typed_shard_sha256": identity[3],
    }


def validate_shared_artifact_metadata(
    index: Any,
    artifact_ids: Sequence[int],
    artifacts: Sequence[Any],
) -> dict[str, Any]:
    """Validate only supplied artifacts against canonical train/train references."""

    identifiers = tuple(int(value) for value in artifact_ids)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("supplied artifact ids must be nonempty and unique")
    if len(identifiers) != len(artifacts):
        raise ValueError("supplied artifact ids and loaded artifacts differ in length")

    references = {_reference_identity(reference): reference for reference in index.shards}
    if len(references) != len(index.shards):
        raise ValueError("canonical typed-sidecar references are not unique")

    artifact_metadata = []
    all_identities: set[tuple[str, int, str, str]] = set()
    calibration_rows = []
    for identifier, artifact in zip(identifiers, artifacts, strict=True):
        entry = artifact.schedule_entry
        if int(entry.get("update", -1)) != identifier:
            raise ValueError("prepared artifact update identity differs from its requested id")
        runtime_year = int(entry.get("runtime_year", -1))
        components = entry.get("components")
        if not isinstance(components, Mapping) or set(components) != set(_COMPONENT_HORIZONS):
            raise ValueError("prepared artifact component registry drift")

        component_counts = {}
        one_day_rows = []
        for component, horizon in _COMPONENT_HORIZONS.items():
            component_records = components[component]
            if not isinstance(component_records, Sequence):
                raise ValueError(f"prepared {component} records are not a sequence")
            component_counts[component] = len(component_records)
            for record in component_records:
                identity = _record_identity(record)
                reference = references.get(identity)
                if reference is None:
                    raise ValueError("prepared artifact record is absent from canonical v6 references")
                parent = reference.parent
                if parent.spatial_split != "train" or parent.temporal_split != "train":
                    raise ValueError("shared calibration attempted to consume a non-train/train reference")
                if identity[1] != runtime_year:
                    raise ValueError("prepared artifact record crosses its retained runtime year")
                starts = record.get("start_rows")
                if not isinstance(starts, Sequence) or not starts:
                    raise ValueError(f"prepared {component} record has no start rows")
                for raw_start in starts:
                    start = int(raw_start)
                    if start < 0 or start + horizon > int(reference.day_count):
                        raise ValueError(f"prepared {component} start row lies outside its v6 shard")
                    if component == "one_day":
                        row = _identity_metadata(identity)
                        row["selected_start_row"] = start
                        one_day_rows.append(row)
                all_identities.add(identity)
        artifact_metadata.append(
            {
                "artifact_id": identifier,
                "runtime_year": runtime_year,
                "component_record_counts": component_counts,
                "one_day_rows": tuple(one_day_rows),
            }
        )
        calibration_rows.extend(
            {"artifact_id": identifier, **row} for row in one_day_rows
        )

    point_years = {(identity[0], identity[1]) for identity in all_identities}
    return {
        "schema_version": "shared_transformer_runtime_metadata_v1",
        "data_product": "canonical_pft14_daily_teacher_v6_prepared",
        "split": "train/train",
        "scope": "bounded_admission_calibration_subset_not_global_representativeness",
        "artifact_ids": identifiers,
        "artifact_count": len(identifiers),
        "artifacts": tuple(artifact_metadata),
        "calibration_daily_rows": tuple(calibration_rows),
        "unique_identity_coverage": {
            "reference_count": len(all_identities),
            "point_year_count": len(point_years),
            "landpoint_count": len({identity[0] for identity in all_identities}),
            "year_count": len({identity[1] for identity in all_identities}),
            "references": tuple(_identity_metadata(value) for value in sorted(all_identities)),
        },
    }


def _load_prepared_artifacts(
    manifest_path: Path, artifact_ids: Sequence[int]
) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "target_oriented_prepared_manifest_v1"
        or manifest.get("status") != "passed"
    ):
        raise ValueError("prepared manifest is not an accepted target-oriented v1 manifest")
    records = manifest.get("entries")
    if not isinstance(records, list):
        raise ValueError("prepared manifest has no entry registry")
    by_id = {}
    for record in records:
        identifier = int(record["update"])
        if identifier in by_id:
            raise ValueError("prepared manifest contains duplicate update identities")
        by_id[identifier] = record

    root = manifest_path.parent
    artifacts = []
    for raw_identifier in artifact_ids:
        identifier = int(raw_identifier)
        record = by_id.get(identifier)
        if record is None:
            raise ValueError(f"prepared artifact {identifier} is absent from the manifest")
        path = (root / str(record["path"])).resolve()
        path.relative_to(root)
        with gzip.open(path, "rb") as handle:
            artifact = pickle.load(handle)
        if int(artifact.schedule_entry.get("update", -1)) != identifier:
            raise ValueError("prepared sample identity differs from the manifest record")
        if int(record.get("runtime_year", -1)) != int(artifact.schedule_entry.get("runtime_year", -2)):
            raise ValueError("prepared sample runtime year differs from the manifest record")
        if str(record.get("trace_signature")) != str(artifact.trace_signature):
            raise ValueError("prepared sample trace signature differs from the manifest record")
        expected_contract = manifest.get("identity", {}).get("training_contract_sha256")
        if expected_contract and str(artifact.training_contract_sha256) != str(expected_contract):
            raise ValueError("prepared sample training contract differs from the manifest")
        artifacts.append(artifact)
    return tuple(artifacts), manifest


def load_shared_runtime(
    args: Any,
    artifact_ids: Sequence[int],
) -> tuple[Any, Any, Any, tuple[Any, ...], Mapping[str, Any]]:
    """Load canonical runtime/templates without initializing any neural weights."""

    identifiers = tuple(int(value) for value in artifact_ids)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("shared runtime artifact ids must be nonempty and unique")

    from scripts.hpc.run_successor_fast_owner_gpu_probe import _load_extended_runtime

    backend = {}
    if getattr(args, "shape_only", False) or getattr(args, "cpu_forward_only", False):

        def cpu_backend() -> tuple[Any, Any]:
            import jax

            return jax, jax.devices("cpu")

        backend["backend_initializer"] = cpu_backend
    runtime = _load_extended_runtime(
        args,
        initialize_probe_parameters=False,
        **backend,
    )

    from research.daily_coarse_graining.daily_operator.mechanistic_output_contract import (
        build_mechanistic_learned_schema,
        load_mechanistic_output_contract,
    )
    from research.daily_coarse_graining.daily_operator.mechanistic_process_head import (
        build_mechanistic_output_layout,
    )
    from research.daily_coarse_graining.daily_operator.strong_baseline_contract import (
        load_strong_baseline_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_contract import (
        load_target_oriented_contract,
    )
    from scripts.hpc.run_target_oriented_attention_gpu_admission import _head
    from scripts.hpc.run_target_oriented_full_step_preflight import _prepare_template

    architecture = load_strong_baseline_contract()
    mechanism = load_mechanistic_output_contract()
    target = load_target_oriented_contract()
    learned_schema = build_mechanistic_learned_schema(runtime["schema"], mechanism)
    target_layout = build_mechanistic_output_layout(
        adapter=runtime["adapter"],
        complete_schema=runtime["schema"],
        learned_schema=learned_schema,
        owner_contract=runtime["owner_contract"],
        architecture_contract=architecture,
        mechanism_contract=mechanism,
        inputs=runtime["probe_assembled"],
    )
    runtime.update(
        architecture_contract=architecture,
        mechanism_contract=mechanism,
        learned_schema=learned_schema,
        target_layout=target_layout,
        target_contract=target,
    )
    runtime.pop("parameters", None)

    artifacts, manifest = _load_prepared_artifacts(Path(args.prepared_manifest), identifiers)
    metadata = validate_shared_artifact_metadata(runtime["index"], identifiers, artifacts)
    metadata["prepared_manifest_schema_version"] = manifest["schema_version"]
    metadata["prepared_manifest_identity"] = dict(manifest.get("identity", {}))

    retained_template = _prepare_template(runtime, artifacts[0].schedule_entry)
    retained_template = replace(
        retained_template,
        static_owner=replace(
            retained_template.static_owner,
            include_modelout_diagnostics=False,
        ),
    )
    parent_head = _head(runtime, target)
    return runtime, parent_head, retained_template, artifacts, metadata


def prepare_shared_trajectory_scales(
    runtime: Mapping[str, Any],
    artifacts: Sequence[Any],
    *,
    flux_floors: Mapping[str, float],
    stock_change_floors: Mapping[str, float],
) -> tuple[Any, Mapping[str, Any]]:
    """Prepare bounded TRAIN-only physical RMS scales from one-day rows."""

    if not artifacts or len(artifacts) > MAX_CALIBRATION_ARTIFACTS:
        raise ValueError("shared scale calibration requires between 1 and 16 artifacts")
    from research.daily_coarse_graining.daily_operator.shared_trajectory_loss import (
        build_shared_trajectory_scales,
        shared_component_trajectories,
    )

    jax = runtime["jax"]
    contract = runtime["contract"]
    assembler = runtime["adapter"].input_assembler
    artifact_ids = tuple(int(artifact.schedule_entry["update"]) for artifact in artifacts)
    validated = validate_shared_artifact_metadata(runtime["index"], artifact_ids, artifacts)
    scale_samples = []
    consumed = []
    for artifact in artifacts:
        update = int(artifact.schedule_entry["update"])
        batch = artifact.batch.one_day
        rows = [
            (record, int(start))
            for record in artifact.schedule_entry["components"]["one_day"]
            for start in record["start_rows"]
        ]
        row_count = int(np.shape(batch.inputs.continuous_state)[0])
        if row_count != len(rows):
            raise ValueError("prepared one-day batch and schedule row counts differ")
        for row_index, (record, start_row) in enumerate(rows):
            inputs = jax.tree_util.tree_map(lambda value: value[row_index], batch.inputs)
            expected_state = jax.tree_util.tree_map(
                lambda value: value[row_index], batch.expected_next_state
            )
            expected_defined = jax.tree_util.tree_map(
                lambda value: value[row_index], batch.expected_next_state_defined
            )
            fast_gpp = batch.process_targets["daily.fast_gpp"]
            target_observation = {
                "state": expected_state.other[None, ...],
                "defined": expected_defined.other[None, ...],
                "gpp": fast_gpp.value[row_index][None, ...],
                "gpp_defined": fast_gpp.defined[row_index][None, ...],
                "fraction": inputs.pft.pft_fraction[None, ...],
                "active": inputs.pft.active_pft_mask[None, ...],
            }
            target = shared_component_trajectories(target_observation, contract)

            assembled = assembler.assemble(inputs)
            initial_observation = dict(target_observation)
            initial_observation["state"] = inputs.continuous_state[None, ...]
            initial_observation["defined"] = assembled.state_defined.other[None, ...]
            initial = shared_component_trajectories(initial_observation, contract)
            initial_stocks = {
                key: type(value)(value.value[0], value.defined[0])
                for key, value in initial.stocks.items()
            }
            scale_samples.append((target, initial_stocks))
            consumed.append(
                {
                    "artifact_id": update,
                    **_identity_metadata(_record_identity(record)),
                    "selected_start_row": start_row,
                    "prepared_day_index": int(np.asarray(inputs.day_index)),
                }
            )

    if len(scale_samples) > MAX_CALIBRATION_ROWS:
        raise ValueError("shared scale calibration exceeds the 512-row admission bound")
    scales = build_shared_trajectory_scales(
        scale_samples,
        split_label="train/train",
        flux_floors=flux_floors,
        stock_change_floors=stock_change_floors,
    )
    unique_references = {
        (
            row["landpoint_id"],
            row["year"],
            row["parent_shard_sha256"],
            row["typed_shard_sha256"],
        )
        for row in consumed
    }
    metadata = {
        "schema_version": "shared_trajectory_scale_calibration_v1",
        "scope": "bounded_train_only_admission_calibration_not_global_scale_qualification",
        "split": "train/train",
        "artifact_ids": artifact_ids,
        "artifact_count": len(artifacts),
        "row_count": len(scale_samples),
        "unique_reference_count": len(unique_references),
        "unique_point_year_count": len({(value[0], value[1]) for value in unique_references}),
        "validated_unique_identity_coverage": validated["unique_identity_coverage"],
        "consumed_rows": tuple(consumed),
        "flux_floors": {key: float(value) for key, value in flux_floors.items()},
        "stock_change_floors": {
            key: float(value) for key, value in stock_change_floors.items()
        },
        "flux_counts": dict(scales.flux_counts),
        "stock_change_counts": dict(scales.stock_change_counts),
        "flux_nonzero_counts": dict(scales.flux_nonzero_counts),
        "stock_change_nonzero_counts": dict(scales.stock_change_nonzero_counts),
        "heterotrophic_respiration_nonzero_count": int(
            scales.flux_nonzero_counts["heterotrophic_respiration"]
        ),
    }
    return scales, metadata


__all__ = [
    "CALIBRATION_SELECTION_SEED",
    "MAX_CALIBRATION_ARTIFACTS",
    "MAX_CALIBRATION_ROWS",
    "load_shared_runtime",
    "prepare_shared_trajectory_scales",
    "select_artifact_ids",
    "validate_shared_artifact_metadata",
]
