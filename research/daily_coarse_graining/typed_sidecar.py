"""Gate-E2 hash-joined supplemental-label sidecars.

The immutable v5 parent shards remain the owner of inputs, endpoints, splits,
and dataset identity.  This module adds only the 29 source-captured label
families that cannot be reconstructed from those endpoints.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, NamedTuple, Sequence

import numpy as np

from research.daily_coarse_graining.daily_flux_capture import CAPTURE_LABEL_PATHS
from research.daily_coarse_graining.daily_markov_contract import load_markov_shard
from research.daily_coarse_graining.markov_dataset import (
    ORCHIDEE_UNDEFINED_THRESHOLD,
    MarkovDatasetIndex,
    MarkovShardRef,
    load_dataset_index,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_PATH = (
    ROOT / "manifests" / "coarse_graining" / "daily_typed_sidecar_v1.json"
)
SIDECAR_DATASET_SCHEMA_VERSION = "daily_typed_sidecar_dataset_v1"
SIDECAR_STATISTICS_SCHEMA_VERSION = "daily_typed_sidecar_statistics_v1"


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _contract_sha256(raw: Mapping[str, Any]) -> str:
    unhashed = dict(raw)
    unhashed.pop("contract_sha256", None)
    return _canonical_json_sha256(unhashed)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TypedSidecarField:
    path: str
    label_id: str
    family: str
    owner: str
    unit: str
    axes: tuple[str, ...]
    shape: tuple[str | int, ...]
    dtype: str
    mask_components: tuple[str, ...]

    @property
    def feature_shape(self) -> tuple[int, ...]:
        return tuple(int(size) for size in self.shape[1:])


@dataclass(frozen=True)
class TypedSidecarContract:
    path: Path
    sha256: str
    inventory_sha256: str
    parent_dataset_id: str
    parent_contract_sha256: str
    fields: tuple[TypedSidecarField, ...]

    @property
    def fields_by_path(self) -> Mapping[str, TypedSidecarField]:
        return {field.path: field for field in self.fields}

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(field.label_id for field in self.fields))


def load_typed_sidecar_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> TypedSidecarContract:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "daily_typed_sidecar_contract_v1":
        raise ValueError("unsupported typed-sidecar contract schema")
    actual_hash = _contract_sha256(raw)
    if raw.get("contract_sha256") != actual_hash:
        raise ValueError("typed-sidecar contract self-hash mismatch")

    inventory_path = ROOT / str(raw["inventory"]["path"])
    inventory_hash = _sha256_file(inventory_path)
    if raw["inventory"].get("sha256") != inventory_hash:
        raise ValueError("typed-sidecar inventory hash mismatch")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("schema_version") != raw["inventory"].get("schema_version"):
        raise ValueError("typed-sidecar inventory schema mismatch")

    inventory_by_id = {item["id"]: item for item in inventory["labels"]}
    axis_names = set(raw["axes"])
    mask_names = set(raw["mask_components"])
    fields: list[TypedSidecarField] = []
    labels: list[str] = []
    for label in raw.get("labels", ()):
        label_id = str(label["id"])
        labels.append(label_id)
        inventory_label = inventory_by_id.get(label_id)
        if inventory_label is None:
            raise ValueError(f"unknown typed-sidecar label {label_id}")
        if label.get("family") != inventory_label.get("capture", {}).get("family"):
            raise ValueError(f"typed-sidecar owner-family drift for {label_id}")
        if not str(inventory_label["units"]).startswith(str(label["unit"])):
            raise ValueError(f"typed-sidecar unit drift for {label_id}")
        if not str(label.get("owner", "")).strip():
            raise ValueError(f"typed-sidecar owner is missing for {label_id}")
        for item in label.get("fields", ()):
            axes = tuple(str(name) for name in item["axes"])
            shape = tuple(item["shape"])
            masks = tuple(str(name) for name in item["mask"])
            if not axes or axes[0] != "day" or len(axes) != len(shape):
                raise ValueError(f"invalid typed-sidecar axes for {item['path']}")
            if shape[0] != "D" or any(name not in axis_names for name in axes):
                raise ValueError(f"invalid typed-sidecar shape for {item['path']}")
            if any(name not in mask_names for name in masks):
                raise ValueError(f"unknown typed-sidecar mask for {item['path']}")
            if item.get("dtype") != "float64":
                raise ValueError(f"typed-sidecar dtype drift for {item['path']}")
            fields.append(
                TypedSidecarField(
                    path=str(item["path"]),
                    label_id=label_id,
                    family=str(label["family"]),
                    owner=str(label["owner"]),
                    unit=str(label["unit"]),
                    axes=axes,
                    shape=shape,
                    dtype="float64",
                    mask_components=masks,
                )
            )

    if len(labels) != len(set(labels)):
        raise ValueError("duplicate typed-sidecar label")
    paths = [field.path for field in fields]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate typed-sidecar field")
    expected_labels = {
        item["id"]
        for item in inventory["labels"]
        if item["availability"] == "missing_non_identifiable"
    }
    expected_paths = {
        path for label_id in expected_labels for path in CAPTURE_LABEL_PATHS[label_id]
    }
    if set(labels) != expected_labels or set(paths) != expected_paths:
        raise ValueError("typed-sidecar contract does not cover the frozen inventory")
    if len(labels) != 29 or len(fields) != 66:
        raise ValueError("typed-sidecar contract must contain 29 labels and 66 fields")

    return TypedSidecarContract(
        path=path,
        sha256=actual_hash,
        inventory_sha256=inventory_hash,
        parent_dataset_id=str(raw["parent"]["dataset_id"]),
        parent_contract_sha256=str(raw["parent"]["markov_contract_sha256"]),
        fields=tuple(fields),
    )


def audit_typed_sidecar_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    contract = load_typed_sidecar_contract(path)
    families = {field.family for field in contract.fields}
    return {
        "schema_version": "daily_typed_sidecar_contract_audit_v1",
        "contract_sha256": contract.sha256,
        "label_count": len(contract.labels),
        "field_count": len(contract.fields),
        "capture_families": sorted(families),
        "all_fields_float64": all(field.dtype == "float64" for field in contract.fields),
        "all_fields_have_defined_mask": all(field.mask_components for field in contract.fields),
        "passed": len(contract.labels) == 29 and len(contract.fields) == 66,
    }


@dataclass(frozen=True)
class TypedSidecarShardRef:
    parent: MarkovShardRef
    path: Path
    sha256: str
    day_count: int


class SupplementalProcessPrediction(NamedTuple):
    """Label-owned supplemental arrays aligned with ``ProcessPrediction.fluxes``."""

    water: Mapping[str, Mapping[str, np.ndarray]]
    carbon: Mapping[str, Mapping[str, np.ndarray]]
    energy: Mapping[str, Mapping[str, np.ndarray]]


class TypedSupplementalBatch(NamedTuple):
    values: SupplementalProcessPrediction
    defined: SupplementalProcessPrediction


@dataclass(frozen=True)
class TypedSidecarIndex:
    contract: TypedSidecarContract
    parent_index: MarkovDatasetIndex
    parent_manifest_sha256: str
    sidecar_manifest_sha256: str
    shards: tuple[TypedSidecarShardRef, ...]

    def select(
        self,
        *,
        spatial_split: str | None = None,
        temporal_split: str | None = None,
    ) -> tuple[TypedSidecarShardRef, ...]:
        return tuple(
            reference
            for reference in self.shards
            if (spatial_split is None or reference.parent.spatial_split == spatial_split)
            and (
                temporal_split is None
                or reference.parent.temporal_split == temporal_split
            )
        )

    def samples(
        self,
        *,
        spatial_split: str | None = None,
        temporal_split: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        for reference in self.select(
            spatial_split=spatial_split,
            temporal_split=temporal_split,
        ):
            parent = load_markov_shard(reference.parent.path)
            values, defined, day_index = read_typed_sidecar_shard(
                reference.path,
                contract=self.contract,
                expected_days=reference.day_count,
            )
            if not np.array_equal(day_index, parent.day_index):
                raise ValueError(
                    "typed-sidecar day_index does not match the parent shard"
                )
            for row in range(parent.days):
                parent_sample = parent.sample(row)
                yield _typed_sample(
                    parent_sample,
                    values={name: value[row] for name, value in values.items()},
                    defined={name: value[row] for name, value in defined.items()},
                    contract=self.contract,
                    spatial_split=reference.parent.spatial_split,
                    temporal_split=reference.parent.temporal_split,
                )


def load_typed_sidecar_index(
    manifest_path: str | Path,
    *,
    parent_manifest_path: str | Path,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    verify_hashes: bool = True,
    require_complete: bool = True,
) -> TypedSidecarIndex:
    contract = load_typed_sidecar_contract(contract_path)
    parent_manifest_path = Path(parent_manifest_path).resolve()
    parent_index = load_dataset_index(
        parent_manifest_path,
        verify_hashes=verify_hashes,
    )
    if parent_index.dataset_id != contract.parent_dataset_id:
        raise ValueError("typed-sidecar parent dataset ID mismatch")
    if parent_index.contract_sha256 != contract.parent_contract_sha256:
        raise ValueError("typed-sidecar parent contract hash mismatch")

    manifest_path = Path(manifest_path).resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SIDECAR_DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported typed-sidecar dataset schema")
    parent_manifest_hash = _sha256_file(parent_manifest_path)
    identity = raw.get("parent", {})
    expected_identity = {
        "dataset_id": parent_index.dataset_id,
        "dataset_manifest_sha256": parent_manifest_hash,
        "markov_contract_sha256": parent_index.contract_sha256,
    }
    if identity != expected_identity:
        raise ValueError("typed-sidecar parent dataset identity mismatch")
    if raw.get("sidecar_contract_sha256") != contract.sha256:
        raise ValueError("typed-sidecar contract hash mismatch")

    parent_by_key = {
        (reference.landpoint_id, reference.year): reference
        for reference in parent_index.shards
    }
    root = manifest_path.parent
    seen: set[tuple[str, int]] = set()
    references: list[TypedSidecarShardRef] = []
    for item in raw.get("shards", ()):
        key = (str(item["landpoint_id"]), int(item["year"]))
        if key in seen:
            raise ValueError(f"duplicate typed-sidecar shard {key[0]}:{key[1]}")
        seen.add(key)
        if key not in parent_by_key:
            raise ValueError(f"typed-sidecar has no parent shard {key[0]}:{key[1]}")
        parent = parent_by_key[key]
        if item.get("parent_shard_sha256") != parent.sha256:
            raise ValueError(f"typed-sidecar parent hash mismatch for {key[0]}:{key[1]}")
        if item.get("spatial_split") != parent.spatial_split or item.get(
            "temporal_split"
        ) != parent.temporal_split:
            raise ValueError(f"typed-sidecar split drift for {key[0]}:{key[1]}")
        path = (root / str(item["sidecar"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"sidecar path escapes dataset root: {path}") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_hash = str(item["sidecar_sha256"])
        if verify_hashes and _sha256_file(path) != expected_hash:
            raise ValueError(f"typed-sidecar hash mismatch for {key[0]}:{key[1]}")
        references.append(
            TypedSidecarShardRef(
                parent=parent,
                path=path,
                sha256=expected_hash,
                day_count=int(item["day_count"]),
            )
        )

    missing = sorted(set(parent_by_key) - seen)
    if require_complete and missing:
        raise ValueError(f"typed-sidecar dataset is missing parent shards: {missing[:3]}")
    if not references:
        raise ValueError("typed-sidecar dataset contains no shards")
    return TypedSidecarIndex(
        contract=contract,
        parent_index=parent_index,
        parent_manifest_sha256=parent_manifest_hash,
        sidecar_manifest_sha256=_sha256_file(manifest_path),
        shards=tuple(references),
    )


def _positive_zero_mask(array: np.ndarray) -> np.ndarray:
    if array.dtype != np.dtype(np.float64):
        raise TypeError("typed-sidecar values must be float64")
    return array.view(np.uint64) == 0


def select_typed_sidecar_layout(array: np.ndarray) -> str:
    """Select a bit-preserving representation from its uncompressed size."""

    positive_zero = _positive_zero_mask(array)
    if positive_zero.all():
        return "constant_zero"
    stored = int(np.count_nonzero(~positive_zero))
    sparse_bytes = stored * (np.dtype(np.uint32).itemsize + array.dtype.itemsize)
    return "coo" if sparse_bytes < array.nbytes else "dense"


def write_typed_sidecar_shard(
    path: str | Path,
    *,
    contract: TypedSidecarContract,
    day_index: np.ndarray,
    values: Mapping[str, np.ndarray],
    defined: Mapping[str, np.ndarray],
    layouts: Mapping[str, str] | None = None,
) -> Path:
    """Write one lossless sidecar shard without changing any parent asset."""

    path = Path(path)
    expected = set(contract.fields_by_path)
    if set(values) != expected or set(defined) != expected:
        raise ValueError("typed-sidecar fields do not match the frozen contract")
    day_index = np.asarray(day_index)
    if day_index.dtype != np.dtype(np.int32) or day_index.ndim != 1:
        raise ValueError("typed-sidecar day_index must be one-dimensional int32")
    if day_index.size != np.unique(day_index).size:
        raise ValueError("typed-sidecar day_index contains duplicate days")
    if day_index.size > 1 and not np.all(np.diff(day_index) == 1):
        raise ValueError("typed-sidecar day_index contains missing days")

    arrays: dict[str, np.ndarray] = {"day_index": day_index}
    requested_layouts = dict(layouts or {})
    unknown_layout_paths = set(requested_layouts) - expected
    if unknown_layout_paths:
        raise ValueError(f"layout requested for unknown fields: {unknown_layout_paths}")
    for field in contract.fields:
        value = np.asarray(values[field.path])
        mask = np.asarray(defined[field.path])
        expected_shape = (day_index.size, *field.feature_shape)
        if value.shape != expected_shape or value.dtype != np.dtype(field.dtype):
            raise ValueError(f"typed-sidecar shape/dtype drift for {field.path}")
        if mask.shape != expected_shape or mask.dtype != np.dtype(np.bool_):
            raise ValueError(f"typed-sidecar defined-mask drift for {field.path}")
        invalid_defined = mask & (
            ~np.isfinite(value) | (np.abs(value) >= ORCHIDEE_UNDEFINED_THRESHOLD)
        )
        if np.any(invalid_defined):
            raise ValueError(
                f"defined typed-sidecar values must be finite/non-sentinel: {field.path}"
            )
        layout = requested_layouts.get(field.path, "auto")
        if layout == "auto":
            layout = select_typed_sidecar_layout(value)
        if layout not in {"dense", "coo", "constant_zero"}:
            raise ValueError(f"unsupported typed-sidecar layout {layout!r}")
        if layout == "dense":
            arrays[field.path] = value
        elif layout == "coo":
            flat = np.ascontiguousarray(value).reshape(-1)
            stored = np.flatnonzero(~_positive_zero_mask(flat)).astype(np.uint32)
            arrays[f"{field.path}__coo_index"] = stored
            arrays[f"{field.path}__coo_value"] = flat[stored]
        else:
            if not _positive_zero_mask(np.ascontiguousarray(value)).all():
                raise ValueError(f"constant-zero layout would lose data: {field.path}")
            arrays[f"{field.path}__constant_zero"] = np.asarray(True)
        arrays[f"{field.path}__defined"] = mask

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    return path


def read_typed_sidecar_shard(
    path: str | Path,
    *,
    contract: TypedSidecarContract,
    expected_days: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    if "day_index" not in arrays:
        raise ValueError("typed-sidecar shard is missing day_index")
    day_index = arrays.pop("day_index")
    if day_index.dtype != np.dtype(np.int32) or day_index.ndim != 1:
        raise ValueError("typed-sidecar day_index must be one-dimensional int32")
    if expected_days is not None and day_index.size != expected_days:
        raise ValueError("typed-sidecar day count mismatch")
    if day_index.size != np.unique(day_index).size:
        raise ValueError("typed-sidecar day_index contains duplicate days")
    if day_index.size > 1 and not np.all(np.diff(day_index) == 1):
        raise ValueError("typed-sidecar day_index contains missing days")

    values: dict[str, np.ndarray] = {}
    defined: dict[str, np.ndarray] = {}
    consumed: set[str] = set()
    for field in contract.fields:
        shape = (day_index.size, *field.feature_shape)
        dense = field.path
        index_name = f"{field.path}__coo_index"
        value_name = f"{field.path}__coo_value"
        zero_name = f"{field.path}__constant_zero"
        modes = [dense in arrays, index_name in arrays or value_name in arrays, zero_name in arrays]
        if sum(modes) != 1:
            raise ValueError(f"typed-sidecar field has ambiguous/missing layout: {field.path}")
        if modes[0]:
            value = arrays[dense]
            consumed.add(dense)
        elif modes[1]:
            if index_name not in arrays or value_name not in arrays:
                raise ValueError(f"incomplete typed-sidecar COO field: {field.path}")
            index = arrays[index_name]
            sparse = arrays[value_name]
            if index.dtype != np.dtype(np.uint32) or index.ndim != 1:
                raise ValueError(f"typed-sidecar COO index drift: {field.path}")
            if sparse.dtype != np.dtype(np.float64) or sparse.shape != index.shape:
                raise ValueError(f"typed-sidecar COO value drift: {field.path}")
            if index.size and (
                np.any(index[1:] <= index[:-1])
                or int(index[-1]) >= int(np.prod(shape, dtype=np.int64))
            ):
                raise ValueError(f"typed-sidecar COO index is invalid: {field.path}")
            value = np.zeros(shape, dtype=np.float64)
            value.reshape(-1)[index] = sparse
            consumed.update((index_name, value_name))
        else:
            marker = arrays[zero_name]
            if marker.shape != () or marker.dtype != np.dtype(np.bool_) or not bool(marker):
                raise ValueError(f"typed-sidecar zero marker drift: {field.path}")
            value = np.zeros(shape, dtype=np.float64)
            consumed.add(zero_name)
        if value.shape != shape or value.dtype != np.dtype(field.dtype):
            raise ValueError(f"typed-sidecar shape/dtype drift for {field.path}")

        mask_name = f"{field.path}__defined"
        if mask_name not in arrays:
            raise ValueError(f"typed-sidecar defined mask is missing: {field.path}")
        mask = arrays[mask_name]
        if mask.shape != shape or mask.dtype != np.dtype(np.bool_):
            raise ValueError(f"typed-sidecar defined-mask drift for {field.path}")
        invalid_defined = mask & (
            ~np.isfinite(value) | (np.abs(value) >= ORCHIDEE_UNDEFINED_THRESHOLD)
        )
        if np.any(invalid_defined):
            raise ValueError(
                f"defined typed-sidecar values must be finite/non-sentinel: {field.path}"
            )
        consumed.add(mask_name)
        values[field.path] = value
        defined[field.path] = mask

    extra = set(arrays) - consumed
    if extra:
        raise ValueError(f"typed-sidecar shard has undeclared arrays: {sorted(extra)}")
    return values, defined, day_index


def _label_tree(
    arrays: Mapping[str, np.ndarray],
    contract: TypedSidecarContract,
) -> SupplementalProcessPrediction:
    domains: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "water": {},
        "carbon": {},
        "energy": {},
    }
    for field in contract.fields:
        domain, label_name = field.label_id.split(".", maxsplit=1)
        field_name = field.path.split(".", maxsplit=1)[1]
        domains[domain].setdefault(label_name, {})[field_name] = arrays[field.path]
    return SupplementalProcessPrediction(**domains)


def _typed_sample(
    parent: Mapping[str, Any],
    *,
    values: Mapping[str, np.ndarray],
    defined: Mapping[str, np.ndarray],
    contract: TypedSidecarContract,
    spatial_split: str,
    temporal_split: str,
) -> dict[str, Any]:
    inputs = {
        name: parent[name]
        for name in (
            "state",
            "forcing_native",
            "parameters",
            "landpoint_static",
            "annual_conditions",
            "year",
            "day_index",
            "discrete_state",
        )
    }
    targets = {
        "supplemental": TypedSupplementalBatch(
            values=_label_tree(values, contract),
            defined=_label_tree(defined, contract),
        ),
        "next_state": parent["next_state"],
        "next_discrete_state": parent["next_discrete_state"],
        "diagnostics": parent["diagnostics"],
    }
    return {
        "inputs": inputs,
        "targets": targets,
        "sample_spatial_split": spatial_split,
        "sample_temporal_split": temporal_split,
    }


def _stack_mapping(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = tuple(values[0])
    if any(tuple(value) != keys for value in values[1:]):
        raise ValueError("typed-sidecar collation schema drift")
    result: dict[str, Any] = {}
    for key in keys:
        children = [value[key] for value in values]
        if isinstance(children[0], Mapping):
            result[key] = _stack_mapping(children)
        else:
            result[key] = np.stack([np.asarray(child) for child in children])
    return result


def _stack_prediction(
    values: Sequence[SupplementalProcessPrediction],
) -> SupplementalProcessPrediction:
    return SupplementalProcessPrediction(
        water=_stack_mapping([value.water for value in values]),
        carbon=_stack_mapping([value.carbon for value in values]),
        energy=_stack_mapping([value.energy for value in values]),
    )


def collate_typed_sidecar_samples(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Collate named inputs and typed targets without the anonymous fast target."""

    if not samples:
        raise ValueError("cannot collate an empty typed-sidecar sample sequence")
    inputs = _stack_mapping([sample["inputs"] for sample in samples])
    targets = [sample["targets"] for sample in samples]
    supplemental = [target["supplemental"] for target in targets]
    result_targets = {
        "supplemental": TypedSupplementalBatch(
            values=_stack_prediction([item.values for item in supplemental]),
            defined=_stack_prediction([item.defined for item in supplemental]),
        ),
        "next_state": np.stack([target["next_state"] for target in targets]),
        "diagnostics": np.stack([target["diagnostics"] for target in targets]),
        "next_discrete_state": _stack_mapping(
            [target["next_discrete_state"] for target in targets]
        ),
    }
    return {"inputs": inputs, "targets": result_targets}


class _MaskedMoments:
    def __init__(self, feature_shape: tuple[int, ...]):
        self.feature_shape = feature_shape
        self.width = int(np.prod(feature_shape, dtype=np.int64))
        self.width = max(self.width, 1)
        self.count = np.zeros(self.width, dtype=np.uint64)
        self.mean = np.zeros(self.width, dtype=np.float64)
        self.m2 = np.zeros(self.width, dtype=np.float64)

    def _merge(
        self,
        batch_count: np.ndarray,
        batch_mean: np.ndarray,
        batch_m2: np.ndarray,
    ) -> None:
        total = self.count + batch_count
        active = batch_count > 0
        delta = batch_mean - self.mean
        weight = np.zeros_like(self.mean)
        np.divide(batch_count, total, out=weight, where=active)
        cross_weight = np.zeros_like(self.mean)
        np.divide(
            self.count * batch_count,
            total,
            out=cross_weight,
            where=active,
        )
        self.mean = np.where(active, self.mean + delta * weight, self.mean)
        self.m2 = np.where(
            active,
            self.m2 + batch_m2 + delta * delta * cross_weight,
            self.m2,
        )
        self.count = total

    def update(self, values: np.ndarray, defined: np.ndarray) -> None:
        rows = np.asarray(values, dtype=np.float64).reshape(-1, self.width)
        mask = np.asarray(defined, dtype=np.bool_).reshape(-1, self.width)
        valid = (
            mask
            & np.isfinite(rows)
            & (np.abs(rows) < ORCHIDEE_UNDEFINED_THRESHOLD)
        )
        safe = np.where(valid, rows, 0.0)
        batch_count = valid.sum(axis=0, dtype=np.uint64)
        batch_mean = np.zeros(self.width, dtype=np.float64)
        np.divide(
            safe.sum(axis=0),
            batch_count,
            out=batch_mean,
            where=batch_count > 0,
        )
        centered = np.where(valid, rows - batch_mean, 0.0)
        self._merge(
            batch_count,
            batch_mean,
            np.sum(centered * centered, axis=0),
        )

    def finalize(self, minimum_scale: float) -> "TypedFieldStatistics":
        variance = np.zeros(self.width, dtype=np.float64)
        np.divide(self.m2, self.count, out=variance, where=self.count > 0)
        variance = np.maximum(variance, 0.0)
        standard_deviation = np.sqrt(variance)
        scale = np.where(
            (self.count > 1) & (standard_deviation > minimum_scale),
            standard_deviation,
            1.0,
        )
        shape = self.feature_shape
        return TypedFieldStatistics(
            count=self.count.reshape(shape),
            mean=self.mean.reshape(shape),
            variance=variance.reshape(shape),
            scale=scale.reshape(shape),
        )


@dataclass(frozen=True)
class TypedFieldStatistics:
    count: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class TypedSidecarStatistics:
    dataset_id: str
    parent_contract_sha256: str
    parent_manifest_sha256: str
    sidecar_contract_sha256: str
    sidecar_manifest_sha256: str
    selection: Mapping[str, str]
    sample_count: int
    source_sidecars: tuple[str, ...]
    fields: Mapping[str, TypedFieldStatistics]


def fit_typed_sidecar_statistics(
    index: TypedSidecarIndex,
    *,
    minimum_scale: float = 1.0e-12,
) -> TypedSidecarStatistics:
    """Fit explicit-mask/finite-only statistics from train/train and nowhere else."""

    if minimum_scale <= 0.0:
        raise ValueError("minimum_scale must be positive")
    references = index.select(spatial_split="train", temporal_split="train")
    if not references:
        raise ValueError("typed-sidecar dataset has no train/train shards")
    moments = {
        field.path: _MaskedMoments(field.feature_shape)
        for field in index.contract.fields
    }
    sample_count = 0
    for reference in references:
        values, defined, _ = read_typed_sidecar_shard(
            reference.path,
            contract=index.contract,
            expected_days=reference.day_count,
        )
        for field in index.contract.fields:
            moments[field.path].update(values[field.path], defined[field.path])
        sample_count += reference.day_count
    return TypedSidecarStatistics(
        dataset_id=index.parent_index.dataset_id,
        parent_contract_sha256=index.parent_index.contract_sha256,
        parent_manifest_sha256=index.parent_manifest_sha256,
        sidecar_contract_sha256=index.contract.sha256,
        sidecar_manifest_sha256=index.sidecar_manifest_sha256,
        selection={"spatial_split": "train", "temporal_split": "train"},
        sample_count=sample_count,
        source_sidecars=tuple(
            f"{ref.parent.landpoint_id}:{ref.parent.year}:{ref.sha256}"
            for ref in references
        ),
        fields={
            path: value.finalize(minimum_scale) for path, value in moments.items()
        },
    )


def write_typed_sidecar_statistics(
    statistics: TypedSidecarStatistics,
    path: str | Path,
) -> Path:
    metadata_path = Path(path)
    if metadata_path.suffix != ".json":
        metadata_path = metadata_path.with_suffix(".json")
    arrays_path = metadata_path.with_suffix(".npz")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"{name}__{member}": np.asarray(getattr(value, member))
        for name, value in statistics.fields.items()
        for member in ("count", "mean", "variance", "scale")
    }
    np.savez_compressed(arrays_path, **arrays)
    payload = {
        "schema_version": SIDECAR_STATISTICS_SCHEMA_VERSION,
        "dataset_id": statistics.dataset_id,
        "parent_contract_sha256": statistics.parent_contract_sha256,
        "parent_manifest_sha256": statistics.parent_manifest_sha256,
        "sidecar_contract_sha256": statistics.sidecar_contract_sha256,
        "sidecar_manifest_sha256": statistics.sidecar_manifest_sha256,
        "selection": dict(statistics.selection),
        "sample_count": statistics.sample_count,
        "source_sidecars": list(statistics.source_sidecars),
        "statistics_npz": arrays_path.name,
        "statistics_npz_sha256": _sha256_file(arrays_path),
        "validity_policy": {
            "explicit_defined_mask": True,
            "finite": True,
            "excluded_absolute_value_gte": ORCHIDEE_UNDEFINED_THRESHOLD,
        },
        "fields": {
            name: {
                "feature_shape": list(value.mean.shape),
                "never_defined_columns": int(np.count_nonzero(value.count == 0)),
            }
            for name, value in statistics.fields.items()
        },
    }
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def load_typed_sidecar_statistics(
    path: str | Path,
    *,
    index: TypedSidecarIndex | None = None,
) -> TypedSidecarStatistics:
    metadata_path = Path(path).resolve()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SIDECAR_STATISTICS_SCHEMA_VERSION:
        raise ValueError("unsupported typed-sidecar statistics schema")
    if payload.get("selection") != {
        "spatial_split": "train",
        "temporal_split": "train",
    }:
        raise ValueError("typed-sidecar statistics are not train/train-only")
    expected_validity = {
        "explicit_defined_mask": True,
        "finite": True,
        "excluded_absolute_value_gte": ORCHIDEE_UNDEFINED_THRESHOLD,
    }
    if payload.get("validity_policy") != expected_validity:
        raise ValueError("typed-sidecar statistics validity policy mismatch")
    arrays_path = metadata_path.parent / str(payload["statistics_npz"])
    if _sha256_file(arrays_path) != payload.get("statistics_npz_sha256"):
        raise ValueError("typed-sidecar statistics array hash mismatch")
    if index is not None:
        identity = (
            payload.get("dataset_id"),
            payload.get("parent_contract_sha256"),
            payload.get("parent_manifest_sha256"),
            payload.get("sidecar_contract_sha256"),
            payload.get("sidecar_manifest_sha256"),
        )
        expected = (
            index.parent_index.dataset_id,
            index.parent_index.contract_sha256,
            index.parent_manifest_sha256,
            index.contract.sha256,
            index.sidecar_manifest_sha256,
        )
        if identity != expected:
            raise ValueError("typed-sidecar statistics dataset identity mismatch")
    with np.load(arrays_path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    fields = {}
    expected_arrays = set()
    for name, metadata in payload.get("fields", {}).items():
        members = {}
        for member in ("count", "mean", "variance", "scale"):
            key = f"{name}__{member}"
            if key not in arrays:
                raise ValueError(f"typed-sidecar statistics field is missing: {key}")
            expected_arrays.add(key)
            members[member] = arrays[key]
        shape = tuple(int(size) for size in metadata["feature_shape"])
        if any(value.shape != shape for value in members.values()):
            raise ValueError(f"typed-sidecar statistics shape drift for {name}")
        fields[name] = TypedFieldStatistics(**members)
    if set(arrays) != expected_arrays:
        raise ValueError("typed-sidecar statistics contain undeclared arrays")
    if index is not None and set(fields) != set(index.contract.fields_by_path):
        raise ValueError("typed-sidecar statistics field contract mismatch")
    return TypedSidecarStatistics(
        dataset_id=str(payload["dataset_id"]),
        parent_contract_sha256=str(payload["parent_contract_sha256"]),
        parent_manifest_sha256=str(payload["parent_manifest_sha256"]),
        sidecar_contract_sha256=str(payload["sidecar_contract_sha256"]),
        sidecar_manifest_sha256=str(payload["sidecar_manifest_sha256"]),
        selection=dict(payload["selection"]),
        sample_count=int(payload["sample_count"]),
        source_sidecars=tuple(payload["source_sidecars"]),
        fields=fields,
    )
