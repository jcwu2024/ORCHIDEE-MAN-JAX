"""Generate and aggregate the bounded Gate-E2 typed-sidecar pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from jax_orchidee.driver import orchestration as teacher
from research.daily_coarse_graining.daily_flux_capture import (
    daily_flux_capture_arrays,
)
from research.daily_coarse_graining.daily_markov_contract import (
    daily_markov_contract_from_metadata,
    extract_state,
    load_markov_shard,
)
from research.daily_coarse_graining.markov_dataset import load_dataset_index
from research.daily_coarse_graining.replay_ceiling import (
    capture_pre_daily_stomate_record,
)
from research.daily_coarse_graining.teacher_shards import (
    COLD_START_BOOTSTRAP,
    REFERENCE_INPUT_NAMES,
    PlanEntry,
    _validate_landpoint_run_def_binding,
    load_plan,
)
from research.daily_coarse_graining.typed_sidecar import (
    TypedSidecarContract,
    load_typed_sidecar_contract,
    read_typed_sidecar_shard,
    write_typed_sidecar_shard,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PILOT_PLAN = (
    ROOT / "manifests" / "coarse_graining" / "gate_e2_typed_sidecar_pilot_v1.json"
)
PILOT_SCHEMA_VERSION = "gate_e2_typed_sidecar_pilot_v1"
TASK_SCHEMA_VERSION = "gate_e2_typed_sidecar_pilot_task_v1"
REPORT_SCHEMA_VERSION = "gate_e2_typed_sidecar_pilot_report_v1"
SIDECAR_DATASET_SCHEMA_VERSION = "daily_typed_sidecar_dataset_v1"
STATE_COMPARISON_SCHEMA_VERSION = "gate_e2_state_comparison_v1"
FULL_PARENT_TRANSITIONS = 12_208_581
MAX_MISMATCH_EXAMPLES = 8


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any], *, omit: str | None = None) -> str:
    payload = dict(value)
    if omit is not None:
        payload.pop(omit, None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _clean_git_head() -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
    ).strip()
    if status:
        raise RuntimeError("typed-sidecar production requires a clean committed worktree")
    return head


@dataclass(frozen=True)
class PilotEntry:
    task_index: int
    landpoint_id: str
    year: int


@dataclass(frozen=True)
class PilotPlan:
    path: Path
    sha256: str
    raw: Mapping[str, Any]
    entries: tuple[PilotEntry, ...]


@dataclass(frozen=True)
class CapabilityMasks:
    active_pft: np.ndarray
    vegetation_pft: np.ndarray
    leak_carbon_pft: np.ndarray
    peat_pft: np.ndarray
    soil_tile: np.ndarray
    snow: bool
    interface: bool
    routing: bool
    peat: bool

    def metadata(self) -> dict[str, Any]:
        return {
            "active_pft": self.active_pft.tolist(),
            "vegetation_pft": self.vegetation_pft.tolist(),
            "leak_carbon_pft": self.leak_carbon_pft.tolist(),
            "peat_pft": self.peat_pft.tolist(),
            "soil_tile": self.soil_tile.tolist(),
            "snow": self.snow,
            "interface": self.interface,
            "routing": self.routing,
            "peat": self.peat,
        }


def load_pilot_plan(path: str | Path = DEFAULT_PILOT_PLAN) -> PilotPlan:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != PILOT_SCHEMA_VERSION:
        raise ValueError("unsupported Gate-E2 pilot schema")
    actual_hash = _canonical_sha256(raw, omit="pilot_sha256")
    if raw.get("pilot_sha256") != actual_hash:
        raise ValueError("Gate-E2 pilot self-hash mismatch")
    for section, path_key, hash_key in (
        (
            raw["parent"],
            "production_spec",
            "production_spec_canonical_sha256",
        ),
        (raw["sidecar_contract"], "path", "canonical_sha256"),
        (raw["selection_source"], "path", "canonical_sha256"),
    ):
        source = (ROOT / str(section[path_key])).resolve()
        try:
            source.relative_to(ROOT)
        except ValueError as error:
            raise ValueError(f"pilot source path escapes repository: {source}") from error
        source_raw = json.loads(source.read_text(encoding="utf-8"))
        if _canonical_sha256(source_raw) != section[hash_key]:
            raise ValueError(f"pilot source hash mismatch: {source}")
    sidecar_contract = load_typed_sidecar_contract(
        ROOT / str(raw["sidecar_contract"]["path"])
    )
    if sidecar_contract.sha256 != raw["sidecar_contract"]["contract_sha256"]:
        raise ValueError("pilot typed-sidecar contract identity mismatch")
    teacher_config = (ROOT / str(raw["teacher_runtime"]["config"])).resolve()
    try:
        teacher_config.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(
            f"pilot Teacher config path escapes repository: {teacher_config}"
        ) from error
    config_raw = yaml.safe_load(teacher_config.read_text(encoding="utf-8"))
    if (
        _canonical_sha256(config_raw)
        != raw["teacher_runtime"]["config_canonical_sha256"]
    ):
        raise ValueError("pilot Teacher config hash mismatch")
    entries = tuple(
        PilotEntry(
            task_index=int(item["task_index"]),
            landpoint_id=str(item["landpoint_id"]),
            year=int(item["year"]),
        )
        for item in raw["entries"]
    )
    if tuple(entry.task_index for entry in entries) != tuple(range(len(entries))):
        raise ValueError("pilot task indices must be contiguous and zero based")
    keys = {(entry.landpoint_id, entry.year) for entry in entries}
    if len(keys) != len(entries):
        raise ValueError("pilot entries must be unique")
    execution = raw["execution"]
    if any(entry.year != int(execution["year"]) for entry in entries):
        raise ValueError("pilot entries must use the frozen execution year")
    return PilotPlan(path=path, sha256=actual_hash, raw=raw, entries=entries)


def _pilot_teacher_config(pilot: PilotPlan) -> Path:
    return (ROOT / str(pilot.raw["teacher_runtime"]["config"])).resolve()


def capability_masks_from_context(context) -> CapabilityMasks:
    active = np.asarray(context.run_scalars.active_pft_mask, dtype=np.bool_)
    entries = context.run_scalars.pft_layout.entries
    if len(entries) != active.size:
        raise ValueError("PFT capability layout does not match the active-PFT axis")
    vegetation = active & np.asarray(
        ["sechiba_vegetation" in entry.capabilities for entry in entries],
        dtype=np.bool_,
    )
    leak_carbon = active & np.asarray(
        ["leak_carbon" in entry.capabilities for entry in entries],
        dtype=np.bool_,
    )
    is_peat = np.asarray(context.run_scalars.is_peat, dtype=np.bool_)
    if is_peat.shape != active.shape:
        raise ValueError("PFT peat traits do not match the active-PFT axis")
    preference = np.asarray(context.run_scalars.pref_soil_veg, dtype=np.int32)
    soil_tile = np.zeros(int(context.run_scalars.nstm), dtype=np.bool_)
    selected = preference[active] - 1
    if selected.size and (np.min(selected) < 0 or np.max(selected) >= soil_tile.size):
        raise ValueError("active PFT soil-tile preference is outside the source layout")
    soil_tile[selected] = True
    perma_peat = str(context.run_def_values.get("PERMA_PEAT", "FALSE")).strip().upper()
    if perma_peat not in {"TRUE", "FALSE"}:
        raise ValueError("PERMA_PEAT must be an explicit logical value")
    return CapabilityMasks(
        active_pft=active,
        vegetation_pft=vegetation,
        leak_carbon_pft=leak_carbon,
        peat_pft=active & is_peat & leak_carbon,
        soil_tile=soil_tile,
        snow=bool(context.ok_explicitsnow),
        interface=True,
        routing=bool(context.river_routing),
        peat=perma_peat == "TRUE",
    )


def build_defined_masks(
    values: Mapping[str, np.ndarray],
    *,
    contract: TypedSidecarContract,
    capability: CapabilityMasks,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for field in contract.fields:
        value = np.asarray(values[field.path])
        expected_shape = (value.shape[0], *field.feature_shape)
        if value.shape != expected_shape or value.dtype != np.dtype(np.float64):
            raise ValueError(f"typed capture shape/dtype drift for {field.path}")
        defined = np.ones(value.shape, dtype=np.bool_)
        for component in field.mask_components:
            if component == "finite":
                defined &= np.isfinite(value) & (np.abs(value) < 0.5e20)
                continue
            if component in {"snow", "interface", "routing", "peat"}:
                defined &= bool(getattr(capability, component))
                continue
            axis_name = "soil_tile" if component == "soil_tile" else "pft"
            if axis_name not in field.axes:
                raise ValueError(
                    f"mask component {component} has no matching axis for {field.path}"
                )
            axis = field.axes.index(axis_name)
            source = np.asarray(getattr(capability, component), dtype=np.bool_)
            shape = [1] * value.ndim
            shape[axis] = source.size
            defined &= source.reshape(shape)
        result[field.path] = defined
    return result


def _json_number(value: float) -> float | str:
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return float(value)


def _float64_ulp_distance(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    observed = np.ascontiguousarray(observed, dtype=np.float64)
    expected = np.ascontiguousarray(expected, dtype=np.float64)
    sign = np.uint64(1 << 63)

    def ordered(values: np.ndarray) -> np.ndarray:
        bits = values.view(np.uint64)
        return np.where((bits & sign) != 0, ~bits, bits | sign)

    observed_ordered = ordered(observed)
    expected_ordered = ordered(expected)
    return np.maximum(observed_ordered, expected_ordered) - np.minimum(
        observed_ordered, expected_ordered
    )


def _continuous_state_mismatches(
    observed: np.ndarray,
    expected: np.ndarray,
    contract,
) -> list[dict[str, Any]]:
    equal = (observed == expected) | (np.isnan(observed) & np.isnan(expected))
    mismatched = ~equal
    reports: list[dict[str, Any]] = []
    for leaf in contract.state_leaves:
        if leaf.discrete:
            continue
        start = int(leaf.start)
        stop = int(leaf.stop)
        local_positions = np.flatnonzero(mismatched[start:stop])
        if not local_positions.size:
            continue
        observed_values = observed[start:stop][local_positions]
        expected_values = expected[start:stop][local_positions]
        absolute = np.abs(observed_values - expected_values)
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = absolute / np.abs(expected_values)
        finite_pairs = np.isfinite(observed_values) & np.isfinite(expected_values)
        ulp = np.full(local_positions.size, np.uint64(0), dtype=np.uint64)
        ulp[finite_pairs] = _float64_ulp_distance(
            observed_values[finite_pairs], expected_values[finite_pairs]
        )
        examples = []
        for index in range(min(local_positions.size, MAX_MISMATCH_EXAMPLES)):
            local_flat = int(local_positions[index])
            examples.append(
                {
                    "compact_index": start + local_flat,
                    "leaf_flat_index": local_flat,
                    "leaf_index": list(np.unravel_index(local_flat, leaf.shape)),
                    "expected": _json_number(float(expected_values[index])),
                    "observed": _json_number(float(observed_values[index])),
                    "absolute_error": _json_number(float(absolute[index])),
                    "relative_error": _json_number(float(relative[index])),
                    "ulp_error": int(ulp[index]) if finite_pairs[index] else None,
                }
            )
        reports.append(
            {
                "key": leaf.key,
                "owner": leaf.component,
                "source_ref": leaf.source_ref,
                "classification": leaf.classification,
                "compact_span": [start, stop],
                "shape": list(leaf.shape),
                "axis_names": list(leaf.axis_names),
                "selected_pft_indices": list(leaf.selected_pft_indices),
                "mismatch_count": int(local_positions.size),
                "max_absolute_error": _json_number(float(np.max(absolute))),
                "max_relative_error": _json_number(float(np.max(relative))),
                "max_ulp_error": (
                    int(np.max(ulp[finite_pairs])) if np.any(finite_pairs) else None
                ),
                "examples": examples,
            }
        )
    return reports


def _discrete_state_mismatches(
    observed: Mapping[str, np.ndarray],
    expected: Mapping[str, np.ndarray],
    contract,
) -> list[dict[str, Any]]:
    leaves = {leaf.key: leaf for leaf in contract.discrete_leaves}
    reports: list[dict[str, Any]] = []
    for name in sorted(set(observed) | set(expected)):
        leaf = leaves.get(name)
        if name not in observed or name not in expected:
            reports.append(
                {
                    "key": name,
                    "owner": None if leaf is None else leaf.component,
                    "source_ref": None if leaf is None else leaf.source_ref,
                    "status": "missing_observed" if name not in observed else "missing_expected",
                }
            )
            continue
        actual = np.asarray(observed[name])
        wanted = np.asarray(expected[name])
        if actual.shape != wanted.shape:
            reports.append(
                {
                    "key": name,
                    "owner": None if leaf is None else leaf.component,
                    "source_ref": None if leaf is None else leaf.source_ref,
                    "status": "shape_drift",
                    "observed_shape": list(actual.shape),
                    "expected_shape": list(wanted.shape),
                }
            )
            continue
        equal = (actual == wanted) | (
            np.issubdtype(actual.dtype, np.floating)
            and np.issubdtype(wanted.dtype, np.floating)
            and np.isnan(actual)
            & np.isnan(wanted)
        )
        positions = np.argwhere(~equal)
        if not positions.size:
            continue
        examples = []
        for position in positions[:MAX_MISMATCH_EXAMPLES]:
            index = tuple(int(item) for item in position)
            expected_value = np.asarray(wanted[index]).item()
            observed_value = np.asarray(actual[index]).item()
            examples.append(
                {
                    "index": list(index),
                    "expected": expected_value,
                    "observed": observed_value,
                }
            )
        reports.append(
            {
                "key": name,
                "owner": None if leaf is None else leaf.component,
                "source_ref": None if leaf is None else leaf.source_ref,
                "status": "different",
                "shape": list(actual.shape),
                "mismatch_count": int(positions.shape[0]),
                "examples": examples,
            }
        )
    return reports


def _maximum_error(
    reports: Sequence[Mapping[str, Any]], key: str
) -> float | str | int | None:
    values = [item[key] for item in reports if item.get(key) is not None]
    if not values:
        return None
    if "inf" in values or "-inf" in values:
        return "inf"
    finite = [float(value) for value in values if value != "nan"]
    if not finite:
        return "nan"
    maximum = max(finite)
    return int(maximum) if key == "max_ulp_error" else maximum


def _state_comparison_report(packet, expected, contract, *, label: str) -> dict[str, Any]:
    continuous, discrete = extract_state(packet, contract)
    wanted_continuous = np.asarray(expected[0], dtype=np.float64)
    if continuous.shape != wanted_continuous.shape:
        raise ValueError(
            f"{label} continuous state shape drift: "
            f"{continuous.shape} != {wanted_continuous.shape}"
        )
    continuous_reports = _continuous_state_mismatches(
        continuous, wanted_continuous, contract
    )
    discrete_reports = _discrete_state_mismatches(discrete, expected[1], contract)
    return {
        "schema_version": STATE_COMPARISON_SCHEMA_VERSION,
        "status": "passed" if not continuous_reports and not discrete_reports else "mismatch",
        "label": label,
        "continuous_width": int(continuous.size),
        "continuous_mismatch_count": int(
            sum(item["mismatch_count"] for item in continuous_reports)
        ),
        "continuous_mismatch_leaf_count": len(continuous_reports),
        "max_absolute_error": _maximum_error(
            continuous_reports, "max_absolute_error"
        ),
        "max_relative_error": _maximum_error(
            continuous_reports, "max_relative_error"
        ),
        "max_ulp_error": _maximum_error(continuous_reports, "max_ulp_error"),
        "continuous_leaves": continuous_reports,
        "discrete_mismatch_leaf_count": len(discrete_reports),
        "discrete_leaves": discrete_reports,
    }


def _assert_state_matches(
    packet,
    expected,
    contract,
    *,
    label: str,
    report_path: Path | None = None,
    report_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = _state_comparison_report(packet, expected, contract, label=label)
    if report["status"] == "passed":
        return report
    persisted = dict(report_context or {}) | report
    if report_path is not None:
        _atomic_json(report_path, persisted)
    if report["continuous_leaves"]:
        first = report["continuous_leaves"][0]
        detail = (
            f"first_owner={first['owner']} first_key={first['key']} "
            f"max_abs={first['max_absolute_error']} max_ulp={first['max_ulp_error']}"
        )
    else:
        first = report["discrete_leaves"][0]
        detail = f"first_discrete_key={first['key']}"
    location = "" if report_path is None else f"; report={report_path}"
    raise ValueError(f"{label} state differs from immutable parent; {detail}{location}")


def _layout_metadata(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        members = {
            item.filename: {
                "compressed_bytes": item.compress_size,
                "uncompressed_bytes": item.file_size,
            }
            for item in archive.infolist()
        }
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "members": members,
    }


def _verify_layout(
    path: Path,
    *,
    contract: TypedSidecarContract,
    values: Mapping[str, np.ndarray],
    defined: Mapping[str, np.ndarray],
    day_index: np.ndarray,
) -> None:
    restored, masks, days = read_typed_sidecar_shard(
        path, contract=contract, expected_days=day_index.size
    )
    if not np.array_equal(days, day_index):
        raise ValueError("typed sidecar changed parent day indices")
    for field in contract.fields:
        if restored[field.path].tobytes() != values[field.path].tobytes():
            raise ValueError(f"typed sidecar is not bit-exact for {field.path}")
        if not np.array_equal(masks[field.path], defined[field.path]):
            raise ValueError(f"typed sidecar changed defined mask for {field.path}")


def _selected_parent(parent_index, entry: PilotEntry):
    matches = tuple(
        ref
        for ref in parent_index.shards
        if ref.landpoint_id == entry.landpoint_id and ref.year == entry.year
    )
    if len(matches) != 1:
        raise ValueError(f"pilot parent selection has {len(matches)} matches for {entry}")
    return matches[0]


def _selected_teacher_entry(plan, entry: PilotEntry) -> PlanEntry:
    matches = tuple(
        item
        for item in plan.entries
        if item.landpoint_id == entry.landpoint_id and item.year == entry.year
    )
    if len(matches) != 1:
        raise ValueError(f"Teacher plan selection has {len(matches)} matches for {entry}")
    selected = matches[0]
    if selected.initialization_mode != COLD_START_BOOTSTRAP:
        raise ValueError("the frozen pilot requires canonical 1961 cold start")
    return selected


def _require_selected_teacher_inputs(
    entry: PlanEntry,
    *,
    teacher_config: Path,
) -> None:
    required = [
        teacher_config,
        entry.run_def,
        *(entry.reference_run_dir / name for name in REFERENCE_INPUT_NAMES),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "selected Teacher inputs are missing: " + ", ".join(missing)
        )
    _validate_landpoint_run_def_binding(entry)


def preflight_pilot(
    pilot: PilotPlan,
    *,
    parent_manifest: Path,
    teacher_plan_path: Path,
    task_index: int | None = None,
    expected_git_head: str | None = None,
) -> dict[str, Any]:
    git_head = _clean_git_head()
    if expected_git_head is not None and git_head != expected_git_head:
        raise ValueError("Gate-E2 preflight Git HEAD mismatch")
    parent_manifest = parent_manifest.resolve()
    if _sha256_file(parent_manifest) != pilot.raw["parent"]["dataset_manifest_sha256"]:
        raise ValueError("pilot parent dataset-manifest hash mismatch")
    parent_raw = json.loads(parent_manifest.read_text(encoding="utf-8"))
    if parent_raw.get("dataset_id") != pilot.raw["parent"]["dataset_id"]:
        raise ValueError("pilot parent dataset ID mismatch")
    if (
        parent_raw.get("markov_contract_sha256")
        != pilot.raw["parent"]["markov_contract_sha256"]
    ):
        raise ValueError("pilot parent Markov contract mismatch")
    parent_index = load_dataset_index(parent_manifest, verify_hashes=False)
    generation_plan = load_plan(teacher_plan_path.resolve(), require_inputs=False)
    if generation_plan.dataset_id != parent_index.dataset_id:
        raise ValueError("Teacher generation plan and parent dataset ID differ")
    contract = load_typed_sidecar_contract()
    if contract.sha256 != pilot.raw["sidecar_contract"]["contract_sha256"]:
        raise ValueError("runtime typed-sidecar contract identity mismatch")
    teacher_config = _pilot_teacher_config(pilot)
    selected = pilot.entries
    if task_index is not None:
        if task_index < 0 or task_index >= len(selected):
            raise ValueError("pilot task index is outside the frozen entry inventory")
        selected = (selected[task_index],)

    expected_start = int(pilot.raw["execution"]["expected_day_start"])
    expected_stop = int(pilot.raw["execution"]["expected_day_stop"])
    expected_days = int(pilot.raw["execution"]["expected_days_per_entry"])
    markov_contract = daily_markov_contract_from_metadata(
        parent_raw["markov_contract"]
    )
    reports = []
    for entry in selected:
        parent_ref = _selected_parent(parent_index, entry)
        if _sha256_file(parent_ref.path) != parent_ref.sha256:
            raise ValueError(f"selected parent shard hash mismatch: {entry}")
        if (parent_ref.spatial_split, parent_ref.temporal_split) != (
            pilot.raw["execution"]["spatial_split"],
            pilot.raw["execution"]["temporal_split"],
        ):
            raise ValueError(f"selected parent shard split drift: {entry}")
        parent = load_markov_shard(parent_ref.path, contract=markov_contract)
        if (
            parent.days != expected_days
            or int(parent.day_index[0]) != expected_start
            or int(parent.day_index[-1]) != expected_stop
        ):
            raise ValueError(f"selected parent shard day inventory drift: {entry}")
        teacher_entry = _selected_teacher_entry(generation_plan, entry)
        _require_selected_teacher_inputs(
            teacher_entry,
            teacher_config=teacher_config,
        )
        reports.append(
            {
                "task_index": entry.task_index,
                "landpoint_id": entry.landpoint_id,
                "year": entry.year,
                "parent_shard_sha256": parent_ref.sha256,
                "run_def_sha256": _sha256_file(teacher_entry.run_def),
                "reference_run_dir": str(teacher_entry.reference_run_dir),
            }
        )
    return {
        "status": "passed",
        "pilot_sha256": pilot.sha256,
        "git_head": git_head,
        "parent_dataset_manifest_sha256": _sha256_file(parent_manifest),
        "teacher_plan_sha256": _sha256_file(generation_plan.path),
        "teacher_config_canonical_sha256": pilot.raw["teacher_runtime"][
            "config_canonical_sha256"
        ],
        "entries": reports,
    }


def _task_layout_path(task_report_path: Path, item: Mapping[str, Any]) -> Path:
    path = (task_report_path.parent / str(item["path"])).resolve()
    try:
        path.relative_to(task_report_path.parent.resolve())
    except ValueError as error:
        raise ValueError(f"pilot sidecar path escapes its task directory: {path}") from error
    if _sha256_file(path) != item["sha256"]:
        raise ValueError(f"pilot sidecar hash mismatch: {path}")
    if path.stat().st_size != int(item["bytes"]):
        raise ValueError(f"pilot sidecar byte count mismatch: {path}")
    return path


def _capability_masks_from_metadata(raw: Mapping[str, Any]) -> CapabilityMasks:
    return CapabilityMasks(
        active_pft=np.asarray(raw["active_pft"], dtype=np.bool_),
        vegetation_pft=np.asarray(raw["vegetation_pft"], dtype=np.bool_),
        leak_carbon_pft=np.asarray(raw["leak_carbon_pft"], dtype=np.bool_),
        peat_pft=np.asarray(raw["peat_pft"], dtype=np.bool_),
        soil_tile=np.asarray(raw["soil_tile"], dtype=np.bool_),
        snow=bool(raw["snow"]),
        interface=bool(raw["interface"]),
        routing=bool(raw["routing"]),
        peat=bool(raw["peat"]),
    )


def generate_task(
    pilot: PilotPlan,
    *,
    task_index: int,
    parent_manifest: Path,
    teacher_plan_path: Path,
    output_root: Path,
    diagnostic_day_limit: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    git_head = _clean_git_head()
    if task_index < 0 or task_index >= len(pilot.entries):
        raise ValueError("pilot task index is outside the frozen entry inventory")
    entry = pilot.entries[task_index]
    parent_manifest = parent_manifest.resolve()
    if _sha256_file(parent_manifest) != pilot.raw["parent"]["dataset_manifest_sha256"]:
        raise ValueError("pilot parent dataset-manifest hash mismatch")
    parent_raw = json.loads(parent_manifest.read_text(encoding="utf-8"))
    if parent_raw.get("dataset_id") != pilot.raw["parent"]["dataset_id"]:
        raise ValueError("pilot parent dataset ID mismatch")
    parent_index = load_dataset_index(parent_manifest, verify_hashes=False)
    parent_ref = _selected_parent(parent_index, entry)
    if _sha256_file(parent_ref.path) != parent_ref.sha256:
        raise ValueError("selected immutable parent shard hash mismatch")
    if parent_ref.spatial_split != "train" or parent_ref.temporal_split != "train":
        raise ValueError("Gate-E2 pilot must remain train/train-only")
    markov_contract = daily_markov_contract_from_metadata(parent_raw["markov_contract"])
    parent = load_markov_shard(parent_ref.path, contract=markov_contract)

    generation_plan = load_plan(teacher_plan_path.resolve(), require_inputs=False)
    if generation_plan.dataset_id != parent_index.dataset_id:
        raise ValueError("Teacher generation plan and parent dataset ID differ")
    teacher_entry = _selected_teacher_entry(generation_plan, entry)
    teacher_config = _pilot_teacher_config(pilot)
    _require_selected_teacher_inputs(
        teacher_entry,
        teacher_config=teacher_config,
    )
    context = teacher.prepare_paper_1961_driver_context(
        teacher_config,
        used_run_def_path=teacher_entry.run_def,
        reference_run_dir=teacher_entry.reference_run_dir,
    )
    bootstrap = teacher.paper_1961_driver_cold_start_day_scaffold(
        teacher_config,
        year=entry.year,
        used_run_def_path=teacher_entry.run_def,
        reference_run_dir=teacher_entry.reference_run_dir,
        prepared_context=context,
    )
    if not bootstrap.ready_for_first_day_end_state:
        raise RuntimeError("pilot cold-start Day 1 did not produce canonical state")
    expected_start = int(pilot.raw["execution"]["expected_day_start"])
    expected_stop = int(pilot.raw["execution"]["expected_day_stop"])
    expected_days = int(pilot.raw["execution"]["expected_days_per_entry"])
    expected_index = np.arange(expected_start, expected_stop + 1, dtype=np.int32)
    if not np.array_equal(parent.day_index, expected_index) or parent.days != expected_days:
        raise ValueError("immutable parent does not match the frozen pilot day inventory")
    if diagnostic_day_limit is not None and not 1 <= diagnostic_day_limit <= parent.days:
        raise ValueError("diagnostic day limit must be within the parent day inventory")

    current = bootstrap.first_day_end_state
    contract = load_typed_sidecar_contract()
    if contract.sha256 != pilot.raw["sidecar_contract"]["contract_sha256"]:
        raise ValueError("runtime typed-sidecar contract identity mismatch")
    values_by_field: dict[str, list[np.ndarray]] = {
        field.path: [] for field in contract.fields
    }
    capability = capability_masks_from_context(context)
    steps_per_day = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    capture_seconds = 0.0
    task_root = output_root.resolve() / "tasks" / entry.landpoint_id / str(entry.year)
    comparison_path = task_root / "state_comparison.json"
    report_context = {
        "pilot_sha256": pilot.sha256,
        "teacher_git_head": git_head,
        "parent_shard_sha256": parent_ref.sha256,
        "task_index": entry.task_index,
        "landpoint_id": entry.landpoint_id,
        "year": entry.year,
        "diagnostic_day_limit": diagnostic_day_limit,
    }
    day_inventory = (
        parent.day_index
        if diagnostic_day_limit is None
        else parent.day_index[:diagnostic_day_limit]
    )
    final_comparison = None
    for row, day_index in enumerate(day_inventory):
        expected_discrete = {
            name: value[row] for name, value in parent.discrete_trajectories.items()
        }
        _assert_state_matches(
            current,
            (parent.state_trajectory[row], expected_discrete),
            markov_contract,
            label=f"Day {int(day_index)} start",
            report_path=comparison_path,
            report_context=report_context,
        )
        forcing = teacher._paper_compiled_forcing_day(
            context,
            year=entry.year,
            start_tstep=(int(day_index) - 1) * steps_per_day,
            steps_per_stomate=steps_per_day,
        )
        capture_started = time.perf_counter()
        record = capture_pre_daily_stomate_record(
            teacher_config,
            previous_state=current,
            year=entry.year,
            day_index=int(day_index),
            start_tstep=(int(day_index) - 1) * steps_per_day,
            used_run_def_path=context.run_def_path,
            prepared_context=context,
            module_jit=True,
            diffuco_local_jit=True,
            use_static_jit_daily_carbon=True,
            use_compiled_sechiba_day=True,
            retain_stomate_step_results=False,
            prebuild_day_payloads=True,
            compiled_forcing_series=forcing,
            capture_daily_flux_labels=True,
        )
        capture_seconds += time.perf_counter() - capture_started
        if record.daily_flux_labels is None:
            raise RuntimeError("Teacher did not return the requested daily flux labels")
        captured = daily_flux_capture_arrays(record.daily_flux_labels)
        for field in contract.fields:
            array = np.asarray(captured[field.path], dtype=np.float64)
            if array.shape != (1, *field.feature_shape):
                raise ValueError(f"captured source shape drift for {field.path}")
            values_by_field[field.path].append(np.ascontiguousarray(array[0]))
        current = record.expected_result.day_end_state
        expected_next_discrete = {
            name: value[row + 1] for name, value in parent.discrete_trajectories.items()
        }
        final_comparison = _assert_state_matches(
            current,
            (parent.state_trajectory[row + 1], expected_next_discrete),
            markov_contract,
            label=f"Day {int(day_index)} end",
            report_path=comparison_path,
            report_context=report_context,
        )

    if diagnostic_day_limit is not None:
        report = report_context | {
            "status": "passed",
            "checked_transition_count": int(diagnostic_day_limit),
            "first_day": int(day_inventory[0]),
            "last_day": int(day_inventory[-1]),
            "last_state_comparison": final_comparison,
            "teacher_capture_seconds": capture_seconds,
        }
        _atomic_json(comparison_path, report)
        return report

    _atomic_json(
        comparison_path,
        report_context
        | {
            "status": "passed",
            "checked_transition_count": int(parent.days),
            "first_day": int(day_inventory[0]),
            "last_day": int(day_inventory[-1]),
            "last_state_comparison": final_comparison,
            "teacher_capture_seconds": capture_seconds,
        },
    )

    values = {
        path: np.stack(rows).astype(np.float64, copy=False)
        for path, rows in values_by_field.items()
    }
    defined = build_defined_masks(values, contract=contract, capability=capability)
    task_root.mkdir(parents=True, exist_ok=True)
    layouts = {
        "dense": {field.path: "dense" for field in contract.fields},
        "hybrid_auto": {field.path: "auto" for field in contract.fields},
    }
    layout_reports = {}
    for name, requested in layouts.items():
        path = task_root / f"sidecar_{name}.npz"
        write_typed_sidecar_shard(
            path,
            contract=contract,
            day_index=parent.day_index,
            values=values,
            defined=defined,
            layouts=requested,
        )
        _verify_layout(
            path,
            contract=contract,
            values=values,
            defined=defined,
            day_index=parent.day_index,
        )
        layout_reports[name] = _layout_metadata(path)
    report = {
        "schema_version": TASK_SCHEMA_VERSION,
        "status": "complete",
        "pilot_sha256": pilot.sha256,
        "teacher_git_head": git_head,
        "task_index": entry.task_index,
        "landpoint_id": entry.landpoint_id,
        "year": entry.year,
        "spatial_split": parent_ref.spatial_split,
        "temporal_split": parent_ref.temporal_split,
        "parent_shard_sha256": parent_ref.sha256,
        "sidecar_contract_sha256": contract.sha256,
        "teacher_config_canonical_sha256": pilot.raw["teacher_runtime"][
            "config_canonical_sha256"
        ],
        "day_count": parent.days,
        "first_day": int(parent.day_index[0]),
        "last_day": int(parent.day_index[-1]),
        "state_replay": "exact_every_day_start_and_next_state",
        "capability_masks": capability.metadata(),
        "defined_counts": {
            field.path: int(np.count_nonzero(defined[field.path]))
            for field in contract.fields
        },
        "element_counts": {
            field.path: int(defined[field.path].size) for field in contract.fields
        },
        "layouts": layout_reports,
        "timing_seconds": {
            "teacher_capture": capture_seconds,
            "total": time.perf_counter() - started,
        },
    }
    _atomic_json(task_root / "task_report.json", report)
    return report


def aggregate_pilot(
    pilot: PilotPlan,
    *,
    parent_manifest: Path,
    output_root: Path,
) -> dict[str, Any]:
    parent_manifest = parent_manifest.resolve()
    output_root = output_root.resolve()
    if _sha256_file(parent_manifest) != pilot.raw["parent"]["dataset_manifest_sha256"]:
        raise ValueError("pilot parent dataset-manifest hash mismatch")
    parent_index = load_dataset_index(parent_manifest, verify_hashes=False)
    contract = load_typed_sidecar_contract()
    if contract.sha256 != pilot.raw["sidecar_contract"]["contract_sha256"]:
        raise ValueError("runtime typed-sidecar contract identity mismatch")
    expected_start = int(pilot.raw["execution"]["expected_day_start"])
    expected_stop = int(pilot.raw["execution"]["expected_day_stop"])
    expected_days = int(pilot.raw["execution"]["expected_days_per_entry"])
    reports = []
    for entry in pilot.entries:
        path = (
            output_root
            / "tasks"
            / entry.landpoint_id
            / str(entry.year)
            / "task_report.json"
        )
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("schema_version") != TASK_SCHEMA_VERSION or report.get("status") != "complete":
            raise ValueError(f"pilot task is incomplete: {path}")
        if report.get("pilot_sha256") != pilot.sha256:
            raise ValueError(f"pilot task plan drift: {path}")
        if (report.get("task_index"), report.get("landpoint_id"), report.get("year")) != (
            entry.task_index,
            entry.landpoint_id,
            entry.year,
        ):
            raise ValueError(f"pilot task identity drift: {path}")
        parent_ref = _selected_parent(parent_index, entry)
        if _sha256_file(parent_ref.path) != parent_ref.sha256:
            raise ValueError(f"pilot parent shard hash mismatch: {parent_ref.path}")
        if report.get("parent_shard_sha256") != parent_ref.sha256:
            raise ValueError(f"pilot task parent identity drift: {path}")
        if (
            report.get("spatial_split"),
            report.get("temporal_split"),
        ) != (parent_ref.spatial_split, parent_ref.temporal_split):
            raise ValueError(f"pilot task split drift: {path}")
        if (parent_ref.spatial_split, parent_ref.temporal_split) != (
            pilot.raw["execution"]["spatial_split"],
            pilot.raw["execution"]["temporal_split"],
        ):
            raise ValueError(f"pilot parent is outside the frozen train/train split: {path}")
        if report.get("sidecar_contract_sha256") != contract.sha256:
            raise ValueError(f"pilot task contract drift: {path}")
        if report.get("teacher_config_canonical_sha256") != pilot.raw[
            "teacher_runtime"
        ]["config_canonical_sha256"]:
            raise ValueError(f"pilot task Teacher config drift: {path}")
        if (
            report.get("day_count"),
            report.get("first_day"),
            report.get("last_day"),
            report.get("state_replay"),
        ) != (
            expected_days,
            expected_start,
            expected_stop,
            "exact_every_day_start_and_next_state",
        ):
            raise ValueError(f"pilot task day/state admission drift: {path}")

        decoded = {}
        for layout in ("dense", "hybrid_auto"):
            item = report["layouts"][layout]
            sidecar = _task_layout_path(path, item)
            decoded[layout] = read_typed_sidecar_shard(
                sidecar,
                contract=contract,
                expected_days=expected_days,
            )
        dense_values, dense_masks, dense_days = decoded["dense"]
        hybrid_values, hybrid_masks, hybrid_days = decoded["hybrid_auto"]
        expected_index = np.arange(expected_start, expected_stop + 1, dtype=np.int32)
        if not np.array_equal(dense_days, expected_index) or not np.array_equal(
            hybrid_days, expected_index
        ):
            raise ValueError(f"pilot sidecar day inventory drift: {path}")
        capability = _capability_masks_from_metadata(report["capability_masks"])
        source_masks = build_defined_masks(
            dense_values,
            contract=contract,
            capability=capability,
        )
        for field in contract.fields:
            name = field.path
            if dense_values[name].tobytes() != hybrid_values[name].tobytes():
                raise ValueError(f"pilot layouts differ for {name}: {path}")
            if not np.array_equal(dense_masks[name], hybrid_masks[name]):
                raise ValueError(f"pilot layout masks differ for {name}: {path}")
            if not np.array_equal(dense_masks[name], source_masks[name]):
                raise ValueError(f"pilot capability mask drift for {name}: {path}")
            if report["defined_counts"].get(name) != int(
                np.count_nonzero(dense_masks[name])
            ):
                raise ValueError(f"pilot defined-count drift for {name}: {path}")
            if report["element_counts"].get(name) != int(dense_masks[name].size):
                raise ValueError(f"pilot element-count drift for {name}: {path}")
        reports.append(report)
    heads = {report["teacher_git_head"] for report in reports}
    if len(heads) != 1:
        raise ValueError("pilot tasks used different source commits")
    totals = {
        layout: sum(int(report["layouts"][layout]["bytes"]) for report in reports)
        for layout in ("dense", "hybrid_auto")
    }
    accepted = min(totals, key=lambda name: (totals[name], name))
    total_days = sum(int(report["day_count"]) for report in reports)
    parent_raw = json.loads(parent_manifest.read_text(encoding="utf-8"))
    sidecar_manifest = {
        "schema_version": SIDECAR_DATASET_SCHEMA_VERSION,
        "pilot_sha256": pilot.sha256,
        "teacher_git_head": next(iter(heads)),
        "sidecar_contract_sha256": pilot.raw["sidecar_contract"][
            "contract_sha256"
        ],
        "parent": {
            "dataset_id": pilot.raw["parent"]["dataset_id"],
            "dataset_manifest_sha256": pilot.raw["parent"]["dataset_manifest_sha256"],
            "markov_contract_sha256": pilot.raw["parent"]["markov_contract_sha256"],
        },
        "shards": [
            {
                "landpoint_id": report["landpoint_id"],
                "year": report["year"],
                "spatial_split": report["spatial_split"],
                "temporal_split": report["temporal_split"],
                "parent_shard_sha256": report["parent_shard_sha256"],
                "sidecar": str(
                    (
                        output_root
                        / "tasks"
                        / report["landpoint_id"]
                        / str(report["year"])
                        / report["layouts"][accepted]["path"]
                    ).relative_to(output_root)
                ).replace("\\", "/"),
                "sidecar_sha256": report["layouts"][accepted]["sha256"],
                "day_count": report["day_count"],
            }
            for report in reports
        ],
    }
    sidecar_manifest_path = _atomic_json(
        output_root / "dataset_manifest.json", sidecar_manifest
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "passed",
        "pilot_sha256": pilot.sha256,
        "teacher_git_head": next(iter(heads)),
        "parent_dataset_manifest_sha256": _sha256_file(parent_manifest),
        "parent_dataset_id": parent_raw["dataset_id"],
        "sidecar_dataset_manifest_sha256": _sha256_file(sidecar_manifest_path),
        "entry_count": len(reports),
        "point_days": total_days,
        "layout_bytes": totals,
        "accepted_layout": accepted,
        "projected_full_generation_gib": (
            totals[accepted] / total_days * FULL_PARENT_TRANSITIONS / 1024**3
        ),
        "capture_wall_seconds": sum(
            float(report["timing_seconds"]["teacher_capture"]) for report in reports
        ),
        "full_generation_authorized": False,
        "tasks": reports,
    }
    _atomic_json(output_root / "pilot_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--pilot-plan", type=Path, default=DEFAULT_PILOT_PLAN)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--pilot-plan", type=Path, default=DEFAULT_PILOT_PLAN)
    preflight.add_argument("--task-index", type=int)
    preflight.add_argument("--parent-manifest", type=Path, required=True)
    preflight.add_argument("--teacher-plan", type=Path, required=True)
    preflight.add_argument("--expected-git-head")
    generate = subparsers.add_parser("generate")
    generate.add_argument("--pilot-plan", type=Path, default=DEFAULT_PILOT_PLAN)
    generate.add_argument("--task-index", type=int, required=True)
    generate.add_argument("--parent-manifest", type=Path, required=True)
    generate.add_argument("--teacher-plan", type=Path, required=True)
    generate.add_argument("--output-root", type=Path, required=True)
    diagnose = subparsers.add_parser("diagnose-state")
    diagnose.add_argument("--pilot-plan", type=Path, default=DEFAULT_PILOT_PLAN)
    diagnose.add_argument("--task-index", type=int, required=True)
    diagnose.add_argument("--parent-manifest", type=Path, required=True)
    diagnose.add_argument("--teacher-plan", type=Path, required=True)
    diagnose.add_argument("--output-root", type=Path, required=True)
    diagnose.add_argument("--max-days", type=int, default=1)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--pilot-plan", type=Path, default=DEFAULT_PILOT_PLAN)
    aggregate.add_argument("--parent-manifest", type=Path, required=True)
    aggregate.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pilot = load_pilot_plan(args.pilot_plan)
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "status": "passed",
                    "pilot_sha256": pilot.sha256,
                    "entries": [entry.__dict__ for entry in pilot.entries],
                },
                indent=2,
            )
        )
        return 0
    if args.command == "preflight":
        result = preflight_pilot(
            pilot,
            task_index=args.task_index,
            parent_manifest=args.parent_manifest,
            teacher_plan_path=args.teacher_plan,
            expected_git_head=args.expected_git_head,
        )
    elif args.command in {"generate", "diagnose-state"}:
        result = generate_task(
            pilot,
            task_index=args.task_index,
            parent_manifest=args.parent_manifest,
            teacher_plan_path=args.teacher_plan,
            output_root=args.output_root,
            diagnostic_day_limit=(
                args.max_days if args.command == "diagnose-state" else None
            ),
        )
    else:
        result = aggregate_pilot(
            pilot,
            parent_manifest=args.parent_manifest,
            output_root=args.output_root,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
