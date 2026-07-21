"""Generate restartable landpoint-year Teacher shards in persistent processes.

The input plan freezes spatial and temporal splits before any samples are
written.  A worker owns complete landpoint chains so consecutive years can
reuse both compiled executables and the preceding year-end state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import re
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import jax
import numpy as np

from jax_orchidee.driver import orchestration as teacher
from research.daily_coarse_graining.persistence_baseline import COMMON_OK_LEAK_FIELDS
from research.daily_coarse_graining.replay_ceiling import _load_state_cache
from research.daily_coarse_graining.supervised_learnability_pilot import (
    _capture_days_compiled_blocks,
    _teacher_target_vector,
    build_boundary_vector_spec,
)
from research.daily_coarse_graining.synthetic_operator_cost import (
    FORCING_FIELDS,
    _synthetic_nroot,
)
from research.daily_coarse_graining.teacher_compatibility_gate import _discrete_arrays

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "daily_teacher_generation_plan_v1"
SHARD_SCHEMA_VERSION = "daily_teacher_landpoint_year_v1"
MANIFEST_SCHEMA_VERSION = "daily_teacher_worker_manifest_v1"
DATASET_SCHEMA_VERSION = "daily_teacher_dataset_manifest_v1"
SPLITS = frozenset({"train", "validation", "test"})
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class PlanEntry:
    landpoint_id: str
    year: int
    days: int
    spatial_split: str
    temporal_split: str
    run_def: Path
    reference_run_dir: Path
    state_cache: Path | None

    @property
    def key(self) -> str:
        return f"{self.landpoint_id}:{self.year}"


@dataclass(frozen=True)
class GenerationPlan:
    path: Path
    raw: dict[str, Any]
    plan_sha256: str
    dataset_id: str
    teacher_config: Path
    block_size: int
    output_root: Path
    entries: tuple[PlanEntry, ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


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
        raise RuntimeError(
            "Teacher shard generation requires a clean committed worktree; "
            "uncommitted source cannot be assigned a reproducible git identity"
        )
    return head


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _validate_safe_id(name: str, value: str) -> None:
    if not value or SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must contain only letters, digits, dot, underscore, or hyphen")


def load_plan(path: Path, *, require_inputs: bool = False) -> GenerationPlan:
    path = path.resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"plan schema_version must be {SCHEMA_VERSION!r}")
    dataset_id = str(raw.get("dataset_id", ""))
    _validate_safe_id("dataset_id", dataset_id)
    block_size = int(raw.get("block_size", 28))
    if block_size < 2:
        raise ValueError("block_size must be at least two")
    teacher_config = _resolve(raw["teacher_config"])
    output_root = _resolve(raw.get("output_root", f"outputs/training/{dataset_id}"))
    entries = []
    seen = set()
    spatial_by_landpoint: dict[str, str] = {}
    temporal_by_year: dict[int, str] = {}
    previous_by_landpoint: dict[str, PlanEntry] = {}
    for item in raw.get("entries", []):
        landpoint_id = str(item["landpoint_id"])
        _validate_safe_id("landpoint_id", landpoint_id)
        year = int(item["year"])
        days = int(item.get("days", 366 if _is_leap_year(year) else 365))
        spatial_split = str(item["spatial_split"])
        temporal_split = str(item["temporal_split"])
        if spatial_split not in SPLITS or temporal_split not in SPLITS:
            raise ValueError(f"{landpoint_id}:{year} uses an unknown split")
        if days < 1 or days > (366 if _is_leap_year(year) else 365):
            raise ValueError(f"{landpoint_id}:{year} has invalid days={days}")
        key = (landpoint_id, year)
        if key in seen:
            raise ValueError(f"duplicate plan entry {landpoint_id}:{year}")
        seen.add(key)
        if landpoint_id in spatial_by_landpoint and spatial_by_landpoint[landpoint_id] != spatial_split:
            raise ValueError(f"landpoint {landpoint_id} leaks across spatial splits")
        if year in temporal_by_year and temporal_by_year[year] != temporal_split:
            raise ValueError(f"year {year} leaks across temporal splits")
        spatial_by_landpoint[landpoint_id] = spatial_split
        temporal_by_year[year] = temporal_split
        state_cache = item.get("state_cache")
        entry = PlanEntry(
            landpoint_id=landpoint_id,
            year=year,
            days=days,
            spatial_split=spatial_split,
            temporal_split=temporal_split,
            run_def=_resolve(item["run_def"]),
            reference_run_dir=_resolve(item["reference_run_dir"]),
            state_cache=None if state_cache is None else _resolve(state_cache),
        )
        previous = previous_by_landpoint.get(landpoint_id)
        if previous is None and entry.state_cache is None:
            raise ValueError(f"first entry for {landpoint_id} requires state_cache")
        if previous is not None and entry.year <= previous.year:
            raise ValueError(f"years for {landpoint_id} must be strictly increasing")
        if previous is not None and entry.state_cache is None and entry.year != previous.year + 1:
            raise ValueError(
                f"{entry.key} requires state_cache because it does not follow {previous.key}"
            )
        previous_by_landpoint[landpoint_id] = entry
        entries.append(entry)
    if not entries:
        raise ValueError("plan entries must not be empty")
    if require_inputs:
        required = [teacher_config]
        for entry in entries:
            required.extend((entry.run_def, entry.reference_run_dir))
            if entry.state_cache is not None:
                required.append(entry.state_cache)
        missing = [str(value) for value in required if not value.exists()]
        if missing:
            raise FileNotFoundError("missing generation inputs: " + ", ".join(missing[:8]))
    return GenerationPlan(
        path=path,
        raw=raw,
        plan_sha256=_sha256_bytes(_canonical_json(raw)),
        dataset_id=dataset_id,
        teacher_config=teacher_config,
        block_size=block_size,
        output_root=output_root,
        entries=tuple(entries),
    )


def _is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def worker_for_landpoint(landpoint_id: str, worker_count: int) -> int:
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    digest = hashlib.sha256(landpoint_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % worker_count


def assigned_entries(plan: GenerationPlan, worker_index: int, worker_count: int) -> tuple[PlanEntry, ...]:
    if worker_index < 0 or worker_index >= worker_count:
        raise ValueError("worker_index must satisfy 0 <= worker_index < worker_count")
    return tuple(
        sorted(
            (
                entry
                for entry in plan.entries
                if worker_for_landpoint(entry.landpoint_id, worker_count) == worker_index
            ),
            key=lambda entry: (entry.landpoint_id, entry.year),
        )
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _worker_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        owner = path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"worker lock already exists at {path}: {owner}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "created_unix": time.time(),
                },
                handle,
            )
        yield
    finally:
        path.unlink(missing_ok=True)


def _leaf_metadata(spec) -> list[dict[str, Any]]:
    return [
        {
            "family": leaf.family,
            "component": leaf.component,
            "name": leaf.name,
            "shape": list(leaf.shape),
            "start": leaf.start,
            "stop": leaf.stop,
        }
        for leaf in spec.leaves
    ]


def _state_input_numpy(state, spec) -> np.ndarray:
    values = []
    for leaf in spec.leaves:
        if leaf.component is None:
            continue
        fields = state.fields_by_component[leaf.component]
        if leaf.name in fields:
            value = fields[leaf.name]
        elif leaf.component == "hydrol_previous_step_state" and leaf.name == "nroot":
            value = _synthetic_nroot(state, np.asarray(0.0))
        else:
            raise ValueError(f"state input is missing {leaf.component}.{leaf.name}")
        values.append(np.asarray(value, dtype=np.float64).reshape(-1))
    slow = state.fields_by_component["slowproc_stomate_previous_step_state"]
    for name in (
        "biomass",
        "npp_daily",
        "resp_maint",
        "resp_growth",
        *COMMON_OK_LEAK_FIELDS,
        "deepC_peat",
    ):
        values.append(np.asarray(slow[name], dtype=np.float64).reshape(-1))
    return np.concatenate(values)


def _forcing_numpy(forcing) -> np.ndarray:
    columns = []
    for name in FORCING_FIELDS:
        value = np.asarray(getattr(forcing, name), dtype=np.float64)
        columns.append(value.reshape((value.shape[0], -1)))
    return np.concatenate(columns, axis=1)


def _parameter_numpy(context) -> np.ndarray:
    values = teacher._compiled_stomate_parameter_values(context)
    return np.concatenate(
        tuple(np.asarray(value, dtype=np.float64).reshape(-1) for value in values)
    )


def _landpoint_physics_numpy(context) -> np.ndarray:
    values = (
        context.hydrol_humcste,
        context.hydrol_throughfall_by_pft,
        context.hydrol_cwrr_ks,
        context.hydrol_zz_mm,
        context.hydrol_dz_mm,
        context.hydrol_reinf_slope,
        context.diaglev,
        context.ext_coeff_vegetfrac,
        context.sechiba_qsint,
        context.run_scalars.pref_soil_veg,
    )
    return np.concatenate(
        tuple(np.asarray(value, dtype=np.float64).reshape(-1) for value in values)
    )


def _calendar_numpy(year: int, day_indices: np.ndarray) -> np.ndarray:
    days = 366.0 if _is_leap_year(year) else 365.0
    angle = 2.0 * np.pi * (day_indices.astype(np.float64) - 1.0) / days
    return np.column_stack(
        (
            np.sin(angle),
            np.cos(angle),
            day_indices.astype(np.float64) / days,
            np.full(day_indices.shape, days / 366.0),
        )
    )


def build_shard_arrays(states, forcings, records, context) -> tuple[dict[str, np.ndarray], Any]:
    if not records:
        raise ValueError("cannot build an empty Teacher shard")
    spec = build_boundary_vector_spec(records[0])
    target = np.stack([_teacher_target_vector(record, spec) for record in records])
    state_input = np.stack([_state_input_numpy(state, spec) for state in states])
    forcing = np.stack([_forcing_numpy(value) for value in forcings])
    day_index = np.asarray([record.day_index for record in records], dtype=np.int32)
    arrays = {
        "day_index": day_index,
        "day_start_state": state_input,
        "day_start_state_finite": np.isfinite(state_input),
        "forcing_48": forcing,
        "forcing_48_finite": np.isfinite(forcing),
        "parameters": _parameter_numpy(context),
        "landpoint_physics": _landpoint_physics_numpy(context),
        "calendar": _calendar_numpy(int(records[0].year), day_index),
        "teacher_target": target,
        "teacher_target_finite": np.isfinite(target),
        **_discrete_arrays(records),
    }
    return arrays, spec


def _array_metadata(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in arrays.items()
    }


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _entry_paths(worker_root: Path, entry: PlanEntry) -> tuple[Path, Path, Path]:
    stem = f"{entry.landpoint_id}_{entry.year}"
    split = f"spatial-{entry.spatial_split}_temporal-{entry.temporal_split}"
    shard_dir = worker_root / "shards" / split
    return (
        shard_dir / f"{stem}.npz",
        shard_dir / f"{stem}.json",
        worker_root / "checkpoints" / entry.landpoint_id / f"{entry.year}.pkl",
    )


def _completed_metadata(
    plan: GenerationPlan,
    worker_root: Path,
    entry: PlanEntry,
    *,
    plan_sha256: str,
    git_head: str,
    preceding_checkpoint_sha256: str | None,
) -> dict[str, Any] | None:
    shard, metadata, checkpoint = _entry_paths(worker_root, entry)
    if not (shard.exists() and metadata.exists() and checkpoint.exists()):
        return None
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    expected = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "plan_sha256": plan_sha256,
        "teacher_git_head": git_head,
        "landpoint_id": entry.landpoint_id,
        "year": entry.year,
        "days": entry.days,
        "preceding_checkpoint_sha256": preceding_checkpoint_sha256,
        "input_hashes": _input_hashes(plan, entry),
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        return None
    if payload.get("shard_sha256") != _sha256_file(shard):
        return None
    if payload.get("checkpoint_sha256") != _sha256_file(checkpoint):
        return None
    return payload


def _initial_state(entry: PlanEntry, chained_state):
    if entry.state_cache is not None:
        return _load_state_cache(entry.state_cache)["state"]
    if chained_state is None:
        raise RuntimeError(f"{entry.key} has no preceding in-process state")
    return chained_state


def _input_hashes(plan: GenerationPlan, entry: PlanEntry) -> dict[str, Any]:
    restart_names = (
        "driver_start.nc",
        "sechiba_start.nc",
        "stomate_start.nc",
        "stomate_restart.nc",
    )
    return {
        "teacher_config": _sha256_file(plan.teacher_config),
        "run_def": _sha256_file(entry.run_def),
        "state_cache": None if entry.state_cache is None else _sha256_file(entry.state_cache),
        "reference_restart": {
            name: _sha256_file(entry.reference_run_dir / name)
            for name in restart_names
            if (entry.reference_run_dir / name).exists()
        },
    }


def _write_entry(
    plan: GenerationPlan,
    entry: PlanEntry,
    worker_root: Path,
    *,
    git_head: str,
    previous_state,
    preceding_checkpoint_sha256: str | None,
) -> tuple[dict[str, Any], Any]:
    context = teacher.prepare_paper_1961_driver_context(
        plan.teacher_config,
        used_run_def_path=entry.run_def,
        reference_run_dir=entry.reference_run_dir,
    )
    initial_state = teacher.rebase_driver_state_for_year_start(
        _initial_state(entry, previous_state)
    )
    started = time.perf_counter()
    states, forcings, records, final_state = _capture_days_compiled_blocks(
        config_path=plan.teacher_config,
        context=context,
        previous_state=initial_state,
        year=entry.year,
        start_day=1,
        days=entry.days,
        block_size=plan.block_size,
    )
    capture_seconds = time.perf_counter() - started
    arrays, spec = build_shard_arrays(states, forcings, records, context)
    shard, metadata_path, checkpoint = _entry_paths(worker_root, entry)
    _atomic_npz(shard, arrays)
    checkpoint_payload = {
        "schema_version": "daily_teacher_year_end_checkpoint_v1",
        "teacher_git_head": git_head,
        "plan_sha256": plan.plan_sha256,
        "landpoint_id": entry.landpoint_id,
        "end_year": entry.year,
        "state": final_state,
    }
    _atomic_pickle(checkpoint, checkpoint_payload)
    metadata = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "status": "complete",
        "provisional_teacher": True,
        "dataset_id": plan.dataset_id,
        "plan_sha256": plan.plan_sha256,
        "teacher_git_head": git_head,
        "landpoint_id": entry.landpoint_id,
        "year": entry.year,
        "days": entry.days,
        "spatial_split": entry.spatial_split,
        "temporal_split": entry.temporal_split,
        "block_size": plan.block_size,
        "capture_seconds": capture_seconds,
        "preceding_checkpoint_sha256": preceding_checkpoint_sha256,
        "input_hashes": _input_hashes(plan, entry),
        "boundary_spec": _leaf_metadata(spec),
        "boundary_spec_sha256": _sha256_bytes(_canonical_json(_leaf_metadata(spec))),
        "arrays": _array_metadata(arrays),
        "shard": _relative(shard, worker_root),
        "shard_sha256": _sha256_file(shard),
        "checkpoint": _relative(checkpoint, worker_root),
        "checkpoint_sha256": _sha256_file(checkpoint),
    }
    _atomic_json(metadata_path, metadata)
    return metadata, final_state


def _load_checkpoint(worker_root: Path, metadata: dict[str, Any]):
    path = worker_root / metadata["checkpoint"]
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("teacher_git_head") != metadata["teacher_git_head"]:
        raise ValueError(f"checkpoint provenance mismatch at {path}")
    return payload["state"]


def generate_worker(
    plan: GenerationPlan,
    *,
    output_root: Path,
    worker_index: int,
    worker_count: int,
    max_entries: int | None = None,
) -> dict[str, Any]:
    git_head = _clean_git_head()
    worker_root = output_root / "workers" / f"worker-{worker_index:03d}-of-{worker_count:03d}"
    selected = assigned_entries(plan, worker_index, worker_count)
    if max_entries is not None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        selected = selected[:max_entries]
    manifest_path = worker_root / "manifest.json"
    completed = []
    chained_state = None
    chained_landpoint = None
    chained_checkpoint_sha256 = None
    with _worker_lock(worker_root / "generation.lock"):
        for entry in selected:
            if entry.landpoint_id != chained_landpoint:
                chained_state = None
                chained_landpoint = entry.landpoint_id
                chained_checkpoint_sha256 = None
            preceding_checkpoint_sha256 = (
                None if entry.state_cache is not None else chained_checkpoint_sha256
            )
            existing = _completed_metadata(
                plan,
                worker_root,
                entry,
                plan_sha256=plan.plan_sha256,
                git_head=git_head,
                preceding_checkpoint_sha256=preceding_checkpoint_sha256,
            )
            if existing is not None:
                chained_state = _load_checkpoint(worker_root, existing)
                chained_checkpoint_sha256 = existing["checkpoint_sha256"]
                completed.append(existing)
                continue
            metadata, chained_state = _write_entry(
                plan,
                entry,
                worker_root,
                git_head=git_head,
                previous_state=chained_state,
                preceding_checkpoint_sha256=preceding_checkpoint_sha256,
            )
            chained_checkpoint_sha256 = metadata["checkpoint_sha256"]
            completed.append(metadata)
            _atomic_json(
                manifest_path,
                _worker_manifest(
                    plan,
                    worker_index=worker_index,
                    worker_count=worker_count,
                    assigned=selected,
                    completed=completed,
                    git_head=git_head,
                ),
            )
    manifest = _worker_manifest(
        plan,
        worker_index=worker_index,
        worker_count=worker_count,
        assigned=selected,
        completed=completed,
        git_head=git_head,
    )
    _atomic_json(manifest_path, manifest)
    return manifest


def _worker_manifest(
    plan: GenerationPlan,
    *,
    worker_index: int,
    worker_count: int,
    assigned: Sequence[PlanEntry],
    completed: Sequence[dict[str, Any]],
    git_head: str,
) -> dict[str, Any]:
    completed_keys = {f"{item['landpoint_id']}:{item['year']}" for item in completed}
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_id": plan.dataset_id,
        "plan_sha256": plan.plan_sha256,
        "teacher_git_head": git_head,
        "worker_index": worker_index,
        "worker_count": worker_count,
        "assigned_entries": [entry.key for entry in assigned],
        "completed_entries": sorted(completed_keys),
        "complete": completed_keys == {entry.key for entry in assigned},
        "shards": list(completed),
        "runtime": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
    }


def aggregate_workers(
    plan: GenerationPlan,
    *,
    output_root: Path,
    worker_count: int,
) -> dict[str, Any]:
    expected = {entry.key for entry in plan.entries}
    observed: dict[str, dict[str, Any]] = {}
    heads = set()
    for index in range(worker_count):
        worker_root = output_root / "workers" / f"worker-{index:03d}-of-{worker_count:03d}"
        manifest_path = worker_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing worker manifest {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"invalid worker manifest schema at {manifest_path}")
        if manifest.get("plan_sha256") != plan.plan_sha256:
            raise ValueError(f"plan drift in {manifest_path}")
        if manifest.get("worker_index") != index or manifest.get("worker_count") != worker_count:
            raise ValueError(f"worker identity mismatch in {manifest_path}")
        heads.add(manifest["teacher_git_head"])
        for shard in manifest["shards"]:
            key = f"{shard['landpoint_id']}:{shard['year']}"
            if key in observed:
                raise ValueError(f"duplicate completed shard {key}")
            shard_path = worker_root / shard["shard"]
            checkpoint_path = worker_root / shard["checkpoint"]
            if _sha256_file(shard_path) != shard["shard_sha256"]:
                raise ValueError(f"shard hash mismatch for {key}")
            if _sha256_file(checkpoint_path) != shard["checkpoint_sha256"]:
                raise ValueError(f"checkpoint hash mismatch for {key}")
            observed[key] = {
                **shard,
                "shard": _relative(shard_path, output_root),
                "checkpoint": _relative(checkpoint_path, output_root),
            }
    if len(heads) != 1:
        raise ValueError("workers used different Teacher git commits")
    missing = sorted(expected - observed.keys())
    unexpected = sorted(observed.keys() - expected)
    if missing or unexpected:
        raise ValueError(f"dataset is incomplete: missing={missing[:8]}, unexpected={unexpected[:8]}")
    preceding_by_landpoint: dict[str, str] = {}
    for entry in sorted(plan.entries, key=lambda value: (value.landpoint_id, value.year)):
        item = observed[entry.key]
        expected_preceding = (
            None
            if entry.state_cache is not None
            else preceding_by_landpoint.get(entry.landpoint_id)
        )
        if item.get("preceding_checkpoint_sha256") != expected_preceding:
            raise ValueError(f"checkpoint chain mismatch for {entry.key}")
        if item.get("input_hashes") != _input_hashes(plan, entry):
            raise ValueError(f"generation input drift for {entry.key}")
        preceding_by_landpoint[entry.landpoint_id] = item["checkpoint_sha256"]
    boundary_hashes = {item["boundary_spec_sha256"] for item in observed.values()}
    if len(boundary_hashes) != 1:
        raise ValueError("Teacher boundary schemas differ across shards")
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_id": plan.dataset_id,
        "status": "complete",
        "provisional_teacher": True,
        "plan": str(plan.path),
        "plan_sha256": plan.plan_sha256,
        "teacher_git_head": next(iter(heads)),
        "worker_count": worker_count,
        "landpoint_count": len({entry.landpoint_id for entry in plan.entries}),
        "year_count": len({entry.year for entry in plan.entries}),
        "shard_count": len(observed),
        "boundary_spec_sha256": next(iter(boundary_hashes)),
        "shards": [observed[key] for key in sorted(observed)],
    }
    _atomic_json(output_root / "dataset_manifest.json", manifest)
    return manifest


def _plan_summary(plan: GenerationPlan, worker_count: int) -> dict[str, Any]:
    return {
        "dataset_id": plan.dataset_id,
        "plan_sha256": plan.plan_sha256,
        "landpoints": len({entry.landpoint_id for entry in plan.entries}),
        "years": len({entry.year for entry in plan.entries}),
        "entries": len(plan.entries),
        "block_size": plan.block_size,
        "worker_loads": [
            len(assigned_entries(plan, index, worker_count)) for index in range(worker_count)
        ],
        "output_root": str(plan.output_root),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--plan", type=Path, required=True)
    validate.add_argument("--worker-count", type=int, default=1)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--plan", type=Path, required=True)
    generate.add_argument("--output-root", type=Path)
    generate.add_argument("--worker-index", type=int, required=True)
    generate.add_argument("--worker-count", type=int, required=True)
    generate.add_argument("--max-entries", type=int)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--plan", type=Path, required=True)
    aggregate.add_argument("--output-root", type=Path)
    aggregate.add_argument("--worker-count", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = load_plan(args.plan, require_inputs=args.command == "generate")
    if args.command == "validate":
        print(json.dumps(_plan_summary(plan, args.worker_count), indent=2))
        return 0
    output_root = plan.output_root if args.output_root is None else args.output_root.resolve()
    if args.command == "generate":
        result = generate_worker(
            plan,
            output_root=output_root,
            worker_index=args.worker_index,
            worker_count=args.worker_count,
            max_entries=args.max_entries,
        )
        print(
            json.dumps(
                {
                    "complete": result["complete"],
                    "completed_entries": len(result["completed_entries"]),
                    "manifest": str(
                        output_root
                        / "workers"
                        / f"worker-{args.worker_index:03d}-of-{args.worker_count:03d}"
                        / "manifest.json"
                    ),
                },
                indent=2,
            )
        )
        return 0 if result["complete"] else 2
    result = aggregate_workers(plan, output_root=output_root, worker_count=args.worker_count)
    print(
        json.dumps(
            {
                "status": result["status"],
                "shard_count": result["shard_count"],
                "manifest": str(output_root / "dataset_manifest.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
