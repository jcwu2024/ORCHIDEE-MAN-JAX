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
from jax_orchidee.driver.init import parse_run_def
from jax_orchidee.driver.paper_binding import expected_paper_domain_limits
from research.daily_coarse_graining.daily_markov_contract import (
    SHARD_SCHEMA_VERSION,
    ConditionLeafSpec,
    assert_markov_continuity,
    build_daily_markov_contract,
    build_state_trajectory,
    estimated_uncompressed_bytes,
    extract_diagnostics,
    extract_fast_day_target,
    extract_state,
    native_forcing_days,
)
from research.daily_coarse_graining.replay_ceiling import _load_state_cache
from research.daily_coarse_graining.supervised_learnability_pilot import (
    _iter_capture_days_compiled_blocks,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "daily_teacher_generation_plan_v2"
MANIFEST_SCHEMA_VERSION = "daily_teacher_worker_manifest_v2"
DATASET_SCHEMA_VERSION = "daily_teacher_dataset_manifest_v3"
WORKER_ASSIGNMENT_STRATEGY = "balanced_landpoint_chains_v1"
SPLITS = frozenset({"train", "validation", "test"})
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
YEAR_START_CHECKPOINT = "year_start_checkpoint"
COLD_START_BOOTSTRAP = "cold_start_bootstrap"
INITIALIZATION_MODES = frozenset({YEAR_START_CHECKPOINT, COLD_START_BOOTSTRAP})
PARAMETER_ORDER = ("alloc_min", "residence_time", "vcmax25", "maint_resp_slope")
LANDPOINT_STATIC_ORDER = (
    "hydrol_humcste",
    "hydrol_throughfall_by_pft",
    "hydrol_cwrr_ks",
    "hydrol_zz_mm",
    "hydrol_dz_mm",
    "hydrol_reinf_slope",
    "diaglev",
    "ext_coeff_vegetfrac",
    "sechiba_qsint",
    "pref_soil_veg",
    "lalo",
    "lon",
    "lat",
    "areas",
    "contfrac",
    "resolution",
    "corners",
    "seglength",
    "measurement_heights",
    "soiltile",
    "njsc",
    "clay_frac",
    "sand_frac",
    "silt_frac",
    "bulk_dens",
    "soil_ph",
    "poor_soils",
    "soilclass",
    "salinity",
    "tide_height",
)
ANNUAL_CONDITION_ORDER = ("annual_co2_ppm",)
REFERENCE_INPUT_NAMES = (
    "driver_start.nc",
    "sechiba_start.nc",
    "stomate_start.nc",
    "stomate_restart.nc",
    "stomate_history_1961.nc",
)

_PARAMETER_SOURCE = (
    "jax_orchidee.driver.orchestration._compiled_stomate_parameter_values; "
    "run.def-controlled PFT vectors"
)
_LANDPOINT_STATIC_SOURCE = (
    "prepared paper driver context and first-step static input interpolation"
)
_ANNUAL_CONDITION_SOURCE = "drivers.co2 annual lookup in the case configuration"


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
    initialization_mode: str
    acceptance_checkpoint: Path | None

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


def _process_memory_bytes(
    status_path: Path = Path("/proc/self/status"),
) -> dict[str, int | None]:
    """Return current and peak resident memory without adding a dependency."""

    if status_path.exists():
        values: dict[str, int] = {}
        for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
            name, separator, raw_value = line.partition(":")
            if separator and name in {"VmRSS", "VmHWM"}:
                fields = raw_value.split()
                if fields:
                    values[name] = int(fields[0]) * 1024
        return {
            "current_rss_bytes": values.get("VmRSS"),
            "peak_rss_bytes": values.get("VmHWM"),
        }

    try:
        import resource
    except ImportError:
        peak_rss = None
    else:
        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if platform.system() != "Darwin":
            peak_rss *= 1024
    return {"current_rss_bytes": None, "peak_rss_bytes": peak_rss}


def _compiled_cache_entries() -> dict[str, int]:
    return {
        "later_day_block": len(teacher._COMPILED_LATER_DAY_BLOCK_CACHE),
        "sechiba_scan": len(teacher._COMPILED_SECHIBA_SCAN_CACHE),
    }


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


def _validate_landpoint_run_def_binding(entry: PlanEntry) -> None:
    values = parse_run_def(entry.run_def)
    expected = expected_paper_domain_limits(entry.landpoint_id)
    observed: dict[str, float | None] = {}
    for name in expected:
        raw = values.get(name)
        try:
            observed[name] = None if raw is None else float(raw)
        except ValueError as exc:
            raise ValueError(
                f"{entry.key} run_def has invalid {name}={raw!r}"
            ) from exc
    mismatches = {
        name: {"expected": target, "observed": observed[name]}
        for name, target in expected.items()
        if observed[name] != target
    }
    if mismatches:
        raise ValueError(
            f"{entry.key} run_def domain does not match landpoint ID: {mismatches}"
        )


def load_plan(path: Path, *, require_inputs: bool = False) -> GenerationPlan:
    path = path.resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"plan schema_version must be {SCHEMA_VERSION!r}")
    dataset_id = str(raw.get("dataset_id", ""))
    _validate_safe_id("dataset_id", dataset_id)
    block_size = int(raw.get("block_size", 7))
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
        initialization_mode = str(item.get("initialization_mode", YEAR_START_CHECKPOINT))
        if initialization_mode not in INITIALIZATION_MODES:
            raise ValueError(
                f"{landpoint_id}:{year} uses unknown initialization_mode "
                f"{initialization_mode!r}"
            )
        acceptance_checkpoint = item.get("acceptance_checkpoint")
        entry = PlanEntry(
            landpoint_id=landpoint_id,
            year=year,
            days=days,
            spatial_split=spatial_split,
            temporal_split=temporal_split,
            run_def=_resolve(item["run_def"]),
            reference_run_dir=_resolve(item["reference_run_dir"]),
            state_cache=None if state_cache is None else _resolve(state_cache),
            initialization_mode=initialization_mode,
            acceptance_checkpoint=(
                None
                if acceptance_checkpoint is None
                else _resolve(acceptance_checkpoint)
            ),
        )
        previous = previous_by_landpoint.get(landpoint_id)
        if entry.initialization_mode == COLD_START_BOOTSTRAP:
            if previous is not None:
                raise ValueError(f"{entry.key} cold_start_bootstrap must begin a landpoint chain")
            if entry.year != 1961:
                raise ValueError(f"{entry.key} cold_start_bootstrap is supported only for 1961")
            if entry.state_cache is not None:
                raise ValueError(f"{entry.key} cold_start_bootstrap must not use state_cache")
            if entry.days < 2:
                raise ValueError(f"{entry.key} cold_start_bootstrap requires at least two days")
        elif previous is None and entry.state_cache is None:
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
            required.append(entry.run_def)
            required.extend(entry.reference_run_dir / name for name in REFERENCE_INPUT_NAMES)
            if entry.state_cache is not None:
                required.append(entry.state_cache)
            if entry.acceptance_checkpoint is not None:
                required.append(entry.acceptance_checkpoint)
        missing = [str(value) for value in required if not value.exists()]
        if missing:
            raise FileNotFoundError("missing generation inputs: " + ", ".join(missing[:8]))
        for entry in entries:
            _validate_landpoint_run_def_binding(entry)
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


def worker_assignment(plan: GenerationPlan, worker_count: int) -> dict[str, int]:
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    entries_by_landpoint: dict[str, int] = {}
    for entry in plan.entries:
        entries_by_landpoint[entry.landpoint_id] = (
            entries_by_landpoint.get(entry.landpoint_id, 0) + 1
        )
    loads = [0] * worker_count
    assignment = {}
    for landpoint_id, entry_count in sorted(
        entries_by_landpoint.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        worker_index = min(range(worker_count), key=lambda index: (loads[index], index))
        assignment[landpoint_id] = worker_index
        loads[worker_index] += entry_count
    return assignment


def assigned_entries(plan: GenerationPlan, worker_index: int, worker_count: int) -> tuple[PlanEntry, ...]:
    if worker_index < 0 or worker_index >= worker_count:
        raise ValueError("worker_index must satisfy 0 <= worker_index < worker_count")
    assignment = worker_assignment(plan, worker_count)
    return tuple(
        sorted(
            (
                entry
                for entry in plan.entries
                if assignment[entry.landpoint_id] == worker_index
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
            np.savez_compressed(handle, **arrays)
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
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "created_unix": time.time(),
                },
                handle,
            )
        yield
    finally:
        path.unlink(missing_ok=True)


def recover_stale_worker_lock(
    plan: GenerationPlan,
    *,
    output_root: Path,
    worker_index: int,
    worker_count: int,
    expected_host: str,
    expected_pid: int,
    expected_slurm_job_id: str | None = None,
) -> dict[str, Any]:
    assigned_entries(plan, worker_index, worker_count)
    worker_root = output_root / "workers" / f"worker-{worker_index:03d}-of-{worker_count:03d}"
    lock_path = worker_root / "generation.lock"
    if not lock_path.is_file():
        raise FileNotFoundError(f"worker lock does not exist: {lock_path}")
    owner = json.loads(lock_path.read_text(encoding="utf-8"))
    expected = {"host": expected_host, "pid": expected_pid}
    if any(owner.get(name) != value for name, value in expected.items()):
        raise ValueError(f"worker lock owner mismatch: expected={expected}, observed={owner}")
    if expected_slurm_job_id is not None and owner.get("slurm_job_id") != expected_slurm_job_id:
        raise ValueError(
            "worker lock Slurm identity mismatch: "
            f"expected={expected_slurm_job_id!r}, observed={owner.get('slurm_job_id')!r}"
        )
    recovered = lock_path.with_name(
        f"generation.lock.stale-{int(time.time())}-{expected_host}-{expected_pid}"
    )
    lock_path.replace(recovered)
    return {
        "status": "recovered",
        "worker_index": worker_index,
        "worker_count": worker_count,
        "owner": owner,
        "preserved_lock": str(recovered),
    }


def _pack_condition_groups(
    groups: Sequence[tuple[str, object]],
    *,
    temporal_role: str,
    source: str,
) -> tuple[np.ndarray, tuple[ConditionLeafSpec, ...]]:
    cursor = 0
    arrays = []
    leaves = []
    for name, value in groups:
        array = np.asarray(value, dtype=np.float64)
        stop = cursor + int(array.size)
        arrays.append(array.reshape(-1))
        leaves.append(
            ConditionLeafSpec(
                name=name,
                shape=tuple(array.shape),
                dtype=str(array.dtype),
                start=cursor,
                stop=stop,
                temporal_role=temporal_role,
                source=source,
            )
        )
        cursor = stop
    return (
        np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float64),
        tuple(leaves),
    )


def _parameter_groups(context) -> tuple[tuple[str, object], ...]:
    values = teacher._compiled_stomate_parameter_values(context)
    return tuple((name, getattr(values, name)) for name in PARAMETER_ORDER)


def _landpoint_physics_groups(context) -> tuple[tuple[str, object], ...]:
    return (
        ("hydrol_humcste", context.hydrol_humcste),
        ("hydrol_throughfall_by_pft", context.hydrol_throughfall_by_pft),
        ("hydrol_cwrr_ks", context.hydrol_cwrr_ks),
        ("hydrol_zz_mm", context.hydrol_zz_mm),
        ("hydrol_dz_mm", context.hydrol_dz_mm),
        ("hydrol_reinf_slope", context.hydrol_reinf_slope),
        ("diaglev", context.diaglev),
        ("ext_coeff_vegetfrac", context.ext_coeff_vegetfrac),
        ("sechiba_qsint", context.sechiba_qsint),
        ("pref_soil_veg", context.run_scalars.pref_soil_veg),
    )


def _landpoint_static_groups(context) -> tuple[tuple[str, object], ...]:
    first = context.first_step_bundle
    if first is None:
        raise ValueError("landpoint static extraction requires a first-step bundle")
    domain = first.domain
    static = first.static_trace_fields
    groups = (
        *_landpoint_physics_groups(context),
        ("lalo", domain.lalo),
        ("lon", domain.lon),
        ("lat", domain.lat),
        ("areas", first.forcing.Areas),
        ("contfrac", first.forcing.contfrac),
        ("resolution", domain.resolution),
        ("corners", domain.corners),
        ("seglength", domain.seglength),
        (
            "measurement_heights",
            np.asarray([first.forcing.Height_Lev1, first.forcing.Height_Levuv]),
        ),
        ("soiltile", first.vegetation.soiltile),
        ("njsc", static.njsc),
        ("clay_frac", static.clay_frac),
        ("sand_frac", static.sand_frac),
        ("silt_frac", static.silt_frac),
        ("bulk_dens", static.bulk_dens),
        ("soil_ph", static.soil_ph),
        ("poor_soils", static.poor_soils),
        ("soilclass", static.soilclass),
        ("salinity", static.salinity),
        ("tide_height", static.tide_height),
    )
    missing = tuple(name for name, value in groups if value is None)
    if missing:
        raise ValueError(f"landpoint neural conditions are missing: {missing}")
    observed_order = tuple(name for name, _value in groups)
    if observed_order != LANDPOINT_STATIC_ORDER:
        raise AssertionError("landpoint static condition order drift")
    return groups


def _annual_condition_groups(context, *, year: int) -> tuple[tuple[str, object], ...]:
    groups = (
        ("annual_co2_ppm", np.asarray([teacher.read_annual_co2(context.config_path, int(year))])),
    )
    if tuple(name for name, _value in groups) != ANNUAL_CONDITION_ORDER:
        raise AssertionError("annual condition order drift")
    return groups


def build_shard_arrays(
    states, forcings, records, context, *, final_state=None
) -> tuple[dict[str, np.ndarray], Any]:
    if not records:
        raise ValueError("cannot build an empty Teacher shard")
    if not (len(states) == len(forcings) == len(records)):
        raise ValueError("Teacher state, forcing, and record counts must match")
    if final_state is None:
        final_state = records[-1].expected_result.day_end_state
    day_index = np.asarray([record.day_index for record in records], dtype=np.int32)
    year = int(records[0].year)
    parameters, parameter_leaves = _pack_condition_groups(
        _parameter_groups(context),
        temporal_role="landpoint_parameter",
        source=_PARAMETER_SOURCE,
    )
    landpoint_static, landpoint_static_leaves = _pack_condition_groups(
        _landpoint_static_groups(context),
        temporal_role="landpoint_static",
        source=_LANDPOINT_STATIC_SOURCE,
    )
    annual_condition, annual_condition_leaves = _pack_condition_groups(
        _annual_condition_groups(context, year=year),
        temporal_role="annual_exogenous",
        source=_ANNUAL_CONDITION_SOURCE,
    )
    forcing_native, forcing_record_indices, native_spec = native_forcing_days(
        context, year=year, day_indices=day_index
    )
    contract = build_daily_markov_contract(
        states[0],
        records[0],
        parameter_leaves=parameter_leaves,
        landpoint_static_leaves=landpoint_static_leaves,
        annual_condition_leaves=annual_condition_leaves,
        native_forcing_spec=native_spec,
    )
    assert_markov_continuity(states, records, final_state, contract)
    state_trajectory, discrete = build_state_trajectory(
        [*states, final_state], contract
    )
    diagnostics = np.stack(
        [extract_diagnostics(record, contract) for record in records]
    )
    fast_day_target = np.stack(
        [
            extract_fast_day_target(record, contract.fast_day_target_leaves)
            for record in records
        ]
    )
    arrays = {
        "day_index": day_index,
        "state_trajectory": state_trajectory,
        "fast_day_target": fast_day_target,
        "forcing_native": forcing_native,
        "forcing_record_indices": forcing_record_indices,
        "parameters": parameters,
        "landpoint_static": landpoint_static,
        "annual_conditions": annual_condition,
        "diagnostics": diagnostics,
        "year": np.asarray(year, dtype=np.int32),
        **{f"state_discrete__{name}": value for name, value in discrete.items()},
    }
    return arrays, contract


def build_shard_arrays_from_blocks(blocks, context):
    """Consume bounded capture blocks into compact annual Markov arrays."""

    parameters, parameter_leaves = _pack_condition_groups(
        _parameter_groups(context),
        temporal_role="landpoint_parameter",
        source=_PARAMETER_SOURCE,
    )
    landpoint_static, landpoint_static_leaves = _pack_condition_groups(
        _landpoint_static_groups(context),
        temporal_role="landpoint_static",
        source=_LANDPOINT_STATIC_SOURCE,
    )
    continuous_rows = []
    discrete_rows: dict[str, list[np.ndarray]] = {}
    diagnostics = []
    fast_day_targets = []
    day_indices = []
    contract = None
    year = None
    previous_final = None
    native_spec = None
    final_state = None

    for states, forcings, records, block_final in blocks:
        if not records or not (len(states) == len(forcings) == len(records)):
            raise ValueError("capture block state, forcing, and record counts must match")
        block_years = {int(record.year) for record in records}
        if len(block_years) != 1:
            raise ValueError("capture block mixes source years")
        block_year = block_years.pop()
        if year is None:
            year = block_year
            _native, _indices, native_spec = native_forcing_days(
                context,
                year=year,
                day_indices=(int(records[0].day_index),),
            )
            annual_conditions, annual_condition_leaves = _pack_condition_groups(
                _annual_condition_groups(context, year=year),
                temporal_role="annual_exogenous",
                source=_ANNUAL_CONDITION_SOURCE,
            )
            contract = build_daily_markov_contract(
                states[0],
                records[0],
                parameter_leaves=parameter_leaves,
                landpoint_static_leaves=landpoint_static_leaves,
                annual_condition_leaves=annual_condition_leaves,
                native_forcing_spec=native_spec,
            )
            discrete_rows = {leaf.key: [] for leaf in contract.discrete_leaves}
        elif block_year != year:
            raise ValueError("capture stream mixes source years")
        assert contract is not None
        if previous_final is not None:
            previous_continuous, previous_discrete = extract_state(
                previous_final, contract
            )
            current_continuous, current_discrete = extract_state(states[0], contract)
            if not np.array_equal(
                previous_continuous, current_continuous, equal_nan=True
            ) or any(
                not np.array_equal(
                    previous_discrete[name], current_discrete[name], equal_nan=True
                )
                for name in previous_discrete
            ):
                raise ValueError("Teacher state continuity failed between capture blocks")
        assert_markov_continuity(states, records, block_final, contract)
        for packet, record in zip(states, records, strict=True):
            day_index = int(record.day_index)
            if day_indices and day_index != day_indices[-1] + 1:
                raise ValueError("capture stream day indices are not consecutive")
            continuous, discrete = extract_state(
                packet,
                contract,
                allow_year_start_missing=not continuous_rows,
            )
            continuous_rows.append(continuous)
            for name, value in discrete.items():
                discrete_rows[name].append(value)
            diagnostics.append(extract_diagnostics(record, contract))
            fast_day_targets.append(
                extract_fast_day_target(record, contract.fast_day_target_leaves)
            )
            day_indices.append(day_index)
        previous_final = block_final
        final_state = block_final

    if contract is None or year is None or final_state is None or native_spec is None:
        raise ValueError("cannot build an empty Teacher shard")
    final_continuous, final_discrete = extract_state(final_state, contract)
    continuous_rows.append(final_continuous)
    for name, value in final_discrete.items():
        discrete_rows[name].append(value)
    forcing_native, forcing_record_indices, observed_native_spec = native_forcing_days(
        context, year=year, day_indices=day_indices
    )
    if observed_native_spec != native_spec:
        raise ValueError("native forcing schema drift within capture stream")
    arrays = {
        "day_index": np.asarray(day_indices, dtype=np.int32),
        "state_trajectory": np.stack(continuous_rows),
        "fast_day_target": np.stack(fast_day_targets),
        "forcing_native": forcing_native,
        "forcing_record_indices": forcing_record_indices,
        "parameters": parameters,
        "landpoint_static": landpoint_static,
        "annual_conditions": annual_conditions,
        "diagnostics": np.stack(diagnostics),
        "year": np.asarray(year, dtype=np.int32),
        **{
            f"state_discrete__{name}": np.stack(values)
            for name, values in discrete_rows.items()
        },
    }
    return arrays, contract, final_state


def _compact_contract_factory(context, *, year: int, first_day_index: int):
    _parameters, parameter_leaves = _pack_condition_groups(
        _parameter_groups(context),
        temporal_role="landpoint_parameter",
        source=_PARAMETER_SOURCE,
    )
    _static, landpoint_static_leaves = _pack_condition_groups(
        _landpoint_static_groups(context),
        temporal_role="landpoint_static",
        source=_LANDPOINT_STATIC_SOURCE,
    )
    _annual, annual_condition_leaves = _pack_condition_groups(
        _annual_condition_groups(context, year=year),
        temporal_role="annual_exogenous",
        source=_ANNUAL_CONDITION_SOURCE,
    )
    _native, _indices, native_spec = native_forcing_days(
        context, year=year, day_indices=(first_day_index,)
    )

    def factory(packet, record):
        return build_daily_markov_contract(
            packet,
            record,
            parameter_leaves=parameter_leaves,
            landpoint_static_leaves=landpoint_static_leaves,
            annual_condition_leaves=annual_condition_leaves,
            native_forcing_spec=native_spec,
        )

    return factory


def build_shard_arrays_from_compact_blocks(blocks, context):
    """Assemble one annual shard from already projected compiled outputs."""

    parameters, _parameter_leaves = _pack_condition_groups(
        _parameter_groups(context),
        temporal_role="landpoint_parameter",
        source=_PARAMETER_SOURCE,
    )
    landpoint_static, _static_leaves = _pack_condition_groups(
        _landpoint_static_groups(context),
        temporal_role="landpoint_static",
        source=_LANDPOINT_STATIC_SOURCE,
    )
    state_rows = []
    fast_day_targets = []
    diagnostics = []
    discrete_rows: dict[str, list[np.ndarray]] = {}
    day_indices = []
    contract = None
    year = None
    final_state = None
    previous_final = None

    for block in blocks:
        if contract is None:
            contract = block.contract
            if contract is None:
                raise ValueError("first compact capture block is missing its contract")
            year = int(block.year)
            discrete_rows = {leaf.key: [] for leaf in contract.discrete_leaves}
        elif int(block.year) != year:
            raise ValueError("compact capture stream mixes source years")
        rows = int(block.day_indices.size)
        if not (
            block.state_rows.shape[0]
            == block.fast_day_targets.shape[0]
            == block.diagnostics.shape[0]
            == rows
        ):
            raise ValueError("compact capture arrays have inconsistent row counts")
        if previous_final is not None:
            expected, expected_discrete = extract_state(previous_final, contract)
            if not np.array_equal(expected, block.state_rows[0], equal_nan=True):
                raise ValueError("compact Teacher state continuity failed between blocks")
            for name, value in expected_discrete.items():
                if not np.array_equal(value, block.discrete_rows[name][0], equal_nan=True):
                    raise ValueError(
                        f"compact Teacher discrete continuity failed for {name}"
                    )
        for offset, day_index in enumerate(block.day_indices):
            day_index = int(day_index)
            if day_indices and day_index != day_indices[-1] + 1:
                raise ValueError("compact capture day indices are not consecutive")
            state_rows.append(np.asarray(block.state_rows[offset]))
            fast_day_targets.append(np.asarray(block.fast_day_targets[offset]))
            diagnostics.append(np.asarray(block.diagnostics[offset]))
            for name in discrete_rows:
                discrete_rows[name].append(np.asarray(block.discrete_rows[name][offset]))
            day_indices.append(day_index)
        previous_final = block.final_state
        final_state = block.final_state

    if contract is None or year is None or final_state is None:
        raise ValueError("cannot build an empty compact Teacher shard")
    final_continuous, final_discrete = extract_state(final_state, contract)
    state_rows.append(final_continuous)
    for name, value in final_discrete.items():
        discrete_rows[name].append(value)
    forcing_native, forcing_record_indices, observed_native_spec = native_forcing_days(
        context, year=year, day_indices=day_indices
    )
    if observed_native_spec != contract.native_forcing:
        raise ValueError("native forcing schema drift within compact capture")
    annual_conditions, _annual_leaves = _pack_condition_groups(
        _annual_condition_groups(context, year=year),
        temporal_role="annual_exogenous",
        source=_ANNUAL_CONDITION_SOURCE,
    )
    arrays = {
        "day_index": np.asarray(day_indices, dtype=np.int32),
        "state_trajectory": np.stack(state_rows),
        "fast_day_target": np.stack(fast_day_targets),
        "forcing_native": forcing_native,
        "forcing_record_indices": forcing_record_indices,
        "parameters": parameters,
        "landpoint_static": landpoint_static,
        "annual_conditions": annual_conditions,
        "diagnostics": np.stack(diagnostics),
        "year": np.asarray(year, dtype=np.int32),
        **{
            f"state_discrete__{name}": np.stack(values)
            for name, values in discrete_rows.items()
        },
    }
    return arrays, contract, final_state


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
        "initialization_mode": entry.initialization_mode,
        "bootstrap_day": 1 if entry.initialization_mode == COLD_START_BOOTSTRAP else None,
        "transition_start_day": 2 if entry.initialization_mode == COLD_START_BOOTSTRAP else 1,
        "transition_count": (
            entry.days - 1
            if entry.initialization_mode == COLD_START_BOOTSTRAP
            else entry.days
        ),
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
        state = _load_state_cache(entry.state_cache)["state"]
    else:
        if chained_state is None:
            raise RuntimeError(f"{entry.key} has no preceding in-process state")
        state = chained_state
    gaps = teacher.driver_year_handoff_state_gaps(state)
    slowproc = state.fields_by_component.get(
        "slowproc_stomate_previous_step_state", {}
    )
    missing_season = tuple(
        name for name in teacher.STOMATE_DAY_SEASON_STATE_FIELDS if name not in slowproc
    )
    if gaps or missing_season:
        missing = [
            f"{gap.component}.{field}"
            for gap in gaps
            for field in gap.fields
        ]
        missing.extend(
            f"slowproc_stomate_previous_step_state.{name}"
            for name in missing_season
        )
        raise ValueError(
            f"{entry.key} year-start checkpoint is incomplete: {missing[:12]}"
        )
    return state


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
        "initialization_mode": entry.initialization_mode,
        "state_cache": None if entry.state_cache is None else _sha256_file(entry.state_cache),
        "acceptance_checkpoint": (
            None
            if entry.acceptance_checkpoint is None
            else _sha256_file(entry.acceptance_checkpoint)
        ),
        "reference_restart": {
            name: _sha256_file(entry.reference_run_dir / name)
            for name in restart_names
            if (entry.reference_run_dir / name).exists()
        },
    }


def _compare_driver_state(
    actual,
    expected,
    *,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
) -> dict[str, Any]:
    """Require exact state schema/discretes and tightly close float64 leaves."""

    if actual.tstep != expected.tstep:
        raise ValueError(
            f"year-end acceptance checkpoint tstep mismatch: {actual.tstep} != {expected.tstep}"
        )
    actual_components = actual.fields_by_component
    expected_components = expected.fields_by_component
    if tuple(actual_components) != tuple(expected_components):
        raise ValueError("year-end acceptance checkpoint component inventory mismatch")
    compared = 0
    exact_mismatch_leaves = 0
    max_abs = 0.0
    max_rel = 0.0

    def compare(left, right, path: str) -> None:
        nonlocal compared, exact_mismatch_leaves, max_abs, max_rel
        if isinstance(right, dict):
            if not isinstance(left, dict) or tuple(left) != tuple(right):
                raise ValueError(f"year-end acceptance checkpoint mapping mismatch at {path}")
            for name in right:
                compare(left[name], right[name], f"{path}.{name}")
            return
        if isinstance(right, (tuple, list)):
            if not isinstance(left, type(right)) or len(left) != len(right):
                raise ValueError(f"year-end acceptance checkpoint sequence mismatch at {path}")
            for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
                compare(left_item, right_item, f"{path}[{index}]")
            return
        try:
            left_array = np.asarray(left)
            right_array = np.asarray(right)
        except (TypeError, ValueError):
            if left != right:
                raise ValueError(f"year-end acceptance checkpoint value mismatch at {path}")
        else:
            if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
                raise ValueError(f"year-end acceptance checkpoint schema mismatch at {path}")
            if left_array.dtype.kind in "fc":
                exact = np.array_equal(left_array, right_array, equal_nan=True)
                if not exact:
                    exact_mismatch_leaves += 1
                if not np.allclose(
                    left_array,
                    right_array,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                ):
                    finite = np.isfinite(left_array) & np.isfinite(right_array)
                    difference = np.abs(left_array[finite] - right_array[finite])
                    field_max_abs = float(np.max(difference)) if difference.size else float("inf")
                    denominator = np.maximum(np.abs(right_array[finite]), np.finfo(np.float64).tiny)
                    relative = difference / denominator
                    field_max_rel = float(np.max(relative)) if relative.size else float("inf")
                    raise ValueError(
                        "year-end acceptance checkpoint float mismatch at "
                        f"{path}: max_abs={field_max_abs:.17g}, "
                        f"max_rel={field_max_rel:.17g}, rtol={rtol}, atol={atol}"
                    )
                finite = np.isfinite(left_array) & np.isfinite(right_array)
                if np.any(finite):
                    difference = np.abs(left_array[finite] - right_array[finite])
                    max_abs = max(max_abs, float(np.max(difference)))
                    denominator = np.maximum(
                        np.abs(right_array[finite]), np.finfo(np.float64).tiny
                    )
                    max_rel = max(max_rel, float(np.max(difference / denominator)))
            elif not np.array_equal(left_array, right_array):
                raise ValueError(f"year-end acceptance checkpoint value mismatch at {path}")
        compared += 1

    for component in expected_components:
        compare(actual_components[component], expected_components[component], component)
    return {
        "status": "exact" if exact_mismatch_leaves == 0 else "numeric_close",
        "compared_state_leaves": compared,
        "exact_mismatch_leaves": exact_mismatch_leaves,
        "rtol": rtol,
        "atol": atol,
        "max_abs_error": max_abs,
        "max_relative_error": max_rel,
    }


def _entry_capture_start(
    plan: GenerationPlan,
    entry: PlanEntry,
    context,
    previous_state,
):
    if entry.initialization_mode == COLD_START_BOOTSTRAP:
        bootstrap = teacher.paper_1961_driver_cold_start_day_scaffold(
            plan.teacher_config,
            year=entry.year,
            used_run_def_path=entry.run_def,
            reference_run_dir=entry.reference_run_dir,
            prepared_context=context,
        )
        if not bootstrap.ready_for_first_day_end_state:
            raise RuntimeError(
                f"{entry.key} cold-start Day 1 did not produce canonical state: "
                f"state_gaps={bootstrap.state_gaps}"
            )
        return bootstrap.first_day_end_state, 2, entry.days - 1, 1

    source_state = _initial_state(entry, previous_state)
    return teacher.rebase_driver_state_for_year_start(source_state), 1, entry.days, None


def _write_entry(
    plan: GenerationPlan,
    entry: PlanEntry,
    worker_root: Path,
    *,
    git_head: str,
    previous_state,
    preceding_checkpoint_sha256: str | None,
) -> tuple[dict[str, Any], Any]:
    entry_started = time.perf_counter()
    memory_before = _process_memory_bytes()
    cache_before = _compiled_cache_entries()
    preparation_started = time.perf_counter()
    context = teacher.prepare_paper_1961_driver_context(
        plan.teacher_config,
        used_run_def_path=entry.run_def,
        reference_run_dir=entry.reference_run_dir,
    )
    initial_state, start_day, transition_days, bootstrap_day = _entry_capture_start(
        plan, entry, context, previous_state
    )
    preparation_seconds = time.perf_counter() - preparation_started
    print(
        f"teacher_entry_prepared key={entry.key} "
        f"seconds={preparation_seconds:.3f} cache={cache_before}",
        flush=True,
    )
    capture_blocks = _iter_capture_days_compiled_blocks(
        config_path=plan.teacher_config,
        context=context,
        previous_state=initial_state,
        year=entry.year,
        start_day=start_day,
        days=transition_days,
        block_size=plan.block_size,
        compact_contract_factory=_compact_contract_factory(
            context,
            year=entry.year,
            first_day_index=start_day,
        ),
    )
    capture_seconds = 0.0

    def timed_blocks():
        nonlocal capture_seconds
        iterator = iter(capture_blocks)
        while True:
            started = time.perf_counter()
            try:
                block = next(iterator)
            except StopIteration:
                capture_seconds += time.perf_counter() - started
                return
            capture_seconds += time.perf_counter() - started
            yield block

    assembly_started = time.perf_counter()
    arrays, contract, final_state = build_shard_arrays_from_compact_blocks(
        timed_blocks(), context
    )
    capture_and_assembly_seconds = time.perf_counter() - assembly_started
    array_assembly_seconds = capture_and_assembly_seconds - capture_seconds
    cache_after = _compiled_cache_entries()
    print(
        f"teacher_entry_captured key={entry.key} seconds={capture_seconds:.3f} "
        f"cache_before={cache_before} cache_after={cache_after}",
        flush=True,
    )
    shard, metadata_path, checkpoint = _entry_paths(worker_root, entry)
    checkpoint_payload = {
        "schema_version": "daily_teacher_year_end_checkpoint_v2",
        "teacher_git_head": git_head,
        "plan_sha256": plan.plan_sha256,
        "landpoint_id": entry.landpoint_id,
        "end_year": entry.year,
        "state": final_state,
    }
    acceptance = None
    if entry.acceptance_checkpoint is not None:
        accepted_state = _load_state_cache(entry.acceptance_checkpoint)["state"]
        try:
            acceptance = {
                **_compare_driver_state(final_state, accepted_state),
                "checkpoint_sha256": _sha256_file(entry.acceptance_checkpoint),
            }
        except ValueError:
            rejected = checkpoint.with_name(f"{checkpoint.stem}.rejected.pkl")
            _atomic_pickle(rejected, checkpoint_payload)
            print(f"teacher_entry_rejected_checkpoint path={rejected}", flush=True)
            raise
    npz_started = time.perf_counter()
    _atomic_npz(shard, arrays)
    npz_write_seconds = time.perf_counter() - npz_started
    checkpoint_started = time.perf_counter()
    _atomic_pickle(checkpoint, checkpoint_payload)
    checkpoint_write_seconds = time.perf_counter() - checkpoint_started
    shard_sha256 = _sha256_file(shard)
    checkpoint_sha256 = _sha256_file(checkpoint)
    memory_after = _process_memory_bytes()
    total_entry_seconds = time.perf_counter() - entry_started
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
        "initialization_mode": entry.initialization_mode,
        "bootstrap_day": bootstrap_day,
        "transition_start_day": start_day,
        "transition_count": transition_days,
        "spatial_split": entry.spatial_split,
        "temporal_split": entry.temporal_split,
        "block_size": plan.block_size,
        "capture_mode": "compact_projected_compiled_blocks",
        "capture_seconds": capture_seconds,
        "timing_seconds": {
            "context_and_state_preparation": preparation_seconds,
            "capture": capture_seconds,
            "array_assembly": array_assembly_seconds,
            "capture_and_incremental_assembly": capture_and_assembly_seconds,
            "npz_write": npz_write_seconds,
            "checkpoint_write": checkpoint_write_seconds,
            "total_entry_before_metadata_write": total_entry_seconds,
        },
        "compiled_cache_entries": {
            "before": cache_before,
            "after": cache_after,
            "delta": {
                name: cache_after[name] - cache_before[name]
                for name in cache_before
            },
        },
        "process_memory": {
            "before": memory_before,
            "after": memory_after,
        },
        "preceding_checkpoint_sha256": preceding_checkpoint_sha256,
        "input_hashes": _input_hashes(plan, entry),
        "year_end_acceptance": acceptance,
        "markov_contract": contract.metadata(),
        "markov_contract_sha256": contract.sha256,
        "arrays": _array_metadata(arrays),
        "uncompressed_array_bytes": estimated_uncompressed_bytes(arrays),
        "omitted_redundancy": [
            "expanded forcing_48",
            "duplicated day_start_state/teacher_target state",
            "persisted finite masks",
            "sechiba_finalize_state diagnostic mirrors",
        ],
        "shard": _relative(shard, worker_root),
        "shard_sha256": shard_sha256,
        "shard_bytes": shard.stat().st_size,
        "checkpoint": _relative(checkpoint, worker_root),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": checkpoint.stat().st_size,
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
                print(f"teacher_entry_reused key={entry.key}", flush=True)
                continue
            print(f"teacher_entry_start key={entry.key}", flush=True)
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
            print(
                f"teacher_entry_complete key={entry.key} "
                f"total_seconds={metadata['timing_seconds']['total_entry_before_metadata_write']:.3f} "
                f"shard_bytes={metadata['shard_bytes']}",
                flush=True,
            )
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
        "worker_assignment_strategy": WORKER_ASSIGNMENT_STRATEGY,
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
        if manifest.get("worker_assignment_strategy") != WORKER_ASSIGNMENT_STRATEGY:
            raise ValueError(f"worker assignment strategy mismatch in {manifest_path}")
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
    contract_hashes = {item["markov_contract_sha256"] for item in observed.values()}
    if len(contract_hashes) != 1:
        raise ValueError("Teacher Markov contracts differ across shards")
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_id": plan.dataset_id,
        "status": "complete",
        "provisional_teacher": True,
        "plan": str(plan.path),
        "plan_sha256": plan.plan_sha256,
        "teacher_git_head": next(iter(heads)),
        "worker_count": worker_count,
        "worker_assignment_strategy": WORKER_ASSIGNMENT_STRATEGY,
        "landpoint_count": len({entry.landpoint_id for entry in plan.entries}),
        "year_count": len({entry.year for entry in plan.entries}),
        "shard_count": len(observed),
        "markov_contract_sha256": next(iter(contract_hashes)),
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
    recover = subparsers.add_parser("recover-lock")
    recover.add_argument("--plan", type=Path, required=True)
    recover.add_argument("--output-root", type=Path)
    recover.add_argument("--worker-index", type=int, required=True)
    recover.add_argument("--worker-count", type=int, required=True)
    recover.add_argument("--expected-host", required=True)
    recover.add_argument("--expected-pid", type=int, required=True)
    recover.add_argument("--expected-slurm-job-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = load_plan(args.plan, require_inputs=args.command in {"validate", "generate"})
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
    if args.command == "recover-lock":
        result = recover_stale_worker_lock(
            plan,
            output_root=output_root,
            worker_index=args.worker_index,
            worker_count=args.worker_count,
            expected_host=args.expected_host,
            expected_pid=args.expected_pid,
            expected_slurm_job_id=args.expected_slurm_job_id,
        )
        print(json.dumps(result, indent=2))
        return 0
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
