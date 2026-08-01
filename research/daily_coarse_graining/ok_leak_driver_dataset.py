"""Hash-bound reader for bounded Teacher OK_LEAK driver captures."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.carbon_budget_ownership import (
    OK_LEAK_DRIVER_SERIES,
    ok_leak_driver_capture_metadata,
)
from research.daily_coarse_graining.daily_markov_contract import (
    daily_markov_contract_from_metadata,
    load_markov_shard,
)
from research.daily_coarse_graining.markov_dataset import load_dataset_index

CAPTURE_MANIFEST_SCHEMA_VERSION = "ok_leak_auxiliary_capture_manifest_v2"
CAPTURE_REPORT_SCHEMA_VERSION = "ok_leak_outer_block_reentry_probe_v2"
CAPTURE_PLAN_SCHEMA_VERSION = "ok_leak_auxiliary_capture_plan_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _record_key(record: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(record["landpoint_id"]),
        int(record["year"]),
        int(record["day_index"]),
    )


def _safe_child(root: Path, relative: str | Path) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"capture path escapes its root: {path}") from error
    return path


def _load_driver_arrays(path: Path) -> dict[str, np.ndarray]:
    expected = tuple(item.field for item in OK_LEAK_DRIVER_SERIES)
    with np.load(path, allow_pickle=False) as payload:
        if tuple(payload.files) != expected:
            raise ValueError("OK_LEAK driver NPZ schema/order drift")
        arrays = {name: np.asarray(payload[name]).copy() for name in expected}
    if {int(value.shape[0]) for value in arrays.values()} != {48}:
        raise ValueError("OK_LEAK driver NPZ must contain exactly 48 steps")
    if not all(np.issubdtype(value.dtype, np.floating) for value in arrays.values()):
        raise ValueError("OK_LEAK driver NPZ fields must be floating point")
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("OK_LEAK driver NPZ contains nonfinite values")
    return arrays


@dataclass(frozen=True)
class OkLeakDriverSample:
    landpoint_id: str
    year: int
    day_index: int
    state: np.ndarray
    next_state: np.ndarray
    fast_day_target: np.ndarray
    forcing_native: np.ndarray
    parameters: np.ndarray
    landpoint_static: np.ndarray
    annual_conditions: np.ndarray
    diagnostics: np.ndarray
    discrete_state: Mapping[str, np.ndarray]
    next_discrete_state: Mapping[str, np.ndarray]
    driver_steps: Mapping[str, np.ndarray]
    capture_npz_sha256: str
    report_path: Path


@dataclass(frozen=True)
class OkLeakDriverDataset:
    dataset_id: str
    teacher_git_head: str
    contract_sha256: str
    capture_plan_sha256: str
    samples: tuple[OkLeakDriverSample, ...]


def load_ok_leak_driver_dataset(
    capture_root: str | Path,
    capture_plan_path: str | Path,
    dataset_manifest_path: str | Path,
    *,
    require_complete: bool = True,
) -> OkLeakDriverDataset:
    """Load every capture only after its source row and all hashes close."""

    capture_root = Path(capture_root).resolve()
    capture_plan_path = Path(capture_plan_path).resolve()
    dataset_manifest_path = Path(dataset_manifest_path).resolve()
    capture_manifest_path = capture_root / "capture_manifest.json"
    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    capture_plan = json.loads(capture_plan_path.read_text(encoding="utf-8"))
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))

    if capture_manifest.get("schema_version") != CAPTURE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported OK_LEAK capture manifest schema")
    if capture_plan.get("schema_version") != CAPTURE_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported OK_LEAK capture plan schema")
    unsigned_plan = {
        name: value for name, value in capture_plan.items() if name != "plan_sha256"
    }
    plan_sha256 = _canonical_sha256(unsigned_plan)
    if capture_plan.get("plan_sha256") != plan_sha256:
        raise ValueError("OK_LEAK capture plan self-hash mismatch")
    if capture_manifest.get("capture_plan_sha256") != plan_sha256:
        raise ValueError("capture manifest does not bind the accepted plan")
    if _sha256_file(dataset_manifest_path) != capture_plan.get(
        "dataset_manifest_sha256"
    ):
        raise ValueError("capture plan dataset-manifest hash mismatch")

    identity = {
        "dataset_id": str(dataset_manifest["dataset_id"]),
        "teacher_git_head": str(dataset_manifest["teacher_git_head"]),
        "markov_contract_sha256": str(dataset_manifest["markov_contract_sha256"]),
    }
    for name, observed in identity.items():
        if capture_plan.get(name) != observed:
            raise ValueError(f"capture plan {name} drift")
    if capture_manifest.get("dataset_id") != identity["dataset_id"]:
        raise ValueError("capture manifest dataset identity drift")
    if capture_plan.get("sealed_test_used") is not False or capture_manifest.get(
        "sealed_test_used"
    ) is not False:
        raise ValueError("OK_LEAK auxiliary captures must remain train-only")

    plan_records = {_record_key(item): item for item in capture_plan["records"]}
    manifest_records = {
        _record_key(item): item for item in capture_manifest.get("records", ())
    }
    if len(plan_records) != len(capture_plan["records"]):
        raise ValueError("duplicate OK_LEAK capture-plan records")
    if len(manifest_records) != len(capture_manifest.get("records", ())):
        raise ValueError("duplicate OK_LEAK capture-manifest records")
    if require_complete:
        if capture_manifest.get("status") != "complete":
            raise ValueError("OK_LEAK capture dataset is not complete")
        if set(manifest_records) != set(plan_records):
            raise ValueError("complete capture manifest does not cover the full plan")
    elif not set(manifest_records) <= set(plan_records):
        raise ValueError("capture manifest contains records outside the accepted plan")
    if capture_manifest.get("capture_interface_failed_count") != 0:
        raise ValueError("capture manifest contains failed scientific records")

    index = load_dataset_index(
        dataset_manifest_path,
        verify_hashes=False,
        verify_files=True,
    )
    references = {(item.landpoint_id, item.year): item for item in index.shards}
    contract = daily_markov_contract_from_metadata(dataset_manifest["markov_contract"])
    samples = []
    verified_source_hashes: set[Path] = set()
    for key in sorted(manifest_records):
        summary = manifest_records[key]
        planned = plan_records[key]
        report_path = _safe_child(capture_root, summary["report"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("schema_version") != CAPTURE_REPORT_SCHEMA_VERSION:
            raise ValueError(f"capture report schema drift for {key}")
        expected_report = {
            "landpoint_id": key[0],
            "year": key[1],
            "day_index": key[2],
            "day_start_state_sha256": planned["day_start_state_sha256"],
            "fast_day_target_sha256": planned["fast_day_target_sha256"],
            "capture_plan_sha256": plan_sha256,
            "source_shard_sha256": planned["source_shard_sha256"],
            "sealed_test_used": False,
            "capture_interface_passed": True,
        }
        drift = {
            name: (report.get(name), value)
            for name, value in expected_report.items()
            if report.get(name) != value
        }
        if drift:
            raise ValueError(f"capture report identity/status drift for {key}: {drift}")
        arrays_path = _safe_child(report_path.parent, report["capture_npz"])
        observed_npz_hash = _sha256_file(arrays_path)
        if observed_npz_hash != report.get("capture_npz_sha256") or (
            observed_npz_hash != summary.get("capture_npz_sha256")
        ):
            raise ValueError(f"capture NPZ hash mismatch for {key}")
        arrays = _load_driver_arrays(arrays_path)
        metadata = ok_leak_driver_capture_metadata(arrays)
        reported_metadata = dict(report.get("capture") or {})
        reported_metadata.pop("path", None)
        reported_metadata.pop("sha256", None)
        if reported_metadata != metadata:
            raise ValueError(f"capture array metadata drift for {key}")

        reference = references.get((key[0], key[1]))
        if reference is None or reference.sha256 != planned["source_shard_sha256"]:
            raise ValueError(f"capture source-shard manifest drift for {key}")
        if reference.path not in verified_source_hashes:
            if _sha256_file(reference.path) != reference.sha256:
                raise ValueError(f"capture source-shard file hash mismatch for {key}")
            verified_source_hashes.add(reference.path)
        shard = load_markov_shard(reference.path, contract=contract)
        row = int(planned["row"])
        if row < 0 or row >= shard.days:
            raise ValueError(f"capture source row out of range for {key}")
        if int(shard.year) != key[1] or int(shard.day_index[row]) != key[2]:
            raise ValueError(f"capture source row identity drift for {key}")
        if _sha256_array(shard.state_trajectory[row]) != planned[
            "day_start_state_sha256"
        ]:
            raise ValueError(f"capture source state hash mismatch for {key}")
        if _sha256_array(shard.fast_day_target[row]) != planned[
            "fast_day_target_sha256"
        ]:
            raise ValueError(f"capture source target hash mismatch for {key}")
        source = shard.sample(row)
        samples.append(
            OkLeakDriverSample(
                landpoint_id=key[0],
                year=key[1],
                day_index=key[2],
                state=np.asarray(source["state"]),
                next_state=np.asarray(source["next_state"]),
                fast_day_target=np.asarray(source["fast_day_target"]),
                forcing_native=np.asarray(source["forcing_native"]),
                parameters=np.asarray(source["parameters"]),
                landpoint_static=np.asarray(source["landpoint_static"]),
                annual_conditions=np.asarray(source["annual_conditions"]),
                diagnostics=np.asarray(source["diagnostics"]),
                discrete_state=dict(source["discrete_state"]),
                next_discrete_state=dict(source["next_discrete_state"]),
                driver_steps=arrays,
                capture_npz_sha256=observed_npz_hash,
                report_path=report_path,
            )
        )

    return OkLeakDriverDataset(
        dataset_id=index.dataset_id,
        teacher_git_head=index.teacher_git_head,
        contract_sha256=index.contract_sha256,
        capture_plan_sha256=plan_sha256,
        samples=tuple(samples),
    )
