"""Atomic checkpoint handoff for the bounded successor fit."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HANDOFF_SCHEMA_VERSION = "successor_small_fit_checkpoint_handoff_v1"
CHECKPOINT_SCHEMA_VERSION = "successor_small_fit_arm_checkpoint_v1"
CHECKPOINT_SET_SCHEMA_VERSION = "successor_small_fit_checkpoint_set_v1"
FROZEN_HANDOFF_SHA256 = "a587b8a93b57bb69f06776aa80fd2268f6ebe2f2e13c07b70e45a3d6632315b0"
DEFAULT_HANDOFF_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "successor_small_fit_checkpoint_handoff_v1.json"
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("contract_sha256", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SuccessorCheckpointHandoff:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def producer(self) -> Mapping[str, Any]:
        return self.payload["producer_artifacts"]

    @property
    def checkpoint(self) -> Mapping[str, Any]:
        return self.payload["checkpoint_policy"]


def load_successor_checkpoint_handoff(
    path: str | Path = DEFAULT_HANDOFF_PATH,
) -> SuccessorCheckpointHandoff:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != HANDOFF_SCHEMA_VERSION:
        raise ValueError("unsupported successor checkpoint handoff schema")
    if payload.get("status") != "frozen_for_local_and_free_gpu_preflight":
        raise ValueError("successor checkpoint handoff is not frozen for preflight")
    observed = _canonical_sha256(payload)
    if payload.get("contract_sha256") != observed:
        raise ValueError("successor checkpoint handoff hash mismatch")
    if observed != FROZEN_HANDOFF_SHA256:
        raise ValueError("successor checkpoint handoff frozen hash drift")

    producer = payload.get("producer_artifacts", {})
    if (
        int(producer.get("updates", 0)) != 500
        or int(producer.get("artifact_count", 0)) != 500
        or int(producer.get("total_size_bytes", 0)) != 16_717_190_876
        or len(str(producer.get("producer_git_head", ""))) != 40
    ):
        raise ValueError("successor checkpoint producer envelope drift")
    policy = payload.get("checkpoint_policy", {})
    if (
        policy.get("arms")
        != [
            "source_routed_process_attention_v2",
            "source_routed_graph_mlp_control_v2",
        ]
        or int(policy.get("required_updates_per_arm", 0)) != 500
        or policy.get("include_optimizer") is not True
        or policy.get("publish") != "atomic_directory_rename_after_complete_manifest"
        or policy.get("checkpoint_on_failed_gate") is not False
        or policy.get("checkpoint_on_submaximal_probe") is not False
    ):
        raise ValueError("successor checkpoint publication policy drift")
    return SuccessorCheckpointHandoff(payload, source, observed)


def admit_prepared_manifest_for_checkpoint_fit(
    handoff: SuccessorCheckpointHandoff,
    manifest_path: str | Path,
    manifest: Mapping[str, Any],
    *,
    runner_contract_sha256: str,
    schedule_sha256: str,
    consumer_git_head: str,
) -> Mapping[str, Any]:
    """Admit one immutable producer manifest without conflating its HEAD with the consumer."""

    producer = handoff.producer
    identity = manifest.get("identity", {})
    observed = {
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "producer_git_head": identity.get("source_git_head"),
        "producer_runner_contract_sha256": identity.get("runner_contract_sha256"),
        "schedule_sha256": identity.get("schedule_sha256"),
        "dataset_manifest_sha256": identity.get("dataset_manifest_sha256"),
        "successor_candidate_contract_sha256": identity.get("successor_candidate_contract_sha256"),
        "fast_owner_contract_sha256": identity.get("fast_owner_contract_sha256"),
        "updates": int(manifest.get("updates", -1)),
        "artifact_count": len(manifest.get("entries", ())),
        "total_size_bytes": int(manifest.get("total_size_bytes", -1)),
        "trace_signature": (
            manifest.get("trace_signatures", [None])[0]
            if len(manifest.get("trace_signatures", ())) == 1
            else None
        ),
    }
    expected = dict(producer)
    drift = {name: (observed.get(name), wanted) for name, wanted in expected.items() if observed.get(name) != wanted}
    if drift:
        raise ValueError(f"successor prepared-manifest handoff identity drift: {drift}")
    if runner_contract_sha256 != producer["producer_runner_contract_sha256"]:
        raise ValueError("successor consumer loaded a different producer runner contract")
    if schedule_sha256 != producer["schedule_sha256"]:
        raise ValueError("successor consumer schedule differs from the admitted producer schedule")
    if len(consumer_git_head) != 40 or consumer_git_head == producer["producer_git_head"]:
        raise ValueError("successor checkpoint consumer git identity is invalid")
    return observed | {
        "consumer_git_head": consumer_git_head,
        "checkpoint_handoff_contract_sha256": handoff.sha256,
    }


def _tree_schema(tree: Any) -> list[Mapping[str, Any]]:
    import jax

    records = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        array = np.asarray(leaf)
        records.append(
            {
                "path": jax.tree_util.keystr(path),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        )
    if not records:
        raise ValueError("successor checkpoint tree is empty")
    return records


def _validate_finite_tree(tree: Any, *, label: str) -> int:
    import jax

    count = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(leaf)
        if array.dtype.kind not in "biufc" or np.any(~np.isfinite(array)):
            raise ValueError(f"successor checkpoint {label} tree is nonnumeric or nonfinite")
        count += int(array.size)
    if count == 0:
        raise ValueError(f"successor checkpoint {label} tree is empty")
    return count


def _write_pickle(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_successor_checkpoint_set(
    output_dir: str | Path,
    *,
    identity: Mapping[str, Any],
    parameters: Mapping[str, Any],
    optimizers: Mapping[str, Any],
    completed_updates: Mapping[str, int],
    parameter_counts: Mapping[str, int],
    final_metrics: Mapping[str, Any],
    loss_reductions: Mapping[str, Any],
    expected_arms: Sequence[str],
    required_updates: int,
    promotion_passed: bool,
) -> Mapping[str, Any]:
    """Publish both arms as one all-or-nothing checkpoint directory."""

    target = Path(output_dir).resolve()
    arms = tuple(expected_arms)
    expected = set(arms)
    if target.exists():
        raise FileExistsError(f"successor checkpoint set already exists: {target}")
    if not promotion_passed:
        raise ValueError("successor checkpoint publication requires passing promotion gates")
    if not arms or any(set(values) != expected for values in (parameters, optimizers, completed_updates, parameter_counts)):
        raise ValueError("successor checkpoint set requires both complete matched arms")
    if any(int(completed_updates[arm]) != required_updates for arm in arms):
        raise ValueError("successor checkpoint set update count is incomplete")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    if staging.exists():
        raise FileExistsError(f"successor checkpoint staging path already exists: {staging}")
    staging.mkdir()
    records = {}
    try:
        for arm in arms:
            parameter_elements = _validate_finite_tree(parameters[arm], label=f"{arm} parameter")
            optimizer_elements = _validate_finite_tree(optimizers[arm], label=f"{arm} optimizer")
            if parameter_elements != int(parameter_counts[arm]):
                raise ValueError(f"successor checkpoint parameter count drift for {arm}")
            payload = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "identity": dict(identity),
                "arm": arm,
                "completed_updates": int(completed_updates[arm]),
                "parameters": parameters[arm],
                "optimizer": optimizers[arm],
                "parameter_tree_schema": _tree_schema(parameters[arm]),
                "optimizer_tree_schema": _tree_schema(optimizers[arm]),
                "parameter_count": parameter_elements,
                "optimizer_element_count": optimizer_elements,
                "final_metrics": final_metrics[arm],
                "loss_reductions": loss_reductions[arm],
            }
            path = staging / f"{arm}.pkl"
            _write_pickle(path, payload)
            records[arm] = {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "completed_updates": int(completed_updates[arm]),
                "parameter_count": parameter_elements,
                "parameter_tree_schema": payload["parameter_tree_schema"],
                "optimizer_tree_schema": payload["optimizer_tree_schema"],
            }
        manifest = {
            "schema_version": CHECKPOINT_SET_SCHEMA_VERSION,
            "status": "passed",
            "identity": dict(identity),
            "arms": list(arms),
            "required_updates_per_arm": int(required_updates),
            "checkpoints": records,
        }
        _write_json(staging / "checkpoint_manifest.json", manifest)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest | {
        "path": str(target),
        "manifest_sha256": sha256_file(target / "checkpoint_manifest.json"),
    }


def load_successor_checkpoint_set(
    checkpoint_dir: str | Path,
    *,
    expected_manifest_sha256: str,
    identity: Mapping[str, Any],
    expected_arms: Sequence[str],
    required_updates: int,
) -> Mapping[str, Mapping[str, Any]]:
    root = Path(checkpoint_dir).resolve()
    manifest_path = root / "checkpoint_manifest.json"
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("successor checkpoint-set manifest hash drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arms = tuple(expected_arms)
    if (
        manifest.get("schema_version") != CHECKPOINT_SET_SCHEMA_VERSION
        or manifest.get("status") != "passed"
        or manifest.get("identity") != dict(identity)
        or manifest.get("arms") != list(arms)
        or int(manifest.get("required_updates_per_arm", -1)) != required_updates
        or set(manifest.get("checkpoints", {})) != set(arms)
    ):
        raise ValueError("successor checkpoint-set manifest identity drift")
    loaded = {}
    for arm in arms:
        record = manifest["checkpoints"][arm]
        path = (root / str(record["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("successor checkpoint escaped its set directory") from error
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"successor checkpoint hash drift for {arm}")
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if (
            payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or payload.get("identity") != dict(identity)
            or payload.get("arm") != arm
            or int(payload.get("completed_updates", -1)) != required_updates
            or payload.get("parameter_tree_schema") != record.get("parameter_tree_schema")
            or payload.get("optimizer_tree_schema") != record.get("optimizer_tree_schema")
            or _tree_schema(payload.get("parameters")) != payload.get("parameter_tree_schema")
            or _tree_schema(payload.get("optimizer")) != payload.get("optimizer_tree_schema")
        ):
            raise ValueError(f"successor checkpoint payload identity or tree-schema drift for {arm}")
        _validate_finite_tree(payload["parameters"], label=f"{arm} parameter")
        _validate_finite_tree(payload["optimizer"], label=f"{arm} optimizer")
        loaded[arm] = payload
    return loaded
