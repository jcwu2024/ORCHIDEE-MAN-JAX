"""Retained-owner context/template helpers used by current shared execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.hpc.canonical_training_runtime import _retained_adapter


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _plan_context(
    manifest: Mapping[str, Any],
    *,
    landpoint_id: str,
    year: int,
) -> Mapping[str, Any]:
    plan_path = Path(str(manifest["plan"])).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if _canonical_sha256(plan) != str(manifest["plan_sha256"]):
        raise ValueError("v6 generation plan hash drift")
    entries = tuple(
        item for item in plan["entries"] if str(item["landpoint_id"]) == landpoint_id and int(item["year"]) == year
    )
    if len(entries) != 1:
        raise ValueError("small-fit reference has no unique generation-plan entry")
    entry = entries[0]
    result = {
        "config": Path(str(plan["teacher_config"])).resolve(),
        "run_def": Path(str(entry["run_def"])).resolve(),
        "reference_run_dir": Path(str(entry["reference_run_dir"])).resolve(),
        "plan": plan_path,
    }
    if any(not path.exists() for path in result.values()):
        raise FileNotFoundError("small-fit generation-plan context is incomplete")
    return result




def _prepare_template(runtime: Mapping[str, Any], entry: Mapping[str, Any]) -> Any:
    from jax_orchidee.driver import orchestration as teacher
    from research.daily_coarse_graining.daily_markov_contract import load_markov_shard

    first = entry["components"]["one_day"][0]
    matches = tuple(
        reference
        for reference in runtime["index"].select(
            spatial_split="train",
            temporal_split="train",
        )
        if str(reference.parent.landpoint_id) == str(first["landpoint_id"])
        and int(reference.parent.year) == int(first["year"])
        and str(reference.parent.sha256) == str(first["parent_shard_sha256"])
        and str(reference.sha256) == str(first["typed_shard_sha256"])
    )
    if len(matches) != 1:
        raise ValueError("preflight retained template reference is ambiguous")
    paths = _plan_context(
        runtime["manifest"],
        landpoint_id=str(first["landpoint_id"]),
        year=int(first["year"]),
    )
    context = teacher.prepare_paper_1961_driver_context(
        paths["config"],
        used_run_def_path=paths["run_def"],
        reference_run_dir=paths["reference_run_dir"],
    )
    parent = load_markov_shard(matches[0].parent.path, contract=runtime["contract"])
    sample = parent.sample(int(first["start_rows"][0]))
    return _retained_adapter(
        runtime | {"context_paths": paths},
        context,
        sample,
        lifecycle_start_tstep=48,
        result_day_index=2,
        fast_owner_materializer=runtime["materializer"],
    )


def build_parent_process_head(runtime: Mapping[str, Any], target_contract: Any) -> Any:
    from research.daily_coarse_graining.daily_operator.target_oriented_process_head import (
        MechanisticTargetOrientedAttentionProcessHead,
    )

    return MechanisticTargetOrientedAttentionProcessHead(
        contract=runtime["contract"],
        architecture_contract=runtime["architecture_contract"],
        mechanism_contract=runtime["mechanism_contract"],
        input_assembler=runtime["adapter"].input_assembler,
        complete_schema=runtime["schema"],
        learned_schema=runtime["learned_schema"],
        layout=runtime["target_layout"],
        input_normalization=runtime["input_normalization"],
        output_normalization=runtime["output_normalization"],
        forcing_layout=runtime["forcing_layout"],
        forcing_indices=runtime["forcing_indices"],
        message_gate_scale=1.0,
        target_contract=target_contract,
    )
