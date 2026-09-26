"""Canonical v6 data/statistics bootstrap, independent of experiment runners.

Extracted without numerical changes from the canonical-tail and fast-owner loaders.
Neural initialization remains optional; shared execution explicitly disables it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
from scripts.hpc.training_artifact_io import _verify

EXPECTED = {
    "dataset_manifest": "5fc2c0691bfa53252f875f7646f3c1a5bea203c2759a6e947b26bb51dc13aa99",
    "acceptance_report": "8784ed42dc64ef77c9642f613eb2e5940635edc6e744a1af4ea8634293baa8e3",
    "base_statistics_json": "ab7f040a930071bd5604c17355a6493b23f3c76cba1e3084dcd5d5653101a4bf",
    "base_statistics_npz": "e88d7cec504c6051aab6c9feacb619dca94961730ffa18f840563a48fc815a3a",
    "process_statistics_json": "9bc2cab43b3279e39a0fc2876ca298b2b371d6890986d44ef035859ca925eaab",
    "process_statistics_npz": "addf957afbd8338f4aa250155dcd299d568c1070b644e30279f9bf0ea37f9bf8",
    "successor_contract": "bb8bb34972e6fdff713f06f7ce48a2faa9eb4991af9f269c9aab26d96b592b5c",
    "fast_owner_contract": "4db5f828442212c58903096c36015f9d769c19aeaaa4eb9a4326fbea2fdfc4c9",
}
FROZEN_LANDPOINT = "001.0-071.0"
ROLLOUT_START_ROW = 0
ROLLOUT_DAYS = 30
ANCHOR_ROWS = (0, 1)
FAST_STATISTICS_JSON_SHA256 = "b2e61a171156071d404d12b833130f2d0e98064564d1c3e007e16f0210734719"
FAST_STATISTICS_NPZ_SHA256 = "890d1fd6110912a835c760b4fe58e9320481a929797858cad5b70b1bf03eb99f"


def _packet(teacher: Any, reconstruct_state_fields: Any, sample: Mapping[str, Any], tstep: int):
    fields = reconstruct_state_fields(
        np.asarray(sample["state"], dtype=np.float64),
        sample["discrete_state"],
        sample["contract"],
    )
    return teacher.DriverPreviousStepStatePacket(
        tstep=int(tstep),
        fields_by_component=fields,
        provenance_by_component={
            component: ("successor real-v6 canonical gate",)
            for component in fields
        },
    )


def _select_references(index: Any) -> tuple[Any, Any]:
    references = tuple(
        sorted(
            index.select(spatial_split="train", temporal_split="train"),
            key=lambda item: (item.parent.landpoint_id, item.parent.year),
        )
    )
    by_key = {
        (item.parent.landpoint_id, int(item.parent.year)): item
        for item in references
    }
    try:
        ordinary = by_key[(FROZEN_LANDPOINT, 1961)]
        restart = by_key[(FROZEN_LANDPOINT, 1962)]
    except KeyError as error:
        raise ValueError("frozen lifecycle references are absent from train/train") from error
    if ordinary.day_count < ROLLOUT_DAYS or restart.day_count < 1:
        raise ValueError("frozen lifecycle references are too short")
    return ordinary, restart


def _plan_context(manifest: Mapping[str, Any], landpoint_id: str) -> dict[str, Path]:
    plan_path = Path(str(manifest["plan"])).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    canonical_plan_sha256 = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if canonical_plan_sha256 != str(manifest["plan_sha256"]):
        raise ValueError("v6 generation plan hash drift")
    entries = tuple(
        item
        for item in plan["entries"]
        if item["landpoint_id"] == landpoint_id and int(item["year"]) == 1961
    )
    if len(entries) != 1:
        raise ValueError("frozen landpoint does not have one 1961 generation entry")
    entry = entries[0]
    result = {
        "config": Path(str(plan["teacher_config"])).resolve(),
        "run_def": Path(str(entry["run_def"])).resolve(),
        "reference_run_dir": Path(str(entry["reference_run_dir"])).resolve(),
        "plan": plan_path,
    }
    if any(not path.exists() for path in result.values()):
        raise FileNotFoundError("v6 plan context is incomplete on this host")
    return result


def _load_runtime(
    args: argparse.Namespace,
    *,
    backend_initializer: Any = initialize_gpu_backend,
    initialize_probe_parameters: bool = True,
) -> dict[str, Any]:
    base_payload = json.loads(args.base_statistics.read_text(encoding="utf-8"))
    base_npz = args.base_statistics.parent / str(base_payload["statistics_npz"])
    process_payload = json.loads(
        args.process_statistics.read_text(encoding="utf-8")
    )
    process_npz = args.process_statistics.parent / str(
        process_payload["statistics_npz"]
    )
    for name, path in (
        ("dataset_manifest", args.dataset_manifest),
        ("acceptance_report", args.acceptance_report),
        ("base_statistics_json", args.base_statistics),
        ("base_statistics_npz", base_npz),
        ("process_statistics_json", args.process_statistics),
        ("process_statistics_npz", process_npz),
    ):
        _verify(path.resolve(), EXPECTED[name], name)

    acceptance = json.loads(args.acceptance_report.read_text(encoding="utf-8"))
    if acceptance.get("status") != "passed":
        raise ValueError("v6 acceptance report is not passed")
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    context_paths = _plan_context(manifest, FROZEN_LANDPOINT)

    jax, devices = backend_initializer()
    jax.config.update("jax_enable_x64", True)

    from research.daily_coarse_graining.daily_markov_contract import (
        daily_markov_contract_from_metadata,
        load_markov_shard,
    )
    from research.daily_coarse_graining.daily_operator.process_derivations import (
        NativeForcingFieldLayout,
    )
    from research.daily_coarse_graining.daily_operator.successor_architecture import (
        SOURCE_ROUTED_ATTENTION,
        build_successor_token_layout,
        initialize_successor_parameters,
        successor_parameter_count,
    )
    from research.daily_coarse_graining.daily_operator.successor_contract import (
        load_successor_candidate_contract,
    )
    from research.daily_coarse_graining.daily_operator.training_contract import (
        load_first_candidate_contract,
    )
    from research.daily_coarse_graining.daily_operator.training_data import (
        DailyProcessTrainingAdapter,
    )
    from research.daily_coarse_graining.daily_operator.training_statistics import (
        load_process_training_statistics,
    )
    from research.daily_coarse_graining.markov_dataset import (
        load_training_statistics,
    )
    from research.daily_coarse_graining.typed_sidecar import (
        load_coherent_typed_sidecar_index,
        read_typed_sidecar_shard,
    )

    successor_contract = load_successor_candidate_contract()
    if successor_contract.sha256 != EXPECTED["successor_contract"]:
        raise ValueError("successor contract hash drift")
    contract = daily_markov_contract_from_metadata(manifest["markov_contract"])
    if contract.sha256 != successor_contract.payload["data"]["markov_contract_sha256"]:
        raise ValueError("successor Markov contract drift")
    adapter = DailyProcessTrainingAdapter.from_contract(contract)
    index = load_coherent_typed_sidecar_index(
        args.dataset_manifest,
        verify_hashes=False,
    )
    base_statistics = load_training_statistics(
        args.base_statistics,
        index=index.parent_index,
    )
    process_statistics = load_process_training_statistics(
        args.process_statistics,
        index=index,
        adapter=adapter,
        candidate_contract=load_first_candidate_contract(),
        base_statistics=base_statistics,
    )
    if process_statistics.sample_count != 8_591_565:
        raise ValueError("formal process-statistics sample count drift")
    successor_data = successor_contract.payload["data"]
    if (
        successor_data["training_statistics_json_sha256"]
        != EXPECTED["process_statistics_json"]
        or successor_data["training_statistics_npz_sha256"]
        != EXPECTED["process_statistics_npz"]
    ):
        raise ValueError("successor process-statistics identity drift")
    ordinary_reference, restart_reference = _select_references(index)
    ordinary_parent = load_markov_shard(
        ordinary_reference.parent.path,
        contract=contract,
    )
    restart_parent = load_markov_shard(
        restart_reference.parent.path,
        contract=contract,
    )
    typed_values, typed_defined, typed_days = read_typed_sidecar_shard(
        ordinary_reference.path,
        contract=index.contract,
        expected_days=ordinary_reference.day_count,
    )
    restart_typed_values, restart_typed_defined, restart_typed_days = (
        read_typed_sidecar_shard(
            restart_reference.path,
            contract=index.contract,
            expected_days=restart_reference.day_count,
        )
    )
    if not np.array_equal(typed_days, ordinary_parent.day_index):
        raise ValueError("frozen typed and parent day indices differ")
    if tuple(np.asarray(ordinary_parent.day_index[:2]).tolist()) != (2, 3):
        raise ValueError("frozen ordinary lifecycle rows are not 1961 Day 2/3")
    if int(np.asarray(restart_parent.day_index[0])) != 1:
        raise ValueError("frozen restart lifecycle row is not Day 1")
    if not np.array_equal(restart_typed_days, restart_parent.day_index):
        raise ValueError("frozen restart typed and parent day indices differ")

    examples = [
        adapter.assemble(
            ordinary_parent.sample(row),
            supplemental_values={
                name: value[row] for name, value in typed_values.items()
            },
            supplemental_defined={
                name: value[row] for name, value in typed_defined.items()
            },
        )
        for row in range(ROLLOUT_START_ROW, ROLLOUT_START_ROW + ROLLOUT_DAYS)
    ]
    assembled = [adapter.input_assembler.assemble(item.inputs) for item in examples]
    schema = process_statistics.schema
    layout = build_successor_token_layout(
        adapter.input_assembler,
        schema,
        successor_contract,
    )
    parameters = (initialize_successor_parameters(
        jax.random.PRNGKey(20260809), schema=schema, layout=layout,
        inputs=assembled[0], arm=SOURCE_ROUTED_ATTENTION,
    ) if initialize_probe_parameters else None)
    if initialize_probe_parameters and successor_parameter_count(parameters) != 8_679_681:
        raise ValueError("successor parameter count drift")
    forcing_fields = contract.native_forcing.fields
    forcing_layout = NativeForcingFieldLayout(
        rain_start=forcing_fields.index("Rainf"),
        rain_stop=forcing_fields.index("Rainf") + 1,
        snow_start=forcing_fields.index("Snowf"),
        snow_stop=forcing_fields.index("Snowf") + 1,
        temperature_start=forcing_fields.index("Tair"),
        temperature_stop=forcing_fields.index("Tair") + 1,
    )
    return {
        "jax": jax,
        "devices": devices,
        "manifest": manifest,
        "context_paths": context_paths,
        "contract": contract,
        "adapter": adapter,
        "index": index,
        "base_statistics": base_statistics,
        "process_statistics": process_statistics,
        "ordinary_reference": ordinary_reference,
        "restart_reference": restart_reference,
        "ordinary_parent": ordinary_parent,
        "restart_parent": restart_parent,
        "restart_typed_values": restart_typed_values,
        "restart_typed_defined": restart_typed_defined,
        "examples": examples,
        "assembled": assembled,
        "schema": schema,
        "layout": layout,
        "parameters": parameters,
        "forcing_layout": forcing_layout,
        "forcing_indices": {
            name: index for index, name in enumerate(forcing_fields)
        },
        "input_normalization": process_statistics.input_normalization(),
        "output_normalization": process_statistics.output_normalization(),
    }


def _retained_adapter(
    runtime: Mapping[str, Any],
    teacher_context: Any,
    sample: Mapping[str, Any],
    *,
    lifecycle_start_tstep: int,
    result_day_index: int,
    fast_owner_materializer: Any | None = None,
) -> Any:
    from jax_orchidee.driver import orchestration as teacher
    from research.daily_coarse_graining.daily_markov_contract import (
        reconstruct_state_fields,
    )
    from research.daily_coarse_graining.daily_operator.canonical_retained_tail import (
        build_canonical_retained_tail_adapter,
    )

    packet_sample = dict(sample)
    packet_sample["contract"] = runtime["contract"]
    initial_state = _packet(
        teacher,
        reconstruct_state_fields,
        packet_sample,
        lifecycle_start_tstep - 1,
    )
    return build_canonical_retained_tail_adapter(
        config_path=runtime["context_paths"]["config"],
        context=teacher_context,
        initial_state=initial_state,
        contract=runtime["contract"],
        input_assembler=runtime["adapter"].input_assembler,
        runtime_year=int(np.asarray(sample["year"])),
        lifecycle_start_tstep=lifecycle_start_tstep,
        result_day_index=result_day_index,
        include_modelout_diagnostics=False,
        fast_owner_materializer=fast_owner_materializer,
    )


def _load_extended_runtime(
    args: argparse.Namespace,
    *,
    backend_initializer: Any | None = None,
    initialize_probe_parameters: bool = True,
) -> dict[str, Any]:
    extra = {} if initialize_probe_parameters else {'initialize_probe_parameters': False}
    runtime = (
        _load_runtime(args, **extra)
        if backend_initializer is None
        else _load_runtime(args, backend_initializer=backend_initializer, **extra)
    )
    payload = json.loads(args.fast_statistics.read_text(encoding="utf-8"))
    arrays_path = args.fast_statistics.parent / str(payload["statistics_npz"])
    _verify(
        args.fast_statistics.resolve(),
        FAST_STATISTICS_JSON_SHA256,
        "successor_fast_owner_statistics_json",
    )
    _verify(
        arrays_path.resolve(),
        FAST_STATISTICS_NPZ_SHA256,
        "successor_fast_owner_statistics_npz",
    )

    from research.daily_coarse_graining.daily_operator.fast_owner_reconstruction import (
        build_successor_fast_owner_materializer,
        load_fast_owner_reconstruction_contract,
    )
    from research.daily_coarse_graining.daily_operator.successor_architecture import (
        build_successor_token_layout,
    )
    from research.daily_coarse_graining.daily_operator.successor_contract import (
        load_successor_candidate_contract,
    )
    from research.daily_coarse_graining.daily_operator.successor_fast_owner_decoder import (
        build_successor_fast_owner_schema,
    )
    from research.daily_coarse_graining.daily_operator.successor_fast_owner_statistics import (
        load_successor_fast_owner_statistics,
    )
    from research.daily_coarse_graining.daily_operator.successor_fast_owner_training import (
        build_successor_fast_owner_targets,
    )

    owner = load_fast_owner_reconstruction_contract(
        markov_contract=runtime["contract"],
    )
    if owner.contract_sha256 != EXPECTED["fast_owner_contract"]:
        raise ValueError("fast-owner contract identity drift")
    if len(owner.leaves) != 160:
        raise ValueError("fast-owner leaf count drift")
    schema = build_successor_fast_owner_schema(
        runtime["process_statistics"].schema,
        markov_contract=runtime["contract"],
        owner_contract=owner,
    )
    successor_contract = load_successor_candidate_contract()
    if successor_contract.sha256 != EXPECTED["successor_contract"]:
        raise ValueError("successor contract identity drift")
    statistics = load_successor_fast_owner_statistics(
        args.fast_statistics,
        index=runtime["index"],
        adapter=runtime["adapter"],
        base=runtime["process_statistics"],
        successor_contract=successor_contract,
        owner_contract=owner,
        schema=schema,
    )
    if statistics.sample_count != 8_591_565 or statistics.source_shard_count != 23_540:
        raise ValueError("fast-owner statistics coverage drift")
    if len(statistics.outputs) != 79:
        raise ValueError("fast-owner statistics field count drift")
    normalization = statistics.output_normalization(runtime["process_statistics"])
    scales = statistics.combined_loss_scales(runtime["process_statistics"])
    if len(schema.fields) != 181 or len(schema.labels) != 48 or len(normalization.mean) != 181 or len(scales) != 180:
        raise ValueError("complete successor decoder registry drift")
    layout = build_successor_token_layout(
        runtime["adapter"].input_assembler,
        schema,
        successor_contract,
    )
    example = runtime["examples"][ANCHOR_ROWS[0]]
    assembled = runtime["adapter"].input_assembler.assemble(example.inputs)
    targets = build_successor_fast_owner_targets(
        example,
        schema=schema,
        markov_contract=runtime["contract"],
        owner_contract=owner,
    )
    materializer = build_successor_fast_owner_materializer(
        runtime["contract"],
        forcing_layout=runtime["forcing_layout"],
    )
    return runtime | {
        "fast_statistics": statistics,
        "successor_contract": successor_contract,
        "owner_contract": owner,
        "schema": schema,
        "layout": layout,
        "output_normalization": normalization,
        "loss_scales": scales,
        "example": example,
        "probe_assembled": assembled,
        "targets": targets,
        "materializer": materializer,
    }
