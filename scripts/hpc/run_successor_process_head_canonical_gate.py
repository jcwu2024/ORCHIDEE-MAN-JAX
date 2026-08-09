"""Real-v6 canonical-tail lifecycle and rollout gates for the successor pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend

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
ROLLOUT_YEAR = 1961
ROLLOUT_START_ROW = 0
ROLLOUT_DAYS = 30
ANCHOR_ROWS = (0, 1)
INITIALIZATION_DECODER_SCALES = (1.0, 0.3, 0.1, 0.03, 0.01, 0.0)
INITIALIZATION_NONNEGATIVE_BASELINE_SCALES = (
    1.0,
    0.3,
    0.1,
    0.03,
    0.01,
    0.003,
    0.001,
)
INITIALIZATION_COHERENT_CARBON_SCALES = (0.3, 0.1, 0.03, 0.01, 0.003)
TARGET_GRADIENT_RATIOS = {
    "cumulative_flux_mse": 0.10,
    "cumulative_flux_bias_squared": 0.15,
    "terminal_stock_mse": 0.10,
    "terminal_stock_bias_squared": 0.15,
}
EXACT_RETAINED_FAST_FIELDS = (
    ("driver_previous_step_state", "old_eair"),
    ("driver_previous_step_state", "old_qair"),
    ("driver_previous_step_state", "old_zlev"),
    ("driver_previous_step_state", "rau_old"),
    ("diffuco_previous_step_state", "frac_nobio"),
    ("diffuco_previous_step_state", "lai"),
    ("diffuco_previous_step_state", "veget"),
    ("diffuco_previous_step_state", "veget_max"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected: str, name: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{name} hash drift: {observed} != {expected}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _json_value(child) for name, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _stack_trees(jax: Any, values: list[Any]) -> Any:
    return jax.tree_util.tree_map(lambda *items: np.stack(items), *values)


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
    parameters = initialize_successor_parameters(
        jax.random.PRNGKey(20260809),
        schema=schema,
        layout=layout,
        inputs=assembled[0],
        arm=SOURCE_ROUTED_ATTENTION,
    )
    if successor_parameter_count(parameters) != 8_679_681:
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


def _head(runtime: Mapping[str, Any], arm: str) -> Any:
    from research.daily_coarse_graining.daily_operator.successor_process_head import (
        SuccessorProcessHead,
    )

    return SuccessorProcessHead(
        arm=arm,
        contract=runtime["contract"],
        input_assembler=runtime["adapter"].input_assembler,
        schema=runtime["schema"],
        token_layout=runtime["layout"],
        input_normalization=runtime["input_normalization"],
        output_normalization=runtime["output_normalization"],
        forcing_layout=runtime["forcing_layout"],
        forcing_indices=runtime["forcing_indices"],
    )


class _NonnegativeScaleProbeHead:
    """Scale bounded nonnegative outputs only for initialization diagnosis."""

    def __init__(self, base: Any, schema: Any):
        self.base = base
        self.nonnegative_fields = frozenset(
            field.name for field in schema.fields if field.constraint == "nonnegative"
        )

    def assemble_context(self, parameters: Mapping[str, Any], inputs: Any) -> Any:
        return self.base.assemble_context(parameters["model"], inputs)

    def __call__(
        self,
        parameters: Mapping[str, Any],
        context: Any,
        inputs: Any,
    ) -> Any:
        outputs = self.base.named_outputs(parameters["model"], context, inputs)
        generic_scale = parameters["nonnegative_scale"]
        scaled = {}
        for name, value in outputs.items():
            scale = generic_scale
            if name == "daily.fast_gpp":
                scale = parameters.get("gpp_scale", generic_scale)
            elif name == "daily.fast_maintenance_respiration":
                scale = parameters.get("maintenance_scale", generic_scale)
            scaled[name] = value * scale if name in self.nonnegative_fields else value
        return self.base.prediction_from_named_outputs(scaled, inputs)


class _TeacherReplacementProbeHead:
    """Replace selected named outputs by same-row Teacher labels."""

    def __init__(self, base: Any, schema: Any):
        self.base = base
        self.field_names = tuple(field.name for field in schema.fields)

    def assemble_context(self, parameters: Mapping[str, Any], inputs: Any) -> Any:
        return self.base.assemble_context(parameters["model"], inputs)

    def __call__(
        self,
        parameters: Mapping[str, Any],
        context: Any,
        inputs: Any,
    ) -> Any:
        import jax.numpy as jnp

        outputs = self.base.named_outputs(parameters["model"], context, inputs)
        replacement_mask = jnp.asarray(parameters["replacement_mask"], dtype=jnp.bool_)
        teacher_outputs = parameters["teacher_outputs"]
        hybrid = {
            name: jnp.where(replacement_mask[index], teacher_outputs[name], outputs[name])
            for index, name in enumerate(self.field_names)
        }
        return self.base.prediction_from_named_outputs(hybrid, inputs)


class _AllTeacherFastOwnerHead:
    """Adapt same-row Teacher process and endpoint values without a network."""

    def __init__(self, base: Any):
        self.base = base

    def assemble_context(self, parameters: Mapping[str, Any], inputs: Any) -> Any:
        import jax.numpy as jnp

        return jnp.asarray(inputs.day_index)

    def __call__(
        self,
        parameters: Mapping[str, Any],
        context: Any,
        inputs: Any,
    ) -> Any:
        return self.base.prediction_from_named_outputs(
            parameters["named_outputs"],
            inputs,
        )

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


def _nested_value(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for name in path:
        current = current[name]
    return current


def _retained_boundary_report(
    prediction: Any,
    sample: Mapping[str, Any],
    contract: Any,
) -> dict[str, Any]:
    from research.daily_coarse_graining.markov_dataset import defined_numeric_mask

    groups = {
        "daily_interface": prediction.retained_boundary.daily_fields,
        "ok_leak": prediction.retained_boundary.ok_leak_result,
        "final_diagnostics": prediction.retained_boundary.final_entry_diagnostics,
    }
    expected = np.asarray(sample["fast_day_target"], dtype=np.float64)
    records = []
    exact_leaf_count = 0
    for leaf in contract.fast_day_target_leaves:
        source = (
            _nested_value(
                prediction.retained_boundary.fast_state_fields[leaf.component],
                leaf.path,
            )
            if leaf.component is not None
            else _nested_value(groups[leaf.family], leaf.path)
        )
        compact = np.asarray(source)
        if compact.shape != leaf.shape:
            for axis, axis_name in enumerate(leaf.axis_names):
                if axis_name == "nvm":
                    compact = np.take(
                        compact,
                        np.asarray(leaf.selected_pft_indices, dtype=np.int64),
                        axis=axis,
                    )
        if compact.shape != leaf.shape:
            raise ValueError(
                f"retained boundary diagnostic shape drift for {leaf.key}: "
                f"{compact.shape} != {leaf.shape}"
            )
        observed = np.asarray(compact, dtype=np.float64).reshape(-1)
        wanted = expected[leaf.start : leaf.stop]
        observed_defined = defined_numeric_mask(observed)
        wanted_defined = defined_numeric_mask(wanted)
        common = observed_defined & wanted_defined
        mask_mismatches = int(np.count_nonzero(observed_defined != wanted_defined))
        maximum_error = (
            float(np.max(np.abs(observed[common] - wanted[common])))
            if np.any(common)
            else 0.0
        )
        exact = (
            mask_mismatches == 0
            and np.allclose(
                observed[common],
                wanted[common],
                atol=1.0e-8,
                rtol=1.0e-10,
            )
        )
        exact_leaf_count += int(exact)
        if not exact:
            record = {
                "leaf": leaf.key,
                "mask_mismatch_count": mask_mismatches,
                "maximum_absolute_error": maximum_error,
            }
            if observed.size <= 16:
                record["observed"] = observed.tolist()
                record["teacher"] = wanted.tolist()
            records.append(record)
    records.sort(
        key=lambda item: (
            item["mask_mismatch_count"],
            item["maximum_absolute_error"],
        ),
        reverse=True,
    )
    return {
        "leaf_count": len(contract.fast_day_target_leaves),
        "exact_leaf_count": exact_leaf_count,
        "nonexact_leaf_count": len(records),
        "atol": 1.0e-8,
        "rtol": 1.0e-10,
        "top_nonexact_leaves": records[:25],
    }


def _flat_fast_target_report(
    observed: Any,
    sample: Mapping[str, Any],
    contract: Any,
) -> dict[str, Any]:
    from research.daily_coarse_graining.markov_dataset import defined_numeric_mask

    observed = np.asarray(observed, dtype=np.float64)
    expected = np.asarray(sample["fast_day_target"], dtype=np.float64)
    if observed.shape != expected.shape:
        raise ValueError(
            f"canonical internal fast-target shape drift: {observed.shape} != {expected.shape}"
        )
    records = []
    exact_leaf_count = 0
    for leaf in contract.fast_day_target_leaves:
        actual = observed[leaf.start : leaf.stop]
        wanted = expected[leaf.start : leaf.stop]
        actual_defined = defined_numeric_mask(actual)
        wanted_defined = defined_numeric_mask(wanted)
        common = actual_defined & wanted_defined
        mask_mismatches = int(np.count_nonzero(actual_defined != wanted_defined))
        maximum_error = (
            float(np.max(np.abs(actual[common] - wanted[common])))
            if np.any(common)
            else 0.0
        )
        exact = (
            mask_mismatches == 0
            and np.allclose(
                actual[common],
                wanted[common],
                atol=1.0e-8,
                rtol=1.0e-10,
            )
        )
        exact_leaf_count += int(exact)
        if not exact:
            record = {
                "leaf": leaf.key,
                "mask_mismatch_count": mask_mismatches,
                "maximum_absolute_error": maximum_error,
            }
            if actual.size <= 16:
                record["observed"] = actual.tolist()
                record["teacher"] = wanted.tolist()
            records.append(record)
    records.sort(
        key=lambda item: (
            item["mask_mismatch_count"],
            item["maximum_absolute_error"],
        ),
        reverse=True,
    )
    return {
        "width": int(observed.size),
        "leaf_count": len(contract.fast_day_target_leaves),
        "exact_leaf_count": exact_leaf_count,
        "nonexact_leaf_count": len(records),
        "atol": 1.0e-8,
        "rtol": 1.0e-10,
        "top_nonexact_leaves": records[:25],
    }


def _case_report(
    result: Any,
    sample: Mapping[str, Any],
    contract: Any,
) -> dict[str, Any]:
    from research.daily_coarse_graining.daily_markov_contract import (
        reconstruct_fast_day_target,
        reconstruct_state_fields,
    )
    from research.daily_coarse_graining.markov_dataset import defined_numeric_mask

    observed = np.asarray(result.next_state.other, dtype=np.float64)
    observed_defined = np.asarray(result.next_state_defined.other, dtype=np.bool_)
    expected = np.asarray(sample["next_state"], dtype=np.float64)
    expected_defined = defined_numeric_mask(expected)
    admitted = expected_defined & observed_defined
    difference = np.abs(observed - expected)
    discrete_mismatches = sum(
        int(
            np.count_nonzero(
                np.asarray(result.next_discrete_state[name])
                != np.asarray(sample["next_discrete_state"][name])
            )
        )
        for name in sample["next_discrete_state"]
    )
    water_residual = float(np.asarray(result.diagnostics["water_budget_residual"]))
    carbon_residual = float(np.asarray(result.diagnostics["carbon_budget_residual"]))
    mask_mismatches = int(np.count_nonzero(observed_defined != expected_defined))
    mask_mismatches_by_leaf = {}
    mask_mismatch_details_by_leaf = {}
    for leaf in contract.state_leaves:
        if leaf.discrete:
            continue
        observed_leaf = observed_defined[leaf.start : leaf.stop].reshape(leaf.shape)
        expected_leaf = expected_defined[leaf.start : leaf.stop].reshape(leaf.shape)
        teacher_defined_model_undefined = expected_leaf & ~observed_leaf
        teacher_undefined_model_defined = ~expected_leaf & observed_leaf
        leaf_mismatches = int(
            np.count_nonzero(
                teacher_defined_model_undefined | teacher_undefined_model_defined
            )
        )
        if leaf_mismatches == 0:
            continue
        mask_mismatches_by_leaf[leaf.key] = leaf_mismatches
        mask_mismatch_details_by_leaf[leaf.key] = {
            "shape": list(leaf.shape),
            "teacher_defined_model_undefined_count": int(
                np.count_nonzero(teacher_defined_model_undefined)
            ),
            "teacher_defined_model_undefined_indices": np.argwhere(
                teacher_defined_model_undefined
            ).tolist(),
            "teacher_undefined_model_defined_count": int(
                np.count_nonzero(teacher_undefined_model_defined)
            ),
            "teacher_undefined_model_defined_indices": np.argwhere(
                teacher_undefined_model_defined
            ).tolist(),
        }
    if sum(mask_mismatches_by_leaf.values()) != mask_mismatches:
        raise ValueError("defined-mask mismatch accounting drift")
    nonfinite_defined = int(
        np.count_nonzero(observed_defined & ~np.isfinite(observed))
    )
    water_minimum = float(np.min(np.asarray(result.fast_update.water.inventory)))
    carbon_minimum = float(np.min(np.asarray(result.fast_update.carbon.inventory)))
    start_fields = reconstruct_state_fields(
        np.asarray(sample["state"], dtype=np.float64),
        sample["discrete_state"],
        contract,
    )
    teacher_fast = reconstruct_fast_day_target(
        np.asarray(sample["fast_day_target"], dtype=np.float64),
        contract.fast_day_target_leaves,
        template_fields=start_fields,
    )
    active_pft_indices = np.asarray(contract.active_pft_indices, dtype=np.int64)
    teacher_gpp = np.take(
        np.asarray(teacher_fast.daily_fields["gpp_daily"], dtype=np.float64),
        active_pft_indices,
        axis=1,
    )
    teacher_maintenance = np.take(
        np.asarray(teacher_fast.daily_fields["resp_maint_part"], dtype=np.float64),
        active_pft_indices,
        axis=1,
    )
    predicted_gpp = np.asarray(
        result.prediction.named_outputs["daily.fast_gpp"], dtype=np.float64
    )
    predicted_maintenance = np.asarray(
        result.prediction.named_outputs["daily.fast_maintenance_respiration"],
        dtype=np.float64,
    )
    if predicted_gpp.shape != teacher_gpp.shape:
        raise ValueError("daily GPP lifecycle comparison shape drift")
    if predicted_maintenance.shape != teacher_maintenance.shape:
        raise ValueError("maintenance lifecycle comparison shape drift")
    vegetation_slots = np.flatnonzero(
        np.asarray(result.prediction.pft.vegetation_process_mask)
    )
    if vegetation_slots.size != 1:
        raise ValueError("frozen lifecycle case must have exactly one vegetation slot")
    vegetation_slot = int(vegetation_slots[0])
    carbon_driver = {
        "vegetation_compact_pft_index": vegetation_slot,
        "predicted_gpp": float(predicted_gpp[0, vegetation_slot]),
        "teacher_gpp": float(teacher_gpp[0, vegetation_slot]),
        "predicted_maintenance_respiration": float(
            np.sum(predicted_maintenance[0, vegetation_slot])
        ),
        "teacher_maintenance_respiration": float(
            np.sum(teacher_maintenance[0, vegetation_slot])
        ),
    }
    carbon_driver["predicted_gpp_minus_maintenance"] = (
        carbon_driver["predicted_gpp"]
        - carbon_driver["predicted_maintenance_respiration"]
    )
    carbon_driver["teacher_gpp_minus_maintenance"] = (
        carbon_driver["teacher_gpp"] - carbon_driver["teacher_maintenance_respiration"]
    )
    predicted_fast = result.prediction.retained_boundary.fast_state_fields
    exact_fast_errors = {}
    for component, field in EXACT_RETAINED_FAST_FIELDS:
        expected_fast = np.asarray(
            teacher_fast.fields_by_component[component][field],
            dtype=np.float64,
        )
        observed_fast = np.asarray(predicted_fast[component][field], dtype=np.float64)
        if observed_fast.shape != expected_fast.shape:
            raise ValueError(f"exact fast-field shape drift for {component}.{field}")
        exact_fast_errors[f"{component}.{field}"] = float(
            np.max(np.abs(observed_fast - expected_fast))
        )
    exact_fast_passed = all(
        np.allclose(
            np.asarray(predicted_fast[component][field]),
            np.asarray(teacher_fast.fields_by_component[component][field]),
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        for component, field in EXACT_RETAINED_FAST_FIELDS
    )
    passed = (
        observed.shape == expected.shape
        and nonfinite_defined == 0
        and mask_mismatches == 0
        and discrete_mismatches == 0
        and abs(water_residual) <= 2.0e-12
        and abs(carbon_residual) <= 2.0e-12
        and water_minimum >= -2.0e-12
        and carbon_minimum >= -2.0e-12
        and int(np.asarray(result.diagnostics["retained_tail_executed"])) == 1
        and exact_fast_passed
    )
    return {
        "status": "passed" if passed else "failed",
        "year": int(np.asarray(sample["year"])),
        "day_index": int(np.asarray(sample["day_index"])),
        "state_width": int(observed.size),
        "defined_count": int(np.count_nonzero(observed_defined)),
        "teacher_defined_count": int(np.count_nonzero(expected_defined)),
        "nonfinite_defined_count": nonfinite_defined,
        "defined_mask_mismatch_count": mask_mismatches,
        "defined_mask_mismatches_by_leaf": mask_mismatches_by_leaf,
        "defined_mask_mismatch_details_by_leaf": mask_mismatch_details_by_leaf,
        "discrete_state_mismatch_count": discrete_mismatches,
        "teacher_state_max_absolute_error": (
            float(np.max(difference[admitted])) if np.any(admitted) else 0.0
        ),
        "teacher_state_error_is_acceptance_gate": False,
        "water_budget_residual": water_residual,
        "carbon_budget_residual": carbon_residual,
        "water_minimum_inventory": water_minimum,
        "carbon_minimum_inventory": carbon_minimum,
        "retained_tail_executed": True,
        "exact_retained_fast_fields": {
            "status": "passed" if exact_fast_passed else "failed",
            "atol": 1.0e-12,
            "rtol": 1.0e-12,
            "maximum_absolute_errors": exact_fast_errors,
        },
        "fast_carbon_driver": carbon_driver,
        "retained_boundary_vs_teacher": _retained_boundary_report(
            result.prediction,
            sample,
            contract,
        ),
    }


def _evaluate_lifecycle_parameters(
    runtime: Mapping[str, Any],
    parameter_sets: Mapping[str, Any],
    *,
    probe_nonnegative_scale: bool = False,
) -> dict[str, Any]:
    from jax_orchidee.driver import orchestration as teacher
    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator
    from research.daily_coarse_graining.daily_operator.successor_architecture import (
        SOURCE_ROUTED_ATTENTION,
        SOURCE_ROUTED_GRAPH_CONTROL,
    )

    jax = runtime["jax"]
    paths = runtime["context_paths"]
    teacher_context = teacher.prepare_paper_1961_driver_context(
        paths["config"],
        used_run_def_path=paths["run_def"],
        reference_run_dir=paths["reference_run_dir"],
    )
    ordinary_parent = runtime["ordinary_parent"]
    restart_parent = runtime["restart_parent"]
    ordinary_samples = [ordinary_parent.sample(row) for row in ANCHOR_ROWS]
    restart_sample = restart_parent.sample(0)
    ordinary_inputs = [runtime["examples"][row].inputs for row in ANCHOR_ROWS]
    restart_input = runtime["adapter"].assemble_input(restart_sample)
    later_tail = _retained_adapter(
        runtime,
        teacher_context,
        ordinary_samples[0],
        lifecycle_start_tstep=48,
        result_day_index=2,
    )
    restart_tail = _retained_adapter(
        runtime,
        teacher_context,
        restart_sample,
        lifecycle_start_tstep=0,
        result_day_index=1,
    )
    arms = {}
    for arm in (SOURCE_ROUTED_ATTENTION, SOURCE_ROUTED_GRAPH_CONTROL):
        head = _head(runtime, arm)
        if probe_nonnegative_scale:
            head = _NonnegativeScaleProbeHead(head, runtime["schema"])
        later = bind_daily_operator(
            input_assembler=runtime["adapter"].input_assembler,
            process_head=head,
            retained_tail=later_tail,
        )
        restart = bind_daily_operator(
            input_assembler=runtime["adapter"].input_assembler,
            process_head=head,
            retained_tail=restart_tail,
        )
        compiled_later = jax.jit(later)
        compiled_restart = jax.jit(restart)
        evaluations = {}
        for name, parameters in parameter_sets.items():
            started = time.perf_counter()
            cold_result = compiled_later(parameters, ordinary_inputs[0])
            jax.block_until_ready(cold_result)
            ordinary_result = compiled_later(parameters, ordinary_inputs[1])
            jax.block_until_ready(ordinary_result)
            restart_result = compiled_restart(parameters, restart_input)
            jax.block_until_ready(restart_result)
            cases = {
                "cold_start_continuation_day2": _case_report(
                    cold_result,
                    ordinary_samples[0],
                    runtime["contract"],
                ),
                "ordinary_later_day3": _case_report(
                    ordinary_result,
                    ordinary_samples[1],
                    runtime["contract"],
                ),
                "restart_year_boundary_1962_day1": _case_report(
                    restart_result,
                    restart_sample,
                    runtime["contract"],
                ),
            }
            passed = all(item["status"] == "passed" for item in cases.values())
            evaluations[name] = {
                "status": "passed" if passed else "failed",
                "seconds": time.perf_counter() - started,
                "cases": cases,
            }
        arms[arm] = evaluations
    return arms


def _scaled_final_decoder(parameters: Mapping[str, Any], scale: float) -> dict[str, Any]:
    result = dict(parameters)
    decoder = dict(parameters["decoder_second"])
    decoder["kernel"] = decoder["kernel"] * float(scale)
    result["decoder_second"] = decoder
    return result


def _lifecycle_stage(runtime: Mapping[str, Any]) -> dict[str, Any]:
    jax = runtime["jax"]
    evaluations = _evaluate_lifecycle_parameters(
        runtime,
        {"declared_initialization": runtime["parameters"]},
    )
    arms = {
        arm: results["declared_initialization"]
        for arm, results in evaluations.items()
    }
    passed = all(item["status"] == "passed" for item in arms.values())
    return {
        "schema_version": "successor_process_head_real_v6_canonical_lifecycle_v1",
        "status": "passed" if passed else "failed",
        "promotion_authorized": False,
        "checkpoint_written": False,
        "historical_v5_used": False,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in runtime["devices"]],
        "successor_contract_sha256": EXPECTED["successor_contract"],
        "data": {
            "dataset_manifest_sha256": EXPECTED["dataset_manifest"],
            "process_statistics_json_sha256": EXPECTED["process_statistics_json"],
            "landpoint_id": FROZEN_LANDPOINT,
            "ordinary_year": 1961,
            "ordinary_rows": list(ANCHOR_ROWS),
            "restart_year": 1962,
            "restart_row": 0,
            "generation_plan_sha256": runtime["manifest"]["plan_sha256"],
        },
        "canonical_boundary": {
            "binding": "bind_daily_operator(canonical_assembler, successor_head, canonical_retained_tail_adapter)",
            "modelout_diagnostics_omitted_for_training": True,
            "native_forcing_only": True,
            "reconstructs_48_forcing_records": False,
            "executes_48_step_state_scan": False,
        },
        "arms": arms,
        "pending_required_gate": "real_30_day_rollout_and_train_train_gradient_ratio_preflight",
    }


def _initialization_stage(runtime: Mapping[str, Any]) -> dict[str, Any]:
    jax = runtime["jax"]
    names_by_scale = {
        scale: f"decoder_kernel_scale_{scale:g}"
        for scale in INITIALIZATION_DECODER_SCALES
    }
    parameter_sets = {
        names_by_scale[scale]: _scaled_final_decoder(runtime["parameters"], scale)
        for scale in INITIALIZATION_DECODER_SCALES
    }
    arms = _evaluate_lifecycle_parameters(runtime, parameter_sets)
    passing_scales = [
        scale
        for scale in INITIALIZATION_DECODER_SCALES
        if all(
            results[names_by_scale[scale]]["status"] == "passed"
            for results in arms.values()
        )
    ]
    positive_passing_scales = [scale for scale in passing_scales if scale > 0.0]
    passed = bool(positive_passing_scales)
    return {
        "schema_version": "successor_process_head_real_v6_initialization_probe_v1",
        "status": "passed" if passed else "failed",
        "promotion_authorized": False,
        "checkpoint_written": False,
        "historical_v5_used": False,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in runtime["devices"]],
        "successor_contract_sha256": EXPECTED["successor_contract"],
        "data": {
            "dataset_manifest_sha256": EXPECTED["dataset_manifest"],
            "process_statistics_json_sha256": EXPECTED["process_statistics_json"],
            "landpoint_id": FROZEN_LANDPOINT,
            "ordinary_year": 1961,
            "ordinary_rows": list(ANCHOR_ROWS),
            "restart_year": 1962,
            "restart_row": 0,
        },
        "probe": {
            "parameter_changed": "decoder_second.kernel_initial_scale",
            "bias_changed": False,
            "parameter_count_changed": False,
            "scales": list(INITIALIZATION_DECODER_SCALES),
            "passing_scales": passing_scales,
            "largest_positive_passing_scale": (
                max(positive_passing_scales) if positive_passing_scales else None
            ),
            "training_updates": 0,
        },
        "canonical_boundary": {
            "binding": "bind_daily_operator(canonical_assembler, successor_head, canonical_retained_tail_adapter)",
            "native_forcing_only": True,
            "reconstructs_48_forcing_records": False,
            "executes_48_step_state_scan": False,
        },
        "arms": arms,
        "pending_required_gate": "freeze_stable_initialization_then_repeat_strict_lifecycle",
    }


def _baseline_stage(runtime: Mapping[str, Any]) -> dict[str, Any]:
    jax = runtime["jax"]
    names_by_scale = {
        scale: f"nonnegative_baseline_scale_{scale:g}"
        for scale in INITIALIZATION_NONNEGATIVE_BASELINE_SCALES
    }
    model = _scaled_final_decoder(runtime["parameters"], 0.01)
    parameter_sets = {
        names_by_scale[scale]: {
            "model": model,
            "nonnegative_scale": np.asarray(scale, dtype=np.float64),
        }
        for scale in INITIALIZATION_NONNEGATIVE_BASELINE_SCALES
    }
    arms = _evaluate_lifecycle_parameters(
        runtime,
        parameter_sets,
        probe_nonnegative_scale=True,
    )
    passing_scales = [
        scale
        for scale in INITIALIZATION_NONNEGATIVE_BASELINE_SCALES
        if all(
            results[names_by_scale[scale]]["status"] == "passed"
            for results in arms.values()
        )
    ]
    passed = bool(passing_scales)
    return {
        "schema_version": "successor_process_head_real_v6_nonnegative_baseline_probe_v1",
        "status": "passed" if passed else "failed",
        "promotion_authorized": False,
        "checkpoint_written": False,
        "historical_v5_used": False,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in runtime["devices"]],
        "successor_contract_sha256": EXPECTED["successor_contract"],
        "data": {
            "dataset_manifest_sha256": EXPECTED["dataset_manifest"],
            "process_statistics_json_sha256": EXPECTED["process_statistics_json"],
            "landpoint_id": FROZEN_LANDPOINT,
            "ordinary_year": 1961,
            "ordinary_rows": list(ANCHOR_ROWS),
            "restart_year": 1962,
            "restart_row": 0,
        },
        "probe": {
            "parameter_changed": "none",
            "diagnostic_transform": "scale_all_nonnegative_named_outputs",
            "candidate_decoder_kernel_scale": 0.01,
            "scales": list(INITIALIZATION_NONNEGATIVE_BASELINE_SCALES),
            "passing_scales": passing_scales,
            "largest_passing_scale": max(passing_scales) if passing_scales else None,
            "training_updates": 0,
        },
        "canonical_boundary": {
            "binding": "bind_daily_operator(canonical_assembler, successor_head, canonical_retained_tail_adapter)",
            "native_forcing_only": True,
            "reconstructs_48_forcing_records": False,
            "executes_48_step_state_scan": False,
        },
        "arms": arms,
        "pending_required_gate": "freeze_coherent_nonnegative_baseline_then_repeat_strict_lifecycle",
    }


def _carbon_baseline_stage(runtime: Mapping[str, Any]) -> dict[str, Any]:
    jax = runtime["jax"]
    names_by_scale = {
        scale: f"coherent_carbon_baseline_scale_{scale:g}"
        for scale in INITIALIZATION_COHERENT_CARBON_SCALES
    }
    model = _scaled_final_decoder(runtime["parameters"], 0.01)
    parameter_sets = {
        names_by_scale[scale]: {
            "model": model,
            "nonnegative_scale": np.asarray(scale, dtype=np.float64),
            "gpp_scale": np.asarray(scale, dtype=np.float64),
            "maintenance_scale": np.asarray(scale / 12.0, dtype=np.float64),
        }
        for scale in INITIALIZATION_COHERENT_CARBON_SCALES
    }
    arms = _evaluate_lifecycle_parameters(
        runtime,
        parameter_sets,
        probe_nonnegative_scale=True,
    )
    passing_scales = [
        scale
        for scale in INITIALIZATION_COHERENT_CARBON_SCALES
        if all(
            results[names_by_scale[scale]]["status"] == "passed"
            for results in arms.values()
        )
    ]
    passed = bool(passing_scales)
    return {
        "schema_version": "successor_process_head_real_v6_coherent_carbon_baseline_probe_v1",
        "status": "passed" if passed else "failed",
        "promotion_authorized": False,
        "checkpoint_written": False,
        "historical_v5_used": False,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in runtime["devices"]],
        "successor_contract_sha256": EXPECTED["successor_contract"],
        "data": {
            "dataset_manifest_sha256": EXPECTED["dataset_manifest"],
            "process_statistics_json_sha256": EXPECTED["process_statistics_json"],
            "landpoint_id": FROZEN_LANDPOINT,
            "ordinary_year": 1961,
            "ordinary_rows": list(ANCHOR_ROWS),
            "restart_year": 1962,
            "restart_row": 0,
        },
        "probe": {
            "parameter_changed": "none",
            "diagnostic_transform": "scale_nonnegative_outputs_with_part_aware_maintenance_baseline",
            "candidate_decoder_kernel_scale": 0.01,
            "maintenance_part_count": 12,
            "maintenance_scale_rule": "generic_nonnegative_scale / maintenance_part_count",
            "scales": list(INITIALIZATION_COHERENT_CARBON_SCALES),
            "passing_scales": passing_scales,
            "largest_passing_scale": max(passing_scales) if passing_scales else None,
            "training_updates": 0,
        },
        "canonical_boundary": {
            "binding": "bind_daily_operator(canonical_assembler, successor_head, canonical_retained_tail_adapter)",
            "native_forcing_only": True,
            "reconstructs_48_forcing_records": False,
            "executes_48_step_state_scan": False,
        },
        "arms": arms,
        "pending_required_gate": "freeze_part_aware_baseline_then_repeat_strict_lifecycle",
    }


def _all_teacher_fast_owner_outputs(
    example: Any,
    sample: Mapping[str, Any],
    owner_contract: Any,
) -> dict[str, Any]:
    from research.daily_coarse_graining.markov_dataset import defined_numeric_mask

    outputs = {
        name: np.asarray(target.value, dtype=np.float64)
        for name, target in example.targets.supervision.items()
    }
    target = np.asarray(sample["fast_day_target"], dtype=np.float64)
    for leaf in owner_contract.leaves:
        if leaf.output_key is None:
            continue
        value = target[leaf.start : leaf.stop].reshape(leaf.shape)
        outputs[leaf.output_key] = value
        if leaf.defined_output_key is not None:
            outputs[leaf.defined_output_key] = np.where(
                defined_numeric_mask(value),
                1.0,
                -1.0,
            )
    return outputs


def _fast_owner_stage(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Prove the non-neural materializer closes all 160 formal v6 leaves."""

    from jax_orchidee.driver import orchestration as teacher
    from research.daily_coarse_graining.daily_operator.fast_owner_reconstruction import (
        build_successor_fast_owner_materializer,
    )
    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator
    from research.daily_coarse_graining.daily_operator.retained_tail import (
        retained_tail_input,
    )
    from research.daily_coarse_graining.daily_operator.successor_architecture import (
        SOURCE_ROUTED_ATTENTION,
    )

    jax = runtime["jax"]
    paths = runtime["context_paths"]
    teacher_context = teacher.prepare_paper_1961_driver_context(
        paths["config"],
        used_run_def_path=paths["run_def"],
        reference_run_dir=paths["reference_run_dir"],
    )
    materializer = build_successor_fast_owner_materializer(
        runtime["contract"],
        forcing_layout=runtime["forcing_layout"],
    )
    if materializer.owner_contract.contract_sha256 != EXPECTED["fast_owner_contract"]:
        raise ValueError("fast-owner materializer contract identity drift")

    ordinary_samples = [
        runtime["ordinary_parent"].sample(row) for row in ANCHOR_ROWS
    ]
    restart_sample = runtime["restart_parent"].sample(0)
    restart_example = runtime["adapter"].assemble(
        restart_sample,
        supplemental_values={
            name: value[0]
            for name, value in runtime["restart_typed_values"].items()
        },
        supplemental_defined={
            name: value[0]
            for name, value in runtime["restart_typed_defined"].items()
        },
    )
    case_inputs = (
        (
            "cold_start_continuation_day2",
            ordinary_samples[0],
            runtime["examples"][ANCHOR_ROWS[0]],
            48,
            2,
        ),
        (
            "ordinary_later_day3",
            ordinary_samples[1],
            runtime["examples"][ANCHOR_ROWS[1]],
            96,
            3,
        ),
        (
            "restart_year_boundary_1962_day1",
            restart_sample,
            restart_example,
            0,
            1,
        ),
    )
    base_head = _head(runtime, SOURCE_ROUTED_ATTENTION)
    head = _AllTeacherFastOwnerHead(base_head)
    cases = {}
    for name, sample, example, lifecycle_start_tstep, result_day_index in case_inputs:
        retained_tail = _retained_adapter(
            runtime,
            teacher_context,
            sample,
            lifecycle_start_tstep=lifecycle_start_tstep,
            result_day_index=result_day_index,
            fast_owner_materializer=materializer,
        )
        transition = bind_daily_operator(
            input_assembler=runtime["adapter"].input_assembler,
            process_head=head,
            retained_tail=retained_tail,
        )
        started = time.perf_counter()
        parameters = {
            "named_outputs": _all_teacher_fast_owner_outputs(
                example,
                sample,
                materializer.owner_contract,
            )
        }
        result = jax.jit(transition)(parameters, example.inputs)
        jax.block_until_ready(result)
        assembled = runtime["adapter"].input_assembler.assemble(example.inputs)
        internal_target = retained_tail._typed_fast_target(
            retained_tail_input(
                assembled,
                result.prediction,
                result.fast_update,
            )
        )
        flat = _flat_fast_target_report(
            internal_target,
            sample,
            runtime["contract"],
        )
        lifecycle = _case_report(result, sample, runtime["contract"])
        rveget = next(
            leaf
            for leaf in runtime["contract"].fast_day_target_leaves
            if leaf.key == "diffuco_previous_step_state.rveget"
        )
        from research.daily_coarse_graining.markov_dataset import defined_numeric_mask

        observed_rveget = np.asarray(internal_target)[rveget.start : rveget.stop]
        teacher_rveget = np.asarray(sample["fast_day_target"])[
            rveget.start : rveget.stop
        ]
        observed_rveget_defined = defined_numeric_mask(observed_rveget)
        teacher_rveget_defined = defined_numeric_mask(teacher_rveget)
        common_rveget = observed_rveget_defined & teacher_rveget_defined
        rveget_mask_mismatches = int(
            np.count_nonzero(observed_rveget_defined != teacher_rveget_defined)
        )
        rveget_exact = (
            rveget_mask_mismatches == 0
            and np.allclose(
                observed_rveget[common_rveget],
                teacher_rveget[common_rveget],
                atol=1.0e-8,
                rtol=1.0e-10,
            )
        )
        rveget_report = {
            "status": "passed" if rveget_exact else "failed",
            "mask_mismatch_count": rveget_mask_mismatches,
            "teacher_defined": teacher_rveget_defined.tolist(),
            "observed_defined": observed_rveget_defined.tolist(),
        }
        passed = (
            flat["exact_leaf_count"] == 160
            and flat["nonexact_leaf_count"] == 0
            and lifecycle["status"] == "passed"
            and lifecycle["defined_mask_mismatch_count"] == 0
            and lifecycle["discrete_state_mismatch_count"] == 0
            and rveget_exact
        )
        cases[name] = {
            "status": "passed" if passed else "failed",
            "seconds": time.perf_counter() - started,
            "fast_target": flat,
            "rveget_defined_status": rveget_report,
            "lifecycle": lifecycle,
        }
    passed = all(case["status"] == "passed" for case in cases.values())
    return {
        "schema_version": "successor_fast_owner_all_teacher_gate_v1",
        "status": "passed" if passed else "failed",
        "promotion_authorized": False,
        "checkpoint_written": False,
        "training_updates": 0,
        "historical_v5_used": False,
        "teacher_data_regenerated": False,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in runtime["devices"]],
        "fast_owner_contract_sha256": EXPECTED["fast_owner_contract"],
        "data": {
            "dataset_manifest_sha256": EXPECTED["dataset_manifest"],
            "markov_contract_sha256": runtime["contract"].sha256,
            "landpoint_id": FROZEN_LANDPOINT,
            "ordinary_year": 1961,
            "ordinary_rows": list(ANCHOR_ROWS),
            "restart_year": 1962,
            "restart_row": 0,
        },
        "canonical_boundary": {
            "binding": "bind_daily_operator(canonical_assembler, all_teacher_process_head, canonical_retained_tail_adapter)",
            "native_forcing_only": True,
            "reconstructs_48_forcing_records": False,
            "executes_48_step_state_scan": False,
            "neural_decoder_invoked": False,
        },
        "acceptance": {
            "required_exact_fast_leaves": 160,
            "atol": 1.0e-8,
            "rtol": 1.0e-10,
            "exact_masks_and_discrete_required": True,
        },
        "cases": cases,
        "pending_required_gate": "architecture_review_then_identical_candidate_control_decoder_extension",
    }


def _attribution_stage(runtime: Mapping[str, Any]) -> dict[str, Any]:
    from jax_orchidee.driver import orchestration as teacher
    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator
    from research.daily_coarse_graining.daily_operator.retained_tail import (
        retained_tail_input,
    )
    from research.daily_coarse_graining.daily_operator.successor_architecture import (
        SOURCE_ROUTED_ATTENTION,
    )

    jax = runtime["jax"]
    paths = runtime["context_paths"]
    teacher_context = teacher.prepare_paper_1961_driver_context(
        paths["config"],
        used_run_def_path=paths["run_def"],
        reference_run_dir=paths["reference_run_dir"],
    )
    sample = runtime["ordinary_parent"].sample(ANCHOR_ROWS[0])
    inputs = runtime["examples"][ANCHOR_ROWS[0]].inputs
    assembled = runtime["adapter"].input_assembler.assemble(inputs)
    retained_tail = _retained_adapter(
        runtime,
        teacher_context,
        sample,
        lifecycle_start_tstep=48,
        result_day_index=2,
    )
    base_head = _head(runtime, SOURCE_ROUTED_ATTENTION)
    head = _TeacherReplacementProbeHead(base_head, runtime["schema"])
    transition = bind_daily_operator(
        input_assembler=runtime["adapter"].input_assembler,
        process_head=head,
        retained_tail=retained_tail,
    )
    compiled = jax.jit(transition)
    field_names = tuple(field.name for field in runtime["schema"].fields)
    teacher_targets = runtime["examples"][ANCHOR_ROWS[0]].targets.supervision
    if set(teacher_targets) != set(field_names):
        raise ValueError("attribution Teacher target registry drift")
    teacher_outputs = {
        name: np.asarray(teacher_targets[name].value, dtype=np.float64)
        for name in field_names
    }

    groups = {
        "daily_carbon": {
            "daily.fast_gpp",
            "daily.fast_maintenance_respiration",
        },
        "daily_temperature": {
            "daily.temperature_context.tsoil",
            "daily.temperature_context.tsurf",
        },
        "water": {
            name
            for name in field_names
            if name.startswith(("inventory.water.", "typed.water."))
        },
        "carbon": {
            name
            for name in field_names
            if name.startswith(("inventory.carbon.", "typed.ok_leak."))
        },
        "thermal_energy": {
            name
            for name in field_names
            if name.startswith(("thermal.", "typed.energy."))
        },
    }
    assigned = set().union(*groups.values())
    if assigned != set(field_names):
        raise ValueError(
            f"attribution group registry drift: {sorted(set(field_names) - assigned)}"
        )
    index = {name: position for position, name in enumerate(field_names)}

    def replacement_mask(names: set[str]) -> np.ndarray:
        mask = np.zeros((len(field_names),), dtype=np.bool_)
        mask[[index[name] for name in names]] = True
        return mask

    modes = {
        "all_predicted": set(),
        "all_teacher": set(field_names),
        **{f"predicted_plus_teacher_{name}": values for name, values in groups.items()},
        **{
            f"teacher_except_{name}": set(field_names) - values
            for name, values in groups.items()
        },
        **{
            f"predicted_plus_teacher_{name.removeprefix('daily.')}": {name}
            for name in groups["daily_carbon"] | groups["daily_temperature"]
        },
    }
    cases = {}
    for name, replacements in modes.items():
        started = time.perf_counter()
        parameters = {
            "model": runtime["parameters"],
            "teacher_outputs": teacher_outputs,
            "replacement_mask": replacement_mask(replacements),
        }
        result = compiled(parameters, inputs)
        jax.block_until_ready(result)
        report = _case_report(result, sample, runtime["contract"])
        internal_target = retained_tail._typed_fast_target(
            retained_tail_input(
                assembled,
                result.prediction,
                result.fast_update,
            )
        )
        report["canonical_internal_fast_target_vs_teacher"] = (
            _flat_fast_target_report(
                internal_target,
                sample,
                runtime["contract"],
            )
        )
        cases[name] = {
            "status": report["status"],
            "seconds": time.perf_counter() - started,
            "teacher_field_count": len(replacements),
            "case": report,
        }
    diagnostic_passed = (
        cases["all_predicted"]["status"] == "failed"
        and cases["all_teacher"]["status"] == "passed"
    )
    return {
        "schema_version": "successor_process_head_real_v6_day2_owner_attribution_v1",
        "status": "passed" if diagnostic_passed else "failed",
        "promotion_authorized": False,
        "checkpoint_written": False,
        "historical_v5_used": False,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in runtime["devices"]],
        "successor_contract_sha256": EXPECTED["successor_contract"],
        "data": {
            "dataset_manifest_sha256": EXPECTED["dataset_manifest"],
            "process_statistics_json_sha256": EXPECTED["process_statistics_json"],
            "landpoint_id": FROZEN_LANDPOINT,
            "year": 1961,
            "row": ANCHOR_ROWS[0],
            "day_index": int(np.asarray(sample["day_index"])),
        },
        "probe": {
            "arm": SOURCE_ROUTED_ATTENTION,
            "field_count": len(field_names),
            "groups": {name: sorted(values) for name, values in groups.items()},
            "training_updates": 0,
        },
        "canonical_boundary": {
            "binding": "bind_daily_operator(canonical_assembler, hybrid_named_outputs, canonical_retained_tail_adapter)",
            "native_forcing_only": True,
            "reconstructs_48_forcing_records": False,
            "executes_48_step_state_scan": False,
        },
        "cases": cases,
        "pending_required_gate": "identify_minimal_restoring_owner_then_repair_initialization_or_boundary",
    }


def _rollout_stage(runtime: Mapping[str, Any]) -> dict[str, Any]:
    import jax.numpy as jnp

    from jax_orchidee.driver import orchestration as teacher
    from research.daily_coarse_graining.daily_operator.long_horizon_objective import (
        long_horizon_biomass_productivity_objective,
        rollout_hard_gate_counts,
    )
    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator
    from research.daily_coarse_graining.daily_operator.successor_architecture import (
        SOURCE_ROUTED_ATTENTION,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        LONG_HORIZON_COMPONENTS,
        TERMINAL_STOCK_FIELDS,
        calibrate_gradient_ratio_coefficients,
        derive_long_horizon_scales,
        extract_biomass_productivity_trajectories,
        rollout_daily_operator,
        trajectories_from_rollout,
    )
    from research.daily_coarse_graining.daily_operator.training_loss import (
        masked_equal_label_macro_mse,
    )
    from research.daily_coarse_graining.markov_dataset import defined_numeric_mask

    jax = runtime["jax"]
    paths = runtime["context_paths"]
    teacher_context = teacher.prepare_paper_1961_driver_context(
        paths["config"],
        used_run_def_path=paths["run_def"],
        reference_run_dir=paths["reference_run_dir"],
    )
    initial_sample = runtime["ordinary_parent"].sample(ROLLOUT_START_ROW)
    retained_tail = _retained_adapter(
        runtime,
        teacher_context,
        initial_sample,
        lifecycle_start_tstep=48,
        result_day_index=2,
    )
    head = _head(runtime, SOURCE_ROUTED_ATTENTION)
    transition = bind_daily_operator(
        input_assembler=runtime["adapter"].input_assembler,
        process_head=head,
        retained_tail=retained_tail,
    )
    templates = _stack_trees(jax, [item.inputs for item in runtime["examples"]])

    forward_started = time.perf_counter()
    forward_rollout = jax.jit(
        lambda parameters: rollout_daily_operator(
            parameters,
            templates,
            transition=transition,
            contract=runtime["contract"],
            rematerialize=False,
        )
    )(runtime["parameters"])
    jax.block_until_ready(forward_rollout)
    forward_seconds = time.perf_counter() - forward_started
    predicted = trajectories_from_rollout(forward_rollout, runtime["contract"])

    parent = runtime["ordinary_parent"]
    start = ROLLOUT_START_ROW
    stop = start + ROLLOUT_DAYS
    expected_continuous = np.asarray(parent.state_trajectory[start + 1 : stop + 1])
    expected_defined = defined_numeric_mask(expected_continuous)
    expected_gpp = np.stack(
        [
            np.asarray(item.targets.supervision["daily.fast_gpp"].value)
            for item in runtime["examples"]
        ]
    )
    expected_gpp_defined = np.stack(
        [
            np.asarray(item.targets.supervision["daily.fast_gpp"].defined)
            for item in runtime["examples"]
        ]
    )
    expected = extract_biomass_productivity_trajectories(
        expected_continuous,
        expected_defined,
        daily_fast_gpp=expected_gpp,
        daily_fast_gpp_defined=expected_gpp_defined,
        pft_fraction=templates.pft.pft_fraction,
        active_pft_mask=templates.pft.active_pft_mask,
        contract=runtime["contract"],
    )
    scales = derive_long_horizon_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    unit_coefficients = {name: 1.0 for name in LONG_HORIZON_COMPONENTS}
    forward_objective = long_horizon_biomass_productivity_objective(
        predicted.fluxes,
        expected.fluxes,
        predicted.stocks,
        expected.stocks,
        flux_scales=scales.fluxes,
        stock_scales=scales.stocks,
        coefficients=unit_coefficients,
    )
    expected_discrete = {
        name: value[start + 1 : stop + 1]
        for name, value in parent.discrete_trajectories.items()
    }
    hard_counts = rollout_hard_gate_counts(
        predicted.stocks,
        expected.stocks,
        nonnegative_stock_fields=TERMINAL_STOCK_FIELDS,
        budget_residuals={
            "water": forward_rollout.result.diagnostics["water_budget_residual"],
            "carbon": forward_rollout.result.diagnostics["carbon_budget_residual"],
        },
        predicted_discrete_state=forward_rollout.result.next_discrete_state,
        expected_discrete_state=expected_discrete,
    )
    hard_counts_host = {
        name: int(np.asarray(value)) for name, value in hard_counts._asdict().items()
    }

    anchor_inputs = _stack_trees(
        jax,
        [runtime["assembled"][row] for row in ANCHOR_ROWS],
    )
    anchor_targets = _stack_trees(
        jax,
        [runtime["examples"][row].targets.supervision for row in ANCHOR_ROWS],
    )

    def one_day_anchor(parameters):
        def one(inputs, targets):
            context = head.assemble_context(parameters, inputs)
            outputs = head.named_outputs(parameters, context, inputs)
            return masked_equal_label_macro_mse(
                outputs,
                targets,
                scales=runtime["process_statistics"].loss_scales,
                field_labels=runtime["process_statistics"].field_labels,
            ).total

        return jnp.mean(jax.vmap(one)(anchor_inputs, anchor_targets))

    def component_vector(parameters):
        trace = rollout_daily_operator(
            parameters,
            templates,
            transition=transition,
            contract=runtime["contract"],
            rematerialize=True,
        )
        current = trajectories_from_rollout(trace, runtime["contract"])
        objective = long_horizon_biomass_productivity_objective(
            current.fluxes,
            expected.fluxes,
            current.stocks,
            expected.stocks,
            flux_scales=scales.fluxes,
            stock_scales=scales.stocks,
            coefficients=unit_coefficients,
        )
        return jnp.stack(
            tuple(getattr(objective, name) for name in LONG_HORIZON_COMPONENTS)
        )

    component_losses = {
        name: (lambda parameters, index=index: component_vector(parameters)[index])
        for index, name in enumerate(LONG_HORIZON_COMPONENTS)
    }
    preflight_started = time.perf_counter()
    preflight = calibrate_gradient_ratio_coefficients(
        runtime["parameters"],
        one_day_anchor,
        component_losses,
        target_gradient_ratios=TARGET_GRADIENT_RATIOS,
        coefficient_minimum=0.001,
        coefficient_maximum=10.0,
        split_label="train/train",
        compile_gradients=True,
        combined_component_loss=component_vector,
    )
    preflight_seconds = time.perf_counter() - preflight_started
    finite_objective = all(
        np.isfinite(float(np.asarray(getattr(forward_objective, name))))
        for name in LONG_HORIZON_COMPONENTS
    )
    hard_passed = all(value == 0 for value in hard_counts_host.values())
    coefficients_interior = all(
        0.001 < value < 10.0
        for value in preflight.resolved_coefficients.values()
    )
    passed = finite_objective and hard_passed and coefficients_interior
    return {
        "schema_version": "successor_process_head_real_v6_30_day_preflight_v1",
        "status": "passed" if passed else "failed",
        "promotion_authorized": False,
        "checkpoint_written": False,
        "historical_v5_used": False,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in runtime["devices"]],
        "successor_contract_sha256": EXPECTED["successor_contract"],
        "data": {
            "dataset_manifest_sha256": EXPECTED["dataset_manifest"],
            "base_statistics_json_sha256": EXPECTED["base_statistics_json"],
            "process_statistics_json_sha256": EXPECTED["process_statistics_json"],
            "split": "train/train",
            "landpoint_id": FROZEN_LANDPOINT,
            "year": ROLLOUT_YEAR,
            "start_row": ROLLOUT_START_ROW,
            "day_indices": np.asarray(parent.day_index[start:stop]),
            "one_day_anchor_rows": list(ANCHOR_ROWS),
            "future_teacher_state_consumed_by_rollout": False,
        },
        "rollout": {
            "days": ROLLOUT_DAYS,
            "rematerialized_for_gradient": True,
            "forward_seconds": forward_seconds,
            "preflight_seconds": preflight_seconds,
            "unit_coefficient_components": {
                name: getattr(forward_objective, name)
                for name in LONG_HORIZON_COMPONENTS
            },
            "hard_gate_counts": hard_counts_host,
        },
        "normalization": {
            "policy": "root_sum_square_propagation_of_accepted_train_train_column_scales",
            "cross_column_covariance_assumed": False,
            "v6_rescan_performed": False,
            "flux_scales": scales.fluxes,
            "stock_scales": scales.stocks,
        },
        "gradient_ratio_preflight": {
            "split": preflight.split_label,
            "one_day_anchor_loss": preflight.anchor_loss,
            "one_day_anchor_gradient_norm": preflight.anchor_gradient_norm,
            "component_losses": preflight.component_losses,
            "component_gradient_norms": preflight.component_gradient_norms,
            "target_gradient_ratios": preflight.target_gradient_ratios,
            "resolved_coefficients": preflight.resolved_coefficients,
            "achieved_gradient_ratios": preflight.achieved_gradient_ratios,
            "coefficients_strictly_inside_bounds": coefficients_interior,
            "combined_gradient_nominal_ratio": sum(
                preflight.achieved_gradient_ratios.values()
            ),
        },
        "canonical_boundary": {
            "binding": "bind_daily_operator(canonical_assembler, successor_head, canonical_retained_tail_adapter)",
            "native_forcing_only": True,
            "reconstructs_48_forcing_records": False,
            "executes_48_step_state_scan": False,
            "daily_operator_steps": ROLLOUT_DAYS,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "attribution",
            "baseline",
            "carbon_baseline",
            "fast_owner",
            "initialization",
            "lifecycle",
            "rollout",
        ),
        required=True,
    )
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime = _load_runtime(args)
    stages = {
        "attribution": _attribution_stage,
        "baseline": _baseline_stage,
        "carbon_baseline": _carbon_baseline_stage,
        "fast_owner": _fast_owner_stage,
        "initialization": _initialization_stage,
        "lifecycle": _lifecycle_stage,
        "rollout": _rollout_stage,
    }
    report = stages[args.stage](runtime)
    _atomic_json(args.output.resolve(), report)
    print(json.dumps(_json_value(report), indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
