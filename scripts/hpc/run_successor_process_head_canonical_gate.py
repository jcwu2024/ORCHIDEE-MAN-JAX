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
}
FROZEN_LANDPOINT = "001.0-071.0"
ROLLOUT_YEAR = 1961
ROLLOUT_START_ROW = 0
ROLLOUT_DAYS = 30
ANCHOR_ROWS = (0, 1)
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


def _load_runtime(args: argparse.Namespace) -> dict[str, Any]:
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

    jax, devices = initialize_gpu_backend()
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
    if not np.array_equal(typed_days, ordinary_parent.day_index):
        raise ValueError("frozen typed and parent day indices differ")
    if tuple(np.asarray(ordinary_parent.day_index[:2]).tolist()) != (2, 3):
        raise ValueError("frozen ordinary lifecycle rows are not 1961 Day 2/3")
    if int(np.asarray(restart_parent.day_index[0])) != 1:
        raise ValueError("frozen restart lifecycle row is not Day 1")

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
        forcing_fields.index("Rainf"),
        forcing_fields.index("Rainf") + 1,
        forcing_fields.index("Snowf"),
        forcing_fields.index("Snowf") + 1,
        forcing_fields.index("Tair"),
        forcing_fields.index("Tair") + 1,
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


def _retained_adapter(
    runtime: Mapping[str, Any],
    teacher_context: Any,
    sample: Mapping[str, Any],
    *,
    lifecycle_start_tstep: int,
    result_day_index: int,
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
    )


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
    mask_mismatches_by_leaf = {
        leaf.key: int(
            np.count_nonzero(
                observed_defined[leaf.start : leaf.stop]
                != expected_defined[leaf.start : leaf.stop]
            )
        )
        for leaf in contract.state_leaves
        if not leaf.discrete
        and np.any(
            observed_defined[leaf.start : leaf.stop]
            != expected_defined[leaf.start : leaf.stop]
        )
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
        "nonfinite_defined_count": nonfinite_defined,
        "defined_mask_mismatch_count": mask_mismatches,
        "defined_mask_mismatches_by_leaf": mask_mismatches_by_leaf,
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
    }


def _lifecycle_stage(runtime: Mapping[str, Any]) -> dict[str, Any]:
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
        started = time.perf_counter()
        head = _head(runtime, arm)
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
        cold_result = compiled_later(runtime["parameters"], ordinary_inputs[0])
        jax.block_until_ready(cold_result)
        ordinary_result = compiled_later(runtime["parameters"], ordinary_inputs[1])
        jax.block_until_ready(ordinary_result)
        compiled_restart = jax.jit(restart)
        restart_result = compiled_restart(runtime["parameters"], restart_input)
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
        arms[arm] = {
            "status": "passed" if passed else "failed",
            "seconds": time.perf_counter() - started,
            "cases": cases,
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
    parser.add_argument("--stage", choices=("lifecycle", "rollout"), required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime = _load_runtime(args)
    report = (
        _lifecycle_stage(runtime)
        if args.stage == "lifecycle"
        else _rollout_stage(runtime)
    )
    _atomic_json(args.output.resolve(), report)
    print(json.dumps(_json_value(report), indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
