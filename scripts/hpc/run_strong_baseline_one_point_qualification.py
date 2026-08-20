"""Memorize one complete train/train point-year with the strong baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np

from scripts.hpc.run_strong_baseline_gpu_admission import _head, _memory_report
from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    PROBE_SEED,
    _load_extended_runtime,
)
from scripts.hpc.run_successor_process_head_canonical_gate import (
    EXPECTED,
    FROZEN_LANDPOINT,
    _atomic_json,
    _retained_adapter,
)

PARAMETER_COUNT = 49_750_681
TOTAL_REDUCTION_GATE = 0.99
EXPERT_REDUCTION_GATE = 0.95
PRIMARY_NRMSE_GATE = 0.05
ISOLATED_EXPERT_REDUCTION_GATE = 0.99
ISOLATED_FIELD_REDUCTION_GATE = 0.95
ISOLATED_FIELD_LOSS_FLOOR = 1.0e-4
DEFAULT_DIAGNOSTIC_TAIL_LEARNING_RATE = 1.0e-5
HIERARCHICAL_GRADIENT_MICROBATCH_SIZE = 2
BUDGET_ATOL = 2.0e-12
STOCK_ATOL = 2.0e-12
PRIMARY_NAMES = (
    "gpp",
    "maintenance_respiration",
    "npp",
    "aboveground_biomass_tendency",
    "belowground_biomass_tendency",
)
HARD_NAMES = (
    "prediction_nonfinite_values",
    "next_state_nonfinite_defined_values",
    "water_budget_failures",
    "carbon_budget_failures",
    "water_negative_inventory_values",
    "carbon_negative_inventory_values",
)


class TrainingStepResult(NamedTuple):
    parameters: Any
    optimizer: Any
    loss: Any
    expert_loss: Any
    gradient_norm: Any
    nonfinite_gradient_values: Any


class CoverageProbeResult(NamedTuple):
    gradients: Any
    updates: Any


class ScienceTargets(NamedTuple):
    next_state: Any
    next_state_defined: Any
    fast_gpp: Any
    fast_gpp_defined: Any


class ScienceBatchStatistics(NamedTuple):
    squared_sum: Any
    signed_sum: Any
    absolute_sum: Any
    count: Any
    hard_counts: Any


class DetailedProcessBatch(NamedTuple):
    total: Any
    expert_mse: Any
    expert_label_count: Any
    label_mse: Any
    label_count: Any
    field_mse: Any
    field_count: Any
    defined_bce: Any
    defined_count: Any
    exact_diagnostic_mse: Any
    exact_diagnostic_count: Any


def _stack(jax: Any, values: Sequence[Any]) -> Any:
    if not values:
        raise ValueError("cannot stack an empty one-point qualification tree")
    return jax.tree_util.tree_map(lambda *items: np.stack(items), *values)


def _split_gradient_microbatches(
    indices: np.ndarray,
    weights: np.ndarray,
    microbatch_size: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Split one logical batch while dropping only zero-weight padding chunks."""

    batch_indices = np.asarray(indices, dtype=np.int32)
    batch_weights = np.asarray(weights, dtype=np.float32)
    if (
        microbatch_size < 1
        or batch_indices.ndim != 1
        or batch_weights.shape != batch_indices.shape
    ):
        raise ValueError("invalid gradient microbatch request")
    result = []
    for start in range(0, batch_indices.size, microbatch_size):
        current_indices = batch_indices[start : start + microbatch_size]
        current_weights = batch_weights[start : start + microbatch_size]
        if np.any(current_weights != 0.0):
            result.append((current_indices, current_weights))
    if not result:
        raise ValueError("logical batch has no admitted gradient samples")
    return tuple(result)


def _prepare_point_year(runtime: Mapping[str, Any]) -> Mapping[str, Any]:
    from research.daily_coarse_graining.daily_operator.successor_fast_owner_training import (
        build_successor_fast_owner_targets,
    )
    from research.daily_coarse_graining.typed_sidecar import read_typed_sidecar_shard

    jax = runtime["jax"]
    reference = runtime["ordinary_reference"]
    parent = runtime["ordinary_parent"]
    if (
        reference.parent.spatial_split != "train"
        or reference.parent.temporal_split != "train"
        or str(reference.parent.landpoint_id) != FROZEN_LANDPOINT
        or int(reference.parent.year) != 1961
    ):
        raise ValueError("one-point qualification reference is not the frozen train/train point-year")
    typed_values, typed_defined, typed_days = read_typed_sidecar_shard(
        reference.path,
        contract=runtime["index"].contract,
        expected_days=reference.day_count,
    )
    if not np.array_equal(typed_days, parent.day_index):
        raise ValueError("one-point qualification typed/base day axes differ")
    day_indices = np.asarray(parent.day_index, dtype=np.int32)
    if day_indices.size != parent.days or np.any(day_indices < 2):
        raise ValueError("one-point qualification includes an invalid lifecycle day")

    raw_inputs = []
    assembled_inputs = []
    targets = []
    science_targets = []
    for row in range(parent.days):
        example = runtime["adapter"].assemble(
            parent.sample(row),
            supplemental_values={name: value[row] for name, value in typed_values.items()},
            supplemental_defined={name: value[row] for name, value in typed_defined.items()},
        )
        target = build_successor_fast_owner_targets(
            example,
            schema=runtime["schema"],
            markov_contract=runtime["contract"],
            owner_contract=runtime["owner_contract"],
        )
        raw_inputs.append(example.inputs)
        assembled_inputs.append(runtime["adapter"].input_assembler.assemble(example.inputs))
        targets.append(target)
        fast_gpp = example.targets.supervision["daily.fast_gpp"]
        science_targets.append(
            ScienceTargets(
                next_state=example.targets.next_state,
                next_state_defined=example.targets.next_state_defined,
                fast_gpp=fast_gpp.value,
                fast_gpp_defined=fast_gpp.defined,
            )
        )
    return {
        "raw_inputs": _stack(jax, raw_inputs),
        "assembled_inputs": _stack(jax, assembled_inputs),
        "targets": _stack(jax, targets),
        "science_targets": _stack(jax, science_targets),
        "day_indices": day_indices,
        "sample_count": int(parent.days),
    }


def _git_head() -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        text=True,
    ).strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_pickle(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _fractional_reduction(initial: float, final: float) -> float | None:
    if initial == 0.0:
        return 0.0 if final == 0.0 else None
    return 1.0 - final / initial


def _compare_loss_breakdowns(
    initial: Mapping[str, Any],
    pre_tail: Mapping[str, Any],
    final: Mapping[str, Any],
) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    counts_identical = True
    for section in ("experts", "labels", "fields", "defined_status"):
        if set(initial[section]) != set(pre_tail[section]) or set(initial[section]) != set(final[section]):
            raise ValueError(f"loss-breakdown {section} registry drift")
        records = {}
        for name in initial[section]:
            first = initial[section][name]
            middle = pre_tail[section][name]
            last = final[section][name]
            count_names = tuple(key for key in first if key.endswith("_count"))
            current_counts_identical = all(first[key] == middle[key] == last[key] for key in count_names)
            counts_identical = counts_identical and current_counts_identical
            records[name] = {
                "initial_mean_daily_loss": first["mean_daily_loss"],
                "pre_tail_mean_daily_loss": middle["mean_daily_loss"],
                "final_mean_daily_loss": last["mean_daily_loss"],
                **{f"initial_{key}": first[key] for key in count_names},
                **{f"pre_tail_{key}": middle[key] for key in count_names},
                **{f"final_{key}": last[key] for key in count_names},
                "counts_identical": current_counts_identical,
                "reduction_from_initialization": _fractional_reduction(
                    first["mean_daily_loss"], last["mean_daily_loss"]
                ),
                "reduction_during_exact_tail": _fractional_reduction(
                    middle["mean_daily_loss"], last["mean_daily_loss"]
                ),
            }
        result[section] = records
    return {"counts_identical": counts_identical, **result}


def _compare_loss_snapshots(
    initial: Mapping[str, Any],
    final: Mapping[str, Any],
) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    counts_identical = True
    for section in ("experts", "labels", "fields", "defined_status", "exact_diagnostics"):
        if set(initial[section]) != set(final[section]):
            raise ValueError(f"loss-snapshot {section} registry drift")
        records = {}
        for name in initial[section]:
            first = initial[section][name]
            last = final[section][name]
            count_names = tuple(key for key in first if key.endswith("_count"))
            current_counts_identical = all(first[key] == last[key] for key in count_names)
            counts_identical = counts_identical and current_counts_identical
            records[name] = {
                "initial_mean_daily_loss": first["mean_daily_loss"],
                "final_mean_daily_loss": last["mean_daily_loss"],
                **{f"initial_{key}": first[key] for key in count_names},
                **{f"final_{key}": last[key] for key in count_names},
                "counts_identical": current_counts_identical,
                "reduction_from_initialization": _fractional_reduction(
                    first["mean_daily_loss"], last["mean_daily_loss"]
                ),
            }
        result[section] = records
    return {"counts_identical": counts_identical, **result}


def _isolated_field_gate(
    loss_breakdown: Mapping[str, Any],
    selected_fields: Sequence[str],
) -> Mapping[str, Any]:
    fields = loss_breakdown["fields"]
    records = {}
    for name in selected_fields:
        record = fields[name]
        initial = float(record["initial_mean_daily_loss"])
        final = float(record["final_mean_daily_loss"])
        reduction = record["reduction_from_initialization"]
        dominant = initial >= ISOLATED_FIELD_LOSS_FLOOR
        passed = (
            not dominant
            or final <= ISOLATED_FIELD_LOSS_FLOOR
            or (reduction is not None and reduction >= ISOLATED_FIELD_REDUCTION_GATE)
        )
        records[name] = {
            "initial_mean_daily_loss": initial,
            "final_mean_daily_loss": final,
            "reduction_from_initialization": reduction,
            "dominant_at_initialization": dominant,
            "passed": passed,
        }
    dominant_records = tuple(record for record in records.values() if record["dominant_at_initialization"])
    return {
        "initial_loss_floor": ISOLATED_FIELD_LOSS_FLOOR,
        "required_dominant_field_reduction": ISOLATED_FIELD_REDUCTION_GATE,
        "dominant_field_count": len(dominant_records),
        "passed": bool(dominant_records and all(record["passed"] for record in dominant_records)),
        "fields": records,
    }


def _one_epoch_training_smoke_passed(
    *,
    failure_reason: str | None,
    completed_epochs: int,
    completed_updates: int,
    expected_updates: int,
    all_gradients_finite: bool,
    initial_process: Mapping[str, Any],
    final_process: Mapping[str, Any],
    final_science: Mapping[str, Any],
    counts_identical: bool,
) -> bool:
    process_values = (
        float(initial_process["total"]),
        float(final_process["total"]),
        *(float(value) for value in final_process["expert_mse"].values()),
    )
    science_values = tuple(
        float(value)
        for record in final_science["primary"].values()
        for key, value in record.items()
        if key != "evaluated_values"
    )
    return (
        failure_reason is None
        and completed_epochs == 1
        and completed_updates == expected_updates
        and all_gradients_finite
        and counts_identical
        and all(np.isfinite(value) for value in (*process_values, *science_values))
        and process_values[1] < process_values[0]
        and all(int(value) == 0 for value in final_science["hard_counts"].values())
    )


def _metric_record(squared: float, signed: float, absolute: float, count: int) -> Mapping[str, Any]:
    if count <= 0:
        raise ValueError("one-point qualification primary metric has no valid values")
    return {
        "normalized_rmse": float(np.sqrt(squared / count)),
        "normalized_signed_bias": float(signed / count),
        "normalized_mean_absolute_error": float(absolute / count),
        "evaluated_values": int(count),
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size < 1 or args.maximum_epochs < 1 or args.evaluation_interval < 1:
        raise ValueError("qualification batch size, epochs, and evaluation interval must be positive")
    isolated_expert = args.isolated_expert
    hierarchical_candidate = bool(args.hierarchical_coordinate_attention)
    coupled_doc_candidate = bool(args.coordinate_coupled_doc_transition)
    if hierarchical_candidate and coupled_doc_candidate:
        raise ValueError("select exactly one coordinate Attention candidate")
    coordinate_candidate = hierarchical_candidate or coupled_doc_candidate
    if coordinate_candidate and not args.mechanistic_output:
        raise ValueError("coordinate Attention requires mechanistic output")
    if coordinate_candidate and isolated_expert not in {None, "litter_poc_doc"}:
        raise ValueError("coordinate Attention only changes litter_poc_doc")
    if isolated_expert is not None and (
        not args.mechanistic_output
        or isolated_expert not in {"litter_poc_doc", "surface_energy"}
    ):
        raise ValueError("isolated-expert mode requires one approved mechanistic diagnostic expert")
    diagnostic_mode = args.diagnostic_tail_updates > 0
    if args.diagnostic_tail_updates < 0 or args.diagnostic_tail_learning_rate <= 0.0:
        raise ValueError("diagnostic tail budget and learning rate are invalid")
    diagnostic_paths = (
        args.named_biomass_tendency_scale_asset,
        args.named_biomass_tendency_scale_acceptance,
        args.diagnostic_state_output,
    )
    if diagnostic_mode and any(path is None for path in diagnostic_paths):
        raise ValueError("loss-floor diagnostic requires tendency-scale and state paths")
    if not diagnostic_mode and any(path is not None for path in diagnostic_paths):
        raise ValueError("diagnostic-only paths require a positive tail-update budget")
    if diagnostic_mode and args.diagnostic_state_output.exists():
        raise FileExistsError(args.diagnostic_state_output)
    if diagnostic_mode and args.mechanistic_output:
        raise ValueError("mechanistic-output qualification does not admit the legacy diagnostic tail")
    if diagnostic_mode and isolated_expert is not None:
        raise ValueError("isolated-expert mode does not admit the legacy diagnostic tail")
    training_smoke = bool(args.training_smoke)
    if training_smoke and (
        not args.mechanistic_output
        or diagnostic_mode
        or args.maximum_epochs != 1
        or args.evaluation_interval != 1
    ):
        raise ValueError("mechanistic training smoke requires one epoch, one-epoch evaluation, and no diagnostic tail")
    started = time.perf_counter()
    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]
    import jax.numpy as jnp

    from jax_orchidee.driver import orchestration as teacher
    from research.daily_coarse_graining.daily_operator.conservative_update import (
        apply_conservative_update,
    )
    from research.daily_coarse_graining.daily_operator.coordinate_coupled_doc_attention import (
        CANDIDATE_NAME as COUPLED_DOC_CANDIDATE_NAME,
    )
    from research.daily_coarse_graining.daily_operator.coordinate_coupled_doc_attention import (
        initialize_coordinate_coupled_doc_transition_parameters,
    )
    from research.daily_coarse_graining.daily_operator.hierarchical_coordinate_attention import (
        CANDIDATE_NAME as HIERARCHICAL_CANDIDATE_NAME,
    )
    from research.daily_coarse_graining.daily_operator.hierarchical_coordinate_attention import (
        initialize_hierarchical_coordinate_attention_parameters,
    )
    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator
    from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import (
        build_strong_baseline_layout,
        initialize_strong_baseline_parameters,
        strong_baseline_parameter_count,
    )
    from research.daily_coarse_graining.daily_operator.strong_baseline_contract import (
        load_strong_baseline_contract,
    )
    from research.daily_coarse_graining.daily_operator.strong_baseline_loss import (
        masked_equal_expert_label_macro_mse,
    )
    from research.daily_coarse_graining.daily_operator.strong_baseline_qualification import (
        accumulate_exact_gradient_microbatch,
        finalize_exact_gradient_accumulator,
        initialize_exact_gradient_accumulator,
        initialize_strong_baseline_adamw,
        padded_epoch_batches,
        qualification_learning_rate,
        strong_baseline_adamw_update,
        strong_baseline_weight_decay_mask,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        DailyOperatorRolloutTrace,
        derive_long_horizon_scales,
        extract_biomass_productivity_trajectories,
        trajectories_from_rollout,
    )
    from research.daily_coarse_graining.daily_operator.successor_training_assets import (
        load_named_biomass_tendency_scale_asset,
    )

    architecture_contract = load_strong_baseline_contract()
    mechanism_contract = None
    parameter_schema = runtime["schema"]
    if args.mechanistic_output:
        from research.daily_coarse_graining.daily_operator.mechanistic_output_contract import (
            build_mechanistic_learned_schema,
            load_mechanistic_output_contract,
        )
        from research.daily_coarse_graining.daily_operator.mechanistic_output_loss import (
            masked_mechanistic_output_loss,
        )
        from research.daily_coarse_graining.daily_operator.mechanistic_process_head import (
            MechanisticCoordinateCoupledDocTransitionProcessHead,
            MechanisticHierarchicalCoordinateAttentionProcessHead,
            MechanisticStrongBaselineProcessHead,
            build_mechanistic_output_layout,
        )

        mechanism_contract = load_mechanistic_output_contract()
        parameter_schema = build_mechanistic_learned_schema(
            runtime["schema"],
            mechanism_contract,
        )
        strong_layout = build_mechanistic_output_layout(
            adapter=runtime["adapter"],
            complete_schema=runtime["schema"],
            learned_schema=parameter_schema,
            owner_contract=runtime["owner_contract"],
            architecture_contract=architecture_contract,
            mechanism_contract=mechanism_contract,
            inputs=runtime["probe_assembled"],
        )
    else:
        strong_layout = build_strong_baseline_layout(
            adapter=runtime["adapter"],
            schema=runtime["schema"],
            owner_contract=runtime["owner_contract"],
            contract=architecture_contract,
            inputs=runtime["probe_assembled"],
        )
    runtime = dict(runtime) | {
        "architecture_contract": architecture_contract,
        "strong_layout": strong_layout,
    }
    head = (
        (
            MechanisticCoordinateCoupledDocTransitionProcessHead
            if coupled_doc_candidate
            else (
                MechanisticHierarchicalCoordinateAttentionProcessHead
                if hierarchical_candidate
                else MechanisticStrongBaselineProcessHead
            )
        )(
            contract=runtime["contract"],
            architecture_contract=architecture_contract,
            mechanism_contract=mechanism_contract,
            input_assembler=runtime["adapter"].input_assembler,
            complete_schema=runtime["schema"],
            learned_schema=parameter_schema,
            layout=strong_layout,
            input_normalization=runtime["input_normalization"],
            output_normalization=runtime["output_normalization"],
            forcing_layout=runtime["forcing_layout"],
            forcing_indices=runtime["forcing_indices"],
            message_gate_scale=0.0,
        )
        if args.mechanistic_output
        else _head(runtime)
    )
    dataset = _prepare_point_year(runtime)
    sample_count = int(dataset["sample_count"])
    batches_per_epoch = int(np.ceil(sample_count / args.batch_size))
    maximum_updates = args.maximum_epochs * batches_per_epoch
    gradient_microbatch_size = (
        min(args.batch_size, HIERARCHICAL_GRADIENT_MICROBATCH_SIZE)
        if coordinate_candidate
        else args.batch_size
    )

    parameter_initializer = (
        initialize_coordinate_coupled_doc_transition_parameters
        if coupled_doc_candidate
        else (
            initialize_hierarchical_coordinate_attention_parameters
            if hierarchical_candidate
            else initialize_strong_baseline_parameters
        )
    )
    parameters = parameter_initializer(
        jax.random.PRNGKey(args.seed),
        schema=parameter_schema,
        layout=strong_layout,
        inputs=runtime["probe_assembled"],
        contract=architecture_contract,
    )
    parameter_count = (
        59_971_257
        if coupled_doc_candidate
        else (
            58_736_698
            if hierarchical_candidate
            else (49_743_001 if args.mechanistic_output else PARAMETER_COUNT)
        )
    )
    if strong_baseline_parameter_count(parameters) != parameter_count:
        raise ValueError("strong-baseline qualification parameter count drift")
    optimizer = initialize_strong_baseline_adamw(parameters)
    decay_mask = strong_baseline_weight_decay_mask(parameters)
    expert_names = tuple(architecture_contract.expert_names)
    isolated_expert_index = (
        None if isolated_expert is None else expert_names.index(isolated_expert)
    )
    active_label_indices = {field.label_index for field in parameter_schema.fields}
    label_names = tuple(
        name for index, name in enumerate(parameter_schema.labels) if index in active_label_indices
    )
    field_names = tuple(
        field.name for field in parameter_schema.fields if field.value_representation != "defined_logit"
    )
    defined_names = tuple(
        field.name for field in parameter_schema.fields if field.value_representation == "defined_logit"
    )
    exact_diagnostic_names = mechanism_contract.exact_fields if args.mechanistic_output else ()
    if set(label_names) != set(architecture_contract.label_ownership):
        raise ValueError("strong-baseline diagnostic label registry drift")
    if len(defined_names) != 1:
        raise ValueError("strong-baseline diagnostic defined-status registry drift")
    selected_field_names = (
        tuple(
            field.name
            for field in parameter_schema.fields
            if architecture_contract.label_ownership[
                parameter_schema.labels[field.label_index]
            ]
            == isolated_expert
            and field.value_representation != "defined_logit"
        )
        if isolated_expert is not None
        else ()
    )
    selected_field_indices = (
        tuple(
            index
            for index, field in enumerate(parameter_schema.fields)
            if architecture_contract.label_ownership[
                parameter_schema.labels[field.label_index]
            ]
            == isolated_expert
            and field.value_representation != "defined_logit"
        )
        if isolated_expert is not None
        else ()
    )
    selected_label_head_indices = (
        tuple(
            index
            for index in active_label_indices
            if architecture_contract.label_ownership[parameter_schema.labels[index]]
            == isolated_expert
        )
        if isolated_expert is not None
        else ()
    )
    if isolated_expert is not None and (
        not selected_field_names or not selected_label_head_indices
    ):
        raise ValueError("isolated expert has no active learned fields")

    assembled_all = jax.device_put(dataset["assembled_inputs"])
    targets_all = jax.device_put(dataset["targets"])
    raw_all = jax.device_put(dataset["raw_inputs"])
    science_all = jax.device_put(dataset["science_targets"])

    def process_loss_result(values: Any, item: Any, wanted: Any) -> Any:
        context = head.assemble_context(values, item)
        if args.mechanistic_output:
            primitive = head(values, context, item)
            update = apply_conservative_update(item, primitive)
            complete = head.materialize_post_update(
                values,
                context,
                item,
                primitive,
                update,
            )
            return masked_mechanistic_output_loss(
                complete.named_outputs,
                wanted,
                inputs=item,
                complete_schema=runtime["schema"],
                learned_schema=parameter_schema,
                scales=runtime["loss_scales"],
                architecture_contract=architecture_contract,
                mechanism_contract=mechanism_contract,
                auxiliary_outputs=complete.auxiliary_outputs,
            )
        outputs = head.named_outputs(values, context, item)
        return masked_equal_expert_label_macro_mse(
            outputs,
            wanted,
            inputs=item,
            schema=runtime["schema"],
            scales=runtime["loss_scales"],
            contract=architecture_contract,
        )

    def one_process_loss(values: Any, item: Any, wanted: Any) -> tuple[Any, Any]:
        loss = process_loss_result(values, item, wanted)
        return loss.total, jnp.stack(tuple(loss.expert_mse[name] for name in expert_names))

    def process_batch(
        values: Any,
        all_inputs: Any,
        all_targets: Any,
        indices: Any,
        weights: Any,
    ) -> tuple[Any, Any]:
        inputs = jax.tree_util.tree_map(lambda value: value[indices], all_inputs)
        wanted = jax.tree_util.tree_map(lambda value: value[indices], all_targets)
        sample_loss, sample_expert = jax.vmap(lambda item, target: one_process_loss(values, item, target))(
            inputs, wanted
        )
        weight = jnp.asarray(weights, dtype=jnp.float64)
        count = jnp.sum(weight, dtype=jnp.float64)
        complete_total = jnp.sum(sample_loss * weight, dtype=jnp.float64) / count
        expert = jnp.sum(sample_expert * weight[:, None], axis=0, dtype=jnp.float64) / count
        objective = (
            complete_total
            if isolated_expert_index is None
            else expert[isolated_expert_index]
        )
        return objective, expert

    def one_process_details(values: Any, item: Any, wanted: Any) -> DetailedProcessBatch:
        loss = process_loss_result(values, item, wanted)
        return DetailedProcessBatch(
            total=loss.total,
            expert_mse=jnp.stack(tuple(loss.expert_mse[name] for name in expert_names)),
            expert_label_count=jnp.stack(tuple(loss.expert_label_count[name] for name in expert_names)),
            label_mse=jnp.stack(tuple(loss.label_mse[name] for name in label_names)),
            label_count=jnp.stack(tuple(loss.label_count[name] for name in label_names)),
            field_mse=jnp.stack(tuple(loss.field_mse[name] for name in field_names)),
            field_count=jnp.stack(tuple(loss.field_count[name] for name in field_names)),
            defined_bce=jnp.stack(tuple(loss.defined_bce[name] for name in defined_names)),
            defined_count=jnp.stack(tuple(loss.defined_count[name] for name in defined_names)),
            exact_diagnostic_mse=(
                jnp.stack(tuple(loss.exact_diagnostic_mse[name] for name in exact_diagnostic_names))
                if exact_diagnostic_names
                else jnp.zeros((0,), dtype=jnp.float64)
            ),
            exact_diagnostic_count=(
                jnp.stack(tuple(loss.exact_diagnostic_count[name] for name in exact_diagnostic_names))
                if exact_diagnostic_names
                else jnp.zeros((0,), dtype=jnp.float64)
            ),
        )

    def detailed_process_batch(
        values: Any,
        all_inputs: Any,
        all_targets: Any,
        indices: Any,
    ) -> DetailedProcessBatch:
        inputs = jax.tree_util.tree_map(lambda value: value[indices], all_inputs)
        wanted = jax.tree_util.tree_map(lambda value: value[indices], all_targets)
        return jax.vmap(lambda item, target: one_process_details(values, item, target))(
            inputs,
            wanted,
        )

    coverage_group_names = (
        (
            (
                "shared",
                "selected_coordinate_blocks",
                "other_expert_blocks",
                "selected_field_heads",
                "other_label_heads",
            )
            if coordinate_candidate
            else (
                "shared",
                "selected_expert_blocks",
                "other_expert_blocks",
                "selected_label_heads",
                "other_label_heads",
            )
        )
        if isolated_expert_index is not None
        else ()
    )

    def coverage_groups(tree: Mapping[str, Any]) -> tuple[Any, ...]:
        if isolated_expert_index is None:
            return ()
        if coordinate_candidate:
            coordinate = tree["hierarchical_litter"]
            shared = {
                name: value
                for name, value in tree.items()
                if name not in {"expert_blocks", "label_heads", "hierarchical_litter"}
            }
            return (
                shared,
                {
                    name: value
                    for name, value in coordinate.items()
                    if name != "field_heads"
                },
                tuple(tree["expert_blocks"]),
                tuple(coordinate["field_heads"][index] for index in selected_field_indices),
                tuple(value for value in tree["label_heads"] if value),
            )
        selected_heads = frozenset(selected_label_head_indices)
        shared = {
            name: value
            for name, value in tree.items()
            if name not in {"expert_blocks", "label_heads"}
        }
        return (
            shared,
            (tree["expert_blocks"][isolated_expert_index],),
            tuple(
                value
                for index, value in enumerate(tree["expert_blocks"])
                if index != isolated_expert_index
            ),
            tuple(
                value
                for index, value in enumerate(tree["label_heads"])
                if index in selected_heads
            ),
            tuple(
                value
                for index, value in enumerate(tree["label_heads"])
                if index not in selected_heads
            ),
        )

    @jax.jit
    def train_step(
        values: Any,
        state: Any,
        all_inputs: Any,
        all_targets: Any,
        indices: Any,
        weights: Any,
        learning_rate: Any,
    ) -> TrainingStepResult:
        (loss, expert_loss), gradients = jax.value_and_grad(
            lambda current: process_batch(
                current,
                all_inputs,
                all_targets,
                indices,
                weights,
            ),
            has_aux=True,
        )(values)
        squared_norm = sum(
            (
                jnp.sum(jnp.square(jnp.asarray(leaf, dtype=jnp.float64)), dtype=jnp.float64)
                for leaf in jax.tree_util.tree_leaves(gradients)
            ),
            jnp.asarray(0.0, dtype=jnp.float64),
        )
        nonfinite = sum(
            (
                jnp.sum(~jnp.isfinite(jnp.asarray(leaf)), dtype=jnp.int64)
                for leaf in jax.tree_util.tree_leaves(gradients)
            ),
            jnp.asarray(0, dtype=jnp.int64),
        )
        next_values, next_state = strong_baseline_adamw_update(
            values,
            gradients,
            state,
            learning_rate,
            decay_mask,
        )
        return TrainingStepResult(
            next_values,
            next_state,
            loss,
            expert_loss,
            jnp.sqrt(squared_norm),
            nonfinite,
        )

    @jax.jit
    def coverage_probe(
        values: Any,
        state: Any,
        all_inputs: Any,
        all_targets: Any,
        indices: Any,
        weights: Any,
        learning_rate: Any,
    ) -> CoverageProbeResult:
        gradients = jax.grad(
            lambda current: process_batch(
                current,
                all_inputs,
                all_targets,
                indices,
                weights,
            )[0]
        )(values)
        next_values, _next_state = strong_baseline_adamw_update(
            values,
            gradients,
            state,
            learning_rate,
            decay_mask,
        )
        updates = jax.tree_util.tree_map(
            lambda after, before: jnp.asarray(after) - jnp.asarray(before),
            next_values,
            values,
        )
        return CoverageProbeResult(gradients, updates)

    @jax.jit
    def accumulate_gradient_microbatch(
        values: Any,
        accumulation: Any,
        all_inputs: Any,
        all_targets: Any,
        indices: Any,
        weights: Any,
    ) -> tuple[Any, Any, Any]:
        (loss, expert_loss), gradients = jax.value_and_grad(
            lambda current: process_batch(
                current,
                all_inputs,
                all_targets,
                indices,
                weights,
            ),
            has_aux=True,
        )(values)
        next_accumulation = accumulate_exact_gradient_microbatch(
            accumulation,
            gradients,
            jnp.sum(jnp.asarray(weights, dtype=jnp.float32), dtype=jnp.float32),
        )
        return next_accumulation, loss, expert_loss

    @jax.jit
    def apply_training_gradient(
        values: Any,
        state: Any,
        gradients: Any,
        learning_rate: Any,
    ) -> tuple[Any, Any]:
        return strong_baseline_adamw_update(
            values,
            gradients,
            state,
            learning_rate,
            decay_mask,
        )

    @jax.jit
    def parameter_updates(after: Any, before: Any) -> Any:
        return jax.tree_util.tree_map(
            lambda current, previous: jnp.asarray(current) - jnp.asarray(previous),
            after,
            before,
        )

    @jax.jit
    def apply_tail_update(
        values: Any,
        state: Any,
        gradients: Any,
    ) -> tuple[Any, Any]:
        return strong_baseline_adamw_update(
            values,
            gradients,
            state,
            jnp.asarray(args.diagnostic_tail_learning_rate, dtype=jnp.float32),
            decay_mask,
        )

    finalize_gradient = jax.jit(finalize_exact_gradient_accumulator)

    @jax.jit
    def parameter_nonfinite_count(values: Any) -> Any:
        return sum(
            (jnp.sum(~jnp.isfinite(jnp.asarray(leaf)), dtype=jnp.int64) for leaf in jax.tree_util.tree_leaves(values)),
            jnp.asarray(0, dtype=jnp.int64),
        )

    evaluate_process_batch = jax.jit(process_batch)
    evaluate_detailed_process_batch = jax.jit(detailed_process_batch)

    def evaluate_process(values: Any) -> Mapping[str, Any]:
        total_sum = 0.0
        expert_sum = np.zeros((len(expert_names),), dtype=np.float64)
        count = 0.0
        for indices, weights in padded_epoch_batches(
            sample_count,
            args.batch_size,
            seed=args.seed,
            epoch=1,
        ):
            total, expert = evaluate_process_batch(
                values,
                assembled_all,
                targets_all,
                jnp.asarray(indices),
                jnp.asarray(weights),
            )
            batch_count = float(np.sum(weights))
            total_sum += float(np.asarray(total)) * batch_count
            expert_sum += np.asarray(expert, dtype=np.float64) * batch_count
            count += batch_count
        return {
            "total": total_sum / count,
            "expert_mse": {name: float(expert_sum[index] / count) for index, name in enumerate(expert_names)},
            "evaluated_days": int(count),
        }

    def evaluate_process_details(values: Any) -> Mapping[str, Any]:
        total_sum = 0.0
        day_count = 0
        shapes = {
            "experts": len(expert_names),
            "labels": len(label_names),
            "fields": len(field_names),
            "defined_status": len(defined_names),
            "exact_diagnostics": len(exact_diagnostic_names),
        }
        loss_sums = {name: np.zeros(size, dtype=np.float64) for name, size in shapes.items()}
        active_days = {name: np.zeros(size, dtype=np.int64) for name, size in shapes.items()}
        defined_values = {name: np.zeros(size, dtype=np.float64) for name, size in shapes.items()}
        expert_label_days = np.zeros((len(expert_names),), dtype=np.float64)
        for indices, weights in padded_epoch_batches(
            sample_count,
            args.batch_size,
            seed=args.seed,
            epoch=1,
        ):
            detail = evaluate_detailed_process_batch(
                values,
                assembled_all,
                targets_all,
                jnp.asarray(indices),
            )
            jax.block_until_ready(detail)
            admitted = np.asarray(weights, dtype=np.bool_)
            total = np.asarray(detail.total, dtype=np.float64)
            total_sum += float(np.sum(total[admitted], dtype=np.float64))
            day_count += int(np.count_nonzero(admitted))
            arrays = {
                "experts": (
                    np.asarray(detail.expert_mse, dtype=np.float64),
                    np.asarray(detail.expert_label_count, dtype=np.float64),
                ),
                "labels": (
                    np.asarray(detail.label_mse, dtype=np.float64),
                    np.asarray(detail.label_count, dtype=np.float64),
                ),
                "fields": (
                    np.asarray(detail.field_mse, dtype=np.float64),
                    np.asarray(detail.field_count, dtype=np.float64),
                ),
                "defined_status": (
                    np.asarray(detail.defined_bce, dtype=np.float64),
                    np.asarray(detail.defined_count, dtype=np.float64),
                ),
                "exact_diagnostics": (
                    np.asarray(detail.exact_diagnostic_mse, dtype=np.float64),
                    np.asarray(detail.exact_diagnostic_count, dtype=np.float64),
                ),
            }
            for section, (losses, counts) in arrays.items():
                active = admitted[:, None] & (counts > 0.0)
                loss_sums[section] += np.sum(np.where(active, losses, 0.0), axis=0, dtype=np.float64)
                active_days[section] += np.sum(active, axis=0, dtype=np.int64)
                defined_values[section] += np.sum(counts * admitted[:, None], axis=0, dtype=np.float64)
            expert_label_days += np.sum(arrays["experts"][1] * admitted[:, None], axis=0, dtype=np.float64)

        names = {
            "experts": expert_names,
            "labels": label_names,
            "fields": field_names,
            "defined_status": defined_names,
            "exact_diagnostics": exact_diagnostic_names,
        }
        sections = {}
        for section, section_names in names.items():
            sections[section] = {
                name: {
                    "mean_daily_loss": (
                        float(loss_sums[section][index] / active_days[section][index])
                        if active_days[section][index]
                        else 0.0
                    ),
                    "active_day_count": int(active_days[section][index]),
                    "defined_value_count": float(defined_values[section][index]),
                }
                for index, name in enumerate(section_names)
            }
        for index, name in enumerate(expert_names):
            owned = tuple(
                label_index
                for label_index, label in enumerate(label_names)
                if architecture_contract.label_ownership[label] == name
            )
            sections["experts"][name]["defined_value_count"] = float(
                np.sum(defined_values["labels"][list(owned)], dtype=np.float64)
            )
            sections["experts"][name]["active_label_day_count"] = float(expert_label_days[index])
        return {
            "total": total_sum / day_count,
            "evaluated_days": day_count,
            **sections,
        }

    long_scales = derive_long_horizon_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    biomass_scales = {
        "aboveground_biomass_carbon": long_scales.stocks["aboveground_biomass_carbon"],
        "belowground_biomass_carbon": long_scales.stocks["belowground_biomass_carbon"],
    }
    tendency_scale_identity = None
    if diagnostic_mode:
        tendency_scale_asset = load_named_biomass_tendency_scale_asset(
            args.named_biomass_tendency_scale_asset,
            args.named_biomass_tendency_scale_acceptance,
        )
        biomass_scales = dict(tendency_scale_asset.scales[1])
        tendency_scale_identity = {
            "asset_path": str(args.named_biomass_tendency_scale_asset.resolve()),
            "asset_file_sha256": tendency_scale_asset.file_sha256,
            "asset_sha256": tendency_scale_asset.payload["asset_sha256"],
            "acceptance_path": str(args.named_biomass_tendency_scale_acceptance.resolve()),
            "acceptance_file_sha256": _sha256_file(args.named_biomass_tendency_scale_acceptance),
            "horizon_days": 1,
            "scales": biomass_scales,
            "selection": "train_only",
        }
    primary_scales = jnp.asarray(
        (
            long_scales.fluxes["gpp"],
            long_scales.fluxes["maintenance_respiration"],
            long_scales.fluxes["npp"],
            biomass_scales["aboveground_biomass_carbon"],
            biomass_scales["belowground_biomass_carbon"],
        ),
        dtype=jnp.float64,
    )
    paths = runtime["context_paths"]
    teacher_context = teacher.prepare_paper_1961_driver_context(
        paths["config"],
        used_run_def_path=paths["run_def"],
        reference_run_dir=paths["reference_run_dir"],
    )
    retained_tail = _retained_adapter(
        runtime,
        teacher_context,
        runtime["ordinary_parent"].sample(0),
        lifecycle_start_tstep=48,
        result_day_index=2,
        fast_owner_materializer=runtime["materializer"],
    )
    transition = bind_daily_operator(
        input_assembler=runtime["adapter"].input_assembler,
        process_head=head,
        retained_tail=retained_tail,
    )

    @jax.jit
    def science_batch(
        values: Any,
        all_inputs: Any,
        all_process_targets: Any,
        expected_all: Any,
        indices: Any,
        weights: Any,
    ) -> ScienceBatchStatistics:
        inputs = jax.tree_util.tree_map(lambda value: value[indices], all_inputs)
        process_targets = jax.tree_util.tree_map(lambda value: value[indices], all_process_targets)
        expected = jax.tree_util.tree_map(lambda value: value[indices], expected_all)
        result = jax.vmap(lambda item: transition(values, item))(inputs)
        trace = DailyOperatorRolloutTrace(
            result=result,
            pft_fraction=inputs.pft.pft_fraction,
            active_pft_mask=inputs.pft.active_pft_mask,
        )
        predicted = trajectories_from_rollout(trace, runtime["contract"])
        wanted = extract_biomass_productivity_trajectories(
            expected.next_state.other,
            expected.next_state_defined.other,
            daily_fast_gpp=expected.fast_gpp,
            daily_fast_gpp_defined=expected.fast_gpp_defined,
            pft_fraction=inputs.pft.pft_fraction,
            active_pft_mask=inputs.pft.active_pft_mask,
            contract=runtime["contract"],
        )
        pairs = (
            (predicted.fluxes["gpp"], wanted.fluxes["gpp"]),
            (
                predicted.fluxes["maintenance_respiration"],
                wanted.fluxes["maintenance_respiration"],
            ),
            (predicted.fluxes["npp"], wanted.fluxes["npp"]),
            (
                predicted.stocks["aboveground_biomass_carbon"],
                wanted.stocks["aboveground_biomass_carbon"],
            ),
            (
                predicted.stocks["belowground_biomass_carbon"],
                wanted.stocks["belowground_biomass_carbon"],
            ),
        )
        squared = []
        signed = []
        absolute = []
        counts = []
        sample_admitted = jnp.asarray(weights, dtype=jnp.bool_)
        for index, (actual, target) in enumerate(pairs):
            left = jnp.asarray(actual.value, dtype=jnp.float64)
            right = jnp.asarray(target.value, dtype=jnp.float64)
            view = (left.shape[0],) + (1,) * (left.ndim - 1)
            admitted = (
                jnp.asarray(actual.defined, dtype=jnp.bool_)
                & jnp.asarray(target.defined, dtype=jnp.bool_)
                & sample_admitted.reshape(view)
            )
            residual = jnp.where(
                admitted,
                (left - right) / primary_scales[index],
                0.0,
            )
            squared.append(jnp.sum(residual * residual, dtype=jnp.float64))
            signed.append(jnp.sum(residual, dtype=jnp.float64))
            absolute.append(jnp.sum(jnp.abs(residual), dtype=jnp.float64))
            counts.append(jnp.sum(admitted, dtype=jnp.int64))

        def admitted_mask(value: Any) -> Any:
            array = jnp.asarray(value)
            return sample_admitted.reshape((array.shape[0],) + (1,) * (array.ndim - 1))

        prediction_nonfinite = sum(
            (
                jnp.sum(
                    (~jnp.isfinite(jnp.asarray(predicted_output)))
                    & jnp.asarray(process_targets[name].defined, dtype=jnp.bool_)
                    & admitted_mask(predicted_output),
                    dtype=jnp.int64,
                )
                for name, predicted_output in result.prediction.named_outputs.items()
            ),
            jnp.asarray(0, dtype=jnp.int64),
        )
        state_nonfinite = sum(
            (
                jnp.sum(
                    jnp.asarray(mask, dtype=jnp.bool_) & ~jnp.isfinite(jnp.asarray(value)) & admitted_mask(value),
                    dtype=jnp.int64,
                )
                for value, mask in zip(result.next_state, result.next_state_defined, strict=True)
            ),
            jnp.asarray(0, dtype=jnp.int64),
        )
        water_budget = jnp.sum(
            sample_admitted
            & (
                ~jnp.isfinite(result.fast_update.water.budget_residual)
                | (jnp.abs(result.fast_update.water.budget_residual) > BUDGET_ATOL)
            ),
            dtype=jnp.int64,
        )
        carbon_budget = jnp.sum(
            sample_admitted
            & (
                ~jnp.isfinite(result.fast_update.carbon.budget_residual)
                | (jnp.abs(result.fast_update.carbon.budget_residual) > BUDGET_ATOL)
            ),
            dtype=jnp.int64,
        )
        water_negative = jnp.sum(
            result.fast_update.water.defined
            & (result.fast_update.water.inventory < -STOCK_ATOL)
            & admitted_mask(result.fast_update.water.inventory),
            dtype=jnp.int64,
        )
        carbon_negative = jnp.sum(
            result.fast_update.carbon.defined
            & (result.fast_update.carbon.inventory < -STOCK_ATOL)
            & admitted_mask(result.fast_update.carbon.inventory),
            dtype=jnp.int64,
        )
        return ScienceBatchStatistics(
            squared_sum=jnp.stack(tuple(squared)),
            signed_sum=jnp.stack(tuple(signed)),
            absolute_sum=jnp.stack(tuple(absolute)),
            count=jnp.stack(tuple(counts)),
            hard_counts=jnp.stack(
                (
                    prediction_nonfinite,
                    state_nonfinite,
                    water_budget,
                    carbon_budget,
                    water_negative,
                    carbon_negative,
                )
            ),
        )

    def evaluate_science(values: Any) -> Mapping[str, Any]:
        squared = np.zeros((len(PRIMARY_NAMES),), dtype=np.float64)
        signed = np.zeros_like(squared)
        absolute = np.zeros_like(squared)
        counts = np.zeros((len(PRIMARY_NAMES),), dtype=np.int64)
        hard = np.zeros((len(HARD_NAMES),), dtype=np.int64)
        for indices, weights in padded_epoch_batches(
            sample_count,
            args.batch_size,
            seed=args.seed,
            epoch=1,
        ):
            result = science_batch(
                values,
                raw_all,
                targets_all,
                science_all,
                jnp.asarray(indices),
                jnp.asarray(weights),
            )
            jax.block_until_ready(result)
            squared += np.asarray(result.squared_sum, dtype=np.float64)
            signed += np.asarray(result.signed_sum, dtype=np.float64)
            absolute += np.asarray(result.absolute_sum, dtype=np.float64)
            counts += np.asarray(result.count, dtype=np.int64)
            hard += np.asarray(result.hard_counts, dtype=np.int64)
        return {
            "primary": {
                name: _metric_record(
                    float(squared[index]),
                    float(signed[index]),
                    float(absolute[index]),
                    int(counts[index]),
                )
                for index, name in enumerate(PRIMARY_NAMES)
            },
            "hard_counts": {name: int(hard[index]) for index, name in enumerate(HARD_NAMES)},
        }

    initial_process = evaluate_process(parameters)
    detail_mode = diagnostic_mode or args.mechanistic_output
    initial_process_details = evaluate_process_details(parameters) if detail_mode else None
    initial_science = evaluate_science(parameters)
    history = []
    science_evaluations = [{"epoch": 0, **initial_science}]
    completed_updates = 0
    completed_epochs = 0
    all_gradients_finite = True
    failure_reason = None
    final_process = initial_process
    final_science = initial_science
    passed = False
    coverage_snapshots: dict[str, Any] = {}

    def coverage_record(result: CoverageProbeResult, update: int) -> Mapping[str, Any]:
        def host_statistics(group: Any) -> tuple[int, int, float]:
            value_count = 0
            nonzero_count = 0
            squared_norm = 0.0
            for leaf in jax.tree_util.tree_leaves(group):
                values = np.asarray(leaf)
                value_count += int(values.size)
                nonzero_count += int(np.count_nonzero(values))
                squared_norm += float(
                    np.sum(
                        np.square(values, dtype=np.float64),
                        dtype=np.float64,
                    )
                )
            return value_count, nonzero_count, float(np.sqrt(squared_norm))

        gradient_statistics = tuple(
            host_statistics(group) for group in coverage_groups(result.gradients)
        )
        update_statistics = tuple(
            host_statistics(group) for group in coverage_groups(result.updates)
        )
        if tuple(record[0] for record in gradient_statistics) != tuple(
            record[0] for record in update_statistics
        ):
            raise ValueError("coverage probe gradient/update shape drift")
        return {
            "update": update,
            "groups": {
                name: {
                    "parameter_value_count": gradient_statistics[index][0],
                    "gradient_nonzero_value_count": gradient_statistics[index][1],
                    "gradient_nonzero_fraction": float(
                        gradient_statistics[index][1] / gradient_statistics[index][0]
                    ),
                    "gradient_l2_norm": gradient_statistics[index][2],
                    "update_nonzero_value_count": update_statistics[index][1],
                    "update_nonzero_fraction": float(
                        update_statistics[index][1] / update_statistics[index][0]
                    ),
                    "update_l2_norm": update_statistics[index][2],
                }
                for index, name in enumerate(coverage_group_names)
            },
        }

    coverage_probe_batch = None
    if isolated_expert is not None and not coordinate_candidate:
        probe_indices, probe_weights = padded_epoch_batches(
            sample_count,
            args.batch_size,
            seed=args.seed,
            epoch=1,
        )[0]
        coverage_probe_batch = (
            jnp.asarray(probe_indices),
            jnp.asarray(probe_weights),
        )
        initial_coverage = coverage_probe(
            parameters,
            optimizer,
            assembled_all,
            targets_all,
            coverage_probe_batch[0],
            coverage_probe_batch[1],
            jnp.asarray(
                qualification_learning_rate(1, maximum_updates),
                dtype=jnp.float32,
            ),
        )
        jax.block_until_ready(initial_coverage)
        coverage_snapshots["initial_probe"] = coverage_record(initial_coverage, 0)

    for epoch in range(1, args.maximum_epochs + 1):
        epoch_started = time.perf_counter()
        losses = []
        gradient_norms = []
        for indices, weights in padded_epoch_batches(
            sample_count,
            args.batch_size,
            seed=args.seed,
            epoch=epoch,
        ):
            update = completed_updates + 1
            rate = qualification_learning_rate(update, maximum_updates)
            gradient_summary = None
            before_parameters = None
            if coordinate_candidate:
                before_parameters = parameters
                accumulation = initialize_exact_gradient_accumulator(parameters)
                batch_loss_sum = 0.0
                logical_batch_count = float(np.sum(weights))
                for micro_indices, micro_weights in _split_gradient_microbatches(
                    indices,
                    weights,
                    gradient_microbatch_size,
                ):
                    accumulation, micro_loss, _micro_expert = accumulate_gradient_microbatch(
                        parameters,
                        accumulation,
                        assembled_all,
                        targets_all,
                        jnp.asarray(micro_indices),
                        jnp.asarray(micro_weights),
                    )
                    jax.block_until_ready(accumulation)
                    micro_count = float(np.sum(micro_weights))
                    batch_loss_sum += float(np.asarray(micro_loss)) * micro_count
                gradient_summary = finalize_gradient(accumulation)
                jax.block_until_ready(gradient_summary)
                admitted_samples = float(np.asarray(gradient_summary.admitted_samples))
                nonfinite = int(np.asarray(gradient_summary.nonfinite_gradient_values))
                gradient_norm = float(np.asarray(gradient_summary.gradient_norm))
                loss_number = batch_loss_sum / admitted_samples
                if admitted_samples != logical_batch_count:
                    failure_reason = f"gradient microbatch coverage drift at update {update}"
                    break
                parameters, optimizer = apply_training_gradient(
                    parameters,
                    optimizer,
                    gradient_summary.gradients,
                    jnp.asarray(rate, dtype=jnp.float32),
                )
                jax.block_until_ready((parameters, optimizer))
            else:
                result = train_step(
                    parameters,
                    optimizer,
                    assembled_all,
                    targets_all,
                    jnp.asarray(indices),
                    jnp.asarray(weights),
                    jnp.asarray(rate, dtype=jnp.float32),
                )
                jax.block_until_ready(result)
                nonfinite = int(np.asarray(result.nonfinite_gradient_values))
                loss_number = float(np.asarray(result.loss))
                gradient_norm = float(np.asarray(result.gradient_norm))
                parameters = result.parameters
                optimizer = result.optimizer
            if nonfinite or not np.isfinite(loss_number) or not np.isfinite(gradient_norm):
                all_gradients_finite = False
                failure_reason = f"nonfinite training update {update}"
                break
            if coordinate_candidate and update in {1, maximum_updates}:
                updates = parameter_updates(parameters, before_parameters)
                jax.block_until_ready(updates)
                snapshot_name = "initial_probe" if update == 1 else "final_probe"
                coverage_snapshots[snapshot_name] = coverage_record(
                    CoverageProbeResult(gradient_summary.gradients, updates),
                    update,
                )
            completed_updates = update
            losses.append(loss_number)
            gradient_norms.append(gradient_norm)
        completed_epochs = epoch if failure_reason is None else epoch - 1
        record: dict[str, Any] = {
            "epoch": epoch,
            "completed_updates": completed_updates,
            "learning_rate": qualification_learning_rate(max(completed_updates, 1), maximum_updates),
            "mean_training_batch_loss": float(np.mean(losses)) if losses else None,
            "gradient_norm": (
                {
                    "minimum": float(np.min(gradient_norms)),
                    "median": float(np.median(gradient_norms)),
                    "maximum": float(np.max(gradient_norms)),
                }
                if gradient_norms
                else None
            ),
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        should_evaluate = (
            epoch == 1
            or epoch % args.evaluation_interval == 0
            or epoch == args.maximum_epochs
            or failure_reason is not None
        )
        if should_evaluate:
            final_process = evaluate_process(parameters)
            total_reduction = 1.0 - final_process["total"] / initial_process["total"]
            expert_reduction = {
                name: 1.0 - final_process["expert_mse"][name] / initial_process["expert_mse"][name]
                for name in expert_names
            }
            process_passed = (
                expert_reduction[isolated_expert] >= ISOLATED_EXPERT_REDUCTION_GATE
                if isolated_expert is not None
                else total_reduction >= TOTAL_REDUCTION_GATE
                and all(value >= EXPERT_REDUCTION_GATE for value in expert_reduction.values())
            )
            record["process_evaluation"] = {
                **final_process,
                "total_reduction": total_reduction,
                "expert_reduction": expert_reduction,
                "passed": process_passed,
            }
            if process_passed or epoch == args.maximum_epochs or failure_reason is not None:
                final_science = evaluate_science(parameters)
                science_evaluations.append({"epoch": epoch, **final_science})
                primary_passed = all(
                    value["normalized_rmse"] <= PRIMARY_NRMSE_GATE for value in final_science["primary"].values()
                )
                hard_passed = all(value == 0 for value in final_science["hard_counts"].values())
                record["science_evaluation"] = final_science | {
                    "primary_passed": primary_passed,
                    "hard_passed": hard_passed,
                }
                passed = (
                    process_passed
                    and (primary_passed or isolated_expert is not None)
                    and hard_passed
                    and all_gradients_finite
                )
        history.append(record)
        print(
            json.dumps(
                {
                    "event": (
                        "mechanistic_output_one_point_epoch"
                        if args.mechanistic_output
                        else "strong_baseline_one_point_epoch"
                    ),
                    "epoch": epoch,
                    "updates": completed_updates,
                    "mean_training_batch_loss": record["mean_training_batch_loss"],
                    "process_total_reduction": record.get("process_evaluation", {}).get("total_reduction"),
                    "passed": passed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if (passed and isolated_expert is None) or failure_reason is not None:
            break

    if isolated_expert is not None and not coordinate_candidate and failure_reason is None:
        final_coverage = coverage_probe(
            parameters,
            optimizer,
            assembled_all,
            targets_all,
            coverage_probe_batch[0],
            coverage_probe_batch[1],
            jnp.asarray(
                qualification_learning_rate(completed_updates, maximum_updates),
                dtype=jnp.float32,
            ),
        )
        jax.block_until_ready(final_coverage)
        coverage_snapshots["final_probe"] = coverage_record(
            final_coverage,
            completed_updates,
        )

    pre_tail_process = final_process
    pre_tail_science = final_science
    pre_tail_process_details = evaluate_process_details(parameters) if diagnostic_mode else None
    tail_history = []
    completed_tail_updates = 0
    if diagnostic_mode and failure_reason is None and not passed:
        tail_optimizer = initialize_strong_baseline_adamw(parameters)
        tail_batches = padded_epoch_batches(
            sample_count,
            args.batch_size,
            seed=args.seed,
            epoch=1,
        )
        for tail_update in range(1, args.diagnostic_tail_updates + 1):
            tail_started = time.perf_counter()
            accumulation = initialize_exact_gradient_accumulator(parameters)
            pre_update_loss_sum = 0.0
            pre_update_expert_sum = np.zeros((len(expert_names),), dtype=np.float64)
            for indices, weights in tail_batches:
                accumulation, batch_loss, batch_expert = accumulate_gradient_microbatch(
                    parameters,
                    accumulation,
                    assembled_all,
                    targets_all,
                    jnp.asarray(indices),
                    jnp.asarray(weights),
                )
                jax.block_until_ready(accumulation)
                batch_count = float(np.sum(weights))
                pre_update_loss_sum += float(np.asarray(batch_loss)) * batch_count
                pre_update_expert_sum += np.asarray(batch_expert, dtype=np.float64) * batch_count
            gradient = finalize_gradient(accumulation)
            jax.block_until_ready(gradient)
            admitted_samples = float(np.asarray(gradient.admitted_samples))
            microbatch_count = int(np.asarray(gradient.microbatch_count))
            nonfinite = int(np.asarray(gradient.nonfinite_gradient_values))
            gradient_norm = float(np.asarray(gradient.gradient_norm))
            mean_microbatch_norm = float(np.asarray(gradient.mean_microbatch_gradient_norm))
            cancellation_ratio = float(np.asarray(gradient.cancellation_ratio))
            coverage_passed = admitted_samples == float(sample_count) and microbatch_count == batches_per_epoch
            pre_update_loss = pre_update_loss_sum / admitted_samples
            if (
                not coverage_passed
                or nonfinite
                or not np.isfinite(pre_update_loss)
                or not np.isfinite(gradient_norm)
                or not np.isfinite(mean_microbatch_norm)
                or not np.isfinite(cancellation_ratio)
            ):
                all_gradients_finite = all_gradients_finite and nonfinite == 0
                failure_reason = f"invalid exact-gradient tail update {tail_update}"
                break

            parameters, tail_optimizer = apply_tail_update(
                parameters,
                tail_optimizer,
                gradient.gradients,
            )
            jax.block_until_ready((parameters, tail_optimizer))
            parameter_nonfinite = int(np.asarray(parameter_nonfinite_count(parameters)))
            if parameter_nonfinite:
                failure_reason = f"nonfinite parameters after exact-gradient tail update {tail_update}"
                break
            completed_tail_updates = tail_update
            final_process = evaluate_process(parameters)
            total_reduction = 1.0 - final_process["total"] / initial_process["total"]
            expert_reduction = {
                name: 1.0 - final_process["expert_mse"][name] / initial_process["expert_mse"][name]
                for name in expert_names
            }
            process_passed = (
                np.isfinite(final_process["total"])
                and total_reduction >= TOTAL_REDUCTION_GATE
                and all(np.isfinite(value) and value >= EXPERT_REDUCTION_GATE for value in expert_reduction.values())
            )
            tail_record: dict[str, Any] = {
                "tail_update": tail_update,
                "learning_rate": args.diagnostic_tail_learning_rate,
                "exact_sample_count": admitted_samples,
                "expected_sample_count": sample_count,
                "microbatch_count": microbatch_count,
                "expected_microbatch_count": batches_per_epoch,
                "coverage_passed": coverage_passed,
                "pre_update_process_loss": pre_update_loss,
                "pre_update_expert_mse": {
                    name: float(pre_update_expert_sum[index] / admitted_samples)
                    for index, name in enumerate(expert_names)
                },
                "exact_gradient_norm": gradient_norm,
                "mean_microbatch_gradient_norm": mean_microbatch_norm,
                "exact_to_mean_microbatch_gradient_norm_ratio": cancellation_ratio,
                "nonfinite_gradient_values": nonfinite,
                "nonfinite_parameter_values": parameter_nonfinite,
                "post_update_process": final_process,
                "total_loss_reduction_from_initialization": total_reduction,
                "expert_loss_reduction_from_initialization": expert_reduction,
                "process_passed": process_passed,
                "elapsed_seconds": time.perf_counter() - tail_started,
            }
            if process_passed or tail_update == args.diagnostic_tail_updates:
                final_science = evaluate_science(parameters)
                science_evaluations.append({"exact_tail_update": tail_update, **final_science})
                primary_passed = all(
                    value["normalized_rmse"] <= PRIMARY_NRMSE_GATE for value in final_science["primary"].values()
                )
                hard_passed = all(value == 0 for value in final_science["hard_counts"].values())
                tail_record["science_evaluation"] = final_science | {
                    "primary_passed": primary_passed,
                    "hard_passed": hard_passed,
                }
                passed = process_passed and primary_passed and hard_passed and all_gradients_finite
            tail_history.append(tail_record)
            print(
                json.dumps(
                    {
                        "event": "strong_baseline_one_point_exact_gradient_tail",
                        "tail_update": tail_update,
                        "exact_gradient_norm": gradient_norm,
                        "cancellation_ratio": cancellation_ratio,
                        "process_total_reduction": total_reduction,
                        "passed": passed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if passed:
                break

    final_process_details = evaluate_process_details(parameters) if detail_mode else None
    loss_floor_breakdown = (
        _compare_loss_breakdowns(
            initial_process_details,
            pre_tail_process_details,
            final_process_details,
        )
        if diagnostic_mode
        else None
    )
    if diagnostic_mode and failure_reason is None and not loss_floor_breakdown["counts_identical"]:
        failure_reason = "loss-floor diagnostic counts changed between evaluations"
    mechanistic_loss_breakdown = (
        _compare_loss_snapshots(initial_process_details, final_process_details)
        if args.mechanistic_output
        else None
    )
    if args.mechanistic_output and failure_reason is None and not mechanistic_loss_breakdown["counts_identical"]:
        failure_reason = "mechanistic loss-breakdown counts changed between evaluations"

    total_reduction = 1.0 - final_process["total"] / initial_process["total"]
    expert_reduction = {
        name: 1.0 - final_process["expert_mse"][name] / initial_process["expert_mse"][name]
        for name in expert_names
    }
    isolated_field_result = (
        _isolated_field_gate(mechanistic_loss_breakdown, selected_field_names)
        if isolated_expert is not None
        else None
    )
    isolated_coverage_gate = None
    if isolated_expert is not None:
        required_groups = (
            ("selected_coordinate_blocks", "selected_field_heads")
            if coordinate_candidate
            else ("selected_expert_blocks", "selected_label_heads")
        )
        coverage_checks = {
            snapshot_name: {
                group: {
                    "gradient_nonzero": (
                        coverage_snapshots[snapshot_name]["groups"][group][
                            "gradient_nonzero_value_count"
                        ]
                        > 0
                    ),
                    "update_nonzero": (
                        coverage_snapshots[snapshot_name]["groups"][group][
                            "update_nonzero_value_count"
                        ]
                        > 0
                    ),
                }
                for group in required_groups
            }
            for snapshot_name in ("initial_probe", "final_probe")
            if snapshot_name in coverage_snapshots
        }
        isolated_coverage_gate = {
            "required_groups": list(required_groups),
            "snapshots": coverage_checks,
            "passed": bool(
                len(coverage_checks) == 2
                and all(
                    item["gradient_nonzero"] and item["update_nonzero"]
                    for snapshot in coverage_checks.values()
                    for item in snapshot.values()
                )
            ),
        }
        if not training_smoke:
            passed = bool(
                failure_reason is None
                and completed_epochs == args.maximum_epochs
                and completed_updates == maximum_updates
                and all_gradients_finite
                and mechanistic_loss_breakdown["counts_identical"]
                and expert_reduction[isolated_expert]
                >= ISOLATED_EXPERT_REDUCTION_GATE
                and isolated_field_result["passed"]
                and isolated_coverage_gate["passed"]
                and all(value == 0 for value in final_science["hard_counts"].values())
            )

    training_smoke_passed = (
        _one_epoch_training_smoke_passed(
            failure_reason=failure_reason,
            completed_epochs=completed_epochs,
            completed_updates=completed_updates,
            expected_updates=batches_per_epoch,
            all_gradients_finite=all_gradients_finite,
            initial_process=initial_process,
            final_process=final_process,
            final_science=final_science,
            counts_identical=mechanistic_loss_breakdown["counts_identical"],
        )
        if training_smoke
        else False
    )

    git_head = _git_head()
    diagnostic_state = None
    if diagnostic_mode and failure_reason is None:
        state_payload = {
            "schema_version": "strong_baseline_one_point_loss_floor_state_v1",
            "identity": {
                "git_head": git_head,
                "dataset_manifest_sha256": EXPECTED["dataset_manifest"],
                "architecture_contract_sha256": architecture_contract.sha256,
                "tendency_scale_asset_sha256": tendency_scale_identity["asset_sha256"],
                "landpoint_id": FROZEN_LANDPOINT,
                "year": 1961,
                "seed": args.seed,
            },
            "usage": {
                "diagnostic_only": True,
                "formal_training_initialization_allowed": False,
                "promotion_checkpoint": False,
                "contains_optimizer_state": False,
            },
            "progress": {
                "completed_minibatch_epochs": completed_epochs,
                "completed_minibatch_updates": completed_updates,
                "completed_exact_gradient_tail_updates": completed_tail_updates,
                "qualification_gate_passed": passed,
            },
            "parameters": jax.device_get(parameters),
        }
        _atomic_pickle(args.diagnostic_state_output, state_payload)
        diagnostic_state = {
            "path": str(args.diagnostic_state_output.resolve()),
            "sha256": _sha256_file(args.diagnostic_state_output),
            **state_payload["usage"],
        }
    execution_status = "completed" if failure_reason is None else "failed"
    report_status = _report_status(
        gate_passed=training_smoke_passed if training_smoke else passed,
        execution_status=execution_status,
    )
    if coupled_doc_candidate:
        report_schema_version = (
            "coordinate_coupled_doc_transition_v4_one_epoch_smoke_v1"
            if training_smoke
            else "coordinate_coupled_doc_transition_v4_memorization_v1"
        )
    elif hierarchical_candidate:
        report_schema_version = (
            "hierarchical_coordinate_query_attention_v3_one_epoch_smoke_v1"
            if training_smoke
            else "hierarchical_coordinate_query_attention_v3_memorization_v1"
        )
    elif diagnostic_mode:
        report_schema_version = "strong_baseline_one_point_loss_floor_diagnostic_v1"
    elif isolated_expert is not None:
        report_schema_version = (
            "mechanistic_output_v2_isolated_expert_one_epoch_smoke_v1"
            if training_smoke
            else "mechanistic_output_v2_isolated_expert_memorization_v1"
        )
    elif training_smoke:
        report_schema_version = "mechanistic_output_one_epoch_training_smoke_v1"
    elif args.mechanistic_output:
        report_schema_version = "mechanistic_output_one_point_year_qualification_v1"
    else:
        report_schema_version = "strong_baseline_one_point_year_qualification_v1"
    report = {
        "schema_version": report_schema_version,
        "status": report_status,
        "execution_status": execution_status,
        "qualification_gate_status": "not_evaluated" if training_smoke else ("passed" if passed else "failed"),
        "one_epoch_training_smoke_gate_status": (
            "passed" if training_smoke_passed else ("failed" if training_smoke else "not_applicable")
        ),
        "failure_reason": failure_reason,
        "identity": {
            "git_head": git_head,
            "dataset_manifest_sha256": EXPECTED["dataset_manifest"],
            "architecture_contract_sha256": architecture_contract.sha256,
            "tendency_scale": tendency_scale_identity,
        },
        "data": {
            "landpoint_id": FROZEN_LANDPOINT,
            "year": 1961,
            "spatial_split": "train",
            "temporal_split": "train",
            "valid_day_count": sample_count,
            "day_index_minimum": int(np.min(dataset["day_indices"])),
            "day_index_maximum": int(np.max(dataset["day_indices"])),
            "every_valid_day_once_per_epoch": True,
            "last_batch_padding_has_zero_weight": True,
        },
        "model": {
            "parameter_count": parameter_count,
            "fresh_initialization_seed": args.seed,
            "message_gate_scale": 0.0,
            "graph_path_executed": False,
            "historical_checkpoint_used": False,
            "training_objective": (
                {
                    "mode": "isolated_existing_expert_loss",
                    "selected_expert": isolated_expert,
                    "unselected_experts_reported": True,
                    "complete_boundary_materialized": True,
                }
                if isolated_expert is not None
                else {"mode": "equal_experts"}
            ),
            **(
                {
                    "candidate": (
                        COUPLED_DOC_CANDIDATE_NAME
                        if coupled_doc_candidate
                        else (
                            HIERARCHICAL_CANDIDATE_NAME
                            if hierarchical_candidate
                            else "strong_baseline_mechanistic_output_v2"
                        )
                    ),
                    "learned_output_field_count": len(parameter_schema.fields),
                    "learned_output_scalar_count": sum(
                        int(np.prod(field.shape, dtype=np.int64)) for field in parameter_schema.fields
                    ),
                    "complete_output_field_count": len(runtime["schema"].fields),
                    "complete_output_scalar_count": sum(
                        int(np.prod(field.shape, dtype=np.int64)) for field in runtime["schema"].fields
                    ),
                    "exact_derived_output_field_count": len(mechanism_contract.exact_fields),
                }
                if args.mechanistic_output
                else {}
            ),
        },
        "optimizer": {
            "name": "adamw",
            "batch_size": args.batch_size,
            "gradient_microbatch_size": gradient_microbatch_size,
            "maximum_gradient_microbatches_per_update": int(
                np.ceil(args.batch_size / gradient_microbatch_size)
            ),
            "gradient_accumulation_preserves_logical_batch": bool(
                gradient_microbatch_size < args.batch_size
            ),
            "maximum_epochs": args.maximum_epochs,
            "completed_epochs": completed_epochs,
            "maximum_updates": maximum_updates,
            "completed_updates": completed_updates,
            "peak_learning_rate": 1.0e-4,
            "final_learning_rate": 1.0e-5,
            "warmup_fraction": 0.02,
            "weight_decay": 1.0e-4,
            "normalization_scales_and_biases_excluded_from_decay": True,
            "gradient_clipping": False,
            "all_observed_gradients_finite": all_gradients_finite,
            "isolated_expert_coverage": (
                {
                    "group_names": list(coverage_group_names),
                    "snapshots": coverage_snapshots,
                    "gate": isolated_coverage_gate,
                }
                if isolated_expert is not None
                else None
            ),
        },
        "exact_gradient_tail": (
            {
                "optimizer": "fresh_adamw",
                "optimizer_state_inherited_from_minibatch_stage": False,
                "maximum_updates": args.diagnostic_tail_updates,
                "completed_updates": completed_tail_updates,
                "microbatch_size": args.batch_size,
                "microbatches_per_update": batches_per_epoch,
                "real_samples_per_update": sample_count,
                "learning_rate": args.diagnostic_tail_learning_rate,
                "weight_decay": 1.0e-4,
                "gradient_clipping": False,
                "history": tail_history,
            }
            if diagnostic_mode
            else None
        ),
        "gates": {
            "required_total_loss_reduction": (
                None if isolated_expert is not None else TOTAL_REDUCTION_GATE
            ),
            "required_every_expert_loss_reduction": (
                None if isolated_expert is not None else EXPERT_REDUCTION_GATE
            ),
            "required_selected_expert_loss_reduction": (
                ISOLATED_EXPERT_REDUCTION_GATE
                if isolated_expert is not None and not training_smoke
                else None
            ),
            "isolated_dominant_field_initial_loss_floor": (
                ISOLATED_FIELD_LOSS_FLOOR if isolated_expert is not None else None
            ),
            "required_isolated_dominant_field_reduction": (
                ISOLATED_FIELD_REDUCTION_GATE if isolated_expert is not None else None
            ),
            "maximum_primary_normalized_rmse": (
                None if isolated_expert is not None else PRIMARY_NRMSE_GATE
            ),
            "required_hard_failure_count": 0,
        },
        "initial": {
            "process": initial_process,
            "science": initial_science,
        },
        "pre_exact_gradient_tail": (
            {"process": pre_tail_process, "science": pre_tail_science} if diagnostic_mode else None
        ),
        "final": {
            "process": final_process,
            "total_loss_reduction": total_reduction,
            "expert_loss_reduction": expert_reduction,
            "isolated_field_gate": isolated_field_result,
            "science": final_science,
        },
        "loss_floor_breakdown": loss_floor_breakdown,
        "mechanistic_loss_breakdown": mechanistic_loss_breakdown,
        "history": history,
        "science_evaluations": science_evaluations,
        "device": {
            "backend": jax.default_backend(),
            "devices": [str(value) for value in runtime["devices"]],
            "memory": [_memory_report(value) for value in runtime["devices"]],
        },
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "checkpoint_written": diagnostic_state is not None,
        "diagnostic_state": diagnostic_state,
        "qualification_checkpoint_initializes_formal_training": False,
        "reconstructs_48_forcing_records": False,
        "executes_48_step_state_scan": False,
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": (
            (
                "run_full_isolated_expert_memorization"
                if training_smoke_passed
                else "repair_isolated_expert_one_epoch_smoke"
            )
            if training_smoke and isolated_expert is not None
            else (
                (
                    "compare_isolated_expert_results_before_joint_objective_change"
                    if passed
                    else "review_isolated_expert_field_and_gradient_failure"
                )
                if isolated_expert is not None
                else (
                    (
                        "run_formal_one_point_year_memorization"
                        if training_smoke_passed
                        else "repair_one_epoch_training_smoke"
                    )
                    if training_smoke
                    else (
                        "design_64_point_year_qualification"
                        if passed
                        else (
                            "use_mechanistic_label_field_attribution_before_model_change"
                            if args.mechanistic_output
                            else "use_label_field_loss_floor_evidence_before_next_model_change"
                        )
                    )
                )
            )
        ),
    }
    _atomic_json(args.output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--maximum-epochs", type=int, default=100)
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=PROBE_SEED)
    parser.add_argument("--mechanistic-output", action="store_true")
    parser.add_argument("--hierarchical-coordinate-attention", action="store_true")
    parser.add_argument("--coordinate-coupled-doc-transition", action="store_true")
    parser.add_argument("--isolated-expert")
    parser.add_argument("--training-smoke", action="store_true")
    parser.add_argument("--diagnostic-tail-updates", type=int, default=0)
    parser.add_argument(
        "--diagnostic-tail-learning-rate",
        type=float,
        default=DEFAULT_DIAGNOSTIC_TAIL_LEARNING_RATE,
    )
    parser.add_argument("--named-biomass-tendency-scale-asset", type=Path)
    parser.add_argument("--named-biomass-tendency-scale-acceptance", type=Path)
    parser.add_argument("--diagnostic-state-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _report_status(*, gate_passed: bool, execution_status: str) -> str:
    if gate_passed:
        return "passed"
    if execution_status == "completed":
        return "completed_failed_gate"
    return "execution_failed"


def _report_exit_code(report: Mapping[str, Any]) -> int:
    return 0 if report["execution_status"] == "completed" else 1


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return _report_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
