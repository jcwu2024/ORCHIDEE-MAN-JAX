"""Memorize one complete train/train point-year with the strong baseline."""

from __future__ import annotations

import argparse
import json
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


def _stack(jax: Any, values: Sequence[Any]) -> Any:
    if not values:
        raise ValueError("cannot stack an empty one-point qualification tree")
    return jax.tree_util.tree_map(lambda *items: np.stack(items), *values)


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
    started = time.perf_counter()
    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]
    import jax.numpy as jnp

    from jax_orchidee.driver import orchestration as teacher
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

    architecture_contract = load_strong_baseline_contract()
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
    head = _head(runtime)
    dataset = _prepare_point_year(runtime)
    sample_count = int(dataset["sample_count"])
    batches_per_epoch = int(np.ceil(sample_count / args.batch_size))
    maximum_updates = args.maximum_epochs * batches_per_epoch

    parameters = initialize_strong_baseline_parameters(
        jax.random.PRNGKey(args.seed),
        schema=runtime["schema"],
        layout=strong_layout,
        inputs=runtime["probe_assembled"],
        contract=architecture_contract,
    )
    if strong_baseline_parameter_count(parameters) != PARAMETER_COUNT:
        raise ValueError("strong-baseline qualification parameter count drift")
    optimizer = initialize_strong_baseline_adamw(parameters)
    decay_mask = strong_baseline_weight_decay_mask(parameters)
    expert_names = tuple(architecture_contract.expert_names)

    assembled_all = jax.device_put(dataset["assembled_inputs"])
    targets_all = jax.device_put(dataset["targets"])
    raw_all = jax.device_put(dataset["raw_inputs"])
    science_all = jax.device_put(dataset["science_targets"])

    def one_process_loss(values: Any, item: Any, wanted: Any) -> tuple[Any, Any]:
        context = head.assemble_context(values, item)
        outputs = head.named_outputs(values, context, item)
        loss = masked_equal_expert_label_macro_mse(
            outputs,
            wanted,
            inputs=item,
            schema=runtime["schema"],
            scales=runtime["loss_scales"],
            contract=architecture_contract,
        )
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
        sample_loss, sample_expert = jax.vmap(
            lambda item, target: one_process_loss(values, item, target)
        )(inputs, wanted)
        weight = jnp.asarray(weights, dtype=jnp.float64)
        count = jnp.sum(weight, dtype=jnp.float64)
        total = jnp.sum(sample_loss * weight, dtype=jnp.float64) / count
        expert = jnp.sum(sample_expert * weight[:, None], axis=0, dtype=jnp.float64) / count
        return total, expert

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

    evaluate_process_batch = jax.jit(process_batch)

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
            "expert_mse": {
                name: float(expert_sum[index] / count)
                for index, name in enumerate(expert_names)
            },
            "evaluated_days": int(count),
        }

    long_scales = derive_long_horizon_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    primary_scales = jnp.asarray(
        (
            long_scales.fluxes["gpp"],
            long_scales.fluxes["maintenance_respiration"],
            long_scales.fluxes["npp"],
            long_scales.stocks["aboveground_biomass_carbon"],
            long_scales.stocks["belowground_biomass_carbon"],
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
        expected_all: Any,
        indices: Any,
        weights: Any,
    ) -> ScienceBatchStatistics:
        inputs = jax.tree_util.tree_map(lambda value: value[indices], all_inputs)
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
                jnp.sum((~jnp.isfinite(jnp.asarray(leaf))) & admitted_mask(leaf), dtype=jnp.int64)
                for leaf in jax.tree_util.tree_leaves(result.prediction.named_outputs)
            ),
            jnp.asarray(0, dtype=jnp.int64),
        )
        state_nonfinite = sum(
            (
                jnp.sum(
                    jnp.asarray(mask, dtype=jnp.bool_)
                    & ~jnp.isfinite(jnp.asarray(value))
                    & admitted_mask(value),
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
            "hard_counts": {
                name: int(hard[index]) for index, name in enumerate(HARD_NAMES)
            },
        }

    initial_process = evaluate_process(parameters)
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
            if nonfinite or not np.isfinite(loss_number) or not np.isfinite(gradient_norm):
                all_gradients_finite = False
                failure_reason = f"nonfinite training update {update}"
                break
            parameters = result.parameters
            optimizer = result.optimizer
            completed_updates = update
            losses.append(loss_number)
            gradient_norms.append(gradient_norm)
        completed_epochs = epoch if failure_reason is None else epoch - 1
        record: dict[str, Any] = {
            "epoch": epoch,
            "completed_updates": completed_updates,
            "learning_rate": qualification_learning_rate(
                max(completed_updates, 1), maximum_updates
            ),
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
                name: 1.0
                - final_process["expert_mse"][name]
                / initial_process["expert_mse"][name]
                for name in expert_names
            }
            process_passed = (
                total_reduction >= TOTAL_REDUCTION_GATE
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
                    value["normalized_rmse"] <= PRIMARY_NRMSE_GATE
                    for value in final_science["primary"].values()
                )
                hard_passed = all(value == 0 for value in final_science["hard_counts"].values())
                record["science_evaluation"] = final_science | {
                    "primary_passed": primary_passed,
                    "hard_passed": hard_passed,
                }
                passed = process_passed and primary_passed and hard_passed and all_gradients_finite
        history.append(record)
        print(
            json.dumps(
                {
                    "event": "strong_baseline_one_point_epoch",
                    "epoch": epoch,
                    "updates": completed_updates,
                    "mean_training_batch_loss": record["mean_training_batch_loss"],
                    "process_total_reduction": record.get("process_evaluation", {}).get(
                        "total_reduction"
                    ),
                    "passed": passed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if passed or failure_reason is not None:
            break

    total_reduction = 1.0 - final_process["total"] / initial_process["total"]
    expert_reduction = {
        name: 1.0
        - final_process["expert_mse"][name] / initial_process["expert_mse"][name]
        for name in expert_names
    }
    report = {
        "schema_version": "strong_baseline_one_point_year_qualification_v1",
        "status": "passed" if passed else "failed",
        "failure_reason": failure_reason,
        "identity": {
            "git_head": _git_head(),
            "dataset_manifest_sha256": EXPECTED["dataset_manifest"],
            "architecture_contract_sha256": architecture_contract.sha256,
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
            "parameter_count": PARAMETER_COUNT,
            "fresh_initialization_seed": args.seed,
            "message_gate_scale": 0.0,
            "graph_path_executed": False,
            "historical_checkpoint_used": False,
        },
        "optimizer": {
            "name": "adamw",
            "batch_size": args.batch_size,
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
        },
        "gates": {
            "required_total_loss_reduction": TOTAL_REDUCTION_GATE,
            "required_every_expert_loss_reduction": EXPERT_REDUCTION_GATE,
            "maximum_primary_normalized_rmse": PRIMARY_NRMSE_GATE,
            "required_hard_failure_count": 0,
        },
        "initial": {
            "process": initial_process,
            "science": initial_science,
        },
        "final": {
            "process": final_process,
            "total_loss_reduction": total_reduction,
            "expert_loss_reduction": expert_reduction,
            "science": final_science,
        },
        "history": history,
        "science_evaluations": science_evaluations,
        "device": {
            "backend": jax.default_backend(),
            "devices": [str(value) for value in runtime["devices"]],
            "memory": [_memory_report(value) for value in runtime["devices"]],
        },
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "checkpoint_written": False,
        "qualification_checkpoint_initializes_formal_training": False,
        "reconstructs_48_forcing_records": False,
        "executes_48_step_state_scan": False,
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": (
            "freeze_64_point_year_qualification"
            if passed
            else "diagnose_strong_baseline_learnability_failure"
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
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
