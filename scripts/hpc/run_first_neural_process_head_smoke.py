"""Bounded real-v6 V100 smoke for the first neural process-head A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend

EXPECTED = {
    "dataset_manifest": "5fc2c0691bfa53252f875f7646f3c1a5bea203c2759a6e947b26bb51dc13aa99",
    "acceptance_report": "8784ed42dc64ef77c9642f613eb2e5940635edc6e744a1af4ea8634293baa8e3",
    "statistics_json": "ab7f040a930071bd5604c17355a6493b23f3c76cba1e3084dcd5d5653101a4bf",
    "statistics_npz": "e88d7cec504c6051aab6c9feacb619dca94961730ffa18f840563a48fc815a3a",
}
PERMUTATION_ATOL = 1.0e-10
PERMUTATION_RTOL = 1.0e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected: str, name: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{name} hash drift: {actual} != {expected}")


def _stack_trees(jax: Any, values: list[Any]) -> Any:
    return jax.tree_util.tree_map(lambda *items: np.stack(items), *values)


def _finite_moments(
    values: list[np.ndarray],
    masks: list[np.ndarray],
    *,
    pft_axis: int | None = None,
    pft_indices: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.stack(values).astype(np.float64, copy=False)
    defined = np.stack(masks).astype(np.bool_, copy=False)
    if pft_axis is not None:
        moved = np.moveaxis(stacked, pft_axis + 1, 1)
        moved_defined = np.moveaxis(defined, pft_axis + 1, 1)
        selected = moved[:, pft_indices]
        selected_defined = moved_defined[:, pft_indices]
        pooled_shape = (-1, *selected.shape[2:])
        pooled_mean, pooled_scale = _finite_moments(
            [value for value in selected.reshape(pooled_shape)],
            [mask for mask in selected_defined.reshape(pooled_shape)],
        )
        mean = np.zeros(moved.shape[1:], dtype=np.float64)
        scale = np.ones(moved.shape[1:], dtype=np.float64)
        mean[np.asarray(pft_indices)] = pooled_mean
        scale[np.asarray(pft_indices)] = pooled_scale
        return np.moveaxis(mean, 0, pft_axis), np.moveaxis(scale, 0, pft_axis)
    count = np.sum(defined, axis=0)
    total = np.sum(np.where(defined, stacked, 0.0), axis=0, dtype=np.float64)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    residual = np.where(defined, stacked - mean, 0.0)
    variance = np.divide(
        np.sum(residual * residual, axis=0, dtype=np.float64),
        count,
        out=np.zeros_like(total),
        where=count > 0,
    )
    scale = np.sqrt(variance)
    scale = np.where((count > 1) & np.isfinite(scale) & (scale > 1.0e-12), scale, 1.0)
    return mean, scale


def _pft_shared_domain_scale(
    values: list[np.ndarray],
    masks: list[np.ndarray],
    element_pft_index: np.ndarray,
) -> np.ndarray:
    _mean, scale = _finite_moments(values, masks)
    flat_scale = scale.reshape(-1)
    pft_index = np.asarray(element_pft_index, dtype=np.int32).reshape(-1)
    slots = tuple(int(slot) for slot in np.unique(pft_index) if slot >= 0)
    if not slots:
        return np.maximum(scale, 1.0e-6)
    indices = tuple(np.flatnonzero(pft_index == slot) for slot in slots)
    if len({len(index) for index in indices}) != 1:
        raise ValueError("PFT domain normalization layout is not pairwise")
    stacked = np.stack(values).astype(np.float64, copy=False)
    defined = np.stack(masks).astype(np.bool_, copy=False)
    paired_values = np.stack(tuple(stacked[:, index] for index in indices), axis=1)
    paired_defined = np.stack(tuple(defined[:, index] for index in indices), axis=1)
    flattened_shape = (-1, paired_values.shape[-1])
    _paired_mean, paired_scale = _finite_moments(
        [value for value in paired_values.reshape(flattened_shape)],
        [mask for mask in paired_defined.reshape(flattened_shape)],
    )
    for index in indices:
        flat_scale[index] = paired_scale
    return np.maximum(flat_scale.reshape(scale.shape), 1.0e-6)


def _report_gradient(jax: Any, gradient: Any) -> dict[str, Any]:
    leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(gradient)]
    return {
        "leaves": len(leaves),
        "finite": all(np.all(np.isfinite(leaf)) for leaf in leaves),
        "norm": float(np.sqrt(sum(np.sum(leaf * leaf, dtype=np.float64) for leaf in leaves))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--statistics-json", type=Path, required=True)
    parser.add_argument("--statistics-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tiny-fit-updates", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size != 16:
        raise ValueError("the first-candidate smoke batch is frozen at 16")
    if args.tiny_fit_updates not in {0, 100}:
        raise ValueError("tiny fit must be disabled or use the frozen 100 updates")

    for name, path in (
        ("dataset_manifest", args.dataset_manifest),
        ("acceptance_report", args.acceptance_report),
        ("statistics_json", args.statistics_json),
        ("statistics_npz", args.statistics_npz),
    ):
        _verify(path.resolve(), EXPECTED[name], name)

    jax, devices = initialize_gpu_backend()
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from research.daily_coarse_graining.canonical_training_run import (
        _adam_init,
        _adam_update,
    )
    from research.daily_coarse_graining.daily_markov_contract import (
        daily_markov_contract_from_metadata,
        load_markov_shard,
    )
    from research.daily_coarse_graining.daily_operator.conservative_update import (
        apply_conservative_update,
        validate_conservative_result,
    )
    from research.daily_coarse_graining.daily_operator.neural_architecture import (
        AXIAL_CANDIDATE,
        FACTORIZED_CONTROL,
        NeuralInputNormalization,
        NeuralOutputNormalization,
        assemble_neural_process_context,
        build_neural_output_schema,
        build_process_axis_layout,
        initialize_neural_process_parameters,
        neural_process_outputs,
        parameter_count,
    )
    from research.daily_coarse_graining.daily_operator.neural_process_head import (
        NeuralProcessHead,
    )
    from research.daily_coarse_graining.daily_operator.process_derivations import (
        NativeForcingFieldLayout,
    )
    from research.daily_coarse_graining.daily_operator.training_data import (
        DailyProcessTrainingAdapter,
    )
    from research.daily_coarse_graining.daily_operator.training_loss import (
        masked_equal_label_macro_mse,
    )
    from research.daily_coarse_graining.typed_sidecar import (
        load_typed_sidecar_contract,
        read_typed_sidecar_shard,
    )

    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    acceptance = json.loads(args.acceptance_report.read_text(encoding="utf-8"))
    if acceptance.get("status") != "passed":
        raise ValueError("v6 acceptance report is not passed")
    contract = daily_markov_contract_from_metadata(manifest["markov_contract"])
    adapter = DailyProcessTrainingAdapter.from_contract(contract)
    typed_contract = load_typed_sidecar_contract()
    item = next(
        shard
        for shard in manifest["shards"]
        if shard["spatial_split"] == "train" and shard["temporal_split"] == "train"
    )
    root = args.dataset_manifest.parent
    base_path = (root / item["shard"]).resolve()
    typed_path = (root / item["typed_shard"]).resolve()
    _verify(base_path, item["shard_sha256"], "selected base shard")
    _verify(typed_path, item["typed_shard_sha256"], "selected typed shard")
    parent = load_markov_shard(base_path, contract=contract)
    typed_values, typed_defined, day_index = read_typed_sidecar_shard(
        typed_path,
        contract=typed_contract,
        expected_days=parent.days,
    )
    if not np.array_equal(day_index, parent.day_index):
        raise ValueError("selected coherent base/typed day index drift")

    examples = []
    for row in range(args.batch_size):
        examples.append(
            adapter.assemble(
                parent.sample(row),
                supplemental_values={name: value[row] for name, value in typed_values.items()},
                supplemental_defined={name: value[row] for name, value in typed_defined.items()},
            )
        )
    assembled = [adapter.input_assembler.assemble(example.inputs) for example in examples]
    first = assembled[0]
    typed_axes = {field.path: field.axes[1:] for field in adapter.typed_contract.fields}
    schema = build_neural_output_schema(
        examples[0].targets.supervision,
        field_labels=adapter.field_labels(),
        typed_axes=typed_axes,
        active_pft_indices=adapter.active_pft_indices,
    )
    axis_layout = build_process_axis_layout(adapter.input_assembler)

    with np.load(args.statistics_npz, allow_pickle=False) as statistics:
        input_normalization = NeuralInputNormalization(
            continuous_mean=statistics["state__mean"],
            continuous_scale=statistics["state__scale"],
            forcing_mean=statistics["forcing_native__mean"],
            forcing_scale=statistics["forcing_native__scale"],
            static_mean=statistics["landpoint_static__mean"],
            static_scale=statistics["landpoint_static__scale"],
            annual_mean=statistics["annual_conditions__mean"],
            annual_scale=statistics["annual_conditions__scale"],
            water_scale=_pft_shared_domain_scale(
                [np.asarray(value.state.water) for value in assembled],
                [np.asarray(value.state_defined.water) for value in assembled],
                axis_layout.domains[0].element_pft_index,
            ),
            carbon_scale=_pft_shared_domain_scale(
                [np.asarray(value.state.carbon) for value in assembled],
                [np.asarray(value.state_defined.carbon) for value in assembled],
                axis_layout.domains[1].element_pft_index,
            ),
            thermal_scale=_pft_shared_domain_scale(
                [np.asarray(value.state.thermal) for value in assembled],
                [np.asarray(value.state_defined.thermal) for value in assembled],
                axis_layout.domains[2].element_pft_index,
            ),
        )
    output_mean = {}
    output_scale = {}
    loss_scale = {}
    for field in schema.fields:
        mean, scale = _finite_moments(
            [np.asarray(example.targets.supervision[field.name].value) for example in examples],
            [np.asarray(example.targets.supervision[field.name].defined) for example in examples],
            pft_axis=field.pft_axis,
            pft_indices=(
                tuple(adapter.active_pft_indices)
                if field.full_pft_axis
                else tuple(range(field.shape[field.pft_axis]))
                if field.pft_axis is not None
                else ()
            ),
        )
        output_mean[field.name] = mean
        output_scale[field.name] = scale
        loss_scale[field.name] = (
            np.ones(field.shape, dtype=np.float64)
            if field.constraint in {"unit_interval", "signed_unit"}
            else scale
        )
    output_normalization = NeuralOutputNormalization(mean=output_mean, scale=output_scale)

    parameters = initialize_neural_process_parameters(
        jax.random.PRNGKey(20260807),
        schema=schema,
        axis_layout=axis_layout,
        inputs=first,
    )
    trainable_count = parameter_count(parameters)
    if not 4_000_000 <= trainable_count <= 6_000_000:
        raise ValueError("candidate parameter count is outside the frozen interval")
    batch_inputs = _stack_trees(jax, assembled)
    batch_targets = _stack_trees(
        jax,
        [example.targets.supervision for example in examples],
    )
    field_labels = adapter.field_labels()

    def one_loss_result(current_parameters, current_inputs, current_targets, arm):
        context = assemble_neural_process_context(
            current_parameters,
            current_inputs,
            normalization=input_normalization,
            axis_layout=axis_layout,
        )
        predicted = neural_process_outputs(
            current_parameters,
            context,
            schema=schema,
            normalization=output_normalization,
            arm=arm,
        )
        return masked_equal_label_macro_mse(
            predicted,
            current_targets,
            scales=loss_scale,
            field_labels=field_labels,
        )

    def one_loss(current_parameters, current_inputs, current_targets, arm):
        return one_loss_result(
            current_parameters,
            current_inputs,
            current_targets,
            arm,
        ).total

    arms = {}
    for arm in (AXIAL_CANDIDATE, FACTORIZED_CONTROL):
        def batch_loss(current_parameters):
            return jnp.mean(
                jax.vmap(lambda x, y: one_loss(current_parameters, x, y, arm))(
                    batch_inputs,
                    batch_targets,
                )
            )

        started = time.perf_counter()
        loss, gradient = jax.jit(jax.value_and_grad(batch_loss))(parameters)
        jax.block_until_ready((loss, gradient))
        reverse_seconds = time.perf_counter() - started
        tangent = jax.tree_util.tree_map(
            lambda value: jnp.ones_like(value) / np.sqrt(float(trainable_count)),
            parameters,
        )
        started = time.perf_counter()
        primal, forward = jax.jit(
            lambda current, direction: jax.jvp(batch_loss, (current,), (direction,))
        )(parameters, tangent)
        jax.block_until_ready((primal, forward))
        forward_seconds = time.perf_counter() - started
        details = jax.jit(
            jax.vmap(lambda x, y: one_loss_result(parameters, x, y, arm))
        )(batch_inputs, batch_targets)
        jax.block_until_ready(details)
        label_mse = {
            name: float(np.mean(np.asarray(value)))
            for name, value in details.label_mse.items()
        }
        field_mse = {
            name: float(np.mean(np.asarray(value)))
            for name, value in details.field_mse.items()
        }
        arms[arm] = {
            "parameter_count": trainable_count,
            "loss": float(loss),
            "loss_finite": bool(np.isfinite(np.asarray(loss))),
            "reverse_gradient": _report_gradient(jax, gradient),
            "forward_directional_derivative": float(forward),
            "forward_derivative_finite": bool(np.isfinite(np.asarray(forward))),
            "jit_reverse_seconds": reverse_seconds,
            "jit_forward_seconds": forward_seconds,
            "largest_label_mse": [
                {"label": name, "mse": value}
                for name, value in sorted(
                    label_mse.items(), key=lambda item: item[1], reverse=True
                )[:10]
            ],
            "largest_field_mse": [
                {"field": name, "mse": value}
                for name, value in sorted(
                    field_mse.items(), key=lambda item: item[1], reverse=True
                )[:10]
            ],
        }
        if args.tiny_fit_updates:
            optimizer_state = _adam_init(parameters)

            @jax.jit
            def tiny_fit_step(current_parameters, current_optimizer_state):
                current_loss, current_gradient = jax.value_and_grad(batch_loss)(
                    current_parameters
                )
                squared_norm = sum(
                    jnp.sum(leaf * leaf, dtype=jnp.float64)
                    for leaf in jax.tree_util.tree_leaves(current_gradient)
                )
                gradient_norm = jnp.sqrt(squared_norm)
                clip_scale = jnp.minimum(1.0, 1.0 / jnp.maximum(gradient_norm, 1.0e-30))
                clipped_gradient = jax.tree_util.tree_map(
                    lambda leaf: leaf * clip_scale,
                    current_gradient,
                )
                next_parameters, next_optimizer_state = _adam_update(
                    current_parameters,
                    clipped_gradient,
                    current_optimizer_state,
                    learning_rate=3.0e-4,
                )
                next_parameters = jax.tree_util.tree_map(
                    lambda updated, previous: updated - 3.0e-4 * 1.0e-5 * previous,
                    next_parameters,
                    current_parameters,
                )
                return next_parameters, next_optimizer_state, current_loss

            fitted_parameters = parameters
            trajectory = {"0": float(loss)}
            checkpoints = {1, 10, 25, 50, int(args.tiny_fit_updates)}
            fit_finite = True
            for update in range(1, int(args.tiny_fit_updates) + 1):
                fitted_parameters, optimizer_state, fit_loss = tiny_fit_step(
                    fitted_parameters,
                    optimizer_state,
                )
                jax.block_until_ready(fit_loss)
                fit_finite &= bool(np.isfinite(np.asarray(fit_loss)))
                if update in checkpoints:
                    trajectory[str(update)] = float(fit_loss)
            final_loss = batch_loss(fitted_parameters)
            jax.block_until_ready(final_loss)
            fit_finite &= bool(np.isfinite(np.asarray(final_loss)))
            reduction = 1.0 - float(final_loss) / float(loss)
            arms[arm]["tiny_fit"] = {
                "status": (
                    "passed" if fit_finite and reduction >= 0.20 else "failed"
                ),
                "updates": int(args.tiny_fit_updates),
                "initial_loss": float(loss),
                "final_loss": float(final_loss),
                "relative_reduction": reduction,
                "finite": fit_finite,
                "loss_trajectory": trajectory,
                "checkpoint_written": False,
            }

    forcing_fields = contract.native_forcing.fields
    candidate_head = NeuralProcessHead(
        arm=AXIAL_CANDIDATE,
        contract=contract,
        input_assembler=adapter.input_assembler,
        schema=schema,
        axis_layout=axis_layout,
        input_normalization=input_normalization,
        output_normalization=output_normalization,
        forcing_layout=NativeForcingFieldLayout(
            forcing_fields.index("Rainf"),
            forcing_fields.index("Rainf") + 1,
            forcing_fields.index("Snowf"),
            forcing_fields.index("Snowf") + 1,
            forcing_fields.index("Tair"),
            forcing_fields.index("Tair") + 1,
        ),
        forcing_indices={name: index for index, name in enumerate(forcing_fields)},
    )
    first_context = candidate_head.assemble_context(parameters, first)
    first_prediction = candidate_head(parameters, first_context, first)
    one_day = apply_conservative_update(first, first_prediction)
    validate_conservative_result(one_day)

    def single_outputs(current_input):
        current_context = assemble_neural_process_context(
            parameters,
            current_input,
            normalization=input_normalization,
            axis_layout=axis_layout,
        )
        return neural_process_outputs(
            parameters,
            current_context,
            schema=schema,
            normalization=output_normalization,
            arm=AXIAL_CANDIDATE,
        )

    def permute_domain(value, layout):
        source = np.asarray(value)
        result = source.copy()
        first_slot = layout.element_pft_index == 0
        second_slot = layout.element_pft_index == 1
        if np.count_nonzero(first_slot) != np.count_nonzero(second_slot):
            raise ValueError(f"{layout.name} PFT element layout is not pairwise")
        result[first_slot] = source[second_slot]
        result[second_slot] = source[first_slot]
        return result

    permutation = np.asarray([1, 0])
    permuted_pft = first.pft._replace(
        **{
            name: np.asarray(getattr(first.pft, name))[permutation]
            for name in first.pft._fields
        }
    )
    permuted_state = first.state._replace(
        **{
            layout.name: permute_domain(getattr(first.state, layout.name), layout)
            for layout in axis_layout.domains
        }
    )
    permuted_defined = first.state_defined._replace(
        **{
            layout.name: permute_domain(getattr(first.state_defined, layout.name), layout)
            for layout in axis_layout.domains
        }
    )
    permuted_input = first._replace(
        pft=permuted_pft,
        state=permuted_state,
        state_defined=permuted_defined,
    )
    compiled_outputs = jax.jit(single_outputs)
    original_outputs = compiled_outputs(first)
    permuted_outputs = compiled_outputs(permuted_input)
    jax.block_until_ready((original_outputs, permuted_outputs))
    permutation_errors = {}
    permutation_tolerance_ratios = {}
    for field in schema.fields:
        original = np.asarray(original_outputs[field.name])
        expected = original.copy()
        if field.pft_axis is not None:
            if field.full_pft_axis:
                slices = [slice(None)] * expected.ndim
                first_slice = list(slices)
                second_slice = list(slices)
                first_slice[field.pft_axis] = adapter.active_pft_indices[0]
                second_slice[field.pft_axis] = adapter.active_pft_indices[1]
                expected[tuple(first_slice)] = original[tuple(second_slice)]
                expected[tuple(second_slice)] = original[tuple(first_slice)]
            else:
                expected = np.flip(expected, axis=field.pft_axis)
        actual = np.asarray(permuted_outputs[field.name])
        difference = np.abs(actual - expected)
        tolerance = PERMUTATION_ATOL + PERMUTATION_RTOL * np.abs(expected)
        permutation_errors[field.name] = float(np.max(difference)) if actual.size else 0.0
        permutation_tolerance_ratios[field.name] = (
            float(np.max(difference / tolerance)) if actual.size else 0.0
        )
    permutation_maximum = max(permutation_errors.values())
    permutation_tolerance_ratio = max(permutation_tolerance_ratios.values())
    permutation_worst_fields = [
        {
            "field": name,
            "maximum_absolute_error": permutation_errors[name],
            "maximum_tolerance_ratio": ratio,
        }
        for name, ratio in sorted(
            permutation_tolerance_ratios.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
    ]

    inactive_pft = first.pft._replace(active_pft_mask=np.asarray([True, False], dtype=np.bool_))
    inactive_input = first._replace(pft=inactive_pft)
    perturbed_pft = inactive_pft._replace(
        pft_state=np.asarray(inactive_pft.pft_state).copy(),
        pft_parameters=np.asarray(inactive_pft.pft_parameters).copy(),
        pft_traits=np.asarray(inactive_pft.pft_traits).copy(),
        pft_fraction=np.asarray(inactive_pft.pft_fraction).copy(),
    )
    perturbed_pft.pft_state[1] = 1.0e6
    perturbed_pft.pft_parameters[1] = -1.0e6
    perturbed_pft.pft_traits[1] = 5.0e5
    perturbed_pft.pft_fraction[1] = 1.0e4
    perturbed_state = inactive_input.state
    for layout in axis_layout.domains:
        value = np.asarray(getattr(perturbed_state, layout.name)).copy()
        value[layout.element_pft_index == 1] = 1.0e8
        perturbed_state = perturbed_state._replace(**{layout.name: value})
    perturbed_input = inactive_input._replace(pft=perturbed_pft, state=perturbed_state)
    inactive_outputs = compiled_outputs(inactive_input)
    perturbed_outputs = compiled_outputs(perturbed_input)
    jax.block_until_ready((inactive_outputs, perturbed_outputs))
    isolation_errors = {
        name: float(np.max(np.abs(np.asarray(perturbed_outputs[name]) - np.asarray(value))))
        if np.asarray(value).size
        else 0.0
        for name, value in inactive_outputs.items()
    }
    isolation_maximum = max(isolation_errors.values())
    pft_gate = permutation_tolerance_ratio <= 1.0 and isolation_maximum <= 1.0e-10

    differentiable = all(
        value["loss_finite"]
        and value["reverse_gradient"]["finite"]
        and value["forward_derivative_finite"]
        for value in arms.values()
    )
    tiny_fit_passed = not args.tiny_fit_updates or all(
        value["tiny_fit"]["status"] == "passed" for value in arms.values()
    )
    water_residual = float(one_day.water.budget_residual)
    carbon_residual = float(one_day.carbon.budget_residual)
    water_minimum = float(jnp.min(one_day.water.inventory))
    carbon_minimum = float(jnp.min(one_day.carbon.inventory))
    conservative = (
        np.isfinite(water_residual)
        and np.isfinite(carbon_residual)
        and abs(water_residual) <= 2.0e-12
        and abs(carbon_residual) <= 2.0e-12
        and water_minimum >= -2.0e-12
        and carbon_minimum >= -2.0e-12
    )
    passed = differentiable and conservative and pft_gate and tiny_fit_passed
    report = {
        "schema_version": "first_neural_process_head_real_v6_smoke_v1",
        "status": "passed" if passed else "failed",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
        "jax_version": jax.__version__,
        "dtype": "float64",
        "batch_size": args.batch_size,
        "tiny_fit_policy": {
            "status": (
                "passed"
                if args.tiny_fit_updates and tiny_fit_passed
                else "not_run"
                if not args.tiny_fit_updates
                else "failed"
            ),
            "updates": int(args.tiny_fit_updates),
            "optimizer": "adamw",
            "learning_rate": 3.0e-4,
            "weight_decay": 1.0e-5,
            "global_gradient_norm_clip": 1.0,
            "minimum_relative_loss_reduction": 0.20,
            "checkpoint_written": False,
        },
        "data": {
            "dataset_manifest_sha256": EXPECTED["dataset_manifest"],
            "acceptance_report_sha256": EXPECTED["acceptance_report"],
            "statistics_json_sha256": EXPECTED["statistics_json"],
            "statistics_npz_sha256": EXPECTED["statistics_npz"],
            "landpoint_id": item["landpoint_id"],
            "year": item["year"],
            "base_shard_sha256": item["shard_sha256"],
            "typed_shard_sha256": item["typed_shard_sha256"],
            "rows": list(range(args.batch_size)),
        },
        "output_schema": {
            "fields": len(schema.fields),
            "labels": len(schema.labels),
            "elements_per_sample": int(
                sum(np.asarray(target.value).size for target in examples[0].targets.supervision.values())
            ),
        },
        "semantic_axis_layout": {
            "field_count": len(axis_layout.field_names),
            "role_names": list(axis_layout.role_names),
            "domain_shapes": {
                layout.name: list(layout.pool_matrix.shape) for layout in axis_layout.domains
            },
            "carbon_32l_pft_axis": 2,
        },
        "pft_gates": {
            "status": "passed" if pft_gate else "failed",
            "permutation_maximum_absolute_error": permutation_maximum,
            "permutation_maximum_tolerance_ratio": permutation_tolerance_ratio,
            "permutation_atol": PERMUTATION_ATOL,
            "permutation_rtol": PERMUTATION_RTOL,
            "permutation_worst_fields": permutation_worst_fields,
            "inactive_slot_isolation_maximum_absolute_error": isolation_maximum,
        },
        "normalization": {
            "accepted_base_statistics": True,
            "typed_output_statistics": "selected_16_row_smoke_only",
            "pft_axis_moments": "shared_across_active_pft_slots",
            "bounded_target_loss_scale": "unit_range",
            "formal_training_blocked": True,
            "reason": "accepted v6 statistics artifact contains no typed-field train/train moments",
        },
        "arms": arms,
        "one_day_construction": {
            "status": "passed" if conservative else "failed",
            "scope": "conservative fast update before canonical retained-tail state comparison",
            "water_budget_residual": water_residual,
            "carbon_budget_residual": carbon_residual,
            "water_minimum_inventory": water_minimum,
            "carbon_minimum_inventory": carbon_minimum,
            "retained_fast_component_count": len(first_prediction.retained_boundary.fast_state_fields),
            "named_output_count": len(first_prediction.named_outputs),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
