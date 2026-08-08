from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_markov_contract import (
    MarkovShard,
    daily_markov_contract_from_metadata,
)
from research.daily_coarse_graining.daily_operator.audit import (
    audit_inference_source_graph,
)
from research.daily_coarse_graining.daily_operator.conservative_update import (
    apply_conservative_update,
    validate_conservative_result,
)
from research.daily_coarse_graining.daily_operator.model import bind_daily_operator
from research.daily_coarse_graining.daily_operator.neural_architecture import (
    AXIAL_CANDIDATE,
    FACTORIZED_CONTROL,
    assemble_neural_process_context,
    build_neural_output_schema,
    build_process_axis_layout,
    initialize_neural_process_parameters,
    neural_process_outputs,
    parameter_count,
    unit_input_normalization,
    unit_output_normalization,
)
from research.daily_coarse_graining.daily_operator.neural_process_head import (
    NeuralProcessHead,
)
from research.daily_coarse_graining.daily_operator.process_derivations import (
    NativeForcingFieldLayout,
)
from research.daily_coarse_graining.daily_operator.training_contract import (
    FirstCandidateContract,
    load_first_candidate_contract,
)
from research.daily_coarse_graining.daily_operator.training_data import (
    DailyProcessTrainingAdapter,
    MaskedProcessTarget,
)
from research.daily_coarse_graining.daily_operator.training_loss import (
    masked_equal_label_macro_mse,
    one_day_acceptance_metrics,
    unit_process_loss_scales,
    validate_local_one_day_hard_gates,
)
from research.daily_coarse_graining.daily_operator.training_statistics import (
    MaskedMoments,
    fit_process_training_statistics,
    load_process_training_statistics,
    vectorized_process_targets,
    vectorized_semantic_state,
    write_process_training_statistics,
)
from research.daily_coarse_graining.daily_operator.types import (
    ConservativeDomainResult,
    ConservativeUpdateResult,
    DailyOperatorResult,
    RetainedTailResult,
    StateDefinedMasks,
    StateFeatureGroups,
)
from research.daily_coarse_graining.markov_dataset import (
    FiniteColumnStatistics,
    MarkovDatasetIndex,
    MarkovShardRef,
    TrainingStatistics,
)
from research.daily_coarse_graining.typed_sidecar import (
    TypedSidecarIndex,
    TypedSidecarShardRef,
    write_typed_sidecar_shard,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_MANIFEST = (
    ROOT
    / "outputs"
    / "research"
    / "daily_coarse_graining"
    / "dataset-manifest-v5"
    / "dataset_manifest.json"
)


def _markov_contract():
    metadata = json.loads(CONTRACT_MANIFEST.read_text(encoding="utf-8"))[
        "markov_contract"
    ]
    contract = daily_markov_contract_from_metadata(metadata)
    assert (
        contract.sha256
        == "6813065b51e95a2ec6663d8bc92f5336148704aea502632a05fd51d7e2e9d442"
    )
    return contract


def _set_condition(vector, leaves, name, value):
    leaf = next(item for item in leaves if item.name == name)
    vector[leaf.start : leaf.stop] = np.asarray(value, dtype=np.float64).reshape(-1)


def _synthetic_parent(adapter: DailyProcessTrainingAdapter):
    contract = adapter.contract
    state = np.zeros((contract.continuous_state_width,), dtype=np.float64)
    veget = next(
        leaf
        for leaf in contract.state_leaves
        if leaf.key == "slowproc_stomate_previous_step_state.veget_max"
    )
    state[veget.start : veget.stop] = np.asarray([0.5, 0.5], dtype=np.float64)
    discrete = {
        leaf.key: np.zeros(leaf.shape, dtype=np.dtype(leaf.dtype))
        for leaf in contract.discrete_leaves
    }
    parameters = np.zeros(
        (contract.static_conditions.parameter_width,), dtype=np.float64
    )
    _set_condition(
        parameters,
        contract.static_conditions.parameter_leaves,
        "maint_resp_slope",
        np.ones((14, 3), dtype=np.float64),
    )
    static = np.zeros(
        (contract.static_conditions.landpoint_static_width,), dtype=np.float64
    )
    _set_condition(
        static,
        contract.static_conditions.landpoint_static_leaves,
        "hydrol_humcste",
        np.ones((14,), dtype=np.float64),
    )
    _set_condition(
        static,
        contract.static_conditions.landpoint_static_leaves,
        "diaglev",
        np.arange(1.0, 12.0, dtype=np.float64),
    )
    _set_condition(
        static,
        contract.static_conditions.landpoint_static_leaves,
        "clay_frac",
        np.asarray([0.3], dtype=np.float64),
    )
    annual = np.zeros(
        (contract.static_conditions.annual_condition_width,), dtype=np.float64
    )
    parent = {
        "state": state,
        "fast_day_target": np.zeros(
            (contract.fast_day_target_width,), dtype=np.float64
        ),
        "forcing_native": np.zeros(
            (
                contract.native_forcing.source_records_per_day,
                contract.native_forcing.width,
            ),
            dtype=np.float64,
        ),
        "parameters": parameters,
        "landpoint_static": static,
        "annual_conditions": annual,
        "next_state": state.copy(),
        "diagnostics": np.zeros((contract.diagnostic_width,), dtype=np.float64),
        "year": np.asarray(1961, dtype=np.int32),
        "day_index": np.asarray(2, dtype=np.int32),
        "discrete_state": discrete,
        "next_discrete_state": {name: value.copy() for name, value in discrete.items()},
    }
    values = {
        field.path: np.zeros(field.feature_shape, dtype=np.float64)
        for field in adapter.typed_contract.fields
    }
    defined = {
        name: np.ones(value.shape, dtype=np.bool_) for name, value in values.items()
    }
    return parent, values, defined


def test_first_candidate_contract_freezes_data_architecture_and_budget(tmp_path):
    contract = load_first_candidate_contract()
    assert contract.candidate_id == "axial_process_operator_v1"
    assert contract.control_id == "factorized_process_control_v1"
    assert contract.payload["training"]["maximum_updates"] == 10000
    assert contract.payload["data"]["historical_v5_allowed"] is False

    changed = dict(contract.payload)
    changed["training"] = dict(changed["training"], maximum_updates=10001)
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="contract hash mismatch"):
        load_first_candidate_contract(path)


def test_host_training_adapter_is_excluded_without_entering_inference_graph():
    graph = audit_inference_source_graph()
    assert graph["passed"], graph["violations"]
    assert graph["host_only_training_files"] == [
        "training_contract.py",
        "training_data.py",
        "training_loss.py",
        "training_statistics.py",
    ]


def test_v6_adapter_builds_only_named_input_and_process_targets():
    adapter = DailyProcessTrainingAdapter.from_contract(_markov_contract())
    parent, values, defined = _synthetic_parent(adapter)
    example = adapter.assemble(
        parent,
        supplemental_values=values,
        supplemental_defined=defined,
    )

    assert example.inputs.forcing.values.shape[0] == 5
    assert example.inputs.pft.pft_state.shape[0] == 2
    assert example.inputs.pft.pft_parameters.shape == (2, 6)
    assert example.inputs.pft.pft_traits.shape == (2, 10)
    assert example.inputs.static_registry.rprof.shape == (1, 2)
    assert example.targets.supervision[
        "inventory.carbon.carbon_32l.outgoing_fraction"
    ].value.shape == (1, 3, 2, 32)
    assert len(example.targets.supervision) == 101
    assert len(set(adapter.field_labels().values())) == 40
    assert set(example.targets.supervision) == set(adapter.field_labels())
    assert all("fast_day_target" not in name for name in example.targets.supervision)
    assert not hasattr(example.inputs, "fast_day_target")
    assert example.targets.retained_boundary.fast_state_fields


def _synthetic_shard_and_typed(adapter, *, days=2):
    parent, values, defined = _synthetic_parent(adapter)
    state_trajectory = np.stack([parent["state"]] * (days + 1))
    discrete = {
        name: np.stack([value] * (days + 1))
        for name, value in parent["discrete_state"].items()
    }
    shard = MarkovShard(
        state_trajectory=state_trajectory,
        fast_day_target=np.stack([parent["fast_day_target"]] * days),
        forcing_native=np.stack([parent["forcing_native"]] * days),
        forcing_record_indices=np.zeros(
            (days, adapter.contract.native_forcing.source_records_per_day),
            dtype=np.int64,
        ),
        parameters=parent["parameters"],
        landpoint_static=parent["landpoint_static"],
        annual_conditions=parent["annual_conditions"],
        diagnostics=np.stack([parent["diagnostics"]] * days),
        year=np.asarray(1961, dtype=np.int32),
        day_index=np.arange(2, days + 2, dtype=np.int32),
        discrete_trajectories=discrete,
    )
    typed_values = {name: np.stack([value] * days) for name, value in values.items()}
    typed_defined = {name: np.stack([value] * days) for name, value in defined.items()}
    return shard, typed_values, typed_defined


def test_vectorized_process_statistics_targets_match_single_day_adapter():
    adapter = DailyProcessTrainingAdapter.from_contract(_markov_contract())
    shard, typed_values, typed_defined = _synthetic_shard_and_typed(adapter)
    targets = vectorized_process_targets(
        shard,
        typed_values,
        typed_defined,
        adapter,
    )
    semantic = vectorized_semantic_state(shard, adapter)
    assert set(targets) == set(adapter.field_labels())
    assert set(semantic) == {"water", "carbon", "thermal"}

    for row in range(shard.days):
        expected = adapter.assemble(
            shard.sample(row),
            supplemental_values={name: value[row] for name, value in typed_values.items()},
            supplemental_defined={name: value[row] for name, value in typed_defined.items()},
        )
        assembled = adapter.input_assembler.assemble(expected.inputs)
        for name, target in targets.items():
            np.testing.assert_array_equal(
                target.value[row],
                expected.targets.supervision[name].value,
                err_msg=name,
            )
            np.testing.assert_array_equal(
                target.defined[row],
                expected.targets.supervision[name].defined,
                err_msg=name,
            )
        for name, target in semantic.items():
            np.testing.assert_array_equal(target.value[row], getattr(assembled.state, name))
            np.testing.assert_array_equal(
                target.defined[row], getattr(assembled.state_defined, name)
            )


def test_masked_moments_merge_and_share_pft_slots():
    values = np.asarray(
        [
            [[1.0, 10.0], [5.0, 50.0]],
            [[3.0, 30.0], [7.0, 70.0]],
        ]
    )
    defined = np.ones_like(values, dtype=np.bool_)
    direct = MaskedMoments((2, 2), pft_axis=0, pft_indices=(0, 1))
    direct.update(values, defined, chunk_rows=1)

    left = MaskedMoments((2, 2), pft_axis=0, pft_indices=(0, 1))
    right = MaskedMoments((2, 2), pft_axis=0, pft_indices=(0, 1))
    left.update(values[:1], defined[:1])
    right.update(values[1:], defined[1:])
    left.merge(right)

    expected = direct.finalize(minimum_scale=1.0e-12)
    actual = left.finalize(minimum_scale=1.0e-12)
    for member in ("count", "mean", "m2", "variance", "scale"):
        np.testing.assert_allclose(getattr(actual, member), getattr(expected, member))
    np.testing.assert_allclose(actual.mean[0], actual.mean[1])
    np.testing.assert_allclose(actual.scale[0], actual.scale[1])
    np.testing.assert_allclose(actual.mean[0], np.asarray([4.0, 40.0]))


def _finite_statistics(shape, count):
    return FiniteColumnStatistics(
        count=np.full(shape, count, dtype=np.uint64),
        mean=np.zeros(shape, dtype=np.float64),
        variance=np.zeros(shape, dtype=np.float64),
        scale=np.ones(shape, dtype=np.float64),
    )


def test_process_statistics_artifact_fits_loads_and_rejects_hash_drift(tmp_path):
    adapter = DailyProcessTrainingAdapter.from_contract(_markov_contract())
    shard, typed_values, typed_defined = _synthetic_shard_and_typed(adapter)
    base_path = tmp_path / "base.npz"
    base_arrays = {
        "state_trajectory": shard.state_trajectory,
        "fast_day_target": shard.fast_day_target,
        "forcing_native": shard.forcing_native,
        "forcing_record_indices": shard.forcing_record_indices,
        "parameters": shard.parameters,
        "landpoint_static": shard.landpoint_static,
        "annual_conditions": shard.annual_conditions,
        "diagnostics": shard.diagnostics,
        "year": shard.year,
        "day_index": shard.day_index,
    }
    base_arrays.update(
        {
            f"state_discrete__{name}": value
            for name, value in shard.discrete_trajectories.items()
        }
    )
    np.savez_compressed(base_path, **base_arrays)
    typed_path = tmp_path / "typed.npz"
    write_typed_sidecar_shard(
        typed_path,
        contract=adapter.typed_contract,
        day_index=shard.day_index,
        values=typed_values,
        defined=typed_defined,
    )

    parent_reference = MarkovShardRef(
        landpoint_id="fixture",
        year=1961,
        spatial_split="train",
        temporal_split="train",
        path=base_path,
        sha256="1" * 64,
        contract_sha256=adapter.contract.sha256,
    )
    parent_index = MarkovDatasetIndex(
        dataset_id="fixture-v6",
        teacher_git_head="fixture-head",
        contract_sha256=adapter.contract.sha256,
        shards=(parent_reference,),
    )
    typed_reference = TypedSidecarShardRef(
        parent=parent_reference,
        path=typed_path,
        sha256="2" * 64,
        day_count=shard.days,
    )
    manifest_sha256 = "3" * 64
    index = TypedSidecarIndex(
        contract=adapter.typed_contract,
        parent_index=parent_index,
        parent_manifest_sha256=manifest_sha256,
        sidecar_manifest_sha256=manifest_sha256,
        shards=(typed_reference,),
    )
    base_statistics = TrainingStatistics(
        dataset_id=parent_index.dataset_id,
        teacher_git_head=parent_index.teacher_git_head,
        contract_sha256=parent_index.contract_sha256,
        sample_count=shard.days,
        source_shards=("fixture",),
        observations_per_column={},
        arrays={
            "state": _finite_statistics((shard.state_trajectory.shape[1],), shard.days),
            "forcing_native": _finite_statistics(
                (shard.forcing_native.shape[-1],),
                shard.days * shard.forcing_native.shape[1],
            ),
            "landpoint_static": _finite_statistics(shard.landpoint_static.shape, shard.days),
            "annual_conditions": _finite_statistics(shard.annual_conditions.shape, shard.days),
        },
    )
    base_json_sha256 = "4" * 64
    base_npz_sha256 = "5" * 64
    candidate = FirstCandidateContract(
        payload={
            "data": {
                "dataset_manifest_sha256": manifest_sha256,
                "markov_contract_sha256": adapter.contract.sha256,
                "typed_contract_sha256": adapter.typed_contract.sha256,
                "training_statistics_json_sha256": base_json_sha256,
                "training_statistics_npz_sha256": base_npz_sha256,
            }
        },
        source_path=tmp_path / "candidate.json",
        sha256="6" * 64,
    )
    statistics = fit_process_training_statistics(
        index,
        adapter,
        base_statistics,
        candidate,
        base_statistics_json_sha256=base_json_sha256,
        base_statistics_npz_sha256=base_npz_sha256,
        chunk_rows=1,
    )
    assert statistics.sample_count == 2
    assert len(statistics.outputs) == 101
    constraints = {field.name: field.constraint for field in statistics.schema.fields}
    assert all(
        np.array_equal(statistics.loss_scales[name], np.ones(value.mean.shape))
        for name, value in statistics.outputs.items()
        if constraints[name] in {"unit_interval", "signed_unit"}
    )

    metadata_path = write_process_training_statistics(
        statistics,
        tmp_path / "process_statistics.json",
    )
    loaded = load_process_training_statistics(
        metadata_path,
        index=index,
        adapter=adapter,
        candidate_contract=candidate,
        base_statistics=base_statistics,
    )
    assert loaded.sample_count == statistics.sample_count
    assert set(loaded.outputs) == set(statistics.outputs)

    arrays_path = metadata_path.with_suffix(".npz")
    arrays_path.write_bytes(arrays_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="NPZ hash mismatch"):
        load_process_training_statistics(
            metadata_path,
            index=index,
            adapter=adapter,
            candidate_contract=candidate,
            base_statistics=base_statistics,
        )


def test_supplemental_contract_drift_fails_closed():
    adapter = DailyProcessTrainingAdapter.from_contract(_markov_contract())
    parent, values, defined = _synthetic_parent(adapter)
    values.pop(next(iter(values)))
    with pytest.raises(ValueError, match="supplemental fields"):
        adapter.assemble(
            parent,
            supplemental_values=values,
            supplemental_defined=defined,
        )


def test_process_loss_is_equal_weighted_by_label_and_differentiable():
    targets = {
        "large": MaskedProcessTarget(
            np.zeros((100,), dtype=np.float64),
            np.ones((100,), dtype=np.bool_),
        ),
        "small": MaskedProcessTarget(
            np.zeros((1,), dtype=np.float64),
            np.ones((1,), dtype=np.bool_),
        ),
    }
    predicted = {
        "large": jnp.ones((100,), dtype=jnp.float64),
        "small": jnp.full((1,), 3.0, dtype=jnp.float64),
    }
    scales = unit_process_loss_scales(targets)
    labels = {"large": "large_label", "small": "small_label"}
    result = masked_equal_label_macro_mse(
        predicted,
        targets,
        scales=scales,
        field_labels=labels,
    )
    assert float(result.label_mse["large_label"]) == pytest.approx(1.0)
    assert float(result.label_mse["small_label"]) == pytest.approx(9.0)
    assert float(result.total) == pytest.approx(5.0)

    gradient = jax.grad(
        lambda value: masked_equal_label_macro_mse(
            {"large": value, "small": predicted["small"]},
            targets,
            scales=scales,
            field_labels=labels,
        ).total
    )(predicted["large"])
    assert np.all(np.isfinite(np.asarray(gradient)))

    nonfinite = dict(predicted)
    nonfinite["small"] = jnp.asarray([jnp.nan], dtype=jnp.float64)
    failed = masked_equal_label_macro_mse(
        nonfinite,
        targets,
        scales=scales,
        field_labels=labels,
    )
    assert not np.isfinite(float(failed.total))

    masked_undefined = {
        "large": MaskedProcessTarget(
            np.concatenate((np.zeros((99,), dtype=np.float64), np.asarray([np.nan]))),
            np.concatenate((np.ones((99,), dtype=np.bool_), np.asarray([False]))),
        ),
        "small": targets["small"],
    }
    masked_result = masked_equal_label_macro_mse(
        predicted,
        masked_undefined,
        scales=scales,
        field_labels=labels,
    )
    masked_gradient = jax.grad(
        lambda value: masked_equal_label_macro_mse(
            {"large": value, "small": predicted["small"]},
            masked_undefined,
            scales=scales,
            field_labels=labels,
        ).total
    )(predicted["large"])
    assert np.isfinite(float(masked_result.total))
    assert np.all(np.isfinite(np.asarray(masked_gradient)))


def test_first_candidate_and_matched_control_share_formal_parameter_tree_and_outputs():
    adapter = DailyProcessTrainingAdapter.from_contract(_markov_contract())
    parent, values, defined = _synthetic_parent(adapter)
    example = adapter.assemble(
        parent,
        supplemental_values=values,
        supplemental_defined=defined,
    )
    assembled = adapter.input_assembler.assemble(example.inputs)
    schema = build_neural_output_schema(
        example.targets.supervision,
        field_labels=adapter.field_labels(),
        typed_axes={field.path: field.axes[1:] for field in adapter.typed_contract.fields},
        active_pft_indices=adapter.active_pft_indices,
    )
    axis_layout = build_process_axis_layout(adapter.input_assembler)
    parameters = initialize_neural_process_parameters(
        jax.random.PRNGKey(20260807),
        schema=schema,
        axis_layout=axis_layout,
        inputs=assembled,
    )
    count = parameter_count(parameters)
    assert 4_000_000 <= count <= 6_000_000
    assert len(schema.fields) == 101
    assert len(schema.labels) == 40

    input_normalization = unit_input_normalization(assembled)
    context = assemble_neural_process_context(
        parameters,
        assembled,
        normalization=input_normalization,
        axis_layout=axis_layout,
    )
    output_normalization = unit_output_normalization(schema)
    predictions = {
        arm: neural_process_outputs(
            parameters,
            context,
            schema=schema,
            normalization=output_normalization,
            arm=arm,
        )
        for arm in (AXIAL_CANDIDATE, FACTORIZED_CONTROL)
    }
    expected_size = sum(np.asarray(target.value).size for target in example.targets.supervision.values())
    for predicted in predictions.values():
        assert set(predicted) == set(example.targets.supervision)
        assert sum(np.asarray(value).size for value in predicted.values()) == expected_size
        for name, value in predicted.items():
            assert value.shape == example.targets.supervision[name].value.shape
            assert np.all(np.isfinite(np.asarray(value)))

    def permute_domain(value, layout):
        source = np.asarray(value)
        result = source.copy()
        first = layout.element_pft_index == 0
        second = layout.element_pft_index == 1
        assert np.count_nonzero(first) == np.count_nonzero(second)
        result[first] = source[second]
        result[second] = source[first]
        return result

    permutation = np.asarray([1, 0])
    permuted_pft = assembled.pft._replace(
        **{
            name: np.asarray(getattr(assembled.pft, name))[permutation]
            for name in assembled.pft._fields
        }
    )
    permuted_state = assembled.state._replace(
        **{
            layout.name: permute_domain(getattr(assembled.state, layout.name), layout)
            for layout in axis_layout.domains
        }
    )
    permuted_defined = assembled.state_defined._replace(
        **{
            layout.name: permute_domain(getattr(assembled.state_defined, layout.name), layout)
            for layout in axis_layout.domains
        }
    )
    permuted_input = assembled._replace(
        pft=permuted_pft,
        state=permuted_state,
        state_defined=permuted_defined,
    )
    permuted_context = assemble_neural_process_context(
        parameters,
        permuted_input,
        normalization=input_normalization,
        axis_layout=axis_layout,
    )
    permuted_prediction = neural_process_outputs(
        parameters,
        permuted_context,
        schema=schema,
        normalization=output_normalization,
        arm=AXIAL_CANDIDATE,
    )
    for field in schema.fields:
        original = np.asarray(predictions[AXIAL_CANDIDATE][field.name])
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
        np.testing.assert_allclose(
            np.asarray(permuted_prediction[field.name]),
            expected,
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg=field.name,
        )

    inactive_pft = assembled.pft._replace(
        active_pft_mask=np.asarray([True, False], dtype=np.bool_)
    )
    inactive_input = assembled._replace(pft=inactive_pft)
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

    def candidate_outputs(current_input):
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

    inactive_prediction = candidate_outputs(inactive_input)
    perturbed_prediction = candidate_outputs(perturbed_input)
    for name in inactive_prediction:
        np.testing.assert_allclose(
            np.asarray(perturbed_prediction[name]),
            np.asarray(inactive_prediction[name]),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def decoder_loss(decoder_second):
        changed = dict(parameters, decoder_second=decoder_second)
        predicted = neural_process_outputs(
            changed,
            context,
            schema=schema,
            normalization=output_normalization,
            arm=AXIAL_CANDIDATE,
        )
        return masked_equal_label_macro_mse(
            predicted,
            example.targets.supervision,
            scales=unit_process_loss_scales(example.targets.supervision),
            field_labels=adapter.field_labels(),
        ).total

    loss, gradient = jax.value_and_grad(decoder_loss)(parameters["decoder_second"])
    assert np.isfinite(float(loss))
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(gradient))

    forcing_fields = adapter.contract.native_forcing.fields
    head = NeuralProcessHead(
        arm=AXIAL_CANDIDATE,
        contract=adapter.contract,
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
    prediction = head(parameters, context, assembled)
    assert set(prediction.named_outputs) == set(example.targets.supervision)
    assert prediction.retained_boundary.fast_state_fields
    update = apply_conservative_update(assembled, prediction)
    validate_conservative_result(update)

    def retained_tail(value):
        return RetainedTailResult(
            state=value.fast_update.state,
            state_defined=value.fast_update.state_defined,
            discrete_state=value.fast_update.discrete_state,
            diagnostics={"test_tail": jnp.asarray(1, dtype=jnp.int32)},
        )

    transition = bind_daily_operator(
        input_assembler=adapter.input_assembler,
        process_head=head,
        retained_tail=retained_tail,
    )
    result = transition(parameters, example.inputs)
    assert int(np.asarray(result.diagnostics["test_tail"])) == 1
    assert set(result.prediction.named_outputs) == set(example.targets.supervision)


def _state_and_masks():
    state = StateFeatureGroups(
        **{
            name: np.asarray([float(index + 1)], dtype=np.float64)
            for index, name in enumerate(StateFeatureGroups._fields)
        }
    )
    masks = StateDefinedMasks(
        **{
            name: np.ones_like(getattr(state, name), dtype=np.bool_)
            for name in StateFeatureGroups._fields
        }
    )
    return state, masks


def test_one_day_hard_metrics_accept_exact_state_and_reject_budget_failure():
    state, masks = _state_and_masks()
    domain = ConservativeDomainResult(
        inventory=np.asarray([1.0]),
        external_input=np.asarray([0.0]),
        external_output=np.asarray([0.0]),
        internal_transfer=np.asarray([[0.0]]),
        budget_residual=np.asarray(0.0),
        defined=np.asarray([True]),
    )
    update = ConservativeUpdateResult(
        state=state,
        state_defined=masks,
        discrete_state={"flag": np.asarray(1, dtype=np.int32)},
        water=domain,
        carbon=domain,
        thermal_tendency=np.asarray([0.0]),
    )
    result = DailyOperatorResult(
        prediction=None,
        fast_update=update,
        next_state=state,
        next_state_defined=masks,
        next_discrete_state={"flag": np.asarray(1, dtype=np.int32)},
        diagnostics={},
    )
    metrics = one_day_acceptance_metrics(
        result,
        expected_state=state,
        expected_state_defined=masks,
        expected_discrete_state={"flag": np.asarray(1, dtype=np.int32)},
    )
    validate_local_one_day_hard_gates(metrics)
    assert all(value["rmse"] == 0.0 for value in metrics["state"].values())

    failed_domain = domain._replace(budget_residual=np.asarray(3.0e-12))
    failed = result._replace(fast_update=update._replace(water=failed_domain))
    failed_metrics = one_day_acceptance_metrics(
        failed,
        expected_state=state,
        expected_state_defined=masks,
        expected_discrete_state={"flag": np.asarray(1, dtype=np.int32)},
    )
    with pytest.raises(ValueError, match="water one-day budget"):
        validate_local_one_day_hard_gates(failed_metrics)
