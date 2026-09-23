from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.shared_transformer import (
    _decoder_dynamic_plan,
    assemble_shared_transformer_context,
    initialize_shared_transformer_parameters,
    shared_transformer_raw_outputs,
    shared_transformer_static_coordinate_indices,
)
from tests.unit.test_daily_operator_hierarchical_coordinate_attention import (
    _permuted_candidate_inputs,
)
from tests.unit.test_shared_transformer import (
    GPP,
    HYDROLOGY,
    MAINTENANCE,
    shared_transformer_setup,
)

THERMAL = "thermal.thermosoil_previous_step_state.ptn.normalized_tendency"


@pytest.fixture(scope="module")
def execution_setup():
    setup = shared_transformer_setup.__wrapped__()
    selected_names = (GPP, MAINTENANCE, HYDROLOGY, THERMAL)
    selected = tuple(
        next(index for index, field in enumerate(setup["learned_schema"].fields) if field.name == name)
        for name in selected_names
    )
    schema = replace(
        setup["learned_schema"],
        fields=tuple(setup["learned_schema"].fields[index] for index in selected),
    )
    full_layout = setup["full_layout"]
    layout = replace(
        full_layout,
        ledger=tuple(full_layout.ledger[index] for index in selected),
        field_expert_index=np.asarray(full_layout.field_expert_index)[list(selected)],
        field_label_index=np.asarray(full_layout.field_label_index)[list(selected)],
        local_state_indices=tuple(full_layout.local_state_indices[index] for index in selected),
    )
    config = replace(setup["config"], coordinate_blocks=2)
    parameters = initialize_shared_transformer_parameters(
        jax.random.PRNGKey(20260921),
        schema=schema,
        layout=layout,
        inputs=setup["assembled"],
        config=config,
        carbon_state_layout=setup["carbon_layout"],
    )
    shared = dict(parameters["shared_transformer"])
    biased_blocks = []
    for block_index, block in enumerate(shared["coordinate_blocks"]):
        current = dict(block)
        attention = dict(current["attention"])
        for projection_index, name in enumerate(("key", "value")):
            projection = dict(attention[name])
            projection["bias"] = jnp.linspace(
                0.01 * (block_index + 1) * (projection_index + 1),
                0.02 * (block_index + 1) * (projection_index + 1),
                config.width,
                dtype=jnp.float32,
            )
            attention[name] = projection
        current["attention"] = attention
        biased_blocks.append(current)
    shared["coordinate_blocks"] = tuple(biased_blocks)
    parameters = dict(parameters)
    parameters["shared_transformer"] = shared
    return setup | {
        "schema": schema,
        "layout": layout,
        "config": config,
        "parameters": parameters,
    }


def _run(setup, inputs, *, execution, chunk_size=3, parameters=None):
    parameters = setup["parameters"] if parameters is None else parameters
    config = replace(
        setup["config"],
        decoder_execution=execution,
        output_chunk_size=chunk_size,
    )
    context = assemble_shared_transformer_context(
        parameters,
        inputs,
        normalization=setup["input_normalization"],
        layout=setup["layout"],
        config=config,
    )
    return shared_transformer_raw_outputs(
        parameters,
        context,
        inputs,
        schema=setup["schema"],
        layout=setup["layout"],
        input_normalization=setup["input_normalization"],
        config=config,
        carbon_state_layout=setup["carbon_layout"],
    )


def _assert_output_equivalent(reference, observed):
    reference_raw, reference_masks = reference
    observed_raw, observed_masks = observed
    assert observed_raw.keys() == reference_raw.keys()
    for name in reference_raw:
        np.testing.assert_allclose(
            np.asarray(observed_raw[name]),
            np.asarray(reference_raw[name]),
            rtol=2.0e-5,
            atol=2.0e-6,
            err_msg=name,
        )
        np.testing.assert_array_equal(
            np.asarray(observed_masks[name]),
            np.asarray(reference_masks[name]),
            err_msg=name,
        )


def test_static_coordinate_plan_matches_canonical_layout(execution_setup):
    setup = execution_setup
    plan = shared_transformer_static_coordinate_indices(
        schema=setup["learned_schema"],
        layout=setup["full_layout"],
    )
    full_count = sum(int(np.prod(field.shape, dtype=np.int64)) for field in setup["learned_schema"].fields)
    eligible_count = sum(indices.size for indices in plan.values())
    assert len(plan) == 161
    assert full_count == 39_006
    assert eligible_count == 8_550

    for field, ownership in zip(
        setup["learned_schema"].fields,
        setup["full_layout"].ledger,
        strict=True,
    ):
        indices = plan[field.name]
        assert np.all(indices[:-1] < indices[1:])
        if ownership.pft_axis is None or not field.full_pft_axis:
            assert indices.size == int(np.prod(field.shape, dtype=np.int64))

    dynamic = _decoder_dynamic_plan(
        schema=setup["learned_schema"],
        layout=setup["full_layout"],
    )
    assert dynamic.axis_expert_index.size == 339
    assert dynamic.local_state_indices.size == 1_798
    assert -1 in dynamic.local_state_indices
    assert sum(indices.size for indices in dynamic.field_axis_bank_indices) == 1_380
    assert sum(indices.size for indices in dynamic.field_local_bank_indices) == 3_249

    width = 768
    microbatch = 4
    blocks_and_key_value = 2 * 2
    global_dense_cotangent_bytes = (
        len(setup["learned_schema"].fields)
        * blocks_and_key_value
        * microbatch
        * (dynamic.axis_expert_index.size + dynamic.local_state_indices.size)
        * width
        * np.dtype(np.float32).itemsize
    )
    field_local_dense_cotangent_bytes = (
        blocks_and_key_value
        * microbatch
        * (
            sum(indices.size for indices in dynamic.field_axis_bank_indices)
            + sum(indices.size for indices in dynamic.field_local_bank_indices)
        )
        * width
        * np.dtype(np.float32).itemsize
    )
    assert global_dense_cotangent_bytes == 16_911_089_664
    assert field_local_dense_cotangent_bytes == 227_524_608


def test_execution_modes_match_with_padding_masks_permutation_and_carbon(execution_setup):
    setup = execution_setup
    inputs = setup["assembled"]
    reference = _run(setup, inputs, execution="original")
    gather_only = _run(setup, inputs, execution="gather_only")
    optimized = _run(setup, inputs, execution="gather_reuse")
    dynamic_reuse = _run(setup, inputs, execution="gather_dynamic_reuse")
    _assert_output_equivalent(reference, gather_only)
    _assert_output_equivalent(reference, optimized)
    _assert_output_equivalent(optimized, dynamic_reuse)
    for block in setup["parameters"]["shared_transformer"]["coordinate_blocks"]:
        assert np.all(np.asarray(block["attention"]["key"]["bias"]) != 0.0)
        assert np.all(np.asarray(block["attention"]["value"]["bias"]) != 0.0)

    permuted = _permuted_candidate_inputs(setup | {"layout": setup["full_layout"]})
    _assert_output_equivalent(
        _run(setup, permuted, execution="gather_reuse"),
        _run(setup, permuted, execution="gather_dynamic_reuse"),
    )
    _assert_output_equivalent(
        _run(setup, permuted, execution="original"),
        _run(setup, permuted, execution="gather_reuse"),
    )

    pft = inputs.pft._replace(
        pft_fraction=np.zeros_like(np.asarray(inputs.pft.pft_fraction)),
        active_pft_mask=np.zeros_like(np.asarray(inputs.pft.active_pft_mask)),
    )
    all_pfts_masked = inputs._replace(
        pft=pft,
        canonical_input=inputs.canonical_input._replace(pft=pft),
    )
    masked_reference = _run(setup, all_pfts_masked, execution="original")
    masked_optimized = _run(setup, all_pfts_masked, execution="gather_reuse")
    masked_dynamic = _run(
        setup,
        all_pfts_masked,
        execution="gather_dynamic_reuse",
    )
    _assert_output_equivalent(masked_reference, masked_optimized)
    _assert_output_equivalent(masked_optimized, masked_dynamic)
    for raw in masked_optimized[0].values():
        assert np.all(np.isfinite(np.asarray(raw)))
    assert not np.any(np.asarray(masked_optimized[1][GPP]))
    assert not np.any(np.asarray(masked_optimized[1][MAINTENANCE]))
    assert not np.any(np.asarray(masked_optimized[1][THERMAL]))

    thermal = np.asarray(optimized[0][THERMAL]).reshape(-1)
    retained = shared_transformer_static_coordinate_indices(schema=setup["schema"], layout=setup["layout"])[THERMAL]
    excluded = np.setdiff1d(np.arange(thermal.size, dtype=np.int32), retained)
    assert retained.size == 64
    assert excluded.size == 384
    np.testing.assert_array_equal(thermal[excluded], 0.0)


def _objective_vector(
    setup,
    execution,
    parameters,
    state_other,
    pft_parameters,
    *,
    base_inputs=None,
):
    inputs = setup["assembled"] if base_inputs is None else base_inputs
    state = inputs.state._replace(other=state_other)
    pft = inputs.pft._replace(pft_parameters=pft_parameters)
    differentiated_inputs = inputs._replace(
        state=state,
        pft=pft,
        canonical_input=inputs.canonical_input._replace(pft=pft),
    )
    raw, _masks = _run(
        setup,
        differentiated_inputs,
        execution=execution,
        parameters=parameters,
    )
    objectives = []
    for name in (GPP, MAINTENANCE, HYDROLOGY, THERMAL):
        flat = raw[name].reshape(-1)
        weights = jnp.linspace(0.5, 1.5, flat.size, dtype=flat.dtype)
        objectives.append(jnp.vdot(flat, weights))
    return jnp.stack(tuple(objectives))


def _assert_jacobian_tree_close(reference, observed):
    reference_leaves, reference_tree = jax.tree_util.tree_flatten(reference)
    observed_leaves, observed_tree = jax.tree_util.tree_flatten(observed)
    assert observed_tree == reference_tree
    for expected, actual in zip(reference_leaves, observed_leaves, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
            rtol=5.0e-4,
            atol=2.0e-5,
        )


def test_per_field_parameter_state_and_pft_input_gradients_match(execution_setup):
    setup = execution_setup
    inputs = setup["assembled"]

    def differentiated(execution):
        return jax.jacrev(
            lambda parameters, state_other, pft_parameters: _objective_vector(
                setup,
                execution,
                parameters,
                state_other,
                pft_parameters,
            ),
            argnums=(0, 1, 2),
        )(
            setup["parameters"],
            jnp.asarray(inputs.state.other),
            jnp.asarray(inputs.pft.pft_parameters),
        )

    reference = differentiated("original")
    gather_only = differentiated("gather_only")
    optimized = differentiated("gather_reuse")
    _assert_jacobian_tree_close(reference, gather_only)
    _assert_jacobian_tree_close(reference, optimized)

    dynamic_parameter_jacobian = jax.jacrev(
        lambda parameters: _objective_vector(
            setup,
            "gather_dynamic_reuse",
            parameters,
            jnp.asarray(inputs.state.other),
            jnp.asarray(inputs.pft.pft_parameters),
        )
    )(setup["parameters"])
    dynamic_state_jacobian = jax.jacrev(
        lambda state_other: _objective_vector(
            setup,
            "gather_dynamic_reuse",
            setup["parameters"],
            state_other,
            jnp.asarray(inputs.pft.pft_parameters),
        )
    )(jnp.asarray(inputs.state.other))
    _assert_jacobian_tree_close(optimized[0], dynamic_parameter_jacobian)
    _assert_jacobian_tree_close(optimized[1], dynamic_state_jacobian)

    parameter_jacobian, state_jacobian, pft_parameter_jacobian = optimized
    assert (
        np.linalg.norm(
            np.asarray(parameter_jacobian["shared_transformer"]["coordinate_blocks"][0]["attention"]["key"]["kernel"])[
                0
            ]
        )
        > 0.0
    )
    carbon_indices = np.asarray(setup["carbon_layout"].state_indices, dtype=np.int32)
    assert np.linalg.norm(np.asarray(state_jacobian)[0, carbon_indices]) > 0.0
    np.testing.assert_array_equal(np.asarray(pft_parameter_jacobian), 0.0)


def test_dynamic_reuse_gradients_match_when_all_pfts_are_inactive(execution_setup):
    setup = execution_setup
    inputs = setup["assembled"]
    pft = inputs.pft._replace(
        pft_fraction=np.zeros_like(np.asarray(inputs.pft.pft_fraction)),
        active_pft_mask=np.zeros_like(np.asarray(inputs.pft.active_pft_mask)),
    )
    masked_inputs = inputs._replace(
        pft=pft,
        canonical_input=inputs.canonical_input._replace(pft=pft),
    )
    state_other = jnp.asarray(masked_inputs.state.other)
    pft_parameters = jnp.asarray(masked_inputs.pft.pft_parameters)

    def scalar_objective(execution, parameters, continuous_state):
        return jnp.sum(
            _objective_vector(
                setup,
                execution,
                parameters,
                continuous_state,
                pft_parameters,
                base_inputs=masked_inputs,
            )
        )

    reference_parameter_gradient = jax.grad(
        lambda parameters: scalar_objective("gather_reuse", parameters, state_other)
    )(setup["parameters"])
    candidate_parameter_gradient = jax.grad(
        lambda parameters: scalar_objective("gather_dynamic_reuse", parameters, state_other)
    )(setup["parameters"])
    _assert_jacobian_tree_close(
        reference_parameter_gradient,
        candidate_parameter_gradient,
    )

    reference_state_gradient = jax.grad(
        lambda continuous_state: scalar_objective("gather_reuse", setup["parameters"], continuous_state)
    )(state_other)
    candidate_state_gradient = jax.grad(
        lambda continuous_state: scalar_objective("gather_dynamic_reuse", setup["parameters"], continuous_state)
    )(state_other)
    np.testing.assert_allclose(
        np.asarray(candidate_state_gradient),
        np.asarray(reference_state_gradient),
        rtol=5.0e-4,
        atol=2.0e-5,
    )
    assert np.linalg.norm(np.asarray(reference_state_gradient)) > 0.0
