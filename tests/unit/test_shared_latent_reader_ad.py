from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.shared_latent_reader import (
    LATENT_SLOTS,
    READER_HEADS,
    READER_WIDTH,
    reader_attention,
    shared_latent_reader_config,
    shared_latent_reader_raw_outputs,
)
from research.daily_coarse_graining.daily_operator.shared_transformer import (
    SharedTransformerConfig,
    _field_active_and_pft_owner,
    assemble_shared_transformer_context,
    initialize_shared_transformer_parameters,
    shared_transformer_raw_outputs,
)
from tests.unit.test_shared_latent_reader_implementation import static_setup

GPP = "daily.fast_gpp"
HYDROLOGY = "inventory.water.qsintveg.storage.outgoing_fraction"
HYDROLOGY_SAME_SHAPE = "endpoint.fast_state.hydrol_previous_step_state.canopy2ground"


def _identity_linear(width: int) -> dict[str, jax.Array]:
    return {
        "kernel": jnp.eye(width, dtype=jnp.float32),
        "bias": jnp.zeros((width,), dtype=jnp.float32),
    }


def _explicit_joint_attention(
    query: jax.Array,
    banks: tuple[jax.Array, ...],
    masks: tuple[jax.Array, ...],
) -> jax.Array:
    """Independent concatenated-bank reference for reader_attention."""
    q = query.reshape(query.shape[:-1] + (READER_HEADS, READER_WIDTH // READER_HEADS))
    q = jnp.swapaxes(q, -3, -2)
    q = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1.0e-6)
    keys = jnp.concatenate(banks, axis=-2)
    values = keys
    keys = keys.reshape(keys.shape[:-1] + (READER_HEADS, READER_WIDTH // READER_HEADS))
    keys = jnp.swapaxes(keys, -3, -2)
    keys = keys / jnp.maximum(jnp.linalg.norm(keys, axis=-1, keepdims=True), 1.0e-6)
    values = values.reshape(values.shape[:-1] + (READER_HEADS, READER_WIDTH // READER_HEADS))
    values = jnp.swapaxes(values, -3, -2)
    logits = jnp.einsum("...hqd,...hkd->...hqk", q, keys) * jnp.sqrt(8.0)
    valid = jnp.concatenate(masks, axis=-1)[:, None, None, :]
    row_valid = jnp.any(valid, axis=-1, keepdims=True)
    weights = jax.nn.softmax(jnp.where(valid, logits, -jnp.inf), axis=-1)
    weights = jnp.where(row_valid & valid, weights, 0.0)
    attended = jnp.einsum("...hqk,...hkd->...hqd", weights, values)
    attended = jnp.swapaxes(attended, -3, -2).reshape(query.shape)
    return attended


def _reader_attention_inputs(seed: int = 12):
    key = jax.random.PRNGKey(seed)
    query_key, latent_key, dynamic_key, process_key, carbon_key = jax.random.split(key, 5)
    query = jax.random.normal(query_key, (2, 1, READER_WIDTH), dtype=jnp.float32)
    latent = jax.random.normal(latent_key, (3, READER_WIDTH), dtype=jnp.float32)
    dynamic = jax.random.normal(dynamic_key, (2, 2, READER_WIDTH), dtype=jnp.float32)
    process = jax.random.normal(process_key, (1, READER_WIDTH), dtype=jnp.float32)
    carbon = jax.random.normal(carbon_key, (4, READER_WIDTH), dtype=jnp.float32)
    return query, latent, dynamic, process, carbon


def test_reader_attention_matches_one_joint_softmax_and_masks_forbidden_carbon():
    query, latent, dynamic, process, carbon = _reader_attention_inputs()
    parameters = {name: _identity_linear(READER_WIDTH) for name in ("query", "key", "value", "output")}
    latent_mask = jnp.asarray([[True, False, True], [False, True, True]])
    dynamic_mask = jnp.asarray([[True, False], [False, True]])
    process_mask = jnp.asarray([[True], [True]])
    carbon_mask = jnp.asarray([[True, False, False, False], [False, False, True, False]])
    latent_by_query = jnp.broadcast_to(latent, (2,) + latent.shape)
    process_by_query = jnp.broadcast_to(process, (2,) + process.shape)
    carbon_by_query = jnp.broadcast_to(carbon, (2,) + carbon.shape)
    expected = _explicit_joint_attention(
        query,
        (latent_by_query, dynamic, process_by_query, carbon_by_query),
        (latent_mask, dynamic_mask, process_mask, carbon_mask),
    )
    observed = reader_attention(
        parameters,
        query,
        latent_key=latent,
        latent_value=latent,
        latent_mask=latent_mask,
        axis_local_key=dynamic,
        axis_local_value=dynamic,
        axis_local_mask=dynamic_mask,
        process_key=process,
        process_value=process,
        process_mask=process_mask,
        carbon_key=carbon,
        carbon_value=carbon,
        carbon_mask=carbon_mask,
    )
    np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), rtol=2.0e-6, atol=2.0e-6)

    def isolated_carbon_loss(value):
        return jnp.sum(reader_attention(
            parameters,
            query,
            latent_key=latent,
            latent_value=latent,
            latent_mask=latent_mask,
            axis_local_key=dynamic,
            axis_local_value=dynamic,
            axis_local_mask=dynamic_mask,
            process_key=process,
            process_value=process,
            process_mask=process_mask,
            carbon_key=value,
            carbon_value=value,
            carbon_mask=jnp.zeros((2, 4), dtype=jnp.bool_),
        ))

    carbon_gradient = jax.grad(isolated_carbon_loss)(carbon)
    np.testing.assert_array_equal(np.asarray(carbon_gradient), np.zeros_like(np.asarray(carbon_gradient)))

    no_sources = reader_attention(
        parameters,
        query,
        latent_key=latent,
        latent_value=latent,
        latent_mask=jnp.zeros((2, 3), dtype=jnp.bool_),
        axis_local_key=dynamic,
        axis_local_value=dynamic,
        axis_local_mask=jnp.zeros((2, 2), dtype=jnp.bool_),
        process_key=process,
        process_value=process,
        process_mask=jnp.zeros((2, 1), dtype=jnp.bool_),
        carbon_key=carbon,
        carbon_value=carbon,
        carbon_mask=jnp.zeros((2, 4), dtype=jnp.bool_),
    )
    assert np.all(np.isfinite(np.asarray(no_sources)))
    np.testing.assert_array_equal(np.asarray(no_sources), np.zeros_like(np.asarray(no_sources)))


@pytest.fixture(scope="module")
def latent_reader_setup():
    base = static_setup()
    full_layout = base["layout"]
    selected_names = (GPP, HYDROLOGY, HYDROLOGY_SAME_SHAPE)
    selected = tuple(
        next(index for index, field in enumerate(base["learned_schema"].fields) if field.name == name)
        for name in selected_names
    )
    schema = replace(
        base["learned_schema"],
        fields=tuple(base["learned_schema"].fields[index] for index in selected),
    )
    layout = replace(
        full_layout,
        ledger=tuple(full_layout.ledger[index] for index in selected),
        field_expert_index=np.asarray(full_layout.field_expert_index)[list(selected)],
        field_label_index=np.asarray(full_layout.field_label_index)[list(selected)],
        local_state_indices=tuple(full_layout.local_state_indices[index] for index in selected),
    )
    carbon_layout = base["carbon_layout"]
    assert layout.field_expert_index[1] == layout.field_expert_index[2]
    assert schema.fields[1].shape == schema.fields[2].shape
    assert not np.array_equal(layout.local_state_indices[1], layout.local_state_indices[2])
    fixture_config = SharedTransformerConfig(
        width=32,
        attention_heads=4,
        ffn_width=128,
        backbone_blocks=1,
        coordinate_blocks=2,
        readout_width=256,
        output_chunk_size=8,
        input_width=384,
        input_attention_heads=8,
        input_ffn_width=1536,
        forcing_width=256,
        forcing_heads=8,
        forcing_ffn_width=1024,
        forcing_blocks=2,
        axial_blocks=1,
        decoder_width=384,
        decoder_ffn_width=1536,
        decoder_attention_heads=8,
        decoder_architecture="latent_reader",
    )
    parameters = initialize_shared_transformer_parameters(
        jax.random.PRNGKey(20261004),
        schema=schema,
        layout=layout,
        inputs=base["assembled"],
        config=fixture_config,
        carbon_state_layout=carbon_layout,
    )
    base.pop("parameters", None)
    return base | {
        "schema": schema,
        "layout": layout,
        "full_layout": full_layout,
        "carbon_layout": carbon_layout,
        "config": fixture_config,
        "parameters": parameters,
    }


def _active_two_pft_inputs(inputs: Any):
    pft = inputs.pft._replace(
        pft_fraction=np.asarray(inputs.pft.pft_fraction).copy(),
        active_pft_mask=np.asarray([True, True], dtype=np.bool_),
    )
    return inputs._replace(pft=pft, canonical_input=inputs.canonical_input._replace(pft=pft))


def _raw_outputs(setup, inputs, parameters=None):
    parameters = setup["parameters"] if parameters is None else parameters
    context = assemble_shared_transformer_context(
        parameters,
        inputs,
        normalization=setup["input_normalization"],
        layout=setup["layout"],
        config=setup["config"],
    )
    return shared_transformer_raw_outputs(
        parameters,
        context,
        inputs,
        schema=setup["schema"],
        layout=setup["layout"],
        input_normalization=setup["input_normalization"],
        config=setup["config"],
        carbon_state_layout=setup["carbon_layout"],
    )


def test_latent_reader_outputs_masks_and_pft_permutation(latent_reader_setup):
    setup = latent_reader_setup
    inputs = setup["assembled"]
    pft = inputs.pft._replace(active_pft_mask=np.asarray([True, False], dtype=np.bool_))
    inputs = inputs._replace(pft=pft, canonical_input=inputs.canonical_input._replace(pft=pft))
    raw, masks = _raw_outputs(setup, inputs)
    assert set(raw) == {field.name for field in setup["schema"].fields}
    for field in setup["schema"].fields:
        values = np.asarray(raw[field.name])
        active = np.asarray(masks[field.name])
        assert values.shape == field.shape
        assert active.shape == field.shape
        assert active.dtype == np.bool_
        assert np.all(np.isfinite(values))
        np.testing.assert_array_equal(values[~active], 0.0)
    assert np.count_nonzero(np.asarray(masks[GPP])) >= 1
    assert np.any(~np.asarray(masks[GPP]))

    shared = setup["parameters"]["shared_transformer"]
    assert shared["latent_slots"].shape == (LATENT_SLOTS, 384)
    assert shared["reader_projection"]["kernel"].shape == (384, READER_WIDTH)
    assert shared["readout_first"]["kernel"].shape == (READER_WIDTH, 256)
    assert "coordinate_blocks" not in shared
    assert setup["config"].input_width == 384
    canonical = shared_latent_reader_config()
    assert (canonical.width, canonical.backbone_blocks, canonical.decoder_width) == (768, 8, 384)
    for axis_name, position_embedding in zip(
        setup["layout"].axis_names,
        shared["output_axis_position_embedding"],
        strict=True,
    ):
        if axis_name == "pft":
            assert position_embedding is None

    inputs = _active_two_pft_inputs(setup["assembled"])
    raw, masks = _raw_outputs(setup, inputs)
    from tests.unit.test_daily_operator_hierarchical_coordinate_attention import _permuted_candidate_inputs

    permuted_inputs = _permuted_candidate_inputs(setup | {"layout": setup["full_layout"]})
    permuted_raw, permuted_masks = _raw_outputs(setup, permuted_inputs)
    for field, ownership in zip(setup["schema"].fields, setup["layout"].ledger, strict=True):
        expected_values = np.asarray(raw[field.name]).copy()
        expected_masks = np.asarray(masks[field.name]).copy()
        if ownership.pft_axis is not None:
            first = [slice(None)] * expected_values.ndim
            second = [slice(None)] * expected_values.ndim
            first[ownership.pft_axis] = setup["layout"].active_pft_indices[0] if field.full_pft_axis else 0
            second[ownership.pft_axis] = setup["layout"].active_pft_indices[1] if field.full_pft_axis else 1
            expected_values[tuple(first)], expected_values[tuple(second)] = (
                expected_values[tuple(second)].copy(), expected_values[tuple(first)].copy()
            )
            expected_masks[tuple(first)], expected_masks[tuple(second)] = (
                expected_masks[tuple(second)].copy(), expected_masks[tuple(first)].copy()
            )
        np.testing.assert_allclose(np.asarray(permuted_raw[field.name]), expected_values, rtol=2e-5, atol=2e-6)
        np.testing.assert_array_equal(np.asarray(permuted_masks[field.name]), expected_masks)


def test_gpp_raw_carbon_mask_excludes_other_pft_and_undefined_values(latent_reader_setup):
    setup = latent_reader_setup
    inputs = _active_two_pft_inputs(setup["assembled"])
    parameters = setup["parameters"]
    context = assemble_shared_transformer_context(
        parameters,
        inputs,
        normalization=setup["input_normalization"],
        layout=setup["layout"],
        config=setup["config"],
    )
    field = setup["schema"].fields[0]
    ownership = setup["layout"].ledger[0]
    _coordinates, active, owners = _field_active_and_pft_owner(
        context,
        field,
        ownership,
        setup["layout"],
    )
    query_owner = int(np.asarray(owners)[0])
    assert bool(np.asarray(active)[0])
    carbon_layout = setup["carbon_layout"]
    carbon_indices = np.asarray(carbon_layout.state_indices, dtype=np.int32)
    carbon_owners = np.asarray(carbon_layout.pft_owner_index, dtype=np.int32)
    carbon_defined = np.asarray(inputs.state_defined.other, dtype=np.bool_)[carbon_indices]
    other_pft = np.flatnonzero(carbon_defined & (carbon_owners != -1) & (carbon_owners != query_owner))
    assert other_pft.size > 0
    legal_carbon = np.flatnonzero(carbon_defined & ((carbon_owners == -1) | (carbon_owners == query_owner)))
    assert legal_carbon.size > 0
    other_coordinate = int(carbon_indices[other_pft[0]])
    undefined_carbon_coordinate = int(carbon_indices[legal_carbon[0]])
    undefined_local_coordinate = int(np.asarray(setup["layout"].local_state_indices[1]).reshape(-1)[0])
    source_state = jnp.asarray(inputs.state.other)

    def output_for_state_value(base_inputs, output_name, coordinate, value):
        current_state = source_state.at[coordinate].set(value)
        current_inputs = base_inputs._replace(state=base_inputs.state._replace(other=current_state))
        outputs, _ = shared_latent_reader_raw_outputs(
            parameters,
            context,
            current_inputs,
            schema=setup["schema"],
            layout=setup["layout"],
            input_normalization=setup["input_normalization"],
            config=setup["config"],
            carbon_state_layout=carbon_layout,
        )
        return outputs[output_name][0, 0]

    derivative = jax.grad(
        lambda value: output_for_state_value(inputs, GPP, other_coordinate, value)
    )(source_state[other_coordinate])
    np.testing.assert_array_equal(np.asarray(derivative), np.asarray(0.0, dtype=derivative.dtype))

    undefined_carbon_state = inputs.state_defined.other.at[undefined_carbon_coordinate].set(False)
    undefined_carbon_inputs = inputs._replace(
        state_defined=inputs.state_defined._replace(other=undefined_carbon_state)
    )
    derivative = jax.grad(
        lambda value: output_for_state_value(
            undefined_carbon_inputs, GPP, undefined_carbon_coordinate, value
        )
    )(source_state[undefined_carbon_coordinate])
    np.testing.assert_array_equal(np.asarray(derivative), np.asarray(0.0, dtype=derivative.dtype))

    undefined_local_state = inputs.state_defined.other.at[undefined_local_coordinate].set(False)
    undefined_local_inputs = inputs._replace(
        state_defined=inputs.state_defined._replace(other=undefined_local_state)
    )
    derivative = jax.grad(
        lambda value: output_for_state_value(
            undefined_local_inputs, HYDROLOGY, undefined_local_coordinate, value
        )
    )(source_state[undefined_local_coordinate])
    np.testing.assert_array_equal(np.asarray(derivative), np.asarray(0.0, dtype=derivative.dtype))


def test_expert_grouping_strict_parameter_state_ad_gate_remains_rejected(latent_reader_setup):
    setup = latent_reader_setup
    inputs = _active_two_pft_inputs(setup["assembled"])
    parameters = setup["parameters"]
    original_state = jnp.asarray(inputs.state.other)
    grouped_config = replace(setup["config"], latent_reader_execution="expert")

    def evaluate(current_parameters, state_other, *, field_reference):
        current_inputs = inputs._replace(state=inputs.state._replace(other=state_other))
        current_context = assemble_shared_transformer_context(
            current_parameters,
            current_inputs,
            normalization=setup["input_normalization"],
            layout=setup["layout"],
            config=setup["config"],
        )
        if field_reference:
            return shared_latent_reader_raw_outputs(
                current_parameters,
                current_context,
                current_inputs,
                schema=setup["schema"],
                layout=setup["layout"],
                input_normalization=setup["input_normalization"],
                config=setup["config"],
                carbon_state_layout=setup["carbon_layout"],
                field_reference=True,
            )
        return shared_transformer_raw_outputs(
            current_parameters,
            current_context,
            current_inputs,
            schema=setup["schema"],
            layout=setup["layout"],
            input_normalization=setup["input_normalization"],
            config=grouped_config,
            carbon_state_layout=setup["carbon_layout"],
        )

    def objective(current_parameters, state_other, *, field_reference):
        raw, masks = evaluate(current_parameters, state_other, field_reference=field_reference)
        terms = []
        for field in setup["schema"].fields:
            weight = jnp.arange(1, 1 + int(np.prod(field.shape)), dtype=jnp.float32).reshape(field.shape)
            terms.append(jnp.sum(raw[field.name] * weight))
        return sum(terms), (raw, masks)

    grouped = jax.value_and_grad(objective, argnums=(0, 1), has_aux=True)(
        parameters, original_state, field_reference=False
    )
    reference = jax.value_and_grad(objective, argnums=(0, 1), has_aux=True)(
        parameters, original_state, field_reference=True
    )
    (grouped_value, (grouped_raw, grouped_masks)), (grouped_parameter_grad, grouped_state_grad) = grouped
    (reference_value, (reference_raw, reference_masks)), (reference_parameter_grad, reference_state_grad) = reference
    np.testing.assert_allclose(np.asarray(grouped_value), np.asarray(reference_value), rtol=1.0e-5, atol=1.0e-6)
    for name in grouped_raw:
        np.testing.assert_array_equal(grouped_masks[name], reference_masks[name])
        np.testing.assert_allclose(grouped_raw[name], reference_raw[name], rtol=1.0e-5, atol=1.0e-6)
    strict_mismatches = []
    for tree_name, grouped_tree, reference_tree in (
        ("parameter", grouped_parameter_grad, reference_parameter_grad),
        ("state", grouped_state_grad, reference_state_grad),
    ):
        grouped_pairs, _ = jax.tree_util.tree_flatten_with_path(grouped_tree)
        reference_pairs, _ = jax.tree_util.tree_flatten_with_path(reference_tree)
        for (grouped_path, grouped_leaf), (reference_path, reference_leaf) in zip(
            grouped_pairs,
            reference_pairs,
            strict=True,
        ):
            assert grouped_path == reference_path
            grouped_array = np.asarray(grouped_leaf)
            reference_array = np.asarray(reference_leaf)
            allowed = 1.0e-6 + 1.0e-5 * np.abs(reference_array)
            mismatch = np.abs(grouped_array - reference_array) > allowed
            if np.any(mismatch):
                strict_mismatches.append((
                    tree_name,
                    jax.tree_util.keystr(grouped_path),
                    int(np.count_nonzero(mismatch)),
                    float(np.max(np.abs(grouped_array - reference_array))),
                    float(np.max(np.abs(grouped_array - reference_array) / np.maximum(np.abs(reference_array), 1.0e-30))),
                ))

    assert strict_mismatches, "expert grouping unexpectedly passed the preserved strict AD gate"
    assert strict_mismatches[0][0:2] == (
        "parameter",
        "['shared_transformer']['backbone_blocks'][0]['attention']['key']['kernel']",
    )
    assert strict_mismatches[0][2] == 1
    assert strict_mismatches[0][3] > 1.0e-6

    field_shared_grad = reference_parameter_grad["shared_transformer"]
    for path in (
        field_shared_grad["latent_slots"],
        field_shared_grad["latent_blocks"][0]["cross_attention"]["query"]["kernel"],
        field_shared_grad["latent_blocks"][0]["self_attention"]["query"]["kernel"],
        field_shared_grad["reader_projection"]["kernel"],
        field_shared_grad["reader_blocks"][0]["attention"]["query"]["kernel"],
        field_shared_grad["readout_first"]["kernel"],
        field_shared_grad["readout_second"]["kernel"],
        field_shared_grad["local_value_input"]["kernel"],
        field_shared_grad["carbon"]["field_embedding"],
    ):
        values = np.asarray(path)
        assert np.all(np.isfinite(values))
        assert np.linalg.norm(values) > 0.0
    assert np.all(np.isfinite(np.asarray(reference_state_grad)))
    assert np.linalg.norm(np.asarray(reference_state_grad)) > 0.0


def test_field_reference_parameter_state_jvp_vjp_and_finite_difference(latent_reader_setup):
    setup = latent_reader_setup
    inputs = _active_two_pft_inputs(setup["assembled"])
    parameters = setup["parameters"]
    shared = parameters["shared_transformer"]
    context = assemble_shared_transformer_context(
        parameters,
        inputs,
        normalization=setup["input_normalization"],
        layout=setup["layout"],
        config=setup["config"],
    )
    field = setup["schema"].fields[0]
    ownership = setup["layout"].ledger[0]
    _coordinates, _active, query_owners = _field_active_and_pft_owner(
        context,
        field,
        ownership,
        setup["layout"],
    )
    query_owner = int(np.asarray(query_owners)[0])
    carbon_layout = setup["carbon_layout"]
    carbon_indices = np.asarray(carbon_layout.state_indices, dtype=np.int32)
    carbon_owners = np.asarray(carbon_layout.pft_owner_index, dtype=np.int32)
    carbon_defined = np.asarray(inputs.state_defined.other, dtype=np.bool_)[carbon_indices]
    legal_carbon = np.flatnonzero(
        carbon_defined & ((carbon_owners == -1) | (carbon_owners == query_owner))
    )
    assert legal_carbon.size > 0
    carbon_coordinate = int(carbon_indices[legal_carbon[0]])
    original_state = jnp.asarray(inputs.state.other)
    assert float(original_state[carbon_coordinate]) == 0.0
    original_state = original_state.at[carbon_coordinate].set(0.25)
    inputs = inputs._replace(state=inputs.state._replace(other=original_state))
    context = assemble_shared_transformer_context(
        parameters,
        inputs,
        normalization=setup["input_normalization"],
        layout=setup["layout"],
        config=setup["config"],
    )
    parameter_point = jnp.stack((
        shared["latent_slots"][0, 0],
        shared["latent_blocks"][0]["cross_attention"]["query"]["kernel"][0, 0],
        shared["reader_projection"]["kernel"][0, 0],
        shared["reader_blocks"][0]["ffn"]["first"]["kernel"][0, 0],
        shared["backbone_blocks"][0]["attention"]["key"]["kernel"][0, 0],
    ))
    state_point = original_state[carbon_coordinate]
    point = (parameter_point, state_point)

    def scalar_output(values):
        parameter_values, state_value = values
        current_shared = dict(shared)
        current_shared["latent_slots"] = shared["latent_slots"].at[0, 0].set(parameter_values[0])
        latent_blocks = list(shared["latent_blocks"])
        latent_block = dict(latent_blocks[0])
        cross = dict(latent_block["cross_attention"])
        cross_query = dict(cross["query"])
        cross_query["kernel"] = cross_query["kernel"].at[0, 0].set(parameter_values[1])
        cross["query"] = cross_query
        latent_block["cross_attention"] = cross
        latent_blocks[0] = latent_block
        current_shared["latent_blocks"] = tuple(latent_blocks)
        reader_projection = dict(shared["reader_projection"])
        reader_projection["kernel"] = reader_projection["kernel"].at[0, 0].set(parameter_values[2])
        current_shared["reader_projection"] = reader_projection
        reader_blocks = list(shared["reader_blocks"])
        reader_block = dict(reader_blocks[0])
        reader_ffn = dict(reader_block["ffn"])
        reader_ffn_first = dict(reader_ffn["first"])
        reader_ffn_first["kernel"] = reader_ffn_first["kernel"].at[0, 0].set(parameter_values[3])
        reader_ffn["first"] = reader_ffn_first
        reader_block["ffn"] = reader_ffn
        reader_blocks[0] = reader_block
        current_shared["reader_blocks"] = tuple(reader_blocks)
        backbone_blocks = list(shared["backbone_blocks"])
        backbone_block = dict(backbone_blocks[0])
        attention = dict(backbone_block["attention"])
        attention_key = dict(attention["key"])
        attention_key["kernel"] = attention_key["kernel"].at[0, 0].set(parameter_values[4])
        attention["key"] = attention_key
        backbone_block["attention"] = attention
        backbone_blocks[0] = backbone_block
        current_shared["backbone_blocks"] = tuple(backbone_blocks)
        current_parameters = dict(parameters)
        current_parameters["shared_transformer"] = current_shared
        current_state = original_state.at[carbon_coordinate].set(state_value)
        current_inputs = inputs._replace(state=inputs.state._replace(other=current_state))
        current_context = assemble_shared_transformer_context(
            current_parameters,
            current_inputs,
            normalization=setup["input_normalization"],
            layout=setup["layout"],
            config=setup["config"],
        )
        outputs, _ = shared_latent_reader_raw_outputs(
            current_parameters,
            current_context,
            current_inputs,
            schema=setup["schema"],
            layout=setup["layout"],
            input_normalization=setup["input_normalization"],
            config=setup["config"],
            carbon_state_layout=carbon_layout,
            field_reference=True,
        )
        return outputs[GPP][0, 0]

    _value, pullback = jax.vjp(scalar_output, point)
    reverse = pullback(jnp.asarray(1.0, dtype=jnp.float32))[0]
    gradient_vector = jnp.concatenate((
        reverse[0].astype(jnp.float64),
        jnp.reshape(reverse[1].astype(jnp.float64), (1,)),
    ))
    gradient_norm = jnp.linalg.norm(gradient_vector)
    assert np.isfinite(float(gradient_norm))
    assert float(gradient_norm) > 0.0
    normalized_vector = gradient_vector / gradient_norm
    tangent = (
        normalized_vector[:5].astype(parameter_point.dtype),
        normalized_vector[5].astype(state_point.dtype),
    )
    np.testing.assert_array_equal(np.count_nonzero(np.asarray(normalized_vector)), 6)
    value, directional = jax.jvp(scalar_output, (point,), (tangent,))
    reverse_inner_product = jnp.vdot(tangent[0], reverse[0]) + tangent[1] * reverse[1]
    np.testing.assert_allclose(
        np.asarray(directional),
        np.asarray(reverse_inner_product),
        rtol=2.0e-4,
        atol=2.0e-5,
    )
    assert abs(float(directional)) > 1.0e-7
    step = jnp.asarray(1.0e-3, dtype=parameter_point.dtype)
    state_step = jnp.asarray(1.0e-3, dtype=state_point.dtype)
    positive = (point[0] + step * tangent[0], point[1] + state_step * tangent[1])
    negative = (point[0] - step * tangent[0], point[1] - state_step * tangent[1])
    finite_difference = (scalar_output(positive) - scalar_output(negative)) / (2.0 * step)
    print("vjp_normalized_direction", {
        "gradient": np.asarray(gradient_vector).tolist(),
        "norm": float(gradient_norm),
        "direction": np.asarray(normalized_vector).tolist(),
        "state_plus_minus": [float(negative[1]), float(positive[1])],
        "parameter_delta_plus": np.asarray(positive[0] - point[0]).tolist(),
        "parameter_delta_minus": np.asarray(negative[0] - point[0]).tolist(),
        "jvp": float(directional),
        "fd_h1e-3": float(finite_difference),
    })
    np.testing.assert_allclose(
        np.asarray(directional),
        np.asarray(finite_difference),
        rtol=2.0e-2,
        atol=2.0e-5,
    )
    assert np.isfinite(float(value))
