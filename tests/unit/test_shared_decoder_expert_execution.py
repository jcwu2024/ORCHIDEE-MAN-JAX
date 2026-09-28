"""Parity and native-AD checks for diagnostic expert128 shared outputs."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.shared_decoder_expert_execution import (
    shared_transformer_raw_outputs_expert128,
)
from research.daily_coarse_graining.daily_operator.shared_transformer import (
    initialize_shared_transformer_parameters,
    shared_transformer_raw_outputs,
)
from tests.unit.test_shared_transformer import (
    _context,
)
from tests.unit.test_shared_transformer import (
    shared_transformer_setup as _shared_transformer_setup,
)


def _expert_setup(base):
    selected_names = (
        "daily.fast_gpp",
        "daily.fast_maintenance_respiration",
        "inventory.water.snow.outgoing_fraction",
        "inventory.water.qsintveg.storage.outgoing_fraction",
    )
    full_schema = base["learned_schema"]
    full_layout = base["full_layout"]
    selected = tuple(
        next(index for index, field in enumerate(full_schema.fields) if field.name == name)
        for name in selected_names
    )
    schema = replace(
        full_schema,
        fields=tuple(full_schema.fields[index] for index in selected),
    )
    layout = replace(
        full_layout,
        ledger=tuple(full_layout.ledger[index] for index in selected),
        field_expert_index=np.asarray(full_layout.field_expert_index)[list(selected)],
        field_label_index=np.asarray(full_layout.field_label_index)[list(selected)],
        local_state_indices=tuple(full_layout.local_state_indices[index] for index in selected),
    )
    config = replace(base["config"], decoder_execution="gather_dynamic_reuse")
    parameters = initialize_shared_transformer_parameters(
        jax.random.PRNGKey(20260925),
        schema=schema,
        layout=layout,
        inputs=base["assembled"],
        config=config,
        carbon_state_layout=base["carbon_layout"],
    )
    return base | {
        "schema": schema,
        "layout": layout,
        "config": config,
        "parameters": parameters,
    }


@pytest.fixture(scope="module")
def expert_setup():
    return _expert_setup(_shared_transformer_setup.__wrapped__())


def _run(setup, function, parameters, context, inputs):
    return function(
        parameters,
        context,
        inputs,
        schema=setup["schema"],
        layout=setup["layout"],
        input_normalization=setup["input_normalization"],
        config=setup["config"],
        carbon_state_layout=setup["carbon_layout"],
    )


def _scalar_objective(outputs):
    terms = []
    for index, value in enumerate(outputs.values()):
        weights = jnp.linspace(0.2, 1.0, value.size, dtype=jnp.float32).reshape(value.shape)
        terms.append(jnp.sum(value * weights) / value.size)
    return sum(terms)


def test_expert128_matches_raw_outputs_and_restores_full_masks(expert_setup):
    setup = expert_setup
    experts = np.asarray(setup["layout"].field_expert_index)
    assert experts[2] == experts[3]
    inputs = setup["assembled"]
    pft = inputs.pft._replace(
        pft_fraction=np.asarray([np.asarray(inputs.pft.pft_fraction)[0], 0.0]),
        active_pft_mask=np.asarray([True, False]),
    )
    inputs = inputs._replace(
        pft=pft,
        canonical_input=inputs.canonical_input._replace(pft=pft),
    )
    context = _context(setup, inputs)
    expected_outputs, expected_masks = _run(
        setup, shared_transformer_raw_outputs,
        setup["parameters"], context, inputs,
    )
    actual_outputs, actual_masks = _run(
        setup, shared_transformer_raw_outputs_expert128,
        setup["parameters"], context, inputs,
    )
    assert tuple(actual_outputs) == tuple(field.name for field in setup["schema"].fields)
    assert tuple(actual_masks) == tuple(actual_outputs)
    for field in setup["schema"].fields:
        np.testing.assert_allclose(
            actual_outputs[field.name], expected_outputs[field.name], rtol=1e-5, atol=1e-6,
        )
        np.testing.assert_array_equal(actual_masks[field.name], expected_masks[field.name])
        assert actual_outputs[field.name].shape == field.shape
    assert not bool(np.asarray(actual_masks["daily.fast_gpp"])[0, 1])
    assert not np.any(np.asarray(actual_masks["daily.fast_maintenance_respiration"])[:, 1, :])


def test_expert128_preserves_parameter_context_and_input_gradients(expert_setup):
    setup = expert_setup
    parameters = setup["parameters"]
    inputs = setup["assembled"]
    context = _context(setup)

    def expected_objective(p, c, x):
        outputs, _ = _run(setup, shared_transformer_raw_outputs, p, c, x)
        return _scalar_objective(outputs)

    def actual_objective(p, c, x):
        outputs, _ = _run(setup, shared_transformer_raw_outputs_expert128, p, c, x)
        return _scalar_objective(outputs)

    expected_gradients = jax.grad(
        expected_objective, argnums=(0, 1, 2), allow_int=True,
    )(parameters, context, inputs)
    actual_gradients = jax.grad(
        actual_objective, argnums=(0, 1, 2), allow_int=True,
    )(parameters, context, inputs)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(actual_gradients),
        jax.tree_util.tree_leaves(expected_gradients),
        strict=True,
    ):
        if np.asarray(actual).dtype == jax.dtypes.float0:
            assert np.asarray(expected).dtype == jax.dtypes.float0
            continue
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_expert128_requires_static_dynamic_reuse_contract(expert_setup):
    setup = expert_setup
    original_config = setup["config"]
    invalid = replace(original_config, decoder_execution="gather_reuse")
    with pytest.raises(ValueError, match="gather_dynamic_reuse"):
        shared_transformer_raw_outputs_expert128(
            setup["parameters"], _context(setup), setup["assembled"],
            schema=setup["schema"], layout=setup["layout"],
            input_normalization=setup["input_normalization"], config=invalid,
            carbon_state_layout=setup["carbon_layout"],
        )


def test_union_projection_preserves_mixed_masks_bias_sentinel_and_native_ad(expert_setup):
    setup = expert_setup
    local_indices = [np.asarray(value).copy() for value in setup['layout'].local_state_indices]
    for value in local_indices:
        if value.size:
            value.reshape(-1)[::3] = -1
    setup = setup | {'layout': replace(setup['layout'], local_state_indices=tuple(local_indices))}
    parameters = jax.tree_util.tree_map_with_path(
        lambda path, value: value + jnp.linspace(0.03, 0.09, value.size).reshape(value.shape)
        if isinstance(path[-1], jax.tree_util.DictKey) and path[-1].key == 'bias' else value,
        setup['parameters'],
    )
    inputs = setup['assembled']
    defined = jnp.asarray(inputs.state_defined.other).at[::4].set(False)
    inputs = inputs._replace(
        state_defined=inputs.state_defined._replace(other=defined),
        state=inputs.state._replace(other=jnp.asarray(inputs.state.other) + 0.25),
    )
    pft = inputs.pft._replace(
        active_pft_mask=jnp.asarray([True, False]),
        pft_fraction=jnp.asarray(inputs.pft.pft_fraction).at[1].set(0.0),
    )
    inputs = inputs._replace(pft=pft, canonical_input=inputs.canonical_input._replace(pft=pft))
    context = _context(setup | {'parameters': parameters}, inputs)
    context = context._replace(expert_axis_mask=jnp.asarray(context.expert_axis_mask).at[:, ::2].set(False))
    union = partial(shared_transformer_raw_outputs_expert128, dynamic_projection_execution='union')

    expected, expected_masks = _run(setup, shared_transformer_raw_outputs_expert128, parameters, context, inputs)
    actual, masks = _run(setup, union, parameters, context, inputs)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(masks[name], expected_masks[name])
        assert np.all(np.isfinite(actual[name]))
    assert not bool(np.asarray(masks['daily.fast_gpp'])[0, 1])

    def objective(function, p, c, x):
        return _scalar_objective(_run(setup, function, p, c, x)[0])
    expected_gradients = jax.grad(partial(objective, shared_transformer_raw_outputs_expert128),
                                  argnums=(0, 1, 2), allow_int=True)(parameters, context, inputs)
    actual_gradients = jax.grad(partial(objective, union), argnums=(0, 1, 2),
                                allow_int=True)(parameters, context, inputs)
    for a, b in zip(jax.tree_util.tree_leaves(actual_gradients),
                    jax.tree_util.tree_leaves(expected_gradients), strict=True):
        if np.asarray(a).dtype == jax.dtypes.float0:
            continue
        assert np.all(np.isfinite(a))
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    pgrad, cgrad, xgrad = actual_gradients
    assert np.linalg.norm(pgrad['shared_transformer']['coordinate_blocks'][0]['attention']['key']['kernel']) > 1e-7
    assert np.linalg.norm(pgrad['shared_transformer']['local_value_input']['kernel']) > 1e-7
    assert np.linalg.norm(cgrad.expert_axis_tokens) > 1e-7
    assert np.linalg.norm(xgrad.state.other) > 1e-7
    np.testing.assert_array_equal(np.asarray(xgrad.state.other)[~np.asarray(defined)], 0.0)
