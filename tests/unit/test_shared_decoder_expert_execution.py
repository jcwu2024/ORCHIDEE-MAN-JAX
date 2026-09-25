"""Parity and native-AD checks for diagnostic expert128 shared outputs."""

from __future__ import annotations

from dataclasses import replace

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
