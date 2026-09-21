from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.shared_transformer import (
    SharedTransformerConfig,
    _tokenizer_config,
    assemble_shared_transformer_context,
    initialize_shared_transformer_parameters,
    shared_transformer_parameter_count,
    shared_transformer_raw_outputs,
)
from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import _attention
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_carbon_feedback import (
    build_carbon_feedback_state_layout,
    load_coupled_gpp_carbon_feedback_contract,
)
from tests.unit.test_daily_operator_hierarchical_coordinate_attention import (
    _permuted_candidate_inputs,
)
from tests.unit.test_daily_operator_mechanistic_output import mechanistic_setup

GPP = "daily.fast_gpp"
MAINTENANCE = "daily.fast_maintenance_respiration"
HYDROLOGY = "inventory.water.snow.outgoing_fraction"


@pytest.fixture(scope="module")
def shared_transformer_setup():
    base = dict(mechanistic_setup.__wrapped__())
    full_layout = base["layout"]
    selected_names = (GPP, MAINTENANCE, HYDROLOGY)
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
    carbon_layout = build_carbon_feedback_state_layout(
        base["adapter"].contract,
        baseline_layout=full_layout,
        architecture_contract=base["architecture_contract"],
        coupled_contract=load_coupled_gpp_carbon_feedback_contract(),
    )
    config = SharedTransformerConfig(
        width=32,
        attention_heads=4,
        ffn_width=128,
        backbone_blocks=2,
        coordinate_blocks=1,
        readout_width=16,
        output_chunk_size=8,
        input_width=16,
        input_attention_heads=4,
        input_ffn_width=64,
        forcing_width=16,
        forcing_heads=4,
        forcing_ffn_width=64,
        forcing_blocks=1,
        axial_blocks=1,
    )
    parameters = initialize_shared_transformer_parameters(
        jax.random.PRNGKey(20260921),
        schema=schema,
        layout=layout,
        inputs=base["assembled"],
        config=config,
        carbon_state_layout=carbon_layout,
    )
    base.pop("parameters", None)
    return base | {
        "schema": schema,
        "layout": layout,
        "full_layout": full_layout,
        "carbon_layout": carbon_layout,
        "config": config,
        "parameters": parameters,
    }


def _mapping_keys(value):
    result = set()
    if isinstance(value, Mapping):
        result.update(value)
        for child in value.values():
            result.update(_mapping_keys(child))
    elif isinstance(value, (tuple, list)):
        for child in value:
            result.update(_mapping_keys(child))
    return result


def _context(setup, inputs=None):
    return assemble_shared_transformer_context(
        setup["parameters"],
        setup["assembled"] if inputs is None else inputs,
        normalization=setup["input_normalization"],
        layout=setup["layout"],
        config=setup["config"],
    )


def _outputs(setup, inputs, context=None, parameters=None, carbon_layout=None):
    current_parameters = setup["parameters"] if parameters is None else parameters
    current_context = _context(setup, inputs) if context is None else context
    return shared_transformer_raw_outputs(
        current_parameters,
        current_context,
        inputs,
        schema=setup["schema"],
        layout=setup["layout"],
        input_normalization=setup["input_normalization"],
        config=setup["config"],
        carbon_state_layout=(
            setup["carbon_layout"] if carbon_layout is None else carbon_layout
        ),
    )


def test_shared_transformer_has_one_shared_stack_and_fresh_fp32_parameters(
    shared_transformer_setup,
):
    parameters = shared_transformer_setup["parameters"]
    shared = parameters["shared_transformer"]
    assert {leaf.dtype for leaf in jax.tree_util.tree_leaves(parameters)} == {jnp.dtype("float32")}
    assert len(shared["backbone_blocks"]) == 2
    assert len(shared["coordinate_blocks"]) == 1
    assert shared["process_query"].shape == (7, 32)
    assert shared["readout_second"]["kernel"].shape == (16, 1)
    assert np.any(np.asarray(shared["readout_first"]["kernel"]) != 0.0)
    assert np.any(np.asarray(shared["readout_second"]["kernel"]) != 0.0)
    assert {
        "expert_blocks",
        "label_heads",
        "structured_gpp",
        "distribution_aware_maintenance",
        "carbon_feedback_adapter",
    }.isdisjoint(_mapping_keys(parameters))


def test_shared_mode_enables_fp32_cosine_attention_without_changing_legacy_default():
    linear = {
        "kernel": jnp.eye(4, dtype=jnp.float32),
        "bias": jnp.zeros((4,), dtype=jnp.float32),
    }
    parameters = {name: linear for name in ("query", "key", "value", "output")}
    query = jnp.asarray([[[1.0, 0.0, 1.0, 0.0]]], dtype=jnp.float32)
    key_value = jnp.asarray(
        [[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]],
        dtype=jnp.float32,
    )
    mask = jnp.ones((1, 1, 2), dtype=jnp.bool_)
    legacy = _attention(parameters, query, key_value, heads=2, mask=mask)
    legacy_scaled = _attention(parameters, 10.0 * query, key_value, heads=2, mask=mask)
    cosine = _attention(
        parameters,
        query,
        key_value,
        heads=2,
        mask=mask,
        score_dtype=jnp.float32,
        normalize_qk=True,
    )
    cosine_scaled = _attention(
        parameters,
        10.0 * query,
        key_value,
        heads=2,
        mask=mask,
        score_dtype=jnp.float32,
        normalize_qk=True,
    )
    assert not np.allclose(np.asarray(legacy_scaled), np.asarray(legacy))
    np.testing.assert_allclose(np.asarray(cosine_scaled), np.asarray(cosine), rtol=1.0e-6)
    assert cosine.dtype == jnp.float32

    default = SharedTransformerConfig()
    tokenizer = _tokenizer_config(default)
    assert default.output_chunk_size == 64
    assert tokenizer.neural_precision.attention_scores_fp32
    assert tokenizer.neural_precision.source_family_pooling_fp32
    assert tokenizer.neural_precision.normalize_attention_qk


def test_default_topology_parameter_count_uses_the_approved_capacity(
    shared_transformer_setup,
):
    setup = shared_transformer_setup
    abstract = jax.eval_shape(
        lambda key: initialize_shared_transformer_parameters(
            key,
            schema=setup["learned_schema"],
            layout=setup["full_layout"],
            inputs=setup["assembled"],
            carbon_state_layout=setup["carbon_layout"],
        ),
        jax.random.PRNGKey(0),
    )
    shared = abstract["shared_transformer"]
    count = shared_transformer_parameter_count(abstract)
    assert len(shared["backbone_blocks"]) == 8
    assert len(shared["coordinate_blocks"]) == 2
    assert count == 75_578_497


def test_source_masks_are_applied_before_the_shared_backbone(
    shared_transformer_setup,
):
    setup = shared_transformer_setup
    inputs = setup["assembled"]
    baseline = _context(setup)
    expected = np.asarray(setup["layout"].source_allowed)[:, np.asarray(baseline.source_family_index)]
    expected &= np.asarray(baseline.source_mask)[None, :]
    np.testing.assert_array_equal(np.asarray(baseline.process_source_mask), expected)

    forcing = np.asarray(inputs.forcing.values).copy()
    for name in ("Rainf", "Snowf"):
        forcing[:, setup["head"].forcing_indices[name]] += 1.0e3
    perturbed_inputs = inputs._replace(forcing=inputs.forcing._replace(values=forcing))
    perturbed = _context(setup, perturbed_inputs)
    canopy = setup["layout"].expert_names.index("canopy_gpp")
    hydrology = setup["layout"].expert_names.index("hydrology_snow")
    np.testing.assert_allclose(
        np.asarray(perturbed.process_tokens[canopy]),
        np.asarray(baseline.process_tokens[canopy]),
        rtol=0.0,
        atol=0.0,
    )
    assert not np.allclose(
        np.asarray(perturbed.process_tokens[hydrology]),
        np.asarray(baseline.process_tokens[hydrology]),
    )


def test_chunked_raw_outputs_have_exact_shapes_masks_and_inactive_pft_behavior(
    shared_transformer_setup,
):
    setup = shared_transformer_setup
    inputs = setup["assembled"]
    pft = inputs.pft._replace(
        pft_fraction=np.asarray([np.asarray(inputs.pft.pft_fraction)[0], 0.0]),
        active_pft_mask=np.asarray([True, False]),
    )
    inactive = inputs._replace(
        pft=pft,
        canonical_input=inputs.canonical_input._replace(pft=pft),
    )
    raw, masks = _outputs(setup, inactive)
    assert set(raw) == {field.name for field in setup["schema"].fields}
    assert set(masks) == set(raw)
    for field in setup["schema"].fields:
        assert raw[field.name].shape == field.shape
        assert masks[field.name].shape == field.shape
        assert masks[field.name].dtype == jnp.bool_
        assert np.all(np.isfinite(np.asarray(raw[field.name])))
    assert not bool(np.asarray(masks[GPP])[0, 1])
    assert not np.any(np.asarray(masks[MAINTENANCE])[:, 1, :])
    assert setup["parameters"]["shared_transformer"]["readout_second"]["kernel"].shape[-1] == 1


def test_coordinate_outputs_are_invariant_to_execution_chunk_size(shared_transformer_setup):
    setup = shared_transformer_setup
    inputs = setup["assembled"]
    context = _context(setup)
    expected, expected_masks = _outputs(setup, inputs, context=context)
    single_coordinate_config = replace(setup["config"], output_chunk_size=1)
    observed, observed_masks = shared_transformer_raw_outputs(
        setup["parameters"],
        context,
        inputs,
        schema=setup["schema"],
        layout=setup["layout"],
        input_normalization=setup["input_normalization"],
        config=single_coordinate_config,
        carbon_state_layout=setup["carbon_layout"],
    )
    for name in expected:
        np.testing.assert_allclose(
            np.asarray(observed[name]),
            np.asarray(expected[name]),
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        np.testing.assert_array_equal(
            np.asarray(observed_masks[name]),
            np.asarray(expected_masks[name]),
        )


def test_gradients_reach_shared_backbone_decoder_readout_and_raw_carbon_tokens(
    shared_transformer_setup,
):
    setup = shared_transformer_setup
    inputs = setup["assembled"]

    def loss(parameters):
        context = assemble_shared_transformer_context(
            parameters,
            inputs,
            normalization=setup["input_normalization"],
            layout=setup["layout"],
            config=setup["config"],
        )
        raw, _masks = _outputs(setup, inputs, context=context, parameters=parameters)
        return sum(jnp.sum(value * value) for value in raw.values())

    gradients = jax.grad(loss)(setup["parameters"])
    leaves = jax.tree_util.tree_leaves(gradients)
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)
    shared = gradients["shared_transformer"]
    assert np.linalg.norm(np.asarray(shared["backbone_blocks"][0]["attention"]["query"]["kernel"])) > 0.0
    assert np.linalg.norm(np.asarray(shared["coordinate_blocks"][0]["attention"]["query"]["kernel"])) > 0.0
    assert np.linalg.norm(np.asarray(shared["readout_second"]["kernel"])) > 0.0
    assert np.linalg.norm(np.asarray(shared["carbon"]["field_embedding"])) > 0.0


def test_carbon_layout_shape_drift_is_rejected(shared_transformer_setup):
    setup = shared_transformer_setup
    invalid = setup["carbon_layout"]._replace(
        field_names=setup["carbon_layout"].field_names[:-1],
    )
    with pytest.raises(ValueError, match="carbon state field layout drift"):
        _outputs(setup, setup["assembled"], carbon_layout=invalid)


def test_shared_transformer_is_pft_permutation_equivariant(shared_transformer_setup):
    setup = shared_transformer_setup
    original_raw, original_masks = _outputs(setup, setup["assembled"])
    permuted_inputs = _permuted_candidate_inputs(
        setup | {"layout": setup["full_layout"]},
    )
    observed_raw, observed_masks = _outputs(setup, permuted_inputs)
    for field, ownership in zip(setup["schema"].fields, setup["layout"].ledger, strict=True):
        expected_raw = np.asarray(original_raw[field.name]).copy()
        expected_mask = np.asarray(original_masks[field.name]).copy()
        if ownership.pft_axis is not None:
            first = [slice(None)] * expected_raw.ndim
            second = [slice(None)] * expected_raw.ndim
            first[ownership.pft_axis] = setup["layout"].active_pft_indices[0] if field.full_pft_axis else 0
            second[ownership.pft_axis] = setup["layout"].active_pft_indices[1] if field.full_pft_axis else 1
            first_value = expected_raw[tuple(first)].copy()
            expected_raw[tuple(first)] = expected_raw[tuple(second)]
            expected_raw[tuple(second)] = first_value
            first_mask = expected_mask[tuple(first)].copy()
            expected_mask[tuple(first)] = expected_mask[tuple(second)]
            expected_mask[tuple(second)] = first_mask
        np.testing.assert_allclose(
            np.asarray(observed_raw[field.name]),
            expected_raw,
            rtol=2.0e-5,
            atol=2.0e-6,
            err_msg=field.name,
        )
        np.testing.assert_array_equal(
            np.asarray(observed_masks[field.name]),
            expected_mask,
            err_msg=field.name,
        )
