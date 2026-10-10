"""Small synthetic CPU implementation checks; independent AD review is separate."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.shared_latent_reader import (
    LATENT_SLOTS,
    READER_WIDTH,
    build_expert_latents,
    shared_latent_reader_config,
    shared_latent_reader_raw_outputs,
)
from research.daily_coarse_graining.daily_operator.shared_transformer import (
    SharedTransformerConfig,
    _project_decoder_context,
    _source_family_bank,
    assemble_shared_transformer_context,
    initialize_shared_transformer_parameters,
    shared_transformer_parameter_count,
    shared_transformer_raw_outputs,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_carbon_feedback import (
    build_carbon_feedback_state_layout,
    load_coupled_gpp_carbon_feedback_contract,
)
from tests.unit.test_daily_operator_mechanistic_output import mechanistic_setup


def static_setup():
    with (
        patch("tests.unit.test_daily_operator_successor_process_head.initialize_successor_parameters", return_value={}),
        patch("tests.unit.test_daily_operator_mechanistic_output.initialize_strong_baseline_parameters", return_value={}),
    ):
        base = dict(mechanistic_setup.__wrapped__())
    carbon = build_carbon_feedback_state_layout(
        base["adapter"].contract, baseline_layout=base["layout"],
        architecture_contract=base["architecture_contract"],
        coupled_contract=load_coupled_gpp_carbon_feedback_contract(),
    )
    return base | {"carbon_layout": carbon}


@pytest.fixture(scope="module")
def latent_setup():
    base = static_setup()
    full_layout = base["layout"]
    # Include a local-state field as well as GPP, maintenance and non-PFT hydrology.
    names = ["daily.fast_gpp", "daily.fast_maintenance_respiration", "inventory.water.snow.outgoing_fraction"]
    local_field = next(field.name for index, field in enumerate(base["learned_schema"].fields)
                       if np.asarray(full_layout.local_state_indices[index]).size
                       and int(np.prod(field.shape)) <= 32 and field.name not in names)
    names.append(local_field)
    selected = [next(i for i, field in enumerate(base["learned_schema"].fields) if field.name == name)
                for name in names]
    schema = replace(base["learned_schema"], fields=tuple(base["learned_schema"].fields[i] for i in selected))
    layout = replace(full_layout, ledger=tuple(full_layout.ledger[i] for i in selected),
                     field_expert_index=full_layout.field_expert_index[selected],
                     field_label_index=full_layout.field_label_index[selected],
                     local_state_indices=tuple(full_layout.local_state_indices[i] for i in selected))
    config = SharedTransformerConfig(
        width=32, attention_heads=4, ffn_width=128, backbone_blocks=1,
        decoder_width=16, decoder_attention_heads=4, decoder_ffn_width=64,
        input_width=16, input_attention_heads=4, input_ffn_width=64,
        forcing_width=16, forcing_heads=4, forcing_ffn_width=64, forcing_blocks=1,
        decoder_architecture="latent_reader", output_chunk_size=8,
    )
    parameters = initialize_shared_transformer_parameters(
        jax.random.PRNGKey(20261004), schema=schema, layout=layout, inputs=base["assembled"],
        config=config, carbon_state_layout=base["carbon_layout"],
    )
    context = assemble_shared_transformer_context(
        parameters, base["assembled"], normalization=base["input_normalization"], layout=layout, config=config,
    )
    return base | dict(full_layout=full_layout, schema=schema, layout=layout, config=config,
                       parameters=parameters, context=context, local_field=local_field)


def raw_outputs(setup, context=None):
    return shared_transformer_raw_outputs(
        setup["parameters"], setup["context"] if context is None else context, setup["assembled"],
        schema=setup["schema"], layout=setup["layout"], input_normalization=setup["input_normalization"],
        config=setup["config"], carbon_state_layout=setup["carbon_layout"],
    )


def test_selected_parameter_shapes_and_full_canonical_count(latent_setup):
    setup = latent_setup
    config = shared_latent_reader_config()
    config.validate()
    assert SharedTransformerConfig().decoder_architecture == "coordinate"
    abstract = jax.eval_shape(lambda key: initialize_shared_transformer_parameters(
        key, schema=setup["learned_schema"], layout=setup["full_layout"], inputs=setup["assembled"],
        config=config, carbon_state_layout=setup["carbon_layout"]), jax.random.PRNGKey(0))
    tree = abstract["shared_transformer"]
    assert "coordinate_blocks" not in tree
    assert tree["latent_slots"].shape == (32, 384)
    assert tree["reader_projection"]["kernel"].shape == (384, 64)
    assert tree["readout_first"]["kernel"].shape == (64, 256)
    assert len(tree["latent_blocks"]) == len(tree["reader_blocks"]) == 2
    assert shared_transformer_parameter_count(abstract) == 66_123_713
    assert {leaf.dtype for leaf in jax.tree_util.tree_leaves(abstract)} == {jnp.dtype("float32")}


def test_small_forward_is_finite_restores_shapes_and_distinguishes_fields(latent_setup):
    setup = latent_setup
    outputs, masks = raw_outputs(setup)
    jax.block_until_ready(outputs)
    for field in setup["schema"].fields:
        assert outputs[field.name].shape == masks[field.name].shape == field.shape
        assert outputs[field.name].dtype == jnp.float32
        assert np.all(np.isfinite(outputs[field.name]))
        assert np.all(np.asarray(outputs[field.name])[~np.asarray(masks[field.name])] == 0)
    # Local value and field identity are consumed even though the latents are shared.
    assert np.any(np.asarray(outputs[setup["local_field"]]) != 0)
    assert not np.all(np.asarray(outputs["daily.fast_maintenance_respiration"]) == 0)


def test_inactive_pft_and_source_masks_remain_finite_and_zero(latent_setup):
    setup = latent_setup
    context = setup["context"]._replace(
        pft_mask=jnp.zeros_like(setup["context"].pft_mask),
        process_source_mask=jnp.zeros_like(setup["context"].process_source_mask),
    )
    outputs, masks = raw_outputs(setup, context)
    jax.block_until_ready(outputs)
    for field in setup["schema"].fields:
        assert np.all(np.isfinite(outputs[field.name]))
        assert np.all(np.asarray(outputs[field.name])[~np.asarray(masks[field.name])] == 0)
    assert not np.any(masks["daily.fast_gpp"])
    assert not np.any(outputs["daily.fast_gpp"])


def test_latent_shape_and_family_permission_expert_isolation(latent_setup):
    setup = latent_setup
    families, family_mask = _source_family_bank(setup["context"], setup["layout"])
    context, families = _project_decoder_context(setup["parameters"], setup["context"], families, family_mask)
    latent = build_expert_latents(setup["parameters"], context, families, family_mask, config=setup["config"])
    assert latent.shape == (len(setup["layout"].expert_names), LATENT_SLOTS, 16)
    assert READER_WIDTH == 64
    # Illegal family values cannot alter this expert's latents.
    changed = jnp.where(family_mask[..., None], families, 1000.0)
    actual = build_expert_latents(setup["parameters"], context, changed, family_mask, config=setup["config"])
    np.testing.assert_allclose(actual, latent, rtol=1e-5, atol=1e-6)
    one_expert = build_expert_latents(setup["parameters"],
        context._replace(process_tokens=context.process_tokens.at[0].add(2.0)), families, family_mask,
        config=setup["config"])
    np.testing.assert_array_equal(one_expert[1:], latent[1:])


def test_expert_grouping_preserves_field_reference_forward(latent_setup):
    setup = latent_setup
    grouped, masks = shared_latent_reader_raw_outputs(
        setup["parameters"], setup["context"], setup["assembled"],
        schema=setup["schema"], layout=setup["layout"],
        input_normalization=setup["input_normalization"], config=setup["config"],
        carbon_state_layout=setup["carbon_layout"], field_reference=False,
    )
    reference, reference_masks = shared_latent_reader_raw_outputs(
        setup["parameters"], setup["context"], setup["assembled"],
        schema=setup["schema"], layout=setup["layout"],
        input_normalization=setup["input_normalization"], config=setup["config"],
        carbon_state_layout=setup["carbon_layout"], field_reference=True,
    )
    jax.block_until_ready(reference)
    for name in grouped:
        np.testing.assert_array_equal(masks[name], reference_masks[name])
        np.testing.assert_allclose(grouped[name], reference[name], rtol=1e-5, atol=1e-6)


def test_public_latent_dispatch_selects_field_default_and_keeps_expert_diagnostic():
    config = shared_latent_reader_config()
    assert config.latent_reader_execution == "field"
    assert SharedTransformerConfig().decoder_architecture == "coordinate"
    schema = SimpleNamespace(fields=())
    layout = SimpleNamespace(ledger=())
    target = "research.daily_coarse_graining.daily_operator.shared_latent_reader.shared_latent_reader_raw_outputs"
    with patch(target, return_value=({}, {})) as decoder:
        shared_transformer_raw_outputs({}, None, None, schema=schema, layout=layout,
                                       input_normalization=None, config=config)
        assert decoder.call_args.kwargs["field_reference"] is True
        shared_transformer_raw_outputs({}, None, None, schema=schema, layout=layout,
            input_normalization=None, config=replace(config, latent_reader_execution="expert"))
        assert decoder.call_args.kwargs["field_reference"] is False
