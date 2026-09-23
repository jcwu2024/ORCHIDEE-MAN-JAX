from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.shared_transformer import (
    _project_decoder_context,
    _source_family_bank,
    assemble_shared_transformer_context,
    initialize_shared_transformer_parameters,
    shared_transformer_raw_outputs,
)
from scripts.hpc.run_shared_transformer_admission import (
    admission_daily_objective,
    shared_config,
)
from tests.unit.test_shared_transformer import (
    GPP,
    _permuted_candidate_inputs,
)
from tests.unit.test_shared_transformer import (
    shared_transformer_setup as base_setup_fixture,  # noqa: F401
)


@pytest.fixture
def split_width_setup(base_setup_fixture):  # noqa: F811
    setup = dict(base_setup_fixture)
    config = replace(
        setup["config"],
        decoder_width=16,
        decoder_ffn_width=64,
        decoder_attention_heads=4,
    )
    parameters = initialize_shared_transformer_parameters(
        jax.random.PRNGKey(20260923),
        schema=setup["schema"],
        layout=setup["layout"],
        inputs=setup["assembled"],
        config=config,
        carbon_state_layout=setup["carbon_layout"],
    )
    return setup | {"config": config, "parameters": parameters}


def _run(setup, parameters=None, inputs=None, config=None):
    parameters = setup["parameters"] if parameters is None else parameters
    inputs = setup["assembled"] if inputs is None else inputs
    config = setup["config"] if config is None else config
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


def test_admission_config_defaults_to_legacy_widths_and_accepts_split_decoder_widths(base_setup_fixture):  # noqa: F811
    common = {
        "width": 32,
        "heads": 4,
        "ffn_width": 128,
        "backbone_blocks": 2,
        "decoder_blocks": 1,
        "query_chunk_size": 8,
    }
    legacy = shared_config(common)
    assert legacy.resolved_decoder_width == 32
    assert legacy.resolved_decoder_ffn_width == 128
    assert legacy.resolved_decoder_attention_heads == 4
    legacy_parameters = base_setup_fixture["parameters"]["shared_transformer"]
    assert base_setup_fixture["config"].resolved_decoder_width == base_setup_fixture["config"].width
    assert "decoder_bridge" not in legacy_parameters

    split = shared_config(
        common
        | {"decoder_width": 16, "decoder_ffn_width": 64, "decoder_heads": 4}
    )
    assert split.width == 32
    assert split.resolved_decoder_width == 16
    assert split.resolved_decoder_ffn_width == 64
    assert split.resolved_decoder_attention_heads == 4


def test_admission_daily_objective_microbatch_mean_sum_hard_and_gradient():
    def daily_objective(parameters, row):
        value = jnp.sum(parameters * row)
        auxiliary = {
            "loss": value,
            "hard": {"violation": jnp.asarray(row[0] > 0.0, dtype=jnp.int32)},
        }
        return value, auxiliary

    parameters = jnp.asarray([0.5, -1.0], dtype=jnp.float32)
    rows = jnp.asarray(
        [[1.0, 2.0], [-2.0, 0.5], [3.0, -1.0], [0.0, 4.0]],
        dtype=jnp.float32,
    )
    microbatch = admission_daily_objective(jax, jnp, daily_objective, 4)
    value, auxiliary = microbatch(parameters, rows)
    expected_rows = rows @ parameters
    np.testing.assert_allclose(float(value), float(jnp.mean(expected_rows)), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(float(auxiliary["loss"]), float(jnp.mean(expected_rows)), rtol=0.0, atol=0.0)
    assert int(auxiliary["hard"]["violation"]) == int(jnp.sum(rows[:, 0] > 0.0))

    gradient = jax.grad(lambda current: microbatch(current, rows)[0])(parameters)
    np.testing.assert_allclose(
        np.asarray(gradient), np.asarray(jnp.mean(rows, axis=0)), rtol=0.0, atol=0.0
    )
    legacy = admission_daily_objective(jax, jnp, daily_objective, 1)
    assert legacy is daily_objective


def test_split_width_keeps_backbone_and_uses_decoder_width_for_all_output_paths(
    split_width_setup,
):
    setup = split_width_setup
    parameters = setup["parameters"]["shared_transformer"]
    assert setup["config"].width == 32
    assert setup["config"].decoder_width == 16
    assert setup["config"].resolved_decoder_width == 16
    assert setup["config"].resolved_decoder_ffn_width == 64
    assert setup["config"].resolved_decoder_attention_heads == 4
    assert parameters["input_projection"]["kernel"].shape == (16, 32)
    assert parameters["decoder_bridge"]["kernel"].shape == (32, 16)
    assert parameters["field_embedding"].shape[-1] == 16
    assert parameters["label_embedding"].shape[-1] == 16
    assert parameters["local_value_input"]["kernel"].shape == (2, 16)
    assert parameters["coordinate_blocks"][0]["attention"]["query"]["kernel"].shape == (16, 16)
    assert parameters["coordinate_blocks"][0]["ffn"]["first"]["kernel"].shape == (16, 64)
    assert parameters["readout_first"]["kernel"].shape == (16, 16)

    raw, masks = _run(setup)
    assert set(raw) == {field.name for field in setup["schema"].fields}
    assert set(masks) == set(raw)
    for field in setup["schema"].fields:
        assert raw[field.name].shape == field.shape
        assert masks[field.name].shape == field.shape
        assert masks[field.name].dtype == jnp.bool_
        assert np.all(np.isfinite(np.asarray(raw[field.name])))


def test_split_width_preserves_pft_equivariance(split_width_setup):
    setup = split_width_setup
    original_raw, original_masks = _run(setup)
    permuted_inputs = _permuted_candidate_inputs(
        setup | {"layout": setup["full_layout"]},
    )
    observed_raw, observed_masks = _run(setup, inputs=permuted_inputs)
    for field, ownership in zip(setup["schema"].fields, setup["layout"].ledger, strict=True):
        expected_raw = np.asarray(original_raw[field.name]).copy()
        expected_mask = np.asarray(original_masks[field.name]).copy()
        if ownership.pft_axis is not None:
            first = [slice(None)] * expected_raw.ndim
            second = [slice(None)] * expected_raw.ndim
            first[ownership.pft_axis] = (
                setup["layout"].active_pft_indices[0] if field.full_pft_axis else 0
            )
            second[ownership.pft_axis] = (
                setup["layout"].active_pft_indices[1] if field.full_pft_axis else 1
            )
            first_value = expected_raw[tuple(first)].copy()
            expected_raw[tuple(first)] = expected_raw[tuple(second)]
            expected_raw[tuple(second)] = first_value
            first_mask = expected_mask[tuple(first)].copy()
            expected_mask[tuple(first)] = expected_mask[tuple(second)]
            expected_mask[tuple(second)] = first_mask
        np.testing.assert_allclose(
            np.asarray(observed_raw[field.name]), expected_raw, rtol=2.0e-5, atol=2.0e-6
        )
        np.testing.assert_array_equal(np.asarray(observed_masks[field.name]), expected_mask)


def test_decoder_bridge_rezeros_masked_tokens_after_nonzero_affine_bias(split_width_setup):
    setup = split_width_setup
    parameters = setup["parameters"]
    shared = parameters["shared_transformer"]
    bridge = dict(shared["decoder_bridge"])
    bridge["bias"] = jnp.full_like(bridge["bias"], 0.75)
    changed_shared = dict(shared)
    changed_shared["decoder_bridge"] = bridge
    parameters = dict(parameters) | {"shared_transformer": changed_shared}
    context = assemble_shared_transformer_context(
        parameters,
        setup["assembled"],
        normalization=setup["input_normalization"],
        layout=setup["layout"],
        config=setup["config"],
    )
    family_tokens, family_mask = _source_family_bank(context, setup["layout"])
    decoder_context, decoder_families = _project_decoder_context(
        parameters, context, family_tokens, family_mask
    )

    assert np.all(np.asarray(decoder_context.expert_axis_tokens)[
        ~np.asarray(context.expert_axis_mask)
    ] == 0.0)
    assert np.all(np.asarray(decoder_context.pft_tokens)[~np.asarray(context.pft_mask)] == 0.0)
    assert np.all(np.asarray(decoder_families)[~np.asarray(family_mask)] == 0.0)
    assert np.any(np.asarray(decoder_context.process_tokens) != 0.0)
    assert np.any(np.asarray(decoder_families)[np.asarray(family_mask)] != 0.0)


def test_split_width_gpp_keeps_global_and_same_pft_carbon_routes_with_finite_vjps(
    split_width_setup,
):
    setup = split_width_setup
    inputs = setup["assembled"]
    carbon_layout = setup["carbon_layout"]
    owners = np.asarray(carbon_layout.pft_owner_index)
    defined_state_indices = np.asarray(carbon_layout.state_indices)

    def objective(parameters, continuous_state):
        differentiable_inputs = inputs._replace(
            state=inputs.state._replace(other=continuous_state)
        )
        raw, _masks = _run(setup, parameters, differentiable_inputs)
        gpp = raw[GPP]
        weights = jnp.linspace(0.5, 1.5, gpp.size, dtype=gpp.dtype).reshape(gpp.shape)
        return jnp.sum(gpp * weights)

    state = jnp.asarray(inputs.state.other)
    value, (parameter_gradient, state_gradient) = jax.value_and_grad(
        objective, argnums=(0, 1)
    )(setup["parameters"], state)
    assert np.isfinite(float(value))
    assert all(
        np.all(np.isfinite(np.asarray(leaf)))
        for leaf in jax.tree_util.tree_leaves(parameter_gradient)
    )
    assert np.all(np.isfinite(np.asarray(state_gradient)))
    assert np.linalg.norm(
        np.asarray(parameter_gradient["shared_transformer"]["coordinate_blocks"][0]
                   ["attention"]["query"]["kernel"])
    ) > 0.0
    assert np.linalg.norm(
        np.asarray(parameter_gradient["shared_transformer"]["decoder_bridge"]["kernel"])
    ) > 0.0
    assert np.linalg.norm(
        np.asarray(parameter_gradient["shared_transformer"]["carbon"]["field_embedding"])
    ) > 0.0
    assert np.linalg.norm(np.asarray(state_gradient).reshape(-1)[defined_state_indices]) > 0.0
    assert np.any(owners == -1)
    assert any(np.any(owners == pft) for pft in range(len(setup["layout"].active_pft_indices)))


def test_gather_reuse_and_dynamic_reuse_match_within_split_width_model(split_width_setup):
    setup = split_width_setup
    inputs = setup["assembled"]
    state = jnp.asarray(inputs.state.other)

    def objective(parameters, continuous_state, execution):
        differentiable_inputs = inputs._replace(
            state=inputs.state._replace(other=continuous_state)
        )
        config = replace(setup["config"], decoder_execution=execution)
        raw, _masks = _run(setup, parameters, differentiable_inputs, config)
        gpp = raw[GPP]
        weights = jnp.linspace(0.5, 1.5, gpp.size, dtype=gpp.dtype).reshape(gpp.shape)
        return jnp.sum(gpp * weights)

    results = {}
    for execution in ("gather_reuse", "gather_dynamic_reuse"):
        results[execution] = jax.value_and_grad(objective, argnums=(0, 1))(
            setup["parameters"], state, execution
        )
    reference_raw, reference_masks = _run(
        setup,
        inputs=inputs,
        config=replace(setup["config"], decoder_execution="gather_reuse"),
    )
    dynamic_raw, dynamic_masks = _run(
        setup,
        inputs=inputs,
        config=replace(setup["config"], decoder_execution="gather_dynamic_reuse"),
    )
    for name in reference_raw:
        np.testing.assert_allclose(
            np.asarray(dynamic_raw[name]), np.asarray(reference_raw[name]), rtol=2.0e-5, atol=2.0e-6
        )
        np.testing.assert_array_equal(np.asarray(dynamic_masks[name]), np.asarray(reference_masks[name]))
    base_value, (base_parameter_gradient, base_state_gradient) = results["gather_reuse"]
    observed_value, (observed_parameter_gradient, observed_state_gradient) = results[
        "gather_dynamic_reuse"
    ]
    np.testing.assert_allclose(float(observed_value), float(base_value), rtol=2.0e-5, atol=2.0e-6)
    for observed, expected in zip(
        jax.tree_util.tree_leaves(observed_parameter_gradient),
        jax.tree_util.tree_leaves(base_parameter_gradient),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), rtol=5.0e-4, atol=2.0e-5)
    np.testing.assert_allclose(
        np.asarray(observed_state_gradient), np.asarray(base_state_gradient), rtol=5.0e-4, atol=2.0e-5
    )
