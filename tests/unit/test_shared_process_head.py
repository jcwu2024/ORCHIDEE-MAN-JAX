from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator.neural_architecture import OutputFieldSpec
from research.daily_coarse_graining.daily_operator.parameter_routing import DEFAULT_PFT_PARAMETER_ROUTING
from research.daily_coarse_graining.daily_operator.shared_process_head import (
    apply_shared_gpp_factor,
    build_shared_process_head,
    transform_shared_output,
)
from research.daily_coarse_graining.daily_operator.shared_training import compose_shared_groups
from research.daily_coarse_graining.daily_operator.shared_transformer import SharedTransformerConfig
from tests.unit.test_daily_operator_mechanistic_output import mechanistic_setup


def _mapping_keys(value):
    keys = set()
    if isinstance(value, Mapping):
        keys.update(value)
        for child in value.values():
            keys.update(_mapping_keys(child))
    elif isinstance(value, (tuple, list)):
        for child in value:
            keys.update(_mapping_keys(child))
    return keys


def test_slow_stock_prior_and_gradient_are_not_half_daily_depletion():
    field = OutputFieldSpec("inventory.carbon.litter.outgoing_fraction", (3,), 0, "unit_interval")
    norm = SimpleNamespace(mean={field.name: jnp.array([0., 2e-5, 1.])})
    def fn(raw):
        return transform_shared_output(
            raw,
            jnp.array([True, True, False]),
            field=field,
            normalization=norm,
            inputs=None,
        )
    actual = jax.jit(fn)(jnp.zeros(3))
    np.testing.assert_allclose(actual, [1e-8, 2e-5, 0.], rtol=1e-6)
    grad = jax.grad(lambda raw: fn(raw).sum())(jnp.zeros(3))
    assert np.all(np.isfinite(grad))
    assert grad[0] > 0 and grad[1] > 0 and grad[2] == 0


def test_explicit_gpp_factor_has_no_other_physical_parameter_route():
    routing = DEFAULT_PFT_PARAMETER_ROUTING
    values = jnp.ones((3, len(routing.channel_names)))
    values = values.at[:, routing.index("vcmax25")].set(jnp.array([100., 200., 300.]))
    traits = jnp.zeros((3, 3)).at[0, 0].set(1.)

    def fn(parameters):
        pft = SimpleNamespace(pft_parameters=parameters, active_pft_mask=jnp.array([True, True, False]),
                              pft_fraction=jnp.array([.1, .9, 0.]), pft_traits=traits)
        return apply_shared_gpp_factor(jnp.array([[2., 2., 2.]]), SimpleNamespace(pft=pft))

    np.testing.assert_allclose(jax.jit(fn)(values), [[0., 4., 0.]])
    gradient = jax.grad(lambda x: fn(x).sum())(values)
    expected = np.zeros_like(values)
    expected[1, routing.index("vcmax25")] = .02
    np.testing.assert_allclose(gradient, expected)


def test_shared_groups_average_horizons_before_three_group_mean():
    actual = compose_shared_groups(3., 6., 9., recurrent_flux=(12., 18.), recurrent_stock=(15., 21.))
    assert actual == (3. + 12. + 15.) / 3.
    assert compose_shared_groups(3., 6., 9.) == 6.


def test_canonical_shared_head_forward_covers_full_learned_schema_without_legacy_weights():
    # Reuse the canonical fixture's metadata construction without allocating any
    # of its superseded successor or 49.7M-parameter baseline trees.
    with (
        patch(
            "tests.unit.test_daily_operator_successor_process_head.initialize_successor_parameters",
            return_value={},
        ),
        patch(
            "tests.unit.test_daily_operator_mechanistic_output.initialize_strong_baseline_parameters",
            return_value={},
        ),
    ):
        setup = mechanistic_setup.__wrapped__()

    config = SharedTransformerConfig(
        width=8,
        attention_heads=2,
        ffn_width=32,
        backbone_blocks=1,
        coordinate_blocks=1,
        readout_width=8,
        output_chunk_size=512,
        input_width=8,
        input_attention_heads=2,
        input_ffn_width=32,
        forcing_width=8,
        forcing_heads=2,
        forcing_ffn_width=32,
        forcing_blocks=1,
        axial_blocks=1,
    )
    active_pft_indices = setup["head"].contract.active_pft_indices
    retained_tail = SimpleNamespace(
        static_owner=SimpleNamespace(
            contract=setup["head"].contract,
            context=SimpleNamespace(
                stomate_static=SimpleNamespace(
                    parameters=SimpleNamespace(
                        coeff_maint_zero=np.zeros(
                            (max(active_pft_indices) + 1, 12),
                            dtype=np.float64,
                        )
                    )
                )
            ),
        )
    )
    head = build_shared_process_head(
        setup["head"],
        retained_tail,
        config=config,
    )
    parameters = head.initialize(setup["assembled"], seed=20260921)

    assert set(parameters) == {"tokenizer", "shared_transformer"}
    assert {"experts", "label_heads"}.isdisjoint(_mapping_keys(parameters))

    context = head.assemble_context(parameters, setup["assembled"])
    outputs = head.learned_outputs(parameters, context, setup["assembled"])
    reference_head = replace(head, shared_config=replace(config, decoder_execution="original"))
    reference_outputs = reference_head.learned_outputs(parameters, context, setup["assembled"])
    fields = head.learned_schema.fields
    assert len(fields) == len(outputs) == 161
    assert sum(int(np.prod(field.shape)) for field in fields) == 39_006
    assert set(outputs) == {field.name for field in fields}
    for field in fields:
        value = np.asarray(outputs[field.name])
        assert value.shape == field.shape
        assert np.all(np.isfinite(value)), field.name
        np.testing.assert_allclose(value, reference_outputs[field.name], rtol=2e-5, atol=2e-6,
                                   err_msg=field.name)
