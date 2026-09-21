from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_markov_contract import ConditionLeafSpec
from research.daily_coarse_graining.daily_operator.neural_architecture import (
    NeuralOutputSchema,
    OutputFieldSpec,
)
from research.daily_coarse_graining.daily_operator.shared_cimean import (
    CIMEAN_FIELD,
    apply_shared_cimean,
)


def _setup(*, full=False, active=(True, True), active_indices=(0, 13), annual=(999.0, 400.0)):
    shape = (1, 14) if full else (1, 2)
    field = OutputFieldSpec(CIMEAN_FIELD, shape, 0, "signed", pft_axis=1, full_pft_axis=full)
    schema = NeuralOutputSchema(fields=(field,), labels=("fast_owner.diffuco",), active_pft_indices=active_indices)
    leaf = ConditionLeafSpec("annual_co2_ppm", (1,), "float64", 1, 2, "annual", "test")
    contract = SimpleNamespace(
        static_conditions=SimpleNamespace(annual_condition_leaves=(leaf,)),
    )
    pft_traits = jnp.asarray([[1.0], [0.0]])
    pft = SimpleNamespace(pft_traits=pft_traits, active_pft_mask=jnp.asarray(active))
    canonical = SimpleNamespace(annual_conditions=jnp.asarray(annual), pft=pft)
    inputs = SimpleNamespace(canonical_input=canonical, pft=pft)
    return schema, contract, inputs


def test_bare_source_owner_has_annual_ca_gradient_and_no_bare_head_gradient():
    schema, contract, inputs = _setup()
    active = {CIMEAN_FIELD: jnp.ones((1, 2), dtype=bool)}

    def loss(raw, annual):
        local = SimpleNamespace(
            canonical_input=SimpleNamespace(
                annual_conditions=annual,
                pft=inputs.pft,
            ),
            pft=inputs.pft,
        )
        out = apply_shared_cimean({CIMEAN_FIELD: raw}, local, schema=schema,
                                  contract=contract, active_masks=active)
        return out[CIMEAN_FIELD].sum()

    raw = jnp.asarray([[11.0, 22.0]])
    np.testing.assert_allclose(loss(raw, jnp.asarray([999.0, 400.0])), 422.0)
    np.testing.assert_allclose(
        jax.grad(loss, argnums=0)(raw, jnp.asarray([999.0, 400.0])),
        [[0.0, 1.0]],
    )
    np.testing.assert_allclose(
        jax.grad(loss, argnums=1)(raw, jnp.asarray([999.0, 400.0])),
        [0.0, 1.0],
    )


def test_compact_pft_permutation_uses_role_trait_not_compact_slot():
    schema, contract, inputs = _setup(active_indices=(13, 0))
    inputs = SimpleNamespace(
        canonical_input=SimpleNamespace(
            annual_conditions=inputs.canonical_input.annual_conditions,
            pft=SimpleNamespace(
                pft_traits=jnp.asarray([[0.0], [1.0]]),
                active_pft_mask=jnp.asarray([True, True]),
            ),
        ),
        pft=inputs.canonical_input.pft,
    )
    out = apply_shared_cimean(
        {CIMEAN_FIELD: jnp.asarray([[11.0, 22.0]])},
        inputs,
        schema=schema,
        contract=contract,
        active_masks={CIMEAN_FIELD: jnp.ones((1, 2), dtype=bool)},
    )
    np.testing.assert_allclose(out[CIMEAN_FIELD], [[11.0, 400.0]])


def test_full_axis_mapping_and_inactive_mask_preserve_vegetation_and_zero_inactive():
    schema, contract, inputs = _setup(full=True, active=(True, False))
    values = jnp.arange(14.0).reshape(1, 14) + 10.0
    mask = jnp.zeros((1, 14), dtype=bool).at[0, 0].set(True).at[0, 13].set(False)
    out = apply_shared_cimean(
        {CIMEAN_FIELD: values}, inputs, schema=schema, contract=contract,
        active_masks={CIMEAN_FIELD: mask},
    )
    expected = np.zeros((1, 14), dtype=np.float64)
    expected[0, 0] = 400.0
    np.testing.assert_allclose(out[CIMEAN_FIELD], expected)


def test_cross_year_day_start_state_does_not_force_zero_delta_endpoint():
    schema, contract, inputs = _setup()
    inputs = SimpleNamespace(
        canonical_input=SimpleNamespace(
            annual_conditions=inputs.canonical_input.annual_conditions,
            continuous_state=jnp.asarray([317.0]),
            pft=inputs.pft,
        ),
        pft=inputs.pft,
    )
    out = apply_shared_cimean(
        {CIMEAN_FIELD: jnp.asarray([[317.0, 318.0]])}, inputs, schema=schema,
        contract=contract, active_masks={CIMEAN_FIELD: jnp.ones((1, 2), dtype=bool)},
    )
    np.testing.assert_allclose(out[CIMEAN_FIELD], [[400.0, 318.0]])

    def endpoint(day_start, annual):
        local = SimpleNamespace(
            canonical_input=SimpleNamespace(
                annual_conditions=annual,
                continuous_state=jnp.asarray([day_start]),
                pft=inputs.pft,
            ),
            pft=inputs.pft,
        )
        return apply_shared_cimean(
            {CIMEAN_FIELD: jnp.asarray([[day_start, 318.0]])},
            local,
            schema=schema,
            contract=contract,
            active_masks={CIMEAN_FIELD: jnp.ones((1, 2), dtype=bool)},
        )[CIMEAN_FIELD][0, 0]

    day_start = jnp.asarray(317.0)
    annual = jnp.asarray([999.0, 400.0])
    np.testing.assert_allclose(jax.grad(endpoint, argnums=0)(day_start, annual), 0.0)
    np.testing.assert_allclose(jax.grad(endpoint, argnums=1)(day_start, annual), [0.0, 1.0])

    def delta(day_start, annual):
        return endpoint(day_start, annual) - day_start

    np.testing.assert_allclose(jax.grad(delta, argnums=0)(day_start, annual), -1.0)
    np.testing.assert_allclose(jax.grad(delta, argnums=1)(day_start, annual), [0.0, 1.0])
