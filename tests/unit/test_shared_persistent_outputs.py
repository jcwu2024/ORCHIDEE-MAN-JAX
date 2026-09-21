from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.neural_architecture import (
    NeuralOutputSchema,
    OutputFieldSpec,
)
from research.daily_coarse_graining.daily_operator.shared_persistent_outputs import (
    PERSISTENT_OUTPUT_OWNERS,
    build_shared_persistent_coordinates,
)
from research.daily_coarse_graining.daily_operator.shared_process_head import transform_shared_output
from research.daily_coarse_graining.daily_operator.successor_fast_owner_decoder import ABSOLUTE_TARGET


def fixture():
    fields, leaves = [], []
    for i, (name, owner) in enumerate(PERSISTENT_OUTPUT_OWNERS.items()):
        fields.append(OutputFieldSpec(name, (1, 2, 2), 0,
                                      "unit_interval" if "lignin" in name or "litterpart" in name
                                      else "nonnegative", pft_axis=1,
                                      value_representation=ABSOLUTE_TARGET))
        leaves.append(SimpleNamespace(key=owner, shape=(1, 2, 2), start=4*i, stop=4*i+4,
                                      selected_pft_indices=(0, 13)))
    schema = NeuralOutputSchema(tuple(fields), ("ok_leak_aux",), (0, 13))
    contract = SimpleNamespace(state_leaves=leaves, active_pft_indices=(0, 13))
    statistics = SimpleNamespace(
        count=np.tile([2, 2, 6, 6], 8),
        mean=np.tile([0., 0., .04, 0.], 8),
        variance=np.tile([0., 0., .0004, 0.], 8),
    )
    return schema, contract, statistics


def test_global_pft_pooling_includes_between_means_and_avoids_unit_fallback():
    schema, contract, statistics = fixture()
    coordinates = build_shared_persistent_coordinates(schema, contract, statistics)
    assert set(coordinates.fields) == set(PERSISTENT_OUTPUT_OWNERS)
    for name, field in coordinates.fields.items():
        np.testing.assert_allclose(coordinates.normalization.mean[name], [[[.03, 0.], [.03, 0.]]])
        # Within variance .0003 + between variance .0003.
        np.testing.assert_allclose(coordinates.normalization.scale[name],
                                   [[[np.sqrt(.0006), 1e-12], [np.sqrt(.0006), 1e-12]]])
        assert field.day_start_owner == PERSISTENT_OUTPUT_OWNERS[name]
    assert all(f.value_representation == ABSOLUTE_TARGET for f in schema.fields)
    # Reordering complete input/metadata/statistics PFT axes commutes with pooling.
    schema = replace(schema, active_pft_indices=(13, 0))
    contract.active_pft_indices = (13, 0)
    for leaf in contract.state_leaves:
        leaf.selected_pft_indices = (13, 0)
    for key in ("count", "mean", "variance"):
        setattr(statistics, key, getattr(statistics, key).reshape(8, 2, 2)[:, ::-1].reshape(-1))
    permuted = build_shared_persistent_coordinates(schema, contract, statistics)
    for name in coordinates.fields:
        np.testing.assert_array_equal(permuted.normalization.scale[name], coordinates.normalization.scale[name])


@pytest.mark.parametrize("name", list(PERSISTENT_OUTPUT_OWNERS))
def test_residual_bounds_state_raw_gradients_and_masks(name):
    schema, contract, statistics = fixture()
    coordinates = build_shared_persistent_coordinates(schema, contract, statistics)
    field = coordinates.fields[name]
    active = jnp.array([[[True, True], [True, False]]])
    def fn(raw, state):
        inputs = SimpleNamespace(canonical_input=SimpleNamespace(continuous_state=state))
        return transform_shared_output(raw, active, field=field,
                                       normalization=coordinates.normalization, inputs=inputs)
    raw, state = jnp.zeros(field.shape), jnp.full(32, .3)
    value = jax.jit(fn)(raw, state)
    np.testing.assert_allclose(value, [[[.33, .3], [.33, 0.]]], atol=1e-12)
    grad_raw, grad_state = jax.grad(lambda r, s: fn(r, s).sum(), argnums=(0, 1))(raw, state)
    assert np.all(np.asarray(grad_raw)[np.asarray(active)] > 0.)
    assert grad_raw[0, 1, 1] == 0.
    expected = np.zeros(32)
    expected[field.day_start_state_start:field.day_start_state_stop] = [1., 1., 1., 0.]
    np.testing.assert_allclose(grad_state, expected, atol=1e-5)
    zero_value = fn(raw, jnp.zeros(32))
    assert np.all(np.isfinite(zero_value)) and np.all(zero_value >= 0)
    for amplitude in (-1e8, 1e8):
        extreme = fn(jnp.full(field.shape, amplitude), state)
        assert np.all(np.isfinite(extreme)) and np.all(extreme >= 0)
        if field.constraint == "unit_interval":
            assert np.all(extreme <= 1.)


def test_rejects_wrong_owner_layout_or_unobserved_statistics():
    schema, contract, statistics = fixture()
    contract.state_leaves[0].selected_pft_indices = (13, 0)
    with pytest.raises(ValueError, match="selection differs"):
        build_shared_persistent_coordinates(schema, contract, statistics)
    contract.state_leaves[0].selected_pft_indices = (0, 13)
    statistics.count[0] = 0
    with pytest.raises(ValueError, match="invalid TRAIN"):
        build_shared_persistent_coordinates(schema, contract, statistics)
