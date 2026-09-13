from types import SimpleNamespace

import numpy as np

from research.daily_coarse_graining.daily_operator.forward_attribution import reduce_pool, split_pool_changes


def test_carbon_element_and_area_weight_are_applied_once():
    leaf = SimpleNamespace(axis_names=('npts', 'ncarb', 'nvm', 'ndeep', 'nelements'))
    a = np.zeros((2, 1, 3, 2, 4, 2))
    a[..., 0] = 2.
    a[..., 1] = 1000.  # Nitrogen cannot enter carbon accounting.
    fraction = np.array([[.25, .75], [.4, .6]])
    result = reduce_pool(a, fraction, np.ones_like(fraction, dtype=bool), leaf, 'litter_above')
    np.testing.assert_allclose(result, [24., 24.])


def test_deep_carbon_is_a_layer_stock_not_a_concentration():
    leaf = SimpleNamespace(axis_names=('npts', 'ncarb', 'nvm', 'ndeep'))
    a = np.ones((1, 1, 3, 2, 4))
    a[:, :, :, 1, :] = 3.
    result = reduce_pool(a, [[.25, .75]], [[True, True]], leaf, 'carbon_32l')
    np.testing.assert_allclose(result, [30.])
    a[:, :, :, 1, :] = np.nan
    result = reduce_pool(a, [[1., 0.]], [[True, False]], leaf, 'carbon_32l')
    np.testing.assert_allclose(result, [12.])
    assert np.isnan(reduce_pool(a, [[.5, .5]], [[True, True]], leaf, 'carbon_32l')).all()


def test_large_drain_can_close_inventory_identity_without_being_scientifically_correct():
    def wrap(v):
        return {'soil': np.array([v])}
    result = split_pool_changes(wrap(1000), wrap(900), wrap(901), wrap(1), wrap(101))['soil']
    np.testing.assert_array_equal(result['endpoint_identity_residual'], [0.])
    np.testing.assert_array_equal(result['fast_change'], [-100.])
    np.testing.assert_array_equal(result['tail_change'], [1.])
