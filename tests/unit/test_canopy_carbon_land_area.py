from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator.successor_training import extract_biomass_productivity_trajectories


def test_canopy_storage_has_no_second_cover_factor_and_selects_carbon():
    shapes = {
        **{key: (1, 2) for key in ('npp_daily', 'resp_maint', 'resp_growth', 'resp_hetero')},
        'biomass': (1, 2, 12, 1), 'litter_above': (1, 2, 2, 2),
        'litter_below': (1, 2, 2, 3, 2), 'carbon_32l': (1, 3, 2, 3),
        'DOC': (1, 2, 3, 2, 2, 2), 'interception_storage': (1, 2, 2)}
    leaves, cursor = [], 0
    for field, shape in shapes.items():
        size = int(np.prod(shape))
        leaves.append(SimpleNamespace(component='slowproc_stomate_previous_step_state', path=(field,),
            start=cursor, stop=cursor+size, shape=shape))
        cursor += size
    contract = SimpleNamespace(state_leaves=leaves, active_pft_indices=(0, 13), continuous_state_width=cursor)
    state = jnp.zeros((1, cursor), dtype=jnp.float64).at[0, -4:].set(jnp.array([10., 1000., 30., 3000.]))

    def ecosystem(fraction, active):
        result = extract_biomass_productivity_trajectories(state, jnp.ones_like(state, dtype=bool),
            daily_fast_gpp=jnp.zeros((1, 1, 2)), daily_fast_gpp_defined=jnp.ones((1, 1, 2), dtype=bool),
            pft_fraction=fraction[None, :], active_pft_mask=active[None, :], contract=contract)
        return result.stocks['ecosystem_carbon'].value.sum()

    active = jnp.array([True, True])
    for fraction in (jnp.array([.25, .75]), jnp.array([.8, .2])):
        np.testing.assert_allclose(ecosystem(fraction, active), 40.)
        np.testing.assert_allclose(jax.grad(ecosystem)(fraction, active), 0.)
    np.testing.assert_allclose(ecosystem(jnp.array([.25, .75]), jnp.array([False, True])), 30.)
