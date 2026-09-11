from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator.fresh_start_training import extract_one_day_trajectories
from tests.unit.test_daily_operator_successor_process_head import _markov_contract


def test_one_day_physical_extraction_preserves_pft_area_and_gradient():
    contract = _markov_contract()
    state = SimpleNamespace(other=jnp.zeros(contract.continuous_state_width))
    defined = SimpleNamespace(other=jnp.ones(contract.continuous_state_width, dtype=bool))
    inputs = SimpleNamespace(pft=SimpleNamespace(pft_fraction=jnp.array([.2, .8]),
                                                active_pft_mask=jnp.array([True, True])))

    def extract(gpp):
        return extract_one_day_trajectories(state, defined, gpp, jnp.ones_like(gpp, dtype=bool), inputs, contract)

    gpp = jnp.array([[2., 10.]])
    result = jax.jit(extract)(gpp)
    np.testing.assert_allclose(result.fluxes['gpp'].value, [[8.4]], rtol=1e-6)
    assert result.fluxes['gpp'].defined.shape == (1, 1)
    assert result.stocks['aboveground_biomass_carbon'].value.shape == (1, 1)
    gradient = jax.grad(lambda v: extract(v).fluxes['gpp'].value.sum())(gpp)
    np.testing.assert_allclose(gradient, [[.2, .8]], rtol=1e-6)
    inputs.pft.active_pft_mask = jnp.array([False, True])
    masked = extract(gpp.at[0, 0].set(jnp.nan))
    np.testing.assert_allclose(masked.fluxes['gpp'].value, [[8.]], rtol=1e-6)
