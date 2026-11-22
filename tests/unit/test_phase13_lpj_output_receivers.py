"""Source-ordered LPJ output receipts on bounded cover and harvest states."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jax_orchidee.stomate import integration
from jax_orchidee.stomate.carbon_kernels import ICARBON, ILEAF, IROOT, NPARTS


def _boundary(value, case):
    nvm, ndeep = 14, 2
    crop = case == "harvest"
    pft = 11 if crop else 13
    shape = (1, nvm)
    biomass = jnp.zeros((*shape, NPARTS, 1))
    area_before = jnp.zeros(shape).at[0, pft].set(0.75)
    biomass_before = biomass.at[0, pft, ILEAF, ICARBON].set(value)
    turnover_before = biomass.at[0, pft, ILEAF, ICARBON].set(1.0)
    flux_before = jnp.zeros(shape).at[0, pft].set(2.0)
    npp = SimpleNamespace(
        npp=flux_before, resp_maint=2.0 * flux_before, resp_growth=3.0 * flux_before, bm_alloc=biomass,
    )
    daily = integration.ExplicitDailyCarbonResult(
        boundary=SimpleNamespace(resp_maint_part=jnp.zeros((*shape, NPARTS))),
        npp_update=npp, age_sla=None, requires_trace=(),
    )
    cover = None
    harvest = None
    establishment = None
    crown = None
    if case in ("cover", "harvest"):
        # Supplied covered state follows LPJ's 0.75/0.5 biomass redistribution.
        cover_turnover = biomass.at[0, pft, ILEAF, ICARBON].set(4.0)
        cover_turnover = cover_turnover.at[0, pft, IROOT, ICARBON].set(2.0)
        cover = SimpleNamespace(
            biomass=1.5 * biomass_before, turnover_daily=cover_turnover,
            veget_max=area_before.at[0, pft].set(0.5), gpp_daily=5.0 * flux_before,
            resp_maint=6.0 * flux_before, resp_growth=7.0 * flux_before,
        )
        if crop:
            # Native harvest of the supplied crop returns 2.2 leaf and 1.1 root.
            harvested_turnover = biomass.at[0, pft, ILEAF, ICARBON].set(2.2)
            harvested_turnover = harvested_turnover.at[0, pft, IROOT, ICARBON].set(1.1)
            harvest = SimpleNamespace(turnover_daily=harvested_turnover)
    if case == "establishment":
        establishment = SimpleNamespace(biomass=2.0 * biomass_before)
        crown = SimpleNamespace(veget_max=area_before.at[0, pft].set(0.6))
    post = integration.ExplicitDailyCarbonKillGapTurnoverResult(
        **dict.fromkeys(integration.ExplicitDailyCarbonKillGapTurnoverResult._fields) | {
            "daily_carbon": daily, "turnover": SimpleNamespace(biomass=biomass_before, turnover=turnover_before),
            "establishment_biomass": establishment, "crown_after_establish": crown,
            "cover": cover, "harvest_agri": harvest, "bm_to_litter": biomass, "requires_trace": (),
        },
    )
    leak = integration.ExplicitOkLeakFromPostNppResult(
        prep=None, requires_trace=(),
        ok_leak=SimpleNamespace(
            littercalc=SimpleNamespace(litter_above=jnp.zeros((1, 2, nvm, 1))),
            soilcarbon=SimpleNamespace(
                litter_below=jnp.zeros((1, 2, nvm, ndeep, 1)),
                carbon_32l=jnp.zeros((1, 3, nvm, ndeep)),
                doc=jnp.zeros((1, nvm, ndeep, 2, 7, 1)),
            ),
        ),
    )
    result = integration.stomate_lpj_outputs_from_post_npp_ok_leak_explicit(
        post_npp=post, ok_leak=leak, veget_max=area_before, gpp_daily=flux_before,
        z_soil=jnp.array([0.5, 1.5]), zf_soil=jnp.array([0.0, 1.0, 2.0]),
        carb_mass_total_old=jnp.zeros(1),
    )
    diag = result.output_diagnostics
    history = result.modelout_fields
    return (
        diag.tot_live_biomass[0, pft, ICARBON], diag.tot_turnover[0, pft, ICARBON], diag.f_veg_litter[0],
        history["LEAF_M"][0, pft], history["GPP"][0, pft], history["MAINT_RESP"][0, pft],
        history["GROWTH_RESP"][0, pft],
    )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("case", ["turn", "establishment", "cover", "harvest"])
def test_lpj_output_receives_final_cover_and_harvest(compiled, case):
    def run(value):
        return _boundary(value, case)

    with jax.enable_x64():
        observed = (jax.jit(run) if compiled else run)(8.0)
    # Original StomateLpj:1616-1635, 2203, 2212, 2284-2286 and 2366.
    if case == "turn":
        expected = (8.0, 1.0, 0.75 / (1000 * 86400), 8.0, 2.0, 4.0, 6.0)
    elif case == "establishment":
        expected = (16.0, 1.0, 0.6 / (1000 * 86400), 16.0, 2.0, 4.0, 6.0)
    else:
        turnover = 3.3 if case == "harvest" else 6.0
        expected = (12.0, turnover, turnover * 0.5 / (1000 * 86400), 12.0, 10.0, 12.0, 14.0)
    np.testing.assert_allclose(np.asarray(observed), expected, rtol=1e-12, atol=1e-14)
