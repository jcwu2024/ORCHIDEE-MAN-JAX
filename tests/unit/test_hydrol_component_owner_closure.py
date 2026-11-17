"""Bounded actual HYDROL return contract against original Fortran values.

Golden numbers are from the verbatim hydrol_var_init+hydrol_soil extraction
in outputs/reviews/main-physics-20261117/hydrol-diffuco-state-closure/hydrol/,
case18: 1 point,14 PFT,6 tiles,7 layers,271.5K, zero melt. No model step.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

from scripts.dev.oracle_hydrol_component_owner import run_owner_case, run_owner_diagnostics


@pytest.mark.parametrize("case_id,drysoil_frac,vegstress", [
    (1, .10504178806323072, 1.1982066838626464),
    (5, .16688044329516771, 1.1677527847519424),
])
def test_custom_soil_parameters_reach_real_wrapper_diagnostics(case_id, drysoil_frac, vegstress):
    # Independent unchanged original Fortran CSV cases 1/5, hydrol_soil
    # 6395-6741 and hydrol_diag_soil 9089-9284. Fixture supplies the formal
    # initialized soil-parameter contract; there are no diagnostic overrides.
    closure = run_owner_case(case_id)
    np.testing.assert_allclose(closure.diagnostics.drysoil_frac,
        [drysoil_frac], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(closure.diagnostics.vegstress,
        [[0.]*13+[vegstress]], rtol=1e-12, atol=1e-14)


@pytest.fixture(scope="module")
def cold_closure():
    assert jax.default_backend() == "cpu"
    assert jax.config.jax_enable_x64
    return run_owner_case(18)


def test_actual_normal_water_and_flux_return_matches_original_fortran(cold_closure):
    soil = cold_closure.module.soil
    expected_mc = [
        7.5156363258476563e-2, 8.0047975308853853e-2, 9.6394097611933655e-2,
        1.3769526433973350e-1, 2.0100295116526148e-1,
        2.6502586602006473e-1, 3.0132568290073364e-1,
    ]
    expected_mcl = [
        7.3131053805620586e-2, 7.7047215503031730e-2, 9.0133710794411556e-2,
        1.2319889370999586e-1, 1.7388221359402278e-1,
        2.2513813583987161e-1, 2.5419930238928135e-1,
    ]
    np.testing.assert_allclose(np.asarray(soil.mc)[0,:,0],expected_mc,rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(np.asarray(soil.mcl)[0,:,0],expected_mcl,rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(np.asarray(soil.tmc)[0],
        [19.556616120108661]+[28.778063578294066]*5,rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(np.asarray(soil.dr_ns)[0],
        [.083383879891340884]+[.021936421705930093]*5,rtol=1e-12,atol=1e-14)


def test_fresh_ice_real_zero_snow_and_independent_topmodel_chain(cold_closure):
    # Source6884 refreshes ice from normal-final mc; source7165 restores water.
    # The original source's fresh tile1/other-tile fractions are independent
    # golden values, and differ from the actual incoming profile.
    actual = np.asarray(cold_closure.module.soil.profil_froz_hydro_ns)
    expected = np.broadcast_to(np.array([.21318273800399534]+[.53448787602149928]*5),actual.shape)
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-14)
    assert np.max(np.abs(actual-np.asarray(cold_closure.module_inputs["profil_froz"]))) > 1e-4
    assert cold_closure.snow_step is None
    np.testing.assert_array_equal(cold_closure.snow_state.snow,np.zeros(1))
    assert cold_closure.diagnostics.fwet_new is None
    # This complete real chain is separate from the wrapper's TOPMODEL=False
    # route. Source7252-7280 makes a frozen surface return zero wetland.
    diagnostics = run_owner_diagnostics(cold_closure,topmodel_new=True)
    np.testing.assert_array_equal(diagnostics.fwet_new,np.zeros(1))
