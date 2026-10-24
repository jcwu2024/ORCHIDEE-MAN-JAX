"""Independent freezing-capacity regression at the real THERMOSOIL wrapper.

Fortran provenance: thermosoil.f90::thermosoil_getdiff2566-2863, especially
2742-2743 (deep latent cutoff) and2760-2762 (natural capacity); source identities
and saved independent oracle values live in fixtures/thermosoil_nslm_fortran.json.
No test imports ignored outputs or evaluates its expected capacity from JAX.

The original oracle point2 has crop slot14. Its 13 natural slots are identical;
the natural14 case role-maps those values to the same natural branch, without
claiming equality to the original crop14. Different wrapper geometry and mcl
are immaterial only for these selected fields: constant mc stays.3 after
humlev, getdiff does not read mcl, and toporganic is disabled. The recurrence
c=T,d=0 and zero snow fractions retain the declared comparison temperature.
This is a component regression, not a paper trajectory or energy-clock test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.thermosoil import thermosoil_explicit_step  # noqa: E402

REFERENCE = json.loads(
    (Path(__file__).parent / 'fixtures/thermosoil_nslm_fortran.json').read_text(encoding='utf-8')
)
ATOL = REFERENCE['tolerance']['atol']
RTOL = REFERENCE['tolerance']['rtol']


def _wrapper_inputs(*, temperature=273.15, nslm=2, long_wetness=.6, crop14=False):
    """Four thermal layers; original material state, source-independent profile."""
    ptn = np.full((1, 4, 14), temperature)
    veget_max = np.zeros((1, 14))
    veget_max[0, [0, 13]] = [.2, .8]
    crop = np.zeros(14, dtype=bool)
    crop[13] = crop14
    tmc_pft = np.full((1, nslm, 14), 100.)
    tmc_pft[:, :, 13] = 150.
    return dict(
        ptn=ptn, cgrnd=np.full((1, 3, 14), temperature), dgrnd=np.zeros((1, 3, 14)),
        cgrnd_snow=np.full((1, 3), temperature), dgrnd_snow=np.zeros((1, 3)),
        temp_sol_new=np.array([temperature]), temp_sol_new_pft=np.full((1, 14), temperature),
        snowrho=np.full((1, 3), 200.), snowtemp=np.full((1, 3), 268.),
        snowdz=np.array([[.01, .02, .04]]), shumdiag_perma=np.full((1, nslm), long_wetness),
        mc_layh=np.full((1, nslm), .3), mcl_layh=np.full((1, nslm), .3),
        tmc_layh=np.full((1, nslm), 100.), mc_layh_pft=np.full((1, nslm, 14), .3),
        mcl_layh_pft=np.full((1, nslm, 14), .3), tmc_layh_pft=tmc_pft,
        njsc=np.array([4]), veget_max=veget_max, pb=np.array([900.]),
        dlt=np.array([.1, .2, .5, 1.]), dz1=np.array([10., 3., 1.]),
        zlt=np.array([.1, .3, .8, 1.8]), znt=np.array([.05, .2, .55, 1.3]),
        dz5=np.array([.2, .3, .4]), dt_sechiba=1800., lambda_thermal=.5,
        frac_snow_veg=np.array([0.]), frac_snow_nobio=np.array([[0.]]),
        totfrac_nobio=np.array([0.]), temp_sol_beg=np.array([temperature]),
        soilcap_initial=np.array([1000.]), pcapa_en_previous=np.full_like(ptn, 1.e6),
        refsoc=np.array([[65000., 32500., 0., 0.]]), ok_laidev=crop,
        veget_mask_2d=np.ones((1, 14), dtype=bool),
        shum_ngrnd_permalong_previous=np.full_like(ptn, long_wetness),
    )


def _capacity_fields(result, slots):
    return np.stack([
        np.asarray(result.getdiff.pcapa)[0][:, slots],
        np.asarray(result.getdiff.pcappa_supp)[0][:, slots],
    ])


@pytest.mark.parametrize('crop14,slots', [(True, list(range(13))), (False, [13])],
                         ids=['original-natural1-13', 'active-natural14-role-map'])
def test_thermosoil_nslm_wrapper_matches_independent_fortran(crop14, slots):
    result = thermosoil_explicit_step(**_wrapper_inputs(crop14=crop14))
    expected = np.broadcast_to(
        np.array([REFERENCE['expected']['pcapa'], REFERENCE['expected']['pcappa_supp']])[:, :, None],
        (2, 4, len(slots)),
    )
    actual = _capacity_fields(result, slots)
    # Assert unchanged shallow values before the boundary regression assertion.
    np.testing.assert_allclose(actual[:, :2], expected[:, :2], atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)
    np.testing.assert_array_equal(actual[1, 2:], 0.)


@pytest.mark.parametrize('temperature,long_wetness,transition', [
    (272.15, .6, True), (274.15, .6, True),
    (270., .6, False), (276., .6, False), (273.15, 0., True),
], ids=['lower-inclusive', 'upper-inclusive', 'cold-outside', 'warm-outside', 'zero-long-wetness'])
def test_thermosoil_nslm_wrapper_transition_boundaries(temperature, long_wetness, transition):
    result = thermosoil_explicit_step(**_wrapper_inputs(
        temperature=temperature, long_wetness=long_wetness))
    np.testing.assert_allclose(np.asarray(result.profile.ptn), temperature, atol=ATOL, rtol=RTOL)
    supp = np.asarray(result.getdiff.pcappa_supp)[0, :, 13]
    expected_shallow = REFERENCE['expected']['pcappa_supp'][0] if transition and long_wetness else 0.
    np.testing.assert_allclose(supp[:2], expected_shallow, atol=ATOL, rtol=RTOL)
    np.testing.assert_array_equal(supp[2:], 0.)


def test_thermosoil_nslm_wrapper_equal_hydrological_thermal_grid():
    result = thermosoil_explicit_step(**_wrapper_inputs(nslm=4))
    # With no thermal layers below nslm, all four remain within the source
    # latent interval; value comes from saved shallow Fortran output.
    np.testing.assert_allclose(np.asarray(result.getdiff.pcappa_supp)[0, :, 13],
                               REFERENCE['expected']['pcappa_supp'][0], atol=ATOL, rtol=RTOL)


def test_thermosoil_nslm_wrapper_jit_uses_original_traced_moisture_shape():
    inputs = _wrapper_inputs()

    @jax.jit
    def capacity(shumdiag_perma):
        result = thermosoil_explicit_step(**{**inputs, 'shumdiag_perma': shumdiag_perma})
        return result.getdiff.pcapa[0, :, 13], result.getdiff.pcappa_supp[0, :, 13]

    pcapa, supp = capacity(jnp.asarray(inputs['shumdiag_perma']))
    np.testing.assert_allclose(np.asarray(pcapa), REFERENCE['expected']['pcapa'], atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(np.asarray(supp), REFERENCE['expected']['pcappa_supp'], atol=ATOL, rtol=RTOL)
