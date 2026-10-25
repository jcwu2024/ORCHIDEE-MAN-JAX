"""Independent bounded regressions for S22-TILE6/FROZEN-DUMMY/TOPMODEL.

References are literal hydrol.f90 arithmetic, not calls to the repaired helpers.
Synthetic coefficients isolate each stage; these tests do not establish a model
day, table-initialization, profile-refresh, or paper TOPMODEL acceptance.
"""
from pathlib import Path
from types import SimpleNamespace
from functools import lru_cache
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jax_orchidee.sechiba import hydrol as h

DZ = np.array([0., 10., 20., 40.])


def _mass_matrix(dz):
    # hydrol_soil_setup 8507-8568 with a=d=0.
    n = len(dz)
    matrix = np.zeros((n, n))
    matrix[0, :2] = [3*dz[1]/8, dz[1]/8]
    for j in range(1, n-1):
        matrix[j, j-1:j+2] = [dz[j]/8, 3*(dz[j]+dz[j+1])/8, dz[j+1]/8]
    matrix[-1, -2:] = [dz[-1]/8, 3*dz[-1]/8]
    return matrix


def _column_water(x, dz=DZ):
    # hydrol_soil 7096-7108, nodal integration, mm.
    total = dz[1]*(3*x[0]+x[1])/8 + dz[-1]*(3*x[-1]+x[-2])/8
    for j in range(1, len(x)-1):
        total += dz[j]*(3*x[j]+x[j-1])/8 + dz[j+1]*(3*x[j]+x[j+1])/8
    return total


def _source_under(x, tile, mask, peat=True):
    # hydrol_soil_smooth_under_mcr 7665-7769: masks are stage-specific.
    x = np.array(x, copy=True)
    full = .15 if peat and tile in (4, 5, 6) else .1
    ends = .15 if peat and tile in (4, 5) else .1
    for j in range(2):
        excess = max(full-x[j], 0.)
        x[j] += excess
        x[j+1] -= excess*(DZ[j]+DZ[j+1])/(DZ[j+1]+DZ[j+2])
    excess = max(full-x[2], 0.)
    x[2] += excess
    x[3] -= excess*(DZ[2]+DZ[3])/DZ[3]
    excess = max(ends-x[3], 0.)
    x[3] += excess
    x[2] -= excess*DZ[3]/(DZ[2]+DZ[3])
    for j in (2, 1):
        excess = max(full-x[j], 0.)
        x[j] += excess
        x[j-1] -= excess*(DZ[j]+DZ[j+1])/(DZ[j-1]+DZ[j])
    top = mask*max(ends-x[0], 0.)
    x[0] += top
    x -= top*DZ[1]/(2*.07*1000)
    return mask*x, top


def _constant_peat_tables():
    # Small explicit table input: zero transport slopes/diffusion, k floor .01.
    zero = jnp.zeros((2, 4), dtype=jnp.float64)
    return SimpleNamespace(a_lin=zero, b_lin=zero, d_lin=zero,
        k_lin=jnp.full((3, 4), .01), mcr=.15, mcs=.9, ks=.01,
        kfact=jnp.ones(4), imin=1, imax=3)


@lru_cache(maxsize=4)
def _all_tiles(mc, rain=0., peat=True):
    # Actual all_tiles -> tile -> infiltration -> solve, without monkeypatches.
    values = np.full((2, 4, 6), mc)
    mask = np.ones((2, 6)); mask[1, 5] = 0.
    area = np.array([[.1, .1, .1, .2, .2, .3], [.1, .1, .1, .35, .35, 0.]])
    zero = jnp.zeros((2, 4, 2), dtype=jnp.float64)
    mineral = SimpleNamespace(a_lin=zero, b_lin=zero, d_lin=zero,
        k_lin=jnp.full((3, 4, 2), .01), imin=1, imax=3)
    return h.hydrol_soil_all_tiles_explicit_step(
        water2infilt=np.zeros((2, 6)), ae_ns=np.zeros((2, 6)), subsinksoil=np.zeros(2),
        precisol_ns=np.full((2, 6), rain), reinfiltration_soil=np.zeros((2, 6)),
        is_crop_soil=False, mc=values, mcl=values, profil_froz=np.zeros_like(values),
        mcr=[.1, .1], mcs=[.6, .6], dz_mm=DZ, dt_days=1/48,
        free_drain_coef=np.zeros((2, 6)), rootsink=np.zeros_like(values),
        resolv=mask > 0, mask_soiltile=mask, kfact_root=np.ones_like(values),
        ks=.01, kfact=np.ones(4), soiltile=area, tmc=np.zeros((2, 6)),
        mineral_tables=mineral, peat_tables=_constant_peat_tables() if peat else None,
        reinf_slope=[0., 0.], peat_hydro=peat, zz_mm=np.cumsum(DZ), zmaxh_m=.07)


@pytest.mark.parametrize("tile", [1, 4, 5, 6])
def test_tile_stage_top_and_deeper_infiltration(tile):
    result = _all_tiles(.5, rain=3.)
    actual = result.tile_results[tile-1]
    # Source 7491-7498: first fill peat only 4/5, unlike deeper 7514-7540.
    sat = .9 if tile in (4, 5) else .6
    first = min((sat-.5)*DZ[1]/2, 3.)
    np.testing.assert_allclose(actual.infilt.wat_inf_first, [first, first], atol=2e-13)
    # Deeper peat6 still uses peat saturation/conductivity; construct a middle
    # node already above mineral saturation so it can accept water only as peat.
    infilt = h.hydrol_soil_infilt_explicit(mc=[[.6, .7, .5, .5]], flux_infilt=[1.],
        k=[[.01]*4], dz_mm=DZ, dt_days=1/48, mcs=[.6], ks=.01,
        kfact=np.ones(4), kfact_root=np.ones((1, 4)), soil_tile_index=tile,
        peat_hydro=True, mcs_peat=.9, ks_peat=.01, kfact_peat=np.ones(4))
    if tile == 1:
        np.testing.assert_allclose(infilt.mc[0, 1], .7, atol=2e-13)
    else:
        # Avoid a zero remaining flux at top for 4/5; this condition primarily
        # checks tile6, whose source mineral top is already saturated.
        assert float(infilt.mc[0, 1]) >= .7
        if tile == 6:
            assert float(infilt.mc[0, 1]) > .7
        assert actual.coef_before_infilt is not None


@pytest.mark.parametrize("tile", [1, 4, 5, 6])
def test_tile_stage_oversaturation_and_area(tile):
    result = _all_tiles(.7)
    threshold = .9 if tile in (4, 5) else .6
    expected = np.full(4, min(.7, threshold))
    np.testing.assert_allclose(result.mc[0, :, tile-1], expected, atol=2e-13)
    removed = _column_water(np.full(4, max(.7-threshold, 0.)))
    routed = result.tile_results[tile-1].solve.over_mcs_routing
    np.testing.assert_allclose(routed.ru_corr_ns+routed.dr_corr_ns,
                               [removed, removed], atol=2e-13)
    # An absent tile reaches the actual smoothing/masking boundary and is zero.
    np.testing.assert_array_equal(result.mc[1, :, 5], np.zeros(4))
    area = np.array([.1, .1, .1, .2, .2, .3])
    expected_tmc = np.array([_column_water(np.full(4, .7 if i in (3, 4) else .6)) for i in range(6)])
    np.testing.assert_allclose(np.dot(np.asarray(result.tmc)[0], area), np.dot(expected_tmc, area), atol=2e-12)


@pytest.mark.parametrize("tile", [1, 4, 5, 6])
def test_tile_stage_residual_top_bottom_masks(tile):
    result = _all_tiles(.12)
    expected, top = _source_under(np.full(4, .12), tile, 1.)
    np.testing.assert_allclose(result.mc[0, :, tile-1], expected, atol=2e-13)
    np.testing.assert_allclose(result.tile_results[tile-1].solve.under_mcr.excess_top[0], top, atol=2e-13)
    np.testing.assert_array_equal(result.mc[1, :, 5], np.zeros(4))


def test_paper_mineral_flag_control():
    # Accepted paper producer keeps HYDROL-local peat False; no peat admission.
    result = _all_tiles(.7, peat=False)
    np.testing.assert_allclose(result.mc[0], .6, atol=2e-13)


@pytest.mark.parametrize("freeze,fraction,resolv,flux", [
    (True, .5, True, .01), (False, .5, True, .01),
    (True, 0., True, .01), (False, 0., True, .01),
    (True, .5, False, 0.), (True, .5, True, 3.),
])
def test_frozen_dummy_literal_order_switch_retry_and_restore(freeze, fraction, resolv, flux):
    # hydrol_soil 7044-7058 then 7080-7092; frozen=False source first sets mc=mcl.
    matrix = _mass_matrix(DZ)
    lower = np.r_[0., matrix.diagonal(-1)][None, :]
    upper = np.r_[matrix.diagonal(1), 0.][None, :]
    diagonal = matrix.diagonal()[None, :]
    setup = h.HydrolSoilSetupResult(e=lower, f=diagonal, g1=upper, ep=lower, fp=diagonal, gp=upper)
    old = np.full((1, 4), .4)
    before = old.copy()
    tile = SimpleNamespace(solve=SimpleNamespace(mc_final=old), setup=setup, k_bottom=np.array([0.]))
    result = h.hydrol_alt_residual_all_tiles_from_results((tile,),
        profil_froz=np.full((1, 4, 1), fraction), mcr=[.1], dz_mm=DZ,
        dt_days=1/48, free_drain_coef=[[0.]], resolv=[[resolv]], mask_soiltile=[[float(resolv)]],
        evapot=[flux], evapot_penm=[flux], frac_bare_ns=[[1.]], ok_freeze_cwrr=freeze)
    liquid = np.minimum(old[0], .1+(1-fraction)*(old[0]-.1))
    rhs = matrix @ liquid; rhs[0] -= flux
    first = np.linalg.solve(matrix, rhs) if resolv else liquid.copy()
    retry = first[0] < .1 and flux > 1e-8
    if retry:
        boundary = matrix.copy(); boundary[0] = 0.; boundary[0, 0] = 1.
        rhs[0] = .1
        final_liquid = np.linalg.solve(boundary, rhs)
    else:
        final_liquid = first
    total = np.maximum(final_liquid, final_liquid+fraction*(old[0]-.1)) if freeze else final_liquid.copy()
    total = np.maximum(final_liquid, final_liquid+fraction*(total-.1))
    np.testing.assert_allclose(result.mc_after[0, :, 0], total, atol=2e-13)
    np.testing.assert_allclose(result.tmc[0, 0], _column_water(total), atol=2e-12)
    assert bool(result.resolv_alt[0, 0]) == retry
    np.testing.assert_array_equal(tile.solve.mc_final, before)


def _topmodel(temp, undef=False, numlayers=11):
    return h.hydrol_peat_water_table_diagnostics(mc=jnp.full((1, 11, 1), .3),
        mcl=jnp.full((1, 11, 1), .3), soiltile=jnp.ones((1, 1)),
        dz_mm=jnp.array([0.]+[10.]*10), dh_mm=jnp.full(11, 10.), wt_ab=jnp.zeros(1),
        topmodel_new=True, numlayers=numlayers, temp_hydro=temp, mcs=jnp.array([.6]),
        param_vp=jnp.array([1e20 if undef else 1.]), param_kp=jnp.ones(1),
        param_qp=jnp.zeros(1), param_fmax=jnp.ones(1))


@pytest.mark.parametrize("first_cold", [0, 3, 6, 7, None])
@pytest.mark.parametrize("compiled", [False, True])
def test_topmodel_contiguous_thaw_prefix(first_cold, compiled):
    # hydrol_soil 7245-7267: stop at first frozen layer; 7308-7342 logistic.
    temp = np.full((1, 11), 280.)
    if first_cold is not None:
        temp[0, first_cold] = 270.
    prefix = 11 if first_cold is None else first_cold
    expected_mean = -.005*prefix if prefix > 6 else 1e20
    expected_fwet = 1/(1+np.exp(-expected_mean)) if prefix > 6 else 0.
    result = jax.jit(_topmodel)(jnp.asarray(temp)) if compiled else _topmodel(jnp.asarray(temp))
    np.testing.assert_allclose(result.meanwt, [expected_mean], atol=2e-13)
    np.testing.assert_allclose(result.fwet_new, [expected_fwet], atol=2e-13)


def test_topmodel_undefined_parameter_control():
    result = _topmodel(jnp.full((1, 11), 280.), undef=True)
    np.testing.assert_array_equal(result.fwet_new, [0.])


@pytest.mark.parametrize("limit,first_cold", [(6, None), (7, None), (2, 3), (15, None)])
def test_topmodel_declared_layer_limit(limit, first_cold):
    # The source scan ends at numlayers: warm nodes below that declared limit
    # cannot extend the prefix. The supported array bound also caps it at nslm.
    temp = np.full((1, 11), 280.)
    if first_cold is not None:
        temp[0, first_cold] = 270.
    prefix = min(limit, 11, 11 if first_cold is None else first_cold)
    mean = -.005*prefix if prefix > 6 else 1e20
    fwet = 1/(1+np.exp(-mean)) if prefix > 6 else 0.
    result = _topmodel(jnp.asarray(temp), numlayers=limit)
    np.testing.assert_allclose(result.meanwt, [mean], atol=2e-13)
    np.testing.assert_allclose(result.fwet_new, [fwet], atol=2e-13)
