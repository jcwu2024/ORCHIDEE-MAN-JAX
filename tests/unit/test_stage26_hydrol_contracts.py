"""Stage26 B: bounded declared contracts, never a model/Fortran run.

Source: hydrol.f90 allowed_err121, tmc_update3512-3584 and
soil_infilt7490-7588. Check-mode assertions do not imply a production caller.
"""
from __future__ import annotations

from pathlib import Path
import sys

import jax
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from jax_orchidee.sechiba import hydrol as h


def _column(mc,dz):
    """Independent literal source3515-3522 order, one column."""
    value = dz[1]*(3*mc[0]+mc[1])/8
    for layer in range(1,len(mc)-1):
        value += dz[layer]*(3*mc[layer]+mc[layer-1])/8 + dz[layer+1]*(3*mc[layer]+mc[layer+1])/8
    value += dz[-1]*(3*mc[-1]+mc[-2])/8
    return value


def _migration_inputs(*,two_tiles=False,old_storage_offset=0.):
    dz = np.array([0.,2.,4.])
    mc = np.array([[[.2,.4],[.3,.5],[.4,.6]]]) if two_tiles else np.full((1,3,1),.25)
    water = np.array([[1.,2.]]) if two_tiles else np.zeros((1,1))
    ntiles = mc.shape[2]
    tmc = np.array([[_column(mc[0,:,tile],dz)+water[0,tile] for tile in range(ntiles)]])
    # Threshold-only cases declare a diagnostic previous-storage residual;
    # this is not alleged to be a generated model trajectory or new physics.
    tmc += old_storage_offset
    return dict(veget_max=np.array([[.5,.5]]),soiltile=np.array([[1.,0.]]) if two_tiles else np.ones((1,1)),
        qsintveg=np.zeros((1,2)),mc=mc,water2infilt=water,tmc=tmc,
        resdist=np.array([[.6,.4]]) if two_tiles else np.ones((1,1)),vegtot_old=np.ones(1),vegtot=np.ones(1),
        pref_soil_veg=np.array([1,ntiles]),dz_mm=dz)


def test_tmc_source_default_accepts_real_migration_roundoff():
    inputs = _migration_inputs(two_tiles=True)
    source = h.hydrol_tmc_update(**inputs,check_cwrr=True,allowed_err=2e-8,raise_on_water_balance_error=False)
    expected_mc = inputs["mc"][:,:,0]*.6+inputs["mc"][:,:,1]*.4
    expected_water = inputs["water2infilt"][0,0]*.6+inputs["water2infilt"][0,1]*.4
    np.testing.assert_allclose(source.mc[0,:,0],expected_mc[0],rtol=0,atol=1e-15)
    np.testing.assert_allclose(source.tmc[0,0],_column(expected_mc[0],inputs["dz_mm"])+expected_water,rtol=0,atol=1e-15)
    residual = float(np.asarray(source.water_balance_error)[0])
    assert 0 < abs(residual) < 2e-7, f"real finite roundoff fixture residual={residual}"
    actual = h.hydrol_tmc_update(**inputs,check_cwrr=True,raise_on_water_balance_error=False)
    assert not bool(np.asarray(actual.water_balance_failed)[0]), f"source-default residual={residual} wrongly marked fatal"
    h.hydrol_tmc_update(**inputs,check_cwrr=True)


@pytest.mark.parametrize("offset",[-3e-7,-1e-7,0.,1e-7,3e-7])
def test_tmc_declared_default_residual_contract_both_sides(offset):
    inputs = _migration_inputs(old_storage_offset=offset)
    result = h.hydrol_tmc_update(**inputs,check_cwrr=True,raise_on_water_balance_error=False)
    # Predicate comes from actual independent stock/flux values, not from an
    # assumed exactly representable offset or an adjusted test tolerance.
    new_total = _column(inputs["mc"][0,:,0],inputs["dz_mm"])
    reference_residual = new_total-inputs["tmc"][0,0]
    np.testing.assert_allclose(result.water_balance_error,[reference_residual],rtol=0,atol=0)
    expected_fatal = abs(reference_residual) > 10*2e-8
    assert bool(np.asarray(result.water_balance_failed)[0]) == expected_fatal, (
        f"source2e-7 threshold: residual={reference_residual}, actual={result.water_balance_failed}, expected={expected_fatal}")
    if expected_fatal:
        with pytest.raises(RuntimeError,match="water-balance"):
            h.hydrol_tmc_update(**inputs,check_cwrr=True)
    else:
        h.hydrol_tmc_update(**inputs,check_cwrr=True)


def test_tmc_disabled_mode_and_explicit_strict_override_remain_distinct():
    inputs = _migration_inputs(old_storage_offset=1e-7)
    disabled = h.hydrol_tmc_update(**inputs)
    assert disabled.water_balance_error is None and disabled.water_balance_failed is None
    with pytest.raises(RuntimeError,match="water-balance"):
        h.hydrol_tmc_update(**inputs,check_cwrr=True,allowed_err=0.)
    permissive = h.hydrol_tmc_update(**inputs,check_cwrr=True,allowed_err=2e-8)
    assert not np.any(np.asarray(permissive.water_balance_failed))


def _source_infilt(mc,flux,dz,dt,sat,k,ks):
    """Literal positive-capacity branch, source7490-7566, one mineral tile."""
    result = mc.copy()
    first = min(max((sat-result[0])*dz[1]/2,0.),flux)
    result[0] += first*2/dz[1]
    total,remaining = first,dt
    flux_tmp = (flux-first)/dt
    for layer in range(1,len(mc)-1):
        km = (k[layer]+ks)/2
        rate = km*(1-np.exp(-flux_tmp/km))
        capacity = max((sat-result[layer])*(dz[layer]+dz[layer+1])/2,0.)
        duration = min(capacity/rate,remaining) if rate>1e-8 else remaining
        if rate>1e-8 and duration*rate>flux-total:
            duration = max(flux-total,0.)/rate
        result[layer] += duration*rate*2/(dz[layer]+dz[layer+1])
        remaining -= duration
        total += duration*rate
    return result,total,remaining


@pytest.mark.parametrize("compiled",[False,True],ids=["eager","jit"])
def test_infilt_valid_domain_bounds_and_source_values(compiled):
    # Positive finite geometry/conductivity/time, nonnegative supply and
    # finite intermediate exponential rates; this is not a proof outside it.
    dz,mc,k = np.array([0.,10.,20.,40.]),np.full((3,4),.3),np.full((3,4),2.)
    flux = np.array([0.,.2,100.])
    kwargs = dict(mc=mc,flux_infilt=flux,k=k,dz_mm=dz,dt_days=.5,mcs=np.full(3,.45),ks=3.,
        kfact=np.ones(4),kfact_root=np.ones((3,4)),check_cwrr2=True)
    call = jax.jit(lambda values: h.hydrol_soil_infilt_explicit(**{**kwargs,"flux_infilt":values})) if compiled else (
        lambda values:h.hydrol_soil_infilt_explicit(**{**kwargs,"flux_infilt":values}))
    result = call(flux)
    references = [_source_infilt(mc[point],flux[point],dz,.5,.45,k[point],3.) for point in range(3)]
    np.testing.assert_allclose(result.mc,np.stack([row[0] for row in references]),rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(result.infilt_tot,[row[1] for row in references],rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(result.dt_remaining,[row[2] for row in references],rtol=1e-13,atol=1e-13)
    assert np.all(np.asarray(result.infilt_tot)>=-1e-8)
    assert np.all(np.asarray(result.infilt_tot)<=flux+1e-8)
    np.testing.assert_allclose(result.ru_infilt+result.qinfilt,flux,rtol=0,atol=1e-13)
    np.testing.assert_allclose(result.check,0.,rtol=0,atol=1e-13)
