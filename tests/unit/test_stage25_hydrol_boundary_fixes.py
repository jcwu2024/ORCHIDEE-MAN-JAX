"""Stage25 A/B/C: finite independent source references, never a model run.

Source truth: hydrol.f90 6884-6905/7165-7174 (A), 8248-8310 (B),
7286-7293 (C), and 8332-8455 for the freezing law. Tables are actual
source-backed builders; the dummy matrices/RHS/solve are independent NumPy.
The signature gate only lets the unchanged baseline run numerically before
the newly required geometry/control interface exists; it is not an oracle.
"""
from __future__ import annotations

from functools import lru_cache
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from jax_orchidee.sechiba import hydrol as h


CONTROLS = dict(fr_dt=3.2, froz_frac_corr=.7, smtot_corr=1.3, max_froz_hydro=.8)


def _source_profile(mc, temperature, dh, residual, saturation, nvg, avg, thermo, controls):
    """Literal hydrol_soil_froz8396-8455, soil lhf333600 remains independent."""
    raw = np.empty_like(mc)
    lower = 273.15-controls["fr_dt"]/2
    upper = 273.15+controls["fr_dt"]/2
    for point in range(mc.shape[0]):
        for tile in range(mc.shape[2]):
            for layer in range(mc.shape[1]):
                moisture, temp = mc[point, layer, tile], temperature[point, layer]
                if (not thermo) or moisture < residual+1e-8:
                    x = 1. if temp >= upper else ((temp-lower)/controls["fr_dt"] if temp >= lower else 0.)
                elif temp >= upper:
                    x = 1.
                elif temp >= lower:
                    base = 2.2*1000*avg*(upper-temp)*333600/273.15/10
                    x = min((saturation-residual)*(base**nvg+1)**(-(1-1/nvg))/(moisture-residual), 1.)
                else:
                    x = 0.
                raw[point,layer,tile] = 1-x
            fm = sum(dh[j]*raw[point,j,tile] for j in range(len(dh)))/sum(dh)
            sm = sum(dh[j]*mc[point,j,tile]/saturation for j in range(len(dh)-1))/sum(dh[:-1])
            raw[point,:,tile] = np.minimum(raw[point,:,tile]*fm**controls["froz_frac_corr"]*sm**controls["smtot_corr"], controls["max_froz_hydro"])
    return raw


def _water(column, dz):
    total = dz[1]*(3*column[0]+column[1])/8 + dz[-1]*(3*column[-1]+column[-2])/8
    for layer in range(1,len(column)-1):
        total += dz[layer]*(3*column[layer]+column[layer-1])/8
        total += dz[layer+1]*(3*column[layer]+column[layer+1])/8
    return total


@lru_cache(maxsize=4)
def _tables(family="mineral", texture=2, saturation=.41):
    z = np.array([0.,.01,.03,.07])
    return (h.build_mineral_cwrr_tables(njsc=texture,mcs=[saturation],z_m=z)
            if family == "mineral" else h.build_peat_cwrr_tables(z))


def _a_inputs(mode, nslm=4):
    residual, saturation = .057,.41
    dz = np.r_[0.,10.*2.**np.arange(nslm-1)]
    dh = np.r_[dz[1]/2,(dz[1:-1]+dz[2:])/2,dz[-1]/2]
    mc = np.broadcast_to(np.linspace(.28,.31,nslm)[None,:,None],(1,nslm,6)).copy()
    temp = np.full((1,nslm),273.15)
    thermo = mode != "linear"
    profile = _source_profile(mc,temp,dh,residual,saturation,2.28,.0124,thermo,CONTROLS) if mode != "off" else np.zeros_like(mc)
    tables = _tables() if nslm==4 else h.build_mineral_cwrr_tables(njsc=2,mcs=[saturation],z_m=np.cumsum(dz)/1000)
    inputs = dict(water2infilt=np.zeros((1,6)),ae_ns=np.zeros((1,6)),subsinksoil=np.zeros(1),
        precisol_ns=np.full((1,6),.4),reinfiltration_soil=np.zeros((1,6)),is_crop_soil=False,
        mc=mc,mcl=np.minimum(mc,residual+(1-profile)*(mc-residual)),profil_froz=profile,
        mcr=np.array([residual]),mcs=np.array([saturation]),dz_mm=dz,dt_days=1/48,
        free_drain_coef=np.ones((1,6)),rootsink=np.zeros_like(mc),resolv=np.ones((1,6),bool),
        mask_soiltile=np.ones((1,6)),kfact_root=np.ones_like(mc),ks=np.array([3501.6]),
        kfact=np.asarray(tables.kfact)[:,0],soiltile=np.full((1,6),1/6),
        tmc=np.array([[_water(mc[0,:,tile],dz) for tile in range(6)]]),evapot=np.array([.05]),
        evapot_corr=np.array([.05]),frac_bare_ns=np.ones((1,6)),njsc=np.array([2]),mineral_tables=tables,
        reinf_slope=np.array([0.]),peat_hydro=False,ok_freeze_cwrr=mode!="off",temp_hydro=temp,
        zz_mm=np.cumsum(dz),zmaxh_m=float(sum(dz))/1000)
    extra = dict(dh_mm=dh,ok_thermodynamical_freezing=thermo,**CONTROLS)
    return inputs,extra


def _call_a(function, inputs, extra):
    accepted = inspect.signature(function).parameters
    return function(**inputs,**{key:value for key,value in extra.items() if key in accepted})


def _source_dummy(moisture, ice, tables, inputs, tile=0):
    """Source6887-7108: independently construct coefficients, matrix and RHS."""
    residual,saturation = float(inputs["mcr"][0]),float(inputs["mcs"][0])
    dz,dt = inputs["dz_mm"],inputs["dt_days"]
    n = len(dz)
    liquid = np.minimum(moisture,residual+(1-ice)*(moisture-residual))
    if inputs["ok_freeze_cwrr"]:
        used = residual+(1-ice)*np.maximum(moisture-residual,0.)
        bins = np.trunc(1+50*(used-residual)/(saturation-residual)).astype(int)
    else:
        used = moisture
        ratio = np.maximum(moisture-residual,0.)/(saturation-residual)
        scaled = 50*ratio
        bins = np.trunc(scaled).astype(int)+1
    bins = np.clip(bins,1,50)
    a,b,d,k = [np.empty(n) for _ in range(4)]
    for layer,offset in enumerate(bins-1):
        a[layer] = np.asarray(tables.a_lin)[offset,layer,0]
        b[layer] = np.asarray(tables.b_lin)[offset,layer,0]
        d[layer] = np.asarray(tables.d_lin)[offset,layer,0]
        k[layer] = max(np.asarray(tables.k_lin)[1,layer,0],a[layer]*used[layer]+b[layer])
    left,right = np.zeros((n,n)),np.zeros((n,n))
    t3,t4,free = dt/2,0.,float(inputs["free_drain_coef"][0,tile])
    for layer in range(n):
        if layer == 0:
            diff = (d[0]+d[1])/dz[1]
            left[0,:2] = [3*dz[1]/8+t3*(diff+a[0]),dz[1]/8-t3*(diff-a[1])]
            right[0,:2] = [3*dz[1]/8-t4*(diff+a[0]),dz[1]/8+t4*(diff-a[1])]
        elif layer == n-1:
            diff = (d[layer]+d[layer-1])/dz[layer]
            left[layer,layer-1:] = [dz[layer]/8-t3*(diff+a[layer-1]),3*dz[layer]/8+t3*(diff-a[layer]*(1-2*free))]
            right[layer,layer-1:] = [dz[layer]/8+t4*(diff+a[layer-1]),3*dz[layer]/8-t4*(diff-a[layer]*(1-2*free))]
        else:
            dl,dr = (d[layer]+d[layer-1])/dz[layer],(d[layer]+d[layer+1])/dz[layer+1]
            left[layer,layer-1:layer+2] = [dz[layer]/8-t3*(dl+a[layer-1]),3*(dz[layer]+dz[layer+1])/8+t3*(dl+dr),dz[layer+1]/8-t3*(dr-a[layer+1])]
            right[layer,layer-1:layer+2] = [dz[layer]/8+t4*(dl+a[layer-1]),3*(dz[layer]+dz[layer+1])/8-t4*(dl+dr),dz[layer+1]/8+t4*(dr-a[layer+1])]
    rhs = right@liquid
    flux_top = float(inputs["evapot_corr"][0])
    rhs[0] -= flux_top+(b[0]+b[1])*dt/2
    for layer in range(1,n-1):
        rhs[layer] += (b[layer-1]-b[layer+1])*dt/2
    rhs[-1] += (b[-2]+b[-1]*(1-2*free))*dt/2
    first = np.linalg.solve(left,rhs)
    retry = bool(first[0]<residual and flux_top>1e-8)
    second = first.copy()
    if retry:
        alt_left,alt_rhs = left.copy(),rhs.copy()
        alt_left[0] = 0.;alt_left[0,0] = 1.;alt_rhs[0] = residual
        second = np.linalg.solve(alt_left,alt_rhs)
    first_total = np.maximum(second,second+ice*(moisture-residual)) if inputs["ok_freeze_cwrr"] else second
    total = np.maximum(second,second+ice*(first_total-residual)) if inputs["ok_freeze_cwrr"] else second
    return dict(rhs=rhs,first=first,second=second,retry=retry,mc_after=total,tmc=_water(total,dz),flux_bottom=k[-1]*free*dt)


def _assert_a(soil, inputs, extra, mode):
    final_mc = np.asarray(soil.mc)
    fresh = (_source_profile(final_mc,inputs["temp_hydro"],extra["dh_mm"],.057,.41,2.28,.0124,
             extra["ok_thermodynamical_freezing"],CONTROLS) if mode!="off" else np.zeros_like(final_mc))
    # Real normal state, not a nonmutation check: compare returned arrays to
    # each actual normal solve and source restore7165-7174 before dummy results.
    for tile,result in enumerate(soil.tile_results):
        normal_mc = np.asarray(result.solve.mc_final)
        normal_mcl = np.minimum(normal_mc,.057+(1-inputs["profil_froz"][:,:,tile])*(normal_mc-.057))
        np.testing.assert_allclose(final_mc[:,:,tile],normal_mc,rtol=0,atol=0)
        np.testing.assert_allclose(np.asarray(soil.mcl)[:,:,tile],normal_mcl,rtol=1e-13,atol=1e-14)
        np.testing.assert_allclose(np.asarray(soil.tmc)[:,tile],
            [_water(normal_mc[0],inputs["dz_mm"])+float(soil.water2infilt[0,tile])],rtol=1e-12,atol=1e-12)
    reference = _source_dummy(final_mc[0,:,0],fresh[0,:,0],inputs["mineral_tables"],inputs)
    alt = soil.evap_bare_limit_alt
    profile_error = float(np.max(np.abs(np.asarray(soil.profil_froz_hydro_ns)-fresh)))
    dummy_error = float(np.max(np.abs(np.asarray(alt.saved.rhs)[0,:,0]-reference["rhs"])))
    assert profile_error < 1e-12 and dummy_error < 1e-10, f"refresh profile_error={profile_error}, dummy_rhs_error={dummy_error}"
    for key,array in (("first",alt.first.mcl),("second",alt.second.mcl),("mc_after",alt.mc_after)):
        np.testing.assert_allclose(np.asarray(array)[0,:,0],reference[key],rtol=1e-10,atol=1e-11)
    np.testing.assert_allclose(np.asarray(alt.tmc)[0,0],reference["tmc"],rtol=1e-10,atol=1e-11)
    np.testing.assert_array_equal(np.asarray(alt.resolv_alt)[0,0],reference["retry"])


@pytest.mark.parametrize("mode",["thermo","linear","off"])
def test_a_final_normal_mc_refresh_dummy_and_restore(mode):
    inputs,extra = _a_inputs(mode)
    soil = _call_a(h.hydrol_soil_all_tiles_explicit_step,inputs,extra)
    _assert_a(soil,inputs,extra,mode)


def _a_module_inputs(inputs):
    module = dict(inputs)
    for key in ("precisol_ns","rootsink","mask_soiltile","frac_bare_ns"):
        module.pop(key)
    module.update(precip_rain=np.array([2.4]),vevapwet=np.zeros((1,6)),veget=np.full((1,6),1/6),
        veget_max=np.full((1,6),1/6),qsintmax=np.zeros((1,6)),qsintveg=np.zeros((1,6)),tot_melt=np.zeros(1),
        vegtot=np.ones(1),throughfall_by_pft=np.zeros(6),pref_soil_veg=np.arange(1,7),
        vevapflo=np.zeros(1),flood_frac=np.zeros(1),flood_res=np.zeros(1),vevapnu=np.zeros(1),
        vevapnu_pft=np.zeros((1,6)),transpir=np.zeros((1,6)),humrel=np.ones((1,6)),humrelv=np.ones((1,6,6)),
        us=np.zeros((1,6,6,inputs["mc"].shape[1])),evap_bare_lim=np.zeros(1),evap_bare_lim_ns=np.zeros((1,6)),tot_bare_soil=np.ones(1))
    return module


def test_a_real_module_boundary_passes_geometry_and_effective_controls():
    inputs,extra = _a_inputs("linear")
    module = _a_module_inputs(inputs)
    result = _call_a(h.hydrol_module_explicit_step,module,extra)
    _assert_a(result.soil,inputs,extra,"linear")


def test_a_enabled_refresh_rejects_missing_actual_temperature_geometry_texture():
    # Added after the immutable common-before receipt: contract admission,
    # not additional baseline numerical-failure evidence.
    inputs,extra = _a_inputs("thermo")
    for missing in ("temp_hydro","dh_mm","njsc"):
        trial,trial_extra = dict(inputs),dict(extra)
        if missing in trial:
            trial[missing] = None
        else:
            trial_extra[missing] = None
        with pytest.raises(ValueError,match=missing):
            h.hydrol_soil_all_tiles_explicit_step(**trial,**trial_extra)


def test_a_narrow_jit_real_all_tiles_refresh_and_restore():
    # Added after common-before: one small real all-tiles component, never a
    # model step. Normal compiled state seeds the independent final-mc ref.
    inputs,extra = _a_inputs("linear")
    call = jax.jit(lambda values: h.hydrol_soil_all_tiles_explicit_step(**{**inputs,**extra,"mc":values}))
    soil = call(jnp.asarray(inputs["mc"]))
    _assert_a(soil,inputs,extra,"linear")


def test_a_actual_precall_closure_to_module_diagnostics_and_downstream(monkeypatch):
    inputs,extra = _a_inputs("linear",7)
    payload = _a_module_inputs(inputs)
    # Only the unrelated snow boundary is stubbed; no module/normal solve/
    # refresh/dummy/diagnostics/output/moisture consumer is replaced.
    payload.update(precip_snow=np.zeros(1),vevapsno=np.zeros(1),snow=np.zeros(1),snowdz=np.zeros((1,3)),
        snowrho=np.full((1,3),100.),snowtemp=np.full((1,3),273.15),snow_age=np.zeros(1),
        snow_nobio=np.zeros((1,1)),snow_nobio_age=np.zeros((1,1)),frac_nobio=np.zeros((1,1)),
        totfrac_nobio=np.zeros(1),irrigation_soil=np.zeros((1,6)),zwt_force=np.full((1,6),1e20),
        wt_ab=np.zeros(1),wt_ab_tide=np.zeros(1))
    signature = inspect.signature(h.hydrol_module_explicit_step).parameters
    for name in h.HYDROL_MODULE_EXPLICIT_STEP_FIELDS:
        if name not in payload and name in signature and signature[name].default is not inspect.Parameter.empty:
            payload[name] = signature[name].default
    monkeypatch.setattr(h,"hydrol_explicit_snow_zero_state",lambda **kwargs: SimpleNamespace(tot_melt=np.zeros(1)))
    precall = h.HydrolFirstStepPrecallAssembly(payload,(),None,(),(),(),(),(),(),("synthetic explicit component payload",))
    kwargs = dict(humcste=np.ones(6),nroot_state=np.full((1,6,7),1/7),dz_mm=inputs["dz_mm"],
        dh_mm=extra["dh_mm"],zz_mm=inputs["zz_mm"],ks=inputs["ks"],reinf_slope=inputs["reinf_slope"],
        zmaxh_m=inputs["zmaxh_m"],ok_freeze_cwrr=True,ok_pc=False,ok_leak=False,peat_hydro=False,
        runtime_static_tables=h.HydrolRuntimeStaticTables(inputs["mineral_tables"],None))
    for key,value in extra.items():
        if key in inspect.signature(h.run_hydrol_first_step_module_from_precall).parameters:
            kwargs[key] = value
    result = h.run_hydrol_first_step_module_from_precall(precall,**kwargs)
    assert result.ok, result.missing_inputs
    _assert_a(result.module.soil,inputs,extra,"linear")
    profile = np.asarray(result.module.soil.profil_froz_hydro_ns)
    expected_grid = np.sum(profile*inputs["soiltile"][:,None,:],axis=2)
    np.testing.assert_allclose(np.asarray(result.diagnostics.profil_froz_hydro),expected_grid,rtol=1e-12,atol=1e-13)
    assert result.thermosoil_moisture is not None
    assert np.all(np.isfinite(np.asarray(result.thermosoil_moisture.shumdiag_perma)))


def test_a_r1_real_closure_writer_packet_and_no_thermosoil_fallback():
    """Source6884-6886/7165-7174 retains fresh ice, restores normal water.

    Unlike the original diagnostics-only closure case, use the real legal
    zero-snow boundary and the actual driver writer/packet implementation.
    No HYDROL process, temperature, result, or writer is replaced.
    """
    from jax_orchidee.driver import orchestration as driver

    inputs,extra = _a_inputs("linear",7)
    payload = _a_module_inputs(inputs)
    payload.update(precip_snow=np.zeros(1),vevapsno=np.zeros(1),snow=np.zeros(1),snowdz=np.zeros((1,3)),
        snowrho=np.full((1,3),100.),snowtemp=np.full((1,3),273.15),snow_age=np.zeros(1),
        snow_nobio=np.zeros((1,1)),snow_nobio_age=np.zeros((1,1)),frac_nobio=np.zeros((1,1)),
        totfrac_nobio=np.zeros(1),irrigation_soil=np.zeros((1,6)),zwt_force=np.full((1,6),1e20),
        wt_ab=np.zeros(1),wt_ab_tide=np.zeros(1))
    signature = inspect.signature(h.hydrol_module_explicit_step).parameters
    for name in h.HYDROL_MODULE_EXPLICIT_STEP_FIELDS:
        if name not in payload and name in signature and signature[name].default is not inspect.Parameter.empty:
            payload[name] = signature[name].default
    precall = h.HydrolFirstStepPrecallAssembly(payload,(),None,(),(),(),(),(),(),
        ("synthetic explicit component payload with legal zero snow",))
    result = h.run_hydrol_first_step_module_from_precall(precall,
        humcste=np.ones(6),nroot_state=np.full((1,6,7),1/7),dz_mm=inputs["dz_mm"],
        zz_mm=inputs["zz_mm"],ks=inputs["ks"],reinf_slope=inputs["reinf_slope"],
        zmaxh_m=inputs["zmaxh_m"],ok_freeze_cwrr=True,ok_pc=False,ok_leak=False,peat_hydro=False,
        runtime_static_tables=h.HydrolRuntimeStaticTables(inputs["mineral_tables"],None),**extra)
    assert result.ok, result.missing_inputs
    assert result.snow_state is not None
    _assert_a(result.module.soil,inputs,extra,"linear")
    soil = result.module.soil
    old_profile = np.asarray(result.module_inputs["profil_froz"])
    new_profile = np.asarray(soil.profil_froz_hydro_ns)
    changed = float(np.max(np.abs(new_profile-old_profile)))
    assert changed > 1e-8, f"fixture must distinguish old/new profiles: {changed}"
    normal = {name:np.asarray(getattr(soil,name)).copy() for name in ("mc","mcl","tmc")}

    fields = driver._empty_previous_step_state_fields()
    driver._add_hydrol_to_previous_fields(fields,result)
    packet = driver._packet_from_previous_step_fields(tstep=1,fields=fields)
    persisted = packet.fields_by_component["hydrol_previous_step_state"]
    for name in ("mc","mcl"):
        np.testing.assert_array_equal(persisted[name],normal[name])
    for name in ("mc","mcl","tmc"):
        np.testing.assert_array_equal(np.asarray(getattr(soil,name)),normal[name])

    # Exercise the actual supported no-THERMOSOIL precall fallback. Ordinary
    # steps with THERMOSOIL recompute ice; this test does not cover that path.
    restart = SimpleNamespace(**persisted,
        moistc=np.swapaxes(np.asarray(persisted["mc"]),1,2),
        moistcl=np.swapaxes(np.asarray(persisted["mcl"]),1,2))
    fallback_template = {**payload,"temp_sol_new":np.full(1,273.15),
        "pgflux":np.zeros(1),"temp_sol_add":np.zeros(1),"transpot":np.zeros((1,6))}
    fallback = h.assemble_hydrol_first_step_precall_payload(
        enerbil_payload=None,hydrol_restart=restart,slowproc_restart=None,
        pref_soil_veg=np.arange(1,7),nstm=6,ext_coeff_vegetfrac=np.ones(6),
        static_payload_template=fallback_template,thermosoil_restart=None,zz_mm=inputs["zz_mm"],
        ok_freeze_cwrr=True,**{key:value for key,value in extra.items() if key!="dh_mm"})
    assert fallback.ok, fallback.missing_inputs
    np.testing.assert_array_equal(fallback.payload["mc"],normal["mc"])
    np.testing.assert_array_equal(fallback.payload["mcl"],normal["mcl"])
    np.testing.assert_array_equal(fallback.payload["temp_hydro"],inputs["temp_hydro"])
    rebuilt_tmc = h.hydrol_initial_tmc_from_restart_mc(mc=fallback.payload["mc"],
        water2infilt=persisted["water2infilt"],dz_mm=inputs["dz_mm"])
    np.testing.assert_allclose(rebuilt_tmc,normal["tmc"],rtol=1e-12,atol=1e-12)
    writer_error = float(np.max(np.abs(np.asarray(persisted["profil_froz_hydro_ns"])-new_profile)))
    fallback_error = float(np.max(np.abs(np.asarray(fallback.payload["profil_froz"])-new_profile)))
    assert writer_error < 1e-12 and fallback_error < 1e-12, (
        f"fresh/old delta={changed}; writer_profile_error={writer_error}; fallback_profile_error={fallback_error}")


def _source_bin_and_coef(mc,ice,tables,residual,saturation,freeze,family):
    # Literal source8291-8301 staging is consequential. Do not rewrite as
    # 50*(mc-residual)/(saturation-residual) or move imin inside INT.
    used = residual+(1-ice)*np.maximum(mc-residual,0.) if freeze else mc
    ratio = ((used-residual)/(saturation-residual) if freeze
             else np.maximum(mc-residual,0.)/(saturation-residual))
    scaled = ((tables.imax-tables.imin)*(used-residual)/(saturation-residual) if freeze
              else (tables.imax-tables.imin)*ratio)
    index = np.trunc(tables.imin+scaled).astype(np.int32) if freeze else np.trunc(scaled).astype(np.int32)+tables.imin
    index = np.clip(index,tables.imin,tables.imax-1)
    outputs = {key:np.empty_like(mc) for key in ("a","b","d","k")}
    for point in range(mc.shape[0]):
        for layer in range(mc.shape[1]):
            offset = int(index[point,layer]-tables.imin)
            at = (offset,layer,0) if family=="mineral" else (offset,layer)
            a,b,d = (np.asarray(getattr(tables,key))[at] for key in ("a_lin","b_lin","d_lin"))
            floor_at = (1,layer,0) if family=="mineral" else (1,layer)
            outputs["a"][point,layer],outputs["b"][point,layer],outputs["d"][point,layer] = a,b,d
            outputs["k"][point,layer] = max(np.asarray(tables.k_lin)[floor_at],a*used[point,layer]+b)
    return index,outputs


def _b_case(family):
    residual,saturation = (.045,.43) if family=="mineral" else (.15,.9)
    tables = _tables(family,1,saturation)
    values = [residual-.01,residual,saturation,saturation+.01]
    for edge in (1,16,25,49):
        bound = residual+edge*(saturation-residual)/50
        values.extend([np.nextafter(bound,-np.inf),bound,np.nextafter(bound,np.inf)])
    # Explicit accepted Stage24 boundary, plus a peat near-boundary search is
    # unnecessary: the small fixed source boundaries above cover its bins.
    mc = np.repeat(np.asarray(values)[:,None],4,axis=1)
    return residual,saturation,tables,mc


@pytest.mark.parametrize("family",["mineral","peat"])
@pytest.mark.parametrize("compiled",[False,True],ids=["eager","narrow_jit"])
def test_b_literal_nonfreeze_indices_real_table_amplitude_and_freeze_control(family,compiled):
    residual,saturation,tables,mc = _b_case(family)
    def evaluate(values,ice,freeze):
        if family=="mineral":
            return h.hydrol_soil_coef_mineral_profile_from_tables(values,ice,jnp.ones_like(values),tables,
                jnp.full((len(mc),),residual),jnp.full((len(mc),),saturation),point_index=0,ok_freeze_cwrr=freeze)
        return h.hydrol_soil_coef_peat_profile_from_tables(values,ice,jnp.ones_like(values),tables,ok_freeze_cwrr=freeze)
    call = jax.jit(evaluate,static_argnums=2) if compiled else evaluate
    for freeze in (False,True):
        ice = np.full_like(mc,.13) if freeze else np.zeros_like(mc)
        expected,coefficients = _source_bin_and_coef(mc,ice,tables,residual,saturation,freeze,family)
        actual = call(jnp.asarray(mc),jnp.asarray(ice),freeze)
        np.testing.assert_array_equal(np.asarray(actual.bin_i),expected)
        for key in ("a","b","d","k"):
            np.testing.assert_allclose(np.asarray(getattr(actual,key)),coefficients[key],rtol=2e-12,atol=1e-11)


def _c_inputs(peat=True,tides=True,equal=False,nstm=6):
    dz = np.array([0.,1.,2.,4.,8.,16.,32.,64.,128.,256.,512.])
    dh = np.r_[dz[1]/2,(dz[1:-1]+dz[2:])/2,dz[-1]/2]
    area = np.full((3,nstm),1/nstm)
    if nstm>=6:
        area[:,5] = [0.,1e-8,np.nextafter(1e-8,np.inf)]
        area[:,0] += 1/nstm-area[:,5]
    return dict(mc=np.full((3,11,nstm),.6),mcl=np.full((3,11,nstm),.6),soiltile=area,
        dz_mm=dz,dh_mm=dh,wt_ab=np.array([10.,20.,30.]),wt_ab_tide=np.array([10.,20.,30.]) if equal else np.array([50.,60.,70.]),
        peat_hydro=peat,tides=tides,liqlayers=10,numlayers=11,topmodel_new=True,
        temp_hydro=np.tile([275.]*7+[270.]*4,(3,1)),mcs=np.full(3,.8),
        param_vp=np.ones(3),param_kp=np.full(3,2.),param_qp=np.zeros(3),param_fmax=np.full(3,.9))


def _source_c(inputs):
    expected = []
    for point in range(3):
        count = 0
        while count<11 and inputs["temp_hydro"][point,count]>=273.15:
            count += 1
        depth = sum(inputs["dh_mm"][:count])
        tiles = []
        for tile in range(inputs["mc"].shape[2]):
            sat = .9 if inputs["peat_hydro"] else .8
            water = sum(inputs["mc"][point,layer,tile]/sat*inputs["dh_mm"][layer] for layer in range(count))/1000-depth/1000
            if water>0:
                water *= sat
            if inputs["peat_hydro"]:
                use_tide = inputs["tides"] and inputs["soiltile"].shape[1]>=6 and inputs["soiltile"][point,5]>1e-8
                water += (inputs["wt_ab_tide"][point] if use_tide else inputs["wt_ab"][point])/1000
            tiles.append(water)
        expected.append(np.dot(inputs["soiltile"][point],tiles))
    mean = np.array(expected)
    return mean,np.maximum(0,np.minimum(.9,(1+np.exp(-2*mean))**(-1)))


@pytest.mark.parametrize("compiled",[False,True],ids=["eager","narrow_jit"])
def test_c_cell_tile6_strict_selector_controls_and_existing_small_axis(compiled):
    for peat,tides,equal,nstm in ((True,True,False,6),(True,False,False,6),(True,True,True,6),(False,True,False,6),(True,True,False,5)):
        inputs = _c_inputs(peat,tides,equal,nstm)
        mean,fwet = _source_c(inputs)
        if compiled:
            call = jax.jit(lambda area,wt,tide: h.hydrol_peat_water_table_diagnostics(**{**inputs,"soiltile":area,"wt_ab":wt,"wt_ab_tide":tide}))
            actual = call(jnp.asarray(inputs["soiltile"]),jnp.asarray(inputs["wt_ab"]),jnp.asarray(inputs["wt_ab_tide"]))
        else:
            actual = h.hydrol_peat_water_table_diagnostics(**inputs)
        np.testing.assert_allclose(np.asarray(actual.meanwt),mean,rtol=1e-12,atol=1e-13)
        np.testing.assert_allclose(np.asarray(actual.fwet_new),fwet,rtol=1e-12,atol=1e-13)
