"""Bounded real HYDROL closure fixture; no model timestep or numerical mocks.

The literal inputs follow oracle_lane_hydrol_soil_owner's original Fortran
driver cases 1/5/17/18/19, except melt is zero (the real legal zero-snow path)
and case 18 uses a uniform cold thermal input. References remain independent.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba import hydrol as h

CASE_IDS = (1, 5, 17, 18, 19)
CASE_NAMES = {1: "normal", 5: "residual_dummy", 17: "default_mineral_normal", 18: "cold_freeze", 19: "inactive_tile_mask"}


def make_owner_inputs(case_id=1):
    """Return precall, wrapper kwargs and the declared 1x14x6x7 contract."""
    if case_id not in CASE_IDS:
        raise ValueError(f"bounded case_id must be one of {CASE_IDS}")
    npts, nvm, nstm, nslm = 1, 14, 6, 7
    dz = np.array([2., 4., 8., 12., 18., 24., 30.])
    zz = np.array([1., 4., 10., 20., 35., 55., 80.])
    veget = np.zeros((npts, nvm)); veget[0, [0, 13]] = [.2, .8]
    soiltile = np.full((npts, nstm), 1 / nstm)
    mc = np.full((npts, nslm, nstm), .30)
    pref = np.ones(nvm, dtype=np.int32)
    class3 = case_id in (17, 18, 19)
    texture = 3 if class3 else 2
    mcr, mcs, mcf, mcw, ks = (.065, .41, .1218, .0657, 1060.8) if class3 else (.057, .41, .30, .12, 106.1)
    parameters = dict(nvan=np.full(12, 1.89), avan=np.full(12, .0075),
        mcr=np.full(12, mcr), mcs_mineral=np.full(12, mcs),
        mcf_mineral=np.full(12, mcf), mcw_mineral=np.full(12, mcw),
        ks=np.full(12, ks), pcent=np.full(12, .8),
        mc_awet=np.full(12, .25 if class3 else .32), mc_adry=np.full(12, .10))
    if case_id == 5:
        mc[:, 0, :] = .055
    if case_id == 18:
        mc[0, :, 0] = [.075, .080, .095, .13, .19, .26, .34]
    if case_id == 19:
        soiltile[:] = 0.; soiltile[:, 3] = 1.
        pref[13] = 4
        mc[:] = 0.; mc[0, :, 3] = [.075, .080, .095, .13, .19, .26, .34]
    njsc = np.full(npts, texture, dtype=np.int32)
    freeze = case_id == 18
    initialized = h.hydrol_var_init(veget=veget, veget_max=veget,
        soiltile=soiltile, njsc=njsc, znh_m=zz/1000., dnh_m=dz/1000.,
        dlh_m=dz/1000., humcste=np.ones(nvm), altmax=np.full((npts,nvm),10.),
        mc=mc, mcl=mc.copy(), water2infilt=np.zeros((npts,nstm)),
        vegtot_old=np.ones(npts), drysoil_frac=np.zeros(npts),
        ok_pc=False, ok_leak=False, peat_hydro=False, tides=False,
        agri_peat=False, ok_freeze_cwrr=freeze, soil_parameters=parameters)
    temp_hydro = np.full((npts, nslm), 271.5 if freeze else 280.)
    profile = np.zeros_like(mc)
    if freeze:
        # Actual pre-call owner; later independent Fortran/NumPy checks certify it.
        profile = h.hydrol_soil_froz_profile(temp_hydro=temp_hydro, mc=initialized.mc,
            njsc=njsc, dh_mm=dz, nvan=parameters["nvan"], avan=parameters["avan"],
            mcr=parameters["mcr"], mcs=parameters["mcs_mineral"])
    pft_evap = np.zeros((npts,nvm)); pft_evap[:,0] = .01
    transpir = np.zeros((npts,nvm)); transpir[:,13] = .02
    payload = dict(precip_rain=np.zeros(npts), vevapwet=np.zeros((npts,nvm)),
        veget=veget, veget_max=veget, qsintmax=np.zeros((npts,nvm)),
        qsintveg=np.zeros((npts,nvm)), tot_melt=np.zeros(npts), vegtot=np.ones(npts),
        throughfall_by_pft=np.ones(nvm), pref_soil_veg=pref, soiltile=soiltile,
        vevapflo=np.zeros(npts), flood_frac=np.zeros(npts), flood_res=np.zeros(npts),
        subsinksoil=np.zeros(npts), vevapnu=np.full(npts,.01), vevapnu_pft=pft_evap,
        transpir=transpir, humrel=np.full((npts,nvm),.5),
        humrelv=np.zeros((npts,nvm,nstm)), us=np.zeros((npts,nvm,nstm,nslm)),
        ae_ns=np.zeros((npts,nstm)), evap_bare_lim=np.full(npts,.5),
        evap_bare_lim_ns=np.zeros((npts,nstm)), evapot=np.full(npts,.2),
        evapot_corr=np.full(npts,1. if case_id==5 else .2), tot_bare_soil=np.full(npts,.2),
        water2infilt=np.zeros((npts,nstm)), reinfiltration_soil=np.zeros((npts,nstm)),
        irrigation_soil=np.zeros((npts,nstm)), mc=np.asarray(initialized.mc),
        mcl=np.asarray(initialized.mcl), profil_froz=np.asarray(profile),
        mcr=np.full(npts,mcr), mcs=np.full(npts,mcs), mcf=np.full(npts,mcf), mcw=np.full(npts,mcw),
        dz_mm=dz, dh_mm=dz, dt_days=1/48., free_drain_coef=np.ones((npts,nstm)),
        resolv=soiltile>0., kfact_root=np.ones_like(mc), ks=parameters["ks"],
        kfact=initialized.mineral_tables.kfact, tmc=np.asarray(initialized.tmc), njsc=njsc,
        mineral_tables=initialized.mineral_tables, peat_tables=None,
        temp_hydro=temp_hydro, zwt_force=np.full((npts,nstm),1e20),
        zz_mm=zz, zmaxh_m=2., wt_ab=np.zeros(npts), wt_ab_tide=np.zeros(npts),
        precip_snow=np.zeros(npts), vevapsno=np.zeros(npts), snow=np.zeros(npts),
        snowdz=np.zeros((npts,3)), snowrho=np.full((npts,3),100.),
        snowtemp=np.full((npts,3),273.15), snow_age=np.zeros(npts),
        snow_nobio=np.zeros((npts,1)), snow_nobio_age=np.zeros((npts,1)),
        frac_nobio=np.zeros((npts,1)), totfrac_nobio=np.zeros(npts))
    signature = inspect.signature(h.hydrol_module_explicit_step).parameters
    for name in h.HYDROL_MODULE_EXPLICIT_STEP_FIELDS:
        if name not in payload and name in signature and signature[name].default is not inspect.Parameter.empty:
            payload[name] = signature[name].default
    precall = h.HydrolFirstStepPrecallAssembly(payload,(),None,(),(),(),(),(),(),
        ("literal bounded original-source soil-owner fixture; real legal explicit zero snow",))
    kwargs = dict(humcste=np.ones(nvm), nroot_state=initialized.nroot,
        dz_mm=dz, dh_mm=dz, zz_mm=zz, ks=parameters["ks"], reinf_slope=np.full(npts,.1),
        zmaxh_m=2., ok_freeze_cwrr=freeze, ok_thermodynamical_freezing=True,
        ok_pc=False, ok_leak=False, peat_hydro=False, dyn_nroot_larix=case_id in (18,19),
        is_tree=np.array([False]*13+[True]), znt=np.array([.01,.03,.07,.15,.30,.60,1.20]),
        runtime_static_tables=h.HydrolRuntimeStaticTables(initialized.mineral_tables,None))
    return dict(precall=precall, kwargs=kwargs, initialized=initialized,
        soil_parameters=parameters, contract=dict(case_id=case_id,name=CASE_NAMES[case_id],
            dimensions=dict(npts=npts,nvm=nvm,nstm=nstm,nslm=nslm,nsnow=3),
            snow="actual explicit zero-state owner", melt=0.,
            topmodel_wrapper_reachable=False, source_case=case_id))


def run_owner_case(case_id=1, *, use_jit=False):
    """Return the unmodified real HydrolFirstStepModuleClosure."""
    values = make_owner_inputs(case_id)
    result = h.run_hydrol_first_step_module_from_precall(values["precall"],
        **values["kwargs"], use_jit=use_jit)
    if not result.ok:
        raise RuntimeError(result.missing_inputs)
    return result


def run_owner_diagnostics(closure, *, topmodel_new=True):
    """Run a complete independent diagnostic chain on the real module result.

    This is explicitly outside the wrapper's admitted TOPMODEL=False route.
    No output field or scalar is manually overwritten.
    """
    values = dict(closure.diagnostic_inputs)
    values.update(liqlayers=closure.module.soil.mc.shape[1]-1,
        numlayers=closure.module.soil.mc.shape[1], topmodel_new=topmodel_new,
        param_vp=np.array([.8]), param_kp=np.array([2.]),
        param_qp=np.array([.1]), param_fmax=np.array([.7]))
    return h.hydrol_module_diagnostics(closure.module, **values)


if __name__ == "__main__":
    import argparse
    import json
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=CASE_IDS, default=1)
    parser.add_argument("--jit", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    result = run_owner_case(args.case, use_jit=args.jit)
    mc = np.asarray(result.module.soil.mc)
    print(json.dumps(dict(case=args.case, jit=args.jit, ok=result.ok,
        elapsed_s=time.perf_counter()-started, shape=list(mc.shape),
        mc_finite=bool(np.all(np.isfinite(mc))), snow=list(np.asarray(result.snow_state.snow)))))
