"""PB-03 finite HYDROL production components and original-Fortran reference.

The public fixture/API executes initialization followed by the actual canopy,
soil, diagnostic and moisture-export components. It never takes a model step.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba import hydrol as h  # noqa: E402

OUTPUT = ROOT / "outputs/reviews/main-physics-20261119/physics-branch-closeout/phase02-hydrol-thermal/hydrol"
FIXTURE = ROOT / "tests/fixtures/physics_branch_hydrol_reference.json"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90"
RTOL, ATOL = 1e-12, 1e-14


def make_cases():
    """Complete declared component inputs; all fluxes are kg m-2 per 1800 s."""
    names = [
        "mineral_rain_canopy",
        "mineral_canopy_overflow",
        "mineral_evaporation",
        "water_table_at_node",
        "water_table_below_domain",
        "water_table_surface",
        "refsoc_zero",
        "refsoc_partial_freeze",
        "refsoc_saturated_warm",
        "tide_rising",
        "tide_falling",
        "tide_nonpositive",
        "tide_zero_change",
        "peat_rain",
        "peat_tide_rising",
        "peat_tide_falling",
        "peat_freeze",
        "mineral_checked_flux",
        "refsoc_mixed_freeze",
        "refsoc_full_organic_warm",
        "mineral_checked_flux_all_tiles",
        "mineral_checked_flux_coherent_zero_et",
    ]
    cases = []
    zz = np.array([0.0, 2.0, 6.0, 14.0, 30.0, 62.0, 126.0, 254.0, 510.0, 1022.0, 2000.0])
    dz = np.r_[2.0, np.diff(zz)]
    dh = np.r_[dz[1] / 2, (dz[1:-1] + dz[2:]) / 2, dz[-1] / 2]
    for index, name in enumerate(names):
        peat = name.startswith("peat_")
        tide = "tide_" in name
        freeze = "freeze" in name
        veget_max = np.zeros((1, 14))
        veget_max[0, [0, 13]] = [0.2, 0.8]
        veget = veget_max.copy()
        veget[0, 13] = 0.65
        soiltile = np.zeros((1, 6))
        soiltile[0, [0, 3]] = [0.2, 0.8]
        mc = np.zeros((1, 11, 6))
        mc[0, :, 0] = np.linspace(0.19, 0.31, 11)
        mc[0, :, 3] = np.linspace(0.46, 0.62, 11) if peat else np.linspace(0.20, 0.32, 11)
        qs = np.zeros((1, 14))
        qs[0, 13] = 0.08
        qmax = np.zeros((1, 14))
        qmax[0, 13] = 0.4
        wet = np.zeros((1, 14))
        wet[0, 13] = 0.015
        transpir = np.zeros((1, 14))
        transpir[0, 13] = 0.025
        evap_pft = np.zeros((1, 14))
        evap_pft[0, 0] = 0.012
        pref = np.ones(14, dtype=np.int32)
        pref[13] = 4
        through = np.full(14, 0.25)
        through[0] = 1.0
        if name in ("mineral_checked_flux_all_tiles", "mineral_checked_flux_coherent_zero_et"):
            veget_max[:] = 0.0
            veget_max[0, [0, 1, 2, 3, 4, 13]] = [0.2, 0.1, 0.1, 0.1, 0.1, 0.4]
            veget = veget_max.copy()
            veget[0, 13] = 0.32
            soiltile[0, :] = [0.2, 0.1, 0.1, 0.4, 0.1, 0.1]
            pref[[1, 2, 3, 4]] = [2, 3, 5, 6]
            for tile in range(6):
                mc[0, :, tile] = np.linspace(0.19 + 0.001 * tile, 0.31 + 0.001 * tile, 11)
        if name == "mineral_checked_flux_coherent_zero_et":
            evap_pft[:] = 0.0
            transpir[:] = 0.0
        temperature = np.full((1, 11), 280.0)
        if freeze:
            temperature[:] = 272.9 if peat else 272.6
        rain = 0.45
        if name == "mineral_canopy_overflow":
            rain = 3.2
            qs[0, 13] = 0.35
        if name == "mineral_evaporation":
            rain = 0.0
            evap_pft[0, 0] = 0.045
        water_table = np.full((1, 6), 1e20)
        if name == "water_table_at_node":
            water_table[:, 3] = 0.126
        if name == "water_table_below_domain":
            water_table[:, 3] = 2.001
        if name == "water_table_surface":
            water_table[:, 3] = 0.0
        refsoc = 0.0 if name == "refsoc_zero" else (35.0 if name == "refsoc_partial_freeze" else 125.0)
        if name == "refsoc_mixed_freeze":
            refsoc = 45500.0
        if name == "refsoc_full_organic_warm":
            refsoc = 162500.0
        tide_height, tide_delta = (0.24, 0.035)
        if "falling" in name:
            tide_delta = -0.045
        if name == "tide_nonpositive":
            tide_height, tide_delta = -0.02, -0.01
        if name == "tide_zero_change":
            tide_height, tide_delta = 0.0, 0.0
        settings = dict(
            case_id=index + 101,
            name=name,
            peat_hydro=peat,
            tides=tide,
            ok_freeze_cwrr=freeze,
            check_cwrr=name.startswith("mineral_checked_flux"),
            use_refSOC_hydrol=name.startswith("refsoc_"),
            refSOC_1d=np.array([refsoc]),
            njsc=np.array([3], dtype=np.int32),
            zz_mm=zz.copy(),
            dz_mm=dz.copy(),
            dh_mm=dh.copy(),
            veget=veget,
            veget_max=veget_max,
            soiltile=soiltile,
            pref_soil_veg=pref,
            humcste=np.ones(14),
            altmax=np.full((1, 14), 10.0),
            mc=mc,
            mcl=mc.copy(),
            water2infilt=np.zeros((1, 6)),
            vegettot_old=np.ones(1),
            drysoil_frac=np.zeros(1),
            precip_rain=np.array([rain]),
            qsintveg=qs,
            qsintmax=qmax,
            vevapwet=wet,
            transpir=transpir,
            vevapnu_pft=evap_pft,
            vevapnu=np.array([evap_pft.sum()]),
            throughfall_by_pft=through,
            temp_hydro=temperature,
            zwt_force=water_table,
            wtp_tide=tide_height,
            dwtp_tide=tide_delta,
            wt_ab=np.array([0.11 if peat else 0.0]),
            wt_ab_tide=np.array([0.07 if tide else 0.0]),
            run2peat=np.array([0.03 if peat else 0.0]),
            run2man=np.zeros(1),
            free_drain_coef=np.array([[1.0, 1.0, 1.0, 0.0 if peat else 1.0, 1.0, 1.0]]),
            soil_parameters={key: np.asarray(value) for key, value in h.hydrol_soil_parameters().items()},
        )
        settings["van_genuchten"] = {
            name: np.asarray(getattr(h, constant))
            for name, constant in (
                ("VG_n", "USDA_NVAN"),
                ("VG_alpha", "USDA_AVAN"),
                ("VG_m", "USDA_VG_M"),
                ("VG_psi_fc", "USDA_VG_PSI_FC"),
                ("VG_psi_wp", "USDA_VG_PSI_WP"),
            )
        }
        cases.append(
            dict(
                name=name,
                inputs=settings,
                domain="explicit private peat SAVE extension; natural producer unresolved"
                if peat
                else "PFT14 finite mineral component tuple",
                phase="var_init -> first canopy/soil consumer -> diagnostics -> THERMOSOIL moisture",
            )
        )
    return cases


def load_fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def run_production(inputs, mode="eager"):
    """Return the actual HydrolFirstStepModuleClosure, including 11-layer exports."""
    if mode not in ("eager", "jit"):
        raise ValueError("mode must be eager or jit")
    v = dict(inputs)
    v.setdefault("check_cwrr", False)
    for key, value in v.items():
        if isinstance(value, list):
            v[key] = np.asarray(value)
    soil = h.hydrol_soil_parameters(v["soil_parameters"])
    initial = h.hydrol_var_init(
        veget=v["veget"],
        veget_max=v["veget_max"],
        soiltile=v["soiltile"],
        njsc=v["njsc"],
        znh_m=v["zz_mm"] / 1000,
        dnh_m=v["dz_mm"] / 1000,
        dlh_m=v["dh_mm"] / 1000,
        humcste=v["humcste"],
        altmax=v["altmax"],
        mc=v["mc"],
        mcl=v["mcl"],
        water2infilt=v["water2infilt"],
        vegtot_old=v["vegettot_old"],
        drysoil_frac=v["drysoil_frac"],
        ok_pc=False,
        ok_leak=False,
        peat_hydro=v["peat_hydro"],
        tides=v["tides"],
        agri_peat=False,
        ok_freeze_cwrr=v["ok_freeze_cwrr"],
        use_refSOC_hydrol=v["use_refSOC_hydrol"],
        refSOC_1d=v["refSOC_1d"],
        soil_parameters=soil,
    )
    profile = np.zeros_like(v["mc"])
    if v["ok_freeze_cwrr"]:
        profile = h.hydrol_soil_froz_profile(
            temp_hydro=v["temp_hydro"],
            mc=initial.mc,
            njsc=v["njsc"],
            dh_mm=v["dh_mm"],
            nvan=soil.nvan,
            avan=soil.avan,
            mcr=soil.mcr,
            mcs=soil.mcs_mineral,
            mcs_point=initial.mcs,
            mcr_point=soil.mcr[np.asarray(v["njsc"]) - 1],
            peat_hydro=v["peat_hydro"],
        )
    payload = {
        key: v[key]
        for key in (
            "precip_rain",
            "vevapwet",
            "veget",
            "veget_max",
            "qsintmax",
            "qsintveg",
            "throughfall_by_pft",
            "pref_soil_veg",
            "soiltile",
            "vevapnu",
            "vevapnu_pft",
            "transpir",
            "water2infilt",
            "temp_hydro",
            "zwt_force",
            "wt_ab",
            "wt_ab_tide",
            "wtp_tide",
            "dwtp_tide",
            "check_cwrr",
        )
    }
    payload.update(
        tot_melt=np.zeros(1),
        vegtot=np.ones(1),
        vevapflo=np.zeros(1),
        flood_frac=np.zeros(1),
        flood_res=np.zeros(1),
        subsinksoil=np.zeros(1),
        humrel=np.full((1, 14), 0.5),
        humrelv=np.zeros((1, 14, 6)),
        us=np.zeros((1, 14, 6, 11)),
        ae_ns=np.zeros((1, 6)),
        evap_bare_lim=np.array([0.5]),
        evap_bare_lim_ns=np.zeros((1, 6)),
        evapot=np.array([0.2]),
        evapot_corr=np.array([0.2]),
        tot_bare_soil=np.array([0.2]),
        reinfiltration_soil=np.zeros((1, 6)),
        irrigation_soil=np.zeros((1, 6)),
        mc=initial.mc,
        mcl=initial.mcl,
        profil_froz=profile,
        mcr=soil.mcr[np.asarray(v["njsc"]) - 1],
        mcs=initial.mcs,
        mcf=initial.mcf,
        mcw=initial.mcw,
        dz_mm=v["dz_mm"],
        dh_mm=v["dh_mm"],
        dt_days=1 / 48,
        free_drain_coef=v["free_drain_coef"],
        resolv=np.asarray(v["soiltile"]) > 0,
        kfact_root=np.ones((1, 11, 6)),
        ks=soil.ks,
        kfact=initial.mineral_tables.kfact,
        tmc=initial.tmc,
        njsc=v["njsc"],
        mineral_tables=initial.mineral_tables,
        peat_tables=initial.peat_tables,
        zz_mm=v["zz_mm"],
        zmaxh_m=2.0,
        precip_snow=np.zeros(1),
        vevapsno=np.zeros(1),
        snow=np.zeros(1),
        snowdz=np.zeros((1, 3)),
        snowrho=np.full((1, 3), 100.0),
        snowtemp=np.full((1, 3), 273.15),
        snow_age=np.zeros(1),
        snow_nobio=np.zeros((1, 1)),
        snow_nobio_age=np.zeros((1, 1)),
        frac_nobio=np.zeros((1, 1)),
        totfrac_nobio=np.zeros(1),
    )
    signature = inspect.signature(h.hydrol_module_explicit_step).parameters
    for name in h.HYDROL_MODULE_EXPLICIT_STEP_FIELDS:
        if name not in payload and name in signature and signature[name].default is not inspect.Parameter.empty:
            payload[name] = signature[name].default
    precall = h.HydrolFirstStepPrecallAssembly(
        payload, (), None, (), (), (), (), (), (), ("PB03 declared finite component",)
    )
    return h.run_hydrol_first_step_module_from_precall(
        precall,
        humcste=v["humcste"],
        nroot_state=initial.nroot,
        dz_mm=v["dz_mm"],
        dh_mm=v["dh_mm"],
        zz_mm=v["zz_mm"],
        ks=soil.ks,
        reinf_slope=np.array([0.17]),
        zmaxh_m=2.0,
        peat_hydro=v["peat_hydro"],
        ok_freeze_cwrr=v["ok_freeze_cwrr"],
        ok_pc=False,
        ok_leak=False,
        tides=v["tides"],
        ok_ru2peat=v["peat_hydro"],
        ok_wt_ab=v["peat_hydro"],
        run2peat=v["run2peat"],
        run2man=v["run2man"],
        dyn_nroot_larix=True,
        is_tree=np.array([False] * 13 + [True]),
        znt=(v["zz_mm"] + 0.5 * v["dh_mm"]) / 1000,
        runtime_static_tables=h.HydrolRuntimeStaticTables(initial.mineral_tables, initial.peat_tables, soil),
        use_jit=mode == "jit",
    )


def flatten_result(result):
    values = {}
    for group, names in (
        (
            "soil",
            (
                "mc",
                "mcl",
                "profil_froz_hydro_ns",
                "tmc",
                "tmc_soil",
                "qflux",
                "ru_ns",
                "dr_ns",
                "water2infilt",
                "wtd_ns",
                "runoff2peat",
                "run2peat",
                "run2man",
                "wt_ab",
                "wt_ab_tide",
            ),
        ),
        (
            "diagnostics",
            (
                "humrel",
                "vegstress",
                "nroot",
                "us",
                "humrelv",
                "vegstressv",
                "soilmoist",
                "soilmoist_liquid",
                "mc_layh",
                "mcl_layh",
                "mc_layh_s",
                "mcl_layh_s",
                "shumdiag",
                "shumdiag_perma",
                "drysoil_frac",
                "evap_bare_lim",
                "evap_bare_lim_ns",
                "runoff",
                "drainage",
                "humtot",
            ),
        ),
        ("canop", ("qsintveg", "precisol", "precip2canopy", "precip2ground", "canopy2ground")),
    ):
        obj = getattr(result.module, group) if group in ("soil", "canop") else result.diagnostics
        for name in names:
            array = np.asarray(getattr(obj, name))
            if name == "canopy2ground":
                array = array[:, 1:]
            values[f"{group}.{name}"] = array
    for name, value in result.thermosoil_moisture._asdict().items():
        values[f"moisture.{name}"] = np.asarray(value)
    for name in ("tmci", "tmcf", "dr_before_corr", "dr_after_corr", "check_tr_ns"):
        values[f"arithmetic.{name}"] = np.stack(
            [np.asarray(getattr(tile.solve, name)) for tile in result.module.soil.tile_results], axis=1
        )
    return values


def _literal(value):
    if isinstance(value, (bool, np.bool_)):
        return ".true." if value else ".false."
    if isinstance(value, (int, np.integer)):
        return str(value)
    return f"{float(value):.17e}_r_std"


def _assign(name, value):
    array = np.asarray(value)
    if array.ndim == 0:
        return f"    {name}={_literal(array.item())}"
    values = ", &\n      & ".join(_literal(x) for x in array.ravel(order="F"))
    shape = ",".join(str(x) for x in array.shape)
    rhs = f"(/{values}/)" if array.ndim == 1 else f"reshape((/{values}/),(/{shape}/))"
    return f"    {name}={rhs}"


def source_dependency_identity(relative, spans):
    path = ROOT / relative
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    return dict(
        source=relative,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        spans=[
            dict(
                start_line=first,
                end_line=last,
                span_sha256=hashlib.sha256(b"".join(lines[first - 1 : last])).hexdigest(),
            )
            for first, last in spans
        ],
    )


def compose_reference(path):
    """Reuse original scientific procedures verbatim; change only finite driver."""
    from scripts.dev import oracle_lane_hydrol_soil_owner as old
    from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes

    procedures = (*old.PROCEDURES, "hydrol_canop")
    spans = {name: extract_procedure_bytes(SOURCE, name) for name in procedures}
    raw = SOURCE.read_bytes()
    vegspan = extract_procedure_bytes(SOURCE, "hydrol_vegupd")
    start = vegspan.span_bytes.index(b"    mask_veget(:,:) = 0")
    stop = vegspan.span_bytes.index(b"    IF (printlev>=3)", start)
    vegblock = vegspan.span_bytes[start:stop]
    soil_bytes = spans["hydrol_soil"].span_bytes
    water_start = soil_bytes.index(b"       !! 3.5 Diagnosing")
    water_stop = soil_bytes.index(b"       !! 3.6 Diagnose", water_start)
    water_block = soil_bytes[water_start:water_stop]
    water_hook = b"       snapshot_wtd_ns(:,jst) = wtd_ns(:,jst) ! PB03 read-only phase snapshot\n"
    instrumented_soil = soil_bytes[:water_stop] + water_hook + soil_bytes[water_stop:]
    arithmetic_marker = b"       !! For water conservation check during redistribution\n       IF (check_cwrr2)"
    arithmetic_hook = b"       snapshot_tmci(:,jst)=tmci; snapshot_tmcf(:,jst)=tmcf\n       snapshot_dr_after(:,jst)=dr_ns(:,jst); snapshot_check_tr(:,jst)=check_tr_ns(:,jst)\n"
    before_marker = b"       !! 2.9 For water conservation check during redistribution AND CORRECTION,"
    before_hook = b"       snapshot_dr_before(:,jst)=dr_ns(:,jst)\n"
    assert instrumented_soil.count(arithmetic_marker) == 1
    assert instrumented_soil.count(before_marker) == 1
    instrumented_soil = instrumented_soil.replace(arithmetic_marker, arithmetic_hook + arithmetic_marker, 1).replace(
        before_marker, before_hook + before_marker, 1
    )
    assert (
        instrumented_soil.replace(water_hook, b"", 1).replace(arithmetic_hook, b"", 1).replace(before_hook, b"", 1)
        == soil_bytes
    )
    sechiba = ROOT / "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90"
    coupling_lines = sechiba.read_bytes().splitlines(keepends=True)
    coupling_start = next(i for i, line in enumerate(coupling_lines) if b"mc_layh_pft(:,:,jv) = mc_layh_s" in line) - 1
    coupling_end = next(i for i in range(coupling_start, len(coupling_lines)) if b"ENDDO" in coupling_lines[i]) + 1
    coupling = b"".join(coupling_lines[coupling_start:coupling_end])
    declaration = old._owner_argument_declarations()
    drivers = [
        f"""  subroutine run_owner_case(case_id,unit_out)
    integer(i_std),intent(in)::case_id,unit_out
    integer(i_std),parameter::kjpindex=1
{declaration}
    integer::jsl,jst,jv,ji
    real(r_std)::veget(1,nvm),mx_eau_var(1),humcste_use(1,nvm),altmax(1,nvm)
    real(r_std)::qsintveg(1,nvm),qsintmax(1,nvm),precip_rain(1),vevapwet(1,nvm)
    real(r_std)::precip2canopy(1,nvm),precip2ground(1,nvm),canopy2ground(1,nvm),frac_bare(1,nvm)
    real(r_std)::mc_layh_pft(1,nslm,nvm),mcl_layh_pft(1,nslm,nvm),soilmoist_pft(1,nslm,nvm)
    call reset_state()
    snapshot_wtd_ns=undef_sechiba
    froz_frac_corr=1._r_std; smtot_corr=2._r_std; max_froz_hydro=1._r_std
    kfact_root=1._r_std; vegtot=1._r_std; vegtot_old=1._r_std
    ok_pc=.false.;ok_leak=.false.;ok_explicitsnow=.true.;ok_LAIdev=.false.
    ok_thermodynamical_freezing=.true.;dyn_nroot_larix=.true.;new_watstress=.false.
    topmodel_new=.false.;do_rsoil=.false.;doponds=.false.;perma_peat=.false.;agri_peat=.false.
    check_cwrr=.false.;check_cwrr2=.false.;liqlayers=nslm-1;numlayers=nslm
    is_tree=.false.;is_tree(14)=.true.;is_crop_soil=.false.
    evapot=.2_r_std;evapot_penm=.2_r_std;evap_bare_lim=.5_r_std;tot_bare_soil=.2_r_std
    humrel=.5_r_std;drysoil_frac=0._r_std;returnflow=0._r_std;reinfiltration=0._r_std
    irrigation=0._r_std;irrig_demand_ratio=0._r_std;tot_melt=0._r_std
    snow=0._r_std;snowdz=0._r_std;fsat=0._r_std;u=2._r_std;v=1._r_std;tq_cdrag=.01_r_std
    reinf_slope=.17_r_std;humcste=1._r_std;altmax=10._r_std
    select case(case_id)
"""
    ]
    mappings = {
        "veget": "veget",
        "veget_max": "veget_max",
        "soiltile": "soiltile",
        "pref_soil_veg": "pref_soil_veg",
        "mc": "mc",
        "mcl": "mcl",
        "water2infilt": "water2infilt",
        "njsc": "njsc",
        "precip_rain": "precip_rain",
        "qsintveg": "qsintveg",
        "qsintmax": "qsintmax",
        "vevapwet": "vevapwet",
        "transpir": "transpir",
        "vevapnu_pft": "vevapnu_pft",
        "vevapnu": "vevapnu",
        "throughfall_by_pft": "throughfall_by_pft",
        "temp_hydro": "stempdiag",
        "zwt_force": "zwt_force",
        "wt_ab": "wt_ab",
        "wt_ab_tide": "wt_ab_tide",
        "run2peat": "run2peat",
        "run2man": "run2man",
        "free_drain_coef": "free_drain_coef",
        "refSOC_1d": "refSOC_1d",
        "use_refSOC_hydrol": "use_refSOC_hydrol",
        "peat_hydro": "peat_hydro",
        "tides": "tides",
        "ok_freeze_cwrr": "ok_freeze_cwrr",
        "check_cwrr": "check_cwrr",
    }
    for case in make_cases():
        v = case["inputs"]
        drivers.append(f"    case({v['case_id']})\n")
        assignments = {name: v[key] for key, name in mappings.items()}
        assignments.update(v["soil_parameters"])
        assignments.update(v["van_genuchten"])
        assignments.update(
            znh=v["zz_mm"] / 1000,
            dnh=v["dz_mm"] / 1000,
            dlh=v["dh_mm"] / 1000,
            diaglev=(v["zz_mm"] + 0.5 * v["dh_mm"]) / 1000,
            znt=(v["zz_mm"] + 0.5 * v["dh_mm"]) / 1000,
            ok_ru2peat=v["peat_hydro"],
            ok_wt_ab=v["peat_hydro"],
        )
        drivers.extend(_assign(name, value) + "\n" for name, value in assignments.items())
        drivers.append(f"    rewind(103);write(103,'(F10.3)'){_literal(v['wtp_tide'])};rewind(103)\n")
        drivers.append(f"    rewind(104);write(104,'(F10.3)'){_literal(v['dwtp_tide'])};rewind(104)\n")
    drivers.append("    end select\n")
    drivers.append("""    call hydrol_var_init(kjpindex,veget,veget_max,soiltile,njsc,mx_eau_var, &
      shumdiag_perma,drysoil_frac,qsintveg,mc_layh,mcl_layh,mc_layh_s,mcl_layh_s, &
      tmc_topgrass,humcste_use,altmax,shumdiag_peat,shumdiag_croppeat,shumdiag_man)
""")
    # var_init resets qsintveg(:,bare), while the PFT14 restart is retained.
    drivers.append(vegblock.decode("latin1"))
    drivers.append("""    canopy2ground=0._r_std
    call hydrol_canop(kjpindex,precip_rain,vevapwet,veget_max,veget,qsintmax, &
      qsintveg,precisol,tot_melt,precip2canopy,precip2ground,canopy2ground)
""")
    call = ", &\n      & ".join(old.OWNER_ARGUMENTS)
    drivers.append(f"    call hydrol_soil({call})\n")
    drivers.append(coupling.decode("latin1"))
    fields = {
        "soil.mc": "mc",
        "soil.mcl": "mcl",
        "soil.profil_froz_hydro_ns": "profil_froz_hydro_ns",
        "soil.tmc": "tmc",
        "soil.qflux": "wat_flux",
        "soil.ru_ns": "ru_ns",
        "soil.dr_ns": "dr_ns",
        "soil.water2infilt": "water2infilt",
        "soil.wtd_ns": "snapshot_wtd_ns",
        "soil.runoff2peat": "runoff2peat",
        "soil.run2peat": "run2peat",
        "soil.run2man": "run2man",
        "soil.wt_ab": "wt_ab",
        "soil.wt_ab_tide": "wt_ab_tide",
        "diagnostics.humrel": "humrel",
        "diagnostics.vegstress": "vegstress",
        "diagnostics.nroot": "nroot",
        "diagnostics.us": "us",
        "diagnostics.humrelv": "humrelv",
        "diagnostics.vegstressv": "vegstressv",
        "diagnostics.soilmoist": "soilmoist",
        "diagnostics.soilmoist_liquid": "soilmoist_liquid",
        "diagnostics.mc_layh": "mc_layh",
        "diagnostics.mcl_layh": "mcl_layh",
        "diagnostics.mc_layh_s": "mc_layh_s",
        "diagnostics.mcl_layh_s": "mcl_layh_s",
        "diagnostics.shumdiag": "shumdiag",
        "diagnostics.shumdiag_perma": "shumdiag_perma",
        "diagnostics.drysoil_frac": "drysoil_frac",
        "diagnostics.evap_bare_lim": "evap_bare_lim",
        "diagnostics.evap_bare_lim_ns": "evap_bare_lim_ns",
        "diagnostics.runoff": "runoff",
        "diagnostics.drainage": "drainage",
        "canop.qsintveg": "qsintveg",
        "canop.precisol": "precisol",
        "canop.precip2canopy": "precip2canopy",
        "canop.precip2ground": "precip2ground",
        "canop.canopy2ground": "canopy2ground(:,2:nvm)",
        "moisture.shumdiag_perma": "shumdiag_perma",
        "moisture.mc_layh": "mc_layh",
        "moisture.mcl_layh": "mcl_layh",
        "moisture.tmc_layh": "soilmoist",
        "moisture.mc_layh_pft": "mc_layh_pft",
        "moisture.mcl_layh_pft": "mcl_layh_pft",
        "moisture.tmc_layh_pft": "soilmoist_pft",
    }
    fields.update(
        {
            f"arithmetic.{name}": f"snapshot_{source}"
            for name, source in (
                ("tmci", "tmci"),
                ("tmcf", "tmcf"),
                ("dr_before_corr", "dr_before"),
                ("dr_after_corr", "dr_after"),
                ("check_tr_ns", "check_tr"),
            )
        }
    )
    three = {
        "mc",
        "mcl",
        "profil_froz_hydro_ns",
        "wat_flux",
        "nroot",
        "humrelv",
        "vegstressv",
        "mc_layh_s",
        "mcl_layh_s",
        "mc_layh_pft",
        "mcl_layh_pft",
        "soilmoist_pft",
    }
    one = {"run2peat", "run2man", "wt_ab", "wt_ab_tide", "drysoil_frac", "evap_bare_lim", "runoff", "drainage"}
    for label, name in fields.items():
        rank = 4 if name == "us" else (3 if name in three else (1 if name in one else 2))
        drivers.append(f"    call emit{rank}(unit_out,case_id,'{label}',{name})\n")
    drivers.append("  end subroutine run_owner_case\n")
    helpers = old._case_driver_source().decode("latin1").split("  subroutine emit1", 1)[1]
    drivers.append("  subroutine emit1" + helpers)
    drivers.append("""
  subroutine emit4(unit_out,case_id,field,x)
    integer,intent(in)::unit_out,case_id
    character(*),intent(in)::field
    real(r_std),intent(in)::x(:,:,:,:)
    integer::i,j,k,l,n
    n=0
    do i=1,size(x,1);do j=1,size(x,2);do k=1,size(x,3);do l=1,size(x,4)
      write(unit_out,'(I0,A,A,A,I0,A,ES24.16E3)')case_id,',',trim(field),',',n,',',x(i,j,k,l);n=n+1
    enddo;enddo;enddo;enddo
  end subroutine emit4
""")
    template = old.TEMPLATE.read_bytes().decode("latin1").replace("nslm=7", "nslm=11")
    # These declarations correct legacy harness-only values to source constants.
    template = template.replace(
        "mc_awet_peat=.5693_r_std,mc_adry_peat=.15_r_std", "mc_awet_peat=.25_r_std,mc_adry_peat=.1_r_std"
    )
    template = template.replace("soilc_max=100._r_std", "soilc_max=130000._r_std")
    template = re.sub(r"(?m)^( real\(r_std\)::(?:diaglev|znt|znh|dnh|dlh)\(nslm\))=.*$", r"\1=0._r_std", template)
    snapshot_declarations = "real(r_std)::throughfall_by_pft(nvm),snapshot_wtd_ns(1,nstm)\nreal(r_std)::snapshot_tmci(1,nstm),snapshot_tmcf(1,nstm),snapshot_dr_before(1,nstm),snapshot_dr_after(1,nstm),snapshot_check_tr(1,nstm)\n"
    template = template.replace("! <MODULE_STATE>", snapshot_declarations + old._module_state().decode("latin1"))
    template = template.replace("! <STATE_SETUP>", old._allocation_and_reset_source().decode("latin1"))
    template = template.replace("! <CASE_DRIVER>", "".join(drivers))
    template = template.replace(
        "! <HYDROL_PROCEDURES>",
        "\n\n".join(
            (instrumented_soil if name == "hydrol_soil" else span.span_bytes).decode("latin1")
            for name, span in spans.items()
        ),
    )
    template = template.replace("do case_id=1,19", f"do case_id=101,{100 + len(make_cases())}")
    template = template.replace("integer::case_id,unit_out", "integer::case_id,unit_out\n character(32)::case_argument")
    template = re.sub(
        r" do case_id=101,\d+\r?\n   call run_owner_case\(case_id,unit_out\)\r?\n enddo",
        " call get_command_argument(1,case_argument)\n read(case_argument,*)case_id\n call run_owner_case(case_id,unit_out)",
        template,
    )
    path.write_bytes(template.encode("latin1"))
    restored = (
        path.read_bytes().replace(water_hook, b"", 1).replace(arithmetic_hook, b"", 1).replace(before_hook, b"", 1)
    )
    assert all(span.span_bytes in restored for span in spans.values())
    return dict(
        source_sha256=hashlib.sha256(raw).hexdigest(),
        procedures={
            name: dict(start_line=span.start_line, end_line=span.end_line, span_sha256=span.span_sha256)
            for name, span in spans.items()
        },
        vegupd_block_sha256=hashlib.sha256(vegblock).hexdigest(),
        local_water_table_diagnostic_block_sha256=hashlib.sha256(water_block).hexdigest(),
        water_table_snapshot=dict(
            hook=water_hook.decode("latin1").strip(),
            source_after_line=raw[: raw.index(soil_bytes) + water_stop].count(b"\n"),
            stripped_body_sha256=hashlib.sha256(
                instrumented_soil.replace(water_hook, b"", 1)
                .replace(arithmetic_hook, b"", 1)
                .replace(before_hook, b"", 1)
            ).hexdigest(),
            original_span_sha256=spans["hydrol_soil"].span_sha256,
            phase="after hydrol_soil section 3.5, before 3.6 under-residual smoothing",
        ),
        arithmetic_snapshots=dict(
            hooks=[before_hook.decode("latin1").strip(), arithmetic_hook.decode("latin1").strip()],
            phase="before section 2.9 and after conservative drainage correction, before optional recheck",
        ),
        harness_constant_sources=dict(
            **source_dependency_identity(
                "fortran_source/ORCHIDEE/src_parameters/constantes_soil_var.f90", ((103, 119), (160, 172))
            ),
            soilc_max=dict(value=130000.0, line=119),
            mc_awet_peat=dict(value=0.25, line=170),
            mc_adry_peat=dict(value=0.1, line=171),
        ),
        check_cwrr_control=source_dependency_identity(
            "fortran_source/ORCHIDEE/src_parameters/constantes_soil.f90", ((463, 473),)
        ),
        coupling=dict(
            source=sechiba.relative_to(ROOT).as_posix(),
            start_line=coupling_start + 1,
            end_line=coupling_end,
            source_sha256=hashlib.sha256(sechiba.read_bytes()).hexdigest(),
            span_sha256=hashlib.sha256(coupling).hexdigest(),
        ),
    )


def build_reference(output_dir=OUTPUT):
    from scripts.dev.fortran_oracle_common import DEFAULT_COMPILER, compile_fortran, compiler_environment
    from scripts.dev.oracle_lane_hydrol_soil_owner import _read_fortran

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pb03_hydrol_") as temporary:
        directory = Path(temporary)
        source = directory / "reference.f90"
        identity = compose_reference(source)
        shutil.copyfile(source, output_dir / "reference.f90")
        try:
            metadata = compile_fortran(source, directory / "reference.exe")
        except subprocess.CalledProcessError as error:
            (output_dir / "reference_compile_failed.log").write_text(error.stderr or "")
            raise RuntimeError(
                "\n".join(line for line in (error.stderr or "").splitlines() if "Error:" in line)
            ) from error
        runs = []
        expected = {}
        csv_lines = ["case_id,field,flat_index,value\n"]
        for case in make_cases():
            case_id = case["inputs"]["case_id"]
            process = subprocess.run(
                [str(directory / "reference.exe"), str(case_id)],
                cwd=directory,
                env=compiler_environment(DEFAULT_COMPILER),
                capture_output=True,
                text=True,
            )
            runs.append(
                dict(
                    case_id=case_id,
                    name=case["name"],
                    run_exit=process.returncode,
                    stdout=process.stdout,
                    stderr=process.stderr,
                )
            )
            shutil.copyfile(directory / "fortran_outputs.txt", output_dir / f"reference-case{case_id}.csv")
            if process.returncode:
                continue
            expected.update(_read_fortran(directory / "fortran_outputs.txt"))
            csv_lines.extend((directory / "fortran_outputs.txt").read_text().splitlines(keepends=True)[1:])
        metadata.update(
            runs=runs,
            identity=identity,
            source_failed_cases=[run["case_id"] for run in runs if run["run_exit"]],
            reference_cases=len(expected),
        )
        (output_dir / "reference_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (output_dir / "fortran_outputs.csv").write_text("".join(csv_lines))
    return expected


def write_fixture(output_dir=OUTPUT):
    from scripts.dev.oracle_lane_hydrol_soil_owner import _read_fortran

    expected = _read_fortran(output_dir / "fortran_outputs.csv")
    metadata = json.loads((output_dir / "reference_metadata.json").read_text())
    fixture = dict(
        schema="physics-branch-hydrol-reference-v1",
        rtol=RTOL,
        atol=ATOL,
        cases=make_cases(),
        expected={
            str(case_id): {name: value.tolist() for name, value in fields.items()}
            for case_id, fields in expected.items()
        },
        reference_identity=metadata["identity"],
        source_failed_cases=metadata["source_failed_cases"],
        compiler={
            name: metadata[name] for name in ("compiler", "compiler_version", "compile_flags", "compile_unit_sha256")
        },
        dimensions=dict(npts=1, nvm=14, nstm=6, nslm=11),
        fixed_contract=dict(
            dt_s=1800.0,
            dt_days=1 / 48,
            zmaxh_m=2.0,
            reinf_slope=0.17,
            fr_dt=2.0,
            froz_frac_corr=1.0,
            smtot_corr=2.0,
            max_froz_hydro=1.0,
            ok_pc=False,
            ok_leak=False,
            agri_peat=False,
            doponds=False,
            new_watstress=False,
            dyn_nroot_larix=True,
            topmodel_wrapper=False,
            evapot=0.2,
            evapot_corr=0.2,
            initial_evap_bare_lim=0.5,
            tot_bare_soil=0.2,
            routing_returnflow=0.0,
            routing_reinfiltration=0.0,
            irrigation=0.0,
            tot_melt=0.0,
            flood_res=0.0,
            flood_frac=0.0,
            vevapflo=0.0,
            subsinksoil=0.0,
            kfact_root=1.0,
            initial_humrel=0.5,
            initial_humrelv=0.0,
            initial_us=0.0,
            initial_ae_ns=0.0,
            initial_evap_bare_lim_ns=0.0,
            zero_snow_state=True,
            compiler_nbint=50,
        ),
        limitations=[
            "Explicit private peat SAVE=true is a bounded component extension; original natural effective producer unresolved.",
            "No whole hydrol_main/model timestep; no exhaustive branch denominator claim.",
            "canopy2ground bare-PFT coordinate is source undefined and excluded.",
            "soil.tmc_soil and diagnostic humtot are not separately returned original-Fortran fields in this fixture.",
        ],
    )
    FIXTURE.write_text(json.dumps(fixture, indent=2, default=lambda value: value.tolist()) + "\n", encoding="utf-8")
    return fixture


def verify_cases(mode, output_dir=OUTPUT, selected=None):
    from scripts.dev.fortran_oracle_common import float_comparison

    fixture = load_fixture()
    output_dir.mkdir(parents=True, exist_ok=True)
    assert jax.default_backend() == "cpu" and jax.config.jax_enable_x64
    if mode == "jit":
        assert not jax.config.jax_disable_jit
    cases = []
    started = time.perf_counter()
    for case in fixture["cases"]:
        if selected and case["name"] not in selected:
            continue
        if case["inputs"]["case_id"] in fixture.get("source_failed_cases", []):
            continue
        before = time.perf_counter()
        result = run_production(case["inputs"], mode)
        if not result.ok:
            raise RuntimeError(result.missing_inputs)
        actual = flatten_result(result)
        expected = fixture["expected"][str(case["inputs"]["case_id"])]
        checks = [
            float_comparison(name, np.asarray(value), actual[name].ravel(order="C"), rtol=RTOL, atol=ATOL)
            for name, value in expected.items()
        ]
        np.savez_compressed(output_dir / f"{case['name']}-{mode}.npz", **actual)
        row = dict(
            name=case["name"],
            case_id=case["inputs"]["case_id"],
            mode=mode,
            checks=checks,
            elapsed_s=time.perf_counter() - before,
            domain=case["domain"],
        )
        cases.append(row)
        print(
            json.dumps(
                dict(
                    case=case["name"],
                    mode=mode,
                    elapsed_s=row["elapsed_s"],
                    passed=sum(check["passed"] for check in checks),
                    failed=[check for check in checks if not check["passed"]],
                )
            ),
            flush=True,
        )
    checks = [check for case in cases for check in case["checks"]]
    receipt = dict(
        mode=mode,
        backend=jax.default_backend(),
        x64=jax.config.jax_enable_x64,
        jax_disable_jit=jax.config.jax_disable_jit,
        cases=cases,
        elapsed_s=time.perf_counter() - started,
        count=len(checks),
        passed=sum(check["passed"] for check in checks),
        failed=sum(not check["passed"] for check in checks),
        scalar_elements=sum(check.get("count", 0) for check in checks),
        identities={
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), FIXTURE, Path(h.__file__))
        },
        command=sys.argv,
        numerical_acceptance=False,
    )
    (output_dir / f"{mode}_comparison.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="mineral_rain_canopy")
    parser.add_argument("--mode", choices=("eager", "jit"), default="eager")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--build-reference", action="store_true")
    parser.add_argument("--write-fixture", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--production-file", type=Path)
    args = parser.parse_args()
    if args.production_file:
        spec = importlib.util.spec_from_file_location("pb03_hydrol_before", args.production_file)
        h = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = h
        spec.loader.exec_module(h)
    if args.build_reference:
        expected = build_reference(args.output_dir)
        print(json.dumps(dict(reference_cases=len(expected))))
        if args.write_fixture:
            write_fixture(args.output_dir)
        raise SystemExit(0)
    if args.write_fixture:
        write_fixture(args.output_dir)
        raise SystemExit(0)
    if args.verify:
        receipt = verify_cases(args.mode, args.output_dir, selected=args.cases)
        print(json.dumps({key: receipt[key] for key in ("count", "passed", "failed", "scalar_elements", "elapsed_s")}))
        raise SystemExit(bool(receipt["failed"]))
    case = next(case for case in make_cases() if case["name"] == args.case)
    result = run_production(case["inputs"], args.mode)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / f"{args.case}-{args.mode}.npz", **flatten_result(result))
    print(
        json.dumps(
            dict(
                case=args.case,
                mode=args.mode,
                ok=result.ok,
                moisture_shape=list(result.thermosoil_moisture.mc_layh.shape),
            )
        )
    )
