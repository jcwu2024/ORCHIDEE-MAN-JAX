"""Original-source daily growth chain; no coupled model integration is run.

The numerical reference is the original Fortran procedures in source order.
Existing oracle templates supply declarations and inert history/IO stubs only.
Each template's modules are namespaced so their global parameter declarations
can coexist; the setup routines assign one consistent non-crop PFT14 tuple.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# The standalone script resolves its sibling oracle helpers after path setup.
from fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
)

PACKAGE = ROOT / "outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon/growth"
FIXTURE = ROOT / "tests/fixtures/phase05_growth_reference.json"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate"
PFT14 = 13
SPANS = {
    "prescribe": ("stomate_prescribe.f90", 90, 348),
    "constraints": ("lpj_constraints.f90", 99, 246),
    "phenology": ("stomate_phenology.f90", 173, 591),
    "alloc": ("stomate_alloc.f90", 146, 834),
    "npp_calc": ("stomate_npp.f90", 116, 725),
    "gap": ("lpj_gap.f90", 117, 364),
    "turn": ("stomate_turnover.f90", 169, 947),
    "vmax": ("stomate_vmax.f90", 105, 363),
    "setlai": ("stomate_lai.f90", 58, 88),
    "crop_bmalloc": ("../src_sticslai/crop_alloc.f90", 18, 384),
}


def encode(value):
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.ravel().tolist()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def decode(value):
    if isinstance(value, dict):
        if set(value) == {"dtype", "shape", "values"}:
            return np.asarray(value["values"], dtype=value["dtype"]).reshape(value["shape"])
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return tuple(value) if value and isinstance(value[0], str) else [decode(v) for v in value]
    return value


def chain_inputs(*, cold_start=False, agriculture=True):
    """Fully specified component-domain tuple, not an observed trajectory.

    Six cells reuse the old allocation/NPP/turnover boundary values and expose
    positive/zero leaf, empty/reserve-rich pools, senescence, and absent cover.
    All PFT14 scientific masks are tree/natural/noncrop, with peat=true explicitly
    admitted by constraints. Values are not reconstructed from endpoints.
    """
    n, v, p, leaf_ages = 6, 14, 12, 4

    def z():
        return np.zeros((n, v), dtype=np.float64)

    def full(x):
        return np.full((n, v), x, dtype=np.float64)

    def pv(x):
        return np.full(v, x, dtype=np.float64)

    natural = np.zeros(v, bool)
    natural[13] = True
    tree = natural.copy()
    peat = natural.copy()
    false = np.zeros(v, bool)
    bio = np.zeros((n, v, p, 1))
    bio[:, 13, 0, 0] = [10.0, 0.0, 1000.0, 100.0, 10.0, 0.0]
    bio[:, 13, 5, 0] = [5.0, 0.0, 5.0, 40.0, 5.0, 0.0]
    bio[:, 13, 7, 0] = [100.0, 0.0, 100.0, 100.0, 2000.0, 0.0]
    bio[3, 13, 1, 0] = 730.0
    bio[3, 13, 3, 0] = 10.0
    bio[3, 13, 6, 0] = 90.0
    la = np.zeros((n, v, leaf_ages))
    la[:5, 13] = [20.0, 30.0, 40.0, 50.0]
    la[3, 13] = [120.0, 130.0, 140.0, 150.0]
    lf = np.zeros_like(la)
    lf[:5, 13] = 0.25
    veg = z()
    veg[:5, 13] = 1.0
    present = veg > 0
    wg = full(5.0)
    wg[:, 13] = [5.0, 299.0, 301.0, 5.0, 5.0, 0.0]
    sen = np.zeros((n, v), bool)
    sen[3, 13] = True
    sla = full(0.02)
    gpp = z()
    gpp[:, 13] = [10.0, 0.0, 10.0, 0.0, 10.0, 0.0]
    rmp = np.zeros((n, v, p))
    rmp[:5, 13, 0] = [0.7, 0.7, 10.0, 0.7, 0.7]
    rmp[:5, 13, 5] = [0.3, 0.3, 0.0, 0.3, 0.3]
    tm = np.asarray([270.0, 285.0, 280.0, 285.0, 275.0, 276.0])
    bm_sapl = np.broadcast_to(np.arange(1.0, 13.0)[None, :, None], (v, p, 1)).copy()
    prescribe = dict(
        veget_max=veg,
        dt_days=1.0,
        pft_present=present,
        everywhere=present.astype(float),
        when_growthinit=wg,
        biomass=bio,
        leaf_frac=lf,
        ind=z(),
        cn_ind=z(),
        co2_to_bm=z(),
        natural=natural,
        pasture=false,
        is_tree=tree,
        bm_sapl=bm_sapl,
        maxdia=pv(0.5),
        pheno_is_none=np.ones(v, bool),
        ok_dgvm=False,
        lpj_gap_const_mort=True,
        firstcall=True,
        stomate_restart_none=bool(cold_start),
        bm_sapl_rescale=40.0,
        min_stomate=1e-8,
    )
    regenerate = full(0.5)
    regenerate[[0, 3], 13] = 1e-4
    constraints = dict(
        t2m_month=tm,
        t2m_min_daily=np.full(n, 260.0),
        adapted=full(0.25),
        regenerate=regenerate,
        tseason=np.asarray([285.0, 275.0, 281.0, 285.0, 280.15, 280.149]),
        natural=natural,
        is_tree=tree,
        is_peat=peat,
        pheno_is_none=np.ones(v, bool),
        tmin_crit=pv(-9999.0),
        tcm_crit=np.where(natural, 275.0, -9999.0),
        agriculture=bool(agriculture),
        undef=-9999.0,
    )
    phenology = dict(
        leaf_age=la,
        gdd_midwinter=full(20.0),
        pheno_model=tuple(["none"] * v),
        active_pft_mask=tuple(bool(x) for x in natural),
        dt_days=1.0,
        undef=-9999.0,
    )
    alloc = dict(
        lai=bio[:, :, 0, 0] * sla,
        senescence=sen,
        moiavail_week=full(0.7),
        tsoil_month=np.tile([283.15, 293.15], (n, 1)),
        soilhum_month=np.tile([0.4, 0.8], (n, 1)),
        age=z(),
        leaf_age=la,
        z_soil=np.asarray([0.0, 1.0, 2.0]),
        sla_calc=sla,
        natural=natural,
        pasture=false,
        is_tree=tree,
        ok_LAIdev=false,
        r0=pv(0.35),
        s0=pv(0.35),
        ext_coeff=pv(0.5),
        lai_max=pv(12.0),
        lai_max_to_happy=pv(0.5),
        tau_leafinit=pv(10.0),
        alloc_min=pv(0.2),
        alloc_max=pv(0.8),
        demi_alloc=pv(100.0),
        alloc_agr_st=np.where(natural, 0.25, 0.0),
        alloc_agr_pn=np.where(natural, 0.05, 0.0),
        when_growthinit_cut=full(100.0),
        is_grassland_manag=false,
    )
    post = dict(
        gpp_daily=gpp,
        resp_maint_part=rmp,
        frac_growthresp=pv(0.25),
        npp_longterm=full(20.0),
        turnover_longterm=np.zeros_like(bio),
        lm_lastyearmax=full(50.0),
        bm_to_litter=np.zeros_like(bio),
        senescence=sen,
        rip_time=full(1.0),
        height=full(1.0),
        t2m_min_daily=np.full(n, 260.0),
        tmin_spring_time=z(),
        herbivores=z(),
        maxmoiavail_lastyear=full(1.0),
        minmoiavail_lastyear=z(),
        moiavail_week=full(0.7),
        t2m_longterm=np.full(n, 293.15),
        t2m_month=tm,
        t2m_week=np.full(n, 293.15),
        veget_max=veg,
        gdd_from_growthinit=z(),
        age=z(),
        turnover_time=full(80.0),
        nrec=np.zeros((n, v), np.int32),
        sla_calc=sla,
        senescence_type=np.zeros(v, np.int32),
        is_tree=tree,
        natural=natural,
        pasture=false,
        is_grassland_manag=false,
        ok_laidev=false,
        availability_fact=pv(0.14),
        residence_time=pv(30.0),
        tmin_crit=pv(-9999.0),
        leaf_tab=np.zeros(v, np.int32),
        pheno_type=np.zeros(v, np.int32),
        maxdia=pv(0.5),
        min_leaf_age_for_senescence=pv(30.0),
        gdd_senescence=pv(100.0),
        senescence_temp=np.zeros((v, 3)),
        hum_frac=pv(0.5),
        senescence_hum=pv(0.2),
        nosenescence_hum=pv(0.8),
        max_turnover_time=pv(80.0),
        min_turnover_time=pv(10.0),
        leaffall=pv(10.0),
        lai_max=pv(12.0),
        leafagecrit=pv(100.0),
        lai_initmin=pv(0.3),
        tau_fruit=pv(90.0),
        tau_sap=pv(730.0),
        sla_age1=full(0.02),
        sla_max=pv(0.03),
        sla_min=pv(0.01),
        lpj_gap_const_mort=True,
        ok_dgvm=False,
        wire_vmax=True,
        ok_nlim_vmax=False,
        vcmax25=pv(50.0),
        n_limfert=full(1.0),
        leaf_timecst=pv(100.0),
        vmax_leafagecrit=pv(100.0),
        vmax_pheno_type=np.zeros(v, np.int32),
        vmax_leaf_tab=np.zeros(v, np.int32),
        vmax_ok_laidev=false,
        coldness_mort=0.04,
        frost_damage_limit=270.15,
        min_stomate=1e-8,
    )
    return dict(
        prescribe_inputs=prescribe,
        constraints_inputs=constraints,
        phenology_inputs=phenology,
        alloc_inputs=alloc,
        post_npp_inputs=post,
        allocation_kwargs=dict(f_fruit=0.1, ecureuil=pv(0.0), max_possible_lai=10.0, min_stomate=1e-8),
    )


def run_current_chain(inputs, *, compiled=False, public_wrapper=False):
    import jax

    jax.config.update("jax_enable_x64", True)
    from jax_orchidee.stomate import integration

    if public_wrapper:
        return integration.stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_static_jit(**inputs)
    fn = integration.stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_explicit
    if not compiled:
        return fn(**inputs)
    # Keep configuration and PFT dispatch as native constants, while every
    # per-cell state/forcing operand crosses the compiled component boundary.
    dynamic = {}
    static = copy.deepcopy(inputs)
    for group, values in inputs.items():
        dynamic[group] = {}
        for key, value in values.items():
            if isinstance(value, np.ndarray) and (value.ndim > 1 or value.shape == (6,)):
                dynamic[group][key] = value
                static[group].pop(key)

    def component(operands):
        result = fn(**{group: {**static[group], **operands[group]} for group in static})
        # String trace requirements are diagnostic metadata, not JIT values.
        boundary = result.post_npp.daily_carbon.boundary._replace(requires_trace=())
        carbon = result.post_npp.daily_carbon._replace(boundary=boundary)
        return result._replace(post_npp=result.post_npp._replace(daily_carbon=carbon))

    return jax.jit(component)(dynamic)


def chain_fields(result):
    out = {}

    def add(prefix, value, names):
        for name in names:
            out[prefix + "." + name] = np.asarray(getattr(value, name))

    add(
        "prescribe",
        result.prescribe,
        ["biomass", "leaf_frac", "when_growthinit", "co2_to_bm", "pft_present", "everywhere", "ind", "cn_ind"],
    )
    add("constraints", result.constraints, ["adapted", "regenerate"])
    add(
        "phenology",
        result.phenology,
        ["biomass", "leaf_frac", "leaf_age", "when_growthinit", "co2_to_bm", "begin_leaves", "gdd_midwinter"],
    )
    add("allocation", result.allocation, ["biomass", "leaf_age", "leaf_frac", "f_alloc"])
    add("npp", result.post_npp.daily_carbon.npp_update, ["biomass", "bm_alloc", "resp_maint", "resp_growth", "npp"])
    add("npp", result.post_npp.daily_carbon.age_sla, ["leaf_age", "leaf_frac", "age", "sla_calc", "sla_age1"])
    add("gap", result.post_npp.gap, ["biomass", "ind", "bm_to_litter", "mortality"])
    add(
        "turnover",
        result.post_npp.turnover,
        ["biomass", "leaf_age", "leaf_frac", "age", "lai", "turnover", "senescence", "turnover_time", "c_export"],
    )
    add("vmax", result.post_npp.vmax, ["leaf_age", "leaf_frac", "vcmax"])
    out["final.lai"] = np.asarray(result.post_npp.lai_after_setlai)
    out["final.bm_to_litter"] = np.asarray(result.post_npp.bm_to_litter)
    return out


def carry_inputs(inputs, result):
    """Expose the real returned carry to a second standalone component entry.

    This is a declared frozen-forcing component reentry, not next-day weather or
    calendar proof. Output rates bm_to_litter/co2_to_bm are reset at the boundary.
    """
    return carry_from_fields(inputs, chain_fields(result))


def carry_from_fields(inputs, fields):
    """Use an actual returned field set (Fortran or JAX), never endpoint algebra."""
    x = copy.deepcopy(inputs)
    pr, co, al, po = [x[k] for k in ("prescribe_inputs", "constraints_inputs", "alloc_inputs", "post_npp_inputs")]
    for k in ("pft_present", "everywhere", "cn_ind"):
        pr[k] = np.asarray(fields["prescribe." + k])
    pr["pft_present"] = pr["pft_present"].astype(bool)
    pr["when_growthinit"] = np.asarray(fields["phenology.when_growthinit"])
    pr.update(
        biomass=np.asarray(fields["turnover.biomass"]),
        leaf_frac=np.asarray(fields["vmax.leaf_frac"]),
        ind=np.asarray(fields["gap.ind"]),
        firstcall=False,
        stomate_restart_none=False,
    )
    co.update(
        adapted=np.asarray(fields["constraints.adapted"]), regenerate=np.asarray(fields["constraints.regenerate"])
    )
    x["phenology_inputs"]["leaf_age"] = np.asarray(fields["vmax.leaf_age"])
    x["phenology_inputs"]["gdd_midwinter"] = np.asarray(fields["phenology.gdd_midwinter"])
    al.update(
        lai=np.asarray(fields["final.lai"]),
        leaf_age=np.asarray(fields["vmax.leaf_age"]),
        age=np.asarray(fields["turnover.age"]),
        senescence=np.asarray(fields["turnover.senescence"], bool),
        sla_calc=np.asarray(fields["npp.sla_calc"]),
    )
    po.update(
        age=np.asarray(fields["turnover.age"]),
        senescence=np.asarray(fields["turnover.senescence"], bool),
        turnover_time=np.asarray(fields["turnover.turnover_time"]),
        sla_calc=np.asarray(fields["npp.sla_calc"]),
        sla_age1=np.asarray(fields["npp.sla_age1"]),
    )
    return x


def _span(filename, first, last):
    return b"".join((SOURCE / filename).read_bytes().splitlines(keepends=True)[first - 1 : last])


def source_identities():
    out = {}
    for name, (filename, a, b) in SPANS.items():
        sf = (SOURCE / filename).resolve()
        out[name] = dict(
            file=sf.relative_to(ROOT).as_posix(),
            subroutine=name,
            source_span=[a, b],
            sha256=hashlib.sha256(sf.read_bytes()).hexdigest(),
            span_sha256=hashlib.sha256(_span(filename, a, b)).hexdigest(),
        )
        if name == "phenology":
            out[name]["compiled_module_span"] = [1, len(sf.read_bytes().splitlines())]
            out[name]["compiled_module_sha256"] = out[name]["sha256"]
    return out


SETUPS = {
    "prescribe": """natural=.false.;pasture=.false.;is_tree=.false.;natural(14)=.true.;is_tree(14)=.true.
maxdia=.5d0;pheno_model='none'
do j=1,nvm;do k=1,nparts;bm_sapl(j,k,1)=dble(k);enddo;enddo
firstcall_prescribe=firstcall;stom_restname_in='restart.nc'
if(cold)stom_restname_in='NONE'
""",
    "constraints": """natural=.false.;is_tree=.false.;is_peat=.false.;agriculture=agri
natural(14)=.true.;is_tree(14)=.true.;is_peat(14)=.true.;pheno_model='none'
tmin_crit=undef;tcm_crit=undef;tcm_crit(14)=275d0
""",
    "phenology": """natural=.false.;is_tree=.false.;ok_LAIdev=.false.;pheno_model='none';lai_initmin=.3d0
natural(14)=.true.;is_tree(14)=.true.
hum_frac=.5d0;hum_min_time=30d0;pheno_gdd_crit=undef;ncdgdd_temp=undef
ncdgdd_moi=undef;ngd_crit=undef;pheno_moigdd_t_crit=undef;senescence_temp=undef
""",
    "allocation": """natural=.false.;pasture=.false.;is_tree=.false.;ok_LAIdev=.false.;is_grassland_manag=.false.
natural(14)=.true.;is_tree(14)=.true.;R0=.35d0;S0=.35d0;tau_leafinit=10d0;ext_coeff=.5d0
lai_max=12d0;lai_max_to_happy=.5d0;ecureuil=0d0;alloc_min=.2d0;alloc_max=.8d0;demi_alloc=100d0
alloc_agr_st=0d0;alloc_agr_pn=0d0;alloc_agr_st(14)=.25d0;alloc_agr_pn(14)=.05d0
""",
    "npp": """ok_LAIdev=.false.;is_grassland_manag=.false.;is_tree=.false.;is_tree(14)=.true.
frac_growthresp=.25d0;sla_max=.03d0;sla_min=.01d0;SP_densitesem=0d0;SP_pgrainmaxi=0d0;SP_tigefeuil=0d0;SP_slamax=.03d0
""",
    "gap": """natural=.false.;is_tree=.false.;pasture=.false.;leaf_tab=0;pheno_type=0
natural(14)=.true.;is_tree(14)=.true.;residence_time=30d0;availability_fact=.14d0;tmin_crit=undef
""",
    "turnover": """senescence_type='none';is_tree=.false.;natural=.false.;is_grassland_manag=.false.;ok_LAIdev=.false.
natural(14)=.true.;is_tree(14)=.true.;min_leaf_age_for_senescence=30d0;gdd_senescence=100d0;senescence_temp=0d0
hum_frac=.5d0;senescence_hum=.2d0;nosenescence_hum=.8d0;max_turnover_time=80d0;min_turnover_time=10d0
leaffall=10d0;lai_max=12d0;leafagecrit=100d0;lai_initmin=.3d0;tau_fruit=90d0;tau_sap=730d0;sla=.02d0
""",
    "vmax": """leaf_timecst=100d0;leafagecrit=100d0;vcmax25=50d0;pheno_type=0;leaf_tab=0;ok_LAIdev=.false.
""",
}


def _family_module(family):
    filenames = {"allocation": "alloc", "npp": "npp_calc", "turnover": "turn"}
    procedure = filenames.get(family, family)
    template_name = "season_phenology" if family == "phenology" else family
    text = (ROOT / f"scripts/dev/oracle_lane_stomate_{template_name}.f90.template").read_text()
    text = re.split(r"(?im)^program main|^\s*program main", text)[0]
    if family == "phenology":
        text = text.replace("nelements=2", "nelements=1").replace("min_stomate=1d-8", "min_stomate=0d0")
        text = text.replace(
            "ileaf=1, iroot=2, isapabove=3, isapbelow=4, iheartabove=5",
            "ileaf=1, iroot=6, isapabove=2, isapbelow=3, iheartabove=4",
        )
        text = text.replace("iheartbelow=6", "iheartbelow=5")
        text = text.replace("! <PHENOLOGY_MODULE>", (SOURCE / "stomate_phenology.f90").read_text())
        text = text.replace("PUBLIC phenology,phenology_clear", "PUBLIC phenology,phenology_clear,setup_phenology")
    else:
        filename, a, b = SPANS[procedure]
        placeholder = {"allocation": "ALLOC", "npp": "NPP", "turnover": "TURNOVER"}.get(family, family.upper())
        text = text.replace(f"! <{placeholder}_PROCEDURE>", _span(filename, a, b).decode())
        if family == "npp":
            cf = ROOT / "fortran_source/ORCHIDEE/src_sticslai/crop_alloc.f90"
            crop = b"".join(cf.read_bytes().splitlines(keepends=True)[17:384]).decode()
            text = text.replace("! <CROP_BMALLOC_PROCEDURE>", crop)
    text = text.replace("undef=1d20", "undef=-9999d0")
    text = re.sub(r"\bmin_stomate=0d0", "min_stomate=1d-8", text, flags=re.I)
    text = re.sub(r"\bmin_stomate=0\.(?!\d)", "min_stomate=1d-8", text, flags=re.I)
    modules = re.findall(r"(?im)^module\s+(\w+)", text)
    oracle = modules[-1]
    i = list(re.finditer(r"(?im)^\s*end\s+module(?:\s+\w+)?\s*$", text))[-1].start()
    setup = f"\nsubroutine setup_{family}(cold,firstcall,agri)\nlogical,intent(in)::cold,firstcall,agri\ninteger::j,k\n{SETUPS[family]}end subroutine\n"
    text = text[:i] + setup + text[i:]
    for name in sorted(modules, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(name)}\b", f"p05_{family}_{name}", text, flags=re.I)
    return text, f"p05_{family}_{oracle}"


def compose_chain_source():
    headers = []
    uses = []
    setups = []
    for family in SETUPS:
        module, name = _family_module(family)
        headers.append(module)
        proc = {"allocation": "alloc", "npp": "npp_calc", "turnover": "turn"}.get(family, family)
        uses.append(f"use {name}, only: {proc}, setup_{family}")
        setups.append(f"call setup_{family}(cold,firstcall,agri)")
    headers.append(
        "module p05_setlai\nuse p05_prescribe_defprec, only:r_std,i_std\n"
        "implicit none\ninteger,parameter::ibare_sechiba=1,ileaf=1,icarbon=1\n"
        "real(r_std),parameter::zero=0d0\ncontains\n"
        + _span("stomate_lai.f90", 58, 88).decode("utf-8")
        + "\nend module p05_setlai\n"
    )
    uses.append("use p05_setlai, only:setlai")
    program = r"""
program main
use p05_prescribe_defprec, only:r_std
!USES
implicit none
integer,parameter::n=6,v=14,p=12,l=4
integer::inp,u,mode,entry,i,j,k,tc
character(len=1024)::input_file,out
logical::cold,firstcall,agri,pr(n,v),sen(n,v),begin(n,v),inc(n,v)
integer::ni(n,v)
real(r_std)::veg(n,v),ev(n,v),wg(n,v),bio(n,v,p,1),la(n,v,l),lf(n,v,l),ind(n,v),cn(n,v),co2(n,v)
real(r_std)::adapt(n,v),regen(n,v),ts(n),tm(n),td(n),tl(n),tw(n),gpp(n,v),rmp(n,v,p)
real(r_std)::maxm(n,v),minm(n,v),mw(n,v),mm(n,v),gddm5(n,v),gddmid(n,v),ncd(n,v),ngd(n,v),thm(n,v)
real(r_std)::pdlai(n,v),slai(n,v),di(n,v),ss(n,v),slac(n,v),sla1(n,v),lai(n,v),soilT(n,2),soilH(n,2),age(n,v),rp(n,v),fa(n,v,p),wgc(n,v)
real(r_std)::ba(n,v,p,1),rm(n,v),rg(n,v),npp(n,v),ds(n,v),pg(n,v),dg(n,v),repr(n,v),cr(n,v),cl(n,v),nf(n,v)
real(r_std)::nlong(n,v),turnlong(n,v,p,1),lm(n,v),lit(n,v,p,1),mort(n,v),spring(n,v),herb(n,v),gdd(n,v),turnout(n,v,p,1),tt(n,v),ce(n,v),vc(n,v)
call get_command_argument(1,input_file);call get_command_argument(2,out)
open(newunit=inp,file=trim(input_file),status='old');read(inp,*)cold,firstcall,agri
!READS
close(inp)
!SETUPS
inc=.false.;ni=0;tc=1;rp=0d0;di=0d0;ds=0d0;ss=1d0;pg=0d0;dg=0d0;repr=0d0;cr=0d0;cl=0d0;pdlai=0d0;slai=0d0
gddm5=10d0;ncd=30d0;ngd=40d0;thm=5d0;maxm=1d0;minm=0d0;mm=.6d0;wgc=100d0;nf=1d0
nlong=20d0;turnlong=0d0;lm=50d0;lit=0d0;mort=0d0;spring=0d0;herb=0d0;gdd=0d0;ce=0d0;vc=0d0
open(newunit=u,file=trim(out),status='replace');write(u,'(A)')'field,value'
call prescribe(n,veg,1d0,pr,ev,wg,bio,lf,ind,cn,co2)
call d4('prescribe.biomass',bio);call d3('prescribe.leaf_frac',lf);call d2('prescribe.when_growthinit',wg)
call d2('prescribe.co2_to_bm',co2);call d2('prescribe.pft_present',merge(1d0,0d0,pr));call d2('prescribe.everywhere',ev)
call d2('prescribe.ind',ind);call d2('prescribe.cn_ind',cn)
call constraints(n,1d0,tm,td,wg,adapt,regen,ts)
call d2('constraints.adapted',adapt);call d2('constraints.regenerate',regen)
call phenology(n,1d0,pr,veg,tl,tm,tw,gpp,maxm,minm,mm,mw,gddm5,gddmid,ncd,ngd,sen,thm,bio,lf,la,wg,co2,pdlai,slai,di,ss,begin,slac)
call d4('phenology.biomass',bio);call d3('phenology.leaf_frac',lf);call d3('phenology.leaf_age',la)
call d2('phenology.when_growthinit',wg);call d2('phenology.co2_to_bm',co2);call d2('phenology.begin_leaves',merge(1d0,0d0,begin));call d2('phenology.gdd_midwinter',gddmid)
call alloc(n,1d0,lai,veg,sen,wg,mw,soilT,soilH,bio,age,la,lf,rp,fa,di,ss,slac,wgc)
call d4('allocation.biomass',bio);call d3('allocation.leaf_age',la);call d3('allocation.leaf_frac',lf);call d3('allocation.f_alloc',fa)
call setlai(bio,slac,slai)
call npp_calc(n,1d0,pr,tl,soilT,lai,rp,gpp,fa,ba,rmp,bio,la,lf,age,rm,rg,npp,inc,di,ds,ss,pg,dg,repr,ni,ni,ni,ni,ni,ni,cr,cl,slai,tc,veg,slac,sla1,nf)
call d4('npp.biomass',bio);call d4('npp.bm_alloc',ba);call d2('npp.resp_maint',rm);call d2('npp.resp_growth',rg);call d2('npp.npp',npp)
call d3('npp.leaf_age',la);call d3('npp.leaf_frac',lf);call d2('npp.age',age);call d2('npp.sla_calc',slac);call d2('npp.sla_age1',sla1)
call gap(n,1d0,nlong,turnlong,lm,pr,bio,ind,lit,mort,td,spring,slac)
call d4('gap.biomass',bio);call d2('gap.ind',ind);call d4('gap.bm_to_litter',lit);call d2('gap.mortality',mort)
! Original stomate_lpj keeps entry lai until final setlai; slai is distinct.
call turn(n,1d0,pr,herb,maxm,minm,mw,mm,tl,tm,tw,veg,gdd,la,lf,age,lai,bio,turnout,sen,tt,ni,ce,slac)
call d4('turnover.biomass',bio);call d3('turnover.leaf_age',la);call d3('turnover.leaf_frac',lf);call d2('turnover.age',age)
call d2('turnover.lai',lai);call d4('turnover.turnover',turnout);call d2('turnover.senescence',merge(1d0,0d0,sen));call d2('turnover.turnover_time',tt);call d2('turnover.c_export',ce)
call setlai(bio,slac,slai)
call setlai(bio,slac,lai)
call vmax(n,1d0,la,lf,vc,nf)
call d3('vmax.leaf_age',la);call d3('vmax.leaf_frac',lf);call d2('vmax.vcmax',vc);call d2('final.lai',lai);call d4('final.bm_to_litter',lit)
close(u)
contains
subroutine dump(name,x)
character(*),intent(in)::name
real(r_std),intent(in)::x(:)
integer::q
do q=1,size(x);write(u,'(A,",",ES26.17)')name,x(q);enddo
end subroutine
subroutine d2(name,x)
character(*),intent(in)::name
real(r_std),intent(in)::x(:,:)
call dump(name,reshape(x,(/size(x)/)))
end subroutine
subroutine d3(name,x)
character(*),intent(in)::name
real(r_std),intent(in)::x(:,:,:)
call dump(name,reshape(x,(/size(x)/)))
end subroutine
subroutine d4(name,x)
character(*),intent(in)::name
real(r_std),intent(in)::x(:,:,:,:)
call dump(name,reshape(x,(/size(x)/)))
end subroutine
end program
"""
    program = program.replace("!USES", "\n".join(uses)).replace("!SETUPS", "\n".join(setups))
    program = program.replace("!READS", "\n".join(f"read(inp,*){v}" for v, _, _ in INPUT_MAP))
    return "\n".join(headers) + program


INPUT_MAP = [
    ("veg", "prescribe_inputs", "veget_max"),
    ("ev", "prescribe_inputs", "everywhere"),
    ("wg", "prescribe_inputs", "when_growthinit"),
    ("bio", "prescribe_inputs", "biomass"),
    ("la", "alloc_inputs", "leaf_age"),
    ("lf", "prescribe_inputs", "leaf_frac"),
    ("ind", "prescribe_inputs", "ind"),
    ("cn", "prescribe_inputs", "cn_ind"),
    ("co2", "prescribe_inputs", "co2_to_bm"),
    ("pr", "prescribe_inputs", "pft_present"),
    ("sen", "alloc_inputs", "senescence"),
    ("adapt", "constraints_inputs", "adapted"),
    ("regen", "constraints_inputs", "regenerate"),
    ("ts", "constraints_inputs", "tseason"),
    ("tm", "constraints_inputs", "t2m_month"),
    ("td", "constraints_inputs", "t2m_min_daily"),
    ("tl", "post_npp_inputs", "t2m_longterm"),
    ("tw", "post_npp_inputs", "t2m_week"),
    ("gpp", "post_npp_inputs", "gpp_daily"),
    ("rmp", "post_npp_inputs", "resp_maint_part"),
    ("mw", "alloc_inputs", "moiavail_week"),
    ("gddmid", "phenology_inputs", "gdd_midwinter"),
    ("slac", "alloc_inputs", "sla_calc"),
    ("sla1", "post_npp_inputs", "sla_age1"),
    ("lai", "alloc_inputs", "lai"),
    ("soilT", "alloc_inputs", "tsoil_month"),
    ("soilH", "alloc_inputs", "soilhum_month"),
    ("age", "alloc_inputs", "age"),
    ("tt", "post_npp_inputs", "turnover_time"),
]


def write_fortran_input(path, inputs):
    rows = [
        " ".join(
            "T" if value else "F"
            for value in (
                inputs["prescribe_inputs"]["stomate_restart_none"],
                inputs["prescribe_inputs"]["firstcall"],
                inputs["constraints_inputs"]["agriculture"],
            )
        )
    ]
    for _, group, name in INPUT_MAP:
        a = np.asarray(inputs[group][name])
        vals = a.ravel(order="F")
        rows.append(" ".join(("T" if x else "F") if a.dtype == bool else format(float(x), ".17e") for x in vals))
    path.write_text("\n".join(rows) + "\n", encoding="ascii")


def read_csv(path):
    out = {}
    with path.open(newline="", encoding="ascii") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["field"], []).append(float(row["value"]))
    return {k: np.asarray(v) for k, v in out.items()}


def reference_shape(name):
    """Shape from the standalone Fortran declarations, independent of JAX."""
    field = name.split(".")[-1]
    if field in {"biomass", "bm_alloc", "bm_to_litter", "turnover"}:
        return (6, 14, 12, 1)
    if field in {"leaf_age", "leaf_frac"}:
        return (6, 14, 4)
    if field == "f_alloc":
        return (6, 14, 12)
    return (6, 14)


def constraints_inputs():
    """Independent mask/temperature matrix, including peat-only admission."""
    cases = {}
    for label, nat, peat, agriculture, none in (
        ("paper_natural_peat", True, True, True, True),
        ("natural_no_agri", True, False, False, True),
        ("peat_only", False, True, False, True),
        ("agriculture_only", False, False, True, True),
        ("excluded_mask", False, False, False, True),
        ("seasonal_tree", True, True, False, False),
    ):
        for defined in (True, False):
            n, v = 3, 14

            def mask(x):
                return np.asarray([False] * 13 + [x], bool)

            wg = np.zeros((n, v))
            wg[:, 13] = [0.0, 2000.0, 1e33]
            reg = np.full((n, v), 0.5)
            reg[0, 13] = 1e-4
            tmin = np.full(v, -9999.0)
            tcm = tmin.copy()
            if defined:
                tcm[13] = 275.0
            else:
                tmin[13] = 265.0
            cases[label + ("_tcm" if defined else "_undef")] = dict(
                t2m_month=np.asarray([270.0, 275.0, 285.0]),
                t2m_min_daily=np.asarray([260.0, 270.0, 275.0]),
                when_growthinit=wg,
                adapted=np.full((n, v), 0.25),
                regenerate=reg,
                tseason=np.asarray([285.0, 280.15, 275.0]),
                natural=mask(nat),
                is_tree=mask(True),
                is_peat=mask(peat),
                pheno_is_none=np.asarray([True] * 13 + [none], bool),
                tmin_crit=tmin,
                tcm_crit=tcm,
                agriculture=agriculture,
                undef=-9999.0,
            )
    return cases


def sla_bound_inputs():
    """Restart boundary values expose each ordered source SLA clamp arm."""
    x = chain_inputs()
    for row, value in ((0, 0.05), (4, 0.005)):
        x["alloc_inputs"]["sla_calc"][row, 13] = value
        x["post_npp_inputs"]["sla_calc"][row, 13] = value
        x["alloc_inputs"]["lai"][row, 13] = x["prescribe_inputs"]["biomass"][row, 13, 0, 0] * value
        x["prescribe_inputs"]["when_growthinit"][row, 13] = 301.0
        x["post_npp_inputs"]["gpp_daily"][row, 13] = 0.0
        x["post_npp_inputs"]["resp_maint_part"][row, 13] = 0.0
    # Original prescribe oracle's high-wood crown arm, retained as actual input.
    for part in (1, 2, 3, 4):
        x["prescribe_inputs"]["biomass"][2, 13, part, 0] = 10000.0
    return x


def run_constraints_matrix(checks, compiler):
    header, module = _family_module("constraints")
    program = f"""
program main
use p05_constraints_defprec,only:r_std
use {module},only:constraints,setup_constraints
use p05_constraints_pft_parameters,only:natural,is_tree,is_peat,agriculture,pheno_model,tmin_crit,tcm_crit
implicit none
integer::inp,u,c,i,j
logical::nat,peat,agri,none,defined
character(len=1024)::a,b
character(len=80)::label
real(r_std)::tm(3),td(3),ts(3),wg(3,14),adapt(3,14),regen(3,14)
call get_command_argument(1,a);call get_command_argument(2,b)
call setup_constraints(.false.,.true.,.false.)
open(newunit=inp,file=trim(a),status='old');open(newunit=u,file=trim(b),status='replace');write(u,'(A)')'field,value'
do c=1,12
read(inp,*)label,nat,peat,agri,none,defined
read(inp,*)tm;read(inp,*)td;read(inp,*)ts;read(inp,*)wg;read(inp,*)adapt;read(inp,*)regen
natural=.false.;natural(14)=nat;is_peat=.false.;is_peat(14)=peat;agriculture=agri
is_tree=.false.;is_tree(14)=.true.;pheno_model='none';if(.not.none)pheno_model(14)='cold'
tmin_crit=-9999d0;tcm_crit=-9999d0
if(defined)then;tcm_crit(14)=275d0;else;tmin_crit(14)=265d0;endif
call constraints(3,1d0,tm,td,wg,adapt,regen,ts)
do j=1,14;do i=1,3
write(u,'(A,",",ES26.17)')trim(label)//'.adapted',adapt(i,j)
write(u,'(A,",",ES26.17)')trim(label)//'.regenerate',regen(i,j)
enddo;enddo
enddo
close(inp);close(u)
end program
"""
    source = checks / "constraints-mask-original.f90"
    source.write_text(header + program)
    inp = checks / "constraints-mask-inputs.txt"
    rows = []
    cases = constraints_inputs()
    for name, x in cases.items():
        flags = [
            x["natural"][13],
            x["is_peat"][13],
            x["agriculture"],
            x["pheno_is_none"][13],
            x["tcm_crit"][13] != -9999.0,
        ]
        rows.append(name + " " + " ".join("T" if y else "F" for y in flags))
        for key in ("t2m_month", "t2m_min_daily", "tseason", "when_growthinit", "adapted", "regenerate"):
            rows.append(" ".join(format(float(z), ".17e") for z in x[key].ravel(order="F")))
    inp.write_text("\n".join(rows) + "\n")
    csvp = checks / "constraints-mask-fortran.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_p05_constraints_") as td:
        b = Path(td)
        unit = b / "oracle.f90"
        unit.write_bytes(source.read_bytes())
        exe = b / "oracle.exe"
        build = compile_fortran(unit, exe, compiler, extra_flags=("-fdefault-double-8",))
        run = subprocess.run(
            [str(exe), str(inp.resolve()), str(csvp.resolve())],
            cwd=b,
            env=compiler_environment(compiler),
            text=True,
            capture_output=True,
            check=True,
        )
    (checks / "constraints-mask-execution.txt").write_text(run.stdout + "\n" + run.stderr)
    from jax_orchidee.stomate.carbon_kernels import constraints_step

    f = read_csv(csvp)
    refs = {}
    comparisons = []
    for name, x in cases.items():
        current = constraints_step(**x)
        refs[name] = {"inputs": encode(x), "expected": {}}
        for field in ("adapted", "regenerate"):
            expected = f[name + "." + field]
            refs[name]["expected"][field] = {"shape": [3, 14], "values": expected.tolist()}
            comparisons.append(
                float_comparison(
                    name + "." + field,
                    expected,
                    np.asarray(getattr(current, field)).ravel(order="F"),
                    rtol=1e-12,
                    atol=1e-14,
                )
            )
    return refs, {
        "build": build,
        "comparisons": comparisons,
        "status": "passed" if all(c["passed"] for c in comparisons) else "failed",
    }


def run_oracle(output_dir=PACKAGE, *, write_fixture=False, compiler=DEFAULT_COMPILER):
    output_dir = Path(output_dir)
    checks = output_dir / "checks"
    checks.mkdir(parents=True, exist_ok=True)
    source = checks / "growth-chain-original.f90"
    source.write_text(compose_chain_source(), encoding="utf-8")
    result = {
        "schema_version": 1,
        "scope": "standalone daily growth component; no full model day",
        "source_identities": source_identities(),
        "cases": {},
        "comparisons": [],
    }
    fixture = {
        "schema_version": 1,
        "reference": "original Fortran procedure chain",
        "source_identities": source_identities(),
        "rtol": 1e-12,
        "atol": 1e-14,
        "cases": {},
    }
    with tempfile.TemporaryDirectory(prefix="orchidee_phase05_growth_") as td:
        b = Path(td)
        unit = b / "oracle.f90"
        unit.write_bytes(source.read_bytes())
        exe = b / "oracle.exe"
        try:
            result["build"] = compile_fortran(
                unit,
                exe,
                compiler,
                base_flags=(
                    "-std=legacy",
                    "-fdefault-real-8",
                    "-fdefault-double-8",
                    "-ffree-line-length-none",
                    "-O0",
                    "-fcheck=all",
                    "-ffpe-trap=invalid,zero,overflow",
                    "-Wall",
                    "-Wextra",
                ),
            )
        except subprocess.CalledProcessError as e:
            (checks / "compile-failure.txt").write_text(e.stderr + "\n" + e.stdout, encoding="utf-8")
            raise
        pending = [
            ("restart", chain_inputs(cold_start=False)),
            ("cold", chain_inputs(cold_start=True)),
            ("sla_bounds", sla_bound_inputs()),
        ]
        while pending:
            name, inputs = pending.pop(0)
            inp = checks / f"{name}-inputs.txt"
            csvp = checks / f"{name}-fortran.csv"
            write_fortran_input(inp, inputs)
            run = subprocess.run(
                [str(exe), str(inp.resolve()), str(csvp.resolve())],
                cwd=b,
                env=compiler_environment(compiler),
                text=True,
                capture_output=True,
            )
            (checks / f"{name}-execution.txt").write_text(run.stdout + "\n" + run.stderr, encoding="utf-8")
            run.check_returncode()
            f = read_csv(csvp)
            eager = run_current_chain(inputs)
            j = chain_fields(eager)
            compiled = chain_fields(run_current_chain(inputs, public_wrapper=True))
            comparisons = []
            for k, values in f.items():
                comparisons += [
                    float_comparison(name + ".eager." + k, values, j[k].ravel(order="F"), rtol=1e-12, atol=1e-14),
                    float_comparison(name + ".jit." + k, values, compiled[k].ravel(order="F"), rtol=1e-12, atol=1e-14),
                ]
            case = {
                "inputs": encode(inputs),
                "expected": {
                    k: {"shape": list(reference_shape(k)), "values": values.tolist()} for k, values in f.items()
                },
            }
            fixture["cases"][name] = case
            result["cases"][name] = {
                "field_count": len(f),
                "coordinates": sum(x.size for x in f.values()),
                "comparisons": comparisons,
            }
            result["comparisons"].extend(comparisons)
            if "reentry" not in name:
                original_fields = {k: values.reshape(reference_shape(k), order="F") for k, values in f.items()}
                pending.append((name + "_reentry", carry_from_fields(inputs, original_fields)))
                actual_carry = carry_inputs(inputs, eager)
                reference_carry = carry_from_fields(inputs, original_fields)
                carry_checks = []
                for group in actual_carry:
                    for k, value in actual_carry[group].items():
                        if isinstance(value, np.ndarray):
                            carry_checks.append(
                                float_comparison(
                                    name + ".carry." + group + "." + k,
                                    reference_carry[group][k],
                                    value,
                                    rtol=1e-12,
                                    atol=1e-14,
                                )
                            )
                result["cases"][name]["carry_checks"] = carry_checks
                result["comparisons"].extend(carry_checks)
        (checks / "compile.json").write_text(json.dumps(result["build"], indent=2) + "\n")
    fixture["constraints_cases"], result["constraints_matrix"] = run_constraints_matrix(checks, compiler)
    result["comparisons"].extend(result["constraints_matrix"]["comparisons"])
    result["status"] = "passed" if all(x["passed"] for x in result["comparisons"]) else "failed"
    (output_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    if write_fixture:
        FIXTURE.write_text(json.dumps(fixture, separators=(",", ":")) + "\n", encoding="ascii")
    (output_dir / "inputs.json").write_text(
        json.dumps({k: v["inputs"] for k, v in fixture["cases"].items()}, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=PACKAGE)
    p.add_argument("--write-fixture", action="store_true")
    args = p.parse_args()
    r = run_oracle(args.output_dir, write_fixture=args.write_fixture)
    print(
        json.dumps(
            {
                "status": r["status"],
                "comparisons": len(r["comparisons"]),
                "failed": [x for x in r["comparisons"] if not x["passed"]],
            },
            indent=2,
        )
    )
    sys.exit(r["status"] != "passed")
