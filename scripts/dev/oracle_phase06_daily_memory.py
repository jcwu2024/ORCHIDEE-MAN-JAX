"""Finite original-season compound conditions and actual compiled consumer joins."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jax_orchidee.stomate import season as sm  # noqa: E402
from scripts.dev import oracle_lane_stomate_season_phenology as legacy  # noqa: E402
from scripts.dev.oracle_phase05_daily_memory import compare, encode, observe  # noqa: E402

jax.config.update("jax_enable_x64", True)
OUT = (
    ROOT / "outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/daily-memory"
)
FIXTURE = ROOT / "tests/fixtures/phase06_daily_memory_reference.json"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90"
N = 8
MEMORY = dict(
    moiavail_month="mm",
    moiavail_week="mw",
    t2m_longterm="tlong",
    tau_longterm="tau",
    t2m_month="tm",
    t2m_week="tw",
    tseason="tsy",
    tseason_length="tsyl",
    tseason_tmp="tsyt",
    tmin_spring_time="tmins",
    onset_date="onset",
    tsoil_month="tsm",
    soilhum_month="shm",
    gdd_from_growthinit="gfrom",
    tsurf_year="tsurfyear",
    t2m_14="t14",
)
ANNUAL = dict(
    npp_longterm="nppl",
    turnover_longterm="turnl",
    gpp_week="gppw",
    maxmoiavail_lastyear="maxl",
    maxmoiavail_thisyear="maxt",
    minmoiavail_lastyear="minl",
    minmoiavail_thisyear="mint",
    maxgppweek_lastyear="mgpl",
    maxgppweek_thisyear="mgpt",
    gdd0_lastyear="g0l",
    gdd0_thisyear="g0t",
    precip_lastyear="pl",
    precip_thisyear="pt",
    lm_lastyearmax="lml",
    lm_thisyearmax="lmt",
    maxfpc_lastyear="mfpl",
    maxfpc_thisyear="mfpt",
)
BIO = dict(
    gdd_m5_dormance="gddm5",
    gdd_midwinter="gddmid",
    ncd_dormance="ncd",
    ngd_minus5="ngd",
    time_hum_min="thm",
    hum_min_dormance="hmd",
)
CONFIG = (
    "natural",
    "pasture",
    "leaf_tab",
    "pheno_type",
    "pft_to_mtc",
    "pheno_model",
    "pheno_gdd_crit",
    "ncdgdd_temp",
    "hum_min_time",
    "leaflife_tab",
    "lai_max",
    "ok_dgvm",
    "ok_stomate",
    "julian_diff",
)
FORCING = (
    "v",
    "vd",
    "mday",
    "t2",
    "ts",
    "tsoil",
    "sh",
    "lalo",
    "pr",
    "npp",
    "bio",
    "turn",
    "gpp",
    "wg",
    "tmin",
    "begin",
    "sla",
    "gid",
    "herb",
)
ORDER = CONFIG + FORCING + tuple(MEMORY.values()) + tuple(ANNUAL.values()) + tuple(BIO.values())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(encode(value), indent=2) + "\n", encoding="utf-8")


def declared_inputs():
    """Complete prescribed tuples, never reconstructed from expected outputs."""

    def field(value, shape=(N, 14)):
        return np.full(shape, value, dtype=float)

    d = dict(
        natural=[False] * 13 + [True],
        pasture=[False] * 14,
        leaf_tab=[1] * 14,
        pheno_type=[1] * 14,
        pft_to_mtc=[2] * 14,
        pheno_model=["none"] * 14,
        pheno_gdd_crit=field(-9999, (14, 4)),
        ncdgdd_temp=field(-9999, (14,)),
        hum_min_time=field(-9999, (14,)),
        leaflife_tab=field(365, (14,)),
        lai_max=field(1, (14,)),
        ok_dgvm=False,
        ok_stomate=True,
        julian_diff=100.0,
    )
    d.update(
        v=field(0),
        vd=field(0),
        mday=field(0.4),
        t2=np.array([268.15, 267.15, 269.15, 273.15, 274.15, 280.0, 290.0, 300.0]),
        ts=field(281, (N,)),
        tsoil=field(278, (N, 2)),
        sh=field(0.4, (N, 2)),
        lalo=field(0, (N, 2)),
        pr=field(2, (N,)),
        npp=field(0),
        bio=field(0, (N, 14, 12, 2)),
        turn=field(0, (N, 14, 12, 2)),
        gpp=field(0),
        wg=field(10),
        tmin=field(275, (N,)),
        begin=np.zeros((N, 14), dtype=bool),
        sla=field(0.02),
        gid=field(0, (N, 2)),
        herb=field(0),
    )
    d["v"][:, 13] = 0.5
    d["vd"][:, 13] = 1
    d["lalo"][:, 0] = 10
    d["wg"][4, 13] = 0
    d["begin"][6:, 13] = True
    d["gid"][:, 0] = 50
    d["gid"][7, 0] = 100
    d.update(
        mm=field(0.4),
        mw=field(0.4),
        tlong=np.array([250.0, 253.1, 270.0, 290.0, 303.1, 310.0, 280.0, 300.0]),
        tau=1093.0,
        tm=np.array([290.0, 290.0, 270.0, 270.0, 280.0, 280.0, 280.0, 280.0]),
        tw=np.array([280.0, 300.0, 260.0, 280.0, 273.15, 270.0, 273.15, 280.0]),
        tsy=field(280, (N,)),
        tsyt=np.array([0.0, 280.0, 0.0, 560.0, 0.0, 280.0, 0.0, 560.0]),
        tsyl=np.array([0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0]),
        tmins=field(0),
        onset=field(0),
        tsm=field(278, (N, 2)),
        shm=field(0.4, (N, 2)),
        gfrom=field(3),
        tsurfyear=field(281, (N,)),
        t14=field(280, (N,)),
        nppl=field(1),
        turnl=field(0.01, (N, 14, 12, 2)),
        gppw=field(0.1),
        maxl=field(0.7),
        maxt=field(0.6),
        minl=field(0.2),
        mint=field(0.3),
        mgpl=field(0.4),
        mgpt=field(0.3),
        g0l=field(1, (N,)),
        g0t=field(2, (N,)),
        pl=field(3, (N,)),
        pt=field(4, (N,)),
        lml=field(1),
        lmt=field(0.8),
        mfpl=field(0.4),
        mfpt=field(0.3),
        gddm5=field(5),
        gddmid=field(5),
        ncd=field(5),
        ngd=field(7),
        thm=field(4),
        hmd=field(0.5),
    )
    d["nppl"][:, 13] = [0, 0, 1, 1, 0, 1, 1, 1]
    d["npp"][:, 13] = [0, 0, 0.1, 0.1, 0, 0.1, 0.1, 0.1]
    d["lml"][:, 13] = [0, 1, 0, 1, 0, 1, 0, 1]
    d["gpp"][:, 13] = 0.2
    d["bio"][:, 13, 0, 0] = [0, 0.5, 0.6, 0.75, 0.8, 1.0, 2.0, 0]
    cases = {}
    for name, tau, year in (
        ("paper_tau_below", 1093.0, False),
        ("paper_tau_equal", 1094.0, False),
        ("paper_tau_above_year", 1095.0, True),
    ):
        case = copy.deepcopy(d)
        case.update(tau=tau, end_of_year=year, domain="fixed paper PFT14 configuration; prescribed finite states")
        cases[name] = case
    equal = copy.deepcopy(d)
    equal.update(tau=1.0, end_of_year=False, domain="fixed target PFT14; exact clamp equalities with tau_after=2")
    equal["t2"] = np.array([250.0, 253.1, 280.0, 303.1, 310.0, 268.15, 273.15, 290.0])
    equal["tlong"] = equal["t2"].copy()
    cases["paper_clamp_equalities"] = equal
    diagnostic = copy.deepcopy(d)
    diagnostic.update(
        end_of_year=False,
        domain="nonpaper diagnostic configuration; no production activation or natural-trajectory claim",
    )
    diagnostic["leaf_tab"][11] = 0
    diagnostic["leaf_tab"][9] = 0
    diagnostic["pheno_type"][11] = 2
    diagnostic["pheno_type"][13] = 2
    diagnostic["pft_to_mtc"][10] = 6
    diagnostic["pft_to_mtc"][13] = 8
    diagnostic["pheno_gdd_crit"][13, :] = [1, 2, 3, 4]
    diagnostic["pheno_gdd_crit"][12, :3] = [1, 2, 3]
    for column in range(4):
        diagnostic["pheno_gdd_crit"][8 + column, :] = [1, 2, 3, 4]
        diagnostic["pheno_gdd_crit"][8 + column, column] = -9999
    diagnostic["ncdgdd_temp"][13] = 0
    diagnostic["hum_min_time"][13] = 1
    diagnostic["gddm5"][[1, 3], 13] = -9999
    diagnostic["gddmid"][[1, 5], 13] = -9999
    diagnostic["ncd"][:, 13] = [1093, 1094, 1095, -9999, 5, 5, 5, 5]
    diagnostic["hmd"][:, 13] = [-9999, 0.3, 0.4, 0.5, 0.5, -9999, 0.4, 0.5]
    diagnostic["tmins"][:, 13] = [-1, 0, 1, 38, 39, 40, 41, 2]
    diagnostic["tmins"][:, 10] = [-1, 0, 1, 38, 39, 40, 41, 2]
    cases["diagnostic_defined_memory"] = diagnostic
    dgvm = copy.deepcopy(d)
    dgvm.update(ok_dgvm=True, end_of_year=False, domain="nonpaper DGVM diagnostic; exact source component only")
    dgvm["natural"][11] = True
    dgvm["pasture"][11] = True
    dgvm["vd"][:, 1] = [0, 1, 1 - 1e-8, 0.5, 1, 0, 0.5, 0]
    dgvm["vd"][:, 13] = 0
    dgvm["lmt"][:, 13] = [0, 1e-8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    cases["diagnostic_dgvm"] = dgvm
    frac_boundary = copy.deepcopy(dgvm)
    # Two positive agricultural covers leave exactly min_stomate after the
    # original sequential subtraction. Inputs are specified before compiling
    # either reference and retained even if current production disagrees.
    frac_boundary["vd"][5, 1] = 1.0 - 2.0**-26
    frac_boundary["vd"][5, 2] = 2.0**-26 - 1e-8
    cases["diagnostic_frac_exact_min"] = frac_boundary
    valid_boundary = copy.deepcopy(frac_boundary)
    valid_boundary["domain"] = "nonpaper DGVM diagnostic; cover-consistent complete tuple, no natural trajectory"
    # A new input case, not a change to the preserved exposed algebra failure:
    # every veget is within veget_max and total maximal cover is one.
    valid_boundary["vd"][:, 13] = [1.0, 0.0, 1.0 - (1.0 - 1e-8), 0.5, 0.0, 1e-8, 0.5, 1.0]
    valid_boundary["v"] = valid_boundary["vd"] * 0.5
    cases["diagnostic_frac_valid_cover_min"] = valid_boundary
    return cases


def current(case, *, compiled=False):
    """Real public season calls with a genuine JIT return boundary."""
    c = {
        key: np.asarray(value) if isinstance(value, list) and key != "pheno_model" else value
        for key, value in case.items()
    }
    memory = sm.SeasonMemoryState(**{k: jnp.asarray(c[v]) for k, v in MEMORY.items()})
    annual = sm.SeasonAnnualState(**{k: jnp.asarray(c[v]) for k, v in ANNUAL.items()})
    bio = sm.SeasonBiometeorologyState(**{k: jnp.asarray(c[v]) for k, v in BIO.items()})

    def transition(memory, annual, bio):
        m = sm.season_memory_step(
            memory,
            dt_days=1.0,
            julian_diff=c["julian_diff"],
            end_of_year=c["end_of_year"],
            firstcall=True,
            moiavail_daily=c["mday"],
            t2m_daily=c["t2"],
            tsurf_daily=c["ts"],
            tsoil_daily=c["tsoil"],
            soilhum_daily=c["sh"],
            begin_leaves=c["begin"],
            when_growthinit=c["wg"],
            natural=c["natural"],
            leaf_tab=c["leaf_tab"],
            pheno_type=c["pheno_type"],
            pft_to_mtc=c["pft_to_mtc"],
        ).state
        gid = sm.season_update_gdd_init_date(
            jnp.asarray(c["gid"]),
            latitude=np.asarray(c["lalo"])[:, 0],
            julian_diff=c["julian_diff"],
            calendar_str="noleap",
        )
        b = sm.season_biometeorology_step(
            bio,
            dt_days=1.0,
            julian_diff=c["julian_diff"],
            firstcall=True,
            t2m_daily=c["t2"],
            t2m_month=m.t2m_month,
            t2m_week=m.t2m_week,
            t2m_longterm=m.t2m_longterm,
            moiavail_month=m.moiavail_month,
            when_growthinit=c["wg"],
            gdd_init_date=gid,
            pheno_gdd_crit=c["pheno_gdd_crit"],
            ncdgdd_temp=c["ncdgdd_temp"],
            hum_min_time=c["hum_min_time"],
        ).state
        a = sm.season_annual_step(
            annual,
            dt_days=1.0,
            tau_longterm=m.tau_longterm,
            end_of_year=c["end_of_year"],
            firstcall=True,
            veget=c["v"],
            veget_max=c["vd"],
            moiavail_daily=c["mday"],
            t2m_daily=c["t2"],
            precip_daily=c["pr"],
            biomass=c["bio"],
            npp_daily=c["npp"],
            turnover_daily=c["turn"],
            gpp_daily=c["gpp"],
            natural=c["natural"],
            pasture=c["pasture"],
            leaflife_tab=c["leaflife_tab"],
            pheno_model=c["pheno_model"],
            hvc1=0.019,
            hvc2=1.38,
            ok_dgvm=c["ok_dgvm"],
            ok_stomate=c["ok_stomate"],
        )
        return m, a.state, b, gid, a.herbivores

    values = jax.jit(transition)(memory, annual, bio) if compiled else transition(memory, annual, bio)
    m, a, b, gid, herb = values
    return {**m._asdict(), **a._asdict(), **b._asdict(), "gdd_init_date": gid, "herbivores": herb}


# Added pure prints before original statements; exact line identities are
# checked during reference creation, never at portable numerical test time.
OBSERVERS = {
    626: [("tau_candidate", "(/tau_longterm+dt/)"), ("tau_max", "(/tau_longterm_max/)")],
    633: [("tlong_raw", "t2m_longterm"), ("tlong_low", "(/tlong_ref_min/)"), ("tlong_high", "(/tlong_ref_max/)")],
    670: [
        ("spring_first_left", "MERGE(1d0,0d0,Tmin_spring_time(:,j)>0)"),
        ("spring_first_right", "MERGE(1d0,0d0,Tmin_spring_time(:,j)<spring_days_max)"),
    ],
    711: [
        ("spring_second_left", "MERGE(1d0,0d0,Tmin_spring_time(:,j)>0)"),
        ("spring_second_right", "MERGE(1d0,0d0,Tmin_spring_time(:,j)<40)"),
    ],
    884: [
        ("gdd_warm", "MERGE(1d0,0d0,t2m_daily>ZeroCelsius-gdd_threshold)"),
        ("gdd_defined", "MERGE(1d0,0d0,gdd_m5_dormance(:,j)/=undef)"),
    ],
    954: [
        ("midsummer_week", "MERGE(1d0,0d0,t2m_month>t2m_week)"),
        ("midsummer_long", "MERGE(1d0,0d0,t2m_month>t2m_longterm)"),
    ],
    965: [
        ("mid_defined", "MERGE(1d0,0d0,gdd_midwinter(:,j)/=undef)"),
        ("mid_warm", "MERGE(1d0,0d0,t2m_daily>ncdgdd_temp(j)+ZeroCelsius)"),
    ],
    1020: [
        ("ncd_defined", "MERGE(1d0,0d0,ncd_dormance(:,j)/=undef)"),
        ("ncd_cold", "MERGE(1d0,0d0,t2m_daily<=ncdgdd_temp(j)+ZeroCelsius)"),
        ("ncd_candidate", "ncd_dormance(:,j)+dt"),
        ("ncd_max", "(/ncd_max/)"),
    ],
    1121: [
        ("hum_defined", "MERGE(1d0,0d0,hum_min_dormance(:,j)/=undef)"),
        ("hum_newmin", "MERGE(1d0,0d0,moiavail_month(:,j)<=hum_min_dormance(:,j))"),
    ],
    1311: [
        ("fpc_frac", "MERGE(1d0,0d0,fracnat>min_stomate)"),
        ("fpc_leaf", "MERGE(1d0,0d0,biomass(:,j,ileaf,icarbon)>lm_lastyearmax(:,j)*.75)"),
    ],
}
OBSERVERS.update(
    {
        666: [
            ("spring_broadleaf", "(/MERGE(un,zero,leaf_tab(j)==1)/)"),
            ("spring_summergreen", "(/MERGE(un,zero,pheno_type(j)==2)/)"),
        ],
        710: [
            ("spring_mtc8", "(/MERGE(un,zero,pft_to_mtc(j)==8)/)"),
            ("spring_mtc6", "(/MERGE(un,zero,pft_to_mtc(j)==6)/)"),
        ],
        852: [("critical_defined", "MERGE(un,zero,pheno_gdd_crit(j,:)/=undef)")],
        1275: [
            ("frac_nonnatural", "(/MERGE(un,zero,.NOT.natural(j))/)"),
            ("frac_pasture", "(/MERGE(un,zero,pasture(j))/)"),
        ],
        1309: [
            ("fpc_natural", "(/MERGE(un,zero,natural(j))/)"),
            ("fpc_dgvm", "(/MERGE(un,zero,ok_dgvm)/)"),
            ("fpc_nonpasture", "(/MERGE(un,zero,.NOT.pasture(j))/)"),
        ],
        1332: [
            ("lm_positive", "MERGE(un,zero,lm_thisyearmax(:,j)>min_stomate)"),
            ("lm_leaf", "MERGE(un,zero,biomass(:,j,ileaf,icarbon)>lm_thisyearmax(:,j)*.75)"),
            ("lm_raw", "lm_thisyearmax(:,j)"),
        ],
        1389: [("endyear", "(/MERGE(un,zero,EndOfYear)/)")],
        1420: [
            ("year_length_positive", "MERGE(un,zero,Tseason_length>zero)"),
            ("year_length", "Tseason_length"),
            ("year_sum", "Tseason_tmp"),
        ],
        1566: [
            ("herb_npp_positive", "MERGE(un,zero,nlflong_nat(:,j)>zero)"),
            ("herb_leaf_weight_positive", "MERGE(un,zero,weighttot(:,j)>zero)"),
            ("herb_npp", "nlflong_nat(:,j)"),
            ("herb_weight", "weighttot(:,j)"),
        ],
    }
)
for line, records in {
    670: [("spring_first_operand", "Tmin_spring_time(:,j)")],
    711: [("spring_second_operand", "Tmin_spring_time(:,j)")],
    884: [("gdd_temp", "t2m_daily"), ("gdd_state", "gdd_m5_dormance(:,j)")],
    954: [("mid_month", "t2m_month"), ("mid_week", "t2m_week"), ("mid_long", "t2m_longterm")],
    965: [("mid_state", "gdd_midwinter(:,j)"), ("mid_temp", "t2m_daily")],
    1020: [("ncd_state", "ncd_dormance(:,j)"), ("ncd_temp", "t2m_daily")],
    1121: [("hum_state", "hum_min_dormance(:,j)"), ("hum_month", "moiavail_month(:,j)")],
    1311: [
        ("fpc_fraction", "fracnat"),
        ("fpc_leaf_mass", "biomass(:,j,ileaf,icarbon)"),
        ("fpc_lastyear_mass", "lm_lastyearmax(:,j)"),
    ],
}.items():
    OBSERVERS[line].extend(records)


def original_unit(*, observed):
    template = legacy.SEASON_TEMPLATE.read_text(encoding="utf-8")
    header = template.split("program main")[0].replace("undef=1d20", "undef=-9999d0")
    header = header.replace("hvc1=0.3d0,hvc2=0.6d0", "hvc1=0.019d0,hvc2=1.38d0")
    source = SOURCE.read_text(encoding="utf-8")
    if observed:
        lines = source.splitlines(keepends=True)
        for line, observers in sorted(OBSERVERS.items(), reverse=True):
            assert lines[line - 1].strip().lower().startswith(("if", "where", "tau_longterm =", "t2m_longterm(:) ="))
            pft = "0" if line in (626, 633, 1389, 1420) else "j"
            lines.insert(
                line - 1,
                "".join(
                    f"call p06_obs('{name}',{expr.replace('MERGE(1d0,0d0,', 'MERGE(un,zero,')},{pft})\n"
                    for name, expr in observers
                ),
            )
        source = "".join(lines)
        marker = "  END SUBROUTINE season\n"
        helper = """
subroutine p06_obs(name,x,pft)
character(*),intent(in)::name
real(r_std),intent(in)::x(:)
integer,intent(in)::pft
integer::q
do q=1,size(x)
write(numout,'(A,A,A,I0,A,I0,A,ES26.17)')'P06,',trim(name),',',pft,',',q,',',x(q)
enddo
end subroutine p06_obs
"""
        assert source.count(marker) == 1
        source = source.replace(marker, marker + helper)
    solar = b"".join(legacy.SOLAR_SOURCE.read_bytes().splitlines(keepends=True)[289:597]).decode()
    header = header.replace("! <DOWNWARD_SOLAR_FLUX>", solar).replace("! <SEASON_MODULE>", source)
    decl = template.split("  integer,parameter::npts=2", 1)[1].split("  call get_command_argument(1,out)", 1)[0]
    program = (
        "program main\nuse defprec;use constantes;use stomate_data;use pft_parameters;use grid;use stomate_season\nimplicit none\ninteger,parameter::npts=8\n"
        + decl
    )
    program += "integer::inp\ncharacter(len=1024)::input_path\nlogical::endyear\n"
    program += "call get_command_argument(1,input_path)\ncall get_command_argument(2,out)\nopen(newunit=inp,file=trim(input_path),status='old')\n"
    program += "read(inp,*)endyear\n" + "".join(f"read(inp,*){name}\n" for name in ORDER) + "close(inp)\n"
    call = next(line for line in template.splitlines() if line.strip().startswith("call season(npts,1.0,.false."))
    program += call.replace(".false.", "endyear", 1) + "\n"
    program += "open(newunit=u,file=trim(out),status='replace');write(u,'(A)')'field,value'\n"
    outputs = {**MEMORY, **ANNUAL, **BIO, "gdd_init_date": "gid", "herbivores": "herb"}
    for name, var in outputs.items():
        expr = f"(/{var}/)" if name == "tau_longterm" else f"reshape({var},(/size({var})/))"
        program += f"call dump('{name}',{expr})\n"
    program += """close(u)
contains
subroutine dump(name,x)
character(*),intent(in)::name
real(r_std),intent(in)::x(:)
integer::q
do q=1,size(x)
write(u,'(A,",",ES26.17)')name,x(q)
enddo
end subroutine
end program
"""
    return header + program


def compile_references(output_dir):
    """Compile plain source and a prints-only observer; compare every return."""
    cases = declared_inputs()
    output_dir.mkdir(parents=True, exist_ok=True)
    refs, observations = {}, {}
    identities = {p.relative_to(ROOT).as_posix(): sha(p) for p in (SOURCE, legacy.SOLAR_SOURCE, legacy.SEASON_TEMPLATE)}
    for instrumented in (False, True):
        label = "observed" if instrumented else "plain"
        unit = original_unit(observed=instrumented)
        (output_dir / f"{label}-season.f90").write_text(unit, encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="phase06_season_") as td:
            src, exe = Path(td) / "oracle.f90", Path(td) / "oracle.exe"
            src.write_text(unit, encoding="utf-8")
            try:
                metadata = legacy._compile_legacy(src, exe, legacy.DEFAULT_COMPILER)
            except subprocess.CalledProcessError as error:
                save(
                    output_dir / f"{label}-compile-failure.json",
                    dict(command=error.cmd, stdout=error.stdout, stderr=error.stderr),
                )
                raise
            save(output_dir / f"{label}-build.json", metadata)
            for name, case in cases.items():
                folder = output_dir / name
                folder.mkdir(exist_ok=True)
                input_path = folder / "complete-input.txt"
                rows = ["T" if case["end_of_year"] else "F"]
                for key in ORDER:
                    arr = np.asarray(case[key])
                    rows.append(
                        " ".join("T" if v else "F" for v in arr.ravel(order="F"))
                        if arr.dtype.kind == "b"
                        else " ".join(str(v) for v in arr.ravel(order="F"))
                    )
                input_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
                csv = folder / f"{label}.csv"
                result = subprocess.run(
                    [str(exe), str(input_path.resolve()), str(csv.resolve())],
                    cwd=td,
                    env=legacy.compiler_environment(legacy.DEFAULT_COMPILER),
                    capture_output=True,
                    text=True,
                    check=True,
                )
                (folder / f"{label}.stdout.txt").write_text(result.stdout, encoding="utf-8")
                (folder / f"{label}.stderr.txt").write_text(result.stderr, encoding="utf-8")
                values = legacy._read_csv(csv)
                if not instrumented:
                    refs[name] = values
                else:
                    assert all(np.array_equal(refs[name][key], val) for key, val in values.items())
                    obs = {}
                    for line in result.stdout.splitlines():
                        if not line.startswith("P06,"):
                            continue
                        _, key, pft, _cell, val = line.split(",")
                        obs.setdefault(key + ".pft" + pft, []).append(float(val))
                    observations[name] = obs
    fixture = dict(
        schema_version=1,
        complete_inputs=cases,
        plain_original_outputs=refs,
        original_observations=observations,
        immutable_assets=identities,
        observer_policy="Original executable lines unchanged; added pure prints only; all plain/observer returns exactly equal.",
        rtol=1e-12,
        atol=1e-14,
    )
    lines = SOURCE.read_bytes().splitlines(keepends=True)
    fixture["condition_sources"] = {}
    for first in sorted(OBSERVERS):
        last = first
        while lines[last - 1].rstrip().endswith(b"&"):
            last += 1
        span = b"".join(lines[first - 1 : last])
        fixture["condition_sources"][str(first)] = dict(
            source_file=SOURCE.relative_to(ROOT).as_posix(),
            source_span=[first, last],
            source_sha256=sha(SOURCE),
            span_sha256=hashlib.sha256(span).hexdigest(),
            statement=span.decode().strip(),
            observation_names=[name for name, _expr in OBSERVERS[first]],
        )
    save(FIXTURE, fixture)
    return fixture


def consumer_join(values):
    """Actual eager/public compiled calls; operand evidence, not carbon oracle."""
    from jax_orchidee.stomate import integration
    from scripts.dev.oracle_phase05_growth_conditions import chain_inputs, run_current_chain

    inputs = chain_inputs()
    mappings = {
        "constraints_inputs": ("t2m_month", "tseason"),
        "alloc_inputs": ("moiavail_week", "tsoil_month", "soilhum_month"),
        "phenology_inputs": ("gdd_midwinter",),
        "post_npp_inputs": (
            "npp_longterm",
            "lm_lastyearmax",
            "tmin_spring_time",
            "maxmoiavail_lastyear",
            "minmoiavail_lastyear",
            "moiavail_week",
            "t2m_longterm",
            "t2m_month",
            "t2m_week",
            "gdd_from_growthinit",
        ),
    }
    for group, names in mappings.items():
        for name in names:
            inputs[group][name] = np.asarray(values[name])[:6].copy()
    inputs["post_npp_inputs"]["turnover_longterm"] = np.asarray(values["turnover_longterm"])[:6, ..., :1].copy()
    inputs["post_npp_inputs"]["herbivores"] = np.asarray(values["herbivores"])[:6].copy()
    with observe(integration, ("constraints_step", "phenology_step", "allocation_step", "turnover_step")) as calls:
        eager = run_current_chain(inputs)
    public = run_current_chain(inputs, public_wrapper=True)
    checks = []
    for owner, group in (
        ("constraints_step", "constraints_inputs"),
        ("phenology_step", "phenology_inputs"),
        ("allocation_step", "alloc_inputs"),
        ("turnover_step", "post_npp_inputs"),
    ):
        for name in mappings[group]:
            if name in calls[owner][0][0]:
                checks.append(compare(owner + ".operand." + name, inputs[group][name], calls[owner][0][0][name]))
    for name, a, b in (
        ("constraints.adapted", eager.constraints.adapted, public.constraints.adapted),
        ("phenology.gdd_midwinter", eager.phenology.gdd_midwinter, public.phenology.gdd_midwinter),
        ("allocation.f_alloc", eager.allocation.f_alloc, public.allocation.f_alloc),
        ("turnover.biomass", eager.post_npp.turnover.biomass, public.post_npp.turnover.biomass),
        ("final.lai", eager.post_npp.lai_after_setlai, public.post_npp.lai_after_setlai),
    ):
        checks.append(compare("public_compiled." + name, a, b))
    return dict(
        checks=checks,
        complete_inputs=inputs,
        observed_actual_calls=calls,
        compiled_returns=dict(
            adapted=public.constraints.adapted,
            f_alloc=public.allocation.f_alloc,
            biomass=public.post_npp.turnover.biomass,
            final_lai=public.post_npp.lai_after_setlai,
        ),
        scope="Actual operands and eager/public compiled returns; declared join is not an independent whole-carbon reference.",
    )


def run(output_dir=OUT):
    fixture = json.loads(FIXTURE.read_text())
    cases, last = {}, None
    for name, inputs in fixture["complete_inputs"].items():
        eager = current(inputs)
        compiled = current(inputs, compiled=True)
        expected = fixture["plain_original_outputs"][name]
        checks = []
        for field, value in eager.items():
            checks.append(compare("original." + field, np.asarray(expected[field]), np.asarray(value).ravel(order="F")))
            checks.append(
                compare(
                    "original_compiled." + field,
                    np.asarray(expected[field]),
                    np.asarray(compiled[field]).ravel(order="F"),
                )
            )
            checks.append(compare("compiled." + field, value, compiled[field]))
        cases[name] = dict(checks=checks, returns=eager)
        if name == "paper_tau_equal":
            last = compiled
    join = consumer_join(last)
    outside = ("diagnostic_frac_exact_min", "diagnostic_frac_valid_cover_min")
    failed = {name: [item for item in case["checks"] if not item["passed"]] for name, case in cases.items()}
    passed = all(not value for name, value in failed.items() if name not in outside) and all(
        c["passed"] for c in join["checks"]
    )
    status = (
        "complete_with_outside_scope_failures"
        if passed and any(failed.values())
        else "passed"
        if passed
        else "failed_preserved"
    )
    result = dict(
        status=status,
        fixed_paper_gate_passed=passed,
        outside_scope_strict_failures={name: failed[name] for name in outside},
        rtol=1e-12,
        atol=1e-14,
        cases=cases,
        consumer_join=join,
        fixture_sha256=sha(FIXTURE),
    )
    save(Path(output_dir) / "execution.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--compile-reference", action="store_true")
    args = parser.parse_args()
    if args.compile_reference:
        compile_references(args.output_dir)
    result = run(args.output_dir)
    print(result["status"])
    return 0 if result["fixed_paper_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
