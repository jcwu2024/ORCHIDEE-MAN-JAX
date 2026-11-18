"""PB-01 bounded original-Fortran canopy branch references and CPU checks.

No model step is run. Original DIFFUCO procedures, a literal firstcall block,
and original CONDVEG main/callees run in finite synthetic boundary harnesses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.dev import oracle_canopy_component_certify as old  # noqa: E402
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    COMPILE_FLAGS,
    DEFAULT_COMPILER,
    compiler_environment,
    float_comparison,
)
from scripts.dev.oracle_canopy_component_fixture import make_canopy_fixture as original_fixture  # noqa: E402

OUTPUT = ROOT / "outputs/reviews/main-physics-20261118/physics-branch-closeout/phase01-canopy-energy/canopy"
NAMES = (
    "water079",
    "water080",
    "water081",
    "water089",
    "water090",
    "water091",
    "water100",
    "lai_at_threshold",
    "lai_above_threshold",
    "growth_at_min",
    "growth_above_min",
    "growth_at_max",
    "growth_below_max",
    "radiation_at_mask",
    "radiation_above_mask",
    "pft_surface_cool",
    "humid_air",
    "low_pressure",
    "salinity_clipped",
    "salinity_unclipped",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return str(value)

    Path(path).write_text(json.dumps(data, indent=2, default=convert) + "\n", encoding="utf-8")


def make_canopy_fixture():
    """Twenty synthetic 14-PFT boundaries; no prior ten-case rerun."""
    base = original_fixture(indices=(0,))
    n = len(NAMES)
    payload = {k: np.repeat(v, n, axis=0) for k, v in base.diffuco_precall.payload.items()}
    payload["humrel"][:, 13] = 0.95
    payload["humrel"][:7, 13] = [0.79, 0.8, 0.81, 0.89, 0.9, 0.91, 1.0]
    payload["lai"][7:9, 13] = [0.001, 0.00100001]
    payload["temp_growth"][9:13] = [-5.0, -4.99999, 50.0, 49.99999]
    payload["swdown"][13:15] = [1e-8, 1.01e-8]
    payload["swnet"][13:15] = payload["swdown"][13:15] * 0.8
    payload["temp_sol_pft"][15, 13] = 291.0
    payload["qair"][16] = 0.019
    payload["pb"][17] = 850.0
    # Source firstcall salinity controls are separately validated below;
    # these source outputs then enter the actual C3 chain before GPP/ci.
    from jax_orchidee.sechiba.diffuco import mangrove_control_salinity

    payload["control_salinity"][18:] = np.asarray(
        mangrove_control_salinity(np.array([10.0, 30.0]), control_salinity_min=0.2)
    )
    precall = base.diffuco_precall._replace(payload=payload)
    return base._replace(
        case_names=NAMES,
        diffuco_precall=precall,
        driver={k: np.repeat(v, n, axis=0) for k, v in base.driver.items()},
        albedo={k: np.repeat(v, n, axis=0) for k, v in base.albedo.items()},
        thermal={k: np.repeat(v, n, axis=0) for k, v in base.thermal.items()},
    )


def run_diffuco(fixture=None, *, use_jit=False):
    from jax_orchidee.sechiba.diffuco import run_pft14_local_enerbil_precall_from_first_step_precall

    fixture = make_canopy_fixture() if fixture is None else fixture
    return run_pft14_local_enerbil_precall_from_first_step_precall(
        fixture.diffuco_precall, run_def_values={}, static_kwargs=fixture.static_kwargs, use_jit=use_jit
    )


def diffuco_values(result):
    after = result.result.payload.payload
    values = {
        "diffuco." + name: np.asarray(after[name]).ravel()
        for name in (
            "q_cdrag",
            "q_cdrag_pft",
            "raero",
            "qsatt",
            "gpp",
            "gsmean",
            "rveget",
            "rstruct",
            "cimean",
            "humrel",
            "vbeta",
            "vbeta_pft",
            "valpha",
            "vbeta1",
            "vbeta2",
            "vbeta3",
            "vbeta3pot",
            "vbeta4",
            "vbeta4_pft",
            "vbeta5",
        )
    }
    chain = result.result.boundary.process_chain
    values["diffuco.vbeta23"] = np.asarray(chain.inter.vbeta23).ravel()
    values["diffuco.cim"] = np.asarray(chain.trans_co2.canopy.cim).ravel()
    return values


def make_control_inputs():
    from jax_orchidee.stomate.carbon_kernels import IAGRHRTPN, IAGRHRTST, IAGRSAPPN, IAGRSAPST

    n, nvm = 8, 14
    biomass = np.zeros((n, nvm, 12, 1))
    for pool in (IAGRSAPST, IAGRHRTST):
        biomass[:, :, pool, 0] = np.array([0.001, 1.0, 10.0, 50.0, 100.0, 300.0, 600.0, 1000.0])[:, None] / 2
    for pool in (IAGRSAPPN, IAGRHRTPN):
        biomass[:, :, pool, 0] = np.array([0.0, 1.0, 5.0, 25.0, 50.0, 100.0, 300.0, 600.0])[:, None] / 2
    return dict(
        salinity=np.array([0.0, 10.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0]),
        z_soil=np.array([0.0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.3, 1.6, 2.0]),
        rprof=np.tile(np.linspace(0.15, 1.2, n)[:, None], (1, nvm)),
        tide_height=np.tile(np.array([-3.0, -0.2, -0.05, 0.0, 0.1, 2.0]), (n, 1)),
        biomass=biomass,
        control_salinity_min=0.2,
        control_inudate_min=0.1,
        agb_agr_ven_all_st=400.0,
        agb_agr_ven_all_pn=200.0,
        h_agr_max_st=1.5,
        h_agr_max_pn=0.4,
    )


def run_controls(inputs=None, *, use_jit=False):
    import jax

    from jax_orchidee.sechiba.diffuco import (
        ControlInundateInputs,
        mangrove_control_inundation,
        mangrove_control_salinity,
    )

    x = make_control_inputs() if inputs is None else inputs

    def calculate(salinity, z_soil, rprof, tide_height, biomass):
        result = mangrove_control_inundation(
            ControlInundateInputs(z_soil=z_soil, rprof=rprof, tide_height=tide_height, biomass=biomass),
            **{
                k: x[k]
                for k in (
                    "control_inudate_min",
                    "agb_agr_ven_all_st",
                    "agb_agr_ven_all_pn",
                    "h_agr_max_st",
                    "h_agr_max_pn",
                )
            },
        )
        return {
            "control_salinity": mangrove_control_salinity(salinity, control_salinity_min=x["control_salinity_min"]),
            "control_inudate": result.control_inudate,
            "frac_root_soil": result.frac_root_soil,
            "n_soil_inudate": result.n_soil_inudate,
            "frac_root_inudate": result.frac_root_inudate,
            "h_agr_st": result.agr.h_agr_st,
            "h_agr_pn": result.agr.h_agr_pn,
            "frac_root_ven_st": result.ventilation.frac_root_ven_st,
            "frac_root_ven_pn": result.ventilation.frac_root_ven_pn,
            "frac_root_ventilate": result.ventilation.frac_root_ventilate,
            "frac_root_anoxia": result.frac_root_anoxia,
        }

    function = jax.jit(calculate) if use_jit else calculate
    return {
        k: np.asarray(v).ravel()
        for k, v in function(**{k: x[k] for k in ("salinity", "z_soil", "rprof", "tide_height", "biomass")}).items()
    }


def make_condveg_inputs():
    n, nvm = 8, 14
    veg = np.zeros((n, nvm))
    vmax = np.zeros_like(veg)
    veg[:, 13] = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.0, 0.6]
    vmax[:, 13] = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.0, 0.7]
    vmax[:, 0] = 1 - 0.1 - vmax[:, 13]
    veg[:, 0] = vmax[:, 0]
    height = np.ones_like(veg)
    height[:, 13] = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 0.0, 15.0]
    lai = np.zeros_like(veg)
    lai[:, 13] = [0.0, 1e-8, 0.1, 0.2, 1.0, 5.0, 0.0, 7.0]
    pft = np.arange(1, nvm + 1)
    return dict(
        snow=np.array([0.0, 0.0, 0.0, 1e-8, 1.0, 10.0, 0.0, 30.0]),
        snow_nobio=np.array([[0.0], [0.0], [0.0], [1e-8], [1.0], [10.0], [0.0], [30.0]]),
        snowrho=np.full((n, 3), 250.0),
        snowdz=np.array(
            [[0.0, 0.0, 0.0]] * 4 + [[0.01, 0.02, 0.03], [0.1, 0.2, 0.3], [0.0, 0.0, 0.0], [0.3, 0.3, 0.3]]
        ),
        snow_age=np.array([0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 0.0, 20.0]),
        snow_nobio_age=np.array([[0.0], [0.0], [0.0], [0.0], [1.0], [5.0], [0.0], [20.0]]),
        veget=veg,
        veget_max=vmax,
        frac_nobio=np.full((n, 1), 0.1),
        totfrac_nobio=np.full(n, 0.1),
        zlev=np.full(n, 35.0),
        height=height,
        temp_air=np.linspace(270.0, 310.0, n),
        pb=np.linspace(850.0, 1013.0, n),
        u=np.array([0.0, 0.099, 0.101, 1.0, 2.0, 3.0, 0.0, 5.0]),
        v=np.zeros(n),
        lai=lai,
        emis_scal=1.0,
        impaze=False,
        rough_dyn=True,
        ok_explicitsnow=True,
        drysoil_frac=np.linspace(0.0, 1.0, n),
        tot_bare_soil=np.full(n, 0.2),
        alb_bg_modis=True,
        soilalb_bg=np.tile([0.15, 0.3], (n, 1)),
        alb_leaf_vis=0.08 + 0.001 * pft,
        alb_leaf_nir=0.25 + 0.002 * pft,
        snowa_aged_vis=np.full(nvm, 0.5),
        snowa_aged_nir=np.full(nvm, 0.3),
        snowa_dec_vis=np.full(nvm, 0.4),
        snowa_dec_nir=np.full(nvm, 0.3),
    )


def run_condveg(inputs=None, *, use_jit=False):
    from types import SimpleNamespace

    from jax_orchidee.sechiba.condveg import run_condveg_first_step_module

    x = make_condveg_inputs() if inputs is None else inputs
    values = {
        "ALB_BG_MODIS": "TRUE",
        "ALB_BARE_MODEL": "FALSE",
        "CONDVEG_SNOWA": "1e20",
        "TCST_SNOWA": "10",
        "ALB_ICE__00001": ".6",
        "ALB_ICE__00002": ".2",
    }
    for name in ("alb_leaf_vis", "alb_leaf_nir", "snowa_aged_vis", "snowa_aged_nir", "snowa_dec_vis", "snowa_dec_nir"):
        values.update({f"{name.upper()}__{j + 1:05d}": str(v) for j, v in enumerate(x[name])})
    restart = {
        k: x[k]
        for k in (
            "snow",
            "snow_nobio",
            "snowrho",
            "snowdz",
            "snow_age",
            "snow_nobio_age",
            "veget",
            "veget_max",
            "frac_nobio",
            "height",
            "lai",
        )
    }
    restart["soilalbedo_bg"] = x["soilalb_bg"]
    closure = run_condveg_first_step_module(
        restart_payload=restart,
        driver_payload={k: x[k] for k in ("zlev", "temp_air", "pb", "u", "v")},
        slowproc_payload={k: x[k] for k in ("totfrac_nobio", "tot_bare_soil")},
        hydrol_diagnostics=SimpleNamespace(drysoil_frac=x["drysoil_frac"]),
        run_def_values=values,
        snowcri_alb=10.0,
        min_wind=0.1,
        height_displacement=0.66,
        z0_ice=0.001,
        sn_dens=330.0,
        ok_explicitsnow=True,
        use_jit=use_jit,
    )
    if closure.missing_inputs:
        raise RuntimeError(closure.missing_inputs)
    result = closure.module
    return {k: np.asarray(getattr(result, k)).ravel() for k in result._fields}


def build_and_run(source_text, target):
    target.mkdir(parents=True, exist_ok=False)
    (target / "oracle.f90").write_text(source_text, encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="orchidee_pb01_canopy_") as temp:
        build = Path(temp)
        source = build / "oracle.f90"
        exe = build / "oracle.exe"
        source.write_text(source_text, encoding="utf-8")
        command = [str(DEFAULT_COMPILER), *COMPILE_FLAGS, "-cpp", str(source), "-o", str(exe)]
        started = time.perf_counter()
        process = subprocess.run(
            command, cwd=build, env=compiler_environment(DEFAULT_COMPILER), capture_output=True, text=True
        )
        save(
            target / "compile.json",
            dict(
                command=command,
                returncode=process.returncode,
                elapsed_s=time.perf_counter() - started,
                stdout=process.stdout,
                stderr=process.stderr,
                source_sha256=sha(source),
            ),
        )
        if process.returncode:
            raise RuntimeError(f"Fortran compile failed: {target}")
        command = [str(exe), str(target / "fortran_outputs.csv")]
        process = subprocess.run(
            command, cwd=build, env=compiler_environment(DEFAULT_COMPILER), capture_output=True, text=True
        )
        save(
            target / "run.json",
            dict(command=command, returncode=process.returncode, stdout=process.stdout, stderr=process.stderr),
        )
        if process.returncode:
            raise RuntimeError(f"Fortran run failed: {target}")


def emit(label, expression, shape):
    if len(shape) == 1:
        return f"do i=1,{shape[0]};write(10,'(a,i0,a,i0,a,es25.17)')'{label},',i,',',0,',',{expression};enddo"
    return f"do i=1,{shape[0]};do j=1,{shape[1]};write(10,'(a,i0,a,i0,a,es25.17)')'{label},',i,',',j,',',{expression};enddo;enddo"


def compose_controls():
    x = make_control_inputs()
    n = len(x["salinity"])
    nt = x["tide_height"].shape[1]
    nslm = len(x["z_soil"]) - 1
    raw = old.DIFFUCO.read_bytes().splitlines(keepends=True)
    # Preserve the exact source calculation; omit only the enclosing firstcall
    # guard/debug printing, which contain out-of-bounds historical tide prints.
    begin = next(i for i, line in enumerate(raw) if b"      !! z1) zhaoz: calculate control_salinity" in line)
    end = next(i for i, line in enumerate(raw) if b"      WRITE (numout,*) 'zhaoz: control_inudate_min'" in line)
    block = b"".join(raw[begin:end])
    header = f"""program oracle
 implicit none
 integer,parameter::r_std=kind(1.0),kjpindex={n},nvm=14,nslm={nslm},itimetide={nt},numout=6
 integer,parameter::iagrsapst=9,iagrhrtst=11,iagrsappn=10,iagrhrtpn=12,icarbon=1
 real(r_std),parameter::zero=0.d0,un=1.d0
 integer::ji,jv,jl,jt,ier,i,j
 real(r_std)::salinity(kjpindex),rprof(kjpindex,nvm),biomass(kjpindex,nvm,12,1),diaglev(nslm)
 real(r_std),allocatable::z_soil(:)
 real(r_std)::control_salinity(kjpindex),control_inudate(kjpindex),rpc(kjpindex)
 real(r_std)::frac_root_soil(kjpindex,nvm,nslm),tide_height(kjpindex,itimetide),tide_height_neg(kjpindex,itimetide),tide_height_pos(kjpindex,itimetide)
 integer::n_soil_inudate(kjpindex,itimetide)
 real(r_std)::frac_root_inudate(kjpindex,nvm,itimetide),frac_root_ven_st(kjpindex,nvm,itimetide),frac_root_ven_pn(kjpindex,nvm,itimetide)
 real(r_std)::frac_root_ventilate(kjpindex,nvm,itimetide),frac_root_anoxia(kjpindex,nvm,itimetide)
 real(r_std)::agb_agr_st(kjpindex,nvm),agb_agr_pn(kjpindex,nvm),h_agr_st(kjpindex,nvm),h_agr_pn(kjpindex,nvm)
 real(r_std)::control_salinity_min,control_inudate_min,agb_agr_ven_all_st,agb_agr_ven_all_pn,h_agr_max_st,h_agr_max_pn
 character(1024)::outpath
"""
    assignments = [
        old.assignment(k, v) if isinstance(v, np.ndarray) else f"{k}={old.literal(v)}"
        for k, v in x.items()
        if k != "z_soil"
    ]
    assignments.append(old.assignment("diaglev", x["z_soil"][1:]))
    tail = [
        "call get_command_argument(1,outpath);open(10,file=trim(outpath),status='replace');write(10,'(a)')'field,i,j,value'"
    ]
    for key, expression, shape in (
        ("control_salinity", "control_salinity(i)", (n,)),
        ("control_inudate", "control_inudate(i)", (n,)),
        ("frac_root_soil", "frac_root_soil(i,14,j)", (n, nslm)),
        ("n_soil_inudate", "real(n_soil_inudate(i,j),r_std)", (n, nt)),
        ("h_agr_st", "h_agr_st(i,14)", (n,)),
        ("h_agr_pn", "h_agr_pn(i,14)", (n,)),
    ):
        tail.append(emit(key, expression, shape))
    for key in ("frac_root_inudate", "frac_root_ven_st", "frac_root_ven_pn", "frac_root_ventilate", "frac_root_anoxia"):
        tail.append(emit(key, f"{key}(i,14,j)", (n, nt)))
    tail += [
        "close(10)",
        "contains",
        "subroutine ipslerr_p(level,a,b,c,d)",
        "integer::level;character(*)::a,b,c,d",
        "error stop 'ipslerr_p'",
        "end subroutine",
        "end program",
    ]
    identity = dict(
        source=str(old.DIFFUCO.relative_to(ROOT)),
        start_line=begin + 1,
        end_line=end,
        source_sha256=sha(old.DIFFUCO),
        span_sha256=hashlib.sha256(block).hexdigest(),
        kind="literal original firstcall calculation block",
    )
    return header + "\n".join(assignments) + "\n" + block.decode() + "\n" + "\n".join(tail), identity


def compose_condveg():
    x = make_condveg_inputs()
    n = len(x["snow"])
    template = (ROOT / "scripts/dev/oracle_lane_surface_condveg.f90.template").read_text(encoding="utf-8")
    prefix = (
        template.split("program oracle_main")[0]
        .replace(
            "soilalb_bg(3,2), soilalb_wet(3,2), soilalb_dry(3,2), soilalb_moy(3,2)",
            f"soilalb_bg({n},2), soilalb_wet({n},2), soilalb_dry({n},2), soilalb_moy({n},2)",
        )
        .replace("emis_scal=.98d0", "emis_scal=1.d0")
    )
    source = ROOT / "fortran_source/ORCHIDEE/src_sechiba/condveg.f90"
    procs = ("condveg_frac_snow", "condveg_albedo", "condveg_z0cdrag", "condveg_z0cdrag_dyn", "condveg_main")
    spans = {k: extract_procedure_bytes(source, k) for k in procs}
    prefix = prefix.replace("! <CONDVEG_PROCEDURES>", b"\n\n".join(s.span_bytes for s in spans.values()).decode())
    prefix = prefix.replace(
        "  implicit none\ncontains\n", "  implicit none\n  real(r_std),allocatable::pb01_snow_mean(:)\ncontains\n", 1
    )
    prefix = prefix.replace(
        "character(*),intent(in)::name; real(r_std),intent(in)::field(:)\n  end subroutine",
        "character(*),intent(in)::name; real(r_std),intent(in)::field(:)\n    if(name=='albedo_snow') pb01_snow_mean=field\n  end subroutine",
        1,
    )
    module_inputs = (
        "soilalb_bg",
        "alb_leaf_vis",
        "alb_leaf_nir",
        "snowa_aged_vis",
        "snowa_aged_nir",
        "snowa_dec_vis",
        "snowa_dec_nir",
    )
    arrays = {k: v for k, v in x.items() if isinstance(v, np.ndarray)}
    lines = [
        "program oracle_main",
        "use defprec;use surface_parameters;use condveg_oracle",
        "implicit none",
        f"integer,parameter::n={n}",
        "integer::i,j,index(n),neighbours(n,NbNeighb)",
        "character(1024)::outpath",
    ]
    lines += [f"real(r_std)::{k}({','.join(map(str, v.shape))})" for k, v in arrays.items() if k not in module_inputs]
    fields = dict(
        frac_snow_veg=(n,),
        frac_snow_nobio=(n, 1),
        emis=(n,),
        z0m=(n,),
        z0h=(n,),
        roughheight=(n,),
        roughheight_pft=(n, 14),
        albedo=(n, 2),
        albedo_snow=(n, 2),
        alb_bare=(n, 2),
        alb_veget=(n, 2),
        albedo_snow_mean=(n,),
    )
    lines += [f"real(r_std)::{k}({','.join(map(str, v))})" for k, v in fields.items()]
    lines += ["real(r_std)::lalo(n,2),resolution(n,2),contfrac(n)"]
    lines += [old.assignment(k, v) for k, v in arrays.items()]
    lines += [
        "do i=1,n;index(i)=i;enddo;neighbours=0;lalo=0.d0;resolution=1.d0;contfrac=1.d0",
        "call condveg_main(1,n,index,lalo,neighbours,resolution,contfrac,veget,veget_max,frac_nobio,totfrac_nobio,zlev,snow,snow_age,snow_nobio,snow_nobio_age,tot_bare_soil,temp_air,pb,u,v,lai,drysoil_frac,height,snowdz,snowrho,emis,albedo,frac_snow_veg,frac_snow_nobio,z0m,z0h,roughheight,roughheight_pft,-1,-1,-1)",
        # Main returns only albedo; obtain helper-owned diagnostics at the same input.
        "call condveg_albedo(n,veget,veget_max,drysoil_frac,frac_nobio,totfrac_nobio,snow,snow_age,snow_nobio,snow_nobio_age,snowdz,snowrho,tot_bare_soil,frac_snow_veg,frac_snow_nobio,albedo,albedo_snow,alb_bare,alb_veget)",
        "albedo_snow_mean=pb01_snow_mean",
        "call get_command_argument(1,outpath);open(10,file=trim(outpath),status='replace');write(10,'(a)')'field,i,j,value'",
    ]
    for k, shape in fields.items():
        lines.append(emit(k, f"{k}(i)" if len(shape) == 1 else f"{k}(i,j)", shape))
    lines += ["close(10)", "end program"]
    return prefix + "\n".join(lines), {
        k: dict(start_line=s.start_line, end_line=s.end_line, sha256=s.span_sha256, source_sha256=s.source_sha256)
        for k, s in spans.items()
    }


def reference():
    fixture = make_canopy_fixture()
    save(
        OUTPUT / "input_tuples.json",
        dict(
            canopy=dict(cases=NAMES, payload=fixture.diffuco_precall.payload, static=fixture.static_kwargs),
            controls=make_control_inputs(),
            condveg=make_condveg_inputs(),
        ),
    )
    old.make_canopy_fixture = make_canopy_fixture
    with tempfile.TemporaryDirectory(prefix="orchidee_pb01_compose_") as temp:
        path = Path(temp) / "all.f90"
        spans = old.compose(path)
        text = path.read_text(encoding="utf-8")
        # The original shared composer includes ENERBIL declarations/procedures;
        # stop before any ENERBIL call. No energy owner is executed here.
        text = text.split("allocate(psold(n)")[0] + "close(10)\nend program\n"
        reuse_or_build(text, OUTPUT / "reference_diffuco")
        save(OUTPUT / "reference_diffuco/source_spans.json", spans)
    text, identity = compose_controls()
    reuse_or_build(text, OUTPUT / "reference_controls")
    save(OUTPUT / "reference_controls/source_spans.json", identity)
    text, identity = compose_condveg()
    reuse_or_build(text, OUTPUT / "reference_condveg")
    save(OUTPUT / "reference_condveg/source_spans.json", identity)


def reuse_or_build(source, target):
    """Reuse only an exact completed reference, including generated constants."""
    if (target / "run.json").exists():
        run = json.loads((target / "run.json").read_text(encoding="utf-8"))
        assert run["returncode"] == 0
        assert (target / "oracle.f90").read_text(encoding="utf-8") == source
    else:
        build_and_run(source, target)


def expected_values():
    d = old.read_csv(OUTPUT / "reference_diffuco/fortran_outputs.csv")
    d = {k: v for k, v in d.items() if k not in ("diffuco.qsintmax", "diffuco.air_qsat", "diffuco.air_dqsat")}
    d["diffuco.cim"] = d["diffuco.cim"].reshape(len(NAMES), 14)[:, 13]
    d["diffuco.q_cdrag_pft"] = d["diffuco.q_cdrag_pft"].reshape(len(NAMES), 14)[:, 1:].ravel()
    c = old.read_csv(OUTPUT / "reference_condveg/fortran_outputs.csv")
    c["roughheight_pft"] = c["roughheight_pft"].reshape(8, 14)[:, 1:].ravel()
    return dict(diffuco=d, controls=old.read_csv(OUTPUT / "reference_controls/fortran_outputs.csv"), condveg=c)


def actual_values(*, use_jit):
    d = diffuco_values(run_diffuco(use_jit=use_jit))
    d["diffuco.q_cdrag_pft"] = d["diffuco.q_cdrag_pft"].reshape(len(NAMES), 14)[:, 1:].ravel()
    c = run_condveg(use_jit=use_jit)
    c["roughheight_pft"] = c["roughheight_pft"].reshape(8, 14)[:, 1:].ravel()
    return dict(diffuco=d, controls=run_controls(use_jit=use_jit), condveg=c)


def certify(use_jit):
    expected = expected_values()
    actual = actual_values(use_jit=use_jit)
    mode = "jit" if use_jit else "eager"
    np.savez(
        OUTPUT / f"actual_{mode}.npz", **{group + "." + k: v for group, vals in actual.items() for k, v in vals.items()}
    )
    comparisons = [
        float_comparison(group + "." + key, value, actual[group][key], rtol=1e-11, atol=1e-12)
        for group, fields in expected.items()
        for key, value in fields.items()
    ]
    return dict(
        status="passed" if all(c["passed"] for c in comparisons) else "failed",
        comparisons=comparisons,
        fields=len(comparisons),
        scalars=sum(c.get("count", 0) for c in comparisons),
        defined_coordinate_exclusions={
            "diffuco.q_cdrag_pft": "Original diffuco_aero never assigns PFT1; cim compares PFT14 only."
        },
    )


def export_fixture():
    expected = expected_values()
    save(
        ROOT / "tests/fixtures/physics_branch_canopy_reference.json",
        dict(
            schema_version=1,
            cases=NAMES,
            producer="Original Fortran literal extraction, PB-01 2026-11-19",
            rtol=1e-11,
            atol=1e-12,
            input_tuples_sha256=sha(OUTPUT / "input_tuples.json"),
            inputs=json.loads((OUTPUT / "input_tuples.json").read_text(encoding="utf-8")),
            reference_identity={
                g: dict(
                    csv_sha256=sha(OUTPUT / f"reference_{g}/fortran_outputs.csv"),
                    spans=json.loads((OUTPUT / f"reference_{g}/source_spans.json").read_text(encoding="utf-8")),
                )
                for g in expected
            },
            expected=expected,
        ),
    )


def diagnose_low_light():
    """Keep original scientific expressions; add observational layer writes."""
    source = (OUTPUT / "reference_diffuco/oracle.f90").read_text(encoding="utf-8")
    anchor = "                  leaf_ci(icinic,jv,jl) = Cc(icinic,jl) + assimi(icinic,jl) / gm(icinic)"
    assert source.count(anchor) == 1
    observation = """
                  if(icinic==15.and.jv==14) write(*,'(a,i0,1x,12(es25.17,1x))') &
                    'LAYER ',jl,vc2(icinic,jl),JJ(icinic,jl),Rd(icinic,jl),assimi(icinic,jl), &
                    x1,x2,Cc(icinic,jl),leaf_ci(icinic,jv,jl),p,q,r,assimi(icinic,jl)+Rd(icinic,jl)
"""
    build_and_run(source.replace(anchor, anchor + observation), OUTPUT / "diagnostic_low_light")
    run = json.loads((OUTPUT / "diagnostic_low_light/run.json").read_text(encoding="utf-8"))
    lines = [line for line in run["stdout"].splitlines() if line.startswith("LAYER ")]
    array = np.array([[float(v) for v in line.split()[1:]] for line in lines])
    columns = ["layer", "vc2", "jj", "rd", "assimi", "x1", "x2", "cc", "leaf_ci", "p", "q", "r", "net_plus_rd"]
    save(OUTPUT / "diagnostic_low_light/fortran_layers.json", dict(columns=columns, data=array))
    result = {}
    for mode in ("eager", "jit"):
        owner = run_diffuco(use_jit=mode == "jit")
        canopy = owner.result.boundary.process_chain.trans_co2.canopy
        actual = {
            k: np.asarray(getattr(canopy, k))[14, : len(array)] for k in ("vc2", "jj", "rd", "assimi", "cc", "leaf_ci")
        }
        actual["net_plus_rd"] = actual["assimi"] + actual["rd"]
        result[mode] = {
            k: dict(fortran=array[:, columns.index(k)], jax=v, absolute_error=np.abs(array[:, columns.index(k)] - v))
            for k, v in actual.items()
        }
    save(OUTPUT / "diagnostic_low_light/analysis.json", result)
    return dict(status="passed", meaning="observations captured; strict cim comparison remains failed")


def diagnose_cubic():
    """Give original scalar cubic and JAX exactly identical saved p/q/r."""
    from jax import jit

    from jax_orchidee.sechiba.diffuco import _yin_cubic_lowest_root

    layers = json.loads((OUTPUT / "diagnostic_low_light/fortran_layers.json").read_text(encoding="utf-8"))
    array = np.asarray(layers["data"])
    pqr = {k: array[:, layers["columns"].index(k)] for k in ("p", "q", "r")}
    raw = old.DIFFUCO.read_bytes().splitlines(keepends=True)
    start = next(i for i, line in enumerate(raw) if i > 2700 and b"QQ = ( (p**2._r_std)" in line)
    end = next(i for i, line in enumerate(raw) if i > start and b"IF (( A_1_tmp .LT. A_1 ))" in line)
    block = b"".join(raw[start:end])
    lines = [
        "program cubic",
        "implicit none",
        "integer,parameter::r_std=kind(1.0)",
        f"integer,parameter::n={len(array)}",
        "integer::i,j",
        "real(r_std)::p,q,r,QQ,UU,PSI,A_1_tmp",
        "real(r_std)::pp(n),pq(n),pr(n),roots(n)",
        "character(1024)::outpath",
    ]
    lines += [old.assignment("p" + k, v) for k, v in pqr.items()]
    lines += [
        "do i=1,n",
        "p=pp(i);q=pq(i);r=pr(i);A_1_tmp=9999.d0",
        block.decode(),
        "endif",
        "roots(i)=A_1_tmp",
        "enddo",
        "call get_command_argument(1,outpath);open(10,file=trim(outpath),status='replace');write(10,'(a)')'field,i,j,value'",
        emit("root", "roots(i)", (len(array),)),
        "close(10)",
        "end program",
    ]
    target = OUTPUT / "diagnostic_cubic"
    build_and_run("\n".join(lines), target)
    truth = old.read_csv(target / "fortran_outputs.csv")["root"]
    result = {
        mode: dict(fortran=truth, jax=np.asarray(fn(**pqr)[0]), absolute_error=np.abs(truth - np.asarray(fn(**pqr)[0])))
        for mode, fn in (("eager", _yin_cubic_lowest_root), ("jit", jit(_yin_cubic_lowest_root)))
    }
    save(
        target / "identical_coefficients.json",
        dict(
            pqr=pqr,
            results=result,
            span=dict(
                source=str(old.DIFFUCO.relative_to(ROOT)),
                start_line=start + 1,
                end_line=end,
                sha256=hashlib.sha256(block).hexdigest(),
                source_sha256=sha(old.DIFFUCO),
            ),
        ),
    )
    return dict(status="passed", meaning="same coefficients cubic observations; no relaxation of strict cim gate")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("reference", "eager", "jit", "export", "diagnose", "cubic"))
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    receipt = dict(command=[sys.executable, *sys.argv], phase=args.phase)
    try:
        import jax

        from jax_orchidee.sechiba import diffuco

        receipt.update(
            backend=jax.default_backend(), x64=bool(jax.config.jax_enable_x64), module_import_path=diffuco.__file__
        )
        assert receipt["backend"] == "cpu" and receipt["x64"]
        receipt["sources"] = {
            str(p.relative_to(ROOT)): sha(p)
            for p in (
                Path(__file__),
                ROOT / "jax_orchidee/sechiba/diffuco.py",
                ROOT / "jax_orchidee/sechiba/condveg.py",
            )
        }
        if args.phase == "reference":
            reference()
            result = dict(status="passed")
        elif args.phase == "export":
            export_fixture()
            result = dict(status="passed")
        elif args.phase == "diagnose":
            result = diagnose_low_light()
        elif args.phase == "cubic":
            result = diagnose_cubic()
        else:
            result = certify(args.phase == "jit")
        receipt.update(result=result, exit_code=int(result["status"] != "passed"))
    except Exception:
        receipt.update(exit_code=1, exception=traceback.format_exc())
    receipt["elapsed_s"] = time.perf_counter() - started
    save(OUTPUT / f"receipt_{args.phase}.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("phase", "exit_code", "elapsed_s")}))
    if receipt.get("exception"):
        print(receipt["exception"])
    return receipt["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
