"""Fixed 10-point PFT14 component gate against original Fortran procedures.

The harness runs the local process sequence, not SECHIBA or a calendar driver.
Scientific procedures are extracted byte-for-byte. Only IO is stubbed. Run
reference once, then eager/JIT against that immutable CSV; old oracles are
identified and reused without rerunning them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
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
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    COMPILE_FLAGS,
    DEFAULT_COMPILER,
    compiler_environment,
    float_comparison,
)
from scripts.dev.oracle_canopy_component_fixture import make_canopy_fixture, run_canopy_components  # noqa: E402

OUTPUT = ROOT / "outputs/reviews/main-physics-20261117/hydrol-diffuco-state-closure/canopy"
DIFFUCO = ROOT / "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90"
ENERBIL = ROOT / "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90"
QSAT = ROOT / "fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90"
DIFF_PROCS = (
    "diffuco_aero",
    "diffuco_snow",
    "diffuco_flood",
    "diffuco_inter",
    "diffuco_trans_co2",
    "diffuco_bare",
    "diffuco_comb",
    "Arrhenius",
    "Arrhenius_modified_0d",
    "Arrhenius_modified_1d",
)
ENER_PROCS = (
    "enerbil_main",
    "enerbil_begin",
    "enerbil_surftemp",
    "enerbil_pottemp",
    "enerbil_flux",
    "enerbil_evapveg",
    "enerbil_t2mdiag",
)
Q_PROCS = ("qsfrict_init", "qsatcalc", "dev_qsatcalc")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def literal(value: float) -> str:
    return f"{float(value):.17e}".replace("e", "d")


def assignment(name: str, array) -> str:
    array = np.asarray(array)
    values = ", &\n".join(
        ",".join(literal(v) for v in array.ravel(order="F")[start : start + 5]) for start in range(0, array.size, 5)
    )
    return f"{name}=reshape((/ &\n{values} /),shape({name}))"


def compose(path: Path) -> dict:
    fixture = make_canopy_fixture()
    payload = fixture.diffuco_precall.payload
    n = len(fixture.case_names)
    spans = {
        name: extract_procedure_bytes(source, name)
        for source, names in ((DIFFUCO, DIFF_PROCS), (ENERBIL, ENER_PROCS), (QSAT, Q_PROCS))
        for name in names
    }
    # Keep the existing audited scaffolding constants and IO interfaces. Add
    # only declarations required by the original full C3 routine and ENERBIL.
    surface = (ROOT / "scripts/dev/diffuco_surface_exchange_oracle_harness.f90.template").read_text()
    prefix = surface.split("module diffuco_oracle", 1)[0]
    prefix = prefix.replace(
        "module IOIPSL\n  implicit none",
        "module IOIPSL\n  use defprec\n  implicit none\n  interface histwrite_p\n    module procedure histwrite_1d,histwrite_2d\n  end interface",
    )
    prefix = prefix.replace(
        "end module\nmodule constantes",
        """  subroutine histwrite_1d(id,name,it,field,n,index)
    integer,intent(in)::id,it,n,index(:)
    character(*),intent(in)::name
    real(r_std),intent(in)::field(:)
  end subroutine
  subroutine histwrite_2d(id,name,it,field,n,index)
    integer,intent(in)::id,it,n,index(:)
    character(*),intent(in)::name
    real(r_std),intent(in)::field(:,:)
  end subroutine
end module
module constantes""",
        1,
    )
    prefix = prefix.replace(
        "  integer::printlev=0",
        """  integer::printlev=0
  integer,parameter::nlai=20,npco2=4,ivcmax=1,testpft=0
  real(r_std),parameter::pb_std=1013.d0,RR=8.314d0,pi=3.1415926535897932384626433832795d0
  real(r_std),parameter::Tetens_1=.622d0,Tetens_2=.378d0,ratio_H2O_to_CO2=1.6d0
  real(r_std),parameter::mol_to_m_1=.0244d0,RG_to_PAR=.5d0,W_to_mol=4.6d0,gb_ref=.04d0,Oi=210000.d0
  real(r_std),parameter::undef_sechiba=1.d20,quatre=4.d0
  real(r_std),parameter::c_stefan=5.6697d-8,chalev0=2.5008d6,chalsu0=2.8345d6,chalfu0=chalsu0-chalev0
  real(r_std)::laimax=8.d0,lai_level_depth=.15d0,downregulation_co2_baselevel=280.d0
  logical::downregulation_co2=.false.,ok_explicitsnow=.false.,almaoutput=.false.,OK_SECHIBA=.true.""",
    )
    prefix = prefix.replace(
        "(/1.d0,0.d0,0.d0,0.d0,0.d0,0.d0/)", "(/.887773d0,.205673d0,.110112d0,.014843d0,.000824d0,.000017d0/)"
    )
    prefix = prefix.replace(
        "module pft_parameters\n  implicit none", "module pft_parameters\n  use defprec\n  implicit none"
    )
    param_decls = "\n".join(
        f"  real(r_std)::{key}(nvm)={literal(value)}"
        for key, value in fixture.static_kwargs["trans_co2_pft_params"].items()
    )
    # C4 is inactive, but original source remains compiled, without substitutions.
    param_decls += "\n  real(r_std)::downregulation_co2_coeff(nvm)=0.d0,fpsir(nvm)=.4d0,fQ(nvm)=1.d0,h_protons(nvm)=4.d0,fpseudo(nvm)=0.d0,gbs(nvm)=.1d0,kp(nvm)=.1d0,alpha(nvm)=.1d0\n  logical::is_c4(nvm)=.false.\n"
    prefix = prefix.replace("  logical::ok_LAIdev(nvm)=.false.", "  logical::ok_LAIdev(nvm)=.false.\n" + param_decls)
    prefix = prefix.replace(
        "! <QSAT_PROCEDURES>", b"\n\n".join(spans[name].span_bytes for name in Q_PROCS).decode("utf-8")
    )
    diffmod = (
        """module diffuco_oracle
 use defprec;use constantes;use constantes_soil;use pft_parameters;use qsat_moisture;use IOIPSL
 implicit none
 integer,parameter::nnobio=2
 real(r_std),allocatable::wind(:),control_salinity(:),control_inudate(:),leaf_ci(:,:,:)
 interface Arrhenius_modified
  module procedure Arrhenius_modified_0d,Arrhenius_modified_1d
 end interface
contains
"""
        + b"\n\n".join(spans[name].span_bytes for name in DIFF_PROCS).decode("utf-8")
        + "\nend module\n"
    )
    energy_template = (ROOT / "scripts/dev/oracle_lane_surface_enerbil_main.f90.template").read_text()
    energymod = (
        "module enerbil_oracle"
        + energy_template.split("module enerbil_oracle", 1)[1].split("program oracle_main", 1)[0]
    )
    energymod = energymod.replace("use qsat_moisture", "use qsat_moisture,only:qsatcalc,dev_qsatcalc")
    energymod = energymod.replace(" implicit none", " use IOIPSL,only:numout,ipslerr_p\n implicit none", 1)
    energymod = energymod.replace(
        "! <ENERBIL_PROCEDURES>", b"\n\n".join(spans[name].span_bytes for name in ENER_PROCS).decode("utf-8")
    )
    # Program aliases only the scientific entry and private work arrays needed
    # to allocate the real ENERBIL module, avoiding namespace collisions.
    energy_names = (
        "enerbil_main",
        "psold",
        "psold_pft",
        "qsol_sat",
        "qsol_sat_pft",
        "pdqsold",
        "pdqsold_pft",
        "psnew",
        "psnew_pft",
        "qsol_sat_new",
        "qsol_sat_new_pft",
        "netrad",
        "netrad_pft",
        "lwabs",
        "lwup",
        "lwnet",
        "fluxsubli",
        "qsat_air",
        "tair",
        "q_sol_pot",
        "temp_sol_pot",
    )
    lines = [
        "program oracle_main",
        " use defprec;use constantes;use constantes_soil;use pft_parameters;use diffuco_oracle",
        " use enerbil_oracle,only:" + ",".join(energy_names),
        " implicit none",
        f" integer,parameter::n={n}",
        " integer::i,j,index(n),indexveg(n*nvm),indexlai(n*(nlai+1)),indexlai0(n*nlai)",
        " character(1024)::outpath",
    ]
    skip = {"control_salinity", "control_inudate", "rstruct"}
    for name, array in payload.items():
        if name not in skip:
            lines.append(f" real(r_std)::{name}({','.join(str(d) for d in array.shape)})")
    gridfields = (
        "q_cdrag",
        "qsatt",
        "vbeta1",
        "vbeta4",
        "vbeta5",
        "valpha",
        "vbeta",
        "raero",
        "air_qsat",
        "air_dqsat",
        "epot",
        "peta",
        "petb",
        "peqa",
        "peqb",
        "emis",
        "soilcap",
        "soilflx",
        "lwdown",
        "rain",
        "fluxsens",
        "fluxlat",
        "vevapp",
        "vevapnu",
        "vevapsno",
        "vevapflo",
        "t2m",
        "tsol_rad",
        "temp_sol_new",
        "temp_sol_add",
        "pgflux",
    )
    pftfields = (
        "q_cdrag_pft",
        "vbeta2",
        "vbeta23",
        "vbeta3",
        "vbeta3pot",
        "vbeta4_pft",
        "vbeta_pft",
        "gpp",
        "gsmean",
        "rveget",
        "rstruct",
        "cimean",
        "cim",
        "soilcap_pft",
        "soilflx_pft",
        "transpir",
        "transpot",
        "vevapwet",
        "vevapnu_pft",
        "temp_sol_new_pft",
    )
    lines += [
        " real(r_std)::" + ",".join(f"{name}(n)" for name in gridfields),
        " real(r_std)::" + ",".join(f"{name}(n,nvm)" for name in pftfields),
        " real(r_std)::lalo(n,2)",
        " allocate(wind(n),control_salinity(n),control_inudate(n),leaf_ci(n,nvm,nlai))",
    ]
    for name, array in payload.items():
        lines.append(assignment(name, array))
    lines += [
        "wind=sqrt(u*u+v*v);ok_LAIdev=.false.;ok_LAIdev(14)=.true.;ok_snowfact=.true.;rough_dyn=.true.",
        "do i=1,n;index(i)=i;enddo",
        "do i=1,n*nvm;indexveg(i)=i;enddo",
        "do i=1,n*(nlai+1);indexlai(i)=i;enddo",
        "do i=1,n*nlai;indexlai0(i)=i;enddo",
        "lalo=0.d0",
        "call diffuco_aero(n,1,u,v,zlev,z0h,z0m,roughheight,roughheight_pft,temp_sol,temp_sol_pft,temp_air,qsurf,qair,snow,q_cdrag,q_cdrag_pft)",
        "raero=1.d0/(max(min_wind,wind)*q_cdrag)",
        "call qsatcalc(n,temp_sol,pb,qsatt)",
        "call qsatcalc(n,temp_air,pb,air_qsat)",
        "call dev_qsatcalc(n,temp_air,pb,air_dqsat)",
        "call diffuco_snow(n,qair,qsatt,rau,u,v,q_cdrag,snow,frac_nobio,totfrac_nobio,snow_nobio,frac_snow_veg,frac_snow_nobio,vbeta1)",
        "call diffuco_flood(n,qair,qsatt,rau,u,v,q_cdrag,evapot,evapot_corr,flood_frac,flood_res,vbeta5)",
        "call diffuco_inter(n,qair,qsatt,rau,u,v,q_cdrag,q_cdrag_pft,humrel,veget,qsintveg,qsintmax,rstruct,vbeta2,vbeta23)",
        "call diffuco_trans_co2(n,lalo,swdown,pb,qsurf,qair,temp_air,temp_growth,rau,u,v,q_cdrag,q_cdrag_pft,humrel,assim_param,ccanopy,veget,veget_max,lai,qsintveg,qsintmax,vbeta3,vbeta3pot,rveget,rstruct,cimean,gsmean,gpp,vbeta23,1,indexveg,indexlai,indexlai0,index,1,cim)",
        "call diffuco_bare(n,u,v,q_cdrag,[(100._r_std,i=1,n)],evap_bare_lim,humrel,veget,veget_max,tot_bare_soil,vbeta2,vbeta3,vbeta4,vbeta4_pft)",
        "call diffuco_comb(n,humrel,rau,u,v,q_cdrag,pb,qair,temp_sol,temp_air,snow,veget,veget_max,lai,tot_bare_soil,vbeta1,vbeta2,vbeta3,vbeta4,vbeta4_pft,valpha,vbeta,vbeta_pft,qsintmax)",
        "call get_command_argument(1,outpath);open(10,file=trim(outpath),status='replace');write(10,'(a)')'field,i,j,value'",
    ]
    diffouts = (
        "q_cdrag",
        "q_cdrag_pft",
        "raero",
        "qsatt",
        "air_qsat",
        "air_dqsat",
        "gpp",
        "gsmean",
        "rveget",
        "rstruct",
        "cimean",
        "cim",
        "humrel",
        "qsintmax",
        "vbeta",
        "vbeta_pft",
        "valpha",
        "vbeta1",
        "vbeta2",
        "vbeta23",
        "vbeta3",
        "vbeta3pot",
        "vbeta4",
        "vbeta4_pft",
        "vbeta5",
    )
    for name in diffouts:
        matrix = name in pftfields or name in {"humrel", "qsintmax"}
        lines += output_lines("diffuco." + name, name, matrix)
    lines += [
        "allocate(psold(n),psold_pft(n,nvm),qsol_sat(n),qsol_sat_pft(n,nvm),pdqsold(n),pdqsold_pft(n,nvm))",
        "allocate(psnew(n),psnew_pft(n,nvm),qsol_sat_new(n),qsol_sat_new_pft(n,nvm),netrad(n),netrad_pft(n,nvm))",
        "allocate(lwabs(n),lwup(n),lwnet(n),fluxsubli(n),qsat_air(n),tair(n),q_sol_pot(n),temp_sol_pot(n))",
        "epot=cp_air*temp_air+cte_grav*zlev;peta=0.d0;peqa=0.d0;petb=epot;peqb=qair;emis=1.d0",
        "soilcap=4.d5;soilcap_pft=4.d5;soilflx=10.d0;soilflx_pft=10.d0;lwdown=350.d0;rain=0.d0;pgflux=soilflx;tsol_rad=temp_sol;q_sol_pot=0.d0;temp_sol_pot=0.d0",
        "call enerbil_main(1,n,index,indexveg,zlev,lwdown,swnet,epot,temp_air,u,v,peta,petb,qair,peqa,peqb,pb,rau,vbeta,vbeta_pft,valpha,vbeta1,vbeta2,vbeta3,vbeta3pot,vbeta4,vbeta4_pft,vbeta5,emis,soilflx,soilflx_pft,soilcap,soilcap_pft,q_cdrag,q_cdrag_pft,veget_max,humrel,fluxsens,fluxlat,vevapp,transpir,transpot,vevapnu,vevapnu_pft,vevapwet,vevapsno,vevapflo,t2m,temp_sol,temp_sol_pft,tsol_rad,temp_sol_new,temp_sol_new_pft,qsurf,evapot,evapot_corr,-1,-1,-1,rain,snowdz,pgflux,temp_sol_add)",
    ]
    eouts = (
        "qsol_sat",
        "qsol_sat_pft",
        "pdqsold",
        "pdqsold_pft",
        "temp_sol_new",
        "temp_sol_new_pft",
        "qsurf",
        "evapot",
        "evapot_corr",
        "fluxsens",
        "fluxlat",
        "vevapp",
        "vevapnu",
        "vevapsno",
        "vevapflo",
        "transpir",
        "transpot",
        "vevapwet",
        "vevapnu_pft",
    )
    for name in eouts:
        lines += output_lines("enerbil." + name, name, name in pftfields or name.endswith("_pft"))
    lines += ["close(10)", "end program"]
    path.write_text(prefix + diffmod + energymod + "\n".join(lines) + "\n", encoding="utf-8")
    return {
        name: {
            "start_line": span.start_line,
            "end_line": span.end_line,
            "sha256": span.span_sha256,
            "source_sha256": span.source_sha256,
        }
        for name, span in spans.items()
    }


def output_lines(label: str, name: str, matrix: bool) -> list[str]:
    if matrix:
        return [f"do i=1,n;do j=1,nvm;write(10,'(a,i0,a,i0,a,es25.17)')'{label},',i,',',j,',',{name}(i,j);enddo;enddo"]
    return [f"do i=1,n;write(10,'(a,i0,a,i0,a,es25.17)')'{label},',i,',',0,',',{name}(i);enddo"]


def run_reference(output: Path) -> dict:
    target = output / "reference"
    target.mkdir(parents=True, exist_ok=False)
    fixture = make_canopy_fixture()
    np.savez(target / "inputs.npz", **fixture.diffuco_precall.payload)
    save(target / "cases.json", fixture.case_names)
    with tempfile.TemporaryDirectory(prefix="orchidee_canopy_component_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        source_spans = compose(source)
        (target / "oracle.f90").write_bytes(source.read_bytes())
        save(target / "source_spans.json", source_spans)
        exe = build / "oracle.exe"
        command = [str(DEFAULT_COMPILER), *COMPILE_FLAGS, "-cpp", str(source), "-o", str(exe)]
        started = time.perf_counter()
        built = subprocess.run(
            command, cwd=build, env=compiler_environment(DEFAULT_COMPILER), capture_output=True, text=True
        )
        result = dict(
            command=command,
            returncode=built.returncode,
            elapsed_s=time.perf_counter() - started,
            stdout=built.stdout,
            stderr=built.stderr,
        )
        save(target / "compile.json", result)
        if built.returncode:
            raise RuntimeError("original component compile failed; see reference/compile.json")
        command = [str(exe), str(target / "fortran_outputs.csv")]
        started = time.perf_counter()
        ran = subprocess.run(
            command, cwd=build, env=compiler_environment(DEFAULT_COMPILER), capture_output=True, text=True
        )
        run = dict(
            command=command,
            returncode=ran.returncode,
            elapsed_s=time.perf_counter() - started,
            stdout=ran.stdout,
            stderr=ran.stderr,
        )
        save(target / "run.json", run)
        if ran.returncode:
            raise RuntimeError("original component run failed; see reference/run.json")
    return dict(
        status="passed",
        fields=len(read_csv(target / "fortran_outputs.csv")),
        reference_sha256=sha(target / "fortran_outputs.csv"),
    )


def read_csv(path: Path) -> dict[str, np.ndarray]:
    values = {}
    for row in csv.DictReader(path.open(encoding="utf-8", newline="")):
        raw = row["value"].strip()
        # gfortran ES25.17 omits E for three-digit exponents. Preserve that
        # captured output without re-executing unchanged scientific routines.
        raw = re.sub(r"(?<=\d)([+-]\d{3})$", r"e\1", raw) if "e" not in raw.lower() else raw
        values.setdefault(row["field"], []).append(float(raw))
    return {name: np.asarray(value) for name, value in values.items()}


def current_values(components) -> dict[str, np.ndarray]:
    from jax_orchidee.sechiba.enerbil import qsat_moisture_dev_qsatcalc, qsat_moisture_qsatcalc

    after = components.diffuco.result.payload.payload
    chain = components.diffuco.result.boundary.process_chain
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
    values["diffuco.vbeta23"] = np.asarray(chain.inter.vbeta23).ravel()
    values["diffuco.cim"] = np.asarray(chain.trans_co2.canopy.cim).ravel()
    source = components.fixture.diffuco_precall.payload
    values["diffuco.air_qsat"] = np.asarray(qsat_moisture_qsatcalc(source["temp_air"], source["pb"]))
    values["diffuco.air_dqsat"] = np.asarray(qsat_moisture_dev_qsatcalc(source["temp_air"], source["pb"]))
    for name in (
        "qsol_sat",
        "qsol_sat_pft",
        "pdqsold",
        "pdqsold_pft",
        "temp_sol_new",
        "temp_sol_new_pft",
        "qsurf",
        "evapot",
        "evapot_corr",
        "fluxsens",
        "fluxlat",
        "vevapp",
        "vevapnu",
        "vevapsno",
        "vevapflo",
        "transpir",
        "transpot",
        "vevapwet",
        "vevapnu_pft",
    ):
        values["enerbil." + name] = np.asarray(components.enerbil.payload[name]).ravel()
    return values


def run_python(output: Path, *, use_jit: bool, actual_npz: Path | None = None) -> dict:
    mode = "jit" if use_jit else "eager"
    target = output / mode
    target.mkdir(parents=True, exist_ok=False)
    fixture = make_canopy_fixture()
    started = time.perf_counter()
    if actual_npz is None:
        result = run_canopy_components(fixture, diffuco_use_jit=use_jit, enerbil_use_jit=use_jit)
        actual = current_values(result)
    else:
        actual = dict(np.load(actual_npz))
    elapsed = time.perf_counter() - started
    np.savez(target / "actual.npz", **actual)
    reference = read_csv(output / "reference/fortran_outputs.csv")
    del reference["diffuco.qsintmax"]  # INTENT(in), not a local owner output.
    # Full fields compare only where both complete owners define them. The
    # single-PFT C3 owner exports cim only for PFT14, unlike its full Fortran
    # counterpart, so the independent cim check is explicitly that column.
    reference["diffuco.cim"] = reference["diffuco.cim"].reshape(len(fixture.case_names), 14)[:, 13]
    excluded = {}
    for name in ("diffuco.q_cdrag_pft", "enerbil.qsol_sat_pft", "enerbil.pdqsold_pft", "enerbil.temp_sol_new_pft"):
        reference[name] = reference[name].reshape(len(fixture.case_names), 14)[:, 1:].ravel()
        actual[name] = actual[name].reshape(len(fixture.case_names), 14)[:, 1:].ravel()
        excluded[name] = (
            "PFT1 has no original assignment: aero and ENERBIL begin/surftemp loop jv=2,nvm; no numerical claim for that slot."
        )
    comparisons = [
        float_comparison(name, expected, actual[name], rtol=1e-11, atol=1e-12) for name, expected in reference.items()
    ]
    result = dict(
        status="passed" if all(c["passed"] for c in comparisons) else "failed",
        actual_owner_call_elapsed_s=elapsed,
        fields=len(comparisons),
        scalars=sum(c.get("count", 0) for c in comparisons),
        comparisons=comparisons,
        reference_sha256=sha(output / "reference/fortran_outputs.csv"),
        component_call_count={"diffuco_wrapper": int(actual_npz is None), "enerbil_wrapper": int(actual_npz is None)},
        reused_actual_npz=str(actual_npz) if actual_npz else None,
        field_comparison_count=len(comparisons),
        pytest_count=0,
        excluded_coordinates=excluded,
        case_names=fixture.case_names,
    )
    save(target / "comparison.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("reference", "eager", "jit"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--reuse-reference", action="store_true")
    parser.add_argument("--actual-npz", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    receipt = dict(
        command=[sys.executable, *sys.argv],
        phase=args.phase,
        production={
            str(p.relative_to(ROOT)): sha(p)
            for p in (
                ROOT / "jax_orchidee/sechiba/diffuco.py",
                ROOT / "jax_orchidee/sechiba/enerbil.py",
                ROOT / "jax_orchidee/sechiba/science_completion.py",
            )
        },
        scripts={
            str(p.relative_to(ROOT)): sha(p)
            for p in (Path(__file__), ROOT / "scripts/dev/oracle_canopy_component_fixture.py")
        },
    )
    try:
        import jax

        from jax_orchidee.sechiba import diffuco as _x64_owner

        receipt["module_import_path"] = _x64_owner.__file__
        receipt["backend"] = jax.default_backend()
        receipt["jax_enable_x64"] = bool(jax.config.jax_enable_x64)
        if receipt["backend"] != "cpu" or not receipt["jax_enable_x64"]:
            raise RuntimeError("component certification requires CPU/x64")
        if args.phase == "reference" and args.reuse_reference:
            run = json.loads((args.output / "reference/run.json").read_text())
            if run["returncode"]:
                raise RuntimeError("cannot reuse failed original run")
            result = dict(
                status="passed",
                reused_completed_original_run=True,
                fields=len(read_csv(args.output / "reference/fortran_outputs.csv")),
                reference_sha256=sha(args.output / "reference/fortran_outputs.csv"),
            )
        else:
            result = (
                run_reference(args.output)
                if args.phase == "reference"
                else run_python(args.output, use_jit=args.phase == "jit", actual_npz=args.actual_npz)
            )
        receipt.update(result=result, exit_code=0 if result["status"] == "passed" else 1)
    except Exception:
        receipt.update(exit_code=1, exception=traceback.format_exc())
    receipt["elapsed_s"] = time.perf_counter() - started
    save(args.output / f"receipt_{args.phase}.json", receipt)
    print(json.dumps({key: receipt.get(key) for key in ("phase", "exit_code", "elapsed_s", "exception")}))
    return receipt["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
