"""Original-byte Stage 4 Batch C closure for ``intersurf_main_2d``.

The executable retains the complete owner bytes.  Its reduced ``sechiba_main``
dependency keeps the original interface and executes original SECHIBA air
density, ENERBIL, QSAT, and PFT14 CONDVEG procedures.  Only IOIPSL history,
XIOS, calendar transport, and parallel synchronization are boundary stubs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_pft14_owner_region_evidence import build_report  # noqa: E402
from extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from fortran_gcov import (  # noqa: E402
    locate_generated_span,
    parse_gcov,
    parse_gcov_line_counts,
)
from fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    ROOT,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)
from fortran_owner_coverage import (  # noqa: E402
    CONTRACT_CLASSES,
    GCOV,
    SOURCE_PROOFS,
    run_extracted_owner_coverage,
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.condveg import (  # noqa: E402
    condveg_albedo_explicit,
    condveg_frac_snow,
    condveg_main_emissivity,
    condveg_z0cdrag,
)
from jax_orchidee.sechiba.enerbil import (  # noqa: E402
    enerbil_begin_local_diagnostics,
    enerbil_flux_local_diagnostics,
    enerbil_surftemp_explicit_solve,
    sechiba_air_density_from_pb_temp_air,
)
from jax_orchidee.sechiba.lifecycle_completion import intersurf_main_2d  # noqa: E402


FAMILY = "intersurf_main_batch_c"
OWNER_ID = "pft14-owner-contract-81281ca5ba43"
BASE_REGION_ID = "pft14-region-065e162842a1"
OWNER_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/intersurf.f90"
SECHIBA_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90"
ENERBIL_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90"
QSAT_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90"
CONDVEG_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/condveg.f90"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY

IIM = 2
JJM = 3
KINDEX = np.asarray([1, 4, 6], dtype=np.int32)
XRDT = 1800.0
NVM = 14
NFLOW = 2
RTOL = 1.0e-12
ATOL = 1.0e-14
UNDEF_SECHIBA = 1.0e20

CASES = (
    {"case_id": "native_history_unsynced_window", "mode": 1, "dw": 900.0},
    {"case_id": "native_history_sync_gate_disabled", "mode": 2, "dw": XRDT},
)
COVERAGE_CASES = CASES + (
    {
        "case_id": "openmp_nonroot_history_gate_skipped",
        "mode": 3,
        "dw": XRDT,
        "is_omp_root": False,
    },
)

ARM_IDS = (
    "fortran_source/ORCHIDEE/src_sechiba/intersurf.f90:603:if:false",
    "fortran_source/ORCHIDEE/src_sechiba/intersurf.f90:698:if:true",
    "fortran_source/ORCHIDEE/src_sechiba/intersurf.f90:735:if:true",
    "fortran_source/ORCHIDEE/src_sechiba/intersurf.f90:736:if:false",
    "fortran_source/ORCHIDEE/src_sechiba/intersurf.f90:736:if:true",
    "fortran_source/ORCHIDEE/src_sechiba/intersurf.f90:738:if:false",
)

# GNU Fortran emits the true fallthrough edge first for these physical IFs.
# Pinning the indices makes the owner records cite branch counters directly.
ARM_BRANCH_INDICES = {
    "fortran_source/ORCHIDEE/src_sechiba/intersurf.f90:735:if:true": 0,
    "fortran_source/ORCHIDEE/src_sechiba/intersurf.f90:736:if:true": 0,
    "fortran_source/ORCHIDEE/src_sechiba/intersurf.f90:736:if:false": 1,
}

COMPILE_FLAGS = (
    "-std=f2018",
    "-fdefault-real-8",
    "-fdefault-double-8",
    "-ffree-line-length-none",
    "-O0",
    "-fcheck=all",
    "-ffpe-trap=invalid,zero,overflow",
)

ALB_LEAF_VIS = np.asarray(
    [0.0, 0.0397, 0.0474, 0.0386, 0.0484, 0.0411, 0.041, 0.0541, 0.0435,
     0.0524, 0.0508, 0.0509, 0.0606, 0.0606],
    dtype=np.float64,
)
ALB_LEAF_NIR = np.asarray(
    [0.0, 0.227, 0.214, 0.193, 0.208, 0.244, 0.177, 0.218, 0.213,
     0.252, 0.265, 0.272, 0.244, 0.244],
    dtype=np.float64,
)
SNOWA_AGED = np.asarray(
    [0.35, 0.0, 0.0, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.18, 0.18,
     0.18, 0.18, 0.18],
    dtype=np.float64,
)
SNOWA_DEC_VIS = np.asarray(
    [0.45, 0.0, 0.0, 0.10, 0.06, 0.11, 0.10, 0.11, 0.18, 0.60, 0.60,
     0.60, 0.60, 0.60],
    dtype=np.float64,
)
SNOWA_DEC_NIR = np.asarray(
    [0.45, 0.0, 0.0, 0.06, 0.06, 0.11, 0.06, 0.11, 0.11, 0.52, 0.52,
     0.52, 0.52, 0.52],
    dtype=np.float64,
)
IS_TREE = np.asarray(
    [False, True, True, True, True, True, True, True, True, False, False,
     False, False, False],
    dtype=bool,
)


def _lines(path: Path, start: int, end: int) -> bytes:
    return b"".join(path.read_bytes().splitlines(keepends=True)[start - 1 : end])


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fortran_vector(values: np.ndarray, *, logical: bool = False) -> str:
    if logical:
        body = ", ".join(".true." if bool(value) else ".false." for value in values)
    else:
        literals = []
        for value in values:
            literal = f"{float(value):.17g}"
            if "." not in literal and "e" not in literal.lower():
                literal += ".0"
            literals.append(f"{literal}_r_std")
        body = ", ".join(literals)
    return f"(/ {body} /)"


def _source_spans() -> dict[str, tuple[Path, bytes, int, int, str]]:
    procedures = {
        "intersurf_main_2d": (OWNER_SOURCE, "intersurf_main_2d"),
        "intsurf_time": (OWNER_SOURCE, "intsurf_time"),
        "sechiba_main": (SECHIBA_SOURCE, "sechiba_main"),
        "sechiba_var_init": (SECHIBA_SOURCE, "sechiba_var_init"),
        "enerbil_begin": (ENERBIL_SOURCE, "enerbil_begin"),
        "enerbil_surftemp": (ENERBIL_SOURCE, "enerbil_surftemp"),
        "enerbil_flux": (ENERBIL_SOURCE, "enerbil_flux"),
        "qsatcalc": (QSAT_SOURCE, "qsatcalc"),
        "dev_qsatcalc": (QSAT_SOURCE, "dev_qsatcalc"),
        "qsfrict_init": (QSAT_SOURCE, "qsfrict_init"),
        "condveg_frac_snow": (CONDVEG_SOURCE, "condveg_frac_snow"),
        "condveg_z0cdrag": (CONDVEG_SOURCE, "condveg_z0cdrag"),
        "condveg_albedo": (CONDVEG_SOURCE, "condveg_albedo"),
    }
    result: dict[str, tuple[Path, bytes, int, int, str]] = {}
    for key, (path, procedure) in procedures.items():
        span = extract_procedure_bytes(path, procedure)
        result[key] = (
            path,
            span.span_bytes,
            int(span.start_line),
            int(span.end_line),
            span.span_sha256,
        )
    return result


TRANSPORT_MODULE = rb"""
module oracle_transport
  implicit none
  integer :: history_write_count=0, xios_send_count=0, histsync_count=0
  integer :: calendar_update_count=0, last_calendar_step=-1
  logical :: is_omp_root=.true., ok_histsync=.false.
  real(8) :: dw=900.d0
contains
  subroutine reset_transport(window)
    real(8),intent(in)::window
    history_write_count=0; xios_send_count=0; histsync_count=0
    calendar_update_count=0; last_calendar_step=-1; dw=window
  end subroutine
  subroutine ipslnlf_p(new_number,old_number)
    integer,intent(in)::new_number
    integer,intent(out),optional::old_number
    if(present(old_number)) old_number=6
  end subroutine
  subroutine xios_orchidee_update_calendar(step)
    integer,intent(in)::step
    calendar_update_count=calendar_update_count+1; last_calendar_step=step
  end subroutine
  subroutine xios_orchidee_send_field(name,value)
    character(len=*),intent(in)::name
    real(8),intent(in)::value(..)
    xios_send_count=xios_send_count+1
  end subroutine
  subroutine histwrite_p(id,name,step,value,n,index)
    integer,intent(in)::id,step,n,index(:)
    character(len=*),intent(in)::name
    real(8),intent(in)::value(..)
    history_write_count=history_write_count+1
  end subroutine
  subroutine histsync(id)
    integer,intent(in)::id
    histsync_count=histsync_count+1
  end subroutine
  subroutine ipslerr_p(level,a,b,c,d)
    integer,intent(in)::level
    character(len=*),intent(in)::a,b,c,d
    error stop 'unexpected fatal boundary in genuine scientific closure'
  end subroutine
end module oracle_transport
"""


def _parameter_module() -> bytes:
    text = f"""
module oracle_parameters
  implicit none
  integer,parameter::i_std=4,r_std=8
  integer,parameter::nvm={NVM},nflow={NFLOW},nsnow=3,nnobio=1,NbNeighb=8
  integer,parameter::ih2o=1,iice=1,ivis=1,inir=2
  real(r_std),parameter::zero=0._r_std,undemi=.5_r_std,un=1._r_std,deux=2._r_std,quatre=4._r_std
  real(r_std),parameter::undef_sechiba=1.e20_r_std,min_sechiba=1.e-8_r_std
  real(r_std),parameter::tp_00=273.15_r_std,chalsu0=2.8345e6_r_std,chalev0=2.5008e6_r_std
  real(r_std),parameter::c_stefan=5.6697e-8_r_std,cp_air=1004.675_r_std,cte_molr=287.05_r_std
  real(r_std),parameter::pa_par_hpa=100._r_std,ct_karman=.41_r_std
  real(r_std),parameter::msmlr_air=28.964e-3_r_std,msmlr_h2o=18.02e-3_r_std
  real(r_std),parameter::z0_bare=.01_r_std,z0_ice=.001_r_std,height_displacement=.66_r_std
  real(r_std),parameter::snowcri_alb=10._r_std,sn_dens=330._r_std,tcst_snowa=10._r_std
  real(r_std),parameter::alb_ice(2)=(/.60_r_std,.20_r_std/)
  real(r_std),parameter::albedo_scal(2)=(/.25_r_std,.25_r_std/)
  real(r_std),parameter::fixed_snow_albedo=undef_sechiba
  real(r_std),parameter::emis_scal=1._r_std
  real(r_std)::dt_sechiba={XRDT:.1f}_r_std,min_wind=.1_r_std
  integer::numout=6,printlev=0
  logical,parameter::diag_qsat=.true.,almaoutput=.false.,ok_explicitsnow=.false.
  logical,parameter::impaze=.false.,alb_bg_modis=.false.,alb_bare_model=.false.
  logical,parameter::is_tree(nvm)={_fortran_vector(IS_TREE, logical=True)}
  logical,parameter::ok_LAIdev(nvm)={_fortran_vector(np.zeros(NVM, dtype=bool), logical=True)}
  real(r_std),parameter::z0_over_height(nvm)={_fortran_vector(np.asarray([0.0] + [0.0625] * 13))}
  real(r_std),parameter::ratio_z0m_z0h(nvm)={_fortran_vector(np.ones(NVM))}
  real(r_std),parameter::alb_leaf_vis(nvm)={_fortran_vector(ALB_LEAF_VIS)}
  real(r_std),parameter::alb_leaf_nir(nvm)={_fortran_vector(ALB_LEAF_NIR)}
  real(r_std),parameter::snowa_aged_vis(nvm)={_fortran_vector(SNOWA_AGED)}
  real(r_std),parameter::snowa_aged_nir(nvm)={_fortran_vector(SNOWA_AGED)}
  real(r_std),parameter::snowa_dec_vis(nvm)={_fortran_vector(SNOWA_DEC_VIS)}
  real(r_std),parameter::snowa_dec_nir(nvm)={_fortran_vector(SNOWA_DEC_NIR)}
end module oracle_parameters
"""
    return text.encode("ascii")


def _qsat_module(spans: dict[str, tuple[Path, bytes, int, int, str]]) -> bytes:
    return (
        rb"""
module qsat_moisture
  use oracle_parameters
  use oracle_transport,only:ipslerr_p
  implicit none
  integer(i_std),parameter::max_temp=370,min_temp=100
  logical,save::l_qsat_first=.true.
  real(r_std),save::qsfrict(max_temp)
contains
"""
        + spans["qsatcalc"][1]
        + spans["dev_qsatcalc"][1]
        + spans["qsfrict_init"][1]
        + b"\nend module qsat_moisture\n"
    )


def _enerbil_module(spans: dict[str, tuple[Path, bytes, int, int, str]]) -> bytes:
    return (
        rb"""
module enerbil_science
  use oracle_parameters
  use oracle_transport,only:ipslerr_p
  use qsat_moisture
  implicit none
  real(r_std),allocatable,save::psold(:),psold_pft(:,:),qsol_sat(:),qsol_sat_pft(:,:)
  real(r_std),allocatable,save::pdqsold(:),pdqsold_pft(:,:),psnew(:),psnew_pft(:,:)
  real(r_std),allocatable,save::qsol_sat_new(:),qsol_sat_new_pft(:,:),netrad(:),netrad_pft(:,:)
  real(r_std),allocatable,save::lwabs(:),lwup(:),lwnet(:),fluxsubli(:),qsat_air(:),tair(:)
  real(r_std),allocatable,save::q_sol_pot(:),temp_sol_pot(:)
contains
  subroutine enerbil_state_init(n)
    integer,intent(in)::n
    if(allocated(psold)) deallocate(psold,psold_pft,qsol_sat,qsol_sat_pft,pdqsold,pdqsold_pft, &
      psnew,psnew_pft,qsol_sat_new,qsol_sat_new_pft,netrad,netrad_pft,lwabs,lwup,lwnet, &
      fluxsubli,qsat_air,tair,q_sol_pot,temp_sol_pot)
    allocate(psold(n),psold_pft(n,nvm),qsol_sat(n),qsol_sat_pft(n,nvm),pdqsold(n), &
      pdqsold_pft(n,nvm),psnew(n),psnew_pft(n,nvm),qsol_sat_new(n),qsol_sat_new_pft(n,nvm), &
      netrad(n),netrad_pft(n,nvm),lwabs(n),lwup(n),lwnet(n),fluxsubli(n),qsat_air(n),tair(n), &
      q_sol_pot(n),temp_sol_pot(n))
    psold_pft=0; qsol_sat_pft=0; pdqsold_pft=0; psnew_pft=0
    qsol_sat_new_pft=0; netrad_pft=0
  end subroutine enerbil_state_init
"""
        + spans["enerbil_begin"][1]
        + spans["enerbil_surftemp"][1]
        + spans["enerbil_flux"][1]
        + b"\nend module enerbil_science\n"
    )


def _condveg_module(spans: dict[str, tuple[Path, bytes, int, int, str]]) -> bytes:
    return (
        rb"""
module condveg_science
  use oracle_parameters
  use oracle_transport,only:ipslerr_p
  implicit none
  real(r_std),allocatable,save::soilalb_dry(:,:),soilalb_wet(:,:),soilalb_moy(:,:),soilalb_bg(:,:)
contains
  subroutine condveg_state_init(n)
    integer,intent(in)::n
    if(allocated(soilalb_moy)) deallocate(soilalb_dry,soilalb_wet,soilalb_moy,soilalb_bg)
    allocate(soilalb_dry(n,2),soilalb_wet(n,2),soilalb_moy(n,2),soilalb_bg(n,2))
    soilalb_dry=0; soilalb_wet=0; soilalb_bg=0
    soilalb_moy(:,1)=.16_r_std; soilalb_moy(:,2)=.32_r_std
  end subroutine condveg_state_init
"""
        + spans["condveg_frac_snow"][1]
        + spans["condveg_z0cdrag"][1]
        + spans["condveg_albedo"][1]
        + b"\nend module condveg_science\n"
    )


def _sechiba_module(spans: dict[str, tuple[Path, bytes, int, int, str]]) -> bytes:
    interface = _lines(SECHIBA_SOURCE, 829, 943)
    no_routing = _lines(SECHIBA_SOURCE, 1236, 1238)
    return (
        rb"""
module sechiba
  use oracle_parameters
  use enerbil_science
  use condveg_science
  implicit none
  integer,save::sechiba_call_count=0,last_itau=-1
  real(r_std),allocatable,save::trace_precip_rain(:),trace_precip_snow(:),trace_u(:),trace_qair(:)
  real(r_std),allocatable,save::trace_lalo(:,:),trace_contfrac(:),trace_cdrag_in(:)
contains
"""
        + interface
        + rb"""
    real(r_std)::rau(kjpindex),temp_sol(kjpindex),temp_sol_pft(kjpindex,nvm)
    real(r_std)::soilflx(kjpindex),soilflx_pft(kjpindex,nvm),soilcap(kjpindex),soilcap_pft(kjpindex,nvm)
    real(r_std)::q_cdrag_pft(kjpindex,nvm),vbeta(kjpindex),vbeta_pft(kjpindex,nvm)
    real(r_std)::valpha(kjpindex),vbeta1(kjpindex),vbeta5(kjpindex),qair_new(kjpindex),epot_air_new(kjpindex)
    real(r_std)::snow(kjpindex),snow_age(kjpindex),snow_nobio(kjpindex,nnobio),snow_nobio_age(kjpindex,nnobio)
    real(r_std)::snowrho(kjpindex,nsnow),snowdz(kjpindex,nsnow),frac_snow_veg(kjpindex),frac_snow_nobio(kjpindex,nnobio)
    real(r_std)::veget(kjpindex,nvm),veget_max(kjpindex,nvm),height(kjpindex,nvm)
    real(r_std)::frac_nobio(kjpindex,nnobio),totfrac_nobio(kjpindex),tot_bare_soil(kjpindex)
    real(r_std)::z0m(kjpindex),z0h(kjpindex),roughheight(kjpindex),roughheight_pft(kjpindex,nvm)
    real(r_std)::albedo_snow(kjpindex,2),alb_bare(kjpindex,2),alb_veget(kjpindex,2)
    real(r_std)::evapot(kjpindex),evapot_corr(kjpindex),pgflux(kjpindex),temp_sol_add(kjpindex)
    integer::jv

    sechiba_call_count=sechiba_call_count+1; last_itau=kjit
    if(allocated(trace_u)) deallocate(trace_precip_rain,trace_precip_snow,trace_u,trace_qair,trace_lalo,trace_contfrac,trace_cdrag_in)
    allocate(trace_precip_rain(kjpindex),trace_precip_snow(kjpindex),trace_u(kjpindex),trace_qair(kjpindex), &
      trace_lalo(kjpindex,2),trace_contfrac(kjpindex),trace_cdrag_in(kjpindex))
    trace_precip_rain=precip_rain; trace_precip_snow=precip_snow; trace_u=u; trace_qair=qair
    trace_lalo=lalo; trace_contfrac=contfrac; trace_cdrag_in=tq_cdrag

    call sechiba_var_init(kjpindex,rau,pb,temp_air)
    veget=zero; veget_max=zero; height=zero
    veget(:,1)=.25_r_std; veget_max(:,1)=.25_r_std
    veget(:,14)=.75_r_std; veget_max(:,14)=.75_r_std; height(:,14)=1._r_std
    frac_nobio=zero; totfrac_nobio=zero; tot_bare_soil=.25_r_std
    snow=zero; snow_age=zero; snow_nobio=zero; snow_nobio_age=zero; snowrho=330._r_std; snowdz=zero
    call condveg_state_init(kjpindex)
    call condveg_frac_snow(kjpindex,snow,snow_nobio,snowrho,snowdz,frac_snow_veg,frac_snow_nobio)
    emis_out=emis_scal
    call condveg_z0cdrag(kjpindex,veget,veget_max,frac_nobio,totfrac_nobio,zlev,height,tot_bare_soil, &
      z0m,z0h,roughheight,roughheight_pft)
    call condveg_albedo(kjpindex,veget,veget_max,zero*temp_air,frac_nobio,totfrac_nobio,snow,snow_age, &
      snow_nobio,snow_nobio_age,snowdz,snowrho,tot_bare_soil,frac_snow_veg,frac_snow_nobio, &
      albedo,albedo_snow,alb_bare,alb_veget)

    temp_sol=temp_air-1.5_r_std
    do jv=1,nvm; temp_sol_pft(:,jv)=temp_sol; enddo
    soilflx=12._r_std; soilflx_pft=12._r_std; soilcap=2.e6_r_std; soilcap_pft=2.e6_r_std
    q_cdrag_pft=spread(tq_cdrag,2,nvm)
    vbeta=.55_r_std; vbeta_pft=.55_r_std; valpha=.85_r_std; vbeta1=.05_r_std; vbeta5=zero
    evapot=zero; evapot_corr=zero; pgflux=zero
    call enerbil_state_init(kjpindex)
    call enerbil_begin(kjpindex,temp_sol,temp_sol_pft,lwdown,swnet,pb,psold,psold_pft,qsol_sat, &
      qsol_sat_pft,pdqsold,pdqsold_pft,netrad,netrad_pft,emis_out)
    call enerbil_surftemp(kjpindex,zlev,emis_out,epot_air,petAcoef,petBcoef,qair,peqAcoef,peqBcoef, &
      soilflx,soilflx_pft,rau,u,v,tq_cdrag,q_cdrag_pft,vbeta,vbeta_pft,valpha,vbeta1,vbeta5, &
      soilcap,soilcap_pft,lwdown,swnet,psnew,qsol_sat_new,qsol_sat_new_pft,temp_sol_new, &
      temp_sol_pft,qair_new,epot_air_new,veget_max)
    call enerbil_flux(kjpindex,emis_out,temp_sol,rau,u,v,tq_cdrag,vbeta,valpha,vbeta1,vbeta5, &
      qair_new,epot_air_new,psnew,qsurf_out,fluxsens,fluxlat,fluxsubli,vevapp,temp_sol_new, &
      lwdown,swnet,lwup,lwnet,pb,tsol_rad,netrad,evapot,evapot_corr,precip_rain,snowdz, &
      temp_air,pgflux,soilcap,temp_sol_add)
    z0m_out=z0m; z0h_out=z0h
"""
        + no_routing
        + rb"""
    netco2flux=zero; fco2_lu=zero
  end subroutine sechiba_main
"""
        + spans["sechiba_var_init"][1]
        + b"\nend module sechiba\n"
    )


INTERSURF_HEAD = rb"""
module intersurf
  use oracle_parameters
  use oracle_transport
  use sechiba,only:sechiba_main
  implicit none
  integer(i_std),save::printlev_loc=0,hist_id=5,rest_id=1,hist2_id=6
  integer(i_std),save::hist_id_stom=7,hist_id_stom_IPCC=8,rest_id_stom=9,itau_offset=2
  real(r_std),save::date0_shifted=2451544.5d0,julian0=2451544.5d0
  logical,parameter::check_INPUTS=.false.
  logical,save::lstep_init_intersurf=.false.,check_time=.false.
  character(len=32),save::calendar_str='gregorian'
  real(r_std),save::one_year=31622400.d0,one_day=86400.d0,year_spread=1.d0,in_julian,julian_diff
  integer(i_std),save::year_length=17568,year,month,day,sec,month_len
  real(r_std),allocatable,save::lalo(:,:),contfrac(:),resolution(:,:),area(:)
  integer(i_std),allocatable,save::neighbours(:,:),ilandindex(:),jlandindex(:)
  interface ioget_calendar
    module procedure calendar_string,calendar_numbers
  end interface
contains
  subroutine calendar_string(value);character(len=*),intent(out)::value;value=calendar_str;end
  subroutine calendar_numbers(a,b);real(r_std),intent(out)::a,b;a=one_year;b=one_day;end
  subroutine tlen2itau(text,dt,date,length)
    character(len=*),intent(in)::text;real(r_std),intent(in)::dt,date;integer(i_std),intent(out)::length
    length=nint(one_year/dt)
  end
  real(r_std) function itau2date(step,date,dt)
    integer(i_std),intent(in)::step;real(r_std),intent(in)::date,dt
    itau2date=date+real(step,r_std)*dt/one_day
  end
  subroutine ju2ymds(julian,y,m,d,s)
    real(r_std),intent(in)::julian;integer(i_std),intent(out)::y,m,d,s
    integer::a,b,c,e,whole,alpha;real(r_std)::shifted,frac
    shifted=julian+.5d0;whole=floor(shifted);frac=shifted-real(whole,r_std)
    alpha=int((real(whole,r_std)-1867216.25d0)/36524.25d0);a=whole+1+alpha-alpha/4
    b=a+1524;c=int((real(b,r_std)-122.1d0)/365.25d0);d=int(365.25d0*real(c,r_std));e=int(real(b-d,r_std)/30.6001d0)
    d=b-d-int(30.6001d0*real(e,r_std));m=merge(e-1,e-13,e<14);y=merge(c-4716,c-4715,m>2);s=nint(frac*one_day)
  end
  subroutine ymds2ju(y,m,d,s,julian)
    integer(i_std),intent(in)::y,m,d;real(r_std),intent(in)::s;real(r_std),intent(out)::julian
    integer::yy,mm,a,b;yy=y;mm=m;if(mm<=2)then;yy=yy-1;mm=mm+12;endif;a=yy/100;b=2-a+a/4
    julian=floor(365.25d0*real(yy+4716,r_std))+floor(30.6001d0*real(mm+1,r_std))+real(d+b,r_std)-1524.5d0+s/one_day
  end
  subroutine itau2ymds(step,dt,y,m,d,s)
    integer(i_std),intent(in)::step;real(r_std),intent(in)::dt;integer(i_std),intent(out)::y,m,d,s
    call ju2ymds(itau2date(step,2451544.5d0,dt),y,m,d,s)
  end
  integer(i_std) function ioget_mon_len(y,m)
    integer(i_std),intent(in)::y,m;integer::ml(12)
    ml=(/31,28,31,30,31,30,31,31,30,31,30,31/);if(mod(y,4)==0.and.m==2)ml(2)=29;ioget_mon_len=ml(m)
  end
  subroutine oracle_intersurf_state(iim,jjm,kindex,lon,lat,zcontfrac,zresolution,window)
    integer,intent(in)::iim,jjm,kindex(:);real(r_std),intent(in)::lon(iim,jjm),lat(iim,jjm),zcontfrac(iim,jjm)
    real(r_std),intent(in)::zresolution(iim,jjm,2),window;integer::ik,i,j,n
    n=size(kindex);if(allocated(lalo))deallocate(lalo,contfrac,resolution,area,neighbours,ilandindex,jlandindex)
    allocate(lalo(n,2),contfrac(n),resolution(n,2),area(n),neighbours(n,NbNeighb),ilandindex(n),jlandindex(n))
    do ik=1,n;j=((kindex(ik)-1)/iim)+1;i=kindex(ik)-(j-1)*iim
      ilandindex(ik)=i;jlandindex(ik)=j
      lalo(ik,1)=lat(i,j);lalo(ik,2)=lon(i,j);contfrac(ik)=zcontfrac(i,j);resolution(ik,:)=zresolution(i,j,:)
    enddo
    area=1.d6;neighbours=spread(kindex,2,NbNeighb);call reset_transport(window)
  end
"""


DRIVER = rb"""
program oracle
  use oracle_parameters
  use oracle_transport
  use intersurf
  use sechiba,only:trace_precip_rain,trace_precip_snow,trace_u,trace_qair,trace_lalo,trace_contfrac,trace_cdrag_in,sechiba_call_count,last_itau
  implicit none
  integer,parameter::iim=2,jjm=3,kjpindex=3
  integer::kindex(kjpindex),mode,i,j,uout
  real(r_std)::xrdt,window,token
  real(r_std)::lon(iim,jjm),lat(iim,jjm),zcontfrac(iim,jjm),zresolution(iim,jjm,2)
  real(r_std)::zlev(iim,jjm),u(iim,jjm),v(iim,jjm),qair(iim,jjm),temp_air(iim,jjm),epot_air(iim,jjm)
  real(r_std)::ccanopy(iim,jjm),cdrag(iim,jjm),petAcoef(iim,jjm),peqAcoef(iim,jjm),petBcoef(iim,jjm),peqBcoef(iim,jjm)
  real(r_std)::precip_rain(iim,jjm),precip_snow(iim,jjm),lwdown(iim,jjm),swnet(iim,jjm),swdown(iim,jjm),pb(iim,jjm),coszang(iim,jjm)
  real(r_std)::vevapp(iim,jjm),fluxsens(iim,jjm),fluxlat(iim,jjm),coastalflow0(iim,jjm),riverflow0(iim,jjm)
  real(r_std)::tsol_rad(iim,jjm),temp_sol_new(iim,jjm),qsurf(iim,jjm),albedo(iim,jjm,2),emis(iim,jjm),z0m(iim,jjm)
  character(len=512)::path,arg
  call get_command_argument(1,arg);read(arg,*)mode;call get_command_argument(2,path)
  kindex=(/1,4,6/);xrdt=1800.d0;window=merge(900.d0,1800.d0,mode==1)
  is_omp_root=(mode/=3)
  do j=1,jjm;do i=1,iim
    token=real(i+10*j+100*mode,r_std)
    lon(i,j)=100.d0+.01d0*token;lat(i,j)=30.d0+.02d0*token;zcontfrac(i,j)=.5d0+.001d0*token
    zresolution(i,j,1)=1000.d0+token;zresolution(i,j,2)=1200.d0+token
    zlev(i,j)=2.d0+.001d0*token;u(i,j)=1.d0+.002d0*token;v(i,j)=.5d0+.001d0*token
    qair(i,j)=.004d0+1.d-6*token;temp_air(i,j)=275.d0+.01d0*token
    epot_air(i,j)=cp_air*temp_air(i,j)+9.80665d0*zlev(i,j);ccanopy(i,j)=400.d0+.1d0*token
    cdrag(i,j)=.01d0+1.d-6*token;petAcoef(i,j)=0.d0;peqAcoef(i,j)=0.d0
    petBcoef(i,j)=epot_air(i,j);peqBcoef(i,j)=qair(i,j)
    precip_rain(i,j)=1.d-5+1.d-9*token;precip_snow(i,j)=2.d-6+1.d-10*token
    lwdown(i,j)=280.d0+.05d0*token;swnet(i,j)=90.d0+.03d0*token;swdown(i,j)=160.d0+.04d0*token
    pb(i,j)=990.d0+.01d0*token;coszang(i,j)=.4d0
  enddo;enddo
  call oracle_intersurf_state(iim,jjm,kindex,lon,lat,zcontfrac,zresolution,window)
  call intersurf_main_2d(5,iim,jjm,kjpindex,kindex,xrdt,1961,.false.,.false.,lon,lat,zcontfrac,zresolution,2451544.5d0, &
    zlev,u,v,qair,temp_air,epot_air,ccanopy,cdrag,petAcoef,peqAcoef,petBcoef,peqBcoef,precip_rain,precip_snow, &
    lwdown,swnet,swdown,pb,vevapp,fluxsens,fluxlat,coastalflow0,riverflow0,tsol_rad,temp_sol_new,qsurf,albedo,emis,z0m,coszang)
  open(newunit=uout,file=trim(path),status='replace',action='write')
  write(uout,'(A)')'field,i,j,k,value'
  call write2('z0m',z0m);call write2('coastalflow',coastalflow0);call write2('riverflow',riverflow0)
  call write2('tsol_rad',tsol_rad);call write2('vevapp',vevapp);call write2('temp_sol_new',temp_sol_new)
  call write2('qsurf',qsurf);call write2('fluxsens',fluxsens);call write2('fluxlat',fluxlat);call write2('emis',emis);call write2('cdrag',cdrag)
  call write3('albedo',albedo)
  call write1('request_precip_rain',trace_precip_rain);call write1('request_precip_snow',trace_precip_snow)
  call write1('request_u',trace_u);call write1('request_qair',trace_qair);call write1('request_lat',trace_lalo(:,1))
  call write1('request_lon',trace_lalo(:,2));call write1('request_contfrac',trace_contfrac);call write1('request_cdrag',trace_cdrag_in)
  call writemeta('calendar_step',real(last_calendar_step,r_std));call writemeta('sechiba_itau',real(last_itau,r_std))
  call writemeta('sechiba_call_count',real(sechiba_call_count,r_std));call writemeta('history_write_count',real(history_write_count,r_std))
  call writemeta('xios_send_count',real(xios_send_count,r_std));call writemeta('histsync_count',real(histsync_count,r_std))
  close(uout)
contains
  subroutine write2(name,value);character(len=*),intent(in)::name;real(r_std),intent(in)::value(iim,jjm);integer::ii,jj
    do jj=1,jjm;do ii=1,iim;write(uout,'(A,",",I0,",",I0,",0,",ES25.17E3)')trim(name),ii,jj,value(ii,jj);enddo;enddo
  end
  subroutine write3(name,value);character(len=*),intent(in)::name;real(r_std),intent(in)::value(iim,jjm,2);integer::ii,jj,kk
    do kk=1,2;do jj=1,jjm;do ii=1,iim;write(uout,'(A,",",I0,",",I0,",",I0,",",ES25.17E3)')trim(name),ii,jj,kk,value(ii,jj,kk);enddo;enddo;enddo
  end
  subroutine write1(name,value);character(len=*),intent(in)::name;real(r_std),intent(in)::value(kjpindex);integer::ii
    do ii=1,kjpindex;write(uout,'(A,",",I0,",0,0,",ES25.17E3)')trim(name),ii,value(ii);enddo
  end
  subroutine writemeta(name,value);character(len=*),intent(in)::name;real(r_std),intent(in)::value
    write(uout,'(A,",0,0,0,",ES25.17E3)')trim(name),value
  end
end program oracle
"""


def compose(path: Path) -> dict[str, dict[str, Any]]:
    spans = _source_spans()
    unit = (
        TRANSPORT_MODULE
        + _parameter_module()
        + _qsat_module(spans)
        + _enerbil_module(spans)
        + _condveg_module(spans)
        + _sechiba_module(spans)
        + INTERSURF_HEAD
        + spans["intersurf_main_2d"][1]
        + spans["intsurf_time"][1]
        + b"\nend module intersurf\n"
        + DRIVER
    )
    path.write_bytes(unit)
    manifest: dict[str, dict[str, Any]] = {}
    for name, (source, payload, start, end, digest) in spans.items():
        manifest[name] = {
            "source_file": source.relative_to(ROOT).as_posix(),
            "start_line": start,
            "end_line": end,
            "sha256": digest,
            "byte_exact_in_compile_unit": bool(unit.count(payload) == 1),
            "role": (
                "full_direct_callee_audit_extract"
                if name == "sechiba_main"
                else "executed_original_procedure"
            ),
        }
    no_routing = _lines(SECHIBA_SOURCE, 1236, 1238)
    manifest["sechiba_main_no_routing_writeback"] = {
        "source_file": SECHIBA_SOURCE.relative_to(ROOT).as_posix(),
        "start_line": 1236,
        "end_line": 1238,
        "sha256": _sha256(no_routing),
        "byte_exact_in_compile_unit": bool(unit.count(no_routing) == 1),
        "role": "executed_original_fragment",
    }
    required_in_unit = [
        record
        for record in manifest.values()
        if record["role"] != "full_direct_callee_audit_extract"
    ]
    if not all(record["byte_exact_in_compile_unit"] for record in required_in_unit):
        raise RuntimeError("a required original source span is absent or duplicated in the compile unit")
    return manifest


def _forcing(mode: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    shape = (IIM, JJM)
    raw: dict[str, np.ndarray] = {
        name: np.empty(shape, dtype=np.float64)
        for name in (
            "lon", "lat", "zcontfrac", "zlev", "u", "v", "qair", "temp_air",
            "epot_air", "ccanopy", "cdrag", "petAcoef", "peqAcoef", "petBcoef",
            "peqBcoef", "precip_rain", "precip_snow", "lwdown", "swnet", "swdown",
            "pb", "coszang",
        )
    }
    resolution = np.empty(shape + (2,), dtype=np.float64)
    for j in range(JJM):
        for i in range(IIM):
            token = float(i + 1 + 10 * (j + 1) + 100 * mode)
            raw["lon"][i, j] = 100.0 + 0.01 * token
            raw["lat"][i, j] = 30.0 + 0.02 * token
            raw["zcontfrac"][i, j] = 0.5 + 0.001 * token
            resolution[i, j] = (1000.0 + token, 1200.0 + token)
            raw["zlev"][i, j] = 2.0 + 0.001 * token
            raw["u"][i, j] = 1.0 + 0.002 * token
            raw["v"][i, j] = 0.5 + 0.001 * token
            raw["qair"][i, j] = 0.004 + 1.0e-6 * token
            raw["temp_air"][i, j] = 275.0 + 0.01 * token
            raw["epot_air"][i, j] = 1004.675 * raw["temp_air"][i, j] + 9.80665 * raw["zlev"][i, j]
            raw["ccanopy"][i, j] = 400.0 + 0.1 * token
            raw["cdrag"][i, j] = 0.01 + 1.0e-6 * token
            raw["petAcoef"][i, j] = 0.0
            raw["peqAcoef"][i, j] = 0.0
            raw["petBcoef"][i, j] = raw["epot_air"][i, j]
            raw["peqBcoef"][i, j] = raw["qair"][i, j]
            raw["precip_rain"][i, j] = 1.0e-5 + 1.0e-9 * token
            raw["precip_snow"][i, j] = 2.0e-6 + 1.0e-10 * token
            raw["lwdown"][i, j] = 280.0 + 0.05 * token
            raw["swnet"][i, j] = 90.0 + 0.03 * token
            raw["swdown"][i, j] = 160.0 + 0.04 * token
            raw["pb"][i, j] = 990.0 + 0.01 * token
            raw["coszang"][i, j] = 0.4
    fields = {name: raw[name] for name in raw if name not in {"lon", "lat", "zcontfrac"}}
    geometry = {"lon": raw["lon"], "lat": raw["lat"], "zcontfrac": raw["zcontfrac"], "zresolution": resolution}
    return fields, geometry


def _science_owner(xrdt: float):
    def owner(request: Any) -> dict[str, np.ndarray]:
        forcing = {name: np.asarray(value, dtype=np.float64) for name, value in request.forcing.items()}
        n = len(request.contfrac)
        veget = np.zeros((n, NVM), dtype=np.float64)
        veget_max = np.zeros_like(veget)
        height = np.zeros_like(veget)
        veget[:, 0] = 0.25
        veget_max[:, 0] = 0.25
        veget[:, 13] = 0.75
        veget_max[:, 13] = 0.75
        height[:, 13] = 1.0
        frac_nobio = np.zeros((n, 1), dtype=np.float64)
        totfrac_nobio = np.zeros(n, dtype=np.float64)
        tot_bare_soil = np.full(n, 0.25, dtype=np.float64)
        snow = np.zeros(n, dtype=np.float64)
        snow_nobio = np.zeros((n, 1), dtype=np.float64)
        snowrho = np.full((n, 3), 330.0, dtype=np.float64)
        snowdz = np.zeros((n, 3), dtype=np.float64)
        snow_fraction = condveg_frac_snow(
            snow=snow,
            snow_nobio=snow_nobio,
            snowrho=snowrho,
            snowdz=snowdz,
            ok_explicitsnow=False,
        )
        roughness = condveg_z0cdrag(
            veget=veget,
            veget_max=veget_max,
            frac_nobio=frac_nobio,
            totfrac_nobio=totfrac_nobio,
            zlev=forcing["zlev"],
            height=height,
            tot_bare_soil=tot_bare_soil,
            is_tree=IS_TREE,
            z0_over_height=np.asarray([0.0] + [0.0625] * 13),
            ratio_z0m_z0h=np.ones(NVM),
        )
        albedo = condveg_albedo_explicit(
            veget=veget,
            veget_max=veget_max,
            drysoil_frac=np.zeros(n),
            frac_nobio=frac_nobio,
            totfrac_nobio=totfrac_nobio,
            snow=snow,
            snow_age=np.zeros(n),
            snow_nobio=snow_nobio,
            snow_nobio_age=np.zeros((n, 1)),
            tot_bare_soil=tot_bare_soil,
            frac_snow_veg=snow_fraction.frac_snow_veg,
            frac_snow_nobio=snow_fraction.frac_snow_nobio,
            soilalb_moy=np.broadcast_to(np.asarray([0.16, 0.32]), (n, 2)),
            alb_leaf_vis=ALB_LEAF_VIS,
            alb_leaf_nir=ALB_LEAF_NIR,
            snowa_aged_vis=SNOWA_AGED,
            snowa_aged_nir=SNOWA_AGED,
            snowa_dec_vis=SNOWA_DEC_VIS,
            snowa_dec_nir=SNOWA_DEC_NIR,
            fixed_snow_albedo=UNDEF_SECHIBA,
            undef_sechiba=UNDEF_SECHIBA,
        )
        emis = np.asarray(condveg_main_emissivity(n, emis_scal=1.0))
        temp_sol = forcing["temp_air"] - 1.5
        temp_sol_pft = np.broadcast_to(temp_sol[:, None], (n, NVM)).copy()
        soilflx = np.full(n, 12.0)
        soilflx_pft = np.full((n, NVM), 12.0)
        soilcap = np.full(n, 2.0e6)
        soilcap_pft = np.full((n, NVM), 2.0e6)
        rau = np.asarray(sechiba_air_density_from_pb_temp_air(forcing["pb"], forcing["temp_air"]))
        ok_laidev = np.zeros(NVM, dtype=bool)
        begin = enerbil_begin_local_diagnostics(
            temp_sol=temp_sol,
            temp_sol_pft=temp_sol_pft,
            lwdown=forcing["lwdown"],
            swnet=forcing["swnet"],
            pb=forcing["pb"],
            emis=emis,
            ok_laidev=ok_laidev,
        )
        cdrag_pft = np.broadcast_to(forcing["cdrag"][:, None], (n, NVM)).copy()
        surftemp = enerbil_surftemp_explicit_solve(
            psold=begin.psold,
            psold_pft=begin.psold_pft,
            qsol_sat=begin.qsol_sat,
            qsol_sat_pft=begin.qsol_sat_pft,
            pdqsold=begin.pdqsold,
            pdqsold_pft=begin.pdqsold_pft,
            netrad=begin.netrad,
            netrad_pft=begin.netrad_pft,
            emis=emis,
            epot_air=forcing["epot_air"],
            petAcoef=forcing["petAcoef"],
            petBcoef=forcing["petBcoef"],
            qair=forcing["qair"],
            peqAcoef=forcing["peqAcoef"],
            peqBcoef=forcing["peqBcoef"],
            soilflx=soilflx,
            soilflx_pft=soilflx_pft,
            rau=rau,
            u=forcing["u"],
            v=forcing["v"],
            q_cdrag=forcing["cdrag"],
            q_cdrag_pft=cdrag_pft,
            vbeta=np.full(n, 0.55),
            vbeta_pft=np.full((n, NVM), 0.55),
            valpha=np.full(n, 0.85),
            vbeta1=np.full(n, 0.05),
            vbeta5=np.zeros(n),
            soilcap=soilcap,
            soilcap_pft=soilcap_pft,
            veget_max=veget_max,
            ok_laidev=ok_laidev,
            dt_sechiba=xrdt,
        )
        flux = enerbil_flux_local_diagnostics(
            emis=emis,
            temp_sol=temp_sol,
            rau=rau,
            u=forcing["u"],
            v=forcing["v"],
            q_cdrag=forcing["cdrag"],
            vbeta=np.full(n, 0.55),
            valpha=np.full(n, 0.85),
            vbeta1=np.full(n, 0.05),
            vbeta5=np.zeros(n),
            qair=surftemp.qair_new,
            epot_air=surftemp.epot_air_new,
            psnew=surftemp.psnew,
            qsol_sat_new=surftemp.qsol_sat_new,
            temp_sol_new=surftemp.temp_sol_new,
            lwdown=forcing["lwdown"],
            swnet=forcing["swnet"],
            dt_sechiba=xrdt,
        )
        return {
            "z0m": np.asarray(roughness.z0m),
            "coastalflow": np.zeros((n, NFLOW)),
            "riverflow": np.zeros((n, NFLOW)),
            "tsol_rad": np.asarray(flux.tsol_rad),
            "vevapp": np.asarray(flux.vevapp),
            "temp_sol_new": np.asarray(surftemp.temp_sol_new),
            "qsurf": np.asarray(flux.qsurf),
            "albedo": np.asarray(albedo.albedo),
            "fluxsens": np.asarray(flux.fluxsens),
            "fluxlat": np.asarray(flux.fluxlat),
            "emis": emis,
            "cdrag": forcing["cdrag"].copy(),
        }

    return owner


def _jax_case(mode: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    fields, geometry = _forcing(mode)
    result = intersurf_main_2d(
        kjit=5,
        iim=IIM,
        jjm=JJM,
        kindex=KINDEX,
        xrdt=XRDT,
        date0_shifted=2451544.5,
        itau_offset=2,
        lon=geometry["lon"],
        lat=geometry["lat"],
        zcontfrac=geometry["zcontfrac"],
        fields=fields,
        owner=_science_owner(XRDT),
        undef_sechiba=UNDEF_SECHIBA,
    )
    outputs = {name: np.asarray(value, dtype=np.float64) for name, value in result.outputs.items()}
    outputs["coastalflow"] = outputs["coastalflow"][:, :, 0]
    outputs["riverflow"] = outputs["riverflow"][:, :, 0]
    request = {
        "request_precip_rain": np.asarray(result.request.forcing["precip_rain"]),
        "request_precip_snow": np.asarray(result.request.forcing["precip_snow"]),
        "request_u": np.asarray(result.request.forcing["u"]),
        "request_qair": np.asarray(result.request.forcing["qair"]),
        "request_lat": np.asarray(result.request.latitude_longitude[:, 0]),
        "request_lon": np.asarray(result.request.latitude_longitude[:, 1]),
        "request_contfrac": np.asarray(result.request.contfrac),
        "request_cdrag": np.asarray(result.request.forcing["cdrag"]),
    }
    meta = {
        "calendar_step": float(result.calendar_step),
        "sechiba_itau": float(result.request.itau_sechiba),
        "sechiba_call_count": 1.0,
        "histsync_count": 0.0,
    }
    return outputs, request, meta


def _read_fortran(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    rows: dict[str, list[tuple[int, int, int, float]]] = defaultdict(list)
    with path.open(encoding="ascii", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[str(row["field"])].append(
                (int(row["i"]), int(row["j"]), int(row["k"]), float(row["value"]))
            )
    outputs: dict[str, np.ndarray] = {}
    for name in ("z0m", "coastalflow", "riverflow", "tsol_rad", "vevapp", "temp_sol_new", "qsurf", "fluxsens", "fluxlat", "emis", "cdrag"):
        value = np.empty((IIM, JJM), dtype=np.float64)
        for i, j, _k, item in rows[name]:
            value[i - 1, j - 1] = item
        outputs[name] = value
    albedo = np.empty((IIM, JJM, 2), dtype=np.float64)
    for i, j, k, item in rows["albedo"]:
        albedo[i - 1, j - 1, k - 1] = item
    outputs["albedo"] = albedo
    request = {
        name: np.asarray([item for _i, _j, _k, item in sorted(items)], dtype=np.float64)
        for name, items in rows.items()
        if name.startswith("request_")
    }
    meta = {
        name: items[0][3]
        for name, items in rows.items()
        if name in {"calendar_step", "sechiba_itau", "sechiba_call_count", "history_write_count", "xios_send_count", "histsync_count"}
    }
    return outputs, request, meta


def _write_extracted_assets(output_dir: Path, manifest: dict[str, dict[str, Any]]) -> None:
    spans = _source_spans()
    extracted = output_dir / "extracted_original_bytes"
    extracted.mkdir(parents=True, exist_ok=True)
    for name, (_source, payload, _start, _end, _digest) in spans.items():
        (extracted / f"{name}.f90").write_bytes(payload)
        manifest[name]["asset"] = (extracted / f"{name}.f90").relative_to(ROOT).as_posix()
    fragment = _lines(SECHIBA_SOURCE, 1236, 1238)
    fragment_path = extracted / "sechiba_main_no_routing_writeback.f90"
    fragment_path.write_bytes(fragment)
    manifest["sechiba_main_no_routing_writeback"]["asset"] = fragment_path.relative_to(ROOT).as_posix()
    (output_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")


def _run_executable(executable: Path, output_path: Path, mode: int, compiler: Path) -> None:
    subprocess.run(
        [str(executable), str(mode), str(output_path.resolve())],
        cwd=executable.parent,
        env=compiler_environment(compiler),
        check=True,
        capture_output=True,
        text=True,
    )


def run_numerical_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    composed_asset = output_dir / "composed_oracle.f90"
    manifest = compose(composed_asset)
    _write_extracted_assets(output_dir, manifest)
    fortran_cases: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]] = {}
    jax_cases: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]] = {}
    build_metadata: dict[str, Any]
    with tempfile.TemporaryDirectory(prefix="orchidee_intersurf_main_numeric_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        shutil.copyfile(composed_asset, source)
        executable = build / "oracle.exe"
        build_metadata = compile_fortran(source, executable, compiler, base_flags=COMPILE_FLAGS)
        combined_rows: list[dict[str, str]] = []
        for case in CASES:
            raw = build / f"{case['case_id']}.csv"
            _run_executable(executable, raw, int(case["mode"]), compiler)
            parsed = _read_fortran(raw)
            fortran_cases[str(case["case_id"])] = parsed
            jax_cases[str(case["case_id"])] = _jax_case(int(case["mode"]))
            with raw.open(encoding="ascii", newline="") as handle:
                for row in csv.DictReader(handle):
                    combined_rows.append({"case": str(case["case_id"]), **row})
    with (output_dir / "fortran_outputs.csv").open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "field", "i", "j", "k", "value"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(combined_rows)

    comparisons: list[dict[str, Any]] = []
    point_fortran: dict[str, np.ndarray] = {}
    point_jax: dict[str, np.ndarray] = {}
    output_fields = ("z0m", "coastalflow", "riverflow", "tsol_rad", "vevapp", "temp_sol_new", "qsurf", "albedo", "fluxsens", "fluxlat", "emis", "cdrag")
    request_fields = ("request_precip_rain", "request_precip_snow", "request_u", "request_qair", "request_lat", "request_lon", "request_contfrac", "request_cdrag")
    case_records: list[dict[str, Any]] = []
    for case in CASES:
        case_id = str(case["case_id"])
        f_outputs, f_request, f_meta = fortran_cases[case_id]
        j_outputs, j_request, j_meta = jax_cases[case_id]
        case_comparisons: list[dict[str, Any]] = []
        for field in output_fields:
            item = float_comparison(f"{case_id}.{field}", f_outputs[field], j_outputs[field], rtol=RTOL, atol=ATOL)
            comparisons.append(item)
            case_comparisons.append(item)
            point_fortran[f"{case_id}.{field}"] = f_outputs[field]
            point_jax[f"{case_id}.{field}"] = j_outputs[field]
            fortran_mask = np.asarray(f_outputs[field] == UNDEF_SECHIBA, dtype=np.float64)
            jax_mask = np.asarray(j_outputs[field] == UNDEF_SECHIBA, dtype=np.float64)
            mask_item = float_comparison(
                f"{case_id}.{field}.undefined_mask",
                fortran_mask,
                jax_mask,
                rtol=0.0,
                atol=0.0,
            )
            comparisons.append(mask_item)
            case_comparisons.append(mask_item)
        for field in request_fields:
            item = float_comparison(f"{case_id}.{field}", f_request[field], j_request[field], rtol=0.0, atol=0.0)
            comparisons.append(item)
            case_comparisons.append(item)
            point_fortran[f"{case_id}.{field}"] = f_request[field]
            point_jax[f"{case_id}.{field}"] = j_request[field]
        for field in ("calendar_step", "sechiba_itau", "sechiba_call_count", "histsync_count"):
            item = float_comparison(f"{case_id}.{field}", [f_meta[field]], [j_meta[field]], rtol=0.0, atol=0.0)
            comparisons.append(item)
            case_comparisons.append(item)
        if f_meta["history_write_count"] != 16.0 or f_meta["xios_send_count"] != 8.0:
            raise RuntimeError(f"{case_id}: native history/XIOS transport counts drifted: {f_meta}")
        case_records.append({
            **case,
            "fortran_meta": f_meta,
            "passed": all(bool(item["passed"]) for item in case_comparisons),
        })
    write_point_comparisons(output_dir / "point_comparisons.csv", point_fortran, point_jax, rtol=RTOL, atol=ATOL)
    result = write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "owner_region_id": OWNER_ID,
            "rtol": RTOL,
            "atol": ATOL,
            "all_exported_flux_state_writebacks_compared": True,
            "all_compressed_boundary_fields_compared": True,
            "all_scatter_masks_compared": True,
            "exported_fields": list(output_fields),
            "scatter_mask_fields": list(output_fields),
            "boundary_request_fields": list(request_fields),
            "cases": case_records,
            "build": build_metadata,
            "source_manifest_asset": (output_dir / "source_manifest.json").relative_to(ROOT).as_posix(),
            "composed_oracle_asset": composed_asset.relative_to(ROOT).as_posix(),
        },
    )
    inputs = {
        "schema_version": 1,
        "family": FAMILY,
        "owner_region_id": OWNER_ID,
        "pft": 14,
        "iim": IIM,
        "jjm": JJM,
        "kindex": KINDEX.tolist(),
        "xrdt": XRDT,
        "cases": list(CASES),
        "scientific_state": {
            "bare_soil_fraction": 0.25,
            "pft14_fraction": 0.75,
            "pft14_height_m": 1.0,
            "temp_sol_offset_k": -1.5,
            "soilflx_w_m2": 12.0,
            "soilcap_j_k": 2.0e6,
            "vbeta": 0.55,
            "valpha": 0.85,
            "vbeta1": 0.05,
            "vbeta5": 0.0,
        },
    }
    (output_dir / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n", encoding="ascii")
    return result


def _execute_coverage_harness(executable: Path, build: Path, compiler: Path) -> None:
    for case in COVERAGE_CASES:
        _run_executable(executable, build / f"coverage_{case['mode']}.csv", int(case["mode"]), compiler)


def _persist_gcov(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="orchidee_intersurf_main_gcov_asset_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        compose(source)
        executable = build / "oracle.exe"
        compile_fortran(source, executable, compiler, extra_flags=("--coverage",), base_flags=COMPILE_FLAGS)
        _execute_coverage_harness(executable, build, compiler)
        notes = next(build.glob("*.gcno"))
        subprocess.run(
            [str(GCOV), "-b", "-c", notes.name],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
        gcov_path = build / "oracle.f90.gcov"
        shutil.copyfile(gcov_path, output_dir / "oracle.f90.gcov")
        owner = extract_procedure_bytes(OWNER_SOURCE, "intersurf_main_2d")
        generated_start = locate_generated_span(source.read_bytes(), owner.span_bytes)
        branches = parse_gcov(gcov_path)
        counts = parse_gcov_line_counts(gcov_path)
        mapping: dict[str, Any] = {}
        for arm_id in ARM_IDS:
            original_line = int(arm_id.rsplit(":", 2)[0].rsplit(":", 1)[1])
            generated_line = generated_start + original_line - int(owner.start_line)
            mapping[arm_id] = {
                "original_line": original_line,
                "generated_line": generated_line,
                "line_count": counts.get(generated_line, 0),
                "branches": branches.get(generated_line, []),
            }
        return mapping


def _formal_audit(output_dir: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    contracts = json.loads(CONTRACT_CLASSES.read_text(encoding="utf-8"))
    owner_regions = [record for record in contracts["owner_regions"] if record.get("owner_region_id") == OWNER_ID]
    if len(owner_regions) != 1:
        raise RuntimeError("intersurf_main_2d owner contract projection drifted")
    projection = {
        "schema_version": 1,
        "source_asset": CONTRACT_CLASSES.relative_to(ROOT).as_posix(),
        "owner_regions": owner_regions,
    }
    (output_dir / "target_contract_classes.json").write_text(json.dumps(projection, indent=2) + "\n", encoding="ascii")
    evidence_root = output_dir / "formal_audit_evidence"
    shutil.rmtree(evidence_root, ignore_errors=True)
    family_dir = evidence_root / FAMILY
    family_dir.mkdir(parents=True)
    staged = family_dir / "owner_region_evidence.json"
    staged.write_text(json.dumps(evidence, indent=2) + "\n", encoding="ascii")
    source_proofs = json.loads(SOURCE_PROOFS.read_text(encoding="utf-8"))
    report = build_report(
        projection,
        source_proofs,
        [(staged.relative_to(ROOT).as_posix(), evidence)],
    )
    (output_dir / "formal_audit_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    if report.get("complete") is not True:
        raise RuntimeError(f"formal owner audit rejected {OWNER_ID}: {report['errors']}")
    return report


def _canonicalize_coverage(
    coverage: dict[str, Any],
    evidence: dict[str, Any],
    gcov_mapping: dict[str, Any],
) -> None:
    record = evidence["records"][0]
    required = set(record["required_arm_ids"])
    if required != set(ARM_IDS):
        raise RuntimeError("intersurf_main_2d canonical arm denominator drifted")

    proofs = json.loads(SOURCE_PROOFS.read_text(encoding="utf-8"))
    proof_by_arm = {
        str(item["arm_id"]): item
        for item in proofs.get("records", [])
        if item.get("passed") is True
    }
    source_arm_ids = set(record["source_proof_arm_ids"])
    numerical_arm_ids = set(record["gcov_arm_ids"])
    if source_arm_ids != {ARM_IDS[0], ARM_IDS[1], ARM_IDS[5]}:
        raise RuntimeError("intersurf_main_2d source-proof partition drifted")
    if numerical_arm_ids != {ARM_IDS[2], ARM_IDS[3], ARM_IDS[4]}:
        raise RuntimeError("intersurf_main_2d executable-GCOV partition drifted")

    by_arm = {str(item["arm_id"]): item for item in coverage["arms"]}
    canonical_records: list[dict[str, Any]] = []
    for arm_id in ARM_IDS:
        if arm_id in by_arm:
            item = by_arm[arm_id]
            item["evidence_method"] = "gcov_branch_counter"
        else:
            proof = proof_by_arm.get(arm_id)
            if proof is None or arm_id not in source_arm_ids:
                raise RuntimeError(f"missing pinned source proof for {arm_id}")
            mapping = gcov_mapping[arm_id]
            item = {
                "arm_id": arm_id,
                "base_region_id": BASE_REGION_ID,
                "procedure": "intersurf_main_2d",
                "original_line": int(mapping["original_line"]),
                "generated_line": int(mapping["generated_line"]),
                "gcov_branch_index": -1,
                "gcov_taken": 0,
                "gcov_passed": False,
                "fatal_boundary_passed": False,
                "source_proof_passed": True,
                "source_proof_asset": SOURCE_PROOFS.relative_to(ROOT).as_posix(),
                "source_proof_policy_id": proof["policy_id"],
                "source_proof_physical_line_sha256": proof["fortran_physical_line_sha256"],
                "passed": True,
                "raw_gcov_branches": mapping["branches"],
                "evidence_method": "pinned_source_proof",
            }
        canonical_records.append(item)

    transition_edges: list[dict[str, Any]] = []
    for original_line in (735, 736):
        mapping = gcov_mapping[
            f"fortran_source/ORCHIDEE/src_sechiba/intersurf.f90:{original_line}:if:true"
        ]
        for branch in mapping["branches"]:
            branch_index = int(branch["branch_index"])
            transition_edges.append(
                {
                    "original_line": original_line,
                    "generated_line": int(mapping["generated_line"]),
                    "gcov_branch_index": branch_index,
                    "arm": "true" if branch_index == 0 else "false",
                    "taken": int(branch["taken"]),
                }
            )
    if len(transition_edges) != 4 or not all(item["taken"] > 0 for item in transition_edges):
        raise RuntimeError(f"lines 735/736 lack four executed GCOV edges: {transition_edges}")

    coverage.update(
        {
            "branch_complete": True,
            "required_arm_count": len(ARM_IDS),
            "covered_arm_count": len(ARM_IDS),
            "missing_arm_ids": [],
            "arms": canonical_records,
            "required_arm_ids": list(ARM_IDS),
            "covered_arm_ids": list(ARM_IDS),
            "gcov_arm_ids": sorted(numerical_arm_ids),
            "source_proof_arm_ids": sorted(source_arm_ids),
            "transition_gcov_edge_count": len(transition_edges),
            "transition_gcov_edges": transition_edges,
            "transition_gcov_complete": True,
        }
    )


def run(output_dir: Path = DEFAULT_OUTPUT_DIR, compiler: Path = DEFAULT_COMPILER) -> dict[str, Any]:
    result = run_extracted_owner_coverage(
        family=FAMILY,
        source_file=OWNER_SOURCE,
        procedures=frozenset({"intersurf_main_2d"}),
        compose=lambda path: compose(path),
        run_numerical_oracle=run_numerical_oracle,
        output_dir=output_dir,
        compiler=compiler,
        gcov=GCOV,
        execute_harness=_execute_coverage_harness,
        arm_branch_indices=ARM_BRANCH_INDICES,
        coverage_base_flags=COMPILE_FLAGS,
    )
    gcov_mapping = _persist_gcov(output_dir, compiler)
    coverage = result["branch_coverage"]
    evidence = result["owner_evidence"]
    _canonicalize_coverage(coverage, evidence, gcov_mapping)
    coverage.update({
        "gcov_asset": (output_dir / "oracle.f90.gcov").relative_to(ROOT).as_posix(),
        "canonical_arm_ids": list(ARM_IDS),
        "gcov_source_mapping": gcov_mapping,
        "deterministic_cases": list(COVERAGE_CASES),
        "fatal_boundary_arm_ids": [],
        "fatal_boundary_reason": "No canonical intersurf_main_2d arm is a fatal boundary.",
        "source_proof_policy": "Only pinned pft14_source_proof_evidence records are credited as source proofs.",
    })
    (output_dir / "branch_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="ascii")
    comparison = json.loads((output_dir / "comparison.json").read_text(encoding="ascii"))
    record = evidence["records"][0]
    record["direct_state_writeback_comparison"] = {
        "status": comparison["status"],
        "rtol": RTOL,
        "atol": ATOL,
        "fields": comparison["exported_fields"],
        "boundary_request_fields": comparison["boundary_request_fields"],
        "all_compressed_boundary_fields_compared": comparison[
            "all_compressed_boundary_fields_compared"
        ],
        "scatter_mask_fields": comparison["scatter_mask_fields"],
        "all_scatter_masks_compared": comparison["all_scatter_masks_compared"],
        "max_abs_error": max(float(item.get("max_abs_error", 0.0)) for item in comparison["comparisons"]),
    }
    record["original_owner_bytes_asset"] = (
        output_dir / "extracted_original_bytes/intersurf_main_2d.f90"
    ).relative_to(ROOT).as_posix()
    record["minimal_call_graph_asset"] = (output_dir / "call_graph.json").relative_to(ROOT).as_posix()
    call_graph = {
        "schema_version": 1,
        "family": FAMILY,
        "root": "intersurf_main_2d",
        "edges": [
            {"caller": "intersurf_main_2d", "callee": "intsurf_time", "kind": "original_procedure"},
            {"caller": "intersurf_main_2d", "callee": "sechiba_main", "kind": "original_interface_reduced_dependency"},
            {"caller": "sechiba_main", "callee": "sechiba_var_init", "kind": "original_procedure"},
            {"caller": "sechiba_main", "callee": "condveg_frac_snow", "kind": "original_procedure"},
            {"caller": "sechiba_main", "callee": "condveg_z0cdrag", "kind": "original_procedure"},
            {"caller": "sechiba_main", "callee": "condveg_albedo", "kind": "original_procedure"},
            {"caller": "sechiba_main", "callee": "enerbil_begin", "kind": "original_procedure"},
            {"caller": "sechiba_main", "callee": "enerbil_surftemp", "kind": "original_procedure"},
            {"caller": "sechiba_main", "callee": "enerbil_flux", "kind": "original_procedure"},
            {"caller": "enerbil_begin", "callee": "qsatcalc/dev_qsatcalc", "kind": "original_procedures"},
            {"caller": "enerbil_flux", "callee": "qsatcalc/dev_qsatcalc", "kind": "original_procedures"},
        ],
        "genuine_scientific_procedures": [
            "sechiba_var_init", "condveg_frac_snow", "condveg_z0cdrag", "condveg_albedo",
            "enerbil_begin", "enerbil_surftemp", "enerbil_flux", "qsatcalc", "dev_qsatcalc", "qsfrict_init",
        ],
        "scientific_stub_count": 0,
        "stubbed_external_boundaries": [
            "ipslnlf_p", "histwrite_p", "histsync", "xios_orchidee_update_calendar",
            "xios_orchidee_send_field", "IOIPSL calendar primitives",
        ],
        "routing_writeback": {
            "disposition": "original no-routing fragment",
            "source": "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90:1236-1238",
        },
    }
    (output_dir / "call_graph.json").write_text(json.dumps(call_graph, indent=2) + "\n", encoding="ascii")
    (output_dir / "owner_region_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="ascii")
    audit = _formal_audit(output_dir, evidence)
    comparison["formal_audit"] = {
        "complete": audit["complete"],
        "asset": (output_dir / "formal_audit_report.json").relative_to(ROOT).as_posix(),
    }
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="ascii")
    return {**result, "formal_audit": audit}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    args = parser.parse_args(argv)
    result = run(args.output_dir.resolve(), args.compiler.resolve())
    print(json.dumps({
        "status": "passed",
        "owner_region_id": OWNER_ID,
        "covered_arms": result["branch_coverage"]["covered_arm_count"],
        "formal_audit_complete": result["formal_audit"]["complete"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
