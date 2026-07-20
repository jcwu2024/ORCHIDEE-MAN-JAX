from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.parameters.control import control_initialize  # noqa: E402
from jax_orchidee.parameters.vertical_soil import vertical_soil_init  # noqa: E402
from scripts.dev.fortran_gcov import parse_gcov, parse_gcov_line_counts  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compiler_environment,
    exact_comparison,
    float_comparison,
    write_result,
)

FAMILY = "control_initialize_owner"
OWNER_ID = "pft14-owner-contract-c41705502dd2"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_parameters/control.f90"
OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles/control_initialize_owner"
GCOV = Path(r"C:\msys64\ucrt64\bin\gcov.exe")
RTOL = 1.0e-12
ATOL = 1.0e-14

# Each nonfatal source edge is exercised by at least one of these configurations.
CASES: dict[str, dict[str, object]] = {
    "paper_cwrr": {
        "SOILTYPE_CLASSIF": "usda", "RIVER_ROUTING": True, "HYDROL_CWRR": True,
        "DO_IRRIGATION": True, "OK_LEAK": True, "STOMATE_OK_STOMATE": True,
        "NVM": 14, "IMPOSE_PARAM": True,
    },
    "choisnel_minimal": {"SOILTYPE_CLASSIF": "zobler", "IMPOSE_PARAM": False},
    "rotation_bvoc_dgvm": {
        "SOILTYPE_CLASSIF": "fao", "OK_ROTATE": False, "DYN_PLNTDT": True,
        "CYC_ROT_MAX": 4, "STOMATE_OK_DGVM": True, "CHEMISTRY_BVOC": True,
        "CANOPY_MULTILAYER": True, "CANOPY_EXTINCTION": False, "IMPOSE_PARAM": True,
    },
    "rotation_active": {
        "SOILTYPE_CLASSIF": "none", "OK_ROTATE": True, "DYN_PLNTDT": True,
        "CYC_ROT_MAX": 3, "STOMATE_OK_CO2": True, "IMPOSE_PARAM": True,
    },
    "routing_fullirr": {
        "RIVER_ROUTING": True, "DO_FLOODPLAINS": True, "DO_FULLIRR": True,
        "CHECK_WATERBAL": True, "OK_PC": True, "CANOPY_MULTILAYER": True,
        "CANOPY_EXTINCTION": False,
    },
    "explicit_overrides": {
        "EROSION_MODULE": True, "OK_DAMRESERVOIR": True, "LIMIT_RIVDEPOS": False,
        "NVM_PLNT": True, "NVM_ROT": True, "NVM_NFERT": True, "ROT_CMD_MAX": 8,
        "OK_EXPLICITSNOW": True, "OK_PEAT": True, "PEAT_OCCUR": True,
        "PERMA_PEAT": True, "CHEMISTRY_LEAFAGE": True, "NOx_RAIN_PULSE": True,
        "NOx_BBG_FERTIL": True, "NOx_FERTILIZERS_USE": True,
        "CO2_FOR_BVOC_POSSELL": True, "CO2_FOR_BVOC_WILKINSON": True,
        "LD_DOC": True, "POOR_SOILS": True, "CHECK_RIVERBAL": True,
        "THERMOSOIL_NBLEV": 11, "DEPTH_MAX_H": 4.0,
    },
}

ERROR_CASES = {
    "cwrr_freeze_error": {"HYDROL_CWRR": True, "FREEZE": True, "DEPTH_MAX_T": 10.0},
    "choisnel_freeze_error": {"FREEZE": True, "THERMOSOIL_NBLEV": 7},
}

LOGICAL_KEYS = {
    "CHECKTIME", "RIVER_ROUTING", "EROSION_MODULE", "OK_DAMRESERVOIR", "LIMIT_RIVDEPOS",
    "HYDROL_CWRR", "DO_IRRIGATION", "DO_FULLIRR", "OK_ROTATE", "DYN_PLNTDT", "NVM_PLNT",
    "NVM_ROT", "NVM_NFERT", "DO_FLOODPLAINS", "CHECK_WATERBAL", "CHECK_RIVERBAL",
    "OK_EXPLICITSNOW", "OK_PC", "OK_PEAT", "PEAT_OCCUR", "PERMA_PEAT", "OK_LEAK",
    "STOMATE_OK_STOMATE", "STOMATE_OK_CO2", "STOMATE_OK_DGVM", "CHEMISTRY_BVOC",
    "CHEMISTRY_LEAFAGE", "CANOPY_EXTINCTION", "CANOPY_MULTILAYER", "NOx_RAIN_PULSE",
    "NOx_BBG_FERTIL", "NOx_FERTILIZERS_USE", "CO2_FOR_BVOC_POSSELL",
    "CO2_FOR_BVOC_WILKINSON", "LD_DOC", "POOR_SOILS", "IMPOSE_PARAM",
}
INTEGER_KEYS = {"CYC_ROT_MAX", "ROT_CMD_MAX", "NVM", "THERMOSOIL_NBLEV"}
REAL_KEYS = {"DEPTH_MAX_H"}

STUBS = r"""module oracle_config
  implicit none
  integer, save :: case_id = 0, call_count = 0, calls(32) = 0, fatal_count = 0
contains
  subroutine note_call(value)
    integer, intent(in) :: value
    call_count=call_count+1; calls(call_count)=value
  end subroutine
end module
module constantes_var
  use oracle_config
  implicit none
  integer, parameter :: i_std=4, r_std=8
  real(r_std), parameter :: deux=2.0_r_std
  integer, save :: printlev=0, numout=6
  real(r_std), save :: dt_sechiba
  logical, save :: check_time, river_routing, erosion_module, ok_damreservoir, limit_rivdepos
  logical, save :: hydrol_cwrr, do_irrigation, do_fullirr, ok_rotate, dyn_plntdt
  logical, save :: nvm_plnt, nvm_rot, nvm_nfert, do_floodplains, check_riverbal
  logical, save :: ok_explicitsnow, ok_pc, ok_peat, peat_occur, perma_peat, ok_leak
  logical, save :: ok_stomate, ok_co2, ok_dgvm, ok_bvoc, ok_leafage, ok_radcanopy
  logical, save :: ok_multilayer, ok_pulse_NOx, ok_bbgfertil_NOx, ok_cropsfertil_NOx
  logical, save :: ok_co2bvoc_poss, ok_co2bvoc_wilk, ld_doc, do_poor_soils
  logical, save :: ok_sechiba, ok_pheno, impose_param=.true.
  integer, save :: cyc_rot_max, rot_cmd_max=5
  interface getin_p
    module procedure getin_l, getin_i, getin_r, getin_c
  end interface
contains
  subroutine getin_l(name,value)
    character(*),intent(in)::name; logical,intent(inout)::value
    select case(case_id)
    case(1); call set_l(name,value,[character(len=24)::'RIVER_ROUTING','HYDROL_CWRR','DO_IRRIGATION','OK_LEAK','STOMATE_OK_STOMATE','IMPOSE_PARAM'])
    case(2); if(trim(name)=='IMPOSE_PARAM') value=.false.
    case(3); call set_l(name,value,[character(len=24)::'DYN_PLNTDT','STOMATE_OK_DGVM','CHEMISTRY_BVOC','CANOPY_MULTILAYER','IMPOSE_PARAM'])
    case(4); call set_l(name,value,[character(len=24)::'OK_ROTATE','DYN_PLNTDT','STOMATE_OK_CO2','IMPOSE_PARAM'])
    case(5); call set_l(name,value,[character(len=24)::'RIVER_ROUTING','DO_FLOODPLAINS','DO_FULLIRR','CHECK_WATERBAL','OK_PC','CANOPY_MULTILAYER'])
    case(6); call set_l(name,value,[character(len=24)::'EROSION_MODULE','OK_DAMRESERVOIR','NVM_PLNT','NVM_ROT','NVM_NFERT','OK_EXPLICITSNOW','OK_PEAT','PEAT_OCCUR','PERMA_PEAT','CHEMISTRY_LEAFAGE','NOx_RAIN_PULSE','NOx_BBG_FERTIL','NOx_FERTILIZERS_USE','CO2_FOR_BVOC_POSSELL','CO2_FOR_BVOC_WILKINSON','LD_DOC','POOR_SOILS','CHECK_RIVERBAL']); if(trim(name)=='LIMIT_RIVDEPOS') value=.false.
    case(7); if(trim(name)=='HYDROL_CWRR') value=.true.
    end select
  end subroutine
  subroutine set_l(name,value,names)
    character(*),intent(in)::name; logical,intent(inout)::value; character(*),intent(in)::names(:)
    integer i; do i=1,size(names); if(trim(name)==trim(names(i))) value=.true.; enddo
  end subroutine
  subroutine getin_i(name,value)
    character(*),intent(in)::name; integer,intent(inout)::value
    if(trim(name)=='NVM' .and. case_id==1) value=14
    if(trim(name)=='CYC_ROT_MAX' .and. case_id==3) value=4
    if(trim(name)=='CYC_ROT_MAX' .and. case_id==4) value=3
    if(trim(name)=='ROT_CMD_MAX' .and. case_id==6) value=8
    if(trim(name)=='THERMOSOIL_NBLEV' .and. case_id==6) value=11
  end subroutine
  subroutine getin_r(name,value)
    character(*),intent(in)::name; real(r_std),intent(inout)::value
    if(trim(name)=='DEPTH_MAX_H' .and. case_id==6) value=4.0_r_std
  end subroutine
  subroutine getin_c(name,value)
    character(*),intent(in)::name; character(*),intent(inout)::value
    if(trim(name)=='SOILTYPE_CLASSIF') then
      select case(case_id); case(1);value='usda';case(3);value='fao';case(4);value='none';case default;value='zobler';end select
    endif
  end subroutine
  subroutine ipslerr_p(level,a,b,c,d)
    integer,intent(in)::level; character(*),intent(in)::a,b,c,d
    if(level>=3) fatal_count=fatal_count+1
  end subroutine
end module
module constantes_soil
  use constantes_var, only:i_std
  use oracle_config
  implicit none
  integer, parameter :: nscm_fao=3, nscm_usda=12
  integer, save :: nscm=nscm_fao
  character(30), save :: soil_classif
  logical, save :: check_waterbal, ok_freeze_thermix=.false.
contains
  subroutine config_soil_parameters(); call note_call(7); end subroutine
end module
module pft_parameters
  use oracle_config
  implicit none
  integer, save :: nvm=13
contains
  subroutine pft_parameters_main(); call note_call(1); end subroutine
  subroutine activate_sub_models(); call note_call(2); end subroutine
  subroutine veget_config(); call note_call(3); end subroutine
  subroutine config_pft_parameters(); call note_call(4); end subroutine
  subroutine config_sechiba_parameters(); call note_call(5); end subroutine
  subroutine config_sechiba_pft_parameters(); call note_call(6); end subroutine
  subroutine config_co2_parameters(); call note_call(8); end subroutine
  subroutine config_stomate_parameters(); call note_call(9); end subroutine
  subroutine config_stomate_pft_parameters(); call note_call(10); end subroutine
  subroutine config_dgvm_parameters(); call note_call(11); end subroutine
end module
module vertical_soil
  use constantes_var, only:r_std
  use oracle_config
  implicit none
  integer, save :: nslm=0, ngrnd=0
  real(r_std), save :: zmaxh=0, zmaxt=38
  real(r_std), allocatable, save :: znt(:), diaglev(:)
contains
  subroutine vertical_soil_init
    integer i
    call note_call(0); nslm=@NSLM@; ngrnd=@NGRND@; zmaxh=2.0_r_std
    zmaxt=38.0_r_std; if(case_id==7) zmaxt=10.0_r_std
    allocate(znt(ngrnd)); znt=(/@ZNT@/)
  end subroutine
end module
"""

PROBE = r"""
program probe
  use control
  use oracle_config
  use constantes_var
  use constantes_soil, only: soil_classif,nscm,check_waterbal,ok_freeze_thermix
  use pft_parameters, only:nvm
  use vertical_soil, only:nslm,ngrnd,zmaxh,zmaxt,znt,diaglev
  implicit none
  character(512)::arg; integer::u,i
  call get_command_argument(1,arg); read(arg,*) case_id
  if(case_id==7 .or. case_id==8) ok_freeze_thermix=.true.
  call control_initialize(1800.0_r_std)
  call get_command_argument(2,arg); open(newunit=u,file=trim(arg),status='replace')
  call wr(u,'dt_sechiba',dt_sechiba); call wl(u,'check_time',check_time)
  call wc(u,'soil_classif',soil_classif); call wi(u,'nscm',nscm)
  call wl(u,'river_routing',river_routing); call wl(u,'erosion_module',erosion_module)
  call wl(u,'ok_damreservoir',ok_damreservoir); call wl(u,'limit_rivdepos',limit_rivdepos)
  call wl(u,'hydrol_cwrr',hydrol_cwrr); call wl(u,'do_irrigation',do_irrigation)
  call wl(u,'do_fullirr',do_fullirr); call wl(u,'ok_rotate',ok_rotate); call wl(u,'dyn_plntdt',dyn_plntdt)
  call wl(u,'nvm_plnt',nvm_plnt); call wl(u,'nvm_rot',nvm_rot); call wl(u,'nvm_nfert',nvm_nfert)
  call wi(u,'cyc_rot_max',cyc_rot_max); call wi(u,'rot_cmd_max',rot_cmd_max)
  call wl(u,'do_floodplains',do_floodplains); call wl(u,'check_waterbal',check_waterbal)
  call wl(u,'check_riverbal',check_riverbal); call wl(u,'ok_explicitsnow',ok_explicitsnow)
  call wl(u,'ok_pc',ok_pc); call wl(u,'ok_peat',ok_peat); call wl(u,'peat_occur',peat_occur)
  call wl(u,'perma_peat',perma_peat); call wl(u,'ok_leak',ok_leak); call wl(u,'ok_stomate',ok_stomate)
  call wl(u,'ok_co2',ok_co2); call wl(u,'ok_dgvm',ok_dgvm); call wl(u,'ok_bvoc',ok_bvoc)
  call wl(u,'ok_leafage',ok_leafage); call wl(u,'ok_radcanopy',ok_radcanopy)
  call wl(u,'ok_multilayer',ok_multilayer); call wl(u,'ok_pulse_nox',ok_pulse_NOx)
  call wl(u,'ok_bbgfertil_nox',ok_bbgfertil_NOx); call wl(u,'ok_cropsfertil_nox',ok_cropsfertil_NOx)
  call wl(u,'ok_co2bvoc_poss',ok_co2bvoc_poss); call wl(u,'ok_co2bvoc_wilk',ok_co2bvoc_wilk)
  call wl(u,'ld_doc',ld_doc); call wl(u,'do_poor_soils',do_poor_soils); call wl(u,'ok_sechiba',ok_sechiba)
  call wl(u,'ok_pheno',ok_pheno); call wi(u,'nvm',nvm); call wl(u,'impose_param',impose_param)
  call wi(u,'nslm',nslm); call wi(u,'ngrnd',ngrnd); call wr(u,'zmaxh',zmaxh)
  do i=1,size(diaglev); call wr(u,'diaglev',diaglev(i)); enddo
  do i=1,call_count; if(calls(i)/=0) call wi(u,'parameter_initializers',calls(i)); enddo
  do i=1,call_count; call wi(u,'owner_call_order',calls(i)); enddo
  call wi(u,'fatal_count',fatal_count); close(u)
contains
  subroutine wr(u,n,v); integer,intent(in)::u;character(*),intent(in)::n;real(r_std),intent(in)::v;write(u,'(A,",",ES25.17E3)')trim(n),v;end subroutine
  subroutine wi(u,n,v); integer,intent(in)::u,v;character(*),intent(in)::n;write(u,'(A,",",I0)')trim(n),v;end subroutine
  subroutine wl(u,n,v); integer,intent(in)::u;logical,intent(in)::v;character(*),intent(in)::n;write(u,'(A,",",I0)')trim(n),merge(1,0,v);end subroutine
  subroutine wc(u,n,v); integer,intent(in)::u;character(*),intent(in)::n,v;write(u,'(A,",",A)')trim(n),trim(v);end subroutine
end program
"""

CALL_IDS = {
    "pft_parameters_main": 1, "activate_sub_models": 2, "veget_config": 3,
    "config_pft_parameters": 4, "config_sechiba_parameters": 5,
    "config_sechiba_pft_parameters": 6, "config_soil_parameters": 7,
    "config_co2_parameters": 8, "config_stomate_parameters": 9,
    "config_stomate_pft_parameters": 10, "config_dgvm_parameters": 11,
}


def compose(path: Path) -> dict[str, object]:
    original = SOURCE.read_bytes()
    grid = vertical_soil_init()
    znt = ",".join(f"{value:.17e}_r_std" for value in grid.znt)
    stubs = STUBS.replace("@NSLM@", str(grid.nslm)).replace("@NGRND@", str(grid.ngrnd)).replace("@ZNT@", znt)
    path.write_bytes(stubs.encode("ascii") + original + PROBE.encode("ascii"))
    return {
        "source_file_sha256": hashlib.sha256(original).hexdigest(),
        "embedded_source_sha256": hashlib.sha256(original).hexdigest(),
        "embedded_start_line": stubs.count("\n") + 1,
        "embedded_byte_count": len(original),
        "byte_exact": original in path.read_bytes(),
    }


def _read(path: Path) -> dict[str, Any]:
    out: dict[str, list[str]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for name, value in csv.reader(handle):
            out.setdefault(name, []).append(value)
    result: dict[str, Any] = {}
    for name, values in out.items():
        if name == "soil_classif":
            result[name] = values[0]
        elif name in {"diaglev"}:
            result[name] = np.asarray(values, dtype=np.float64)
        elif name == "parameter_initializers":
            result[name] = tuple(int(v) for v in values)
        elif name == "owner_call_order":
            result[name] = tuple(int(v) for v in values)
        else:
            result[name] = float(values[0])
    return result


def _expected(config: dict[str, object]) -> dict[str, Any]:
    grid = vertical_soil_init(config) if config.get("HYDROL_CWRR", False) else None
    state = control_initialize(
        config,
        dt=1800.0,
        znt=None if grid is None else grid.znt,
        nslm=None if grid is None else grid.nslm,
        zmaxt=None if grid is None else grid.depth_max_t,
        ok_freeze_thermix=False,
    )
    expected = {field.name: getattr(state, field.name) for field in fields(state)}
    expected.pop("provenance")
    expected["parameter_initializers"] = tuple(CALL_IDS[name] for name in state.parameter_initializers)
    return expected


def _compile(source: Path, executable: Path, compiler: Path) -> dict[str, object]:
    flags = ["-std=f2008", "-fdefault-real-8", "-ffree-line-length-none", "-O0", "-fcheck=all", "-fprofile-arcs", "-ftest-coverage"]
    command = [str(compiler), *flags, str(source), "-o", str(executable)]
    done = subprocess.run(command, cwd=source.parent, env=compiler_environment(compiler), text=True, capture_output=True)
    if done.returncode != 0:
        raise RuntimeError(f"gfortran failed ({done.returncode}):\n{done.stderr}")
    return {"compiler": str(compiler), "compile_flags": flags, "compile_stderr": done.stderr}


def _arm_coverage(gcov_path: Path, owner: dict[str, Any], line_offset: int) -> dict[str, Any]:
    branches = parse_gcov(gcov_path)
    line_counts = parse_gcov_line_counts(gcov_path)
    records = []
    for arm_id in owner["arm_ids"]:
        line = int(str(arm_id).split(":")[-3])
        kind, side = str(arm_id).rsplit(":", 2)[-2:]
        rows = branches.get(line + line_offset, [])
        if kind == "case":
            mapped_line = line + line_offset + 1
            passed = line_counts.get(mapped_line, 0) > 0
        elif len(rows) >= 2:
            selected = rows[-2] if side == "true" else rows[-1]
            passed = selected["taken"] > 0
        else:
            passed = False
        records.append({
            "arm_id": arm_id,
            "passed": passed,
            "gcov_mapped_line": mapped_line if kind == "case" else line + line_offset,
            "gcov_line_count": line_counts.get(mapped_line if kind == "case" else line + line_offset, 0),
            "raw_gcov_branches": rows,
        })
    return {"records": records, "covered": {r["arm_id"] for r in records if r["passed"]}}


def run_oracle(output_dir: Path = OUTPUT_DIR, compiler: Path = DEFAULT_COMPILER, gcov: Path = GCOV) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    contracts = json.loads((ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json").read_text(encoding="utf-8"))
    owner = next(item for item in contracts["owner_regions"] if item["owner_region_id"] == OWNER_ID)
    comparisons: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="orchidee_control_owner_") as td:
        build = Path(td)
        source = build / "control_owner_oracle.f90"
        executable = build / "oracle.exe"
        source_meta = compose(source)
        compiler_meta = _compile(source, executable, compiler)
        for case_id, (name, config) in enumerate(CASES.items(), 1):
            path = build / f"{name}.csv"
            subprocess.run([str(executable), str(case_id), str(path)], cwd=build, env=compiler_environment(compiler), check=True, capture_output=True, text=True)
            actual, expected = _read(path), _expected(config)
            for field_name, expected_value in expected.items():
                actual_value = actual[field_name]
                label = f"{name}.{field_name}"
                if field_name == "soil_classif":
                    comparisons.append({"name": label, "comparison": "exact", "passed": actual_value == expected_value})
                elif field_name == "diaglev":
                    comparisons.append(float_comparison(label, actual_value, expected_value, rtol=RTOL, atol=ATOL))
                elif field_name == "parameter_initializers":
                    comparisons.append(exact_comparison(label, actual_value, expected_value))
                elif isinstance(expected_value, (float, np.floating)):
                    comparisons.append(float_comparison(label, [actual_value], [expected_value], rtol=RTOL, atol=ATOL))
                else:
                    comparisons.append(exact_comparison(label, [int(actual_value)], [int(expected_value)]))
            expected_owner_calls = ((0,) if config.get("HYDROL_CWRR", False) else ()) + expected["parameter_initializers"]
            comparisons.append(exact_comparison(f"{name}.owner_call_order", actual["owner_call_order"], expected_owner_calls))
        for case_id, name in ((7, "cwrr_freeze_error"), (8, "choisnel_freeze_error")):
            path = build / f"{name}.csv"
            subprocess.run([str(executable), str(case_id), str(path)], cwd=build, env=compiler_environment(compiler), check=True, capture_output=True, text=True)
            comparisons.append(exact_comparison(f"{name}.fatal_boundary", [_read(path)["fatal_count"]], [1]))
        notes = next(build.glob("*.gcno"))
        subprocess.run([str(gcov), "-b", "-c", notes.name], cwd=build, env=compiler_environment(compiler), check=True, capture_output=True, text=True)
        listing = build / f"{source.name}.gcov"
        coverage = _arm_coverage(listing, owner, int(source_meta["embedded_start_line"]) - 1)
        (output_dir / "control.f90").write_bytes(SOURCE.read_bytes())
        (output_dir / "control_owner_oracle.gcov").write_bytes(listing.read_bytes())

    required = set(owner["arm_ids"])
    covered = coverage["covered"]
    branch_doc = {
        "schema_version": 1, "family": FAMILY, "branch_complete": required == covered,
        "required_arm_count": len(required), "covered_arm_count": len(covered),
        "missing_arm_ids": sorted(required - covered), "records": coverage["records"],
        "coverage_kind": "gcov_original_source_statement_and_branch_execution",
        "classification_errors": [],
    }
    evidence = {
        "schema_version": 1, "family": FAMILY, "complete": required == covered,
        "records": [{
            "owner_region_id": OWNER_ID, "base_region_id": owner["base_region_id"],
            "fortran_procedure": owner["fortran_procedure"], "jax_owners": owner["jax_owners"],
            "required_arm_ids": sorted(required), "covered_arm_ids": sorted(covered),
            "missing_arm_ids": sorted(required - covered), "passed": required == covered,
            "classification_errors": [],
            "comparison_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/comparison.json",
            "branch_coverage_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/branch_coverage.json",
        }],
    }
    (output_dir / "branch_coverage.json").write_text(json.dumps(branch_doc, indent=2) + "\n", encoding="ascii")
    (output_dir / "owner_region_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="ascii")
    (output_dir / "inputs.json").write_text(json.dumps({"cases": CASES, "error_cases": ERROR_CASES}, indent=2) + "\n", encoding="ascii")
    return write_result(output_dir, FAMILY, comparisons, {
        **source_meta, **compiler_meta, "owner_region_id": OWNER_ID,
        "state_contract_fields": [field.name for field in fields(control_initialize({}, dt=1800.0)) if field.name != "provenance"],
        "tolerance": {"float_rtol": RTOL, "float_atol": ATOL, "discrete": "exact"},
        "save_call_boundary": "Original control module and USE-associated module SAVE state; one initialization per process because diaglev is allocated without re-entry deallocation.",
        "branch_complete": required == covered,
        "classification_errors": [],
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the byte-exact control_initialize owner oracle.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    parser.add_argument("--gcov", type=Path, default=GCOV)
    args = parser.parse_args(argv)
    try:
        result = run_oracle(args.output_dir.resolve(), args.compiler.resolve(), args.gcov.resolve())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
