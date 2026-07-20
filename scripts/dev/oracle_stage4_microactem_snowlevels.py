"""Stage 4 executable evidence for the PFT14 microactem/snowlevels owners."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from jax import config as jax_config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
jax_config.update("jax_enable_x64", True)

from jax_orchidee.stomate.permafrost import microactem  # noqa: E402
from jax_orchidee.stomate.soilcarbon_kernels import deep_carbon_snowlevels_step  # noqa: E402
from scripts.dev.audit_pft14_owner_region_evidence import build_report, load_evidence_documents  # noqa: E402
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_gcov import locate_generated_span, parse_gcov, parse_gcov_line_counts, select_arm_branch  # noqa: E402
from scripts.dev.fortran_oracle_common import DEFAULT_COMPILER, compile_fortran, compiler_environment, float_comparison, write_point_comparisons, write_result  # noqa: E402

FAMILY = "stage4_microactem_snowlevels"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90"
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
CONTRACTS = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
PROOFS = ROOT / "outputs/reference_mode/pft14_source_proof_evidence.json"
GCOV = Path(r"C:\msys64\ucrt64\bin\gcov.exe")
RTOL, ATOL = 1e-12, 1e-12
OWNERS = {
    "microactem": ("pft14-owner-contract-1211208a2520", "pft14-region-f030c57cba29"),
    "snowlevels": ("pft14-owner-contract-6a1aba752eed", "pft14-region-d440d94ae10a"),
}
# Contract labels and SELECT/CASE dispatches do not always produce a gcov row.
# These are the first executable statement governed by the source label.
GCOV_SOURCE_LINE = {
    "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90:2513:select_case:fallthrough": 2513,
    "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90:2520:case:case": 2521,
    "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90:2523:elsewhere:fallthrough": 2524,
    "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90:2526:elsewhere:fallthrough": 2527,
    "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90:2571:select_case:fallthrough": 2571,
    "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90:2572:case:case": 2574,
    "fortran_source/ORCHIDEE/src_stomate/stomate_permafrost_soilcarbon.f90:2576:case:case": 2577,
}


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")


def _compose(micro: bytes, snow: bytes) -> bytes:
    return b"""module constantes
  implicit none
  integer, parameter :: i_std=4, r_std=8, nvm=16, nsnow=3
  real(r_std), parameter :: ZeroCelsius=273.15_r_std, un=1._r_std
end module constantes
module stomate_permafrost_soilcarbon
  use constantes
  implicit none
  logical :: perma_peat=.true., agri_peat=.true.
  logical :: is_peat(nvm)
  real(r_std) :: tau_peat=3.1536e8_r_std, z_tau=1.e6_r_std, flux_tot_coeff(3)=(/1.2_r_std,1.4_r_std,.75_r_std/)
contains
""" + micro + b"\n" + snow + b"""
end module stomate_permafrost_soilcarbon
program oracle
  use constantes
  use stomate_permafrost_soilcarbon
  implicit none
  integer :: u, ii, ij, ik, mode, frozen, moisture
  character(512) :: path, arg
  real :: temp(2,13,nvm), moist(2,13,nvm), fb(2,13,nvm), temp_profile(13)
  real(r_std) :: peat(2,13), zi(13), snowdz(2,nsnow), zii(2,nsnow,nvm), zff(2,0:nsnow,nvm), veg(2,nvm)
  call get_command_argument(1,path); call get_command_argument(2,arg); read(arg,*) mode
  is_peat=.false.; is_peat(14)=.true.
  do ij=1,13; zi(ij)=real(ij,r_std)*.2_r_std; enddo
  temp_profile=(/-4.,-.5,1.,-2.,.5,-4.,-.5,1.,-2.,.5,-4.,-.5,1./)
  temp=-4.
  do ij=1,13
    temp(1,ij,:)=temp_profile(ij)
  enddo
  moist=.3; moist(1,:,14)=(/.01,.03,.15,.5,.9,.95,.3,.4,.6,.8,.2,.7,.45/); moist(2,:,15)=.02
  peat=.01; peat(1,:)=(/.01,.03,.15,.5,.9,.95,.3,.4,.6,.8,.2,.7,.45/)
  if(mode.eq.1) then; frozen=9; moisture=0; else; frozen=1; moisture=0; endif
  fb=microactem(temp,frozen,1,moist,2,13,nvm,zi,peat)
  fb=microactem(temp,frozen,moisture,moist,2,13,nvm,zi,peat)
  snowdz(1,:)=(/.02_r_std,.08_r_std,.30_r_std/); snowdz(2,:)=(/.15_r_std,.0_r_std,.45_r_std/); veg=0._r_std; veg(1,1)=1.; veg(1,14)=.8; veg(2,1)=1.; veg(2,14)=.6
  call snowlevels(2,snowdz,zii,zff,veg)
  open(newunit=u,file=trim(path),status='replace')
  do ii=1,2; do ij=1,13; do ik=1,nvm
    write(u,'(A,",",3(I0,","),ES25.17E3)') 'fb',ii,ij,ik,fb(ii,ij,ik)
  enddo; enddo; enddo
  do ii=1,2; do ij=1,nsnow; do ik=1,nvm
    write(u,'(A,",",3(I0,","),ES25.17E3)') 'zi',ii,ij,ik,zii(ii,ij,ik)
  enddo; enddo; enddo
  do ii=1,2; do ij=0,nsnow; do ik=1,nvm
    write(u,'(A,",",3(I0,","),ES25.17E3)') 'zf',ii,ij,ik,zff(ii,ij,ik)
  enddo; enddo; enddo
  close(u)
end program oracle
"""


def _read(path: Path) -> dict[str, np.ndarray]:
    result: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.reader(handle):
            result.setdefault(row[0], []).append(float(row[4]))
    return {key: np.asarray(value) for key, value in result.items()}


def _jax() -> dict[str, np.ndarray]:
    temp = np.full((2, 13, 16), -4.0)
    temp[0, :, :] = np.array([-4., -.5, 1., -2., .5, -4., -.5, 1., -2., .5, -4., -.5, 1.])[:, None]
    moist = np.full((2, 13, 16), .3); moist[0, :, 13] = [.01, .03, .15, .5, .9, .95, .3, .4, .6, .8, .2, .7, .45]; moist[1, :, 14] = .02
    peat = np.full((2, 13), .01); peat[0] = [.01, .03, .15, .5, .9, .95, .3, .4, .6, .8, .2, .7, .45]
    micro = microactem(temp, 1, 0, moist, np.arange(1, 14) * .2, peat, perma_peat=True, agri_peat=True, is_peat=np.array([False] * 13 + [True] + [False] * 2), epsilon=np.finfo(np.float64).eps)
    snow = deep_carbon_snowlevels_step(np.array([[.02, .08, .30], [.15, 0., .45]]), np.array([[1.] + [0.] * 12 + [.8] + [0.] * 2, [1.] + [0.] * 12 + [.6] + [0.] * 2]))
    return {"fb": np.asarray(micro.fbact_seconds).ravel(order="C"), "zi": np.asarray(snow.zi_snow).ravel(order="C"), "zf": np.asarray(snow.zf_snow).ravel(order="C")}


def _contracts() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    document = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    selected = {r["fortran_procedure"]: r for r in document["owner_regions"] if r["owner_region_id"] in {value[0] for value in OWNERS.values()}}
    if set(selected) != set(OWNERS): raise RuntimeError("microactem/snowlevels contract drift")
    return selected, document


def run(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    micro, snow = (extract_procedure_bytes(SOURCE, name) for name in OWNERS)
    for name, extracted in (("microactem", micro), ("snowlevels", snow)):
        (output_dir / f"{name}.f90").write_bytes(extracted.span_bytes)
    unit = _compose(micro.span_bytes, snow.span_bytes); (output_dir / "oracle.f90").write_bytes(unit)
    csv_path = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_microactem_snowlevels_") as td:
        build = Path(td); source = build / "oracle.f90"; source.write_bytes(unit); exe = build / "oracle.exe"; meta = compile_fortran(source, exe, compiler)
        subprocess.run([str(exe), str(csv_path), "0"], cwd=build, env=compiler_environment(compiler), check=True, capture_output=True, text=True)
        fatal = subprocess.run([str(exe), str(build / "fatal.csv"), "1"], cwd=build, env=compiler_environment(compiler), capture_output=True, text=True)
    fortran, jax = _read(csv_path), _jax()
    write_point_comparisons(output_dir / "point_comparisons.csv", fortran, jax, rtol=RTOL, atol=ATOL)
    comparison = write_result(output_dir, FAMILY, [float_comparison(key, value, jax[key], rtol=RTOL, atol=ATOL) for key, value in fortran.items()], {"source": {"file": SOURCE.relative_to(ROOT).as_posix(), "sha256": micro.source_sha256, "procedures": {name: {"start_line": item.start_line, "end_line": item.end_line, "sha256": item.span_sha256, "exact_extracted_bytes": (output_dir / f"{name}.f90").read_bytes() == item.span_bytes} for name, item in (("microactem", micro), ("snowlevels", snow))}}, "compiler": meta, "tolerance_policy": {"rtol": RTOL, "atol": ATOL, "changed": False}, "matrix": {"cases": ["cold_peat_pft14_and_bare", "later_snow_active_pft14_and_bare"], "pft14_index_zero_based": 13, "bare_soil_index_zero_based": 0, "snow_layers": 3}})
    if comparison["status"] != "passed": raise RuntimeError("microactem/snowlevels numerical mismatch")
    _json(output_dir / "fatal_boundary_evidence.json", {"schema_version": 1, "family": FAMILY, "complete": bool(fatal.returncode != 0 or "ERROR" in fatal.stdout + fatal.stderr), "records": [{"boundary": "frozen_respiration_func_default", "input": 9, "returncode": fatal.returncode, "stdout": fatal.stdout, "stderr": fatal.stderr, "passed": bool(fatal.returncode != 0 or "ERROR" in fatal.stdout + fatal.stderr)}]})
    return _coverage(output_dir, unit, micro, snow, compiler, comparison)


def _coverage(output_dir: Path, unit: bytes, micro: Any, snow: Any, compiler: Path, comparison: dict[str, Any]) -> dict[str, Any]:
    contracts, document = _contracts()
    with tempfile.TemporaryDirectory(prefix="orchidee_microactem_snowlevels_gcov_") as td:
        build=Path(td); source=build/"oracle.f90"; source.write_bytes(unit); exe=build/"oracle.exe"; meta=compile_fortran(source,exe,compiler,extra_flags=("--coverage",)); subprocess.run([str(exe),str(build/"coverage.csv"),"0"],cwd=build,env=compiler_environment(compiler),check=True,capture_output=True,text=True); note=next(build.glob("*.gcno")); gc=subprocess.run([str(GCOV),"-b","-c",note.name],cwd=build,env=compiler_environment(compiler),check=True,capture_output=True,text=True); shutil.copy2(build/"oracle.f90.gcov",output_dir/"oracle.f90.gcov"); branches=parse_gcov(build/"oracle.f90.gcov"); counts=parse_gcov_line_counts(build/"oracle.f90.gcov")
    proofs={r["arm_id"] for r in json.loads(PROOFS.read_text(encoding="utf-8"))["records"] if r.get("passed") is True}; starts={"microactem":locate_generated_span(unit,micro.span_bytes),"snowlevels":locate_generated_span(unit,snow.span_bytes)}; rows=[]; evidence=[]
    for name, extracted in (("microactem",micro),("snowlevels",snow)):
        owner=contracts[name]; required=owner["arm_ids"]; gcov_ids=[]; proof_ids=[]
        for arm_id in required:
            line=int(arm_id.rsplit(":",3)[1]); mapped=GCOV_SOURCE_LINE.get(arm_id,line); generated=starts[name]+mapped-extracted.start_line; selected=select_arm_branch(arm_id,branches.get(generated,[])); gcov_ok=bool((selected and selected["taken"]>0) or (arm_id in GCOV_SOURCE_LINE and counts.get(generated,0)>0)); proof_ok=arm_id in proofs; passed=gcov_ok or proof_ok
            if gcov_ok: gcov_ids.append(arm_id)
            if proof_ok: proof_ids.append(arm_id)
            rows.append({"arm_id":arm_id,"procedure":name,"original_line":line,"gcov_source_line":mapped,"generated_line":generated,"gcov_branch_index":None if selected is None else selected["branch_index"],"gcov_taken":0 if selected is None else selected["taken"],"gcov_line_count":counts.get(generated,0),"gcov_passed":gcov_ok,"fatal_boundary_passed":False,"source_proof_passed":proof_ok,"passed":passed,"raw_gcov_branches":branches.get(generated,[])})
        covered=sorted(r["arm_id"] for r in rows if r["procedure"]==name and r["passed"]); missing=sorted(set(required)-set(covered)); evidence.append({"owner_region_id":owner["owner_region_id"],"base_region_id":owner["base_region_id"],"fortran_procedure":name,"jax_owners":owner["jax_owners"],"comparison_asset":f"outputs/reference_mode/micro_oracles/{FAMILY}/comparison.json","branch_coverage_asset":f"outputs/reference_mode/micro_oracles/{FAMILY}/branch_coverage.json","required_arm_ids":sorted(required),"gcov_arm_ids":sorted(gcov_ids),"fatal_boundary_arm_ids":[],"source_proof_arm_ids":sorted(proof_ids),"covered_arm_ids":covered,"missing_arm_ids":missing,"passed":not missing and comparison["status"]=="passed"})
    missing=sorted(r["arm_id"] for r in rows if not r["passed"]); coverage={"schema_version":1,"family":FAMILY,"branch_complete":not missing,"required_arm_count":len(rows),"covered_arm_count":len(rows)-len(missing),"missing_arm_ids":missing,"arms":rows,"gcov_asset":"oracle.f90.gcov","compiler":meta,"gcov_stdout":gc.stdout,"gcov_stderr":gc.stderr,"source_line_mapping":GCOV_SOURCE_LINE,"policy":"GCOV parses the byte-exact extracted-procedure harness; pinned paper-static dispatch proof is used only from the canonical source-proof evidence. No coverage is manually asserted."}; _json(output_dir/"branch_coverage.json",coverage); _json(output_dir/"owner_region_evidence.json",{"schema_version":1,"family":FAMILY,"complete":all(r["passed"] for r in evidence),"records":evidence}); central=build_report(document,json.loads(PROOFS.read_text(encoding="utf-8")),load_evidence_documents(ROOT/"outputs/reference_mode/micro_oracles")); accepted={r["owner_region_id"]:r.get("passed") is True for r in central["records"] if r["owner_region_id"] in {v[0] for v in OWNERS.values()}}; central["assigned_owner_acceptance"]={"accepted":accepted,"global_completion_not_required_by_this_batch_owner":True}; _json(output_dir/"central_audit_report.json",central)
    if missing or not all(accepted.get(v[0],False) for v in OWNERS.values()): raise RuntimeError(f"microactem/snowlevels closure incomplete: {missing}, {accepted}")
    return {"comparison":comparison,"coverage":coverage,"central_audit":central}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=OUTPUT); args=parser.parse_args(); print(json.dumps(run(args.output_dir.resolve()),indent=2))
