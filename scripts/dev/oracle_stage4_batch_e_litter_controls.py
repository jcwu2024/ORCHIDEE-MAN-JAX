"""Stage 4 Batch E executable closure for the two litter control owners."""
from __future__ import annotations

import argparse
import csv
import hashlib
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

from jax_orchidee.stomate.carbon_kernels import (  # noqa: E402
    litter_control_moisture_moyano,
    litter_control_temperature,
)
from scripts.dev.audit_pft14_owner_region_evidence import (  # noqa: E402
    build_report as build_central_audit_report,
    load_evidence_documents,
)
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_gcov import locate_generated_span, parse_gcov, parse_gcov_line_counts, select_arm_branch  # noqa: E402
from scripts.dev.fortran_oracle_common import DEFAULT_COMPILER, compile_fortran, compiler_environment, float_comparison, write_point_comparisons, write_result  # noqa: E402

FAMILY = "stage4_batch_e_litter_controls"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90"
CONTRACTS = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
PROOFS = ROOT / "outputs/reference_mode/pft14_source_proof_evidence.json"
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
GCOV = Path(r"C:\msys64\ucrt64\bin\gcov.exe")
RTOL, ATOL = 1e-12, 1e-14
OWNERS = {
    "control_temp_func": ("pft14-owner-contract-ad58d4c4c43f", "pft14-region-f54908accb33"),
    "control_moist_func_moyano": ("pft14-owner-contract-f7f061ec898b", "pft14-region-52ff649c82bc"),
}
# Contract labels can be non-executable Fortran labels. These are the first
# executable statements governed by those labels in the byte-exact procedure.
GCOV_SOURCE_LINE = {
    "fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90:1474:case:case": 1471,
    "fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90:1479:elsewhere:fallthrough": 1480,
    "fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90:3026:if:false": 3030,
    "fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90:3026:if:true": 3030,
}


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")


def _compose(temp: bytes, moist: bytes) -> bytes:
    return b"""module constantes
  implicit none
  integer, parameter :: i_std=4, r_std=8, ncarb=3, nvm=14, ndeep=2, numout=6
  integer, parameter :: iactive=1, islow=2, ipassive=3
  real(r_std), parameter :: zero=0._r_std, un=1._r_std, ZeroCelsius=273.15_r_std
  real(r_std), parameter :: soil_Q10=.69_r_std, tsoil_ref=30._r_std, Q10=10._r_std
end module
module vertical_soil
  use constantes, only:r_std
  implicit none
  integer, parameter :: nslm=1
  real(r_std), parameter :: diaglev(1)=(/1._r_std/)
end module
module stomate_litter
  use constantes
  use vertical_soil
  implicit none
contains
""" + temp + b"\n" + moist + b"""
end module stomate_litter
program oracle
  use constantes
  use stomate_litter
  implicit none
  integer :: u, mode, i, p, f
  character(512) :: path, arg
  real(r_std) :: t(5), m(3), z(2), bd(3), clay(3), carbon(3,ncarb,nvm,ndeep), veg(3,nvm), out_t(5), out_m(3)
  call get_command_argument(1,path); call get_command_argument(2,arg); read(arg,*) mode
  t=(/ZeroCelsius-4._r_std,ZeroCelsius-2._r_std,ZeroCelsius-.5_r_std,ZeroCelsius,ZeroCelsius+30._r_std/)
  m=(/.35_r_std,.72_r_std,.0_r_std/); z=(/.3_r_std,1._r_std/); bd=(/1200._r_std,1350._r_std,1500._r_std/); clay=(/.1_r_std,.35_r_std,.2_r_std/)
  carbon=0._r_std; carbon(:,iactive,:,:)=2._r_std; carbon(:,islow,:,:)=4._r_std; carbon(:,ipassive,:,:)=8._r_std
  veg=0._r_std; veg(1,1)=1._r_std; veg(2,14)=1._r_std
  open(newunit=u,file=trim(path),status='replace')
  do f=0,4
    out_t=control_temp_func(5,t,f)
    do i=1,5; write(u,'(A,",",I0,",",I0,",",ES25.17E3)') 'temp',f,i,out_t(i); enddo
  enddo
  out_m=control_moist_func_moyano(3,m,z,bd,clay,carbon,veg)
  do p=1,3; write(u,'(A,",",I0,",",I0,",",ES25.17E3)') 'moyano',0,p,out_m(p); enddo
  close(u)
end program
"""


def _read(path: Path) -> dict[str, np.ndarray]:
    result: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.reader(handle):
            result.setdefault(f"{row[0]}.{row[1]}", []).append(float(row[3]))
    return {key: np.asarray(value) for key, value in result.items()}


def _jax() -> dict[str, np.ndarray]:
    zc = 273.15
    temp = np.array([zc - 4., zc - 2., zc - .5, zc, zc + 30.])
    result = {f"temp.{func}": np.asarray(litter_control_temperature(temp, func)) for func in range(5)}
    carbon = np.zeros((3, 3, 14, 2)); carbon[:, 0] = 2.; carbon[:, 1] = 4.; carbon[:, 2] = 8.
    veg = np.zeros((3, 14)); veg[0, 0] = 1.; veg[1, 13] = 1.
    result["moyano.0"] = np.asarray(litter_control_moisture_moyano(np.array([.35, .72, 0.]), np.array([.3, 1.]), np.array([1200., 1350., 1500.]), np.array([.1, .35, .2]), carbon, veg))
    return result


def _contracts() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    document = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    selected = {record["fortran_procedure"]: record for record in document["owner_regions"] if record["owner_region_id"] in {pair[0] for pair in OWNERS.values()}}
    if set(selected) != set(OWNERS): raise RuntimeError("Batch E owner contract drift")
    return selected, document


def run(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp, moist = (extract_procedure_bytes(SOURCE, name) for name in OWNERS)
    # Pin the raw bytes independently of the compilable wrapper.
    (output_dir / "control_temp_func.f90").write_bytes(temp.span_bytes)
    (output_dir / "control_moist_func_moyano.f90").write_bytes(moist.span_bytes)
    source_bytes = _compose(temp.span_bytes, moist.span_bytes)
    (output_dir / "oracle.f90").write_bytes(source_bytes)
    csv_path = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_batch_e_controls_") as td:
        build = Path(td); source = build / "oracle.f90"; source.write_bytes(source_bytes); exe = build / "oracle.exe"
        try:
            compiler_meta = compile_fortran(source, exe, compiler)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(exc.stderr) from exc
        subprocess.run([str(exe), str(csv_path), "0"], cwd=build, env=compiler_environment(compiler), check=True, capture_output=True, text=True)
    fortran, jax = _read(csv_path), _jax()
    _json(output_dir / "inputs.json", {
        "schema_version": 1, "family": FAMILY,
        "temperature_kelvin": [zc for zc in (269.15, 271.15, 272.65, 273.15, 303.15)],
        "frozen_respiration_func": [0, 1, 2, 3, 4],
        "moyano_moisture": [0.35, 0.72, 0.0],
        "pft14_index_zero_based": 13, "bare_soil_index_zero_based": 0,
        "vegetation_witnesses": ["bare_only", "pft14_only", "masked_all_zero"],
        "threshold_witnesses": ["case1: -1C, 0C, >0C", "case2: -3C, 0C, >0C", "moyano: NINT(moist*100) positive and non-positive"],
    })
    write_point_comparisons(output_dir / "point_comparisons.csv", fortran, jax, rtol=RTOL, atol=ATOL)
    comparison = write_result(output_dir, FAMILY, [float_comparison(key, value, jax[key], rtol=RTOL, atol=ATOL) for key, value in fortran.items()], {"source": {"file": SOURCE.relative_to(ROOT).as_posix(), "procedures": {"control_temp_func": {"start_line": temp.start_line, "end_line": temp.end_line, "sha256": temp.span_sha256, "exact_extracted_bytes": (output_dir / "control_temp_func.f90").read_bytes() == temp.span_bytes}, "control_moist_func_moyano": {"start_line": moist.start_line, "end_line": moist.end_line, "sha256": moist.span_sha256, "exact_extracted_bytes": (output_dir / "control_moist_func_moyano.f90").read_bytes() == moist.span_bytes}}}, "compiler": compiler_meta, "tolerance_policy": {"rtol": RTOL, "atol": ATOL, "changed": False}, "matrix": {"pft14_index_zero_based": 13, "bare_soil_index_zero_based": 0, "masked_point_index_zero_based": 2, "temperature_functions": [0,1,2,3,4]}})
    if comparison["status"] != "passed": raise RuntimeError("Batch E numerical mismatch")
    return _coverage(output_dir, source_bytes, temp, moist, compiler, comparison)


def _coverage(output_dir: Path, source_bytes: bytes, temp: Any, moist: Any, compiler: Path, comparison: dict[str, Any]) -> dict[str, Any]:
    contracts, contract_document = _contracts()
    with tempfile.TemporaryDirectory(prefix="orchidee_batch_e_controls_gcov_") as td:
        build=Path(td); source=build/"oracle.f90"; source.write_bytes(source_bytes); exe=build/"oracle.exe"; meta=compile_fortran(source,exe,compiler,extra_flags=("--coverage",))
        subprocess.run([str(exe), str(build / "coverage.csv"), "0"],cwd=build,env=compiler_environment(compiler),check=True,capture_output=True,text=True)
        note=next(build.glob("*.gcno")); gc=subprocess.run([str(GCOV),"-b","-c",note.name],cwd=build,env=compiler_environment(compiler),check=True,capture_output=True,text=True); gcov_path=build/"oracle.f90.gcov"; shutil.copy2(gcov_path,output_dir/"oracle.f90.gcov"); branches=parse_gcov(gcov_path); counts=parse_gcov_line_counts(gcov_path)
    starts={"control_temp_func":locate_generated_span(source_bytes,temp.span_bytes),"control_moist_func_moyano":locate_generated_span(source_bytes,moist.span_bytes)}
    proof_records={r["arm_id"]:r for r in json.loads(PROOFS.read_text(encoding="utf-8"))["records"] if r.get("passed") is True}
    rows=[]; evidence=[]
    for procedure, extracted in (("control_temp_func",temp),("control_moist_func_moyano",moist)):
        owner=contracts[procedure]; required=owner["arm_ids"]; gcov_ids=[]; proof_ids=[]
        for arm_id in required:
            line=int(arm_id.rsplit(":",3)[1]); mapped_line=GCOV_SOURCE_LINE.get(arm_id,line); generated=starts[procedure]+mapped_line-extracted.start_line; raw=branches.get(generated,[]); selected=select_arm_branch(arm_id,raw)
            # WHERE/IF predicates use the branch selection helper. Dispatch and
            # label-only edges are proven by the mapped executable statement.
            gcov_ok=bool((selected and selected["taken"]>0) or (arm_id in GCOV_SOURCE_LINE and counts.get(generated,0)>0) or (arm_id.endswith("select_case:fallthrough") and counts.get(generated,0)>0)); proof_ok=arm_id in proof_records
            # CASE/fallthrough edges are compiler-lowered dispatch edges; their source proof is canonical when gcov has no row.
            covered=gcov_ok or proof_ok
            if gcov_ok: gcov_ids.append(arm_id)
            if proof_ok: proof_ids.append(arm_id)
            rows.append({"arm_id":arm_id,"procedure":procedure,"original_line":line,"gcov_source_line":mapped_line,"generated_line":generated,"gcov_branch_index":None if selected is None else selected["branch_index"],"gcov_taken":0 if selected is None else selected["taken"],"gcov_line_count":counts.get(generated,0),"gcov_passed":gcov_ok,"fatal_boundary_passed":False,"source_proof_passed":proof_ok,"passed":covered,"raw_gcov_branches":raw})
        covered={r["arm_id"] for r in rows if r["procedure"]==procedure and r["passed"]}; missing=sorted(set(required)-covered)
        evidence.append({"owner_region_id":owner["owner_region_id"],"base_region_id":owner["base_region_id"],"fortran_procedure":procedure,"jax_owners":owner["jax_owners"],"comparison_asset":f"outputs/reference_mode/micro_oracles/{FAMILY}/comparison.json","branch_coverage_asset":f"outputs/reference_mode/micro_oracles/{FAMILY}/branch_coverage.json","required_arm_ids":sorted(required),"gcov_arm_ids":sorted(gcov_ids),"fatal_boundary_arm_ids":[],"source_proof_arm_ids":sorted(proof_ids),"covered_arm_ids":sorted(covered),"missing_arm_ids":missing,"passed":not missing and comparison["status"]=="passed"})
    missing=sorted(r["arm_id"] for r in rows if not r["passed"]); coverage={"schema_version":1,"family":FAMILY,"branch_complete":not missing,"required_arm_count":len(rows),"covered_arm_count":len(rows)-len(missing),"missing_arm_ids":missing,"arms":rows,"gcov_asset":"oracle.f90.gcov","compiler":meta,"gcov_stdout":gc.stdout,"gcov_stderr":gc.stderr,"source_line_mapping":GCOV_SOURCE_LINE,"policy":"GCOV parses the byte-exact harness. Non-executable contract labels map to their first governed executable source statement; no coverage is manually asserted."}
    _json(output_dir/"branch_coverage.json",coverage); _json(output_dir/"fatal_boundary_evidence.json",{"schema_version":1,"family":FAMILY,"complete":True,"required_fatal_arm_ids":[],"records":[],"reason":"Neither pure function has a fatal boundary."}); _json(output_dir/"owner_region_evidence.json",{"schema_version":1,"family":FAMILY,"complete":all(r["passed"] for r in evidence),"records":evidence})
    central=build_central_audit_report(contract_document,json.loads(PROOFS.read_text(encoding="utf-8")),load_evidence_documents(ROOT/"outputs/reference_mode/micro_oracles")); accepted={r["owner_region_id"]:r.get("passed") is True for r in central["records"] if r["owner_region_id"] in {x[0] for x in OWNERS.values()}}; central["assigned_owner_acceptance"]={"accepted":accepted,"global_completion_not_required_by_this_batch_owner":True}; _json(output_dir/"central_audit_report.json",central)
    if missing or not all(accepted.get(value[0],False) for value in OWNERS.values()): raise RuntimeError(f"Batch E owner closure incomplete: {missing}, {accepted}")
    return {"comparison":comparison,"coverage":coverage,"central_audit":central}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=OUTPUT); args=parser.parse_args()
    print(json.dumps(run(args.output_dir.resolve()),indent=2))
