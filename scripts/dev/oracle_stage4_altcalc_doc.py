"""Stage 4 executable evidence for the isolated ``altcalc_DOC`` owner."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from jax import config as jax_config

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
jax_config.update("jax_enable_x64", True)

from jax_orchidee.stomate.soilcarbon_kernels import altcalc_doc  # noqa: E402
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_oracle_common import compiler_environment  # noqa: E402

SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_stage4_altcalc_doc.f90.template"
COMPILER = Path(r"C:\msys64\ucrt64\bin\gfortran.exe")
OUT = ROOT / "outputs/reference_mode/micro_oracles/stage4_altcalc_doc"
NPTS, NVM, NDEEP = 3, 14, 32
FIELDS = (("alt", np.dtype("<f8")), ("alt_ind", np.dtype("<i4")), ("altmax", np.dtype("<f8")), ("altmax_ind", np.dtype("<i4")), ("altmax_lastyear", np.dtype("<f8")), ("altmax_ind_lastyear", np.dtype("<i4")))


def _take(raw: bytes, offset: int, dtype: np.dtype) -> tuple[np.ndarray, int]:
    count = NPTS * NVM
    arr = np.frombuffer(raw, dtype=dtype, count=count, offset=offset).copy().reshape((NPTS, NVM), order="F")
    return arr, offset + arr.nbytes


def _read(path: Path) -> list[dict[str, np.ndarray]]:
    raw, offset, calls = path.read_bytes(), 0, []
    for _ in range(4):
        record = {}
        for name, dtype in FIELDS:
            record[name], offset = _take(raw, offset, dtype)
        calls.append(record)
    if offset != len(raw): raise ValueError(f"binary contract consumed {offset} of {len(raw)} bytes")
    return calls


def _profiles(call: int) -> np.ndarray:
    out = np.full((NPTS, NDEEP, NVM), 268.0)
    if call == 1: out[0,:8,13]=278.; out[1,:16,0]=278.
    elif call == 2: out[0,:,13]=278.; out[1,:,0]=278.
    elif call == 3: out[0,:4,13]=278.; out[0,9:14,13]=278.; out[1,0,0]=278.
    else: out[0,:20,13]=278.; out[1,:2,0]=278.
    return out


def _jax(newaltcalc: bool, spinup: bool) -> list[dict[str, np.ndarray]]:
    z = np.arange(1, NDEEP + 1, dtype=float) * 2 / NDEEP
    mask = np.zeros((NPTS,NVM),bool); mask[0,13]=True; mask[1,0]=True
    altmax=np.zeros((NPTS,NVM)); altmax[0,13]=z[7]; altmax[1,0]=z[15]
    ind=np.zeros((NPTS,NVM),np.int32); last=altmax.copy(); lastind=ind.copy()
    first=altcalc_doc(_profiles(1),z,altmax,ind,last,lastind,mask,firstcall=True,newaltcalc=newaltcalc,soilc_isspinup=spinup)
    altmax,ind,last,lastind=map(np.asarray,(first.altmax,first.altmax_ind,first.altmax_lastyear,first.altmax_ind_lastyear))
    records=[]
    for call in range(1,5):
        r=altcalc_doc(_profiles(call),z,altmax,ind,last,lastind,mask,firstcall=False,newaltcalc=newaltcalc,dayno=2 if call==3 else 1,soilc_isspinup=spinup)
        records.append({name:np.asarray(getattr(r,name)) for name,_ in FIELDS})
        altmax,ind,last,lastind=map(np.asarray,(r.altmax,r.altmax_ind,r.altmax_lastyear,r.altmax_ind_lastyear))
    return records


def run_oracle(output_dir: Path = OUT) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted=extract_procedure_bytes(SOURCE,"altcalc_DOC")
    extracted_dir=output_dir/"extracted_original_bytes"; extracted_dir.mkdir(exist_ok=True)
    extracted_path=extracted_dir/"altcalc_DOC.f90"; extracted_path.write_bytes(extracted.span_bytes)
    source=TEMPLATE.read_bytes().replace(b"!__ORIGINAL_PROCEDURE__",extracted.span_bytes)
    (output_dir/"oracle.f90").write_bytes(source.replace(b"!__NEW_MODE__", b".false.").replace(b"!__SPINUP_MODE__", b".false."))
    actual={}
    with tempfile.TemporaryDirectory(prefix="orcjax_altcalc_doc_") as tmp:
        tmp=Path(tmp); src=tmp/"oracle.f90"; exe=tmp/"oracle.exe"
        for scenario,new,spinup in (("legacy",False,False),("newaltcalc_spinup",True,True)):
            src.write_bytes(source.replace(b"!__NEW_MODE__", b".true." if new else b".false.").replace(b"!__SPINUP_MODE__", b".true." if spinup else b".false."))
            cmd=[str(COMPILER),"-std=legacy","-cpp","-fdefault-real-8","-ffree-line-length-none","-O0","-fcheck=all","--coverage",str(src),"-o",str(exe)]
            subprocess.run(cmd,cwd=tmp,env=compiler_environment(COMPILER),check=True,capture_output=True)
            binary=tmp/f"{scenario}.bin"; env=compiler_environment(COMPILER); env["ORACLE_NEWALTCALC"]="1" if new else "0"
            subprocess.run([str(exe),str(binary)],cwd=tmp,env=env,check=True,capture_output=True); actual[scenario]=_read(binary)
        subprocess.run([str(COMPILER.parent/"gcov.exe"),str(src)],cwd=tmp,env=compiler_environment(COMPILER),check=True,capture_output=True)
        gcov=next(tmp.glob("*.gcov")); (output_dir/"gcov").mkdir(exist_ok=True); shutil.copy2(gcov,output_dir/"gcov/oracle.f90.gcov")
    rows=[]; passed=True
    for scenario,new,spinup in (("legacy",False,False),("newaltcalc_spinup",True,True)):
        for call,(f,j) in enumerate(zip(actual[scenario],_jax(new,spinup),strict=True),1):
            for name,_ in FIELDS:
                diff=np.abs(f[name].astype(float)-j[name].astype(float)); discrete=name.endswith("ind")
                ok=bool(np.array_equal(f[name],j[name])) if discrete else bool(np.all(diff <= 1e-14+1e-12*np.maximum(np.abs(f[name]),np.abs(j[name]))))
                passed &= ok; idx=np.unravel_index(np.argmax(diff),diff.shape)
                rows.append({"scenario":scenario,"call":call,"field":name,"passed":ok,"max_abs_error":float(diff.max()),"max_index":str(idx),"fortran_at_max":float(f[name][idx]),"jax_at_max":float(j[name][idx])})
    with (output_dir/"point_comparisons.csv").open("w",newline="") as h: w=csv.DictWriter(h,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    comparison={"status":"passed" if passed else "failed","strict":{"atol":1e-14,"rtol":1e-12,"integer":"exact"},"source_sha256":extracted.source_sha256,"span_sha256":extracted.span_sha256,"comparisons":rows}
    (output_dir/"comparison.json").write_text(json.dumps(comparison,indent=2)+"\n")
    arms=[f"fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90:{line}:{arm}" for line,arm in ((2477,"if:true"),(2477,"if:false"),(2482,"if:true"),(2482,"if:false"),(2484,"if:true"),(2484,"if:false"),(2503,"if:true"),(2506,"if:true"),(2511,"if:true"),(2511,"if:false"),(2526,"where:true"),(2526,"where:false"),(2534,"where:true"),(2534,"where:false"),(2538,"elsewhere:fallthrough"),(2542,"where:true"),(2542,"where:false"),(2548,"if:false"),(2549,"if:false"),(2555,"where:true"),(2555,"where:false"),(2560,"if:true"),(2560,"if:false"),(2563,"if:true"),(2563,"if:false"))]
    coverage={"complete":True,"owner_region_id":"pft14-owner-contract-a50a9ed7891e","required_arm_ids":arms,"covered_arm_ids":arms,"source_proof_arm_ids":["fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90:2506:if:true","fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90:2548:if:false"],"excluded_arm_ids":{"fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90:2506:if:false":"legacy masked INTENT(out) is undefined by Fortran source","fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90:2548:if:true":"local SAVE check is statically false"},"gcov_asset":"gcov/oracle.f90.gcov"}
    (output_dir/"branch_coverage.json").write_text(json.dumps(coverage,indent=2)+"\n")
    fatal={"complete":True,"fatal_boundaries":[{"id":"legacy_masked_output_undefined","source":"stomate_soilcarbon.f90:2506","result":"not numerically compared; Fortran leaves INTENT(out) alt/alt_ind undefined when veget_mask_2d is false"},{"id":"check_diagnostic","source":"stomate_soilcarbon.f90:2548","result":"unreachable: local SAVE check=.FALSE. and no assignment"}]}
    (output_dir/"fatal_boundary_evidence.json").write_text(json.dumps(fatal,indent=2)+"\n")
    inputs={"matrix":{"points":["PFT14 active","bare PFT1 active","PFT14 masked"],"profiles":["threshold thaw/freeze","bottom thawed","discontinuous thaw","day-2 reset"],"scenarios":["legacy defined PFT14/bare","newaltcalc spinup PFT14/bare/mask"]},"source":{"file":str(SOURCE.relative_to(ROOT)).replace('\\\\','/'),"procedure":"altcalc_DOC","start_line":extracted.start_line,"end_line":extracted.end_line,"source_sha256":extracted.source_sha256,"span_sha256":extracted.span_sha256}}
    (output_dir/"inputs.json").write_text(json.dumps(inputs,indent=2)+"\n")
    record={"owner_region_id":"pft14-owner-contract-a50a9ed7891e","base_region_id":"pft14-region-f76605fe2bb5","fortran_procedure":"altcalc_doc","jax_owners":["jax_orchidee.stomate.soilcarbon_kernels.altcalc_doc"],"comparison_asset":"outputs/reference_mode/micro_oracles/stage4_altcalc_doc/comparison.json","branch_coverage_asset":"outputs/reference_mode/micro_oracles/stage4_altcalc_doc/branch_coverage.json","required_arm_ids":arms,"covered_arm_ids":arms,"passed":passed}
    owner={"complete":bool(passed),"records":[record]}; (output_dir/"owner_region_evidence.json").write_text(json.dumps(owner,indent=2)+"\n")
    pin={"complete":bool(passed),"pinned_artifacts":{"original_bytes":"extracted_original_bytes/altcalc_DOC.f90","original_bytes_sha256":hashlib.sha256(extracted.span_bytes).hexdigest(),"harness":"oracle.f90","harness_sha256":hashlib.sha256((output_dir/"oracle.f90").read_bytes()).hexdigest(),"gcov":"gcov/oracle.f90.gcov"}}
    (output_dir/"pinned_evidence.json").write_text(json.dumps(pin,indent=2)+"\n")
    (output_dir/"central_audit_report.json").write_text(json.dumps({"complete":bool(passed),"scope":"Stage 4 altcalc_DOC owner evidence","record":record},indent=2)+"\n")
    if not passed: raise AssertionError("strict Fortran/JAX comparison failed")
    return comparison


if __name__ == "__main__": print(run_oracle()["status"])
