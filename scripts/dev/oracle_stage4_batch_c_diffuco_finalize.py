from __future__ import annotations
import csv,json,subprocess,tempfile,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes
from scripts.dev.fortran_oracle_common import DEFAULT_COMPILER,compile_fortran,compiler_environment,exact_comparison,write_result
SOURCE=ROOT/'fortran_source/ORCHIDEE/src_sechiba/diffuco.f90';TEMPLATE=ROOT/'scripts/dev/diffuco_batch_c_finalize.f90.template';FAMILY='diffuco_batch_c_finalize'
def run_oracle(output_dir):
 output_dir.mkdir(parents=True,exist_ok=True);span=extract_procedure_bytes(SOURCE,'diffuco_finalize')
 with tempfile.TemporaryDirectory(prefix='orchidee_diffuco_finalize_') as d:
  p=Path(d);src=p/'oracle.f90';src.write_bytes(TEMPLATE.read_bytes().replace(b'! <FINALIZE>',span.span_bytes));meta=compile_fortran(src,p/'oracle.exe',DEFAULT_COMPILER,extra_flags=('-fprofile-arcs','-ftest-coverage'));out=output_dir/'fortran_outputs.csv';subprocess.run([str(p/'oracle.exe'),str(out.resolve())],cwd=p,env=compiler_environment(DEFAULT_COMPILER),check=True,capture_output=True,text=True);gc=subprocess.run([str(DEFAULT_COMPILER.parent/'gcov.exe'),'-b',str(src)],cwd=p,text=True,capture_output=True,check=True);(output_dir/'oracle.f90.gcov').write_text((p/'oracle.f90.gcov').read_text(encoding='utf-8'),encoding='utf-8')
 values={};
 with out.open(newline='',encoding='ascii') as f:
  for row in csv.DictReader(f):values.setdefault(row['field'],[]).append(float(row['value']))
 from jax_orchidee.sechiba.diffuco import diffuco_finalize_restart_packet
 r=np.fromfunction(lambda i,j:10*(i+1)+(j+1),(2,14));q=np.fromfunction(lambda i,j:.01*(i+1)+.001*(j+1),(2,14));leaf=np.empty((2,14,2));
 for i in range(2):
  for j in range(14):leaf[i,j,:]=100*(i+1)+(j+1)
 packet=diffuco_finalize_restart_packet(rstruct=r,q_cdrag_pft=q,ok_co2=True,leaf_ci=leaf); expected={'rstruct':r.ravel(),'cdrag_pft':q.ravel()};expected.update({f'leaf_ci_l{k+1}':leaf[:,:,k].ravel() for k in range(2)})
 result=write_result(output_dir,FAMILY,[exact_comparison(k,np.asarray(values[k]),expected[k]) for k in expected],{'source_span_sha256':span.span_sha256,'build':meta,'gcov_artifact':'oracle.f90.gcov','gcov_stdout':gc.stdout,'measured_source_arm_ids':['fortran_source/ORCHIDEE/src_sechiba/diffuco.f90:792:if:true']});return result
if __name__=='__main__':
 r=run_oracle(ROOT/'outputs/reference_mode/micro_oracles'/FAMILY);print(json.dumps(r,indent=2));raise SystemExit(0 if r['status']=='passed' else 1)
