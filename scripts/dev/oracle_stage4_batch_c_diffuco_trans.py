from __future__ import annotations
import csv, json, re, shutil, subprocess, tempfile
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes
from scripts.dev.fortran_oracle_common import DEFAULT_COMPILER, compile_fortran, compiler_environment, float_comparison, write_point_comparisons, write_result

SOURCE=ROOT/'fortran_source/ORCHIDEE/src_sechiba/diffuco.f90'; QSAT=ROOT/'fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90'; TEMPLATE=ROOT/'scripts/dev/diffuco_batch_c_trans.f90.template'; FAMILY='diffuco_batch_c_trans'
def _read(path):
 d={}
 with path.open(newline='',encoding='ascii') as f:
  for r in csv.DictReader(f): d.setdefault(r['field'],[]).append(float(r['value']))
 return {k:np.asarray(v) for k,v in d.items()}
def _jax():
 from jax_orchidee.sechiba.diffuco import diffuco_trans_source_routed
 n,nvm=3,14; hum=np.full((n,nvm),.7); veg=np.zeros((n,nvm)); vmax=np.zeros((n,nvm)); lai=np.zeros((n,nvm)); qsv=np.zeros((n,nvm)); qsm=np.full((n,nvm),.3); rs=np.full((n,nvm),50.); vb23=np.full((n,nvm),.2)
 veg[:,13]=[.5,.6,.7]; vmax[:,13]=.8; lai[:,13]=[2,2,2]; qsv[:,13]=[.15,.1,.5]; qsm[2,13]=0
 r=diffuco_trans_source_routed(swnet=np.array([200.,0.,150.]),temp_air=np.array([290.,285.,295.]),pb=np.full(n,1000.),qair=np.array([.004,.005,.2]),rau=np.array([1.2,1.1,1.3]),u=np.array([.02,2.,4.]),v=np.zeros(n),q_cdrag=np.array([.01,.02,.03]),humrel=hum,veget=veg,veget_max=vmax,lai=lai,qsintveg=qsv,qsintmax=qsm,rstruct=rs,vbeta23=vb23,kzero=np.full(nvm,.01),rveg_pft=np.full(nvm,1.1))
 return {'vbeta3':np.asarray(r.vbeta3).ravel(),'vbeta3pot':np.asarray(r.vbeta3pot).ravel(),'rveget':np.asarray(r.rveget).ravel(),'cimean':np.asarray(r.cimean).ravel(),'vbetaco2':np.asarray(r.vbetaco2).ravel()}
def run_oracle(output_dir):
 output_dir.mkdir(parents=True,exist_ok=True); q=b'\n\n'.join(extract_procedure_bytes(QSAT,n).span_bytes for n in ('qsfrict_init','qsatcalc')); t=extract_procedure_bytes(SOURCE,'diffuco_trans')
 with tempfile.TemporaryDirectory(prefix='orchidee_diffuco_trans_') as d:
  p=Path(d); src=p/'oracle.f90'; src.write_bytes(TEMPLATE.read_bytes().replace(b'! <QSAT>',q).replace(b'! <TRANS>',t.span_bytes)); meta=compile_fortran(src,p/'oracle.exe',DEFAULT_COMPILER,extra_flags=('-fprofile-arcs','-ftest-coverage')); out=output_dir/'fortran_outputs.csv'; subprocess.run([str(p/'oracle.exe'),str(out.resolve())],cwd=p,env=compiler_environment(DEFAULT_COMPILER),check=True,capture_output=True,text=True)
  gcov=subprocess.run([str(DEFAULT_COMPILER.parent/'gcov.exe'),'-b',str(src)],cwd=p,text=True,capture_output=True,check=True)
  shutil.copy2(p/'oracle.f90.gcov',output_dir/'oracle.f90.gcov')
 gcov_text=(output_dir/'oracle.f90.gcov').read_text(encoding='utf-8')
 conditions={'1846':'IF (qsintmax(ji,jv) .GT. min_sechiba) THEN','1854':'IF ( ( veget(ji,jv)*lai(ji,jv) .GT. min_sechiba ) .AND. &'}
 executed=[]
 for source_line,condition in conditions.items():
  at=gcov_text.index(condition); nearby=gcov_text[at:at+1200]
  taken=[int(value) for value in re.findall(r'branch\s+\d+\s+taken\s+(\d+)%',nearby)]
  if not any(value>0 for value in taken) or not any(value==0 for value in taken):
   raise RuntimeError(f'gcov does not show both outcomes for original diffuco_trans line {source_line}')
  executed.extend((f'{source_line}:if:true',f'{source_line}:if:false'))
 f=_read(out); j=_jax(); write_point_comparisons(output_dir/'point_comparisons.csv',f,j,rtol=1e-12,atol=1e-14); result=write_result(output_dir,FAMILY,[float_comparison(k,f[k],j[k],rtol=1e-12,atol=1e-14) for k in f],{'source_spans':{'diffuco_trans':t.span_sha256},'build':meta,'gcov_artifact':'oracle.f90.gcov','gcov_returncode':gcov.returncode,'gcov_stdout':gcov.stdout,'gcov_stderr':gcov.stderr,'measured_source_arm_ids':[f'fortran_source/ORCHIDEE/src_sechiba/diffuco.f90:{key}' for key in executed]}); return result
if __name__=='__main__':
 r=run_oracle(ROOT/'outputs/reference_mode/micro_oracles'/FAMILY); print(json.dumps(r,indent=2)); raise SystemExit(0 if r['status']=='passed' else 1)
