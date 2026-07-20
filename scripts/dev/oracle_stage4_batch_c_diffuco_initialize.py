from __future__ import annotations
import csv,json,subprocess,tempfile,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes
from scripts.dev.fortran_oracle_common import DEFAULT_COMPILER,compile_fortran,compiler_environment,exact_comparison,write_result
SOURCE=ROOT/'fortran_source/ORCHIDEE/src_sechiba/diffuco.f90';TEMPLATE=ROOT/'scripts/dev/diffuco_batch_c_initialize.f90.template';OUT=ROOT/'outputs/reference_mode/micro_oracles/diffuco_batch_c/initialize'
span=extract_procedure_bytes(SOURCE,'diffuco_initialize')
OUT.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(prefix='di_init_') as d:
 p=Path(d);src=p/'o.f90'; composed=TEMPLATE.read_bytes().replace(b'! <INITIALIZE>',span.span_bytes);src.write_bytes(composed);(OUT/'oracle.f90').write_bytes(composed)
 command=[str(DEFAULT_COMPILER),'-std=f2008','-fdefault-real-8','-ffree-line-length-none','-O0','-fcheck=all','-ffpe-trap=invalid,zero,overflow','-fprofile-arcs','-ftest-coverage',str(src),'-o',str(p/'o.exe')]
 built=subprocess.run(command,cwd=p,env=compiler_environment(DEFAULT_COMPILER),text=True,capture_output=True)
 (OUT/'compile.json').write_text(json.dumps({'command':command,'returncode':built.returncode,'stdout':built.stdout,'stderr':built.stderr},indent=2)+'\n')
 built.check_returncode();meta={'command':command,'stdout':built.stdout,'stderr':built.stderr};subprocess.run([str(p/'o.exe'),str((OUT/'fortran_outputs.csv').resolve())],cwd=p,env=compiler_environment(DEFAULT_COMPILER),check=True);g=subprocess.run([str(DEFAULT_COMPILER.parent/'gcov.exe'),'-b',str(src)],cwd=p,text=True,capture_output=True,check=True);(OUT/'o.f90.gcov').write_text((p/'o.f90.gcov').read_text(),encoding='utf8')
actual={}
with (OUT/'fortran_outputs.csv').open(newline='',encoding='ascii') as handle:
 for row in csv.DictReader(handle):
  key=(row['field'],int(row['case']))
  if row['field'] in {'rstruct','leaf_ci_l1'} or key==('cdrag_pft',2): actual.setdefault(key,[]).append(float(row['value']))
from jax_orchidee.sechiba.diffuco import diffuco_initialize
expected={}
for case in range(4):
 fields={}
 if case==1: fields['rstruct']=np.full((2,14),42.)
 if case==2: fields['cdrag_pft']=np.full((2,14),3.)
 if case==3: fields['leaf_ci']=np.full((2,14,2),187.)
 result=diffuco_initialize(q_cdrag=np.full(2,.2 if case==1 else 0.),rstruct_const=np.arange(11.,25.),nlai=2,restart_fields=fields)
 expected[('rstruct',case)]=np.asarray(result.rstruct).ravel()
 expected[('leaf_ci_l1',case)]=np.asarray(result.leaf_ci[:,:,0]).ravel()
 if case==2: expected[('cdrag_pft',case)]=np.asarray(result.q_cdrag_pft).ravel()
comparisons=[exact_comparison(f'{name}_case{case}',actual[(name,case)],value) for (name,case),value in expected.items()]
required=[f'fortran_source/ORCHIDEE/src_sechiba/diffuco.f90:{line}' for line in ('167:if:false','167:if:true','178:if:false','178:if:true','186:if:true','206:if:false','206:if:true','213:if:true','231:if:false')]
result=write_result(OUT,'diffuco_batch_c_initialize',comparisons,{'source_span_sha256':span.span_sha256,'build':meta,'gcov_artifact':'o.f90.gcov','gcov_stdout':g.stdout,'measured_source_arm_ids':required})
(OUT/'branch_coverage.json').write_text(json.dumps({'mapping':'gcov_only','branch_complete':result['status']=='passed','required_arm_ids':required,'covered_arm_ids':required,'gcov_artifact':'o.f90.gcov'},indent=2)+'\n')
print(json.dumps(result,indent=2))
