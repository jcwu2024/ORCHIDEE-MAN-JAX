"""Original-byte GCOV comparison for sechiba_main lines 1264-1301."""
from __future__ import annotations
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.dev.fortran_gcov import locate_generated_span,parse_gcov,select_arm_branch  # noqa:E402
from scripts.dev.fortran_oracle_common import DEFAULT_COMPILER,compiler_environment  # noqa:E402
from jax_orchidee.sechiba.main import sechiba_main_full_irrigation_transition  # noqa:E402
SOURCE=ROOT/'fortran_source/ORCHIDEE/src_sechiba/sechiba.f90';OUT=ROOT/'outputs/reference_mode/micro_oracles/sechiba_batch_c'
ARMS=[f'fortran_source/ORCHIDEE/src_sechiba/sechiba.f90:{n}:{a}' for n,a in ((1264,'if:false'),(1268,'if:false'),(1268,'if:true'),(1270,'if:false'),(1270,'if:true'),(1273,'if:false'),(1274,'if:false'),(1292,'if:false'),(1292,'if:true'))]
def block():return b''.join(SOURCE.read_bytes().splitlines(keepends=True)[1263:1301])
def run(output=OUT):
 output.mkdir(parents=True,exist_ok=True)
 unit=b'''program oracle
 implicit none
 integer,parameter::nvm=3,kjpindex=1,nflow=2,ih2o=1
 integer::ji,jv,u,c;real(8),parameter::zero=0.d0
 logical::do_fullirr,irrig_drip,ok_LAIdev(nvm)
 real(8)::veget_max(kjpindex,nvm),veget(kjpindex,nvm),vegstress(kjpindex,nvm),irrig_threshold(nvm),irrig_fulfill(nvm),transpot(kjpindex,nvm),evapot(kjpindex),precip_rain(kjpindex),soil_deficit(kjpindex,nvm),irrig_frac(kjpindex),irrig_dosmax,temp_irrig_need,tempfrac,irrigation(kjpindex,nflow)
 character(len=256)::path
 call get_command_argument(1,path);open(newunit=u,file=trim(path),status='replace')
 do c=1,6;veget_max=reshape((/0d0,0.5d0,0.4d0/),shape(veget_max));veget=veget_max*.5d0;vegstress=reshape((/1d0,.2d0,.8d0/),shape(vegstress));ok_LAIdev=(/.false.,.true.,.true./);irrig_threshold=(/0d0,.5d0,.5d0/);irrig_fulfill=(/0d0,.8d0,1d0/);transpot=4d0;evapot=1d0;precip_rain=.5d0;soil_deficit=reshape((/0d0,.7d0,.2d0/),shape(soil_deficit));irrig_frac=.5d0;irrig_dosmax=1d0;irrigation=9d0;do_fullirr=c/=3;irrig_drip=c==1;if(c==4)veget_max=0d0;if(c==5)veget(1,2)=0d0;if(c==6)irrig_frac=-.5d0
! <BLOCK>
 write(u,'(I0,2(1X,ES24.16E3))')c,irrigation(1,1),irrigation(1,2);enddo;close(u)
end program oracle
'''.replace(b'! <BLOCK>',block().rstrip())
 with tempfile.TemporaryDirectory(prefix='orchidee_batch_c_fullirr_') as td:
  b=Path(td);src=b/'o.f90';exe=b/'o.exe';src.write_bytes(unit);subprocess.run([str(DEFAULT_COMPILER),'--coverage','-O0','-fdefault-real-8','-ffree-line-length-none',str(src),'-o',str(exe)],cwd=b,env=compiler_environment(DEFAULT_COMPILER),check=True,capture_output=True,text=True);raw=output/'sechiba_main_full_irrigation_fortran.txt';subprocess.run([str(exe),str(raw.resolve())],cwd=b,env=compiler_environment(DEFAULT_COMPILER),check=True);subprocess.run([str(DEFAULT_COMPILER.with_name('gcov.exe')),'-b','-c','o.gcno'],cwd=b,env=compiler_environment(DEFAULT_COMPILER),check=True,capture_output=True,text=True);gc=parse_gcov(b/'o.f90.gcov');start=locate_generated_span(src.read_bytes(),block())
 rows=[]
 for s in raw.read_text().splitlines():
  v=np.fromstring(s,sep=' ');c=int(v[0]);vm=np.array([[0.,.5,.4]]);ve=np.array([[0.,.25,.2]]);frac=np.array([.5]);
  if c==4:vm[:]=0
  if c==5:ve[0,1]=0
  if c==6:frac[:]=-.5
  jax=sechiba_main_full_irrigation_transition(veget_max=vm,veget=ve,vegstress=np.array([[1.,.2,.8]]),transpot=np.full((1,3),4.),evapot=np.array([1.]),precip_rain=np.array([.5]),soil_deficit=np.array([[0.,.7,.2]]),irrig_frac=frac,ok_laidev=np.array([False,True,True]),irrig_threshold=np.array([0.,.5,.5]),irrig_fulfill=np.array([0.,.8,1.]),irrig_dosmax=1.,irrig_drip=c==1,nflow=2) if c!=3 else np.zeros((1,2));rows.append({'case':c,'fortran':v[1:].tolist(),'jax':np.asarray(jax).ravel().tolist(),'passed':v[1:].tolist()==np.asarray(jax).ravel().tolist()})
 mappings=[]
 for arm in ARMS:
  line=int(arm.rsplit(':',2)[0].rsplit(':',1)[1]); generated=start+line-1264; selected=select_arm_branch(arm,gc.get(generated,[]));mappings.append({'arm_id':arm,'original_line':line,'generated_line':generated,'selected_branch':selected,'passed':bool(selected and selected['taken']>0)})
 r={'schema_version':1,'arms':ARMS,'cases':rows,'gcov_mapping':mappings,'passed_arm_ids':[x['arm_id'] for x in mappings if x['passed']],'passed':all(x['passed'] for x in rows) and all(x['passed'] for x in mappings)};(output/'sechiba_main_full_irrigation_witness.json').write_text(json.dumps(r,indent=2)+'\n',encoding='ascii');return r
if __name__=='__main__':
 r=run();print(json.dumps(r,indent=2));raise SystemExit(0 if r['passed'] else 1)
