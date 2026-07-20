"""Original-byte GCOV oracle for ``sechiba_main`` lines 1494-1501."""
from __future__ import annotations
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.dev.fortran_gcov import parse_gcov  # noqa:E402
from scripts.dev.fortran_oracle_common import DEFAULT_COMPILER,compiler_environment  # noqa:E402
from scripts.dev.oracle_stage4_batch_c_main_output_dispatch import _state  # noqa:E402
from jax_orchidee.sechiba.main import build_sechiba_main_output_packet_pft14  # noqa:E402
SOURCE=ROOT/'fortran_source/ORCHIDEE/src_sechiba/sechiba.f90'; OUT=ROOT/'outputs/reference_mode/micro_oracles/sechiba_batch_c'
ARMS=['fortran_source/ORCHIDEE/src_sechiba/sechiba.f90:1496:if:false','fortran_source/ORCHIDEE/src_sechiba/sechiba.f90:1496:if:true']
def block(): return b''.join(SOURCE.read_bytes().splitlines(keepends=True)[1493:1501])
def run(output=OUT):
 output.mkdir(parents=True,exist_ok=True)
 unit=b'''program oracle
 implicit none
 integer,parameter::nvm=14,kjpindex=2
 integer::ji,jv,u,c; real(8),parameter::zero=0.d0
 real(8)::veget_max(kjpindex,nvm),lai(kjpindex,nvm),histvar(kjpindex)
 character(len=256)::path
 call get_command_argument(1,path);open(newunit=u,file=trim(path),status='replace')
 do c=1,2; veget_max=0; lai=0; if(c==1)then;veget_max(:,2)=(/.2d0,.4d0/);lai(:,2)=(/3d0,5d0/);endif
! <BLOCK>
 write(u,'(I0,2(1X,ES24.16E3))')c,histvar(1),histvar(2);enddo;close(u)
end program oracle
'''.replace(b'! <BLOCK>',block().rstrip())
 with tempfile.TemporaryDirectory(prefix='orchidee_batch_c_lai_') as td:
  b=Path(td);src=b/'o.f90';exe=b/'o.exe';src.write_bytes(unit);subprocess.run([str(DEFAULT_COMPILER),'--coverage','-O0','-fdefault-real-8',str(src),'-o',str(exe)],cwd=b,env=compiler_environment(DEFAULT_COMPILER),check=True,capture_output=True,text=True)
  raw=output/'sechiba_main_lai_mean_fortran.txt';subprocess.run([str(exe),str(raw.resolve())],cwd=b,env=compiler_environment(DEFAULT_COMPILER),check=True);subprocess.run([str(DEFAULT_COMPILER.with_name('gcov.exe')),'-b','-c','o.gcno'],cwd=b,env=compiler_environment(DEFAULT_COMPILER),check=True,capture_output=True,text=True); gc=parse_gcov(b/'o.f90.gcov'); ln=src.read_bytes()[:src.read_bytes().index(b'IF (SUM(veget_max')].count(b'\n')+1
 cases=[]
 for text in raw.read_text().splitlines():
  v=np.fromstring(text,sep=' '); veg=np.zeros((2,14)); lai=np.zeros((2,14));
  if int(v[0])==1:veg[:,1]=[.2,.4];lai[:,1]=[3,5]
  st=_state(veg);st['lai']=lai;p=build_sechiba_main_output_packet_pft14(state=st,dt_sechiba=1800.,is_tree=np.zeros(14,bool),natural=np.zeros(14,bool)); got=next(np.asarray(x.value) for x in p.xios if x.name=='LAImean');cases.append({'case':int(v[0]),'passed':bool(np.array_equal(v[1:],got)),'fortran':v[1:].tolist(),'jax':got.tolist()})
 branches=gc.get(ln,[]);r={'schema_version':1,'arms':ARMS,'cases':cases,'gcov_mapping':{'original_line':1496,'generated_line':ln,'branches':branches},'passed':all(x['passed'] for x in cases) and len(branches)>=2 and all(int(x['taken'])>0 for x in branches[:2])};(output/'sechiba_main_lai_mean_witness.json').write_text(json.dumps(r,indent=2)+'\n',encoding='ascii');return r
if __name__=='__main__':
 r=run();print(json.dumps(r,indent=2));raise SystemExit(0 if r['passed'] else 1)
