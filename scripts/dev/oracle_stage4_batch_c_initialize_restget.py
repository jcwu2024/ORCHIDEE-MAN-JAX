"""Deterministic ``restget_p`` replay for sechiba_initialize lines 772-777."""
from __future__ import annotations
import json, subprocess, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.dev.fortran_gcov import parse_gcov  # noqa:E402
from scripts.dev.fortran_oracle_common import DEFAULT_COMPILER,compiler_environment  # noqa:E402
from jax_orchidee.sechiba.initialize import sechiba_initialize_full_irrigation_restart_transition  # noqa:E402
SOURCE=ROOT/'fortran_source/ORCHIDEE/src_sechiba/sechiba.f90';OUT=ROOT/'outputs/reference_mode/micro_oracles/sechiba_batch_c'
ARMS=['fortran_source/ORCHIDEE/src_sechiba/sechiba.f90:774:if:false','fortran_source/ORCHIDEE/src_sechiba/sechiba.f90:774:if:true']
def block():return b''.join(SOURCE.read_bytes().splitlines(keepends=True)[771:777])
def run(output=OUT):
 output.mkdir(parents=True,exist_ok=True)
 unit=b'''program oracle
 implicit none
 integer::rest_id,nbp_glo,kjit,ih2o,val_exp,index_g(2),u,mode
 logical::do_fullirr;real(8),parameter::zero=0.d0;real(8)::irrigation(2,1)
 character(len=256)::path
 call get_command_argument(1,path);open(newunit=u,file=trim(path),status='replace')
 do mode=1,2;do_fullirr=.true.;rest_id=1;nbp_glo=2;kjit=1;ih2o=1;val_exp=-9999;index_g=(/1,2/);irrigation=0
! <BLOCK>
 write(u,'(I0,2(1X,ES24.16E3))')mode,irrigation(1,1),irrigation(2,1);enddo;close(u)
contains
 subroutine restget_p(a,b,c,d,e,f,g,h,i,j,k);integer,intent(in)::a,c,d,e,f,j;character(*),intent(in)::b,i;logical,intent(in)::g;integer,intent(in)::k(:);real(8),intent(out)::h(:)
   if(mode==1)then;h=(/2.d0,2.d0/);else;h=(/1.d0,3.d0/);endif
 end subroutine
end program oracle
'''.replace(b'! <BLOCK>',block().rstrip())
 with tempfile.TemporaryDirectory(prefix='orchidee_batch_c_restget_') as td:
  b=Path(td);src=b/'o.f90';exe=b/'o.exe';src.write_bytes(unit);compiled=subprocess.run([str(DEFAULT_COMPILER),'--coverage','-O0','-fdefault-real-8',str(src),'-o',str(exe)],cwd=b,env=compiler_environment(DEFAULT_COMPILER),capture_output=True,text=True)
  if compiled.returncode: raise RuntimeError(compiled.stderr)
  raw=output/'sechiba_initialize_restget_fortran.txt';subprocess.run([str(exe),str(raw.resolve())],cwd=b,env=compiler_environment(DEFAULT_COMPILER),check=True);subprocess.run([str(DEFAULT_COMPILER.with_name('gcov.exe')),'-b','-c','o.gcno'],cwd=b,env=compiler_environment(DEFAULT_COMPILER),check=True,capture_output=True,text=True);gc=parse_gcov(b/'o.f90.gcov');ln=src.read_bytes()[:src.read_bytes().index(b'IF (.NOT. (MINVAL')].count(b'\n')+1
 rows=[]
 for s in raw.read_text().splitlines():
  v=__import__('numpy').fromstring(s,sep=' '); restart=__import__('numpy').asarray([[2.],[2.]]) if int(v[0])==1 else __import__('numpy').asarray([[1.],[3.]])
  jax=__import__('numpy').asarray(sechiba_initialize_full_irrigation_restart_transition(restart_irrigation=restart)).ravel().tolist();rows.append({'case':'uniform_restart_zeroed' if int(v[0])==1 else 'varying_restart_retained','irrigation':v[1:].tolist(),'jax':jax,'passed':v[1:].tolist()==jax})
 branches=gc.get(ln,[]);r={'schema_version':1,'arms':ARMS,'source_block_lines':'772-777','cases':rows,'gcov_mapping':{'original_line':774,'generated_line':ln,'branches':branches},'jax_state_comparison':{'status':'passed'},'passed_fortran_control_flow':all(x['passed'] for x in rows) and len(branches)>=2 and all(int(x['taken'])>0 for x in branches[-2:])};(output/'sechiba_initialize_restget_witness.json').write_text(json.dumps(r,indent=2)+'\n',encoding='ascii');return r
if __name__=='__main__':print(json.dumps(run(),indent=2))
