"""Original-byte GCOV oracle for ``sechiba_main`` output classification."""
from __future__ import annotations
import csv, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.dev.fortran_gcov import parse_gcov  # noqa: E402
from scripts.dev.fortran_oracle_common import DEFAULT_COMPILER, compiler_environment  # noqa: E402
from jax_orchidee.sechiba.main import build_sechiba_main_output_packet_pft14  # noqa: E402

SOURCE=ROOT/'fortran_source/ORCHIDEE/src_sechiba/sechiba.f90'
OUT=ROOT/'outputs/reference_mode/micro_oracles/sechiba_batch_c'
ARMS={f'fortran_source/ORCHIDEE/src_sechiba/sechiba.f90:{line}:{edge}' for line,edge in ((1425,'if:false'),(1425,'if:true'),(1427,'else_if:false'),(1427,'else_if:true'))}

def _block(): return b''.join(SOURCE.read_bytes().splitlines(keepends=True)[1420:1432])
def _unit():
    return b'''program oracle
 implicit none
 integer,parameter :: nvm=14,kjpindex=2
 integer :: jv,unit,case_id
 real(8),parameter :: zero=0.d0
 logical :: is_tree(nvm),natural(nvm)
 real(8) :: veget_max(kjpindex,nvm),sum_treefrac(kjpindex),sum_grassfrac(kjpindex),sum_cropfrac(kjpindex)
 character(len=256)::path
 call get_command_argument(1,path); open(newunit=unit,file=trim(path),status='replace')
 do case_id=1,3
   veget_max=0; is_tree=.false.; natural=.false.
   if(case_id==1) then; is_tree(2)=.true.; natural(2)=.true.; veget_max(:,2)=(/0.2d0,0.3d0/); endif
   if(case_id==2) then; natural(2)=.true.; veget_max(:,2)=(/0.4d0,0.1d0/); endif
   if(case_id==3) then; veget_max(:,2)=(/0.5d0,0.6d0/); endif
! <BLOCK>
   write(unit,'(I0,6(1X,ES24.16E3))') case_id,sum_treefrac(1),sum_treefrac(2),sum_grassfrac(1),sum_grassfrac(2),sum_cropfrac(1),sum_cropfrac(2)
 enddo
 close(unit)
end program oracle
'''.replace(b'! <BLOCK>',_block().rstrip())

def _state(veget_max):
    names='temp_sol_new fluxsens fluxlat vevapnu snow snow_age reinf_slope njsc drysoil_frac vevapflo k_litt vbeta vbeta1 vbeta2 vbeta3 vbeta4 vbeta5 rsol tsol_rad qsurf emis z0m z0h roughheight vevapsno vevapp fusion tq_cdrag soilflx soilcap grndflux pgflux totfrac_nobio frac_snow_veg'.split()
    state={name:np.zeros(2) for name in names}; state['veget_max']=veget_max; state['contfrac']=np.ones(2)
    for name in ('veget','gpp','co2_flux','vbeta_pft','vbeta2','vbeta3','vbeta4_pft','gsmean','cimean','rveget','rstruct','vevapwet','transpir','roughheight_pft','lai','tq_cdrag_pft','soilflx_pft','soilcap_pft','temp_sol_pft','vevapnu_pft') : state[name]=np.zeros((2,14))
    state['frac_nobio']=np.zeros((2,1)); state['frac_snow_nobio']=np.zeros((2,1)); state['snow_nobio']=np.zeros((2,1)); state['snow_nobio_age']=np.zeros((2,1)); state['soiltile']=np.zeros((2,1)); state['irrigation']=np.zeros((2,1))
    for name in ('snowtemp','snowliq','snowdz','snowrho','snowgrain','snowheat'): state[name]=np.zeros((2,3))
    return state

def run(output=OUT):
 output.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='orchidee_batch_c_output_') as td:
  build=Path(td); source=build/'oracle.f90'; exe=build/'oracle.exe'; source.write_bytes(_unit())
  cmd=[str(DEFAULT_COMPILER),'--coverage','-O0','-fdefault-real-8','-ffree-line-length-none',str(source),'-o',str(exe)]
  subprocess.run(cmd,cwd=build,env=compiler_environment(DEFAULT_COMPILER),check=True,capture_output=True,text=True)
  raw=output/'sechiba_main_output_dispatch_fortran.txt'; subprocess.run([str(exe),str(raw.resolve())],cwd=build,env=compiler_environment(DEFAULT_COMPILER),check=True)
  subprocess.run([str(DEFAULT_COMPILER.with_name('gcov.exe')),'-b','-c','oracle.gcno'],cwd=build,env=compiler_environment(DEFAULT_COMPILER),check=True,capture_output=True,text=True)
  gcov=parse_gcov(build/'oracle.f90.gcov'); branch_lines={line:source.read_bytes()[:source.read_bytes().index(f'IF (is_tree'.encode())].count(b'\n')+1 if line==1425 else source.read_bytes()[:source.read_bytes().index(b'ELSE IF')].count(b'\n')+1 for line in (1425,1427)}
 rows=[]
 for text in raw.read_text(encoding='ascii').splitlines():
  values=np.fromstring(text.replace('D','E'),sep=' '); case=int(values[0]); actual=values[1:]
  veg=np.zeros((2,14)); tree=np.zeros(14,bool); natural=np.zeros(14,bool)
  if case==1: tree[1]=natural[1]=True; veg[:,1]=[.2,.3]
  elif case==2: natural[1]=True; veg[:,1]=[.4,.1]
  else: veg[:,1]=[.5,.6]
  packet=build_sechiba_main_output_packet_pft14(state=_state(veg),dt_sechiba=1800.,is_tree=tree,natural=natural)
  fields={x.name:np.asarray(x.value) for x in packet.xios}; expected=np.r_[fields['treeFrac']/100,fields['grassFrac']/100,fields['cropFrac']/100]
  rows.append({'case':case,'passed':bool(np.array_equal(actual,expected)),'fortran':actual.tolist(),'jax':expected.tolist()})
 branches={str(line):{'original_line':line,'generated_line':generated,'branches':gcov.get(generated,[])} for line,generated in branch_lines.items()}
 passed=all(x['passed'] for x in rows) and all(len(x['branches'])>=2 and all(int(b['taken'])>0 for b in x['branches'][:2]) for x in branches.values())
 result={'schema_version':1,'arms':sorted(ARMS),'source_block_sha256':__import__('hashlib').sha256(_block()).hexdigest(),'cases':rows,'gcov_mapping':branches,'passed':passed}
 (output/'sechiba_main_output_dispatch_witness.json').write_text(json.dumps(result,indent=2)+'\n',encoding='ascii'); return result
if __name__=='__main__':
 r=run(); print(json.dumps(r,indent=2)); raise SystemExit(0 if r['passed'] else 1)
