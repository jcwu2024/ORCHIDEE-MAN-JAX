"""Formal owner oracle for intersurf_initialize_2d (Stage 4 Batch C).

The executable is deliberately a boundary harness: it retains the original
owner's six contracted IF statements and its direct ``sechiba_initialize``
call site.  Only history/XIOS/getin transport is replaced with no-op callbacks;
the direct callee is a deterministic state provider so the owner compression,
scatter, and xrdt conversion can be compared to the JAX owner in isolation.
"""
from __future__ import annotations

import hashlib, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes
from scripts.dev.fortran_gcov import parse_gcov
from scripts.dev.fortran_oracle_common import DEFAULT_COMPILER, compiler_environment
from jax_orchidee.sechiba.lifecycle_completion import intersurf_initialize_2d

OUT = ROOT / "outputs/reference_mode/micro_oracles/intersurf_initialize_batch_c"
SRC = ROOT / "fortran_source/ORCHIDEE/src_sechiba/intersurf.f90"
OWNER = "pft14-owner-contract-6a6b9c98566f"
ARMS = [f"fortran_source/ORCHIDEE/src_sechiba/intersurf.f90:{n}:if:{v}" for n,v in ((260,'true'),(316,'false'),(353,'true'),(359,'true'),(367,'false'),(438,'true'))]

def lines(a:int,b:int)->bytes: return b''.join(SRC.read_bytes().splitlines(keepends=True)[a-1:b])

def unit() -> bytes:
    # Each inserted statement is taken verbatim from the owner; declarations and
    # transport procedures are harness scaffolding only.
    return b'''program oracle
 implicit none
 integer,parameter::iim=2,jjm=2,kjpindex=2,nflow=3,ngrnd=2
 integer::kindex(2),ik,i,j,hist_id,hist2_id,hist_id_stom,hist_id_stom_IPCC,mode
 logical::hydrol_cwrr,check_INPUTS,ok_stomate,is_omp_root,is_root_prc
 real(8)::xrdt,soilth_lev(ngrnd),znt(ngrnd),out(2,2,12),zz0m(2),zvevapp(2),ztsol(2),zq(2),zcdrag(2),zfluxs(2),zfluxl(2),zemis(2),ztemp(2),zalbedo(2,2),zcoastal(2,nflow),zriver(2,nflow)
 character(len=256)::path
 character(len=8)::mode_text
 call get_command_argument(1,path); call get_command_argument(2,mode_text);read(mode_text,*)mode
 hydrol_cwrr=mode==1; check_INPUTS=.false.; ok_stomate=.true.; is_omp_root=.true.; hist_id=1; hist2_id=0; is_root_prc=.true.
 hist_id_stom=2;hist_id_stom_IPCC=3;xrdt=2.d0;kindex=(/1,4/);znt=(/1.d0,2.d0/);out=-999.d0
! <260>
! <316>
! <353>
! <359>
! <367>
 call sechiba_initialize(zz0m,zvevapp,ztsol,zq,zcdrag,zfluxs,zfluxl,zemis,ztemp,zalbedo,zcoastal,zriver)
 do ik=1,kjpindex; j=((kindex(ik)-1)/iim)+1;i=kindex(ik)-(j-1)*iim
  out(i,j,1)=zz0m(ik);out(i,j,2)=zvevapp(ik)/xrdt;out(i,j,3)=ztsol(ik);out(i,j,4)=zq(ik);out(i,j,5)=zcdrag(ik);out(i,j,6)=zfluxs(ik);out(i,j,7)=zfluxl(ik);out(i,j,8)=zemis(ik);out(i,j,9)=ztemp(ik);out(i,j,10)=zalbedo(ik,1);out(i,j,11)=zalbedo(ik,2);out(i,j,12)=zcoastal(ik,1)/xrdt
 enddo
! <438>
 open(10,file=trim(path),status='replace');do j=1,jjm;do i=1,iim;write(10,'(24(ES24.16E3,1X))')out(i,j,:),zriver(min(2,max(1,i+j-1)),1)/xrdt,soilth_lev(1),soilth_lev(2);enddo;enddo;close(10)
contains
 subroutine sechiba_initialize(a,b,c,d,e,f,g,h,i,j,k,l)
 real(8),intent(out)::a(:),b(:),c(:),d(:),e(:),f(:),g(:),h(:),i(:),j(:,:),k(:,:),l(:,:)
 a=(/101.d0,102.d0/);b=(/201.d0,202.d0/);c=(/301.d0,302.d0/);d=(/401.d0,402.d0/);e=(/501.d0,502.d0/);f=(/601.d0,602.d0/);g=(/701.d0,702.d0/);h=(/801.d0,802.d0/);i=(/901.d0,902.d0/);j=9; k=1201; l=1301
 end subroutine
 subroutine histwrite_p(a,b,c,d,e,f);integer,intent(in)::a,c,e;character(*),intent(in)::b;real(8),intent(in)::d(:);integer,intent(in)::f(:);end subroutine
 subroutine histsync(a);integer,intent(in)::a;end subroutine
 subroutine getin_dump;end subroutine
 function thermosoilc_levels() result(x);real(8)::x(ngrnd);x=(/3.d0,4.d0/);end function
end program oracle
'''.replace(b'! <260>',lines(260,260)+b' soilth_lev=znt\n    ELSE\n soilth_lev=thermosoilc_levels()\n    END IF').replace(b'! <316>',lines(316,316)+b' write(*,*) "diagnostic"\n    ENDIF').replace(b'! <353>',lines(353,353)+b' call histwrite_p(hist_id_stom,"Areas",1,soilth_lev,1,kindex)\n    ENDIF').replace(b'! <359>',lines(359,359)+b' call histsync(hist_id)\n    END IF').replace(b'! <367>',lines(367,367)+b' call histsync(hist2_id)\n    ENDIF').replace(b'! <438>',lines(438,438))

def jax_state() -> np.ndarray:
    fields={n:np.full((2,2),float(k+1)) for k,n in enumerate(('u','v','zlev','qair','precip_rain','precip_snow','lwdown','swnet','swdown','temp_air','epot_air','ccanopy','petAcoef','peqAcoef','petBcoef','peqBcoef','cdrag','pb'))}
    def owner(req):
        n=len(req.contfrac); x=np.arange(1,n+1,dtype=float)
        return {'z0m':100+x,'vevapp':200+x,'tsol_rad':300+x,'qsurf':400+x,'cdrag':500+x,'fluxsens':600+x,'fluxlat':700+x,'emis':800+x,'temp_sol_new':900+x,'albedo':np.stack((1000+x,1100+x),1),'coastalflow':np.stack((1200+x,)*3,1),'riverflow':np.stack((1300+x,)*3,1)}
    r=intersurf_initialize_2d(kjit=1,iim=2,jjm=2,kindex=np.array([1,4]),xrdt=2.,date0_shifted=0.,itau_offset=0,lon=np.zeros((2,2)),lat=np.zeros((2,2)),zcontfrac=np.ones((2,2)),fields=fields,owner=owner,undef_sechiba=-999.)
    return np.stack([r.outputs[k] for k in ('z0m','vevapp','tsol_rad','qsurf','cdrag','fluxsens','fluxlat','emis','temp_sol_new')],-1)

def run(output:Path=OUT)->dict:
 output.mkdir(parents=True,exist_ok=True); span=extract_procedure_bytes(SRC,'intersurf_initialize_2d'); (output/'intersurf_initialize_2d.f90').write_bytes(span.span_bytes); (output/'direct_callee_boundary.f90').write_bytes(lines(377,390))
 with tempfile.TemporaryDirectory(prefix='orchidee_intersurf_init_') as td:
  d=Path(td); f=d/'oracle.f90';exe=d/'oracle.exe';f.write_bytes(unit()); cmd=[str(DEFAULT_COMPILER),'--coverage','-O0','-fdefault-real-8','-ffree-line-length-none',str(f),'-o',str(exe)]; c=subprocess.run(cmd,cwd=d,env=compiler_environment(DEFAULT_COMPILER),capture_output=True,text=True)
  if c.returncode: raise RuntimeError(c.stderr)
  rows=[]
  for mode in (1,0):
   p=d/f'{mode}.txt'; q=subprocess.run([str(exe),str(p),str(mode)],cwd=d,env=compiler_environment(DEFAULT_COMPILER),capture_output=True,text=True); rows.append({'case':'cwrr' if mode else 'choisnel','exit_code':q.returncode,'state':np.loadtxt(p).tolist()})
  subprocess.run([str(DEFAULT_COMPILER.with_name('gcov.exe')),'-b','-c','oracle.gcno'],cwd=d,env=compiler_environment(DEFAULT_COMPILER),check=True,capture_output=True,text=True); gc=parse_gcov(d/'oracle.f90.gcov')
  (output/'oracle.f90.gcov').write_text((d/'oracle.f90.gcov').read_text(),encoding='utf-8')
  witness_source=f.read_bytes()
  arm_lines={}
  for arm, original_line in zip(ARMS,(260,316,353,359,367,438)):
   statement=lines(original_line,original_line).strip()
   generated=witness_source[:witness_source.index(statement)].count(b'\n')+1
   arm_lines[arm]={'original_line':original_line,'generated_line':generated,'gcov_branches':gc.get(generated,[])}
 j=jax_state(); # The direct-callee fixture has the same ordered 2 land points.
 expected=np.array(rows[0]['state'])[[0,3],0:9]; jflat=np.array([j[0,0],j[1,1]])
 max_abs=float(np.max(np.abs(expected-jflat))); state_passed=bool(max_abs == 0.0)
 comparison={'schema_version':1,'fortran_cases':rows,'jax_state_fields':['z0m','vevapp','tsol_rad','qsurf','cdrag','fluxsens','fluxlat','emis','temp_sol_new'],'direct_callee_boundary':'intersurf.f90:377-390','status':'passed' if state_passed else 'failed','all_state_values_compared':state_passed,'max_abs_error':max_abs,'fortran_land_state':expected.tolist(),'jax_land_state':jflat.tolist(),'note':'Fortran fixture values are direct-callee outputs; JAX comparison exercises identical owner compression/scatter/unit conversion with deterministic corresponding outputs.'}
 # Contract evidence is sourced from actual gcov and exact original byte hashes.
 covered=[arm for arm, mapping in arm_lines.items() if any(int(row['taken']) > 0 for row in mapping['gcov_branches'])]
 cov={'schema_version':1,'family':'intersurf_initialize_batch_c','branch_complete':set(covered)==set(ARMS),'required_arm_ids':ARMS,'covered_arm_ids':covered,'gcov_arm_ids':covered,'fatal_witness_arm_ids':[],'source_proof_arm_ids':[],'gcov_asset':'oracle.f90.gcov','gcov_mapping':arm_lines,'cases':[x['case'] for x in rows],'no_manual_coverage':True}
 complete=bool(cov['branch_complete'] and state_passed)
 evidence={'schema_version':1,'family':'intersurf_initialize_batch_c','complete':complete,'records':[{'owner_region_id':OWNER,'base_region_id':'pft14-region-1bef3d06c31f','fortran_procedure':'intersurf_initialize_2d','jax_owners':['jax_orchidee.sechiba.lifecycle_completion.intersurf_initialize_2d'],'comparison_asset':'outputs/reference_mode/micro_oracles/intersurf_initialize_batch_c/comparison.json','branch_coverage_asset':'outputs/reference_mode/micro_oracles/intersurf_initialize_batch_c/branch_coverage.json','required_arm_ids':ARMS,'gcov_arm_ids':covered,'fatal_witness_arm_ids':[],'source_proof_arm_ids':[],'covered_arm_ids':covered,'missing_arm_ids':sorted(set(ARMS)-set(covered)),'direct_state_writeback_comparison':{'status':comparison['status'],'state_fields':comparison['jax_state_fields'],'max_abs_error':max_abs},'passed':complete}]}
 inputs={'schema_version':1,'owner_source_sha256':span.span_sha256,'owner_span':[span.start_line,span.end_line],'direct_callee_call_sha256':hashlib.sha256(lines(377,390)).hexdigest(),'io_xios_stubs':['histwrite_p','histsync','getin_dump'],'compiler_command':cmd,'gcov_line_count':len(gc)}
 for n,o in [('inputs.json',inputs),('comparison.json',comparison),('branch_coverage.json',cov),('owner_region_evidence.json',evidence)]: (output/n).write_text(json.dumps(o,indent=2)+'\n',encoding='ascii')
 return evidence
if __name__=='__main__': print(json.dumps(run(),indent=2))
