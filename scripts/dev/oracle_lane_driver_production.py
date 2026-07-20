from __future__ import annotations

import subprocess
import tempfile
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMPILER = Path(r"C:\msys64\ucrt64\bin\gfortran.exe")
SECHIBA = ROOT / "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90"
STOMATE_IO = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90"


def _lines(path: Path, start: int, end: int) -> bytes:
    return b"".join(path.read_bytes().splitlines(keepends=True)[start - 1 : end])


def restart_call_signatures() -> list[tuple[str, int, tuple[str, ...]]]:
    text = STOMATE_IO.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"!.*", "", text)
    result: list[tuple[str, int, tuple[str, ...]]] = []
    for match in re.finditer(r"CALL\s+(rest(?:put|get)_p)\s*\(", text, re.I):
        start = match.end()
        depth = 1
        quoted = False
        end = start
        while depth and end < len(text):
            char = text[end]
            if char == "'":
                quoted = not quoted
            elif not quoted:
                depth += int(char == "(") - int(char == ")")
            end += 1
        body = text[start : end - 1]
        args: list[str] = []
        item_start = 0
        depth = 0
        quoted = False
        for index, char in enumerate(body):
            if char == "'":
                quoted = not quoted
            elif not quoted:
                depth += int(char == "(") - int(char == ")")
                if char == "," and depth == 0:
                    args.append(body[item_start:index].strip())
                    item_start = index + 1
        args.append(body[item_start:].strip())
        result.append((match.group(1).lower(), len(args), tuple(args)))
    return result


def _restart_io_stub() -> bytes:
    return rb"""
module restart_memory
  implicit none
  integer, parameter :: max_fields=256, max_values=200000
  integer :: field_count=0, put_count=0, get_count=0
  character(len=80) :: names(max_fields)=''
  integer :: lengths(max_fields)=0
  real(8) :: values(max_values,max_fields)=0
contains
  integer function slot(name)
    character(len=*),intent(in)::name
    integer::i
    slot=0
    do i=1,field_count
      if(trim(names(i))==trim(name)) then; slot=i; return; endif
    enddo
    field_count=field_count+1; names(field_count)=trim(name); slot=field_count
  end function
  subroutine save_value(name,value)
    character(len=*),intent(in)::name
    real(8),intent(in)::value(..)
    integer::s
    s=slot(name); put_count=put_count+1
    select rank(value)
    rank(0); lengths(s)=1; values(1,s)=value
    rank(1); lengths(s)=size(value); values(1:lengths(s),s)=value
    rank(2); lengths(s)=size(value); values(1:lengths(s),s)=reshape(value,[size(value)])
    rank(3); lengths(s)=size(value); values(1:lengths(s),s)=reshape(value,[size(value)])
    rank(4); lengths(s)=size(value); values(1:lengths(s),s)=reshape(value,[size(value)])
    rank(5); lengths(s)=size(value); values(1:lengths(s),s)=reshape(value,[size(value)])
    rank(6); lengths(s)=size(value); values(1:lengths(s),s)=reshape(value,[size(value)])
    end select
  end subroutine
  subroutine load_value(name,value,default)
    character(len=*),intent(in)::name
    real(8),intent(out)::value(..)
    real(8),intent(in)::default
    integer::s
    s=slot(name); get_count=get_count+1
    select rank(value)
    rank(0); value=merge(values(1,s),default,lengths(s)==1)
    rank(1); if(lengths(s)==size(value))then;value=values(1:lengths(s),s);else;value=default;endif
    rank(2); if(lengths(s)==size(value))then;value=reshape(values(1:lengths(s),s),shape(value));else;value=default;endif
    rank(3); if(lengths(s)==size(value))then;value=reshape(values(1:lengths(s),s),shape(value));else;value=default;endif
    rank(4); if(lengths(s)==size(value))then;value=reshape(values(1:lengths(s),s),shape(value));else;value=default;endif
    rank(5); if(lengths(s)==size(value))then;value=reshape(values(1:lengths(s),s),shape(value));else;value=default;endif
    rank(6); if(lengths(s)==size(value))then;value=reshape(values(1:lengths(s),s),shape(value));else;value=default;endif
    end select
  end subroutine
end module restart_memory
module ioipsl_para
  use restart_memory
  implicit none
  interface restput_p
    module procedure put_real_scalar,put_int_scalar,put_rank3,put_rank4,put_rank5
  end interface
  interface restget_p
    module procedure get_real_scalar,get_int_scalar,get_rank3,get_rank4,get_rank5
  end interface
contains
  subroutine put_real_scalar(id,name,time,value)
    integer,intent(in)::id,time;character(len=*),intent(in)::name;real(8),intent(in)::value
    call save_value(name,value)
  end
  subroutine put_int_scalar(id,name,time,value)
    integer,intent(in)::id,time,value;character(len=*),intent(in)::name
    call save_value(name,real(value,8))
  end
  subroutine put_rank3(id,name,d0,d1,d2,time,value,mode,ng,index)
    integer,intent(in)::id,d0,d1,d2,time,ng,index(:);character(len=*),intent(in)::name,mode
    real(8),intent(in)::value(..);call save_value(name,value)
  end
  subroutine put_rank4(id,name,d0,d1,d2,d3,time,value,mode,ng,index)
    integer,intent(in)::id,d0,d1,d2,d3,time,ng,index(:);character(len=*),intent(in)::name,mode
    real(8),intent(in)::value(..);call save_value(name,value)
  end
  subroutine put_rank5(id,name,d0,d1,d2,d3,d4,time,value,mode,ng,index)
    integer,intent(in)::id,d0,d1,d2,d3,d4,time,ng,index(:);character(len=*),intent(in)::name,mode
    real(8),intent(in)::value(..);call save_value(name,value)
  end
  subroutine get_real_scalar(id,name,time,required,default,value)
    integer,intent(in)::id,time;logical,intent(in)::required;character(len=*),intent(in)::name
    real(8),intent(in)::default;real(8),intent(out)::value;call load_value(name,value,default)
  end
  subroutine get_int_scalar(id,name,time,required,default,value)
    integer,intent(in)::id,time;logical,intent(in)::required;character(len=*),intent(in)::name
    real(8),intent(in)::default;integer,intent(out)::value;real(8)::tmp
    call load_value(name,tmp,default);value=nint(tmp)
  end
  subroutine get_rank3(id,name,d0,d1,d2,time,required,value,mode,ng,index)
    integer,intent(in)::id,d0,d1,d2,time,ng,index(:);logical,intent(in)::required
    character(len=*),intent(in)::name,mode;real(8),intent(out)::value(..)
    call load_value(name,value,9.96921e36_8)
  end
  subroutine get_rank4(id,name,d0,d1,d2,d3,time,required,value,mode,ng,index)
    integer,intent(in)::id,d0,d1,d2,d3,time,ng,index(:);logical,intent(in)::required
    character(len=*),intent(in)::name,mode;real(8),intent(out)::value(..)
    call load_value(name,value,9.96921e36_8)
  end
  subroutine get_rank5(id,name,d0,d1,d2,d3,d4,time,required,value,mode,ng,index)
    integer,intent(in)::id,d0,d1,d2,d3,d4,time,ng,index(:);logical,intent(in)::required
    character(len=*),intent(in)::name,mode;real(8),intent(out)::value(..)
    call load_value(name,value,9.96921e36_8)
  end
end module ioipsl_para
"""


def _restart_owner_unit() -> bytes:
    support = rb"""
module constantes
  implicit none
  integer,parameter::i_std=4,r_std=8
  real(8),parameter::zero=0.d0,un=1.d0,val_exp=9.96921d36,large_value=1.d33
  real(8),parameter::undef=-9999.d0,undef_sechiba=-9999.d0
  integer::printlev=0,numout=6
end module constantes
module constantes_soil
  implicit none
  integer,parameter::nslm=11,ndeep=32,nsnow=3,nvert=20,ns=20
end module constantes_soil
module mod_orchidee_para; implicit none; end module
module stomate_data
  implicit none
  integer,parameter::nvm=14,nparts=8,nelements=1,nleafages=4,nlitt=2,nlevs=2,ncarb=3
  integer,parameter::nbpools=7,npco2=6,npool=4,ndoc=2,nwp=1,months_num_const=12
  integer,parameter::imetabolic=1,istructural=2,iabove=1,ibelow=2,icarbon=1
  integer,parameter::ifree=1,iadsorbed=2
  integer::rest_id_stomate=1,itime=1,nbp_glo=2,index_g(2)=[1,2]
  logical::reset_thawed_humidity=.false.,spinup_analytic=.false.
  real(8)::gdd_crit_estab=0.d0,O2_init_conc=0.d0,CH4_init_conc=0.d0
end module stomate_data
"""
    module_head = rb"""
module stomate_io
  use stomate_data
  use constantes
  use constantes_soil
  use mod_orchidee_para
  use ioipsl_para
  implicit none
  private
  public readstart,writerestart
  real(8),allocatable,save::trefe(:)
contains
"""
    return (
        _restart_io_stub()
        + support
        + module_head
        + _lines(STOMATE_IO, 34, 1747)
        + b"\n"
        + _lines(STOMATE_IO, 1751, 2944)
        + b"\nend module stomate_io\n"
    )


def compile_restart_owners() -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="orchidee_restart_owner_") as td:
        build = Path(td)
        source = build / "restart_owner.f90"
        source.write_bytes(_restart_owner_unit())
        environment = {
            **os.environ,
            "PATH": str(COMPILER.parent) + os.pathsep + os.environ.get("PATH", ""),
        }
        result = subprocess.run(
            [str(COMPILER), "-O0", "-fno-frontend-optimize", "-std=f2018",
             "-fdefault-real-8", "-ffree-line-length-none", "-c", str(source),
             "-o", str(build / "restart_owner.o")],
            cwd=td, env=environment, capture_output=True, text=True,
        )
        if result.returncode:
            diagnostic = ROOT / "outputs/reference_mode/micro_oracles/driver_restart_handoff/restart_owner_compile_diagnostic.f90"
            diagnostic.parent.mkdir(parents=True, exist_ok=True)
            diagnostic.write_bytes(source.read_bytes())
        return result


def _module_order_unit() -> bytes:
    """Build a compile unit containing the byte-exact sechiba_main owner."""
    stubs = rb"""
module trace_log
  implicit none
  integer :: trace_count=0
  character(len=32) :: trace_names(32)
contains
  subroutine trace(name)
    character(len=*), intent(in) :: name
    trace_count=trace_count+1
    trace_names(trace_count)=name
  end subroutine trace
end module trace_log
module ioipsl; implicit none; end module
module xios_orchidee; implicit none; end module
module constantes
  implicit none
  integer, parameter :: i_std=4, r_std=8
  integer, parameter :: nvm=14, nflow=1, nsnow=3, nslm=11, nstm=3
  integer, parameter :: nlai=20, nnobio=1, ngrnd=7, nbdl=11, nbneighb=8
  integer, parameter :: ih2o=1, ncarb=3, npco2=6
  real(r_std), parameter :: zero=0._r_std, un=1._r_std, huit=8._r_std
  real(r_std), parameter :: mille=1000._r_std, min_sechiba=1.e-8_r_std
  real(r_std), parameter :: dt_sechiba=1800._r_std, one_day=86400._r_std
  real(r_std) :: min_wind=0.1_r_std
  integer :: printlev=0, numout=6, nbp_glo=1
  logical :: almaoutput=.false., ok_co2=.true.
end module constantes
module constantes_soil
  use constantes, only: i_std, r_std, nstm, nslm
  implicit none
  integer, parameter :: ndeep=32, ndoc=2, npool=4, nelements=1
  integer :: pref_soil_veg(14)=1
end module constantes_soil
module pft_parameters
  implicit none
  logical :: ok_LAIdev(14)=.false.
  real(8) :: irrig_threshold(14)=0.5d0
  real(8) :: irrig_fulfill(14)=1.d0, ext_coeff(14)=0.5d0
  logical :: is_tree(14)=.false., natural(14)=.true.
end module pft_parameters
module grid; implicit none; end module
module diffuco; implicit none; end module
module condveg; implicit none; end module
module enerbil; implicit none; end module
module hydrol
  implicit none
  logical :: hydrol_cwrr=.true., ok_explicitsnow=.true.
end module hydrol
module hydrolc; implicit none; end module
module thermosoil; implicit none; end module
module thermosoilc; implicit none; end module
module sechiba_io; implicit none; end module
module slowproc; implicit none; end module
module erosion
  implicit none
  logical :: erosion_module=.false.
end module erosion
module routing
  implicit none
  logical :: river_routing=.false., do_fullirr=.false., irrig_drip=.false.
  logical :: do_floodplains=.false.
  real(8) :: irrig_dosmax=1.d0
end module routing
module ioipsl_para; implicit none; end module
"""
    prefix = _lines(SECHIBA, 28, 468).replace(
        b"PUBLIC sechiba_main, sechiba_initialize, sechiba_clear",
        b"PUBLIC sechiba_main, oracle_init_state",
    )
    prefix = prefix.replace(
        b"  IMPLICIT NONE\n",
        b"  IMPLICIT NONE\n"
        b"  logical :: dynpeat_PWT=.false., ok_rotate=.false., ok_stomate=.true.\n"
        b"  logical :: done_stomate_lcchange=.false., dyn_peat=.false., use_age_class=.false.\n"
        b"  integer :: rot_cmd_max=1\n",
        1,
    )
    owner = _lines(SECHIBA, 829, 1786)
    return stubs + prefix + owner + _module_state_initializer(prefix)


def _module_state_initializer(prefix: bytes) -> bytes:
    statements: list[bytes] = []
    declaration = re.compile(
        rb"^\s*(INTEGER|REAL|LOGICAL)[^\n]*ALLOCATABLE[^\n]*::\s*([A-Za-z][A-Za-z0-9_]*)",
        re.IGNORECASE,
    )
    dimension = re.compile(rb"DIMENSION\s*\(([^)]*)\)", re.IGNORECASE)
    for line in prefix.splitlines():
        match = declaration.match(line)
        if not match:
            continue
        dims = dimension.search(line)
        if not dims:
            continue
        rank = dims.group(1).count(b",") + 1
        name = match.group(2)
        shape = b",".join([b"32"] * rank)
        value = b".false." if match.group(1).upper() == b"LOGICAL" else b"0"
        statements.append(b"    if (.not. allocated(" + name + b")) allocate(" + name + b"(" + shape + b"))")
        statements.append(b"    " + name + b" = " + value)
    return (
        b"\n  subroutine oracle_init_state\n"
        + b"\n".join(statements)
        + b"\n  end subroutine oracle_init_state\nEND MODULE sechiba\n"
    )


def _module_order_driver() -> bytes:
    return rb"""
subroutine diffuco_main(); use trace_log; call trace('diffuco_main'); end
subroutine enerbil_main(); use trace_log; call trace('enerbil_main'); end
subroutine hydrol_main(); use trace_log; call trace('hydrol_main'); end
subroutine hydrolc_main(); use trace_log; call trace('hydrolc_main'); end
subroutine enerbil_fusion(); use trace_log; call trace('enerbil_fusion'); end
subroutine condveg_main(); use trace_log; call trace('condveg_main'); end
subroutine thermosoil_main(); use trace_log; call trace('thermosoil_main'); end
subroutine thermosoilc_main(); use trace_log; call trace('thermosoilc_main'); end
subroutine slowproc_main(); use trace_log; call trace('slowproc_main'); end
subroutine sechiba_var_init(); end
subroutine sechiba_end(); end
subroutine sechiba_finalize(); end
subroutine sechiba_get_cmd(); end
subroutine slowproc_change_frac(); end
subroutine hydrol_rotation_update(); end
subroutine thermosoil_rotation_update(); end
subroutine erosion_main(); end
subroutine routing_main(); end
subroutine histwrite_p(); end
subroutine xios_orchidee_send_field(); end
program module_order_oracle
  use sechiba
  use trace_log
  implicit none
  integer :: index(1), neighbours(1,8), i
  real(8) :: lalo(1,2), contfrac(1), resolution(1,2), zlev(1), u(1), v(1)
  real(8) :: qair(1),q2m(1),t2m(1),temp_air(1),epot_air(1),ccanopy(1)
  real(8) :: tq(1),pa(1),qa(1),pbt(1),qbt(1),rain(1),snow(1),lw(1),swn(1),swd(1),cosz(1),pb(1)
  real(8) :: vev(1),fs(1),fl(1),coast(1,1),river(1,1),netco2(1),fco2(1)
  real(8) :: trad(1),tnew(1),qsurf(1),alb(1,2),emis(1),z0m(1),z0h(1)
  call oracle_init_state()
  index=1; neighbours=1; lalo=0; contfrac=1; resolution=1; zlev=2; u=1; v=1
  qair=0.01d0; q2m=qair; t2m=280; temp_air=280; epot_air=280; ccanopy=400
  tq=0.1d0; pa=1; qa=1; pbt=0; qbt=0; rain=0; snow=0; lw=300; swn=0; swd=0; cosz=0; pb=101325
  call sechiba_main(1,1,1,index,0.0,1961,.false.,.false.,lalo,contfrac,neighbours,resolution, &
    zlev,u,v,qair,q2m,t2m,temp_air,epot_air,ccanopy,tq,pa,qa,pbt,qbt,rain,snow,lw,swn,swd,cosz,pb, &
    vev,fs,fl,coast,river,netco2,fco2,trad,tnew,qsurf,alb,emis,z0m,z0h,1,1,1,1,1,1)
  do i=1,trace_count
    write(*,'(A)') trim(trace_names(i))
  end do
end program module_order_oracle
"""


def compile_module_order_owner() -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="orchidee_sechiba_owner_") as td:
        build = Path(td)
        source = build / "owner.f90"
        source.write_bytes(_module_order_unit())
        driver = build / "driver.f90"
        driver.write_bytes(_module_order_driver())
        environment = {
            **os.environ,
            "PATH": str(COMPILER.parent) + os.pathsep + os.environ.get("PATH", ""),
        }
        common = [
            str(COMPILER), "-O0", "-fno-frontend-optimize", "-fdefault-real-8",
            "-ffree-line-length-none",
        ]
        result = subprocess.run(
            [
                *common,
                "-c",
                str(source),
                "-o",
                str(build / "owner.o"),
            ],
            cwd=td,
            env=environment,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            driver_result = subprocess.run(
                [*common, "-c", str(driver), "-o", str(build / "driver.o")],
                cwd=td, env=environment, capture_output=True, text=True,
            )
            if driver_result.returncode:
                result = driver_result
            else:
                result = subprocess.run(
                    [str(COMPILER), str(build / "owner.o"), str(build / "driver.o"), "-o", str(build / "oracle.exe")],
                    cwd=td, env=environment, capture_output=True, text=True,
                )
                if result.returncode == 0:
                    result = subprocess.run(
                        [str(build / "oracle.exe")], cwd=td, env=environment,
                        capture_output=True, text=True,
                    )
        if result.returncode and not result.stderr:
            diagnostic = ROOT / "outputs/reference_mode/micro_oracles/driver_sechiba_module_order/owner_compile_diagnostic.f90"
            diagnostic.parent.mkdir(parents=True, exist_ok=True)
            diagnostic.write_bytes(source.read_bytes())
        return result


if __name__ == "__main__":
    result = compile_module_order_owner()
    print(result.stdout)
    print(result.stderr)
    raise SystemExit(result.returncode)
