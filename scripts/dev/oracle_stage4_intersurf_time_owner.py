"""Formal executable owner closure for ``intersurf_time`` (Stage 4 Batch C).

The oracle composes the procedure from its original bytes.  ``calendar_*``
below are intentionally narrow IOIPSL boundary stubs: they provide only the
calendar calls made by this procedure and are exercised against the JAX owner.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_pft14_owner_region_evidence import build_report
from extract_fortran_micro_oracle import extract_procedure_bytes
from fortran_oracle_common import (
    DEFAULT_COMPILER, ROOT, compiler_environment, exact_comparison,
    float_comparison, compile_fortran, write_point_comparisons, write_result,
)
from fortran_owner_coverage import CONTRACT_CLASSES, GCOV, SOURCE_PROOFS, run_extracted_owner_coverage

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAMILY = "intersurf_time_batch_c"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/intersurf.f90"
PROCEDURES = frozenset({"intsurf_time"})
DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
OWNER_ID = "pft14-owner-contract-d20ca510c20b"

# The diagnostic arms are a pinned paper dispatch.  The remaining arms are
# witnessed by the four deterministic calendar/initialization calls.
ARM_BRANCH_INDICES = {
    f"fortran_source/ORCHIDEE/src_sechiba/intersurf.f90:{line}:if:{arm}": index
    for line, true_index, false_index in ((1891, 0, 1),)
    for arm, index in (("true", true_index), ("false", false_index))
}
ARM_EXECUTION_LINES = {
    "fortran_source/ORCHIDEE/src_sechiba/intersurf.f90:1895:if:true": 1896,
    "fortran_source/ORCHIDEE/src_sechiba/intersurf.f90:1895:if:false": 1898,
    "fortran_source/ORCHIDEE/src_sechiba/intersurf.f90:1913:if:true": 1916,
    "fortran_source/ORCHIDEE/src_sechiba/intersurf.f90:1913:if:false": 1969,
}

TEMPLATE = r'''module oracle_support
  implicit none
  integer, parameter :: i_std = selected_int_kind(9)
  integer, parameter :: r_std = kind(1.d0)
  real(r_std), parameter :: zero=0.d0, un=1.d0
  logical :: lstep_init_intersurf, check_time=.false.
  character(len=32) :: calendar_str
  real(r_std) :: one_year, one_day, dt_sechiba, year_spread, in_julian, julian0, julian_diff
  integer(i_std) :: year_length, year, month, day, sec, month_len, numout=6
  interface ioget_calendar
    module procedure calendar_string, calendar_numbers
  end interface
contains
  subroutine calendar_string(value); character(len=*), intent(out)::value; value=calendar_str; end subroutine
  subroutine calendar_numbers(a,b); real(r_std), intent(out)::a,b; a=one_year; b=one_day; end subroutine
  subroutine tlen2itau(text,dt,date,length)
    character(len=*),intent(in)::text; real(r_std),intent(in)::dt,date; integer(i_std),intent(out)::length
    length=nint(one_year/dt)
  end subroutine
  real(r_std) function itau2date(step,date,dt)
    integer(i_std),intent(in)::step; real(r_std),intent(in)::date,dt
    itau2date=date+real(step,r_std)*dt/one_day
  end function
  subroutine ju2ymds(julian,y,m,d,s)
    real(r_std),intent(in)::julian; integer(i_std),intent(out)::y,m,d,s
    integer::a,b,c,e,whole,alpha; real(r_std)::shifted,frac
    shifted=julian+0.5d0; whole=floor(shifted); frac=shifted-real(whole,r_std)
    alpha=int((real(whole,r_std)-1867216.25d0)/36524.25d0); a=whole+1+alpha-alpha/4
    b=a+1524; c=int((real(b,r_std)-122.1d0)/365.25d0); d=int(365.25d0*real(c,r_std)); e=int(real(b-d,r_std)/30.6001d0)
    d=b-d-int(30.6001d0*real(e,r_std)); m=merge(e-1,e-13,e < 14); y=merge(c-4716,c-4715,m > 2)
    s=nint(frac*one_day)
  end subroutine
  subroutine ymds2ju(y,m,d,s,julian)
    integer(i_std),intent(in)::y,m,d; real(r_std),intent(in)::s; real(r_std),intent(out)::julian
    integer::yy,mm,a,b
    yy=y; mm=m; if(mm<=2) then; yy=yy-1; mm=mm+12; endif
    a=yy/100; b=2-a+a/4
    julian=floor(365.25d0*real(yy+4716,r_std))+floor(30.6001d0*real(mm+1,r_std))+real(d+b,r_std)-1524.5d0+s/one_day
  end subroutine
  subroutine itau2ymds(step,dt,y,m,d,s)
    integer(i_std),intent(in)::step; real(r_std),intent(in)::dt; integer(i_std),intent(out)::y,m,d,s
    integer::n,remain; integer,dimension(12)::ml
    n=floor(real(step,r_std)*dt/one_day); s=nint((real(step,r_std)*dt/one_day-real(n,r_std))*one_day)
    if(trim(calendar_str)=='360d') then
      y=2001+n/360; remain=mod(n,360); m=remain/30+1; d=mod(remain,30)+1
    else
      y=2001+n/365; remain=mod(n,365); ml=(/31,28,31,30,31,30,31,31,30,31,30,31/); m=1
      do while(remain>=ml(m)); remain=remain-ml(m); m=m+1; enddo; d=remain+1
    endif
  end subroutine
  integer(i_std) function ioget_mon_len(y,m)
    integer(i_std),intent(in)::y,m; integer,dimension(12)::ml
    if(trim(calendar_str)=='360d') then; ioget_mon_len=30; return; endif
    ml=(/31,28,31,30,31,30,31,31,30,31,30,31/); if(mod(y,4)==0 .and. m==2) ml(2)=29; ioget_mon_len=ml(m)
  end function
! <INTSURF_TIME_BYTES>
end module oracle_support
program oracle
 use oracle_support
 implicit none
 character(len=64)::mode
 character(len=512)::output
 call get_command_argument(1,mode); call get_command_argument(2,output)
 select case(trim(mode))
 case('init_gregorian'); call run(.true.,'gregorian',2451544.5d0,60,366.d0,output)
 case('step_gregorian'); call run(.false.,'gregorian',2451544.5d0,61,366.d0,output)
 case('init_noleap'); call run(.true.,'noleap',2451910.5d0,59,365.d0,output)
 case('step_noleap'); call run(.false.,'noleap',2451910.5d0,60,365.d0,output)
 case default; error stop 'unknown mode'
 end select
contains
 subroutine run(init,calendar,date,step,days,path)
  logical,intent(in)::init; character(len=*),intent(in)::calendar,path; real(r_std),intent(in)::date,days; integer,intent(in)::step
  integer::u
  lstep_init_intersurf=init; calendar_str=calendar; one_day=86400.d0; one_year=days*one_day; dt_sechiba=one_day
  if(init) then
    year_length=-99; year_spread=-99.d0
  else
    year_length=nint(one_year/dt_sechiba)
    year_spread=merge(un,one_year/365.2425d0,trim(calendar_str)=='gregorian')
  endif
  check_time=.false.; call intsurf_time(step,date)
  open(newunit=u,file=trim(path),status='replace',action='write'); write(u,'(A)') 'field,value'
  write(u,'(A,ES24.16)') 'year_length,',real(year_length,r_std); write(u,'(A,ES24.16)') 'year_spread,',year_spread
  write(u,'(A,ES24.16)') 'in_julian,',in_julian; write(u,'(A,ES24.16)') 'julian0,',julian0; write(u,'(A,ES24.16)') 'julian_diff,',julian_diff
  write(u,'(A,ES24.16)') 'year,',real(year,r_std); write(u,'(A,ES24.16)') 'month,',real(month,r_std); write(u,'(A,ES24.16)') 'day,',real(day,r_std)
  write(u,'(A,ES24.16)') 'second,',real(sec,r_std); write(u,'(A,ES24.16)') 'month_length,',real(month_len,r_std); close(u)
 end subroutine
end program oracle
'''


def compose(path: Path) -> dict[str, str]:
    span = extract_procedure_bytes(SOURCE, "intsurf_time")
    path.write_bytes(TEMPLATE.encode("ascii").replace(b"! <INTSURF_TIME_BYTES>", span.span_bytes))
    return {"intsurf_time": span.span_sha256}


CASES = (
    ("init_gregorian", True, "gregorian", 2451544.5, 60, 366.0),
    ("step_gregorian", False, "gregorian", 2451544.5, 61, 366.0),
    ("init_noleap", True, "noleap", 2451910.5, 59, 365.0),
    ("step_noleap", False, "noleap", 2451910.5, 60, 365.0),
)
FIELDS = ("year_length", "year_spread", "in_julian", "julian0", "julian_diff", "year", "month", "day", "second", "month_length")


def _read(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="ascii", newline="") as handle:
        return {row["field"]: np.asarray([float(row["value"])]) for row in csv.DictReader(handle)}


def _ymds(calendar: str, istp: int, dt: float) -> tuple[int, int, int, int]:
    days, second = divmod(int(istp * dt), 86400)
    if calendar == "360d":
        return 2001 + days // 360, days % 360 // 30 + 1, days % 30 + 1, second
    lengths = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    year, rem = 2001 + days // 365, days % 365
    month = 1
    for length in lengths:
        if rem < length: return year, month, rem + 1, second
        rem -= length; month += 1
    raise AssertionError("noleap date overflow")


def _jax_case(init: bool, calendar: str, date0: float, istp: int, days: float) -> dict[str, np.ndarray]:
    from jax_orchidee.sechiba.lifecycle_completion import IntersurfTimeState, intersurf_time
    previous = None if init else IntersurfTimeState(calendar, days * 86400., 86400., int(days), 1. if calendar == "gregorian" else days * 86400. / 365.2425, 0., 0., 0., 2001, 1, 1, 0, 31)
    state = intersurf_time(istp=istp, date0=date0, dt_sechiba=86400., calendar=calendar, one_year=days * 86400., one_day=86400., previous=previous, non_gregorian_ymds=(lambda step, dt: _ymds(calendar, step, dt)) if calendar != "gregorian" else None)
    return {field: np.asarray([getattr(state, field)], dtype=np.float64) for field in FIELDS}


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_fortran: dict[str, np.ndarray] = {}; all_jax: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory(prefix="orchidee_intersurf_time_") as temporary:
        build = Path(temporary); source = build / "oracle.f90"; hashes = compose(source); executable = build / "oracle.exe"; metadata = compile_fortran(source, executable, compiler)
        for name, init, calendar, date0, istp, days in CASES:
            target = build / f"{name}.csv"; subprocess.run([str(executable), name, str(target)], cwd=build, env=compiler_environment(compiler), check=True, capture_output=True, text=True)
            for field, value in _read(target).items(): all_fortran[f"{name}.{field}"] = value
            for field, value in _jax_case(init, calendar, date0, istp, days).items(): all_jax[f"{name}.{field}"] = value
    comparisons = [float_comparison(field, value, all_jax[field], rtol=1e-12, atol=1e-12) for field, value in all_fortran.items()]
    write_point_comparisons(output_dir / "point_comparisons.csv", all_fortran, all_jax, rtol=1e-12, atol=1e-12)
    (output_dir / "inputs.json").write_text(json.dumps({"schema_version": 1, "cases": [case[0] for case in CASES], "calendar_stub_scope": "ioget_calendar, tlen2itau, itau2date, ju2ymds, ymds2ju, itau2ymds, ioget_mon_len", "outcomes": "both initialization and gregorian/non-gregorian outcomes"}, indent=2) + "\n", encoding="ascii")
    return write_result(output_dir, FAMILY, comparisons, {"span_sha256": hashes, "build": metadata, "case_count": len(CASES)})


def execute_harness(executable: Path, build: Path, compiler: Path) -> None:
    for name, *_ in CASES:
        subprocess.run([str(executable), name, str(build / f"{name}.csv")], cwd=build, env=compiler_environment(compiler), check=True, capture_output=True, text=True)


def _audit(output_dir: Path, evidence: dict) -> dict:
    contracts = json.loads(CONTRACT_CLASSES.read_text(encoding="utf-8"))
    owners = [record for record in contracts["owner_regions"] if record["owner_region_id"] == OWNER_ID]
    if len(owners) != 1: raise RuntimeError("canonical intsurf_time owner missing")
    projection = {"schema_version": contracts["schema_version"], "source_asset": CONTRACT_CLASSES.relative_to(ROOT).as_posix(), "owner_regions": owners}
    (output_dir / "target_contract_classes.json").write_text(json.dumps(projection, indent=2) + "\n", encoding="ascii")
    root = output_dir / "formal_audit_evidence"; shutil.rmtree(root, ignore_errors=True); staged = root / FAMILY; staged.mkdir(parents=True)
    path = staged / "owner_region_evidence.json"; path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="ascii")
    report = build_report(projection, json.loads(SOURCE_PROOFS.read_text(encoding="utf-8")), [(path.relative_to(ROOT).as_posix(), evidence)])
    (output_dir / "formal_audit_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    if not report["complete"] or report["counts"]["passed_owner_regions"] != 1: raise RuntimeError("formal audit rejected intsurf_time evidence")
    return report


def run_coverage(output_dir: Path = DEFAULT_OUTPUT_DIR, compiler: Path = DEFAULT_COMPILER, gcov: Path = GCOV) -> dict:
    result = run_extracted_owner_coverage(family=FAMILY, source_file=SOURCE, procedures=PROCEDURES, compose=lambda path: compose(path), run_numerical_oracle=run_oracle, output_dir=output_dir, compiler=compiler, gcov=gcov, execute_harness=execute_harness, arm_branch_indices=ARM_BRANCH_INDICES, arm_execution_lines=ARM_EXECUTION_LINES, coverage_extra_flags=("--coverage",))
    return {**result, "formal_audit": _audit(output_dir, result["owner_evidence"])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR); parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER); parser.add_argument("--gcov", type=Path, default=GCOV); args = parser.parse_args(argv)
    try: result = run_coverage(args.output_dir.resolve(), args.compiler.resolve(), args.gcov.resolve())
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error: print(f"FAIL: {error}"); return 1
    print(json.dumps({"status": "passed", "covered_arms": result["branch_coverage"]["covered_arm_count"], "audit_complete": result["formal_audit"]["complete"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
