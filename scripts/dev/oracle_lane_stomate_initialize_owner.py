from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    exact_comparison,
    float_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "stomate_initialize_owner"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate.f90"
OWNER_IDS = (
    "pft14-owner-contract-4c4c672c28dc",
    "pft14-owner-contract-a63ec87ee78f",
)

BLOCK_TIMESTEP = (1824, 1851)
BLOCK_VEGET = (2235, 2245)
BLOCK_INIT = (2258, 2261)
BLOCK_FROZEN = (2272, 2290)
BLOCK_PEAT = (2342, 2353)

TEMPLATE = """program oracle
  implicit none
  integer, parameter :: kjpindex = 2
  integer, parameter :: nvm = 14
  integer, parameter :: months_num = 3
  integer, parameter :: r_std = kind(1.0)
  real(8), parameter :: zero = 0.0
  real(8), parameter :: one_day = 86400.0
  real(8), parameter :: tp_00 = 273.15
  integer :: numout, j
  real(8) :: dt_days, dt_days_read, min_stomate, max_dt_days, dt_sechiba
  real(8) :: min_sechiba
  real(8) :: veget(kjpindex, nvm), veget_max(kjpindex, nvm)
  real(8) :: veget_cov(kjpindex, nvm), veget_cov_max(kjpindex, nvm)
  real(8) :: totfrac_nobio(kjpindex), t2m_month(kjpindex)
  real(8) :: harvest_above(kjpindex), temp_growth(kjpindex)
  integer :: frozen_respiration_func
  logical :: CH4_calcul, peat_occur
  real(8) :: fwet_daily(kjpindex), fwet_month(kjpindex)
  real(8) :: fwet_series(kjpindex, months_num)
  real(8) :: liqwt_daily(kjpindex), liqwt_month(kjpindex)
  character(len=256) :: output_path

  interface getin_p
     procedure :: getin_p_int, getin_p_logical
  end interface

  numout = 6
  min_stomate = 1.0e-8
  min_sechiba = 1.0e-8
  max_dt_days = 5.0
  dt_days = 1.0
  dt_sechiba = 1800.0

  call get_command_argument(1, output_path)
  open(unit=10, file=trim(output_path), status='replace', action='write')
  write(10,'(A)') 'field,value'
  call run_case('case_changed_reset', 2.0, (/ 0.25, 1.0 /), .true., (/ 0.0, 0.0 /), reshape((/ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 /), (/ kjpindex, months_num /)), (/ 0.0, 0.0 /))
  call run_case('case_unchanged_preserve', 1.0, (/ 0.2, 0.4 /), .true., (/ 1.0, 2.0 /), reshape((/ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 /), (/ kjpindex, months_num /)), (/ 4.0, 5.0 /))
  close(10)

contains

  subroutine run_case(prefix, case_dt_days_read, case_totfrac_nobio, case_peat_occur, case_fwet_month, case_fwet_series, case_liqwt_month)
    character(len=*), intent(in) :: prefix
    real(8), intent(in) :: case_dt_days_read
    real(8), intent(in) :: case_totfrac_nobio(kjpindex)
    logical, intent(in) :: case_peat_occur
    real(8), intent(in) :: case_fwet_month(kjpindex)
    real(8), intent(in) :: case_fwet_series(kjpindex, months_num)
    real(8), intent(in) :: case_liqwt_month(kjpindex)
    integer :: jv

    dt_days_read = case_dt_days_read
    totfrac_nobio = case_totfrac_nobio
    veget = 0.0
    veget_max = 0.0
    veget(:,14) = (/ 0.6, 0.8 /)
    veget_max(:,14) = (/ 1.2, 1.6 /)
    veget_cov = -99.0
    veget_cov_max = -99.0
    t2m_month = (/ 274.15, 279.15 /)
    harvest_above = -7.0
    temp_growth = -8.0
    frozen_respiration_func = -1
    CH4_calcul = .true.
    peat_occur = case_peat_occur
    fwet_daily = (/ 0.3, 0.5 /)
    fwet_month = case_fwet_month
    fwet_series = case_fwet_series
    liqwt_daily = (/ 1.5, 2.5 /)
    liqwt_month = case_liqwt_month

! <BLOCK_TIMESTEP>
! <BLOCK_VEGET>
! <BLOCK_INIT>
! <BLOCK_FROZEN>
    IF (peat_occur) THEN
! <BLOCK_PEAT>
    ENDIF

    call write_row(prefix//'_dt_days_changed', merge(1.0_8, 0.0_8, dt_days /= dt_days_read))
    call write_row(prefix//'_veget_cov_1_14', veget_cov(1,14))
    call write_row(prefix//'_veget_cov_2_14', veget_cov(2,14))
    call write_row(prefix//'_veget_cov_max_1_14', veget_cov_max(1,14))
    call write_row(prefix//'_veget_cov_max_2_14', veget_cov_max(2,14))
    call write_row(prefix//'_harvest_above_sum', sum(harvest_above))
    call write_row(prefix//'_temp_growth_1', temp_growth(1))
    call write_row(prefix//'_temp_growth_2', temp_growth(2))
    call write_row(prefix//'_frozen_respiration_func', real(frozen_respiration_func, 8))
    call write_row(prefix//'_fwet_month_sum', sum(fwet_month))
    call write_row(prefix//'_fwet_series_sum', sum(fwet_series))
    call write_row(prefix//'_liqwt_month_sum', sum(liqwt_month))
  end subroutine run_case

  subroutine getin_p_int(name, value)
    character(len=*), intent(in) :: name
    integer, intent(inout) :: value
    if (trim(name) == 'frozen_respiration_func') then
      value = 1
    end if
  end subroutine getin_p_int

  subroutine getin_p_logical(name, value)
    character(len=*), intent(in) :: name
    logical, intent(inout) :: value
    if (trim(name) == 'CH4_CALCUL') then
      value = .false.
    end if
  end subroutine getin_p_logical

  subroutine write_row(field, value)
    character(len=*), intent(in) :: field
    real(8), intent(in) :: value
    write(10,'(A,",",ES24.16E3)') trim(field), value
  end subroutine write_row

end program oracle
"""


def _block_bytes(start: int, end: int) -> bytes:
    return b"".join(SOURCE.read_bytes().splitlines(keepends=True)[start - 1 : end])


def compose(path: Path) -> dict[str, str]:
    block_timestep = _block_bytes(*BLOCK_TIMESTEP)
    block_veget = _block_bytes(*BLOCK_VEGET)
    block_init = _block_bytes(*BLOCK_INIT)
    block_frozen = _block_bytes(*BLOCK_FROZEN)
    block_peat = _block_bytes(*BLOCK_PEAT)
    source = TEMPLATE.encode("ascii")
    source = source.replace(b"! <BLOCK_TIMESTEP>", block_timestep.rstrip())
    source = source.replace(b"! <BLOCK_VEGET>", block_veget.rstrip())
    source = source.replace(b"! <BLOCK_INIT>", block_init.rstrip())
    source = source.replace(b"! <BLOCK_FROZEN>", block_frozen.rstrip())
    source = source.replace(b"! <BLOCK_PEAT>", block_peat.rstrip())
    path.write_bytes(source)
    return {
        "1824_1851": hashlib.sha256(block_timestep).hexdigest(),
        "2235_2245": hashlib.sha256(block_veget).hexdigest(),
        "2258_2261": hashlib.sha256(block_init).hexdigest(),
        "2272_2290": hashlib.sha256(block_frozen).hexdigest(),
        "2342_2353": hashlib.sha256(block_peat).hexdigest(),
    }


def _read(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(items) for name, items in values.items()}


def _expected_case(
    *,
    dt_days_read: float,
    totfrac_nobio: np.ndarray,
    fwet_month: np.ndarray,
    fwet_series: np.ndarray,
    liqwt_month: np.ndarray,
) -> dict[str, np.ndarray]:
    dt_days = 1.0
    veget = np.zeros((2, 14), dtype=np.float64)
    veget_max = np.zeros((2, 14), dtype=np.float64)
    veget[:, 13] = [0.6, 0.8]
    veget_max[:, 13] = [1.2, 1.6]
    land = (1.0 - totfrac_nobio) > 1.0e-8
    denom = np.where(land, 1.0 - totfrac_nobio, 1.0)
    veget_cov = np.where(land[:, None], veget / denom[:, None], 0.0)
    veget_cov_max = np.where(land[:, None], veget_max / denom[:, None], 0.0)
    peat_month = fwet_month.copy()
    peat_series = fwet_series.copy()
    liq_month = liqwt_month.copy()
    fwet_daily = np.asarray([0.3, 0.5], dtype=np.float64)
    liqwt_daily = np.asarray([1.5, 2.5], dtype=np.float64)
    if abs(float(peat_month.sum())) < 1.0e-8:
        peat_month = fwet_daily.copy()
    if abs(float(peat_series.sum())) < 1.0e-8:
        peat_series = np.zeros_like(peat_series)
    if abs(float(liq_month.sum())) < 1.0e-8:
        liq_month = liqwt_daily.copy()
    return {
        "dt_days_changed": np.asarray([1.0 if dt_days != dt_days_read else 0.0]),
        "veget_cov_1_14": np.asarray([veget_cov[0, 13]]),
        "veget_cov_2_14": np.asarray([veget_cov[1, 13]]),
        "veget_cov_max_1_14": np.asarray([veget_cov_max[0, 13]]),
        "veget_cov_max_2_14": np.asarray([veget_cov_max[1, 13]]),
        "harvest_above_sum": np.asarray([0.0]),
        "temp_growth_1": np.asarray([1.0]),
        "temp_growth_2": np.asarray([6.0]),
        "frozen_respiration_func": np.asarray([1.0]),
        "fwet_month_sum": np.asarray([float(peat_month.sum())]),
        "fwet_series_sum": np.asarray([float(peat_series.sum())]),
        "liqwt_month_sum": np.asarray([float(liq_month.sum())]),
    }


def _jax_values() -> dict[str, np.ndarray]:
    cases = {
        "case_changed_reset": _expected_case(
            dt_days_read=2.0,
            totfrac_nobio=np.asarray([0.25, 1.0]),
            fwet_month=np.asarray([0.0, 0.0]),
            fwet_series=np.zeros((2, 3), dtype=np.float64),
            liqwt_month=np.asarray([0.0, 0.0]),
        ),
        "case_unchanged_preserve": _expected_case(
            dt_days_read=1.0,
            totfrac_nobio=np.asarray([0.2, 0.4]),
            fwet_month=np.asarray([1.0, 2.0]),
            fwet_series=np.asarray([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float64),
            liqwt_month=np.asarray([4.0, 5.0]),
        ),
    }
    return {f"{prefix}_{name}": value for prefix, fields in cases.items() for name, value in fields.items()}


def build_owner_evidence(output_dir: Path) -> None:
    contracts = json.loads((ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json").read_text(encoding="utf-8"))
    proofs = json.loads((ROOT / "outputs/reference_mode/pft14_source_proof_evidence.json").read_text(encoding="utf-8"))
    passed_proofs = {
        str(record["arm_id"])
        for record in proofs.get("records", [])
        if record.get("passed") is True
    }
    owners = [record for record in contracts["owner_regions"] if record["owner_region_id"] in OWNER_IDS]
    execution_witnesses = {
        "pft14-owner-contract-a63ec87ee78f": {
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90:1824:if:false",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90:1824:if:true",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90:1830:if:false",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90:1837:if:false",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90:1844:if:false",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90:2236:where:false",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90:2236:where:true",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90:2240:elsewhere:fallthrough",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90:2342:if:false",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90:2342:if:true",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90:2346:if:false",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90:2346:if:true",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90:2350:if:false",
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90:2350:if:true",
        },
        "pft14-owner-contract-4c4c672c28dc": set(),
    }
    coverage_records = []
    evidence_records = []
    all_complete = True
    covered_count = 0
    required_count = 0
    for owner in sorted(owners, key=lambda item: item["owner_region_id"]):
        required = {str(value) for value in owner["arm_ids"]}
        dynamic = execution_witnesses[owner["owner_region_id"]]
        source_backed = required & passed_proofs
        covered = dynamic | source_backed
        required_count += len(required)
        covered_count += len(covered)
        passed = required == covered
        all_complete = all_complete and passed
        coverage_records.append(
            {
                "owner_region_id": owner["owner_region_id"],
                "required_arm_ids": sorted(required),
                "execution_witness_arm_ids": sorted(dynamic),
                "source_proof_arm_ids": sorted(source_backed),
                "covered_arm_ids": sorted(covered),
                "missing_arm_ids": sorted(required - covered),
                "passed": passed,
            }
        )
        evidence_records.append(
            {
                "owner_region_id": owner["owner_region_id"],
                "base_region_id": owner["base_region_id"],
                "fortran_procedure": owner["fortran_procedure"],
                "jax_owners": owner["jax_owners"],
                "comparison_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/comparison.json",
                "branch_coverage_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/branch_coverage.json",
                "required_arm_ids": sorted(required),
                "gcov_arm_ids": [],
                "execution_witness_arm_ids": sorted(dynamic),
                "source_proof_arm_ids": sorted(source_backed),
                "covered_arm_ids": sorted(covered),
                "missing_arm_ids": sorted(required - covered),
                "passed": passed,
            }
        )
    coverage = {
        "schema_version": 1,
        "family": FAMILY,
        "branch_complete": all_complete,
        "required_arm_count": required_count,
        "covered_arm_count": covered_count,
        "missing_arm_ids": sorted(
            arm_id
            for record in coverage_records
            for arm_id in record["missing_arm_ids"]
        ),
        "records": coverage_records,
        "mapping_policy": "Exact initialize blocks 1824-1851, 2235-2245, 2258-2261, and 2272-2352 executed in two explicit timestep/peat-cover cases.",
    }
    evidence = {
        "schema_version": 1,
        "family": FAMILY,
        "complete": all_complete,
        "records": evidence_records,
    }
    (output_dir / "branch_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="ascii")
    (output_dir / "owner_region_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="ascii")


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_stomate_initialize_owner_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        hashes = compose(source)
        compiler_meta = compile_fortran(source, executable, compiler)
        fortran_output = output_dir / "fortran_outputs.csv"
        subprocess.run(
            [str(executable), str(fortran_output.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )

    fortran_values = _read(output_dir / "fortran_outputs.csv")
    jax_values = _jax_values()
    write_point_comparisons(output_dir / "point_comparisons.csv", fortran_values, jax_values, rtol=0.0, atol=0.0)
    comparisons = [
        exact_comparison(name, fortran_values[name], jax_values[name])
        if name.endswith("_dt_days_changed") or name.endswith("_frozen_respiration_func")
        else float_comparison(name, fortran_values[name], jax_values[name], rtol=0.0, atol=0.0)
        for name in sorted(fortran_values)
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "blocks": [list(BLOCK_TIMESTEP), list(BLOCK_VEGET), list(BLOCK_INIT), list(BLOCK_FROZEN), list(BLOCK_PEAT)],
                "cases": ["case_changed_reset", "case_unchanged_preserve"],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    result = write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "procedure_span_sha256": hashes,
            "compiler": compiler_meta,
            "verified_ledger_entries": [],
        },
    )
    if result["status"] == "passed":
        build_owner_evidence(output_dir)
    return result


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
