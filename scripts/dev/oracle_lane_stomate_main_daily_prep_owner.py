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

from jax_orchidee.stomate.daily import stomate_instant_daily_prep  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    exact_comparison,
    float_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "stomate_main_daily_prep_owner"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate.f90"
OWNER_ID = "pft14-owner-contract-7df013fb2bd0"
BLOCK = (3128, 3168)

TEMPLATE = """program oracle
  implicit none
  integer, parameter :: kjpindex = 2
  integer, parameter :: nvm = 14
  integer :: day, pixid, jv
  real(8) :: sec, dt_sechiba
  real(8) :: t2m(kjpindex), swdown(kjpindex), st2m(kjpindex)
  real(8) :: ugdh(kjpindex, nvm), uphoi(kjpindex)
  real(8) :: SP_tdmin(nvm), SP_tdmax(nvm)
  logical :: EndOfMonth, ok_LAIdev(nvm)
  character(len=256) :: output_path

  call get_command_argument(1, output_path)
  open(unit=10, file=trim(output_path), status='replace', action='write')
  write(10,'(A)') 'field,value'

  call run_case('case_eom_true', 1, 0.0, (/ 273.15, 278.15 /), (/ -1.0, 10.0 /))
  call run_case('case_eom_false', 2, 1800.0, (/ 275.15, 279.15 /), (/ 5.0, 0.0 /))
  close(10)

contains

  subroutine run_case(prefix, case_day, case_sec, case_t2m, case_swdown)
    character(len=*), intent(in) :: prefix
    integer, intent(in) :: case_day
    real(8), intent(in) :: case_sec
    real(8), intent(in) :: case_t2m(kjpindex), case_swdown(kjpindex)

    dt_sechiba = 1800.0
    day = case_day
    sec = case_sec
    t2m = case_t2m
    swdown = case_swdown
    SP_tdmin = 0.0
    SP_tdmax = 20.0
    ok_LAIdev = .false.
    ok_LAIdev(3) = .true.
    ugdh = 0.0
    uphoi = 0.0
    EndOfMonth = .false.

! <BLOCK>

    call write_row(prefix//'_end_of_month', merge(1.0_8, 0.0_8, EndOfMonth))
    call write_row(prefix//'_st2m_1', st2m(1))
    call write_row(prefix//'_st2m_2', st2m(2))
    call write_row(prefix//'_uphoi_1', uphoi(1))
    call write_row(prefix//'_uphoi_2', uphoi(2))
    call write_row(prefix//'_ugdh_1_3', ugdh(1,3))
    call write_row(prefix//'_ugdh_2_3', ugdh(2,3))
  end subroutine run_case

  subroutine write_row(field, value)
    character(len=*), intent(in) :: field
    real(8), intent(in) :: value
    write(10,'(A,",",ES24.16E3)') trim(field), value
  end subroutine write_row

end program oracle
"""


def _block_bytes() -> bytes:
    return b"".join(SOURCE.read_bytes().splitlines(keepends=True)[BLOCK[0] - 1 : BLOCK[1]])


def compose(path: Path) -> dict[str, str]:
    block = _block_bytes()
    path.write_bytes(TEMPLATE.encode("ascii").replace(b"! <BLOCK>", block.rstrip()))
    return {"3128_3168": hashlib.sha256(block).hexdigest()}


def _read(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(items) for name, items in values.items()}


def _jax_case(day: int, sec: float, t2m: np.ndarray, swdown: np.ndarray) -> dict[str, np.ndarray]:
    result = stomate_instant_daily_prep(
        day=day,
        sec=sec,
        dt_sechiba=1800.0,
        t2m=t2m,
        ok_LAIdev=np.asarray([False, False, True] + [False] * 11),
        SP_tdmin=np.zeros(14, dtype=np.float64),
        SP_tdmax=np.full(14, 20.0, dtype=np.float64),
        swdown=swdown,
    )
    return {
        "end_of_month": np.asarray([1.0 if result.end_of_month else 0.0]),
        "st2m_1": np.asarray([float(np.asarray(result.st2m)[0])]),
        "st2m_2": np.asarray([float(np.asarray(result.st2m)[1])]),
        "uphoi_1": np.asarray([float(np.asarray(result.uphoi)[0])]),
        "uphoi_2": np.asarray([float(np.asarray(result.uphoi)[1])]),
        "ugdh_1_3": np.asarray([float(np.asarray(result.ugdh)[0, 2])]),
        "ugdh_2_3": np.asarray([float(np.asarray(result.ugdh)[1, 2])]),
    }


def _jax_values() -> dict[str, np.ndarray]:
    cases = {
        "case_eom_true": _jax_case(1, 0.0, np.asarray([273.15, 278.15]), np.asarray([-1.0, 10.0])),
        "case_eom_false": _jax_case(2, 1800.0, np.asarray([275.15, 279.15]), np.asarray([5.0, 0.0])),
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
    owner = next(record for record in contracts["owner_regions"] if record["owner_region_id"] == OWNER_ID)
    required = {str(value) for value in owner["arm_ids"]}
    execution_witnesses = {
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90:3130:if:false",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90:3130:if:true",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90:3163:if:false",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90:3163:if:true",
    }
    source_backed = required & passed_proofs
    covered = execution_witnesses | source_backed
    coverage = {
        "schema_version": 1,
        "family": FAMILY,
        "branch_complete": required == covered,
        "required_arm_count": len(required),
        "covered_arm_count": len(covered),
        "missing_arm_ids": sorted(required - covered),
        "execution_witness_arm_ids": sorted(execution_witnesses),
        "source_proof_arm_ids": sorted(source_backed),
        "mapping_policy": "Exact stomate_main source block 3128-3168 executed in two explicit calendar/radiation cases.",
    }
    evidence = {
        "schema_version": 1,
        "family": FAMILY,
        "complete": required == covered,
        "records": [
            {
                "owner_region_id": owner["owner_region_id"],
                "base_region_id": owner["base_region_id"],
                "fortran_procedure": owner["fortran_procedure"],
                "jax_owners": owner["jax_owners"],
                "comparison_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/comparison.json",
                "branch_coverage_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/branch_coverage.json",
                "required_arm_ids": sorted(required),
                "execution_witness_arm_ids": sorted(execution_witnesses),
                "source_proof_arm_ids": sorted(source_backed),
                "covered_arm_ids": sorted(covered),
                "missing_arm_ids": sorted(required - covered),
                "passed": required == covered,
            }
        ],
    }
    (output_dir / "branch_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="ascii")
    (output_dir / "owner_region_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="ascii")


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_stomate_main_daily_prep_") as temporary:
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
        if name.endswith("end_of_month")
        else float_comparison(name, fortran_values[name], jax_values[name], rtol=0.0, atol=0.0)
        for name in sorted(fortran_values)
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps({"schema_version": 1, "block": list(BLOCK), "cases": ["case_eom_true", "case_eom_false"]}, indent=2)
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
