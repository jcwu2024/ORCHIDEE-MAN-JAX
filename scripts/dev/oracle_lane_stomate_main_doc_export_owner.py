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

from jax_orchidee.stomate.carbon_kernels import IROOT  # noqa: E402
from jax_orchidee.stomate.soilcarbon_kernels import (  # noqa: E402
    soilcarbon_leak_doc_export_aggregate,
)
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "stomate_main_doc_export_owner"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate.f90"
OWNER_ID = "pft14-owner-contract-3b3635ab31d5"
BLOCK = (3419, 3458)

TEMPLATE = """program oracle
  implicit none
  integer, parameter :: kjpindex = 2
  integer, parameter :: nvm = 4
  integer, parameter :: nexp = 3
  integer, parameter :: iact = 5
  integer, parameter :: islo = 6
  integer, parameter :: ipas = 7
  integer, parameter :: nelements = 1
  integer, parameter :: icarbon = 1
  integer, parameter :: idocl = 2
  integer, parameter :: idocr = 3
  integer, parameter :: iCO2aq = 4
  integer, parameter :: irunoff = 1
  integer, parameter :: idrainage = 2
  integer, parameter :: iflooded = 3
  integer, parameter :: ibelow = 2
  integer, parameter :: iroot = 6
  real(8), parameter :: zero = 0.0
  real(8), parameter :: un = 1.0
  real(8), parameter :: min_sechiba = 1.0e-8
  real(8), parameter :: one_day = 86400.0
  real(8), parameter :: dt_sechiba = 43200.0
  integer :: k, m, l, i, unit
  integer :: pref_soil_veg(nvm)
  real(8) :: soil_resp_modif
  real(8) :: veget_max(kjpindex, nvm)
  real(8) :: flood_frac(kjpindex)
  real(8) :: resp_hetero_litter(kjpindex, nvm, 2)
  real(8) :: resp_hetero_soil(kjpindex, nvm)
  real(8) :: resp_hetero_flood(kjpindex, nvm)
  real(8) :: resp_flood_soil(kjpindex, nvm)
  real(8) :: resp_maint_part_radia(kjpindex, nvm, 6)
  real(8) :: flood_root_radia(kjpindex, nvm)
  real(8) :: runoff_per_soil(kjpindex, 2)
  real(8) :: drainage_per_soil(kjpindex, 2)
  real(8) :: DOC_EXP(kjpindex, nvm, nexp, ipas, nelements)
  real(8) :: DOC_EXP_agg(kjpindex, nexp, iCO2aq)
  real(8) :: DOC_EXP_b(kjpindex, nvm, nexp, iCO2aq, nelements)
  character(len=256) :: output_path

  pref_soil_veg = (/ 1, 1, 2, 2 /)
  veget_max = reshape((/ &
    0.0, 0.0, 0.5, 0.4, &
    0.25, 0.3, 0.75, 0.2 /), shape(veget_max), order=(/1,2/))
  flood_frac = (/ 0.2, 1.0 /)
  resp_hetero_litter = 0.0
  resp_hetero_soil = 0.0
  resp_hetero_flood = 0.0
  resp_flood_soil = 0.0
  resp_maint_part_radia = 0.0
  flood_root_radia = 0.0
  runoff_per_soil = reshape((/ 10.0, 30.0, 20.0, 40.0 /), shape(runoff_per_soil), order=(/1,2/))
  drainage_per_soil = reshape((/ 5.0, 7.0, 6.0, 8.0 /), shape(drainage_per_soil), order=(/1,2/))
  DOC_EXP = 0.0
  DOC_EXP(1,2,1,iact,1) = 1.0
  DOC_EXP(1,2,1,islo,1) = 2.0
  DOC_EXP(1,2,1,ipas,1) = 3.0
  DOC_EXP(1,3,2,iact,1) = 4.0
  DOC_EXP(1,3,2,islo,1) = 5.0
  DOC_EXP(1,3,2,ipas,1) = 6.0
  DOC_EXP(2,2,1,iact,1) = 7.0
  DOC_EXP(2,2,1,islo,1) = 8.0
  DOC_EXP(2,2,1,ipas,1) = 9.0
  resp_hetero_litter(:,2,ibelow) = (/ 4.0, 5.0 /)
  resp_hetero_litter(:,3,ibelow) = (/ 2.0, 1.0 /)
  resp_hetero_soil(:,2) = (/ 2.0, 3.0 /)
  resp_hetero_soil(:,3) = (/ 1.0, 4.0 /)
  resp_maint_part_radia(:,2,iroot) = (/ 1.0, 2.0 /)
  resp_maint_part_radia(:,3,iroot) = (/ 2.0, 1.0 /)
  resp_hetero_flood(:,2) = (/ 0.25, 0.5 /)
  resp_hetero_flood(:,3) = (/ 0.75, 0.25 /)
  resp_flood_soil(:,2) = (/ 0.1, 0.2 /)
  resp_flood_soil(:,3) = (/ 0.3, 0.4 /)
  flood_root_radia(:,2) = (/ 0.5, 0.6 /)
  flood_root_radia(:,3) = (/ 0.7, 0.8 /)

! <BLOCK>

  call get_command_argument(1, output_path)
  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit,'(A)') 'field,value'
  call emit_tensor(unit, 'doc_exp_agg', DOC_EXP_agg)
  call emit_scalar(unit, 'doc_exp_b_1_2_1_idocl', DOC_EXP_b(1,2,1,idocl,icarbon))
  call emit_scalar(unit, 'doc_exp_b_1_2_1_idocr', DOC_EXP_b(1,2,1,idocr,icarbon))
  call emit_scalar(unit, 'doc_exp_b_2_2_1_idocl', DOC_EXP_b(2,2,1,idocl,icarbon))
  call emit_scalar(unit, 'doc_exp_b_2_2_1_idocr', DOC_EXP_b(2,2,1,idocr,icarbon))
  close(unit)

contains

  subroutine emit_tensor(unit, prefix, values)
    integer, intent(in) :: unit
    character(len=*), intent(in) :: prefix
    real(8), intent(in) :: values(:,:,:)
    integer :: i0, i1, i2, index
    index = 0
    do i0 = 1, size(values,1)
      do i1 = 1, size(values,2)
        do i2 = 1, size(values,3)
          write(unit,'(A,",",ES24.16E3)') trim(prefix)//'_'//trim(adjustl(itoa(index))), values(i0,i1,i2)
          index = index + 1
        end do
      end do
    end do
  end subroutine emit_tensor

  subroutine emit_scalar(unit, field, value)
    integer, intent(in) :: unit
    character(len=*), intent(in) :: field
    real(8), intent(in) :: value
    write(unit,'(A,",",ES24.16E3)') trim(field), value
  end subroutine emit_scalar

  function itoa(value) result(text)
    integer, intent(in) :: value
    character(len=16) :: text
    write(text,'(I0)') value
  end function itoa

  logical function isnan(value)
    use, intrinsic :: ieee_arithmetic
    real(8), intent(in) :: value
    isnan = ieee_is_nan(value)
  end function isnan

end program oracle
"""


def _block_bytes() -> bytes:
    return b"".join(SOURCE.read_bytes().splitlines(keepends=True)[BLOCK[0] - 1 : BLOCK[1]])


def compose(path: Path) -> dict[str, str]:
    block = _block_bytes()
    path.write_bytes(TEMPLATE.encode("ascii").replace(b"! <BLOCK>", block.rstrip()))
    return {"3419_3458": hashlib.sha256(block).hexdigest()}


def _read(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(items) for name, items in values.items()}


def _jax_values() -> dict[str, np.ndarray]:
    doc_exp = np.zeros((2, 4, 3, 7, 1), dtype=np.float64)
    doc_exp[0, 1, 0, 4, 0] = 1.0
    doc_exp[0, 1, 0, 5, 0] = 2.0
    doc_exp[0, 1, 0, 6, 0] = 3.0
    doc_exp[0, 2, 1, 4, 0] = 4.0
    doc_exp[0, 2, 1, 5, 0] = 5.0
    doc_exp[0, 2, 1, 6, 0] = 6.0
    doc_exp[1, 1, 0, 4, 0] = 7.0
    doc_exp[1, 1, 0, 5, 0] = 8.0
    doc_exp[1, 1, 0, 6, 0] = 9.0
    result = soilcarbon_leak_doc_export_aggregate(
        doc_exp,
        np.asarray([[0.0, 0.5, 0.25, 0.75], [0.0, 0.4, 0.3, 0.2]], dtype=np.float64),
        np.asarray(
            [
                [[0.0, 0.0], [0.0, 4.0], [0.0, 2.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 5.0], [0.0, 1.0], [0.0, 0.0]],
            ],
            dtype=np.float64,
        ),
        np.asarray([[0.0, 2.0, 1.0, 0.0], [0.0, 3.0, 4.0, 0.0]], dtype=np.float64),
        np.asarray([[0.0, 0.25, 0.75, 0.0], [0.0, 0.5, 0.25, 0.0]], dtype=np.float64),
        np.asarray([[0.0, 0.1, 0.3, 0.0], [0.0, 0.2, 0.4, 0.0]], dtype=np.float64),
        np.pad(
            np.asarray(
                [
                    [[0.0], [1.0], [2.0], [0.0]],
                    [[0.0], [2.0], [1.0], [0.0]],
                ],
                dtype=np.float64,
            ),
            ((0, 0), (0, 0), (IROOT, 0)),
        ),
        np.asarray([[0.0, 0.5, 0.7, 0.0], [0.0, 0.6, 0.8, 0.0]], dtype=np.float64),
        np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64),
        np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64),
        np.asarray([0, 0, 1, 1], dtype=np.int32),
        np.asarray([0.2, 1.0], dtype=np.float64),
        dt_days=0.5,
    )
    flat = np.asarray(result.doc_exp_agg).reshape(-1)
    values = {f"doc_exp_agg_{index}": np.asarray([value]) for index, value in enumerate(flat)}
    values.update(
        {
            "doc_exp_b_1_2_1_idocl": np.asarray([float(np.asarray(result.doc_exp_b)[0, 1, 0, 1, 0])]),
            "doc_exp_b_1_2_1_idocr": np.asarray([float(np.asarray(result.doc_exp_b)[0, 1, 0, 2, 0])]),
            "doc_exp_b_2_2_1_idocl": np.asarray([float(np.asarray(result.doc_exp_b)[1, 1, 0, 1, 0])]),
            "doc_exp_b_2_2_1_idocr": np.asarray([float(np.asarray(result.doc_exp_b)[1, 1, 0, 2, 0])]),
        }
    )
    return values


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
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90:3425:if:true",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90:3433:if:true",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90:3442:if:false",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90:3442:if:true",
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
        "mapping_policy": "Exact stomate_main DOC export block 3419-3458 executed with one dry-fraction true point and one fully flooded false point.",
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
                "gcov_arm_ids": [],
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
    with tempfile.TemporaryDirectory(prefix="orchidee_stomate_main_doc_export_owner_") as temporary:
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
    write_point_comparisons(
        output_dir / "point_comparisons.csv", fortran_values, jax_values,
        rtol=1e-12, atol=1e-14,
    )
    comparisons = [
        float_comparison(
            name, fortran_values[name], jax_values[name], rtol=1e-12, atol=1e-14
        )
        for name in sorted(fortran_values)
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps({"schema_version": 1, "block": list(BLOCK)}, indent=2) + "\n",
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
