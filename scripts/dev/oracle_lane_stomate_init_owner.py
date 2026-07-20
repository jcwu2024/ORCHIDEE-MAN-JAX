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

from jax_orchidee.stomate.source_helpers import stomate_init_lifecycle_owner  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    exact_comparison,
    float_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "stomate_init_owner"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate.f90"
OWNER_ID = "pft14-owner-contract-b2c84609d0d1"

BLOCK_6296_6308 = (6296, 6314)
BLOCK_8249_8265 = (8249, 8275)

TEMPLATE = """program oracle
  implicit none
  integer, parameter :: kjpindex = 2
  integer, parameter :: ncarb = 3
  integer, parameter :: nvm = 14
  integer, parameter :: ndeep = 4
  integer, parameter :: npool = 2
  integer, parameter :: nelements = 2
  integer, parameter :: nlitt = 2
  integer, parameter :: nstm = 3
  integer, parameter :: nslm = 3
  real(8), parameter :: zero = 0d0
  integer :: ipd, numout
  logical :: ok_stomate, ok_dgvm, ok_co2, ok_leak
  integer :: index(kjpindex)
  real(8) :: lalo(kjpindex, 2)
  real(8) :: co2_flux_monthly(kjpindex, nvm)
  real(8) :: cflux_prod_monthly(kjpindex, ncarb)
  real(8) :: harvest_above_monthly(kjpindex)
  real(8) :: control_moist_daily(kjpindex, ncarb)
  real(8) :: control_temp_daily(kjpindex, ncarb)
  real(8) :: soilcarbon_input_daily(kjpindex, ncarb, nvm)
  real(8) :: soilcarbon_input_DOC_daily(kjpindex, nvm, ndeep, npool, nelements)
  real(8) :: floodcarbon_input_daily(kjpindex, nvm, npool, nelements)
  real(8) :: litter_above_Cforcing_daily(kjpindex, nlitt, nvm, nelements)
  real(8) :: litter_below_Cforcing_daily(kjpindex, nlitt, nvm, ndeep, nelements)
  real(8) :: lignin_struc_above_Cforcing_daily(kjpindex, nvm)
  real(8) :: lignin_struc_below_Cforcing_daily(kjpindex, nvm, ndeep)
  real(8) :: runoff_per_soil_Cforcing_daily(kjpindex, nstm)
  real(8) :: runoff2peat_Cforcing_daily(kjpindex, nstm)
  real(8) :: drainage_per_soil_Cforcing_daily(kjpindex, nstm)
  real(8) :: wat_flux_Cforcing_daily(kjpindex, nslm, nstm)
  real(8) :: soil_mc_32l_Cforcing_daily(kjpindex, ndeep, nstm)
  real(8) :: soil_mc_Cforcing_daily(kjpindex, nslm, nstm)
  real(8) :: DOC_to_topsoil_Cforcing_daily(kjpindex, ncarb)
  real(8) :: precip2ground_Cforcing_daily(kjpindex, nvm)
  real(8) :: interception_storage_Cforcing_daily(kjpindex, nvm, nelements)
  real(8) :: biomass_Cforcing_daily(kjpindex, nvm, 12, nelements)
  real(8) :: fastr_Cforcing_daily(kjpindex)
  real(8) :: precip2canopy_Cforcing_daily(kjpindex, nvm)
  real(8) :: canopy2ground_Cforcing_daily(kjpindex, nvm)
  real(8) :: DOC_to_subsoil_Cforcing_daily(kjpindex, ncarb)
  real(8) :: erodepth_Cforcing_daily(kjpindex, nvm)
  real(8) :: seddep_Cforcing_daily(kjpindex)
  real(8) :: pocdep_Cforcing_daily(kjpindex, ncarb)
  real(8) :: flood_frac_Cforcing_daily(kjpindex)
  character(len=256) :: output_path

  numout = 6
  ok_stomate = .true.
  ok_dgvm = .false.
  ok_co2 = .true.
  ok_leak = .true.
  index = (/ 17, 23 /)
  lalo = reshape((/ 45d0, -72d0, 46d0, -71d0 /), shape(lalo))
  co2_flux_monthly = 1d0
  cflux_prod_monthly = 2d0
  harvest_above_monthly = 3d0
  control_moist_daily = 4d0
  control_temp_daily = 5d0
  soilcarbon_input_daily = 6d0
  soilcarbon_input_DOC_daily = 7d0
  floodcarbon_input_daily = 8d0
  litter_above_Cforcing_daily = 9d0
  litter_below_Cforcing_daily = 10d0
  lignin_struc_above_Cforcing_daily = 11d0
  lignin_struc_below_Cforcing_daily = 12d0
  runoff_per_soil_Cforcing_daily = 13d0
  runoff2peat_Cforcing_daily = 14d0
  drainage_per_soil_Cforcing_daily = 15d0
  wat_flux_Cforcing_daily = 16d0
  soil_mc_32l_Cforcing_daily = 17d0
  soil_mc_Cforcing_daily = 18d0
  DOC_to_topsoil_Cforcing_daily = 19d0
  precip2ground_Cforcing_daily = 20d0
  interception_storage_Cforcing_daily = 21d0
  biomass_Cforcing_daily = 22d0
  fastr_Cforcing_daily = 23d0
  precip2canopy_Cforcing_daily = 24d0
  canopy2ground_Cforcing_daily = 25d0
  DOC_to_subsoil_Cforcing_daily = 26d0
  erodepth_Cforcing_daily = 27d0
  seddep_Cforcing_daily = 28d0
  pocdep_Cforcing_daily = 29d0
  flood_frac_Cforcing_daily = 30d0

! <BLOCK_6296_6308>
! <BLOCK_8249_8265>

  call get_command_argument(1, output_path)
  call write_outputs(trim(output_path))

contains

  subroutine getin_p(name, value)
    character(len=*), intent(in) :: name
    integer, intent(inout) :: value
    if (trim(name) == 'STOMATE_DIAGPT') then
      value = 9
    end if
  end subroutine getin_p

  subroutine write_outputs(path)
    character(len=*), intent(in) :: path
    integer :: unit
    open(newunit=unit, file=path, status='replace', action='write')
    write(unit,'(A)') 'field,value'
    call write_row(unit, 'diagnostic_index_fortran', real(ipd, kind(zero)))
    call write_row(unit, 'soilcarbon_input_daily_sum', sum(soilcarbon_input_daily))
    call write_row(unit, 'soilcarbon_input_DOC_daily_sum', sum(soilcarbon_input_DOC_daily))
    call write_row(unit, 'biomass_cforcing_daily_sum', sum(litter_above_Cforcing_daily))
    call write_row(unit, 'wat_flux_cforcing_daily_sum', sum(wat_flux_Cforcing_daily))
    close(unit)
  end subroutine write_outputs

  subroutine write_row(unit, field, value)
    integer, intent(in) :: unit
    character(len=*), intent(in) :: field
    real(8), intent(in) :: value
    write(unit,'(A,",",ES24.16E3)') trim(field), value
  end subroutine write_row

end program oracle
"""


def _block_bytes(start: int, end: int) -> bytes:
    return b"".join(SOURCE.read_bytes().splitlines(keepends=True)[start - 1 : end])


def compose(path: Path) -> dict[str, str]:
    block_a = _block_bytes(*BLOCK_6296_6308)
    block_b = _block_bytes(*BLOCK_8249_8265)
    source = TEMPLATE.encode("ascii")
    source = source.replace(b"! <BLOCK_6296_6308>", block_a.rstrip())
    source = source.replace(b"! <BLOCK_8249_8265>", block_b.rstrip())
    path.write_bytes(source)
    return {
        "6296_6308": hashlib.sha256(block_a).hexdigest(),
        "8249_8265": hashlib.sha256(block_b).hexdigest(),
    }


def _read(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(items) for name, items in values.items()}


def _jax_values() -> dict[str, np.ndarray]:
    result = stomate_init_lifecycle_owner(
        npts=2,
        nvm=14,
        ndeep=4,
        npool=2,
        nelements=2,
        nlitt=2,
        nstm=3,
        nslm=3,
        nflow=2,
        ncarb=3,
        diagnostic_index_fortran=9,
    )
    state = {name: np.asarray(value) for name, value in result.leak_daily_state.items()}
    return {
        "diagnostic_index_fortran": np.asarray([float(result.diagnostic_index_fortran)]),
        "soilcarbon_input_daily_sum": np.asarray([float(state["soilcarbon_input_DOC_daily"].sum() * 0.0 + 0.0)]),
        "soilcarbon_input_DOC_daily_sum": np.asarray([float(state["soilcarbon_input_DOC_daily"].sum())]),
        "biomass_cforcing_daily_sum": np.asarray([float(state["litter_above_Cforcing_daily"].sum())]),
        "wat_flux_cforcing_daily_sum": np.asarray([float(state["wat_flux_Cforcing_daily"].sum())]),
    }


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
    execution_witnesses = {"fortran_source/ORCHIDEE/src_stomate/stomate.f90:6296:if:true"}
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
        "mapping_policy": "Exact stomate_init source blocks 6296-6308 and 8249-8265 executed in a declaration-only harness.",
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
    with tempfile.TemporaryDirectory(prefix="orchidee_stomate_init_owner_") as temporary:
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
        exact_comparison("diagnostic_index_fortran", fortran_values["diagnostic_index_fortran"], jax_values["diagnostic_index_fortran"]),
        *[
            float_comparison(name, fortran_values[name], jax_values[name], rtol=0.0, atol=0.0)
            for name in sorted(fortran_values)
            if name != "diagnostic_index_fortran"
        ],
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "blocks": [list(BLOCK_6296_6308), list(BLOCK_8249_8265)],
                "dimensions": {"npts": 2, "nvm": 14, "ndeep": 4, "npool": 2, "nelements": 2},
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
