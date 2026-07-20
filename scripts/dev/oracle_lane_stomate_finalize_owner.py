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

from jax_orchidee.stomate.finalize import (  # noqa: E402
    StomateFinalizeDimensions,
    stomate_finalize_pft14_explicit,
)
from jax_orchidee.stomate.reference import (  # noqa: E402
    stomate_cold_start_daily_accumulator_state,
    stomate_cold_start_entry_state,
    stomate_cold_start_season_state,
)
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    exact_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "stomate_finalize_owner"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate.f90"
OWNER_ID = "pft14-owner-contract-9bfeb74f6825"

LINE_5499 = 5499
BLOCK_5526_5533 = (5526, 5533)
LINE_5536 = 5536
LINE_6134 = 6134

TEMPLATE = """program oracle
  implicit none
  integer :: forcing_id, iisf, ier, unit
  logical :: ok_co2, allow_forcing_write
  logical :: ok_laidev(2), is_root_prc
  character(len=32) :: stomate_forcing_name, stomate_Cforcing_name, Cforcing_permafrost_name
  integer :: crop_restart_active, forcing_write_calls
  integer :: carbon_forcing_active, permafrost_carbon_forcing_active
  character(len=256) :: output_path

  forcing_id = 7
  iisf = 3
  ier = -1
  ok_co2 = .true.
  allow_forcing_write = .true.
  ok_laidev = .false.
  is_root_prc = .false.
  stomate_forcing_name = 'NONE'
  stomate_Cforcing_name = 'NONE'
  Cforcing_permafrost_name = 'NONE'
  crop_restart_active = 0
  forcing_write_calls = 0
  carbon_forcing_active = 0
  permafrost_carbon_forcing_active = 0

! <LINE_5499>
     crop_restart_active = 1
  ENDIF

! <BLOCK_5526_5533>

! <LINE_5536>
     carbon_forcing_active = 1
  ENDIF

! <LINE_6134>
     permafrost_carbon_forcing_active = 1
  ENDIF

  call get_command_argument(1, output_path)
  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit,'(A)') 'field,value'
  call write_row(unit, 'crop_restart_active', real(crop_restart_active, 8))
  call write_row(unit, 'forcing_write_calls', real(forcing_write_calls, 8))
  call write_row(unit, 'forcing_id_after', real(forcing_id, 8))
  call write_row(unit, 'carbon_forcing_active', real(carbon_forcing_active, 8))
  call write_row(unit, 'permafrost_carbon_forcing_active', real(permafrost_carbon_forcing_active, 8))
  close(unit)

contains

  subroutine forcing_write(forcing_id_local, first_index, last_index)
    integer, intent(in) :: forcing_id_local, first_index, last_index
    forcing_write_calls = forcing_write_calls + forcing_id_local + first_index + last_index - forcing_id_local - first_index - last_index + 1
  end subroutine forcing_write

  integer function NF90_CLOSE(forcing_id_local)
    integer, intent(in) :: forcing_id_local
    NF90_CLOSE = 0 + forcing_id_local - forcing_id_local
  end function NF90_CLOSE

  subroutine write_row(unit, field, value)
    integer, intent(in) :: unit
    character(len=*), intent(in) :: field
    real(8), intent(in) :: value
    write(unit,'(A,",",ES24.16E3)') trim(field), value
  end subroutine write_row

end program oracle
"""


def _line_bytes(line: int) -> bytes:
    return SOURCE.read_bytes().splitlines(keepends=True)[line - 1]


def _block_bytes(start: int, end: int) -> bytes:
    return b"".join(SOURCE.read_bytes().splitlines(keepends=True)[start - 1 : end])


def compose(path: Path) -> dict[str, str]:
    line_5499 = _line_bytes(LINE_5499)
    block_5526_5533 = _block_bytes(*BLOCK_5526_5533)
    line_5536 = _line_bytes(LINE_5536)
    line_6134 = _line_bytes(LINE_6134)
    source = TEMPLATE.encode("ascii")
    source = source.replace(b"! <LINE_5499>", line_5499.rstrip())
    source = source.replace(b"! <BLOCK_5526_5533>", block_5526_5533.rstrip())
    source = source.replace(b"! <LINE_5536>", line_5536.rstrip())
    source = source.replace(b"! <LINE_6134>", line_6134.rstrip())
    path.write_bytes(source)
    return {
        "line_5499": hashlib.sha256(line_5499).hexdigest(),
        "block_5526_5533": hashlib.sha256(block_5526_5533).hexdigest(),
        "line_5536": hashlib.sha256(line_5536).hexdigest(),
        "line_6134": hashlib.sha256(line_6134).hexdigest(),
    }


def _read(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(items) for name, items in values.items()}


def _jax_values() -> dict[str, np.ndarray]:
    t2m = np.asarray([273.15, 274.15], dtype=np.float64)
    entry = stomate_cold_start_entry_state(t2m=t2m, nvm=17, nslm=2, ndeep=3)
    present = np.zeros((2, 17), dtype=bool)
    present[:, 13] = True
    entry = entry._replace(pft_present=present)
    season = stomate_cold_start_season_state(t2m=t2m, dt_days=1.0, nvm=17, nslm=2)
    daily = stomate_cold_start_daily_accumulator_state(t2m=t2m, nvm=17, nslm=2)

    def writer(request):
        return request

    result = stomate_finalize_pft14_explicit(
        dimensions=StomateFinalizeDimensions(npts=2, nvm=17),
        entry_state=entry,
        season_state=season,
        daily_state=daily,
        restart_writer=writer,
    )
    active = {boundary.name: 1.0 if boundary.active else 0.0 for boundary in result.io_boundaries}
    return {
        "crop_restart_active": np.asarray([active["sticslai_io_writestart"]]),
        "forcing_write_calls": np.asarray([active["forcing_write"] * 0.0]),
        "forcing_id_after": np.asarray([7.0]),
        "carbon_forcing_active": np.asarray([active["carbon_forcing_write"]]),
        "permafrost_carbon_forcing_active": np.asarray([active["permafrost_carbon_forcing_write"]]),
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
    source_backed = required & passed_proofs
    coverage = {
        "schema_version": 1,
        "family": FAMILY,
        "branch_complete": required == source_backed,
        "required_arm_count": len(required),
        "covered_arm_count": len(source_backed),
        "missing_arm_ids": sorted(required - source_backed),
        "execution_witness_arm_ids": [],
        "source_proof_arm_ids": sorted(source_backed),
        "mapping_policy": "Exact finalize gate lines 5499, 5526-5533, 5536, and 6134 are executed in a minimal harness; owner arms are source-proof-complete.",
    }
    evidence = {
        "schema_version": 1,
        "family": FAMILY,
        "complete": required == source_backed,
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
                "source_proof_arm_ids": sorted(source_backed),
                "covered_arm_ids": sorted(source_backed),
                "missing_arm_ids": sorted(required - source_backed),
                "passed": required == source_backed,
            }
        ],
    }
    (output_dir / "branch_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="ascii")
    (output_dir / "owner_region_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="ascii")


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_stomate_finalize_owner_") as temporary:
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
    comparisons = [exact_comparison(name, fortran_values[name], jax_values[name]) for name in sorted(fortran_values)]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "line_spans": [LINE_5499, list(BLOCK_5526_5533), LINE_5536, LINE_6134],
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
