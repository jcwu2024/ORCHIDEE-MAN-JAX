from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from jax import config as jax_config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

jax_config.update("jax_enable_x64", True)

from jax_orchidee.sticslai_completion import (  # noqa: E402
    STICS_FALSE_FIELDS,
    STICS_ONE_REAL_FIELDS,
    STICS_TRUE_FIELDS,
    STICS_ZERO_INTEGER_FIELDS,
    STICS_ZERO_REAL_FIELDS,
    SticsInitPftParameters,
    stics_init_source_routed,
)
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.audit_pft14_owner_region_evidence import (  # noqa: E402
    build_report as build_central_audit_report,
    load_evidence_documents,
)
from scripts.dev.fortran_gcov import (  # noqa: E402
    locate_generated_span,
    parse_gcov,
    parse_gcov_line_counts,
    select_arm_branch,
)
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    exact_comparison,
    float_comparison,
    write_result,
)

FAMILY = "stics_init_batch_d"
OWNER_REGION_ID = "pft14-owner-contract-5499b4b53358"
BASE_REGION_ID = "pft14-region-6fd8f0049374"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sticslai/Stics_init.f90"
CONTRACTS = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
CENTRAL_SOURCE_PROOFS = ROOT / "outputs/reference_mode/pft14_source_proof_evidence.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
GCOV = Path(r"C:\msys64\ucrt64\bin\gcov.exe")

NPTS = 2
NVM = 14
NBOXMAX = 5
RTOL = 1e-12
ATOL = 1e-14

PARAMETER_FIELDS = (
    "stpltger",
    "R_stamflax",
    "R_stlaxsen",
    "R_stsenlan",
    "R_stlevdrp",
    "R_stflodrp",
    "R_stdrpmat",
    "R_stdrpdes",
    "R_stlevamf",
)
REAL_2D_FIELDS = (
    *STICS_ZERO_REAL_FIELDS,
    *STICS_ONE_REAL_FIELDS,
    *PARAMETER_FIELDS,
    "stlevflo",
)
INTEGER_2D_FIELDS = (
    *STICS_ZERO_INTEGER_FIELDS,
    "nrecbutoir",
    "hist_sencourset",
    "hist_latestset",
    "doyhiststset",
)
LOGICAL_2D_FIELDS = (*STICS_FALSE_FIELDS, *STICS_TRUE_FIELDS)
BOX_REAL_FIELDS = (
    "box_lai",
    "box_lairem",
    "box_tdev",
    "box_biom",
    "box_biomrem",
    "box_durage",
    "box_somsenbase",
)
STATE_FIELDS = (
    *REAL_2D_FIELDS,
    *INTEGER_2D_FIELDS,
    *LOGICAL_2D_FIELDS,
    "v_dltams",
    "histgrowthset",
    "box_ndays",
    *BOX_REAL_FIELDS,
    "box_ulai",
)
CASES = (
    {
        "id": 1,
        "name": "cold_global_pft14_and_bare",
        "f_crop_init": True,
        "recycle": (),
        "witness": "cold call initializes every point/PFT, including bare soil and PFT14",
    },
    {
        "id": 2,
        "name": "later_pft14_recycle_mask",
        "f_crop_init": False,
        "recycle": ((0, 13),),
        "witness": "later call initializes only active PFT14 point 1",
    },
    {
        "id": 3,
        "name": "later_bare_recycle_mask",
        "f_crop_init": False,
        "recycle": ((1, 0),),
        "witness": "later call initializes only active bare-soil point 2",
    },
    {
        "id": 4,
        "name": "later_inactive_mask_noop",
        "f_crop_init": False,
        "recycle": (),
        "witness": "later call with an empty recycle mask preserves all state",
    },
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="ascii")


def _argument_names(span: bytes) -> list[str]:
    header: list[str] = []
    depth = 0
    started = False
    for raw_line in span.decode("ascii").splitlines():
        code = raw_line.split("!", 1)[0]
        if not started and re.search(r"\bsubroutine\s+stics_init\b", code, re.I):
            started = True
        if not started:
            continue
        header.append(code.replace("&", " "))
        depth += code.count("(") - code.count(")")
        if depth == 0:
            break
    match = re.search(r"\bsubroutine\s+stics_init\s*\((.*)\)", " ".join(header), re.I)
    if match is None:
        raise RuntimeError("could not parse Stics_init argument list")
    return [item.strip() for item in match.group(1).split(",") if item.strip()]


def _minimal_dependencies() -> bytes:
    return (
        "module pft_parameters\n"
        "  implicit none\n"
        "  integer, parameter :: nvm = 14\n"
        "  logical :: ok_LAIdev(nvm)\n"
        "  integer :: SP_codlainet(nvm), SP_nbox(nvm)\n"
        "  real :: SP_stpltger(nvm), SP_stamflax(nvm), SP_stlaxsen(nvm)\n"
        "  real :: SP_stsenlan(nvm), SP_stlevdrp(nvm), SP_stflodrp(nvm)\n"
        "  real :: SP_stdrpmat(nvm), SP_stdrpdes(nvm), SP_stlevamf(nvm)\n"
        "  real :: SP_vlaimax(nvm)\n"
        "end module pft_parameters\n\n"
        "module constantes\n"
        "  implicit none\n"
        "end module constantes\n\n"
    ).encode("ascii")


def _fortran_driver(arguments: list[str]) -> bytes:
    expected = {
        "kjpindex",
        "nboxmax",
        "f_crop_init",
        "f_crop_recycle",
        *STATE_FIELDS,
    }
    actual = {name.lower() for name in arguments}
    if {name.lower() for name in expected} != actual:
        missing = sorted({name.lower() for name in expected} - actual)
        extra = sorted(actual - {name.lower() for name in expected})
        raise RuntimeError(f"Stics_init argument contract drift: missing={missing}, extra={extra}")

    declarations = [
        "  logical :: f_crop_init",
        "  logical :: f_crop_recycle(kjpindex,nvm)",
        *(f"  real :: {name}(kjpindex,nvm)" for name in REAL_2D_FIELDS),
        *(f"  integer :: {name}(kjpindex,nvm)" for name in INTEGER_2D_FIELDS),
        *(f"  logical :: {name}(kjpindex,nvm)" for name in LOGICAL_2D_FIELDS),
        "  real :: v_dltams(kjpindex,nvm,60)",
        "  real :: histgrowthset(kjpindex,nvm,300,5)",
        "  integer :: box_ndays(kjpindex,nvm,nboxmax)",
        *(f"  real :: {name}(kjpindex,nvm,nboxmax)" for name in BOX_REAL_FIELDS),
        "  real :: box_ulai(nvm,nboxmax)",
    ]
    initializers = [
        *(f"    {name} = real(case_id * 10000 + {index})" for index, name in enumerate(REAL_2D_FIELDS, 1)),
        *(f"    {name} = case_id * 10000 + {index}" for index, name in enumerate(INTEGER_2D_FIELDS, 201)),
        *(f"    {name} = mod(case_id + {index}, 2) == 0" for index, name in enumerate(LOGICAL_2D_FIELDS, 301)),
        "    v_dltams = real(case_id * 10000 + 401)",
        "    histgrowthset = real(case_id * 10000 + 402)",
        "    box_ndays = case_id * 10000 + 403",
        *(f"    {name} = real(case_id * 10000 + {index})" for index, name in enumerate(BOX_REAL_FIELDS, 404)),
        "    box_ulai = real(case_id * 10000 + 411)",
    ]
    writes = [
        *(f"    write(unit) {name}" for name in STATE_FIELDS),
        "    write(unit) f_crop_init",
        "    write(unit) f_crop_recycle",
    ]
    call = "  call Stics_init( &\n" + ", &\n".join(f"    {name}" for name in arguments) + ")"
    driver = f"""
program stics_init_owner_oracle
  use pft_parameters
  implicit none
  integer, parameter :: kjpindex = {NPTS}, nboxmax = {NBOXMAX}
  integer :: case_id, j, unit
  character(len=1024) :: output_dir, output_path
{chr(10).join(declarations)}

  call get_command_argument(1, output_dir)
  if (len_trim(output_dir) == 0) error stop 'missing output directory'
  do case_id = 1, 4
    call initialize_parameters()
    call initialize_state(case_id)
    f_crop_init = case_id == 1
    f_crop_recycle = .false.
    if (case_id == 2) f_crop_recycle(1,14) = .true.
    if (case_id == 3) f_crop_recycle(2,1) = .true.
{call}
    write(output_path, '(A,"/case_",I1,".bin")') trim(output_dir), case_id
    open(newunit=unit, file=trim(output_path), access='stream', form='unformatted', &
         status='replace', action='write')
    call write_state(unit)
    close(unit)
  end do

contains

  subroutine initialize_parameters()
    ok_LAIdev = .false.
    SP_codlainet = 1
    SP_nbox = 0
    do j = 1, nvm
      SP_stpltger(j) = 10.0 + real(j) / 16.0
      SP_stamflax(j) = 20.0 + real(j) / 16.0
      SP_stlaxsen(j) = 30.0 + real(j) / 16.0
      SP_stsenlan(j) = 40.0 + real(j) / 16.0
      SP_stlevdrp(j) = 50.0 + real(j) / 16.0
      SP_stflodrp(j) = 5.0 + real(j) / 32.0
      SP_stdrpmat(j) = 60.0 + real(j) / 16.0
      SP_stdrpdes(j) = 70.0 + real(j) / 16.0
      SP_stlevamf(j) = 80.0 + real(j) / 16.0
      SP_vlaimax(j) = -9999.0
    end do
  end subroutine initialize_parameters

  subroutine initialize_state(case_id)
    integer, intent(in) :: case_id
{chr(10).join(initializers)}
  end subroutine initialize_state

  subroutine write_state(unit)
    integer, intent(in) :: unit
{chr(10).join(writes)}
  end subroutine write_state
end program stics_init_owner_oracle
"""
    return driver.lstrip().encode("ascii")


def compose(path: Path) -> dict[str, Any]:
    extracted = extract_procedure_bytes(SOURCE, "stics_init")
    dependencies = _minimal_dependencies()
    driver = _fortran_driver(_argument_names(extracted.span_bytes))
    payload = dependencies + extracted.span_bytes + b"\n" + driver
    path.write_bytes(payload)
    return {
        "source_sha256": extracted.source_sha256,
        "span_sha256": extracted.span_sha256,
        "start_line": extracted.start_line,
        "end_line": extracted.end_line,
        "dependency_sha256": _sha256(dependencies),
        "driver_sha256": _sha256(driver),
        "compile_unit_sha256": _sha256(payload),
    }


def _parameters() -> SticsInitPftParameters:
    j = np.arange(1, NVM + 1, dtype=np.float64)
    return SticsInitPftParameters(
        ok_laidev=np.zeros(NVM, dtype=bool),
        sp_codlainet=np.ones(NVM, dtype=np.int32),
        sp_stpltger=10.0 + j / 16.0,
        sp_stamflax=20.0 + j / 16.0,
        sp_stlaxsen=30.0 + j / 16.0,
        sp_stsenlan=40.0 + j / 16.0,
        sp_stlevdrp=50.0 + j / 16.0,
        sp_stflodrp=5.0 + j / 32.0,
        sp_stdrpmat=60.0 + j / 16.0,
        sp_stdrpdes=70.0 + j / 16.0,
        sp_stlevamf=80.0 + j / 16.0,
        sp_nbox=np.zeros(NVM, dtype=np.int32),
        sp_vlaimax=np.full(NVM, -9999.0),
    )


def _initial_state(case_id: int) -> dict[str, np.ndarray]:
    state: dict[str, np.ndarray] = {}
    for index, name in enumerate(REAL_2D_FIELDS, 1):
        state[name] = np.full((NPTS, NVM), case_id * 10000 + index, dtype=np.float64)
    for index, name in enumerate(INTEGER_2D_FIELDS, 201):
        state[name] = np.full((NPTS, NVM), case_id * 10000 + index, dtype=np.int32)
    for index, name in enumerate(LOGICAL_2D_FIELDS, 301):
        state[name] = np.full((NPTS, NVM), (case_id + index) % 2 == 0, dtype=bool)
    state["v_dltams"] = np.full((NPTS, NVM, 60), case_id * 10000 + 401.0)
    state["histgrowthset"] = np.full((NPTS, NVM, 300, 5), case_id * 10000 + 402.0)
    state["box_ndays"] = np.full((NPTS, NVM, NBOXMAX), case_id * 10000 + 403, dtype=np.int32)
    for index, name in enumerate(BOX_REAL_FIELDS, 404):
        state[name] = np.full((NPTS, NVM, NBOXMAX), case_id * 10000 + float(index))
    state["box_ulai"] = np.full((NVM, NBOXMAX), case_id * 10000 + 411.0)
    return state


def _field_layout() -> list[tuple[str, tuple[int, ...], np.dtype[Any]]]:
    layout: list[tuple[str, tuple[int, ...], np.dtype[Any]]] = []
    for name in STATE_FIELDS:
        if name in REAL_2D_FIELDS:
            layout.append((name, (NPTS, NVM), np.dtype("<f8")))
        elif name in INTEGER_2D_FIELDS:
            layout.append((name, (NPTS, NVM), np.dtype("<i4")))
        elif name in LOGICAL_2D_FIELDS:
            layout.append((name, (NPTS, NVM), np.dtype("<i4")))
        elif name == "v_dltams":
            layout.append((name, (NPTS, NVM, 60), np.dtype("<f8")))
        elif name == "histgrowthset":
            layout.append((name, (NPTS, NVM, 300, 5), np.dtype("<f8")))
        elif name == "box_ndays":
            layout.append((name, (NPTS, NVM, NBOXMAX), np.dtype("<i4")))
        elif name in BOX_REAL_FIELDS:
            layout.append((name, (NPTS, NVM, NBOXMAX), np.dtype("<f8")))
        elif name == "box_ulai":
            layout.append((name, (NVM, NBOXMAX), np.dtype("<f8")))
        else:
            raise RuntimeError(f"missing binary layout for {name}")
    layout.extend(
        (
            ("f_crop_init", (1,), np.dtype("<i4")),
            ("f_crop_recycle", (NPTS, NVM), np.dtype("<i4")),
        )
    )
    return layout


def _read_binary(path: Path) -> dict[str, np.ndarray]:
    payload = path.read_bytes()
    offset = 0
    result: dict[str, np.ndarray] = {}
    logical = {*LOGICAL_2D_FIELDS, "f_crop_init", "f_crop_recycle"}
    for name, shape, dtype in _field_layout():
        count = int(np.prod(shape))
        size = count * dtype.itemsize
        values = np.frombuffer(payload, dtype=dtype, count=count, offset=offset).reshape(shape, order="F")
        result[name] = values != 0 if name in logical else values.copy()
        offset += size
    if offset != len(payload):
        raise RuntimeError(f"unexpected binary payload size for {path}: {len(payload)} != {offset}")
    return result


def _jax_outputs(case: dict[str, Any]) -> dict[str, np.ndarray]:
    recycle = np.zeros((NPTS, NVM), dtype=bool)
    for point, pft in case["recycle"]:
        recycle[point, pft] = True
    result = stics_init_source_routed(
        state=_initial_state(int(case["id"])),
        f_crop_init=bool(case["f_crop_init"]),
        f_crop_recycle=recycle,
        parameters=_parameters(),
    )
    output = {name: np.asarray(value) for name, value in result.state.items()}
    output["f_crop_init"] = np.asarray([result.f_crop_init], dtype=bool)
    output["f_crop_recycle"] = np.asarray(result.f_crop_recycle, dtype=bool)
    return output


def _run_executable(executable: Path, output_dir: Path, compiler: Path) -> subprocess.CompletedProcess[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [str(executable), str(output_dir.resolve())],
        cwd=executable.parent,
        env=compiler_environment(compiler),
        check=True,
        capture_output=True,
        text=True,
    )


def run_numerical_oracle(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    compiler: Path = DEFAULT_COMPILER,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = extract_procedure_bytes(SOURCE, "stics_init")
    (output_dir / "Stics_init.f90").write_bytes(extracted.span_bytes)
    (output_dir / "minimal_dependencies.f90").write_bytes(_minimal_dependencies())
    with tempfile.TemporaryDirectory(prefix="orchidee_stics_init_batch_d_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        source_metadata = compose(source)
        compiler_metadata = compile_fortran(source, executable, compiler)
        run = _run_executable(executable, output_dir, compiler)
        shutil.copy2(source, output_dir / "oracle.f90")

    comparisons: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    with (output_dir / "point_comparisons.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("case", "field", "count", "max_abs_error", "max_rel_error", "passed"))
        for case in CASES:
            actual = _read_binary(output_dir / f"case_{case['id']}.bin")
            expected = _jax_outputs(case)
            if set(actual) != set(expected):
                raise RuntimeError(f"case {case['name']} output field mismatch")
            for name in sorted(actual):
                label = f"{case['name']}.{name}"
                if actual[name].dtype == bool or np.issubdtype(actual[name].dtype, np.integer):
                    comparison = exact_comparison(label, actual[name], expected[name])
                    max_abs = 0.0 if comparison["passed"] else float("inf")
                    max_rel = max_abs
                else:
                    comparison = float_comparison(label, actual[name], expected[name], rtol=RTOL, atol=ATOL)
                    max_abs = float(comparison["max_abs_error"])
                    max_rel = float(comparison["max_rel_error"])
                comparisons.append(comparison)
                writer.writerow((case["name"], name, comparison["count"], max_abs, max_rel, str(comparison["passed"]).lower()))
            summaries.append(
                {
                    "case": case["name"],
                    "pft14_index_zero_based": 13,
                    "bare_soil_index_zero_based": 0,
                    "f_crop_init": case["f_crop_init"],
                    "recycle_true_indices_zero_based": [list(value) for value in case["recycle"]],
                    "witness": case["witness"],
                    "binary_asset": f"case_{case['id']}.bin",
                    "binary_sha256": _sha256((output_dir / f"case_{case['id']}.bin").read_bytes()),
                }
            )

    _json(
        output_dir / "inputs.json",
        {
            "schema_version": 1,
            "family": FAMILY,
            "npts": NPTS,
            "nvm": NVM,
            "nboxmax": NBOXMAX,
            "paper_switches": {"ok_laidev_all_false": True, "sp_codlainet": [1] * NVM},
            "cases": summaries,
        },
    )
    result = write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "owner_region_id": OWNER_REGION_ID,
            "source": {
                "file": SOURCE.relative_to(ROOT).as_posix(),
                **source_metadata,
                "extracted_asset": "Stics_init.f90",
                "exact_extracted_bytes": (output_dir / "Stics_init.f90").read_bytes() == extracted.span_bytes,
            },
            "dependencies": {
                "asset": "minimal_dependencies.f90",
                "modules": ["pft_parameters", "constantes"],
                "scientific_callees": [],
                "note": "Only module state named by the byte-exact procedure is provided; no process logic is stubbed.",
            },
            "compiler": compiler_metadata,
            "normal_stdout": run.stdout,
            "normal_stderr": run.stderr,
            "writeback_field_count": len(STATE_FIELDS) + 2,
            "case_count": len(CASES),
            "tolerance_policy": {"rtol": RTOL, "atol": ATOL, "changed": False},
        },
    )
    if result["status"] != "passed":
        raise RuntimeError("Stics_init numerical comparison failed")
    return result


def _target_contract() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    contracts = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    owner = next(
        record for record in contracts["owner_regions"] if record["owner_region_id"] == OWNER_REGION_ID
    )
    arms = [record for record in contracts["arm_classifications"] if record["base_region_id"] == BASE_REGION_ID]
    proofs = json.loads(CENTRAL_SOURCE_PROOFS.read_text(encoding="utf-8"))
    proof_by_arm = {record["arm_id"]: record for record in proofs["records"] if record.get("passed") is True}
    selected_proofs = [proof_by_arm[arm["arm_id"]] for arm in arms if arm["arm_id"] in proof_by_arm]
    required = set(owner["arm_ids"])
    if required != {arm["arm_id"] for arm in arms} or required != {record["arm_id"] for record in selected_proofs}:
        raise RuntimeError("Stics_init owner contract/source-proof drift")
    return owner, arms, selected_proofs


def run_coverage(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    compiler: Path = DEFAULT_COMPILER,
    gcov: Path = GCOV,
) -> dict[str, Any]:
    comparison = run_numerical_oracle(output_dir, compiler)
    owner, arms, source_proofs = _target_contract()
    extracted = extract_procedure_bytes(SOURCE, "stics_init")
    with tempfile.TemporaryDirectory(prefix="orchidee_stics_init_batch_d_gcov_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        executable = build / "oracle_gcov.exe"
        compose(source)
        coverage_compiler = compile_fortran(source, executable, compiler, extra_flags=("--coverage",))
        coverage_output = build / "outputs"
        run = _run_executable(executable, coverage_output, compiler)
        notes = next(build.glob("*.gcno"))
        gcov_run = subprocess.run(
            [str(gcov), "-b", "-c", notes.name],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
        gcov_path = build / f"{source.name}.gcov"
        raw_branches = parse_gcov(gcov_path)
        line_counts = parse_gcov_line_counts(gcov_path)
        generated_start = locate_generated_span(source.read_bytes(), extracted.span_bytes)
        shutil.copy2(gcov_path, output_dir / "oracle.f90.gcov")

    arm_records: list[dict[str, Any]] = []
    for arm in arms:
        original_line = int(arm["line"])
        generated_line = generated_start + original_line - extracted.start_line
        branch_rows = raw_branches.get(generated_line, [])
        selected = select_arm_branch(str(arm["arm_id"]), branch_rows)
        gcov_passed = bool(selected is not None and selected["taken"] > 0)
        proof_passed = any(record["arm_id"] == arm["arm_id"] for record in source_proofs)
        arm_records.append(
            {
                "arm_id": arm["arm_id"],
                "base_region_id": BASE_REGION_ID,
                "procedure": "stics_init",
                "original_line": original_line,
                "generated_line": generated_line,
                "source": arm["source"],
                "gcov_branch_index": -1 if selected is None else selected["branch_index"],
                "gcov_taken": 0 if selected is None else selected["taken"],
                "gcov_line_count": line_counts.get(generated_line, 0),
                "gcov_passed": gcov_passed,
                "fatal_boundary_passed": False,
                "source_proof_passed": proof_passed,
                "procedure_span_sha256": extracted.span_sha256,
                "raw_gcov_branches": branch_rows,
                "passed": gcov_passed and proof_passed,
            }
        )

    missing = sorted(record["arm_id"] for record in arm_records if not record["passed"])
    required = sorted(owner["arm_ids"])
    gcov_arms = sorted(record["arm_id"] for record in arm_records if record["gcov_passed"])
    proof_arms = sorted(record["arm_id"] for record in arm_records if record["source_proof_passed"])
    coverage = {
        "schema_version": 1,
        "family": FAMILY,
        "mapping": "parsed_gcov_plus_canonical_source_proof",
        "branch_complete": not missing,
        "required_arm_count": len(required),
        "covered_arm_count": len(required) - len(missing),
        "missing_arm_ids": missing,
        "arms": arm_records,
        "gcov_asset": "oracle.f90.gcov",
        "gcov_stdout": gcov_run.stdout,
        "gcov_stderr": gcov_run.stderr,
        "coverage_stdout": run.stdout,
        "coverage_stderr": run.stderr,
        "compiler": coverage_compiler,
        "mapping_policy": "GNU Fortran -O0 --coverage; canonical true/false edge selected by fortran_gcov.select_arm_branch.",
    }
    _json(output_dir / "branch_coverage.json", coverage)
    _json(
        output_dir / "gcov.json",
        {
            "schema_version": 1,
            "family": FAMILY,
            "generated_procedure_start_line": generated_start,
            "branches_by_generated_line": {str(key): value for key, value in sorted(raw_branches.items())},
            "line_counts": {str(key): value for key, value in sorted(line_counts.items())},
        },
    )
    _json(
        output_dir / "source_proof_evidence.json",
        {
            "schema_version": 1,
            "family": FAMILY,
            "complete": len(source_proofs) == len(required),
            "canonical_asset": CENTRAL_SOURCE_PROOFS.relative_to(ROOT).as_posix(),
            "records": source_proofs,
        },
    )
    _json(
        output_dir / "fatal_boundary_evidence.json",
        {
            "schema_version": 1,
            "family": FAMILY,
            "complete": True,
            "required_fatal_arm_ids": [],
            "records": [],
            "reason": "Stics_init has no fatal boundary and all four required owner arms are normal-return edges.",
        },
    )
    owner_evidence = {
        "schema_version": 1,
        "family": FAMILY,
        "complete": not missing and comparison["status"] == "passed",
        "records": [
            {
                "owner_region_id": OWNER_REGION_ID,
                "base_region_id": BASE_REGION_ID,
                "fortran_procedure": "stics_init",
                "jax_owners": owner["jax_owners"],
                "comparison_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/comparison.json",
                "branch_coverage_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/branch_coverage.json",
                "required_arm_ids": required,
                "gcov_arm_ids": gcov_arms,
                "fatal_boundary_arm_ids": [],
                "source_proof_arm_ids": proof_arms,
                "covered_arm_ids": required if not missing else sorted(set(gcov_arms) & set(proof_arms)),
                "missing_arm_ids": missing,
                "passed": not missing and comparison["status"] == "passed",
            }
        ],
    }
    _json(output_dir / "owner_region_evidence.json", owner_evidence)
    if not coverage["branch_complete"] or not owner_evidence["complete"]:
        raise RuntimeError(f"Stics_init owner closure incomplete: {missing}")
    central_audit = build_central_audit_report(
        json.loads(CONTRACTS.read_text(encoding="utf-8")),
        json.loads(CENTRAL_SOURCE_PROOFS.read_text(encoding="utf-8")),
        load_evidence_documents(ROOT / "outputs/reference_mode/micro_oracles"),
    )
    accepted = next(
        (
            record
            for record in central_audit["records"]
            if record["owner_region_id"] == OWNER_REGION_ID
        ),
        None,
    )
    central_audit["assigned_owner_acceptance"] = {
        "owner_region_id": OWNER_REGION_ID,
        "accepted": bool(accepted is not None and accepted.get("passed") is True),
        "global_completion_not_required_by_this_batch_owner": True,
    }
    _json(output_dir / "central_audit_report.json", central_audit)
    if not central_audit["assigned_owner_acceptance"]["accepted"]:
        raise RuntimeError("central audit did not accept the Stics_init owner record")
    return {
        "comparison": comparison,
        "coverage": coverage,
        "owner_evidence": owner_evidence,
        "central_audit": central_audit,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Stage 4 Batch D Stics_init owner closure.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    try:
        result = run_coverage(args.output_dir.resolve())
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(
        json.dumps(
            {
                "status": "passed",
                "owner_region_id": OWNER_REGION_ID,
                "comparisons": len(result["comparison"]["comparisons"]),
                "covered_arms": result["coverage"]["covered_arm_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
