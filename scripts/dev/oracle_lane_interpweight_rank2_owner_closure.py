"""Executable original-byte closure for the two rank-2 interpweight owners."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.interpolation_aggregate import aggregate_2d  # noqa: E402
from jax_orchidee.driver.interpolation_core12 import (  # noqa: E402
    AggregatePacket,
    InterpolationSource,
    InterpolationTarget,
    interpweight_2d,
)
from jax_orchidee.driver.interpolation_core4cont import (  # noqa: E402
    interpweight_2dcont_routed,
)
from scripts.dev.extract_fortran_micro_oracle import (  # noqa: E402
    ExtractedProcedure,
    extract_procedure_bytes,
)
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    exact_comparison,
    float_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "interpweight_rank2_owner_closure"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/interpweight.f90"
AGGREGATE_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/interpol_help.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_interpweight_rank2_owner_closure.f90.template"
CONTRACT_CLASSES = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
OWNER_IDS = {
    "interpweight_2d": "pft14-owner-contract-252914bbabc7",
    "interpweight_2dcont": "pft14-owner-contract-372b01d64eab",
}
PROCEDURES = {
    "interpweight_2D": SOURCE,
    "interpweight_2Dcont": SOURCE,
    "interpweight_get_var3dims_file": SOURCE,
    "interpweight_get_var4dims_file": SOURCE,
    "interpweight_get_varNdims_file": SOURCE,
    "interpweight_calc_resolution_in": SOURCE,
    "interpweight_modifying_input2D": SOURCE,
    "interpweight_modifying_input3D": SOURCE,
    "interpweight_modifying_input4D": SOURCE,
    "interpweight_masking_input2D": SOURCE,
    "interpweight_masking_input3D": SOURCE,
    "interpweight_masking_input4D": SOURCE,
    "interpweight_provide_fractions2D": SOURCE,
    "interpweight_provide_interpolation2D": SOURCE,
    "aggregate_2d": AGGREGATE_SOURCE,
}
FLOAT_FIELDS = {"output", "availability", "overlap_area"}
DISCRETE_FIELDS = {"attempt_count", "attempt_width", "mask", "overlap_index"}
CONTRACT_MODES = (
    "frac_default",
    "frac_rank3",
    "frac_rank4_zero",
    "frac_rank4_selected",
    "frac_rank4_late",
    "frac_rank4_full",
    "cont_default",
    "cont_rank3",
    "cont_rank4_zero",
    "cont_rank4_selected",
    "cont_rank4_late",
    "cont_rank4_full",
)


@dataclass(frozen=True)
class Case:
    owner: str
    fixture_id: int
    masktype: str
    noneg: bool
    target_lon: float
    target_resolution: float
    typefrac: str = "default"
    default_types: bool = False


CASES = {
    "frac_east_nomask": Case(
        "fraction", 1, "nomask", False, 179.75, 180000.0, default_types=True
    ),
    "frac_west_var": Case("fraction", 2, "var", True, -179.75, 180000.0),
    "frac_boundary_mbelow": Case("fraction", 3, "mbelow", False, 30.0, 120000.0),
    "frac_mabove": Case("fraction", 3, "mabove", False, 0.0, 120000.0),
    "frac_msumrange": Case("fraction", 3, "msumrange", False, 0.0, 120000.0),
    "frac_dense_retry": Case("fraction", 4, "nomask", False, 0.0, 200000.0),
    "frac_large_capacity": Case("fraction", 3, "nomask", False, 0.0, 2000000.0),
    "cont_east_default": Case("continuous", 1, "nomask", False, 179.75, 180000.0),
    "cont_west_slope_var": Case(
        "continuous", 2, "var", True, -179.75, 180000.0, typefrac="slopecalc"
    ),
    "cont_boundary_mbelow": Case("continuous", 3, "mbelow", False, 30.0, 120000.0),
    "cont_mabove": Case("continuous", 3, "mabove", False, 0.0, 120000.0),
    "cont_msumrange": Case("continuous", 3, "msumrange", False, 0.0, 120000.0),
    "cont_dense_retry": Case("continuous", 4, "nomask", False, 0.0, 200000.0),
}


def _marker(procedure: str) -> bytes:
    return f"! <{procedure.upper()}>".encode("ascii")


def compose(source_path: Path, extracted_dir: Path) -> dict[str, dict[str, Any]]:
    data = TEMPLATE.read_bytes()
    records: dict[str, dict[str, Any]] = {}
    extracted_dir.mkdir(parents=True, exist_ok=True)
    for procedure, source in PROCEDURES.items():
        span = extract_procedure_bytes(source, procedure)
        marker = _marker(procedure)
        if data.count(marker) != 1:
            raise ValueError(f"expected exactly one marker for {procedure}")
        data = data.replace(marker, span.span_bytes)
        extracted_file = extracted_dir / f"{procedure}.f90"
        extracted_file.write_bytes(span.span_bytes)
        records[procedure] = _span_record(span, source, extracted_file)
    if b"! <" in data:
        raise ValueError("unreplaced extracted-procedure marker")
    source_path.write_bytes(data)
    return records


def _span_record(
    span: ExtractedProcedure, source: Path, extracted_file: Path
) -> dict[str, Any]:
    try:
        extracted_asset = extracted_file.relative_to(ROOT).as_posix()
    except ValueError:
        extracted_asset = str(extracted_file.resolve())
    return {
        "source_file": source.relative_to(ROOT).as_posix(),
        "source_sha256": span.source_sha256,
        "start_line": span.start_line,
        "end_line": span.end_line,
        "span_sha256": span.span_sha256,
        "extracted_file": extracted_asset,
        "byte_count": len(span.span_bytes),
    }


def _read_fortran(path: Path) -> dict[str, dict[str, np.ndarray]]:
    fields: dict[str, dict[str, list[float]]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            fields.setdefault(row["case"], {}).setdefault(row["field"], []).append(
                float(row["value"])
            )
    return {
        case: {field: np.asarray(values) for field, values in case_fields.items()}
        for case, case_fields in fields.items()
    }


def _shape(values: np.ndarray) -> tuple[int, int]:
    width = int(round(np.sqrt(values.size)))
    if width * width != values.size:
        raise ValueError(f"fixture source field is not square: {values.size}")
    return width, width


def _target(case: Case) -> InterpolationTarget:
    return InterpolationTarget(
        lalo=np.asarray([[0.0, case.target_lon], [45.0, 90.0]]),
        resolution=np.full((2, 2), case.target_resolution),
        neighbours=np.zeros((2, 8), dtype=np.int32),
        contfrac=np.asarray([0.75, 0.5]),
    )


def _mask_variable(shape: tuple[int, int]) -> np.ndarray:
    i, j = np.indices(shape)
    return ((i + j) % 2 == 0).astype(np.float64)


def _source(case: Case, actual: dict[str, np.ndarray]) -> InterpolationSource:
    shape = _shape(actual["source_values"])
    values = actual["source_values"].reshape(shape, order="F")
    lon = actual["source_lon"].reshape(shape, order="F")
    lat = actual["source_lat"].reshape(shape, order="F")
    if case.fixture_id == 2:
        source_lon: np.ndarray = lon
        source_lat: np.ndarray = lat
    else:
        source_lon = lon[:, 0]
        source_lat = lat[0, :]
    return InterpolationSource(
        values=values,
        longitude=source_lon,
        latitude=source_lat,
        mask_variable=_mask_variable(shape),
        variable_name="soilcolor" if case.owner == "fraction" else "topography",
    )


def _aggregate_callback(captured: dict[str, Any]):
    def callback(request):
        result = aggregate_2d(
            request.lalo.shape[0],
            request.lalo,
            request.neighbours,
            request.resolution,
            request.contfrac,
            request.longitude.shape[0],
            request.longitude.shape[1],
            request.longitude,
            request.latitude,
            request.mask,
            request.callsign,
            request.nbvmax,
            global_grid=True,
        )
        captured.setdefault("widths", []).append(request.nbvmax)
        if result.ok:
            captured["packet"] = result
            captured["mask"] = request.mask.copy()
        return AggregatePacket(result.indinc, result.areaoverlap, result.ok)

    return callback


def _jax_case(case: Case, actual: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    source = _source(case, actual)
    target = _target(case)
    captured: dict[str, Any] = {}
    callback = _aggregate_callback(captured)
    if case.owner == "fraction":
        variabletypes = [-1.0, -1.0, -1.0] if case.default_types else [1.0, 2.0, 3.0]
        result = interpweight_2d(
            source,
            target,
            variabletypes,
            callback,
            varmin=1.0,
            varmax=3.0,
            noneg=case.noneg,
            masktype=case.masktype,
            maskvalues=(1.5, 1.0, 5.0),
            max_resolution_lon=100000.0 if case.default_types else -1.0,
            max_resolution_lat=100000.0 if case.default_types else -1.0,
        )
        output = result.fractions.ravel(order="F")
        availability = result.availability
    else:
        result = interpweight_2dcont_routed(
            source,
            target,
            callback,
            noneg=case.noneg,
            masktype=case.masktype,
            maskvalues=(1.5, 1.0, 5.0),
            typefrac=case.typefrac,
            defaultvalue=7.5,
            default_no_value=4.0,
        )
        output = result.output
        availability = result.availability
    packet = captured["packet"]
    return {
        "output": np.asarray(output),
        "availability": np.asarray(availability),
        "attempt_count": np.asarray([len(captured["widths"])]),
        "attempt_width": np.asarray(captured["widths"]),
        "mask": np.asarray(captured["mask"]).ravel(order="F"),
        "overlap_index": np.asarray(packet.indinc).ravel(order="F"),
        "overlap_area": np.asarray(packet.areaoverlap).ravel(order="F"),
    }


def _run_contracts(
    executable: Path, build: Path, compiler: Path
) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for mode in CONTRACT_MODES:
        completed = subprocess.run(
            [str(executable), mode],
            cwd=build,
            env=compiler_environment(compiler),
            text=True,
            capture_output=True,
            check=False,
        )
        combined = completed.stdout + completed.stderr
        if "default" in mode:
            expected = "IPSLERR"
            contract = "explicit_ipslerr_p_fatal"
        else:
            expected = "not allocated"
            contract = "unallocated_invar2D_undefined"
        outcomes.append(
            {
                "mode": mode,
                "owner_region_id": OWNER_IDS[
                    "interpweight_2dcont" if mode.startswith("cont_") else "interpweight_2d"
                ],
                "source_contract": contract,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "passed": completed.returncode != 0 and expected.lower() in combined.lower(),
            }
        )
    return outcomes


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "branch_coverage.json",
        "owner_region_evidence.json",
        "formal_audit_report.json",
        "formal_audit_report_verification.json",
        "target_contract_classes.json",
        "oracle_interpweight_rank2_owner.f90.gcov",
    ):
        (output_dir / stale_name).unlink(missing_ok=True)
    shutil.rmtree(output_dir / "formal_audit_evidence", ignore_errors=True)
    extracted_dir = output_dir / "extracted_original_bytes"
    fortran_output = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_interpweight_rank2_owner_") as temporary:
        build = Path(temporary)
        source = build / "oracle_interpweight_rank2_owner.f90"
        executable = build / "oracle_interpweight_rank2_owner.exe"
        spans = compose(source, extracted_dir)
        shutil.copy2(source, output_dir / "composed_oracle.f90")
        compiler_meta = compile_fortran(
            source,
            executable,
            compiler,
            extra_flags=("-cpp",),
        )
        subprocess.run(
            [str(executable), "valid", str(fortran_output.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            text=True,
            capture_output=True,
            check=True,
        )
        contracts = _run_contracts(executable, build, compiler)

    if not all(record["passed"] for record in contracts):
        failed = [record["mode"] for record in contracts if not record["passed"]]
        raise AssertionError(f"fatal/undefined contracts failed: {failed}")
    fortran = _read_fortran(fortran_output)
    if set(fortran) != set(CASES):
        raise AssertionError(f"Fortran case drift: {sorted(fortran)}")
    jax = {name: _jax_case(case, fortran[name]) for name, case in CASES.items()}
    flat_fortran: dict[str, np.ndarray] = {}
    flat_jax: dict[str, np.ndarray] = {}
    comparisons: list[dict[str, Any]] = []
    for case_name in sorted(CASES):
        for field in sorted(FLOAT_FIELDS | DISCRETE_FIELDS):
            name = f"{case_name}.{field}"
            flat_fortran[name] = fortran[case_name][field]
            flat_jax[name] = jax[case_name][field]
            if field in DISCRETE_FIELDS:
                comparisons.append(
                    exact_comparison(
                        name, fortran[case_name][field].astype(np.int64), jax[case_name][field]
                    )
                )
            else:
                comparisons.append(
                    float_comparison(
                        name,
                        fortran[case_name][field],
                        jax[case_name][field],
                        rtol=1e-12,
                        atol=1e-14,
                    )
                )
    write_point_comparisons(
        output_dir / "point_comparisons.csv",
        flat_fortran,
        flat_jax,
        rtol=1e-12,
        atol=1e-14,
    )
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": {name: case.__dict__ for name, case in CASES.items()},
                "fixture_transport": "deterministic rank-2 soilcolor/topography arrays",
                "tolerance": {"rtol": 1e-12, "atol": 1e-14, "discrete": "exact"},
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
            "aggregate_source_file_sha256": hashlib.sha256(
                AGGREGATE_SOURCE.read_bytes()
            ).hexdigest(),
            "procedure_spans": spans,
            "compiler": compiler_meta,
            "fatal_and_undefined_contracts": contracts,
            "tolerance": {"rtol": 1e-12, "atol": 1e-14, "discrete": "exact"},
        },
    )
    if result["status"] != "passed":
        failed = [item["name"] for item in comparisons if not item["passed"]]
        raise AssertionError(f"Fortran/JAX comparisons failed: {failed}")
    return result


if __name__ == "__main__":
    output = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
    oracle_result = run_oracle(output)
    print(json.dumps(oracle_result, indent=2))
    raise SystemExit(0 if oracle_result["status"] == "passed" else 1)
