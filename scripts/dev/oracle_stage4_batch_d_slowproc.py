"""Stage 4 Batch D closure for the two SLOWPROC owners."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.bundle import load_paper_1961_first_step_bundle  # noqa: E402
from jax_orchidee.driver.static import (  # noqa: E402
    read_paper_usda_soil_static_fields,
    slowproc_soilt_from_explicit_overlap,
)
from jax_orchidee.driver.trace import read_fixed_format_soil_trace  # noqa: E402
from jax_orchidee.sechiba.slowproc import slowproc_init_pft14_explicit  # noqa: E402
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.audit_pft14_owner_region_evidence import (  # noqa: E402
    build_report as build_owner_report,
    load_evidence_documents,
)
from scripts.dev.fortran_gcov import (  # noqa: E402
    locate_generated_span,
    parse_gcov,
    parse_gcov_line_counts,
    select_arm_branch,
)
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    COMPILE_FLAGS,
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    exact_comparison,
    float_comparison,
    write_point_comparisons,
)
from scripts.dev.oracle_lane_slowproc_soilt_owner import (  # noqa: E402
    SPANS as SOILT_SPANS,
    compose as compose_soilt,
)


FAMILY = "slowproc_batch_d"
OWNER_IDS = (
    "pft14-owner-contract-1efd6322fc73",
    "pft14-owner-contract-463396c23861",
)
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90"
CONTRACTS = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
SOURCE_PROOFS = ROOT / "outputs/reference_mode/pft14_source_proof_evidence.json"
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
INIT_TEMPLATE = ROOT / "scripts/dev/oracle_stage4_batch_d_slowproc_init.f90.template"
SOILT_TEMPLATE = ROOT / "scripts/dev/oracle_lane_slowproc_soilt_owner.f90.template"
GCOV = Path(r"C:\msys64\ucrt64\bin\gcov.exe")
RTOL = 1.0e-12
ATOL = 1.0e-14
CONFIG = ROOT / "configs/orchidee_man_250919.yaml"
SOIL_TRACE = ROOT / "outputs/server_1961_trace_full_20260623/traces/orchjax_slowproc_soilt_trace.txt"


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _compose_init(path: Path) -> dict[str, Any]:
    init = extract_procedure_bytes(SOURCE, "slowproc_init")
    veget = extract_procedure_bytes(SOURCE, "slowproc_veget")
    payload = INIT_TEMPLATE.read_bytes()
    payload = payload.replace(b"! <SLOWPROC_INIT_PROCEDURE>", init.span_bytes)
    payload = payload.replace(b"! <SLOWPROC_VEGET_PROCEDURE>", veget.span_bytes)
    _write_bytes(path, payload)
    return {
        "slowproc_init": _span_metadata(init),
        "slowproc_veget": _span_metadata(veget),
    }


def _span_metadata(span) -> dict[str, Any]:
    return {
        "start_line": span.start_line,
        "end_line": span.end_line,
        "sha256": span.span_sha256,
    }


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    rows: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for name, value in csv.reader(handle):
            rows.setdefault(name, []).append(float(value))
    return {name: np.asarray(values, dtype=np.float64) for name, values in rows.items()}


def _init_kwargs(case: int) -> dict[str, Any]:
    height = np.asarray(
        [0.0, 30.0, 30.0, 20.0, 20.0, 20.0, 15.0, 15.0, 15.0, 0.5, 0.6, 1.0, 1.0, 30.0],
        dtype=np.float64,
    )
    default = np.zeros((3, 14), dtype=np.float64)
    default[:, 13] = 1.0
    restart: dict[str, Any] = {}
    if case == 2:
        veget_max = np.zeros((3, 14), dtype=np.float64)
        veget_max[:2, 13] = 1.0
        veget = np.zeros_like(veget_max)
        veget[0, 13] = 0.97318184
        lai = np.zeros_like(veget_max)
        lai[0, 13] = 3.6186761
        frac_age = np.zeros((3, 14, 4), dtype=np.float64)
        frac_age[:, :, 0] = 1.0
        frac_age[0, 13, :] = [0.23, 0.25, 0.26, 0.26]
        restart = {
            "veget": veget,
            "veget_max": veget_max,
            "frac_nobio": np.zeros((3, 1), dtype=np.float64),
            "lai": lai,
            "height": np.broadcast_to(height, (3, 14)).copy(),
            "frac_age": frac_age,
            "njsc": np.asarray([2, 5, 12], dtype=np.int32),
            "clay_frac": np.asarray([0.2, 0.1, 0.55]),
            "sand_frac": np.asarray([0.4, 0.06, 0.15]),
            "bulk_dens": np.asarray([1650.0, 1450.0, 1380.0]),
            "soil_ph": np.asarray([7.0, 6.5, 5.8]),
            "poor_soils": np.asarray([0.0, 0.2, 0.1]),
            "reinf_slope": np.asarray([0.0, 0.2, 0.3]),
            "peatPET_last": np.asarray([2.0, 3.0, 4.0]),
            "precipitation_last": np.asarray([10.0, 20.0, 30.0]),
            "precipitation_this": np.asarray([1.0, 2.0, 3.0]),
            "peatPET_this": np.asarray([0.1, 0.2, 0.3]),
            "growth_day": np.asarray([17.0, 18.0, 19.0]),
            "GSL": np.asarray([0.1, 0.2, 0.3]),
            "summerp_longterm": np.asarray([5.0, 6.0, 7.0]),
            "summerpet_longterm": np.asarray([8.0, 9.0, 10.0]),
            "peatC": np.asarray([11.0, 12.0, 13.0]),
            "peatC_ok": np.asarray([1.0, 0.0, 1.0]),
        }
    soilclass = np.zeros((3, 12), dtype=np.float64)
    soilclass[:, 1] = 1.0
    return {
        "restart": restart,
        "veget_max_default": default,
        "frac_nobio_default": 0.0,
        "height_presc": height,
        "pref_soil_veg": np.asarray([1] * 13 + [4], dtype=np.int32),
        "ext_coeff_vegetfrac": np.asarray([0.0] + [0.5] * 13),
        "nstm": 6,
        "diaglev": np.asarray([0.1, 0.3, 1.0]),
        "soil_boundary": {
            "soilclass": soilclass,
            "clay_frac": np.full(3, 0.2),
            "sand_frac": np.full(3, 0.4),
            "silt_frac": np.full(3, 0.4),
            "bulk_dens": np.full(3, 1650.0),
            "soil_ph": np.full(3, 7.0),
            "poor_soils": np.zeros(3),
        },
        "salinity_data": np.asarray([31.25, 18.0, 0.0]),
        "tide_height_data": np.asarray([[0.1, -0.2, 0.3], [0.2, 0.0, -0.1], [0.3, 0.2, 0.4]]),
    }


def _flatten_init_result(case: int) -> dict[str, np.ndarray]:
    result = slowproc_init_pft14_explicit(**_init_kwargs(case))
    fields = {
        "lai": result.lai,
        "height": result.height,
        "frac_age": result.frac_age,
        "veget": result.veget,
        "veget_max": result.veget_max,
        "frac_nobio": result.frac_nobio,
        "totfrac_nobio": result.totfrac_nobio,
        "soiltile": result.soiltile,
        "tot_bare_soil": result.tot_bare_soil,
        "reinf_slope": result.reinf_slope,
        "njsc": result.njsc,
        "clayfraction": result.clayfraction,
        "sandfraction": result.sandfraction,
        "siltfraction": result.siltfraction,
        "bulk_density": result.bulk_density,
        "soil_ph": result.soil_ph,
        "poor_soils": result.poor_soils,
        "fc_grazing": result.fc_grazing,
        "salinity": result.salinity,
        "tide_height": result.tide_height,
        "peatPET_lastyear": result.peat.peatPET_lastyear,
        "precipitation_lastsummer": result.peat.precipitation_lastsummer,
        "precipitation_thissummer": result.peat.precipitation_thissummer,
        "peatPET_thisyear": result.peat.peatPET_thisyear,
        "growth_day": result.peat.growth_day,
        "GSL": result.peat.GSL,
        "summerp_long": result.peat.summerp_long,
        "summerpet_long": result.peat.summerpet_long,
        "peatC": result.peat.peatC,
        "peatC_ok": result.peat.peatC_ok,
        "vegetnew_firstday": result.no_lcc.vegetnew_firstday,
        "glccNetLCC": result.no_lcc.glccNetLCC,
        "glccSecondShift": result.no_lcc.glccSecondShift,
        "glccPrimaryShift": result.no_lcc.glccPrimaryShift,
        "harvest_matrix": result.no_lcc.harvest_matrix,
        "bound_spa": result.no_lcc.bound_spa,
    }
    # These SAVE arrays are source-defined zeros on the fixed FIRE_DISABLE path.
    for name, shape in {
        "m_lightn": (3, 12),
        "proxy_anidens": (3, 12),
        "m_observed_ba": (3, 12),
        "m_cf_coarse": (3, 12),
        "m_cf_fine": (3, 12),
        "m_ratio": (3, 12),
        "m_ratio_flag": (3, 12),
    }.items():
        fields[name] = np.zeros(shape)
    fields["popd"] = np.asarray(result.fire.popd)
    fields["humign"] = np.asarray(result.fire.humign)
    fields["scalars"] = np.asarray([result.lcanop, result.veget_update])
    return {name: np.asarray(value).ravel(order="F") for name, value in fields.items()}


def run_init_comparison(output: Path, compiler: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_stage4_slowproc_init_") as td:
        build = Path(td)
        source = build / "slowproc_init_oracle.f90"
        executable = build / "slowproc_init_oracle.exe"
        spans = _compose_init(source)
        metadata = compile_fortran(source, executable, compiler)
        actual: dict[str, np.ndarray] = {}
        expected: dict[str, np.ndarray] = {}
        for case, label in ((1, "cold"), (2, "restart")):
            case_output = output / f"slowproc_init_{label}_fortran.csv"
            subprocess.run(
                [str(executable), str(case), str(case_output.resolve())],
                cwd=build,
                env=compiler_environment(compiler),
                check=True,
                capture_output=True,
                text=True,
            )
            for name, values in _read_csv(case_output).items():
                actual[f"{label}_{name}"] = values
            for name, values in _flatten_init_result(case).items():
                expected[f"{label}_{name}"] = values
    if set(actual) != set(expected):
        raise RuntimeError(
            f"slowproc_init state field drift: missing={sorted(set(expected)-set(actual))}, "
            f"unexpected={sorted(set(actual)-set(expected))}"
        )
    comparisons = []
    discrete = {name for name in actual if name.endswith("_njsc") or name.endswith("_scalars")}
    for name in sorted(actual):
        if name in discrete:
            comparisons.append(exact_comparison(name, actual[name].astype(np.int64), expected[name].astype(np.int64)))
        else:
            comparisons.append(float_comparison(name, actual[name], expected[name], rtol=RTOL, atol=ATOL))
    write_point_comparisons(output / "slowproc_init_point_comparisons.csv", actual, expected, rtol=RTOL, atol=ATOL)
    result = {
        "schema_version": 2,
        "component": "slowproc_init",
        "status": "passed" if comparisons and all(item["passed"] for item in comparisons) else "failed",
        "comparison_passed": bool(comparisons and all(item["passed"] for item in comparisons)),
        "tolerance_policy": {"rtol": RTOL, "atol": ATOL, "changed": False},
        "case_matrix": [
            {"id": "cold", "restart": False, "pft14": True, "defined_state_complete": True},
            {"id": "restart", "restart": True, "pft14": True, "bare_diagnostic_points": [2, 3]},
        ],
        "comparisons": comparisons,
        "source_fragments": spans,
        "compiler": metadata,
        "undefined_source_state": {
            "fields": ["veget_year", "laimap", "veget_max_new", "frac_nobio_new", "veget_max_adjusted"],
            "policy": "Not compared: original slowproc_init leaves these allocated/INTENT(out) objects undefined on the fixed imposed-vegetation path.",
        },
    }
    (output / "slowproc_init_comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    if result["status"] != "passed":
        failed = [item["name"] for item in comparisons if not item["passed"]]
        raise RuntimeError(f"slowproc_init comparison failed: {failed}")
    return result


def _soilt_overlap_expected() -> dict[str, np.ndarray]:
    soiltext = np.asarray([[1.0, 2.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    soilbd = np.asarray([[1400.0, 1500.0, 1450.0], [1500.0, 1600.0, 1500.0], [1700.0, 1550.0, 1520.0]])
    soil_ph = np.asarray([[6.5, 6.8, 6.9], [7.0, 7.2, 6.7], [5.5, 7.1, 6.6]])
    poor = np.asarray([[0.1, 0.2, 0.0], [0.0, 0.3, 0.0], [0.2, 0.1, 0.0]])
    sub_index = np.zeros((3, 2, 2), dtype=np.int64)
    sub_area = np.zeros((3, 2), dtype=np.float64)
    sub_index[1] = [[1, 1], [2, 2]]
    sub_area[1] = [0.25, 0.75]
    sub_index[2] = [[1, 3], [3, 1]]
    sub_area[2] = [0.6, 0.4]
    result = slowproc_soilt_from_explicit_overlap(
        soiltext,
        soilbd,
        soil_ph,
        poor,
        sub_index,
        sub_area,
        soil_classif="usda",
        index_base=1,
    )
    return {
        "active_soilclass": np.asarray(result["soilclass"]).ravel(order="F"),
        "active_clayfraction": np.asarray(result["clay_frac"]),
        "active_sandfraction": np.asarray(result["sand_frac"]),
        "active_siltfraction": np.asarray(result["silt_frac"]),
        "active_bulk_density": np.asarray(result["bulk_dens"]),
        "active_soil_ph": np.asarray(result["soil_ph"]),
        "active_poor_soils": np.asarray(result["poor_soils"]),
    }


def _paper_soilt_trace_expected() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    trace = read_fixed_format_soil_trace(SOIL_TRACE)
    if len(trace) != 1:
        raise RuntimeError(f"expected one fixed-format slowproc_soilt trace row, found {len(trace)}")
    row = trace[0]
    bundle = load_paper_1961_first_step_bundle(CONFIG, year=1961)
    fields, overlap = read_paper_usda_soil_static_fields(
        CONFIG,
        lalo=bundle.domain.lalo,
        resolution_m=bundle.domain.resolution,
    )
    expected_class = np.zeros((1, 12), dtype=np.float64)
    expected_class[0, int(round(float(row["soiltext"]))) - 1] = 1.0
    actual = {
        "paper_soilclass": np.asarray(fields["soilclass"]).ravel(order="F"),
        "paper_njsc": np.asarray(fields["njsc"]),
        "paper_clayfraction": np.asarray(fields["clay_frac"]),
        "paper_sandfraction": np.asarray(fields["sand_frac"]),
        "paper_siltfraction": np.asarray(fields["silt_frac"]),
        "paper_bulk_density": np.asarray(fields["bulk_dens"]),
        "paper_soil_ph": np.asarray(fields["soil_ph"]),
        "paper_poor_soils": np.asarray(fields["poor_soils"]),
        "paper_sub_index": np.asarray(overlap["soilclass_sub_index"][0, 0]),
        "paper_sub_area": np.asarray([overlap["soilclass_sub_area"][0, 0]]),
    }
    expected = {
        "paper_soilclass": expected_class.ravel(order="F"),
        "paper_njsc": np.asarray([int(round(float(row["soiltext"])))], dtype=np.int32),
        "paper_clayfraction": np.asarray([row["clay_frac"]]),
        "paper_sandfraction": np.asarray([row["sand_frac"]]),
        "paper_siltfraction": np.asarray([row["silt_frac"]]),
        "paper_bulk_density": np.asarray([row["bulk_dens"]]),
        "paper_soil_ph": np.asarray([row["soil_ph"]]),
        "paper_poor_soils": np.asarray([row["poor_soils"]]),
        "paper_sub_index": np.asarray([row["source_i"], row["source_j"]], dtype=np.int32),
        "paper_sub_area": np.asarray([row["sub_area"]]),
    }
    return {**actual, **{f"expected::{name}": value for name, value in expected.items()}}, {
        "netcdf_path": str((ROOT / "data/INPUTDIR_ZZ/test.nc").resolve()),
        "netcdf_sha256": hashlib.sha256((ROOT / "data/INPUTDIR_ZZ/test.nc").read_bytes()).hexdigest(),
        "trace_path": SOIL_TRACE.relative_to(ROOT).as_posix(),
        "trace_sha256": hashlib.sha256(SOIL_TRACE.read_bytes()).hexdigest(),
    }


def run_soilt_comparison(output: Path, compiler: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="orchidee_stage4_slowproc_soilt_") as td:
        build = Path(td)
        source = build / "slowproc_soilt_oracle.f90"
        executable = build / "slowproc_soilt_oracle.exe"
        spans = compose_soilt(source)
        metadata = compile_fortran(source, executable, compiler)
        raw_output = output / "slowproc_soilt_fortran.csv"
        subprocess.run(
            [str(executable), "normal", str(raw_output.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
        _write_bytes(output / "slowproc_soilt_oracle.f90", source.read_bytes())
    local_actual = _read_csv(raw_output)
    local_expected = _soilt_overlap_expected()
    paper_packet, real_format = _paper_soilt_trace_expected()
    paper_actual = {name: value for name, value in paper_packet.items() if not name.startswith("expected::")}
    paper_expected = {name.removeprefix("expected::"): value for name, value in paper_packet.items() if name.startswith("expected::")}
    actual = {**local_actual, **paper_actual}
    expected = {**local_expected, **paper_expected}
    if set(actual) != set(expected):
        raise RuntimeError(f"slowproc_soilt field drift: {sorted(set(actual) ^ set(expected))}")
    discrete = {"paper_njsc", "paper_sub_index"}
    comparisons = [
        exact_comparison(name, actual[name].astype(np.int64), expected[name].astype(np.int64))
        if name in discrete
        else float_comparison(name, actual[name], expected[name], rtol=RTOL, atol=ATOL)
        for name in sorted(actual)
    ]
    write_point_comparisons(output / "slowproc_soilt_point_comparisons.csv", actual, expected, rtol=RTOL, atol=ATOL)
    result = {
        "schema_version": 2,
        "component": "slowproc_soilt",
        "status": "passed" if comparisons and all(item["passed"] for item in comparisons) else "failed",
        "comparison_passed": bool(comparisons and all(item["passed"] for item in comparisons)),
        "tolerance_policy": {"rtol": RTOL, "atol": ATOL, "changed": False},
        "case_matrix": [
            {"id": "paper_netcdf_trace", "format": "NetCDF", "classification": "usda", "pft14_landpoint": True},
            {"id": "masked_multipoint_retry_default", "format": "IO-boundary packet from deterministic NetCDF variables", "masked_cells": True},
        ],
        "comparisons": comparisons,
        "source_fragments": spans,
        "real_format_inputs": real_format,
        "compiler": metadata,
    }
    (output / "slowproc_soilt_comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    if result["status"] != "passed":
        raise RuntimeError(f"slowproc_soilt comparison failed: {[x['name'] for x in comparisons if not x['passed']]}")
    return result


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _owner_contracts() -> dict[str, dict[str, Any]]:
    records = {
        str(record["owner_region_id"]): record
        for record in _load_json(CONTRACTS).get("owner_regions", [])
        if record.get("owner_region_id") in OWNER_IDS
    }
    if set(records) != set(OWNER_IDS):
        raise RuntimeError("Batch D slowproc owner IDs drifted from the canonical contract")
    expected = {
        "pft14-owner-contract-1efd6322fc73": ("slowproc_init", 84),
        "pft14-owner-contract-463396c23861": ("slowproc_soilt", 51),
    }
    for owner_id, (procedure, count) in expected.items():
        owner = records[owner_id]
        if owner.get("fortran_procedure") != procedure or len(owner.get("arm_ids", [])) != count:
            raise RuntimeError(f"canonical owner contract drifted: {owner_id}")
    return records


def _passed_source_proofs() -> set[str]:
    return {
        str(record["arm_id"])
        for record in _load_json(SOURCE_PROOFS).get("records", [])
        if record.get("passed") is True
    }


def _run_gcov(compiler: Path, gcov: Path, build: Path, source: Path, executable: Path) -> Path:
    before = set(build.glob("*.gcno"))
    compile_fortran(
        source,
        executable,
        compiler,
        extra_flags=("--coverage",),
        base_flags=tuple(flag for flag in COMPILE_FLAGS if flag != "-fcheck=all"),
    )
    created = set(build.glob("*.gcno")) - before
    if len(created) != 1:
        raise RuntimeError(f"expected one new gcov notes file for {source.name}, found {sorted(created)}")
    return created.pop()


def _gcov_listing(gcov: Path, build: Path, notes: Path, source: Path) -> Path:
    subprocess.run(
        [str(gcov), "-b", "-c", notes.name],
        cwd=build,
        env=compiler_environment(gcov),
        check=True,
        capture_output=True,
        text=True,
    )
    listing = build / f"{source.name}.gcov"
    if not listing.is_file():
        raise RuntimeError(f"gcov did not write {listing.name}")
    return listing


def _map_gcov_arms(
    *,
    owner: dict[str, Any],
    proof_ids: set[str],
    source: Path,
    source_span: bytes,
    original_start: int,
    listing: Path,
    fatal_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    raw = parse_gcov(listing)
    line_counts = parse_gcov_line_counts(listing)
    generated_start = locate_generated_span(source.read_bytes(), source_span)
    fatal_ids = fatal_ids or set()
    records = []
    passed: set[str] = set()
    for arm_id in owner["arm_ids"]:
        arm_id = str(arm_id)
        if arm_id in proof_ids or arm_id in fatal_ids:
            continue
        line = int(arm_id.rsplit(":", 3)[-3])
        generated_line = generated_start + line - original_start
        condition_line = generated_line + {1997: 3, 2175: 4}.get(line, 0)
        branch_rows = raw.get(condition_line, [])
        if arm_id.endswith(":case:case"):
            witness_line = next(
                (candidate for candidate in range(generated_line + 1, generated_line + 8) if line_counts.get(candidate, 0) > 0),
                None,
            )
            branch = None if witness_line is None else {"branch_index": -2, "taken": line_counts[witness_line]}
        else:
            witness_line = condition_line
            branch = select_arm_branch(arm_id, branch_rows)
        record = {
            "arm_id": arm_id,
            "original_line": line,
            "generated_line": witness_line,
            "gcov_branch_index": -1 if branch is None else branch["branch_index"],
            "gcov_taken": 0 if branch is None else branch["taken"],
            "raw_gcov_branches": branch_rows,
            "passed": bool(branch and branch["taken"] > 0),
        }
        records.append(record)
        if record["passed"]:
            passed.add(arm_id)
    return records, passed


def _map_soilt_gcov_arms(
    *,
    owner: dict[str, Any],
    proof_ids: set[str],
    source: Path,
    listing: Path,
    fatal_ids: set[str],
) -> tuple[list[dict[str, Any]], set[str]]:
    raw = parse_gcov(listing)
    line_counts = parse_gcov_line_counts(listing)
    original_lines = SOURCE.read_bytes().splitlines(keepends=True)
    generated_starts = {}
    for name, (start, end) in SOILT_SPANS.items():
        fragment = b"".join(original_lines[start - 1 : end])
        generated_starts[name] = locate_generated_span(source.read_bytes(), fragment)
    records = []
    passed: set[str] = set()
    for raw_arm_id in owner["arm_ids"]:
        arm_id = str(raw_arm_id)
        if arm_id in proof_ids or arm_id in fatal_ids:
            continue
        line = int(arm_id.rsplit(":", 3)[-3])
        selected = next(
            ((name, span) for name, span in SOILT_SPANS.items() if span[0] <= line <= span[1]),
            None,
        )
        if selected is None:
            raise RuntimeError(f"no extracted slowproc_soilt span contains {arm_id}")
        name, span = selected
        generated_line = generated_starts[name] + line - span[0]
        branch_rows = raw.get(generated_line, [])
        if arm_id.endswith(":case:case"):
            witness_line = next(
                (candidate for candidate in range(generated_line + 1, generated_line + 8) if line_counts.get(candidate, 0) > 0),
                None,
            )
            branch = None if witness_line is None else {"branch_index": -2, "taken": line_counts[witness_line]}
        else:
            witness_line = generated_line
            branch = select_arm_branch(arm_id, branch_rows)
        record = {
            "arm_id": arm_id,
            "original_line": line,
            "generated_line": witness_line,
            "gcov_branch_index": -1 if branch is None else branch["branch_index"],
            "gcov_taken": 0 if branch is None else branch["taken"],
            "raw_gcov_branches": branch_rows,
            "passed": bool(branch and branch["taken"] > 0),
        }
        records.append(record)
        if record["passed"]:
            passed.add(arm_id)
    return records, passed


def run_branch_witnesses(output: Path, compiler: Path, gcov: Path) -> dict[str, Any]:
    owners = _owner_contracts()
    proof_ids = _passed_source_proofs()
    init_span = extract_procedure_bytes(SOURCE, "slowproc_init")
    fatal_ids = {
        "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90:4548:if:true",
        "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90:4863:if:true",
    }
    fatal_witnesses: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="orchidee_stage4_slowproc_gcov_") as td:
        build = Path(td)

        init_source = build / "slowproc_init_coverage.f90"
        init_exe = build / "slowproc_init_coverage.exe"
        _compose_init(init_source)
        init_notes = _run_gcov(compiler, gcov, build, init_source, init_exe)
        init_cases = []
        for case in range(1, 14):
            completed = subprocess.run(
                [str(init_exe), str(case), str((build / f"init_{case}.csv").resolve())],
                cwd=build,
                env=compiler_environment(compiler),
                capture_output=True,
                text=True,
            )
            init_cases.append({"case": case, "returncode": completed.returncode, "passed": completed.returncode == 0})
        if not all(case["passed"] for case in init_cases):
            raise RuntimeError(f"slowproc_init GCOV case failed: {init_cases}")
        init_listing = _gcov_listing(gcov, build, init_notes, init_source)
        init_records, init_passed = _map_gcov_arms(
            owner=owners[OWNER_IDS[0]],
            proof_ids=proof_ids,
            source=init_source,
            source_span=init_span.span_bytes,
            original_start=init_span.start_line,
            listing=init_listing,
        )
        _write_bytes(output / "slowproc_init_coverage_oracle.f90", init_source.read_bytes())
        _write_bytes(output / "slowproc_init_coverage_oracle.f90.gcov", init_listing.read_bytes())

        soilt_source = build / "slowproc_soilt_coverage.f90"
        soilt_exe = build / "slowproc_soilt_coverage.exe"
        compose_soilt(soilt_source)
        soilt_notes = _run_gcov(compiler, gcov, build, soilt_source, soilt_exe)
        soilt_cases = []
        for mode in ("normal", "complete", "impsoilt"):
            completed = subprocess.run(
                [str(soilt_exe), mode, str((build / f"soilt_{mode}.csv").resolve())],
                cwd=build,
                env=compiler_environment(compiler),
                capture_output=True,
                text=True,
            )
            soilt_cases.append({"case": mode, "returncode": completed.returncode, "passed": completed.returncode == 0})
        for mode, expected_text, arm_id in (
            ("retry", "slowproc_soilt|Error in allocation for solbd||", "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90:4548:if:true"),
            ("fatal", "slowproc_soilt|Problem soil color class incompatible 2||", "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90:4863:if:true"),
        ):
            completed = subprocess.run(
                [str(soilt_exe), mode, str((build / f"soilt_{mode}.csv").resolve())],
                cwd=build,
                env=compiler_environment(compiler),
                capture_output=True,
                text=True,
            )
            combined = completed.stdout + completed.stderr
            matched = completed.returncode != 0 and expected_text in combined
            fatal_witnesses.append(
                {
                    "case": mode,
                    "arm_id": arm_id,
                    "expected_message": expected_text,
                    "returncode": completed.returncode,
                    "matched": matched,
                    "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
                    "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
                }
            )
        if not all(case["passed"] for case in soilt_cases) or not all(item["matched"] for item in fatal_witnesses):
            raise RuntimeError(f"slowproc_soilt witness failure: cases={soilt_cases}, fatal={fatal_witnesses}")
        soilt_listing = _gcov_listing(gcov, build, soilt_notes, soilt_source)
        soilt_records, soilt_passed = _map_soilt_gcov_arms(
            owner=owners[OWNER_IDS[1]],
            proof_ids=proof_ids,
            source=soilt_source,
            listing=soilt_listing,
            fatal_ids=fatal_ids,
        )
        _write_bytes(output / "slowproc_soilt_coverage_oracle.f90", soilt_source.read_bytes())
        _write_bytes(output / "slowproc_soilt_coverage_oracle.f90.gcov", soilt_listing.read_bytes())

    witness = {
        "schema_version": 1,
        "init_cases": init_cases,
        "soilt_cases": soilt_cases,
        "fatal_witnesses": fatal_witnesses,
        "gcov_records": {"slowproc_init": init_records, "slowproc_soilt": soilt_records},
        "gcov_arm_ids": sorted(init_passed | soilt_passed),
        "fatal_witness_arm_ids": sorted(fatal_ids),
        "passed": all(item["matched"] for item in fatal_witnesses),
    }
    (output / "execution_witnesses.json").write_text(json.dumps(witness, indent=2) + "\n", encoding="ascii")
    return witness


def assemble_evidence(
    output: Path,
    init_comparison: dict[str, Any],
    soilt_comparison: dict[str, Any],
    witnesses: dict[str, Any],
) -> dict[str, Any]:
    owners = _owner_contracts()
    proof_ids = _passed_source_proofs()
    gcov_ids = set(witnesses["gcov_arm_ids"])
    fatal_ids = set(witnesses["fatal_witness_arm_ids"])
    records = []
    arm_records = []
    component = {
        "slowproc_init": init_comparison,
        "slowproc_soilt": soilt_comparison,
    }
    for owner_id in OWNER_IDS:
        owner = owners[owner_id]
        procedure = str(owner["fortran_procedure"])
        required = set(map(str, owner["arm_ids"]))
        pinned = required & proof_ids
        executed = required & gcov_ids
        fatal = required & fatal_ids
        covered = pinned | executed | fatal
        missing = required - covered
        comparison = component[procedure]
        records.append(
            {
                "owner_region_id": owner_id,
                "base_region_id": owner["base_region_id"],
                "fortran_procedure": procedure,
                "jax_owners": owner["jax_owners"],
                "comparison_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/{procedure}_comparison.json",
                "branch_coverage_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/branch_coverage.json",
                "required_arm_ids": sorted(required),
                "gcov_arm_ids": sorted(executed),
                "fatal_witness_arm_ids": sorted(fatal),
                "source_proof_arm_ids": sorted(pinned),
                "covered_arm_ids": sorted(covered),
                "missing_arm_ids": sorted(missing),
                "direct_state_writeback_comparison": {
                    "status": comparison["status"],
                    "comparison_count": len(comparison["comparisons"]),
                    "tolerance_policy": comparison["tolerance_policy"],
                },
                "passed": not missing and comparison["status"] == "passed",
            }
        )
        for arm_id in sorted(required):
            support = (
                "pinned_source_proof"
                if arm_id in pinned
                else "exact_fatal_witness"
                if arm_id in fatal
                else "real_gcov"
                if arm_id in executed
                else "unclosed"
            )
            arm_records.append({"owner_region_id": owner_id, "arm_id": arm_id, "support": support})
    missing = [item["arm_id"] for item in arm_records if item["support"] == "unclosed"]
    coverage = {
        "schema_version": 1,
        "family": FAMILY,
        "branch_complete": not missing and all(record["passed"] for record in records),
        "required_arm_count": len(arm_records),
        "covered_arm_count": len(arm_records) - len(missing),
        "required_by_owner": {OWNER_IDS[0]: 84, OWNER_IDS[1]: 51},
        "missing_arm_ids": missing,
        "gcov_arm_ids": sorted(gcov_ids),
        "fatal_witness_arm_ids": sorted(fatal_ids),
        "source_proof_arm_ids": sorted({item["arm_id"] for item in arm_records if item["support"] == "pinned_source_proof"}),
        "arms": arm_records,
        "policy": "Only accumulated real GCOV, exact fatal witnesses, or the pinned canonical source-proof ledger support an arm.",
    }
    evidence = {
        "schema_version": 1,
        "family": FAMILY,
        "complete": coverage["branch_complete"],
        "records": records,
    }
    (output / "branch_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="ascii")
    (output / "owner_region_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="ascii")
    if not coverage["branch_complete"]:
        raise RuntimeError(f"{FAMILY} canonical arm coverage incomplete: {missing}")
    return {"coverage": coverage, "evidence": evidence}


def write_inputs(output: Path, init: dict[str, Any], soilt: dict[str, Any]) -> None:
    spans = {}
    for procedure in ("slowproc_init", "slowproc_soilt"):
        extracted = extract_procedure_bytes(SOURCE, procedure)
        asset = f"owner_{procedure}.f90"
        _write_bytes(output / asset, extracted.span_bytes)
        spans[procedure] = {"asset": asset, **_span_metadata(extracted)}
    applicability = {
        "schema_version": 1,
        "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "procedure_spans": spans,
        "defined_state_policy": {
            "compared": "Every defined INTENT(out), INTENT(inout), and SAVE write represented by the production owner is emitted.",
            "source_undefined": ["veget_year", "laimap", "veget_max_new", "frac_nobio_new", "veget_max_adjusted"],
            "proof": "Pinned original procedure bytes contain allocation/declaration but no fixed-path defining assignment before return.",
        },
        "batch_attention_applicability": {
            "cold_start": "slowproc_init case cold",
            "restart": "slowproc_init case restart",
            "daily_boundary": "not referenced by either owned procedure; slowproc_main owns the daily gate",
            "pft14_and_bare": "restart case contains occupied and zero-canopy PFT14 points; no non-PFT14 process is claimed",
            "mask": "slowproc_soilt mask and masked-source-cell cases",
            "snow_and_active_layer": "not referenced by either owned procedure; thermosoil/stomate owners hold those branches",
        },
    }
    (output / "source_applicability_proof.json").write_text(json.dumps(applicability, indent=2) + "\n", encoding="ascii")
    inputs = {
        "schema_version": 1,
        "family": FAMILY,
        "owner_ids": list(OWNER_IDS),
        "source_file": SOURCE.relative_to(ROOT).as_posix(),
        "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "owner_procedure_spans": spans,
        "case_matrix": {"slowproc_init": init["case_matrix"], "slowproc_soilt": soilt["case_matrix"]},
        "minimal_genuine_dependencies": [
            "byte-exact slowproc_veget",
            "byte-exact get_soilcorr_usda",
            "restart/config transport callbacks",
            "certified aggregate_p overlap boundary",
            "IOIPSL NetCDF transport boundary",
        ],
        "tolerance_policy": {"rtol": RTOL, "atol": ATOL, "changed": False},
    }
    (output / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n", encoding="ascii")


def run_all(output: Path, compiler: Path, gcov: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    init = run_init_comparison(output, compiler)
    soilt = run_soilt_comparison(output, compiler)
    witnesses = run_branch_witnesses(output, compiler, gcov)
    write_inputs(output, init, soilt)
    assembled = assemble_evidence(output, init, soilt, witnesses)
    comparison = {
        "schema_version": 2,
        "family": FAMILY,
        "status": "passed",
        "components": ["slowproc_init_comparison.json", "slowproc_soilt_comparison.json"],
        "comparison_count": len(init["comparisons"]) + len(soilt["comparisons"]),
        "tolerance_policy": {"rtol": RTOL, "atol": ATOL, "changed": False},
    }
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="ascii")
    aggregate = build_owner_report(
        _load_json(CONTRACTS),
        _load_json(SOURCE_PROOFS),
        load_evidence_documents(output.parent),
    )
    accepted = {
        str(record["owner_region_id"])
        for record in aggregate["records"]
        if record.get("owner_region_id") in OWNER_IDS
    }
    if accepted != set(OWNER_IDS) or any(aggregate["errors"].values()):
        raise RuntimeError(
            f"aggregate rejected Batch D slowproc evidence: accepted={sorted(accepted)}, "
            f"errors={aggregate['errors']}"
        )
    (output / "aggregate_audit.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="ascii")
    return {**assembled, "comparison": comparison, "aggregate": aggregate}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    parser.add_argument("--gcov", type=Path, default=GCOV)
    parser.add_argument("--init-only", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists() and output != OUTPUT.resolve():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    if args.init_only:
        result = run_init_comparison(output, args.compiler.resolve())
        summary = {"status": result["status"], "comparisons": len(result["comparisons"])}
    else:
        result = run_all(output, args.compiler.resolve(), args.gcov.resolve())
        summary = {
            "status": result["comparison"]["status"],
            "covered_arms": result["coverage"]["covered_arm_count"],
            "required_arms": result["coverage"]["required_arm_count"],
            "owner_regions": len(result["evidence"]["records"]),
        }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
