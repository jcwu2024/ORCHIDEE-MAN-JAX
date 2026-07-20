from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.science_completion import slowproc_main_history_routing  # noqa: E402
from scripts.dev.fortran_gcov import locate_generated_span, parse_gcov, select_arm_branch  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    COMPILE_FLAGS,
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "slowproc_main_owners"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_slowproc_main_owners.f90.template"
GCOV = Path(r"C:\msys64\ucrt64\bin\gcov.exe")
CONTRACT_CLASSES = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
SOURCE_PROOFS = ROOT / "outputs/reference_mode/pft14_source_proof_evidence.json"
OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY

CALENDAR_SPAN = (556, 675)
LANDUSE_SPAN = (685, 708)
HISTORY_SPAN = (1017, 1068)
SPANS = {
    "calendar": CALENDAR_SPAN,
    "landuse": LANDUSE_SPAN,
    "history": HISTORY_SPAN,
}


def _span(start: int, end: int) -> bytes:
    lines = SOURCE.read_bytes().splitlines(keepends=True)
    return b"".join(lines[start - 1 : end])


def compose(path: Path) -> dict[str, Any]:
    content = TEMPLATE.read_bytes()
    span_hashes = {}
    for name, (start, end) in SPANS.items():
        fragment = _span(start, end)
        content = content.replace(f"! <SLOWPROC_MAIN_{name.upper()}_SPAN>".encode(), fragment)
        span_hashes[name] = {
            "start_line": start,
            "end_line": end,
            "span_sha256": hashlib.sha256(fragment).hexdigest(),
        }
    path.write_bytes(content)
    return span_hashes


def _read(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.reader(handle):
            values.setdefault(row[0], []).append(float(row[1]))
    return {name: np.asarray(value, dtype=np.float64) for name, value in values.items()}


def _calendar_expected() -> dict[str, np.ndarray]:
    def payload(
        *,
        sec: float,
        dt_sechiba: float,
        month: int,
        day: int,
        dynpeat_pc: bool,
        dynpeat_pwt: bool,
        peatc: float,
        pc_lim: float,
        growth_day: np.ndarray,
        temp_growth: np.ndarray,
        precip_this: np.ndarray,
        precip_last: np.ndarray,
        peatpet_this: np.ndarray,
        peatpet_last: np.ndarray,
        gsl: np.ndarray,
        date: int,
    ) -> dict[str, np.ndarray]:
        first_ts_year = sec == dt_sechiba and month == 1 and day == 1
        last_ts_day = sec == 0.0
        end_of_year = last_ts_day and month == 1 and day == 1
        dyn_peat = dynpeat_pc or dynpeat_pwt
        first_ts_month = dyn_peat and sec == dt_sechiba and day == 1
        gd = growth_day.copy()
        peatc_ok = np.zeros_like(gd)
        precip_this_out = precip_this.copy()
        precip_last_out = precip_last.copy()
        peatpet_this_out = peatpet_this.copy()
        peatpet_last_out = peatpet_last.copy()
        gsl_out = gsl.copy()
        summerp_long = np.array([5.0, 6.0], dtype=np.float64)
        summerpet_long = np.array([7.0, 8.0], dtype=np.float64)
        if first_ts_year:
            gd[:] = 0.0
        if sec == dt_sechiba and month == 1 and day == 2:
            if not dynpeat_pc:
                peatc_ok[:] = 1.0
            elif peatc >= pc_lim:
                peatc_ok[:] = 1.0
        if last_ts_day:
            gd = gd + np.where(temp_growth > 5.0, 1.0, 0.0)
        if dyn_peat and dynpeat_pwt and 5 <= month <= 9:
            precip_this_out = precip_this_out + np.array([1.5, 2.25], dtype=np.float64)
            peatpet_this_out = peatpet_this_out + np.array([0.1, 0.2], dtype=np.float64)
        if dyn_peat and dynpeat_pwt and end_of_year:
            peatpet_last_out = peatpet_this_out.copy()
            peatpet_this_out[:] = 0.0
            precip_last_out = precip_this_out.copy()
            precip_this_out[:] = 0.0
            gsl_out = gd / 365.0
        update_peatfrac = first_ts_month
        sat_duration = None
        if dyn_peat:
            sat_length = np.trunc(30.0 * 12.0 * gsl_out).astype(np.float64)
            sat_duration = np.minimum(360.0, np.maximum(1.0, 360.0 - sat_length))
        do_slow = last_ts_day
        date_out = date + 1 if do_slow else date
        if dyn_peat and dynpeat_pwt and end_of_year:
            if date_out == 365:
                summerp_long = precip_last_out.copy()
                summerpet_long = peatpet_last_out.copy()
            summerp_long = (summerp_long * 29.0 + precip_last_out) / 30.0
            summerpet_long = (summerpet_long * 29.0 + peatpet_last_out) / 30.0
        result = {
            "first_ts_year": np.array([1.0 if first_ts_year else 0.0]),
            "last_ts_day": np.array([1.0 if last_ts_day else 0.0]),
            "end_of_year": np.array([1.0 if end_of_year else 0.0]),
            "first_ts_month": np.array([1.0 if first_ts_month else 0.0]),
            "update_peatfrac": np.array([1.0 if update_peatfrac else 0.0]),
            "do_slow": np.array([1.0 if do_slow else 0.0]),
            "date": np.array([float(date_out)]),
            "growth_day": gd,
            "peatC_ok": peatc_ok,
            "precipitation_thissummer": precip_this_out,
            "precipitation_lastsummer": precip_last_out,
            "peatPET_thisyear": peatpet_this_out,
            "peatPET_lastyear": peatpet_last_out,
            "GSL": gsl_out,
            "summerp_long": summerp_long,
            "summerpet_long": summerpet_long,
        }
        if sat_duration is not None:
            result["sat_duration"] = sat_duration
        return result

    cases = {
        "cal_base_": dict(
            sec=1800.0,
            dt_sechiba=1800.0,
            month=6,
            day=15,
            dynpeat_pc=False,
            dynpeat_pwt=False,
            peatc=0.0,
            pc_lim=50.0,
            growth_day=np.array([10.0, 20.0]),
            temp_growth=np.array([4.0, 6.0]),
            precip_this=np.array([0.0, 0.0]),
            precip_last=np.array([10.0, 20.0]),
            peatpet_this=np.array([0.0, 0.0]),
            peatpet_last=np.array([1.0, 2.0]),
            gsl=np.array([0.1, 0.2]),
            date=10,
        ),
        "cal_newyear_": dict(
            sec=1800.0,
            dt_sechiba=1800.0,
            month=1,
            day=1,
            dynpeat_pc=False,
            dynpeat_pwt=False,
            peatc=0.0,
            pc_lim=50.0,
            growth_day=np.array([10.0, 20.0]),
            temp_growth=np.array([4.0, 6.0]),
            precip_this=np.array([0.0, 0.0]),
            precip_last=np.array([10.0, 20.0]),
            peatpet_this=np.array([0.0, 0.0]),
            peatpet_last=np.array([1.0, 2.0]),
            gsl=np.array([0.1, 0.2]),
            date=10,
        ),
        "cal_pc_false_": dict(
            sec=1800.0,
            dt_sechiba=1800.0,
            month=1,
            day=2,
            dynpeat_pc=True,
            dynpeat_pwt=False,
            peatc=40.0,
            pc_lim=50.0,
            growth_day=np.array([0.0, 0.0]),
            temp_growth=np.array([4.0, 6.0]),
            precip_this=np.array([0.0, 0.0]),
            precip_last=np.array([10.0, 20.0]),
            peatpet_this=np.array([0.0, 0.0]),
            peatpet_last=np.array([1.0, 2.0]),
            gsl=np.array([0.1, 0.2]),
            date=10,
        ),
        "cal_pc_true_": dict(
            sec=1800.0,
            dt_sechiba=1800.0,
            month=1,
            day=2,
            dynpeat_pc=True,
            dynpeat_pwt=False,
            peatc=60.0,
            pc_lim=50.0,
            growth_day=np.array([0.0, 0.0]),
            temp_growth=np.array([4.0, 6.0]),
            precip_this=np.array([0.0, 0.0]),
            precip_last=np.array([10.0, 20.0]),
            peatpet_this=np.array([0.0, 0.0]),
            peatpet_last=np.array([1.0, 2.0]),
            gsl=np.array([0.1, 0.2]),
            date=10,
        ),
        "cal_pwt_monthly_": dict(
            sec=1800.0,
            dt_sechiba=1800.0,
            month=6,
            day=1,
            dynpeat_pc=False,
            dynpeat_pwt=True,
            peatc=0.0,
            pc_lim=50.0,
            growth_day=np.array([7.0, 8.0]),
            temp_growth=np.array([4.0, 6.0]),
            precip_this=np.array([1.0, 2.0]),
            precip_last=np.array([10.0, 20.0]),
            peatpet_this=np.array([0.5, 0.75]),
            peatpet_last=np.array([1.0, 2.0]),
            gsl=np.array([0.1, 0.2]),
            date=10,
        ),
        "cal_pc_gate_false_": dict(
            sec=1800.0,
            dt_sechiba=1800.0,
            month=1,
            day=2,
            dynpeat_pc=True,
            dynpeat_pwt=False,
            peatc=40.0,
            pc_lim=50.0,
            growth_day=np.array([0.0, 0.0]),
            temp_growth=np.array([4.0, 6.0]),
            precip_this=np.array([0.0, 0.0]),
            precip_last=np.array([10.0, 20.0]),
            peatpet_this=np.array([0.0, 0.0]),
            peatpet_last=np.array([1.0, 2.0]),
            gsl=np.array([0.1, 0.2]),
            date=10,
        ),
        "cal_eoy_not365_": dict(
            sec=0.0,
            dt_sechiba=1800.0,
            month=1,
            day=1,
            dynpeat_pc=False,
            dynpeat_pwt=True,
            peatc=0.0,
            pc_lim=50.0,
            growth_day=np.array([7.0, 8.0]),
            temp_growth=np.array([4.0, 6.0]),
            precip_this=np.array([1.0, 2.0]),
            precip_last=np.array([10.0, 20.0]),
            peatpet_this=np.array([0.5, 0.75]),
            peatpet_last=np.array([1.0, 2.0]),
            gsl=np.array([0.1, 0.2]),
            date=10,
        ),
        "cal_eoy_365_": dict(
            sec=0.0,
            dt_sechiba=1800.0,
            month=1,
            day=1,
            dynpeat_pc=False,
            dynpeat_pwt=True,
            peatc=0.0,
            pc_lim=50.0,
            growth_day=np.array([7.0, 8.0]),
            temp_growth=np.array([4.0, 6.0]),
            precip_this=np.array([1.0, 2.0]),
            precip_last=np.array([10.0, 20.0]),
            peatpet_this=np.array([0.5, 0.75]),
            peatpet_last=np.array([1.0, 2.0]),
            gsl=np.array([0.1, 0.2]),
            date=364,
        ),
        "cal_lastday_not_eoy_": dict(
            sec=0.0,
            dt_sechiba=1800.0,
            month=1,
            day=2,
            dynpeat_pc=False,
            dynpeat_pwt=False,
            peatc=0.0,
            pc_lim=50.0,
            growth_day=np.array([10.0, 20.0]),
            temp_growth=np.array([4.0, 6.0]),
            precip_this=np.array([0.0, 0.0]),
            precip_last=np.array([10.0, 20.0]),
            peatpet_this=np.array([0.0, 0.0]),
            peatpet_last=np.array([1.0, 2.0]),
            gsl=np.array([0.1, 0.2]),
            date=20,
        ),
    }
    out = {}
    for prefix, kwargs in cases.items():
        result = payload(**kwargs)
        out.update({prefix + name: value for name, value in result.items()})
    return out


def _landuse_expected() -> dict[str, np.ndarray]:
    def packet(map_pft_format: bool, veget_update: int, date: int, veget_year: int, veget_year_orig: int) -> dict[str, np.ndarray]:
        calls = 0.0
        veget_year_tmp = 0.0
        veget_nextyear = np.full(3, -1.0)
        frac_nobio_nextyear = np.full(2, -1.0)
        if map_pft_format and veget_update > 0:
            if date == 1:
                veget_year_tmp = float(veget_year + 1)
                if ((veget_year + 1) - veget_year_orig) % veget_update == 0:
                    calls = 1.0
                    leading = 0.1 * float(veget_year - 1998)
                    veget_nextyear = np.array([leading, 0.25, 0.75 - leading], dtype=np.float64)
                    frac_nobio_nextyear = np.array([0.1, 0.0], dtype=np.float64)
        return {
            "readveget_calls": np.array([calls]),
            "veget_year_tmp": np.array([veget_year_tmp]),
            "veget_nextyear": veget_nextyear,
            "frac_nobio_nextyear": frac_nobio_nextyear,
        }

    cases = {
        "landuse_off_": packet(True, 0, 1, 1999, 1999),
        "landuse_datefalse_": packet(True, 1, 2, 1999, 1999),
        "landuse_modtrue_": packet(True, 1, 1, 1999, 1999),
        "landuse_modfalse_": packet(True, 2, 1, 1999, 1999),
    }
    out = {}
    for prefix, values in cases.items():
        out.update({prefix + name: value for name, value in values.items()})
    return out


def _history_expected() -> dict[str, np.ndarray]:
    gpp = np.array([[9.0, 10.0, 20.0], [8.0, 12.0, 24.0]], dtype=np.float64)
    resp_maint = np.array([[3.0, 2.0, 4.0], [2.0, 1.0, 2.0]], dtype=np.float64)
    resp_growth = np.array([[2.0, 1.0, 3.0], [1.0, 1.0, 2.0]], dtype=np.float64)
    resp_hetero = np.array([[1.0, 5.0, 6.0], [2.0, 4.0, 3.0]], dtype=np.float64)
    bulkdens = np.array([1400.0, 1650.0], dtype=np.float64)
    textfrac = np.array([[0.2, 0.3, 0.5], [0.3, 0.1, 0.6]], dtype=np.float64)
    active = slowproc_main_history_routing(
        gpp=gpp,
        resp_maint=resp_maint,
        resp_growth=resp_growth,
        resp_hetero=resp_hetero,
        hist2_id=2,
    )
    inactive = slowproc_main_history_routing(
        gpp=gpp,
        resp_maint=resp_maint,
        resp_growth=resp_growth,
        resp_hetero=resp_hetero,
        hist2_id=0,
    )
    return {
        "hist_active_bulkdens": bulkdens,
        "hist_active_textfrac": textfrac.ravel(order="F"),
        "hist_active_npp": np.asarray(active.npp).ravel(order="F"),
        "hist_inactive_bulkdens": bulkdens,
        "hist_inactive_textfrac": textfrac.ravel(order="F"),
        "hist_inactive_npp": np.asarray(inactive.npp).ravel(order="F"),
    }


def expected_outputs() -> dict[str, np.ndarray]:
    expected = {}
    expected.update(_calendar_expected())
    expected.update(_landuse_expected())
    expected.update(_history_expected())
    return expected


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_slowproc_main_owners_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        span_hashes = compose(source)
        metadata = compile_fortran(source, executable, compiler)
        output = output_dir / "fortran_outputs.csv"
        subprocess.run(
            [str(executable), str(output.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    actual = _read(output_dir / "fortran_outputs.csv")
    expected = expected_outputs()
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise ValueError(f"missing expected oracle outputs: {missing}")
    comparable_actual = {name: actual[name] for name in expected}
    write_point_comparisons(output_dir / "point_comparisons.csv", comparable_actual, expected, rtol=1e-12, atol=1e-14)
    comparisons = [
        float_comparison(name, comparable_actual[name], expected[name], rtol=1e-12, atol=1e-14)
        for name in sorted(expected)
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps({"schema_version": 1, "families": ["main_surface", "main_history"]}, indent=2) + "\n",
        encoding="ascii",
    )
    return write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "span_sha256": span_hashes,
            "compiler": metadata,
        },
    )


def _collect_target_arms() -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    contract = json.loads(CONTRACT_CLASSES.read_text(encoding="utf-8"))
    source_proofs = json.loads(SOURCE_PROOFS.read_text(encoding="utf-8"))
    source_key = SOURCE.relative_to(ROOT).as_posix()
    owner_regions = {
        str(record["base_region_id"]): record
        for record in contract["owner_regions"]
        if record.get("fortran_file") == source_key
        and record.get("fortran_procedure") == "slowproc_main"
        and record.get("owner_region_id")
        in {"pft14-owner-contract-f9cf3838e7eb", "pft14-owner-contract-22f52ac38e14"}
    }
    target_arms = [
        record
        for record in contract["arm_classifications"]
        if str(record.get("base_region_id")) in owner_regions
    ]
    passed_source_proofs = {
        str(record["arm_id"])
        for record in source_proofs.get("records", [])
        if record.get("passed") is True
    }
    return owner_regions, target_arms, passed_source_proofs


def run_coverage(output_dir: Path = OUTPUT_DIR, compiler: Path = DEFAULT_COMPILER, gcov: Path = GCOV) -> dict[str, Any]:
    comparison = run_oracle(output_dir, compiler)
    if comparison.get("status") != "passed":
        raise RuntimeError("slowproc_main numerical comparison did not pass")

    owner_regions, target_arms, passed_source_proofs = _collect_target_arms()
    gcov_targets = [record for record in target_arms if str(record["arm_id"]) not in passed_source_proofs]

    with tempfile.TemporaryDirectory(prefix="orchidee_slowproc_main_gcov_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        executable = build / "oracle_gcov.exe"
        span_hashes = compose(source)
        compile_fortran(
            source,
            executable,
            compiler,
            extra_flags=("--coverage",),
            base_flags=tuple(flag for flag in COMPILE_FLAGS if flag != "-fcheck=all"),
        )
        subprocess.run(
            [str(executable), str((build / "out.csv").resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
        notes = next(build.glob("*.gcno"))
        subprocess.run(
            [str(gcov), "-b", "-c", notes.name],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
        raw_branches = parse_gcov(build / f"{source.name}.gcov")
        generated = source.read_bytes()
        generated_starts = {
            name: locate_generated_span(generated, _span(start, end))
            for name, (start, end) in SPANS.items()
        }
        arm_records = []
        for arm in gcov_targets:
            arm_id = str(arm["arm_id"])
            line = int(arm["line"])
            if CALENDAR_SPAN[0] <= line <= CALENDAR_SPAN[1]:
                span_name, start_line = "calendar", CALENDAR_SPAN[0]
            elif LANDUSE_SPAN[0] <= line <= LANDUSE_SPAN[1]:
                span_name, start_line = "landuse", LANDUSE_SPAN[0]
            elif HISTORY_SPAN[0] <= line <= HISTORY_SPAN[1]:
                span_name, start_line = "history", HISTORY_SPAN[0]
            else:
                raise RuntimeError(f"no generated span for {arm_id}")
            generated_line = generated_starts[span_name] + line - start_line
            branch_rows = raw_branches.get(generated_line, [])
            if not branch_rows and ":elsewhere:" in arm_id:
                branch_rows = raw_branches.get(generated_line - 2, [])
            branch = select_arm_branch(arm_id, branch_rows)
            arm_records.append(
                {
                    "arm_id": arm_id,
                    "base_region_id": arm["base_region_id"],
                    "procedure": "slowproc_main",
                    "original_line": line,
                    "generated_line": generated_line,
                    "gcov_branch_index": -1 if branch is None else branch["branch_index"],
                    "gcov_taken": 0 if branch is None else branch["taken"],
                    "procedure_span_sha256": span_hashes[span_name]["span_sha256"],
                    "passed": bool(branch and branch["taken"] > 0),
                    "raw_gcov_branches": branch_rows,
                }
            )

    gcov_arm_ids = {record["arm_id"] for record in arm_records if record["passed"]}
    combined_passed = gcov_arm_ids | passed_source_proofs
    branch_coverage = {
        "schema_version": 1,
        "family": FAMILY,
        "branch_complete": all(str(record["arm_id"]) in combined_passed for record in target_arms),
        "required_arm_count": len(target_arms),
        "covered_arm_count": sum(str(record["arm_id"]) in combined_passed for record in target_arms),
        "gcov_arm_count": len(gcov_arm_ids),
        "source_proof_arm_count": sum(str(record["arm_id"]) in passed_source_proofs for record in target_arms),
        "missing_arm_ids": sorted(str(record["arm_id"]) for record in target_arms if str(record["arm_id"]) not in combined_passed),
        "arms": arm_records,
        "compiler": str(compiler),
        "gcov": str(gcov),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "branch_coverage.json").write_text(json.dumps(branch_coverage, indent=2) + "\n", encoding="ascii")

    contract_by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in target_arms:
        contract_by_region[str(record["base_region_id"])].append(record)

    evidence_records = []
    for region_id, owner in sorted(owner_regions.items()):
        required = {str(record["arm_id"]) for record in contract_by_region[region_id]}
        source_backed = required & passed_source_proofs
        numerical = required & gcov_arm_ids
        covered = source_backed | numerical
        evidence_records.append(
            {
                "owner_region_id": owner["owner_region_id"],
                "base_region_id": region_id,
                "fortran_procedure": owner["fortran_procedure"],
                "jax_owners": owner["jax_owners"],
                "comparison_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/comparison.json",
                "branch_coverage_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/branch_coverage.json",
                "required_arm_ids": sorted(required),
                "gcov_arm_ids": sorted(numerical),
                "source_proof_arm_ids": sorted(source_backed),
                "covered_arm_ids": sorted(covered),
                "missing_arm_ids": sorted(required - covered),
                "passed": required == covered,
            }
        )
    owner_evidence = {
        "schema_version": 1,
        "family": FAMILY,
        "complete": all(record["passed"] for record in evidence_records),
        "records": evidence_records,
    }
    (output_dir / "owner_region_evidence.json").write_text(json.dumps(owner_evidence, indent=2) + "\n", encoding="ascii")
    if not branch_coverage["branch_complete"] or not owner_evidence["complete"]:
        raise RuntimeError(f"{FAMILY} owner coverage incomplete: {branch_coverage['missing_arm_ids']}")
    return {"branch_coverage": branch_coverage, "owner_evidence": owner_evidence}


if __name__ == "__main__":
    result = run_coverage()
    print(json.dumps({"status": "passed", "covered_arms": result["branch_coverage"]["covered_arm_count"]}, indent=2))
