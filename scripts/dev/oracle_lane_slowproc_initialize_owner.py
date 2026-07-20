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

from jax_orchidee.sechiba.science_completion import slowproc_initialize_source_routed  # noqa: E402
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

FAMILY = "slowproc_initialize_owner"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_slowproc_initialize_owner.f90.template"
GCOV = Path(r"C:\msys64\ucrt64\bin\gcov.exe")
CONTRACT_CLASSES = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
SOURCE_PROOFS = ROOT / "outputs/reference_mode/pft14_source_proof_evidence.json"
OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY

INITIALIZE_SPAN = (257, 302)


def _span(start: int, end: int) -> bytes:
    lines = SOURCE.read_bytes().splitlines(keepends=True)
    return b"".join(lines[start - 1 : end])


def compose(path: Path) -> dict[str, Any]:
    fragment = _span(*INITIALIZE_SPAN)
    payload = TEMPLATE.read_bytes().replace(b"! <SLOWPROC_INITIALIZE_SPAN>", fragment)
    path.write_bytes(payload)
    return {
        "start_line": INITIALIZE_SPAN[0],
        "end_line": INITIALIZE_SPAN[1],
        "span_sha256": hashlib.sha256(fragment).hexdigest(),
    }


def _read(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.reader(handle):
            values.setdefault(row[0], []).append(float(row[1]))
    return {name: np.asarray(value, dtype=np.float64) for name, value in values.items()}


def expected_outputs() -> dict[str, np.ndarray]:
    pref_soil_veg = np.asarray([1] * 13 + [4], dtype=np.int32)
    ext_coeff = np.asarray([0.0] + [0.5] * 13, dtype=np.float64)
    height_presc = np.asarray(
        [0.0, 30.0, 30.0, 20.0, 20.0, 20.0, 15.0, 15.0, 15.0, 0.5, 0.6, 1.0, 1.0, 30.0],
        dtype=np.float64,
    )
    veget = np.zeros((1, 14), dtype=np.float64)
    veget[0, 13] = 0.97318184
    veget_max = np.zeros((1, 14), dtype=np.float64)
    veget_max[0, 13] = 1.0
    lai = np.zeros((1, 14), dtype=np.float64)
    lai[0, 13] = 3.6186761
    frac_age = np.zeros((1, 14, 4), dtype=np.float64)
    frac_age[0, 13, :] = [0.23069987, 0.256, 0.257, 0.25630013]
    initialized, qsintmax, _ = slowproc_initialize_source_routed(
        init_kwargs={
            "restart": {
                "veget": veget,
                "veget_max": veget_max,
                "frac_nobio": np.asarray([[0.0]], dtype=np.float64),
                "lai": lai,
                "height": height_presc[None, :],
                "frac_age": frac_age,
                "njsc": np.asarray([2], dtype=np.int32),
                "clay_frac": np.asarray([0.2], dtype=np.float64),
                "sand_frac": np.asarray([0.4], dtype=np.float64),
                "bulk_dens": np.asarray([1650.0], dtype=np.float64),
                "soil_ph": np.asarray([7.0], dtype=np.float64),
                "poor_soils": np.asarray([0.0], dtype=np.float64),
                "reinf_slope": np.asarray([0.0], dtype=np.float64),
                "growth_day": np.asarray([17.0], dtype=np.float64),
            },
            "veget_max_default": np.asarray([0.0] * 13 + [1.0], dtype=np.float64),
            "frac_nobio_default": 0.0,
            "height_presc": height_presc,
            "pref_soil_veg": pref_soil_veg,
            "ext_coeff_vegetfrac": ext_coeff,
            "nstm": 6,
            "diaglev": np.asarray([0.1, 0.3, 1.0], dtype=np.float64),
            "salinity_data": np.asarray([31.25], dtype=np.float64),
            "tide_height_data": np.asarray([[0.1, -0.2, 0.3]], dtype=np.float64),
        },
        dt_stomate=86400.0,
        dt_sechiba=1800.0,
        ok_stomate=True,
        qsintcst=0.1,
    )
    return {
        "active_lai": np.asarray(initialized.lai).ravel(order="F"),
        "active_veget": np.asarray(initialized.veget).ravel(order="F"),
        "active_veget_max": np.asarray(initialized.veget_max).ravel(order="F"),
        "active_frac_nobio": np.asarray(initialized.frac_nobio).ravel(order="F"),
        "active_soiltile": np.asarray(initialized.soiltile).ravel(order="F"),
        "active_reinf_slope": np.asarray(initialized.reinf_slope),
        "active_totfrac_nobio": np.asarray(initialized.totfrac_nobio),
        "active_tot_bare_soil": np.asarray(initialized.tot_bare_soil),
        "active_qsintmax": np.asarray(qsintmax).ravel(order="F"),
        "active_lcanop": np.asarray([float(initialized.lcanop)]),
        "active_veget_update": np.asarray([float(initialized.veget_update)]),
        "active_dt_days": np.asarray([1.0]),
    }


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_slowproc_initialize_owner_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        span_hash = compose(source)
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
    write_point_comparisons(output_dir / "point_comparisons.csv", actual, expected, rtol=1e-12, atol=1e-14)
    comparisons = [
        float_comparison(name, actual[name], expected[name], rtol=1e-12, atol=1e-14)
        for name in sorted(actual)
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps({"schema_version": 1, "cases": ["active_ok_stomate"]}, indent=2) + "\n",
        encoding="ascii",
    )
    return write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "span_sha256": span_hash,
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
        and record.get("fortran_procedure") == "slowproc_initialize"
        and record.get("owner_region_id") == "pft14-owner-contract-c436137b6218"
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


def run_coverage(
    output_dir: Path = OUTPUT_DIR,
    compiler: Path = DEFAULT_COMPILER,
    gcov: Path = GCOV,
) -> dict[str, Any]:
    comparison = run_oracle(output_dir, compiler)
    if comparison.get("status") != "passed":
        raise RuntimeError("slowproc_initialize numerical comparison did not pass")

    owner_regions, target_arms, passed_source_proofs = _collect_target_arms()
    gcov_targets = [record for record in target_arms if str(record["arm_id"]) not in passed_source_proofs]

    with tempfile.TemporaryDirectory(prefix="orchidee_slowproc_initialize_gcov_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        executable = build / "oracle_gcov.exe"
        span_hash = compose(source)
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
        generated_start = locate_generated_span(source.read_bytes(), _span(*INITIALIZE_SPAN))
        arm_records = []
        for arm in gcov_targets:
            line = int(arm["line"])
            generated_line = generated_start + line - INITIALIZE_SPAN[0]
            branch_rows = raw_branches.get(generated_line, [])
            branch = select_arm_branch(str(arm["arm_id"]), branch_rows)
            arm_records.append(
                {
                    "arm_id": str(arm["arm_id"]),
                    "base_region_id": arm["base_region_id"],
                    "procedure": "slowproc_initialize",
                    "original_line": line,
                    "generated_line": generated_line,
                    "gcov_branch_index": -1 if branch is None else branch["branch_index"],
                    "gcov_taken": 0 if branch is None else branch["taken"],
                    "procedure_span_sha256": span_hash["span_sha256"],
                    "passed": bool(branch and branch["taken"] > 0),
                    "raw_gcov_branches": branch_rows,
                }
            )

    target_ids = {str(record["arm_id"]) for record in target_arms}
    gcov_arm_ids = {record["arm_id"] for record in arm_records if record["passed"]}
    combined_passed = gcov_arm_ids | (passed_source_proofs & target_ids)
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
