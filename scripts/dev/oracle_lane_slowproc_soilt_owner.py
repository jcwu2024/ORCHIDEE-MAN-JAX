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

from jax_orchidee.driver.static import slowproc_soilt_from_explicit_overlap  # noqa: E402
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
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

FAMILY = "slowproc_soilt_owner"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_slowproc_soilt_owner.f90.template"
GCOV = Path(r"C:\msys64\ucrt64\bin\gcov.exe")
CONTRACT_CLASSES = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
SOURCE_PROOFS = ROOT / "outputs/reference_mode/pft14_source_proof_evidence.json"
OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY

READ_SPAN = (4377, 4450)
MASK_SPAN = (4457, 4555)
SELECT_SPAN = (4559, 4908)
SPANS = {
    "read": READ_SPAN,
    "mask": MASK_SPAN,
    "select": SELECT_SPAN,
}


def _span(start: int, end: int) -> bytes:
    lines = SOURCE.read_bytes().splitlines(keepends=True)
    return b"".join(lines[start - 1 : end])


def compose(path: Path) -> dict[str, Any]:
    payload = TEMPLATE.read_bytes()
    span_hash = {}
    for name, span in SPANS.items():
        fragment = _span(*span)
        payload = payload.replace(f"! <SLOWPROC_SOILT_{name.upper()}_SPAN>".encode(), fragment)
        span_hash[name] = {
            "start_line": span[0],
            "end_line": span[1],
            "span_sha256": hashlib.sha256(fragment).hexdigest(),
        }
    dependency = extract_procedure_bytes(SOURCE, "get_soilcorr_usda")
    payload = payload.replace(b"! <GET_SOILCORR_USDA_PROCEDURE>", dependency.span_bytes)
    span_hash["get_soilcorr_usda"] = {
        "start_line": dependency.start_line,
        "end_line": dependency.end_line,
        "span_sha256": dependency.span_sha256,
    }
    path.write_bytes(payload)
    return span_hash


def _read(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.reader(handle):
            values.setdefault(row[0], []).append(float(row[1]))
    return {name: np.asarray(value, dtype=np.float64) for name, value in values.items()}


def expected_outputs() -> dict[str, np.ndarray]:
    soiltext = np.asarray(
        [
            [1.0, 2.0, 0.0],
            [0.0, 4.0, 0.0],
            [13.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    soilbd = np.asarray(
        [
            [1400.0, 1500.0, 1450.0],
            [1500.0, 1600.0, 1500.0],
            [1700.0, 1550.0, 1520.0],
        ],
        dtype=np.float64,
    )
    soil_ph_source = np.asarray(
        [
            [6.5, 6.8, 6.9],
            [7.0, 7.2, 6.7],
            [5.5, 7.1, 6.6],
        ],
        dtype=np.float64,
    )
    poor_soils_source = np.asarray(
        [
            [0.1, 0.2, 0.0],
            [0.0, 0.3, 0.0],
            [0.2, 0.1, 0.0],
        ],
        dtype=np.float64,
    )
    sub_index = np.zeros((3, 2, 2), dtype=np.int64)
    sub_area = np.zeros((3, 2), dtype=np.float64)
    sub_index[1, 0] = [1, 1]
    sub_area[1, 0] = 0.25
    sub_index[1, 1] = [2, 2]
    sub_area[1, 1] = 0.75
    sub_index[2, 0] = [1, 3]
    sub_area[2, 0] = 0.6
    sub_index[2, 1] = [3, 1]
    sub_area[2, 1] = 0.4
    result = slowproc_soilt_from_explicit_overlap(
        soiltext,
        soilbd,
        soil_ph_source,
        poor_soils_source,
        sub_index,
        sub_area,
        soil_classif="usda",
        index_base=1,
        do_poor_soils=True,
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


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_slowproc_soilt_owner_") as td:
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
        json.dumps({"schema_version": 1, "cases": ["active_usda_retry_default", "impsoilt_none_coverage_only"]}, indent=2) + "\n",
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
        and record.get("fortran_procedure") == "slowproc_soilt"
        and record.get("owner_region_id") == "pft14-owner-contract-463396c23861"
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
        raise RuntimeError("slowproc_soilt numerical comparison did not pass")

    owner_regions, target_arms, passed_source_proofs = _collect_target_arms()
    gcov_targets = [record for record in target_arms if str(record["arm_id"]) not in passed_source_proofs]

    with tempfile.TemporaryDirectory(prefix="orchidee_slowproc_soilt_gcov_") as td:
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
        generated = source.read_bytes()
        generated_starts = {name: locate_generated_span(generated, _span(*span)) for name, span in SPANS.items()}
        arm_records = []
        for arm in gcov_targets:
            line = int(arm["line"])
            if READ_SPAN[0] <= line <= READ_SPAN[1]:
                span_name, start_line = "read", READ_SPAN[0]
            elif MASK_SPAN[0] <= line <= MASK_SPAN[1]:
                span_name, start_line = "mask", MASK_SPAN[0]
            elif SELECT_SPAN[0] <= line <= SELECT_SPAN[1]:
                span_name, start_line = "select", SELECT_SPAN[0]
            else:
                raise RuntimeError(f"no generated span for {arm['arm_id']}")
            generated_line = generated_starts[span_name] + line - start_line
            branch_rows = raw_branches.get(generated_line, [])
            branch = select_arm_branch(str(arm["arm_id"]), branch_rows)
            arm_records.append(
                {
                    "arm_id": str(arm["arm_id"]),
                    "base_region_id": arm["base_region_id"],
                    "procedure": "slowproc_soilt",
                    "original_line": line,
                    "generated_line": generated_line,
                    "gcov_branch_index": -1 if branch is None else branch["branch_index"],
                    "gcov_taken": 0 if branch is None else branch["taken"],
                    "procedure_span_sha256": span_hash[span_name]["span_sha256"],
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
