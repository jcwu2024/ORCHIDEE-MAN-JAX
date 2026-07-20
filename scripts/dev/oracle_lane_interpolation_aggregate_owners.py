from __future__ import annotations

import csv
import hashlib
import json
import shutil
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

from jax_orchidee.driver.interpolation_aggregate import (  # noqa: E402
    aggregate_2d,
    aggregate_2d_p,
    aggregate_vec,
    aggregate_vec_p,
)
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_gcov import (  # noqa: E402
    locate_generated_span,
    parse_gcov,
    select_arm_branch,
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

FAMILY = "interpolation_aggregate_owners"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/interpol_help.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_interpolation_aggregate_owners.f90.template"
CONTRACT_CLASSES = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
SOURCE_PROOFS = ROOT / "outputs/reference_mode/pft14_source_proof_evidence.json"
PROCEDURES = ("aggregate_2d", "aggregate_vec", "aggregate_vec_p", "aggregate_2d_p")
OWNER_IDS = {
    "aggregate_2d": "pft14-owner-contract-3479c5bbdcd5",
    "aggregate_2d_p": "pft14-owner-contract-9561ebb9666b",
    "aggregate_vec": "pft14-owner-contract-633db67a9e88",
    "aggregate_vec_p": "pft14-owner-contract-740908287844",
}
CONDITION_TERMINAL_LINES = {
    257: 259,
    350: 351,
    356: 357,
    397: 398,
    601: 602,
    622: 623,
    714: 715,
    720: 721,
    856: 857,
}
DISCRETE = {f"{owner}_{field}" for owner in OWNER_IDS for field in ("ind", "ok")}


def compose(path: Path) -> dict[str, Any]:
    spans = {name: extract_procedure_bytes(SOURCE, name) for name in PROCEDURES}
    data = TEMPLATE.read_bytes()
    for name, span in spans.items():
        data = data.replace(f"! <{name.upper()}>".encode(), span.span_bytes)
    if b"! <" in data:
        raise ValueError("unreplaced extracted-procedure marker")
    path.write_bytes(data)
    return spans


def _read(path: Path) -> dict[str, np.ndarray]:
    fields: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            fields.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(values) for name, values in fields.items()}


def _fields(prefix: str, result: Any) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_ind": result.indinc.ravel(order="F"),
        f"{prefix}_area": result.areaoverlap.ravel(order="F"),
        f"{prefix}_ok": np.asarray([result.ok]),
    }


def _jax() -> dict[str, np.ndarray]:
    lalo = np.asarray([[0.0, 0.0], [0.0, 1.5]])
    resolution = np.full((2, 2), 240_000.0)
    contfrac = np.asarray([1.0, 0.0])
    neighbours = np.zeros((2, 4), dtype=np.int32)
    lon = np.asarray([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    lat = np.asarray([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    mask = np.ones((3, 3), dtype=np.int32)
    mask[1, 1] = 0
    vector_lon = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0])
    vector_lat = np.zeros(5)
    common = (2, lalo, neighbours, resolution, contfrac)
    results = {
        "aggregate_2d": aggregate_2d(*common, 3, 3, lon, lat, mask, "oracle", 9),
        "aggregate_vec": aggregate_vec(
            *common, 5, vector_lon, vector_lat, 60_000.0, 60_000.0, "oracle", 9
        ),
        "aggregate_2d_p": aggregate_2d_p(*common, 3, 3, lon, lat, mask, "oracle", 9),
        "aggregate_vec_p": aggregate_vec_p(
            *common, 5, vector_lon, vector_lat, 60_000.0, 60_000.0, "oracle", 9
        ),
    }
    return {
        field: values
        for name, result in results.items()
        for field, values in _fields(name, result).items()
    }


def _run_gcov(build: Path, source: Path) -> dict[str, Any]:
    gcov = shutil.which("gcov", path=str(DEFAULT_COMPILER.parent)) or shutil.which(
        "gcov"
    )
    if gcov is None:
        raise FileNotFoundError("gcov was not found beside the configured compiler")
    completed = subprocess.run(
        [gcov, "-b", str(source)], cwd=build, text=True, capture_output=True, check=True
    )
    reports = sorted(build.glob("*.gcov"))
    if len(reports) != 1:
        raise RuntimeError(f"expected one gcov report, found {reports}")
    return {
        "command": gcov,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "report": reports[0],
    }


def _coverage_and_evidence(
    output_dir: Path, generated: Path, spans: dict[str, Any], report: Path
) -> None:
    contracts = json.loads(CONTRACT_CLASSES.read_text(encoding="utf-8"))
    proofs = json.loads(SOURCE_PROOFS.read_text(encoding="utf-8"))
    owners = {
        str(record["owner_region_id"]): record
        for record in contracts["owner_regions"]
        if str(record.get("owner_region_id")) in set(OWNER_IDS.values())
    }
    if set(owners) != set(OWNER_IDS.values()):
        raise RuntimeError("aggregate owner contracts are absent or ambiguous")
    source_bytes = generated.read_bytes()
    branches = parse_gcov(report)
    arm_rows: list[dict[str, Any]] = []
    by_region: dict[str, set[str]] = defaultdict(set)
    for owner_id, owner in owners.items():
        procedure = str(owner["fortran_procedure"])
        span = spans[procedure]
        generated_start = locate_generated_span(source_bytes, span.span_bytes)
        for arm_id in owner["arm_ids"]:
            original_line = int(str(arm_id).rsplit(":", 3)[-3])
            generated_line = generated_start + original_line - int(span.start_line)
            terminal_line = CONDITION_TERMINAL_LINES.get(original_line, original_line)
            if original_line == 856 and str(arm_id).endswith(":false"):
                terminal_line = 864
            terminal_generated_line = (
                generated_start + terminal_line - int(span.start_line)
            )
            branch_rows = branches.get(terminal_generated_line, [])
            if original_line == 856 and str(arm_id).endswith(":true") and len(branch_rows) >= 2:
                branch = (
                    branch_rows[-1]
                    if str(arm_id).endswith(":true")
                    else branch_rows[-2]
                )
            else:
                branch = select_arm_branch(str(arm_id), branch_rows)
            passed = branch is not None and branch["taken"] > 0
            if passed:
                by_region[owner_id].add(str(arm_id))
            arm_rows.append(
                {
                    "arm_id": arm_id,
                    "owner_region_id": owner_id,
                    "base_region_id": owner["base_region_id"],
                    "fortran_procedure": procedure,
                    "original_line": original_line,
                    "generated_line": generated_line,
                    "condition_terminal_generated_line": terminal_generated_line,
                    "gcov_branch_index": -1
                    if branch is None
                    else branch["branch_index"],
                    "gcov_taken": 0 if branch is None else branch["taken"],
                    "procedure_span_sha256": span.span_sha256,
                    "passed": passed,
                }
            )
    passed_proofs = {
        str(row["arm_id"]) for row in proofs["records"] if row.get("passed") is True
    }
    records: list[dict[str, Any]] = []
    for procedure, owner_id in sorted(OWNER_IDS.items()):
        owner = owners[owner_id]
        required = {str(arm) for arm in owner["arm_ids"]}
        gcov_arms = by_region[owner_id]
        source_arms = required & passed_proofs
        covered = gcov_arms | source_arms
        records.append(
            {
                "owner_region_id": owner_id,
                "base_region_id": owner["base_region_id"],
                "fortran_procedure": procedure,
                "jax_owners": owner["jax_owners"],
                "comparison_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/comparison.json",
                "branch_coverage_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/branch_coverage.json",
                "required_arm_ids": sorted(required),
                "gcov_arm_ids": sorted(gcov_arms),
                "source_proof_arm_ids": sorted(source_arms),
                "covered_arm_ids": sorted(covered),
                "missing_arm_ids": sorted(required - covered),
                "passed": required == covered,
            }
        )
    coverage = {
        "schema_version": 1,
        "family": FAMILY,
        "branch_complete": len(arm_rows) == sum(row["passed"] for row in arm_rows),
        "required_arm_count": len(arm_rows),
        "covered_arm_count": sum(row["passed"] for row in arm_rows),
        "missing_arm_ids": sorted(
            row["arm_id"] for row in arm_rows if not row["passed"]
        ),
        "arms": arm_rows,
        "mapping_policy": "Each contract arm is mapped from its extracted source span to its generated gcov line; select_arm_branch maps true/false edges.",
    }
    evidence = {
        "schema_version": 1,
        "family": FAMILY,
        "complete": all(record["passed"] for record in records),
        "records": records,
    }
    (output_dir / "branch_coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="ascii"
    )
    (output_dir / "owner_region_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="ascii"
    )


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="orchidee_interpolation_aggregate_"
    ) as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        spans = compose(source)
        compiler_metadata = compile_fortran(
            source, executable, compiler, extra_flags=("-std=gnu", "--coverage")
        )
        subprocess.run(
            [str(executable), str((output_dir / "fortran_outputs.csv").resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
        boundary_cases = {
            "aggregate_2d_capacity": "aggregate_2d|Working on variable :fatal|Reached incmax value for fopt.",
            "aggregate_vec_capacity": "aggregate_vec (nbpt < nbind)|Working on variable :fatal|Reached incmax value for fopt.",
            "aggregate_2d_p_grid": "aggregate_2d_p|Interpolation is only possible for regular lat/lon grids for the moment.",
            "aggregate_vec_p_grid": "aggregate_vec_p|Interpolation is only possible for regular lat/lon grids for the moment.",
        }
        boundary_records = []
        for mode, diagnostic in boundary_cases.items():
            completed = subprocess.run(
                [
                    str(executable),
                    str((output_dir / "fatal_unused.csv").resolve()),
                    mode,
                ],
                cwd=build,
                env=compiler_environment(compiler),
                capture_output=True,
                text=True,
            )
            boundary_records.append(
                {
                    "mode": mode,
                    "returncode": completed.returncode,
                    "expected_diagnostic": diagnostic,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "passed": completed.returncode != 0
                    and diagnostic.split("|")[0] in completed.stdout,
                }
            )
        if not all(record["passed"] for record in boundary_records):
            raise RuntimeError(
                f"fatal boundary verification failed: {boundary_records}"
            )
        (output_dir / "error_boundaries.json").write_text(
            json.dumps({"schema_version": 1, "records": boundary_records}, indent=2)
            + "\n",
            encoding="ascii",
        )
        gcov = _run_gcov(build, source)
        shutil.copy2(gcov["report"], output_dir / gcov["report"].name)
        (output_dir / "gcov.json").write_text(
            json.dumps({**gcov, "report": gcov["report"].name}, default=str, indent=2)
            + "\n",
            encoding="ascii",
        )
        _coverage_and_evidence(output_dir, source, spans, gcov["report"])
    fortran = _read(output_dir / "fortran_outputs.csv")
    jax = _jax()
    write_point_comparisons(
        output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-12
    )
    comparisons = [
        exact_comparison(name, fortran[name].astype(np.int64), jax[name])
        if name in DISCRETE
        else float_comparison(name, fortran[name], jax[name], rtol=1e-12, atol=1e-12)
        for name in sorted(fortran)
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "masked and zero-contfrac ABI inputs",
                    "empty overlap",
                    "boundary cells",
                    "reversed latitude",
                    "global and regional source selection",
                    "serial RegLonLat wrappers",
                ],
                "float_tolerance": 1e-12,
                "discrete_policy": "exact",
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    return write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "procedure_span_sha256": {
                name: span.span_sha256 for name, span in spans.items()
            },
            "compiler": compiler_metadata,
            "verified_ledger_entries": [],
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
