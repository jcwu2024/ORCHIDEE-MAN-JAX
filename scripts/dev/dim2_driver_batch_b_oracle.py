"""Formal closure for the four Stage 4 Batch B DIM2 driver owners.

The runner extracts immutable owner bytes and minimal compilable source
segments from dim2_driver.f90.  The executable fixture uses the original
Fortran decisions; only getin/ipslerr transport is supplied by the harness.
The wider state/call contract is checked by the focused production tests.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_fortran_micro_oracle import extract_source_fragments  # noqa: E402
from fortran_gcov import (  # noqa: E402
    locate_generated_span,
    parse_gcov,
    parse_gcov_line_counts,
    select_arm_branch,
)
from fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    exact_comparison,
    float_comparison,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90"
CONTRACTS = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
AUDIT = ROOT / "scripts/dev/audit_pft14_owner_region_evidence.py"
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/dim2_driver_batch_b_formal"
TEMPLATE = ROOT / "scripts/dev/dim2_driver_batch_b_harness.f90.template"
FAMILY = "dim2_driver_batch_b_formal"

OWNERS = {
    "pft14-owner-contract-8b2f034e7b0a": ["600", "639-645", "659-676"],
    "pft14-owner-contract-9d83d36c3d1b": ["361-401", "827-931", "1441-1445"],
    "pft14-owner-contract-2f2f83908ced": ["443-458", "569-572"],
    "pft14-owner-contract-f6d7f1a4e631": [
        "285-311",
        "379-390",
        "415-430",
        "470-490",
        "519-529",
        "724-806",
        "835-914",
        "995-1008",
        "1051",
        "1141-1158",
        "1403-1426",
    ],
}

# Exact source segments inserted into the deterministic executable.  Segments
# ending inside a surrounding loop are narrowed to the complete decision.
SEGMENTS = {
    "TIME_STEP": ["285-311"],
    "ROOT_RESTART": ["361-401"],
    "LANDPOINT_GUARD": ["424-430"],
    "DIMENSIONS": ["443-458"],
    "CHRONOLOGY": ["470-490"],
    "SPLIT_START": ["519-529"],
    "TIME_GUARD": ["569-572"],
    "CONTROLS": ["598-696"],
    "FALLBACKS": ["724-796"],
    "FORCING_INDEX": ["806-810"],
    "LAST_STEP": ["835-837"],
    "ROOT_PRINT": ["827-831"],
    "WATCHOUT_PREP": ["914-921"],
    "RELAXATION": ["931-939"],
    "PBL_DEFAULTS": ["995-1004", "1007-1008"],
    "INITIAL_STEP": ["1051-1058"],
    "ALBEDO_RESTART": ["1139-1157"],
    "LOAD_BALANCE": ["1403-1407"],
    "RESTART_WRITES": ["1415-1430"],
    "ROOT_CLOSE": ["1441-1446"],
}

ERROR_BOUNDARY_WITNESSES = {
    # The source call is a continued statement; GNU gcov assigns execution to
    # its terminal continuation line 571.
    "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90:569:if:true": 571,
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _contracts() -> dict[str, dict[str, Any]]:
    document = json.loads(CONTRACTS.read_text(encoding="ascii"))
    wanted = set(OWNERS)
    records = {
        item["owner_region_id"]: item
        for item in document["owner_regions"]
        if item["owner_region_id"] in wanted
    }
    if set(records) != wanted:
        raise AssertionError("canonical contract asset does not contain exactly four DIM2 owners")
    return records


def _compose(output_dir: Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    data = TEMPLATE.read_bytes()
    metadata: dict[str, dict[str, Any]] = {}
    segment_dir = output_dir / "minimal_original_call_segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    for marker, spans in SEGMENTS.items():
        extracted = extract_source_fragments(SOURCE, spans)
        token = f"! <{marker}>".encode("ascii")
        if data.count(token) != 1:
            raise AssertionError(f"template marker drift: {marker}")
        data = data.replace(token, extracted.span_bytes)
        target = segment_dir / f"{marker.lower()}.f90"
        target.write_bytes(extracted.span_bytes)
        metadata[marker] = {
            "source_fragments": spans,
            "start_line": extracted.start_line,
            "end_line": extracted.end_line,
            "span_sha256": extracted.span_sha256,
            "byte_count": len(extracted.span_bytes),
            "asset": target.relative_to(ROOT).as_posix(),
        }
    if b"! <" in data:
        raise AssertionError("unreplaced source marker")
    target = output_dir / "composed_oracle.f90"
    target.write_bytes(data)
    return target, metadata


def _extract_owners(output_dir: Path) -> dict[str, dict[str, Any]]:
    target_dir = output_dir / "extracted_original_bytes"
    target_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for owner_id, spans in OWNERS.items():
        extracted = extract_source_fragments(SOURCE, spans)
        target = target_dir / f"{owner_id}.f90"
        target.write_bytes(extracted.span_bytes)
        records[owner_id] = {
            "source_file": SOURCE.relative_to(ROOT).as_posix(),
            "source_sha256": extracted.source_sha256,
            "source_fragments": spans,
            "start_line": extracted.start_line,
            "end_line": extracted.end_line,
            "span_sha256": extracted.span_sha256,
            "byte_count": len(extracted.span_bytes),
            "extracted_file": target.relative_to(ROOT).as_posix(),
        }
    return records


def _load_fortran(path: Path) -> dict[str, np.ndarray]:
    rows: dict[str, list[list[float]]] = {}
    for raw in path.read_text(encoding="ascii").splitlines():
        fields = raw.split()
        rows.setdefault(fields[0], []).append([float(value) for value in fields[1:]])
    return {name: np.asarray(values, dtype=np.float64) for name, values in rows.items()}


def _collect_gcov(
    build: Path, output_dir: Path, compiler: Path
) -> dict[str, Any]:
    gcov = compiler.with_name("gcov.exe")
    notes = sorted(build.glob("*.gcno"))
    if not gcov.is_file() or not notes:
        raise FileNotFoundError("coverage compiler did not provide gcov/.gcno")
    completed = subprocess.run(
        [str(gcov), "-b", "-c", notes[0].name],
        cwd=build,
        env=compiler_environment(compiler),
        text=True,
        capture_output=True,
        check=True,
    )
    reports = []
    for source in sorted(build.glob("*.gcov")):
        target = output_dir / source.name
        shutil.copy2(source, target)
        reports.append(target.relative_to(ROOT).as_posix())
    if not reports:
        raise FileNotFoundError("gcov did not emit a branch report")
    return {
        "executable": str(gcov),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "reports": reports,
    }


def _generated_line(
    original_line: int, generated_bytes: bytes
) -> tuple[str, str, int]:
    for marker, spans in SEGMENTS.items():
        for raw_span in spans:
            bounds = [int(value) for value in raw_span.split("-")]
            start, end = bounds[0], bounds[-1]
            if not start <= original_line <= end:
                continue
            extracted = extract_source_fragments(SOURCE, [raw_span])
            generated_start = locate_generated_span(
                generated_bytes, extracted.span_bytes
            )
            return marker, raw_span, generated_start + original_line - start
    raise RuntimeError(f"no compiled source segment contains original line {original_line}")


def _mapped_coverage(
    owner_contracts: dict[str, dict[str, Any]],
    generated_source: Path,
    gcov_report: Path,
    gcov_metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    generated_bytes = generated_source.read_bytes()
    branches = parse_gcov(gcov_report)
    line_counts = parse_gcov_line_counts(gcov_report)
    arm_records = []
    for owner_id, owner in owner_contracts.items():
        for arm_id in owner["arm_ids"]:
            original_line = int(arm_id.rsplit(":", 3)[-3])
            marker, source_span, generated_line = _generated_line(
                original_line, generated_bytes
            )
            rows = branches.get(generated_line, [])
            branch = select_arm_branch(arm_id, rows)
            witness_original = ERROR_BOUNDARY_WITNESSES.get(arm_id)
            witness_generated = None
            witness_count = None
            if witness_original is not None:
                _, _, witness_generated = _generated_line(
                    witness_original, generated_bytes
                )
                witness_count = line_counts.get(witness_generated, 0)
            gcov_passed = bool(branch is not None and branch["taken"] > 0)
            error_boundary_passed = bool(
                witness_count is not None and witness_count > 0
            )
            arm_records.append(
                {
                    "owner_region_id": owner_id,
                    "base_region_id": owner["base_region_id"],
                    "arm_id": arm_id,
                    "original_line": original_line,
                    "source_segment": marker,
                    "source_span": source_span,
                    "generated_line": generated_line,
                    "gcov_branch_index": -1
                    if branch is None
                    else branch["branch_index"],
                    "gcov_taken": 0 if branch is None else branch["taken"],
                    "raw_gcov_branches": rows,
                    "error_boundary_witness_original_line": witness_original,
                    "error_boundary_witness_generated_line": witness_generated,
                    "error_boundary_witness_count": witness_count,
                    "gcov_passed": gcov_passed,
                    "error_boundary_passed": error_boundary_passed,
                    "passed": gcov_passed or error_boundary_passed,
                }
            )

    required_ids = [record["arm_id"] for record in arm_records]
    covered_ids = [record["arm_id"] for record in arm_records if record["passed"]]
    missing_ids = [record["arm_id"] for record in arm_records if not record["passed"]]
    coverage = {
        "schema_version": 1,
        "family": FAMILY,
        "branch_complete": not missing_ids,
        "required_arm_count": len(required_ids),
        "covered_arm_count": len(covered_ids),
        "required_arm_ids": required_ids,
        "covered_arm_ids": covered_ids,
        "missing_arm_ids": missing_ids,
        "arms": arm_records,
        "gcov": gcov_metadata,
        "mapping_policy": (
            "GNU Fortran -O0 --coverage branch counts mapped from each original "
            "source fragment to its unique generated span; explicit ipslerr "
            "execution lines witness error boundaries."
        ),
    }
    evidence_records = []
    for owner_id, owner in sorted(owner_contracts.items()):
        owner_arms = [
            record for record in arm_records if record["owner_region_id"] == owner_id
        ]
        required = {record["arm_id"] for record in owner_arms}
        gcov_arms = {
            record["arm_id"] for record in owner_arms if record["gcov_passed"]
        }
        boundary_arms = {
            record["arm_id"]
            for record in owner_arms
            if record["error_boundary_passed"]
        }
        covered = gcov_arms | boundary_arms
        evidence_records.append(
            {
                "owner_region_id": owner_id,
                "base_region_id": owner["base_region_id"],
                "fortran_procedure": owner["fortran_procedure"],
                "jax_owners": owner["jax_owners"],
                "comparison_asset": (
                    OUTPUT / "comparison.json"
                ).relative_to(ROOT).as_posix(),
                "branch_coverage_asset": (
                    OUTPUT / "branch_coverage.json"
                ).relative_to(ROOT).as_posix(),
                "required_arm_ids": sorted(required),
                "gcov_arm_ids": sorted(gcov_arms),
                "error_boundary_arm_ids": sorted(boundary_arms),
                "source_proof_arm_ids": [],
                "covered_arm_ids": sorted(covered),
                "missing_arm_ids": sorted(required - covered),
                "passed": bool(required and required == covered),
            }
        )
    evidence = {
        "schema_version": 1,
        "family": FAMILY,
        "complete": bool(evidence_records)
        and all(record["passed"] for record in evidence_records),
        "records": evidence_records,
    }
    return coverage, evidence


def _expected() -> dict[str, np.ndarray]:
    from jax_orchidee.driver.dim2 import (
        Dim2RestartTime,
        resolve_dim2_forcing_controls,
        resolve_dim2_time_control,
    )
    from jax_orchidee.driver.driver_forcing_completion import (
        driver_output_status,
        validate_driver_dimensions,
    )

    controls = []
    arguments = (
        dict(dt_force=7200.0, dt=1800.0, split=4),
        dict(
            dt_force=21600.0,
            dt=1800.0,
            split=12,
            inter_lin=True,
            spread_override=99,
            netrad_cons_override=False,
        ),
        dict(
            dt_force=21600.0,
            dt=21600.0,
            split=1,
            inter_lin=True,
            spread_override=2,
        ),
        dict(dt_force=3600.0, dt=1800.0, split=2, inter_lin=True),
        dict(dt_force=1800.0, dt=1800.0, split=1, inter_lin=True),
    )
    for case_id, argument in enumerate(arguments, 1):
        value = resolve_dim2_forcing_controls(**argument)
        controls.append(
            [case_id, value.no_inter, value.inter_lin, value.nb_spread, value.netrad_cons]
        )

    time_cases = (
        resolve_dim2_time_control(
            dt_force=21600.0,
            dt_sechiba=1800.0,
            forcing_length=5,
            date0=1.0,
        ),
        resolve_dim2_time_control(
            dt_force=21600.0,
            dt_sechiba=1800.0,
            forcing_length=5,
            date0=1.0,
            restart=Dim2RestartTime(
                itau_dep_rest=7, date0_rest=3.0, dt_rest=900.0
            ),
        ),
        resolve_dim2_time_control(
            dt_force=21600.0,
            dt_sechiba=1800.0,
            forcing_length=5,
            date0=1.0,
            restart=Dim2RestartTime(
                itau_dep_rest=7,
                date0_rest=3.0,
                dt_rest=900.0,
                driver_reset_time=True,
            ),
        ),
        resolve_dim2_time_control(
            dt_force=21600.0,
            dt_sechiba=1800.0,
            forcing_length=5,
            date0=1.0,
        ),
    )
    chronology = np.asarray(
        [
            [
                index,
                item.itau_dep,
                item.itau_fin,
                item.date0_rest,
                item.dt_rest,
                item.for_offset,
                item.split_start,
            ]
            for index, item in enumerate(time_cases, 1)
        ],
        dtype=np.float64,
    )

    # Exercise the production error boundaries as part of the deterministic
    # fixture.  The Fortran rows carry the ipslerr severity/count.
    errors = []
    for index, args in enumerate(
        (
            ((2, 2, 3), (2, 2, 3), 4, 4),
            ((2, 2, 3), (9, 8, 7), 4, 4),
            ((2, 2, 3), (2, 2, 3), 2, 3),
            ((1, 1, 1), (1, 1, 1), 4, 4),
        ),
        1,
    ):
        failed = 0
        try:
            validate_driver_dimensions(
                forcing_shape=args[0],
                model_shape=args[1],
                forcing_steps=args[2],
                requested_steps=args[3],
            )
        except ValueError:
            failed = 1
        errors.append([index, failed])

    output = []
    for index, (watchout, root) in enumerate(
        ((False, True), (True, True), (False, False)), 1
    ):
        status = driver_output_status(is_watchout=watchout, is_root_prc=root)
        output.append(
            [
                index,
                status.write_standard_restart_fields,
                status.write_pbl_restart_fields,
                status.close_forcing_history,
                status.close_restart_and_dump_config,
            ]
        )
    return {
        "controls": np.asarray(controls, dtype=np.float64),
        "chronology": chronology,
        "dimension_errors": np.asarray(errors, dtype=np.float64),
        "output_status": np.asarray(output, dtype=np.float64),
    }


def _run_tests() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/unit/test_dim2_driver.py",
        "tests/unit/test_driver_forcing_completion.py",
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "passed": completed.returncode == 0,
    }


def _run_formal_audit(
    output_dir: Path, evidence: dict[str, Any], owners: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    evidence_root = output_dir / "formal_audit_evidence"
    family_dir = evidence_root / FAMILY
    family_dir.mkdir(parents=True, exist_ok=True)
    (family_dir / "owner_region_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="ascii"
    )
    projection = {
        "schema_version": 1,
        "owner_regions": list(owners.values()),
    }
    contract_path = output_dir / "target_contract_classes.json"
    contract_path.write_text(
        json.dumps(projection, indent=2) + "\n", encoding="ascii"
    )
    source_proofs = output_dir / "empty_source_proofs.json"
    source_proofs.write_text(
        '{"schema_version": 1, "records": []}\n', encoding="ascii"
    )
    report = output_dir / "formal_audit_report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(AUDIT),
            "--contract-classes",
            str(contract_path),
            "--source-proofs",
            str(source_proofs),
            "--evidence-root",
            str(evidence_root),
            "--output",
            str(report),
            "--require-complete",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(report.read_text(encoding="ascii"))
    result["stdout"] = completed.stdout
    return result


def run_oracle(
    output_dir: Path = OUTPUT, compiler: Path = DEFAULT_COMPILER
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    owner_contracts = _contracts()
    provenance = _extract_owners(output_dir)
    composed, segments = _compose(output_dir)
    fortran_output = output_dir / "fortran_outputs.txt"
    with tempfile.TemporaryDirectory(prefix="orchidee_dim2_driver_batch_b_") as td:
        build = Path(td)
        source = build / composed.name
        executable = build / "dim2_driver_batch_b_oracle.exe"
        shutil.copy2(composed, source)
        compiler_metadata = compile_fortran(
            source,
            executable,
            compiler,
            extra_flags=("-cpp", "-fprofile-arcs", "-ftest-coverage"),
        )
        subprocess.run(
            [str(executable), str(fortran_output.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            text=True,
            capture_output=True,
            check=True,
        )
        gcov_metadata = _collect_gcov(build, output_dir, compiler)

    actual = _load_fortran(fortran_output)
    expected = _expected()
    comparisons = []
    for name in sorted(expected):
        if name not in actual:
            raise AssertionError(f"Fortran fixture omitted {name}")
        comparisons.append(
            float_comparison(
                name, actual[name], expected[name], rtol=1e-12, atol=1e-14
            )
        )
        comparisons.append(
            exact_comparison(
                f"{name}.shape",
                np.asarray(actual[name].shape, dtype=np.int64),
                np.asarray(expected[name].shape, dtype=np.int64),
            )
        )
    tests = _run_tests()
    comparisons.append(
        {
            "name": "focused_dim2_state_mask_diagnostic_error_tests",
            "comparison": "pytest",
            "passed": tests["passed"],
        }
    )
    gcov_report = ROOT / gcov_metadata["reports"][0]
    coverage, evidence = _mapped_coverage(
        owner_contracts, composed, gcov_report, gcov_metadata
    )
    (output_dir / "branch_coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="ascii"
    )
    for record in evidence["records"]:
        owner_id = record["owner_region_id"]
        record["extracted_owner_asset"] = provenance[owner_id]["extracted_file"]
        record["owner_span_sha256"] = provenance[owner_id]["span_sha256"]
    evidence["transport_runtime_stubs"] = [
        "getin_p",
        "ipslerr_p",
        "restini/restget_p/restput_p/restclo",
        "bcast",
        "forcing_READ",
        "timer/history close transport",
    ]
    evidence["source_provenance"] = provenance
    evidence_path = output_dir / "owner_region_evidence.json"
    incomplete_path = output_dir / "owner_region_evidence.incomplete.json"
    for path in (evidence_path, incomplete_path):
        if path.exists():
            path.unlink()
    audit = None
    if evidence["complete"]:
        evidence_path.write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="ascii"
        )
        audit = _run_formal_audit(output_dir, evidence, owner_contracts)
    else:
        incomplete_path.write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="ascii"
        )
    passed = (
        all(item["passed"] for item in comparisons)
        and evidence["complete"]
        and audit is not None
        and audit["complete"]
        and audit["counts"]["passed_owner_regions"] == 4
    )
    result = {
        "schema_version": 2,
        "family": FAMILY,
        "status": "passed" if passed else "failed",
        "comparisons": comparisons,
        "source_file_sha256": _sha256(SOURCE.read_bytes()),
        "source_provenance": provenance,
        "minimal_original_call_segments": segments,
        "compiler": compiler_metadata,
        "gcov": gcov_metadata,
        "coverage_complete": coverage["branch_complete"],
        "coverage_missing_arm_ids": coverage["missing_arm_ids"],
        "formal_audit": {
            "complete": False if audit is None else audit["complete"],
            "passed_owner_regions": 0 if audit is None else audit["counts"]["passed_owner_regions"],
            "required_owner_regions": 4 if audit is None else audit["counts"]["required_owner_regions"],
            "asset": (output_dir / "formal_audit_report.json")
            .relative_to(ROOT)
            .as_posix(),
        },
        "tolerance": {"rtol": 1e-12, "atol": 1e-14, "discrete": "exact"},
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="ascii"
    )
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_file": SOURCE.relative_to(ROOT).as_posix(),
                "owners": sorted(OWNERS),
                "owner_source_fragments": OWNERS,
                "minimal_original_call_segments": SEGMENTS,
                "fixture_policy": "only external transport/runtime stubs",
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    if not passed:
        raise RuntimeError(
            "formal DIM2 Batch B owner closure incomplete; "
            f"missing arms={coverage['missing_arm_ids']}"
        )
    return result


if __name__ == "__main__":
    result = run_oracle()
    print(json.dumps(result, indent=2))
