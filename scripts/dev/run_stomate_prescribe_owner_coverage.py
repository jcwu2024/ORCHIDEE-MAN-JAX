from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from fortran_gcov import locate_generated_span, parse_gcov, parse_gcov_line_counts, select_arm_branch  # noqa: E402
from fortran_oracle_common import DEFAULT_COMPILER, ROOT, compile_fortran, compiler_environment  # noqa: E402
from fortran_owner_coverage import CONTRACT_CLASSES, GCOV  # noqa: E402
from audit_pft14_owner_region_evidence import build_report, load_evidence_documents  # noqa: E402
from oracle_lane_stomate_carbon import compose_prescribe_oracle, run_prescribe_oracle  # noqa: E402

FAMILY = "stomate_prescribe_restart_gate"
OWNER_ID = "pft14-owner-contract-261c1baede4c"
SOURCE_FILE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_prescribe.f90"
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
ELSEWHERE_WITNESS = (
    "fortran_source/ORCHIDEE/src_stomate/stomate_prescribe.f90:242:elsewhere:fallthrough"
)


def run_coverage(
    output_dir: Path = OUTPUT,
    compiler: Path = DEFAULT_COMPILER,
    gcov: Path = GCOV,
) -> dict[str, object]:
    comparison = run_prescribe_oracle(output_dir, compiler)
    if comparison.get("status") != "passed":
        raise RuntimeError("prescribe numerical comparison did not pass")
    contracts = json.loads(CONTRACT_CLASSES.read_text(encoding="utf-8"))
    owner = next(
        row for row in contracts["owner_regions"] if row["owner_region_id"] == OWNER_ID
    )
    required = {str(arm_id) for arm_id in owner["arm_ids"]}
    arms = [
        row
        for row in contracts["arm_classifications"]
        if str(row["arm_id"]) in required
    ]
    if {str(row["arm_id"]) for row in arms} != required:
        raise RuntimeError("contract-class arm mapping is incomplete")

    with tempfile.TemporaryDirectory(prefix="orchidee_prescribe_gcov_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        compose_prescribe_oracle(source)
        executable = build / "oracle_gcov.exe"
        compile_fortran(source, executable, compiler, extra_flags=("--coverage",))
        subprocess.run(
            [str(executable), str(build / "fortran_outputs.csv")],
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
            check=True,
            capture_output=True,
            text=True,
        )
        listing = build / f"{source.name}.gcov"
        branches = parse_gcov(listing)
        line_counts = parse_gcov_line_counts(listing)
        extracted = extract_procedure_bytes(SOURCE_FILE, "prescribe")
        generated_start = locate_generated_span(source.read_bytes(), extracted.span_bytes)
        records: list[dict[str, object]] = []
        for arm in arms:
            arm_id = str(arm["arm_id"])
            original_line = int(arm["line"])
            generated_line = generated_start + original_line - extracted.start_line
            witness_line = 244 if arm_id == ELSEWHERE_WITNESS else None
            witness_generated = (
                None if witness_line is None else generated_start + witness_line - extracted.start_line
            )
            witness_count = None if witness_generated is None else line_counts.get(witness_generated, 0)
            branch_rows = branches.get(generated_line, [])
            branch = None if witness_line is not None else select_arm_branch(arm_id, branch_rows)
            gcov_passed = bool(
                witness_count > 0 if witness_count is not None else branch and branch["taken"] > 0
            )
            records.append(
                {
                    "arm_id": arm_id,
                    "base_region_id": owner["base_region_id"],
                    "procedure": "prescribe",
                    "original_line": original_line,
                    "generated_line": generated_line,
                    "gcov_branch_index": -1 if branch is None else branch["branch_index"],
                    "gcov_taken": 0 if branch is None else branch["taken"],
                    "execution_witness_original_line": witness_line,
                    "execution_witness_generated_line": witness_generated,
                    "execution_witness_count": witness_count,
                    "procedure_span_sha256": extracted.span_sha256,
                    "raw_gcov_branches": branch_rows,
                    "passed": gcov_passed,
                }
            )

    covered = {str(row["arm_id"]) for row in records if row["passed"]}
    missing = sorted(required - covered)
    coverage = {
        "schema_version": 1,
        "family": FAMILY,
        "owner_region_id": OWNER_ID,
        "branch_complete": not missing,
        "required_arm_count": len(required),
        "covered_arm_count": len(covered),
        "missing_arm_ids": missing,
        "arms": records,
        "mapping_policy": (
            "GCOV maps the byte-exact extracted prescribe procedure. The ELSEWHERE "
            "fallthrough is witnessed by its executable assignment at original line 244; "
            "all other canonical arms require a taken GCOV edge."
        ),
    }
    evidence_record = {
        "owner_region_id": OWNER_ID,
        "base_region_id": owner["base_region_id"],
        "fortran_procedure": "prescribe",
        "jax_owners": owner["jax_owners"],
        "comparison_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/comparison.json",
        "branch_coverage_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/branch_coverage.json",
        "required_arm_ids": sorted(required),
        "gcov_arm_ids": sorted(covered),
        "fatal_boundary_arm_ids": [],
        "source_proof_arm_ids": [],
        "covered_arm_ids": sorted(covered),
        "missing_arm_ids": missing,
        "passed": not missing,
    }
    evidence = {"schema_version": 1, "family": FAMILY, "complete": not missing, "records": [evidence_record]}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "branch_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="ascii")
    (output_dir / "owner_region_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="ascii")
    other_evidence = [
        item
        for item in load_evidence_documents(ROOT / "outputs/reference_mode/micro_oracles")
        if all(record.get("owner_region_id") != OWNER_ID for record in item[1].get("records", []))
    ]
    acceptance = build_report(
        contracts,
        json.loads((ROOT / "outputs/reference_mode/pft14_source_proof_evidence.json").read_text(encoding="utf-8")),
        [*other_evidence, (str(output_dir / "owner_region_evidence.json"), evidence)],
    )
    accepted = next(
        (row.get("passed") is True for row in acceptance["records"] if row["owner_region_id"] == OWNER_ID),
        False,
    )
    acceptance["assigned_owner_acceptance"] = {
        "owner_region_id": OWNER_ID,
        "accepted": accepted,
        "global_completion_not_required_by_this_local_owner_lane": True,
    }
    (output_dir / "central_audit_report.json").write_text(
        json.dumps(acceptance, indent=2) + "\n", encoding="ascii"
    )
    if missing or not accepted:
        raise RuntimeError(f"{OWNER_ID} incomplete GCOV coverage: {missing}")
    return {"branch_coverage": coverage, "owner_evidence": evidence}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run local prescribe owner GCOV evidence.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    parser.add_argument("--gcov", type=Path, default=GCOV)
    args = parser.parse_args(argv)
    try:
        result = run_coverage(args.output_dir.resolve(), args.compiler.resolve(), args.gcov.resolve())
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(json.dumps({"status": "passed", "covered_arms": result["branch_coverage"]["covered_arm_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
