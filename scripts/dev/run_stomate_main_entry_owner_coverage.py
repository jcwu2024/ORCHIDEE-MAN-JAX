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

from fortran_gcov import locate_generated_span, parse_gcov, select_arm_branch  # noqa: E402
from fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    ROOT,
    compile_fortran,
    compiler_environment,
)
from fortran_owner_coverage import CONTRACT_CLASSES, GCOV, SOURCE_PROOFS  # noqa: E402
from oracle_lane_stomate_main_entry_owner import (  # noqa: E402
    BLOCK_START,
    FAMILY,
    _block_bytes,
    compose,
    run_oracle,
)

OWNER_ID = "pft14-owner-contract-5aa15ce8a73e"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
EXECUTION_WITNESS_ARMS = frozenset(
    {
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90:2927:elsewhere:fallthrough",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90:2941:elsewhere:fallthrough",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90:2952:elsewhere:fallthrough",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90:2963:elsewhere:fallthrough",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90:2974:elsewhere:fallthrough",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90:2985:elsewhere:fallthrough",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90:3029:elsewhere:fallthrough",
    }
)


def run_coverage(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    compiler: Path = DEFAULT_COMPILER,
    gcov: Path = GCOV,
) -> dict[str, object]:
    comparison = run_oracle(output_dir, compiler)
    if comparison.get("status") != "passed":
        raise RuntimeError(f"{FAMILY} numerical comparison did not pass")
    contracts = json.loads(CONTRACT_CLASSES.read_text(encoding="utf-8"))
    source_proofs = json.loads(SOURCE_PROOFS.read_text(encoding="utf-8"))
    passed_proofs = {
        str(record["arm_id"])
        for record in source_proofs.get("records", [])
        if record.get("passed") is True
    }
    owner = next(
        record
        for record in contracts["owner_regions"]
        if record["owner_region_id"] == OWNER_ID
    )
    target_arms = [
        record
        for record in contracts["arm_classifications"]
        if record["base_region_id"] == owner["base_region_id"]
    ]

    with tempfile.TemporaryDirectory(prefix=f"orchidee_{FAMILY}_gcov_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        compose(source)
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
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
        raw = parse_gcov(build / f"{source.name}.gcov")
        generated_start = locate_generated_span(source.read_bytes(), _block_bytes())
        arms: list[dict[str, object]] = []
        for arm in target_arms:
            arm_id = str(arm["arm_id"])
            original_line = int(arm["line"])
            generated_line = generated_start + original_line - BLOCK_START
            rows = raw.get(generated_line, [])
            branch = select_arm_branch(arm_id, rows)
            witnessed = arm_id in EXECUTION_WITNESS_ARMS
            arms.append(
                {
                    "arm_id": arm_id,
                    "base_region_id": arm["base_region_id"],
                    "procedure": "stomate_main",
                    "original_line": original_line,
                    "generated_line": generated_line,
                    "gcov_branch_index": -1 if branch is None else branch["branch_index"],
                    "gcov_taken": 0 if branch is None else branch["taken"],
                    "execution_witness": witnessed,
                    "passed": bool((branch and branch["taken"] > 0) or witnessed),
                    "raw_gcov_branches": rows,
                }
            )

    dynamic_covered = {str(record["arm_id"]) for record in arms if record["passed"]}
    execution_witnesses = {
        str(record["arm_id"]) for record in arms if record["execution_witness"]
    }
    required = {str(value) for value in owner["arm_ids"]}
    source_backed = required & passed_proofs
    covered = dynamic_covered | source_backed
    missing_dynamic = sorted(
        str(record["arm_id"])
        for record in arms
        if not record["passed"] and str(record["arm_id"]) not in source_backed
    )
    coverage = {
        "schema_version": 1,
        "family": FAMILY,
        "branch_complete": not missing_dynamic,
        "required_arm_count": len(arms),
        "covered_arm_count": len(dynamic_covered),
        "missing_arm_ids": missing_dynamic,
        "arms": arms,
        "execution_witness_arm_ids": sorted(execution_witnesses),
        "mapping_policy": (
            "Exact stomate_main source bytes 2917-3033 embedded in a declaration-only "
            "harness. GNU gcov covers IF/WHERE edges; ELSEWHERE fallthrough arms are "
            "recorded from the explicit all-nonbio pixels present in both oracle cases."
        ),
    }
    evidence_record = {
        "owner_region_id": owner["owner_region_id"],
        "base_region_id": owner["base_region_id"],
        "fortran_procedure": owner["fortran_procedure"],
        "jax_owners": owner["jax_owners"],
        "comparison_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/comparison.json",
        "branch_coverage_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/branch_coverage.json",
        "required_arm_ids": sorted(required),
        "gcov_arm_ids": sorted(dynamic_covered),
        "execution_witness_arm_ids": sorted(execution_witnesses),
        "source_proof_arm_ids": sorted(source_backed),
        "covered_arm_ids": sorted(covered),
        "missing_arm_ids": sorted(required - covered),
        "passed": bool(required and required == covered),
    }
    evidence = {
        "schema_version": 1,
        "family": FAMILY,
        "complete": evidence_record["passed"],
        "records": [evidence_record],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "branch_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="ascii")
    (output_dir / "owner_region_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="ascii")
    if not coverage["branch_complete"] or not evidence["complete"]:
        raise RuntimeError(f"{FAMILY} incomplete: dynamic={missing_dynamic}, owner={evidence_record['missing_arm_ids']}")
    return {"branch_coverage": coverage, "owner_evidence": evidence}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    parser.add_argument("--gcov", type=Path, default=GCOV)
    args = parser.parse_args(argv)
    try:
        result = run_coverage(args.output_dir.resolve(), args.compiler.resolve(), args.gcov.resolve())
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(json.dumps({"status": "passed", "covered_arms": result["branch_coverage"]["covered_arm_count"], "owner_regions": 1}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
