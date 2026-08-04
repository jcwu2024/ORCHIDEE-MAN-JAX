from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "manifests/coarse_graining/teacher_physical_parameter_gradient_v1.json"
EVIDENCE_ROOT = (
    ROOT / "outputs/research/daily_coarse_graining/gate_d1_physical_parameter_gradients"
)
DEFAULT_OUTPUT = EVIDENCE_ROOT / "comparison.json"
TEACHER_SCOPES = {
    "complete_teacher_one_day": EVIDENCE_ROOT / "one_day_comparison.json",
    "complete_teacher_multiday": EVIDENCE_ROOT / "multiday_comparison.json",
    "complete_teacher_restart_year_boundary": EVIDENCE_ROOT / "restart_comparison.json",
}
FORTRAN_EVIDENCE = EVIDENCE_ROOT / "selected_fortran_finite_differences.json"


def _load(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate the complete Gate-D1 physical-parameter gradient evidence"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    current_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    checks: list[dict[str, object]] = []
    teacher_reports = {}
    for result_key, path in TEACHER_SCOPES.items():
        report = _load(path)
        cases = report.get(result_key, [])
        passed = bool(
            report.get("status") == "partial_passed"
            and report.get("manifest_sha256") == manifest_sha
            and report.get("git_head") == current_head
            and cases
            and all(case.get("passed") is True for case in cases)
            and all(case.get("all_finite") is True for case in cases)
        )
        checks.append(
            {
                "name": result_key,
                "passed": passed,
                "asset": _relative(path),
                "case_count": len(cases),
                "git_head": report.get("git_head"),
            }
        )
        teacher_reports[result_key] = report

    process_case_sets = [
        tuple(case.get("pair_id") for case in report.get("process_cases", []))
        for report in teacher_reports.values()
    ]
    process_passed = bool(
        process_case_sets
        and all(items == process_case_sets[0] for items in process_case_sets)
        and process_case_sets[0]
        and all(
            all(case.get("passed") is True for case in report.get("process_cases", []))
            for report in teacher_reports.values()
        )
    )
    checks.append(
        {
            "name": "process_case_matrix_consistent",
            "passed": process_passed,
            "case_count": len(process_case_sets[0]) if process_case_sets else 0,
        }
    )

    fortran = _load(FORTRAN_EVIDENCE)
    fortran_cases = fortran.get("cases", [])
    fortran_passed = bool(
        fortran.get("status") == "passed"
        and fortran_cases
        and all(case.get("passed") is True for case in fortran_cases)
    )
    checks.append(
        {
            "name": "selected_fortran_finite_differences",
            "passed": fortran_passed,
            "asset": _relative(FORTRAN_EVIDENCE),
            "case_count": len(fortran_cases),
        }
    )

    report = {
        "schema_version": "gate_d1_physical_parameter_gradient_acceptance_v1",
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_head": current_head,
        "manifest": _relative(MANIFEST),
        "manifest_sha256": manifest_sha,
        "checks": checks,
        "scope": {
            "process_case_count": len(process_case_sets[0]) if process_case_sets else 0,
            "complete_teacher_pair_count": sum(
                len(report.get(key, [])) for key, report in teacher_reports.items()
            ),
            "selected_fortran_pair_count": len(fortran_cases),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
