from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.runtime import configure_jax_compilation_cache  # noqa: E402
from research.daily_coarse_graining.physical_parameter_process_cases import (  # noqa: E402
    run_process_gradient_cases,
)
from research.daily_coarse_graining.teacher_parameter_gradient_gate import (  # noqa: E402
    TeacherGradientPairSpec,
    prepare_later_day_gradient_runtime,
    prepare_restart_year_gradient_runtime,
    run_teacher_gradient_matrix,
)

MANIFEST = ROOT / "manifests/coarse_graining/teacher_physical_parameter_gradient_v1.json"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/research/daily_coarse_graining/gate_d1_physical_parameter_gradients/comparison.json"
)


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")


def _one_day_specs() -> tuple[TeacherGradientPairSpec, ...]:
    return (
        TeacherGradientPairSpec(
            "day2.npp__maint_resp_slope_c",
            "smooth_active",
            "maint_resp_slope_c",
            "NPP_model",
            1.0e-6,
        ),
        TeacherGradientPairSpec(
            "day2.maintenance__maint_resp_slope_b",
            "smooth_active",
            "maint_resp_slope_b",
            "MAINT_RESP",
            1.0e-7,
        ),
        TeacherGradientPairSpec(
            "day2.allocation_sapwood__alloc_min",
            "smooth_active",
            "alloc_min",
            "BM_ALLOC_SAP_AB",
            2.0e-6,
        ),
        TeacherGradientPairSpec(
            "day2.vcmax__vcmax25",
            "smooth_active",
            "vcmax25",
            "VCMAX",
            6.32061836e-4,
        ),
        TeacherGradientPairSpec(
            "day2.agb__residence_time",
            "smooth_active",
            "residence_time",
            "AGB_model",
            5.06583452e-4,
        ),
        TeacherGradientPairSpec(
            "day2.gpp__g0",
            "smooth_active",
            "g0",
            "GPP_model",
            1.0e-7,
        ),
        TeacherGradientPairSpec(
            "day2.gpp__alloc_min_inactive",
            "inactive",
            "alloc_min",
            "GPP_model",
            2.0e-6,
        ),
    )


def _multiday_specs() -> tuple[TeacherGradientPairSpec, ...]:
    return (
        TeacherGradientPairSpec(
            "day3.gpp__vcmax25",
            "smooth_active",
            "vcmax25",
            "GPP_model",
            6.32061836e-4,
            1,
        ),
        TeacherGradientPairSpec(
            "day3.agb__residence_time",
            "smooth_active",
            "residence_time",
            "AGB_model",
            5.06583452e-4,
            1,
        ),
        TeacherGradientPairSpec(
            "day3.gpp__alloc_min_inactive",
            "inactive",
            "alloc_min",
            "GPP_model",
            2.0e-6,
            1,
        ),
    )


def _restart_specs() -> tuple[TeacherGradientPairSpec, ...]:
    return (
        TeacherGradientPairSpec(
            "restart1962.day1.gpp__g0",
            "smooth_active",
            "g0",
            "GPP_model",
            1.0e-7,
        ),
        TeacherGradientPairSpec(
            "restart1962.day1.vcmax__vcmax25",
            "smooth_active",
            "vcmax25",
            "VCMAX",
            6.32061836e-4,
        ),
        TeacherGradientPairSpec(
            "restart1962.day1.agb__residence_time",
            "smooth_active",
            "residence_time",
            "AGB_model",
            5.06583452e-4,
        ),
        TeacherGradientPairSpec(
            "restart1962.day1.gpp__alloc_min_inactive",
            "inactive",
            "alloc_min",
            "GPP_model",
            2.0e-6,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Gate-D1 physical-parameter gradients")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/orchidee_man_250919.yaml")
    parser.add_argument("--run-def", type=Path, default=ROOT / "outputs/reference_mode/used_run.def")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--scope",
        choices=("process", "one-day", "multiday", "restart"),
        default="one-day",
        help="each complete-Teacher scope includes the process matrix",
    )
    args = parser.parse_args()

    configure_jax_compilation_cache(ROOT)
    json.loads(MANIFEST.read_text(encoding="utf-8"))
    process = run_process_gradient_cases()
    report: dict[str, object] = {
        "schema_version": "gate_d1_physical_parameter_gradients_v1",
        "status": "running" if args.scope == "one-day" else "partial_passed",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_head": _git_head(),
        "manifest": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "scope": args.scope,
        "process_cases": [item.as_dict() for item in process],
        "complete_teacher_one_day": [],
        "complete_teacher_multiday": [],
        "complete_teacher_restart_year_boundary": [],
        "remaining_required_scopes": [
            "complete_teacher_multiday",
            "complete_teacher_restart_year_boundary",
            "selected_fortran_finite_differences",
        ],
    }
    _write_report(args.output, report)
    if args.scope == "process":
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.scope == "one-day":
        runtime = prepare_later_day_gradient_runtime(
            args.config,
            used_run_def_path=args.run_def,
            seed_days=1,
            horizon=1,
        )
        specs = _one_day_specs()
        result_key = "complete_teacher_one_day"
    elif args.scope == "multiday":
        runtime = prepare_later_day_gradient_runtime(
            args.config,
            used_run_def_path=args.run_def,
            seed_days=1,
            horizon=2,
        )
        specs = _multiday_specs()
        result_key = "complete_teacher_multiday"
    else:
        runtime = prepare_restart_year_gradient_runtime(
            args.config,
            used_run_def_path=args.run_def,
        )
        specs = _restart_specs()
        result_key = "complete_teacher_restart_year_boundary"
    results = run_teacher_gradient_matrix(runtime, specs)
    report[result_key] = [item.as_dict() for item in results]
    report["status"] = (
        "partial_passed"
        if all(item.passed for item in (*process, *results))
        else "failed"
    )
    report["generated_at_utc"] = datetime.now(UTC).isoformat()
    report["teacher_runtime"] = {
        "year": runtime.year,
        "day_indices": list(runtime.day_indices),
        "parameter_count": len({item.parameter_id for item in specs}),
        "declared_pair_count": len(results),
        "candidate_model": "canonical complete-day Teacher",
        "half_hour_steps_retained": 48,
        "lifecycle_scope": args.scope,
    }
    _write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
