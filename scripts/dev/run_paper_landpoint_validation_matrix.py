from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.paper_validation import annual_delta_passes  # noqa: E402


def _selection_ids(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(str(value) for value in payload["selected_landpoint_ids"])


def _run_one(
    landpoint_id: str,
    *,
    args: argparse.Namespace,
    result_root: Path,
) -> dict[str, object]:
    point_root = result_root / landpoint_id
    output = point_root / f"multiyear_{args.start_year}_{args.start_year + args.years - 1}.json"
    stdout_path = point_root / "runner.stdout.log"
    stderr_path = point_root / "runner.stderr.log"
    checkpoint_dir = point_root / "checkpoints"
    point_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "dev" / "run_multiyear_modelout_lite.py"),
        "--start-year",
        str(args.start_year),
        "--years",
        str(args.years),
        "--days-per-year",
        str(args.days_per_year),
        "--initial-state",
        args.initial_state,
        "--landpoint-id",
        landpoint_id,
        "--field",
        "GPP",
        "--field",
        "NPP",
        "--field",
        "LEAF_M",
        "--field",
        "LAI",
        "--field",
        "MAINT_RESP",
        "--field",
        "GROWTH_RESP",
        "--field",
        "VCMAX",
        "--field",
        "MAINT_RESP_AGRSAPST",
        "--field",
        "MAINT_RESP_AGRSAPPN",
        "--field",
        "MAINT_RESP_AGRHRTST",
        "--field",
        "MAINT_RESP_AGRHRTPN",
        "--field",
        "BM_ALLOC_LEAF",
        "--field",
        "BM_ALLOC_SAP_AB",
        "--field",
        "BM_ALLOC_SAP_BE",
        "--field",
        "BM_ALLOC_ROOT",
        "--output",
        str(output),
        "--year-checkpoint-dir",
        str(checkpoint_dir),
        "--single-pass-daily-fold",
        "off",
        "--use-static-jit-daily-carbon",
        "on",
        "--prebuild-day-payloads",
        "on",
        "--module-jit",
        "on",
        "--diffuco-local-jit",
        "on",
    ]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("PYTHONPYCACHEPREFIX", str(Path(os.environ.get("TEMP", str(result_root))) / "orchjax_pycache"))
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    row: dict[str, object] = {
        "landpoint_id": landpoint_id,
        "returncode": int(completed.returncode),
        "elapsed_seconds": elapsed,
        "output": str(output),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "ready": False,
        "tolerance_pass": False,
        "max_abs_error": None,
    }
    if not output.exists():
        return row
    payload = json.loads(output.read_text(encoding="utf-8"))
    years = payload.get("years", [])
    modelout_deltas = [year.get("annual_reference_delta") for year in years]
    field_deltas = [year.get("annual_reference_field_delta") for year in years]
    diagnostic_deltas = [year.get("annual_reference_diagnostic_delta") for year in years]
    modelout_errors = [
        float(values["abs_error"])
        for delta in modelout_deltas
        if delta
        for values in delta.values()
    ]
    field_errors = [
        float(values["abs_error"])
        for delta in field_deltas
        if delta
        for values in delta.values()
    ]
    diagnostic_errors = [
        float(values["abs_error"])
        for delta in diagnostic_deltas
        if delta
        for values in delta.values()
    ]
    errors = modelout_errors + field_errors + diagnostic_errors
    row.update(
        ready=bool(payload.get("ready_for_requested_years")),
        completed_years=len(years),
        tolerance_pass=(
            bool(payload.get("ready_for_requested_years"))
            and len(modelout_deltas) == args.years
            and len(field_deltas) == args.years
            and len(diagnostic_deltas) == args.years
            and all(
                annual_delta_passes(delta, atol=args.atol, rtol=args.rtol)
                for delta in modelout_deltas
            )
            and all(
                annual_delta_passes(delta, atol=args.atol, rtol=args.rtol)
                for delta in field_deltas
            )
            and all(
                annual_delta_passes(delta, atol=args.atol, rtol=args.rtol)
                for delta in diagnostic_deltas
            )
        ),
        max_abs_error=max(errors) if errors else None,
        max_modelout_abs_error=max(modelout_errors) if modelout_errors else None,
        max_field_abs_error=max(field_errors) if field_errors else None,
        max_diagnostic_abs_error=max(diagnostic_errors) if diagnostic_errors else None,
        year_max_abs_errors=[
            max(
                [float(values["abs_error"]) for values in (modelout_delta or {}).values()]
                + [float(values["abs_error"]) for values in (field_delta or {}).values()]
                + [float(values["abs_error"]) for values in (diagnostic_delta or {}).values()],
                default=None,
            )
            for modelout_delta, field_delta, diagnostic_delta in zip(
                modelout_deltas, field_deltas, diagnostic_deltas, strict=True
            )
        ],
        year_max_modelout_abs_errors=[
            None if not delta else max(float(values["abs_error"]) for values in delta.values())
            for delta in modelout_deltas
        ],
        year_max_field_abs_errors=[
            None if not delta else max(float(values["abs_error"]) for values in delta.values())
            for delta in field_deltas
        ],
        year_max_diagnostic_abs_errors=[
            None if not delta else max(float(values["abs_error"]) for values in delta.values())
            for delta in diagnostic_deltas
        ],
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and aggregate strict multi-landpoint paper validation.")
    parser.add_argument("--selection", type=Path, default=None)
    parser.add_argument("--landpoint-id", action="append", default=[])
    parser.add_argument("--start-year", type=int, default=1961)
    parser.add_argument("--years", type=int, default=1)
    parser.add_argument("--days-per-year", default="365")
    parser.add_argument("--initial-state", choices=("cold-start", "restart-backed"), default="cold-start")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--atol", type=float, default=2.0e-6)
    parser.add_argument("--rtol", type=float, default=1.0e-7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "reference_mode" / "paper_landpoint_validation",
    )
    args = parser.parse_args()
    if args.years < 1 or args.workers < 1:
        raise ValueError("--years and --workers must be positive")
    ids = list(args.landpoint_id)
    if args.selection is not None:
        ids.extend(_selection_ids(args.selection))
    ids = list(dict.fromkeys(ids))
    if not ids:
        raise ValueError("provide --selection or at least one --landpoint-id")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run_one, landpoint_id, args=args, result_root=args.output_dir): landpoint_id
            for landpoint_id in ids
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row), flush=True)
    rows.sort(key=lambda row: ids.index(str(row["landpoint_id"])))
    payload = {
        "description": (
            f"{args.initial_state} paper-landpoint validation against annual Fortran stomate_history truth. "
            "Each landpoint uses its own canonical raw output package and materialized run.def; "
            "the gate covers 12 underlying modelout fields, 12 process diagnostics, "
            "and four derived modelout values."
        ),
        "start_year": args.start_year,
        "years": args.years,
        "days_per_year": args.days_per_year,
        "initial_state": args.initial_state,
        "atol": args.atol,
        "rtol": args.rtol,
        "landpoint_count": len(rows),
        "ready_count": sum(bool(row["ready"]) for row in rows),
        "pass_count": sum(bool(row["tolerance_pass"]) for row in rows),
        "elapsed_seconds": time.perf_counter() - started,
        "results": rows,
    }
    summary = args.output_dir / "summary.json"
    summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "results"}, indent=2))
    return 0 if payload["pass_count"] == payload["landpoint_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
