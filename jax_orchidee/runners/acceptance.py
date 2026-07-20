from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(
    os.environ.get("ORCHIDEE_REPO_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.paper_validation import annual_delta_passes  # noqa: E402
from jax_orchidee.runners.multiyear import main as run_multiyear_main  # noqa: E402

KEY_MODEL_OUTPUTS = ("AGB_model", "BGB_model", "GPP_model", "NPP_model")
DEFAULT_OUTPUT_DIR = (
    Path(os.environ.get("ORCHIDEE_OUTPUT_ROOT", ROOT / "outputs")).expanduser()
    / "acceptance"
    / "paper_landpoints"
)


def _multiyear_command(
    landpoint_id: str,
    *,
    args: argparse.Namespace,
    compiled: bool,
    output: Path,
    checkpoint_dir: Path,
) -> list[str]:
    """Build one production acceptance command with the accepted fast-path policy."""

    return [
        "--start-year", str(args.start_year),
        "--years", str(args.years),
        "--days-per-year", str(args.days_per_year),
        "--initial-state", args.initial_state,
        "--landpoint-id", landpoint_id,
        "--compiled-sechiba-day", "on" if compiled else "off",
        "--compiled-day-block-size", "7" if compiled else "0",
        # The three-point annual gate rejected this companion optimization at
        # fixed atol=1e-8 on 319.0-057.0 DOC. Keep it off in production.
        "--numpy-accumulator", "off",
        "--single-pass-daily-fold", "off",
        "--use-static-jit-daily-carbon", "on" if compiled else "off",
        "--prebuild-day-payloads", "on",
        "--module-jit", "on",
        "--diffuco-local-jit", "on",
        "--field", "GPP",
        "--field", "NPP",
        "--field", "LEAF_M",
        "--field", "LAI",
        "--field", "MAINT_RESP",
        "--field", "GROWTH_RESP",
        "--field", "VCMAX",
        "--field", "MAINT_RESP_AGRSAPST",
        "--field", "MAINT_RESP_AGRSAPPN",
        "--field", "MAINT_RESP_AGRHRTST",
        "--field", "MAINT_RESP_AGRHRTPN",
        "--field", "BM_ALLOC_LEAF",
        "--field", "BM_ALLOC_SAP_AB",
        "--field", "BM_ALLOC_SAP_BE",
        "--field", "BM_ALLOC_ROOT",
        "--output", str(output),
        "--year-checkpoint-dir", str(checkpoint_dir),
        "--resume-checkpoints", "on",
    ]


def _selection_ids(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(str(value) for value in payload["selected_landpoint_ids"])


def _delta_errors(deltas: list[dict[str, object] | None]) -> list[float]:
    return [
        float(values["abs_error"])
        for delta in deltas
        if delta
        for values in delta.values()
    ]


def _series_statistics(years: list[dict[str, object]]) -> dict[str, dict[str, float | int | None]]:
    statistics: dict[str, dict[str, float | int | None]] = {}
    for name in KEY_MODEL_OUTPUTS:
        pairs = [
            year["annual_reference_delta"][name]
            for year in years
            if year.get("annual_reference_delta") and name in year["annual_reference_delta"]
        ]
        if not pairs:
            statistics[name] = {"count": 0}
            continue
        actual = np.asarray([pair["jax"] for pair in pairs], dtype=np.float64)
        reference = np.asarray([pair["reference"] for pair in pairs], dtype=np.float64)
        error = actual - reference
        reference_rms = max(float(np.sqrt(np.mean(reference * reference))), 1.0e-30)
        reference_l1 = max(float(np.sum(np.abs(reference))), 1.0e-30)
        correlation = None
        if actual.size >= 2 and float(np.std(actual)) > 0.0 and float(np.std(reference)) > 0.0:
            correlation = float(np.corrcoef(actual, reference)[0, 1])
        statistics[name] = {
            "count": int(actual.size),
            "max_abs_error": float(np.max(np.abs(error))),
            "mean_bias": float(np.mean(error)),
            "normalized_bias": float(np.sum(error) / reference_l1),
            "normalized_rmse": float(np.sqrt(np.mean(error * error)) / reference_rms),
            "correlation": correlation,
        }
    return statistics


def _evaluate(
    payload: dict[str, object],
    *,
    requested_years: int,
    science_atol: float,
    science_rtol: float,
    diagnostic_atol: float,
    diagnostic_rtol: float,
    max_normalized_bias: float,
    max_normalized_rmse: float,
) -> dict[str, object]:
    years = list(payload.get("years", []))
    modelout_deltas = [year.get("annual_reference_delta") for year in years]
    field_deltas = [year.get("annual_reference_field_delta") for year in years]
    diagnostic_deltas = [year.get("annual_reference_diagnostic_delta") for year in years]
    ready = bool(payload.get("ready_for_requested_years")) and len(years) == requested_years
    complete_truth = (
        len(modelout_deltas) == requested_years
        and all(delta and all(name in delta for name in KEY_MODEL_OUTPUTS) for delta in modelout_deltas)
    )
    annual_science_pass = ready and complete_truth and all(
        annual_delta_passes(delta, atol=science_atol, rtol=science_rtol)
        for delta in modelout_deltas
    )
    series = _series_statistics(years)
    series_pass = bool(series) and all(
        int(values.get("count", 0)) == requested_years
        and abs(float(values.get("normalized_bias", np.inf))) <= max_normalized_bias
        and float(values.get("normalized_rmse", np.inf)) <= max_normalized_rmse
        for values in series.values()
    )
    numerical_diagnostic_pass = (
        ready
        and len(field_deltas) == requested_years
        and len(diagnostic_deltas) == requested_years
        and all(annual_delta_passes(delta, atol=diagnostic_atol, rtol=diagnostic_rtol) for delta in modelout_deltas)
        and all(annual_delta_passes(delta, atol=diagnostic_atol, rtol=diagnostic_rtol) for delta in field_deltas)
        and all(annual_delta_passes(delta, atol=diagnostic_atol, rtol=diagnostic_rtol) for delta in diagnostic_deltas)
    )
    modelout_errors = _delta_errors(modelout_deltas)
    field_errors = _delta_errors(field_deltas)
    diagnostic_errors = _delta_errors(diagnostic_deltas)
    return {
        "ready": ready,
        "annual_truth_complete": complete_truth,
        "scientific_acceptance_pass": bool(annual_science_pass and series_pass),
        "annual_scientific_tolerance_pass": bool(annual_science_pass),
        "series_statistics_pass": bool(series_pass),
        "numerical_diagnostic_pass": bool(numerical_diagnostic_pass),
        "max_modelout_abs_error": max(modelout_errors, default=None),
        "max_field_abs_error": max(field_errors, default=None),
        "max_diagnostic_abs_error": max(diagnostic_errors, default=None),
        "series": series,
    }


def _payload_matches_request(
    payload: dict[str, object],
    *,
    landpoint_id: str,
    args: argparse.Namespace,
) -> bool:
    """Reject cached runs produced for a different scientific protocol."""

    requested_days = [
        int(value.strip())
        for value in str(args.days_per_year).split(",")
        if value.strip()
    ]
    if len(requested_days) == 1:
        requested_days *= int(args.years)
    return bool(
        payload.get("landpoint_id") == landpoint_id
        and int(payload.get("start_year", -1)) == int(args.start_year)
        and int(payload.get("requested_years", -1)) == int(args.years)
        and payload.get("days_per_year") == requested_days
        and payload.get("initial_state") == args.initial_state
        and payload.get("ready_for_requested_years") is True
        and len(payload.get("years", [])) == int(args.years)
    )


def _run_point(
    landpoint_id: str,
    *,
    args: argparse.Namespace,
    compiled: bool,
    force: bool,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    mode = "compiled" if compiled else "strict"
    point_dir = args.output_dir / landpoint_id
    point_dir.mkdir(parents=True, exist_ok=True)
    output = point_dir / f"{mode}_{args.start_year}_{args.start_year + args.years - 1}.json"
    log = point_dir / f"{mode}.stdout.log"
    if output.exists() and not force:
        payload = json.loads(output.read_text(encoding="utf-8"))
        if _payload_matches_request(payload, landpoint_id=landpoint_id, args=args):
            return payload, {"mode": mode, "resumed": True, "output": str(output), "elapsed_seconds": 0.0}

    command = _multiyear_command(
        landpoint_id,
        args=args,
        compiled=compiled,
        output=output,
        checkpoint_dir=point_dir / f"{mode}_checkpoints",
    )
    stdout = io.StringIO()
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(stdout):
            returncode = run_multiyear_main(command)
        error = None
    except Exception:
        returncode = 1
        error = traceback.format_exc()
    elapsed = time.perf_counter() - started
    log.write_text(stdout.getvalue() + ("\n" + error if error else ""), encoding="utf-8")
    payload = json.loads(output.read_text(encoding="utf-8")) if output.exists() else None
    return payload, {
        "mode": mode,
        "resumed": False,
        "returncode": returncode,
        "output": str(output),
        "log": str(log),
        "elapsed_seconds": elapsed,
    }


def _write_summary(args: argparse.Namespace, rows: list[dict[str, object]], started: float) -> Path:
    accepted = sum(bool(row.get("scientific_acceptance_pass")) for row in rows)
    payload = {
        "description": (
            "Paper PFT14 annual scientific acceptance. Compiled mode runs first in one process so "
            "landpoints reuse the compiled SECHIBA executable; strict mode runs only for scientific failures."
        ),
        "tolerance_policy": {
            "scientific_key_outputs": list(KEY_MODEL_OUTPUTS),
            "scientific_atol": args.science_atol,
            "scientific_rtol": args.science_rtol,
            "max_normalized_bias": args.max_normalized_bias,
            "max_normalized_rmse": args.max_normalized_rmse,
            "diagnostic_atol": args.diagnostic_atol,
            "diagnostic_rtol": args.diagnostic_rtol,
            "policy_fixed_before_pilot": True,
        },
        "start_year": args.start_year,
        "years": args.years,
        "days_per_year": args.days_per_year,
        "initial_state": args.initial_state,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "landpoint_count": len(rows),
        "scientific_acceptance_count": accepted,
        "elapsed_seconds": time.perf_counter() - started,
        "results": rows,
    }
    summary_name = (
        "summary.json"
        if args.shard_count == 1
        else f"summary_shard_{args.shard_index:03d}_of_{args.shard_count:03d}.json"
    )
    summary = args.output_dir / summary_name
    temporary = summary.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="orchidee-jax validate-landpoints",
        description="Run resumable paper-landpoint annual scientific acceptance.",
    )
    parser.add_argument("--selection", type=Path, default=None)
    parser.add_argument("--landpoint-id", action="append", default=[])
    parser.add_argument("--start-year", type=int, default=1961)
    parser.add_argument("--years", type=int, default=1)
    parser.add_argument("--days-per-year", default="365")
    parser.add_argument("--initial-state", choices=("cold-start", "restart-backed"), default="cold-start")
    parser.add_argument("--strict-on-failure", choices=("on", "off"), default="on")
    parser.add_argument("--science-atol", type=float, default=1.0e-5)
    parser.add_argument("--science-rtol", type=float, default=1.0e-3)
    parser.add_argument("--max-normalized-bias", type=float, default=1.0e-3)
    parser.add_argument("--max-normalized-rmse", type=float, default=1.0e-3)
    parser.add_argument("--diagnostic-atol", type=float, default=2.0e-6)
    parser.add_argument("--diagnostic-rtol", type=float, default=1.0e-7)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    if args.years < 1 or args.shard_count < 1:
        raise ValueError("--years and --shard-count must be positive")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must be in [0, shard-count)")
    ids = list(args.landpoint_id)
    if args.selection is not None:
        ids.extend(_selection_ids(args.selection))
    ids = list(dict.fromkeys(ids))
    if not ids:
        raise ValueError("provide --selection or at least one --landpoint-id")
    ids = [landpoint_id for index, landpoint_id in enumerate(ids) if index % args.shard_count == args.shard_index]
    if args.max_points is not None:
        if args.max_points < 1:
            raise ValueError("--max-points must be positive")
        ids = ids[: args.max_points]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    for landpoint_id in ids:
        compiled_payload, compiled_run = _run_point(
            landpoint_id, args=args, compiled=True, force=args.force
        )
        row: dict[str, object] = {"landpoint_id": landpoint_id, "compiled_run": compiled_run}
        if compiled_payload is None:
            row.update(scientific_acceptance_pass=False, classification="compiled_execution_failure")
        else:
            compiled_evaluation = _evaluate(
                compiled_payload,
                requested_years=args.years,
                science_atol=args.science_atol,
                science_rtol=args.science_rtol,
                diagnostic_atol=args.diagnostic_atol,
                diagnostic_rtol=args.diagnostic_rtol,
                max_normalized_bias=args.max_normalized_bias,
                max_normalized_rmse=args.max_normalized_rmse,
            )
            row["compiled_evaluation"] = compiled_evaluation
            row["scientific_acceptance_pass"] = compiled_evaluation["scientific_acceptance_pass"]
            row["classification"] = (
                "accepted_compiled"
                if compiled_evaluation["scientific_acceptance_pass"]
                else "requires_strict_diagnostic"
            )
            if not compiled_evaluation["scientific_acceptance_pass"] and args.strict_on_failure == "on":
                strict_payload, strict_run = _run_point(
                    landpoint_id, args=args, compiled=False, force=args.force
                )
                row["strict_run"] = strict_run
                if strict_payload is None:
                    row["classification"] = "strict_execution_failure"
                else:
                    strict_evaluation = _evaluate(
                        strict_payload,
                        requested_years=args.years,
                        science_atol=args.science_atol,
                        science_rtol=args.science_rtol,
                        diagnostic_atol=args.diagnostic_atol,
                        diagnostic_rtol=args.diagnostic_rtol,
                        max_normalized_bias=args.max_normalized_bias,
                        max_normalized_rmse=args.max_normalized_rmse,
                    )
                    row["strict_evaluation"] = strict_evaluation
                    row["classification"] = (
                        "compiled_path_regression"
                        if strict_evaluation["scientific_acceptance_pass"]
                        else "shared_semantic_input_or_reference_mismatch"
                    )
        rows.append(row)
        summary = _write_summary(args, rows, started)
        print(json.dumps({"landpoint_id": landpoint_id, "classification": row["classification"]}), flush=True)

    summary = _write_summary(args, rows, started)
    print(json.dumps({"summary": str(summary), "accepted": sum(bool(row.get("scientific_acceptance_pass")) for row in rows), "total": len(rows)}, indent=2))
    return 0 if all(bool(row.get("scientific_acceptance_pass")) for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
