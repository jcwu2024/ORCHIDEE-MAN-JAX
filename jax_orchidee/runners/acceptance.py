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
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(
    os.environ.get("ORCHIDEE_REPO_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.paper_validation import annual_delta_passes  # noqa: E402
from jax_orchidee.runners.multiyear import (  # noqa: E402
    EXECUTION_OPTIONS,
    RunRequest,
    completed_years_match,
    parse_run_args,
    prepare_run,
)
from jax_orchidee.runners.multiyear import main as run_multiyear_main  # noqa: E402
from jax_orchidee.runners.run_identity import (  # noqa: E402
    RESULT_PROTOCOL,
    assigned_site_ids,
    read_selection_ids,
    write_json,
)

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
    force: bool = False,
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
        "--release-jit-caches", getattr(args, "release_jit_caches", "off") if compiled else "off",
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
        "--resume-checkpoints", "off" if force else "on",
    ]


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
    payload: object,
    request: RunRequest,
) -> bool:
    args = request.args
    return bool(
        isinstance(payload, dict)
        and payload.get("run_identity") == request.identity
        and payload.get("landpoint_id") == args.landpoint_id
        and payload.get("start_year") == args.start_year
        and payload.get("requested_years") == args.years
        and payload.get("days_per_year") == request.days
        and payload.get("initial_state") == args.initial_state
        and all(payload.get(key) == getattr(args, key) for key in EXECUTION_OPTIONS)
        and payload.get("ready_for_requested_years") is True
        and completed_years_match(payload.get("years"), request, args.years)
    )


def _read_result(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


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
    stdout = io.StringIO()
    started = time.perf_counter()
    receipt = {"mode": mode, "resumed": False, "output": None, "log": str(log)}
    payload = None
    with TemporaryDirectory(prefix=f".{mode}-", dir=point_dir) as temporary:
        staged = Path(temporary) / output.name
        command = _multiyear_command(
            landpoint_id, args=args, compiled=compiled, output=staged,
            checkpoint_dir=point_dir / f"{mode}_checkpoints", force=force,
        )
        try:
            request = prepare_run(parse_run_args(command))
            receipt["batch_identity"] = request.identity["batch"]
            if args.initial_state == "restart-backed" and request.identity["initial_source"] != "restart-files":
                raise ValueError("restart-backed acceptance requires a declared driver/SECHIBA/STOMATE restart triplet")
            cached = None if force else _read_result(output)
            if _payload_matches_request(cached, request):
                return cached, {
                    **receipt, "resumed": True, "output": str(output), "returncode": 0,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            with contextlib.redirect_stdout(stdout):
                receipt["returncode"] = run_multiyear_main(command, request=request)
            current = _read_result(staged)
            if isinstance(current, dict) and isinstance(current.get("years"), list):
                receipt["reported_years"] = len(current["years"])
            if receipt["returncode"] != 0:
                receipt["error"] = "multiyear execution did not complete"
            elif not _payload_matches_request(current, request):
                receipt.update(returncode=1, error="current attempt produced no matching complete result")
            else:
                staged.replace(output)
                payload = current
                receipt["output"] = str(output)
        except KeyboardInterrupt:
            receipt.update(returncode=130, error="interrupted")
        except Exception:
            receipt.update(returncode=1, error=traceback.format_exc())
    log.write_text(stdout.getvalue() + "\n" + receipt.get("error", ""), encoding="utf-8")
    receipt["elapsed_seconds"] = time.perf_counter() - started
    return payload, receipt


def _write_summary(
    args: argparse.Namespace, selection: tuple[str, ...], rows: list[dict[str, object]], started: float,
) -> Path:
    accepted = sum(bool(row.get("scientific_acceptance_pass")) for row in rows)
    payload = {
        "result_protocol": RESULT_PROTOCOL,
        "description": (
            "Paper PFT14 annual scientific acceptance. Compiled mode runs first in one process. "
            "Strict mode runs only for scientific failures."
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
        "days_per_year": [int(part) for part in args.days_per_year.split(",") if part.strip()],
        "initial_state": args.initial_state,
        "release_jit_caches": args.release_jit_caches,
        "strict_on_failure": args.strict_on_failure,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "selection_ids": selection,
        "max_points": args.max_points,
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
    write_json(summary, payload)
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
    parser.add_argument(
        "--release-jit-caches", choices=("on", "off"), default="off",
        help="Use process-local JIT cache cleanup at compiled day-block and year boundaries.",
    )
    parser.add_argument("--science-atol", type=float, default=1.0e-5)
    parser.add_argument("--science-rtol", type=float, default=1.0e-3)
    parser.add_argument("--max-normalized-bias", type=float, default=1.0e-3)
    parser.add_argument("--max-normalized-rmse", type=float, default=1.0e-3)
    parser.add_argument("--diagnostic-atol", type=float, default=2.0e-6)
    parser.add_argument("--diagnostic-rtol", type=float, default=1.0e-7)
    parser.add_argument("--force", action="store_true", help="Run from the declared start without automatic result or checkpoint reuse.")
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
        ids.extend(read_selection_ids(args.selection))
    selection = tuple(dict.fromkeys(ids))
    if not selection:
        raise ValueError("provide --selection or at least one --landpoint-id")
    if args.max_points is not None and args.max_points < 1:
        raise ValueError("--max-points must be positive")
    ids = assigned_site_ids(selection, args.shard_index, args.shard_count, args.max_points)
    if not ids:
        parser.error("this shard selects no landpoints")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows: list[dict[str, object]] = [
        {"landpoint_id": landpoint_id, "scientific_acceptance_pass": False, "classification": "pending"}
        for landpoint_id in ids
    ]
    # Replace any prior summary before starting physics. After an external
    # interruption unvisited points still appear as pending in this batch.
    _write_summary(args, selection, rows, started)
    for index, landpoint_id in enumerate(ids):
        compiled_payload, compiled_run = _run_point(
            landpoint_id, args=args, compiled=True, force=args.force
        )
        row: dict[str, object] = {"landpoint_id": landpoint_id, "compiled_run": compiled_run}
        interrupted = compiled_run["returncode"] == 130
        if compiled_payload is None:
            row.update(
                scientific_acceptance_pass=False,
                classification="compiled_interrupted" if interrupted else "compiled_execution_failure",
            )
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
                    interrupted = strict_run["returncode"] == 130
                    row["classification"] = "strict_interrupted" if interrupted else "strict_execution_failure"
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
        rows[index] = row
        summary = _write_summary(args, selection, rows, started)
        print(json.dumps({"landpoint_id": landpoint_id, "classification": row["classification"]}), flush=True)
        if interrupted:
            return 130

    summary = _write_summary(args, selection, rows, started)
    print(json.dumps({"summary": str(summary), "accepted": sum(bool(row.get("scientific_acceptance_pass")) for row in rows), "total": len(rows)}, indent=2))
    return 0 if all(bool(row.get("scientific_acceptance_pass")) for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
