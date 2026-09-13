"""Summarize canopy and nonliving-carbon drift from saved forward-screen JSON.

This is a host-only reader.  It does not load a checkpoint, Teacher payload, or
JAX runtime.  The canopy quantity is the sum of the saved source-direct daily
``feedback_diagnostics.canopy_stock_change.physical_residual`` series.  The
remaining nonliving discrepancy is reported as an accounting remainder, not a
causal attribution.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (ROOT / "outputs").resolve()
NONLIVING = "nonliving_ecosystem_carbon"


def _scalar(value: Any, *, name: str) -> float:
    while isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"{name} must contain exactly one landpoint value")
        value = value[0]
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _series(value: Any, *, name: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty daily series")
    return [_scalar(item, name=name) for item in value]


def _selection_summary(selection: dict[str, Any]) -> dict[str, Any]:
    required = ("artifact_id", "landpoint_id", "year", "start_row", "length")
    missing = [name for name in required if name not in selection]
    if missing:
        raise ValueError(f"window selection lacks {missing}")
    return {name: selection[name] for name in required}


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = _mean(left), _mean(right)
    numerator = math.fsum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_sum = math.fsum((x - left_mean) ** 2 for x in left)
    right_sum = math.fsum((y - right_mean) ** 2 for y in right)
    if left_sum == 0.0 or right_sum == 0.0:
        return None
    return numerator / math.sqrt(left_sum * right_sum)


def _window_summary(window: dict[str, Any]) -> dict[str, Any]:
    selection = window.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("window lacks selection")
    selection_summary = _selection_summary(selection)
    horizon = int(selection_summary["length"])
    physical = window.get("physical", {})
    stocks = physical.get("stocks", {})
    nonliving = stocks.get(NONLIVING, {})
    predicted_final = _scalar(nonliving.get("predicted_final"), name="nonliving predicted_final")
    teacher_final = _scalar(nonliving.get("teacher_final"), name="nonliving teacher_final")
    feedback = window.get("feedback_diagnostics", {})
    canopy = feedback.get("canopy_stock_change", {})
    daily_residual = _series(canopy.get("physical_residual"), name="canopy physical_residual")
    if len(daily_residual) != horizon:
        raise ValueError("canopy physical_residual length differs from selected horizon")
    terminal_nonliving = predicted_final - teacher_final
    cumulative_canopy = math.fsum(daily_residual)
    return {
        "selection": selection_summary,
        "terminal_nonliving_discrepancy_gC_m2": terminal_nonliving,
        "cumulative_canopy_residual_gC_m2": cumulative_canopy,
        "remaining_nonliving_discrepancy_gC_m2": terminal_nonliving - cumulative_canopy,
        "canopy_daily_residual_gC_m2_day": {
            "count": len(daily_residual),
            "mean": _mean(daily_residual),
            "first": daily_residual[0],
            "last": daily_residual[-1],
        },
    }


def _horizon_summary(windows: list[dict[str, Any]]) -> dict[str, Any]:
    terminal = [row["terminal_nonliving_discrepancy_gC_m2"] for row in windows]
    canopy = [row["cumulative_canopy_residual_gC_m2"] for row in windows]
    remainder = [row["remaining_nonliving_discrepancy_gC_m2"] for row in windows]
    dominant = max(windows, key=lambda row: abs(row["terminal_nonliving_discrepancy_gC_m2"]))
    signed_total = math.fsum(terminal)
    absolute_total = math.fsum(abs(value) for value in terminal)
    dominant_value = dominant["terminal_nonliving_discrepancy_gC_m2"]
    return {
        "window_count": len(windows),
        "mean_terminal_nonliving_discrepancy_gC_m2": _mean(terminal),
        "mean_cumulative_canopy_residual_gC_m2": _mean(canopy),
        "mean_remaining_nonliving_discrepancy_gC_m2": _mean(remainder),
        "terminal_vs_canopy_correlation": _correlation(terminal, canopy),
        "dominant_window": {
            "selection": dominant["selection"],
            "terminal_nonliving_discrepancy_gC_m2": dominant_value,
            "share_of_signed_terminal_discrepancy": (
                dominant_value / signed_total if signed_total else None
            ),
            "share_of_absolute_terminal_discrepancy": (
                abs(dominant_value) / absolute_total if absolute_total else None
            ),
        },
        "windows": windows,
    }


def summarize_report(report: dict[str, Any], *, source: str) -> dict[str, Any]:
    windows = report.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("forward report lacks windows")
    by_horizon: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for window in windows:
        row = _window_summary(window)
        by_horizon[int(row["selection"]["length"])].append(row)
    return {
        "source": source,
        "checkpoint_update": report.get("checkpoint_update"),
        "checkpoint": report.get("checkpoint"),
        "source_git_head": report.get("source_git_head"),
        "horizons": {str(horizon): _horizon_summary(rows) for horizon, rows in sorted(by_horizon.items())},
    }


def summarize_paths(paths: list[Path]) -> dict[str, Any]:
    reports = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"forward report must be a JSON object: {path}")
        reports.append(summarize_report(payload, source=str(path)))
    return {
        "schema_version": "feedback_carbon_drift_summary_v2",
        "scope": "saved_forward_report_accounting_not_causal_attribution",
        "reports": reports,
    }


def _output_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(OUTPUT_ROOT)
    except ValueError as error:
        raise ValueError(f"--output must resolve under {OUTPUT_ROOT}") from error
    return resolved


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--report", type=Path, action="append", required=True,
                        help="Saved completed forward-screen JSON; repeat for multiple checkpoints.")
    result.add_argument("--output", type=Path, required=True,
                        help="Explicit JSON destination under outputs/.")
    return result


def main() -> None:
    args = parser().parse_args()
    output = _output_path(args.output)
    result = summarize_paths(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
