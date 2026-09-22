"""Analyze saved shared-pilot physical curves without running model inference."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

DEVELOPMENT_PREFIX = "development_"
TOLERANCE = 1.0e-12
TARGETS = {
    "gpp": ("fluxes", "gpp"),
    "npp": ("fluxes", "npp"),
    "autotrophic_respiration": ("fluxes", "autotrophic_respiration"),
    "maintenance_respiration": ("fluxes", "maintenance_respiration"),
    "growth_respiration": ("fluxes", "growth_respiration"),
    "heterotrophic_respiration": ("fluxes", "heterotrophic_respiration"),
    "aboveground_biomass_change": ("stocks", "aboveground_biomass_carbon"),
    "belowground_biomass_change": ("stocks", "belowground_biomass_carbon"),
    "canopy_organic_carbon_change": (
        "stocks",
        "canopy_interception_organic_carbon",
    ),
    "aboveground_litter_change": ("stocks", "aboveground_litter_carbon"),
    "belowground_litter_change": ("stocks", "belowground_litter_carbon"),
    "dissolved_organic_carbon_change": ("stocks", "dissolved_organic_carbon"),
}


def _development_t30(monitor: Mapping[str, Any]) -> list[dict[str, Any]]:
    windows = []
    for panel_name, panel in monitor["panels"].items():
        if not panel_name.startswith(DEVELOPMENT_PREFIX):
            continue
        for window in panel["windows"]:
            if int(window["horizon"]) == 30:
                windows.append({"panel": panel_name, **window})
    if len(windows) != 24:
        raise ValueError(f"expected 24 held-out development T30 windows, found {len(windows)}")
    return windows


def _window_identity(window: Mapping[str, Any]) -> dict[str, Any]:
    selection = window["selection"]
    return {
        "panel": window["panel"],
        "split": window["split"],
        "artifact_id": selection["artifact_id"],
        "landpoint_id": selection.get("landpoint_id"),
        "year": selection.get("year"),
        "start_row": selection.get("start_row"),
        "first_teacher_day_key": selection.get("teacher_day_keys", [None])[0],
    }


def _metric_arrays(
    window: Mapping[str, Any], kind: str, key: str, day: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curve = window["physical_curves"][kind][key]
    defined = np.asarray(curve["defined"], dtype=bool)
    if day < 1 or day > defined.shape[0]:
        raise ValueError(f"day {day} outside curve length {defined.shape[0]}")
    if kind == "fluxes":
        predicted = np.asarray(curve["predicted"], dtype=np.float64)[:day].sum(axis=0)
        teacher = np.asarray(curve["teacher"], dtype=np.float64)[:day].sum(axis=0)
        valid = np.all(defined[:day], axis=0)
    else:
        predicted = np.asarray(
            curve["predicted_change_from_common_initial"], dtype=np.float64
        )[day - 1]
        teacher = np.asarray(
            curve["teacher_change_from_common_initial"], dtype=np.float64
        )[day - 1]
        valid = defined[day - 1]
    valid &= np.isfinite(predicted) & np.isfinite(teacher)
    return predicted.reshape(-1), teacher.reshape(-1), valid.reshape(-1)


def _pooled_arrays(
    windows: Sequence[Mapping[str, Any]], kind: str, key: str, day: int
) -> tuple[np.ndarray, np.ndarray]:
    predicted, teacher = [], []
    for window in windows:
        p_value, t_value, valid = _metric_arrays(window, kind, key, day)
        predicted.extend(p_value[valid].tolist())
        teacher.extend(t_value[valid].tolist())
    return np.asarray(predicted), np.asarray(teacher)


def _summary(predicted: np.ndarray, teacher: np.ndarray) -> dict[str, Any]:
    if teacher.size == 0:
        return {"count": 0}
    error = predicted - teacher
    return {
        "count": int(teacher.size),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
        "mae": float(np.mean(np.abs(error))),
        "teacher_rms": float(np.sqrt(np.mean(np.square(teacher)))),
        "teacher_mean": float(np.mean(teacher)),
        "teacher_nonzero": int(np.sum(np.abs(teacher) > TOLERANCE)),
    }


def _development(
    windows: Sequence[Mapping[str, Any]], kind: str, key: str
) -> list[dict[str, Any]]:
    return [
        {"day": day, **_summary(*_pooled_arrays(windows, kind, key, day))}
        for day in range(1, 31)
    ]


def _window_concentration(
    windows: Sequence[Mapping[str, Any]], kind: str, key: str
) -> dict[str, Any]:
    records = []
    for window in windows:
        predicted, teacher, valid = _metric_arrays(window, kind, key, 30)
        error = predicted[valid] - teacher[valid]
        records.append(
            {
                **_window_identity(window),
                "coordinate_count": int(error.size),
                "signed_error_mean": float(np.mean(error)),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "sum_error": float(np.sum(error)),
                "sse": float(np.sum(np.square(error))),
                "teacher_mean": float(np.mean(teacher[valid])),
                "predicted_mean": float(np.mean(predicted[valid])),
            }
        )
    total_sse = sum(record["sse"] for record in records)
    total_abs_error = sum(abs(record["sum_error"]) for record in records)
    for record in records:
        record["sse_fraction"] = record["sse"] / total_sse if total_sse else 0.0
        record["absolute_error_fraction"] = (
            abs(record["sum_error"]) / total_abs_error if total_abs_error else 0.0
        )
    by_sse = sorted(records, key=lambda record: record["sse"], reverse=True)
    positive = sorted(records, key=lambda record: record["sum_error"], reverse=True)
    negative = sorted(records, key=lambda record: record["sum_error"])
    weights = np.asarray([record["sse_fraction"] for record in records])
    positive_error_sum = sum(max(record["sum_error"], 0.0) for record in records)
    negative_error_sum = sum(min(record["sum_error"], 0.0) for record in records)
    absolute_error_sum = positive_error_sum - negative_error_sum
    net_error_sum = positive_error_sum + negative_error_sum
    panels = {}
    for panel in sorted({record["panel"] for record in records}):
        selected = [record for record in records if record["panel"] == panel]
        count = sum(record["coordinate_count"] for record in selected)
        panel_sse = sum(record["sse"] for record in selected)
        panels[panel] = {
            "coordinate_count": count,
            "rmse": float(np.sqrt(panel_sse / count)),
            "bias": sum(record["sum_error"] for record in selected) / count,
            "sse_fraction": panel_sse / total_sse if total_sse else 0.0,
        }
    return {
        "window_count": len(records),
        "total_sse": total_sse,
        "positive_error_window_count": sum(record["sum_error"] > 0 for record in records),
        "negative_error_window_count": sum(record["sum_error"] < 0 for record in records),
        "positive_error_sum": positive_error_sum,
        "negative_error_sum": negative_error_sum,
        "net_error_sum": net_error_sum,
        "signed_error_cancellation_fraction": (
            1.0 - abs(net_error_sum) / absolute_error_sum if absolute_error_sum else 0.0
        ),
        "by_panel": panels,
        "effective_window_count_from_sse": (
            float(1.0 / np.sum(np.square(weights))) if np.any(weights) else 0.0
        ),
        "top1_sse_fraction": float(sum(value["sse_fraction"] for value in by_sse[:1])),
        "top3_sse_fraction": float(sum(value["sse_fraction"] for value in by_sse[:3])),
        "top5_sse_fraction": float(sum(value["sse_fraction"] for value in by_sse[:5])),
        "top_sse_windows": by_sse[:8],
        "largest_positive_error_windows": [
            value for value in positive if value["sum_error"] > 0
        ][:5],
        "largest_negative_error_windows": [
            value for value in negative if value["sum_error"] < 0
        ][:5],
        "all_windows": sorted(
            records,
            key=lambda value: (
                value["panel"],
                value["artifact_id"],
            ),
        ),
    }


def _aligned_flux_errors(
    windows: Sequence[Mapping[str, Any]], keys: Iterable[str], day: int = 30
) -> dict[str, np.ndarray]:
    result = {key: [] for key in keys}
    for window in windows:
        arrays = {
            key: _metric_arrays(window, "fluxes", key, day) for key in result
        }
        valid = np.logical_and.reduce([value[2] for value in arrays.values()])
        for key, (predicted, teacher, _mask) in arrays.items():
            result[key].extend((predicted[valid] - teacher[valid]).tolist())
    return {key: np.asarray(value) for key, value in result.items()}


def _respiration_compensation(windows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors = _aligned_flux_errors(
        windows,
        ("maintenance_respiration", "growth_respiration", "autotrophic_respiration"),
    )
    maintenance = errors["maintenance_respiration"]
    growth = errors["growth_respiration"]
    autotrophic = errors["autotrophic_respiration"]
    component_sum = maintenance + growth
    denominator = np.sum(np.abs(maintenance) + np.abs(growth))
    return {
        "count": int(autotrophic.size),
        "maintenance_bias": float(np.mean(maintenance)),
        "growth_bias": float(np.mean(growth)),
        "autotrophic_bias": float(np.mean(autotrophic)),
        "maintenance_rmse": float(np.sqrt(np.mean(np.square(maintenance)))),
        "growth_rmse": float(np.sqrt(np.mean(np.square(growth)))),
        "autotrophic_rmse": float(np.sqrt(np.mean(np.square(autotrophic)))),
        "maintenance_growth_error_correlation": float(np.corrcoef(maintenance, growth)[0, 1]),
        "maintenance_growth_mean_cross_product": float(np.mean(maintenance * growth)),
        "opposite_sign_fraction": float(np.mean(np.signbit(maintenance) != np.signbit(growth))),
        "absolute_error_cancellation_fraction": (
            float(
                np.sum(np.abs(maintenance) + np.abs(growth) - np.abs(component_sum))
                / denominator
            )
            if denominator
            else 0.0
        ),
        "ra_equals_component_sum_max_abs_error": float(
            np.max(np.abs(autotrophic - component_sum), initial=0.0)
        ),
        "mse_identity": {
            "autotrophic_mse": float(np.mean(np.square(autotrophic))),
            "maintenance_mse": float(np.mean(np.square(maintenance))),
            "growth_mse": float(np.mean(np.square(growth))),
            "two_times_mean_cross_product": float(2.0 * np.mean(maintenance * growth)),
        },
        "interpretation_limit": (
            "This is an arithmetic decomposition and cross-window association in saved "
            "errors; it does not establish a causal compensation mechanism."
        ),
    }


def _primary_error_association(
    attribution: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    targets = (
        "gpp",
        "npp",
        "aboveground_biomass_change",
        "belowground_biomass_change",
    )
    identities = [
        (value["panel"], value["artifact_id"])
        for value in attribution[targets[0]]["all_windows"]
    ]
    errors = []
    for target in targets:
        records = attribution[target]["all_windows"]
        current = [(value["panel"], value["artifact_id"]) for value in records]
        if current != identities:
            raise ValueError("primary target window order mismatch")
        errors.append([value["signed_error_mean"] for value in records])
    correlation = np.corrcoef(np.asarray(errors, dtype=np.float64))
    return {
        "targets": targets,
        "pearson_correlation": correlation.tolist(),
        "interpretation_limit": (
            "These correlations describe co-location of signed endpoint errors across the "
            "24 fixed windows. They do not establish that one target error causes another."
        ),
    }


def _rh_coverage(windows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    defined_values = 0
    teacher_nonzero = 0
    predicted_nonzero = 0
    teacher_nonzero_windows = 0
    for window in windows:
        curve = window["physical_curves"]["fluxes"]["heterotrophic_respiration"]
        defined = np.asarray(curve["defined"], dtype=bool)
        teacher = np.asarray(curve["teacher"], dtype=np.float64)
        predicted = np.asarray(curve["predicted"], dtype=np.float64)
        valid = defined & np.isfinite(teacher) & np.isfinite(predicted)
        window_nonzero = int(np.sum(valid & (np.abs(teacher) > TOLERANCE)))
        defined_values += int(np.sum(valid))
        teacher_nonzero += window_nonzero
        predicted_nonzero += int(np.sum(valid & (np.abs(predicted) > TOLERANCE)))
        teacher_nonzero_windows += int(window_nonzero > 0)
    return {
        "window_count": len(windows),
        "defined_daily_values": defined_values,
        "teacher_nonzero_daily_values": teacher_nonzero,
        "teacher_nonzero_windows": teacher_nonzero_windows,
        "predicted_nonzero_daily_values": predicted_nonzero,
        "nonzero_tolerance": TOLERANCE,
        "interpretation": (
            "Zero Teacher coverage means this panel cannot evaluate a learned nonzero RH response."
        ),
    }


def analyze(report: Mapping[str, Any], source: Path) -> dict[str, Any]:
    monitor_by_update = {int(monitor["update"]): monitor for monitor in report["monitors"]}
    if set(monitor_by_update) != {0, 20, 100}:
        raise ValueError(f"expected monitors at updates 0, 20, and 100: {monitor_by_update}")
    windows_by_update = {
        update: _development_t30(monitor) for update, monitor in monitor_by_update.items()
    }
    daily = {}
    for update, windows in windows_by_update.items():
        daily[str(update)] = {
            target: _development(windows, kind, key)
            for target, (kind, key) in TARGETS.items()
        }
    final_windows = windows_by_update[100]
    attribution = {
        target: _window_concentration(final_windows, kind, key)
        for target, (kind, key) in TARGETS.items()
    }
    return {
        "schema_version": "shared_pilot_curve_diagnosis_v1",
        "source": {
            "path": source.as_posix(),
            "status": report["status"],
            "completed_updates": report["completed_updates"],
        },
        "scope": {
            "monitor_updates": [0, 20, 100],
            "panel_prefix": DEVELOPMENT_PREFIX,
            "horizon_days": 30,
            "held_out_window_count": len(final_windows),
            "inference_run": False,
            "sealed_test_used": False,
            "causal_claim": False,
        },
        "day_by_day_pooled_metrics": daily,
        "update100_window_attribution": attribution,
        "update100_primary_error_association": _primary_error_association(attribution),
        "update100_respiration_compensation": _respiration_compensation(final_windows),
        "update100_rh_coverage": _rh_coverage(final_windows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = analyze(report, args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "held_out_window_count": result["scope"]["held_out_window_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
