"""JSON-friendly physical summaries for monitored one-day training outputs."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

FLUX_UNIT = "gC_m-2_day-1"
STOCK_UNIT = "gC_m-2"
EXPECTED_FLUX_FIELDS = (
    "gpp",
    "maintenance_respiration",
    "growth_respiration",
    "heterotrophic_respiration",
    "npp",
    "autotrophic_respiration",
)


def _group(record: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = record.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"one-day monitor group {name!r} must be a mapping")
    return value


def _field_names(records: Sequence[Mapping[str, Any]], group_name: str) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for record in records:
        for name in _group(record, group_name):
            if not isinstance(name, str):
                raise ValueError(f"one-day monitor {group_name} field names must be strings")
            if name not in seen:
                names.append(name)
                seen.add(name)
    return tuple(names)


def _empty_statistics(*, missing_records: int, nonfinite_defined: int) -> dict[str, Any]:
    return {
        "n": 0,
        "missing_records": int(missing_records),
        "nonfinite_defined": int(nonfinite_defined),
        "rmse": None,
        "mae": None,
        "bias": None,
        "absolute_error_p50": None,
        "absolute_error_p90": None,
        "absolute_error_p99": None,
        "teacher_mean": None,
    }


def _statistics(
    records: Sequence[Mapping[str, Any]],
    group_name: str,
    field_name: str,
    *,
    predicted_key: str,
    teacher_key: str,
    optional_pair: bool,
) -> dict[str, Any]:
    errors: list[np.ndarray] = []
    teachers: list[np.ndarray] = []
    missing_records = 0
    nonfinite_defined = 0
    for record_index, record in enumerate(records):
        fields = _group(record, group_name)
        if field_name not in fields:
            missing_records += 1
            continue
        field = fields[field_name]
        if not isinstance(field, Mapping):
            raise ValueError(f"one-day monitor field {group_name}.{field_name} must be a mapping")
        if predicted_key not in field or teacher_key not in field:
            if optional_pair:
                missing_records += 1
                continue
            raise ValueError(
                f"one-day monitor field {group_name}.{field_name} record {record_index} "
                f"lacks {predicted_key!r} or {teacher_key!r}"
            )
        if "defined" not in field:
            raise ValueError(f"one-day monitor field {group_name}.{field_name} lacks 'defined'")
        predicted = np.asarray(field[predicted_key], dtype=np.float64)
        teacher = np.asarray(field[teacher_key], dtype=np.float64)
        defined = np.asarray(field["defined"], dtype=np.bool_)
        if predicted.shape != teacher.shape or defined.shape != predicted.shape:
            raise ValueError(
                f"one-day monitor shape drift for {group_name}.{field_name}: "
                f"predicted={predicted.shape}, teacher={teacher.shape}, defined={defined.shape}"
            )
        finite = np.isfinite(predicted) & np.isfinite(teacher)
        nonfinite_defined += int(np.count_nonzero(defined & ~finite))
        valid = defined & finite
        errors.append((predicted[valid] - teacher[valid]).reshape(-1))
        teachers.append(teacher[valid].reshape(-1))

    error = np.concatenate(errors) if errors else np.empty((0,), dtype=np.float64)
    teacher_values = np.concatenate(teachers) if teachers else np.empty((0,), dtype=np.float64)
    if error.size == 0:
        return _empty_statistics(
            missing_records=missing_records,
            nonfinite_defined=nonfinite_defined,
        )
    absolute = np.abs(error)
    return {
        "n": int(error.size),
        "missing_records": int(missing_records),
        "nonfinite_defined": int(nonfinite_defined),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(absolute)),
        "bias": float(np.mean(error)),
        "absolute_error_p50": float(np.quantile(absolute, 0.50)),
        "absolute_error_p90": float(np.quantile(absolute, 0.90)),
        "absolute_error_p99": float(np.quantile(absolute, 0.99)),
        "teacher_mean": float(np.mean(teacher_values)),
    }


def summarize_monitor(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize provided physical fields without deriving or replacing any field."""

    records = tuple(records)
    if any(not isinstance(record, Mapping) for record in records):
        raise ValueError("one-day monitor records must be mappings")
    flux_names = _field_names(records, "fluxes")
    stock_names = _field_names(records, "stocks")
    return {
        "sample_count": len(records),
        "fluxes": {
            "unit": FLUX_UNIT,
            "missing_expected_fields": [name for name in EXPECTED_FLUX_FIELDS if name not in flux_names],
            "fields": {
                name: _statistics(
                    records,
                    "fluxes",
                    name,
                    predicted_key="predicted",
                    teacher_key="teacher",
                    optional_pair=False,
                )
                for name in flux_names
            },
        },
        "stocks": {
            "unit": STOCK_UNIT,
            "change_unit": STOCK_UNIT,
            "fields": {
                name: {
                    "level": _statistics(
                        records,
                        "stocks",
                        name,
                        predicted_key="predicted",
                        teacher_key="teacher",
                        optional_pair=False,
                    ),
                    "change": _statistics(
                        records,
                        "stocks",
                        name,
                        predicted_key="predicted_change",
                        teacher_key="teacher_change",
                        optional_pair=True,
                    ),
                }
                for name in stock_names
            },
        },
    }


__all__ = ["summarize_monitor"]
