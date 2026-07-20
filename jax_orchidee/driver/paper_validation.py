"""Deterministic selection and evaluation helpers for paper landpoints.

These helpers operate only at the reference/validation boundary. They do not
implement or infer ORCHIDEE process logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from jax_orchidee.driver.reference_layout import (
    LEGACY_LANDPOINT_ID,
    inventory_paper_references,
)
from jax_orchidee.driver.run_def_materialization import read_run_def_values
from jax_orchidee.stomate.reference import read_paper_modelout_csv


FEATURE_NAMES = (
    "grid_x",
    "grid_y",
    "vcmax25",
    "maint_resp_slope_c",
    "alloc_min",
    "residence_time",
    "AGB_model",
    "BGB_model",
    "GPP_model",
    "NPP_model",
)


@dataclass(frozen=True)
class PaperValidationPoint:
    landpoint_id: str
    iteration_id: str
    param_set: str
    target_year: int
    grid_x: float
    grid_y: float
    vcmax25: float
    maint_resp_slope_c: float
    alloc_min: float
    residence_time: float
    AGB_model: float
    BGB_model: float
    GPP_model: float
    NPP_model: float

    def feature_vector(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in FEATURE_NAMES)

    def to_json_row(self) -> dict[str, object]:
        return asdict(self)


def _parse_landpoint_id(landpoint_id: str) -> tuple[float, float]:
    parts = landpoint_id.split("-")
    if len(parts) != 2:
        raise ValueError(f"invalid paper landpoint id: {landpoint_id}")
    return float(parts[0]), float(parts[1])


def load_paper_validation_points(root: str | Path) -> tuple[PaperValidationPoint, ...]:
    """Load one compact validation-feature row per complete paper landpoint."""

    points: list[PaperValidationPoint] = []
    for reference in inventory_paper_references(root):
        if not reference.has_minimum_annual_truth:
            continue
        if reference.iteration_id is None or reference.param_set is None:
            continue
        csv_rows = read_paper_modelout_csv(root=root, landpoint_id=reference.landpoint_id)
        if not csv_rows:
            raise ValueError(f"no paper modelout rows for {reference.landpoint_id}")
        target_year = max(row.target_year for row in csv_rows)
        target_rows = tuple(row for row in csv_rows if row.target_year == target_year)
        grid_x, grid_y = _parse_landpoint_id(reference.landpoint_id)
        if reference.run_def is None:
            raise ValueError(f"no archived run.def for {reference.landpoint_id}")
        run_def = read_run_def_values(reference.run_def)
        vcmax25 = float(run_def["VCMAX25__00014"])
        maint_resp_slope_c = float(run_def["MAINT_RESP_SLOPE_C__00014"])
        alloc_min = float(run_def["ALLOC_MIN__00014"])
        residence_time = float(run_def["RESIDENCE_TIME__00014"])
        points.append(
            PaperValidationPoint(
                landpoint_id=reference.landpoint_id,
                iteration_id=reference.iteration_id,
                param_set=reference.param_set,
                target_year=int(target_year),
                grid_x=grid_x,
                grid_y=grid_y,
                vcmax25=vcmax25,
                maint_resp_slope_c=maint_resp_slope_c,
                alloc_min=alloc_min,
                residence_time=residence_time,
                AGB_model=float(np.mean([row.AGB_model for row in target_rows])),
                BGB_model=float(np.mean([row.BGB_model for row in target_rows])),
                GPP_model=float(np.mean([row.GPP_model for row in target_rows])),
                NPP_model=float(np.mean([row.NPP_model for row in target_rows])),
            )
        )
    return tuple(sorted(points, key=lambda point: point.landpoint_id))


def select_representative_paper_points(
    points: Iterable[PaperValidationPoint],
    *,
    count: int,
    include_ids: Iterable[str] = (LEGACY_LANDPOINT_ID,),
) -> tuple[PaperValidationPoint, ...]:
    """Select a deterministic maximin sample in standardized feature space."""

    rows = tuple(points)
    if not rows:
        return ()
    if count < 1:
        raise ValueError("count must be positive")
    if count >= len(rows):
        return rows

    by_id = {row.landpoint_id: index for index, row in enumerate(rows)}
    selected = [by_id[item] for item in include_ids if item in by_id]
    selected = list(dict.fromkeys(selected))
    matrix = np.asarray([row.feature_vector() for row in rows], dtype=np.float64)
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    standardized = (matrix - mean) / scale

    if not selected:
        selected.append(int(np.argmax(np.sum(standardized * standardized, axis=1))))
    while len(selected) < count:
        distances = np.min(
            np.sum((standardized[:, None, :] - standardized[selected][None, :, :]) ** 2, axis=2),
            axis=1,
        )
        distances[np.asarray(selected, dtype=np.int64)] = -1.0
        selected.append(int(np.argmax(distances)))
    return tuple(rows[index] for index in selected)


def annual_delta_passes(
    delta: Mapping[str, Mapping[str, float]] | None,
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Apply the validation tolerance to one annual modelout delta mapping."""

    if not delta:
        return False
    return all(
        float(values["abs_error"])
        <= float(atol) + float(rtol) * abs(float(values["reference"]))
        for values in delta.values()
    )
