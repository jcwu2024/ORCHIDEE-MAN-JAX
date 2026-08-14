"""Named AGB/BGB tendency normalization for successor training."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Sequence

import jax.numpy as jnp
import numpy as np

from .long_horizon_objective import DEFAULT_HORIZONS, MaskedTrajectory, NormalizedError

NAMED_BIOMASS_TENDENCY_FIELDS = (
    "aboveground_biomass_carbon",
    "belowground_biomass_carbon",
)
NAMED_BIOMASS_TENDENCY_COMPONENTS = (
    "tendency_mse",
    "tendency_bias_squared",
)


class NamedBiomassTendencyHorizon(NamedTuple):
    errors: Mapping[str, NormalizedError]
    macro_mse: Any
    macro_bias_squared: Any


class NamedBiomassTendencyResult(NamedTuple):
    total: Any
    tendency_mse: Any
    tendency_bias_squared: Any
    by_horizon: Mapping[int, NamedBiomassTendencyHorizon]


class NamedBiomassTendencyScaleStatistics(NamedTuple):
    scales: Mapping[int, Mapping[str, float]]
    audit: Mapping[str, Any]


class BiomassIdentityResidual(NamedTuple):
    value: Any
    defined: Any


class BiomassIdentityRoundoffAudit(NamedTuple):
    residual: Any
    tolerance: Any
    tolerance_ratio: Any
    defined: Any


def _horizons(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or any(value <= 0 for value in result):
        raise ValueError("biomass tendency horizons must be positive")
    if len(set(result)) != len(result):
        raise ValueError("biomass tendency horizons must be unique")
    return result


def _normalized_error(
    predicted: Any,
    expected: Any,
    defined: Any,
    scale: float,
) -> NormalizedError:
    predicted = jnp.asarray(predicted, dtype=jnp.float64)
    expected = jnp.asarray(expected, dtype=jnp.float64)
    defined = jnp.asarray(defined, dtype=jnp.bool_)
    valid = defined & jnp.isfinite(predicted) & jnp.isfinite(expected)
    residual = jnp.where(valid, (predicted - expected) / scale, 0.0)
    count = jnp.sum(valid, dtype=jnp.float64)
    denominator = jnp.maximum(count, 1.0)
    mean = jnp.where(count > 0.0, jnp.sum(residual) / denominator, 0.0)
    mse = jnp.where(
        count > 0.0,
        jnp.sum(residual * residual, dtype=jnp.float64) / denominator,
        0.0,
    )
    return NormalizedError(mse, mean, mean * mean, count)


def _macro(errors: Sequence[NormalizedError], attribute: str) -> Any:
    active = jnp.stack(tuple(error.count > 0.0 for error in errors))
    values = jnp.stack(tuple(getattr(error, attribute) for error in errors))
    count = jnp.sum(active, dtype=jnp.float64)
    return jnp.where(
        count > 0.0,
        jnp.sum(jnp.where(active, values, 0.0), dtype=jnp.float64)
        / jnp.maximum(count, 1.0),
        0.0,
    )


def _validate_trajectory_registry(
    predicted: Mapping[str, MaskedTrajectory],
    expected: Mapping[str, MaskedTrajectory],
) -> int:
    required = set(NAMED_BIOMASS_TENDENCY_FIELDS)
    for label, registry in (("prediction", predicted), ("teacher", expected)):
        missing = sorted(required - set(registry))
        if missing:
            raise ValueError(f"biomass tendency {label} fields missing: {missing}")
    available = None
    for name in NAMED_BIOMASS_TENDENCY_FIELDS:
        actual = predicted[name]
        wanted = expected[name]
        actual_value = jnp.asarray(actual.value)
        wanted_value = jnp.asarray(wanted.value)
        if actual_value.shape != wanted_value.shape or actual_value.ndim < 1:
            raise ValueError(f"biomass tendency trajectory shape drift for {name}")
        if (
            jnp.asarray(actual.defined).shape != actual_value.shape
            or jnp.asarray(wanted.defined).shape != wanted_value.shape
        ):
            raise ValueError(f"biomass tendency mask shape drift for {name}")
        if available is None:
            available = int(actual_value.shape[0])
        elif available != int(actual_value.shape[0]):
            raise ValueError("biomass tendency trajectory lengths differ")
    if available is None:
        raise ValueError("biomass tendency field registry is empty")
    return available


def _validate_scales(
    scales: Mapping[int, Mapping[str, Any]],
    horizons: Sequence[int],
) -> dict[int, dict[str, float]]:
    horizon_tuple = _horizons(horizons)
    if set(scales) != set(horizon_tuple):
        raise ValueError("biomass tendency scale horizon registry drift")
    result = {}
    for horizon in horizon_tuple:
        current = scales[horizon]
        if tuple(current) != NAMED_BIOMASS_TENDENCY_FIELDS:
            raise ValueError("biomass tendency scale field registry drift")
        resolved = {name: float(current[name]) for name in current}
        if any(not np.isfinite(value) or value <= 0.0 for value in resolved.values()):
            raise ValueError("biomass tendency scales must be finite and positive")
        result[horizon] = resolved
    return result


def named_biomass_tendency_objective(
    predicted_stocks: Mapping[str, MaskedTrajectory],
    expected_stocks: Mapping[str, MaskedTrajectory],
    *,
    scales: Mapping[int, Mapping[str, Any]],
    coefficients: Mapping[str, Any],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> NamedBiomassTendencyResult:
    """Compare named biomass changes using train-only tendency scales.

    Candidate and Teacher rollouts share the same day-start state. Therefore
    ``(predicted_H - start) - (teacher_H - start)`` is exactly the terminal
    residual used here. The new training signal comes from the frozen
    tendency scale, not from redundant subtraction of the common start state.
    AGB/BGB source ownership is inherited from
    ``derive_area_weighted_biomass_carbon`` and its audited Fortran routing.
    """

    horizon_tuple = _horizons(horizons)
    available = _validate_trajectory_registry(predicted_stocks, expected_stocks)
    if max(horizon_tuple) > available:
        raise ValueError("biomass tendency horizon exceeds available trajectory")
    resolved_scales = _validate_scales(scales, horizon_tuple)
    if tuple(coefficients) != NAMED_BIOMASS_TENDENCY_COMPONENTS:
        raise ValueError("biomass tendency coefficient registry drift")

    by_horizon = {}
    for horizon in horizon_tuple:
        errors = {}
        for name in NAMED_BIOMASS_TENDENCY_FIELDS:
            actual = predicted_stocks[name]
            wanted = expected_stocks[name]
            defined = (
                jnp.asarray(actual.defined[horizon - 1], dtype=jnp.bool_)
                & jnp.asarray(wanted.defined[horizon - 1], dtype=jnp.bool_)
            )
            errors[name] = _normalized_error(
                actual.value[horizon - 1],
                wanted.value[horizon - 1],
                defined,
                resolved_scales[horizon][name],
            )
        values = tuple(errors.values())
        by_horizon[horizon] = NamedBiomassTendencyHorizon(
            errors=errors,
            macro_mse=_macro(values, "mse"),
            macro_bias_squared=_macro(values, "bias_squared"),
        )

    denominator = jnp.asarray(len(horizon_tuple), dtype=jnp.float64)
    tendency_mse = sum(value.macro_mse for value in by_horizon.values()) / denominator
    tendency_bias_squared = (
        sum(value.macro_bias_squared for value in by_horizon.values()) / denominator
    )
    total = (
        jnp.asarray(coefficients["tendency_mse"], dtype=jnp.float64) * tendency_mse
        + jnp.asarray(
            coefficients["tendency_bias_squared"], dtype=jnp.float64
        )
        * tendency_bias_squared
    )
    return NamedBiomassTendencyResult(
        total=total,
        tendency_mse=tendency_mse,
        tendency_bias_squared=tendency_bias_squared,
        by_horizon=by_horizon,
    )


def fit_named_biomass_tendency_scales(
    initial_stocks: Mapping[str, MaskedTrajectory],
    teacher_stocks: Mapping[str, MaskedTrajectory],
    *,
    stock_scales: Mapping[str, Any],
    floor_ratio: float,
    minimum_scale: float = 1.0e-12,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> NamedBiomassTendencyScaleStatistics:
    """Fit finite train-only AGB/BGB change scales with a stock-relative floor."""

    horizon_tuple = _horizons(horizons)
    if not 0.0 < floor_ratio <= 1.0:
        raise ValueError("biomass tendency floor ratio must be in (0, 1]")
    if not np.isfinite(minimum_scale) or minimum_scale <= 0.0:
        raise ValueError("biomass tendency minimum scale must be positive")
    required = set(NAMED_BIOMASS_TENDENCY_FIELDS)
    for label, registry in (
        ("initial", initial_stocks),
        ("teacher", teacher_stocks),
        ("stock scale", stock_scales),
    ):
        missing = sorted(required - set(registry))
        if missing:
            raise ValueError(f"biomass tendency {label} fields missing: {missing}")

    scales = {}
    records = {}
    for horizon in horizon_tuple:
        horizon_scales = {}
        horizon_records = {}
        for name in NAMED_BIOMASS_TENDENCY_FIELDS:
            initial = initial_stocks[name]
            teacher = teacher_stocks[name]
            initial_value = np.asarray(initial.value, dtype=np.float64)
            initial_defined = np.asarray(initial.defined, dtype=np.bool_)
            teacher_value = np.asarray(teacher.value, dtype=np.float64)
            teacher_defined = np.asarray(teacher.defined, dtype=np.bool_)
            if teacher_value.ndim < 1 or max(horizon_tuple) > teacher_value.shape[0]:
                raise ValueError("biomass tendency scale trajectory is too short")
            terminal = teacher_value[horizon - 1]
            terminal_defined = teacher_defined[horizon - 1]
            if (
                initial_value.shape != terminal.shape
                or initial_defined.shape != initial_value.shape
                or teacher_defined.shape != teacher_value.shape
            ):
                raise ValueError(f"biomass tendency scale shape drift for {name}")
            valid = (
                initial_defined
                & terminal_defined
                & np.isfinite(initial_value)
                & np.isfinite(terminal)
            )
            tendency = (terminal - initial_value)[valid]
            count = int(tendency.size)
            empirical = (
                float(np.std(tendency, ddof=1, dtype=np.float64))
                if count > 1
                else 0.0
            )
            stock_scale = float(stock_scales[name])
            if not np.isfinite(stock_scale) or stock_scale <= 0.0:
                raise ValueError("biomass tendency stock scales must be positive")
            floor = floor_ratio * stock_scale
            resolved = max(empirical, floor, minimum_scale)
            horizon_scales[name] = resolved
            horizon_records[name] = {
                "count": count,
                "empirical_standard_deviation": empirical,
                "stock_scale": stock_scale,
                "stock_relative_floor": floor,
                "resolved_scale": resolved,
                "floor_applied": empirical < floor,
            }
        scales[horizon] = horizon_scales
        records[horizon] = horizon_records
    return NamedBiomassTendencyScaleStatistics(
        scales=scales,
        audit={
            "policy": "max_empirical_standard_deviation_stock_relative_floor_and_minimum",
            "horizons": list(horizon_tuple),
            "fields": list(NAMED_BIOMASS_TENDENCY_FIELDS),
            "floor_ratio": float(floor_ratio),
            "minimum_scale": float(minimum_scale),
            "records": records,
        },
    )


def biomass_stock_identity_residual(
    stocks: Mapping[str, MaskedTrajectory],
) -> BiomassIdentityResidual:
    """Return total-live minus AGB+BGB under their common defined mask."""

    required = {
        "aboveground_biomass_carbon",
        "belowground_biomass_carbon",
        "total_live_biomass_carbon",
    }
    missing = sorted(required - set(stocks))
    if missing:
        raise ValueError(f"biomass identity fields missing: {missing}")
    above = stocks["aboveground_biomass_carbon"]
    below = stocks["belowground_biomass_carbon"]
    total = stocks["total_live_biomass_carbon"]
    shapes = {
        jnp.asarray(above.value).shape,
        jnp.asarray(below.value).shape,
        jnp.asarray(total.value).shape,
    }
    if len(shapes) != 1:
        raise ValueError("biomass identity value shapes differ")
    defined_shapes = {
        jnp.asarray(above.defined).shape,
        jnp.asarray(below.defined).shape,
        jnp.asarray(total.defined).shape,
    }
    if defined_shapes != shapes:
        raise ValueError("biomass identity mask shapes differ")
    defined = (
        jnp.asarray(above.defined, dtype=jnp.bool_)
        & jnp.asarray(below.defined, dtype=jnp.bool_)
        & jnp.asarray(total.defined, dtype=jnp.bool_)
    )
    residual = jnp.where(
        defined,
        jnp.asarray(total.value, dtype=jnp.float64)
        - jnp.asarray(above.value, dtype=jnp.float64)
        - jnp.asarray(below.value, dtype=jnp.float64),
        0.0,
    )
    return BiomassIdentityResidual(value=residual, defined=defined)


def biomass_stock_identity_roundoff_audit(
    stocks: Mapping[str, MaskedTrajectory],
    *,
    roundoff_units: int,
    magnitude_floor: float = 1.0,
) -> BiomassIdentityRoundoffAudit:
    """Audit the AGB/BGB identity against a scale-aware float64 error bound."""

    if roundoff_units < 1:
        raise ValueError("biomass identity roundoff units must be positive")
    if not np.isfinite(magnitude_floor) or magnitude_floor <= 0.0:
        raise ValueError("biomass identity magnitude floor must be positive")
    identity = biomass_stock_identity_residual(stocks)
    above = jnp.asarray(
        stocks["aboveground_biomass_carbon"].value,
        dtype=jnp.float64,
    )
    below = jnp.asarray(
        stocks["belowground_biomass_carbon"].value,
        dtype=jnp.float64,
    )
    total = jnp.asarray(
        stocks["total_live_biomass_carbon"].value,
        dtype=jnp.float64,
    )
    magnitude = jnp.maximum(
        jnp.maximum(jnp.abs(total), jnp.abs(above) + jnp.abs(below)),
        jnp.asarray(magnitude_floor, dtype=jnp.float64),
    )
    tolerance = (
        jnp.asarray(roundoff_units, dtype=jnp.float64)
        * jnp.asarray(np.finfo(np.float64).eps, dtype=jnp.float64)
        * magnitude
    )
    ratio = jnp.where(
        identity.defined,
        jnp.abs(identity.value) / tolerance,
        0.0,
    )
    return BiomassIdentityRoundoffAudit(
        residual=identity.value,
        tolerance=tolerance,
        tolerance_ratio=ratio,
        defined=identity.defined,
    )
