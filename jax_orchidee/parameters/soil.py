"""Soil parameter initialization from ``config_soil_parameters``.

Fortran provenance: ``src_parameters/constantes_soil.f90``, subroutine
``config_soil_parameters``, lines 54-485.  Reads, dependent defaults, and
guards below intentionally follow source order.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np


class SoilParameterError(ValueError):
    """Fatal ``ipslerr_p`` outcome from ``config_soil_parameters``."""


@dataclass(frozen=True)
class SoilParameters:
    """Saved variables touched by ``config_soil_parameters``."""

    so_capa_dry: float = 1.80e6
    so_cond_dry: float = 0.40
    so_capa_wet: float = 3.03e6
    so_cond_wet: float = 1.89
    sn_cond: float = 0.3
    sn_dens: float = 330.0
    sn_capa: float = 2100.0 * 330.0
    mx_eau_nobio: float = 150.0
    qsintcst: float = 0.1
    so_discretization_method: int = 0
    min_drain: float = 0.001
    max_drain: float = 0.1
    exp_drain: float = 1.5
    rsol_cste: float = 33.0e3
    hcrit_litter: float = 0.08
    tau_peat: float = 3.1536e8
    z_tau: float = 1.0e6
    read_reftemp: bool = False
    ok_freeze_thermix: bool = False
    ok_ecorr: bool = False
    poros: float = 0.41
    fr_dt: float = 2.0
    ok_snowfact: bool = False
    smcmax_fao: tuple[float, float, float] = (0.41, 0.43, 0.41)
    ok_freeze_cwrr: bool = False
    ok_thermodynamical_freezing: bool = True
    check_cwrr: bool = False
    check_cwrr2: bool = False


def _normalized(overrides: Mapping[str, object] | None) -> dict[str, object]:
    return {} if overrides is None else {str(key).strip().upper(): value for key, value in overrides.items()}


def _boolean(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    token = str(value).strip().strip(".").upper()
    if token in {"T", "TRUE", "Y", "YES", "1"}:
        return True
    if token in {"F", "FALSE", "N", "NO", "0"}:
        return False
    raise ValueError(f"invalid Fortran logical value {value!r}")


def _fortran_float(value: object) -> float:
    parsed = float(str(value).strip().replace("D", "E").replace("d", "e"))
    if not np.isfinite(parsed):
        raise ValueError(f"invalid finite Fortran real value {value!r}")
    return parsed


def _scalar(values: Mapping[str, object], key: str, current: float) -> float:
    return _fortran_float(values[key]) if key in values else current


def _integer(values: Mapping[str, object], key: str, current: int) -> int:
    if key not in values:
        return current
    parsed = _fortran_float(values[key])
    if not parsed.is_integer():
        raise ValueError(f"{key} must be an integer, got {values[key]!r}")
    return int(parsed)


def _logical(values: Mapping[str, object], key: str, current: bool) -> bool:
    return _boolean(values[key]) if key in values else current


def _vector(values: Mapping[str, object], key: str, current: Sequence[float]) -> tuple[float, ...]:
    result = list(current)
    if key in values:
        raw = values[key]
        if isinstance(raw, str):
            raw = raw.replace("(/", "").replace("/)", "").replace(",", " ").split()
        array = np.asarray(raw, dtype=object)
        if array.ndim != 1:
            raise ValueError(f"{key} must be a vector")
        result = [_fortran_float(value) for value in array.tolist()]
        if len(result) != len(current):
            raise ValueError(f"{key} must contain {len(current)} values")
    for index in range(1, len(result) + 1):
        indexed = f"{key}__{index:05d}"
        if indexed in values:
            result[index - 1] = _fortran_float(values[indexed])
    return tuple(result)


def _positive(key: str, value: float) -> None:
    if value <= 0.0:
        raise SoilParameterError(f"{key} must be positive")


def config_soil_parameters(
    overrides: Mapping[str, object] | None = None,
    *,
    ok_sechiba: bool,
    impose_param: bool,
    hydrol_cwrr: bool,
    initial: SoilParameters | None = None,
) -> SoilParameters:
    """Apply ``getin_p`` overrides and guards in exact Fortran source order."""

    values = _normalized(overrides)
    state = SoilParameters() if initial is None else initial

    # constantes_soil.f90:70-317
    if ok_sechiba and impose_param:
        value = _scalar(values, "DRY_SOIL_HEAT_CAPACITY", state.so_capa_dry)
        _positive("DRY_SOIL_HEAT_CAPACITY", value)
        state = replace(state, so_capa_dry=value)

        value = _scalar(values, "DRY_SOIL_HEAT_COND", state.so_cond_dry)
        _positive("DRY_SOIL_HEAT_COND", value)
        state = replace(state, so_cond_dry=value)

        value = _scalar(values, "WET_SOIL_HEAT_CAPACITY", state.so_capa_wet)
        _positive("WET_SOIL_HEAT_CAPACITY", value)
        state = replace(state, so_capa_wet=value)

        value = _scalar(values, "WET_SOIL_HEAT_COND", state.so_cond_wet)
        _positive("WET_SOIL_HEAT_COND", value)
        state = replace(state, so_cond_wet=value)

        value = _scalar(values, "SNOW_HEAT_COND", state.sn_cond)
        _positive("SNOW_HEAT_COND", value)
        state = replace(state, sn_cond=value)

        density = _scalar(values, "SNOW_DENSITY", state.sn_dens)
        _positive("SNOW_DENSITY", density)
        state = replace(state, sn_dens=density, sn_capa=2100.0 * density)

        value = _scalar(values, "NOBIO_WATER_CAPAC_VOLUMETRI", state.mx_eau_nobio)
        _positive("NOBIO_WATER_CAPAC_VOLUMETRI", value)
        state = replace(state, mx_eau_nobio=value)

        value = _scalar(values, "SECHIBA_QSINT", state.qsintcst)
        _positive("SECHIBA_QSINT", value)
        state = replace(state, qsintcst=value)

        method = _integer(values, "SOIL_LAYERS_DISCRE_METHOD", 0)
        if method < 0 or method > 2:
            raise SoilParameterError("SOIL_LAYERS_DISCRE_METHOD must be in [0, 2]")
        state = replace(state, so_discretization_method=method)

        if not hydrol_cwrr:
            minimum = _scalar(values, "CHOISNEL_DIFF_MIN", state.min_drain)
            _positive("CHOISNEL_DIFF_MIN", minimum)
            state = replace(state, min_drain=minimum)

            maximum = _scalar(values, "CHOISNEL_DIFF_MAX", state.max_drain)
            if maximum <= 0.0 or maximum <= minimum:
                raise SoilParameterError("CHOISNEL_DIFF_MAX must be positive and greater than CHOISNEL_DIFF_MIN")
            state = replace(state, max_drain=maximum)

            exponent = _scalar(values, "CHOISNEL_DIFF_EXP", state.exp_drain)
            _positive("CHOISNEL_DIFF_EXP", exponent)
            state = replace(state, exp_drain=exponent)

            resistance = _scalar(values, "CHOISNEL_RSOL_CSTE", state.rsol_cste)
            _positive("CHOISNEL_RSOL_CSTE", resistance)
            state = replace(state, rsol_cste=resistance)

            litter = _scalar(values, "HCRIT_LITTER", state.hcrit_litter)
            _positive("HCRIT_LITTER", litter)
            state = replace(state, hcrit_litter=litter)

    # Lines 320-336: these reads occur regardless of the outer option gate.
    state = replace(
        state,
        tau_peat=_scalar(values, "TAU_PEAT", state.tau_peat),
        z_tau=_scalar(values, "Z_TAU", state.z_tau),
    )
    ok_freeze = _logical(values, "OK_FREEZE", False)

    # Lines 346-379: each default follows OK_FREEZE, then its own key wins.
    state = replace(
        state,
        read_reftemp=_logical(values, "READ_REFTEMP", ok_freeze),
        ok_freeze_thermix=_logical(values, "OK_FREEZE_THERMIX", ok_freeze),
        ok_ecorr=_logical(values, "OK_ECORR", ok_freeze),
    )
    if state.ok_ecorr and not state.ok_freeze_thermix:
        raise SoilParameterError("OK_ECORR cannot be activated without OK_FREEZE_THERMIX")

    # Lines 385-444.
    state = replace(
        state,
        poros=_scalar(values, "POROS", 0.41),
        fr_dt=_scalar(values, "FR_DT", 2.0),
        ok_snowfact=_logical(values, "OK_SNOWFACT", ok_freeze),
        smcmax_fao=_vector(values, "SMCMAX_FAO", state.smcmax_fao),
        ok_freeze_cwrr=_logical(values, "OK_FREEZE_CWRR", ok_freeze),
    )
    if state.ok_freeze_cwrr:
        state = replace(
            state,
            ok_thermodynamical_freezing=_logical(values, "OK_THERMODYNAMICAL_FREEZING", True),
        )

    # Lines 463-483 explicitly reset these defaults on every call.
    return replace(
        state,
        check_cwrr=_logical(values, "CHECK_CWRR", False),
        check_cwrr2=_logical(values, "CHECK_CWRR2", False),
    )


__all__ = ["SoilParameterError", "SoilParameters", "config_soil_parameters"]
