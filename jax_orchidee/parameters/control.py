"""Source-ordered owner for ``control.f90::control_initialize``.

Only configuration semantics owned by ``control_initialize`` live here.
Scientific parameter tables remain owned by the downstream initializers named
in :class:`ControlInitializationState.parameter_initializers`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from jax_orchidee.driver.init import parse_run_def, parse_run_def_bool


CONTROL_INITIALIZE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_parameters/control.f90::control_initialize lines 48-627",
)

PAPER_CONTROL_VALUES: dict[str, object] = {
    "soil_classif": "usda",
    "nscm": 12,
    "river_routing": True,
    "hydrol_cwrr": True,
    "do_irrigation": False,
    "do_fullirr": False,
    "ok_rotate": False,
    "cyc_rot_max": 1,
    "do_floodplains": False,
    "check_waterbal": False,
    "ok_explicitsnow": True,
    "ok_pc": False,
    "ok_peat": False,
    "peat_occur": False,
    "perma_peat": True,
    "ok_leak": True,
    "ok_stomate": True,
    "ok_co2": True,
    "ok_dgvm": False,
    "ok_bvoc": False,
    "ld_doc": False,
    "do_poor_soils": True,
    "nvm": 14,
    "impose_param": True,
}


class ControlInitializationError(ValueError):
    """A configuration cannot follow a valid ``control_initialize`` arm."""


@dataclass(frozen=True)
class ControlInitializationState:
    """Resolved module controls and downstream parameter-call contract."""

    dt_sechiba: float
    check_time: bool
    soil_classif: str
    nscm: int
    river_routing: bool
    erosion_module: bool
    ok_damreservoir: bool
    limit_rivdepos: bool
    hydrol_cwrr: bool
    do_irrigation: bool
    do_fullirr: bool
    ok_rotate: bool
    dyn_plntdt: bool
    nvm_plnt: bool
    nvm_rot: bool
    nvm_nfert: bool
    cyc_rot_max: int
    rot_cmd_max: int
    do_floodplains: bool
    check_waterbal: bool
    check_riverbal: bool
    ok_explicitsnow: bool
    ok_pc: bool
    ok_peat: bool
    peat_occur: bool
    perma_peat: bool
    ok_leak: bool
    ok_stomate: bool
    ok_co2: bool
    ok_dgvm: bool
    ok_bvoc: bool
    ok_leafage: bool
    ok_radcanopy: bool
    ok_multilayer: bool
    ok_pulse_nox: bool
    ok_bbgfertil_nox: bool
    ok_cropsfertil_nox: bool
    ok_co2bvoc_poss: bool
    ok_co2bvoc_wilk: bool
    ld_doc: bool
    do_poor_soils: bool
    ok_sechiba: bool
    ok_pheno: bool
    nvm: int
    impose_param: bool
    nslm: int
    ngrnd: int
    zmaxh: float
    diaglev: np.ndarray
    parameter_initializers: tuple[str, ...]
    provenance: tuple[str, ...] = CONTROL_INITIALIZE_PROVENANCE


def _casefold_values(values: Mapping[str, object]) -> dict[str, object]:
    return {str(key).casefold(): value for key, value in values.items()}


def _bool(values: Mapping[str, object], name: str, default: bool) -> bool:
    value = values.get(name.casefold(), default)
    try:
        return parse_run_def_bool(value)  # type: ignore[arg-type]
    except ValueError as exc:
        raise ControlInitializationError(f"invalid {name} boolean: {value!r}") from exc


def _parse_fortran_number(value: object) -> float:
    """Parse a finite scalar, including Fortran ``D`` exponent notation."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError("booleans are not numeric configuration values")
    normalized = value
    if isinstance(value, str):
        token = value.strip()
        if not token:
            raise ValueError("empty numeric configuration value")
        normalized = token.replace("D", "E").replace("d", "e")
    number = float(normalized)
    if not np.isfinite(number):
        raise ValueError("numeric configuration value must be finite")
    return number


def _int(values: Mapping[str, object], name: str, default: int) -> int:
    value = values.get(name.casefold(), default)
    try:
        number = _parse_fortran_number(value)
    except (TypeError, ValueError) as exc:
        raise ControlInitializationError(f"invalid {name} integer: {value!r}") from exc
    if not number.is_integer():
        raise ControlInitializationError(f"invalid {name} integer: {value!r}")
    return int(number)


def _float(values: Mapping[str, object], name: str, default: float) -> float:
    value = values.get(name.casefold(), default)
    try:
        return _parse_fortran_number(value)
    except (TypeError, ValueError) as exc:
        raise ControlInitializationError(f"invalid {name} number: {value!r}") from exc


def _diaglev_choisnel(zmaxh: float, nslm: int) -> np.ndarray:
    diaglev = np.empty(nslm, dtype=np.float64)
    denominator = 2 ** (nslm - 1) - 1
    for jv in range(1, nslm):
        diaglev[jv - 1] = zmaxh / denominator * ((2 ** (jv - 1) - 1) + (2**jv - 1)) / 2.0
    diaglev[-1] = zmaxh
    return diaglev


def control_initialize(
    values: Mapping[str, object],
    *,
    dt: float,
    znt: np.ndarray | None = None,
    nslm: int | None = None,
    zmaxt: float | None = None,
    ok_freeze_thermix: bool = False,
) -> ControlInitializationState:
    """Resolve the 37 PFT14-reachable owner arms in Fortran source order.

    ``znt`` and ``nslm`` are outputs of ``vertical_soil_init`` for CWRR, not
    reimplemented here. Invalid source configurations raise immediately.
    Fortran provenance: ``control.f90::control_initialize`` lines 48-627.
    """

    raw = _casefold_values(values)
    dt_sechiba = float(dt)
    if not np.isfinite(dt_sechiba) or dt_sechiba <= 0.0:
        raise ControlInitializationError("dt must be finite and positive")

    check_time = _bool(raw, "CHECKTIME", False)
    soil_classif = str(raw.get("soiltype_classif", "zobler")).strip().casefold()
    if soil_classif in {"zobler", "fao", "none"}:
        nscm = 3
    elif soil_classif == "usda":
        nscm = 12
    else:
        raise ControlInitializationError(
            "unsupported SOILTYPE_CLASSIF; expected zobler, fao, none, or usda"
        )

    river_routing = _bool(raw, "RIVER_ROUTING", False)
    erosion_module = _bool(raw, "EROSION_MODULE", False)
    ok_damreservoir = _bool(raw, "OK_DAMRESERVOIR", False)
    limit_rivdepos = _bool(raw, "LIMIT_RIVDEPOS", True)
    hydrol_cwrr = _bool(raw, "HYDROL_CWRR", False)
    do_irrigation = _bool(raw, "DO_IRRIGATION", False) if river_routing else False
    do_fullirr = _bool(raw, "DO_FULLIRR", False)

    ok_rotate = _bool(raw, "OK_ROTATE", False)
    dyn_plntdt = _bool(raw, "DYN_PLNTDT", False)
    nvm_plnt = _bool(raw, "NVM_PLNT", False)
    nvm_rot = _bool(raw, "NVM_ROT", False)
    nvm_nfert = _bool(raw, "NVM_NFERT", False)
    cyc_rot_max = _int(raw, "CYC_ROT_MAX", 1)
    rot_cmd_max = _int(raw, "ROT_CMD_MAX", 5)
    if not ok_rotate:
        cyc_rot_max = 1
    else:
        dyn_plntdt = False

    do_floodplains = _bool(raw, "DO_FLOODPLAINS", False) if river_routing else False
    check_waterbal = _bool(raw, "CHECK_WATERBAL", False)
    if check_waterbal and do_fullirr:
        check_waterbal = False
    # Line 243 resets check_waterbal before CHECK_RIVERBAL is read. Preserve
    # that source-order assignment rather than repairing the apparent typo.
    check_waterbal = False
    check_riverbal = _bool(raw, "CHECK_RIVERBAL", False)

    ok_explicitsnow = _bool(raw, "OK_EXPLICITSNOW", False)
    ok_pc = _bool(raw, "OK_PC", False)
    ok_peat = _bool(raw, "OK_PEAT", False)
    peat_occur = _bool(raw, "PEAT_OCCUR", False)
    perma_peat = _bool(raw, "PERMA_PEAT", False)
    ok_leak = _bool(raw, "OK_LEAK", False)
    if ok_leak:
        ok_pc = False

    ok_stomate = _bool(raw, "STOMATE_OK_STOMATE", False)
    ok_co2 = True if ok_stomate else _bool(raw, "STOMATE_OK_CO2", False)
    ok_dgvm = _bool(raw, "STOMATE_OK_DGVM", False)

    ok_bvoc = _bool(raw, "CHEMISTRY_BVOC", False)
    ok_leafage = _bool(raw, "CHEMISTRY_LEAFAGE", ok_bvoc)
    ok_radcanopy = _bool(raw, "CANOPY_EXTINCTION", ok_bvoc)
    ok_multilayer = _bool(raw, "CANOPY_MULTILAYER", ok_bvoc)
    ok_pulse_nox = _bool(raw, "NOx_RAIN_PULSE", ok_bvoc)
    ok_bbgfertil_nox = _bool(raw, "NOx_BBG_FERTIL", ok_bvoc)
    ok_cropsfertil_nox = _bool(raw, "NOx_FERTILIZERS_USE", ok_bvoc)
    ok_co2bvoc_poss = _bool(raw, "CO2_FOR_BVOC_POSSELL", False)
    ok_co2bvoc_wilk = _bool(raw, "CO2_FOR_BVOC_WILKINSON", False)
    ld_doc = _bool(raw, "LD_DOC", False)
    do_poor_soils = _bool(raw, "POOR_SOILS", False)

    ok_sechiba = True
    if ok_dgvm:
        ok_stomate = True
    if ok_multilayer and not ok_radcanopy:
        ok_radcanopy = True
    if ok_dgvm and ok_rotate:
        raise ControlInitializationError("STOMATE_OK_DGVM and OK_ROTATE cannot both be true")
    ok_pheno = True

    nvm = _int(raw, "NVM", 13)
    impose_param = _bool(raw, "IMPOSE_PARAM", True)
    if nvm < 1:
        raise ControlInitializationError("NVM must be positive")

    if hydrol_cwrr:
        if znt is None or nslm is None:
            raise ControlInitializationError("CWRR requires vertical_soil_init outputs znt and nslm")
        znt_array = np.asarray(znt, dtype=np.float64)
        if znt_array.ndim != 1 or nslm < 1 or nslm > znt_array.size:
            raise ControlInitializationError("CWRR znt must be one-dimensional with at least nslm levels")
        if not np.all(np.isfinite(znt_array)) or np.any(np.diff(znt_array) <= 0.0):
            raise ControlInitializationError("CWRR znt levels must be finite and strictly increasing")
        ngrnd = int(znt_array.size)
        zmaxh = _float(raw, "DEPTH_MAX_H", 2.0)
        diaglev = znt_array[:nslm].copy()
        if ok_freeze_thermix:
            thermal_depth = _float(raw, "DEPTH_MAX_T", 38.0) if zmaxt is None else float(zmaxt)
            if thermal_depth < 11.0:
                raise ControlInitializationError(
                    "OK_FREEZE_THERMIX with CWRR requires DEPTH_MAX_T >= 11 m"
                )
    else:
        zmaxh = _float(raw, "DEPTH_MAX_H", 4.0)
        ngrnd = _int(raw, "THERMOSOIL_NBLEV", 7)
        nslm = 11
        if ngrnd < 1 or zmaxh <= 0.0:
            raise ControlInitializationError("Choisnel soil depth and level count must be positive")
        if ok_freeze_thermix and ngrnd < 11:
            raise ControlInitializationError(
                "OK_FREEZE_THERMIX with Choisnel requires THERMOSOIL_NBLEV >= 11"
            )
        diaglev = _diaglev_choisnel(zmaxh, nslm)

    parameter_initializers = ["pft_parameters_main", "activate_sub_models", "veget_config"]
    if impose_param:
        parameter_initializers.append("config_pft_parameters")
    if ok_sechiba and impose_param:
        parameter_initializers.extend(("config_sechiba_parameters", "config_sechiba_pft_parameters"))
    parameter_initializers.append("config_soil_parameters")
    if ok_co2 and impose_param:
        parameter_initializers.append("config_co2_parameters")
    if ok_stomate and impose_param:
        parameter_initializers.extend(("config_stomate_parameters", "config_stomate_pft_parameters"))
    if ok_dgvm and impose_param:
        parameter_initializers.append("config_dgvm_parameters")

    return ControlInitializationState(
        dt_sechiba=dt_sechiba,
        check_time=check_time,
        soil_classif=soil_classif,
        nscm=nscm,
        river_routing=river_routing,
        erosion_module=erosion_module,
        ok_damreservoir=ok_damreservoir,
        limit_rivdepos=limit_rivdepos,
        hydrol_cwrr=hydrol_cwrr,
        do_irrigation=do_irrigation,
        do_fullirr=do_fullirr,
        ok_rotate=ok_rotate,
        dyn_plntdt=dyn_plntdt,
        nvm_plnt=nvm_plnt,
        nvm_rot=nvm_rot,
        nvm_nfert=nvm_nfert,
        cyc_rot_max=cyc_rot_max,
        rot_cmd_max=rot_cmd_max,
        do_floodplains=do_floodplains,
        check_waterbal=check_waterbal,
        check_riverbal=check_riverbal,
        ok_explicitsnow=ok_explicitsnow,
        ok_pc=ok_pc,
        ok_peat=ok_peat,
        peat_occur=peat_occur,
        perma_peat=perma_peat,
        ok_leak=ok_leak,
        ok_stomate=ok_stomate,
        ok_co2=ok_co2,
        ok_dgvm=ok_dgvm,
        ok_bvoc=ok_bvoc,
        ok_leafage=ok_leafage,
        ok_radcanopy=ok_radcanopy,
        ok_multilayer=ok_multilayer,
        ok_pulse_nox=ok_pulse_nox,
        ok_bbgfertil_nox=ok_bbgfertil_nox,
        ok_cropsfertil_nox=ok_cropsfertil_nox,
        ok_co2bvoc_poss=ok_co2bvoc_poss,
        ok_co2bvoc_wilk=ok_co2bvoc_wilk,
        ld_doc=ld_doc,
        do_poor_soils=do_poor_soils,
        ok_sechiba=ok_sechiba,
        ok_pheno=ok_pheno,
        nvm=nvm,
        impose_param=impose_param,
        nslm=nslm,
        ngrnd=ngrnd,
        zmaxh=zmaxh,
        diaglev=diaglev,
        parameter_initializers=tuple(parameter_initializers),
    )


def validate_paper_control_configuration(state: ControlInitializationState) -> None:
    """Reject any control state outside the fixed paper protocol."""

    mismatches = {
        name: (getattr(state, name), expected)
        for name, expected in PAPER_CONTROL_VALUES.items()
        if getattr(state, name) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{name}={actual!r} (expected {expected!r})"
            for name, (actual, expected) in mismatches.items()
        )
        raise ControlInitializationError(f"not the fixed paper control configuration: {details}")


def paper_control_initialize(
    used_run_def_path: str | Path,
    *,
    dt: float = 1800.0,
) -> ControlInitializationState:
    """Initialize and validate the materialized paper CWRR configuration."""

    from jax_orchidee.driver.stomate_boundary import (
        paper_case_cwrr_vertical_soil_grid_from_used_run_def,
    )

    path = Path(used_run_def_path)
    values = parse_run_def(path)
    grid = paper_case_cwrr_vertical_soil_grid_from_used_run_def(path)
    state = control_initialize(
        values,
        dt=dt,
        znt=np.asarray(grid.znt),
        nslm=int(np.asarray(grid.znh).size),
        zmaxt=float(np.asarray(grid.zlt)[-1]),
        ok_freeze_thermix=_bool(_casefold_values(values), "OK_FREEZE_THERMIX", False),
    )
    validate_paper_control_configuration(state)
    return state
