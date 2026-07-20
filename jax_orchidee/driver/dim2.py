"""Source-order offline DIM2 driver transition for the paper PFT14 path.

The module owns only ``dim2_driver.f90::driver`` lines 285-1441.  NetCDF
forcing/domain IO remains in :mod:`jax_orchidee.driver.bundle`, while the
SECHIBA/STOMATE process transition is supplied through explicit callbacks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
import math
import re
from typing import Any

import numpy as np

from jax_orchidee.driver.bundle import DriverStepBundle
from jax_orchidee.driver.sechiba_boundary import IntersurfFirstStepPayload, build_intersurf_first_step_payload


ONE_DAY = 86400.0
CP_AIR = 1004.675
GRAVITY = 9.80665
MOLAR_GAS_CONSTANT_AIR = 287.05
VAL_EXP = 999999.0


@dataclass(frozen=True)
class Dim2Switches:
    """Audited switches at the DIM2 boundary.

    ``river_routing`` records the paper run value but routing remains owned by
    the intersurf/SECHIBA callback. ``driver_routing`` rejects attempts to run
    routing in this source-owner module.
    """

    weathergen: bool = False
    relaxation: bool = False
    watchout: bool = False
    river_routing: bool = True
    driver_routing: bool = False


@dataclass(frozen=True)
class Dim2RestartTime:
    """Time metadata returned by IOIPSL ``restini`` (lines 361-401)."""

    itau_dep_rest: int = 0
    date0_rest: float | None = None
    dt_rest: float | None = None
    driver_reset_time: bool = False


@dataclass(frozen=True)
class Dim2TimeControl:
    """Resolved forcing/restart chronology from lines 285-305 and 379-573."""

    dt: float
    dt_force: float
    split: int
    itau_dep_rest: int
    itau_dep: int
    itau_fin: int
    itau_len: int
    date0: float
    date0_rest: float
    dt_rest: float
    for_offset: int
    istp: int
    istp_old: int
    split_start: int


@dataclass(frozen=True)
class Dim2ForcingControls:
    """Forcing interpolation and precipitation spread, lines 598-696."""

    no_inter: bool
    inter_lin: bool
    nb_spread: int
    netrad_cons: bool


@dataclass(frozen=True)
class Dim2TimeIndex:
    """One iteration of the nested ``it``/``is`` loops (lines 802-837)."""

    it: int
    it_force: int
    isplit: int
    istp: int
    forcing_model_tstep: int
    julian: float
    lstep_init: bool
    lstep_last: bool


@dataclass(frozen=True)
class Dim2DriverRestart:
    """Driver-owned restart fields written at lines 1415-1429."""

    fluxsens: np.ndarray | None = None
    vevapp: np.ndarray | None = None
    old_zlev: np.ndarray | None = None
    old_qair: np.ndarray | None = None
    old_eair: np.ndarray | None = None
    rau_old: np.ndarray | None = None
    albedo_vis: np.ndarray | None = None
    albedo_nir: np.ndarray | None = None
    z0: np.ndarray | None = None
    petAcoef: np.ndarray | None = None
    petBcoef: np.ndarray | None = None
    peqAcoef: np.ndarray | None = None
    peqBcoef: np.ndarray | None = None


@dataclass(frozen=True)
class Dim2IntersurfBoundary:
    """Exact DIM2 values adjacent to an intersurf call."""

    index: Dim2TimeIndex
    payload: IntersurfFirstStepPayload
    eair: np.ndarray
    swnet: np.ndarray
    coszang: np.ndarray | None
    rau: np.ndarray
    cdrag: np.ndarray
    petAcoef: np.ndarray
    petBcoef: np.ndarray
    peqAcoef: np.ndarray
    peqBcoef: np.ndarray


@dataclass(frozen=True)
class Dim2IntersurfOutput:
    """Fields returned to DIM2 by initialize/main intersurf calls."""

    vevapp: np.ndarray
    fluxsens: np.ndarray
    fluxlat: np.ndarray
    coastalflow: np.ndarray
    riverflow: np.ndarray
    tsol_rad: np.ndarray
    temp_sol_new: np.ndarray
    qsurf: np.ndarray
    albedo: np.ndarray
    emis: np.ndarray
    z0: np.ndarray
    component_state: Any = None


@dataclass(frozen=True)
class Dim2StepOutput:
    index: Dim2TimeIndex
    initialize_boundary: Dim2IntersurfBoundary | None
    main_boundary: Dim2IntersurfBoundary
    output: Dim2IntersurfOutput
    dtdt: np.ndarray


@dataclass(frozen=True)
class Dim2DriverResult:
    """Final source-order state and driver restart packet."""

    time: Dim2TimeControl
    steps: tuple[Dim2StepOutput, ...]
    final_output: Dim2IntersurfOutput
    initial_restart: Dim2DriverRestart
    restart: Dim2DriverRestart
    istp_old: int
    component_state: Any


def validate_dim2_switches(switches: Dim2Switches) -> None:
    """Reject branches outside the audited offline paper transition."""

    unsupported = []
    if switches.weathergen:
        unsupported.append("WEATHERGEN=y")
    if switches.relaxation:
        unsupported.append("RELAXATION=y")
    if switches.watchout:
        unsupported.append("WATCHOUT forcing")
    if switches.driver_routing:
        unsupported.append("driver-owned routing")
    if not switches.river_routing:
        unsupported.append("RIVER_ROUTING=n (paper run requires y; execution is delegated to intersurf)")
    if unsupported:
        raise NotImplementedError("unsupported dim2 paper-path switch(es): " + ", ".join(unsupported))


def resolve_dim2_forcing_controls(
    *,
    dt_force: float,
    dt: float,
    split: int,
    inter_lin: bool = False,
    spread_override: int | None = None,
    netrad_cons_override: bool | None = None,
) -> Dim2ForcingControls:
    """Reproduce ``dim2_driver.f90::driver`` lines 620-696.

    Fortran assigns real quotients to integer ``nb_spread`` and therefore
    truncates toward zero. ``NO_INTER`` is read but then overwritten solely
    from ``INTER_LIN`` at lines 639-643.
    """

    if dt_force <= 0.0 or dt <= 0.0:
        raise ValueError("dt_force and dt must be positive")
    if split < 1:
        raise ValueError("split must be positive")
    linear = bool(inter_lin)
    no_inter = not linear
    if split == 1:
        no_inter = True
        linear = False
    if dt_force >= 3.0 * 3600.0:
        nb_spread = int(0.5 * (dt_force / dt))
    else:
        nb_spread = int(dt_force / dt)
    if spread_override is not None:
        nb_spread = int(spread_override)
    if nb_spread > split:
        nb_spread = split
    elif split == 1:
        nb_spread = 1
    netrad_cons = bool(netrad_cons_override) if netrad_cons_override is not None else True
    if not linear:
        netrad_cons = False
    return Dim2ForcingControls(no_inter, linear, nb_spread, netrad_cons)


def _fortran_nint(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0.0 else math.ceil(value - 0.5)


_TIME_PART = re.compile(r"([+-]?\d+(?:\.\d*)?)([SDMY]?)", re.IGNORECASE)


def time_length_to_forcing_steps(value: str | int, *, dt_force: float) -> int:
    """Implement the documented ``tlen2itau`` DIM2 forms (lines 495-556).

    Bare integers are forcing steps. S/D/M/Y suffixes use seconds, days, and
    the documented fixed 365-day year; one month is one twelfth of that year.
    """

    if isinstance(value, int):
        return value
    token = str(value).strip().upper().replace(" ", "")
    if re.fullmatch(r"[+-]?\d+", token):
        return int(token)
    position = 0
    seconds = 0.0
    factors = {"S": 1.0, "D": ONE_DAY, "M": 365.0 * ONE_DAY / 12.0, "Y": 365.0 * ONE_DAY}
    for match in _TIME_PART.finditer(token):
        if match.start() != position or not match.group(2):
            raise ValueError(f"unsupported DIM2 time length: {value!r}")
        seconds += float(match.group(1)) * factors[match.group(2)]
        position = match.end()
    if position != len(token) or position == 0:
        raise ValueError(f"unsupported DIM2 time length: {value!r}")
    return _fortran_nint(seconds / float(dt_force))


def resolve_dim2_time_control(
    *,
    dt_force: float,
    dt_sechiba: float,
    forcing_length: int,
    date0: float,
    restart: Dim2RestartTime = Dim2RestartTime(),
    time_skip: str | int = 0,
    time_length: str | int | None = None,
) -> Dim2TimeControl:
    """Resolve DIM2 chronology without performing IO.

    Provenance: ``dim2_driver.f90::driver`` lines 285-305, 361-401 and
    464-573. As in Fortran line 296, ``INT`` truncates a non-integral split.
    """

    dt_force = float(dt_force)
    dt = float(dt_sechiba)
    if dt_force <= 0.0 or dt <= 0.0:
        raise ValueError("dt_force and dt_sechiba must be positive")
    ratio = dt_force / dt
    split = int(ratio)
    if split < 1:
        raise ValueError("DT_SECHIBA cannot exceed the forcing timestep")
    if forcing_length < 0:
        raise ValueError("forcing_length must be non-negative")

    date0_rest = float(date0 if restart.date0_rest is None else restart.date0_rest)
    dt_rest = float(dt if restart.dt_rest is None else restart.dt_rest)
    itau_dep_rest = int(restart.itau_dep_rest)
    if itau_dep_rest == 0 or restart.driver_reset_time:
        itau_dep = 0
        date0_rest = float(date0)
    else:
        itau_dep = itau_dep_rest
    if dt_rest != dt_force and itau_dep > 1:
        itau_dep = _fortran_nint(itau_dep * dt_rest / dt_force)

    for_offset = 0 if float(date0) == date0_rest else _fortran_nint((date0_rest - float(date0)) * ONE_DAY / dt_force)
    skip_steps = time_length_to_forcing_steps(time_skip, dt_force=dt_force)
    itau_dep += skip_steps
    # Lines 519 and 533: TIME_SKIP advances itau_dep while the pre-existing
    # itau_fin remains fixed, so the default integration length shrinks.
    default_length = int(forcing_length) - skip_steps
    itau_len = default_length if time_length is None else time_length_to_forcing_steps(time_length, dt_force=dt_force)
    if itau_len < 0:
        raise ValueError("TIME_LENGTH must be non-negative")
    itau_fin = itau_dep + itau_len
    istp = itau_dep * split + 1
    split_start = (istp - 1) % split + 1
    return Dim2TimeControl(
        dt=dt,
        dt_force=dt_force,
        split=split,
        itau_dep_rest=itau_dep_rest,
        itau_dep=itau_dep,
        itau_fin=itau_fin,
        itau_len=itau_len,
        date0=float(date0),
        date0_rest=date0_rest,
        dt_rest=dt_rest,
        for_offset=for_offset,
        istp=istp,
        istp_old=itau_dep_rest,
        split_start=split_start,
    )


def iter_dim2_time_indices(time: Dim2TimeControl) -> Iterator[Dim2TimeIndex]:
    """Yield the exact nested-loop order from lines 802-837 and 1392-1408."""

    istp = time.istp
    first = True
    split_start = time.split_start
    for it in range(time.itau_dep + 1, time.itau_fin + 1):
        it_force = it + time.for_offset
        if it_force < 0:
            raise IndexError(f"forcing chronology index is negative: {it_force}")
        for isplit in range(split_start, time.split + 1):
            yield Dim2TimeIndex(
                it=it,
                it_force=it_force,
                isplit=isplit,
                istp=istp,
                forcing_model_tstep=(it_force - 1) * time.split + isplit - 1,
                julian=time.date0_rest + istp * time.dt / ONE_DAY,
                lstep_init=first,
                lstep_last=it == time.itau_fin and isplit == time.split,
            )
            first = False
            istp += 1
        split_start = 1


def _vector(value: Any, npts: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1 and npts != 1:
        arr = np.full(npts, float(arr[0]), dtype=np.float64)
    if arr.shape != (npts,):
        raise ValueError(f"{name} must contain {npts} landpoint values, got {arr.shape}")
    return arr.copy()


def _albedo(value: Any, npts: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (npts, 2):
        return arr.copy()
    if arr.shape == (2, npts):
        return arr.T.copy()
    raise ValueError(f"albedo must have shape ({npts}, 2), got {arr.shape}")


def _restart_value(value: Any | None, fallback: np.ndarray, npts: int, name: str) -> np.ndarray:
    return fallback.copy() if value is None else _vector(value, npts, name)


def resolve_dim2_restart_fields(
    restart: Dim2DriverRestart,
    bundle: DriverStepBundle,
) -> Dim2DriverRestart:
    """Resolve driver restart fallbacks from lines 722-787.

    Albedo and roughness remain optional here because Fortran reads them only
    after ``intersurf_initialize_2d`` (lines 1135-1157).
    """

    npts = int(bundle.domain.nbindex)
    if npts < 1:
        raise ValueError("DIM2 requires at least one landpoint")
    zlev = _vector(bundle.forcing.zlev, npts, "zlev")
    qair = _vector(bundle.forcing.qair, npts, "qair")
    temp_air = _vector(bundle.forcing.temp_air, npts, "temp_air")
    pb_pa = _vector(bundle.forcing.pb, npts, "pb")
    eair = CP_AIR * temp_air + GRAVITY * zlev
    rau = pb_pa / (MOLAR_GAS_CONSTANT_AIR * temp_air)
    zero = np.zeros(npts, dtype=np.float64)
    return replace(
        restart,
        fluxsens=_restart_value(restart.fluxsens, zero, npts, "fluxsens"),
        vevapp=_restart_value(restart.vevapp, zero, npts, "vevapp"),
        old_zlev=_restart_value(restart.old_zlev, zlev, npts, "old_zlev"),
        old_qair=_restart_value(restart.old_qair, qair, npts, "old_qair"),
        old_eair=_restart_value(restart.old_eair, eair, npts, "old_eair"),
        rau_old=_restart_value(restart.rau_old, rau, npts, "rau_old"),
        petAcoef=_restart_value(restart.petAcoef, zero, npts, "petAcoef"),
        petBcoef=_restart_value(restart.petBcoef, eair, npts, "petBcoef"),
        peqAcoef=_restart_value(restart.peqAcoef, zero, npts, "peqAcoef"),
        peqBcoef=_restart_value(restart.peqBcoef, qair, npts, "peqBcoef"),
    )


def _boundary(
    bundle: DriverStepBundle,
    index: Dim2TimeIndex,
    *,
    z0: np.ndarray,
    albedo: np.ndarray,
    restart: Dim2DriverRestart,
    dt: float,
    coszang: np.ndarray | None,
) -> Dim2IntersurfBoundary:
    npts = int(bundle.domain.nbindex)
    payload = build_intersurf_first_step_payload(bundle, driver_z0_for_wind=z0, dt_sechiba=dt)
    temp_air = _vector(bundle.forcing.temp_air, npts, "temp_air")
    zlev = _vector(bundle.forcing.zlev, npts, "zlev")
    pb_pa = _vector(bundle.forcing.pb, npts, "pb")
    eair = CP_AIR * temp_air + GRAVITY * zlev
    rau = pb_pa / (MOLAR_GAS_CONSTANT_AIR * temp_air)
    swdown = _vector(bundle.forcing.swdown, npts, "swdown")
    swnet = (1.0 - np.mean(albedo, axis=1)) * swdown
    zero = np.zeros(npts, dtype=np.float64)
    return Dim2IntersurfBoundary(
        index=index,
        payload=payload,
        eair=eair,
        swnet=swnet,
        coszang=None if coszang is None else _vector(coszang, npts, "coszang"),
        rau=rau,
        cdrag=zero,
        petAcoef=zero,
        # Non-relaxation paper path, lines 974-1003: forcing observations
        # overwrite all four coefficients on every substep.
        petBcoef=eair.copy(),
        peqAcoef=zero,
        peqBcoef=payload.qair.copy(),
    )


def _validate_output(output: Dim2IntersurfOutput, npts: int) -> Dim2IntersurfOutput:
    return replace(
        output,
        vevapp=_vector(output.vevapp, npts, "vevapp"),
        fluxsens=_vector(output.fluxsens, npts, "fluxsens"),
        fluxlat=_vector(output.fluxlat, npts, "fluxlat"),
        coastalflow=_vector(output.coastalflow, npts, "coastalflow"),
        riverflow=_vector(output.riverflow, npts, "riverflow"),
        tsol_rad=_vector(output.tsol_rad, npts, "tsol_rad"),
        temp_sol_new=_vector(output.temp_sol_new, npts, "temp_sol_new"),
        qsurf=_vector(output.qsurf, npts, "qsurf"),
        albedo=_albedo(output.albedo, npts),
        emis=_vector(output.emis, npts, "emis"),
        z0=_vector(output.z0, npts, "z0"),
    )


def run_dim2_driver(
    *,
    time: Dim2TimeControl,
    forcing_reader: Callable[[int], DriverStepBundle],
    intersurf_initialize: Callable[[Dim2IntersurfBoundary, Any], Dim2IntersurfOutput],
    intersurf_main: Callable[[Dim2IntersurfBoundary, Any], Dim2IntersurfOutput],
    restart: Dim2DriverRestart = Dim2DriverRestart(),
    switches: Dim2Switches = Dim2Switches(),
    component_state: Any = None,
    coszang_reader: Callable[[int], np.ndarray | None] | None = None,
) -> Dim2DriverResult:
    """Run the audited source-order transition over independent landpoints.

    ``forcing_reader`` receives the zero-based expanded model-step index used
    by ``read_forcing_model_step_cached``. The callbacks are the explicit
    intersurf ownership boundary; this module never substitutes SECHIBA or
    STOMATE process logic.
    """

    validate_dim2_switches(switches)
    indices = tuple(iter_dim2_time_indices(time))
    if not indices:
        raise ValueError("DIM2 transition requires at least one model timestep")

    steps: list[Dim2StepOutput] = []
    output: Dim2IntersurfOutput | None = None
    old_zlev: np.ndarray | None = None
    old_qair: np.ndarray | None = None
    old_eair: np.ndarray | None = None
    rau_old: np.ndarray | None = None
    temp_sol_old: np.ndarray | None = None
    final_boundary: Dim2IntersurfBoundary | None = None
    initial_restart: Dim2DriverRestart | None = None

    for index in indices:
        bundle = forcing_reader(index.forcing_model_tstep)
        npts = int(bundle.domain.nbindex)
        if npts < 1:
            raise ValueError("DIM2 requires at least one landpoint")
        if output is None:
            initial_restart = resolve_dim2_restart_fields(restart, bundle)
            # Lines 794-796 initialize these before the first intersurf call;
            # restart albedo/z0 is not read until lines 1135-1157.
            albedo = np.full((npts, 2), 0.13, dtype=np.float64)
            z0 = np.full(npts, 0.1, dtype=np.float64)
            temp_sol_old = _vector(bundle.forcing.temp_air, npts, "temp_air")
        else:
            if output.z0.shape != (npts,):
                raise ValueError("landpoint count changed during a DIM2 transition")
            albedo = output.albedo
            z0 = output.z0

        coszang = None if coszang_reader is None else coszang_reader(index.forcing_model_tstep)
        initialize_boundary = None
        boundary = _boundary(bundle, index, z0=z0, albedo=albedo, restart=restart, dt=time.dt, coszang=coszang)
        if index.lstep_init:
            initialize_boundary = boundary
            initialized = _validate_output(intersurf_initialize(boundary, component_state), npts)
            component_state = initialized.component_state
            # Lines 1133-1180: initialize output/restart albedo and z0 alter
            # swnet and wind before main is called on this same timestep.
            albedo = initialized.albedo
            if restart.albedo_vis is not None:
                albedo[:, 0] = _vector(restart.albedo_vis, npts, "albedo_vis")
            if restart.albedo_nir is not None:
                albedo[:, 1] = _vector(restart.albedo_nir, npts, "albedo_nir")
            # Preserve the source exactly, including its asymmetric line 1157:
            # tmp_z0 starts from initialized z0, and z0 is assigned only when
            # tmp_z0 is entirely val_exp. A normal present restart z0 therefore
            # does not overwrite the initialized value.
            tmp_z0 = initialized.z0 if restart.z0 is None else _vector(restart.z0, npts, "z0")
            z0 = tmp_z0 if np.all(tmp_z0 == VAL_EXP) else initialized.z0
            temp_sol_old = initialized.temp_sol_new
            boundary = _boundary(bundle, index, z0=z0, albedo=albedo, restart=restart, dt=time.dt, coszang=coszang)

        output = _validate_output(intersurf_main(boundary, component_state), npts)
        component_state = output.component_state
        assert temp_sol_old is not None
        dtdt = np.abs(output.temp_sol_new - temp_sol_old) / time.dt
        temp_sol_old = output.temp_sol_new.copy()
        old_zlev = _vector(bundle.forcing.zlev, npts, "zlev")
        old_qair = boundary.payload.qair.copy()
        old_eair = boundary.eair.copy()
        rau_old = boundary.rau.copy()
        final_boundary = boundary
        steps.append(Dim2StepOutput(index, initialize_boundary, boundary, output, dtdt))

    assert output is not None and final_boundary is not None and initial_restart is not None
    assert old_zlev is not None and old_qair is not None and old_eair is not None and rau_old is not None
    final_restart = Dim2DriverRestart(
        fluxsens=output.fluxsens,
        vevapp=output.vevapp,
        old_zlev=old_zlev,
        old_qair=old_qair,
        old_eair=old_eair,
        rau_old=rau_old,
        albedo_vis=output.albedo[:, 0],
        albedo_nir=output.albedo[:, 1],
        z0=output.z0,
        petAcoef=final_boundary.petAcoef,
        petBcoef=final_boundary.petBcoef,
        peqAcoef=final_boundary.peqAcoef,
        peqBcoef=final_boundary.peqBcoef,
    )
    return Dim2DriverResult(
        time=time,
        steps=tuple(steps),
        final_output=output,
        initial_restart=initial_restart,
        restart=final_restart,
        istp_old=indices[-1].istp,
        component_state=component_state,
    )


def dim2_switches_from_run_def(values: Mapping[str, str]) -> Dim2Switches:
    """Read only switches owned or constrained by this DIM2 transition."""

    def flag(name: str, default: bool) -> bool:
        token = values.get(name)
        if token is None:
            return default
        normalized = token.strip().strip(".").upper()
        if normalized in {"Y", "YES", "TRUE", "T", "1"}:
            return True
        if normalized in {"N", "NO", "FALSE", "F", "0"}:
            return False
        raise ValueError(f"invalid run.def boolean {name}={token!r}")

    return Dim2Switches(
        weathergen=flag("WEATHERGEN", False),
        relaxation=flag("RELAXATION", False),
        watchout=flag("WATCHOUT", False),
        river_routing=flag("RIVER_ROUTING", True),
        driver_routing=flag("DRIVER_ROUTING", False),
    )
