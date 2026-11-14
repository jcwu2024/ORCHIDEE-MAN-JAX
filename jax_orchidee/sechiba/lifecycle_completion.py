"""Source-routed offline SECHIBA lifecycle, time, history, and finalization.

Only transport is delegated to callbacks.  Grid compression, unit conversion,
calendar-derived state, output activation, finalizer dispatch, routing
writeback, and CO2 aggregation follow the cited Fortran routines here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, NamedTuple

import numpy as np

from jax_orchidee.sechiba.ioipsl_calendar import IOIPSLCalendar, fortran_nint


INTERSURF_SOURCE = "fortran_source/ORCHIDEE/src_sechiba/intersurf.f90"
HISTORY_SOURCE = "fortran_source/ORCHIDEE/src_sechiba/ioipslctrl.f90"
SECHIBA_SOURCE = "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90"


@dataclass(frozen=True)
class IntersurfTimeState:
    """Time state assigned by ``intsurf_time``1884-1982 plus adapter binding.

    ``clock_binding`` is an immutable-run domain guard for this pure adapter,
    not a field or a parameter-validation branch in the original Fortran.
    Previous unbound/astronomical/seconds-valued payloads require migration.
    """

    calendar: str
    one_year: float
    one_day: float
    year_length: int
    year_spread: float
    in_julian: float
    julian0: float
    julian_diff: float
    year: int
    month: int
    day: int
    second: float
    month_length: int
    initialized: bool = True
    clock_binding: tuple[object, ...] | None = None


def _julian_to_gregorian(julian: float) -> tuple[int, int, int, float]:
    """IOIPSL Gregorian Lilian decode, calendar.f90::ju2ymds201-309."""
    return IOIPSLCalendar("gregorian").ju2ymds(julian)


def _gregorian_to_julian(year: int, month: int, day: int) -> float:
    """IOIPSL Gregorian Lilian encode, calendar.f90::ymds2ju112-197."""
    return IOIPSLCalendar("gregorian").ymds2ju(year, month, day, 0.0)


def _gregorian_month_length(year: int, month: int) -> int:
    """IOIPSL Gregorian ioget_mon_len calendar.f90:858-909."""
    return IOIPSLCalendar("gregorian").month_length(year, month)


NonGregorianYmds = Callable[[int, float], tuple[int, int, int, float]]


def intersurf_time(
    *,
    istp: int,
    date0: float,
    dt_sechiba: float,
    calendar: str,
    one_year: float,
    one_day: float,
    previous: IntersurfTimeState | None = None,
    non_gregorian_ymds: NonGregorianYmds | None = None,
    clock: IOIPSLCalendar | None = None,
    date0_convention: str | None = None,
) -> IntersurfTimeState:
    """Apply ``intersurf.f90::intsurf_time``1884-1982 using native clock units.

    Explicitly declare ``date0_convention='ioipsl_calendar'``: Gregorian date0
    is Lilian (2000-01-01=152385), noleap date0 is year*365 at January 1.
    No astronomical-JD conversion is inferred. ``one_year`` is days and
    ``one_day`` is seconds. Only measured Gregorian/noleap providers are
    admitted. Noleap needs an explicit locked ``clock``; that startdate is
    independently owned by IOIPSL, not reconstructed from owner ``date0``.

    Source first-call year_length/year_spread persist on later calls. The
    additional previous-state binding checks restrict this adapter to one
    immutable run; they do not claim that Fortran checks arbitrary later
    argument changes. They prevent silent reinitialization or reinterpretation
    of legacy payloads. The old unbound callback interface is rejected.
    """
    if date0_convention != "ioipsl_calendar":
        raise ValueError("declare date0_convention='ioipsl_calendar'; astronomical-JD/legacy origins require explicit migration")
    if non_gregorian_ymds is not None:
        raise ValueError("legacy non_gregorian_ymds callback is unbound; provide a locked IOIPSLCalendar")
    if not all(math.isfinite(float(v)) for v in (date0, dt_sechiba, one_year, one_day)):
        raise ValueError("date0, dt_sechiba, one_year, and one_day must be finite")
    if dt_sechiba <= 0 or one_day <= 0 or one_year <= 0:
        raise ValueError("dt_sechiba, one_day, and one_year must be positive")
    if int(istp) != istp:
        raise ValueError("istp must be an integer")
    name = calendar.strip()
    if clock is None:
        clock = IOIPSLCalendar(name)
    if not isinstance(clock, IOIPSLCalendar) or clock.calendar != name:
        raise ValueError("clock calendar must match the owner calendar")
    if one_year != clock.one_year_days:
        raise ValueError("one_year must use the IOIPSL calendar constant in days, not seconds")
    if one_day != clock.one_day_seconds:
        raise ValueError("one_day must use the IOIPSL day constant in seconds")
    if name != "gregorian" and clock.startdate_day is None:
        raise ValueError("noleap requires an explicitly locked IOIPSLCalendar startdate")
    binding = (date0_convention, name, clock.startdate_day, clock.startdate_second,
               float(date0), float(dt_sechiba), float(one_year), float(one_day))
    if previous is not None:
        if not previous.initialized or previous.clock_binding is None:
            raise ValueError("previous must be initialized and carry an explicit IOIPSL clock binding")
        if previous.clock_binding != binding:
            raise ValueError("previous clock binding changed; this adapter requires an immutable run")
        if (previous.calendar, previous.one_year, previous.one_day) != (name, one_year, one_day):
            raise ValueError("previous calendar constants disagree with its clock binding")
        year_length, year_spread = previous.year_length, previous.year_spread
    else:
        year_length = clock.year_steps(float(dt_sechiba), float(date0))
        year_spread = 1.0 if name == "gregorian" else float(one_year) / 365.2425
    in_julian = clock.itau2date(int(istp), float(date0), float(dt_sechiba))
    if name == "gregorian":
        year, month, day, _ = clock.ju2ymds(in_julian)
    else:
        year, month, day, second = clock.itau2ymds(int(istp), float(dt_sechiba))
    julian0 = clock.ymds2ju(year, 1, 1, 0.0)
    julian_diff = in_julian - julian0
    if name == "gregorian":
        second = float(fortran_nint((julian_diff - int(julian_diff)) * one_day))
    return IntersurfTimeState(
        calendar=name, one_year=float(one_year), one_day=float(one_day),
        year_length=year_length, year_spread=float(year_spread),
        in_julian=in_julian, julian0=julian0, julian_diff=julian_diff,
        year=int(year), month=int(month), day=int(day), second=float(second),
        month_length=clock.month_length(year, month), clock_binding=binding,
    )


def _non_gregorian_month_length(calendar: str, year: int, month: int) -> int:
    """Only the native-measured noleap domain is supported by this adapter."""
    if calendar != "noleap":
        raise ValueError(f"unsupported non-Gregorian calendar {calendar!r}")
    return IOIPSLCalendar(calendar).month_length(year, month)


def non_gregorian_ymds_to_year_offset(
    *, month: int, day: int, second: float, one_day: float, calendar: str, year: int
) -> float:
    """Pure negative offset only; this cannot establish an IOIPSL origin lock.

    Month lengths follow calendar.f90::ioget_mon_len858-909. The source owner
    computes julian0 by ymds2ju; it does not reconstruct it from this offset.
    """
    days = sum(_non_gregorian_month_length(calendar, year, m) for m in range(1, month))
    days += day - 1 + second / one_day
    return -float(days)


@dataclass(frozen=True)
class HistorySwitches:
    """Configuration values read by ``ioipslctrl_history`` lines 167-2468."""

    xios_orchidee_ok: bool
    almaoutput: bool = False
    write_step: float = 86400.0
    sechiba_histfile2: bool = False
    write_step2: float = 1800.0
    sechiba_histlevel: int = 5
    sechiba_histlevel2: int = 5
    ok_stomate: bool = True
    stomate_hist_days: float = 10.0
    stomate_ipcc_hist_days: float = 0.0
    dt_stomate: float = 86400.0
    is_omp_root: bool = True
    grid_type: str = "RegLonLat"
    hydrol_cwrr: bool = True
    river_routing: bool = False
    do_irrigation: bool = False
    do_floodplains: bool = False
    ok_co2: bool = True
    ok_bvoc: bool = False
    ok_radcanopy: bool = False
    ok_multilayer: bool = False


@dataclass(frozen=True)
class HistoryStreamRequest:
    name: str
    active: bool
    filename: str
    frequency: float
    history_level: int | None
    grid_kind: str
    longitude: np.ndarray
    latitude: np.ndarray
    operations: Mapping[str, tuple[str, ...]]
    selections: Mapping[str, bool]
    source_lines: tuple[int, int]


class HistoryRoutingResult(NamedTuple):
    streams: tuple[HistoryStreamRequest, ...]
    handles: Mapping[str, object]
    provenance: tuple[str, ...] = (
        f"{HISTORY_SOURCE}::ioipslctrl_history lines 167-1419",
        f"{HISTORY_SOURCE}::ioipslctrl_history lines 1422-2252",
        f"{HISTORY_SOURCE}::ioipslctrl_history lines 2258-2509",
    )


HistoryTransport = Callable[[HistoryStreamRequest], object]


def _history_operations(
    level: int, *, dt: float, one_day: float
) -> dict[str, tuple[str, ...]]:
    if not 0 <= level <= 11:
        raise ValueError("history level must be between 0 and 11")

    def active(operation: str) -> tuple[str, ...]:
        return tuple(operation if index <= level else "never" for index in range(1, 12))

    return {
        "ave": active("ave(scatter(X))"),
        "avecels": active("ave(cels(scatter(X)))"),
        "tmincels": active("t_min(cels(scatter(X)))"),
        "tmaxcels": active("t_max(cels(scatter(X)))"),
        "once": active("once(scatter(X))"),
        "sumscatter": active("t_sum(scatter(X))"),
        "fluxop": active(f"ave(scatter(X*{one_day / dt:.10g}))"),
        "fluxop_scinsec": active(f"ave(scatter(X*{1.0 / dt:.10g}))"),
    }


def ioipslctrl_history(
    *,
    lon: object,
    lat: object,
    dt: float,
    one_day: float,
    switches: HistorySwitches,
    transport: HistoryTransport | None = None,
    filenames: Mapping[str, str] | None = None,
) -> HistoryRoutingResult:
    """Resolve every history stream before crossing the IO transport boundary.

    Fortran provenance: ``ioipslctrl.f90::ioipslctrl_history`` lines 65-2514.
    """

    longitude, latitude = np.asarray(lon), np.asarray(lat)
    if longitude.shape != latitude.shape or longitude.ndim != 2:
        raise ValueError("lon and lat must share a rank-2 grid shape")
    if dt <= 0 or one_day <= 0:
        raise ValueError("dt and one_day must be positive")
    names = {
        "sechiba": "sechiba_history.nc",
        "sechiba2": "sechiba_out_2.nc",
        "stomate": "stomate_history.nc",
        "stomate_ipcc": "stomate_ipcc_history.nc",
    }
    names.update(dict(filenames or {}))
    dw = 0.0 if switches.xios_orchidee_ok else float(switches.write_step)
    sechiba_active = dw != 0.0
    second_active = (
        switches.sechiba_histfile2 and switches.write_step2 != 0 and sechiba_active
    )

    if not switches.ok_stomate or not sechiba_active:
        stomate_dt = 0.0
    elif switches.stomate_hist_days == -1:
        stomate_dt = -1.0
    elif switches.stomate_hist_days == 0:
        stomate_dt = 0.0
    else:
        stomate_dt = float(np.rint(switches.stomate_hist_days)) * one_day
    if stomate_dt > 0 and switches.dt_stomate > stomate_dt:
        raise ValueError("DT_STOMATE must be less than or equal to STOMATE_HIST_DT")

    ipcc_dt = (
        -1.0
        if switches.stomate_ipcc_hist_days == -1
        else (float(np.rint(switches.stomate_ipcc_hist_days)) * one_day)
    )
    if not switches.ok_stomate or (ipcc_dt != 0 and not sechiba_active):
        ipcc_dt = 0.0
    if ipcc_dt > 0 and switches.dt_stomate > ipcc_dt:
        raise ValueError(
            "DT_STOMATE must be less than or equal to STOMATE_IPCC_HIST_DT"
        )

    if switches.grid_type == "RegLonLat":
        out_lon, out_lat, grid_kind = longitude[:, 0], latitude[0, :], "rectilinear"
    else:
        out_lon, out_lat, grid_kind = longitude, latitude, "curvilinear"
    selections = {
        name: bool(getattr(switches, name))
        for name in (
            "almaoutput",
            "hydrol_cwrr",
            "river_routing",
            "do_irrigation",
            "do_floodplains",
            "ok_co2",
            "ok_stomate",
            "ok_bvoc",
            "ok_radcanopy",
            "ok_multilayer",
        )
    }
    empty_ops: Mapping[str, tuple[str, ...]] = {}
    streams = (
        HistoryStreamRequest(
            "sechiba",
            sechiba_active,
            names["sechiba"],
            dw,
            switches.sechiba_histlevel,
            grid_kind,
            out_lon,
            out_lat,
            _history_operations(switches.sechiba_histlevel, dt=dt, one_day=one_day),
            selections,
            (167, 1419),
        ),
        HistoryStreamRequest(
            "sechiba2",
            second_active,
            names["sechiba2"],
            float(switches.write_step2),
            switches.sechiba_histlevel2,
            grid_kind,
            out_lon,
            out_lat,
            _history_operations(switches.sechiba_histlevel2, dt=dt, one_day=one_day),
            selections,
            (1422, 2252),
        ),
        HistoryStreamRequest(
            "stomate",
            switches.ok_stomate and stomate_dt != 0,
            names["stomate"],
            stomate_dt,
            None,
            grid_kind,
            out_lon,
            out_lat,
            empty_ops,
            selections,
            (2258, 2397),
        ),
        HistoryStreamRequest(
            "stomate_ipcc",
            switches.ok_stomate and ipcc_dt != 0,
            names["stomate_ipcc"],
            ipcc_dt,
            None,
            grid_kind,
            out_lon,
            out_lat,
            empty_ops,
            selections,
            (2402, 2509),
        ),
    )
    handles: dict[str, object] = {}
    for stream in streams:
        handles[stream.name] = (
            transport(stream)
            if stream.active and switches.is_omp_root and transport
            else -1
        )
    return HistoryRoutingResult(streams, handles)


_FORCING_FIELDS = (
    "u",
    "v",
    "zlev",
    "qair",
    "precip_rain",
    "precip_snow",
    "lwdown",
    "swnet",
    "swdown",
    "temp_air",
    "epot_air",
    "ccanopy",
    "petAcoef",
    "peqAcoef",
    "petBcoef",
    "peqBcoef",
    "cdrag",
    "pb",
)
_OUTPUT_FIELDS = (
    "z0m",
    "coastalflow",
    "riverflow",
    "tsol_rad",
    "vevapp",
    "temp_sol_new",
    "qsurf",
    "albedo",
    "fluxsens",
    "fluxlat",
    "emis",
    "cdrag",
)


@dataclass(frozen=True)
class Intersurf2DRequest:
    itau_sechiba: int
    date0_shifted: float
    forcing: Mapping[str, np.ndarray]
    latitude_longitude: np.ndarray
    contfrac: np.ndarray
    provenance: str


class Intersurf2DResult(NamedTuple):
    outputs: Mapping[str, np.ndarray]
    request: Intersurf2DRequest
    owner_result: object
    calendar_step: int
    initialization: bool


ScientificOwner = Callable[[Intersurf2DRequest], object]


def _land_positions(
    kindex: object, iim: int, jjm: int
) -> tuple[np.ndarray, np.ndarray]:
    index = np.asarray(kindex, dtype=np.int64)
    if index.ndim != 1 or index.size == 0:
        raise ValueError("kindex must be a non-empty rank-1 array")
    if (
        np.any(index < 1)
        or np.any(index > iim * jjm)
        or np.unique(index).size != index.size
    ):
        raise ValueError(
            "kindex must contain unique one-based positions inside the 2-D grid"
        )
    zero = index - 1
    return zero % iim, zero // iim


def _owner_outputs(result: object) -> Mapping[str, object]:
    if isinstance(result, Mapping):
        return result
    if hasattr(result, "outputs"):
        return result.outputs
    if hasattr(result, "state"):
        return result.state
    raise TypeError(
        "scientific owner must return a mapping or an object with outputs/state"
    )


def _intersurf_2d(
    *,
    initialization: bool,
    kjit: int,
    iim: int,
    jjm: int,
    kindex: object,
    xrdt: float,
    date0_shifted: float,
    itau_offset: int,
    lon: object,
    lat: object,
    zcontfrac: object,
    fields: Mapping[str, object],
    owner: ScientificOwner,
    undef_sechiba: float,
) -> Intersurf2DResult:
    if xrdt <= 0:
        raise ValueError("xrdt must be positive")
    ii, jj = _land_positions(kindex, iim, jjm)
    required = set(_FORCING_FIELDS) | ({"coszang"} if not initialization else set())
    missing = sorted(required - fields.keys())
    if missing:
        raise ValueError(f"missing 2-D forcing fields: {missing}")
    compressed: dict[str, np.ndarray] = {}
    for name in required:
        array = np.asarray(fields[name])
        if array.shape != (iim, jjm):
            raise ValueError(f"{name} must have shape ({iim}, {jjm})")
        compressed[name] = array[ii, jj].copy()
    compressed["precip_rain"] *= xrdt
    compressed["precip_snow"] *= xrdt
    longitude, latitude, contfrac = (
        np.asarray(lon),
        np.asarray(lat),
        np.asarray(zcontfrac),
    )
    for name, array in (("lon", longitude), ("lat", latitude), ("zcontfrac", contfrac)):
        if array.shape != (iim, jjm):
            raise ValueError(f"{name} must have shape ({iim}, {jjm})")
    itau = int(kjit) + int(itau_offset)
    request = Intersurf2DRequest(
        itau,
        float(date0_shifted),
        compressed,
        np.stack((latitude[ii, jj], longitude[ii, jj]), axis=1),
        contfrac[ii, jj].copy(),
        f"{INTERSURF_SOURCE}::{'intersurf_initialize_2d' if initialization else 'intersurf_main_2d'} "
        f"lines {113 if initialization else 458}-{443 if initialization else 761}",
    )
    owner_result = owner(request)
    land_outputs = _owner_outputs(owner_result)
    missing_outputs = sorted(set(_OUTPUT_FIELDS) - land_outputs.keys())
    if missing_outputs:
        raise ValueError(f"scientific owner omitted outputs: {missing_outputs}")
    outputs: dict[str, np.ndarray] = {}
    nland = ii.size
    for name in _OUTPUT_FIELDS:
        values = np.asarray(land_outputs[name])
        trailing = (2,) if name == "albedo" else values.shape[1:]
        expected = (nland,) + trailing
        if values.shape != expected:
            raise ValueError(
                f"owner output {name} must have leading land shape {nland}"
            )
        grid = np.full((iim, jjm) + trailing, undef_sechiba, dtype=values.dtype)
        grid[ii, jj] = values
        if name in {"vevapp", "coastalflow", "riverflow"}:
            grid[ii, jj] /= xrdt
        outputs[name] = grid
    return Intersurf2DResult(
        outputs,
        request,
        owner_result,
        itau + 1 if initialization else itau,
        initialization,
    )


def intersurf_initialize_2d(**kwargs: object) -> Intersurf2DResult:
    """Compress, initialize, scatter, and convert per lines 113-443."""

    return _intersurf_2d(initialization=True, **kwargs)


def intersurf_main_2d(**kwargs: object) -> Intersurf2DResult:
    """Compress, step, scatter, and convert per lines 458-761."""

    return _intersurf_2d(initialization=False, **kwargs)


@dataclass(frozen=True)
class SechibaFinalizeSwitches:
    hydrol_cwrr: bool = True
    river_routing: bool = True
    nbp_glo: int = 1
    do_fullirr: bool = False
    ih2o_fortran: int = 1


class SechibaFinalizeResult(NamedTuple):
    state: Mapping[str, object]
    netco2flux: np.ndarray
    process_order: tuple[str, ...]
    owner_results: Mapping[str, object]
    provenance: tuple[str, ...] = (
        f"{SECHIBA_SOURCE}::sechiba_finalize lines 1838-1880",
        f"{SECHIBA_SOURCE}::sechiba_finalize lines 1882-1904",
        f"{SECHIBA_SOURCE}::sechiba_finalize lines 1906-1919",
    )


FinalizeOwner = Callable[[Mapping[str, object]], object]


def sechiba_finalize(
    *,
    state: Mapping[str, object],
    finalizers: Mapping[str, FinalizeOwner],
    switches: SechibaFinalizeSwitches = SechibaFinalizeSwitches(),
    restart_transport: Callable[[Mapping[str, object]], object] | None = None,
) -> SechibaFinalizeResult:
    """Run source-ordered component finalizers and compute restart writeback.

    Component callbacks are process owners, not serializers: each receives the
    current routed state.  The optional restart callback is invoked only after
    all scientific choices and values have been resolved.

    Fortran provenance: ``sechiba.f90::sechiba_finalize`` lines 1800-1923.
    """

    required_owners = [
        "diffuco",
        "enerbil",
        "hydrol" if switches.hydrol_cwrr else "hydrolc",
        "condveg",
        "thermosoil" if switches.hydrol_cwrr else "thermosoilc",
        "slowproc",
    ]
    missing = [name for name in required_owners if name not in finalizers]
    if missing:
        raise ValueError(f"missing source finalizers: {missing}")
    updated = dict(state)
    results: dict[str, object] = {}
    order: list[str] = []
    for name in required_owners[:2]:
        results[name] = finalizers[name](updated)
        order.append(f"{name}_finalize")
    hydrology = required_owners[2]
    results[hydrology] = finalizers[hydrology](updated)
    order.append(f"{hydrology}_finalize")
    if switches.hydrol_cwrr:
        if "rsol" not in updated:
            raise ValueError("state must contain rsol for CWRR finalize writeback")
        updated["rsol"] = np.full_like(np.asarray(updated["rsol"]), -1.0)
    else:
        for name in ("evap_bare_lim", "k_litt", "shumdiag", "shumdiag_perma"):
            if name not in updated:
                raise ValueError(
                    f"state must contain {name} for Choisnel finalize writeback"
                )
        updated["evap_bare_lim"] = np.full_like(
            np.asarray(updated["evap_bare_lim"]), -1.0
        )
        updated["k_litt"] = np.full_like(np.asarray(updated["k_litt"]), 8.0)
        updated["shumdiag_perma"] = np.asarray(updated["shumdiag"]).copy()
    for name in required_owners[3:5]:
        results[name] = finalizers[name](updated)
        order.append(f"{name}_finalize")

    if switches.river_routing and switches.nbp_glo > 1:
        if "routing" not in finalizers:
            raise ValueError("missing source finalizer: routing")
        results["routing"] = finalizers["routing"](updated)
        order.append("routing_finalize")
    else:
        zero_fields = (
            "reinfiltration",
            "returnflow",
            "sed_deposition",
            "poc_deposition",
            "sed_deposition_d",
            "poc_deposition_d",
            "flood_res",
            "flood_frac",
            "streamfl_frac",
            "fastr",
        )
        for name in zero_fields:
            if name not in updated:
                raise ValueError(f"state must contain routing writeback field {name}")
            updated[name] = np.zeros_like(np.asarray(updated[name]))
        if "irrigation" not in updated:
            raise ValueError("state must contain irrigation")
        if switches.do_fullirr:
            water = switches.ih2o_fortran - 1
            irrigation = np.asarray(updated["irrigation"])
            if irrigation.ndim != 2 or not 0 <= water < irrigation.shape[1]:
                raise ValueError("ih2o_fortran is outside irrigation's flow dimension")
            updated["irrigation_restart"] = irrigation[:, water].copy()
        else:
            updated["irrigation"] = np.zeros_like(np.asarray(updated["irrigation"]))
        order.append("routing_finalize[no-call-writeback]")

    results["slowproc"] = finalizers["slowproc"](updated)
    order.append("slowproc_finalize")
    if "co2_flux" not in updated or "veget_max" not in updated:
        raise ValueError("state must contain co2_flux and veget_max")
    co2_flux, vegetation = (
        np.asarray(updated["co2_flux"]),
        np.asarray(updated["veget_max"]),
    )
    if (
        co2_flux.shape != vegetation.shape
        or co2_flux.ndim != 2
        or co2_flux.shape[1] < 2
    ):
        raise ValueError(
            "co2_flux and veget_max must share shape (land, at least 2 PFTs)"
        )
    netco2flux = np.sum(co2_flux[:, 1:] * vegetation[:, 1:], axis=1)
    updated["netco2flux"] = netco2flux
    order.append("netco2flux")
    if restart_transport is not None:
        results["restart_transport"] = restart_transport(updated)
        order.append("restart_transport")
    return SechibaFinalizeResult(updated, netco2flux, tuple(order), results)
