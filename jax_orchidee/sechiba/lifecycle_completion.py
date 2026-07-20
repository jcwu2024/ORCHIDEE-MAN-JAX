"""Source-routed offline SECHIBA lifecycle, time, history, and finalization.

Only transport is delegated to callbacks.  Grid compression, unit conversion,
calendar-derived state, output activation, finalizer dispatch, routing
writeback, and CO2 aggregation follow the cited Fortran routines here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, NamedTuple

import numpy as np


INTERSURF_SOURCE = "fortran_source/ORCHIDEE/src_sechiba/intersurf.f90"
HISTORY_SOURCE = "fortran_source/ORCHIDEE/src_sechiba/ioipslctrl.f90"
SECHIBA_SOURCE = "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90"


@dataclass(frozen=True)
class IntersurfTimeState:
    """Module time state assigned by ``intsurf_time`` lines 1884-1982."""

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
    second: int
    month_length: int
    initialized: bool = True


def _julian_to_gregorian(julian: float) -> tuple[int, int, int, int]:
    """Convert an astronomical Julian day using the standard Gregorian split."""

    shifted = float(julian) + 0.5
    whole = int(np.floor(shifted))
    frac = shifted - whole
    alpha = int((whole - 1867216.25) / 36524.25)
    a = whole + 1 + alpha - alpha // 4
    b = a + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)
    day = b - d - int(30.6001 * e)
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    second = int(np.rint(frac * 86400.0))
    if second == 86400:
        return _julian_to_gregorian(float(whole) + 0.5)
    return year, month, day, second


def _gregorian_to_julian(year: int, month: int, day: int) -> float:
    y, m = (year - 1, month + 12) if month <= 2 else (year, month)
    a = y // 100
    b = 2 - a + a // 4
    return (
        np.floor(365.25 * (y + 4716)) + np.floor(30.6001 * (m + 1)) + day + b - 1524.5
    )


def _gregorian_month_length(year: int, month: int) -> int:
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    return (31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[month - 1]


NonGregorianYmds = Callable[[int, float], tuple[int, int, int, int]]


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
) -> IntersurfTimeState:
    """Apply the exact first-call and calendar branches of ``intsurf_time``.

    ``date0`` and ``in_julian`` use IOIPSL Julian-day units.  IOIPSL's
    calendar-specific ``itau2ymds`` is an explicit primitive for non-Gregorian
    calendars; all subsequent scientific time values are calculated here.

    Fortran provenance: ``intersurf.f90::intsurf_time`` lines 1884-1982.
    """

    if dt_sechiba <= 0 or one_day <= 0 or one_year <= 0:
        raise ValueError("dt_sechiba, one_day, and one_year must be positive")
    name = calendar.strip()
    initialized = previous is not None
    year_length = (
        previous.year_length if initialized else int(np.rint(one_year / dt_sechiba))
    )
    year_spread = (
        previous.year_spread
        if initialized
        else (1.0 if name == "gregorian" else one_year / 365.2425)
    )
    in_julian = float(date0) + int(istp) * float(dt_sechiba) / float(one_day)

    if name == "gregorian":
        year, month, day, _ = _julian_to_gregorian(in_julian)
        julian0 = _gregorian_to_julian(year, 1, 1)
        julian_diff = in_julian - julian0
        second = int(np.rint((julian_diff - int(julian_diff)) * one_day))
        month_length = _gregorian_month_length(year, month)
    else:
        if non_gregorian_ymds is None:
            raise ValueError(
                "non_gregorian_ymds is required for a non-Gregorian IOIPSL calendar"
            )
        year, month, day, second = non_gregorian_ymds(int(istp), float(dt_sechiba))
        julian0 = in_julian + non_gregorian_ymds_to_year_offset(
            month=month,
            day=day,
            second=second,
            one_day=one_day,
            calendar=name,
            year=year,
        )
        julian_diff = in_julian - julian0
        month_length = _non_gregorian_month_length(name, year, month)

    return IntersurfTimeState(
        calendar=name,
        one_year=float(one_year),
        one_day=float(one_day),
        year_length=year_length,
        year_spread=float(year_spread),
        in_julian=in_julian,
        julian0=float(julian0),
        julian_diff=float(julian_diff),
        year=int(year),
        month=int(month),
        day=int(day),
        second=int(second),
        month_length=int(month_length),
    )


def _non_gregorian_month_length(calendar: str, year: int, month: int) -> int:
    if calendar in {"360d", "360_day"}:
        return 30
    if calendar in {"noleap", "365d", "365_day"}:
        return (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[month - 1]
    if calendar in {"all_leap", "366d", "366_day"}:
        return (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[month - 1]
    raise ValueError(f"unsupported non-Gregorian calendar {calendar!r}")


def non_gregorian_ymds_to_year_offset(
    *, month: int, day: int, second: int, one_day: float, calendar: str, year: int
) -> float:
    """Return the negative Julian offset from a date to January 1."""

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
