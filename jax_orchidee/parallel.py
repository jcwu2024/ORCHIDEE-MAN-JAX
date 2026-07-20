"""Source-backed MPI and XIOS completion boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class MpiInitializationState:
    communicator: object | None
    mpi_size: int
    mpi_rank: int
    mpi_rank_root: int
    is_mpi_root: bool
    is_ok_mpi: bool
    xios_orchidee_ok: bool


def init_orchidee_mpi(
    *,
    xios_orchidee_ok: bool,
    cpp_para: bool,
    communicator: object | None = None,
    mpi_init: Callable[[], object] | None = None,
    xios_comm_init: Callable[[], object] | None = None,
    communicator_info: Callable[[object], tuple[int, int]] | None = None,
) -> MpiInitializationState:
    """Resolve ``Init_orchidee_mpi`` without inventing MPI handles or ranks."""

    if not cpp_para:
        return MpiInitializationState(None, 1, 0, 0, True, False, False)

    if communicator_info is None:
        raise ValueError("communicator_info is required for a parallel build")
    if xios_orchidee_ok:
        if mpi_init is None or xios_comm_init is None:
            raise ValueError("XIOS initialization requires mpi_init and xios_comm_init")
        mpi_init()
        resolved_communicator = xios_comm_init()
    elif communicator is not None:
        resolved_communicator = communicator
    else:
        if mpi_init is None:
            raise ValueError("mpi_init is required when no communicator is supplied")
        resolved_communicator = mpi_init()

    mpi_size, mpi_rank = map(int, communicator_info(resolved_communicator))
    if mpi_size < 1 or not 0 <= mpi_rank < mpi_size:
        raise ValueError("communicator_info returned an invalid MPI size/rank")
    return MpiInitializationState(
        resolved_communicator,
        mpi_size,
        mpi_rank,
        0,
        mpi_rank == 0,
        True,
        bool(xios_orchidee_ok),
    )


@dataclass(frozen=True)
class XiosDimensions:
    iim_g: int
    jjm_g: int
    jj_begin: int
    jj_nb: int
    nbp_mpi: int
    nvm: int
    nlai: int
    ngrnd: int
    nstm: int
    nnobio: int
    nslm: int
    nwp: int
    ndeep: int
    nsnow: int
    ncarb: int


@dataclass(frozen=True)
class XiosInitControls:
    off_line_mode: bool
    river_routing: bool
    hydrol_cwrr: bool
    ok_freeze_cwrr: bool
    check_cwrr2: bool
    do_floodplains: bool
    ok_stomate: bool
    do_irrigation: bool
    ok_co2: bool
    ok_bvoc: bool
    ok_radcanopy: bool
    ok_multilayer: bool
    ok_bbgfertil_nox: bool
    ok_cropsfertil_nox: bool
    check_waterbal: bool
    impaze: bool
    ok_explicitsnow: bool


@dataclass(frozen=True)
class XiosDomainDefinition:
    ni_glo: int
    nj_glo: int
    ibegin: int
    ni: int
    jbegin: int
    nj: int
    data_dim: int
    data_ibegin: int
    data_ni: int
    data_i_index: np.ndarray
    lonvalue_1d: np.ndarray
    latvalue_1d: np.ndarray


@dataclass(frozen=True)
class XiosAxisDefinition:
    axis_id: str
    n_glo: int
    values: np.ndarray


@dataclass(frozen=True)
class XiosInitializationPlan:
    communicator: object
    calendar: str
    start_date: tuple[int, int, int, int, int, int]
    time_origin: tuple[int, int, int, int, int, int]
    timestep_seconds: float
    domain: XiosDomainDefinition
    axes: tuple[XiosAxisDefinition, ...]
    disabled_fields: tuple[str, ...]


@dataclass(frozen=True)
class XiosInitializationState:
    xios_orchidee_ok: bool
    calendar: str
    time_origin: tuple[int, int, int]
    almaoutput: bool
    plan: XiosInitializationPlan | None


_NO_ROUTING_FIELDS = (
    "basinmap",
    "nbrivers",
    "riversret",
    "hydrographs",
    "fastr",
    "slowr",
    "streamr",
    "laker",
    "lake_overflow",
    "mask_coast",
    "pondr",
    "slowflow",
    "delfastr",
    "delslowr",
    "delstreamr",
    "dellaker",
    "delpondr",
    "delfloodr",
    "irrigmap",
    "swampmap",
    "wbr_stream",
    "wbr_fast",
    "wbr_slow",
    "wbr_lake",
    "reinfiltration",
    "irrigation",
    "netirrig",
    "SurfStor",
)
_CWRR_FIELDS = ("dss", "gqsb", "bqsb", "rsol")
_CHOISNEL_FIELDS = (
    "frac_bare",
    "twbr",
    "nroot",
    "dh",
    "mcs",
    "water2infilt",
    "reinf_slope",
    "evapnu_soil",
    "drainage_soil",
    "transpir_soil",
    "runoff_soil",
    "tmc",
    "njsc",
    "k_litt",
    "soilmoist",
    "mc",
    "kfact_root",
    "vegetmax_soil",
    "undermcr",
    "wtd",
    "ru_corr",
    "ru_corr2",
    "dr_corr",
    "dr_force",
    "qinfilt",
    "ru_infilt",
    "tws",
)
_BVOC_FIELDS = (
    "PAR",
    "flx_fertil_no",
    "flx_iso",
    "flx_mono",
    "flx_ORVOC",
    "flx_MBO",
    "flx_methanol",
    "flx_acetone",
    "flx_acetal",
    "flx_formal",
    "flx_acetic",
    "flx_formic",
    "flx_no_soil",
    "flx_no",
    "flx_apinen",
    "flx_bpinen",
    "flx_limonen",
    "flx_myrcen",
    "flx_sabinen",
    "flx_camphen",
    "flx_3caren",
    "flx_tbocimen",
    "flx_othermono",
    "flx_sesquiter",
    "CRF",
    "fco2",
)
_ALMA_FIELDS = frozenset(("delsoilmoist", "delintercept", "delswe", "soilwet", "twbr"))


def _one_based_axis(axis_id: str, size: int) -> XiosAxisDefinition:
    if size < 1:
        raise ValueError(f"axis {axis_id} must have positive size")
    return XiosAxisDefinition(axis_id, size, np.arange(1, size + 1, dtype=np.float64))


def _disabled_xios_fields(controls: XiosInitControls) -> tuple[str, ...]:
    fields: list[str] = []
    if controls.off_line_mode:
        fields.extend(("q2m", "t2m"))
    if not controls.river_routing:
        fields.extend(_NO_ROUTING_FIELDS)
    fields.extend(_CWRR_FIELDS if controls.hydrol_cwrr else _CHOISNEL_FIELDS)
    if not controls.ok_freeze_cwrr:
        fields.extend(("profil_froz_hydro", "temp_hydro", "kk_moy"))
    if not controls.check_cwrr2:
        fields.extend(("check_infilt", "check_tr", "check_over", "check_under"))
    if not controls.do_floodplains:
        fields.extend(("floodmap", "floodh", "floodr", "floodout"))
    if not controls.ok_stomate:
        fields.extend(("nee", "maint_resp", "hetero_resp", "growth_resp", "npp"))
    if not controls.do_irrigation:
        fields.extend(("irrigation", "netirrig", "irrigmap"))
    if not controls.ok_co2:
        fields.extend(("vbetaco2", "cimean", "cim", "gpp"))
    if not controls.ok_bvoc:
        fields.extend(_BVOC_FIELDS)
    if not controls.ok_bvoc or not controls.ok_radcanopy:
        fields.extend(("PARdf", "PARdr"))
    if not controls.ok_bvoc or not controls.ok_radcanopy or not controls.ok_multilayer:
        fields.extend(("PARsuntab", "PARshtab"))
    if not controls.ok_bvoc or not controls.ok_radcanopy or controls.ok_multilayer:
        fields.extend(("PARsun", "PARsh", "laisun", "laish"))
    if not controls.ok_bvoc or not controls.ok_bbgfertil_nox:
        fields.append("flx_co2_bbg_year")
    if not controls.ok_bvoc or not controls.ok_cropsfertil_nox:
        fields.extend(("N_qt_WRICE_year", "N_qt_OTHER_year"))
    if not controls.check_waterbal:
        fields.append("tot_flux")
    if controls.impaze:
        fields.extend(("soilalb_vis", "soilalb_nir", "vegalb_vis", "vegalb_nir"))
    if controls.ok_explicitsnow:
        fields.append("Qf")
    else:
        fields.extend(
            ("pkappa_snow", "pcapa_snow", "snowliq", "snowrho", "snowheat", "snowgrain")
        )
    return tuple(fields)


def xios_orchidee_init(
    *,
    communicator: object,
    date0: float,
    year: int,
    month: int,
    day: int,
    lon_mpi: np.ndarray,
    lat_mpi: np.ndarray,
    soilth_lev: np.ndarray,
    kindex_mpi: np.ndarray,
    dimensions: XiosDimensions,
    controls: XiosInitControls,
    calendar: str,
    dt_sechiba: float,
    xios_orchidee_ok: bool,
    xios_compiled: bool,
    is_omp_root: bool,
    almaoutput: bool,
    julian_to_date: Callable[[float], tuple[int, int, int, float]],
    transport: Callable[[XiosInitializationPlan], Mapping[str, bool]] | None = None,
    broadcast_almaoutput: Callable[[bool], bool] | None = None,
) -> XiosInitializationState:
    """Build and optionally transport the definitions from source lines 149-511."""

    if xios_orchidee_ok and not xios_compiled:
        raise RuntimeError("XIOS_ORCHIDEE_OK requires a build compiled with XIOS")
    calendar_xios = "d360" if calendar == "360d" else str(calendar)
    year0, month0, day0, _sec0 = julian_to_date(float(date0))
    origin = (int(year0), int(month0), int(day0))
    expected_grid = (dimensions.iim_g, dimensions.jj_nb)
    lon = np.asarray(lon_mpi)
    lat = np.asarray(lat_mpi)
    soil = np.asarray(soilth_lev)
    kindex = np.asarray(kindex_mpi)
    if lon.shape != expected_grid or lat.shape != expected_grid:
        raise ValueError(
            f"lon_mpi and lat_mpi must have Fortran local-domain shape {expected_grid}"
        )
    if soil.shape != (dimensions.ngrnd,):
        raise ValueError("soilth_lev shape does not match ngrnd")
    if kindex.shape != (dimensions.nbp_mpi,):
        raise ValueError("kindex_mpi shape does not match nbp_mpi")
    if np.any(kindex < 1):
        raise ValueError("kindex_mpi uses one-based Fortran indices")
    if not xios_orchidee_ok:
        return XiosInitializationState(
            False, calendar_xios, origin, bool(almaoutput), None
        )

    domain = XiosDomainDefinition(
        dimensions.iim_g,
        dimensions.jjm_g,
        0,
        dimensions.iim_g,
        dimensions.jj_begin - 1,
        dimensions.jj_nb,
        1,
        0,
        dimensions.nbp_mpi,
        kindex.astype(np.int64, copy=False) - 1,
        lon[:, 0],
        lat[0, :],
    )
    axis_sizes = (
        ("nvm", dimensions.nvm),
        ("nlaip1", dimensions.nlai + 1),
        ("nstm", dimensions.nstm),
        ("nnobio", dimensions.nnobio),
        ("albtyp", 2),
        ("nslm", dimensions.nslm),
        ("nwp", dimensions.nwp),
        ("ndeep", dimensions.ndeep),
        ("P10", 10),
        ("P100", 100),
        ("P11", 11),
        ("P101", 101),
        ("nsnow", dimensions.nsnow if controls.ok_explicitsnow else 1),
        ("ncarb", dimensions.ncarb),
    )
    axes = [_one_based_axis(axis_id, size) for axis_id, size in axis_sizes]
    axes.insert(2, XiosAxisDefinition("ngrnd", dimensions.ngrnd, soil))
    plan = XiosInitializationPlan(
        communicator,
        calendar_xios,
        (int(year), int(month), int(day), 0, 0, 0),
        (*origin, 0, 0, 0),
        float(dt_sechiba),
        domain,
        tuple(axes),
        _disabled_xios_fields(controls),
    )

    resolved_alma = bool(almaoutput)
    if is_omp_root:
        if transport is None:
            raise ValueError("root XIOS initialization requires an explicit transport")
        active = transport(plan)
        missing_active = _ALMA_FIELDS.difference(active)
        if missing_active:
            raise ValueError(
                f"XIOS transport omitted active-field states: {sorted(missing_active)}"
            )
        resolved_alma = resolved_alma or any(
            bool(active[field]) for field in _ALMA_FIELDS
        )
    if broadcast_almaoutput is None:
        raise ValueError(
            "enabled XIOS initialization requires an ALMA broadcast callback"
        )
    resolved_alma = bool(broadcast_almaoutput(resolved_alma))
    return XiosInitializationState(True, calendar_xios, origin, resolved_alma, plan)


@dataclass(frozen=True)
class XiosFieldSend:
    field_id: str
    rank: int
    gathered_shape: tuple[int, ...]
    transported: bool


def _send_field_ranked(
    field_id: str,
    field: np.ndarray,
    *,
    rank: int,
    nbp_mpi: int,
    xios_orchidee_ok: bool,
    is_omp_root: bool,
    gather_omp: Callable[[np.ndarray], np.ndarray] | None,
    transport: Callable[[str, np.ndarray], None] | None,
) -> XiosFieldSend:
    array = np.asarray(field)
    if array.ndim != rank:
        raise ValueError(f"{field_id} requires a rank-{rank} field")
    if not xios_orchidee_ok:
        return XiosFieldSend(str(field_id), rank, (), False)
    if gather_omp is None:
        raise ValueError("enabled XIOS sends require an explicit gather_omp callback")
    gathered = np.asarray(gather_omp(array))
    expected = (int(nbp_mpi), *array.shape[1:])
    if gathered.shape != expected:
        raise ValueError(f"gather_omp returned {gathered.shape}, expected {expected}")
    transported = False
    if is_omp_root:
        if transport is None:
            raise ValueError("root XIOS sends require an explicit transport callback")
        transport(str(field_id), gathered)
        transported = True
    return XiosFieldSend(str(field_id), rank, gathered.shape, transported)


def xios_orchidee_send_field_r1d(
    field_id: str, field: np.ndarray, **kwargs: object
) -> XiosFieldSend:
    return _send_field_ranked(field_id, field, rank=1, **kwargs)  # type: ignore[arg-type]


def xios_orchidee_send_field_r2d(
    field_id: str, field: np.ndarray, **kwargs: object
) -> XiosFieldSend:
    return _send_field_ranked(field_id, field, rank=2, **kwargs)  # type: ignore[arg-type]


def xios_orchidee_send_field_r3d(
    field_id: str, field: np.ndarray, **kwargs: object
) -> XiosFieldSend:
    return _send_field_ranked(field_id, field, rank=3, **kwargs)  # type: ignore[arg-type]


def xios_orchidee_send_field_r4d(
    field_id: str, field: np.ndarray, **kwargs: object
) -> XiosFieldSend:
    return _send_field_ranked(field_id, field, rank=4, **kwargs)  # type: ignore[arg-type]


def xios_orchidee_send_field_r5d(
    field_id: str, field: np.ndarray, **kwargs: object
) -> XiosFieldSend:
    return _send_field_ranked(field_id, field, rank=5, **kwargs)  # type: ignore[arg-type]


def xios_orchidee_send_field(
    field_id: str, field: np.ndarray, **kwargs: object
) -> XiosFieldSend:
    """Dispatch the Fortran generic interface by array rank."""

    senders = {
        1: xios_orchidee_send_field_r1d,
        2: xios_orchidee_send_field_r2d,
        3: xios_orchidee_send_field_r3d,
        4: xios_orchidee_send_field_r4d,
        5: xios_orchidee_send_field_r5d,
    }
    array = np.asarray(field)
    try:
        sender = senders[array.ndim]
    except KeyError as exc:
        raise ValueError("XIOS field rank must be between 1 and 5") from exc
    return sender(field_id, array, **kwargs)
