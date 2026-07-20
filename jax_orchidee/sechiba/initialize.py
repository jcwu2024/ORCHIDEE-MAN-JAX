"""Source-ordered SECHIBA initialization for the audited PFT14 path.

This module keeps ``sechiba_init`` allocation state separate from the
``sechiba_initialize`` component dispatcher.  Allocation without a Fortran
assignment is represented as an explicit gap, never as a NumPy/JAX default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, NamedTuple

import jax.numpy as jnp

from jax_orchidee.sechiba.enerbil import enerbil_cold_start_surface_state
from jax_orchidee.sechiba.diffuco import diffuco_initialize
from jax_orchidee.sechiba.condveg import condveg_initialize
from jax_orchidee.sechiba.hydrol import hydrol_cold_start_state
from jax_orchidee.sechiba.slowproc import slowproc_init_pft14_explicit
from jax_orchidee.sechiba.thermosoil import thermosoil_initialize


SECHIBA_INIT_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_init lines 1983-2553",
)

SECHIBA_INITIALIZE_ORDER_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_initialize lines 599-698",
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_initialize lines 700-740",
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_initialize lines 742-783",
)

_OWNER_ORDER = (
    "slowproc_initialize",
    "diffuco_initialize",
    "enerbil_initialize",
    "hydrol_initialize",
    "condveg_initialize",
    "thermosoil_initialize",
)

_ALLOCATION_ONLY_FIELDS = (
    # Lines 1993-2022: index values are not populated until lines 2761-2806.
    "indexveg",
    "indexlai",
    "indexlai0",
    "indexsoil",
    "indexnobio",
    "indexgrnd",
    "indexsnow",
    "indexlayer",
    "indexnslm",
    "indexalb",
    # Lines 2041-2121 and 2134-2209 allocate these without assigning values.
    "drysoil_frac",
    "rsol",
    "evap_bare_lim",
    "soil_deficit",
    "evapot_corr",
    "soiltile",
    "is_crop_soil",
    "reinf_slope",
    "vbeta1",
    "vbeta4",
    "vbeta4_pft",
    "vbeta5",
    "soilcap",
    "soilcap_pft",
    "soilflx",
    "soilflx_pft",
    "tq_cdrag_pft",
    "vbeta_pft",
    "vbeta2",
    "vbeta3",
    "vbeta3pot",
    "gsmean",
    "cimean",
    "veget_max",
    "tot_bare_soil",
    "assim_param",
    # Lines 2249-2318 allocate these without assigning values.
    "liqwt_ratio",
    "mc_peat_above",
    "mc_croppeat_above",
    "mc_man_above",
    "soil_mc",
    "wat_flux",
    "drainage_per_soil",
    "runoff_per_soil",
    "runoff2peat",
    # Lines 2446-2451 assign erodepth only in the allocation-error branch.
    "erodepth",
    # Lines 2483-2499 and 2517-2539 allocate these without assignment.
    "precip2canopy",
    "precip2ground",
    "canopy2ground",
    "vevapsno",
    "vevapnu",
    "vevapnu_pft",
    "t2mdiag",
    "totfrac_nobio",
    "floodout",
    "runoff",
    "drainage",
)


class SechibaInitializationError(RuntimeError):
    """Invalid first-call state or an unresolved exact initialization boundary."""


class SechibaInitializationBoundaryError(SechibaInitializationError):
    """A source-reachable component has no callable exact owner."""


@dataclass(frozen=True)
class SechibaInitDimensions:
    """Runtime dimensions needed by assignments in ``sechiba_init`` 1983-2553."""

    npts: int
    nvm: int
    nslm: int
    nstm: int
    nnobio: int
    itimetide: int
    nflow: int
    nexp: int
    nctext: int
    ncarb: int
    nparts: int
    nelements: int
    nlitt: int
    ndeep: int
    ndoc: int
    npool: int
    nleafages: int
    nlai: int
    ngrnd: int
    nsnow: int
    npco2: int

    def validate(self) -> None:
        for name, value in vars(self).items():
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if self.nvm < 14:
            raise ValueError("the PFT14 path requires nvm >= 14")


@dataclass(frozen=True)
class SechibaInitializePFT14Switches:
    """Paper-run switches controlling branches in ``sechiba_initialize``."""

    hydrol_cwrr: bool = True
    erosion_module: bool = False
    river_routing: bool = True
    nbp_glo: int = 1
    do_fullirr: bool = False
    active_pft_fortran: int = 14


Owner = Callable[..., object]
OwnerInputs = Mapping[str, object] | Callable[[Mapping[str, object], Mapping[str, object]], Mapping[str, object]]


@dataclass(frozen=True)
class SechibaInitializationOwners:
    """Exact local owners and explicit boundaries in Fortran call order.

    Defaults are the existing source-backed initialization owners. Callers may
    replace any owner explicitly for alternate integration boundaries.
    """

    slowproc_initialize: Owner | None = slowproc_init_pft14_explicit
    diffuco_initialize: Owner | None = diffuco_initialize
    enerbil_initialize: Owner | None = enerbil_cold_start_surface_state
    hydrol_initialize: Owner | None = hydrol_cold_start_state
    condveg_initialize: Owner | None = condveg_initialize
    thermosoil_initialize: Owner | None = thermosoil_initialize


class SechibaInitStateResult(NamedTuple):
    state: Mapping[str, object]
    l_first: bool
    assignment_order: tuple[str, ...]
    allocation_only_fields: tuple[str, ...] = _ALLOCATION_ONLY_FIELDS
    allocation_shapes: Mapping[str, tuple[int, ...]] = {}
    provenance: tuple[str, ...] = SECHIBA_INIT_PROVENANCE


class SechibaInitializePFT14Result(NamedTuple):
    state: Mapping[str, object]
    owner_results: Mapping[str, object]
    l_first: bool
    process_order: tuple[str, ...]
    provenance: tuple[str, ...] = SECHIBA_INIT_PROVENANCE + SECHIBA_INITIALIZE_ORDER_PROVENANCE
    gaps: tuple[str, ...] = (
        "Restart/history/XIOS/MPI operations remain caller-owned boundaries inside component owners.",
        "sechiba_init lines 2554-2818 are outside the owned allocation span.",
    )


def _full(state: dict[str, object], order: list[str], name: str, shape: tuple[int, ...], value, dtype) -> None:
    state[name] = jnp.full(shape, value, dtype=dtype)
    order.append(name)


def _allocation_shape_contract(dimensions: SechibaInitDimensions) -> dict[str, tuple[int, ...]]:
    """Shapes allocated by ``sechiba_init`` without a subsequent assignment.

    This is the JAX owner for successful Fortran ``ALLOCATE(...,stat=ier)``
    paths.  Unassigned Fortran storage has no scientific value, so materializing
    zeros or sentinels would be a semantic invention; callers instead get the
    exact allocation contract and component initializers must produce values.

    Fortran provenance: ``sechiba.f90::sechiba_init`` lines 2041-2539,
    including active allocation-success arms at 2135, 2260-2318, 2447, and
    2484-2499.
    """

    d = dimensions
    p, v, s, t = d.npts, d.nvm, d.nslm, d.nstm
    shapes = {
        "indexveg": (p * v,),
        "indexlai": (p * (d.nlai + 1),),
        "indexlai0": (p * d.nlai,),
        "indexsoil": (p * t,),
        "indexnobio": (p * d.nnobio,),
        "indexgrnd": (p * d.ngrnd,),
        "indexsnow": (p * d.nsnow,),
        "indexlayer": (p * s,),
        "indexnslm": (p * s,),
        "indexalb": (p * 2,),
        "drysoil_frac": (p,),
        "rsol": (p,),
        "evap_bare_lim": (p,),
        "soil_deficit": (p, v),
        "evapot_corr": (p,),
        "soiltile": (p, t),
        "is_crop_soil": (t,),
        "reinf_slope": (p,),
        "vbeta1": (p,),
        "vbeta4": (p,),
        "vbeta4_pft": (p, v),
        "vbeta5": (p,),
        "soilcap": (p,),
        "soilcap_pft": (p, v),
        "soilflx": (p,),
        "soilflx_pft": (p, v),
        "tq_cdrag_pft": (p, v),
        "vbeta_pft": (p, v),
        "vbeta2": (p, v),
        "vbeta3": (p, v),
        "vbeta3pot": (p, v),
        "gsmean": (p, v),
        "cimean": (p, v),
        "veget_max": (p, v),
        "tot_bare_soil": (p,),
        "assim_param": (p, v, d.npco2),
        "liqwt_ratio": (p,),
        "mc_peat_above": (p,),
        "mc_croppeat_above": (p,),
        "mc_man_above": (p,),
        "soil_mc": (p, s, t),
        "wat_flux": (p, s, t),
        "drainage_per_soil": (p, t),
        "runoff_per_soil": (p, t),
        "runoff2peat": (p, t),
        "erodepth": (p, v),
        "precip2canopy": (p, v),
        "precip2ground": (p, v),
        "canopy2ground": (p, v),
        "vevapsno": (p,),
        "vevapnu": (p,),
        "vevapnu_pft": (p, v),
        "t2mdiag": (p,),
        "totfrac_nobio": (p,),
        "floodout": (p,),
        "runoff": (p,),
        "drainage": (p,),
    }
    if set(shapes) != set(_ALLOCATION_ONLY_FIELDS):
        raise AssertionError("sechiba_init allocation contract is incomplete")
    return shapes


def sechiba_init_index_tables(
    *,
    index: object,
    kjpij: int,
    nvm: int,
    nstm: int,
    nnobio: int,
    nlai: int,
    ngrnd: int,
    nsnow: int,
    nslm: int,
    offset_omp: int = 0,
    offset_mpi: int = 0,
) -> dict[str, object]:
    """Build flattened Fortran index tables in literal loop/storage order.

    Fortran provenance: ``sechiba.f90::sechiba_init`` lines 2761-2806.
    """

    index_array = jnp.asarray(index, dtype=jnp.int32)
    if index_array.ndim != 1 or index_array.size < 1:
        raise ValueError("index must be a non-empty rank-1 Fortran global-index array")
    if int(kjpij) < int(index_array.size):
        raise ValueError("kjpij must be at least the number of compressed land points")
    counts = {
        "indexlai0": nlai,
        "indexlai": nlai + 1,
        "indexveg": nvm,
        "indexsoil": nstm,
        "indexnobio": nnobio,
        "indexgrnd": ngrnd,
        "indexsnow": nsnow,
        "indexnslm": nslm,
        "indexlayer": nslm,
        "indexalb": 2,
    }
    shift = int(offset_omp) - int(offset_mpi)
    return {
        name: jnp.concatenate(tuple(index_array + level * int(kjpij) + shift for level in range(count)))
        for name, count in counts.items()
    }


def sechiba_prepare_thermosoil_pft_state(
    *,
    mc_layh_s: object,
    mcl_layh_s: object,
    soilmoist: object,
    pref_soil_veg: object,
    ok_laidev: object,
) -> dict[str, object]:
    """Map HYDROL soil tiles to PFT arrays before THERMOSOIL initialization.

    ``pref_soil_veg`` uses Fortran's one-based soil-tile indices.  A PFT may
    have zero cover at any land point; the source maps every configured PFT
    independently of its current vegetation fraction.

    Fortran provenance: ``sechiba.f90::sechiba_initialize`` lines 700-714.
    """

    mc_tiles = jnp.asarray(mc_layh_s)
    mcl_tiles = jnp.asarray(mcl_layh_s)
    moisture = jnp.asarray(soilmoist)
    preferences = jnp.asarray(pref_soil_veg, dtype=jnp.int32)
    lai_development = jnp.asarray(ok_laidev, dtype=jnp.bool_)
    if mc_tiles.ndim != 3 or mcl_tiles.shape != mc_tiles.shape:
        raise ValueError("mc_layh_s and mcl_layh_s must share shape (npts,nslm,nstm)")
    if moisture.shape != mc_tiles.shape[:2]:
        raise ValueError("soilmoist must have shape (npts,nslm)")
    if preferences.ndim != 1 or lai_development.shape != preferences.shape:
        raise ValueError("pref_soil_veg and ok_laidev must share shape (nvm,)")
    tile_index = preferences - 1
    if bool(jnp.any(tile_index < 0)) or bool(jnp.any(tile_index >= mc_tiles.shape[2])):
        raise ValueError("pref_soil_veg contains an out-of-range Fortran soil-tile index")

    is_crop_soil = jnp.zeros((mc_tiles.shape[2],), dtype=jnp.bool_)
    is_crop_soil = is_crop_soil.at[tile_index].max(lai_development)
    return {
        "mc_layh_pft": jnp.take(mc_tiles, tile_index, axis=2),
        "mcl_layh_pft": jnp.take(mcl_tiles, tile_index, axis=2),
        "soilmoist_pft": jnp.broadcast_to(moisture[:, :, None], moisture.shape + (preferences.size,)),
        "is_crop_soil": is_crop_soil,
    }


def sechiba_init_state(
    *,
    l_first: bool,
    dimensions: SechibaInitDimensions,
    undef_sechiba: float = 1.0e20,
    undef_int: int = 999999999,
    dtype=jnp.float64,
) -> SechibaInitStateResult:
    """Materialize only values actually assigned in ``sechiba_init`` 1983-2553.

    Fortran provenance: ``sechiba.f90::sechiba_init`` lines 1983-2553.
    The returned ``l_first`` is false, matching line 1984.  Passing false
    models a second call without ``sechiba_clear`` and raises, matching the
    fatal ``ipslerr_p`` branch at lines 1985-1987.
    """

    dimensions.validate()
    if not bool(l_first):
        raise SechibaInitializationError(
            "sechiba_init requires l_first_sechiba=.TRUE.; lines 1985-1987 stop on a repeated call"
        )

    d = dimensions
    p, v, s, b = d.npts, d.nvm, d.nslm, d.nnobio
    state: dict[str, object] = {}
    order: list[str] = []

    # Restart sentinels, lines 2024-2206, in literal assignment order.
    for name in ("flood_res", "flood_frac", "snow", "snow_age", "evapot"):
        _full(state, order, name, (p,), undef_sechiba, dtype)
    for name in ("humrel", "vegstress"):
        _full(state, order, name, (p, v), undef_sechiba, dtype)
    _full(state, order, "vegstress_old", (p, v), 1.0, dtype)
    _full(state, order, "njsc", (p,), undef_int, jnp.int32)
    _full(state, order, "temp_sol", (p,), undef_sechiba, dtype)
    for name in ("temp_sol_pft", "temp_sol_new_pft"):
        _full(state, order, name, (p, v), undef_sechiba, dtype)
    _full(state, order, "qsurf", (p,), undef_sechiba, dtype)
    _full(state, order, "qsintveg", (p, v), undef_sechiba, dtype)
    _full(state, order, "gpp", (p, v), undef_sechiba, dtype)
    _full(state, order, "salinity", (p,), 30.0, dtype)
    _full(state, order, "tide_height", (p, d.itimetide), 0.0, dtype)
    _full(state, order, "temp_growth", (p,), undef_sechiba, dtype)
    _full(state, order, "veget", (p, v), undef_sechiba, dtype)
    _full(state, order, "lai", (p, v), undef_sechiba, dtype)
    _full(state, order, "frac_age", (p, v, d.nleafages), undef_sechiba, dtype)
    _full(state, order, "height", (p, v), undef_sechiba, dtype)
    for name in ("frac_nobio", "snow_nobio", "snow_nobio_age"):
        _full(state, order, name, (p, b), undef_sechiba, dtype)

    # Explicit scalar/diagnostic initialization, lines 2211-2381.
    _full(state, order, "f_rot_sech", (p,), False, jnp.bool_)
    _full(state, order, "fwet_out", (p,), undef_sechiba, dtype)
    _full(state, order, "drunoff_tot", (p,), 0.0, dtype)
    for name in ("wtp", "fwet_new", "fpeat"):
        _full(state, order, name, (p,), undef_sechiba, dtype)
    for name in ("shumdiag_peat", "shumdiag_croppeat", "shumdiag_man"):
        _full(state, order, name, (p, s), 1.0, dtype)
    _full(state, order, "DOC_EXP_agg", (p, d.nexp, d.nflow), 0.0, dtype)
    for name in ("std_totsed", "gridarea", "effgrid_perc", "bulkdens"):
        _full(state, order, name, (p,), 0.0, dtype)
    _full(state, order, "textfrac", (p, d.nctext), 0.0, dtype)
    for name in ("POC_EXP_agg", "DOC_ERO_agg"):
        _full(state, order, name, (p, d.ncarb), 0.0, dtype)
    _full(state, order, "SED_EXP_agg", (p, d.nctext), 0.0, dtype)

    # Erosion/carbon state, lines 2383-2481.
    _full(state, order, "biomass", (p, v, d.nparts, d.nelements), 0.0, dtype)
    _full(state, order, "litter_above", (p, d.nlitt, v, d.nelements), 0.0, dtype)
    _full(state, order, "litter_below", (p, d.nlitt, v, d.ndeep, d.nelements), 0.0, dtype)
    _full(state, order, "carbon_32l", (p, d.ncarb, v, d.ndeep), 0.0, dtype)
    _full(state, order, "DOC", (p, v, d.ndeep, d.ndoc, d.npool, d.nelements), 0.0, dtype)
    _full(state, order, "lignin_struc_above", (p, v), 0.0, dtype)
    _full(state, order, "lignin_struc_below", (p, v, d.ndeep), 0.0, dtype)
    for name in ("depth_deepsoil", "seddep_rate"):
        _full(state, order, name, (p, v), 0.0, dtype)
    _full(state, order, "sed_deposition", (p, d.nctext), 0.0, dtype)
    _full(state, order, "poc_deposition", (p, d.ncarb), 0.0, dtype)
    _full(state, order, "sed_deposition_d", (p,), 0.0, dtype)
    _full(state, order, "poc_deposition_d", (p, d.ncarb), 0.0, dtype)
    for name in ("DOC_to_topsoil", "DOC_to_subsoil"):
        _full(state, order, name, (p, d.nflow), 0.0, dtype)

    # Routing exchange allocation state, lines 2501-2551.
    for name in ("fastr", "streamfl_frac", "stream_frac"):
        _full(state, order, name, (p,), undef_sechiba, dtype)
    _full(state, order, "vevapflo", (p,), 0.0, dtype)
    for name in ("returnflow", "reinfiltration", "irrigation"):
        _full(state, order, name, (p, d.nflow), 0.0, dtype)

    return SechibaInitStateResult(
        state=state,
        l_first=False,
        assignment_order=tuple(order),
        allocation_shapes=_allocation_shape_contract(dimensions),
    )


def _validate_switches(switches: SechibaInitializePFT14Switches) -> None:
    expected = SechibaInitializePFT14Switches()
    for name in vars(expected):
        if name == "do_fullirr":
            continue
        if getattr(switches, name) != getattr(expected, name):
            raise NotImplementedError(
                f"unsupported sechiba_initialize paper branch: {name}={getattr(switches, name)!r}; "
                f"required {getattr(expected, name)!r}"
            )


def sechiba_initialize_full_irrigation_restart_transition(
    *, restart_irrigation, val_exp: float = -9999.0
):
    """Replay ``sechiba_initialize`` lines 772-777 after ``restget_p``.

    The caller owns restart transport and supplies the materialized land by
    flow field. Uniform valid restart values are reset to zero; spatially
    varying values, or values below ``val_exp``, are retained exactly.

    Fortran provenance: ``src_sechiba/sechiba.f90::sechiba_initialize``
    lines 772-777.
    """
    irrigation = jnp.asarray(restart_irrigation)
    if irrigation.ndim != 2 or irrigation.shape[1] < 1:
        raise ValueError("restart_irrigation must have shape (land, positive flow)")
    first_flow = irrigation[:, 0]
    reset = jnp.logical_not(
        jnp.logical_or(jnp.min(first_flow) < jnp.max(first_flow), jnp.max(first_flow) < val_exp)
    )
    return jnp.where(reset, jnp.zeros_like(irrigation), irrigation)


def _payload(result: object) -> dict[str, object]:
    if isinstance(result, Mapping):
        values = dict(result)
    elif hasattr(result, "as_payload"):
        values = dict(result.as_payload())
    elif hasattr(result, "_asdict"):
        values = dict(result._asdict())
    else:
        raise TypeError("initialization owner must return a mapping, NamedTuple, or object with as_payload()")
    values.pop("provenance", None)
    values.pop("notes", None)
    return values


def _owner_kwargs(
    spec: OwnerInputs,
    state: Mapping[str, object],
    results: Mapping[str, object],
) -> dict[str, object]:
    values = spec(state, results) if callable(spec) else spec
    if not isinstance(values, Mapping):
        raise TypeError("owner input factory must return a mapping")
    return dict(values)


def sechiba_initialize_pft14_explicit(
    *,
    l_first: bool,
    dimensions: SechibaInitDimensions,
    forcing: Mapping[str, object],
    pft_parameters: Mapping[str, object],
    owner_inputs: Mapping[str, OwnerInputs],
    owners: SechibaInitializationOwners = SechibaInitializationOwners(),
    switches: SechibaInitializePFT14Switches = SechibaInitializePFT14Switches(),
    restart_irrigation: object | None = None,
    val_exp: float = -9999.0,
) -> SechibaInitializePFT14Result:
    """Compose exact initialization owners in ``sechiba_initialize`` order.

    ``forcing`` and ``pft_parameters`` are validation/input boundaries rather
    than hidden globals.  Owner input factories receive the current state and
    prior owner results, allowing HYDROL output to feed THERMOSOIL without an
    out-of-order precomputation.  No restart, history, XIOS, map interpolation,
    or MPI value is synthesized here.

    Fortran provenance: ``sechiba_initialize`` lines 599-783, with local
    allocation state from ``sechiba_init`` lines 1983-2553.
    """

    _validate_switches(switches)
    dimensions.validate()
    if not forcing:
        raise ValueError("forcing must be a non-empty explicit mapping")
    for name, value in forcing.items():
        array = jnp.asarray(value)
        if array.ndim < 1 or array.shape[0] != dimensions.npts:
            raise ValueError(f"forcing[{name!r}] must have leading land dimension {dimensions.npts}")

    if "veget_max_default" not in pft_parameters:
        raise ValueError("pft_parameters must include veget_max_default")
    vegetation = jnp.asarray(pft_parameters["veget_max_default"])
    if vegetation.ndim == 1:
        vegetation = jnp.broadcast_to(vegetation[None, :], (dimensions.npts, vegetation.shape[0]))
    if vegetation.shape != (dimensions.npts, dimensions.nvm):
        raise ValueError("veget_max_default must have shape (nvm,) or (npts,nvm)")
    active = switches.active_pft_fortran - 1
    if active < 0 or active >= dimensions.nvm:
        raise ValueError("active_pft_fortran is outside the configured nvm")

    allocation = sechiba_init_state(l_first=l_first, dimensions=dimensions)
    state = dict(allocation.state)
    results: dict[str, object] = {}
    order = ["sechiba_init"]

    for name in _OWNER_ORDER:
        owner = getattr(owners, name)
        if owner is None:
            raise SechibaInitializationBoundaryError(
                f"{name} is source-reachable and requires an explicit exact owner boundary"
            )
        if name not in owner_inputs:
            raise SechibaInitializationBoundaryError(f"missing explicit inputs for {name}")
        result = owner(**_owner_kwargs(owner_inputs[name], state, results))
        results[name] = result
        state.update(_payload(result))
        order.append(name)

        if name == "slowproc_initialize":
            if "co2_flux" not in state or "veget_max" not in state:
                raise SechibaInitializationBoundaryError(
                    "slowproc_initialize must write co2_flux and veget_max before lines 620-623"
                )
            co2_flux = jnp.asarray(state["co2_flux"])
            veget_max = jnp.asarray(state["veget_max"])
            if co2_flux.shape != (dimensions.npts, dimensions.nvm) or veget_max.shape != co2_flux.shape:
                raise ValueError("co2_flux and veget_max must have shape (npts,nvm)")
            state["netco2flux"] = jnp.sum(co2_flux[:, 1:] * veget_max[:, 1:], axis=1)
            order.append("netco2flux_writeback")
            if "rot_cmd_max" not in state:
                raise SechibaInitializationBoundaryError(
                    "slowproc_initialize must write rot_cmd_max before the allocation at lines 625-632"
                )
            rot_cmd_max = int(state["rot_cmd_max"])
            if rot_cmd_max < 1:
                raise ValueError("slowproc_initialize rot_cmd_max must be positive")
            state["rot_cmd"] = jnp.zeros((dimensions.npts, rot_cmd_max), dtype=jnp.int32)
            order.append("rot_cmd_allocation")

        if name == "hydrol_initialize":
            required = ("mc_layh_s", "mcl_layh_s", "soilmoist")
            if all(field in state for field in required):
                if "pref_soil_veg" not in pft_parameters or "ok_LAIdev" not in pft_parameters:
                    raise SechibaInitializationBoundaryError(
                        "PFT soil mapping requires pref_soil_veg and ok_LAIdev before thermosoil_initialize"
                    )
                mapped = sechiba_prepare_thermosoil_pft_state(
                    mc_layh_s=state["mc_layh_s"],
                    mcl_layh_s=state["mcl_layh_s"],
                    soilmoist=state["soilmoist"],
                    pref_soil_veg=pft_parameters["pref_soil_veg"],
                    ok_laidev=pft_parameters["ok_LAIdev"],
                )
                state.update(mapped)
                order.append("hydrol_soil_tiles_to_pft")

    # Paper case: EROSION_MODULE=n and RIVER_ROUTING=y with nbp_glo=1.
    order.append("erosion_initialize[fixed-off]")
    for name, shape in (
        ("riverflow", (dimensions.npts, dimensions.nflow)),
        ("coastalflow", (dimensions.npts, dimensions.nflow)),
        ("returnflow", (dimensions.npts, dimensions.nflow)),
        ("reinfiltration", (dimensions.npts, dimensions.nflow)),
        ("irrigation", (dimensions.npts, dimensions.nflow)),
        ("sed_deposition", (dimensions.npts, dimensions.nctext)),
        ("poc_deposition", (dimensions.npts, dimensions.ncarb)),
        ("flood_frac", (dimensions.npts,)),
        ("streamfl_frac", (dimensions.npts,)),
        ("stream_frac", (dimensions.npts,)),
        ("flood_res", (dimensions.npts,)),
        ("fastr", (dimensions.npts,)),
    ):
        state[name] = jnp.zeros(shape, dtype=jnp.float64)
    order.append("routing_initialize[single-point-no-call]")
    if switches.do_fullirr:
        if restart_irrigation is None:
            raise SechibaInitializationBoundaryError(
                "DO_FULLIRR requires explicit replayed restart_irrigation at sechiba_initialize lines 772-777"
            )
        replayed = sechiba_initialize_full_irrigation_restart_transition(
            restart_irrigation=restart_irrigation, val_exp=val_exp
        )
        if replayed.shape != state["irrigation"].shape:
            raise ValueError("restart_irrigation shape must match initialized irrigation")
        state["irrigation"] = replayed
        order.append("full_irrigation[restart-replay]")
    else:
        order.append("full_irrigation[fixed-off]")

    missing_outputs = tuple(name for name in ("z0m", "z0h", "emis", "qsurf") if name not in state)
    if missing_outputs:
        raise SechibaInitializationBoundaryError(
            f"component owners did not produce initialization output fields: {missing_outputs}"
        )
    for source, target in (("z0m", "z0m_out"), ("z0h", "z0h_out"), ("emis", "emis_out"), ("qsurf", "qsurf_out")):
        state[target] = jnp.asarray(state[source])
    order.append("internal_output_writeback")

    return SechibaInitializePFT14Result(state, results, allocation.l_first, tuple(order))


# Public short name is deliberately scoped by the function's PFT14 contract.
sechiba_initialize = sechiba_initialize_pft14_explicit
