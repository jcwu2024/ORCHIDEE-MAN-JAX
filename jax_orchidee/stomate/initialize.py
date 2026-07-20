"""Source-ordered STOMATE initialization for the fixed paper PFT14 path.

This module owns only ``stomate.f90::stomate_initialize`` lines 1465-2350.
Physical restart I/O, the large ``stomate_init`` allocation routine, and
parameter materialization remain explicit owner boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, NamedTuple

import jax.numpy as jnp

from jax_orchidee.stomate.carbon_kernels import deadleaf_cover_from_litter, vmax_step


STOMATE_INITIALIZE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_initialize lines 1465-2350",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_initialize lines 1491-1500 calls stomate_init then data",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_initialize lines 1503-1561 zeros fluxes then calls readstart",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_initialize lines 1586-1818 initializes deep-carbon state and takes the no-OK_LAIDEV arm",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_initialize lines 1821-1848 validates the STOMATE time step",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_initialize lines 1853-2232 gates three forcing writers",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_initialize lines 2234-2245 computes veget_cov and veget_cov_max",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_initialize lines 2249-2254 calls stomate_var_init",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_initialize lines 2258-2261 initializes harvest_above and temp_growth",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_initialize lines 2272-2350 fixes frozen respiration and gates CH4/peat initialization",
)

STOMATE_VAR_INIT_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_var_init lines 9249-9321",
    "fortran_source/ORCHIDEE/src_stomate/stomate_vmax.f90::vmax lines 105-363",
    "fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90::deadleaf lines 1314-1353",
)


class StomateInitializationBoundaryError(RuntimeError):
    """Raised when a source-reachable owner or value has not been supplied."""


@dataclass(frozen=True)
class StomateInitializeDimensions:
    """Dynamic dimensions needed by the owned initialization span."""

    npts: int
    nvm: int
    active_pft_fortran: int = 14

    def validate(self) -> None:
        if self.npts < 1:
            raise ValueError("npts must be positive")
        if self.nvm < self.active_pft_fortran:
            raise ValueError(
                f"nvm={self.nvm} cannot contain active Fortran PFT {self.active_pft_fortran}"
            )


@dataclass(frozen=True)
class StomateInitializePFT14Switches:
    """Structural switches fixed by the materialized paper run.

    Provenance: ``outputs/server_1961_trace_full_20260623/run/used_run.def``
    lines 137-199, 339-341, 1510-1525, and 4763-4785.
    ``ok_leak`` is true but its initialization allocation is nested beneath
    the disabled carbon-forcing writer at lines 2059-2185.
    """

    spinup_analytic: bool = False
    ok_laidev: bool | tuple[bool, ...] = False
    ok_rotate: bool = False
    ok_leak: bool = True
    peat_occur: bool = False
    ch4_calcul: bool = False
    stomate_forcing_name: str = "NONE"
    stomate_cforcing_name: str = "NONE"
    cforcing_permafrost_name: str = "NONE"


Owner = Callable[..., object]
OwnerInputs = Mapping[str, object] | Callable[[Mapping[str, object], Mapping[str, object]], Mapping[str, object]]


@dataclass(frozen=True)
class StomateInitializationOwners:
    """Exact process owners dispatched in Fortran source order."""

    stomate_init: Owner | None = None
    data: Owner | None = None
    readstart: Owner | None = None
    season: Owner | None = None
    carbon: Owner | None = None


class StomateVarInitResult(NamedTuple):
    assim_param: jnp.ndarray
    deadleaf_cover: jnp.ndarray
    provenance: tuple[str, ...] = STOMATE_VAR_INIT_PROVENANCE


class StomateInitializePFT14Result(NamedTuple):
    state: Mapping[str, object]
    owner_results: Mapping[str, object]
    process_order: tuple[str, ...]
    provenance: tuple[str, ...] = STOMATE_INITIALIZE_PROVENANCE
    source_undefined_fields: tuple[str, ...] = ("N_limfert",)


class StomateInitializeBoundaryDisposition(NamedTuple):
    line: int
    condition: str
    taken: bool
    classification: str
    reason: str


def stomate_initialize_forcing_boundaries(
    switches: StomateInitializePFT14Switches,
) -> tuple[StomateInitializeBoundaryDisposition, ...]:
    """Classify the eight reachable forcing gates without inventing state.

    Provenance: ``stomate.f90::stomate_initialize`` lines 1866, 2059,
    2115, 2150, 2180, 2199, 2205, and 2220.  In the paper configuration
    every file writer is disabled; the leak/peat sub-gates remain nested
    inside those disabled I/O branches and therefore assign no model state.
    """

    carbon_writer = switches.stomate_cforcing_name.strip() != "NONE"
    permafrost_writer = switches.cforcing_permafrost_name.strip() != "NONE"
    return (
        StomateInitializeBoundaryDisposition(
            1866,
            "TRIM(stomate_forcing_name) /= 'NONE'",
            switches.stomate_forcing_name.strip() != "NONE",
            "io_gate",
            "teststomate forcing writer; disabled in the paper configuration",
        ),
        StomateInitializeBoundaryDisposition(
            2059,
            "TRIM(stomate_Cforcing_name) /= 'NONE'",
            carbon_writer,
            "io_gate",
            "forcesoil serialization branch; disabled, so no scientific state is assigned",
        ),
        StomateInitializeBoundaryDisposition(
            2115, "ok_leak", carbon_writer and switches.ok_leak, "nested_allocation_gate",
            "nested under the disabled carbon forcing writer",
        ),
        StomateInitializeBoundaryDisposition(
            2150, "ok_leak", carbon_writer and switches.ok_leak, "nested_io_state_gate",
            "zeroes serialization buffers only when the carbon writer is enabled",
        ),
        StomateInitializeBoundaryDisposition(
            2180, "ok_peat", carbon_writer and switches.peat_occur, "nested_io_state_gate",
            "peat forcing buffer is nested under the disabled carbon writer",
        ),
        StomateInitializeBoundaryDisposition(
            2199,
            "TRIM(Cforcing_permafrost_name) /= 'NONE'",
            permafrost_writer,
            "io_gate",
            "permafrost forcing writer; disabled in the paper configuration",
        ),
        StomateInitializeBoundaryDisposition(
            2205, ".NOT. ok_leak", permafrost_writer and not switches.ok_leak,
            "nested_allocation_gate", "nested under the disabled permafrost writer",
        ),
        StomateInitializeBoundaryDisposition(
            2220, ".NOT. ok_leak", permafrost_writer and not switches.ok_leak,
            "nested_io_state_gate", "zeroes writer buffers only inside the disabled writer",
        ),
    )


_OWNER_PREFIX = ("stomate_init", "data", "readstart", "season")


def _validate_switches(switches: StomateInitializePFT14Switches, nvm: int) -> None:
    expected = StomateInitializePFT14Switches()
    for name in (
        "spinup_analytic",
        "ok_rotate",
        "ok_leak",
        "peat_occur",
        "ch4_calcul",
        "stomate_forcing_name",
        "stomate_cforcing_name",
        "cforcing_permafrost_name",
    ):
        actual = getattr(switches, name)
        wanted = getattr(expected, name)
        if actual != wanted:
            raise NotImplementedError(
                f"unsupported stomate_initialize paper branch: {name}={actual!r}; required {wanted!r}"
            )
    ok_laidev = switches.ok_laidev
    if isinstance(ok_laidev, bool):
        active = ok_laidev
    else:
        if len(ok_laidev) != nvm:
            raise ValueError(f"ok_laidev must contain nvm={nvm} entries")
        active = any(bool(value) for value in ok_laidev)
    if active:
        raise NotImplementedError(
            "unsupported stomate_initialize paper branch: ANY(ok_LAIdev) must be false (lines 1563-1818)"
        )


def _payload(value: object) -> dict[str, object]:
    """Flatten owner state contracts while retaining ``None`` boundaries."""

    if isinstance(value, Mapping):
        items = value.items()
    elif hasattr(value, "_asdict"):
        items = value._asdict().items()
    elif hasattr(value, "__dataclass_fields__"):
        items = ((name, getattr(value, name)) for name in value.__dataclass_fields__)
    elif hasattr(value, "as_payload"):
        items = value.as_payload().items()
    else:
        raise TypeError("initialization owner must return a mapping, dataclass, NamedTuple, or as_payload object")

    result: dict[str, object] = {}
    for name, item in items:
        if name in {"provenance", "notes", "report"}:
            continue
        if name.endswith("_state") and (
            isinstance(item, Mapping) or hasattr(item, "_asdict") or hasattr(item, "__dataclass_fields__")
        ):
            result.update(_payload(item))
        else:
            result[name] = item
    return result


def _owner_kwargs(
    spec: OwnerInputs,
    state: Mapping[str, object],
    results: Mapping[str, object],
) -> dict[str, object]:
    values = spec(state, results) if callable(spec) else spec
    if not isinstance(values, Mapping):
        raise TypeError("owner input factory must return a mapping")
    return dict(values)


def _require_shape(state: Mapping[str, object], name: str, shape: tuple[int, ...]) -> jnp.ndarray:
    if name not in state:
        raise StomateInitializationBoundaryError(f"owner state must define {name!r}")
    value = jnp.asarray(state[name])
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    return value


def _require_deep_carbon(state: Mapping[str, object], name: str, npts: int, nvm: int) -> jnp.ndarray:
    if name not in state:
        raise StomateInitializationBoundaryError(f"owner state must define {name!r}")
    value = jnp.asarray(state[name])
    if value.ndim != 3 or value.shape[0] != npts or value.shape[2] != nvm:
        raise ValueError(f"{name} must have shape (npts,ndeep,nvm)")
    return value


def stomate_var_init_carbon_owner(
    *,
    assim_param,
    leaf_age,
    leaf_frac,
    dead_leaves,
    veget_cov_max,
    sla_calc,
    val_exp: float,
    vmax_inputs: Mapping[str, object] | None = None,
) -> StomateVarInitResult:
    """Execute the carbon-owned work of ``stomate_var_init`` exactly.

    ``assim_param`` is preserved unless every entry equals ``val_exp``.  The
    fallback requires explicit ``vmax_inputs``; this avoids manufacturing the
    source-uninitialized ``N_limfert`` value from the no-crop arm at lines
    1814-1818.  ``vmax`` receives ``dt_days=0`` as at lines 9292-9309.
    """

    assim = jnp.asarray(assim_param)
    if bool(jnp.all(assim == val_exp)):
        if vmax_inputs is None:
            raise StomateInitializationBoundaryError(
                "all assim_param values are val_exp; explicit vmax_inputs including n_limfert are required"
            )
        inputs = dict(vmax_inputs)
        ivcmax = int(inputs.pop("ivcmax", 0))
        inputs.update(leaf_age=leaf_age, leaf_frac=leaf_frac, dt_days=0.0)
        vmax = vmax_step(**inputs)
        if assim.ndim != 3:
            raise ValueError("assim_param must have shape (npts,nvm,npco2)")
        if ivcmax < 0 or ivcmax >= assim.shape[2]:
            raise ValueError("ivcmax is outside the assim_param axis")
        assim = assim.at[:, :, ivcmax].set(vmax.vcmax)
        assim = assim.at[:, 0, ivcmax].set(0.0)
    deadleaf_cover = deadleaf_cover_from_litter(dead_leaves, veget_cov_max, sla_calc)
    return StomateVarInitResult(assim, deadleaf_cover)


def stomate_initialize_pft14_explicit(
    *,
    dimensions: StomateInitializeDimensions,
    owner_inputs: Mapping[str, OwnerInputs],
    owners: StomateInitializationOwners,
    dt_days: float,
    dt_sechiba_seconds: float,
    max_dt_days: float,
    switches: StomateInitializePFT14Switches = StomateInitializePFT14Switches(),
    min_stomate: float = 1.0e-8,
    one_day_seconds: float = 86400.0,
    dtype=jnp.float64,
) -> StomateInitializePFT14Result:
    """Compose the fixed-switch PFT14 lifecycle in source order.

    Owners receive only caller-supplied keyword arguments. Input factories may
    inspect current state and prior owner results, which lets the season and
    carbon boundaries consume the normalized readstart contracts without an
    out-of-order precomputation. No missing restart or parameter value is
    synthesized.
    """

    dimensions.validate()
    _validate_switches(switches, dimensions.nvm)
    if max_dt_days <= 0.0:
        raise ValueError("max_dt_days must be positive")

    state: dict[str, object] = {}
    results: dict[str, object] = {}
    order: list[str] = []

    for name in _OWNER_PREFIX:
        owner = getattr(owners, name)
        if owner is None:
            raise StomateInitializationBoundaryError(
                f"{name} is source-reachable and requires an explicit exact owner boundary"
            )
        if name not in owner_inputs:
            raise StomateInitializationBoundaryError(f"missing explicit inputs for {name}")
        if name == "readstart":
            state["co2_flux"] = jnp.zeros((dimensions.npts, dimensions.nvm), dtype=dtype)
            state["fco2_lu"] = jnp.zeros((dimensions.npts,), dtype=dtype)
            order.append("flux_zero")
        result = owner(**_owner_kwargs(owner_inputs[name], state, results))
        results[name] = result
        state.update(_payload(result))
        order.append(name)

    p, v = dimensions.npts, dimensions.nvm
    veget = _require_shape(state, "veget", (p, v))
    veget_max = _require_shape(state, "veget_max", (p, v))
    totfrac_nobio = _require_shape(state, "totfrac_nobio", (p,))
    deep_a = _require_deep_carbon(state, "deepC_a", p, v)
    deep_s = _require_shape(state, "deepC_s", deep_a.shape)
    deep_p = _require_shape(state, "deepC_p", deep_a.shape)
    state["soilc_total"] = deep_a + deep_s + deep_p
    state["heat_Zimov"] = jnp.zeros_like(deep_a)
    state["N_limfert"] = None
    order.extend(("deep_carbon_initialization", "crop_initialization[fixed-off]"))

    if "dt_days_read" not in state:
        raise StomateInitializationBoundaryError("readstart/season owner state must define 'dt_days_read'")
    if float(dt_days) - float(jnp.floor(float(dt_days) + 0.5)) > float(min_stomate):
        raise ValueError("dt_days must be a multiple of a full day (stomate_initialize lines 1829-1834)")
    if float(dt_days) > float(max_dt_days):
        raise ValueError("dt_days exceeds max_dt_days (stomate_initialize lines 1836-1841)")
    if float(dt_sechiba_seconds) > float(dt_days) * float(one_day_seconds):
        raise ValueError("dt_days is smaller than the forcing time step (stomate_initialize lines 1843-1848)")
    state["dt_days"] = float(dt_days)
    state["dt_days_changed"] = float(dt_days) != float(state["dt_days_read"])
    order.append("time_step_checks")
    state["forcing_boundary_dispositions"] = stomate_initialize_forcing_boundaries(switches)
    order.extend(("stomate_forcing[fixed-off]", "carbon_forcing[fixed-off]", "permafrost_forcing[fixed-off]"))

    denominator = 1.0 - totfrac_nobio
    land = denominator > float(min_stomate)
    safe_denominator = jnp.where(land, denominator, 1.0)
    state["veget_cov"] = jnp.where(land[:, None], veget / safe_denominator[:, None], 0.0)
    state["veget_cov_max"] = jnp.where(land[:, None], veget_max / safe_denominator[:, None], 0.0)
    order.append("vegetation_cover")

    carbon = owners.carbon
    if carbon is None:
        raise StomateInitializationBoundaryError(
            "carbon is source-reachable through stomate_var_init and requires an explicit exact owner boundary"
        )
    if "carbon" not in owner_inputs:
        raise StomateInitializationBoundaryError("missing explicit inputs for carbon")
    carbon_result = carbon(**_owner_kwargs(owner_inputs["carbon"], state, results))
    results["carbon"] = carbon_result
    state.update(_payload(carbon_result))
    order.append("carbon")
    if "assim_param" not in state:
        raise StomateInitializationBoundaryError("carbon owner must define 'assim_param'")
    assim_param = jnp.asarray(state["assim_param"])
    if assim_param.ndim != 3 or assim_param.shape[:2] != (p, v):
        raise ValueError("assim_param must have shape (npts,nvm,npco2)")
    _require_shape(state, "deadleaf_cover", (p,))

    t2m_month = _require_shape(state, "t2m_month", (p,))
    state["harvest_above"] = jnp.zeros((p,), dtype=t2m_month.dtype)
    state["temp_growth"] = t2m_month - 273.15
    state["frozen_respiration_func"] = 1
    order.extend(("harvest_above_zero", "temp_growth", "frozen_respiration_func", "ch4[fixed-off]", "peat[fixed-off]"))

    return StomateInitializePFT14Result(state, results, tuple(order))


stomate_initialize = stomate_initialize_pft14_explicit
StomateInitializeResult = StomateInitializePFT14Result
StomateInitDimensions = StomateInitializeDimensions
