"""PFT14 fixed-switch orchestration for the active ``sechiba_main`` path.

The process owners remain in their existing modules.  This module composes
them and closes the single-landpoint paper-case tail without implementing IO
or silently accepting inactive process branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, NamedTuple

import jax.numpy as jnp
import numpy as np

from jax_orchidee.sechiba.hydrol_thermosoil_completion import (
    hydrol_rotation_update,
)
from jax_orchidee.sechiba.science_completion import (
    slowproc_change_frac_source_routed,
)
from jax_orchidee.sechiba.sechiba_step import (
    SechibaExplicitCoupledStepResult,
    sechiba_explicit_coupled_step,
)
from jax_orchidee.sechiba.thermosoil import thermosoil_rotation_update


SECHIBA_MAIN_PFT14_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 997-1216",
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1219-1780",
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_end lines 3114-3140",
)

SECHIBA_MAIN_OUTPUT_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1420-1525 "
    "classifies vegetation, computes diagnostics, and sends XIOS fields",
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1527-1625 "
    "writes the non-ALMA primary history stream",
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1668-1758 "
    "guards the secondary history stream with hist2_id > 0",
)

_SECHIBA_SOURCE_FILE = "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90"
_SECHIBA_MAIN_SUBROUTINE = "sechiba_main"

_ROUTING_ZERO_FIELDS = (
    "riverflow",
    "coastalflow",
    "returnflow",
    "reinfiltration",
    "irrigation",
    "sed_deposition",
    "poc_deposition",
    "flood_frac",
    "stream_frac",
    "streamfl_frac",
    "flood_res",
    "fastr",
)


@dataclass(frozen=True)
class SechibaMainPFT14FixedSwitches:
    """Audited fixed switches for the single-point paper path."""

    nvm: int = 14
    hydrol_cwrr: bool = True
    ok_explicitsnow: bool = True
    dynpeat_pwt: bool = False
    erosion_module: bool = False
    river_routing: bool = True
    nbp_glo: int = 1
    do_fullirr: bool = False
    ok_rotate: bool = False
    done_stomate_lcchange: bool = False
    dyn_peat: bool = False
    use_age_class: bool = False
    ldrestart_write: bool = False
    almaoutput: bool = False
    hist2_active: bool = False
    do_floodplains: bool = False
    ok_co2: bool = True
    ok_stomate: bool = True


@dataclass(frozen=True)
class SechibaOutputField:
    """One source-ordered output call, represented without performing IO."""

    name: str
    value: object
    source_line: int
    source_file: str = _SECHIBA_SOURCE_FILE
    source_subroutine: str = _SECHIBA_MAIN_SUBROUTINE

    @property
    def provenance(self) -> str:
        return f"{self.source_file}::{self.source_subroutine} line {self.source_line}"


@dataclass(frozen=True)
class SechibaMainOutputPacket:
    """Pure-data model-output packet for ``sechiba_main`` lines 1420-1758."""

    xios: tuple[SechibaOutputField, ...]
    history: tuple[SechibaOutputField, ...]
    history2: tuple[SechibaOutputField, ...]
    diagnostics: Mapping[str, object]
    history2_active: bool
    provenance: tuple[str, ...] = SECHIBA_MAIN_OUTPUT_PROVENANCE

    @staticmethod
    def as_mapping(fields: tuple[SechibaOutputField, ...]) -> dict[str, object]:
        return {field.name: field.value for field in fields}


class SechibaMainTailResult(NamedTuple):
    """State after the source-ordered ``sechiba_main`` lines 1219-1780 tail."""

    state: Mapping[str, object]
    xios_result: object | None
    history_result: object | None
    modelout: SechibaMainOutputPacket
    process_order: tuple[str, ...]
    restart_result: object | None
    provenance: tuple[str, ...] = SECHIBA_MAIN_PFT14_PROVENANCE[1:]


class RoutingSlowprocWriteback(NamedTuple):
    """Daily routing fields passed from SECHIBA to slow processes."""

    doc_to_topsoil: jnp.ndarray
    doc_to_subsoil: jnp.ndarray
    sed_deposition_d: jnp.ndarray
    poc_deposition_d: jnp.ndarray


class SechibaRotationResult(NamedTuple):
    """All state mutated by ``sechiba_main`` crop rotation."""

    state: Mapping[str, object]
    matrices: tuple[jnp.ndarray, ...]
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1309-1407",
        "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_get_cmd lines 3144-3190",
        "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_rotation_update lines 3591-3807",
        "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_rotation_update lines 3790-3858",
    )


class SechibaPostOutputResult(NamedTuple):
    """LCC and restart state after ``sechiba_main`` model output."""

    state: Mapping[str, object]
    process_order: tuple[str, ...]
    restart_result: object | None
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1760-1784",
    )


def routing_to_slowproc_writeback(
    *, reinfiltration, irrigation, returnflow, sed_deposition, poc_deposition, dt_sechiba
) -> RoutingSlowprocWriteback:
    """Reproduce ``sechiba_main`` 1252-1256 for ``RIVER_ROUTING=true``."""

    scale = jnp.asarray(1000.0 * 86400.0) / jnp.asarray(dt_sechiba)
    day_scale = jnp.asarray(86400.0) / jnp.asarray(dt_sechiba)
    return RoutingSlowprocWriteback(
        (jnp.asarray(reinfiltration) + jnp.asarray(irrigation)) * scale,
        jnp.asarray(returnflow) * scale,
        jnp.sum(jnp.asarray(sed_deposition), axis=1) * day_scale,
        jnp.asarray(poc_deposition) * day_scale,
    )


class SechibaMainPFT14Result(NamedTuple):
    """Existing local process chain followed by the fixed paper-case tail."""

    step: SechibaExplicitCoupledStepResult
    tail: SechibaMainTailResult
    provenance: tuple[str, ...] = SECHIBA_MAIN_PFT14_PROVENANCE

    @property
    def state(self) -> Mapping[str, object]:
        return self.tail.state


OutputBoundary = Callable[[tuple[SechibaOutputField, ...]], object]
StepOwner = Callable[..., SechibaExplicitCoupledStepResult]


def validate_sechiba_main_pft14_switches(switches: SechibaMainPFT14FixedSwitches) -> None:
    """Reject every branch outside the audited PFT14 single-point path."""

    expected = SechibaMainPFT14FixedSwitches()
    for name in (
        "nvm",
        "hydrol_cwrr",
        "ok_explicitsnow",
        "dynpeat_pwt",
        "erosion_module",
        "river_routing",
        "nbp_glo",
        "almaoutput",
        "hist2_active",
        "do_floodplains",
        "ok_co2",
        "ok_stomate",
    ):
        actual = getattr(switches, name)
        required = getattr(expected, name)
        if actual != required:
            raise NotImplementedError(
                f"unsupported sechiba_main PFT14 fixed-switch branch: {name}={actual!r}; "
                f"the audited paper path requires {required!r}"
            )


def sechiba_main_full_irrigation_transition(
    *, veget_max, veget, vegstress, transpot, evapot, precip_rain, soil_deficit,
    irrig_frac, ok_laidev, irrig_threshold, irrig_fulfill, irrig_dosmax,
    irrig_drip: bool, nflow: int,
):
    """Compute ``irrigation`` from ``sechiba_main`` lines 1264-1301.

    Fortran provenance: ``src_sechiba/sechiba.f90::sechiba_main`` lines
    1264-1301. Crop rotation and all later tail transitions are deliberately
    outside this owner.
    """
    maximum = jnp.asarray(veget_max)
    cover = jnp.asarray(veget)
    stress = jnp.asarray(vegstress)
    potential = jnp.asarray(transpot)
    evaporation = jnp.asarray(evapot)
    deficit = jnp.asarray(soil_deficit)
    rain = jnp.asarray(precip_rain)
    fraction = jnp.asarray(irrig_frac)
    development = jnp.asarray(ok_laidev, dtype=jnp.bool_)
    threshold = jnp.asarray(irrig_threshold)
    fulfill = jnp.asarray(irrig_fulfill)
    if maximum.ndim != 2 or any(value.shape != maximum.shape for value in (cover, stress, potential, deficit)):
        raise ValueError("vegetation, stress, transpiration, and deficit fields must share (land,pft)")
    if evaporation.shape != maximum.shape[:1] or rain.shape != maximum.shape[:1] or fraction.shape != maximum.shape[:1]:
        raise ValueError("evapot, precip_rain, and irrig_frac must have land shape")
    if any(value.shape != maximum.shape[1:] for value in (development, threshold, fulfill)):
        raise ValueError("irrigation PFT parameters must have PFT shape")
    demand = jnp.zeros(maximum.shape[:1], dtype=maximum.dtype)
    for pft in range(1, maximum.shape[1]):
        present = maximum[:, pft] > 0
        relative_cover = jnp.where(present, cover[:, pft] / maximum[:, pft], 0.0)
        relative_cover = jnp.maximum(relative_cover, 0.0)
        eligible = present & development[pft] & (stress[:, pft] < threshold[pft])
        if irrig_drip:
            raw = potential[:, pft] * relative_cover + evaporation * (1.0 - relative_cover) - rain
            component = fraction * jnp.minimum(irrig_dosmax, fulfill[pft] * jnp.maximum(0.0, raw)) * maximum[:, pft]
        else:
            component = fraction * jnp.minimum(irrig_dosmax, jnp.maximum(0.0, deficit[:, pft])) * maximum[:, pft]
        demand = demand + jnp.where(eligible, component, 0.0)
    demand = jnp.maximum(demand, 0.0)
    irrigation = jnp.zeros((maximum.shape[0], nflow), dtype=maximum.dtype)
    return irrigation.at[:, 0].set(demand)


def sechiba_get_cmd(cmdin: int, *, ok_laidev, nvm: int | None = None) -> tuple[int, int, float]:
    """Decode one rotation command exactly as ``sechiba_get_cmd``.

    Source and target indices remain Fortran one-based in the return value.
    Fortran provenance: ``src_sechiba/sechiba.f90::sechiba_get_cmd`` lines
    3144-3190.
    """

    command = int(cmdin)
    development = np.asarray(ok_laidev, dtype=bool)
    pft_count = int(development.size if nvm is None else nvm)
    if development.shape != (pft_count,):
        raise ValueError("ok_laidev must have shape (nvm,)")
    if command > 1_010_000 or command < 0:
        raise RuntimeError("cmd error in sechiba_get_cmd")
    if command == 0:
        return 0, 0, 0.0

    target = command % 100
    source = (command // 100) % 100
    percent = float(command // 10_000) / 100.0
    if 1.0 < percent < 1.01:
        percent = 1.0
    if percent > 1.0:
        raise RuntimeError("incorrect percent rotation, sechiba_get_cmd")
    if not 1 <= target <= pft_count or not (development[target - 1] or target == 1):
        raise RuntimeError("incorrect rotation target, sechiba_get_cmd")
    if not 1 <= source <= pft_count or not (development[source - 1] or source == 1):
        raise RuntimeError("incorrect rotation source, sechiba_get_cmd")
    return source, target, percent


def _rotate_pft_field(
    field: np.ndarray,
    *,
    matrix: np.ndarray,
    old_veget_max: np.ndarray,
    new_veget_max: np.ndarray,
    min_sechiba: float,
) -> np.ndarray:
    """Apply the source loops at ``sechiba_main`` lines 1363-1390."""

    old_field = np.asarray(field, dtype=np.float64).copy()
    rotated = old_field.copy()
    for target in range(matrix.shape[1]):
        if np.sum(matrix[:, target]) > min_sechiba:
            value = (
                old_field[target]
                * old_veget_max[target]
                * (1.0 - np.sum(matrix[target, :]))
            )
            for source in range(matrix.shape[0]):
                if matrix[source, target] > min_sechiba:
                    value += (
                        old_veget_max[source]
                        * matrix[source, target]
                        * old_field[source]
                    )
            rotated[target] = value / new_veget_max[target]
    return rotated


def sechiba_main_crop_rotation_transition(
    *,
    state: Mapping[str, object],
    ok_laidev,
    pref_soil_veg,
    ext_coeff,
    dz,
    min_sechiba: float = 1.0e-8,
    check_cwrr: bool = False,
    allowed_err: float = 1.0e-8,
) -> SechibaRotationResult:
    """Run the integrated crop-rotation transition in source call order.

    The returned mapping includes every ``sechiba_main``, HYDROL, and
    THERMOSOIL state writeback made by lines 1309-1407. No rotation state is
    reconstructed: all module SAVE fields used by the Fortran callees are
    mandatory inputs.
    """

    required = (
        "veget_max", "veget", "lai", "totfrac_nobio", "soiltile",
        "f_rot_sech", "rot_cmd", "qsintveg", "mc", "water2infilt", "tmc",
        "humtot", "resdist", "ptn", "cgrnd", "dgrnd", "temp_sol_new_pft",
        "soilcap_pft", "soilflx_pft",
    )
    _require_fields(state, required, boundary="crop-rotation state")
    updated = dict(state)
    veget_max = np.asarray(state["veget_max"], dtype=np.float64).copy()
    veget = np.asarray(state["veget"], dtype=np.float64).copy()
    lai = np.asarray(state["lai"], dtype=np.float64)
    totfrac_nobio = np.asarray(state["totfrac_nobio"], dtype=np.float64)
    soiltile = np.asarray(state["soiltile"], dtype=np.float64).copy()
    f_rot_sech = np.asarray(state["f_rot_sech"], dtype=bool).copy()
    rot_cmd = np.asarray(state["rot_cmd"], dtype=np.int64).copy()
    qsintveg = np.asarray(state["qsintveg"], dtype=np.float64).copy()
    mc = np.asarray(state["mc"], dtype=np.float64).copy()
    water2infilt = np.asarray(state["water2infilt"], dtype=np.float64).copy()
    tmc = np.asarray(state["tmc"], dtype=np.float64).copy()
    humtot = np.asarray(state["humtot"], dtype=np.float64).copy()
    resdist = np.asarray(state["resdist"], dtype=np.float64).copy()
    ptn = np.asarray(state["ptn"], dtype=np.float64).copy()
    cgrnd = np.asarray(state["cgrnd"], dtype=np.float64).copy()
    dgrnd = np.asarray(state["dgrnd"], dtype=np.float64).copy()
    temp_sol_new_pft = np.asarray(state["temp_sol_new_pft"], dtype=np.float64).copy()
    soilcap_pft = np.asarray(state["soilcap_pft"], dtype=np.float64).copy()
    soilflx_pft = np.asarray(state["soilflx_pft"], dtype=np.float64).copy()

    if veget_max.ndim != 2:
        raise ValueError("veget_max must have shape (land,pft)")
    npts, nvm = veget_max.shape
    if any(value.shape != (npts, nvm) for value in (veget, lai, qsintveg)):
        raise ValueError("veget, lai, and qsintveg must share veget_max shape")
    if f_rot_sech.shape != (npts,) or rot_cmd.ndim != 2 or rot_cmd.shape[0] != npts:
        raise ValueError("f_rot_sech and rot_cmd must have land-point dimensions")
    if totfrac_nobio.shape != (npts,) or soiltile.ndim != 2 or soiltile.shape[0] != npts:
        raise ValueError("totfrac_nobio and soiltile must share the land dimension")
    nstm = soiltile.shape[1]
    pref = np.asarray(pref_soil_veg, dtype=np.int64)
    extinction = np.asarray(ext_coeff, dtype=np.float64)
    development = np.asarray(ok_laidev, dtype=bool)
    if pref.shape != (nvm,) or extinction.shape != (nvm,) or development.shape != (nvm,):
        raise ValueError("rotation PFT parameters must have shape (nvm,)")
    if np.any(pref < 1) or np.any(pref > nstm):
        raise ValueError("pref_soil_veg must contain Fortran indices in 1..nstm")
    if any(value.shape[:1] != (npts,) or value.shape[-1:] != (nvm,) for value in (ptn, cgrnd, dgrnd)):
        raise ValueError("ptn, cgrnd, and dgrnd must have land and trailing PFT dimensions")
    if any(value.shape != (npts, nvm) for value in (temp_sol_new_pft, soilcap_pft, soilflx_pft)):
        raise ValueError("SECHIBA thermal PFT fields must share veget_max shape")

    matrices: list[jnp.ndarray] = []
    for point in range(npts):
        if not f_rot_sech[point]:
            continue
        matrix = np.zeros((nvm, nvm), dtype=np.float64)
        for command in rot_cmd[point]:
            if int(command) <= 0:
                break
            source, target, percent = sechiba_get_cmd(
                int(command), ok_laidev=development, nvm=nvm
            )
            matrix[source - 1, target - 1] = percent
        if np.sum(matrix) <= 0.0:
            raise RuntimeError("sechiba_rotation: active flag has no positive rotation command")

        old_veget_max = veget_max[point].copy()
        new_veget_max = old_veget_max.copy()
        for source in range(nvm):
            for target in range(nvm):
                fraction = matrix[source, target]
                if fraction > 0.0 and fraction < 1.0 + min_sechiba:
                    amount = fraction * old_veget_max[source]
                    new_veget_max[source] -= amount
                    new_veget_max[target] += amount
        new_veget_max[new_veget_max < min_sechiba] = 0.0
        veget_max[point] = new_veget_max
        veget[point, 0] = new_veget_max[0]
        veget[point, 1:] = new_veget_max[1:] * (
            1.0 - np.exp(-lai[point, 1:] * extinction[1:])
        )
        soiltile[point] = 0.0
        soiltile[point, 0] = totfrac_nobio[point]
        for source in range(nvm):
            soiltile[point, pref[source] - 1] += new_veget_max[source]
        tile_sum = float(np.sum(soiltile[point]))
        if tile_sum < 1.0 - min_sechiba or tile_sum > 1.0 + min_sechiba:
            raise RuntimeError("sechiba_rotation: sum of soiltile not equal to 1")

        hydrol = hydrol_rotation_update(
            ip_fortran=point + 1,
            rot_matrix=matrix,
            old_veget_max=old_veget_max,
            veget_max=veget_max,
            soiltile=soiltile,
            qsintveg=qsintveg,
            pref_soil_veg=pref,
            mc=mc,
            water2infilt=water2infilt,
            tmc=tmc,
            humtot=humtot,
            resdist=resdist,
            dz=dz,
            min_sechiba=min_sechiba,
            check_cwrr=check_cwrr,
            allowed_err=allowed_err,
        )
        mc, water2infilt, qsintveg = hydrol.mc, hydrol.water2infilt, hydrol.qsintveg
        tmc, humtot, resdist = hydrol.tmc, hydrol.humtot, hydrol.resdist

        thermal = thermosoil_rotation_update(
            matrix_rot=matrix,
            old_veget_max=old_veget_max,
            ptn=ptn[point],
            cgrnd=cgrnd[point],
            dgrnd=dgrnd[point],
            min_sechiba=min_sechiba,
        )
        ptn[point], cgrnd[point], dgrnd[point] = tuple(np.asarray(value) for value in thermal)
        temp_sol_new_pft[point] = _rotate_pft_field(
            temp_sol_new_pft[point], matrix=matrix, old_veget_max=old_veget_max,
            new_veget_max=new_veget_max, min_sechiba=min_sechiba,
        )
        soilcap_pft[point] = _rotate_pft_field(
            soilcap_pft[point], matrix=matrix, old_veget_max=old_veget_max,
            new_veget_max=new_veget_max, min_sechiba=min_sechiba,
        )
        soilflx_pft[point] = _rotate_pft_field(
            soilflx_pft[point], matrix=matrix, old_veget_max=old_veget_max,
            new_veget_max=new_veget_max, min_sechiba=min_sechiba,
        )
        f_rot_sech[point] = False
        rot_cmd[point, :] = 0
        matrices.append(jnp.asarray(matrix))

    for name, value in (
        ("veget_max", veget_max), ("veget", veget), ("soiltile", soiltile),
        ("f_rot_sech", f_rot_sech), ("rot_cmd", rot_cmd), ("qsintveg", qsintveg),
        ("mc", mc), ("water2infilt", water2infilt), ("tmc", tmc),
        ("humtot", humtot), ("resdist", resdist), ("ptn", ptn),
        ("cgrnd", cgrnd), ("dgrnd", dgrnd),
        ("temp_sol_new_pft", temp_sol_new_pft), ("soilcap_pft", soilcap_pft),
        ("soilflx_pft", soilflx_pft),
    ):
        updated[name] = jnp.asarray(value)
    return SechibaRotationResult(updated, tuple(matrices))


def sechiba_main_lcc_transition(
    *, state: Mapping[str, object], lcc_inputs: Mapping[str, object]
) -> Mapping[str, object]:
    """Apply ``slowproc_change_frac`` and retain every source writeback.

    Fortran provenance: ``sechiba_main`` lines 1768-1774,
    ``slowproc_change_frac`` lines 5967-6026, and ``slowproc_veget`` lines
    2820-2925. The five carbon fields are explicit identity writebacks at
    ``slowproc_veget`` lines 2883-2887 in this source revision.
    """

    _require_fields(
        state,
        ("lai", "biomass", "litter_above", "litter_below", "carbon_32l", "DOC"),
        boundary="land-cover-change state",
    )
    required = {
        "agri_peat", "veget_max_new", "frac_nobio_new", "pref_soil_veg",
        "ext_coeff_vegetfrac", "nstm",
    }
    missing = sorted(required - set(lcc_inputs))
    if missing:
        raise ValueError(f"lcc_inputs missing source fields: {missing}")
    result = slowproc_change_frac_source_routed(
        lai=state["lai"],
        **dict(lcc_inputs),
    )
    updated = dict(state)
    updated.update(result.vegetation._asdict())
    updated["tot_bare_soil"] = result.tot_bare_soil
    for name in ("biomass", "litter_above", "litter_below", "carbon_32l", "DOC"):
        updated[name] = jnp.asarray(state[name])
    return updated


def sechiba_main_post_output_transition(
    *,
    state: Mapping[str, object],
    done_stomate_lcchange: bool,
    dyn_peat: bool,
    use_age_class: bool,
    ldrestart_write: bool,
    lcc_inputs: Mapping[str, object] | None = None,
    finalize_owner: Callable[[Mapping[str, object]], object] | None = None,
) -> SechibaPostOutputResult:
    """Execute lines 1768-1784 after all model-output calls have completed."""

    updated = dict(state)
    order: list[str] = []
    if done_stomate_lcchange and not dyn_peat:
        if use_age_class:
            order.append("slowproc_change_frac[age-class-no-call]")
        else:
            if lcc_inputs is None:
                raise ValueError(
                    "active land-cover change requires explicit lcc_inputs for sechiba_main lines 1768-1773"
                )
            updated = dict(sechiba_main_lcc_transition(state=updated, lcc_inputs=lcc_inputs))
            order.append("slowproc_change_frac[source-routed]")
    else:
        order.append("slowproc_change_frac[fixed-off]")
    updated["done_stomate_lcchange"] = False

    restart_result = None
    if ldrestart_write:
        if finalize_owner is None:
            raise ValueError("LDRESTART_WRITE requires an explicit sechiba_finalize owner")
        restart_result = finalize_owner(updated)
        finalized_state = getattr(restart_result, "state", None)
        if finalized_state is None and isinstance(restart_result, Mapping):
            finalized_state = restart_result
        if not isinstance(finalized_state, Mapping):
            raise TypeError("sechiba_finalize owner must return a state mapping or an object with .state")
        updated = dict(finalized_state)
        order.append("sechiba_finalize[source-routed]")
    else:
        order.append("sechiba_finalize[fixed-off]")
    return SechibaPostOutputResult(updated, tuple(order), restart_result)


def _output_field(name: str, value: object, line: int) -> SechibaOutputField:
    return SechibaOutputField(name=name, value=jnp.asarray(value), source_line=line)


def build_sechiba_main_output_packet_pft14(
    *,
    state: Mapping[str, object],
    dt_sechiba: object,
    is_tree: object,
    natural: object,
    switches: SechibaMainPFT14FixedSwitches = SechibaMainPFT14FixedSwitches(),
) -> SechibaMainOutputPacket:
    """Build the exact fixed paper-path output calls from lines 1420-1758.

    This is a pure model boundary: values, scales, masks, names, and call order
    are retained, while XIOS/IOIPSL handles and file operations are omitted.
    """

    validate_sechiba_main_pft14_switches(switches)
    required = (
        "temp_sol_new", "fluxsens", "fluxlat", "vevapnu", "snow", "snow_age",
        "snow_nobio", "snow_nobio_age", "frac_snow_nobio", "totfrac_nobio",
        "frac_snow_veg", "reinf_slope", "njsc", "veget", "veget_max",
        "frac_nobio", "soiltile", "rstruct", "gpp", "co2_flux", "drysoil_frac",
        "vevapflo", "k_litt", "vbeta", "vbeta_pft", "vbeta1", "vbeta2",
        "vbeta3", "vbeta4", "vbeta4_pft", "vbeta5", "gsmean", "cimean",
        "rveget", "rsol", "vevapwet", "vevapsno", "transpir", "contfrac",
        "tsol_rad", "qsurf", "emis", "z0m", "z0h", "roughheight",
        "roughheight_pft", "lai", "vevapp", "fusion", "irrigation",
        "tq_cdrag", "tq_cdrag_pft", "soilflx", "soilcap", "soilflx_pft",
        "soilcap_pft", "temp_sol_pft", "vevapnu_pft", "grndflux", "snowtemp",
        "snowliq", "snowdz", "snowrho", "snowgrain", "snowheat", "pgflux",
    )
    _require_fields(state, required, boundary="sechiba_main output")

    veget_max = jnp.asarray(state["veget_max"])
    if veget_max.ndim != 2 or veget_max.shape[0] < 1 or veget_max.shape[1] != switches.nvm:
        raise ValueError("veget_max must have shape (positive landpoints, 14)")
    npts = veget_max.shape[0]
    tree = jnp.asarray(is_tree, dtype=bool)
    natural_arr = jnp.asarray(natural, dtype=bool)
    if tree.shape != (switches.nvm,) or natural_arr.shape != (switches.nvm,):
        raise ValueError("is_tree and natural must each have shape (14,)")

    def land_pft(name: str) -> jnp.ndarray:
        value = jnp.asarray(state[name])
        if value.shape != (npts, switches.nvm):
            raise ValueError(f"{name} must have shape ({npts}, 14)")
        return value

    for name in ("veget", "gpp", "co2_flux", "vbeta_pft", "vbeta2", "vbeta3", "vbeta4_pft", "gsmean", "cimean", "rveget", "rstruct", "vevapwet", "transpir", "roughheight_pft", "lai", "tq_cdrag_pft", "soilflx_pft", "soilcap_pft", "temp_sol_pft", "vevapnu_pft"):
        land_pft(name)

    # sechiba_main 1423-1431 uses a source-ordered PFT loop, not a reduction.
    sum_treefrac = jnp.zeros(npts, dtype=veget_max.dtype)
    sum_grassfrac = jnp.zeros(npts, dtype=veget_max.dtype)
    sum_cropfrac = jnp.zeros(npts, dtype=veget_max.dtype)
    for jv in range(1, switches.nvm):
        if bool(tree[jv]) and bool(natural_arr[jv]):
            sum_treefrac = sum_treefrac + veget_max[:, jv]
        elif not bool(tree[jv]) and bool(natural_arr[jv]):
            sum_grassfrac = sum_grassfrac + veget_max[:, jv]
        else:
            sum_cropfrac = sum_cropfrac + veget_max[:, jv]
    wet_sum = jnp.sum(jnp.asarray(state["vevapwet"]), axis=1)
    transpir_sum = jnp.sum(jnp.asarray(state["transpir"]), axis=1)
    soil_evap_sum = jnp.asarray(state["vevapnu"]) + jnp.asarray(state["vevapsno"])
    contfrac = jnp.asarray(state["contfrac"])
    frac_nobio = jnp.asarray(state["frac_nobio"])
    if frac_nobio.ndim != 2 or frac_nobio.shape[0] != npts:
        raise ValueError(f"frac_nobio must have shape ({npts}, nnobio)")
    frac_snow_nobio = jnp.asarray(state["frac_snow_nobio"])
    if frac_snow_nobio.ndim != 2 or frac_snow_nobio.shape[0] != npts:
        raise ValueError(f"frac_snow_nobio must have shape ({npts}, nnobio)")
    denom = jnp.sum(veget_max, axis=1)
    # Lines 1494-1501 divide and accumulate inside the PFT loop.
    lai_mean = jnp.zeros(npts, dtype=veget_max.dtype)
    lai = jnp.asarray(state["lai"])
    for jv in range(1, switches.nvm):
        term = jnp.where(denom > 0, veget_max[:, jv] * lai[:, jv] / denom, 0)
        lai_mean = lai_mean + term
    frac_snow = (
        jnp.sum(frac_snow_nobio, axis=1) * jnp.asarray(state["totfrac_nobio"])
        + jnp.asarray(state["frac_snow_veg"]) * (1 - jnp.asarray(state["totfrac_nobio"]))
    )
    tree_frac = sum_treefrac * 100 * contfrac
    grass_frac = sum_grassfrac * 100 * contfrac
    crop_frac = sum_cropfrac * 100 * contfrac
    baresoil_frac = veget_max[:, 0] * 100 * contfrac
    residual_frac = jnp.sum(frac_nobio, axis=1) * 100 * contfrac
    real_dtype = veget_max.dtype
    njsc_real = jnp.asarray(state["njsc"], dtype=real_dtype)
    irrigation = jnp.asarray(state["irrigation"])
    if irrigation.ndim != 2 or irrigation.shape[0] != npts or irrigation.shape[1] < 1:
        raise ValueError(f"irrigation must have shape ({npts}, at least 1)")
    irrigation_water = irrigation[:, 0]
    dt = jnp.asarray(dt_sechiba)
    day_scale = jnp.asarray(86400.0) / dt

    xios: list[SechibaOutputField] = []
    add_xios = lambda name, value, line: xios.append(_output_field(name, value, line))
    add_xios("temp_sol_new", state["temp_sol_new"], 1434)
    add_xios("fluxsens", state["fluxsens"], 1435)
    add_xios("fluxlat", state["fluxlat"], 1436)
    add_xios("evapnu", jnp.asarray(state["vevapnu"]) * day_scale, 1437)
    add_xios("snow", state["snow"], 1438)
    add_xios("snowage", state["snow_age"], 1439)
    add_xios("snownobio", state["snow_nobio"], 1440)
    add_xios("snownobioage", state["snow_nobio_age"], 1441)
    add_xios("frac_snow", frac_snow, 1442)
    add_xios("frac_snow_veg", state["frac_snow_veg"], 1443)
    add_xios("frac_snow_nobio", state["frac_snow_nobio"], 1444)
    add_xios("reinf_slope", state["reinf_slope"], 1445)
    add_xios("njsc", njsc_real, 1446)
    add_xios("vegetfrac", state["veget"], 1447)
    add_xios("maxvegetfrac", state["veget_max"], 1448)
    add_xios("nobiofrac", state["frac_nobio"], 1449)
    add_xios("soiltile", state["soiltile"], 1450)
    add_xios("rstruct", state["rstruct"], 1451)
    add_xios("gpp", jnp.asarray(state["gpp"]) / dt, 1452)
    add_xios("nee", jnp.asarray(state["co2_flux"]) / dt, 1453)
    for name, source, line in (
        ("drysoil_frac", "drysoil_frac", 1454), ("evapflo", "vevapflo", 1455),
        ("evapflo_alma", "vevapflo", 1456), ("k_litt", "k_litt", 1457),
        ("beta", "vbeta", 1458), ("vbeta1", "vbeta1", 1459),
        ("vbeta2", "vbeta2", 1460), ("vbeta3", "vbeta3", 1461),
        ("vbeta4", "vbeta4", 1462), ("vbeta5", "vbeta5", 1463),
        ("gsmean", "gsmean", 1464), ("cimean", "cimean", 1465),
        ("rveget", "rveget", 1466), ("rsol", "rsol", 1467),
    ):
        value = state[source]
        if name == "evapflo":
            value = jnp.asarray(value) * day_scale
        elif name == "evapflo_alma":
            value = jnp.asarray(value) / dt
        add_xios(name, value, line)
    for name, value, line in (
        ("evspsblveg", wet_sum / dt, 1470), ("evspsblsoi", soil_evap_sum / dt, 1472),
        ("tran", transpir_sum / dt, 1474), ("treeFrac", tree_frac, 1476),
        ("grassFrac", grass_frac, 1478), ("cropFrac", crop_frac, 1480),
        ("baresoilFrac", baresoil_frac, 1482), ("residualFrac", residual_frac, 1484),
        ("tsol_rad", jnp.asarray(state["tsol_rad"]) - 273.15, 1486),
    ):
        add_xios(name, value, line)
    for name, source, line in (
        ("qsurf", "qsurf", 1487), ("emis", "emis", 1488), ("z0m", "z0m", 1489),
        ("z0h", "z0h", 1490), ("roughheight", "roughheight", 1491),
        ("roughheight_pft", "roughheight_pft", 1492), ("lai", "lai", 1493),
    ):
        add_xios(name, state[source], line)
    add_xios("LAImean", lai_mean, 1503)
    for name, value, line in (
        ("vevapsno", jnp.asarray(state["vevapsno"]) / dt, 1504),
        ("vevapp", jnp.asarray(state["vevapp"]) / dt, 1505),
        ("vevapnu", jnp.asarray(state["vevapnu"]) * day_scale, 1506),
        ("vevapnu_alma", jnp.asarray(state["vevapnu"]) / dt, 1507),
        ("transpir", jnp.asarray(state["transpir"]) * day_scale, 1508),
        ("inter", jnp.asarray(state["vevapwet"]) * day_scale, 1509),
        ("Qf", state["fusion"], 1510),
        ("irrigation", irrigation_water * day_scale, 1512),
        ("ECanop", wet_sum / dt, 1518), ("TVeg", transpir_sum / dt, 1523),
        ("ACond", state["tq_cdrag"], 1524), ("ACond_pft", state["tq_cdrag_pft"], 1525),
    ):
        add_xios(name, value, line)

    history: list[SechibaOutputField] = []
    add_hist = lambda name, value, line: history.append(_output_field(name, value, line))
    for name, source, line in (
        ("beta", "vbeta", 1529), ("beta_pft", "vbeta_pft", 1530),
        ("z0m", "z0m", 1531), ("z0h", "z0h", 1532), ("soilflx", "soilflx", 1533),
        ("soilcap", "soilcap", 1534), ("soilflx_pft", "soilflx_pft", 1535),
        ("soilcap_pft", "soilcap_pft", 1536), ("roughheight", "roughheight", 1537),
        ("roughheight_pft", "roughheight_pft", 1538), ("temp_sol_pft", "temp_sol_pft", 1539),
        ("vegetfrac", "veget", 1540), ("maxvegetfrac", "veget_max", 1541),
        ("nobiofrac", "frac_nobio", 1542), ("lai", "lai", 1543),
        ("subli", "vevapsno", 1544), ("evapnu", "vevapnu", 1545),
        ("evapnu_pft", "vevapnu_pft", 1546), ("transpir", "transpir", 1547),
        ("inter", "vevapwet", 1548), ("vbeta1", "vbeta1", 1549),
        ("vbeta2", "vbeta2", 1550), ("vbeta3", "vbeta3", 1551),
        ("vbeta4", "vbeta4", 1552), ("vbeta4_pft", "vbeta4_pft", 1553),
        ("vbeta5", "vbeta5", 1554), ("drysoil_frac", "drysoil_frac", 1555),
        ("rveget", "rveget", 1556), ("rstruct", "rstruct", 1557),
        ("snow", "snow", 1562), ("snowage", "snow_age", 1563),
        ("snownobio", "snow_nobio", 1564), ("snownobioage", "snow_nobio_age", 1565),
    ):
        add_hist(name, state[source], line)
    add_hist("irrigation", irrigation_water, 1569)
    for name, source, line in (
        ("grndflux", "grndflux", 1573), ("snowtemp", "snowtemp", 1574),
        ("snowliq", "snowliq", 1575), ("snowdz", "snowdz", 1576),
        ("snowrho", "snowrho", 1577), ("snowgrain", "snowgrain", 1578),
        ("snowheat", "snowheat", 1579), ("pgflux", "pgflux", 1582),
        ("soiltile", "soiltile", 1583),
    ):
        add_hist(name, state[source], line)
    add_hist("soilindex", njsc_real, 1586)
    add_hist("reinf_slope", state["reinf_slope"], 1587)
    add_hist("k_litt", state["k_litt"], 1588)
    add_hist("gsmean", state["gsmean"], 1595)
    add_hist("gpp", state["gpp"], 1596)
    add_hist("cimean", state["cimean"], 1597)
    add_hist("nee", state["co2_flux"], 1600)
    for name, value, line in (
        ("evspsblveg", wet_sum, 1604), ("evspsblsoi", soil_evap_sum, 1607),
        ("tran", transpir_sum, 1610), ("treeFrac", tree_frac, 1613),
        ("grassFrac", grass_frac, 1616), ("cropFrac", crop_frac, 1619),
        ("baresoilFrac", baresoil_frac, 1622), ("residualFrac", residual_frac, 1625),
    ):
        add_hist(name, value, line)

    diagnostics = {
        "sum_treefrac": sum_treefrac,
        "sum_grassfrac": sum_grassfrac,
        "sum_cropfrac": sum_cropfrac,
        "evspsblveg": wet_sum,
        "evspsblsoi": soil_evap_sum,
        "tran": transpir_sum,
        "LAImean": lai_mean,
    }
    return SechibaMainOutputPacket(
        xios=tuple(xios),
        history=tuple(history),
        history2=(),
        diagnostics=diagnostics,
        history2_active=False,
    )


def _require_fields(state: Mapping[str, object], names: tuple[str, ...], *, boundary: str) -> None:
    missing = tuple(name for name in names if name not in state)
    if missing:
        raise ValueError(f"missing {boundary} fields: {missing}")


def _write_step_outputs(
    state: Mapping[str, object],
    step: SechibaExplicitCoupledStepResult,
) -> dict[str, object]:
    """Apply owner outputs in the same order as their Fortran calls."""

    updated = dict(state)
    updated.update(step.diffuco_payload)
    updated.update(step.enerbil_payload)
    updated.update(step.hydrol_outputs._asdict())
    updated.update(step.hydrol_diagnostics._asdict())
    updated.update(step.condveg._asdict())
    updated.update(step.thermosoil_payload)

    if step.slowproc is not None:
        surface = step.slowproc.surface if hasattr(step.slowproc, "surface") else step.slowproc
        vegetation = surface.vegetation
        updated.update(vegetation._asdict())
        updated["tot_bare_soil"] = surface.tot_bare_soil
    return updated


def sechiba_main_tail_pft14_fixed(
    *,
    state: Mapping[str, object],
    dt_sechiba: object,
    is_tree: object,
    natural: object,
    switches: SechibaMainPFT14FixedSwitches = SechibaMainPFT14FixedSwitches(),
    full_irrigation_inputs: Mapping[str, object] | None = None,
    rotation_inputs: Mapping[str, object] | None = None,
    lcc_inputs: Mapping[str, object] | None = None,
    finalize_owner: Callable[[Mapping[str, object]], object] | None = None,
    xios_boundary: OutputBoundary | None = None,
    history_boundary: OutputBoundary | None = None,
) -> SechibaMainTailResult:
    """Close ``sechiba_main`` lines 1219-1780 for the paper fixed switches.

    XIOS and history are deliberately caller-owned serialization boundaries.
    They receive source-ordered immutable field tuples, never scientific state.
    """

    validate_sechiba_main_pft14_switches(switches)
    _require_fields(state, _ROUTING_ZERO_FIELDS, boundary="routing-state")
    _require_fields(
        state,
        (
            "co2_flux",
            "veget_max",
            "temp_sol_new",
            "temp_sol_new_pft",
            "z0m",
            "z0h",
            "emis",
            "qsurf",
        ),
        boundary="post-process state",
    )

    updated = dict(state)
    order = ["erosion_main[fixed-off]"]

    # sechiba_main 1227-1249: RIVER_ROUTING=y but nbp_glo=1 enters ELSE.
    for name in _ROUTING_ZERO_FIELDS:
        updated[name] = jnp.zeros_like(jnp.asarray(updated[name]))
    order.append("routing_main[single-point-no-call]")

    # Lines 1251-1262 still test RIVER_ROUTING alone, not the routing call gate.
    routing_writeback = routing_to_slowproc_writeback(
        reinfiltration=updated["reinfiltration"],
        irrigation=updated["irrigation"],
        returnflow=updated["returnflow"],
        sed_deposition=updated["sed_deposition"],
        poc_deposition=updated["poc_deposition"],
        dt_sechiba=dt_sechiba,
    )
    updated["DOC_to_topsoil"] = routing_writeback.doc_to_topsoil
    updated["DOC_to_subsoil"] = routing_writeback.doc_to_subsoil
    updated["sed_deposition_d"] = routing_writeback.sed_deposition_d
    updated["poc_deposition_d"] = routing_writeback.poc_deposition_d
    order.append("routing_to_slowproc_writeback")

    # Lines 1264-1301: explicit packet owns the full-irrigation state transition.
    if switches.do_fullirr:
        if full_irrigation_inputs is None:
            raise ValueError("DO_FULLIRR requires explicit full_irrigation_inputs for sechiba_main lines 1264-1301")
        required_irrigation = {
            "vegstress", "transpot", "evapot", "precip_rain", "soil_deficit",
            "irrig_frac", "ok_laidev", "irrig_threshold", "irrig_fulfill",
            "irrig_dosmax", "irrig_drip",
        }
        missing = sorted(required_irrigation - set(full_irrigation_inputs))
        if missing:
            raise ValueError(f"full_irrigation_inputs missing source fields: {missing}")
        updated["irrigation"] = sechiba_main_full_irrigation_transition(
            **{name: full_irrigation_inputs[name] for name in required_irrigation},
            veget_max=updated["veget_max"],
            veget=updated["veget"],
            nflow=jnp.asarray(updated["irrigation"]).shape[1],
        )
        order.append("full_irrigation[source-routed]")
    else:
        updated["irrigation"] = jnp.zeros_like(updated["irrigation"])
        order.append("full_irrigation[fixed-off]")

    co2_flux = jnp.asarray(updated["co2_flux"])
    veget_max = jnp.asarray(updated["veget_max"])
    if co2_flux.ndim != 2 or veget_max.shape != co2_flux.shape:
        raise ValueError("co2_flux and veget_max must have the same (land, pft) shape")
    if veget_max.shape[1] != switches.nvm:
        raise ValueError(f"PFT14 path requires 14 PFT columns, got {veget_max.shape[1]}")
    updated["netco2flux"] = jnp.sum(co2_flux[:, 1:] * veget_max[:, 1:], axis=1)
    order.append("netco2flux")

    # Lines 1309-1407: module SAVE state is supplied explicitly and every
    # HYDROL/THERMOSOIL/SECHIBA writeback is merged before sechiba_end.
    if switches.ok_rotate:
        if rotation_inputs is None:
            raise ValueError("OK_ROTATE requires explicit rotation_inputs for sechiba_main lines 1309-1407")
        required_rotation = {
            "ok_laidev", "pref_soil_veg", "ext_coeff", "dz",
        }
        missing = sorted(required_rotation - set(rotation_inputs))
        if missing:
            raise ValueError(f"rotation_inputs missing source fields: {missing}")
        rotation = sechiba_main_crop_rotation_transition(
            state=updated,
            **dict(rotation_inputs),
        )
        updated = dict(rotation.state)
        order.append("crop_rotation[source-routed]")
    else:
        order.append("crop_rotation[fixed-off]")

    # sechiba_end, lines 3114-3140, is an exact state swap.
    updated["temp_sol"] = jnp.asarray(updated["temp_sol_new"])
    updated["temp_sol_pft"] = jnp.asarray(updated["temp_sol_new_pft"])
    order.append("sechiba_end")

    for source, target in (
        ("z0m", "z0m_out"),
        ("z0h", "z0h_out"),
        ("emis", "emis_out"),
        ("qsurf", "qsurf_out"),
    ):
        updated[target] = jnp.asarray(updated[source])
    order.append("internal_output_writeback")

    modelout = build_sechiba_main_output_packet_pft14(
        state=updated,
        dt_sechiba=dt_sechiba,
        is_tree=is_tree,
        natural=natural,
        switches=switches,
    )
    order.append("modelout_packet")

    xios_result = xios_boundary(modelout.xios) if xios_boundary is not None else None
    order.append("xios_boundary" if xios_boundary is not None else "xios_boundary[external]")
    history_result = history_boundary(modelout.history) if history_boundary is not None else None
    order.append("history_boundary" if history_boundary is not None else "history_boundary[external]")

    post_output = sechiba_main_post_output_transition(
        state=updated,
        done_stomate_lcchange=switches.done_stomate_lcchange,
        dyn_peat=switches.dyn_peat,
        use_age_class=switches.use_age_class,
        ldrestart_write=switches.ldrestart_write,
        lcc_inputs=lcc_inputs,
        finalize_owner=finalize_owner,
    )
    updated = dict(post_output.state)
    order.extend(post_output.process_order)
    return SechibaMainTailResult(
        updated, xios_result, history_result, modelout, tuple(order), post_output.restart_result
    )


def sechiba_main_pft14_fixed(
    *,
    step_inputs: Mapping[str, object],
    state: Mapping[str, object],
    dt_sechiba: object,
    is_tree: object,
    natural: object,
    switches: SechibaMainPFT14FixedSwitches = SechibaMainPFT14FixedSwitches(),
    full_irrigation_inputs: Mapping[str, object] | None = None,
    rotation_inputs: Mapping[str, object] | None = None,
    lcc_inputs: Mapping[str, object] | None = None,
    finalize_owner: Callable[[Mapping[str, object]], object] | None = None,
    xios_boundary: OutputBoundary | None = None,
    history_boundary: OutputBoundary | None = None,
    step_owner: StepOwner = sechiba_explicit_coupled_step,
) -> SechibaMainPFT14Result:
    """Run existing SECHIBA owners, then the lines 1219-1780 fixed tail."""

    validate_sechiba_main_pft14_switches(switches)
    step = step_owner(**dict(step_inputs))
    post_step_state = _write_step_outputs(state, step)
    tail = sechiba_main_tail_pft14_fixed(
        state=post_step_state,
        dt_sechiba=dt_sechiba,
        is_tree=is_tree,
        natural=natural,
        switches=switches,
        full_irrigation_inputs=full_irrigation_inputs,
        rotation_inputs=rotation_inputs,
        lcc_inputs=lcc_inputs,
        finalize_owner=finalize_owner,
        xios_boundary=xios_boundary,
        history_boundary=history_boundary,
    )
    return SechibaMainPFT14Result(step=step, tail=tail)


# The short name is the public fixed-path entry point; no generic branch claim.
sechiba_main = sechiba_main_pft14_fixed
