"""Thin explicit-input SECHIBA orchestration for implemented PFT14 kernels.

This is not a complete ``sechiba_main`` port. It preserves the Fortran process
order for the currently implemented local kernels and requires audited payloads
for upstream pieces that are not yet locally computed.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple

from jax_orchidee.sechiba.condveg import condveg_main_minimal
from jax_orchidee.sechiba.coupling import (
    assemble_enerbil_precall_payload,
    assemble_thermosoil_explicit_kwargs,
    enerbil_downstream_payload,
    thermosoil_downstream_payload,
)
from jax_orchidee.sechiba.enerbil import enerbil_explicit_local_step, enerbil_fusion_step
from jax_orchidee.sechiba.diffuco import (
    DiffucoPFT14AfterMainPayload,
    DiffucoPFT14LocalEnerbilPrecallResult,
    diffuco_pft14_local_enerbil_precall_explicit,
)
from jax_orchidee.sechiba.hydrol import (
    hydrol_module_diagnostics,
    hydrol_module_explicit_step,
    hydrol_module_outputs,
    hydrol_to_thermosoil_moisture_inputs,
)
from jax_orchidee.sechiba.slowproc import (
    slowproc_surface_transition_explicit,
    slowproc_surface_update_explicit,
)
from jax_orchidee.sechiba.thermosoil import thermosoil_explicit_step


SECHIBA_EXPLICIT_STEP_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 997-1216",
    "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_main lines 485-551",
    "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_main lines 1206-1296",
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1077-1082",
    "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_fusion lines 1881-1953",
    "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_main lines 398-435",
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_main lines 893-1032",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 1103-1121",
)

ENERBIL_LOCAL_STEP_FIELDS = (
    "temp_sol",
    "temp_sol_pft",
    "lwdown",
    "swnet",
    "pb",
    "emis",
    "ok_laidev",
    "epot_air",
    "petAcoef",
    "petBcoef",
    "qair",
    "peqAcoef",
    "peqBcoef",
    "soilflx",
    "soilflx_pft",
    "rau",
    "u",
    "v",
    "q_cdrag",
    "q_cdrag_pft",
    "vbeta",
    "vbeta_pft",
    "valpha",
    "vbeta1",
    "vbeta2",
    "vbeta3",
    "vbeta3pot",
    "vbeta4",
    "vbeta4_pft",
    "vbeta5",
    "soilcap",
    "soilcap_pft",
    "veget_max",
    "dt_sechiba",
    "q_sol_pot",
    "temp_sol_pot",
    "precip_rain",
    "snowdz",
    "temp_air",
    "pgflux",
)


class SechibaExplicitCoupledStepResult(NamedTuple):
    """Outputs from the current explicit-input SECHIBA local chain."""

    diffuco_payload: Mapping[str, object]
    diffuco_local_fields: tuple[str, ...]
    diffuco_passthrough_fields: tuple[str, ...]
    enerbil: object
    enerbil_payload: Mapping[str, object]
    hydrol: object
    hydrol_outputs: object
    hydrol_diagnostics: object
    hydrol_thermosoil_moisture: object
    fusion: object | None
    condveg: object
    thermosoil: object
    thermosoil_payload: Mapping[str, object]
    slowproc: object | None
    provenance: tuple[str, ...] = SECHIBA_EXPLICIT_STEP_PROVENANCE
    notes: tuple[str, ...] = (
        "DIFFUCO may be supplied as either an audited after-diffuco/pre-ENERBIL payload or a source-backed PFT14 payload object.",
        "ENERBIL, HYDROL, CONDVEG, THERMOSOIL, and the optional closed SLOWPROC surface update are executed only from explicit caller-provided inputs and previous module outputs.",
        "stomate_main, routing_main, sechiba_end, restart, and modelout are not executed by this helper.",
    )


class SechibaExplicitLocalDiffucoStepResult(NamedTuple):
    """Outputs from SECHIBA when DIFFUCO is built from local explicit inputs."""

    diffuco: DiffucoPFT14LocalEnerbilPrecallResult
    step: SechibaExplicitCoupledStepResult
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 997-1216",
        "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_main lines 629-710",
        *SECHIBA_EXPLICIT_STEP_PROVENANCE[1:],
    )


def _normalize_diffuco_payload(
    diffuco_payload: Mapping[str, object] | DiffucoPFT14AfterMainPayload,
) -> tuple[dict[str, object], tuple[str, ...], tuple[str, ...]]:
    if isinstance(diffuco_payload, DiffucoPFT14AfterMainPayload):
        return (
            dict(diffuco_payload.payload),
            diffuco_payload.local_fields,
            diffuco_payload.passthrough_fields,
        )
    return dict(diffuco_payload), (), tuple(diffuco_payload.keys())


def sechiba_explicit_coupled_step(
    *,
    diffuco_payload: Mapping[str, object] | DiffucoPFT14AfterMainPayload,
    enerbil_inputs: Mapping[str, object],
    hydrol_inputs: Mapping[str, object],
    hydrol_diagnostic_inputs: Mapping[str, object],
    condveg_inputs: Mapping[str, object],
    thermosoil_inputs: Mapping[str, object],
    slowproc_inputs: Mapping[str, object] | None = None,
) -> SechibaExplicitCoupledStepResult:
    """Run the current local SECHIBA chain in Fortran process order.

    Fortran provenance: ``sechiba_main`` calls ``diffuco_main`` at lines
    997-1005, ``enerbil_main`` at lines 1013-1019, ``hydrol_main`` at lines
        1049-1072, conditionally calls ``enerbil_fusion`` at lines 1077-1082,
        then calls ``condveg_main`` at lines 1085-1091 and ``thermosoil_main``
        at lines 1109-1118. The SLOWPROC surface transition exposes its
        ``do_slow`` gate when that field is supplied. This helper starts after
        a caller supplies the audited DIFFUCO boundary payload.
    """

    diffuco_source, diffuco_local_fields, diffuco_passthrough_fields = _normalize_diffuco_payload(diffuco_payload)
    enerbil_boundary = assemble_enerbil_precall_payload(diffuco_source, enerbil_inputs)
    enerbil_source = {**diffuco_source, **enerbil_inputs}
    missing_enerbil_step = tuple(
        field for field in ENERBIL_LOCAL_STEP_FIELDS if field not in enerbil_source and field not in {"q_sol_pot", "temp_sol_pot"}
    )
    if missing_enerbil_step:
        raise ValueError(f"missing ENERBIL local-step fields: {missing_enerbil_step}")
    enerbil_kwargs = {field: enerbil_source[field] for field in ENERBIL_LOCAL_STEP_FIELDS if field in enerbil_source}
    enerbil = enerbil_explicit_local_step(**enerbil_kwargs)
    e_payload = enerbil_downstream_payload(enerbil)

    hydrol_kwargs = dict(hydrol_inputs)
    for name in (
        "vevapwet",
        "vevapnu",
        "vevapnu_pft",
        "vevapflo",
        "transpir",
    ):
        if name not in hydrol_kwargs:
            hydrol_kwargs[name] = e_payload[name]
    hydrol = hydrol_module_explicit_step(**hydrol_kwargs)
    hydrol_outputs = hydrol_module_outputs(hydrol, soiltile=hydrol_kwargs["soiltile"])
    hydrol_diag = hydrol_module_diagnostics(hydrol, **hydrol_diagnostic_inputs)
    moisture = hydrol_to_thermosoil_moisture_inputs(
        hydrol_diag,
        pref_soil_veg=hydrol_kwargs["pref_soil_veg"],
    )

    condveg_kwargs = dict(condveg_inputs)
    fusion = None
    if not bool(condveg_kwargs.get("ok_explicitsnow", True)):
        fusion_sources = {
            "tot_melt": hydrol_kwargs.get("tot_melt"),
            "soilcap": enerbil_source.get("soilcap"),
            "soilcap_pft": enerbil_source.get("soilcap_pft"),
            "snowdz": condveg_kwargs.get("snowdz"),
            "temp_sol_new": e_payload.get("temp_sol_new"),
            "temp_sol_new_pft": e_payload.get("temp_sol_new_pft"),
            "ok_laidev": enerbil_source.get("ok_laidev"),
            "dt_sechiba": enerbil_source.get("dt_sechiba"),
        }
        missing_fusion = tuple(name for name, value in fusion_sources.items() if value is None)
        if missing_fusion:
            raise ValueError(f"missing post-HYDROL ENERBIL fusion fields: {missing_fusion}")
        fusion = enerbil_fusion_step(
            **fusion_sources,
            ok_explicitsnow=False,
        )
        e_payload = {
            **e_payload,
            "temp_sol_new": fusion.temp_sol_new,
            "temp_sol_new_pft": fusion.temp_sol_new_pft,
            "fusion": fusion.fusion,
        }
    condveg = condveg_main_minimal(**condveg_kwargs)

    thermosoil_kwargs = dict(thermosoil_inputs)
    thermosoil_boundary = assemble_thermosoil_explicit_kwargs(
        moisture=moisture,
        temp_sol_new=e_payload["temp_sol_new"],
        temp_sol_new_pft=e_payload["temp_sol_new_pft"],
        snowrho=condveg_kwargs["snowrho"],
        snowtemp=thermosoil_kwargs.pop("snowtemp"),
        snowdz=condveg_kwargs["snowdz"],
        frac_snow_veg=condveg.frac_snow_veg,
        frac_snow_nobio=condveg.frac_snow_nobio,
        **thermosoil_kwargs,
    )
    thermosoil = thermosoil_explicit_step(**thermosoil_boundary.kwargs)
    t_payload = thermosoil_downstream_payload(thermosoil)

    slowproc = None
    if slowproc_inputs is not None:
        slowproc_kwargs = dict(slowproc_inputs)
        if "do_slow" in slowproc_kwargs:
            slowproc = slowproc_surface_transition_explicit(**slowproc_kwargs)
        else:
            slowproc = slowproc_surface_update_explicit(**slowproc_kwargs)

    return SechibaExplicitCoupledStepResult(
        diffuco_payload=diffuco_source,
        diffuco_local_fields=diffuco_local_fields,
        diffuco_passthrough_fields=diffuco_passthrough_fields,
        enerbil=enerbil,
        enerbil_payload=e_payload,
        hydrol=hydrol,
        hydrol_outputs=hydrol_outputs,
        hydrol_diagnostics=hydrol_diag,
        hydrol_thermosoil_moisture=moisture,
        fusion=fusion,
        condveg=condveg,
        thermosoil=thermosoil,
        thermosoil_payload=t_payload,
        slowproc=slowproc,
    )


def sechiba_explicit_coupled_step_from_local_diffuco(
    *,
    diffuco_inputs: Mapping[str, object],
    pft_output_backgrounds: Mapping[str, object],
    diffuco_passthrough: Mapping[str, object] | None = None,
    enerbil_inputs: Mapping[str, object],
    hydrol_inputs: Mapping[str, object],
    hydrol_diagnostic_inputs: Mapping[str, object],
    condveg_inputs: Mapping[str, object],
    thermosoil_inputs: Mapping[str, object],
    slowproc_inputs: Mapping[str, object] | None = None,
) -> SechibaExplicitLocalDiffucoStepResult:
    """Run SECHIBA local chain after computing the PFT14 DIFFUCO boundary.

    Fortran provenance: ``sechiba_main`` calls ``diffuco_main`` at lines
    997-1005 before ``enerbil_main`` at lines 1013-1019. This wrapper locally
    computes the currently implemented PFT14 DIFFUCO boundary from explicit
    inputs, then delegates the downstream ENERBIL/HYDROL/CONDVEG/THERMOSOIL
    order to ``sechiba_explicit_coupled_step``.
    """

    diffuco = diffuco_pft14_local_enerbil_precall_explicit(
        **dict(diffuco_inputs),
        pft_output_backgrounds=pft_output_backgrounds,
        passthrough=diffuco_passthrough,
    )
    step = sechiba_explicit_coupled_step(
        diffuco_payload=diffuco.payload,
        enerbil_inputs=enerbil_inputs,
        hydrol_inputs=hydrol_inputs,
        hydrol_diagnostic_inputs=hydrol_diagnostic_inputs,
        condveg_inputs=condveg_inputs,
        thermosoil_inputs=thermosoil_inputs,
        slowproc_inputs=slowproc_inputs,
    )
    return SechibaExplicitLocalDiffucoStepResult(diffuco=diffuco, step=step)
