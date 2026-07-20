"""First-step SECHIBA coverage aggregation for the paper reference case."""

from __future__ import annotations

from dataclasses import dataclass

from jax_orchidee.sechiba.diffuco import (
    DiffucoControlInundationAssembly,
    DiffucoControlInundationInputCoverage,
    DiffucoControlSalinityAssembly,
    DiffucoFirstStepPrecallAssembly,
    DiffucoPFT14FirstStepLocalEnerbilPrecallAssembly,
)
from jax_orchidee.sechiba.enerbil import EnerbilFirstStepInputCoverage
from jax_orchidee.sechiba.condveg import CondvegFirstStepModuleClosure
from jax_orchidee.sechiba.hydrol import HydrolFirstStepModuleClosure, HydrolFirstStepPrecallAssembly
from jax_orchidee.sechiba.thermosoil import ThermosoilFirstStepModuleClosure


SECHIBA_FIRST_STEP_INPUT_COVERAGE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 984-1216",
    "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_main lines 470-710",
    "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_main lines 390-598",
    "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90::hydrol_main lines 1206-1296",
    "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_main lines 398-435",
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_main lines 893-1032",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 973-1121",
)


@dataclass(frozen=True)
class SechibaFirstStepInputCoverage:
    """Module-level strict coverage for first-step ``sechiba_main``.

    This object is an audit contract, not an execution adapter. It records only
    exact sources already proven at the first-step boundary and keeps every
    same-step upstream-process gap explicit.
    """

    diffuco_control: DiffucoControlInundationInputCoverage
    diffuco_control_salinity: DiffucoControlSalinityAssembly | None
    diffuco_control_inundation: DiffucoControlInundationAssembly | None
    diffuco_precall: DiffucoFirstStepPrecallAssembly | None
    diffuco_local_enerbil_precall: DiffucoPFT14FirstStepLocalEnerbilPrecallAssembly | None
    enerbil: EnerbilFirstStepInputCoverage
    hydrol_precall: HydrolFirstStepPrecallAssembly | None
    hydrol_module: HydrolFirstStepModuleClosure | None
    condveg_module: CondvegFirstStepModuleClosure | None
    thermosoil_module: ThermosoilFirstStepModuleClosure | None
    hydrol_restart_anchors_available: bool
    condveg_initialize_available: bool
    thermosoil_restart_recurrence_available: bool
    slowproc_restart_state_available: bool
    closed_module_boundaries: tuple[str, ...]
    missing_by_module: dict[str, tuple[str, ...]]
    provenance: tuple[str, ...] = SECHIBA_FIRST_STEP_INPUT_COVERAGE_PROVENANCE
    notes: tuple[str, ...] = (
        "This coverage target starts at the first reference timestep after restart initialization.",
        "HYDROL/THERMOSOIL restart anchors cover initial recurrence state only; same-step diagnostics still require local process execution.",
        "DIFFUCO control coverage does not imply full DIFFUCO closure unless all beta, drag, photosynthesis, salinity, tide, and biomass inputs are exact.",
        "ENERBIL coverage is complete only after same-step DIFFUCO produces its pre-call beta/drag/surface payload and ENERBIL local source kernels are executable.",
    )

    @property
    def ready_for_sechiba_explicit_step(self) -> bool:
        """True only when every module boundary required for the local chain is exact."""

        return not any(self.missing_by_module.values())

    @property
    def missing_components(self) -> tuple[str, ...]:
        """Flatten missing module fields into stable ``module:field`` labels."""

        return tuple(
            f"{module}:{field}"
            for module, fields in self.missing_by_module.items()
            for field in fields
        )


def first_step_sechiba_input_coverage(
    *,
    diffuco_control: DiffucoControlInundationInputCoverage,
    diffuco_control_salinity: DiffucoControlSalinityAssembly | None = None,
    diffuco_control_inundation: DiffucoControlInundationAssembly | None = None,
    diffuco_precall: DiffucoFirstStepPrecallAssembly | None = None,
    diffuco_local_enerbil_precall: DiffucoPFT14FirstStepLocalEnerbilPrecallAssembly | None = None,
    enerbil: EnerbilFirstStepInputCoverage,
    hydrol_precall: HydrolFirstStepPrecallAssembly | None = None,
    hydrol_module: HydrolFirstStepModuleClosure | None = None,
    condveg_module: CondvegFirstStepModuleClosure | None = None,
    thermosoil_module: ThermosoilFirstStepModuleClosure | None = None,
    hydrol_restart_anchors_available: bool,
    condveg_initialize_available: bool,
    thermosoil_restart_recurrence_available: bool,
    slowproc_restart_state_available: bool,
) -> SechibaFirstStepInputCoverage:
    """Aggregate exact first-step SECHIBA coverage without filling values.

    Fortran provenance follows ``sechiba_main`` ordering: ``rprof`` setup and
    ``diffuco_main`` before ``enerbil_main`` at lines 984-1019, ``hydrol_main``
    at lines 1049-1072, ``condveg_main`` at lines 1085-1091,
    ``thermosoil_main`` at lines 1109-1118, then ``slowproc_main`` and
    ``stomate_main`` at lines 1184-1216.
    """

    closed: list[str] = []
    missing: dict[str, tuple[str, ...]] = {}

    if diffuco_control.complete:
        closed.append("diffuco_control_inundation_inputs")
    missing_diffuco = list(diffuco_control.missing_inputs)
    if diffuco_control_salinity is not None and diffuco_control_salinity.ok:
        closed.append("diffuco_control_salinity")
    else:
        missing_diffuco.append("salinity_control")
    if diffuco_control_inundation is not None and diffuco_control_inundation.ok:
        closed.append("diffuco_control_inundation")
        missing_diffuco = [field for field in missing_diffuco if field != "tide_height"]
    else:
        missing_diffuco.append("inundation_control")
    if diffuco_precall is not None:
        closed.append("diffuco_first_step_precall_partial_payload")
        missing_diffuco.extend(diffuco_precall.missing_local_process_inputs)
    else:
        missing_diffuco.append("first_step_precall_payload")
    if diffuco_local_enerbil_precall is not None:
        closed.append("diffuco_first_step_local_enerbil_precall")
    else:
        missing_diffuco.extend(("pft14_trans_co2_gpp", "diffuco_comb_final_beta_bundle", "after_diffuco_precall_payload"))
    missing["DIFFUCO"] = tuple(dict.fromkeys(missing_diffuco))

    if enerbil.ok:
        closed.append("enerbil_precall_inputs")
    missing["ENERBIL"] = tuple(enerbil.missing)

    if hydrol_restart_anchors_available:
        closed.append("hydrol_restart_anchors")
    missing["HYDROL"] = () if hydrol_restart_anchors_available else ("restart_anchor_state",)
    if hydrol_restart_anchors_available:
        missing_hydrol = []
        if hydrol_precall is not None:
            closed.append("hydrol_first_step_precall_partial_payload")
            if hydrol_precall.enerbil_boundary.ok:
                closed.append("enerbil_to_hydrol_first_step_boundary")
            missing_hydrol.extend(hydrol_precall.missing_inputs)
        else:
            missing_hydrol.append("same_step_enerbil_evaporation_inputs")
        if hydrol_module is not None:
            if hydrol_module.ok:
                closed.extend(
                    (
                        "hydrol_first_step_module_outputs",
                        "hydrol_to_stomate_diagnostics",
                        "hydrol_to_thermosoil_moisture",
                    )
                )
            missing_hydrol.extend(hydrol_module.missing_inputs)
        else:
            missing_hydrol.extend(
                (
                    "hydrol_main_outputs",
                    "hydrol_to_stomate_diagnostics",
                    "hydrol_to_thermosoil_moisture",
                )
            )
        missing["HYDROL"] = tuple(dict.fromkeys(missing_hydrol))

    if condveg_initialize_available:
        closed.append("condveg_initialize_state")
    missing["CONDVEG"] = () if condveg_initialize_available else ("initialize_state",)
    if condveg_initialize_available:
        missing_condveg = []
        if condveg_module is not None:
            if condveg_module.ok:
                closed.append("condveg_to_thermosoil_snow_fractions")
            missing_condveg.extend(condveg_module.missing_inputs)
            if "albedo" in condveg_module.covered_fields:
                closed.append("condveg_same_step_albedo_update")
            else:
                missing_condveg.append("same_step_snow_albedo_update")
        else:
            missing_condveg.extend(("same_step_snow_albedo_update", "condveg_to_thermosoil_snow_fractions"))
        missing["CONDVEG"] = tuple(dict.fromkeys(missing_condveg))

    if thermosoil_restart_recurrence_available:
        closed.append("thermosoil_restart_recurrence_state")
    missing["THERMOSOIL"] = () if thermosoil_restart_recurrence_available else ("restart_recurrence_state",)
    if thermosoil_restart_recurrence_available:
        missing_thermosoil = []
        if thermosoil_module is not None:
            if thermosoil_module.ok:
                closed.extend(
                    (
                        "thermosoil_main_temperature_profile",
                        "thermosoil_to_stomate_stempdiag",
                        "next_step_soil_thermal_coefficients",
                    )
                )
            missing_thermosoil.extend(thermosoil_module.missing_inputs)
        else:
            missing_thermosoil.extend(
                (
                    "same_step_hydrol_moisture_inputs",
                    "thermosoil_main_temperature_profile",
                    "thermosoil_to_stomate_stempdiag",
                    "next_step_soil_thermal_coefficients",
                )
            )
        missing["THERMOSOIL"] = tuple(dict.fromkeys(missing_thermosoil))

    if slowproc_restart_state_available:
        closed.append("slowproc_restart_state")
    missing["SLOWPROC"] = () if slowproc_restart_state_available else ("restart_state",)
    if slowproc_restart_state_available:
        missing["SLOWPROC"] = (
            "post_stomate_surface_update",
            "slowproc_to_stomate_dynamic_surface_state",
        )

    return SechibaFirstStepInputCoverage(
        diffuco_control=diffuco_control,
        diffuco_control_salinity=diffuco_control_salinity,
        diffuco_control_inundation=diffuco_control_inundation,
        diffuco_precall=diffuco_precall,
        diffuco_local_enerbil_precall=diffuco_local_enerbil_precall,
        enerbil=enerbil,
        hydrol_precall=hydrol_precall,
        hydrol_module=hydrol_module,
        condveg_module=condveg_module,
        thermosoil_module=thermosoil_module,
        hydrol_restart_anchors_available=bool(hydrol_restart_anchors_available),
        condveg_initialize_available=bool(condveg_initialize_available),
        thermosoil_restart_recurrence_available=bool(thermosoil_restart_recurrence_available),
        slowproc_restart_state_available=bool(slowproc_restart_state_available),
        closed_module_boundaries=tuple(closed),
        missing_by_module={module: tuple(fields) for module, fields in missing.items()},
    )
