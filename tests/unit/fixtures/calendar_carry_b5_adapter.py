"""Saved baseline full adapter; never import the old physical model.

Baseline: b5eff879e00154c936a1709d5ab91aae65f158d6
Source: jax_orchidee/driver/orchestration.py::_runtime_next_state_from_components
"""

def _runtime_next_state_from_components(
    *,
    tstep: int,
    previous_state: DriverPreviousStepStatePacket,
    enerbil_local: EnerbilFirstStepLocalAssembly | None,
    diffuco_local: DiffucoPFT14FirstStepLocalEnerbilPrecallAssembly | None,
    diffuco_day_static_cache: DriverDiffucoDayStaticCache | None = None,
    thermosoil_module: ThermosoilFirstStepModuleClosure | None,
    condveg_module: CondvegFirstStepModuleClosure | None,
    hydrol_module: HydrolFirstStepModuleClosure | None,
    ok_laidev: tuple[bool, ...] | None = None,
) -> DriverPreviousStepStatePacket:
    """Extract source-backed next-step state from compact runtime components."""

    fields = _empty_previous_step_state_fields()
    _add_enerbil_local_to_previous_fields(
        fields, enerbil_local, thermosoil_module=thermosoil_module,
    )
    _add_diffuco_local_to_previous_fields(fields, diffuco_local)
    if diffuco_day_static_cache is not None:
        _add_diffuco_saved_controls_to_previous_fields(
            fields,
            control_salinity=diffuco_day_static_cache.control_salinity,
            control_inundation=diffuco_day_static_cache.control_inundation,
        )
    _add_diffuco_saved_controls_to_previous_fields(fields, precall=getattr(diffuco_local, "kwargs", None))
    _add_thermosoil_to_previous_fields(fields, thermosoil_module)
    _add_condveg_to_previous_fields(fields, condveg_module)
    _add_hydrol_to_previous_fields(fields, hydrol_module)
    prior_finalize = _previous_state_component_fields(previous_state, "sechiba_finalize_state")
    if prior_finalize:
        fields["sechiba_finalize_state"] = _sechiba_finalize_state_from_components(
            previous=prior_finalize,
            enerbil_local=enerbil_local,
            diffuco_local=diffuco_local,
            thermosoil_module=thermosoil_module,
            condveg_module=condveg_module,
            hydrol_module=hydrol_module,
        )
    prior_slowproc = _previous_state_component_fields(previous_state, "slowproc_stomate_previous_step_state")
    for name in STOMATE_HALF_HOUR_CARRY_FIELDS:
        if name in prior_slowproc:
            fields["slowproc_stomate_previous_step_state"][name] = prior_slowproc[name]
    for name in ("veget", "veget_max", "lai", "frac_nobio", "height"):
        if name in prior_slowproc:
            fields["diffuco_previous_step_state"][name] = prior_slowproc[name]
    _write_maintenance_lai_to_previous_fields(fields, prior_slowproc, ok_laidev)
    prior_therm = _previous_state_component_fields(previous_state, "thermosoil_previous_step_state")
    for name in ("refSOC", "e_soil_lat"):
        if name in prior_therm:
            fields["thermosoil_previous_step_state"][name] = prior_therm[name]
    prior_hydrol = _previous_state_component_fields(previous_state, "hydrol_previous_step_state")
    for name in ("free_drain_coef", "zwt_force", "njsc", "fwet_new"):
        if name in prior_hydrol:
            fields["hydrol_previous_step_state"][name] = prior_hydrol[name]
    prior_diffuco = _previous_state_component_fields(previous_state, "diffuco_previous_step_state")
    if "soilalbedo_bg" in prior_diffuco:
        fields["diffuco_previous_step_state"]["soilalbedo_bg"] = prior_diffuco["soilalbedo_bg"]
    if "height" in prior_diffuco:
        fields["diffuco_previous_step_state"]["height"] = prior_diffuco["height"]
    for name in ("control_salinity", "control_inudate"):
        if name in prior_diffuco and name not in fields["diffuco_previous_step_state"]:
            fields["diffuco_previous_step_state"][name] = prior_diffuco[name]
    prior_driver = _previous_state_component_fields(previous_state, "driver_previous_step_state")
    if "albedo" in prior_driver and "albedo" not in fields.get("driver_previous_step_state", {}):
        fields.setdefault("driver_previous_step_state", {})["albedo"] = prior_driver["albedo"]
    if hasattr(previous_state, "component_fields"):
        return fast_state_from_previous_fields(
            tstep=int(tstep),
            fields_by_component=fields,
            provenance_by_component=_previous_step_state_provenance(),
        )
    return _packet_from_previous_step_fields(tstep=int(tstep), fields=fields)
