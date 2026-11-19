"""PB-03 real HYDROL/THERMOSOIL producers and selected driver consumers.

Only the two named physical components execute. Unrelated forcing, canopy,
energy and vegetation values are explicit adapter boundary data. No SECHIBA
timestep or STOMATE process is called.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from jax_orchidee.driver import orchestration as driver
from scripts.dev import oracle_physics_branch_hydrol as hydrol_oracle
from scripts.dev import oracle_physics_branch_thermal as thermal_oracle

HYDROL_CASE = "mineral_rain_canopy"
THERMAL_CASE = "hydrol_mineral_rain_canopy"
MOISTURE_FIELDS = ("shumdiag_perma", "mc_layh", "mcl_layh", "tmc_layh", "mc_layh_pft", "mcl_layh_pft", "tmc_layh_pft")


def run_components(hydrol_inputs, thermal_inputs, mode="eager"):
    """Feed the actual seven moisture exports into the actual thermal owner."""
    hydrol = hydrol_oracle.run_production(hydrol_inputs, mode)
    if not hydrol.ok:
        raise ValueError(hydrol.missing_inputs)
    thermal_entry = dict(thermal_inputs)
    thermal_entry.update({name: getattr(hydrol.thermosoil_moisture, name) for name in MOISTURE_FIELDS})
    thermal = thermal_oracle.run_production(thermal_entry, mode)
    return hydrol, thermal, thermal_entry


def write_and_map(hydrol, thermal):
    """Execute current production writers, packet transport and DIFFUCO map."""
    fields = driver._empty_previous_step_state_fields()
    driver._add_hydrol_to_previous_fields(fields, hydrol)
    driver._add_thermosoil_to_previous_fields(fields, thermal)
    packet = driver._packet_from_previous_step_fields(tstep=1, fields=fields)
    mapped, used, missing = driver._diffuco_restart_payload_from_previous_state(packet)
    return packet, mapped, used, missing


def stomate_entry(hydrol, thermal, packet, hydrol_inputs, thermal_inputs):
    """Run the real input adapter with separately declared unrelated seams.

    sechiba_main:1184-1212 passes vegstress as slowproc's humrel and wat_flux
    unchanged. This checks that boundary rename, not carbon computation.
    """
    fields = {key: dict(value) for key, value in packet.fields_by_component.items()}
    fields["slowproc_stomate_previous_step_state"] = {
        "veget": np.asarray(hydrol_inputs["veget"]),
        "veget_max": np.asarray(hydrol_inputs["veget_max"]),
        "totfrac_nobio": np.zeros(1),
    }
    fields["hydrol_previous_step_state"]["fwet_new"] = np.zeros(1)
    # These three prerequisites are declared input seams, not component outputs.
    fields["diffuco_previous_step_state"]["temp_sol"] = thermal_inputs["temp_sol_new"]
    packet = driver.DriverPreviousStepStatePacket(1, fields, packet.provenance_by_component)
    forcing = SimpleNamespace(
        u=np.array([2.]), v=np.array([1.]), lalo=np.array([[45., 120.]]),
        precip_rain=np.asarray(hydrol_inputs["precip_rain"]),
        precip_snow=np.zeros(1), pb=thermal_inputs["pb"],
    )
    canopy_boundary = SimpleNamespace(result=SimpleNamespace(
        payload=SimpleNamespace(payload={"gpp": np.zeros((1, 14))})))
    energy_boundary = SimpleNamespace(payload={"t2mdiag": np.array([280.]), "evapot_corr": np.array([.2])})
    context_boundary = SimpleNamespace(
        topmodel_new=False, min_wind=.1,
        cwrr_grid=SimpleNamespace(znt=thermal_inputs["znt"], zlt=thermal_inputs["zlt"]),
    )
    return driver._paper_next_step_runtime_entry_payload_from_components(
        tstep=1, driver_payload=forcing, previous_state=packet,
        diffuco_local=canopy_boundary, enerbil_local=energy_boundary,
        hydrol=hydrol, thermosoil=thermal, prepared_context=context_boundary,
    )
