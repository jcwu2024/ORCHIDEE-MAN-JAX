"""THERMOSOIL's final grid clamp must survive every driver state writer.

The thermal wrapper runs its real small CPU numerical kernel. Other completed
components are synthetic, with no model/forcing/restart-file integration.
"""

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace as NS

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_orchidee.driver import orchestration as driver
from jax_orchidee.driver.fast_state import fast_state_from_previous_packet
from jax_orchidee.sechiba import thermosoil
from jax_orchidee.sechiba.hydrol import HydrolThermosoilMoistureInputs
from jax_orchidee.sechiba.restart_lifecycle import SECHIBA_FINALIZE_SOURCE_FIELDS


def _thermal_closure(temp, snowdz, *, explicit=True, compiled=False):
    """Small real THERMOSOIL call with current HYDROL snow, not ENERBIL snow."""
    temp = jnp.asarray(temp, dtype=jnp.float64)
    snowdz = jnp.asarray(snowdz, dtype=jnp.float64)
    npts = temp.shape[0]
    def grid(value):
        return jnp.full((npts,), value, dtype=jnp.float64)

    def layers(value):
        return jnp.full((npts, 3), value, dtype=jnp.float64)
    pft = temp[:, None] + 4.0
    moisture = HydrolThermosoilMoistureInputs(
        shumdiag_perma=jnp.full((npts, 2), 0.3),
        mc_layh=jnp.full((npts, 2), 0.3),
        mcl_layh=jnp.full((npts, 2), 0.3),
        tmc_layh=jnp.full((npts, 2), 30.0),
        mc_layh_pft=jnp.full((npts, 2, 1), 0.3),
        mcl_layh_pft=jnp.full((npts, 2, 1), 0.3),
        tmc_layh_pft=jnp.full((npts, 2, 1), 30.0),
    )
    restart = NS(
        ptn=jnp.full((npts, 4, 1), 280.0),
        cgrnd=jnp.full((npts, 3, 1), 275.0),
        dgrnd=jnp.full((npts, 3, 1), 0.2),
        cgrnd_snow=layers(260.0), dgrnd_snow=layers(0.1),
        lambda_snow=grid(0.5), gtemp=grid(278.0),
        refsoc=jnp.zeros((npts, 4)),
    )
    payload = {
        "temp_sol_new": temp, "temp_sol_new_pft": pft,
        "soilcap": grid(1000.0),
        # Deliberately stale and opposite to current snow at selected points.
        "snowdz": jnp.where(snowdz > 0.0, 0.0, 1.0),
    }
    closure = thermosoil.run_thermosoil_first_step_module(
        moisture=moisture, thermosoil_restart=restart,
        enerbil_payload=payload,
        condveg_result=NS(frac_snow_veg=grid(0.4), frac_snow_nobio=jnp.zeros((npts, 1))),
        snowrho=layers(200.0), snowtemp=layers(270.0), snowdz=snowdz,
        snow=grid(6.6), njsc=jnp.ones((npts,), dtype=jnp.int32),
        veget_max=jnp.ones((npts, 1)), pb=grid(1000.0), totfrac_nobio=grid(0.0),
        dlt=jnp.asarray([0.1, 0.2, 0.4, 0.8]), dz1=jnp.asarray([10.0, 3.0, 1.0]),
        zlt=jnp.asarray([0.1, 0.3, 0.7, 1.5]), znt=jnp.asarray([0.05, 0.2, 0.5, 1.1]),
        dz5=jnp.asarray([0.2, 0.4, 0.8]), dt_sechiba=1800.0,
        ok_laidev=(False,), ok_explicitsnow=explicit, use_jit=compiled,
    )
    return closure, payload


def _components(closure, payload):
    """Complete small post-process components for real packet/finalize writers."""
    npts = payload["temp_sol_new"].shape[0]
    grid = jnp.zeros((npts,))
    tile = jnp.zeros((npts, 1))
    ep = {name: grid for name in (
        "qsurf evapot evapot_corr tsol_rad vevapp fluxlat fluxsens petAcoef petBcoef "
        "peqAcoef peqBcoef zlev qair temp_air rau"
    ).split()}
    ep.update(payload)
    enerbil = NS(payload=ep, step=NS(pottemp=None))
    diffuco = NS(
        kwargs={"pft_index": 0},
        result=NS(payload=NS(payload={name: tile for name in ("rstruct", "gpp", "gsmean", "rveget", "cimean")}),
                  boundary=NS(drag=NS(q_cdrag_pft=tile),
                              process_chain=NS(trans_co2=NS(canopy=NS(leaf_ci=tile))))),
    )
    snow = NS(**{name: grid for name in ("snow", "snow_age")},
              **{name: tile for name in ("snow_nobio", "snow_nobio_age")},
              **{name: jnp.zeros((npts, 3)) for name in ("snowdz", "snowrho", "snowtemp", "snowheat", "snowgrain")})
    soil = NS(**{name: tile for name in (
        "mc mcl water2infilt run2peat run2man wt_ab wt_ab_tide profil_froz_hydro_ns"
    ).split()})
    diag = NS(**{name: grid for name in (
        "ae_ns us vegstress evap_bare_lim drysoil_frac humrel wtp fwet_new liqwt_ratio humtot"
    ).split()}, evap_bare_lim_ns=tile, mc_layh_s=tile)
    outputs = NS(**{name: tile for name in (
        "drainage_per_soil runoff_per_soil wat_flux runoff2peat canopy2ground precip2ground precip2canopy"
    ).split()})
    hydrol = NS(ok=True, snow_state=snow, diagnostics=diag, outputs=outputs,
                module=NS(soil=soil, canop=NS(qsintveg=tile), flood=NS(flood_res=grid)),
                module_inputs={}, snow_step=None)
    condveg = NS(ok=True, module=NS(
        **{name: grid for name in ("z0m", "z0h", "roughheight", "frac_snow_veg")},
        roughheight_pft=tile, frac_snow_nobio=tile, albedo=None))
    prior_finalize = {name: tile for name in SECHIBA_FINALIZE_SOURCE_FIELDS}
    prior_finalize["leaf_ci"] = jnp.zeros((npts, 1, 1))
    previous = driver._packet_from_previous_step_fields(tstep=0, fields={
        "sechiba_finalize_state": prior_finalize,
        "slowproc_stomate_previous_step_state": {"lai": tile},
    })
    return enerbil, diffuco, hydrol, condveg, previous


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_real_thermal_closure_gate_threshold_and_current_snow(explicit, compiled):
    temp = np.asarray([272.0, 273.15, 280.0, 280.0, 280.0, 280.0])
    # Strict sum, not any positive layer: fifth point sums to zero.
    snowdz = [[0.1, 0, 0], [0.1, 0, 0], [0.1, 0, 0], [0, 0, 0], [0.1, -0.1, 0], [-0.1, 0, 0]]
    closure, ep = _thermal_closure(temp, snowdz, explicit=explicit, compiled=compiled)
    assert closure.ok
    expected = temp.copy()
    if explicit:
        expected[1:3] = 273.15
    np.testing.assert_array_equal(closure.downstream_payload["temp_sol_new"], expected)
    np.testing.assert_array_equal(ep["temp_sol_new"], temp)
    np.testing.assert_array_equal(ep["temp_sol_new_pft"], temp[:, None] + 4.0)
    # THERMOSOIL must solve with the original ENERBIL values before its final clamp.
    np.testing.assert_array_equal(closure.boundary_payload["temp_sol_new"], temp)


@pytest.mark.parametrize("route", ["runtime", "compiled_fast", "later_scaffold", "first_scaffold", "cold"])
def test_all_packet_routes_finalize_and_next_diffuco_read_clamped_grid(monkeypatch, route):
    temp = jnp.asarray([280.0, 281.0])
    closure, ep = _thermal_closure(temp, [[0.1, 0, 0], [0, 0, 0]])
    enerbil, diffuco, hydrol, condveg, previous = _components(closure, ep)
    # The clamp was already computed in the successful THERMOSOIL closure.
    def unexpected_second_clamp(**kwargs):
        pytest.fail("driver writers must reuse the closed downstream temperature")
    monkeypatch.setattr(thermosoil, "thermosoil_explicit_snow_surface_clamp", unexpected_second_clamp)

    if route in {"runtime", "compiled_fast"}:
        def write(temp_override):
            local = NS(payload={**enerbil.payload, "temp_sol_new": temp_override}, step=enerbil.step)
            return driver._runtime_next_state_from_components(
                tstep=1, previous_state=(fast_state_from_previous_packet(previous) if route == "compiled_fast" else previous),
                enerbil_local=local, diffuco_local=diffuco, thermosoil_module=closure,
                condveg_module=condveg, hydrol_module=hydrol)
        packet = jax.jit(write)(temp) if route == "compiled_fast" else write(temp)
    elif route == "later_scaffold":
        packet = driver.driver_previous_step_state_from_next_step_hydrol_scaffold(NS(
            ready_for_next_state=True, tstep=1,
            enerbil=NS(enerbil_local=enerbil, diffuco_local_enerbil_precall=diffuco, previous_state=previous),
            thermosoil_module=closure, hydrol_module=hydrol, condveg_module=condveg))
    elif route == "first_scaffold":
        # All restart-specific carry/readstart owners are orthogonal to this writer.
        packet = driver.driver_previous_step_state_from_timestep_scaffold(NS(
            step=NS(tstep=0), enerbil_first_step_local=enerbil,
            diffuco_first_step_local_enerbil_precall=diffuco,
            diffuco_control_salinity=None, diffuco_control_assembly=None,
            diffuco_first_step_precall=None, thermosoil_first_step_module=closure,
            hydrol_first_step_module=hydrol, condveg_first_step_module=condveg,
            first_step_restart_state=None))
    else:
        # Cold initialization needs all declared static/carry owners, even when
        # only the two thermal writeback fields are being asserted here.
        npts = temp.shape[0]
        tile = jnp.zeros((npts, 1))
        full = NS(**{name: tile for name in driver._SECHIBA_SLOWPROC_FINALIZE_SOURCE_FIELDS})
        full.veget = tile
        full.peat = full
        cold_hydrol = NS(**{name: tile for name in (
            "free_drain_coef zwt_force fwet_out fwet_new liqwt_ratio wtp run2peat run2man wt_ab wt_ab_tide resdist vegtot_old"
        ).split()})
        packet = driver._previous_step_state_from_cold_start_payloads({
            "enerbil_first_step_local": enerbil, "diffuco_first_step_local_enerbil_precall": diffuco,
            "thermosoil_first_step_module": closure, "hydrol_first_step_module": hydrol,
            "condveg_first_step_module": condveg, "slowproc_init_pft14": full,
            "hydrol_cold_start_state": cold_hydrol, "condveg_soilalbedo_bg": tile,
            "thermosoil_refSOC": tile,
            "enerbil_surface_state": NS(q_sol_pot=tile, temp_sol_pot=tile),
        })
    state = packet.fields_by_component
    expected_components = ["diffuco_previous_step_state", "enerbil_previous_step_state"]
    if route != "first_scaffold":
        expected_components.append("sechiba_finalize_state")
    for component in expected_components:
        np.testing.assert_array_equal(state[component]["temp_sol"], [273.15, 281.0])
        np.testing.assert_array_equal(state[component]["temp_sol_pft"], temp[:, None] + 4.0)
    # Exercise the real next-step DIFFUCO state-to-precall mapper.
    next_payload, _, _ = driver._diffuco_restart_payload_from_previous_state(packet)
    np.testing.assert_array_equal(next_payload["temp_sol"], [273.15, 281.0])
    np.testing.assert_array_equal(enerbil.payload["temp_sol_new"], temp)
    np.testing.assert_array_equal(enerbil.payload["temp_sol_new_pft"], temp[:, None] + 4.0)


def test_incomplete_thermosoil_does_not_publish_final_clamp():
    fields = driver._empty_previous_step_state_fields()
    temp = jnp.asarray([280.0])
    closure, ep = _thermal_closure(temp, [[0.1, 0, 0]])
    enerbil, _, _, _, _ = _components(closure, ep)
    incomplete = closure._replace(module=None, missing_inputs=("failed thermal closure",))
    driver._add_enerbil_local_to_previous_fields(fields, enerbil, thermosoil_module=incomplete)
    np.testing.assert_array_equal(fields["diffuco_previous_step_state"]["temp_sol"], temp)


@pytest.mark.parametrize("explicit", [False, True])
def test_restart_scaffold_calls_thermal_owner_with_current_hydrol_snow_and_gate(monkeypatch, explicit):
    """Traverse the actual first/restart caller with upstream processes stubbed.

    Stop at the THERMOSOIL call: no forcing/restart IO or integration is run.
    The separate tests execute the real thermal owner and its state consumers.
    """
    grid = np.zeros((2,))
    tile = np.zeros((2, 1))
    closure, ep = _thermal_closure([280.0, 281.0], [[0.1, 0, 0], [0, 0, 0]])
    enerbil, diffuco, hydrol, condveg, _ = _components(closure, ep)
    # Current snow intentionally differs from restart snow in every field.
    for name in ("snow", "snowrho", "snowtemp", "snowdz"):
        setattr(hydrol.snow_state, name, np.full_like(getattr(hydrol.snow_state, name), 0.125))
    hydrol.thermosoil_moisture = object()
    restart_snow = NS(
        snow=grid, snowrho=np.zeros((2, 3)), snowtemp=np.zeros((2, 3)), snowdz=np.zeros((2, 3)),
        veget=tile, veget_max=tile, frac_nobio=tile, as_payload=lambda: {},
    )
    restart = NS(
        driver_albedo=NS(z0=grid, as_coverage_payload=lambda: {}),
        enerbil_thermal=NS(as_coverage_payload=lambda: {}),
        stomate=NS(biomass=np.zeros((2, 1, 1, 1)), altmax=grid),
        diffuco_precall=restart_snow, slowproc=NS(veget=tile, lai=tile),
        hydrol=NS(njsc=grid, run2peat=grid, run2man=grid), thermosoil=NS(),
    )
    values = defaultdict(lambda: "FALSE", {"OK_EXPLICITSNOW": "TRUE" if explicit else "FALSE"})
    context = NS(run_def_path=Path("unused.def"), runtime=NS(dt_sechiba=1800.0),
                 run_def_values=values, dt_sechiba=1800.0, first_step_restart_state=restart)
    step = NS(run_scalars=NS(fortran_pft_ids=(14,), pref_soil_veg=tile,
                            is_tree=(False,), z0_over_height=0.1, ratio_z0m_z0h=1.0))
    payload = NS(kjpindex=2, **{name: grid for name in (
        "u v zlev temp_air qair pb swdown ccanopy precip_rain precip_snow lalo neighbours resolution salinity tide_height"
    ).split()})
    cwrr = NS(**{name: grid for name in ("znh", "dlh", "dnh", "dlt", "dz1", "zlt", "znt", "dz5")})
    for name, result in {
        "_paper_1961_step_bundle_from_context": step,
        "build_intersurf_first_step_payload": payload,
        "paper_case_vertical_grids_from_used_run_def": (grid, grid, grid),
        "paper_case_cwrr_vertical_soil_grid_from_used_run_def": cwrr,
        "stomate_driver_entry_source": {}, "stomate_no_routing_entry_source": {},
        "_enerbil_driver_coverage_payload": {}, "enerbil_first_step_input_coverage": NS(),
        "assemble_enerbil_first_step_precall_payload": NS(),
        "diffuco_control_inundation_input_coverage": NS(),
        "assemble_pft14_control_salinity_first_step": NS(),
        "assemble_pft14_control_inundation_first_step": NS(),
        "slowproc_derivvar_explicit": NS(qsintmax=tile, assim_param=tile, height=tile, temp_growth=tile),
        "assemble_diffuco_first_step_precall_payload": NS(ok=True, payload={}),
        "run_pft14_local_enerbil_precall_from_first_step_precall": diffuco,
        "run_enerbil_first_step_local_from_precall": enerbil,
        "assemble_hydrol_first_step_precall_payload": NS(),
        "run_hydrol_first_step_module_from_precall": hydrol,
        "run_condveg_first_step_module": condveg,
        "parse_run_def_float": 1.0, "parse_run_def_int": 1,
        "parse_run_def_indexed_selection": np.ones((1,)),
        "parse_run_def_indexed_vector": np.ones((1,)),
        "_select_run_def_pft_vector": np.ones((1,)), "_paper_mangrove_pft_index": 0,
    }.items():
        monkeypatch.setattr(driver, name, lambda *args, _result=result, **kwargs: _result)

    class ReachedThermalOwner(Exception):
        pass

    def check_call(**kwargs):
        for name in ("snow", "snowrho", "snowtemp", "snowdz"):
            assert kwargs[name] is getattr(hydrol.snow_state, name)
        assert kwargs["ok_explicitsnow"] is explicit
        assert kwargs["moisture"] is hydrol.thermosoil_moisture
        raise ReachedThermalOwner

    monkeypatch.setattr(driver, "run_thermosoil_first_step_module", check_call)
    with pytest.raises(ReachedThermalOwner):
        driver.paper_1961_driver_timestep_scaffold("unused.yaml", prepared_context=context)
