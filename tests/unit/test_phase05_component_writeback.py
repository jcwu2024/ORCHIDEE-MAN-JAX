"""Real finite component returns through shared writers and next operands.

The independently declared growth and litter tuples are joined on two cells.
This verifies transport/order at actual production functions, not a new
original-Fortran oracle for the combined tuple or a natural model day.
"""
from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace as NS

import jax
import numpy as np
import pytest

from jax_orchidee.driver import orchestration as driver
from jax_orchidee.sechiba.condveg import run_condveg_first_step_module
from jax_orchidee.sechiba.diffuco import ControlInundateInputs, mangrove_control_inundation
from jax_orchidee.sechiba.slowproc import slowproc_surface_transition_explicit
from jax_orchidee.stomate import integration
from scripts.dev import oracle_phase05_canopy_conditions as canopy
from scripts.dev import oracle_phase05_energy_conditions as energy
from scripts.dev import oracle_phase05_growth_conditions as growth
from scripts.dev import oracle_phase05_litter_soilcarbon as litter


def first_two(value):
    if isinstance(value, dict):
        return {key: first_two(item) for key, item in value.items()}
    if isinstance(value, np.ndarray) and value.shape and value.shape[0] == 6:
        return value[:2].copy()
    return value


@pytest.fixture(scope="module")
def actual_components():
    args, _ = litter.case_inputs(litter.SPECS[0])
    _, leak, _, _ = litter.actual_outputs(args)
    inputs = first_two(growth.chain_inputs())
    for group in ("prescribe_inputs", "post_npp_inputs"):
        inputs[group]["veget_max"] = np.asarray(args["veget_max"])
    carbon = growth.run_current_chain(inputs, public_wrapper=True)
    envelope = integration.ExplicitOkLeakFromPostNppResult(
        prep=None, ok_leak=leak, requires_trace=(),
    )
    output = integration.stomate_lpj_outputs_from_post_npp_ok_leak_explicit(
        post_npp=carbon.post_npp, ok_leak=envelope,
        veget_max=inputs["prescribe_inputs"]["veget_max"],
        z_soil=args["z_soil"], zf_soil=args["zf_soil_b"],
        carb_mass_total_old=np.zeros(2),
        gpp_daily=inputs["post_npp_inputs"]["gpp_daily"],
    )
    pref = (1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4)
    scalars = NS(pref_soil_veg=pref, nstm=6, nvm=14,
                 fortran_pft_ids=tuple(range(1, 15)))
    old = driver.DriverPreviousStepStatePacket(
        tstep=47,
        fields_by_component={
            "slowproc_stomate_previous_step_state": {
                "lai": np.zeros((2, 14)), "veget_max": args["veget_max"],
                "frac_nobio": np.zeros((2, 1)),
                "gdd_init_date": np.asarray([[0., 2000.], [0., 2000.]]),
                "deepC_peat": np.zeros((2, 14, 32)),
                "depth_deepsoil": args["zf_soil_b"],
            },
            "diffuco_previous_step_state": {},
        },
        provenance_by_component={},
    )
    values = {"STOMATE_OK_DGVM": "n", **{
        f"EXT_COEFF_VEGETFRAC__{i:05d}": "1.0" for i in range(1, 15)
    }}
    surface = driver._paper_later_day_slowproc_surface_update_from_runtime(
        previous_state=old, run_def_values=values, daily_carbon=carbon,
        metadata=scalars, run_scalars=scalars,
    )
    return inputs, carbon, envelope, output, old, surface, scalars


@pytest.mark.parametrize("fresh", (False, True))
def test_actual_daily_packet_retains_fresh_source_date_and_carbon_returns(actual_components, fresh):
    inputs, carbon, leak, output, old, surface, _ = actual_components
    current = np.asarray([[100., 863.8901616842861], [100., 863.8901616842861]])
    # This field comes from the independently checked season producer in the
    # adjacent test file; here its distinct value isolates writer ownership.
    season_fields = {"gdd_init_date": current, "t2m_month": np.asarray([280., 290.])} if fresh else None
    reset = {
        "gpp_daily": np.zeros_like(inputs["post_npp_inputs"]["gpp_daily"]),
        "resp_maint_part": np.zeros_like(inputs["post_npp_inputs"]["resp_maint_part"]),
    }
    packet = driver._paper_day_first_stomate_state_packet(
        daily_carbon=carbon, ok_leak=leak, outputs=output,
        daily_reset_fields=reset, season_memory_fields=season_fields,
        slowproc_surface_update=surface, previous_state=old, dt_days=1.,
    )
    expected = current if fresh else old.fields_by_component["slowproc_stomate_previous_step_state"]["gdd_init_date"]
    np.testing.assert_array_equal(packet.fields["gdd_init_date"], expected)
    np.testing.assert_array_equal(packet.fields["sla_calc"], carbon.post_npp.daily_carbon.age_sla.sla_calc)
    np.testing.assert_array_equal(packet.fields["lai"], carbon.post_npp.lai_after_setlai)
    np.testing.assert_array_equal(packet.fields["assim_param"][:, :, 0], carbon.post_npp.vmax.vcmax)
    end = driver._paper_day_end_state_packet(previous_state=old, stomate_state=packet)
    slow = end.fields_by_component["slowproc_stomate_previous_step_state"]
    np.testing.assert_array_equal(slow["gdd_init_date"], expected)
    np.testing.assert_array_equal(slow["DOC"], leak.ok_leak.soilcarbon.doc)
    for name in ("lai", "veget", "veget_max", "frac_nobio", "tot_bare_soil"):
        np.testing.assert_array_equal(end.fields_by_component["diffuco_previous_step_state"][name], packet.fields[name])
    _, operands = driver._paper_later_day_slowproc_derivvar_payload(
        slow, vcmax_fix=np.full(14, 50.), height_presc=np.full(14, 1.), qsintcst=.1,
    )
    expected_qsint = .1 * np.asarray(slow["veget"]) * np.asarray(slow["lai"])
    expected_qsint[:, 0] = 0.
    np.testing.assert_allclose(operands["qsintmax"], expected_qsint, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(operands["assim_param"], packet.fields["assim_param"])


def test_daily_and_nondaily_surface_use_actual_growth_return(actual_components):
    _, carbon, _, _, old, surface, scalars = actual_components
    veg = surface.vegetation
    result = slowproc_surface_transition_explicit(
        do_slow=False, lai=carbon.post_npp.lai_after_setlai,
        frac_nobio=veg.frac_nobio, veget_max=veg.veget_max, veget=veg.veget,
        totfrac_nobio=veg.totfrac_nobio, soiltile=veg.soiltile,
    )
    for name in veg._fields:
        np.testing.assert_array_equal(getattr(result.surface.vegetation, name), getattr(veg, name))
    np.testing.assert_array_equal(result.surface.tot_bare_soil, surface.tot_bare_soil)
    assert old.tstep == 47 and scalars.nstm == 6


def test_outer_compiled_dispatch_preserves_explicit_source_flags():
    inputs = first_two(growth.chain_inputs(cold_start=True, agriculture=False))
    dispatch = {}
    for source, prefix, keys in (
        ("prescribe_inputs", "pres_", ("ok_dgvm", "lpj_gap_const_mort", "natural", "pasture", "is_tree", "pheno_is_none", "firstcall", "stomate_restart_none")),
        ("constraints_inputs", "cons_", ("natural", "is_tree", "is_peat", "pheno_is_none", "tmin_crit", "tcm_crit", "agriculture", "undef")),
        ("post_npp_inputs", "post_", ("ok_dgvm", "lpj_gap_const_mort", "wire_vmax", "ok_nlim_vmax")),
    ):
        for key in keys:
            value = inputs[source][key]
            dispatch[prefix + key] = tuple(value.tolist()) if isinstance(value, np.ndarray) else value
    dispatch["phenology_active_pft_mask"] = tuple(np.any(inputs["prescribe_inputs"]["veget_max"] > 0., axis=0).tolist())
    dispatch["pheno_model"] = inputs["phenology_inputs"]["pheno_model"]
    # Inputs' arrays cross the outer boundary. PFT/source gates are immutable
    # closure values, as in the maintained driver specialization contract.
    dynamic = {group: {} for group in inputs}
    constants = {group: dict(values) for group, values in inputs.items()}
    for group, values in inputs.items():
        for key, value in values.items():
            if isinstance(value, np.ndarray) and value.ndim > 1:
                dynamic[group][key] = value
                constants[group].pop(key)

    def outer(operands):
        result = integration.stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_outer_compiled(
            **{group: {**constants[group], **operands[group]} for group in inputs},
            static_dispatch=dispatch,
        )
        return integration._daily_carbon_without_boundary_trace(result)

    result = jax.jit(outer)(dynamic)
    eager = growth.run_current_chain(inputs)
    for name, expected in growth.chain_fields(eager).items():
        np.testing.assert_allclose(growth.chain_fields(result)[name], expected, rtol=1e-12, atol=1e-14, err_msg=name)


def test_all_maintained_driver_bundle_calls_pass_solar_producer_operands():
    # Only inspect the actual orchestration module; no model step is run.
    tree = ast.parse(inspect.getsource(driver))
    names = {"_paper_first_step_stomate_restart_input_bundles",
             "_paper_cold_start_stomate_input_bundles", "_paper_later_day_stomate_input_bundles"}
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id in names]
    assert len(calls) == 7
    for call in calls:
        keys = {keyword.arg for keyword in call.keywords}
        assert {"latitude", "calendar_str"} <= keys, (call.func.id, call.lineno)


@pytest.mark.parametrize("use_jit", (False, True))
def test_actual_condveg_wrapper_return_reaches_writer_and_next_diffuco_operands(use_jit):
    inputs = canopy.make_albedo_inputs()
    values = {"ALB_BG_MODIS": "y", "ALB_BARE_MODEL": "n", "ALB_ICE__00001": ".6", "ALB_ICE__00002": ".2"}
    for name in ("alb_leaf_vis", "alb_leaf_nir", "snowa_aged_vis", "snowa_aged_nir", "snowa_dec_vis", "snowa_dec_nir"):
        values.update({f"{name.upper()}__{index:05d}": str(value) for index, value in enumerate(inputs[name], 1)})
    current = run_condveg_first_step_module(
        restart_payload={**inputs, "soilalbedo_bg": inputs["soilalb_bg"]},
        driver_payload=inputs, slowproc_payload=inputs,
        hydrol_diagnostics=NS(drysoil_frac=inputs["drysoil_frac"]),
        run_def_values=values, use_jit=use_jit,
    )
    assert current.ok and not current.missing_inputs
    # The same complete independent source tuple also reaches the real wrapper.
    expected = canopy.condveg_values(inputs, use_jit=use_jit, kind="modis")
    for name in current.module._fields:
        actual = np.asarray(getattr(current.module, name))
        if name == "roughheight_pft":
            actual = actual[:, 1:]
        np.testing.assert_allclose(actual.ravel(), expected[name], rtol=1e-11, atol=1e-12, err_msg=name)
    fields = driver._empty_previous_step_state_fields()
    driver._add_condveg_to_previous_fields(fields, current)
    operands, _, missing = driver._diffuco_restart_payload_from_state_fields(
        diffuco_state=fields["diffuco_previous_step_state"], hydrol_state={}, slowproc_state={},
    )
    assert missing  # This real mapper intentionally lacks unrelated next-step inputs.
    for name in ("z0m", "z0h", "roughheight", "roughheight_pft", "frac_snow_veg", "frac_snow_nobio"):
        np.testing.assert_array_equal(fields["diffuco_previous_step_state"][name], getattr(current.module, name))
        np.testing.assert_array_equal(operands[name], getattr(current.module, name))
    np.testing.assert_array_equal(fields["enerbil_previous_step_state"]["albedo"], current.module.albedo)
    np.testing.assert_array_equal(fields["driver_previous_step_state"]["albedo_vis"], current.module.albedo[:, 0])


@pytest.mark.parametrize("cap", tuple(canopy.CAPS))
@pytest.mark.parametrize("use_jit", (False, True))
def test_actual_root_aeration_return_enters_real_diffuco_and_energy(cap, use_jit):
    from scripts.dev.oracle_canopy_component_fixture import run_canopy_components

    inputs = canopy.make_control_inputs(cap)
    roots = mangrove_control_inundation(
        ControlInundateInputs(**{name: inputs[name] for name in ("z_soil", "rprof", "tide_height", "biomass")}),
        **{name: inputs[name] for name in (
            "control_inudate_min", "agb_agr_ven_all_st", "agb_agr_ven_all_pn", "h_agr_max_st", "h_agr_max_pn",
        )},
    )
    fixture = energy.joint_fixture()
    payload = {**fixture.diffuco_precall.payload, "control_inudate": roots.control_inudate[:4]}
    fixture = fixture._replace(diffuco_precall=fixture.diffuco_precall._replace(payload=payload))
    # This explicitly declared component join has no new combined Fortran oracle
    # and makes no natural-trajectory claim. Both numerical owners really run.
    current = run_canopy_components(fixture, diffuco_use_jit=use_jit, enerbil_use_jit=use_jit)
    assert current.enerbil_precall.ok
    diffuco = current.diffuco.result.payload.payload
    np.testing.assert_array_equal(current.diffuco.kwargs["trans_co2_inputs"]["control_inudate"], roots.control_inudate[:4])
    assert np.all(np.isfinite(np.asarray(diffuco["gpp"])))
    for name in ("vbeta", "vbeta1", "vbeta2", "vbeta3", "vbeta4", "vbeta5"):
        np.testing.assert_array_equal(current.enerbil.payload[name], diffuco[name])
