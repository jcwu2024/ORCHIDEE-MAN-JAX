"""Bounded CONDVEG -> DIFFUCO argument/state carry, not a driver rollout.

Source: sechiba.f90::sechiba_main 997-1005 passes the already computed
frac_snow_veg/nobio; condveg.f90::condveg_frac_snow 878-893 owns their algebra.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jax_orchidee.sechiba.condveg import condveg_frac_snow
from jax_orchidee.sechiba.diffuco import (
    DIFFUCO_FIRST_STEP_PRECALL_RESTART_FIELDS,
    assemble_diffuco_first_step_precall_payload,
)


def _upstream():
    fields = dict(snow=np.array([10.]), snow_nobio=np.array([[5.]]),
                  snowrho=np.full((1, 3), 330.), snowdz=np.full((1, 3), .01))
    fractions = condveg_frac_snow(**fields, ok_explicitsnow=False,
                                  sn_dens=660., snowcri_alb=20.)
    return fields, fractions


def _cold_mapping_expression(fields, fractions):
    """Evaluate the actual cold mapping expression with bounded upstream data."""
    from jax_orchidee.driver import orchestration as orch

    tree = ast.parse(Path(orch.__file__).read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "paper_1961_driver_cold_start_first_step_coverage")
    value = next(node.value for node in ast.walk(owner) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "diffuco_restart_payload"
                         for target in node.targets))
    geometry = dict(veget=np.zeros((1, 14)), veget_max=np.zeros((1, 14)),
                    frac_nobio=np.zeros((1, 1)), lai=np.zeros((1, 14)))
    hydrol = SimpleNamespace(**{key: fields[key] for key in ("snow", "snow_nobio")},
        explicit_snow=SimpleNamespace(**fields, snowtemp=np.full((1, 3), 270.)),
        snow_age=np.zeros(1), snow_nobio_age=np.zeros((1, 1)), qsintveg=np.zeros((1, 14)),
        humrel=np.ones((1, 14)), evap_bare_lim=np.ones(1))
    surface = SimpleNamespace(**fractions._asdict(), roughheight=np.zeros(1),
        roughheight_pft=np.zeros((1, 14)), z0h=np.full(1, .001), z0m=np.full(1, .01))
    env = dict(vars(orch))
    env.update(enerbil_surface=SimpleNamespace(as_payload=lambda: {}), hydrol_cold=hydrol,
        slowproc_veg_cold=SimpleNamespace(vegetation=SimpleNamespace(**geometry), lai=geometry["lai"]),
        slowproc_derivvar=SimpleNamespace(height=np.ones((1, 14))),
        condveg_initial_surface=surface, payload=SimpleNamespace(kjpindex=1), nvm=14,
        run_def_values={}, fortran_pft_ids=np.arange(1, 15), soilalbedo_bg=np.zeros((1, 2)),
        parse_run_def_indexed_selection=lambda *args: np.ones(14))
    return eval(compile(ast.Expression(value), "<actual-cold-diffuco-mapping>", "eval"), env)


@pytest.mark.parametrize("boundary", ["direct-assembly", "previous-state", "cold-mapping"])
def test_source_effective_upstream_fractions_reach_diffuco(boundary):
    from jax_orchidee.driver import orchestration as orch

    fields, fractions = _upstream()
    supplied = {**fields, **fractions._asdict()}
    if boundary == "previous-state":
        supplied, _, _ = orch._diffuco_restart_payload_from_state_fields(
            diffuco_state=supplied, hydrol_state={}, slowproc_state={})
    elif boundary == "cold-mapping":
        supplied = _cold_mapping_expression(fields, fractions)
    actual = assemble_diffuco_first_step_precall_payload(
        driver_or_intersurf_payload={}, restart_payload=supplied, ok_explicitsnow=False)
    # Independent source reductions, not expectations taken from the producer.
    np.testing.assert_allclose(actual.payload["frac_snow_veg"], [10 / (10 + 20 * 660 / 100)],
                               rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(actual.payload["frac_snow_nobio"], [[5 / (5 + 20)]],
                               rtol=1e-12, atol=1e-14)
    assert {"frac_snow_veg", "frac_snow_nobio"} <= set(actual.restart_inputs)


@pytest.mark.parametrize("explicit", [False, True])
def test_missing_upstream_fractions_keep_existing_default_fallback(explicit):
    fields, _ = _upstream()
    actual = assemble_diffuco_first_step_precall_payload(
        driver_or_intersurf_payload={}, restart_payload=fields, ok_explicitsnow=explicit)
    expected = condveg_frac_snow(**fields, ok_explicitsnow=explicit)
    np.testing.assert_array_equal(actual.payload["frac_snow_veg"], expected.frac_snow_veg)
    np.testing.assert_array_equal(actual.payload["frac_snow_nobio"], expected.frac_snow_nobio)
    assert {"frac_snow_veg", "frac_snow_nobio"} <= set(actual.source_kernel_inputs)


def test_fraction_carry_does_not_expand_serialized_restart_requirements():
    assert "frac_snow_veg" not in DIFFUCO_FIRST_STEP_PRECALL_RESTART_FIELDS
    assert "frac_snow_nobio" not in DIFFUCO_FIRST_STEP_PRECALL_RESTART_FIELDS


def test_real_condveg_module_writer_to_next_diffuco_preserves_source_fractions():
    """Real main -> actual state writer -> actual mapper/assembler, no rollout."""
    from jax_orchidee.driver import orchestration as orch
    from jax_orchidee.sechiba.condveg import condveg_main_minimal

    snow, _ = _upstream()
    vegetation = np.zeros((1, 14))
    vegetation[:, 0] = 1.
    module = condveg_main_minimal(**snow, snowcri_alb=20., sn_dens=660.,
        ok_explicitsnow=False, veget=vegetation, veget_max=vegetation,
        frac_nobio=np.zeros((1, 1)), totfrac_nobio=np.zeros(1),
        zlev=np.array([30.]), height=np.zeros((1, 14)),
        temp_air=np.array([290.]), pb=np.array([1000.]), u=np.array([.05]), v=np.zeros(1),
        lai=np.zeros((1, 14)), emis_scal=1.)
    state = orch._empty_previous_step_state_fields()
    state["hydrol_previous_step_state"].update(snow)
    orch._add_condveg_to_previous_fields(state, SimpleNamespace(ok=True, module=module))
    mapped, _, _ = orch._diffuco_restart_payload_from_state_fields(
        diffuco_state=state["diffuco_previous_step_state"],
        hydrol_state=state["hydrol_previous_step_state"], slowproc_state={})
    actual = assemble_diffuco_first_step_precall_payload(
        driver_or_intersurf_payload={}, restart_payload=mapped, ok_explicitsnow=False)
    np.testing.assert_allclose(actual.payload["frac_snow_veg"], [10 / (10 + 20 * 660 / 100)],
                               rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(actual.payload["frac_snow_nobio"], [[5 / (5 + 20)]],
                               rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(state["diffuco_previous_step_state"]["frac_snow_veg"], module.frac_snow_veg)


def _r1_boundary_snow(changed=False):
    """Finite positive pre/post HYDROL boundary arrays, not a HYDROL rollout."""
    density, depth = (300., .02) if changed else (200., .01)
    return dict(snow=np.array([density * depth * 3]),
        snow_nobio=np.array([[8. if changed else 2.]]),
        snowrho=np.full((1, 3), density), snowdz=np.full((1, 3), depth),
        snow_age=np.array([.5 if changed else 3.]),
        snow_nobio_age=np.array([[1. if changed else 4.]]))


def _r1_actual_restart_condveg_call(pre, post, *, explicit, hydrol_ok=True):
    """Replay only the actual restart-scaffold call and real CONDVEG component."""
    from jax_orchidee.driver import orchestration as orch

    vegetation = np.zeros((1, 14))
    vegetation[:, 0] = .8
    restart = {**pre, "veget": vegetation, "veget_max": vegetation.copy(),
        "frac_nobio": np.array([[.2]]), "height": np.full((1, 14), 2.),
        "lai": np.ones((1, 14)), "soilalbedo_bg": np.array([[.1, .2]])}
    values = dict(ALB_BG_MODIS="TRUE", ALB_BARE_MODEL="FALSE", CONDVEG_SNOWA="1e20",
        TCST_SNOWA="10", ALB_ICE__00001=".6", ALB_ICE__00002=".2",
        OK_EXPLICITSNOW="TRUE" if explicit else "FALSE", IMPOSE_AZE="FALSE", ROUGH_DYN="TRUE")
    for name, scalar in (("ALB_LEAF_VIS", .1), ("ALB_LEAF_NIR", .2),
                         ("SNOWA_AGED_VIS", .6), ("SNOWA_AGED_NIR", .2),
                         ("SNOWA_DEC_VIS", .1), ("SNOWA_DEC_NIR", .1)):
        values.update({f"{name}__{index:05d}": str(scalar) for index in range(1, 15)})
    scalar_inputs = SimpleNamespace(is_tree=np.zeros(14, dtype=bool),
        z0_over_height=np.full(14, .0625), ratio_z0m_z0h=np.full(14, 10.))
    env = dict(vars(orch))
    env.update(first_step_restart_state=SimpleNamespace(diffuco_precall=SimpleNamespace(
        **restart, as_payload=lambda: dict(restart))),
        hydrol_first_step_module=SimpleNamespace(ok=hydrol_ok,
            snow_state=None if post is None else SimpleNamespace(**post),
            diagnostics=SimpleNamespace(drysoil_frac=np.array([.5]))),
        payload=SimpleNamespace(zlev=np.array([10.]), temp_air=np.array([285.]),
            pb=np.array([1013.]), u=np.array([1.]), v=np.zeros(1)),
        step=SimpleNamespace(run_scalars=scalar_inputs), run_def_values=values)
    tree = ast.parse(Path(orch.__file__).read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "paper_1961_driver_timestep_scaffold")
    calls = [node for node in ast.walk(owner) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "run_condveg_first_step_module"]
    assert len(calls) == 1
    return eval(compile(ast.Expression(calls[0]), "<actual-restart-scaffold-condveg-call>", "eval"), env)


def _r1_source_fractions(snow, explicit):
    # condveg.f90:878-893; default SNOWCRI_ALB=10 and SNOW_DENSITY=330.
    if explicit:
        depth = float(np.sum(snow["snowdz"]))
        average_density = float(np.sum(snow["snowdz"] * snow["snowrho"])) / depth
        vegetation = np.tanh(depth / (.025 * (average_density / 50.)))
    else:
        mass = float(snow["snow"][0])
        vegetation = mass / (mass + 10. * 330. / 100.)
    nobio_mass = float(snow["snow_nobio"][0, 0])
    return dict(frac_snow_veg=np.array([vegetation]),
                frac_snow_nobio=np.array([[nobio_mass / (nobio_mass + 10.)]]))


@pytest.mark.parametrize("field", ["frac_snow_veg", "frac_snow_nobio"])
@pytest.mark.parametrize("changed", [False, True], ids=["unchanged-default", "updated-snow"])
@pytest.mark.parametrize("explicit", [False, True], ids=["mass", "explicit"])
def test_r1_restart_scaffold_call_and_packet_use_post_hydrol_snow(explicit, changed, field):
    """Actual call/component -> writer/packet -> mapper/assembler; no full scaffold."""
    from jax_orchidee.driver import orchestration as orch

    pre, post = _r1_boundary_snow(), _r1_boundary_snow(changed)
    closure = _r1_actual_restart_condveg_call(pre, post, explicit=explicit)
    assert closure.ok and closure.module.albedo is not None
    assert {"frac_snow_veg", "frac_snow_nobio", "albedo"} <= set(closure.covered_fields)
    expected = _r1_source_fractions(post, explicit)
    # Separate before assertions establish both affected fractions independently.
    np.testing.assert_allclose(getattr(closure.module, field), expected[field], rtol=1e-12, atol=1e-14)
    for name in ("snow", "snow_nobio", "snowrho", "snowdz", "snow_age", "snow_nobio_age"):
        np.testing.assert_array_equal(closure.inputs[name], post[name])
    state = orch._empty_previous_step_state_fields()
    # Declared post-HYDROL state; HYDROL itself is not executed by this fixture.
    state["hydrol_previous_step_state"].update(post)
    orch._add_condveg_to_previous_fields(state, closure)
    packet = orch._packet_from_previous_step_fields(tstep=0, fields=state)
    mapped, _, _ = orch._diffuco_restart_payload_from_previous_state(packet)
    next_call = assemble_diffuco_first_step_precall_payload(
        driver_or_intersurf_payload={}, restart_payload=mapped, ok_explicitsnow=explicit)
    np.testing.assert_array_equal(next_call.payload["snow"], post["snow"])
    for name, value in expected.items():
        np.testing.assert_allclose(next_call.payload[name], value, rtol=1e-12, atol=1e-14)
        assert name in next_call.restart_inputs and name not in next_call.source_kernel_inputs


@pytest.mark.parametrize("hydrol_ok", [False, True])
def test_r1_missing_snow_state_preserves_original_restart_fallback(hydrol_ok):
    pre = _r1_boundary_snow()
    closure = _r1_actual_restart_condveg_call(pre, None, explicit=True, hydrol_ok=hydrol_ok)
    assert closure.module is not None
    assert {"frac_snow_veg", "frac_snow_nobio"} <= set(closure.covered_fields)
    for name, expected in _r1_source_fractions(pre, True).items():
        np.testing.assert_allclose(getattr(closure.module, name), expected, rtol=1e-12, atol=1e-14)
    if not hydrol_ok:
        assert "albedo" not in closure.covered_fields and closure.module.albedo is None


def test_r1_missing_explicit_geometry_retains_strict_missing_input_boundary():
    pre = {**_r1_boundary_snow(), "snowrho": None, "snowdz": None}
    closure = _r1_actual_restart_condveg_call(pre, None, explicit=True)
    assert closure.module is None and not closure.ok
    assert {"snowrho", "snowdz"} <= set(closure.missing_inputs)
